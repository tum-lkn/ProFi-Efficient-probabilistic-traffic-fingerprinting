import argparse
import logging
import sys

import torch

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, TraceTMC_ELBO
from pyro.optim import DCTAdam
from pyro.util import ignore_jit_warnings
from typing import Any, List, Dict, Tuple
import utils

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.DEBUG)

# Add another handler for logging debugging events (e.g. for profiling)
# in a separate stream that can be captured.
log = logging.getLogger()
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(filter=lambda record: record.levelno <= logging.DEBUG)
log.addHandler(debug_handler)

# Pyro distributions have two different shapes:
# - batch-shape
# - event shape.
# The batch shape denotes conditionally independent random variables. event-shape
# denotes dependent variables, i.e., one draw from the distribution.


def phmm_model(sequences: torch.Tensor, lengths: torch.Tensor, args: Any,
              num_steps: int=None, batch_size: int=None, include_prior: bool=True):
    """
    Create a normal, basic HMM model.

    Args:
        sequences: The observed evidence, has shape (num_sequences, max_length, data_dim).
        lengths: The length of each sequence in `sequences`, has shape (,num_sequences).
        args: Object returned by the argument handler.
        num_steps: The number of steps that the HMM should have.
        batch_size: The size of minibatches.
        include_prior:

    Returns:

    """
    # Sometimes it is safe to ignore jit warnings. Here we use the
    # pyro.util.ignore_jit_warnings context manager to silence warnings about
    # conversion to integer, since we know all three numbers will be the same
    # across all invocations to the model.
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length

    if num_steps is None:
        num_steps = torch.mean(lengths).int()

    # Mask out sample statements. The effect is the same as adding an if-statement
    # that includes the sampling of the prior or does not include it, i.e.,
    # if `include_prior` is true, then the sample statements are masked out.
    # See: http://docs.pyro.ai/en/0.3.0-release/poutine.html
    with poutine.mask(mask=include_prior):
        # `pyro.sample` samples from a distribution and stores the sample
        # statement in pyros backend. The backend uses these names to uniquely
        # identify the statements and change their behavior at runtime based
        # on how the enclosing stochastic function is used.
        # I guess in this case I always generate probs_x and probs_y, but depending
        # on `include_prior` the distributions will be considered when infering
        # parameters.
        #
        # The method `.to_event` is used to declare the given dimensions as
        # dependent from the right. This allows treating univariate distribution
        # as multivariate distributions. In this case, the batch-shape of
        # probs_x is now `args.hidden_dim` and the event shape is also
        # `args.hidden_dim`. More generally, if the matrix passed to the
        # dirichlet distribution would be m x n, then the event-shape would be
        # n and the batch-shape would be m.
        #
        # Define a uniform prior over transitions. In total, there are:
        # - T - 1 DELETE states
        # - T - 1 MATCH states
        # - T     INSERT states
        # - 1     START state
        # that have three neighbors each, i.e., 2(T - 1) + T + 1 transitions
        # with three destinations each.
        probs_x = pyro.sample(
            "probs_x",
            dist.Dirichlet(100. * torch.ones(2 * (num_steps - 1) + num_steps + 1, 3)).to_event(1)
        )
        # The last hidden states before the end states have two transitions each.
        # The insert state to itself and the END state, the match and delete
        # state to the INSERT and the END state.
        probs_e = pyro.sample(
            "probs_e",
            dist.Dirichlet(100. * torch.ones(3, 2)).to_event(1)
        )
        # In this case the batch-shape is now [()], whereas the event-shape is
        # (args.hidden_dim, data_dim). That is, there are no independent variables.
        # All variables are dependent.
        #
        # Define a prior over the observations with shape (hidden_dim, data_dim),
        # i.e., the probability of drawing each observation at every state.
        # Why is dim of event shape 2 here?
        probs_y = pyro.sample(
            "probs_y",
            dist.Beta(0.1, 0.9).expand([args.hidden_dim, data_dim]).to_event(2)
        )

    # Create a plate over the dimension of the observations. Declare the most
    # right dimension as independent. `tones_plate` when iterates over `data_dim`
    # and returns numbers just as a python `range` would.
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    # We subsample batch_size items out of num_sequences items. Note that since
    # we're using dim=-1 for the notes plate, we need to batch over a different
    # dimension, here dim=-2.
    with pyro.plate("sequences", num_sequences, batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        x = 0
        # If we are not using the jit, then we can vary the program structure
        # each call by running for a dynamically determined number of time
        # steps, lengths.max(). However if we are using the jit, then we try to
        # keep a single program structure for all minibatches; the fixed
        # structure ends up being faster since each program structure would
        # need to trigger a new jit compile stage.
        #
        # The markov tells pyro that variables in the loop depend only on
        # variables outside of the loop, variables from the previous and the
        # current step of the loop (markov assumption). This allows pyro to
        # recyle dimensions during the enumeration of discrete variables.
        # See http://pyro.ai/examples/enumeration.html for more info.
        for t in pyro.markov(range(max_length if args.jit else lengths.max())):
            # Again, mask out the sample statements for time indices that are
            # smaller than the the corresponding lengths of those sequences.
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                # Draw next hidden state.
                x = pyro.sample(
                    name="x_{}".format(t),
                    fn=dist.Categorical(probs_x[x]),
                    infer={"enumerate": "parallel"}
                )
                with tones_plate:
                    # Condition on the evidence and the prior for the
                    # observation models. In inference, this amounts to inferring
                    # a distribution over the output given the observations and
                    # our guess.
                    pyro.sample(
                        name="y_{}".format(t),
                        fn=dist.Bernoulli(probs_y[x.squeeze(-1)]),
                        obs=sequences[batch, t]
                    )

def hmm_model(sequences: torch.Tensor, lengths: torch.Tensor, args: Any,
              batch_size: int=None, include_prior: bool=True):
    """
    Create a normal, basic HMM model.

    Args:
        sequences: The observed evidence, has shape (num_sequences, max_length, data_dim).
        lengths: The length of each sequence in `sequences`, has shape (,num_sequences).
        args: Object returned by the argument handler.
        batch_size: The size of minibatches.
        include_prior:

    Returns:

    """
    # Sometimes it is safe to ignore jit warnings. Here we use the
    # pyro.util.ignore_jit_warnings context manager to silence warnings about
    # conversion to integer, since we know all three numbers will be the same
    # across all invocations to the model.
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length

    # Mask out sample statements. The effect is the same as adding an if-statement
    # that includes the sampling of the prior or does not include it, i.e.,
    # if `include_prior` is true, then the sample statements are masked out.
    # See: http://docs.pyro.ai/en/0.3.0-release/poutine.html
    with poutine.mask(mask=include_prior):
        # `pyro.sample` samples from a distribution and stores the sample
        # statement in pyros backend. The backend uses these names to uniquely
        # identify the statements and change their behavior at runtime based
        # on how the enclosing stochastic function is used.
        # I guess in this case I always generate probs_x and probs_y, but depending
        # on `include_prior` the distributions will be considered when infering
        # parameters.
        #
        # The method `.to_event` is used to declare the given dimensions as
        # dependent from the right. This allows treating univariate distribution
        # as multivariate distributions. In this case, the batch-shape of
        # probs_x is now `args.hidden_dim` and the event shape is also
        # `args.hidden_dim`. More generally, if the matrix passed to the
        # dirichlet distribution would be m x n, then the event-shape would be
        # n and the batch-shape would be m.
        #
        # Define a prior for the state transitions of shape (hidden_dim, hidden_dim),
        # where the probability of staying in a specific state is 0.9, and
        # transitioning to another state is 0.1.
        probs_x = pyro.sample(
            "probs_x",
            dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1).to_event(1)
        )
        # In this case the batch-shape is now [()], whereas the event-shape is
        # (args.hidden_dim, data_dim). That is, there are no independent variables.
        # All variables are dependent.
        #
        # Define a prior over the observations with shape (hidden_dim, data_dim),
        # i.e., the probability of drawing each observation at every state.
        # Why is dim of event shape 2 here?
        probs_y = pyro.sample(
            "probs_y",
            dist.Beta(0.1, 0.9).expand([args.hidden_dim, data_dim]).to_event(2)
        )

    # Create a plate over the dimension of the observations. Declare the most
    # right dimension as independent. `tones_plate` when iterates over `data_dim`
    # and returns numbers just as a python `range` would.
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    # We subsample batch_size items out of num_sequences items. Note that since
    # we're using dim=-1 for the notes plate, we need to batch over a different
    # dimension, here dim=-2.
    with pyro.plate("sequences", num_sequences, batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        x = 0
        # If we are not using the jit, then we can vary the program structure
        # each call by running for a dynamically determined number of time
        # steps, lengths.max(). However if we are using the jit, then we try to
        # keep a single program structure for all minibatches; the fixed
        # structure ends up being faster since each program structure would
        # need to trigger a new jit compile stage.
        #
        # The markov tells pyro that variables in the loop depend only on
        # variables outside of the loop, variables from the previous and the
        # current step of the loop (markov assumption). This allows pyro to
        # recyle dimensions during the enumeration of discrete variables.
        # See http://pyro.ai/examples/enumeration.html for more info.
        for t in pyro.markov(range(max_length if args.jit else lengths.max())):
            # Again, mask out the sample statements for time indices that are
            # smaller than the the corresponding lengths of those sequences.
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                # Draw next hidden state.
                x = pyro.sample(
                    name="x_{}".format(t),
                    fn=dist.Categorical(probs_x[x]),
                    infer={"enumerate": "parallel"}
                )
                with tones_plate:
                    # Condition on the evidence and the prior for the
                    # observation models. In inference, this amounts to inferring
                    # a distribution over the output given the observations and
                    # our guess.
                    pyro.sample(
                        name="y_{}".format(t),
                        fn=dist.Bernoulli(probs_y[x.squeeze(-1)]),
                        obs=sequences[batch, t]
                    )
                    
                    
def main(args):
    data = {
        'train': utils.dummy_phmm_data(1000, 1),
        'test': utils.dummy_phmm_data(100, 10000)
    }
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logging.info('Loading data')

    logging.info('-' * 40)
    sequences = data['train']['sequences']
    lengths = data['train']['sequence_lengths']

    if args.truncate:
        lengths = lengths.clamp(max=args.truncate)
        sequences = sequences[:, :args.truncate]
    num_observations = float(lengths.sum())
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()
    pyro.enable_validation(__debug__)

    # We'll train using MAP Baum-Welch, i.e. MAP estimation while marginalizing
    # out the hidden state x. This is accomplished via an automatic guide that
    # learns point estimates of all of our conditional probability tables,
    # named probs_*.
    model = hmm_model
    guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("probs_")))

    # To help debug our tensor shapes, let's print the shape of each site's
    # distribution, value, and log_prob tensor. Note this information is
    # automatically printed on most errors inside SVI.
    if args.print_shapes:
        first_available_dim = -3
        guide_trace = poutine.trace(guide).get_trace(
            sequences,
            lengths,
            args=args,
            batch_size=args.batch_size
        )
        model_trace = poutine.trace(
            poutine.replay(
                poutine.enum(model, first_available_dim),
                guide_trace
            )
        ).get_trace(
            sequences,
            lengths,
            args=args,
            batch_size=args.batch_size
        )
        logging.info(model_trace.format_shapes())

    # Enumeration requires a TraceEnum elbo and declaring the max_plate_nesting.
    # All of our models have two plates: "data" and "tones".
    optim = DCTAdam({'lr': args.learning_rate})
    if args.tmc:
        if args.jit:
            raise NotImplementedError("jit support not yet added for TraceTMC_ELBO")
        elbo = TraceTMC_ELBO(max_plate_nesting=2)
        tmc_model = poutine.infer_config(
            model,
            lambda msg: {
                "num_samples": args.tmc_num_samples,
                "expand": False
            } if msg["infer"].get("enumerate", None) == "parallel" else {}
        )  # noqa: E501
        svi = SVI(tmc_model, guide, optim, elbo)
    else:
        Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
        elbo = Elbo(
            max_plate_nesting=2,
            strict_enumeration_warning=True,
            jit_options={"time_compilation": args.time_compilation}
        )
        svi = SVI(model, guide, optim, elbo)

    # We'll train on small minibatches.
    logging.info('Step\tLoss')
    for step in range(args.num_steps):
        loss = svi.step(sequences, lengths, args=args, batch_size=args.batch_size)
        logging.info('{: >5d}\t{}'.format(step, loss / num_observations))

    if args.jit and args.time_compilation:
        logging.debug('time to compile: {} s.'.format(elbo._differentiable_loss.compile_time))

    # We evaluate on the entire training dataset,
    # excluding the prior term so our results are comparable across models.
    train_loss = elbo.loss(model, guide, sequences, lengths, args, include_prior=False)
    logging.info('training loss = {}'.format(train_loss / num_observations))

    # Finally we evaluate on the test dataset.
    logging.info('-' * 40)
    logging.info('Evaluating on {} test sequences'.format(len(data['test']['sequences'])))
    sequences = data['test']['sequences']
    lengths = data['test']['sequence_lengths']
    if args.truncate:
        lengths = lengths.clamp(max=args.truncate)
    num_observations = float(lengths.sum())

    # note that since we removed unseen notes above (to make the problem a bit easier and for
    # numerical stability) this test loss may not be directly comparable to numbers
    # reported on this dataset elsewhere.
    test_loss = elbo.loss(model, guide, sequences, lengths, args=args, include_prior=False)
    logging.info('test loss = {}'.format(test_loss / num_observations))

    # We expect models with higher capacity to perform better,
    # but eventually overfit to the training set.
    capacity = sum(value.reshape(-1).size(0)
                   for value in pyro.get_param_store().values())
    logging.info('{} capacity = {} parameters'.format(model.__name__, capacity))


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.4.0')
    parser = argparse.ArgumentParser(description="MAP Baum-Welch learning Bach Chorales")
    parser.add_argument("-n", "--num-steps", default=50, type=int)
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("-d", "--hidden-dim", default=16, type=int)
    parser.add_argument("-nn", "--nn-dim", default=48, type=int)
    parser.add_argument("-nc", "--nn-channels", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-t", "--truncate", type=int)
    parser.add_argument("-p", "--print-shapes", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--time-compilation', action='store_true')
    parser.add_argument('-rp', '--raftery-parameterization', action='store_true')
    parser.add_argument('--tmc', action='store_true',
                        help="Use Tensor Monte Carlo instead of exact enumeration "
                             "to estimate the marginal likelihood. You probably don't want to do this, "
                             "except to see that TMC makes Monte Carlo gradient estimation feasible "
                             "even with very large numbers of non-reparametrized variables.")
    parser.add_argument('--tmc-num-samples', default=10, type=int)
    args = parser.parse_args()
    main(args)

