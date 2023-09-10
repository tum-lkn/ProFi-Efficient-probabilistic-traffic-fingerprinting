import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_probability as tfp

from data_conversion import data_aggregation
from data_conversion import constants

tfd = tfp.distributions

def set_constant_states_tf() -> None:

    set_constants()

    for key, state in constants.tls_states_hmm_c.items():
        constants.tls_states_hmm_c[key] = state * constants.bins

    for key, state in constants.tls_states_hmm_s.items():
        constants.tls_states_hmm_s[key] = state * constants.bins

    constants.default_state *= constants.bins
    constants.num_emissions = constants.default_state + 1


def shape_trace_tf(trace: List[int]) -> List[int]:
    """
    Cuts the trace to a specified length

    Args:
        trace (list): list with Server Messages and TLS traffic only

    Returns:
        trace (list): original list with randomly dropped packets
    """

    if constants.number_packets <= 0:
        return trace

    if len(trace) < constants.number_packets:
        trace.extend([constants.default_state] * (constants.number_packets - len(trace)))

    return trace[:constants.number_packets]


def convert_trace_to_states_tf(pcap_file: str, num_packets: int, centers: np.array=None,
                               included_packets: str='main_flow') -> List[int]:
    """
    Converts the pcap representation of a trace into the state representation of HMMs

    Args:
        pcap_file (string): path to the pcap file

    Returns:
        traces (list): list of states
    """

    trace = []
    pd_data, convertible = convert_pcap_to_dataframe(pcap_file)

    if not convertible:
        return trace

    pd_data = pd_data[:num_packets]
    if included_packets == constants.FLOW_MAIN:
        pd_data = extract_main_flow(pd_data)
    elif included_packets == constants.FLOW_CO:
        pass
    else:
        raise ValueError("Value {} not known for argument included_packets".format(included_packets))

    pd_data = pd_data[['ip.src', '_ws.col.Info', 'frame.len', 'frame.time_epoch']]
    pd_data['frame.time_epoch'] = pd_data['frame.time_epoch'] - pd_data['frame.time_epoch'][0]
    pd_data['frame.time_epoch'] /= pd_data['frame.time_epoch'].iloc[-1]
    client_ip = pd_data[pd_data['_ws.col.Info'].str.contains('Client Hello')].iloc[0].values[0]

    for info, source, length, time_rel in zip(pd_data['_ws.col.Info'], pd_data['ip.src'], pd_data['frame.len'], pd_data['frame.time_epoch']):
        info_splited = info.split(',')
        length_orig = length / len(info_splited)

        if centers is None:
            packet_size = int(np.rint(length_orig))
        elif centers.size == 1:
            packet_size = 0
        else:
            packet_size = int(np.argmin(np.abs(centers - length_orig)))

        # time_relative = np.rint(100 * time_rel)
        for info_state in info_splited:
            if 'Application Data' in info_state:
                info_state = 'Application Data'
            if 'Certificate' in info_state:
                info_state = 'Certificate'
            if 'Continuation Data' in info_state:
                info_state = 'Continuation Data'
            if source == client_ip:
                original_state = constants.tls_states_hmm_c.get(info_state.strip(), constants.default_state)
            else:
                original_state = constants.tls_states_hmm_s.get(info_state.strip(), constants.default_state)

            if original_state == constants.default_state:
                packet_size = 0
                time_relative = 0

            trace.append(original_state + packet_size)
            # trace.append(original_state)
    trace = shape_trace_tf(trace)
    return trace


def convert_pcap_to_states_tf(company: str, num_packets: int, browser=None, centers: np.array=None,
                              included_packets=constants.FLOW_MAIN,
                              data_set='train') -> List[int]:
    """
    Collects the traces for a specific application and converts them into a state representation

    Args:
        company (string): name of application

    Returns:
        traces (list): list of traces in state representation
    """
    pcap_files = acquire_pcap_files(company, browser=browser, data_set=data_set)
    traces = []
    for pcap_file in pcap_files:
        pcap = os.path.join(constants.trace_path, pcap_file + '.pcapng')
        trace = convert_trace_to_states_tf(pcap, num_packets, centers, included_packets)
        if trace:
            traces.append(trace)

    return traces


def save_hmm_tf(hmm: tfd.HiddenMarkovModel, company: str, priors: Tuple[tfd.Dirichlet, tfd.Dirichlet, tfd.Dirichlet]=None) -> None:
    """
    Writes a HMM and the priors to a hdf5 file

    Args:
        hmm (tfd.HiddenMarkovModel): Hidden Markov Model
        priors (list): list of init, transition and observation priors
        company (string): company name

    Returns:
        /
    """

    init = hmm.initial_distribution.logits_parameter().numpy()
    trans = hmm.transition_distribution.logits_parameter().numpy()
    obs = hmm.observation_distribution.logits_parameter().numpy()

    if priors is not None:

        obs_prior = priors[0].concentration.numpy()
        transition_prior = priors[1].concentration.numpy()
        init_prior = priors[2].concentration.numpy()

    hdf5_file = os.path.join(constants.model_path, company, 'hmm_' + company + '.h5')

    with h5py.File(hdf5_file, 'w') as hf:

        hf.create_dataset('init', data = init)

    with h5py.File(hdf5_file, 'a') as hf:

        hf.create_dataset('trans', data = trans)
        hf.create_dataset('obs', data = obs)

        if priors is not None:
            hf.create_dataset('init_prior', data = init_prior)
            hf.create_dataset('transition_prior', data = transition_prior)
            hf.create_dataset('obs_prior', data = obs_prior)


def load_hmm_tf(company: str) -> tfd.HiddenMarkovModel:
    """
    Loads a HMM of a hdf5 file

    Args:
        company (string): comapny name

    Returns:
        hmm (tfd.HiddenMarkovModel): Hidden Markov Model
    """

    hdf5_file = os.path.join(constants.model_path, company, 'hmm_' + company + '.h5') 

    if not os.path.exists(hdf5_file):

        return None

    init_prior = None
    transition_prior = None
    obs_prior = None

    with h5py.File(hdf5_file, 'r') as hf:

        init = tf.convert_to_tensor(hf['init'][:], dtype = tf.float64)
        trans = tf.convert_to_tensor(hf['trans'][:], dtype = tf.float64)
        obs = tf.convert_to_tensor(hf['obs'][:], dtype = tf.float64)

        try:

            init_prior = tf.convert_to_tensor(hf['init_prior'][:], dtype = tf.float64)
            transition_prior = tf.convert_to_tensor(hf['transition_prior'][:], dtype = tf.float64)
            obs_prior = tf.convert_to_tensor(hf['obs_prior'][:], dtype = tf.float64)

        except:

            pass

    hmm = tfd.HiddenMarkovModel(
            initial_distribution = tfd.Categorical(logits = init),
            transition_distribution = tfd.Categorical(logits = trans),
            observation_distribution = tfd.Categorical(logits = obs),
            num_steps = constants.number_packets,
            name = company
        )

    if obs_prior is not None or transition_prior is not None or init_prior is not None:
        obs_prior = tfd.Dirichlet(concentration = obs_prior)
        transition_prior = tfd.Dirichlet(concentration = transition_prior)
        init_prior = tfd.Dirichlet(concentration = init_prior)
        prior = [obs_prior, transition_prior, init_prior]

        return hmm, prior

    return hmm


def load_fingerprints_hmm_tf() -> Dict[str, tfd.HiddenMarkovModel]:
    """
    Loads all HMMs

    Args:
        /

    Returns:
        hmms (dict): dict of all HMMs with company names as keys
    """
    companys = get_services()
    hmms = {}
    for company in companys:
        hmm = load_hmm_tf(company)
        if hmm is not None:
            hmms[company] = hmm
    return hmms

