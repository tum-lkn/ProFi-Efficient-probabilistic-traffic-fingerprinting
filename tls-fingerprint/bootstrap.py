from __future__ import annotations
from typing import Tuple
import scipy.stats as stats
import numpy as np
# import matplotlib.pyplot as plt


def _test_statistics(a: np.array, b: np.array, mu_a: float | None = None, mu_b: float | None = None) -> float:
    if mu_a is None:
        mu_a = np.mean(a)
    if mu_b is None:
        mu_b = np.mean(b)
    var_a = np.var(a)
    var_b = np.var(b)
    statistics = (mu_a - mu_b) / np.sqrt((var_a / a.size) + (var_b / b.size))
    return float(statistics)


def _sample_statistic(sample: np.array, statistic: callable, num_samples: int, seed: int) -> np.array:

    sampled = np.zeros(num_samples)
    for i in range(num_samples):
        sampled[i] = statistic(sample[r.randint(0, sample.size, sample.size)])
    return sampled


def resample(r: np.random.RandomState, arr: np.array) -> np.array:
    return arr[r.randint(0, arr.size, arr.size)]


def interpret_p_val(p_val: float, confidence_level: float):
    if p_val < confidence_level:
        print("p_val < confidence_level --> reject H0, i.e., the two groups are different.")
    else:
        print("p_val >= confidence_level --> accept H0, i.e., there is no difference between the groups.")


def hypothesis_test(a: np.array, b: np.array, num_samples: int = 1000,
                     alternative='two-sided', seed: int = 1) -> float:
    """
    Use bootstrapping to determine the test-statistics for the mean.

    Args:
        a: First sample.
        b: Second sample.
        num_samples: Number of bootstrap samples to perform.
        alternative: Alternative hypothesis: {less, greater, two-sided}.
        seed: Seed for the random number generator.

    Returns:
        p_val: The p-value for the selected test. The fraction of bootstrap samples
            that match the null-hypothesis, i.e., equal means.

    Note:
        Interpretation of alternatives:
            - less: The mean of a is less than the mean of b.
            - greater: THe mean of a is greater than the mean of b.
            - two-sided: The mean of a and b are different.

        Interpretation p-value:
            In general the p-value give sthe likelihood of observing a specific
            set of observations under the assumption that the null-hypothesis
            is true. The smaller the p-value the less likely the null hypothesis
            and the more likely the alternative hypothesis.

    """
    if alternative == 'two-sided':
        p1 = hypothesis_test(a, b, num_samples, 'greater', seed)
        p2 = hypothesis_test(a, b, num_samples, 'less', seed)
        # Taken from here: https://www.stat.umn.edu/geyer/5601/examp/tests.html
        p_val = float(2 * np.min([p1, p2]))
    else:
        comparisons = {
            'greater': lambda t, t_prime: float(t_prime > t),
            'less': lambda t, t_prime: float(t_prime < t)
        }
        r = np.random.RandomState(seed=seed)
        mu_a = float(np.mean(a))
        mu_b = float(np.mean(b))
        mu_ab = np.mean(np.concatenate([a, b]))
        a_prime = a - mu_a + mu_ab
        b_prime = b - mu_b + mu_ab
        t = _test_statistics(a, b, mu_a, mu_b)

        true_conditions = 0
        for i in range(num_samples):
            resampled_a = resample(r, a_prime)
            resampled_b = resample(r, b_prime)
            t_prime = _test_statistics(resampled_a, resampled_b)
            true_conditions += comparisons[alternative](t, t_prime)
        p_val = true_conditions / num_samples
    return p_val


def bootstrap_ci(a: np.array, alpha: float, statistic: callable,
                 bootstrap_samples: int = 1000, seed: int = 1) -> Tuple[float, float]:
    u = statistic(a)
    u_p = np.zeros(bootstrap_samples)
    r = np.random.RandomState(seed=seed)
    for i in range(bootstrap_samples):
        u_p[i] = statistic(resample(r, a)) - u
    perc = (100 - alpha) / 2.
    lower = np.percentile(u_p, perc)
    upper = np.percentile(u_p, 100 - perc)
    # No, the order of upper and lower used here and both times subtracted is correct.
    return u - upper, u - lower


if __name__ == '__main__':
    r = np.random.RandomState(seed=1)
    s1 = r.normal(loc=0., size=1000)
    s2 = r.normal(loc=0., size=1000)
    s3 = r.normal(loc=1., size=1000)
    print(f"mu(s1) > mu(s2):  {hypothesis_test(s1, s2):.4f} vs {stats.ttest_ind(s1, s2, alternative='greater')[1]:.4f}")
    print(f"mu(s1) > mu(s3):  {hypothesis_test(s1, s3):.4f} vs {stats.ttest_ind(s1, s3, alternative='greater')[1]:.4f}")
    print(f"mu(s1) < mu(s2):  {hypothesis_test(s1, s2, alternative='less'):.4f} vs {stats.ttest_ind(s1, s2, alternative='less')[1]:.4f}")
    print(f"mu(s1) < mu(s3):  {hypothesis_test(s1, s3, alternative='less'):.4f} vs {stats.ttest_ind(s1, s3, alternative='less')[1]:.4f}")
    print(f"mu(s1) != mu(s2): {hypothesis_test(s1, s2, alternative='two-sided'):.4f} vs {stats.ttest_ind(s1, s2, alternative='two-sided')[1]:.4f}")
    print(f"mu(s1) != mu(s3): {hypothesis_test(s1, s3, alternative='two-sided'):.4f} vs {stats.ttest_ind(s1, s3, alternative='two-sided')[1]:.4f}")
    print(f"mu(s2) != mu(s1): {hypothesis_test(s2, s1, alternative='two-sided'):.4f} vs {stats.ttest_ind(s2, s1, alternative='two-sided')[1]:.4f}")
    print(f"mu(s2) != mu(s3): {hypothesis_test(s2, s3, alternative='two-sided'):.4f} vs {stats.ttest_ind(s2, s3, alternative='two-sided')[1]:.4f}")
    print(f"mu(s3) != mu(s1): {hypothesis_test(s3, s1, alternative='two-sided'):.4f} vs {stats.ttest_ind(s3, s1, alternative='two-sided')[1]:.4f}")
    print(f"mu(s3) != mu(s2): {hypothesis_test(s3, s2, alternative='two-sided'):.4f} vs {stats.ttest_ind(s3, s2, alternative='two-sided')[1]:.4f}")

    print("Confidence interval: ", bootstrap_ci(s1, 90, np.mean), stats.t.interval(0.9, loc=np.mean(s1), df=len(s1) - 1, scale=stats.sem(s1)))