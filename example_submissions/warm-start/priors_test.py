
import matplotlib.pyplot as plt
from warm_start.hyperparameters import HyperParameters, hparam
from dataclasses import dataclass
from warm_start.priors import LogUniformPrior, UniformPrior
import numpy as np
from warm_start.utils import set_seed
import math

@dataclass
class A(HyperParameters):
    learning_rate: float = hparam(default=0.001, prior=LogUniformPrior(min=1e-6, max=1))


def test_log_uniform():
    n_bins = 5
    n_points = 100
    set_seed(123)
    x = [
        A.sample().to_array() for i in range(n_points)
    ]
    hist, bins, _ = plt.hist(x, bins=n_bins)
    # histogram on log scale. 
    # Use non-equal bin sizes, such that they look equal on log scale.
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    hist, bins, _ = plt.hist(x, bins=logbins)
    plt.xscale("log")
    
    counts = np.sum(hist, axis=0)
    mean = np.mean(counts)
    std = np.std(counts)
    
    error_to_mean_ratio = (std / mean)
    # TODO: This is not ideal, since changing the seed might break the test.
    # For this particular seed  (123), the variance is about 5.
    assert error_to_mean_ratio < 0.25, f"Variance is too large!{error_to_mean_ratio} {std}, {counts}, {plt.show()}"
    

@dataclass
class B(A):
    momentum: float = hparam(default=0., prior=UniformPrior(min=-2., max=2.))


def test_to_array():
    b = B.sample()
    array = b.to_array()
    assert array[0] == b.learning_rate
    assert array[1] == b.momentum


def test_log_uniform_and_uniform():
    n_bins = 5
    n_points = 100
    set_seed(123)
    x_samples = [
        B.sample() for i in range(n_points)
    ]

    assert all([0. < x.learning_rate < 1 for x in x_samples]), x_samples
    assert all([-2 < x.momentum < 2 for x in x_samples]), x_samples

    x = np.stack([
        x.to_array() for x in x_samples
    ])

    x0 = x[:, 0] # learning rate
    x1 = x[:, 1] # momentum
    print(x0.mean(), x0.std())
    print(x1.mean(), x1.std())

    assert np.abs(x1.mean()) <= 0.5
    assert x1.max() - 2 <= 0.2    
    assert x1.min() - (-2) <= 0.2    


def test_loguniform_prior():
    prior = LogUniformPrior(min=1, max=1e5, base=10)
    samples = [prior.sample() for _ in range(1000)]
    assert all(1 < x < 1e5 for x in samples)
    log_samples = np.log10(samples)
    mean = np.mean(log_samples)
    # mean base-10 exponent should be around 2.5 
    assert 2.4 <= mean <= 2.6
    