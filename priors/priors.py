"""Parameter Priors.

Functions to sample from prior distributions.
"""

# Required libraries
from typing import Callable
import numpy as np


# Uniform priors
def generate_uniform_prior_sampler(
        low: float,
        high: float
) -> Callable:
    """Generate a uniform prior sampler function.

    Parameters
    ----------
    low : float
        The lower bound of the prior.
    high : float
        The upper bound of the prior.

    Returns
    -------
    prior_sampler : Callable
        Function that takes a number of samples and returns the sampled
        values as an array.
    """
    def _prior_sampler(num_samples: int) -> np.ndarray:
        """Sample from a uniform prior."""
        return np.random.uniform(low=low, high=high, size=num_samples)
    return _prior_sampler


# Log uniform priors
def generate_log_uniform_prior_sampler(
        low: float,
        high: float
) -> Callable:
    """Generate a log uniform prior sampler function.

    Parameters
    ----------
    low : float
        The lower bound of the prior.
    high : float
        The upper bound of the prior.

    Returns
    -------
    prior_sampler : Callable
        Function that takes a number of samples and returns the sampled
        values as an array.
    """
    def _prior_sampler(num_samples: int) -> np.ndarray:
        """Sample from a log uniform prior."""
        return np.exp(np.random.uniform(low=np.log(low),
                                        high=np.log(high),
                                        size=num_samples))
    return _prior_sampler


# Gaussian priors
def generate_gaussian_prior_sampler(
        mean: float,
        std: float
) -> Callable:
    """Generate a Gaussian prior sampler function.

    Parameters
    ----------
    mean : float
        The mean of the prior.
    std : float
        The standard deviation of the prior.

    Returns
    -------
    prior_sampler : Callable
        Function that takes a number of samples and returns the sampled
        values as an array.
    """
    def _prior_sampler(num_samples: int) -> np.ndarray:
        """Sample from a Gaussian prior."""
        return np.random.normal(loc=mean, scale=std, size=num_samples)
    return _prior_sampler


# Truncated Gaussian priors
def generate_truncated_gaussian_prior_sampler(
        mean: float,
        std: float,
        low: float,
        high: float
) -> Callable:
    """Generate a truncated Gaussian prior sampler function.

    Parameters
    ----------
    mean : float
        The mean of the prior.
    std : float
        The standard deviation of the prior.
    low : float
        The lower bound of the prior.
    high : float
        The upper bound of the prior.

    Returns
    -------
    prior_sampler : Callable
        Function that takes a number of samples and returns the sampled
        values as an array.
    """
    def _prior_sampler(num_samples: int) -> np.ndarray:
        """Sample from a truncated Gaussian prior."""
        samples = np.empty(num_samples)
        out_of_range = np.array([True] * num_samples)
        while np.any(out_of_range):
            samples[out_of_range] = np.random.normal(loc=mean,
                                                     scale=std,
                                                     size=np.sum(out_of_range))
            out_of_range = np.logical_or(samples < low, samples > high)

        return samples
    return _prior_sampler


# Composite priors
def generate_composite_prior_sampler(
        *prior_sampler_functions: Callable
) -> Callable:
    """Generate a composite prior sampler function.

    Parameters
    ----------
    *prior_sampler_functions : Callable
        The prior sampler functions to combine.

    Returns
    -------
    prior_sampler : Callable
        Function that takes a number of samples and returns the sampled
        values as an array. One row per sample, one column per parameter.
    """
    def _prior_sampler(num_samples: int) -> np.ndarray:
        """Sample from a composite prior."""
        samples = np.empty((num_samples, len(prior_sampler_functions)))
        for idx, prior_sampler_function in enumerate(prior_sampler_functions):
            samples[:, idx] = prior_sampler_function(num_samples)
        return samples
    return _prior_sampler
