"""Noise simulators and simulator generators.

This module contains functions that generate noise simulators.
"""

# Import libraries
from __future__ import annotations
from typing import Tuple, Callable
import numpy as np
from pandas import DataFrame
from .core import Simulator


# White noise
def white_noise(
        shape: np.ndarray | int,
        sigma_noise: np.ndarray
) -> np.ndarray:
    """Return white noise with given sigma value.

    Parameters
    ----------
    shape : np.ndarray
        The shape of the noise array to generate.
    sigma_noise : np.ndarray
        The noise standard deviation(s).

    Returns
    -------
    noise : np.ndarray
        White noise.
    """
    if isinstance(shape, int):
        shape = (shape,)
    normalized_noise = np.random.normal(loc=0.0, size=shape)
    if len(sigma_noise) == 1:
        return normalized_noise * sigma_noise
    return np.einsum('i,ij...->ij...', sigma_noise,
                     normalized_noise)


def generate_white_noise_simulator(
        noise_size: int,
        sigma_sampler: Callable
) -> Simulator:
    """Return a simulator function for white noise.

    Parameters
    ----------
    noise_size : int
        Size of noise vector for each data simulation.
    sigma_sampler : Callable
        Function that returns samples of the noise standard deviation.

    Returns
    -------
    noise_simulator : Callable
        Function that takes a number of data simulations and returns
        the generated white noise data for each simulation as an array
        and an empty data frame.
    """
    def _noise_simulator(num_data_simulations: int
                         ) -> Tuple[np.ndarray, DataFrame]:
        """Return a simulator for a white noise model."""
        sigma_samples = sigma_sampler(num_data_simulations)
        noise = white_noise(np.array([num_data_simulations, noise_size]),
                            sigma_samples)
        return noise, DataFrame(sigma_samples[:, np.newaxis],
                                columns=['sigma'])
    return _noise_simulator
