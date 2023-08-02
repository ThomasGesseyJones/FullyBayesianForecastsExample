"""Noise simulators and simulator generators.

This module contains functions that generate noise simulators.
"""

# Import libraries
from __future__ import annotations
from typing import Tuple
import numpy as np
from pandas import DataFrame
from .core import Simulator


# White noise
def white_noise(shape: np.ndarray | int, sigma_noise: float) -> np.ndarray:
    """Return white noise with given sigma value.

    Parameters
    ----------
    shape : np.ndarray
        The shape of the noise array to generate.
    sigma_noise : float
        The noise standard deviation.

    Returns
    -------
    noise : np.ndarray
        White noise.
    """
    if isinstance(shape, int):
        shape = (shape,)
    noise = np.random.normal(loc=0.0, scale=sigma_noise, size=shape)
    return noise


def generate_white_noise_simulator(
        noise_size: int,
        sigma_noise: int
) -> Simulator:
    """Returns a simulator function for white noise.

    Parameters
    ----------
    noise_size : int
        Size of noise vector for each data simulation.
    sigma_noise : float
        Noise standard deviation.

    Returns
    -------
    noise_simulator : Callable
        Function that takes a number of data simulations and returns
        the generated white noise data for each simulation as an array
        and an empty data frame.
    """
    def _noise_simulator(num_data_simulations: int
                         ) -> Tuple[np.ndarray, DataFrame]:
        """Simulator for a white noise model."""
        noise = white_noise(np.array([num_data_simulations, noise_size]),
                            sigma_noise)
        return noise, DataFrame()
    return _noise_simulator
