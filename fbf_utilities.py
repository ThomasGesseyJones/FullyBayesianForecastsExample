"""Fully Bayesian Forecast Utilities.

This file contains a series of utility functions for our fully Bayesian
forecasting pipeline. These functions are used in several of the main scripts,
and mainly deal with the construction of the simulators used in the Evidence
Network and IO.
"""

# Required imports
from typing import Callable, Collection, Tuple
import argparse
from simulators import additive_simulator_combiner, Simulator
from simulators.noise import generate_white_noise_simulator
from simulators.twenty_one_cm import load_globalemu_emulator, \
    GLOBALEMU_PARAMETER_RANGES, \
    global_signal_experiment_measurement_redshifts, \
    generate_global_signal_simulator, generate_foreground_simulator
from priors import generate_uniform_prior_sampler, \
    generate_log_uniform_prior_sampler, \
    generate_gaussian_prior_sampler, \
    generate_truncated_gaussian_prior_sampler
from copy import deepcopy
import yaml
import os
import pickle as pkl

# Parameters
NOISE_DEFAULT = 0.079  # K, taken from REACH mission paper


# IO
def get_noise_sigma() -> float:
    """Get the noise sigma from the command line arguments.

    Returns
    -------
    noise_sigma : float
        The noise sigma in K.
    """
    parser = argparse.ArgumentParser(
        description="Train the Evidence Network."
    )
    parser.add_argument(
        "noise",
        type=float,
        default=NOISE_DEFAULT,
        help="The noise sigma in K.",
        nargs='?'
    )
    args = parser.parse_args()
    return args.noise


def load_configuration_dict() -> dict:
    """Load the configuration dictionary from the config.yaml file.

    Returns
    -------
    config_dict : dict
        Dictionary containing the configuration parameters.
    """
    with open("configuration.yaml", 'r') as file:
        return yaml.safe_load(file)


def timing_filename(noise_sigma: float) -> str:
    """Get the filename for the timing data.

    Parameters
    ----------
    noise_sigma : float
        The noise sigma in K.

    Returns
    -------
    filename : str
        The filename for the timing data.
    """
    folder = os.path.join('figures_and_results', 'timing_data')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'en_noise_{noise_sigma:.4f}_timing_data.pkl')


def clear_timing_data(timing_file: str):
    """Clear the timing data file.

    Parameters
    ----------
    timing_file : str
        The filename for the timing data.
    """
    with open(timing_file, 'wb') as file:
        pkl.dump({}, file)


def add_timing_data(timing_file: str, entry_name: str, time_s: float):
    """Add timing data to the timing data file.

    Parameters
    ----------
    timing_file : str
        The filename for the timing data.
    entry_name : str
        The name of the entry to add.
    time_s : float
        The time to add in seconds.
    """
    if os.path.isfile(timing_file):
        with open(timing_file, 'rb') as file:
            timing_data = pkl.load(file)
    else:
        timing_data = {}
    timing_data[entry_name] = time_s
    with open(timing_file, 'wb') as file:
        pkl.dump(timing_data, file)


def _get_prior_generator(prior_type: str) -> Callable:
    """Get the appropriate prior generator given a prior type.

    Parameters
    ----------
    prior_type : str
        String ID of the prior type

    Returns
    -------
    prior_generator : Callable
        Prior generator corresponding to the given prior type
    """
    if prior_type == 'uniform':
        return generate_uniform_prior_sampler
    elif prior_type == 'log_uniform':
        return generate_log_uniform_prior_sampler
    elif prior_type == 'gaussian':
        return generate_gaussian_prior_sampler
    elif prior_type == 'truncated_gaussian':
        return generate_truncated_gaussian_prior_sampler
    else:
        raise ValueError("Unknown prior type.")


# Priors
def create_globalemu_prior_samplers(config_dict: dict) -> Collection[Callable]:
    """Create a prior sampler over the globalemu parameters.

    Parameters
    ----------
    config_dict : dict
        Dictionary containing the configuration parameters.

    Returns
    -------
    individual_priors : Collection[Callable]
        For each parameter, a function that takes a number of samples and
        returns the sampled values as an array.
    """
    # Loop over parameters constructing individual priors
    individual_priors = []
    for param in config_dict['priors']['global_signal'].keys():
        # Get prior info
        prior_info = deepcopy(config_dict['priors']['global_signal'][param])

        # Replace emu_min and emu_max with the min and max value globalemu
        # can take for this parameter
        for k, v in prior_info.items():
            if v == 'emu_min':
                prior_info[k] = GLOBALEMU_PARAMETER_RANGES[param][0]
            elif v == 'emu_max':
                prior_info[k] = GLOBALEMU_PARAMETER_RANGES[param][1]

        # Get prior the type
        prior_type = prior_info.pop('type')
        prior_generator = _get_prior_generator(prior_type)

        # Generate prior sampler
        prior_sampler = prior_generator(**prior_info)
        individual_priors.append(prior_sampler)

    # Combine priors
    return individual_priors


def create_foreground_prior_samplers(config_dict: dict
                                     ) -> Collection[Callable]:
    """Create a prior samplers over the foreground parameters.

    Parameters
    ----------
    config_dict : dict
        Dictionary containing the configuration parameters.

    Returns
    -------
    individual_priors : Collection[Callable]
        For each parameter, a function that takes a number of samples and
        returns the sampled values as an array.
    """
    # Loop over parameters constructing individual priors
    individual_priors = []
    for param in config_dict['priors']['foregrounds'].keys():
        # Get prior info
        prior_info = deepcopy(config_dict['priors']['foregrounds'][param])

        # Get prior the type
        prior_type = prior_info.pop('type')
        prior_generator = _get_prior_generator(prior_type)

        # Generate prior sampler
        prior_sampler = prior_generator(**prior_info)
        individual_priors.append(prior_sampler)

    # Combine priors
    return individual_priors


# Assemble simulators
def assemble_simulators(
        config_dict: dict,
        sigma_noise: float
) -> Tuple[Simulator, Simulator]:
    """Assemble the simulator functions for the Evidence Network.

    Parameters
    ----------
    config_dict : dict
        Dictionary containing the configuration parameters.
    sigma_noise : float
        The noise sigma in K.

    Returns
    -------
    no_signal_simulator : Simulator
        Function that generates data from noise + foreground model.
    with_signal_simulator : Simulator
        Function that generates data from noise + foreground + signal model.
    """
    # Set-up globalemu
    globalemu_priors = create_globalemu_prior_samplers(config_dict)
    globalemu_redshifts = global_signal_experiment_measurement_redshifts(
        config_dict['frequency_resolution'])
    globalemu_predictor = load_globalemu_emulator(globalemu_redshifts)

    # Set-up foreground model
    foreground_priors = create_foreground_prior_samplers(config_dict)

    # Build no signal simulator
    noise_simulator = generate_white_noise_simulator(
        len(globalemu_redshifts), sigma_noise)
    foreground_simulator = generate_foreground_simulator(
        globalemu_redshifts, *foreground_priors)
    no_signal_simulator = additive_simulator_combiner(noise_simulator,
                                                      foreground_simulator)

    # Build with signal simulator
    noise_simulator = generate_white_noise_simulator(
        len(globalemu_redshifts), sigma_noise)
    foreground_simulator = generate_foreground_simulator(
        globalemu_redshifts, *foreground_priors)
    signal_simulator = generate_global_signal_simulator(
        globalemu_predictor, *globalemu_priors)
    with_signal_simulator = additive_simulator_combiner(
        noise_simulator, foreground_simulator, signal_simulator
    )

    return no_signal_simulator, with_signal_simulator
