"""Train Evidence Network.

Script to train the example Evidence Network for the paper. The created
Evidence Network is trained to predict the Bayes ratio between a model
with a noisy 21-cm signal and a model with only noise. The network is
saved to the `models` directory after training is complete.

If using this script it is recommended to train on a GPU for speed.
Plus some CPUs will not have enough memory to train the network.

The script can take an optional command line argument to specify the
noise sigma in K. The default is 0.025 K.
"""

# Required imports
from typing import Callable, Collection
import argparse
from simulators import additive_simulator_combiner
from simulators.noise import generate_white_noise_simulator
from simulators.twenty_one_cm import load_globalemu_emulator, \
    GLOBALEMU_INPUTS, GLOBALEMU_PARAMETER_RANGES, \
    global_signal_experiment_measurement_redshifts, \
    generate_global_signal_simulator
from priors import generate_uniform_prior_sampler, \
    generate_log_uniform_prior_sampler,\
    generate_gaussian_prior_sampler, \
    generate_truncated_gaussian_prior_sampler
from evidence_networks import EvidenceNetwork
from copy import deepcopy
import yaml
import os


# Parameters
NOISE_DEFAULT = 0.025  # K


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
        help="The noise sigma in K."
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
    for param in GLOBALEMU_INPUTS:
        # Get prior info
        prior_info = deepcopy(config_dict['priors'][param])

        # Replace emu_min and emu_max with the min and max value globalemu
        # can take for this parameter
        for k, v in prior_info.items():
            if v == 'emu_min':
                prior_info[k] = GLOBALEMU_PARAMETER_RANGES[param][0]
            elif v == 'emu_max':
                prior_info[k] = GLOBALEMU_PARAMETER_RANGES[param][1]

        # Get prior type
        prior_type = prior_info.pop('type')
        if prior_type == 'uniform':
            prior_generator = generate_uniform_prior_sampler
        elif prior_type == 'log_uniform':
            prior_generator = generate_log_uniform_prior_sampler
        elif prior_type == 'gaussian':
            prior_generator = generate_gaussian_prior_sampler
        elif prior_type == 'truncated_gaussian':
            prior_generator = generate_truncated_gaussian_prior_sampler
        else:
            raise ValueError("Unknown prior type.")

        # Generate prior sampler
        prior_sampler = prior_generator(**prior_info)
        individual_priors.append(prior_sampler)

    # Combine priors
    return individual_priors


def main():
    """Train the Evidence Network."""
    # IO
    sigma_noise = get_noise_sigma()
    config_dict = load_configuration_dict()

    # Set-up globalemu
    globalemu_priors = create_globalemu_prior_samplers(config_dict)
    globalemu_redshifts = global_signal_experiment_measurement_redshifts(
        config_dict['frequency_resolution'])
    globalemu_predictor = load_globalemu_emulator(globalemu_redshifts)

    # Build simulators
    noise_only_simulator = generate_white_noise_simulator(
        len(globalemu_redshifts), sigma_noise)
    noise_for_signal = generate_white_noise_simulator(
        len(globalemu_redshifts), sigma_noise)
    signal_simulator = generate_global_signal_simulator(
        globalemu_predictor, *globalemu_priors)
    noisy_signal_simulator = additive_simulator_combiner(
        signal_simulator, noise_for_signal)

    # Create and train evidence network
    en = EvidenceNetwork(noise_only_simulator, noisy_signal_simulator)
    en.train()

    # Save the network
    network_folder = os.path.join("models", f'en_noise_{sigma_noise:.4f}')
    os.makedirs(network_folder, exist_ok=True)
    network_file = os.path.join(network_folder, "global_signal_en.h5")
    en.save(network_file)


if __name__ == "__main__":
    main()
