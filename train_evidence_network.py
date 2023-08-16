"""Train Evidence Network.

Script to train the example Evidence Network for the paper. The created
Evidence Network is trained to predict the Bayes ratio between a model
with a noisy 21-cm signal and a model with only noise. The network is
saved to the `models` directory after training is complete.

If using this script it is recommended to train on a GPU for speed.
Plus some CPUs will not have enough memory to train the network.

The script can take an optional command line argument to specify the
noise sigma in K. The default is 0.079 K.
"""

# Required imports
from typing import Callable, Collection, Tuple
import argparse
import numpy as np
from simulators import additive_simulator_combiner, Simulator
from simulators.noise import generate_white_noise_simulator
from simulators.twenty_one_cm import load_globalemu_emulator, \
    GLOBALEMU_INPUTS, GLOBALEMU_PARAMETER_RANGES, \
    global_signal_experiment_measurement_redshifts, \
    generate_global_signal_simulator
from priors import generate_uniform_prior_sampler, \
    generate_log_uniform_prior_sampler, \
    generate_gaussian_prior_sampler, \
    generate_truncated_gaussian_prior_sampler
from evidence_networks import EvidenceNetwork
from copy import deepcopy
import yaml
import os
import matplotlib.pyplot as plt
import pickle as pkl
import time


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
    noise_only_simulator : Simulator
        Function that generates data from noise only model.
    noisy_signal_simulator : Simulator
        Function that generates data from noise + signal model.
    """
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

    return noise_only_simulator, noisy_signal_simulator


def main():
    """Train the Evidence Network."""
    # IO
    sigma_noise = get_noise_sigma()
    config_dict = load_configuration_dict()
    timing_file = timing_filename(sigma_noise)

    # Set-up simulators
    start = time.time()
    noise_only_simulator, noisy_signal_simulator = assemble_simulators(
        config_dict, sigma_noise)
    end = time.time()
    add_timing_data(timing_file, 'simulator_assembly', end - start)

    # Create and train evidence network
    start = time.time()
    en = EvidenceNetwork(noise_only_simulator, noisy_signal_simulator)
    en.train()
    end = time.time()
    add_timing_data(timing_file, 'network_training', end - start)

    # Save the network
    network_folder = os.path.join("models", f'en_noise_{sigma_noise:.4f}')
    os.makedirs(network_folder, exist_ok=True)
    network_file = os.path.join(network_folder, "global_signal_en.h5")
    en.save(network_file)

    # Perform blind coverage test
    start = time.time()
    plt.style.use(os.path.join('figures_and_results', 'mnras_single.mplstyle'))
    fig, ax = plt.subplots()
    _ = en.blind_coverage_test(plotting_ax=ax, num_validation_samples=10_000)
    figure_folder = os.path.join('figures_and_results', 'blind_coverage_tests')
    os.makedirs(figure_folder, exist_ok=True)
    fig.savefig(os.path.join(figure_folder,
                             f'en_noise_{sigma_noise:.4f}_blind_coverage.pdf'))
    end = time.time()
    add_timing_data(timing_file, 'bct', end - start)

    # Verification evaluations for comparison with other methods
    verification_ds_per_model = config_dict['verification_data_sets_per_model']
    data, labels = en.get_simulated_data(verification_ds_per_model)
    log_bayes_ratios = en.evaluate_log_bayes_ratio(data)
    os.makedirs('verification_data', exist_ok=True)
    np.savez(os.path.join('verification_data',
                          f'noise_{sigma_noise:.4f}_verification_data.npz'),
             data=data, labels=labels, log_bayes_ratios=log_bayes_ratios)

    # Verification evaluations for comparison with other methods
    verification_ds_per_model = config_dict['verification_data_sets_per_model']
    data, labels = en.get_simulated_data(verification_ds_per_model)
    log_bayes_ratios = en.evaluate_log_bayes_ratio(data)
    os.makedirs('verification_data', exist_ok=True)
    np.savez(os.path.join('verification_data',
                          f'noise_{sigma_noise:.4f}_verification_data.npz'),
             data=np.squeeze(data),
             labels=np.squeeze(labels),
             log_bayes_ratios=np.squeeze(log_bayes_ratios))


if __name__ == "__main__":
    main()
