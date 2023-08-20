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
import numpy as np
from evidence_networks import EvidenceNetwork
from fbf_utilities import load_configuration_dict, get_noise_sigma, \
    assemble_simulators, timing_filename, add_timing_data
import os
import matplotlib.pyplot as plt
import time


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
    en.train(epochs=30, roll_back=True)
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
    _ = en.blind_coverage_test(plotting_ax=ax, num_validation_samples=100_000)
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
