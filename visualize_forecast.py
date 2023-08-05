"""Run and Visualize Fully Bayesian Forecast.

This script runs a fully Bayesian forecast for the detectability of the
global 21-cm signal and visualizes the results. The evidence network it
uses needs to be trained and saved first using
`train_evidence_network.py` or this script with raise an error.

If using this script it is recommended to run on a GPU for speed.
Plus some CPUs will not have enough memory for the mock data.

The script can take an optional command line argument to specify the
noise sigma in K. The default is 0.025 K. Each different noise level
will require a different network to be trained.
"""

# Required imports
from evidence_networks import EvidenceNetwork
from train_evidence_network import get_noise_sigma, load_configuration_dict, \
    assemble_simulators
import os
import numpy as np


def main():
    """Run and visualize fully Bayesian forecast."""
    # IO
    sigma_noise = get_noise_sigma()
    config_dict = load_configuration_dict()

    # Set up simulators
    noise_only_simulator, noisy_signal_simulator = assemble_simulators(
        config_dict, sigma_noise)

    # Load evidence network
    en = EvidenceNetwork(noise_only_simulator, noisy_signal_simulator)
    network_folder = os.path.join("models", f'en_noise_{sigma_noise:.4f}')
    network_file = os.path.join(network_folder, "global_signal_en.h5")
    en.load(network_file)

    # Generate mock data for forecast and evaluate log Bayes ratio
    num_data_sets = config_dict["br_evaluations_for_forecast"]
    mock_data_w_signal, signal_params = \
        noisy_signal_simulator(num_data_sets)
    log_bayes_ratios = en.evaluate_log_bayes_ratio(mock_data_w_signal)
    print(np.mean(log_bayes_ratios))


if __name__ == "__main__":
    main()
