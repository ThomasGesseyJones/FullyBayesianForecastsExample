"""Train Evidence Network.

Script to train the Evidence Networks for the paper. The created
Evidence Network is trained to predict the Bayes ratio between a model
with a 21-cm signal and a model with only noise + foreground. The network is
saved to the `models` directory after training is complete. After training
the network is tested using the blind coverage test and the results are
saved to the `figures_and_results` directory. The network is also evaluated
on precomputed verification data sets, for which Polychord results are
available, and the comparison is plot and saved to the
`figures_and_results` directory.

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
from math import erf

# Parameters
EN_ALPHA = 2.0


def sigma_to_log_k(sigma: float) -> float:
    """Convert statistical significance in sigma to log Bayes ratio.

    Parameters
    ----------
    sigma : float
        The statistical significance in sigma.

    Returns
    -------
    log_k : float
        The equivalent log Bayes ratio.
    """
    probability = (1 + erf(sigma / np.sqrt(2)) - 1)
    inv_probability = 1 - probability
    log_k = np.log(probability / inv_probability)
    return log_k


def main():
    """Train the Evidence Network."""
    # IO
    sigma_noise = get_noise_sigma()
    config_dict = load_configuration_dict()
    timing_file = timing_filename(sigma_noise)

    # Set-up simulators
    start = time.time()
    no_signal_simulator, signal_simulator = assemble_simulators(
        config_dict, sigma_noise)
    end = time.time()
    add_timing_data(timing_file, 'simulator_assembly', end - start)

    # Create and train evidence network
    start = time.time()
    en = EvidenceNetwork(no_signal_simulator, signal_simulator,
                         alpha=EN_ALPHA)
    en.train(epochs=20,
             train_data_samples_per_model=2_000_000,
             initial_learning_rate=2e-4,
             decay_steps=100_000,
             batch_size=1000,
             roll_back=True)
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

    # Load verification data
    verification_data_file = os.path.join(
        'verification_data', f'noise_{sigma_noise:.4f}_verification_data.npz')
    verification_file_contents = np.load(verification_data_file)
    pc_log_bayes_ratios = verification_file_contents['log_bayes_ratios']
    v_data = verification_file_contents['data']
    v_labels = verification_file_contents['labels']
    pc_nlike = verification_file_contents['nlike']

    # Evaluate network on verification data
    start = time.time()
    en_log_bayes_ratios = np.squeeze(en.evaluate_log_bayes_ratio(v_data))
    end = time.time()
    add_timing_data(timing_file, 'verification_evaluations',
                    end - start)

    # In case useful save the log bayes ratios computed by the network
    en_bayes_ratio_file = os.path.join(
        'verification_data', f'noise_{sigma_noise:.4f}_en_log_k.npz')
    np.savez(en_bayes_ratio_file, log_bayes_ratios=en_log_bayes_ratios)

    # Create output directory for results of verification comparison
    os.makedirs(os.path.join("figures_and_results",
                             "polychord_verification"), exist_ok=True)
    numeric_results_filename = os.path.join(
        "figures_and_results",
        "polychord_verification",
        f"polychord_verification_"
        f"en_noise_{sigma_noise:.4f}_K_results.txt")
    numeric_results_file = open(numeric_results_filename, 'w')

    # Print results
    numeric_results_file.write('Polychord Verification Results\n')
    numeric_results_file.write('-----------------------------\n\n')

    # Mean difference and rmse error in log Z
    error = en_log_bayes_ratios - pc_log_bayes_ratios
    numeric_results_file.write(f"Mean log K error: "
                               f"{np.mean(error):.4f}\n")
    numeric_results_file.write(f"RMSE in log K: "
                               f"{np.sqrt(np.mean(error ** 2)):.4f}\n")
    numeric_results_file.write("\n")

    # Detection changes
    en_log_bayes_w_signal = en_log_bayes_ratios[v_labels == 1]
    pc_log_bayes_w_signal = pc_log_bayes_ratios[v_labels == 1]
    for detection_sigma in [2, 3, 5]:
        detection_threshold = sigma_to_log_k(detection_sigma)
        en_detected = en_log_bayes_w_signal > detection_threshold
        pc_detected = pc_log_bayes_w_signal > detection_threshold
        en_percent_detected = np.mean(en_detected) * 100
        pc_percent_detected = np.mean(pc_detected) * 100
        percent_difference = en_percent_detected - pc_percent_detected
        percent_changed = np.mean(en_detected != pc_detected) * 100

        numeric_results_file.write(
            f"{detection_sigma} sigma detection statistics:\n")
        numeric_results_file.write(
            f"{en_percent_detected:.2f}% of signals were detectable "
            f"according to Polychord\n")
        numeric_results_file.write(
            f"{pc_percent_detected:.2f}% of signals were detectable "
            f"according to the network\n")
        numeric_results_file.write(
            f"{percent_difference:.2f}% difference in detection rate\n")
        numeric_results_file.write(
            f"{percent_changed:.2f}% of signals changed detection status\n")
        numeric_results_file.write("\n")

    # And mean and total number of live point
    pc_nlike = np.array(pc_nlike)
    numeric_results_file.write(f"Mean number of likelihood evaluations: "
                               f"{np.mean(pc_nlike):.4f}\n")
    numeric_results_file.write(f"Total number of likelihood evaluations: "
                               f"{np.sum(pc_nlike):.4f}\n")
    numeric_results_file.close()

    # Plot results
    plt.style.use(os.path.join('figures_and_results', 'mnras_single.mplstyle'))
    fig, ax = plt.subplots()
    ax.scatter(en_log_bayes_ratios[v_labels == 0],
               pc_log_bayes_ratios[v_labels == 0],
               c='C0', label='No signal',
               s=2, marker='x', zorder=1, alpha=0.8)
    ax.scatter(en_log_bayes_ratios[v_labels == 1],
               pc_log_bayes_ratios[v_labels == 1],
               c='C1', label='With Signal',
               s=2, marker='x', zorder=1, alpha=0.8)
    min_log_z = np.min([np.min(en_log_bayes_ratios),
                        np.min(pc_log_bayes_ratios)])
    max_log_z = np.max([np.max(en_log_bayes_ratios),
                        np.max(pc_log_bayes_ratios)])
    ax.plot([min_log_z, max_log_z], [min_log_z, max_log_z], c='k', ls='--',
            zorder=0)
    ax.set_xlabel(r'$\log K_{\rm EN}$')
    ax.set_ylabel(r'$\log K_{\rm PolyChord}$')
    ax.set_xlim(-15, 30)
    ax.set_ylim(-15, 30)

    # Add lines at 2, 3 and 5 sigma to guide the eye as to where we need
    # the network to be accurate, coloured using default matplotlib
    # colour cycle skipping the first colour
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color'][1:]
    for detection_sigma, c in zip([2, 3, 5], colors):
        detection_threshold = sigma_to_log_k(detection_sigma)
        ax.axvline(detection_threshold, ls=':',
                   zorder=-1, label=rf'{detection_sigma}$\sigma$',
                   c=c)
        ax.axhline(detection_threshold, ls=':', zorder=-1, c=c)
    ax.legend()

    # Save figure
    fig.tight_layout()
    filename = os.path.join(
        "figures_and_results",
        "polychord_verification",
        f"polychord_verification_"
        f"en_noise_{sigma_noise:.4f}_K.pdf")
    fig.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    main()
