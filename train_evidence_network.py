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
noise sigma in K. The default is 0.015 K.
"""

# Required imports
import numpy as np
from evidence_networks import EvidenceNetwork
from fbf_utilities import load_configuration_dict, \
    assemble_simulators, \
    generate_preprocessing_function, NOISE_DEFAULT
import os
import matplotlib.pyplot as plt
from typing import Tuple
import keras
from keras import layers
from math import erf
import argparse

# Parameters
EN_ALPHA = 2.0


def get_inputs():
    """Get the inputs from the command line arguments.

    Returns
    -------
    run_id : int
        The run id for the training run.
    noise_sigma : float
        The noise sigma in K.
    """
    parser = argparse.ArgumentParser(
        description="Forecast with Evidence Network."
    )
    parser.add_argument(
        "run_id",
        type=int,
        help="The run id for the training run."
    )

    parser.add_argument(
        "noise",
        type=float,
        default=NOISE_DEFAULT,
        help="The noise sigma in K.",
        nargs='?'
    )
    args = parser.parse_args()
    return args.run_id, args.noise


def get_settings(run_id: int) -> Tuple:
    """Get network settings to use for this test.

    Parameters
    ----------
    run_id : int
        The run id for the training run.

    Returns
    -------
    settings : Tuple
        A tuple of the network settings to use for this test.
    """
    # Defaults
    epochs = 500
    training_size = 2_000_000
    initial_learning_rate = 1e-3
    decay_steps = 100_000
    batch_size = 8092
    for_network_width = 256
    back_network_width = 64
    additional_for_layers = 0
    additional_back_layers = 2
    whitening_transform = 'Cholesky'
    whitening_number = 100_000
    alpha = 2.0

    # This particular run
    if run_id == 0:
        pass
    elif run_id == 1:
        epochs = 250
    elif run_id == 2:
        epochs = 1000
    elif run_id == 3:
        epochs = 2000
    elif run_id == 4:
        batch_size = 4096
    elif run_id == 5:
        batch_size = 16_384
    elif run_id == 6:
        batch_size = 32_768
    elif run_id == 7:
        back_network_width = 32
        additional_back_layers = 1
    elif run_id == 8:
        back_network_width = 64
        additional_back_layers = 1
    elif run_id == 9:
        back_network_width = 128
        additional_back_layers = 1
    elif run_id == 10:
        back_network_width = 32
    elif run_id == 11:
        back_network_width = 128
    elif run_id == 12:
        back_network_width = 32
        additional_back_layers = 3
    elif run_id == 13:
        back_network_width = 64
        additional_back_layers = 3
    elif run_id == 14:
        back_network_width = 128
        additional_back_layers = 3
    else:
        raise ValueError(f"Run id {run_id} not recognised.")

    return (epochs, training_size, initial_learning_rate, decay_steps,
            batch_size, for_network_width, back_network_width,
            additional_for_layers, additional_back_layers, whitening_transform,
            whitening_number, alpha)


def default_nn_model(
        input_size: int,
        for_network_width: int,
        back_network_width: int,
        additional_for_layers: int,
        additional_back_layers: int
) -> keras.Model:
    """Return a neural network model.

    This is a modified version of the model from the appendix of
    arXiv:2305.11241.

    Parameters
    ----------
    input_size: int
        The number of input features
    for_network_width: int
        The width of the for-network
    back_network_width: int
        The width of the back-network
    additional_for_layers: int
        The number of additional layers in the for-network
    additional_back_layers: int
        The number of additional layers in the back-network

    Returns
    -------
    keras.Model
        The default neural network model
    """
    inputs = layers.Input(shape=(input_size,))
    x = layers.Dense(for_network_width)(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    for _ in range(additional_for_layers):
        x = layers.Dense(for_network_width)(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
    x = layers.Dense(back_network_width)(x)
    x = layers.LeakyReLU()(x)
    x_batch_norm_1 = layers.BatchNormalization()(x)  # Save for skip
    x = layers.Dense(back_network_width)(x_batch_norm_1)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(back_network_width)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Add()([x, x_batch_norm_1])  # Skip connection
    x = layers.BatchNormalization()(x)
    x = layers.Dense(back_network_width)(x)
    x = layers.LeakyReLU()(x)
    for _ in range(additional_back_layers):
        x = layers.Dense(back_network_width)(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name="jeffrey_wandelt_23_network")
    return model


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
    config_dict = load_configuration_dict()
    run_id, noise_sigma = get_inputs()
    settings = get_settings(run_id)

    # Unpack settings
    (epochs, training_size, initial_learning_rate, decay_steps, batch_size,
     for_network_width, back_network_width, additional_for_layers,
     additional_back_layers, whitening_transform, whitening_number,
     alpha) = settings
    config_dict['whitening_transform'] = whitening_transform
    config_dict['covariance_samples'] = whitening_number

    # Set-up simulators
    no_signal_simulator, signal_simulator = assemble_simulators(
        config_dict, noise_sigma)

    # Generate our preprocessing function
    model_dir = os.path.join("models", f'en_noise_{noise_sigma}')
    data_preprocessing = generate_preprocessing_function(
        config_dict,
        noise_sigma,
        model_dir,
        overwrite=True)

    # Create and train evidence network
    en = EvidenceNetwork(no_signal_simulator,
                         signal_simulator,
                         alpha=alpha,
                         data_preprocessing=data_preprocessing)
    nn_model = default_nn_model(
        en._data_size,
        for_network_width,
        back_network_width,
        additional_for_layers,
        additional_back_layers)
    en.train(epochs=epochs,
             train_data_samples_per_model=training_size,
             validation_data_samples_per_model=int(training_size*0.4),
             initial_learning_rate=initial_learning_rate,
             decay_steps=decay_steps,
             batch_size=batch_size,
             roll_back=True,
             nn_model=nn_model)

    # Save the network
    os.makedirs(model_dir, exist_ok=True)
    network_file = os.path.join(
        model_dir,
        f"global_signal_en_noise_{noise_sigma}.h5")
    en.save(network_file)

    # Perform blind coverage test
    plt.style.use(os.path.join('figures_and_results', 'mnras_single.mplstyle'))
    fig, ax = plt.subplots()
    _ = en.blind_coverage_test(plotting_ax=ax,
                               num_validation_samples=int(training_size*0.4))
    figure_folder = os.path.join('figures_and_results', 'blind_coverage_tests')
    os.makedirs(figure_folder, exist_ok=True)
    fig.savefig(os.path.join(
        figure_folder,
        f'en_blind_coverage_noise_{noise_sigma}_{run_id}.pdf'))

    # Load verification data
    verification_data_file = os.path.join(
        'verification_data',
        f'verification_data_noise_{noise_sigma}.npz')
    verification_file_contents = np.load(verification_data_file)
    pc_log_bayes_ratios = verification_file_contents['log_bayes_ratios']
    v_data = verification_file_contents['data']
    v_labels = verification_file_contents['labels']

    # Evaluate network on verification data
    en_log_bayes_ratios = np.squeeze(en.evaluate_log_bayes_ratio(v_data))

    # In case useful save the log bayes ratios computed by the network
    en_bayes_ratio_file = os.path.join(
        'verification_data',
        f'en_log_k_noise_{noise_sigma}_{run_id}.npz')
    np.savez(en_bayes_ratio_file, log_bayes_ratios=en_log_bayes_ratios)

    # Create output directory for results of verification comparison
    os.makedirs(os.path.join("figures_and_results",
                             "polychord_verification"), exist_ok=True)
    numeric_results_filename = os.path.join(
        "figures_and_results",
        "polychord_verification",
        f"polychord_verification_results_noise_{noise_sigma}_{run_id}.txt")
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
            f"{pc_percent_detected:.2f}% of signals were detectable "
            f"according to Polychord\n")
        numeric_results_file.write(
            f"{en_percent_detected:.2f}% of signals were detectable "
            f"according to the network\n")
        numeric_results_file.write(
            f"{percent_difference:.2f}% difference in detection rate\n")
        numeric_results_file.write(
            f"{percent_changed:.2f}% of signals changed detection status\n")
        numeric_results_file.write("\n")

    # Network loss
    test_loss = en.calculate_testing_loss(
        int(training_size*0.4), batch_size)
    numeric_results_file.write(f"Network testing loss: {test_loss:.4f}\n")
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
        f"polychord_verification_noise_{noise_sigma}_{run_id}.pdf")
    fig.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    main()
