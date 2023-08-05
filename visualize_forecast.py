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
from __future__ import annotations
from typing import Collection
from evidence_networks import EvidenceNetwork
from train_evidence_network import get_noise_sigma, load_configuration_dict, \
    assemble_simulators
import os
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from math import erf


# Parameters
default_parameter_labels = {
    "f_star": r"$f_{*}$",
    "v_c": r"$V_{\rm c}$",
    "f_x": r"$f_{\rm x}$",
    "tau": r"$\tau$",
    "alpha": r"$\alpha$",
    "nu_min": r"$\nu_{\rm min}$",
    "R_mfp": r"$R_{\rm mfp}$"}


# Plotting functions
def detectability_corner_plot(
        log_bayes_ratios: np.ndarray,
        detection_threshold: float | str,
        parameter_values: DataFrame,
        parameters_to_plot: Collection[str] = None,
        parameter_labels: dict[str, str] = None,
        parameters_to_log: Collection[str] = None,
        line_kwargs: dict = None,
        pcolormesh_kwargs: dict = None,
) -> plt.Figure:
    """Plot a fully Bayesian forecast of the detectability of a signal.

    Parameters
    ----------
    log_bayes_ratios : ndarray
        The log Bayes ratios between the model with a signal and the
        model without a signal for a number of data sets generated
        from the model with a signal.
    detection_threshold : float | str
        The detection threshold for the log bayes ratio. We conclude a
        detection is made if the log bayes ratio is greater than this value.
        Alternatively, can be given as a string for some common values:
            'X sigma' for X sigma detection threshold
    parameter_values : DataFrame
        Parameters to plot conditional detectability for. Can either be
        input parameters or derived parameters.
    parameters_to_plot : Collection[str], optional
         Which of the parameters in `parameter_values` to plot. If None,
         all parameters are plotted.
    parameter_labels : dict[str, str], optional
        Labels for the plotted parameters. If one is not given
        the default is to the parameter name from the DataFrame column.
    parameters_to_log : Collection[str], optional
        Which of the parameters that are plotted to plot as the log10 of
        their value. log10 is automatically added to the parameter label.
    line_kwargs : dict, optional
        Keyword arguments to pass to the on digonal line plots.
    pcolormesh_kwargs : dict, optional
        Keyword arguments to pass to the below diagonal pcolormesh plots.

    Returns
    -------
    fig : Figure
        The detectability corner plot.
    """
    # Input sanitization
    if len(log_bayes_ratios) != parameter_values.shape[0]:
        raise ValueError(
            f'log_bayes_ratios and parameter_values must have the same '
            f'length. Given lengths were {len(log_bayes_ratios)} and '
            f'{parameter_values.shape[0]} respectively.')

    # Input processing
    if isinstance(detection_threshold, str):
        if 'sigma' in detection_threshold:
            detection_threshold = detection_threshold.replace('sigma', '')
            detection_threshold = detection_threshold.strip()
            detection_sigma = float(detection_threshold)

            probability = (1 + erf(detection_sigma / np.sqrt(2)) - 1)
            inv_probability = 1 - probability
            detection_threshold = np.log(probability / inv_probability)
        else:
            raise ValueError(
                f'Invalid detection threshold_string: {detection_threshold}')

    if parameters_to_plot is None:
        parameters_to_plot = parameter_values.columns

    if parameter_labels is None:
        parameter_labels = {}
    labels = [parameter_labels.get(param, param) for
              param in parameters_to_plot]

    # Default line and pcolormesh kwargs
    if line_kwargs is None:
        line_kwargs = {}
    if pcolormesh_kwargs is None:
        pcolormesh_kwargs = {}

    if 'ls' not in line_kwargs:
        line_kwargs['ls'] = '-'
    if 'lw' not in line_kwargs:
        line_kwargs['lw'] = 1

    if 'vmin' not in pcolormesh_kwargs:
        pcolormesh_kwargs['vmin'] = 0
    if 'vmax' not in pcolormesh_kwargs:
        pcolormesh_kwargs['vmax'] = 1
    if 'linewidth' not in pcolormesh_kwargs:
        pcolormesh_kwargs['linewidth'] = 0
    if 'rasterized' not in pcolormesh_kwargs:
        pcolormesh_kwargs['rasterized'] = True
    if 'cmap' not in pcolormesh_kwargs:
        pcolormesh_kwargs['cmap'] = 'inferno'

    # Change to Log parameters
    if parameters_to_log is not None:
        for param in parameters_to_log:
            if param not in parameters_to_plot:
                raise ValueError(
                    f'Parameter {param} is not in parameters_to_plot. '
                    f'parameters_to_log must be a subset of '
                    f'parameters_to_plot.')
            # Update labels
            param_idx = parameters_to_plot.index(param)
            current_label = labels[param_idx]
            stripped_label = current_label.strip('$')
            labels[param_idx] = rf'$\log_{{\rm 10}}({stripped_label})$'

            # Update values
            parameter_values[param] = np.log10(parameter_values[param])

    # Create figure and grid for subplots
    num_params = len(parameters_to_plot)
    fig = plt.figure()
    grid = fig.add_gridspec(num_params, num_params, hspace=0, wspace=0)

    # Create axes and shared axes where appropriate
    axes = grid.subplots(sharex='col')

    # Hide the above diagonal axes
    for row in range(num_params):
        for col in range(row+1, num_params):
            axes[row, col].set_visible(False)

    # Separate Formatting for diagonal axes
    for i in range(num_params):
        ax = axes[i, i]
        ax.set_ylim(0, 1.1)

        # Put axis label on the right
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.yaxis.set_ticks([0.25, 0.5, 0.75, 1.00])
        ax.set_ylabel('P(Detection)')

        ax.tick_params('y', which='both', direction='in', right=True)
        ax.tick_params('x', which='both', direction='inout', top=False,
                       bottom=True)

    # Format off-diagonal axes
    for row in range(num_params):
        for col in range(row):
            ax = axes[row, col]
            ax.tick_params('both', which='both', direction='inout',
                           bottom=True,  left=True, right=True, top=True)

    # Set axis labels
    for row, label in enumerate(labels):
        if row > 0:
            ax = axes[row, 0]
            ax.set_ylabel(label)

        for col in range(1, row):
            ax = axes[row, col]
            # turn off tick labels and axis labels
            ax.set_yticklabels([])
            ax.set_ylabel('')

        ax = axes[-1, row]
        ax.set_xlabel(label)

    # Align said labels
    fig.align_ylabels(axes[:, 0])
    fig.align_xlabels(axes[-1, :])
    fig.subplots_adjust(bottom=0.00)

    # Determine which signals are detectable
    detectable = log_bayes_ratios > detection_threshold

    # Set figure title to the total detection probability
    total_detection_probability = np.mean(detectable)
    fig.suptitle(
        f'Total Detection Probability: {total_detection_probability:.3f}')

    # Plot the off-diagonal
    parameter_resolution = 30
    cmap = None
    for row_idx, row_param in enumerate(parameters_to_plot):
        for col_idx, col_param in enumerate(parameters_to_plot):
            # Only plot the lower triangle
            if row_idx <= col_idx:
                continue

            # Get parameter values
            row_values = parameter_values[row_param].to_numpy()
            col_values = parameter_values[col_param].to_numpy()

            # Get parameter values
            row_bin_edges = np.linspace(
                np.min(row_values), np.max(row_values),
                parameter_resolution + 1)
            row_bin_centers = (row_bin_edges[:-1] + row_bin_edges[1:]) / 2

            col_bin_edges = np.linspace(
                np.min(col_values), np.max(col_values),
                parameter_resolution + 1)
            col_bin_centers = (col_bin_edges[:-1] + col_bin_edges[1:]) / 2

            # Loop over bins determining detection probability
            detection_probability = np.zeros((parameter_resolution,
                                              parameter_resolution))
            for row_bin_idx in range(parameter_resolution):
                for col_bin_idx in range(parameter_resolution):
                    # Determine which data sets are in the bin
                    in_row_bin = np.logical_and(
                        row_values >= row_bin_edges[row_bin_idx],
                        row_values < row_bin_edges[row_bin_idx + 1]
                    )
                    in_col_bin = np.logical_and(
                        col_values >= col_bin_edges[col_bin_idx],
                        col_values < col_bin_edges[col_bin_idx + 1]
                    )
                    in_bin = np.logical_and(in_row_bin, in_col_bin)

                    # and if data set is detectable
                    bin_detectable = detectable[in_bin]

                    # Combine to get detection probability
                    detection_probability[row_bin_idx, col_bin_idx] = np.mean(
                        bin_detectable)

            # Plot
            ax = axes[row_idx, col_idx]
            x_mesh, y_mesh = np.meshgrid(col_bin_centers, row_bin_centers)
            cmap = ax.pcolormesh(x_mesh, y_mesh, detection_probability,
                                 **pcolormesh_kwargs)

    # Plot the diagonal
    for idx, param in enumerate(parameters_to_plot):
        # Get parameter values
        diag_param_values = parameter_values[param].to_numpy()

        # Get bins
        bin_edges = np.linspace(
            np.min(diag_param_values), np.max(diag_param_values),
            parameter_resolution + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Loop over bins determining detection probability
        detection_probability = np.zeros(parameter_resolution)
        for bin_idx in range(parameter_resolution):
            # Determine which data sets are in bin
            in_bin = np.logical_and(
                diag_param_values >= bin_edges[bin_idx],
                diag_param_values < bin_edges[bin_idx + 1])

            # and if data sets are detectable
            bin_detectable = detectable[in_bin]

            # Determine detection probability
            detection_probability[bin_idx] = np.mean(bin_detectable)

        # Plot
        ax = axes[idx, idx]
        ax.plot(bin_centers, detection_probability, **line_kwargs)

        # Add a reference line at 100% and the mean value
        ax.axhline(1, color='k', linestyle='--', linewidth=0.5, zorder=-1,
                   alpha=0.5)
        ax.axhline(total_detection_probability,
                   color=cmap.get_cmap()(total_detection_probability),
                   linestyle='--', linewidth=0.5, zorder=-1, alpha=0.5)

    # Further formatting
    fig.tight_layout()

    # TODO: Add tick placements
    # TODO: Manually enforced ranges

    # Add colorbar at the bottom of the figure
    cbar = fig.colorbar(cmap, ax=axes.ravel().tolist(),
                        orientation='horizontal',
                        location='bottom')
    cbar.ax.set_xlabel("Detection probability")
    cbar.ax.set_ylim(0, 1)

    # Convert colorbar ticks to percentages
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_ticklabels(["{:.0f}%".format(tick * 100) for tick in
                         cbar.get_ticks()])

    # Return the figure handle
    return fig


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

    # Set-up plotting style and variables
    plt.style.use(os.path.join('figures', 'mnras_single.mplstyle'))
    plt.rcParams.update({'ytick.labelsize': 6})
    plt.rcParams.update({'xtick.labelsize': 6})
    plt.rcParams.update({'axes.labelsize': 6})

    detection_thresholds = config_dict["detection_thresholds"]
    if not isinstance(detection_thresholds, list):
        detection_thresholds = [detection_thresholds]
    parameters_to_plot = config_dict["parameters_to_plot"]
    parameters_to_log = config_dict["parameters_to_log"]

    # Plotting
    os.makedirs(os.path.join("figures",
                             "detectability_triangle_plots"), exist_ok=True)
    for detection_threshold in detection_thresholds:
        fig = detectability_corner_plot(
            log_bayes_ratios,
            detection_threshold,
            signal_params,
            parameters_to_plot,
            default_parameter_labels,
            parameters_to_log)
        filename = os.path.join(
            "figures",
            "detectability_triangle_plots",
            f"detectability_triangle_"
            f"{str(detection_threshold).replace(' ', '_')}_"
            f"noise_{sigma_noise:.4f}_K.pdf")
        fig.savefig(filename)
        plt.close(fig)


if __name__ == "__main__":
    main()
