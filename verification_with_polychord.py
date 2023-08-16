"""Verify accuracy of evidence network with Polychord.

This script evaluates the Bayes ratio for a range of mock data sets
using Polychord. These values are then compared to the values previously
computed by the evidence network. The script outputs summary statistics
as well as plots of the Bayes ratio values.

This script is intended to be run after train_evidence_network.py has
been run to train the evidence network. If not run, the script will
fail.

It is recommended to run this script on a CPU since PolyChord does not
derive any benefit from GPUs.

The script can take an optional command line argument to specify the
noise sigma in K. The default is 0.079 K. This indicates the evidence
network to use for the comparison.
"""

# Required imports
from __future__ import annotations
from typing import Callable, Tuple
from train_evidence_network import get_noise_sigma, load_configuration_dict, \
    timing_filename, add_timing_data
from simulators.twenty_one_cm import load_globalemu_emulator, \
    global_signal_experiment_measurement_redshifts, GLOBALEMU_INPUTS, \
    GLOBALEMU_PARAMETER_RANGES
import os
import shutil
from mpi4py import MPI
import numpy as np
from pypolychord import PolyChordSettings, run_polychord
from pypolychord.priors import UniformPrior, GaussianPrior, LogUniformPrior
from copy import deepcopy
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import time
from math import erf

# Parameters
CHAIN_DIR = "chains"


# Prior, likelihood, and evidence functions
def noise_only_log_evidence(data: np.ndarray, sigma_noise: float) -> float:
    """Evaluate the log evidence for a noise only model.

    Parameters
    ----------
    data : np.ndarray
        The mock data to evaluate the log evidence for.
    sigma_noise : float
        The noise sigma in K.

    Returns
    -------
    log_evidence : float
        The log evidence for the noise only model.
    """
    num_data_points = data.size
    log_evidence = -0.5*num_data_points*np.log(2*np.pi*sigma_noise**2) \
        - 0.5 * np.sum(data**2) / sigma_noise**2
    return log_evidence


def generate_noisy_signal_loglikelihood(data: np.ndarray,
                                        sigma_noise: float,
                                        globalemu_emulator: Callable
                                        ) -> Callable:
    """Generate a loglikelihood function for a noisy signal model.

    Parameters
    ----------
    data : np.ndarray
        The mock data to evaluate the loglikelihood for.
    sigma_noise : float
        The noise sigma in K.
    globalemu_emulator : Callable
        The emulator for the global signal.

    Returns
    -------
    loglikelihood : Callable
        The loglikelihood function for the noisy signal model.
    """
    def loglikelihood(theta: np.ndarray) -> Tuple[float, list]:
        """Evaluate the loglikelihood for a noisy signal model."""
        global_signal_mk, _ = globalemu_emulator(theta)
        global_signal_k = global_signal_mk / 1000
        num_data_points = data.size
        log_evidence = -0.5*num_data_points*np.log(2*np.pi*sigma_noise**2) \
                       - 0.5*np.sum((data-global_signal_k)**2)/sigma_noise**2
        return log_evidence, []
    return loglikelihood


class TruncatedGaussianPrior:
    """Truncated Gaussian prior.

    This prior is a Gaussian distribution with mean mu and standard deviation
    sigma truncated to the range [low, high].
    """

    def __init__(self, mu, sigma, low, high):
        """Initialize the truncated Gaussian prior.

        Parameters
        ----------
        mu : float
            The mean of the would have been Gaussian prior.
        sigma : float
            The standard deviation of the would have been Gaussian prior.
        low : float
            The lower bound of the truncated Gaussian prior.
        high : float
            The upper bound of the truncated Gaussian prior.
        """
        self.mu = mu
        self.sigma = sigma
        self.low = low
        self.high = high

        # Define utility variables
        self.alpha = (self.low - self.mu) / self.sigma
        self.beta = (self.high - self.mu) / self.sigma

    def __call__(self, x: np.ndarray | float) -> np.ndarray | float:
        """Sample the prior.

        Parameters
        ----------
        x : np.ndarray or float
            Sample from a unit hypercube.

        Returns
        -------
        prior : np.ndarray or float
            The equivalent sampled parameter value(s) in the original parameter
            space.
        """
        scaled_values = truncnorm.ppf(x, self.alpha, self.beta)
        return self.mu + self.sigma * scaled_values


def generate_prior(config_dict: dict) -> Callable:
    """Generate a prior function for the global signal model.

    Parameters
    ----------
    config_dict : dict
        The configuration dictionary for the pipeline.

    Returns
    -------
    prior : Callable
        The prior callable for the global signal model.
    """
    # Loop over parameters constructing individual prior objects
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
            param_prior = UniformPrior(prior_info['low'], prior_info['high'])
        elif prior_type == 'log_uniform':
            param_prior = LogUniformPrior(prior_info['low'],
                                          prior_info['high'])
        elif prior_type == 'gaussian':
            param_prior = GaussianPrior(prior_info['mean'], prior_info['std'])
        elif prior_type == 'truncated_gaussian':
            param_prior = TruncatedGaussianPrior(prior_info['mean'],
                                                 prior_info['std'],
                                                 prior_info['low'],
                                                 prior_info['high'])
        else:
            raise ValueError("Unknown prior type.")

        # Add to list of priors
        individual_priors.append(param_prior)

    # Define prior function
    def prior(x: np.ndarray) -> np.ndarray:
        """Sample the prior.

        Parameters
        ----------
        x : np.ndarray
            Sample from a unit hypercube.

        Returns
        -------
        theta : np.ndarray
            The equivalent sampled parameter values in the original parameter
            space.
        """
        theta = np.zeros_like(x)
        for idx, sample_transform in enumerate(individual_priors):
            theta[idx] = sample_transform(x[idx])
        return theta
    return prior


def main():
    """Verify accuracy of evidence network with Polychord."""
    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Get noise sigma and configuration data
    sigma_noise = get_noise_sigma()
    config_dict = load_configuration_dict()
    timing_file = timing_filename(sigma_noise)

    # Load verification data
    verification_data_file = (
        os.path.join('verification_data',
                     f'noise_{sigma_noise:.4f}_verification_data.npz'))
    verification_data_file_contents = np.load(verification_data_file)
    v_data = verification_data_file_contents['data']
    en_log_bayes_ratios = verification_data_file_contents['log_bayes_ratios']

    # Set up global emu
    globalemu_redshifts = global_signal_experiment_measurement_redshifts(
        config_dict['frequency_resolution'])
    globalemu_predictor = load_globalemu_emulator(globalemu_redshifts)

    # Generate priors
    prior = generate_prior(config_dict)

    if rank == 0:
        # Make sure chains directory exists
        os.makedirs(CHAIN_DIR, exist_ok=True)

        # Loop over data using Polychord to evaluate the evidence
        pc_log_bayes_ratios = []
        pc_nlike = []

    settings = None
    start = time.time()
    for data in v_data:
        # Can find noise only evidence analytically
        log_z_noise_only = noise_only_log_evidence(data, sigma_noise)

        # Use Polychord to find evidence for noise + global signal
        loglikelihood = generate_noisy_signal_loglikelihood(
            data, sigma_noise, globalemu_predictor
        )

        # Set Polychord properties
        n_dims = len(GLOBALEMU_INPUTS)
        n_derived = 0
        settings = PolyChordSettings(n_dims, n_derived)
        settings.nlive = 25 * n_dims  # As recommended
        settings.base_dir = os.path.join(CHAIN_DIR, f'noise_{sigma_noise:.4f}')
        settings.file_root = f'noise_{sigma_noise:.4f}'
        settings.do_clustering = True
        settings.read_resume = False

        # Clear out base directory ready for the run
        if rank == 0:
            try:
                shutil.rmtree(settings.base_dir)
            except OSError:
                pass
            try:
                os.mkdir(settings.base_dir)
            except OSError:
                pass

        # Run polychord
        comm.Barrier()
        output = run_polychord(loglikelihood, n_dims,
                               n_derived, settings, prior)

        # Compute log bayes ratio
        if rank == 0:
            log_z_noisy_signal = output.logZ

            # Compute log bayes ratio
            log_bayes_ratio = log_z_noisy_signal - log_z_noise_only
            pc_log_bayes_ratios.append(log_bayes_ratio)
            pc_nlike.append(output.nlike)
        comm.Barrier()

    # Clean up now finished
    if rank != 0:
        return

    # Record timing data
    end = time.time()
    add_timing_data(timing_file, 'total_polychord_log_k',
                    end - start)
    add_timing_data(timing_file, 'average_polychord_log_k',
                    (end - start) / v_data.shape[0])

    try:
        shutil.rmtree(settings.base_dir)
    except OSError:
        pass

    # Save PolyChord log bayes ratios if needed for later comparison
    pc_log_bayes_ratios = np.squeeze(np.array(pc_log_bayes_ratios))
    polychord_data_file = (
        os.path.join('verification_data',
                     f'noise_{sigma_noise:.4f}_polychord_log_k.npz'))
    np.savez(polychord_data_file, log_bayes_ratios=pc_log_bayes_ratios)

    # Create output directory for results of comparison
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
                               f"{np.sqrt(np.mean(error**2)):.4f}\n")

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
    ax.scatter(en_log_bayes_ratios, pc_log_bayes_ratios, c='C0')
    min_log_z = np.min([np.min(en_log_bayes_ratios),
                        np.min(pc_log_bayes_ratios)])
    max_log_z = np.max([np.max(en_log_bayes_ratios),
                        np.max(pc_log_bayes_ratios)])
    ax.plot([min_log_z, max_log_z], [min_log_z, max_log_z], c='k', ls='--')
    ax.set_xlabel(r'$\log K_{\rm EN}$')
    ax.set_ylabel(r'$\log K_{\rm PolyChord}$')
    ax.set_xlim(-15, 30)
    ax.set_ylim(-15, 30)

    # Add lines at 2, 3 and 5 sigma to guide the eye as to where we need
    # the network to be accurate
    for detection_sigma in [2, 3, 5]:
        probability = (1 + erf(detection_sigma / np.sqrt(2)) - 1)
        inv_probability = 1 - probability
        detection_threshold = np.log(probability / inv_probability)
        ax.axvline(detection_threshold, ls=':',
                   zorder=-1, label=rf'{detection_sigma}$\sigma$')
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
