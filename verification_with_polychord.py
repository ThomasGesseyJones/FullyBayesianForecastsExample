"""Bayes Ratios from Polychord.

This script generates a range of mock data sets, and then evaluates the Bayes
ratio between the no signal model and the with signal model using
Polychord. These results are then stored in the verification_data directory
for later comparison with the results from the evidence network.

This script should be run before train_evidence_network.py, and only needs to
be run once for each noise sigma (it is not necessary to rerun this script
if changes are made to the evidence network). The script can be run in
parallel using MPI (which is recommended for speed). It is recommended to run
this script on a CPU since PolyChord does not derive any benefit from GPUs.

The script can take an optional command line argument to specify the
noise sigma in K. The default is 0.079 K. This should be the same as the
noise sigma used later in train_evidence_network.py.
"""

# Required imports
from __future__ import annotations
from typing import Callable, Tuple
from fbf_utilities import get_noise_sigma, load_configuration_dict, \
    timing_filename, add_timing_data, clear_timing_data, assemble_simulators
from simulators.twenty_one_cm import load_globalemu_emulator, \
    global_signal_experiment_measurement_redshifts, GLOBALEMU_INPUTS, \
    GLOBALEMU_PARAMETER_RANGES, foreground_model, FREQ_21CM_MHZ
import os
import shutil
from mpi4py import MPI
import numpy as np
from pypolychord import PolyChordSettings, run_polychord
from pypolychord.priors import UniformPrior, GaussianPrior, LogUniformPrior
from copy import deepcopy
from scipy.stats import truncnorm
import time

# Parameters
CHAIN_DIR = "chains"


# Prior, likelihood, and evidence functions
def generate_loglikelihood(data: np.ndarray,
                           sigma_noise: float,
                           globalemu_emulator: Callable,
                           include_signal: bool = True) -> Callable:
    """Generate a loglikelihood function.

    Parameters
    ----------
    data : np.ndarray
        The mock data to evaluate the loglikelihood for.
    sigma_noise : float
        The noise sigma in K.
    globalemu_emulator : Callable
        The emulator for the global signal.
    include_signal : bool
        Whether to include the signal in the loglikelihood.

    Returns
    -------
    loglikelihood : Callable
        The loglikelihood function for the data model.
    """
    # Get redshifts (and corresponding frequencies) from the global signal
    # emulator
    _, zs = globalemu_emulator(np.ones(len(GLOBALEMU_INPUTS)))
    freqs = FREQ_21CM_MHZ / (1 + zs)

    def loglikelihood(theta: np.ndarray) -> Tuple[float, list]:
        """Evaluate the loglikelihood for a noisy signal model."""
        # Global signal component
        if include_signal:
            global_signal_parameters = theta[:len(GLOBALEMU_INPUTS)]
            foreground_parameters = theta[len(GLOBALEMU_INPUTS):]
            global_signal_mk, _ = globalemu_emulator(global_signal_parameters)
            global_signal_k = global_signal_mk / 1000
        else:
            global_signal_k = np.zeros_like(freqs)
            foreground_parameters = theta

        # Foreground component
        foreground = foreground_model(freqs, foreground_parameters)

        # Model of data
        data_model = global_signal_k + foreground

        # Evaluate loglikelihood
        num_data_points = data.size
        log_evidence = -0.5*num_data_points*np.log(2*np.pi*sigma_noise**2) \
                       - 0.5*np.sum((data-data_model)**2)/sigma_noise**2
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


def generate_prior(config_dict: dict,
                   include_signal: bool = True) -> Callable:
    """Generate a prior function.

    Parameters
    ----------
    config_dict : dict
        The configuration dictionary for the pipeline.
    include_signal : bool
        Whether the signal parameters are included in the prior.

    Returns
    -------
    prior : Callable
        The prior callable for the model.
    """
    # Loop over parameters constructing individual prior objects
    individual_priors = []

    # Construct parameter list
    for param_list in config_dict['priors'].keys():
        for param in config_dict['priors'][param_list].keys():
            # Get prior info
            prior_info = deepcopy(config_dict['priors'][param_list][param])

            # Replace emu_min and emu_max with the min and max value globalemu
            # can take for this parameter (if applicable)
            if param in GLOBALEMU_INPUTS:
                for k, v in prior_info.items():
                    if v == 'emu_min':
                        prior_info[k] = GLOBALEMU_PARAMETER_RANGES[param][0]
                    elif v == 'emu_max':
                        prior_info[k] = GLOBALEMU_PARAMETER_RANGES[param][1]

            # Get the prior type
            prior_type = prior_info.pop('type')
            if prior_type == 'uniform':
                param_prior = UniformPrior(prior_info['low'],
                                           prior_info['high'])
            elif prior_type == 'log_uniform':
                param_prior = LogUniformPrior(prior_info['low'],
                                              prior_info['high'])
            elif prior_type == 'gaussian':
                param_prior = GaussianPrior(prior_info['mean'],
                                            prior_info['std'])
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

    # Get noise sigma, configuration data, and timing file
    sigma_noise = get_noise_sigma()
    config_dict = load_configuration_dict()
    timing_file = timing_filename(sigma_noise)
    if rank == 0:
        clear_timing_data(timing_file)

    # Generate verification data
    if rank == 0:
        verification_ds_per_model = \
            config_dict['verification_data_sets_per_model']

        no_signal_simulator, with_signal_simulator = assemble_simulators(
            config_dict, sigma_noise)

        no_signal_data, _ = no_signal_simulator(verification_ds_per_model)
        with_signal_data, _ = with_signal_simulator(verification_ds_per_model)
        v_data = np.concatenate(
            [no_signal_data, with_signal_data], axis=0)
        v_labels = np.concatenate(
            [np.zeros(no_signal_data.shape[0]),
             np.ones(with_signal_data.shape[0])], axis=0)
    else:
        v_data = None
        v_labels = None
    v_data = comm.bcast(v_data, root=0)

    # Set up global emu
    globalemu_redshifts = global_signal_experiment_measurement_redshifts(
        config_dict['frequency_resolution'])
    globalemu_predictor = load_globalemu_emulator(globalemu_redshifts)

    # Generate priors
    no_signal_prior = generate_prior(config_dict, include_signal=False)
    with_signal_prior = generate_prior(config_dict, include_signal=True)

    if rank == 0:
        # Make sure chains directory exists
        os.makedirs(CHAIN_DIR, exist_ok=True)

        # Loop over data using Polychord to evaluate the evidence
        pc_log_bayes_ratios = []
        pc_nlike = []

    # Loop over mock data sets
    settings = None
    start = time.time()
    for data in v_data:
        # Data structure to store evidences to compute log bayes ratio
        # from
        log_zs = []

        # Use Polychord to fit data with and without signal
        for with_signal, prior in zip([False, True],
                                      [no_signal_prior, with_signal_prior]):
            # Assemble loglikelihood function
            loglikelihood = generate_loglikelihood(
                data, sigma_noise, globalemu_predictor,
                include_signal=with_signal)

            # Set Polychord properties
            if with_signal:
                n_dims = len(config_dict['priors'].keys())
            else:
                n_dims = len(config_dict['priors'].keys()) - \
                         len(GLOBALEMU_INPUTS)
            n_derived = 0
            settings = PolyChordSettings(n_dims, n_derived)
            settings.nlive = 25 * n_dims  # As recommended
            settings.base_dir = os.path.join(
                CHAIN_DIR,
                f'noise_{sigma_noise:.4f}_with_signal_{with_signal}')
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

            # Append log evidence to list
            comm.Barrier()
            if rank == 0:
                log_zs.append(output.logZ)

        # Compute log Bayes ratio
        if rank == 0:
            log_z_no_signal = log_zs[0]
            log_z_with_signal = log_zs[1]
            log_bayes_ratio = log_z_with_signal - log_z_no_signal
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

    # Save verification data for later comparison, alongside log bayes ratios
    # computed by Polychord and the labels of the models used to generate
    # the data
    os.makedirs('verification_data', exist_ok=True)
    np.savez(os.path.join('verification_data',
                          f'noise_{sigma_noise:.4f}_verification_data.npz'),
             data=v_data,
             labels=v_labels,
             log_bayes_ratios=np.squeeze(np.array(pc_log_bayes_ratios)),
             nlike=np.squeeze(np.array(pc_nlike)))


if __name__ == "__main__":
    main()
