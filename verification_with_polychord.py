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

The script takes two command line arguments:
- `--batch_num`: An integer that specifies the batch of verification runs
    to perform. As many HPCs have a limit on the job length, the script
    allows the verification runs to be split into batches. The number of
    mock data sets analysed in each batch is set in the configuration file.
    Batches should run sequentially, starting from 0.
- `--noise_sigma` (optional): A float value that allows you to specify the
   noise sigma in K. The default is 0.020 K. This should be the same as the
    noise sigma used later in train_evidence_network.py.
"""

# Required imports
from __future__ import annotations
from typing import Callable, Tuple
from fbf_utilities import load_configuration_dict, \
    timing_filename, add_timing_data, clear_timing_data, assemble_simulators, \
    NOISE_DEFAULT
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
import argparse
import anesthetic

# Parameters
CHAIN_DIR = "chains"


# IO
def get_command_line_arguments():
    """Get command line arguments of the script.

    Returns
    -------
    batch_num : int
        The batch number of the verification runs to perform.
    noise_sigma : float
        The noise sigma in K.
    """
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_num",
                        type=int,
                        nargs='?')
    parser.add_argument("noise_sigma",
                        type=float,
                        default=NOISE_DEFAULT,
                        nargs='?')
    args = parser.parse_args()
    return args.batch_num, args.noise_sigma


# Prior, likelihood, and evidence functions
def generate_loglikelihood(data: np.ndarray,
                           globalemu_emulator: Callable,
                           noise_sigma: float,
                           include_signal: bool = True) -> Callable:
    """Generate a loglikelihood function.

    Parameters
    ----------
    data : np.ndarray
        The mock data to evaluate the loglikelihood for.
    globalemu_emulator : Callable
        The emulator for the global signal.
    noise_sigma : float
        The standard deviation of the noise in K.
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
        log_evidence = -0.5*num_data_points*np.log(2*np.pi*noise_sigma**2) \
                       - 0.5*np.sum((data-data_model)**2)/noise_sigma**2
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
        if param_list == 'global_signal' and not include_signal:
            continue
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

    # Configuration data, script params, and timing file
    config_dict = load_configuration_dict()
    batch_num, noise_sigma = get_command_line_arguments()
    timing_file = timing_filename(noise_sigma)
    if rank == 0 and batch_num == 0:
        clear_timing_data(timing_file)

    # Remove old verification data (if first batch)
    verification_data_dir = 'verification_data'
    os.makedirs(verification_data_dir, exist_ok=True)
    verification_data_file = os.path.join(
        verification_data_dir,
        f'verification_data_noise_{noise_sigma}.npz')
    if rank == 0 and batch_num == 0:
        try:
            os.remove(verification_data_file)
        except FileNotFoundError:
            pass

    # Generate verification data
    verification_ds_per_model = \
        config_dict['verification_data_sets_per_model']
    verification_ds_per_batch = \
        config_dict['verification_data_set_batch_size']
    ds_left = verification_ds_per_model - verification_ds_per_batch * batch_num
    ds_per_model = min(verification_ds_per_model, ds_left)
    if ds_per_model <= 0:
        raise ValueError("No verification data set left to analyse.")

    if rank == 0:
        no_signal_simulator, with_signal_simulator = assemble_simulators(
            config_dict, noise_sigma)

        no_signal_data, _ = no_signal_simulator(ds_left)
        with_signal_data, _ = with_signal_simulator(ds_left)
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

    batch_chains_dir = os.path.join(CHAIN_DIR, f'noise_{noise_sigma}_'
                                               f'batch_{batch_num}')
    if rank == 0:
        # Make sure chains directory exists
        os.makedirs(CHAIN_DIR, exist_ok=True)
        os.makedirs(batch_chains_dir, exist_ok=True)

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

        # Use Polychord to fit data with and without the signal
        for with_signal, prior in zip([False, True],
                                      [no_signal_prior, with_signal_prior]):
            # Initial high noise run to roughly find the posterior peak
            # Assemble loglikelihood function
            loglikelihood = generate_loglikelihood(
                data,
                globalemu_predictor,
                config_dict['high_noise_value'],
                include_signal=with_signal)

            # Set Polychord properties
            n_dims = len(config_dict['priors']['foregrounds'].keys())
            if with_signal:
                n_dims += len(config_dict['priors']['global_signal'].keys())
            n_derived = 0
            settings = PolyChordSettings(n_dims, n_derived)
            settings.nlive = 25 * n_dims  # As recommended
            settings.base_dir = batch_chains_dir
            settings.file_root = 'verification'
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

            # Find the peak of the posterior and use it as the starting point
            # for the low noise run
            comm.Barrier()
            log_volume_correction = None
            new_mins = None
            new_maxs = None
            if rank == 0:
                chains = anesthetic.read_chains(
                    os.path.join(settings.base_dir, settings.file_root))

                # Only do this for the foregrounds as they are the only
                # parameters expected to have large volume corrections
                if with_signal:
                    parameter_names = np.arange(
                        len(config_dict['priors']['global_signal'].keys()),
                        n_dims)
                else:
                    parameter_names = np.arange(n_dims)
                param_iter = zip(
                    parameter_names,
                    config_dict['priors']['foregrounds'].values()
                )

                new_mins = []
                new_maxs = []
                log_volume_correction = 1
                for param_name, param_info in param_iter:
                    # Only implemented for uniform priors so skip for none
                    # uniform priors
                    if param_info['type'] != 'uniform':
                        new_mins.append(None)
                        new_maxs.append(None)
                        continue
                    old_min = param_info['low']
                    old_max = param_info['high']

                    # Get the 1D marginalized posterior
                    samples = chains[param_name]
                    weights = chains.get_weights()

                    # Find the mean and std of the 1D marginalized posterior
                    mean = np.average(samples, weights=weights)
                    std = np.sqrt(np.average((samples - mean) ** 2,
                                             weights=weights))

                    # Set new min and max to 3 sigma from the mean
                    # or old min and max if the new min and max are outside
                    # the old min and max
                    new_min = mean - 3 * std
                    new_max = mean + 3 * std
                    new_min = max(new_min, old_min)
                    new_max = min(new_max, old_max)

                    # Calculate the volume correction (this assumes a uniform
                    # prior, hence the condition above)
                    log_volume_correction += \
                        np.log(old_max - old_min) - \
                        np.log(new_max - new_min)

                    new_mins.append(new_min)
                    new_maxs.append(new_max)

                new_mins = np.array(new_mins)
                new_maxs = np.array(new_maxs)

                print('High noise fit complete', flush=True)
                print(f'log_volume_correction: {log_volume_correction}',
                      flush=True)
                print(f'new_mins: {new_mins}', flush=True)
                print(f'new_maxs: {new_maxs}', flush=True)

            # Broadcast the new mins and maxs to all ranks
            comm.Barrier()
            new_mins = comm.bcast(new_mins, root=0)
            new_maxs = comm.bcast(new_maxs, root=0)
            log_volume_correction = comm.bcast(log_volume_correction, root=0)

            # Deepcopy the configuration and update with the new mins and maxs
            new_config_dict = deepcopy(config_dict)
            for idx, param in enumerate(
                    new_config_dict['priors']['foregrounds']):
                if new_mins[idx] is None:
                    continue
                new_config_dict['priors']['foregrounds'][param]['low'] = \
                    new_mins[idx]
                new_config_dict['priors']['foregrounds'][param]['high'] = \
                    new_maxs[idx]

            # Assemble new loglikelihood and prior function
            new_loglikelihood = generate_loglikelihood(
                data,
                globalemu_predictor,
                noise_sigma,
                include_signal=with_signal)
            new_prior = generate_prior(new_config_dict,
                                       include_signal=with_signal)

            # Reset Polychord properties
            settings = PolyChordSettings(n_dims, n_derived)
            settings.nlive = 25 * n_dims  # As recommended
            settings.base_dir = batch_chains_dir
            settings.file_root = 'verification'
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
            output = run_polychord(new_loglikelihood, n_dims,
                                   n_derived, settings, new_prior)

            # Append log evidence to list
            comm.Barrier()
            if rank == 0:
                log_zs.append(output.logZ + log_volume_correction)

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
    add_timing_data(timing_file,
                    f'total_polychord_log_k_batch_{batch_num}',
                    end - start)

    # Clean up
    try:
        shutil.rmtree(settings.base_dir)
    except OSError:
        pass

    # Save verification data for later comparison, alongside log bayes ratios
    # computed by Polychord and the labels of the models used to generate
    # the data. Creating if doesn't exist already, appending if it does.
    if not os.path.exists(verification_data_file):
        np.savez(verification_data_file,
                 data=v_data,
                 labels=v_labels,
                 log_bayes_ratios=np.squeeze(np.array(pc_log_bayes_ratios)),
                 nlike=np.squeeze(np.array(pc_nlike)))
        return

    # Load existing data
    verification_file_contents = np.load(verification_data_file)
    existing_log_bayes_ratios = verification_file_contents['log_bayes_ratios']
    existing_nlike = verification_file_contents['nlike']
    existing_data = verification_file_contents['data']
    existing_labels = verification_file_contents['labels']
    verification_file_contents.close()

    # Append new data
    new_log_bayes_ratios = np.concatenate(
        [existing_log_bayes_ratios, np.squeeze(np.array(pc_log_bayes_ratios))])
    new_nlike = np.concatenate(
        [existing_nlike, np.squeeze(np.array(pc_nlike))])
    new_data = np.concatenate(
        [existing_data, v_data], axis=0)
    new_labels = np.concatenate(
        [existing_labels, v_labels], axis=0)

    # Save new data
    np.savez(verification_data_file,
             data=new_data,
             labels=new_labels,
             log_bayes_ratios=new_log_bayes_ratios,
             nlike=new_nlike)


if __name__ == "__main__":
    main()
