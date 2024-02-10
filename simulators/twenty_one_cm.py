"""21cm Simulators.

Functions and classes to simulate the cosmological 21-cm signal.
"""

# Required libraries
from typing import Callable, Tuple
from globalemu.downloads import download as globalemu_download
from globalemu.eval import evaluate
from tensorflow import keras
import numpy as np
import os
from priors import generate_composite_prior_sampler
from pandas import DataFrame
from .core import Simulator


# Globalemu parameters
GLOBALEMU_BASE_DIR = f'T_release{os.sep}'  # Globalemu requires the seperator
GLOBALEMU_INPUTS = ['f_star', 'v_c', 'f_x', 'tau', 'alpha', 'nu_min', 'R_mfp']
GLOBALEMU_PARAMETER_RANGES = {
    'f_star': (0.0001, 0.5),
    'v_c': (4.2, 100.0),  # km/s
    'f_x': (0.0001, 1000.0),
    'tau': (0.04, 0.17),
    'alpha': (1.0, 1.5),
    'nu_min': (0.1, 3.0),  # keV
    'R_mfp': (10.0, 50.0),  # cMpc
}


def download_globalemu() -> None:
    """Download the globalemu 21-cm sky averaged signal emulator."""
    globalemu_download().model()


def load_globalemu_emulator(redshifts: np.ndarray) -> evaluate:
    """Load the pre-trained globalemu 21-cm sky averaged signal emulator.

    Parameters
    ----------
    redshifts : np.ndarray
        The redshifts at which the emulator will evaluate the 21-cm signal

    Returns
    -------
    predictor : evaluate object
        The pre-trained globalemu 21-cm sky averaged signal emulator
    """
    if not os.path.exists(GLOBALEMU_BASE_DIR):
        download_globalemu()

    # Preloading the model leads to a significant speed-up with global emu
    model = keras.models.load_model(
        os.path.join(GLOBALEMU_BASE_DIR, 'model.h5'),
        compile=False)
    predictor = evaluate(base_dir=GLOBALEMU_BASE_DIR,
                         model=model,
                         z=redshifts)
    return predictor


# REACH parameters, from de Lera Acedo et al. 2022
REACH_MIN_Z = 7.5
REACH_MAX_Z = 28.0

FREQ_21CM_MHZ = 1420.0


def global_signal_experiment_measurement_redshifts(
        freq_binning_mhz,
        min_z: float = REACH_MIN_Z,
        max_z: float = REACH_MAX_Z
) -> np.ndarray:
    """Get the redshifts at which our experiment will measure the 21-cm signal.

    Parameters
    ----------
    freq_binning_mhz : float
        The frequency binning in MHz of the experiment.
    min_z : float
        Minimum redshift experiment can observe.
    max_z : float
        Maximum redshift experiment can observe.

    Returns
    -------
    redshifts : np.ndarray
        The redshifts it will measure.
    """
    max_nu = FREQ_21CM_MHZ / (1 + min_z)
    min_nu = FREQ_21CM_MHZ / (1 + max_z)
    range_nu = max_nu - min_nu
    num_bins = np.floor(range_nu / freq_binning_mhz)
    nus = min_nu + np.arange(num_bins)*freq_binning_mhz
    redshifts = FREQ_21CM_MHZ / nus - 1
    return redshifts


# Foreground model
def foreground_model(
        frequencies_mhz: np.ndarray,
        coefficients: np.ndarray,
        lower_norm_freq: float = 55.0,
        upper_norm_freq: float = 85.0,
) -> np.ndarray:
    """Foreground model for the sky-averaged radio temperature.

    Model is that used by SARAS 3. See Singh et al. 2021 and
    Bevins et al. 2022
    log10(T) = sum_i a_i * R(log10(nu/1 MHz))^i,
    where R linearly scales the frequency to -1 at lower_norm_freq and 1 at
    upper_norm_freq.

    In the original papers, the lower and upper normalization frequencies
    are set to the bounding frequencies of their data.
    To keep our coefficient values consistent with the literature,
    we use the same normalization frequencies of 55 and 85 MHz by default.

    Parameters
    ----------
    frequencies_mhz : np.ndarray
        The frequencies at which to calculate the foreground model in MHz.
    coefficients : np.ndarray
        The coefficients for the foreground model (a_i). Coefficient index
        should be the first dimension in the array. The number of coefficients
        is set implicitly by the shape of the array.
    lower_norm_freq : float
        The lower normalization frequency in MHz. Default is 55 MHz.
    upper_norm_freq : float
        The upper normalization frequency in MHz. Default is 85 MHz.

    Returns
    -------
    for_temp : np.ndarray
        The foreground model for the sky-averaged radio temperature in K.
        Frequency is in the same order as the input frequencies, and is indexed
        along the first dimension of the array.
    """
    # Scale the frequencies
    log_freq = np.log10(frequencies_mhz)
    norm_log_freq = (2 * log_freq - np.log10(lower_norm_freq) -
                     np.log10(upper_norm_freq)) / \
                    (np.log10(upper_norm_freq) - np.log10(lower_norm_freq))

    # Exponentiate the frequencies
    num_coefficients = coefficients.shape[0]
    log_freq_powers = np.array([norm_log_freq**i for i in
                                range(num_coefficients)])

    # Calculate the foreground model
    log_temp = np.einsum('i...,ij->j...', coefficients,
                         log_freq_powers, optimize=True)
    for_temp = 10**log_temp
    return for_temp


# Simulators
def generate_global_signal_simulator(
        global_emu_predictor: evaluate,
        f_star_sampler: Callable,
        v_c_sampler: Callable,
        f_x_sampler: Callable,
        tau_sampler: Callable,
        alpha_sampler: Callable,
        nu_min_sampler: Callable,
        r_mfp_sampler: Callable
) -> Simulator:
    """Return a simulator function for the global 21-cm signal.

    Parameters
    ----------
    global_emu_predictor : evaluate object
        The pre-trained globalemu 21-cm sky averaged signal emulator
    f_star_sampler : Callable
        Function that returns a sample of f_star
    v_c_sampler : Callable
        Function that returns a sample of v_c
    f_x_sampler : Callable
        Function that returns a sample of f_x
    tau_sampler : Callable
        Function that returns a sample of tau
    alpha_sampler : Callable
        Function that returns a sample of alpha
    nu_min_sampler : Callable
        Function that returns a sample of nu_min
    r_mfp_sampler : Callable
        Function that returns a sample of R_mfp

    Returns
    -------
    signal_simulator : Callable
        Function that takes a number of data simulations and returns
        that number of global 21-cm signals (in K) as an array
        plus the corresponding parameters as a dataframe.
    """
    # Generate the composite prior sampler
    prior_sampler = generate_composite_prior_sampler(
        f_star_sampler, v_c_sampler, f_x_sampler, tau_sampler,
        alpha_sampler, nu_min_sampler, r_mfp_sampler)

    # Define the simulator function
    def signal_simulator(num_sims: int) -> Tuple[np.ndarray, DataFrame]:
        """Simulate the global 21-cm signal.

        Parameters
        ----------
        num_sims : int
            The number of simulations to run

        Returns
        -------
        signals : np.ndarray
            The simulated global 21-cm signals (in K) as an array
        parameters : DataFrame
            The corresponding parameters as a dataframe.
        """
        # Sample the parameters
        params = prior_sampler(num_sims)

        # Run the simulator in batches to avoid memory issues
        batch_size = 100_000
        remaining_sims = num_sims
        signals = []
        while remaining_sims > 0:
            batch_sims = min(batch_size, remaining_sims)
            batch_params = params[num_sims - remaining_sims:
                                  num_sims - remaining_sims + batch_sims]
            # Remember converting mK -> K
            signals.append(global_emu_predictor(batch_params)[0]/1000)
            remaining_sims -= batch_sims
        signals = np.concatenate(signals)

        # Convert params to DataFrame
        params = DataFrame(params, columns=GLOBALEMU_INPUTS)

        return signals, params
    return signal_simulator
