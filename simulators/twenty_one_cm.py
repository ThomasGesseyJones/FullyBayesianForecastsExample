"""21cm Simulators.

Functions and classes to simulate the cosmological 21-cm signal.
"""

# Required libraries
from globalemu.downloads import download as globalemu_download
from globalemu.eval import evaluate
from tensorflow import keras
import numpy as np
import os


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
