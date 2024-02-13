"""Core simulator functionality.

Functions for manipulating simulators, e.g. combining simulators.
"""

# Import libraries
from typing import Callable, Tuple
from pandas import DataFrame
import numpy as np


# Simulator type
Simulator = Callable[[int], Tuple[np.ndarray, DataFrame]]


# Combine simulators
def additive_simulator_combiner(
        *simulators: Simulator
):
    """Combine simulators by adding their data outputs.

    Simulator parameters must have different names.

    Parameters
    ----------
    simulators : Iterable of Simulator
        The simulators to combine

    Returns
    -------
    combined_simulator : Simulator
        The combined simulator.
    """
    # Catch edge-cases
    if len(simulators) == 0:
        raise ValueError('No simulators given.')
    elif len(simulators) == 1:
        return simulators[0]

    def _combined_simulator(num_data_simulations: int
                            ) -> Tuple[np.ndarray, DataFrame]:
        """Combine simulators by adding their data outputs."""
        data_plus_params = [simulator(num_data_simulations) for simulator in
                            simulators]
        combined_data = data_plus_params[0][0]
        combined_params = data_plus_params[0][1]
        for data, params in combined_params[1:]:
            combined_params = combined_params.join(params, how='outer')
            combined_data += data
        return combined_data, combined_params
    return _combined_simulator
