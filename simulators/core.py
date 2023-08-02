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
        simulator_1: Simulator,
        simulator_2: Simulator
):
    """Combine two simulators by adding their data outputs.

    Parameters
    ----------
    simulator_1 : Simulator
        The first simulator to combine.
    simulator_2 : Simulator
        The second simulator to combine.

    Returns
    -------
    combined_simulator : Simulator
        The combined simulator.
    """
    def _combined_simulator(num_data_simulations: int
                            ) -> Tuple[np.ndarray, DataFrame]:
        """Combine two simulators by adding their data outputs."""
        data_1, params_1 = simulator_1(num_data_simulations)
        data_2, params_2 = simulator_2(num_data_simulations)

        # Combine parameters
        params = params_1.join(params_2, how='outer',
                               lsuffix='_1', rsuffix='_2')
        return data_1 + data_2, params
    return _combined_simulator
