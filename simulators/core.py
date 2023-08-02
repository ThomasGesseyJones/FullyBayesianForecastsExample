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
    """"""
    pass
