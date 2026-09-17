## Test for condcution solver in pyTHM

import numpy as np
import pandas as pd
from pyTHM.Conduction import conduction
from conftest import (
    DATA_DIR,
    OUTPUTS_DIR
)

def analytical_sol():
    """
    Return analytical solution for temperature profile
    """


def test_instantiate_conduction_solver():

    solver = conduction.HeatConductionInFuelPin()