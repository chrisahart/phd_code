from __future__ import division, print_function
import time
import numpy as np
from matplotlib import pyplot as plt
import copy
import pandas as pd

"""
    Functions reading CP2K hirshfeld analysis
"""


def read_hirsh(filename):
    """
    Read Hirshfeld analysis from CP2K output file (requires removal from CP2K output file to filename)
    """

    # Read number of atoms and labels from .xyz file
    cols = ['Atom', 'Element', 'Kind', 'Ref Charge', 'Pop 1', 'Pop 2', 'Spin', 'Charge']
    file_spec1 = pd.read_csv(filename, names=cols, delim_whitespace=True, skiprows=1)
    species = file_spec1['Element']

    H_index = [i for i, e in enumerate(species) if e == 'H']
    O_index = [i for i, e in enumerate(species) if e == 'O']
    Fe_index = [i for i, e in enumerate(species) if e == 'Fe_a' or e == 'Fe_b' or e == 'Fe']

    # Force database to numeric, assigning any non-numeric as NaN
    file_spec1 = file_spec1.apply(pd.to_numeric, errors='coerce')

    Fe_db1 = file_spec1.loc[Fe_index]
    O_db1 = file_spec1.loc[O_index]
    H_db1 = file_spec1.loc[H_index]

    return Fe_db1, O_db1, H_db1, file_spec1
