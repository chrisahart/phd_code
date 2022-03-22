from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh
import matplotlib.pyplot as plt
from scripts.general import parameters

"""
    Convergence multigrid. 
    Plot convergence of energy and energy difference with multigrid._
"""

test = np.array([28.4251, 22.7401  , 24.1195 ])
print(test/parameters.angstrom_to_bohr)

folder = '/media/chris/DATA/Storage/University/PhD/Programming/Shuttle/work/personal_files/exercises/cdft/MgO/cpmd_restart/cell-110-240-4nn/ref'
file_in = '/input-bohr.xyz'
file_out = '/input.xyz'

# Assign column identities
cols = ['Num', 'Species', 'X', 'Y', 'Z']

# Read as csv file with whitespace delimiter
file_coord = pd.read_csv('{}{}'.format(folder, file_in), names=cols, delim_whitespace=True)
file_coord = file_coord.drop(file_coord.columns[0], axis=1)
file_coord['X'] = file_coord['X'] / parameters.angstrom_to_bohr
file_coord['Y'] = file_coord['Y'] / parameters.angstrom_to_bohr
file_coord['Z'] = file_coord['Z'] / parameters.angstrom_to_bohr

num_atoms = 238
print(file_coord)

print_xyz.print_from_pandas(file_coord, num_atoms, '{}/{}'.format(folder, file_out))
