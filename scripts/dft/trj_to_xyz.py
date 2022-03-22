from __future__ import division, print_function
import time
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from scripts.general import functions
from scripts.general import parameters
from scripts.formatting import load_coordinates
from scripts.formatting import load_energy
from scripts.formatting import load_forces_out
from scripts.formatting import load_forces
from scripts.formatting import print_xyz

"""
    Read CPMD .trj and convert to .xyz with random element allocation 
"""


def flatten(t):
    return [item for sublist in t for item in sublist]


folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru'
filename_output = 'test.xyz'
num_atoms = 191
# species = ['Ru'] + ['Ru'] + ['O'] + ['H'] + ['H'] + ['O'] + ['H'] + ['H'] + (num_atoms-8) * ['X']
# species = 63 * ['H'] + 126 * ['O'] + ['Ru'] + ['Ru']

# Build species list
species = ['Ru'] + ['Ru']
for i in range(int((num_atoms-2)/3)):
    species += ['O'] + ['H'] + ['H']

# Read .trj and convert to numpy array of arrays size 3 (x,y,z)
data_in = np.genfromtxt('{}/final-step.trj'.format(folder_1),delimiter='_', dtype=str)
data_list = []
for i in range(0, data_in.shape[0]):
    data_list.append(data_in[i].split())
data = np.array(flatten(data_list), dtype=float)
coord = np.array_split(data, num_atoms)

# Create pandas dataframe from species and coordinates
coord_xyz = pd.DataFrame(data=coord)
coord_xyz.insert(loc=0, column='A', value=species)

# Print pandas dataframe to file
print_xyz.print_from_pandas(coord_xyz, num_atoms, '{}/{}'.format(folder_1, filename_output))
