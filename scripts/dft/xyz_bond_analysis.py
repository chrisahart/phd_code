from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh
import matplotlib.pyplot as plt

""" xyz average bonds. 
    Calculates average bond lengths for each bulk iron oxide. """


def rmse(x):
    return np.sqrt(np.mean((x - np.mean(x)) ** 2))


# Filename
folder = '/scratch/cahart/work/personal_files/feIV_bulk/hematite/convergence/multigrid/221/exp_vesta_m300/'
input_filename = 'geom_61.xyz'

# Read coordinates
coordinates, coord_x, coord_y, coord_z, species, num_atoms, num_timesteps = \
    load_coordinates.load_values_coord(folder, input_filename)
species = species.reset_index(drop=True)

# Parameters
neighbours = {'Fe': 6, 'Fe_a': 6, 'Fe_b': 6, 'O': 3, 'H': 1}
decimal_places = 5

# Determine to nearest neighbours
fe_atom = np.array([10]) - 1
atoms_o = np.zeros(6*fe_atom.size)
bond_lengths = functions.calc_bond_lengths(coordinates, num_atoms)
for i in range(0, fe_atom.size):
    bond_lengths_atom = bond_lengths[fe_atom[i]]
    bond_lengths_atom[bond_lengths_atom == 0] = 1E6
    atoms_o[6*i:6*(i+1)] = np.argsort(bond_lengths_atom)[0:6]

print(atoms_o+1)

# Create connectivity matrix (Fe bonded to what O), accounting for iron atoms which may not be fully bonded
# todo

# Use connectivity matrix to calculate RMSE
print('coordinates', coordinates.shape)
fe_coord = coordinates[0, :, fe_atom]

coord_diff = coordinates[0, :, fe_atom] - coordinates[0, :, int(atoms_o[0])]
bond_length = np.linalg.norm(coord_diff)
rmse = np.sqrt((1.94 - np.linalg.norm(coord_diff))**2)
print(bond_length, rmse)
