from __future__ import division, print_function
import time
import numpy as np
import copy
from matplotlib import pyplot as plt
import pandas as pd
from scripts.formatting import load_coordinates
from scripts.formatting import print_xyz
from scripts.general import parameters
from scripts.general import functions


"""
   .xyz extract atoms.
   Extract atoms from .xyz file
   Used to extract clusters from bulk crystal.
"""

folder = '/scratch/cahart/work/personal_files/feIV_bulk/hematite/cluster/hole/neutral/geometries/'
input_filename = 'hole_c37_s.xyz'
output_filename = 'cluster_neutral.xyz'

# Number of header lines in .xyz file to ignore
header_lines = 2

# Read number of atoms and labels from .xyz file
cols = ['Species', 'X', 'Y', 'Z']
file_coord = pd.read_csv('{}{}'.format(folder, input_filename), names=cols, delim_whitespace=True)
coordinates, coord_x, coord_y, coord_z, species, num_atoms, num_timesteps = \
    load_coordinates.load_values_coord(folder, input_filename)

# Determine to nearest neighbours
fe_atom = np.array([37]) - 1
atoms_o = np.zeros(6*fe_atom.size)
bond_lengths = functions.calc_bond_lengths(coordinates, num_atoms)
for i in range(0, fe_atom.size):
    bond_lengths_atom = bond_lengths[fe_atom[i]]
    bond_lengths_atom[bond_lengths_atom == 0] = 1E6
    atoms_o[6*i:6*(i+1)] = np.argsort(bond_lengths_atom)[0:6]

# Cut to nearest neighbours
atoms = np.unique(np.append(fe_atom, atoms_o))
num_atoms = atoms.size
file_coord = file_coord.loc[atoms + 1, :]

# Print pandas dataframe to file
print('fe_atom', fe_atom)
print('atoms_o', atoms_o)
print('file_coord\n', file_coord)
print_xyz.print_from_pandas(file_coord, num_atoms, '{}/{}'.format(folder, output_filename))
