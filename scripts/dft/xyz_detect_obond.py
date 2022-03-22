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

# Separate into hydrogen bonded and not hydrogen bonded O atoms for DOS analysis

# Filename
folder = '/media/chris/DATA/Storage/University/PhD/Programming/Shuttle/work/personal_files/feIV_bulk/pdos/hole/relaxed/lepidocrocite_hse_pdos_fine'
input_filename = 'geom.xyz'
filename_output = 'geom_label.xyz'

# Read coordinates
coordinates, coord_x, coord_y, coord_z, species, num_atoms, num_timesteps = \
    load_coordinates.load_values_coord(folder, input_filename)
species = species.reset_index(drop=True)

# Parameters
neighbours = {'Fe': 6, 'Fe_a': 6, 'Fe_b': 6, 'O': 3, 'H': 1}
decimal_places = 5

# Detect unique atoms
atom_types, atom_indices = np.unique(species, return_inverse=True)

# Calculate bond lengths
bond_lengths = functions.calc_bond_lengths(coordinates, num_atoms)
bond_lengths[bond_lengths == 0] = 1E6

# Calculate indexes of different elements
H_index = np.array([i for i, e in enumerate(species) if e == 'H'])
O_index = [i for i, e in enumerate(species) if e == 'O']
Fe_index = [i for i, e in enumerate(species) if e == 'Fe_a' or e == 'Fe_b' or e == 'Fe']

# Separate bond lengths matrix into individual species
Fe_bonds = bond_lengths[Fe_index, :]
O_bonds = bond_lengths[O_index, :]
H_bonds = bond_lengths[H_index, :]

# If hydrogen bonded replace O with oh
for i in range(0, O_bonds.shape[0]):
    if np.min(O_bonds[i]) < 1.5:
        species[O_index[i]] = 'Oh'

# Create pandas dataframe from species and coordinates
coord = np.column_stack((coord_x.ravel(), coord_y.ravel(), coord_z.ravel()))
coord_xyz = pd.DataFrame(data=coord)
coord_xyz.insert(loc=0, column='A', value=pd.Series(species).values)

# Print pandas dataframe to file
print_xyz.print_from_pandas(coord_xyz, num_atoms, '{}/{}'.format(folder, filename_output))
