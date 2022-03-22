from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh
import matplotlib.pyplot as plt

""" .xyz modify hydrogen bonds. 
    Change hydrogen bond length or angles.
     Used to test different hydrogen bond angles in lepidocrocite, or to make OH bond lengths longer. """

# Files
folder = '/scratch/cahart/work/personal_files/feIV_bulk/white_rust/222_supercell/neutral_14_h9656_cg/'
input_filename = 'geom.xyz'
output_filename = 'geom_hlong.xyz'

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

# Truncate bond lengths to bonded atoms
Fe_bonds_trunc = np.sort(Fe_bonds)[:, 0:neighbours['Fe']]
Fe_bonds_trunc[Fe_bonds_trunc > 2.3] = np.nan
O_bonds_trunc = np.sort(O_bonds)[:, 0:neighbours['O']]
O_bonds_trunc[O_bonds_trunc > 2.5] = np.nan
H_bonds_trunc = np.sort(H_bonds)[:, 0:neighbours['H']]


# Change hydrogen bond length or angle
unique_index = functions.layer_identifier(coordinates, species, 3, 2, 'H')
for i in range(0, len(H_index)):

    if True:

        # Determine atom pair
        atom1 = H_index[i]
        atom2 = np.argmin(bond_lengths[H_index[i]])

        # Calculate angle with axes
        diff = (coordinates[0, :, atom1] - coordinates[0, :, atom2])
        angle = np.arccos((np.abs(diff[:]) / H_bonds_trunc[i]))

        angle[1] = -np.deg2rad(45)

        # print('angle', np.degrees(angle))
        # print('diff', diff)

        coordinates[0, 2, atom1] = coordinates[0, 2, atom2] + np.sign(diff[2]) *0.9657 #0.96723 #(H_bonds_trunc[i])*np.sin(angle[1])
        # coordinates[0, 2, atom1] = coordinates[0, 2, atom2] + np.sign(diff[2]) * (H_bonds_trunc[i])*np.cos(angle[1])

        # Change hydrogen positions to right
        # coordinates[0, 1, atom1] = coordinates[0, 1, atom2] + np.sign(diff[1]) * (H_bonds_trunc[i])*np.sin(angle[2])
        # coordinates[0, 2, atom1] = coordinates[0, 2, atom2] + np.sign(diff[2]) * (H_bonds_trunc[i])*np.cos(angle[2])

# Create pandas dataframe to save to file
coord = np.column_stack((coordinates[0, 0, :], coordinates[0, 1, :], coordinates[0, 2, :]))
coord_xyz = pd.DataFrame(data=coord)
coord_xyz.insert(loc=0, column='A', value=pd.Series(species).values)  # Insert column with insert

# Print to file
print_xyz.print_from_pandas(coord_xyz, num_atoms, '{}/{}'.format(folder, output_filename))
