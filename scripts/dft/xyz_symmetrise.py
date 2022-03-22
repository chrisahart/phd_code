from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh
import matplotlib.pyplot as plt

""" .xyz symmetrize. 
    Symmetrise atoms of a particular type along particular Cartesian direction.  
    Used in geometry optimisation to fix distortions during BFGS optimisation such as in lepidocrocite neutral """

# Files
folder = '/scratch/cahart/work/personal_files/feIV_bulk/white_rust/332_supercell/neutral_hlong/'
input_filename = 'white_rust-SPIN_DENSITY-1_35.xyz'
output_filename = 'white_rust-SPIN_DENSITY-1_35_symm.xyz'

# Atom type, axis to detect and axis to symmetrise (should be the same)
atoms_list = ['H']
axis_detect = 1
axis_symm = 'Y'

# Set number of decimal places to round coordinates to, in case layers differ slightly
decimal_places = 3

# Number of header lines in .xyz file to ignore
header_lines = 1

# Read number of atoms and labels from .xyz file
cols = ['Species', 'X', 'Y', 'Z']
file_coord = pd.read_csv('{}{}'.format(folder, input_filename), names=cols, delim_whitespace=True)
file_coord = file_coord.dropna(axis='rows', thresh=2)
file_coord = file_coord.dropna(axis='columns', thresh=1)
coordinates, coord_x, coord_y, coord_z, species, num_atoms, num_timesteps = \
    load_coordinates.load_values_coord(folder, input_filename)

# Detect unique atoms
atom_types, atom_indices = np.unique(species, return_inverse=True)

# Detect unique layers
layers, layers_indices = np.unique(coordinates[0, axis_detect, :], return_inverse=True)

# Extract locations of iron atoms
iron_layers = np.zeros(num_atoms)
count = 0
for atom in range(num_atoms):

    if species.iloc[atom] in atoms_list:
        iron_layers[count] = layers[layers_indices[atom]]
        count = count + 1

iron_layers = iron_layers[:count]
iron_layer, layer_indices = np.unique(iron_layers, return_inverse=True)
layer_diff = np.zeros(iron_layer.shape[0]-1)
layers_diff = np.zeros(iron_layers.shape[0]-1)
iron_layer_truncated = np.copy(iron_layer)

# If layers are within tolerance set coordinate to bottom of layer
for i in range(0, iron_layer.shape[0]-1):

    layer_diff[i] = np.round(iron_layer[i + 1] - iron_layer[i], decimal_places)

    if layer_diff[i] < 0.2:

        iron_layer_truncated[i+1] = iron_layer_truncated[i]

iron_layer_final = iron_layer_truncated[layer_indices]
unique_layer, unique_index, unique_counts = np.unique(iron_layer_final, return_counts=True, return_inverse=True)
layer_dict = dict(zip(unique_layer, unique_counts))

# print('iron_layer_truncated', iron_layer_truncated)
# print('layer_dict', layer_dict)

fe_count = 0
# Assign coordinates
for atom in range(0, num_atoms):

    if species.iloc[atom] in atoms_list:

        file_coord[axis_symm][atom+1] = \
            np.average(iron_layer[
                       unique_index[fe_count]*unique_counts[unique_index[fe_count]]:
                       unique_index[fe_count]*unique_counts[unique_index[fe_count]]+unique_counts[unique_index[fe_count]]])
        fe_count = fe_count + 1

# Print to file
print_xyz.print_from_pandas(file_coord, num_atoms, '{}/{}'.format(folder, output_filename))
