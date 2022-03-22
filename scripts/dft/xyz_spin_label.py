from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh
import matplotlib.pyplot as plt


""" Crystal layer label. 
    Script to label atoms according to layer in Cartesian direction. """


# folder = '/media/chris/DATA1/Storage/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/441_supercell_cdft/structures/dft'
folder= '/media/chris/Elements/Backup/Archer-2/bulk/hematite/221_supercell_cdft/cdft/couplings/geo_opt/0_from-neutral'
input_filename = 'input.xyz'
output_filename = 'input_label.xyz'

# Import coordinates
coordinates, coord_x, coord_y, coord_z, species, num_atoms, num_timesteps = \
    load_coordinates.load_values_coord(folder, input_filename)

# Atom type, axis to detect
atom_type = 'Fe'
axis = 2

# Spin layer (e.g. AFM abab, FM aaaa), repeat as desired
spin_layer = 5 * ['Fe_a', 'Fe_b', 'Fe_a', 'Fe_b']
# spin_layer = 5 * ['Fe_b', 'Fe_a', 'Fe_b', 'Fe_a']
# spin_layer = 5 * ['Fe_a', 'X', 'Fe_a', 'X']
# Decimal places
decimal_places = 3

# Number of header lines in .xyz file to ignore
header_lines = 2

# Detect unique atoms
atom_types, atom_indices = np.unique(species, return_inverse=True)

# Extract locations of iron atoms
unique_index = functions.layer_identifier(coordinates, species, decimal_places, axis, atom_type)

# Assign spin depending on layer
fe_count = 0
for atom in range(0, num_atoms):

    if species.iloc[atom] == atom_type:

        species.iloc[atom] = spin_layer[unique_index[fe_count]]

        fe_count = fe_count + 1

# Read number of atoms and labels from .xyz file
cols = ['Species', 'X', 'Y', 'Z']
file_coord = pd.read_csv('{}/{}'.format(folder, input_filename), names=cols, delim_whitespace=True)
file_coord = file_coord.drop(['Species'], 1)
file_coord = file_coord.apply(pd.to_numeric, errors='coerce')
file_coord = file_coord.dropna(axis='rows', thresh=2)
file_coord = file_coord.dropna(axis='columns', thresh=1)

# Insert atom labels to database of coordinates
file_coord.insert(loc=0, column='A', value=pd.Series(species).values)

# Print to file
print_xyz.print_from_pandas(file_coord, num_atoms, '{}/{}'.format(folder, output_filename))
