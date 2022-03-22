from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.formatting import load_cube
from scripts.formatting import cp2k_hirsh
import matplotlib.pyplot as plt

""" .xyz bond lengths. 
    Calculates difference in bond lengths between two .cube files for specific atom. """

# Files
folder1 = '/scratch/cahart/work/personal_files/feIV_bulk/lepidocrocite/snapshots/313_supercell/nodisp/'
filename1 = 'neutral_c81_s.cube'
folder2 = '/scratch/cahart/work/personal_files/feIV_bulk/lepidocrocite/snapshots/313_supercell/nodisp/'
filename2 = 'hole_c81_s.cube'
atom = 81 - 1

# Parameters
neighbours = {'Fe': 6, 'Fe_a': 6, 'Fe_b': 6, 'O': 3, 'H': 1}
decimal_places = 5

# Load cube files
coordinates1, coord_x1, coord_y1, coord_z1, num_atoms1, num_timesteps1 = load_cube.load_values_coord(folder1, filename1)
coordinates2, coord_x2, coord_y2, coord_z2, num_atoms2, num_timesteps2 = load_cube.load_values_coord(folder2, filename2)

# Calculate bond length differences
bond_lengths1 = functions.calc_bond_lengths(coordinates1, num_atoms1)
bond_lengths2 = functions.calc_bond_lengths(coordinates2, num_atoms1)
bonds_1 = np.sort(bond_lengths1[atom, :])[1:1+neighbours['Fe']]
bonds_2 = np.sort(bond_lengths2[atom, :])[1:1+neighbours['Fe']]
diff = bonds_2 - bonds_1

# Printing
print('coordinates1', coordinates1[0, :, atom])
print('coordinates2', coordinates2[0, :, atom])
print('bonds_1', bonds_1)
print('bonds_2', bonds_2)
print(atom, diff, np.average(diff))
