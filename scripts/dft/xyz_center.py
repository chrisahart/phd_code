from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh
import matplotlib.pyplot as plt

"""
    .xyz center.
    Center .xyz files around particular atom (not present in Zdenek coordshift).
    Used for cluster calculations, cut from bulk with xyz_extract_atoms then centred using .xyz center 
"""

folder = '/scratch/cahart/work/personal_files/feIV_bulk/hematite/cluster/hole/neutral/geometries/'
input_filename = 'cluster_neutral.xyz'
output_filename = 'cluster_neutral_center2.xyz'
#
# Number of header lines in .xyz file to ignore
header_lines = 2

# Read number of atoms and labels from .xyz file
cols = ['Species', 'X', 'Y', 'Z']
file_coord = pd.read_csv('{}{}'.format(folder, input_filename), names=cols, delim_whitespace=True)
num_atoms = int(float(file_coord['Species'][0]))

# Center around atom given cell size
atom = 1
cell = 20

# Translate all other atoms
for i in range(2, num_atoms+1):
    file_coord['X'][i] = (-file_coord['X'][1] + file_coord['X'][i]) + cell/2
    file_coord['Y'][i] = (-file_coord['Y'][1] + file_coord['Y'][i]) + cell/2
    file_coord['Z'][i] = (-file_coord['Z'][1] + file_coord['Z'][i]) + cell/2

# Set atom to center of simulation cell
file_coord['X'][1] = cell/2
file_coord['Y'][1] = cell/2
file_coord['Z'][1] = cell/2

# Print to file
print_xyz.print_from_pandas(file_coord, num_atoms, '{}/{}'.format(folder, output_filename))

