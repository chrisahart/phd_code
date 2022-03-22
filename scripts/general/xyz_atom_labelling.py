from __future__ import division, print_function
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
import scipy
import re
import pickle
import pandas as pd
from distutils.dir_util import copy_tree
import copy
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh

""" .xyz. 
    .xyz.  """

# Filename
folder = '/scratch/cahart/work/personal_files/fe_bulk/pdos/all_label/polaron/lepidocrocite_hse_pdos_fine_all_hole'
input_filename = 'input.xyz'
output_filename = 'input.xyz'

# Read coordinates
file_coord, num_atoms, species = load_coordinates.load_file_coord(folder, input_filename)
species = species.reset_index(drop=True)
file_coord= file_coord.reset_index(drop=True)


# Species
species_unique = list(set(species.values))
count = np.zeros(len(species_unique))

for i in range(0, num_atoms):

    for species_type in range(len(species_unique)):

        if species[i] == species_unique[species_type]:
            species[i] = '{}_{} '.format(species_unique[species_type], int(count[species_type]+1))
            print(species[i])
            count[species_type] = count[species_type] + 1

print('species_unique', species_unique)
print('count', count)
print('species\n', species)

# Print pandas dataframe to file
file_coord.insert(loc=0, column='Species', value=species)

print(file_coord)
print_xyz.print_from_pandas(file_coord, num_atoms, '{}/{}'.format(folder, output_filename))
