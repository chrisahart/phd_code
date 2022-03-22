from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
import glob
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
import random


""" xyz modify """

# Experimental structure
# folder_in = 'E:/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/441_supercell_cdft/density_difference/dft/nn-0'
# folder_out = 'E:/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/441_supercell_cdft/density_difference/dft/nn-0'
# filename_in_1 = 'struct.xyz'
# filename_in_2 = 'hole_s_331_2.xyz'
# filename_out = 'hole_s_331_3.xyz'
# file_coord1, num_atoms1, species1 = load_coordinates.load_file_coord(folder_in, filename_in_1)
# file_coord2, num_atoms2, species2 = load_coordinates.load_file_coord(folder_in, filename_in_2)
# index_extract = np.ones(np.shape(file_coord2)[0])
# for i in range(0, np.shape(file_coord2)[0]):  # Loop over 221
#     for j in range(0, np.shape(file_coord1)[0]):  # Loop over 441
#         if file_coord2['X'].values[i] == file_coord1['X'].values[j]:
#             index_extract[i] = j
# np.savetxt('{}/331_index.txt'.format(folder_out), index_extract, delimiter=',')

# Use
folder_in = '/media/chris/Elements/Backup/Archer-2/surfin/hematite/from-guido-neutral/ref/geometries/prodA'
folder_out = folder_in
filename_in_1 = 'step-0.xyz'
filename_in_2 = 'step-0-cube.xyz'
filename_out = 'input.xyz'
file_coord1, num_atoms1, species1 = load_coordinates.load_file_coord(folder_in, filename_in_1)
file_coord2, num_atoms2, species2 = load_coordinates.load_file_coord(folder_in, filename_in_2)
file_coord2.insert(loc=0, column='Species', value=species1)
print_xyz.print_from_pandas(file_coord2, num_atoms2, '{}/{}'.format(folder_out, filename_out))
