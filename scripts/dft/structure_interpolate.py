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
# folder1 = '/media/chris/DATA1/Storage/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/331_supercell_cdft/structures/dft_force-1e-4_from-441'
folder1 = '/media/chris/DATA1/Storage/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/441_supercell_cdft/structures/dft'
# folder1='/media/chris/DATA1/Storage/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/331_supercell_cdft/structures/dft_force-1e-4_from-441'
# folder1='/media/chris/DATA1/Storage/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/221_supercell_cdft/structures/electron/dft/hf50'
# folder1='/media/chris/DATA1/Storage/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/441_supercell_cdft/structures/electron/dft/hf12'
# folder1='/media/chris/DATA1/Storage/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/441_supercell_cdft/structures/electron/dft/hf50'
input_filename1 = 'nn-0_r3.xyz'
input_filename2 = 'nn-1-3.xyz'
output_filename = 'nn-1-3_0r3_3r1_coupling.xyz'

# atoms = np.array([104, 34, 72, 70, 92, 36, 8, 68, 32])

# Read coordinates
file_coord1, num_atoms1, species1 = load_coordinates.load_file_coord(folder1, input_filename1)
file_coord2, num_atoms2, species2 = load_coordinates.load_file_coord(folder1, input_filename2)

# for i in range(0, int(np.shape(atoms)[0])):
#     print(i)
#     file_coord1['X'][atoms[i]] = file_coord2['X'][atoms[i]]
#     file_coord1['Y'][atoms[i]] = file_coord2['Y'][atoms[i]]
#     file_coord1['Z'][atoms[i]] = file_coord2['Z'][atoms[i]]

# interpolated = file_coord1
interpolated = 0.5 * file_coord1 + 0.5 * file_coord2
interpolated.insert(loc=0, column='Species', value=species1)
print_xyz.print_from_pandas(interpolated, num_atoms1, '{}/{}'.format(folder1, output_filename))
