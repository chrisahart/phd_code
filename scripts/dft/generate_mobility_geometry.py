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

""" Generate 1D chain for mobility calculation  """

# Filename
# folder1 = 'E:/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/mobility/1d_chain'
# output_filename = 'nn-1-1-cg_coupling.xyz'
# coord = np.zeros((1500, 3))
# for i in range(0, np.shape(coord)[0]):
#     coord[i, 0] = float(i * 2.97)
# np.savetxt('{}/coordinates.dat'.format(folder1), coord, fmt='%1.4f')

folder1 = 'E:/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/mobility/2d_plane'
# input_filename1 = '700_1_1_supercell_2d_chain.xyz'
input_filename1 = '40_40_1_supercell_2d.xyz'
file_coord1, num_atoms1, species1 = load_coordinates.load_file_coord(folder1, input_filename1)
fe_only = pd.DataFrame(np.nan, index=np.arange(0, int(np.shape(species1)[0]), 1), columns=['X', 'Y', 'Z'])

count = 0
for i in range(0, int(np.shape(species1)[0])):
    if species1.iloc[i] == 'Fe':
        count += 1

        fe_only['X'][i] = file_coord1['X'][i+1].copy()
        fe_only['Y'][i] = file_coord1['Y'][i+1].copy()
        fe_only['Z'][i] = file_coord1['Z'][i+1].copy()

fe_only = fe_only.dropna(axis='rows', thresh=2)
fe_only = fe_only.dropna(axis='columns', thresh=1)
np.savetxt('{}/40_40_1_supercell_2d_fe.dat'.format(folder1), fe_only, fmt='%1.4f')

fe_only.insert(loc=0, column='Species', value=species1)
num_atoms1 = count
print_xyz.print_from_pandas(fe_only, num_atoms1, '{}/40_40_1_supercell_2d_fe.xyz'.format(folder1))
# print_xyz.print_from_pandas(fe_only, num_atoms1, '{}/coordinates.dat'.format(folder1))
# np.savetxt('{}/50_50_1_supercell_2d_fe.dat'.format(folder1), fe_only, fmt='%1.4f')
