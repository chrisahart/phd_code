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
    General Hirshfeld analysis. 
    Prints total iron spin moment.
"""

# Filenames
# filename1 = '/scratch/cahart/work/personal_files/feIV_bulk/hematite/final/221_supercell/exp_vesta_neutral_cubes_s4/hirsh.out'
# filename2 = '/scratch/cahart/work/personal_files/feIV_bulk/hematite/final/221_supercell/exp_vesta_neutral_cubes_s4_hole/hirsh_init.out'

# filename1 = '/scratch/cahart/work/personal_files/feIV_bulk/lepidocrocite/final/313_supercell/neutral_18_cubes_symm_tight/hirsh_neut.out'
# filename2 = '/scratch/cahart/work/personal_files/feIV_bulk/lepidocrocite/final/313_supercell/neutral_18_cubes_symm_tight_hole/hirsh_init.out'

filename1 = '/scratch/cahart/work/personal_files/feIV_bulk/geothite/final/hf18/313_supercell/positive_17_0_tight_neutral_cubes/hirsh_neut.out'
filename2 = '/scratch/cahart/work/personal_files/feIV_bulk/geothite/final/hf18/313_supercell/positive_17_0_tight_neutral_cubes_hole/hirsh_init.out'

# Read files
Fe_db1, O_db1, H_db1, file_spec1 = cp2k_hirsh.read_hirsh(filename1)
Fe_db2, O_db2, H_db2, file_spec2 = cp2k_hirsh.read_hirsh(filename2)

# Calculate difference
Fe_db_diff = Fe_db1 - Fe_db2
O_db_diff = O_db1 - O_db2
H_db_diff = H_db1 - H_db2
file_spec_diff = file_spec1 - file_spec2

# File 1
Fe_db1_positive = Fe_db1[Fe_db2['Spin'] > 0]
Fe_db1_negative = Fe_db1[Fe_db2['Spin'] < 0]
O_db1_positive = O_db1[O_db2['Spin'] > 0]
O_db1_negative = O_db1[O_db2['Spin'] < 0]

# File 2
Fe_db2_positive = Fe_db2[Fe_db1['Spin'] > 0]
Fe_db2_negative = Fe_db2[Fe_db2['Spin'] < 0]
O_db2_positive = O_db2[O_db2['Spin'] >= 0]
O_db2_negative = O_db2[O_db2['Spin'] < 0]

# File 1
print('Fe_db1_positive', np.average(Fe_db1_positive['Spin']))
print('Fe_db1_negative', np.average(Fe_db1_negative['Spin']))
# print('O_db1_positive', np.average(O_db1_positive['Spin']))
# print('O_db1_negative', np.average(O_db1_negative['Spin']))

# File 2
print('\nFe_db2_positive', np.average(Fe_db2_positive['Spin']))
print('Fe_db2_negative', np.average(Fe_db2_negative['Spin']))
# print('O_db2_positive', np.average(O_db2_positive['Spin']))
# print('O_db2_negative', np.average(O_db2_negative['Spin']))

# Difference
print('\nFe difference average for positive spin', np.average(Fe_db2_positive['Spin'] - Fe_db1_positive['Spin']))
print('Fe difference average for negative spin', np.average(Fe_db2_negative['Spin'] - Fe_db1_negative['Spin']))
# print('O difference average for positive spin', np.average(O_db2_positive['Spin'] - O_db1_positive['Spin']))
# print('O difference average for negative spin', np.average(O_db2_negative['Spin'] - O_db1_negative['Spin']))

# Sum
print('\nFe difference sum for positive spin', np.sum(Fe_db2_positive['Spin'] - Fe_db1_positive['Spin']))
print('Fe difference sum for negative spin', np.sum(Fe_db2_negative['Spin'] - Fe_db1_negative['Spin']))
# print('O difference sum for positive spin', np.sum(O_db2_positive['Spin'] - O_db1_positive['Spin']))
# print('O difference sum for negative spin', np.sum(O_db2_negative['Spin'] - O_db1_negative['Spin']))

# Sum for all
# print('\nSum for positive spin', np.sum(Fe_db2_positive['Spin'] - Fe_db1_positive['Spin']) +
#       np.sum(O_db2_positive['Spin'] - O_db1_positive['Spin']))
#
# print('Sum for negative spin', np.sum(Fe_db2_negative['Spin'] - Fe_db1_negative['Spin']) +
#       np.sum(O_db2_negative['Spin'] - O_db1_negative['Spin']))

print('Sum for all', np.sum(Fe_db2_negative['Spin'] - Fe_db1_negative['Spin']) +
      np.sum(O_db2_negative['Spin'] - O_db1_negative['Spin']) +
      np.sum(Fe_db2_positive['Spin'] - Fe_db1_positive['Spin']) +
      np.sum(O_db2_positive['Spin'] - O_db1_positive['Spin'])
      )
