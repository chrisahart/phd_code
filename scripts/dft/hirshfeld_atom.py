from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scripts.formatting import load_coordinates
from scripts.formatting import load_cube
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh
from scripts.general import parameters
from scripts.general import functions

"""
    Atomic Hirshfeld analysis. 
    Prints change in spin moment of specific atoms.
"""

# CDFT
filename1 = 'E:/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/331_supercell_cdft/hirshfeld/dft/ts_nn4/hirsh.out'

Fe_db1, O_db1, H_db1, file_spec1 = cp2k_hirsh.read_hirsh(filename1)

hole_4o = np.array([52, 204, 209, 229, 128]) - 1
nn1_4o = np.array([89, 172, 179, 228, 175]) - 1
nn4_4o = np.array([61, 170, 138, 143, 207]) - 1

hole_6o = np.array([52, 134, 204, 209, 229, 177, 128]) - 1
nn1_6o = np.array([89, 172, 179, 228, 175, 134, 177]) - 1
nn2_6o = np.array([98, 175, 170, 207, 233, 174, 172]) - 1
nn3_6o = np.array([101, 164, 222, 183, 185, 181, 202]) - 1
nn4_60 = np.array([61, 170, 226, 207, 147, 143, 138]) - 1

hole = hole_6o[0]
nn1 = nn1_6o[0]
nn2 = nn2_6o[0]
nn3 = nn3_6o[0]
nn4 = nn4_60[0]

one = hole_6o
# two = nn2

print('\nSpin')
for i in range(0, np.shape(one)[0]):
    print(str(np.round(file_spec1['Spin'].values[one[i]], 2)))


print('\nCharge')
for i in range(0, np.shape(one)[0]):
    print(str(np.round(file_spec1['Charge'].values[one[i]], 2)))


# print(file_spec1['Spin'].values[one])
# print('Charge', np.sum(file_spec1['Charge'].values[one]))

# print('Charge', np.sum(file_spec1['Charge'].values[one]), np.sum(file_spec1['Charge'].values[two]))
# print('Charge diff', np.sum(file_spec1['Charge'].values[one]) - np.sum(file_spec1['Charge'].values[two]))
# print('\nSpin', np.sum(file_spec1['Spin'].values[one]), np.sum(file_spec1['Spin'].values[two]))
# print('Spin diff', np.sum(file_spec1['Spin'].values[one]) - np.sum(file_spec1['Spin'].values[two]))

# spin_o = file_spec1['Spin'][atoms_o].values
# average_o = np.average(spin_o)
# print('\nspin_o', spin_o)
# print('average_o', average_o)
#
# spin_fe = file_spec1['Spin'][atoms_fe].values
# average_fe = np.average(spin_fe)
# print('\nspin_fe', spin_fe)
# print('average_fe', average_fe)

# Difference of two files

# Filenames
# filename1 = '/scratch/cahart/work/personal_files/feIV_bulk/white_rust/final/332_supercell/hf29/neutral_hlong_cg_offset1_29cg/hirsh_final.out'
# filename2 = '/scratch/cahart/work/personal_files/feIV_bulk/white_rust/final/332_supercell/hf29/neutral_hlong_cg29/hirsh_neut.out'
#
# # Read files
# Fe_db1, O_db1, H_db1, file_spec1 = cp2k_hirsh.read_hirsh(filename1)
# Fe_db2, O_db2, H_db2, file_spec2 = cp2k_hirsh.read_hirsh(filename2)
#
# # Specify atoms manually
# atoms_o = np.array([75, 61, 90, 76, 85, 64]) - 1
# atoms_fe = np.array([48, 53, 47, 45, 54, 38, 49, 46, 50]) - 1
#
# neut_o = file_spec1['Spin'][atoms_o].values
# relaxed_o = file_spec2['Spin'][atoms_o].values
# change_o = relaxed_o - neut_o
# average_o = np.average(change_o)
# print('\nneut_o', neut_o)
# print('relaxed_o', relaxed_o)
# print('change_o', change_o)
# print('average_o', average_o)
#
# neut_fe = file_spec1['Spin'][atoms_fe].values
# relaxed_fe = file_spec2['Spin'][atoms_fe].values
# change_fe = relaxed_fe - neut_fe
# average_fe = np.average(change_fe[1:-2])
# print('\nneut_fe', neut_fe)
# print('relaxed_fe', relaxed_fe)
# print('change_fe', change_fe)
# print('average_fe', average_fe)
