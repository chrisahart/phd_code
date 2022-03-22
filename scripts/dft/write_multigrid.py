from __future__ import division, print_function
import numpy as np
import shutil
import os
from distutils.dir_util import copy_tree
import matplotlib.pyplot as plt
import scipy

""" Write force constants. 
    Copy folder contents to new folder, modifying input and submit files.  """

# Files
directory = '/scratch/cahart/work/personal_files/feIV_bulk/force_constants/goethite/hole/test/'
folder_in = 'neutral'
folder_out = 'neutral_offset'

# Positions
pos_Fe = np.array([2.53061,  6.47301,  8.29892])  # Fe goethite
pos_O = np.array([3.21517,  4.47859,  8.29893])  # O goethite
pos_H = np.array([4.15529,  4.17201,  8.29895])  # H goethite
line = (119 + 107) - 2
line2 = (119 + 108) - 2

percent_range = np.array([0.25, 0.5, 0.75, 1.5, 1, 2, 3, 4, 5])
values = np.shape(percent_range)[0]

# Loop over desired folders
for i in range(0, values):

    # Calculate new coordinate
    percent = pos_O + (pos_Fe - pos_O) * percent_range[i]/100
    percentH = pos_H + (pos_Fe - pos_O) * percent_range[i]/100

    # Make new directory
    copy_tree('{}{}'.format(directory, folder_in), '{}{}{}'.format(directory, folder_out, percent_range[i]))

    # Edit input.inp with new coordinate
    lines = open('{}{}{}/input.inp'.format(directory, folder_out, percent_range[i])).read().splitlines()
    lines[line] = 'O {} {} {}'.format('O ', percent[0], percent[1],  percent[2])
    lines[line2] = 'H {} {} {}'.format(percentH[0], percentH[1], percentH[2])
    lines[31] = ' CHARGE  0'
    open('{}{}{}/input.inp'.format(directory, folder_out, percent_range[i]), 'w').write('\n'.join(lines))

    # Edit submit.pbs
    lines = open('{}neutral_offset{}/submit.pbs'.format(directory, percent_range[i])).read().splitlines()
    lines[2] = '#PBS -N g_{}'.format(i)
    open('{}{}{}/submit.pbs'.format(directory, folder_out, percent_range[i]), 'w').write('\n'.join(lines))
