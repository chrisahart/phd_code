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
directory = '/scratch/cahart/work/personal_files/feIV_bulk/hematite/convergence/multigrid_m400restart/221/'
folder_in = 'exp_vesta_m400'
folder_out = 'exp_vesta_m'

percent_range = np.arange(450, 800, step=50)
values = np.shape(percent_range)[0]

# Loop over desired folders
for i in range(0, values):

    # Make new directory
    copy_tree('{}{}'.format(directory, folder_in), '{}{}{}'.format(directory, folder_out, percent_range[i]))

    # Edit input.inp with new coordinate
    lines = open('{}{}{}/input.inp'.format(directory, folder_out, percent_range[i])).read().splitlines()
    lines[54-1] = 'CUTOFF  {}'.format(percent_range[i])
    open('{}{}{}/input.inp'.format(directory, folder_out, percent_range[i]), 'w').write('\n'.join(lines))

    # Edit submit.pbs
    lines = open('{}{}{}/submit.pbs'.format(directory, folder_out, percent_range[i])).read().splitlines()
    lines[2] = '#PBS -N hem_m{}'.format(percent_range[i])
    open('{}{}{}/submit.pbs'.format(directory, folder_out, percent_range[i]), 'w').write('\n'.join(lines))
