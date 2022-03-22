#!/usr/bin/env python

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
import re


""" Write force constants.                                                                                                                                                                                         
    Copy folder contents to new folder, modifying input and submit files.  """

# Directory
directory = ''
folder_out = 'neutral_'

# Grid
cutoff = np.array([600, 1200, 1800, 2400, 3000])
rel_cutoff = np.array([140, 200, 250])
cutoff_rel_cutoff = []
for i in range(cutoff.shape[0]):
    for j in range(rel_cutoff.shape[0]):
        cutoff_rel_cutoff.append('{}_{}'.format(cutoff[i], rel_cutoff[j]))
folder_out_append = np.array(cutoff_rel_cutoff)

# Search string
search_string = ' HOMO - LUMO gap [eV] : '
index = 0

# Allocation
store = []

# Loop over desired folders
for i in range(0, np.shape(folder_out_append)[0]):

    temp = []
    # Search for string and return numbers from each line
    for line in open('{}{}{}/cp2k_log.log'.format(directory, folder_out, folder_out_append[i])).read().splitlines():

        if search_string in line:

            line_string = str(line.split())
            line_numeric = float(re.sub("[^0-9.]", "", str(line.split())))
            temp.append(line_numeric)

    store.append(np.array(temp[0]))

print('store\n', store)
np.savetxt('{}band_gap.out'.format(directory), store, delimiter=',')
