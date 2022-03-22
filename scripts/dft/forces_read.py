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
folder1 = 'E:/University/PhD/Programming/dft_ml_md/output/pyrene-cof/supercell_33/pbe/cdft/geo_opt/linker-yes/hirshfeld/absolute_1-dimer/forces'

# input_filename1 = 'dft-force.out'
# input_filename1 = 'cdft_force-1.out'
# input_filename1 = 'cdft_force-2.out'

atoms = np.array([108, 120, 119, 124, 123, 130]) + 1

# Read coordinates
file_coord1, num_atoms1, species1 = load_coordinates.load_file_coord(folder1, input_filename1)

for i in range(0, np.shape(atoms)[0]):
    print("{0:.0E},".format(file_coord1['X'].values[atoms[i]]), "{0:.0E}".format(file_coord1['Y'].values[atoms[i]]))
