from __future__ import division, print_function
import numpy as np
import shutil
import os
from distutils.dir_util import copy_tree
import matplotlib.pyplot as plt
import scipy
import glob
import copy
import shutil

""" Rename .pdos files. 
    Read .pdos files and rename to spin channel and species.  """

folder = '/scratch/cahart/work/personal_files/fe_bulk/pdos/all_label/polaron/lepidocrocite_hse_pdos_fine_all_electron'
extension = ".pdos"
filename_alpha = []
filename_beta = []
species = []

# Detect files ending in extension
for subdir, dirs, files in os.walk(folder):
    for file_name in files:

        if extension in os.path.splitext(file_name)[-1]:

            if "ALPHA" in os.path.splitext(file_name)[0]:
                filename_alpha.append(file_name)

            elif "BETA" in os.path.splitext(file_name)[0]:
                filename_beta.append(file_name)

# Read species
for i in range(len(filename_alpha)):
    with open('{}/{}'.format(folder, filename_alpha[i])) as f:
        words = f.readlines()[0].split()
        species.append(words[6])

# Rename files with spin channel and species
for i in range(len(filename_alpha)):
    os.rename('{}/{}'.format(folder, filename_alpha[i]), '{}/ALPHA_{}{}'.format(folder, species[i], extension))
    os.rename('{}/{}'.format(folder, filename_beta[i]), '{}/BETA_{}{}'.format(folder, species[i], extension))

print (filename_alpha)
print (filename_beta)
print (species)
