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
import pandas as pd

""" Sum .pdos occupancies. 
    Read .pdos from file and create files of summed occupancies per species.  """


def sum_all(x, label):
    """Sum of all orbitals"""

    total = 0
    for i in range(3, len(label)):
        total += x[label[i]]

    return total


def sum_species(filename, indices, label, power):
    """ Loop over filename and add together """

    data = pd.read_csv('{}/{}'.format(folder, filename[indices[0]]), names=label[:], skiprows=[0, 1],
                       delim_whitespace=True)
    data[label[3]] = sum_all(data, label) ** power

    for loop in range(1, len(indices)):
        data_new = pd.read_csv('{}/{}'.format(folder, filename[indices[loop]]), names=label, skiprows=[0, 1],
                               delim_whitespace=True)
        data_new[label[3]] = sum_all(data_new, label) ** power
        data[label[3]] = data[label[3]] + data_new[label[3]]
        data = data[label[0:4]]

    return data


def print_sum(filename, data, fermi_energy):
    """ Print data to filename """

    data.loc[-1] = ['MO', 'Eigenvalue', 'Occupation', 'Sum']
    data.index = data.index + 1
    data = data.sort_index()
    data.loc[-1] = ['Total Sum. Fermi = {}'.format(fermi_energy)] + [None]*3
    data.index = data.index + 1
    data = data.sort_index()
    data.to_csv(filename, sep='\t', index=False, header=False)


folder = '/scratch/cahart/work/personal_files/fe_bulk/pdos/all_label/polaron/lepidocrocite_hse_pdos_fine_all_electron'
filename_alpha = []
filename_beta = []

labels_h = ['MO', 'Eigenvalue', 'Occupation', 's', 'py', 'pz', 'px']
labels_o = labels_h + ['d-2', 'd-1', 'd0', 'd+1', 'd+2']
labels_fe = labels_o + ['f-3', 'f-2', 'f-1', 'f0', 'f+1', 'f+2', 'f+3']

# Detect files ending in extension
for subdir, dirs, files in os.walk(folder):
    for file_name in files:

        if ".pdos" in os.path.splitext(file_name)[-1]:

            if "ALPHA" in os.path.splitext(file_name)[0]:
                filename_alpha.append(file_name)

            elif "BETA" in os.path.splitext(file_name)[0]:
                filename_beta.append(file_name)

# Fermi energy
input_file = open('{}/{}'.format(folder, filename_alpha[0]))
line_first = input_file.readline().strip().split()
fermi = float(line_first[15])

# Read species
species = []
for i in range(len(filename_alpha)):
    with open('{}/{}'.format(folder, filename_alpha[i])) as f:
        words = f.readlines()[0].split()
        species.append(words[6])
indices_alpha_fe = [i for i, s in enumerate(species) if 'Fe' in s]
indices_alpha_o = [i for i, s in enumerate(species) if 'O' in s]
indices_alpha_h = [i for i, s in enumerate(species) if 'H' in s]

# Read species
species = []
for i in range(len(filename_beta)):
    with open('{}/{}'.format(folder, filename_beta[i])) as f:
        words = f.readlines()[0].split()
        species.append(words[6])
indices_beta_fe = [i for i, s in enumerate(species) if 'Fe' in s]
indices_beta_o = [i for i, s in enumerate(species) if 'O' in s]
indices_beta_h = [i for i, s in enumerate(species) if 'H' in s]

# Sum over ALPHA ** 1
pdos_alpha_o_power1 = sum_species(filename_alpha, indices_alpha_o, labels_o, 1)
pdos_alpha_fe_power1 = sum_species(filename_alpha, indices_alpha_fe, labels_fe, 1)
pdos_alpha_h_power1 = sum_species(filename_alpha, indices_alpha_h, labels_h, 1)

#  Sum over ALPHA ** 2
pdos_alpha_o_power2 = sum_species(filename_alpha, indices_alpha_o, labels_o, 2)
pdos_alpha_fe_power2 = sum_species(filename_alpha, indices_alpha_fe, labels_fe, 2)
pdos_alpha_h_power2 = sum_species(filename_alpha, indices_alpha_h, labels_h, 2)
pdos_alpha_power2 = copy.copy(pdos_alpha_o_power2)
pdos_alpha_power2[labels_o[3]] = \
    pdos_alpha_o_power2[labels_o[3]] + pdos_alpha_fe_power2[labels_fe[3]] + pdos_alpha_h_power2[labels_h[3]]

# Sum over ALPHA ** 4
pdos_alpha_o_power4 = sum_species(filename_alpha, indices_alpha_o, labels_o, 4)
pdos_alpha_fe_power4 = sum_species(filename_alpha, indices_alpha_fe, labels_fe, 4)
pdos_alpha_h_power4 = sum_species(filename_alpha, indices_alpha_h, labels_h, 4)
pdos_alpha_power4 = copy.copy(pdos_alpha_o_power4)
pdos_alpha_power4[labels_o[3]] = \
    pdos_alpha_o_power4[labels_o[3]] + pdos_alpha_fe_power4[labels_fe[3]] + pdos_alpha_h_power4[labels_h[3]]

# Sum over BETA ** 1
pdos_beta_o_power1 = sum_species(filename_beta, indices_beta_o, labels_o, 1)
pdos_beta_fe_power1 = sum_species(filename_beta, indices_beta_fe, labels_fe, 1)
pdos_beta_h_power1 = sum_species(filename_beta, indices_beta_h, labels_h, 1)

# Sum over BETA ** 2
pdos_beta_o_power2 = sum_species(filename_beta, indices_beta_o, labels_o, 2)
pdos_beta_fe_power2 = sum_species(filename_beta, indices_beta_fe, labels_fe, 2)
pdos_beta_h_power2 = sum_species(filename_beta, indices_beta_h, labels_h, 2)
pdos_beta_power2 = copy.copy(pdos_beta_o_power2)
pdos_beta_power2[labels_o[3]] = \
    pdos_beta_o_power2[labels_o[3]] + pdos_beta_fe_power2[labels_fe[3]]+ pdos_beta_h_power2[labels_h[3]]

# Sum over BETA ** 4
pdos_beta_o_power4 = sum_species(filename_beta, indices_beta_o, labels_o, 4)
pdos_beta_fe_power4 = sum_species(filename_beta, indices_beta_fe, labels_fe, 4)
pdos_beta_h_power4 = sum_species(filename_beta, indices_beta_h, labels_h, 4)
pdos_beta_power4 = copy.copy(pdos_beta_o_power4)
pdos_beta_power4[labels_o[3]] = \
    pdos_beta_o_power4[labels_o[3]] + pdos_beta_fe_power4[labels_fe[3]]+ pdos_beta_h_power4[labels_h[3]]

# Print ALPHA
print_sum('{}/ALPHA_fe_power1.out'.format(folder), pdos_alpha_fe_power1, fermi)
print_sum('{}/ALPHA_o_power1.out'.format(folder), pdos_alpha_o_power1, fermi)
print_sum('{}/ALPHA_h_power1.out'.format(folder), pdos_alpha_h_power1, fermi)
print_sum('{}/ALPHA_fe_power2.out'.format(folder), pdos_alpha_fe_power2, fermi)
print_sum('{}/ALPHA_o_power2.out'.format(folder), pdos_alpha_o_power2, fermi)
print_sum('{}/ALPHA_h_power2.out'.format(folder), pdos_alpha_h_power2, fermi)
print_sum('{}/ALPHA_power2.out'.format(folder), pdos_alpha_power2, fermi)
print_sum('{}/ALPHA_fe_power4.out'.format(folder), pdos_alpha_fe_power4, fermi)
print_sum('{}/ALPHA_o_power4.out'.format(folder), pdos_alpha_o_power4, fermi)
print_sum('{}/ALPHA_h_power4.out'.format(folder), pdos_alpha_h_power4, fermi)
print_sum('{}/ALPHA_power4.out'.format(folder), pdos_alpha_power4, fermi)

# Print BETA
print_sum('{}/beta_fe_power1.out'.format(folder), pdos_beta_fe_power1, fermi)
print_sum('{}/beta_o_power1.out'.format(folder), pdos_beta_o_power1, fermi)
print_sum('{}/beta_h_power1.out'.format(folder), pdos_beta_h_power1, fermi)
print_sum('{}/beta_fe_power2.out'.format(folder), pdos_beta_fe_power2, fermi)
print_sum('{}/beta_o_power2.out'.format(folder), pdos_beta_o_power2, fermi)
print_sum('{}/beta_h_power2.out'.format(folder), pdos_beta_h_power2, fermi)
print_sum('{}/beta_power2.out'.format(folder), pdos_beta_power2, fermi)
print_sum('{}/beta_fe_power4.out'.format(folder), pdos_beta_fe_power4, fermi)
print_sum('{}/beta_o_power4.out'.format(folder), pdos_beta_o_power4, fermi)
print_sum('{}/beta_h_power4.out'.format(folder), pdos_beta_h_power4, fermi)
print_sum('{}/beta_power4.out'.format(folder), pdos_beta_power4, fermi)
