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
from scripts.general import parameters
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh

""" Calculate PDOS for 1D hematite chain """

def gaussian(energy_grid, eigenvalue, width):
    """Return Gaussian centred on eigenvalue over energy_grid with width"""

    x = -((energy_grid - eigenvalue) / width) ** 2

    return np.exp(x) / (np.sqrt(np.pi) * width)


def smearing(eigenvalues, pdos, energy_grid, width):
    """ Calculate convoluted PDOS by summing Gaussian distribution centred on each eigenvalue"""

    cpdos = np.zeros(eigenvalues.shape[0])
    for i in range(eigenvalues.shape[0]):
        cpdos += pdos[i] * gaussian(energy_grid, eigenvalues[i], width)

    return cpdos


def sum_s(x, label):
    """Sum of s orbitals"""

    total = 0
    total += x[label[3]]

    return total


def sum_p(x, label):
    """Sum of p orbitals"""

    total = 0
    for i in range(4, 7):
        total += x[label[i]]

    return total


def sum_d(x, label):
    """Sum of d orbitals"""

    total = 0
    for i in range(7, 12):
        total += x[label[i]]

    return total


def sum_f(x, label):
    """Sum of d orbitals"""

    total = 0
    for i in range(12, 18):
        total += x[label[i]]

    return total


def sum_all(x, label):
    """Sum of all orbitals"""

    total = 0
    for i in range(3, len(label)):
        total += x[label[i]]

    return total


# Projected Density of states
width = 0.1  # Gaussian width, 0.2 works for m400, 0.1 for m800, 0.05 for m1200
y_lim = 180  # Y axis limit
x_lim = [-8, 6]  # x axis limit

folder = 'E:/University/PhD/Programming/dft_ml_md/output/fe_chain/dos/hse_admm'
filename_pdos_fea_a = '{}/fea_alpha.pdos'.format(folder)
filename_pdos_o_a = '{}/o_alpha.pdos'.format(folder)
filename_pdos_fea_b = '{}/fea_beta.pdos'.format(folder)
filename_pdos_o_b = '{}/o_beta.pdos'.format(folder)
filename_pdos_save = '{}/hf_iron_oxygen.png'.format(folder)

# Read Fermi energy from file
input_file = open(filename_pdos_fea_a, 'r')
line_first = input_file.readline().strip().split()
fermi = float(line_first[15])
print('Fermi energy:', fermi)

# Read energy and DOS from files
labels_o = ['MO', 'Eigenvalue', 'Occupation', 's', 'py', 'pz', 'px', 'd-2', 'd-1', 'd0', 'd+1', 'd+2']
labels_fe = ['MO', 'Eigenvalue', 'Occupation', 's', 'py', 'pz', 'px', 'd-2', 'd-1', 'd0', 'd+1', 'd+2', 'f-3', 'f-2',
             'f-1', 'f0', 'f+1', 'f+2', 'f+3']
# labels_o = ['MO', 'Eigenvalue', 'Occupation', 's', 'py', 'pz', 'px']
# labels_fe = ['MO', 'Eigenvalue', 'Occupation', 's', 'py', 'pz', 'px', 'd-2', 'd-1', 'd0', 'd+1', 'd+2']
pdos_fea_a = pd.read_csv(filename_pdos_fea_a, names=labels_fe, skiprows=[0, 1], delim_whitespace=True)
pdos_o_a = pd.read_csv(filename_pdos_o_a, names=labels_o, skiprows=[0, 1], delim_whitespace=True)
pdos_fea_b = pd.read_csv(filename_pdos_fea_b, names=labels_fe, skiprows=[0, 1], delim_whitespace=True)
pdos_o_b = pd.read_csv(filename_pdos_o_b, names=labels_o, skiprows=[0, 1], delim_whitespace=True)

# Calculate PDOS for Fe (summing Fe_a, Fe_b for both spin channels)
pdos_fe_a_3s = sum_s(pdos_fea_a, labels_fe)
pdos_fe_b_3s = sum_s(pdos_fea_b, labels_fe)
pdos_fe_a_3p = sum_p(pdos_fea_a, labels_fe)
pdos_fe_b_3p = sum_p(pdos_fea_b, labels_fe)
pdos_fe_a_3d = sum_d(pdos_fea_a, labels_fe)
pdos_fe_b_3d = sum_d(pdos_fea_b, labels_fe)
pdos_fe_a_3f = sum_f(pdos_fea_a, labels_fe)
pdos_fe_b_3f = sum_f(pdos_fea_b, labels_fe)
pdos_fe_a_all = sum_all(pdos_fea_a, labels_fe)
pdos_fe_b_all = sum_all(pdos_fea_b, labels_fe)

# Calculate PDOS for O
pdos_o_a_2s = sum_s(pdos_o_a, labels_o)
pdos_o_b_2s = sum_s(pdos_o_b, labels_o)
pdos_o_a_2p = sum_p(pdos_o_a, labels_o)
pdos_o_b_2p = sum_p(pdos_o_b, labels_o)
pdos_o_a_2d = sum_d(pdos_o_a, labels_o)
pdos_o_b_2d = sum_d(pdos_o_b, labels_o)
pdos_o_a_all = sum_all(pdos_o_a, labels_o)
pdos_o_b_all = sum_all(pdos_o_b, labels_o)

# Calculate convoluted PDOS
num_points = (pdos_o_a['s']).shape[0]
eigenvalues = (pdos_o_a['Eigenvalue'] - fermi) * parameters.hartree_to_ev
energy_grid = np.linspace(np.min(eigenvalues), np.max(eigenvalues), num=num_points)

# Plotting convoluted PDOS
fig_cpdos, ax_cpdos = plt.subplots()

# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_o_a_2s.values, energy_grid, width), 'b', label='O(2s)')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_o_b_2s.values, energy_grid, width), 'b')

# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_o_a_2p.values, energy_grid, width), 'r', label='O(2p)')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_o_b_2p.values, energy_grid, width), 'r')

# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_o_a_2d.values, energy_grid, width), 'g', label='O(2d)')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_o_b_2d.values, energy_grid, width), 'g')

# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_fe_a_3s.values, energy_grid, width), 'r', label='Fe(3s)')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_fe_b_3s.values, energy_grid, width), 'r')

# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_fe_a_3p.values, energy_grid, width), 'g', label='Fe(3p)')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_fe_b_3p.values, energy_grid, width), 'g')

# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_fe_a_3d.values, energy_grid, width), 'b', label='Fe(3d)')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_fe_b_3d.values, energy_grid, width), 'b')

# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_fe_a_3f.values, energy_grid, width), 'y', label='Fe(3f)')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_fe_b_3f.values, energy_grid, width), 'y')

ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_o_a_all.values, energy_grid, width), 'r', label='O')
# print(pdos_o_a_all)
ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_o_b_all.values, energy_grid, width), 'r')

ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_fe_a_all.values, energy_grid, width), 'b', label='Fe')
ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_fe_b_all.values, energy_grid, width), 'b')

# ax_cpdos.set_xlim([x_lim[0], x_lim[1]])
# ax_cpdos.set_ylim([-y_lim, y_lim])
ax_cpdos.set_xlim([-7, 24])
ax_cpdos.set_ylim([-33, 33])

ax_cpdos.set_xlabel(r'E - E$_\mathrm{f}$ (eV)')
ax_cpdos.set_ylabel('DOS (arb units)')
ax_cpdos.legend(frameon=True)
fig_cpdos.tight_layout()
# fig_cpdos.savefig(filename_pdos_save, dpi=parameters.save_dpi, bbbox_inches='tight')

# Print HOMO alpha
LUMO = np.where(pdos_o_a['Occupation'].values==0)[0][0]
print('\nHOMO O_a(2p)', pdos_o_a_2p[LUMO-1],
      (pdos_o_a_2p[LUMO-1] / (pdos_o_a_all[LUMO-1]+pdos_fe_a_all[LUMO-1])) * 100, '%')
# print('HOMO O_a(2d)', pdos_o_a_2d[LUMO-1],
#       (pdos_o_a_2d[LUMO-1] / (pdos_o_a_all[LUMO-1]+pdos_fe_a_all[LUMO-1])) * 100, '%')
print('HOMO Fe_a(3d)', pdos_fe_a_3d[LUMO-1],
      (pdos_fe_a_3d[LUMO-1] / (pdos_o_a_all[LUMO-1]+pdos_fe_a_all[LUMO-1])) * 100, '%')
# print('HOMO Fe_a(3f)', pdos_fe_a_3f[LUMO-1],
#       (pdos_fe_a_3f[LUMO-1] / (pdos_o_a_all[LUMO-1]+pdos_fe_a_all[LUMO-1])) * 100, '%')

# Print HOMO beta
print('\nHOMO O_b(2p)', pdos_o_b_2p[LUMO-1],
      (pdos_o_b_2p[LUMO-1] / (pdos_o_b_all[LUMO-1]+pdos_fe_b_all[LUMO-1])) * 100, '%')
# print('HOMO O_b(2d)', pdos_o_b_2d[LUMO-1],
#       (pdos_o_b_2d[LUMO-1] / (pdos_o_b_all[LUMO-1]+pdos_fe_b_all[LUMO-1])) * 100, '%')
print('HOMO Fe_b(3d)', pdos_fe_b_3d[LUMO-1],
      (pdos_fe_b_3d[LUMO-1] / (pdos_o_b_all[LUMO-1]+pdos_fe_b_all[LUMO-1])) * 100, '%')
# print('HOMO Fe_b(3f)', pdos_fe_b_3f[LUMO-1],
#       (pdos_fe_b_3f[LUMO-1] / (pdos_o_b_all[LUMO-1]+pdos_fe_b_all[LUMO-1])) * 100, '%')

# Print LUMO alpha
print('\nLUMO O_a(2s)', pdos_o_a_2s[LUMO],
      (pdos_o_b_2s[LUMO] / (pdos_o_b_all[LUMO]+pdos_fe_b_all[LUMO])) * 100, '%')
print('LUMO O_a(2p)', pdos_o_a_2p[LUMO],
      (pdos_o_b_2p[LUMO] / (pdos_o_b_all[LUMO]+pdos_fe_b_all[LUMO])) * 100, '%')
# print('LUMO O_a(2d)', pdos_o_a_2d[LUMO],
#       (pdos_o_b_2d[LUMO] / (pdos_o_b_all[LUMO]+pdos_fe_b_all[LUMO])) * 100, '%')
print('LUMO Fe_a(3d)', pdos_fe_a_3d[LUMO],
      (pdos_fe_b_3d[LUMO] / (pdos_o_b_all[LUMO]+pdos_fe_b_all[LUMO])) * 100, '%')
# print('LUMO Fe_a(3f)', pdos_fe_a_3f[LUMO],
#       (pdos_fe_b_3f[LUMO] / (pdos_o_b_all[LUMO]+pdos_fe_b_all[LUMO])) * 100, '%')

# Print LUMO beta
print('\nLUMO O_b(2s)', pdos_o_b_2s[LUMO],
      (pdos_o_b_2s[LUMO] / (pdos_o_a_all[LUMO]+pdos_fe_a_all[LUMO])) * 100, '%')
print('LUMO O_b(2p)', pdos_o_b_2p[LUMO],
      (pdos_o_b_2p[LUMO] / (pdos_o_a_all[LUMO]+pdos_fe_a_all[LUMO])) * 100, '%')
# print('LUMO O_b(2d)', pdos_o_b_2d[LUMO],
#       (pdos_o_b_2d[LUMO] / (pdos_o_a_all[LUMO]+pdos_fe_a_all[LUMO])) * 100, '%')
print('LUMO Fe_b(3d)', pdos_fe_b_3d[LUMO],
      (pdos_fe_b_3d[LUMO] / (pdos_o_a_all[LUMO]+pdos_fe_a_all[LUMO])) * 100, '%')
# print('LUMO Fe_b(3f)', pdos_fe_b_3f[LUMO],
#       (pdos_fe_b_3f[LUMO] / (pdos_o_a_all[LUMO]+pdos_fe_a_all[LUMO])) * 100, '%')

# Plotting PDOS with dots
# fig_pdos, ax_pdos = plt.subplots()
# ax_pdos.plot(eigenvalues, pdos_o_a_2p.values, 'r.-', label='O(2p)')
# ax_pdos.plot(eigenvalues, -pdos_o_b_2p.values, 'r.-')
# ax_pdos.plot(eigenvalues, pdos_fe_a_3d.values, 'b.-', label='Fe(3d)')
# ax_pdos.plot(eigenvalues, -pdos_fe_b_3d.values, 'b.-')
# ax_pdos.plot(eigenvalues, pdos_o_a_all.values, 'grey', label='O')
# ax_pdos.plot(eigenvalues, pdos_fe_a_all.values, 'k', label='Fe')
# ax_pdos.set_xlim([x_lim[0], x_lim[1]])
# ax_pdos.set_xlabel(r'E - E$_\mathrm{f}$ (eV)')
# ax_pdos.set_ylabel('DOS (states / eV)')
# ax_pdos.legend(frameon=True)
# fig_pdos.tight_layout()


if __name__ == "__main__":
    print('Finished.')
    plt.show()