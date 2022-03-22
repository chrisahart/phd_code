from __future__ import division, print_function
import time
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from scripts.general import functions
from scripts.general import parameters

"""
    Plot .pdos file from CP2K
"""


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
width = 0.2  # Gaussian width
x_lim = [-8, 6]  # x axis limit
atoms = 36  # 36 Fe atoms in

# Filenames
filename_pdos_save = '/scratch/cahart/work/personal_files/dft_ml_md/output/fe_bulk/pdos/images/polaron_single/hole_hematite.png'

# Read energy and DOS from files
labels_o = ['MO', 'Eigenvalue', 'Occupation', 's', 'py', 'pz', 'px', 'd-2', 'd-1', 'd0', 'd+1', 'd+2']
labels_fe = ['MO', 'Eigenvalue', 'Occupation', 's', 'py', 'pz', 'px', 'd-2', 'd-1', 'd0', 'd+1', 'd+2', 'f-3', 'f-2',
             'f-1', 'f0', 'f+1', 'f+2', 'f+3']

# Calculate PDOS for Fe neutral
filename_pdos_fean_a = '/scratch/cahart/work/personal_files/dft_ml_md/output/fe_bulk/pdos/hole/hematite/neutral/feah_alpha.pdos'
filename_pdos_fean_b = '/scratch/cahart/work/personal_files/dft_ml_md/output/fe_bulk/pdos/hole/hematite/neutral/feah_beta.pdos'
pdos_fean_a = pd.read_csv(filename_pdos_fean_a, names=labels_fe, skiprows=[0, 1], delim_whitespace=True)
pdos_fean_b = pd.read_csv(filename_pdos_fean_b, names=labels_fe, skiprows=[0, 1], delim_whitespace=True)
pdos_fenut_a_3s = sum_s(pdos_fean_a, labels_fe)
pdos_fenut_b_3s = sum_s(pdos_fean_b, labels_fe)
pdos_fenut_a_3p = sum_p(pdos_fean_a, labels_fe)
pdos_fenut_b_3p = sum_p(pdos_fean_b, labels_fe)
pdos_fenut_a_3d = sum_d(pdos_fean_a, labels_fe)
pdos_fenut_b_3d = sum_d(pdos_fean_b, labels_fe)
pdos_fenut_a_3f = sum_f(pdos_fean_a, labels_fe)
pdos_fenut_b_3f = sum_f(pdos_fean_b, labels_fe)

# Read Fermi energy from file
input_file = open(filename_pdos_fean_a, 'r')
line_first = input_file.readline().strip().split()
fermi = float(line_first[15])

# Calculate PDOS for Fe hole vertical
filename_pdos_feav_a = '/scratch/cahart/work/personal_files/dft_ml_md/output/fe_bulk/pdos/hole/hematite/vertical/feah_alpha.pdos'
filename_pdos_feav_b = '/scratch/cahart/work/personal_files/dft_ml_md/output/fe_bulk/pdos/hole/hematite/vertical/feah_beta.pdos'
pdos_feav_a = pd.read_csv(filename_pdos_feav_a, names=labels_fe, skiprows=[0, 1], delim_whitespace=True)
pdos_feav_b = pd.read_csv(filename_pdos_feav_b, names=labels_fe, skiprows=[0, 1], delim_whitespace=True)
pdos_fehv_a_3s = sum_s(pdos_feav_a, labels_fe)
pdos_fehv_b_3s = sum_s(pdos_feav_b, labels_fe)
pdos_fehv_a_3p = sum_p(pdos_feav_a, labels_fe)
pdos_fehv_b_3p = sum_p(pdos_feav_b, labels_fe)
pdos_fehv_a_3d = sum_d(pdos_feav_a, labels_fe)
pdos_fehv_b_3d = sum_d(pdos_feav_b, labels_fe)
pdos_fehv_a_3f = sum_f(pdos_feav_a, labels_fe)
pdos_fehv_b_3f = sum_f(pdos_feav_b, labels_fe)

# Calculate PDOS for Fe hole relaxed
filename_pdos_fearh_a = '/scratch/cahart/work/personal_files/dft_ml_md/output/fe_bulk/pdos/hole/hematite/relaxed/feh_alpha.pdos'
filename_pdos_fearh_b = '/scratch/cahart/work/personal_files/dft_ml_md/output/fe_bulk/pdos/hole/hematite/relaxed/feh_beta.pdos'
pdos_fearh_a = pd.read_csv(filename_pdos_fearh_a, names=labels_fe, skiprows=[0, 1], delim_whitespace=True)
pdos_fearh_b = pd.read_csv(filename_pdos_fearh_b, names=labels_fe, skiprows=[0, 1], delim_whitespace=True)
pdos_fehr_a_3s = sum_s(pdos_fearh_a, labels_fe)
pdos_fehr_b_3s = sum_s(pdos_fearh_b, labels_fe)
pdos_fehr_a_3p = sum_p(pdos_fearh_a, labels_fe)
pdos_fehr_b_3p = sum_p(pdos_fearh_b, labels_fe)
pdos_fehr_a_3d = sum_d(pdos_fearh_a, labels_fe)
pdos_fehr_b_3d = sum_d(pdos_fearh_b, labels_fe)
pdos_fehr_a_3f = sum_f(pdos_fearh_a, labels_fe)
pdos_fehr_b_3f = sum_f(pdos_fearh_b, labels_fe)

# Calculate convoluted PDOS
eigenvalues_n = (pdos_fean_a['Eigenvalue'] - fermi) * parameters.hartree_to_ev
energy_grid_n = np.linspace(np.min(eigenvalues_n), np.max(eigenvalues_n), num=(pdos_fean_a['s']).shape[0])
eigenvalues_v = (pdos_feav_a['Eigenvalue'] - fermi) * parameters.hartree_to_ev
energy_grid_v = np.linspace(np.min(eigenvalues_v), np.max(eigenvalues_v), num=(pdos_feav_a['s']).shape[0])
eigenvalues_r = (pdos_fearh_a['Eigenvalue'] - fermi) * parameters.hartree_to_ev
energy_grid_r = np.linspace(np.min(eigenvalues_r), np.max(eigenvalues_r), num=(pdos_fearh_a['s']).shape[0])

# Plotting convoluted PDOS
fig_cpdos, ax_cpdos = plt.subplots()

# No smearing
# ax_cpdos.plot(eigenvalues, pdos_fenut_a_3d.values, 'rx', label='Neutral')

# Charged
ax_cpdos.plot(energy_grid_n, smearing(eigenvalues_n, pdos_fenut_a_3d.values, energy_grid_n, width)/atoms, 'r', label='Neutral')
ax_cpdos.plot(energy_grid_n, smearing(eigenvalues_n, -pdos_fenut_b_3d.values, energy_grid_n, width)/atoms, 'r')

ax_cpdos.plot(energy_grid_v, smearing(eigenvalues_v, pdos_fehv_a_3d.values, energy_grid_v, width)/atoms, 'g', label='Vertical')
ax_cpdos.plot(energy_grid_v, smearing(eigenvalues_v, -pdos_fehv_b_3d.values, energy_grid_v, width)/atoms, 'g')

ax_cpdos.plot(energy_grid_r, smearing(eigenvalues_r, pdos_fehr_a_3d.values, energy_grid_r, width)/atoms, 'b', label='Relaxed')
ax_cpdos.plot(energy_grid_r, smearing(eigenvalues_r, -pdos_fehr_b_3d.values, energy_grid_r, width)/atoms, 'b')

ax_cpdos.set_xlim([x_lim[0], x_lim[1]])
# ax_cpdos.set_ylim([-2, 5])  # Hole
# ax_cpdos.set_ylim([-2, 5])  # Electron
ax_cpdos.set_xlabel(r'E - E$_\mathrm{f}$ (eV)')
ax_cpdos.set_ylabel('DOS (states / eV)')
ax_cpdos.legend(frameon=True)
fig_cpdos.tight_layout()
fig_cpdos.savefig(filename_pdos_save, dpi=parameters.save_dpi, bbbox_inches='tight')


if __name__ == "__main__":
    print('Finished.')
    plt.show()
