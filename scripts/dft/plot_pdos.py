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
# width = 0.3  # Gaussian width
# y_lim = 25  # Y axis limit
# x_lim = [-11, 9]  # x axis limit
width = 0.2 # Gaussian width, 0.2 works for m400, 0.1 for m800, 0.05 for m1200
y_lim = 180  # Y axis limit
x_lim = [-8, 6]  # x axis limit

# Filenames
folder = 'E:/University/PhD/Programming/dft_ml_md/output/fe_bulk'
filename_pdos_fea_a = '{}/pdos/neutral/hematite/fea_alpha.pdos'.format(folder)
filename_pdos_feb_a = '{}/pdos/neutral/hematite/feb_alpha.pdos'.format(folder)
filename_pdos_o_a = '{}/pdos/neutral/hematite/o_alpha.pdos'.format(folder)
filename_pdos_fea_b = '{}/pdos/neutral/hematite/fea_beta.pdos'.format(folder)
filename_pdos_feb_b = '{}/pdos/neutral/hematite/feb_beta.pdos'.format(folder)
filename_pdos_o_b = '{}/pdos/neutral/hematite/o_beta.pdos'.format(folder)
filename_pdos_save = '{}/pdos/images/neutral/hematite_oxygen_orbitals_zoom.png'.format(folder)

# Read Fermi energy from file
input_file = open(filename_pdos_fea_a, 'r')
line_first = input_file.readline().strip().split()
fermi = float(line_first[15])

# Read energy and DOS from files
labels_o = ['MO', 'Eigenvalue', 'Occupation', 's', 'py', 'pz', 'px', 'd-2', 'd-1', 'd0', 'd+1', 'd+2']
labels_fe = ['MO', 'Eigenvalue', 'Occupation', 's', 'py', 'pz', 'px', 'd-2', 'd-1', 'd0', 'd+1', 'd+2', 'f-3', 'f-2',
             'f-1', 'f0', 'f+1', 'f+2', 'f+3']
pdos_fea_a = pd.read_csv(filename_pdos_fea_a, names=labels_fe, skiprows=[0, 1], delim_whitespace=True)
pdos_feb_a = pd.read_csv(filename_pdos_feb_a, names=labels_fe, skiprows=[0, 1], delim_whitespace=True)
pdos_o_a = pd.read_csv(filename_pdos_o_a, names=labels_o, skiprows=[0, 1], delim_whitespace=True)
pdos_fea_b = pd.read_csv(filename_pdos_fea_b, names=labels_fe, skiprows=[0, 1], delim_whitespace=True)
pdos_feb_b = pd.read_csv(filename_pdos_feb_b, names=labels_fe, skiprows=[0, 1], delim_whitespace=True)
pdos_o_b = pd.read_csv(filename_pdos_o_b, names=labels_o, skiprows=[0, 1], delim_whitespace=True)

# Calculate PDOS for OH
# filename_pdos_oh_a = '/scratch/cahart/work/personal_files/dft_ml_md/output/fe_bulk/pdos/hole/hematite/neutral/oh_alpha.pdos'
# filename_pdos_oh_b = '/scratch/cahart/work/personal_files/dft_ml_md/output/fe_bulk/pdos/hole/hematite/neutral/oh_beta.pdos'
# pdos_oh_a = pd.read_csv(filename_pdos_oh_a, names=labels_o, skiprows=[0, 1], delim_whitespace=True)
# pdos_oh_b = pd.read_csv(filename_pdos_oh_b, names=labels_o, skiprows=[0, 1], delim_whitespace=True)
# pdos_oh_a_2s = sum_s(pdos_oh_a, labels_o)
# pdos_oh_b_2s = sum_s(pdos_oh_b, labels_o)
# pdos_oh_a_2p = sum_p(pdos_oh_a, labels_o)
# pdos_oh_b_2p = sum_p(pdos_oh_b, labels_o)
# pdos_oh_a_all = sum_all(pdos_oh_a, labels_o)
# pdos_oh_b_all = sum_all(pdos_oh_b, labels_o)

# Calculate PDOS for Fe_h
filename_pdos_feh_a = '{}/pdos/hole/hematite/relaxed2/feh_alpha.pdos'.format(folder)
filename_pdos_feh_b = '{}/pdos/hole/hematite/relaxed2/feh_beta.pdos'.format(folder)
pdos_feh_a = pd.read_csv(filename_pdos_feh_a, names=labels_fe, skiprows=[0, 1], delim_whitespace=True)
pdos_feh_b = pd.read_csv(filename_pdos_feh_b, names=labels_fe, skiprows=[0, 1], delim_whitespace=True)
pdos_feh_a_3s = sum_s(pdos_feh_a, labels_fe)
pdos_feh_b_3s = sum_s(pdos_feh_b, labels_fe)
pdos_feh_a_3p = sum_p(pdos_feh_a, labels_fe)
pdos_feh_b_3p = sum_p(pdos_feh_b, labels_fe)
pdos_feh_a_3d = sum_d(pdos_feh_a, labels_fe)
pdos_feh_b_3d = sum_d(pdos_feh_b, labels_fe)
pdos_feh_a_3f = sum_f(pdos_feh_a, labels_fe)
pdos_feh_b_3f = sum_f(pdos_feh_b, labels_fe)
pdos_feh_a_all = sum_all(pdos_feh_a, labels_fe)
pdos_feh_b_all = sum_all(pdos_feh_b, labels_fe)

input_file = open(filename_pdos_feh_a, 'r')
line_first = input_file.readline().strip().split()
fermi2 = float(line_first[15])

# Calculate PDOS for Fe (summing Fe_a, Fe_b for both spin channels)
pdos_fe_a_3s = sum_s(pdos_fea_a, labels_fe) + sum_s(pdos_feb_a, labels_fe)
pdos_fe_b_3s = sum_s(pdos_fea_b, labels_fe) + sum_s(pdos_feb_b, labels_fe)
pdos_fe_a_3p = sum_p(pdos_fea_a, labels_fe) + sum_p(pdos_feb_a, labels_fe)
pdos_fe_b_3p = sum_p(pdos_fea_b, labels_fe) + sum_p(pdos_feb_b, labels_fe)
pdos_fe_a_3d = sum_d(pdos_fea_a, labels_fe) + sum_d(pdos_feb_a, labels_fe)
pdos_fe_b_3d = sum_d(pdos_fea_b, labels_fe) + sum_d(pdos_feb_b, labels_fe)
pdos_fe_a_3f = sum_f(pdos_fea_a, labels_fe) + sum_f(pdos_feb_a, labels_fe)
pdos_fe_b_3f = sum_f(pdos_fea_b, labels_fe) + sum_f(pdos_feb_b, labels_fe)
pdos_fe_a_all = sum_all(pdos_fea_a, labels_fe) + sum_all(pdos_feb_a, labels_fe)
pdos_fe_b_all = sum_all(pdos_fea_b, labels_fe) + sum_all(pdos_feb_b, labels_fe)

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

# Calculate convoluted PDOS
eigenvalues_h = (pdos_feh_a['Eigenvalue'] - fermi) * parameters.hartree_to_ev
energy_grid_h = np.linspace(np.min(eigenvalues_h), np.max(eigenvalues_h), num=num_points)

# Plotting convoluted PDOS
fig_cpdos, ax_cpdos = plt.subplots()

# Hematite
ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_o_a_2s.values, energy_grid, width), 'b', label='O(2s)')
ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_o_b_2s.values, energy_grid, width), 'b')

ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_o_a_2p.values, energy_grid, width), 'r', label='O(2p)')
ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_o_b_2p.values, energy_grid, width), 'r')

ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_o_a_2d.values, energy_grid, width), 'g', label='O(2d)')
ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_o_b_2d.values, energy_grid, width), 'g')


# Lepidocrocite, goethite
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_o_a_2p.values, energy_grid, width)+smearing(eigenvalues, pdos_oh_a_2p.values, energy_grid, width),
#               'r', label='O(2p) total')
# ax_cpdos.plot(energy_grid, -smearing(eigenvalues, pdos_o_b_2p.values, energy_grid, width)-smearing(eigenvalues, pdos_oh_b_2p.values, energy_grid, width), 'r')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_oh_a_2p.values, energy_grid, width), 'g', label='O(2p) H donor')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_oh_b_2p.values, energy_grid, width), 'g')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_o_a_2p.values, energy_grid, width), 'm', label='O(2p) H acceptor')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_o_b_2p.values, energy_grid, width), 'm')

# White rust
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_o_a_2s.values, energy_grid, width), 'g', label='O(2s)')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_fe_a_3s.values, energy_grid, width), 'm', label='Fe(3s+4s)')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_fe_b_3s.values, energy_grid, width), 'm')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_o_b_2s.values, energy_grid, width), 'g')

# All
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_fe_a_3s.values, energy_grid, width), 'r', label='Fe(3s)')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_fe_b_3s.values, energy_grid, width), 'r')
#
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_fe_a_3p.values, energy_grid, width), 'g', label='Fe(3p)')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_fe_b_3p.values, energy_grid, width), 'g')
#
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_fe_a_3d.values, energy_grid, width), 'b', label='Fe(3d)')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_fe_b_3d.values, energy_grid, width), 'b')
#
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_fe_a_3f.values, energy_grid, width), 'y', label='Fe(3f)')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, -pdos_fe_b_3f.values, energy_grid, width), 'y')

# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_fe_a_all.values+pdos_o_a_all.values, energy_grid, width), 'k', label='Total')


# Hole
# ax_cpdos.plot(energy_grid_h, 10 * smearing(eigenvalues_h, pdos_feh_a_3d.values, energy_grid, width), 'k', label=r'Fe$_{\rm h}$(3d) x 10')
# ax_cpdos.plot(energy_grid_h, 10 * smearing(eigenvalues_h, -pdos_feh_b_3d.values, energy_grid, width), 'k')

# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_o_a_all.values, energy_grid, width), 'grey', label='O')
# ax_cpdos.plot(energy_grid, smearing(eigenvalues, pdos_fe_a_all.values, energy_grid, width), 'k', label='Fe')
# ax_cpdos.set_xlim([x_lim[0], x_lim[1]])
ax_cpdos.set_xlim([1.8, 4.6])
# ax_cpdos.set_ylim([-y_lim, y_lim])
ax_cpdos.set_ylim([-50, 50])
ax_cpdos.set_xlabel(r'E - E$_\mathrm{f}$ (eV)')
ax_cpdos.set_ylabel('DOS (arb units)')
ax_cpdos.legend(frameon=True)
fig_cpdos.tight_layout()
fig_cpdos.savefig(filename_pdos_save, dpi=parameters.save_dpi, bbbox_inches='tight')

# Print HOMO and LUMO DOS (hematite)
LUMO = np.where(pdos_o_a['Occupation'].values==0)[0][0]
print('HOMO O_a(2p)', pdos_o_a_2p[LUMO-1],
      (pdos_o_a_2p[LUMO-1] / (pdos_o_a_all[LUMO-1]+pdos_fe_a_all[LUMO-1])) * 100, '%')
print('HOMO O_a(2d)', pdos_o_a_2d[LUMO-1],
      (pdos_o_a_2d[LUMO-1] / (pdos_o_a_all[LUMO-1]+pdos_fe_a_all[LUMO-1])) * 100, '%')

print('HOMO Fe_a(3d)', pdos_fe_a_3d[LUMO-1],
      (pdos_fe_a_3d[LUMO-1] / (pdos_o_a_all[LUMO-1]+pdos_fe_a_all[LUMO-1])) * 100, '%')
print('HOMO Fe_a(3f)', pdos_fe_a_3f[LUMO-1],
      (pdos_fe_a_3f[LUMO-1] / (pdos_o_a_all[LUMO-1]+pdos_fe_a_all[LUMO-1])) * 100, '%')

print('LUMO O_a(2p)', pdos_o_a_2p[LUMO],
      (pdos_o_a_2p[LUMO] / (pdos_o_a_all[LUMO]+pdos_fe_a_all[LUMO])) * 100, '%')
print('LUMO O_a(2d)', pdos_o_a_2d[LUMO],
      (pdos_o_a_2d[LUMO] / (pdos_o_a_all[LUMO]+pdos_fe_a_all[LUMO])) * 100, '%')

print('LUMO Fe_a(3d)', pdos_fe_a_3d[LUMO],
      (pdos_fe_a_3d[LUMO] / (pdos_o_a_all[LUMO]+pdos_fe_a_all[LUMO])) * 100, '%')
print('LUMO Fe_a(3f)', pdos_fe_a_3f[LUMO],
      (pdos_fe_a_3f[LUMO] / (pdos_o_a_all[LUMO]+pdos_fe_a_all[LUMO])) * 100, '%')
# Print HOMO and LUMO DOS Oh (lepidocrocite, goethite)
# print('HOMO Oh_a(2p)', pdos_oh_a_2p[LUMO-1],
#       (pdos_oh_a_2p[LUMO-1] / (pdos_o_a_all[LUMO-1]+pdos_fe_a_all[LUMO-1]+pdos_oh_a_all[LUMO-1])) * 100, '%')
# print('HOMO O_a(2p)', pdos_o_a_2p[LUMO-1],
#       (pdos_o_a_2p[LUMO-1] / (pdos_o_a_all[LUMO-1]+pdos_fe_a_all[LUMO-1]+pdos_oh_a_all[LUMO-1])) * 100, '%')
# print('HOMO Fe_a(3d)', pdos_fe_a_3d[LUMO-1],
#       (pdos_fe_a_3d[LUMO-1] / (pdos_o_a_all[LUMO-1]+pdos_fe_a_all[LUMO-1]+pdos_oh_a_all[LUMO-1])) * 100, '%')
# 
# print('\nLUMO Oh_a(2p)', pdos_oh_a_2p[LUMO],
#       (pdos_oh_a_2p[LUMO] / (pdos_o_a_all[LUMO]+pdos_fe_a_all[LUMO]+pdos_oh_a_all[LUMO])) * 100, '%')
# print('LUMO O_a(2p)', pdos_o_a_2p[LUMO],
#       (pdos_o_a_2p[LUMO] / (pdos_o_a_all[LUMO]+pdos_fe_a_all[LUMO]+pdos_oh_a_all[LUMO])) * 100, '%')
# print('LUMO Fe_a(3d)', pdos_fe_a_3d[LUMO],
#       (pdos_fe_a_3d[LUMO] / (pdos_o_a_all[LUMO]+pdos_fe_a_all[LUMO]+pdos_oh_a_all[LUMO])) * 100, '%')


# # White rust
# print('HOMO O_a(2s)', pdos_o_a_2s[LUMO-1],
#       (pdos_o_a_2s[LUMO-1] / (pdos_o_a_all[LUMO-1]+pdos_fe_a_all[LUMO-1])) * 100, '%')
# print('HOMO Fe_a(3s+4s)', pdos_fe_a_3s[LUMO-1],
#       (pdos_fe_a_3s[LUMO-1] / (pdos_o_a_all[LUMO-1]+pdos_fe_a_all[LUMO-1])) * 100, '%')

# print('LUMO O_a(2s)', pdos_o_a_2s[LUMO],
#       (pdos_o_a_2s[LUMO] / (pdos_o_a_all[LUMO]+pdos_fe_a_all[LUMO])) * 100, '%')
# print('LUMO Fe_a(3s+4s)', pdos_fe_a_3s[LUMO],
#       (pdos_fe_a_3s[LUMO] / (pdos_o_a_all[LUMO]+pdos_fe_a_all[LUMO])) * 100, '%')


# Plotting PDOS with dots
fig_pdos, ax_pdos = plt.subplots()
# ax_pdos.plot(eigenvalues_h, pdos_feh_a_all.values, 'k.-', label='Feh(all)')
# ax_pdos.plot(eigenvalues_h, -pdos_feh_b_all.values, 'k.-')
# ax_pdos.plot(eigenvalues, pdos_o_a_2s.values, 'g.-', label='O(2s)')
# ax_cpdos.plot(eigenvalues, pdos_o_a_2p.values+pdos_oh_a_2p.values, 'r.-', label='O(2p)')
ax_pdos.plot(eigenvalues, pdos_o_a_2p.values, 'r.-', label='O(2p)')
ax_pdos.plot(eigenvalues, -pdos_o_b_2p.values, 'r.-')
ax_pdos.plot(eigenvalues, pdos_fe_a_3d.values, 'b.-', label='Fe(3d)')
ax_pdos.plot(eigenvalues, -pdos_fe_b_3d.values, 'b.-')
# ax_pdos.plot(eigenvalues, pdos_fe_a_3s.values, '.-', label='Fe(3s)')
ax_pdos.plot(eigenvalues, pdos_o_a_all.values, 'grey', label='O')
ax_pdos.plot(eigenvalues, pdos_fe_a_all.values, 'k', label='Fe')
ax_pdos.set_xlim([x_lim[0], x_lim[1]])
ax_pdos.set_xlabel(r'E - E$_\mathrm{f}$ (eV)')
ax_pdos.set_ylabel('DOS (states / eV)')
ax_pdos.legend(frameon=True)
fig_pdos.tight_layout()

# Hematite:
# HOMO O_a(2p) 0.5896086899999999 58.9608684103913 %
# HOMO Fe_a(3d) 0.38722654 38.72265361277346 %
# LUMO O_a(2p) 0.0878855 8.788550087885499 %
# LUMO Fe_a(3d) 0.86825897 86.82589786825896 %

# Hematite SCAN m800:
# HOMO O_a(2p) 0.62398273 62.39827611991381 %
# HOMO Fe_a(3d) 0.33720397 33.720398686019934 %
# LUMO O_a(2p) 0.07299632 7.299631854007364 %
# LUMO Fe_a(3d) 0.8789672 87.89671824206565 %

# Hematite SCAN m1200:
# HOMO O_a(2p) 0.62428895 62.4288925028443 %
# HOMO Fe_a(3d) 0.33700062999999997 33.70006165199753 %
# LUMO O_a(2p) 0.0707903 7.07903 %
# LUMO Fe_a(3d) 0.88009414 88.00941399999999 %

# Hematite SCAN m1200 (GTH_SCAN):
# HOMO O_a(2p) 0.6347505 63.47505190425156 %
# HOMO Fe_a(3d) 0.32738408 32.73840898215227 %
# LUMO O_a(2p) 0.07157471 7.157470928425291 %
# LUMO Fe_a(3d) 0.88134655 88.13465411865346 %

# Lepidocrocite:
# HOMO Oh_a(2p) 0.06260264 6.27828840095318 %
# HOMO O_a(2p) 0.60783075 60.95808016191765 %
# HOMO Fe_a(3d) 0.28757310999999997 28.840108355478826 %
#
# LUMO Oh_a(2p) 0.01492429 1.5012156300951027 %
# LUMO O_a(2p) 0.02521673 2.536519272668119 %
# LUMO Fe_a(3d) 0.91025092 91.56099944536369 %


# Goethite:
# HOMO Oh_a(2p) 0.31255295 31.393311298058336 %
# HOMO O_a(2p) 0.38339808999999997 38.509108906030114 %
# HOMO Fe_a(3d) 0.28745152999999996 28.87208507996214 %

# LUMO Oh_a(2p) 0.00537064 0.5411452634625079 %
# LUMO O_a(2p) 0.0484422 4.881032257180429 %
# LUMO Fe_a(3d) 0.88722966 89.39719067233166 %


# White rust
# HOMO O_a(2s) 0.00163237 0.16542400124234452 %
# HOMO O_a(2p) 0.02022236 2.0493293222511677 %
# HOMO Fe_a(3s+4s) 0.00305662 0.3097571694391438 %
# HOMO Fe_a(3d) 0.95023123 96.29621481161413 %

# LUMO O_a(2s) 0.3889501 55.80537203220272 %
# LUMO O_a(2p) 0.0078416 1.1250888104353767 %
# LUMO Fe_a(3s+4s) 0.26551889 38.09583912699214 %
# LUMO Fe_a(3d) 0.03234948 4.641404556646986 %


if __name__ == "__main__":
    print('Finished.')
    plt.show()
