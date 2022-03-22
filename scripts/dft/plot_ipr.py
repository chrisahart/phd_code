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


def calc_ipr(a_4, a_2):
    
    return a_4 / a_2 ** 2


# Plotting
x_lim = [-8, 6]
y_lim = 0.3
labels_sum = ['MO', 'Eigenvalue', 'Occupation', 'Sum']
# fermi = 0.281628  # Neutral 331 hematite
fermi = 0.168483  # Neutral 613 lepidocrocite
# fermi = 0.201845  # Neutral 316 goethite

# Folders
folder_in1 = '/scratch/cahart/work/personal_files/fe_bulk/pdos/all_label/polaron/lepidocrocite_hse_pdos_fine_all_electron'
folder_in2 = '/scratch/cahart/work/personal_files/fe_bulk/pdos/all_label/lepidocrocite_hse_pdos_fine_all'
folder_out = '/scratch/cahart/work/personal_files/dft_ml_md/output/fe_bulk/ipr/lepidocrocite'
save_name = 'polaron_electron'

# 1
sum_1_alpha_power2 = pd.read_csv('{}/ALPHA_power2.out'.format(folder_in1), names=labels_sum, skiprows=[0, 1], delim_whitespace=True)
sum_1_alpha_power4 = pd.read_csv('{}/ALPHA_power4.out'.format(folder_in1), names=labels_sum, skiprows=[0, 1], delim_whitespace=True)
sum_1_beta_power2 = pd.read_csv('{}/beta_power2.out'.format(folder_in1), names=labels_sum, skiprows=[0, 1], delim_whitespace=True)
sum_1_beta_power4 = pd.read_csv('{}/beta_power4.out'.format(folder_in1), names=labels_sum, skiprows=[0, 1], delim_whitespace=True)

# 2
sum_2_alpha_power2 = pd.read_csv('{}/ALPHA_power2.out'.format(folder_in2), names=labels_sum, skiprows=[0, 1], delim_whitespace=True)
sum_2_alpha_power4 = pd.read_csv('{}/ALPHA_power4.out'.format(folder_in2), names=labels_sum, skiprows=[0, 1], delim_whitespace=True)
sum_2_beta_power2 = pd.read_csv('{}/beta_power2.out'.format(folder_in2), names=labels_sum, skiprows=[0, 1], delim_whitespace=True)
sum_2_beta_power4 = pd.read_csv('{}/beta_power4.out'.format(folder_in2), names=labels_sum, skiprows=[0, 1], delim_whitespace=True)

# Grid
num_points = (sum_1_alpha_power2['Eigenvalue']).shape[0]
eigenvalues_1 = (sum_1_alpha_power2['Eigenvalue'] - fermi) * parameters.hartree_to_ev
eigenvalues_2 = (sum_2_alpha_power2['Eigenvalue'] - fermi) * parameters.hartree_to_ev

# Calculate total IPR
ipr_1_alpha = calc_ipr(sum_1_alpha_power4['Sum'],  sum_1_alpha_power2['Sum'])
ipr_1_beta = calc_ipr(sum_1_beta_power4['Sum'],  sum_1_beta_power2['Sum'])
ipr_2_alpha = calc_ipr(sum_2_alpha_power4['Sum'],  sum_2_alpha_power2['Sum'])
ipr_2_beta = calc_ipr(sum_2_beta_power4['Sum'],  sum_2_beta_power2['Sum'])

# Plot total IPR
fig_ipr, ax_ipr = plt.subplots()
ax_ipr.vlines(eigenvalues_1, 0, ipr_1_alpha, 'k', linestyles="-", linewidths=0.8, label='Relaxed')
ax_ipr.vlines(eigenvalues_1, 0, -ipr_1_beta, 'k', linestyles="-", linewidths=0.8, label='')
ax_ipr.vlines(eigenvalues_2, 0, ipr_2_alpha, 'b', linestyles="-", linewidths=0.8, label='Neutral')
ax_ipr.vlines(eigenvalues_2, 0, -ipr_2_beta, 'b', linestyles="-", linewidths=0.8, label='')
ax_ipr.set_xlim([x_lim[0], x_lim[1]])
ax_ipr.set_ylim([-y_lim, y_lim])
ax_ipr.set_xlabel(r'E - E$_\mathrm{f}$ (eV)')
ax_ipr.set_ylabel('IPR (arb units)')
ax_ipr.legend(frameon=True)
fig_ipr.tight_layout()
fig_ipr.savefig('{}/{}.png'.format(folder_out, save_name), dpi=parameters.save_dpi, bbbox_inches='tight')
ax_ipr.set_ylim([-1, 1])
fig_ipr.tight_layout()
fig_ipr.savefig('{}/{}_ylim1.png'.format(folder_out, save_name), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
