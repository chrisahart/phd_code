from __future__ import division, print_function
import time
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from scripts.general import functions
from scripts.general import parameters
from scripts.formatting import load_coordinates
from scripts.formatting import load_energy
from scripts.formatting import load_forces_out
from scripts.formatting import load_forces
# from scripts.dft import cdft_beta


"""
    Plot energy and forces for hematite interface 
"""

def read_hirsh(folder, filename, num_atoms, filename_brent, filename_mnbrack):
    """
    Read Hirshfeld
    """

    cols_hirsh = ['Atom', 'Element', 'Kind', 'Ref Charge', 'Pop 1', 'Pop 2', 'Spin', 'Charge']
    data_hirsh = pd.read_csv('{}{}'.format(folder, filename), names=cols_hirsh, delim_whitespace=True)
    species = data_hirsh['Element']
    data_hirsh = data_hirsh.apply(pd.to_numeric, errors='coerce')
    num_data = int(np.floor((len(data_hirsh) + 1) / (num_atoms + 2)))
    step = np.linspace(start=0, stop=(num_data - 1), num=num_data, dtype=int)
    brent = np.zeros(num_data)
    mnbrack = np.zeros(num_data)

    if filename_brent:
        cols_brent = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        data_brent = pd.read_csv('{}{}'.format(folder, filename_brent), names=cols_brent, delim_whitespace=True)
        data_mnbrack = pd.read_csv('{}{}'.format(folder, filename_mnbrack), names=cols_brent, delim_whitespace=True)
        brent = data_brent['9']
        mnbrack = data_mnbrack['9']
        num_data = len(brent)
        step = np.linspace(start=0, stop=(num_data - 1), num=num_data, dtype=int)

    return data_hirsh, species, num_data, step, brent, mnbrack

atoms = 435

folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/electron/hawk/hse/nve/cdft-polaron/analysis/'
energy_kinetic1_1, energy_potential1_1, energy_total1_1, temperature1_1, time_val1_1, time_per_step1_1 = load_energy.load_values_energy(folder_1, '/energy/dft.out')
file_spec1_1, species1_1, num_data1_1, step1_1, brent1_1, mnbrack1_1 = read_hirsh(folder_1, '/hirshfeld/dft.out', atoms, None, None)

folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/electron/hawk/hse/nve/cdft-polaron/analysis/'
energy_kinetic1_2, energy_potential1_2, energy_total1_2, temperature1_2, time_val1_2, time_per_step1_2 = load_energy.load_values_energy(folder_2, 'energy/single-96-3p8158_cdft-1e-2.out')
energy_kinetic2_2, energy_potential2_2, energy_total2_2, temperature2_2, time_val2_2, time_per_step2_2 = load_energy.load_values_energy(folder_2, 'energy/single-96-3p8158_cdft-1e-3.out')
energy_kinetic3_2, energy_potential3_2, energy_total3_2, temperature3_2, time_val3_2, time_per_step3_2 = load_energy.load_values_energy(folder_2, 'energy/single-96-3p8158_cdft-1e-4.out')
file_spec1_2, species1_2, num_data1_2, step1_2, brent1_2, mnbrack1_2 = read_hirsh(folder_2, '/hirshfeld/single-96-3p8158_cdft-1e-2.out', atoms, None, None)
file_spec2_2, species2_2, num_data2_2, step2_2, brent2_2, mnbrack2_2 = read_hirsh(folder_2, '/hirshfeld/single-96-3p8158_cdft-1e-3.out', atoms, None, None)

index_fe_2 = np.array([96, 134]) - 1
folder_save = folder_2

# Plot total energy
time_plot = 80
energy_end = time_plot*2
fig_energy, ax_energy = plt.subplots()
ax_energy.plot(time_val1_1-time_val1_1[0], (energy_total1_1-energy_total1_1[0])/atoms, 'k-', label='DFT')
ax_energy.plot(time_val1_2-time_val1_2[0], (energy_total1_2-energy_total1_2[0])/atoms, 'r-', label='CDFT 1e-2')
ax_energy.plot(time_val2_2-time_val2_2[0], (energy_total2_2-energy_total2_2[0])/atoms, 'g-', label='CDFT 1e-3')
ax_energy.plot(time_val3_2-time_val3_2[0], (energy_total3_2-energy_total3_2[0])/atoms, 'b-', label='CDFT 1e-4')
ax_energy.set_xlabel('Time / fs')
ax_energy.set_ylabel('Energy change per atom / Ha')
ax_energy.set_xlim([0, time_plot])
# ax_energy.set_ylim([-1e-6, 4e-6])
ax_energy.set_ylim([-1e-6, 1e-5])
ax_energy.legend(frameon=False)
fig_energy.tight_layout()
fig_energy.savefig('{}/energy_{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot Hirshfeld analysis of selected atoms
# time_plot = 500
skip = 0
skip_line = 2
plot_index = index_fe_2
plot_quantity = 'Spin'
fig_hirshfeld, ax_hirshfeld = plt.subplots()
temp1 = np.zeros(num_data1_1)
temp2 = np.zeros(num_data1_1)
temp3 = np.zeros(num_data1_1)
i = -1
for n in range(num_data1_1):
    i = i + 1
    temp1[n] = (file_spec1_1.loc[atoms * i + skip_line * i + plot_index[0], plot_quantity])
    temp2[n] = (file_spec1_1.loc[atoms * i + skip_line * i + plot_index[1], plot_quantity])
    # temp3[n] = (file_spec1_1.loc[atoms * i + skip_line * i + plot_index[3], plot_quantity])
ax_hirshfeld.plot(time_val1_1[skip:]-time_val1_1[skip], temp1[skip:-1], 'k-', label='DFT')
ax_hirshfeld.plot(time_val1_1[skip:]-time_val1_1[skip], temp2[skip:-1], 'k-', alpha=0.4)
# ax_hirshfeld.plot(time_val1_1[skip:]-time_val1_1[skip], temp3[skip:], 'k-', alpha=0.4)
ax_hirshfeld.plot(time_val1_1[skip:] - time_val1_1[skip], (temp1[skip:-1]+temp2[skip:-1])/2, 'k--')
temp1 = np.zeros(num_data1_2)
temp2 = np.zeros(num_data1_2)
temp3 = np.zeros(num_data1_2)
i = -1
for n in range(num_data1_2):
    i = i + 1
    temp1[n] = (file_spec1_2.loc[atoms * i + skip_line * i + plot_index[0], 'Spin'])
    temp2[n] = (file_spec1_2.loc[atoms * i + skip_line * i + plot_index[1], 'Spin'])
    # temp3[n] = (file_spec1_2.loc[atoms * i + skip_line * i + plot_index[2], 'Spin'])
ax_hirshfeld.plot(time_val1_2[skip:] - time_val1_2[skip], temp1[skip:-1], 'r-', label='CDFT 3.72 3e-3')
ax_hirshfeld.plot(time_val1_2[skip:] - time_val1_2[skip], temp2[skip:-1], 'r-', alpha=0.4)
# ax_hirshfeld.plot(time_val1_2[skip:] - time_val1_2[skip], temp3[skip:], 'r-', alpha=0.4)
ax_hirshfeld.plot(time_val1_2[skip:] - time_val1_2[skip], (temp1[skip:-1]+temp2[skip:-1])/2, 'r--')
ax_hirshfeld.set_xlabel('Time / fs')
ax_hirshfeld.set_ylabel('Hirshfeld spin moment')
ax_hirshfeld.set_xlim([0, time_plot])
# ax_hirshfeld.set_ylim([-3.78, -3.67])
ax_hirshfeld.legend(frameon=False)
fig_hirshfeld.tight_layout()
fig_hirshfeld.savefig('{}/spin_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
