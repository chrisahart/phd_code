from __future__ import division, print_function
import time
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from pylab import MaxNLocator
from scripts.general import functions
from scripts.general import parameters
from scripts.formatting import load_coordinates
from scripts.formatting import load_energy
from scripts.formatting import load_forces_out
from scripts.formatting import load_forces


"""
    Plot energy for ru-ru benchmark 
"""


def read_energy(folder, filename):
    """
        Return CP2K MD .ener file as re-structured Numpy array.
    """

    files = ['{}/{}'.format(folder, filename)]
    cols = ['Step', 'Time', 'E_kin', 'Temp', 'E_pot', 'E_tot', 'Time_per_step']
    file_energy = pd.read_csv(files[0], delim_whitespace=True, names=cols, skiprows=[0])

    # Load energy data from Pandas database
    energy_kinetic = file_energy['E_kin'].values
    energy_potential = file_energy['E_pot'].values
    energy_total = file_energy['E_tot'].values
    temperature = file_energy['Temp'].values
    time = file_energy['Time'].values
    time_per_step = file_energy['Time_per_step'].values

    return energy_kinetic, energy_potential, energy_total, temperature, time, time_per_step


def read_hirsh(folder, filename):
    """
    Read Hirshfeld
    """

    cols_hirsh = ['Atom', 'Element', 'Kind', 'Ref Charge', 'Pop 1', 'Pop 2', 'Spin', 'Charge']
    data_hirsh = pd.read_csv('{}{}'.format(folder, filename), names=cols_hirsh, delim_whitespace=True)
    species = data_hirsh['Element']
    data_hirsh = data_hirsh.apply(pd.to_numeric, errors='coerce')

    return data_hirsh, species


atoms = 432

folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/pentacene/supercell_321/pbe/cdft/geo_opt/analysis'
folder_save_1 = folder_1
energy1_1 = np.loadtxt('{}/energy/absolute_1-dimer_molopt.out'.format(folder_1))
iasd1_1 = np.loadtxt('{}/iasd/absolute_1-dimer_molopt.out'.format(folder_1))
steps = np.linspace(start=1, stop=iasd1_1.shape[-1], num=iasd1_1.shape[-1])

# Plot total energy
start = 3
x_end = iasd1_1.shape[-1]
fig_energy, ax_energy = plt.subplots()
ax_energy.plot(steps[start:]-steps[start], energy1_1[start:]-energy1_1[start], 'kx-')
ax_energy.set_xlabel('Geometry optimisation step')
ax_energy.set_ylabel('Energy change / Ha')
xa = ax_energy.get_xaxis()
xa.set_major_locator(plt.MaxNLocator(integer=True))
ax_energy.set_xlim([0, 13])
ax_energy.legend(frameon=False)
fig_energy.tight_layout()
fig_energy.savefig('{}/energy.png'.format(folder_save_1), dpi=300, bbbox_inches='tight')

# Plot iasd
fig_iasd, ax_iasd = plt.subplots()
ax_iasd.plot(steps[start:]-steps[start], iasd1_1[start:], 'kx-')
ax_iasd.set_xlabel('Geometry optimisation step')
ax_iasd.set_ylabel('IASD')
xa = ax_iasd.get_xaxis()
xa.set_major_locator(plt.MaxNLocator(integer=True))
ax_iasd.set_xlim([0, 13])
# ax_iasd.set_ylim([start, 8])
fig_iasd.tight_layout()
fig_iasd.savefig('{}/iasd.png'.format(folder_save_1), dpi=300, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
