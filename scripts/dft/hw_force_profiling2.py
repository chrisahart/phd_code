from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
import random
from numpy import nan as Nan
import matplotlib.pyplot as plt
import scipy
from scripts.general import parameters
from scripts.general import functions
from scripts.formatting import load_coordinates
from scripts.formatting import load_cube

"""
    Plotting of HW force profile data.
"""


# General
folder_input = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hw_forces/he2'
# cores = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 30])
cores = np.array([6, 10, 11, 12, 13, 14, 24])
# labels = ['CP2K', 'CDFT', 'Integrate', 'Weight', 'Force']
labels = ['CP2K', 'CDFT', 'CDFT integrate', 'CDFT calculate']

# Data
cols = ['Filename', 'A', 'B', 'C', 'D', 'E', 'F', 'Time']
hw_cutoff_atom_energy = pd.read_csv('{}/he2_hw-dev_cutoff-atom_energy.out'.format(folder_input), names=cols, delim_whitespace=True)
hw_cutoff_atom_energy = hw_cutoff_atom_energy.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)
hw_cutoff_atom_energy_energy = pd.read_csv('{}/he2_hw-dev_cutoff-atom-energy_energy.out'.format(folder_input), names=cols, delim_whitespace=True)
hw_cutoff_atom_energy_energy = hw_cutoff_atom_energy_energy.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)
hw_cutoff_eps_energy = pd.read_csv('{}/he2_hw-dev_cutoff-eps_energy.out'.format(folder_input), names=cols, delim_whitespace=True)
hw_cutoff_eps_energy = hw_cutoff_eps_energy.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)
hw_cutoff_eps_energy_energy = pd.read_csv('{}/he2_hw-dev_cutoff-eps_energy_energy.out'.format(folder_input), names=cols, delim_whitespace=True)
hw_cutoff_eps_energy_energy = hw_cutoff_eps_energy_energy.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)

# Specific
print(np.shape(cores))
d = 4
x = np.shape(cores)[0] * d
name = 'cutoff_eps'

# Plot energy (relative time)
fig_hw_cutoff_eps_energy, ax_hw_cutoff_eps_energy = plt.subplots()
ax_hw_cutoff_eps_energy.plot(cores, hw_cutoff_eps_energy['Time'].values[0:x:d]/hw_cutoff_eps_energy['Time'].values[0], 'ko-', label=labels[0])
ax_hw_cutoff_eps_energy.plot(cores, hw_cutoff_eps_energy['Time'].values[1:x+1:d]/hw_cutoff_eps_energy['Time'].values[1], 'ro-', label=labels[1])
ax_hw_cutoff_eps_energy.plot(cores, hw_cutoff_eps_energy['Time'].values[2:x+2:d]/hw_cutoff_eps_energy['Time'].values[2], 'go-', label=labels[2])
ax_hw_cutoff_eps_energy.plot(cores, hw_cutoff_eps_energy['Time'].values[3:x+3:d]/hw_cutoff_eps_energy['Time'].values[3], 'bo-', label=labels[3])
ax_hw_cutoff_eps_energy.set_xlabel('Cutoff')
ax_hw_cutoff_eps_energy.set_ylabel('Time divided by time on single core')
ax_hw_cutoff_eps_energy.legend(frameon=True)
# ax_hw_cutoff_eps_energy.set_ylim([0, 1.05])
fig_hw_cutoff_eps_energy.tight_layout()
fig_hw_cutoff_eps_energy.savefig('{}/plotted/cutoff/{}.png'.format(folder_input, name), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot energy (real time)
fig_hw_cutoff_eps_energy_time, ax_hw_cutoff_eps_energy_time = plt.subplots()
# ax_hw_cutoff_eps_energy_time.plot(cores, hw_cutoff_eps_energy['Time'].values[0:x:d], 'ko-', label=labels[0])
# ax_hw_cutoff_eps_energy_time.plot(cores, hw_cutoff_eps_energy['Time'].values[1:x+1:d], 'ro-', label=labels[1])
# ax_hw_cutoff_eps_energy_time.plot(cores, hw_cutoff_eps_energy['Time'].values[2:x+2:d], 'go-', label=labels[2])
ax_hw_cutoff_eps_energy_time.plot(cores, hw_cutoff_eps_energy['Time'].values[3:x+3:d], 'ko-', label=labels[3])
# ax_hw_cutoff_eps_energy_time.set_ylim([-0.05, 5])
ax_hw_cutoff_eps_energy_time.set_xlabel('Cutoff')
ax_hw_cutoff_eps_energy_time.set_ylabel('Time / s')
# ax_hw_cutoff_eps_energy_time.legend(frameon=True)
fig_hw_cutoff_eps_energy_time.tight_layout()
fig_hw_cutoff_eps_energy_time.savefig('{}/plotted/cutoff/{}_time.png'.format(folder_input, name), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot energy
fig_energy, ax_energy = plt.subplots()
ax_energy.plot(cores[:], hw_cutoff_eps_energy_energy['Time'].values[:], 'ko-', label=labels[3])
# ax_energy.set_ylim([-0.05, 5])
ax_energy.set_xlabel('Cutoff')
ax_energy.set_ylabel('Energy / au')
# ax_energy.legend(frameon=True)
fig_energy.tight_layout()
fig_energy.savefig('{}/plotted/cutoff/{}_energy.png'.format(folder_input, name), dpi=parameters.save_dpi, bbbox_inches='tight')


if __name__ == "__main__":
    print('Finished.')
    plt.show()
