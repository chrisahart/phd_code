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
cores = np.array([1, 2, 3, 4, 5], dtype='int')
# labels = ['CP2K', 'CDFT', 'Integrate', 'Weight', 'Force']
labels = ['CP2K', 'CDFT', 'CDFT integrate', 'CDFT calculate']

# Data
cols = ['Filename', 'A', 'B', 'C', 'D', 'E', 'F', 'Time']
hw_energy = pd.read_csv('{}/he2_hw_energy.out'.format(folder_input), names=cols, delim_whitespace=True)
hw_energy = hw_energy.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)
hw_atom_energy = pd.read_csv('{}/he2_hw-dev-atom_energy.out'.format(folder_input), names=cols, delim_whitespace=True)
hw_atom_energy = hw_atom_energy.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)
hw_atom_energy2 = pd.read_csv('{}/he2_hw-dev-atom2_energy.out'.format(folder_input), names=cols, delim_whitespace=True)
hw_atom_energy2 = hw_atom_energy2.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)
hw_grid_energy = pd.read_csv('{}/he2_hw-dev-grid_energy.out'.format(folder_input), names=cols, delim_whitespace=True)
hw_grid_energy = hw_grid_energy.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)

# Specific
d = 4
x = 5 * d
name = 'hw_atom_energy2'

# Plot energy (relative time)
fig_hw_atom_energy2, ax_hw_atom_energy2 = plt.subplots()
ax_hw_atom_energy2.plot(cores, hw_atom_energy2['Time'].values[0:x:d]/hw_atom_energy2['Time'].values[0], 'ko-', label=labels[0])
ax_hw_atom_energy2.plot(cores, hw_atom_energy2['Time'].values[1:x+1:d]/hw_atom_energy2['Time'].values[1], 'ro-', label=labels[1])
ax_hw_atom_energy2.plot(cores, hw_atom_energy2['Time'].values[2:x+2:d]/hw_atom_energy2['Time'].values[2], 'go-', label=labels[2])
ax_hw_atom_energy2.plot(cores, hw_atom_energy2['Time'].values[3:x+3:d]/hw_atom_energy2['Time'].values[3], 'bo-', label=labels[3])
ax_hw_atom_energy2.set_xlabel('Cores')
ax_hw_atom_energy2.set_ylabel('Time divided by time on single core')
ax_hw_atom_energy2.legend(frameon=True)
ax_hw_atom_energy2.set_ylim([0, 1.05])
fig_hw_atom_energy2.tight_layout()
fig_hw_atom_energy2.savefig('{}/plotted/{}.png'.format(folder_input, name), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot energy (real time)
fig_hw_atom_energy2_time, ax_hw_atom_energy2_time = plt.subplots()
# ax_hw_atom_energy2_time.plot(cores, hw_atom_energy2['Time'].values[0:x:d], 'ko-', label=labels[0])
ax_hw_atom_energy2_time.plot(cores, hw_atom_energy2['Time'].values[1:x+1:d], 'ro-', label=labels[1])
ax_hw_atom_energy2_time.plot(cores, hw_atom_energy2['Time'].values[2:x+2:d], 'go-', label=labels[2])
ax_hw_atom_energy2_time.plot(cores, hw_atom_energy2['Time'].values[3:x+3:d], 'bo-', label=labels[3])
ax_hw_atom_energy2_time.set_ylim([-0.05, 5])
ax_hw_atom_energy2_time.set_xlabel('Cores')
ax_hw_atom_energy2_time.set_ylabel('Time / s')
ax_hw_atom_energy2_time.legend(frameon=True)
fig_hw_atom_energy2_time.tight_layout()
fig_hw_atom_energy2_time.savefig('{}/plotted/{}_time.png'.format(folder_input, name), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
