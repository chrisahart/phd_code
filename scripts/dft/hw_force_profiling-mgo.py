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
folder_input = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hw_forces/mgo'
cores = np.array([2, 4, 6, 8, 10], dtype='int')
# labels = ['CP2K', 'CDFT', 'Integrate', 'Weight', 'Force']
labels = ['CP2K', 'CDFT', 'CDFT calculate', 'CDFT force', 'CDFT integrate']

# Data
cols = ['Filename', 'A', 'B', 'C', 'D', 'E', 'F', 'Time']
hw_energy = pd.read_csv('{}/mgo_hw_energy.out'.format(folder_input), names=cols, delim_whitespace=True)
hw_energy = hw_energy.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)
hw_atom_energy = pd.read_csv('{}/mgo_hw-dev-atom_energy.out'.format(folder_input), names=cols, delim_whitespace=True)
hw_atom_energy = hw_atom_energy.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)
hw_atom_energy_e12 = pd.read_csv('{}/mgo_hw-dev-atom_e12_energy.out'.format(folder_input), names=cols, delim_whitespace=True)
hw_atom_energy_e12 = hw_atom_energy_e12.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)
hw_atom_energy_e14 = pd.read_csv('{}/mgo_hw-dev-atom_e14_energy.out'.format(folder_input), names=cols, delim_whitespace=True)
hw_atom_energy_e14 = hw_atom_energy_e14.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)
hw_atom_force = pd.read_csv('{}/mgo_hw-dev-atom_force.out'.format(folder_input), names=cols, delim_whitespace=True)
hw_atom_force = hw_atom_force.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)

# Specific
d = 5
x = np.shape(cores)[0] * d
name = 'hw_atom_force'

# Plot force (relative time)
fig_hw_atom_force, ax_hw_atom_force = plt.subplots()
ax_hw_atom_force.plot(cores, hw_atom_force['Time'].values[0:x:d]/hw_atom_force['Time'].values[0], 'ko-', label=labels[0])
ax_hw_atom_force.plot(cores, hw_atom_force['Time'].values[1:x+1:d]/hw_atom_force['Time'].values[1], 'ro-', label=labels[1])
ax_hw_atom_force.plot(cores, hw_atom_force['Time'].values[2:x+2:d]/hw_atom_force['Time'].values[2], 'bo-', label=labels[2])
ax_hw_atom_force.plot(cores, hw_atom_force['Time'].values[3:x+3:d]/hw_atom_force['Time'].values[3], 'mo-', label=labels[3])
# ax_hw_atom_force.plot(cores, hw_atom_force['Time'].values[4:x+4:d]/hw_atom_force['Time'].values[4], 'mo-', label=labels[4])
ax_hw_atom_force.set_xlabel('Nodes')
ax_hw_atom_force.set_ylabel('Time divided by time on single core')
ax_hw_atom_force.legend(frameon=True)
ax_hw_atom_force.set_ylim([0, 1.05])
fig_hw_atom_force.tight_layout()
fig_hw_atom_force.savefig('{}/plotted/{}.png'.format(folder_input, name), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot force (real time)
fig_hw_atom_force_time, ax_hw_atom_force_time = plt.subplots()
# ax_hw_atom_force_time.plot(cores, hw_atom_force['Time'].values[0:x:d], 'ko-', label=labels[0])
ax_hw_atom_force_time.plot(cores, hw_atom_force['Time'].values[1:x+1:d], 'ro-', label=labels[1])
ax_hw_atom_force_time.plot(cores, hw_atom_force['Time'].values[2:x+2:d], 'bo-', label=labels[2])
ax_hw_atom_force_time.plot(cores, hw_atom_force['Time'].values[3:x+3:d], 'mo-', label=labels[3])
# ax_hw_atom_force_time.plot(cores, hw_atom_force['Time'].values[4:x+4:d], 'mo-', label=labels[4])
# ax_hw_atom_force_time.set_ylim([-0.05, 5])
ax_hw_atom_force_time.set_xlabel('Nodes')
ax_hw_atom_force_time.set_ylabel('Time / s')
ax_hw_atom_force_time.legend(frameon=True)
fig_hw_atom_force_time.tight_layout()
fig_hw_atom_force_time.savefig('{}/plotted/{}_time.png'.format(folder_input, name), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
