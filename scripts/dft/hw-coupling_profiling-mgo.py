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
folder_input = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hw_forces/mgo/old'
cores = np.array([6, 8, 10, 12, 14, 16, 18], dtype='int')
labels = ['CP2K', 'CDFT', 'CDFT integrate', 'CDFT calculate']
labels2 = ['+1', '-1']
cols1 = ['Filename', 'A', 'B', 'C', 'D', 'E', 'F', 'Time']

# Data
# hw_atom_cutoff_atomic1 = pd.read_csv('{}/mgo_hw-dev-atom_cutoff-atomic_energy1.out'.format(folder_input), names=cols1, delim_whitespace=True)
# hw_atom_cutoff_atomic1 = hw_atom_cutoff_atomic1.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)
# hw_atom_cutoff_atomic2 = pd.read_csv('{}/mgo_hw-dev-atom_cutoff-atomic_energy2.out'.format(folder_input), names=cols1, delim_whitespace=True)
# hw_atom_cutoff_atomic2 = hw_atom_cutoff_atomic2.drop(['A', 'B', 'C', 'D', 'E', 'Time'], axis=1)
# hw_atom_cutoff_atomic3 = pd.read_csv('{}/mgo_hw-dev-atom_cutoff-atomic_energy3.out'.format(folder_input), names=cols1, delim_whitespace=True)
# hw_atom_cutoff_atomic3 = hw_atom_cutoff_atomic3.drop(['A', 'B', 'D', 'E', 'F', 'Time'], axis=1)
#
# hw_atom_cutoff_eps1 = pd.read_csv('{}/mgo_hw-dev-atom_cutoff-eps_energy1.out'.format(folder_input), names=cols1, delim_whitespace=True)
# hw_atom_cutoff_eps1 = hw_atom_cutoff_eps1.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)
# hw_atom_cutoff_eps2 = pd.read_csv('{}/mgo_hw-dev-atom_cutoff-eps_energy2.out'.format(folder_input), names=cols1, delim_whitespace=True)
# hw_atom_cutoff_eps2 = hw_atom_cutoff_eps2.drop(['A', 'B', 'C', 'D', 'E', 'Time'], axis=1)
# hw_atom_cutoff_eps3 = pd.read_csv('{}/mgo_hw-dev-atom_cutoff-eps_energy3.out'.format(folder_input), names=cols1, delim_whitespace=True)
# hw_atom_cutoff_eps3 = hw_atom_cutoff_eps3.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)

hw_atom_cutoff_atomic1 = pd.read_csv('{}/mgo_hw-dev-atom_cutoff-atomic_energy1-scf5.out'.format(folder_input), names=cols1, delim_whitespace=True)
hw_atom_cutoff_atomic1 = hw_atom_cutoff_atomic1.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)
hw_atom_cutoff_atomic2 = pd.read_csv('{}/mgo_hw-dev-atom_cutoff-atomic_energy2-scf5.out'.format(folder_input), names=cols1, delim_whitespace=True)
hw_atom_cutoff_atomic2 = hw_atom_cutoff_atomic2.drop(['A', 'B', 'C', 'D', 'E', 'Time'], axis=1)
hw_atom_cutoff_atomic3 = pd.read_csv('{}/mgo_hw-dev-atom_cutoff-atomic_energy3-scf5.out'.format(folder_input), names=cols1, delim_whitespace=True)
hw_atom_cutoff_atomic3 = hw_atom_cutoff_atomic3.drop(['A', 'B', 'D', 'E', 'F', 'Time'], axis=1)

hw_atom_cutoff_eps1 = pd.read_csv('{}/mgo_hw-dev-atom_cutoff-eps_energy1-scf5.out'.format(folder_input), names=cols1, delim_whitespace=True)
hw_atom_cutoff_eps1 = hw_atom_cutoff_eps1.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)
hw_atom_cutoff_eps2 = pd.read_csv('{}/mgo_hw-dev-atom_cutoff-eps_energy2-scf5.out'.format(folder_input), names=cols1, delim_whitespace=True)
hw_atom_cutoff_eps2 = hw_atom_cutoff_eps2.drop(['A', 'B', 'C', 'D', 'E', 'Time'], axis=1)
hw_atom_cutoff_eps3 = pd.read_csv('{}/mgo_hw-dev-atom_cutoff-eps_energy3-scf5.out'.format(folder_input), names=cols1, delim_whitespace=True)
hw_atom_cutoff_eps3 = hw_atom_cutoff_eps3.drop(['A', 'B', 'D', 'E', 'F', 'Time'], axis=1)


# Specific
d = 2
start = 0
x = np.shape(cores)[0] * d
name = 'hw_atom_cutoff_eps-scf5'

# Plot energy (relative time)
# average = (hw_atom_cutoff_eps1['Time'].values[0:x:d] + hw_atom_cutoff_eps1['Time'].values[1:x+1:d]) / 2
fig_hw_atom_cutoff_eps, ax_hw_atom_cutoff_eps = plt.subplots()
ax_hw_atom_cutoff_eps.plot(cores[start:], hw_atom_cutoff_eps1['Time'].values[0:x:d], 'ko-', label=labels2[0])
ax_hw_atom_cutoff_eps.plot(cores[start:], hw_atom_cutoff_eps1['Time'].values[1:x+1:d], 'ko-', label=labels2[1])
# ax_hw_atom_cutoff_eps.plot(cores[start:], average, 'ko--', label='Average')
# ax_hw_atom_cutoff_eps.plot(cores, hw_atom_cutoff_eps1['Time'].values[0:x:d]/hw_atom_cutoff_eps1['Time'].values[0], 'ko-', label=labels[0])
# ax_hw_atom_cutoff_eps.plot(cores, -hw_atom_cutoff_eps3['C'].values[0:x:d]/-hw_atom_cutoff_eps3['C'].values[0], 'ro-', label=labels[0])
ax_hw_atom_cutoff_eps.set_xlabel('Cutoff')
ax_hw_atom_cutoff_eps.set_ylabel('Time / s')
# ax_hw_atom_cutoff_eps.legend(frameon=True)
# ax_hw_atom_cutoff_eps.set_ylim([0, 1.05])
fig_hw_atom_cutoff_eps.tight_layout()
fig_hw_atom_cutoff_eps.savefig('{}/plotted/{}.png'.format(folder_input, name), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot energy (real time)
average = (-hw_atom_cutoff_eps3['C'].values[0:x:d]-hw_atom_cutoff_eps3['C'].values[1:x+1:d]) / 2
fig_hw_atom_cutoff_eps_time, ax_hw_atom_cutoff_eps_time = plt.subplots()
# ax_hw_atom_cutoff_eps_time.plot(cores, -hw_atom_cutoff_eps3['C'].values[0:x:d], 'ro-', label=labels2[0])
ax_hw_atom_cutoff_eps_time.plot(cores, -hw_atom_cutoff_eps3['C'].values[1:x+1:d], 'ko-', label=labels2[0])
# ax_hw_atom_cutoff_eps_time.plot(cores, average, 'ko--', label='Average')
# ax_hw_atom_cutoff_eps_time.set_ylim([-0.05, 5])
ax_hw_atom_cutoff_eps_time.set_xlabel('Cutoff')
ax_hw_atom_cutoff_eps_time.set_ylabel('Time / s')
# ax_hw_atom_cutoff_eps_time.legend(frameon=True)
fig_hw_atom_cutoff_eps_time.tight_layout()
fig_hw_atom_cutoff_eps_time.savefig('{}/plotted/{}_time.png'.format(folder_input, name), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot energy (real time)
fig_hw_atom_coupling_eps_time, ax_hw_atom_coupling_eps_time = plt.subplots()
ax_hw_atom_coupling_eps_time.plot(cores[start:], hw_atom_cutoff_eps2['F'].values[start:], 'ko-', label=labels[0])
# ax_hw_atom_coupling_eps_time.set_ylim([-0.05, 5])
ax_hw_atom_coupling_eps_time.set_xlabel('Cutoff')
ax_hw_atom_coupling_eps_time.set_ylabel('Coupling / au')
# ax_hw_atom_coupling_eps_time.legend(frameon=True)
fig_hw_atom_coupling_eps_time.tight_layout()
fig_hw_atom_coupling_eps_time.savefig('{}/plotted/{}_coupling.png'.format(folder_input, name), dpi=parameters.save_dpi, bbbox_inches='tight')


if __name__ == "__main__":
    print('Finished.')
    plt.show()
