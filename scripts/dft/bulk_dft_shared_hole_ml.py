from __future__ import division, print_function
import time
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from scripts.general import functions
from scripts.general import parameters
from scripts.formatting import load_coordinates
from scripts.formatting import load_energy
from scripts.formatting import load_forces_out
from scripts.formatting import load_forces
from scripts.formatting import load_cube
# from scripts.dft import cdft_beta


"""
    Plot energy and forces for bulk hematite CDFT
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


    return data_hirsh, species


def func_metric(a, b, c):
    index = [0, 1, 2, 3, 4, 5]
    metric = np.average(a.flat[index])
    return metric


skip = 2
atoms = 120
run = '0'
print('run', run)
value = 'Spin'
folder_4 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/bulk/hole/ml/analysis'
folder_save = folder_4

plot_color = 'r', 'b', 'g', 'c', 'm', 'orange', 'y', 'peru','yellowgreen', 'lightgreen'
topology_file = '{}/position/topology.xyz'.format(folder_4)
trajectory_file = '{}/position/{}.xyz'.format(folder_4, run)
_, _, _, _, species1_4, _, _ = load_coordinates.load_values_coord(folder_4, '/position/topology.xyz')

# Plotting
draw_polaron = False
draw_legend = False
polaron_size = 4
polaron_alpha = 1
ylim_1 = [2.16, 1.89]
xlim_1 = [0, 3000]

# System
box_size = [10.071, 10.071, 13.747, 90, 90, 120]
timestep = 0.5
h_all = np.NaN
water = np.NaN
fe_beta = np.array([1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 25, 26, 27, 28, 29, 30, 37, 38, 41, 42, 45, 46]) - 1
fe_alpha = np.array([7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24, 31, 32, 33, 34, 35, 36, 39, 40, 43, 44, 47, 48]) - 1
fe_b = np.array([14, 16, 18, 42, 27, 45, 25, 29]) - 1
fe_d = np.array([6, 2, 13, 17, 38, 4, 15, 41]) - 1
fe_f = np.array([46, 28, 5, 1, 30, 26, 37, 3]) - 1
o_all = np.linspace(start=49, stop=120, num=120-49+1, dtype=int) - 1
num_species = np.array([len(o_all), len(fe_alpha), len(fe_beta)])

# Get indexes for mdanalysis
species_numpy = species1_4[3:atoms+3].to_numpy()
fe_only_b = np.zeros(fe_b.shape[0])
fe_only_d = np.zeros(fe_d.shape[0])
fe_only_f = np.zeros(fe_f.shape[0])
for i in range(fe_b.shape[0]):
    fe_only_b[i] = np.count_nonzero(species_numpy[:fe_b[i]] == 'Fe')
    fe_only_d[i] = np.count_nonzero(species_numpy[:fe_d[i]] == 'Fe')
    fe_only_f[i] = np.count_nonzero(species_numpy[:fe_f[i]] == 'Fe')

# Setup md analysis environment
universe = mda.Universe(topology_file, trajectory_file)
atoms_fe = universe.select_atoms('name Fe Co')
atoms_o = universe.select_atoms('name O')
dist_arr = distances.distance_array(atoms_fe.positions, atoms_o.positions, box=box_size)
bond_lengths_time = np.zeros((len(universe.trajectory), len(atoms_fe), len(atoms_o)))
bond_lengths_mean_1 = np.zeros((len(universe.trajectory)))
bond_lengths_mean_2 = np.zeros((len(universe.trajectory)))
for ts in universe.trajectory:
    frame = universe.trajectory.frame
    bond_lengths_time[frame] = distances.distance_array(atoms_fe.positions, atoms_o.positions, box=box_size)
    bond_lengths_mean_1[frame] = np.average(np.sort(bond_lengths_time[frame])[:, 0:3])
    bond_lengths_mean_2[frame] = np.average(np.sort(bond_lengths_time[frame])[:, 3:6])

# Plot  metric (all)
time_plot = np.linspace(start=0, stop=len(universe.trajectory)*timestep, num=len(universe.trajectory))
metric = np.zeros((len(universe.trajectory)))
fig_4, ax_4 = plt.subplots()
for i in range(len(atoms_fe)):
    for j in range(len(universe.trajectory)):
        sorted = np.sort(bond_lengths_time[j, i])[0:6]
        metric[j] = func_metric(sorted, bond_lengths_mean_1[j], bond_lengths_mean_2[j])
    ax_4.plot(time_plot, metric, 'k')
ax_4.set_xlabel('Time / s')
ax_4.set_ylabel('Average Fe-O bond length / A')
# ax_4.set_xlim([0, len(universe.trajectory)*timestep])
ax_4.set_xlim(xlim_1)
ax_4.set_ylim(ylim_1)
if draw_legend: ax_4.legend(frameon=False)
fig_4.tight_layout()
fig_4.savefig('{}/metric_all_{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')

# Plot  metric (color coded by layer)
time_plot = np.linspace(start=0, stop=len(universe.trajectory)*timestep, num=len(universe.trajectory))
metric = np.zeros((len(universe.trajectory)))
fig_6, ax_6 = plt.subplots()
temp1 = np.zeros((len(fe_only_b), len(universe.trajectory)))
temp2 = np.zeros((len(fe_only_d), len(universe.trajectory)))
temp3 = np.zeros((len(fe_only_f), len(universe.trajectory)))
for i in range(len(fe_only_b)):
    for j in range(len(universe.trajectory)):
        sorted1 = np.sort(bond_lengths_time[j, int(fe_only_b[i])])[0:6]
        sorted2 = np.sort(bond_lengths_time[j, int(fe_only_d[i])])[0:6]
        sorted3 = np.sort(bond_lengths_time[j, int(fe_only_f[i])])[0:6]
        temp1[i, j] = func_metric(sorted1, bond_lengths_mean_1[j], bond_lengths_mean_2[j])
        temp2[i, j] = func_metric(sorted2, bond_lengths_mean_1[j], bond_lengths_mean_2[j])
        temp3[i, j] = func_metric(sorted3, bond_lengths_mean_1[j], bond_lengths_mean_2[j])
    ax_6.plot(time_plot, temp1[i, :], 'r')
    ax_6.plot(time_plot, temp2[i, :], 'g')
    ax_6.plot(time_plot, temp3[i, :], 'b')
ax_6.plot(time_plot[0], temp1[0, 0], 'r-', label='Fe B')
ax_6.plot(time_plot[0], temp2[0, 0], 'g-', label='Fe D')
ax_6.plot(time_plot[0], temp3[0, 0], 'b-', label='Fe F')
ax_6.set_xlabel('Time / s')
ax_6.set_ylabel('Average Fe-O bond length / A')
# ax_6.set_xlim([0, len(universe.trajectory)*timestep])
ax_6.set_xlim(xlim_1)
ax_6.set_ylim(ylim_1)
if draw_legend: ax_6.legend(frameon=False)
fig_6.tight_layout()
fig_6.savefig('{}/metric_color_layer_{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')

# Plot  metric (color coded by atom)
time_plot = np.linspace(start=0, stop=len(universe.trajectory)*timestep, num=len(universe.trajectory))
metric = np.zeros((len(universe.trajectory)))
fig_7, ax_7 = plt.subplots()
temp3 = np.zeros((len(fe_only_f), len(universe.trajectory)))
for i in range(len(fe_only_f)):
    for j in range(len(universe.trajectory)):
        sorted3 = np.sort(bond_lengths_time[j, int(fe_only_f[i])])[0:6]
        temp3[i, j] = func_metric(sorted3, bond_lengths_mean_1[j], bond_lengths_mean_2[j])
    ax_7.plot(time_plot, temp3[i, :], '-', color=plot_color[i], label='Fe {}'.format(i + 1))
ax_7.set_xlabel('Time / s')
ax_7.set_ylabel('Average Fe-O bond length / A')
# ax_7.set_xlim([0, len(universe.trajectory)*timestep])
ax_7.set_xlim(xlim_1)
ax_7.set_ylim(ylim_1)
# ax_7.set_xlim([700, 1500])
if draw_legend: ax_7.legend(frameon=False)
fig_7.tight_layout()
fig_7.savefig('{}/metric_color_atom_{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
