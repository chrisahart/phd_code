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
from scripts.formatting import load_cube
# from scripts.dft import cdft_beta
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import matplotlib.pyplot as plt


def func_metric(a, b, c):
    index = [0, 1, 2, 3, 4, 5]
    metric = np.average(a.flat[index])
    return metric

# 221 supercell hole
# folder_data = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hematite-conservation/221_supercell_cdft/hole-struct1/md/analysis'
# folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/analysis/structure/bulk'
# topology_file = '{}/position/topology.xyz'.format(folder_data)
# trajectory_file = '{}/position/dft.xyz'.format(folder_data)
# box_size = [10.071, 10.071, 13.747, 90, 90, 120]
# atom_polaron = 13 - 1
# timestep = 0.5
# ylim_1 = [1.7, 2.5]
# ylim_2 = [2.1, 1.9]

# Interface Guido structure
# folder_data = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/hole/archer2/analysis'
# folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/analysis/structure/interface_guido'
# topology_file = '{}/position/topology.xyz'.format(folder_data)
# trajectory_file = '{}/position/dft.xyz'.format(folder_data)
# box_size = [10.232192, 10.257230, 47.518755, 91.050370,  86.745222, 119.818827]
# atom_polaron = 39 - 1
# timestep = 0.5
# ylim_1 = [1.6, 3.0]
# ylim_2 = [2.150, 1.975]

# Interface Philipp structure
folder_data = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/hole/analysis'
folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/analysis/structure/interface_philipp'
topology_file = '{}/position/topology.xyz'.format(folder_data)
trajectory_file = '{}/position/04.xyz'.format(folder_data)
box_size = [10.241000, 10.294300, 47.342300,  91.966000, 87.424000, 119.738000]
# atom_polaron = np.array([63, 134]) - 1
atom_polaron = np.array([15, 44]) - 1
plot_color = 'y', 'm'
timestep = 0.5
ylim_1 = [1.6, 3.0]
ylim_2 = [2.150, 1.90]

universe = mda.Universe(topology_file, trajectory_file)
atoms_fe = universe.select_atoms('name Fe')
atoms_o = universe.select_atoms('name O')
dist_arr = distances.distance_array(atoms_fe.positions, atoms_o.positions, box=box_size)

# Collect bond lengths over trajectory
bond_lengths_time = np.zeros((len(universe.trajectory), len(atoms_fe), len(atoms_o)))
bond_lengths_mean_1 = np.zeros((len(universe.trajectory)))
bond_lengths_mean_2 = np.zeros((len(universe.trajectory)))
for ts in universe.trajectory:
    frame = universe.trajectory.frame
    bond_lengths_time[frame] = distances.distance_array(atoms_fe.positions, atoms_o.positions, box=box_size)
    bond_lengths_mean_1[frame] = np.average(np.sort(bond_lengths_time[frame])[:, 0:3])
    bond_lengths_mean_2[frame] = np.average(np.sort(bond_lengths_time[frame])[:, 3:6])
print('np.average(bond_lengths_mean_1)', np.average(bond_lengths_mean_1))
print('np.average(bond_lengths_mean_2)', np.average(bond_lengths_mean_2))

# Plot all bond lengths for single frame
# frame_plot = 0
frame_plot = len(universe.trajectory)-1
index = np.linspace(start=1, stop=6, num=6)
fig_1, ax_1 = plt.subplots()
for i in range(len(atoms_fe)):
    ax_1.plot(index, np.sort(bond_lengths_time[frame_plot, i])[0:6], 'kx')
for i in range(len(atom_polaron)):
    ax_1.plot(index, np.sort(bond_lengths_time[frame_plot, atom_polaron[i]])[0:6], 'x', color=plot_color[i], label='Polaron')
ax_1.plot([0, 50],[bond_lengths_mean_1[0], bond_lengths_mean_1[0]], 'k--', alpha=0.4)
ax_1.plot([0, 50],[bond_lengths_mean_2[0], bond_lengths_mean_2[0]], 'k--', alpha=0.4)
ax_1.set_xlabel('Fe-O index')
ax_1.set_ylabel('Fe-O distance / A')
ax_1.set_xlim([0.8, 6.2])
ax_1.set_ylim(ylim_1)
ax_1.legend(frameon=True)
fig_1.tight_layout()
fig_1.savefig('{}/bond_lengths_frame_{}.png'.format(folder_save, frame_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot metric for single frame
metric = np.zeros((len(atoms_fe)))
index = np.linspace(start=1, stop=len(atoms_fe), num=len(atoms_fe))
fig_2, ax_2 = plt.subplots()
for i in range(len(atoms_fe)):
    sorted = np.sort(bond_lengths_time[frame_plot, i])[0:6]
    metric[i] = func_metric(sorted, bond_lengths_mean_1[frame_plot], bond_lengths_mean_2[frame_plot])
ax_2.plot(index, metric, 'kx')
for i in range(len(atom_polaron)):
    ax_2.plot(index[atom_polaron[i]], metric[atom_polaron[i]],  'x', color=plot_color[i], label='Polaron')
ax_2.set_xlabel('Fe index')
ax_2.set_ylabel('Average Fe-O bond length / A')
ax_2.set_xlim([0.8, len(atoms_fe)+0.2])
ax_2.legend(frameon=True)
ax_2.set_ylim(ylim_2)
fig_2.tight_layout()
fig_2.savefig('{}/metric_frame_{}.png'.format(folder_save, frame_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot all bond lengths for all frame
time_plot = np.linspace(start=0, stop=len(universe.trajectory)*timestep, num=len(universe.trajectory))
fe_o_1 = np.zeros((len(universe.trajectory)))
fe_o_2 = np.zeros((len(universe.trajectory)))
fe_o_3 = np.zeros((len(universe.trajectory)))
fe_o_4 = np.zeros((len(universe.trajectory)))
fe_o_5 = np.zeros((len(universe.trajectory)))
fe_o_6 = np.zeros((len(universe.trajectory)))
fig_3, ax_3 = plt.subplots()
for i in range(len(atoms_fe)):
    for j in range(len(universe.trajectory)):
        fe_o_1[j] = np.sort(bond_lengths_time[j, i])[0]
        fe_o_2[j] = np.sort(bond_lengths_time[j, i])[1]
        fe_o_3[j] = np.sort(bond_lengths_time[j, i])[2]
        fe_o_4[j] = np.sort(bond_lengths_time[j, i])[3]
        fe_o_5[j] = np.sort(bond_lengths_time[j, i])[4]
        fe_o_6[j] = np.sort(bond_lengths_time[j, i])[5]
    # ax_3.plot(time_plot, fe_o_1, 'k')
    # ax_3.plot(time_plot, fe_o_2, 'k')
    # ax_3.plot(time_plot, fe_o_3, 'k')
    ax_3.plot(time_plot, fe_o_4, 'k')
    ax_3.plot(time_plot, fe_o_5, 'k')
    ax_3.plot(time_plot, fe_o_6, 'k')
for i in range(len(atom_polaron)):
    for j in range(len(universe.trajectory)):
        fe_o_1[j] = np.sort(bond_lengths_time[j, atom_polaron[i]])[0]
        fe_o_2[j] = np.sort(bond_lengths_time[j, atom_polaron[i]])[1]
        fe_o_3[j] = np.sort(bond_lengths_time[j, atom_polaron[i]])[2]
        fe_o_4[j] = np.sort(bond_lengths_time[j, atom_polaron[i]])[3]
        fe_o_5[j] = np.sort(bond_lengths_time[j, atom_polaron[i]])[4]
        fe_o_6[j] = np.sort(bond_lengths_time[j, atom_polaron[i]])[5]
    # ax_3.plot(time_plot, fe_o_1, color=plot_color[i])
    # ax_3.plot(time_plot, fe_o_2, color=plot_color[i])
    # ax_3.plot(time_plot, fe_o_3, color=plot_color[i])
    ax_3.plot(time_plot, fe_o_4, color=plot_color[i])
    ax_3.plot(time_plot, fe_o_5, color=plot_color[i])
    ax_3.plot(time_plot, fe_o_6, color=plot_color[i])
ax_3.set_xlabel('Time / s')
ax_3.set_ylabel('Fe-O distance / A')
ax_3.set_xlim([0, len(universe.trajectory)*timestep])
ax_3.set_ylim(ylim_1)
ax_3.legend(frameon=False)
fig_3.tight_layout()
fig_3.savefig('{}/bond_lengths_all.png'.format(folder_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot all metric
time_plot = np.linspace(start=0, stop=len(universe.trajectory)*timestep, num=len(universe.trajectory))
metric = np.zeros((len(universe.trajectory)))
fig_4, ax_4 = plt.subplots()
for i in range(len(atoms_fe)):
    for j in range(len(universe.trajectory)):
        sorted = np.sort(bond_lengths_time[j, i])[0:6]
        metric[j] = func_metric(sorted, bond_lengths_mean_1[j], bond_lengths_mean_2[j])
    ax_4.plot(time_plot, metric, 'k')
for i in range(len(atom_polaron)):
    for j in range(len(universe.trajectory)):
        sorted = np.sort(bond_lengths_time[j, atom_polaron[i]])[0:6]
        metric[j] = func_metric(sorted, bond_lengths_mean_1[j], bond_lengths_mean_2[j])
    ax_4.plot(time_plot, metric, '-', color=plot_color[i])
# ax_4.plot([0, 10000], [np.average(metric), np.average(metric)], 'r--', alpha=0.4)
# print('np.average(metric)', np.average(metric))
ax_4.set_xlabel('Time / s')
ax_4.set_ylabel('Average Fe-O bond length / A')
ax_4.set_xlim([0, len(universe.trajectory)*timestep])
ax_4.set_ylim(ylim_2)
ax_4.legend(frameon=False)
fig_4.tight_layout()
fig_4.savefig('{}/metric_all.png'.format(folder_save), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
