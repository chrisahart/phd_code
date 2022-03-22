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
run = '400K'
value = 'Spin'
folder_4 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/bulk/hole/400K/analysis'
folder_save = folder_4
cols_hirsh = ['Atom', 'Element', 'Kind', 'Ref Charge', 'Pop 1', 'Pop 2', 'Spin', 'Charge']
polaron_index_fe = {'330K': [13-1, 27-1, 46-1, 5-1], '400K': [13-1, 27-1, 38-1, 1-1, 28-1, 14-1, 29-1, 14-1, 42-1]}
labels = '0'
# plot_color = 'y', 'm', 'orange', 'hotpink', 'skyblue', 'peru', 'cyan', 'brown', 'yellowgreen', 'lightgreen'
# plot_color = 'r', 'g', 'b', 'y', 'm', 'orange', 'c', 'peru','yellowgreen', 'lightgreen'
plot_color = 'r', 'b', 'g', 'c', 'm', 'orange', 'y', 'peru','yellowgreen', 'lightgreen'
index_fe_4 = polaron_index_fe[run]
energy_kinetic1_4, energy_potential1_4, energy_total1_4, temperature1_4, time_val1_4, time_per_step1_4 = read_energy(folder_4, '/energy/{}.out'.format(run))
file_spec1_4, species1_4 = read_hirsh(folder_4, '/hirshfeld/{}.out'.format(run))
file_spec1_4 = file_spec1_4.apply(pd.to_numeric, errors='coerce')
num_data1_4 = energy_kinetic1_4.shape[0]
topology_file = '{}/position/topology.xyz'.format(folder_4)
trajectory_file = '{}/position/{}.xyz'.format(folder_4, run)

# Plotting
draw_polaron = False
draw_legend = False
polaron_size = 4
polaron_alpha = 1
ylim_1 = [-4, -3.0]
ylim_2 = [2.16, 1.89]
ylim_3 = [-31.6, -30.4]
strength_limit = -31.0

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

# Plot all iron spin
skip_start_2 = 3
skip_end_2 = 5
fig_spin2, ax_spin2 = plt.subplots()
x_end = time_val1_4[-1]-time_val1_4[0]
if draw_polaron:
    temp4 = np.zeros(num_data1_4)
    for j in range(len(index_fe_4)):
        for n in range(num_data1_4):
            temp4[n] = (file_spec1_4.loc[skip_start_2 + atoms * n + skip_end_2 * n + index_fe_4[j], 'Spin'])
        ax_spin2.plot(time_val1_4-time_val1_4[0], temp4, '--', color=plot_color[j], linewidth=polaron_size, alpha=polaron_alpha)
    ax_spin2.plot(time_val1_4[0], temp4[0], '--', label='Polaron', color=plot_color[0], linewidth=polaron_size, alpha=polaron_alpha)
temp1 = np.zeros((8, num_data1_4))
temp2 = np.zeros((8, num_data1_4))
temp3 = np.zeros((8, num_data1_4))
for j in range(len(fe_b)):
    for n in range(num_data1_4):
        temp1[j, n] = (file_spec1_4.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_b[j], 'Spin'])
        temp2[j, n] = (file_spec1_4.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_d[j], 'Spin'])
        temp3[j, n] = (file_spec1_4.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_f[j], 'Spin'])
    ax_spin2.plot(time_val1_4 - time_val1_4[0], temp1[j, :], 'r-')
    ax_spin2.plot(time_val1_4 - time_val1_4[0], temp2[j, :], 'g-')
    ax_spin2.plot(time_val1_4 - time_val1_4[0], temp3[j, :], 'b-')
ax_spin2.plot(time_val1_4[0] - time_val1_4[0], temp1[0, 0], 'r-', label='Fe, B')
ax_spin2.plot(time_val1_4[0] - time_val1_4[0], temp2[0, 0], 'g-', label='Fe, D')
ax_spin2.plot(time_val1_4[0] - time_val1_4[0], temp3[0, 0], 'b-', label='Fe, F')
# ax_spin2.plot(time_val1_4 - time_val1_4[0], np.sum(temp1, axis=0)/8, 'r--')
# ax_spin2.plot(time_val1_4 - time_val1_4[0], np.sum(temp2, axis=0)/8, 'g--')
# ax_spin2.plot(time_val1_4 - time_val1_4[0], np.sum(temp3, axis=0)/8, 'b--')
if draw_legend: ax_spin2.legend(frameon=True)
ax_spin2.plot([0, x_end], [-3.29, -3.29], '--', label='Bulk', color='grey')
ax_spin2.set_xlabel('Time / fs')
ax_spin2.set_ylabel('Spin moment')
ax_spin2.set_ylim(ylim_1)
# ax_spin2.set_xlim([0, x_end])
ax_spin2.set_xlim([1500, 1600])
fig_spin2.tight_layout()
fig_spin2.savefig('{}/spin_{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')

# Plot iron spin
skip_start_2 = 3
skip_end_2 = 5
fig_spin3, ax_spin3 = plt.subplots()
x_end = time_val1_4[-1]-time_val1_4[0]
temp1 = np.zeros((8, num_data1_4))
temp2 = np.zeros((8, num_data1_4))
temp3 = np.zeros((8, num_data1_4))
ylim_3 = [-4.14, -3.0]  # Hole
for j in range(len(fe_f)):
    for n in range(num_data1_4):
        temp1[j, n] = (file_spec1_4.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_b[j], 'Spin'])
        temp2[j, n] = (file_spec1_4.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_d[j], 'Spin'])
        temp3[j, n] = (file_spec1_4.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_f[j], 'Spin'])
    ax_spin3.plot(time_val1_4 - time_val1_4[0], temp3[j, :], '-', color=plot_color[j], label='Fe {}'.format(j+1))
ax_spin3.plot(time_val1_4 - time_val1_4[0], (temp3[1, :]+temp3[3, :])/2, 'k--', label='Sum polaron')
# ax_spin3.legend(frameon=True, loc='upper left')
ax_spin3.set_ylim(ylim_3)
ax_spin3.set_xlim([1290, 1330])
# ax_spin3.plot([0, x_end], [-3.29, -3.29], '--', label='Bulk', color='grey')
print('mean Fe B', np.mean(temp1, axis=0)[700*2:1500*2])
print('mean Fe D', np.mean(temp2, axis=0)[700*2:1500*2])
print('mean Fe F', np.mean(temp3, axis=0)[700*2:1500*2])
print('mean Fe F min', np.mean(np.max(temp3, axis=0)[700*2:1500*2]))
ax_spin3.set_xlabel('Time / fs')
ax_spin3.set_ylabel('Spin moment')
fig_spin3.tight_layout()
fig_spin3.savefig('{}/spin_hopping-bd_{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')

# f_sum = np.sum(temp3, axis=0)
# ax_spin3.plot(time_val1_4 - time_val1_4[0], f_sum, 'k-', label='Fe, F')
# ax_spin3.set_ylim([-31.4, -30.6])  # Fe F total
# ax_spin3.set_xlim([700, 1550])
# fig_spin3.tight_layout()
# fig_spin3.savefig('{}/spin_fe_f2_{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')
# print(np.mean(f_sum[700*2:1500*2]))
# print(np.max(f_sum[700*2:1500*2]))
# print(np.min(f_sum[700*2:1500*2]))

# bd_sum = np.sum(temp1, axis=0) + np.sum(temp2, axis=0)
# bd_sum_mean = np.mean(bd_sum[700*2:1500*2])
# ax_spin3.plot(time_val1_4 - time_val1_4[0], bd_sum, 'k-', label='Sum (all)')
# ax_spin3.set_ylim([-63.12, -62.6])  # Fe BD total
# ax_spin3.set_xlim([700, 1550])  # Fe BD total
# fig_spin3.tight_layout()
# fig_spin3.savefig('{}/spin_fe_bd2_{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')
# print(bd_sum_mean)
# print(np.max(bd_sum[700*2:1500*2]))
# print(np.min(bd_sum[700*2:1500*2]))

# f_sum = np.sum(temp3, axis=0)
# bd_sum = np.sum(temp1, axis=0) + np.sum(temp2, axis=0)
# bd_sum_mean = np.mean(bd_sum[700*2:1500*2])
# ax_spin3.plot(time_val1_4 - time_val1_4[0], bd_sum - f_sum, 'k-', label='Fe, F')
# ax_spin3.set_ylim([-32.4, -31.6])  # Fe F total
# ax_spin3.set_xlim([700, 1550])
# fig_spin3.tight_layout()
# fig_spin3.savefig('{}/spin_fe_f-bd_{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')
# print(np.mean(f_sum[700*2:1500*2]))
# print(np.max(f_sum[700*2:1500*2]))
# print(np.min(f_sum[700*2:1500*2]))


# Setup md analysis environment
# universe = mda.Universe(topology_file, trajectory_file)
# atoms_fe = universe.select_atoms('name Fe')
# atoms_o = universe.select_atoms('name O')
# dist_arr = distances.distance_array(atoms_fe.positions, atoms_o.positions, box=box_size)
# bond_lengths_time = np.zeros((len(universe.trajectory), len(atoms_fe), len(atoms_o)))
# bond_lengths_mean_1 = np.zeros((len(universe.trajectory)))
# bond_lengths_mean_2 = np.zeros((len(universe.trajectory)))
# for ts in universe.trajectory:
#     frame = universe.trajectory.frame
#     bond_lengths_time[frame] = distances.distance_array(atoms_fe.positions, atoms_o.positions, box=box_size)
#     bond_lengths_mean_1[frame] = np.average(np.sort(bond_lengths_time[frame])[:, 0:3])
#     bond_lengths_mean_2[frame] = np.average(np.sort(bond_lengths_time[frame])[:, 3:6])

# Plot  metric
# time_plot = np.linspace(start=0, stop=len(universe.trajectory)*timestep, num=len(universe.trajectory))
# metric = np.zeros((len(universe.trajectory)))
# fig_4, ax_4 = plt.subplots()
# if draw_polaron:
#     for i in range(len(index_fe_4)):
#         for j in range(len(universe.trajectory)):
#             sorted = np.sort(bond_lengths_time[j, index_fe_4[i]])[0:6]
#             metric[j] = func_metric(sorted, bond_lengths_mean_1[j], bond_lengths_mean_2[j])
#         ax_4.plot(time_plot, metric, '--', color=plot_color[i], linewidth=polaron_size, alpha=polaron_alpha)
# for i in range(len(atoms_fe)):
#     for j in range(len(universe.trajectory)):
#         sorted = np.sort(bond_lengths_time[j, i])[0:6]
#         metric[j] = func_metric(sorted, bond_lengths_mean_1[j], bond_lengths_mean_2[j])
#     ax_4.plot(time_plot, metric, 'k')
# ax_4.set_xlabel('Time / s')
# ax_4.set_ylabel('Average Fe-O bond length / A')
# ax_4.set_xlim([0, len(universe.trajectory)*timestep])
# ax_4.set_ylim(ylim_2)
# if draw_legend: ax_4.legend(frameon=False)
# fig_4.tight_layout()
# fig_4.savefig('{}/metric_all_{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')

# Plot correlation plot of spin moment against metric
# fig_3, ax_3 = plt.subplots()
# temp = np.zeros(num_data1_4)
# metric = np.zeros((len(universe.trajectory)))
# print(temp.shape)
# print(metric.shape)
# for j in range(len(fe_beta)):
#     for n in range(len(universe.trajectory)):
#         temp[n] = (file_spec1_4.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_beta[j], value])
#         sorted = np.sort(bond_lengths_time[n, j])[0:6]
#         metric[n] = func_metric(sorted, bond_lengths_mean_1[n], bond_lengths_mean_2[n])
#     ax_3.plot(metric, temp, 'k.', markersize=4)
# temp = np.zeros(num_data1_4)
# metric = np.zeros((len(universe.trajectory)))
# for j in range(len(index_fe_4)):
#     for n in range(len(universe.trajectory)):
#         temp[n] = (file_spec1_4.loc[skip_start_2 + atoms * n + skip_end_2 * n + index_fe_4[j], value])
#         sorted = np.sort(bond_lengths_time[n, index_fe_4[j]])[0:6]
#         metric[n] = func_metric(sorted, bond_lengths_mean_1[n], bond_lengths_mean_2[n])
#     ax_3.plot(metric, temp, '.', color=plot_color[j], markersize=4)
# ax_3.plot([0, x_end], [-3.29, -3.29], '--', color='grey')
# ax_3.set_xlim([0, len(universe.trajectory)*timestep])
# ax_3.set_xlabel('Average Fe-O bond length / A')
# ax_3.set_ylabel('Spin moment')
# ax_3.set_ylim(-4.13, -3.1)
# ax_3.set_ylabel('Charge')
# ax_3.set_ylim(0.41, 0.54)
# ax_3.set_xlim(ylim_2)
# ax_3.legend(frameon=False)
# fig_3.tight_layout()
# fig_3.savefig('{}/spin_correlation{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')
# fig_3.savefig('{}/charge_correlation{}.png'.format(folder_save_2, run), dpi=300, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
