from __future__ import division, print_function
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scripts.general import parameters
import MDAnalysis as mda
from MDAnalysis.analysis import distances

"""
    Plot Hirshfeld analysis for hematite/water interface hole 
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


def func_metric(a, b, c):
    index = [0, 1, 2, 3, 4, 5]
    metric = np.average(a.flat[index])
    return metric


# Atomic index (interface)
h_top = np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]) - 1
h_bot = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) - 1
h_all = np.concatenate([h_top, h_bot])
water = np.linspace(start=157, stop=435, num=435-157+1, dtype=int) - 1
o_a = np.array([42, 45, 48, 75, 78, 81, 108, 111, 114, 141, 144, 147]) - 1
o_b = np.array([49, 52, 55, 82, 85, 88, 115, 118, 121, 148, 151, 154]) - 1
o_c = np.array([41, 44, 47, 74, 77, 80, 107, 110, 113, 140, 143, 146]) - 1
o_d = np.array([51, 54, 57, 84, 87, 90, 117, 120, 123, 150, 153, 156]) - 1
o_e = np.array([40, 43, 46, 73, 76, 79, 106, 109, 112, 139, 142, 145]) - 1
o_f = np.array([50, 53, 56, 83, 86, 89, 116, 119, 122, 149, 152, 155]) - 1
o_g = np.array([25, 26, 27, 58, 59, 60, 91, 92, 93, 124, 125, 126]) - 1
o_all = np.concatenate([o_a, o_b, o_c, o_d, o_e, o_f, o_g])
fe_a = np.array([33, 37, 66, 70, 99, 103, 132, 136]) - 1
fe_b = np.array([29, 34, 62, 67, 95, 100, 128, 133]) - 1
fe_c = np.array([32, 39, 65, 72, 98, 105, 131, 138]) - 1
fe_d = np.array([28, 36, 61, 69, 94, 102, 127, 135]) - 1
fe_e = np.array([31, 38, 64, 71, 97, 104, 130, 137]) - 1
fe_f = np.array([30, 35, 63, 68, 96, 101, 129, 134]) - 1
fe_f1 = np.array([134, 68, 101, 35]) - 1
fe_f2 = np.array([30, 96, 129, 63]) - 1
fe_alpha = np.concatenate([fe_a, fe_c, fe_e])
fe_beta = np.concatenate([fe_b, fe_d, fe_f])
fe_all = np.sort(np.concatenate([fe_a, fe_b, fe_c, fe_d, fe_e, fe_f]))

# Interface Philipp structure
atoms = 435
folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/hole/analysis'
# folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/cdft/interface/fixing-delocalisation/hole-cdft/analysis'
# folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/electron/analysis'
# folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/cdft/interface/fixing-delocalisation/electron-cdft/analysis'
folder_save_2 = folder_2
run = '05'
value = 'Spin'
draw_polaron = False
draw_legend = True
plot_save = True
energy_kinetic1_2, energy_potential1_2, energy_total1_2, temperature1_2, time_val1_2, time_per_step1_2 = read_energy(folder_2, '/energy/{}.out'.format(run))
file_spec1_2, species1_2 = read_hirsh(folder_2, '/hirshfeld/{}.out'.format(run))
num_data1_2 = energy_kinetic1_2.shape[0]
skip_start_2 = 3
skip_end_2 = 5
num_hirsh = len(file_spec1_2)/(atoms + skip_end_2)
# polaron_index = {'00': 'fe_b', '01': 'fe_f', '02': 'fe_f', '03': 'fe_d', '04': 'fe_f', '05': 'fe_f', '06': 'fe_f',
#                  '07': 'fe_f', '08': 'fe_f', '09': 'fe_f'}
# polaron_index_fe = {'00': [100-1], '01': [30-1, 63-1, 134-1], '02': [61-1, 36-1, 96-1, 134-1], '03': [95-1, 62-1, 63-1, 36-1, 135-1],
#                     '04': [134-1], '05': [29-1, 68-1], '06': [63-1], '07': [134-1], '08': [35-1], '09': [129-1, 30-1]}
polaron_index = {'00': 'fe_b', '01': 'fe_f', '02': 'fe_f', '03': 'fe_d', '04': 'fe_f', '05': 'fe_f', '06': 'fe_f',
                 '07': 'fe_f', '08': 'fe_f', '09': 'fe_f'}
polaron_index_fe = {'00': [129-1, 134-1], '01': [95-1, 62-1, 29-1]}
# plot_color = 'y', 'm', 'orange', 'hotpink', 'skyblue', 'peru', 'cyan', 'brown', 'yellowgreen', 'lightgreen'
# plot_color = 'r', 'g', 'b', 'y', 'm', 'orange', 'c', 'peru','yellowgreen', 'lightgreen'
plot_color = 'r', 'g', 'c', 'b', 'm', 'orange', 'y', 'peru','yellowgreen', 'lightgreen'
topology_file = '{}/position/topology.xyz'.format(folder_2)
trajectory_file = '{}/position/{}.xyz'.format(folder_save_2, run)
box_size = [10.241000, 10.294300, 47.342300,  91.966000, 87.424000, 119.738000]
timestep = 0.5

# Printing and plotting arrays
ylim_1 = [1.6, 3.0]
ylim_2 = [2.150, 1.85]  # Hole
# ylim_2 = [2.04, 2.20]  # Electron
ylim_3 = [-4.14, -3.0]  # Hole
# ylim_3 = [-4.12, -3.5]  # Electron
# ylim_3 = [-4.12, -3.7]  # Electron F only
ylim_4 = [-0.4, 1.2]
polaron_spin = -3.29  # Hole
# polaron_spin = -3.72  # Electron
plot_fe = globals()[str(polaron_index[run])]
print(run, str(polaron_index[run]))

# Get indexes for mdanalysis
species_numpy = species1_2[3:atoms+3].to_numpy()
species_numpy_fe = np.where(species_numpy == 'Fe')
fe_only_b = np.zeros(fe_b.shape[0])
fe_only_d = np.zeros(fe_d.shape[0])
fe_only_f1 = np.zeros(fe_f1.shape[0])
fe_only_f2 = np.zeros(fe_f2.shape[0])
for i in range(fe_b.shape[0]):
    fe_only_b[i] = np.count_nonzero(species_numpy[:fe_b[i]] == 'Fe')
    fe_only_d[i] = np.count_nonzero(species_numpy[:fe_d[i]] == 'Fe')
for i in range(fe_f1.shape[0]):
    fe_only_f1[i] = np.count_nonzero(species_numpy[:fe_f1[i]] == 'Fe')
    fe_only_f2[i] = np.count_nonzero(species_numpy[:fe_f2[i]] == 'Fe')
fe_only_f = np.concatenate([fe_only_f1, fe_only_f2])
if draw_polaron:
    fe_only_polaron = np.zeros(len(polaron_index_fe[run]))
    for i in range(len(polaron_index_fe[run])):
        fe_only_polaron[i] = np.count_nonzero(species_numpy[:polaron_index_fe[run][i]] == 'Fe')

# Plot all iron spin 1
fig_spin2, ax_spin2 = plt.subplots()
x_end = time_val1_2[-1]-time_val1_2[0]
temp4 = np.zeros(num_data1_2)
if draw_polaron:
    for j in range(len(polaron_index_fe[run])):
        for n in range(num_data1_2):
            temp4[n] = (file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + polaron_index_fe[run][j], value])
        ax_spin2.plot(time_val1_2-time_val1_2[0], temp4, '--', color=plot_color[j], linewidth=5)
temp1 = np.zeros((8, num_data1_2))
temp2 = np.zeros((8, num_data1_2))
temp3 = np.zeros((8, num_data1_2))
for j in range(len(fe_b)):
    for n in range(num_data1_2):
        temp1[j, n] = (file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_b[j], value])
        temp2[j, n] = (file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_d[j], value])
        # temp3[n] = (file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_f[j], value])
    ax_spin2.plot(time_val1_2-time_val1_2[0], temp1[j, :], 'r-')
    ax_spin2.plot(time_val1_2-time_val1_2[0], temp2[j, :], 'g-')
    # ax_spin2.plot(time_val1_2-time_val1_2[0], temp3, 'b-')
ax_spin2.plot(time_val1_2[0], temp1[0, 0], 'r-', label='Fe B')
ax_spin2.plot(time_val1_2[0], temp2[0, 0], 'g-', label='Fe D')
# ax_spin2.plot(time_val1_2[0], temp3[0], 'b-', label='Fe, F')
temp4 = np.zeros((4, num_data1_2))
temp5 = np.zeros((4, num_data1_2))
for j in range(len(fe_f1)):
    for n in range(num_data1_2):
        temp4[j, n] = (file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_f1[j], value])
        temp5[j, n] = (file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_f2[j], value])
    ax_spin2.plot(time_val1_2 - time_val1_2[0], temp4[j, :], 'b-')
    ax_spin2.plot(time_val1_2 - time_val1_2[0], temp5[j, :], 'c-')
ax_spin2.plot(time_val1_2[0], temp4[0, 0], 'b-', label=r'Fe F$_{\mathrm{i}}$')
ax_spin2.plot(time_val1_2[0], temp5[0, 0], 'c-', label=r'Fe F$_{\mathrm{b}}$')
if draw_legend: ax_spin2.legend(frameon=False)
ax_spin2.plot([0, x_end], [polaron_spin, polaron_spin], '--', label='Bulk', color='grey')
ax_spin2.set_xlabel('Time / fs')
ax_spin2.set_xlim([0, x_end])
ax_spin2.set_ylabel('Spin moment')
ax_spin2.set_ylim(ylim_3)
# ax_spin2.set_ylabel('Charge')
# ax_spin2.set_ylim(0.41, 0.54)
ax_spin2.set_xlim([0, x_end])
fig_spin2.tight_layout()
if plot_save: fig_spin2.savefig('{}/fe_spin_all_{}.png'.format(folder_save_2, run), dpi=300, bbbox_inches='tight')
# if plot_save: fig_spin2.savefig('{}/fe_spin_f_{}.png'.format(folder_save_2, run), dpi=300, bbbox_inches='tight')
# fig_spin2.savefig('{}/fe_charge_all_{}.png'.format(folder_save_2, run), dpi=300, bbbox_inches='tight')

time_avg = np.arange(start=70*2, stop=2000*2)
# print('mean Fe B', np.mean(temp1[:, time_avg]))
# print('mean Fe D', np.mean(temp2[:, time_avg]))
print('mean Fe Fb', np.mean(temp5[:, time_avg]))
print('mean Fe Fi', np.mean(temp4[:, time_avg]))

print('\nmean Fe Fb max', np.mean(np.max(temp5[:, time_avg], axis=0)))
print('mean Fe Fi max', np.mean(np.max(temp4[:, time_avg], axis=0)))
index_max_i = np.argmax(temp4[:, time_avg], axis=0)[0]
index_max_b = np.argmax(temp5[:, time_avg], axis=0)[0]
index_i = np.delete(np.array([0, 1, 2, 3]), index_max_i)
index_b = np.delete(np.array([0, 1, 2, 3]), index_max_b)
print('mean Fe Fi - max',  np.mean(temp4[np.ix_(index_i, time_avg)]))
print('mean Fe Fb - max',  np.mean(temp5[np.ix_(index_b, time_avg)]))

print('\nmean Fe Fb min', np.mean(np.min(temp5[:, time_avg], axis=0)))
print('mean Fe Fi min', np.mean(np.min(temp4[:, time_avg], axis=0)))
index_max_i = np.argmin(temp4[:, time_avg], axis=0)[0]
index_max_b = np.argmin(temp5[:, time_avg], axis=0)[0]
index_i = np.delete(np.array([0, 1, 2, 3]), index_max_i)
index_b = np.delete(np.array([0, 1, 2, 3]), index_max_b)
print('mean Fe Fi - min',  np.mean(temp4[np.ix_(index_i, time_avg)]))
print('mean Fe Fb - min',  np.mean(temp5[np.ix_(index_b, time_avg)]))

# Plot iron spin
skip_start_2 = 3
skip_end_2 = 5
fig_spin3, ax_spin3 = plt.subplots()
x_end = time_val1_2[-1]-time_val1_2[0]
temp1 = np.zeros((8, num_data1_2))
temp2 = np.zeros((8, num_data1_2))
temp3 = np.zeros((8, num_data1_2))
for j in range(len(fe_f)):
    for n in range(num_data1_2):
        temp1[j, n] = (file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_b[j], 'Spin'])
        temp2[j, n] = (file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_d[j], 'Spin'])
        temp3[j, n] = (file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_f[j], 'Spin'])
    ax_spin3.plot(time_val1_2 - time_val1_2[0], temp3[j, :], '-', color=plot_color[j], label='Fe {}'.format(j+1))
ax_spin3.plot(time_val1_2 - time_val1_2[0], (temp3[2, :]+temp3[3,:])/2, 'k--', label='Sum polaron')
# ax_spin3.legend(frameon=True, loc='upper left')
ax_spin3.set_xlabel('Time / fs')
ax_spin3.set_ylabel('Spin moment')
ax_spin3.set_ylim(ylim_3)
ax_spin3.set_xlim([1580-50, 1580+50])
fig_spin3.tight_layout()
if plot_save: fig_spin3.savefig('{}/spin_single_{}.png'.format(folder_save_2, run), dpi=300, bbbox_inches='tight')

# Print first Hirshfeld spin
# n = 0
# print('\nHirshfeld step', n)
# temp1 = np.zeros(8)
# for j in range(len(fe_f)):
#     temp1[j] = (file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + plot_fe[j], 'Spin'])
# print('All iron atoms of spin layer', temp1)
# print('Max', np.max(temp1))

# Print last Hirshfeld spin
# n = num_hirsh - 1
# print('\nHirshfeld step', n)
# temp1 = np.zeros(8)
# for j in range(len(fe_f)):
#     temp1[j] = (file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + plot_fe[j], 'Spin'])
# print('All iron atoms of spin layer', temp1)
# print('Max', np.max(temp1))

# Setup md analysis environment
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

# Plot  metric
time_plot = np.linspace(start=0, stop=len(universe.trajectory)*timestep, num=len(universe.trajectory))
metric = np.zeros((len(universe.trajectory)))
fig_4, ax_4 = plt.subplots()
# temp1 = np.zeros((len(fe_only_polaron), len(universe.trajectory)))
# if draw_polaron:
#     for i in range(len(fe_only_polaron)):
#         for j in range(len(universe.trajectory)):
#             sorted1 = np.sort(bond_lengths_time[j, int(fe_only_polaron[i])])[0:6]
#             temp1[i, j] = func_metric(sorted1, bond_lengths_mean_1[j], bond_lengths_mean_2[j])
#     ax_4.plot(time_plot, temp1[i, :], '--', color=plot_color[i], linewidth=5)
# temp1 = np.zeros((len(fe_only_b), len(universe.trajectory)))
# temp2 = np.zeros((len(fe_only_d), len(universe.trajectory)))
# for i in range(len(fe_only_b)):
#     for j in range(len(universe.trajectory)):
#         sorted1 = np.sort(bond_lengths_time[j, int(fe_only_b[i])])[0:6]
#         sorted2 = np.sort(bond_lengths_time[j, int(fe_only_d[i])])[0:6]
#         temp1[i, j] = func_metric(sorted1, bond_lengths_mean_1[j], bond_lengths_mean_2[j])
#         temp2[i, j] = func_metric(sorted2, bond_lengths_mean_1[j], bond_lengths_mean_2[j])
#     ax_4.plot(time_plot, temp1[i, :], 'r')
#     ax_4.plot(time_plot, temp2[i, :], 'g')
# ax_4.plot(time_plot[0], temp1[0, 0], 'r-', label='Fe B')
# ax_4.plot(time_plot[0], temp2[0, 0], 'g-', label='Fe D')
temp3 = np.zeros((len(fe_only_f1), len(universe.trajectory)))
temp4 = np.zeros((len(fe_only_f2), len(universe.trajectory)))
for i in range(len(fe_only_f1)):
    for j in range(len(universe.trajectory)):
        sorted1 = np.sort(bond_lengths_time[j, int(fe_only_f1[i])])[0:6]
        sorted2 = np.sort(bond_lengths_time[j, int(fe_only_f2[i])])[0:6]
        temp3[i, j] = func_metric(sorted1, bond_lengths_mean_1[j], bond_lengths_mean_2[j])
        temp4[i, j] = func_metric(sorted2, bond_lengths_mean_1[j], bond_lengths_mean_2[j])
    ax_4.plot(time_plot, temp3[i, :], 'b')
    ax_4.plot(time_plot, temp4[i, :], 'c')
ax_4.plot(time_plot[0], temp3[0, 0], 'b-', label=r'Fe F$_{\mathrm{i}}$')
ax_4.plot(time_plot[0], temp4[0, 0], 'c-', label=r'Fe F$_{\mathrm{b}}$')
ax_4.set_xlabel('Time / s')
ax_4.set_ylabel('Average Fe-O bond length / A')
ax_4.set_xlim([0, len(universe.trajectory)*timestep])
ax_4.set_ylim(ylim_2)
if draw_legend: ax_4.legend(frameon=False)
fig_4.tight_layout()
fig_4.savefig('{}/metric_all_{}.png'.format(folder_save_2, run), dpi=300, bbbox_inches='tight')
fig_4.savefig('{}/metric_fonly_{}.png'.format(folder_save_2, run), dpi=300, bbbox_inches='tight')

# time_avg = np.arange(start=70*2, stop=2000*2)
# print('mean Fe B', np.mean(temp1[:, time_avg]))
# print('mean Fe D', np.mean(temp2[:, time_avg]))
print('mean Fe Fi', np.mean(temp3[:, time_avg]))
print('mean Fe Fb', np.mean(temp4[:, time_avg]))

print('\nmean Fe Fi min', np.mean(np.min(temp3[:, time_avg])))
print('mean Fe Fb min', np.mean(np.min(temp4[:, time_avg])))
print('mean Fe Fi - min',  np.mean(temp3[np.ix_(index_i, time_avg)]))
print('mean Fe Fb - min',  np.mean(temp4[np.ix_(index_b, time_avg)]))

print('\nmean Fe Fi max', np.mean(np.max(temp3[:, time_avg])))
print('mean Fe Fb max', np.mean(np.max(temp4[:, time_avg])))
print('mean Fe Fi - max',  np.mean(temp3[np.ix_(index_i, time_avg)]))
print('mean Fe Fb - max',  np.mean(temp4[np.ix_(index_b, time_avg)]))

# Allocate arrays
print('Allocating 1')
mean_fe_alpha_spin1 = np.zeros(num_data1_2)
mean_fe_beta_spin1 = np.zeros(num_data1_2)
mean_fe_alpha_a_spin1 = np.zeros(num_data1_2)
mean_fe_alpha_c_spin1 = np.zeros(num_data1_2)
mean_fe_alpha_e_spin1 = np.zeros(num_data1_2)
mean_fe_beta_b_spin1 = np.zeros(num_data1_2)
mean_fe_beta_d_spin1 = np.zeros(num_data1_2)
mean_fe_beta_f_spin1 = np.zeros(num_data1_2)
mean_fe_alpha_a_charge1 = np.zeros(num_data1_2)
mean_fe_alpha_c_charge1 = np.zeros(num_data1_2)
mean_fe_alpha_e_charge1 = np.zeros(num_data1_2)
mean_fe_beta_b_charge1 = np.zeros(num_data1_2)
mean_fe_beta_d_charge1 = np.zeros(num_data1_2)
mean_fe_beta_f_charge1 = np.zeros(num_data1_2)
mean_fe_alpha_charge1 = np.zeros(num_data1_2)
mean_fe_beta_charge1 = np.zeros(num_data1_2)
mean_water_spin1 = np.zeros(num_data1_2)
mean_water_charge1 = np.zeros(num_data1_2)
mean_o_spin1 = np.zeros(num_data1_2)
mean_o_charge1 = np.zeros(num_data1_2)
mean_h_spin1 = np.zeros(num_data1_2)
mean_h_charge1 = np.zeros(num_data1_2)

# Build arrays
for n in range(num_data1_2):
    # mean_fe_alpha_spin1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_alpha, 'Spin']))
    # mean_fe_alpha_spin1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_alpha, 'Spin']))
    # mean_o_spin1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + o_all, 'Spin']))
    # mean_o_charge1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + o_all, 'Charge']))
    # mean_water_spin1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + water, 'Spin']))
    # mean_water_charge1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + water, 'Charge']))
    # mean_h_spin1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + h_all, 'Spin']))
    # mean_h_charge1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + h_all, 'Charge']))

    mean_fe_alpha_spin1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_alpha, 'Spin']))
    mean_fe_alpha_charge1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_alpha, 'Charge']))
    mean_fe_beta_spin1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_beta, 'Spin']))
    mean_fe_alpha_a_spin1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_a, 'Spin']))
    mean_fe_alpha_c_spin1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_c, 'Spin']))
    mean_fe_alpha_e_spin1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_e, 'Spin']))
    mean_fe_beta_b_spin1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_b, 'Spin']))
    mean_fe_beta_d_spin1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_d, 'Spin']))
    mean_fe_beta_f_spin1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_f, 'Spin']))
    mean_fe_alpha_a_charge1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_a, 'Charge']))
    mean_fe_alpha_c_charge1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_c, 'Charge']))
    mean_fe_alpha_e_charge1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_e, 'Charge']))
    mean_fe_beta_charge1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_beta, 'Charge']))
    mean_fe_beta_b_charge1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_b, 'Charge']))
    mean_fe_beta_d_charge1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_d, 'Charge']))
    mean_fe_beta_f_charge1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_f, 'Charge']))
    mean_o_spin1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + o_all, 'Spin']))
    mean_o_charge1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + o_all, 'Charge']))
    mean_water_spin1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + water, 'Spin']))
    mean_water_charge1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + water, 'Charge']))
    mean_h_spin1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + h_all, 'Spin']))
    mean_h_charge1[n] = np.mean((file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + h_all, 'Charge']))

# Plot charge and spin difference for hematite slab and water
equil = int(500 / timestep)
fig_charge_total, ax_charge_total = plt.subplots()
temp1_1 = len(o_all)*(mean_o_charge1) +len(fe_alpha)*(mean_fe_alpha_charge1) + len(fe_beta)*(mean_fe_beta_charge1)
temp2_1 = len(h_all) * (mean_h_charge1)
temp3_1 = len(water)*(mean_water_charge1)
charge_mean = np.mean(temp1_1[equil:] + temp2_1[equil:] - temp3_1[equil:])
print('charge_mean', charge_mean)
ax_charge_total.plot([time_val1_2[0], time_val1_2[-1]], [charge_mean, charge_mean], 'r--', alpha=0.5)
ax_charge_total.plot(time_val1_2-time_val1_2[0], (temp1_1 + temp2_1 - temp3_1), 'r-', label='Charge')
temp1_1 = len(o_all)*(mean_o_spin1) +len(fe_alpha)*(mean_fe_alpha_spin1) + len(fe_beta)*(mean_fe_beta_spin1)
temp2_1 = len(h_all) * (mean_h_spin1)
temp3_1 = len(water)*(mean_water_spin1)
spin_mean = np.mean(temp1_1[equil:] + temp2_1[equil:] - temp3_1[equil:])
print('spin_mean', spin_mean)
ax_charge_total.plot([time_val1_2[0], time_val1_2[-1]], [spin_mean, spin_mean], 'g--', alpha=0.5)
ax_charge_total.plot(time_val1_2-time_val1_2[0], (temp1_1 + temp2_1 - temp3_1), 'g-', label='Spin')
ax_charge_total.set_xlabel('Time / fs')
ax_charge_total.set_ylabel('Hematite - water')
ax_charge_total.set_xlim([0, x_end])
ax_charge_total.set_ylim(ylim_4)
if draw_legend: ax_charge_total.legend(frameon=False)
fig_charge_total.tight_layout()
if plot_save: fig_charge_total.savefig('{}/charge_spin_{}.png'.format(folder_save_2, run), dpi=300, bbbox_inches='tight')

# Plot total charge for hematite slab and water
equil = int(500 / timestep)
fig_charge2_total, ax_charge2_total = plt.subplots()
temp1_1 = len(o_all)*(mean_o_charge1) +len(fe_alpha)*(mean_fe_alpha_charge1) + len(fe_beta)*(mean_fe_beta_charge1)
temp2_1 = len(h_all) * (mean_h_charge1)
temp3_1 = len(water)*(mean_water_charge1)
charge_mean = np.mean(temp1_1[equil:] + temp2_1[equil:] - temp3_1[equil:])
print('charge_mean', charge_mean)
charge_mean1 = np.mean(temp1_1[equil:] + temp2_1[equil:])
charge_mean2 = np.mean(temp3_1[equil:])
print('charge_mean1', charge_mean1)
print('charge_mean2', charge_mean2)
ax_charge2_total.plot([time_val1_2[0], time_val1_2[-1]], [charge_mean1, charge_mean1], 'r--', alpha=0.5)
ax_charge2_total.plot([time_val1_2[0], time_val1_2[-1]], [charge_mean2, charge_mean2], 'g--', alpha=0.5)
ax_charge2_total.plot(time_val1_2-time_val1_2[0], (temp1_1 + temp2_1), 'r-', label='Hematite')
ax_charge2_total.plot(time_val1_2-time_val1_2[0], (temp3_1), 'g-', label='Water')
# temp1_1 = len(o_all)*(mean_o_spin1) +len(fe_alpha)*(mean_fe_alpha_spin1) + len(fe_beta)*(mean_fe_beta_spin1)
# temp2_1 = len(h_all) * (mean_h_spin1)
# temp3_1 = len(water)*(mean_water_spin1)
# spin_mean = np.mean(temp1_1[equil:] + temp2_1[equil:] - temp3_1[equil:])
# print('spin_mean', spin_mean)
# ax_charge2_total.plot([time_val1_2[0], time_val1_2[-1]], [spin_mean, spin_mean], 'g--', alpha=0.5)
# ax_charge2_total.plot(time_val1_2-time_val1_2[0], (temp1_1 + temp2_1 - temp3_1), 'g-', label='Spin')
ax_charge2_total.set_xlabel('Time / fs')
ax_charge2_total.set_ylabel('Total charge')
ax_charge2_total.set_xlim([0, x_end])
ax_charge2_total.set_ylim(ylim_4)
if draw_legend: ax_charge2_total.legend(frameon=False)
fig_charge2_total.tight_layout()
if plot_save: fig_charge2_total.savefig('{}/charge_{}.png'.format(folder_save_2, run), dpi=300, bbbox_inches='tight')

# Plot total charge for hematite slab and water
equil = int(500 / timestep)
fig_charge3_total, ax_charge3_total = plt.subplots()
# temp1_1 = len(o_all)*(mean_o_charge1) +len(fe_alpha)*(mean_fe_alpha_charge1) + len(fe_beta)*(mean_fe_beta_charge1)
# temp2_1 = len(h_all) * (mean_h_charge1)
# temp3_1 = len(water)*(mean_water_charge1)
# charge_mean = np.mean(temp1_1[equil:] + temp2_1[equil:] - temp3_1[equil:])
# print('charge_mean', charge_mean)
# ax_charge3_total.plot([time_val1_2[0], time_val1_2[-1]], [charge_mean, charge_mean], 'r--', alpha=0.5)
# ax_charge3_total.plot(time_val1_2-time_val1_2[0], (temp1_1 + temp2_1), 'r-', label='Hematite')
# ax_charge3_total.plot(time_val1_2-time_val1_2[0], (temp3_1), 'g-', label='Water')
temp1_1 = len(o_all)*(mean_o_spin1) +len(fe_alpha)*(mean_fe_alpha_spin1) + len(fe_beta)*(mean_fe_beta_spin1)
temp2_1 = len(h_all) * (mean_h_spin1)
temp3_1 = len(water)*(mean_water_spin1)
spin_mean = np.mean(temp1_1[equil:] + temp2_1[equil:] - temp3_1[equil:])
print('spin_mean', spin_mean)
spin_mean1 = np.mean(temp1_1[equil:] + temp2_1[equil:] - temp3_1[equil:])
spin_mean2 = np.mean(temp3_1[equil:])
print('spin_mean1', spin_mean1)
print('spin_mean2', spin_mean2)
ax_charge3_total.plot([time_val1_2[0], time_val1_2[-1]], [spin_mean1, spin_mean1], 'r--', alpha=0.5)
ax_charge3_total.plot([time_val1_2[0], time_val1_2[-1]], [spin_mean2, spin_mean2], 'g--', alpha=0.5)
ax_charge3_total.plot(time_val1_2-time_val1_2[0], (temp1_1 + temp2_1), 'r-', label='Hematite')
ax_charge3_total.plot(time_val1_2-time_val1_2[0], (temp3_1), 'g-', label='Water')
ax_charge3_total.set_xlabel('Time / fs')
ax_charge3_total.set_ylabel('Total spin')
ax_charge3_total.set_xlim([0, x_end])
ax_charge3_total.set_ylim(ylim_4)
if draw_legend: ax_charge3_total.legend(frameon=False)
fig_charge3_total.tight_layout()
if plot_save: fig_charge3_total.savefig('{}/spin_{}.png'.format(folder_save_2, run), dpi=300, bbbox_inches='tight')

# Plot CDFT energy
figenergy_total, axenergy_total = plt.subplots()
# axenergy_total.plot(time_val1_2-time_val1_2[0], energy_total1_2, 'g-', label='Potential')
# axenergy_total.plot(time_val1_2-time_val1_2[0], energy_total1_2, 'g-', label='Kinetic')
axenergy_total.plot(time_val1_2-time_val1_2[0], energy_total1_2, 'kx-')
axenergy_total.set_xlabel('Time / fs')
axenergy_total.set_ylabel('Total energy / H')
axenergy_total.set_xlim([0, x_end])
# axenergy_total.set_ylim(ylim_4)
if draw_legend: axenergy_total.legend(frameon=False)
figenergy_total.tight_layout()
if plot_save: figenergy_total.savefig('{}/energy_cdft_{}.png'.format(folder_save_2, run), dpi=300, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
