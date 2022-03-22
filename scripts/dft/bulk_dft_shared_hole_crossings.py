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


# folder_4 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/cdft/bulk/cdft/prevent-crossing/400K/extrap-0/constraint-f/analysis'
folder_4 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/cdft/bulk/cdft/prevent-crossing/400K/extrap-0/constraint-bd/analysis'
folder_save = folder_4
skip = 2
atoms = 120
run = 'step-3000_cdft-newton-63_eps-0.2_dft-cg'
value = 'Spin'
cols_hirsh = ['Atom', 'Element', 'Kind', 'Ref Charge', 'Pop 1', 'Pop 2', 'Spin', 'Charge']
# polaron_index_fe = {'330K': [13-1, 27-1, 46-1, 5-1], '400K': [13-1, 27-1, 38-1, 1-1, 28-1, 14-1, 29-1, 14-1, 42-1]}
labels = '0'
plot_color = 'y', 'm', 'orange', 'hotpink', 'skyblue', 'peru', 'cyan', 'brown', 'yellowgreen', 'lightgreen'
# index_fe_4 = polaron_index_fe[run]
index_fe_4 = [13-1, 27-1, 38-1, 1-1, 28-1, 14-1, 29-1, 14-1, 42-1]
energy_kinetic1_4, energy_potential1_4, energy_total1_4, temperature1_4, time_val1_4, time_per_step1_4 = read_energy(folder_4, '/energy/{}.out'.format(run))
file_spec1_4, species1_4 = read_hirsh(folder_4, '/hirshfeld/{}.out'.format(run))
file_spec1_4 = file_spec1_4.apply(pd.to_numeric, errors='coerce')
num_data1_4 = energy_kinetic1_4.shape[0]
topology_file = '{}/position/topology.xyz'.format(folder_4)
trajectory_file = '{}/position/{}.xyz'.format(folder_4, run)
strength1_4 = np.loadtxt('{}/strength/{}.out'.format(folder_4, run))

# Plotting
draw_polaron = False
draw_legend = False
draw_limit = True
polaron_size = 4
polaron_alpha = 1
ylim_1 = [-4, -3.0]
ylim_2 = [2.16, 1.89]
ylim_3 = [-31.6, -30.4]
strength_limit = -30.8 - 0.2
time_start = 1500
x_start = time_start
x_end = time_start + time_val1_4[-1]-time_val1_4[0]
if draw_limit: x_end = 1600

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
temp1 = np.zeros((8, num_data1_4))
temp2 = np.zeros((8, num_data1_4))
temp3 = np.zeros((8, num_data1_4))
for j in range(len(fe_b)):
    for n in range(num_data1_4):
        temp1[j, n] = (file_spec1_4.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_b[j], 'Spin'])
        temp2[j, n] = (file_spec1_4.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_d[j], 'Spin'])
        temp3[j, n] = (file_spec1_4.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_f[j], 'Spin'])
    ax_spin2.plot(time_start+time_val1_4 - time_val1_4[0], temp1[j, :], 'r-')
    ax_spin2.plot(time_start+time_val1_4 - time_val1_4[0], temp2[j, :], 'g-')
    ax_spin2.plot(time_start+time_val1_4 - time_val1_4[0], temp3[j, :], 'b-')
ax_spin2.plot(time_start+time_val1_4[0] - time_val1_4[0], temp1[0, 0], 'r-', label='Fe, B')
ax_spin2.plot(time_start+time_val1_4[0] - time_val1_4[0], temp2[0, 0], 'g-', label='Fe, D')
ax_spin2.plot(time_start+time_val1_4[0] - time_val1_4[0], temp3[0, 0], 'b-', label='Fe, F')
# ax_spin2.plot(time_start+time_val1_4 - time_val1_4[0], np.sum(temp1, axis=0)/8, 'r-')
# ax_spin2.plot(time_start+time_val1_4 - time_val1_4[0], np.sum(temp2, axis=0)/8, 'g-')
# ax_spin2.plot(time_start+time_val1_4 - time_val1_4[0], np.sum(temp3, axis=0)/8, 'b-')
if draw_legend: ax_spin2.legend(frameon=True)
ax_spin2.plot([x_start, x_end], [-3.29, -3.29], '--', label='Bulk', color='grey')
ax_spin2.set_xlabel('Time / fs')
ax_spin2.set_ylabel('Spin moment')
ax_spin2.set_ylim(ylim_1)
ax_spin2.set_xlim([x_start, x_end])
fig_spin2.tight_layout()
fig_spin2.savefig('{}/spin_{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')

# Plot iron spin
skip_start_2 = 3
skip_end_2 = 5
fig_spin3, ax_spin3 = plt.subplots()
# x_end = time_val1_4[-1]-time_val1_4[0]
temp1 = np.zeros((8, num_data1_4))
plot_fe = fe_f
for j in range(len(fe_f)):
    for n in range(num_data1_4):
        temp1[j, n] = (file_spec1_4.loc[skip_start_2 + atoms * n + skip_end_2 * n + plot_fe[j], 'Spin'])
    # ax_spin3.plot(time_val1_4 - time_val1_4[0], temp1[j, :], '-', color=plot_color[j], label='Fe {}'.format(j))
ax_spin3.plot(time_start + time_val1_4 - time_val1_4[0], np.sum(temp1, axis=0), 'k-', label='Sum (all)')
# ax_spin3.plot(time_val1_4 - time_val1_4[0], (temp1[1, :]+temp1[3,:])/2, 'r-', label='Sum (polaron)')
# ax_spin3.plot(time_val1_4 - time_val1_4[0], (np.sum(temp1, axis=0)-temp1[1, :]-temp1[3,:])/6, 'g-')
# ax_spin3.plot(time_val1_4[0] - time_val1_4[0], temp3[0], 'b-', label='Fe, F')
# ax_spin3.legend(frameon=True, loc='upper left')
# ax_spin3.plot([0, x_end], [-3.29, -3.29], '--', label='Bulk', color='grey')
ax_spin3.plot([0, x_end], [strength_limit, strength_limit], '--', label='Bulk', color='grey')
ax_spin3.set_xlabel('Time / fs')
ax_spin3.set_ylabel('Spin moment')
ax_spin3.set_ylim(ylim_3)
ax_spin3.set_xlim([x_start, x_end])
# ax_spin3.set_xlim([1310-20, 1310+20])
# ax_spin3.set_xlim([1600, 3000])
# ax_spin3.set_xlim([2310-20, 2310+20])
fig_spin3.tight_layout()
fig_spin3.savefig('{}/spin_single_{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')

# Plot iron spin
skip_start_2 = 3
skip_end_2 = 5
fig_spin4, ax_spin4 = plt.subplots()
# x_end = time_val1_4[-1]-time_val1_4[0]
temp1 = np.zeros((8, num_data1_4))
temp2 = np.zeros((8, num_data1_4))
temp3 = np.zeros((8, num_data1_4))
for j in range(len(fe_f)):
    for n in range(num_data1_4):
        temp1[j, n] = (file_spec1_4.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_b[j], 'Spin'])
        temp2[j, n] = (file_spec1_4.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_d[j], 'Spin'])
        temp3[j, n] = (file_spec1_4.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_f[j], 'Spin'])
    # ax_spin3.plot(time_val1_4 - time_val1_4[0], temp1[j, :], '-', color=plot_color[j], label='Fe {}'.format(j))
ax_spin4.plot(time_start + time_val1_4 - time_val1_4[0], np.sum(temp1, axis=0)+np.sum(temp2, axis=0), 'k-', label='Sum (all)')
# ax_spin4.plot([0, x_end], [strength_limit, strength_limit], '--', label='Bulk', color='grey')
ax_spin4.set_xlabel('Time / fs')
ax_spin4.set_ylabel('Spin moment')
# ax_spin4.set_ylim(ylim_3)
ax_spin4.set_xlim([x_start, x_end])
# ax_spin4.set_xlim([1310-20, 1310+20])
# ax_spin4.set_xlim([1600, 3000])
# ax_spin4.set_xlim([2310-20, 2310+20])
fig_spin4.tight_layout()
fig_spin4.savefig('{}/spin_fe_bd_{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')

# Plot strength
skip_start_2 = 3
skip_end_2 = 5
fig_strength, ax_strength = plt.subplots()
ax_strength.plot(time_start + time_val1_4 - time_val1_4[0], strength1_4[1:], 'k-')
ax_strength.set_xlabel('Time / fs')
ax_strength.set_ylabel('Lagrange multiplier')
# ax_strength.set_ylim(ylim_1)
ax_strength.set_xlim([x_start, x_end])
fig_strength.tight_layout()
fig_strength.savefig('{}/strength{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')

# Plot CDFT energy
fig_energy_total, ax_energy_total = plt.subplots()
ax_energy_total.plot(time_start + time_val1_4-time_val1_4[0], energy_total1_4 - energy_total1_4[0], 'k-')
ax_energy_total.set_xlabel('Time / fs')
ax_energy_total.set_ylabel('Energy change / H')
ax_energy_total.set_xlim([x_start, x_end])
# axenergy_total.set_ylim(ylim_4)
fig_energy_total.tight_layout()
fig_energy_total.savefig('{}/energy_{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')
ax_energy_total.plot(time_start + time_val1_4-time_val1_4[0], energy_potential1_4 - energy_potential1_4[0], 'r-', label='Potential')
ax_energy_total.plot(time_start + time_val1_4-time_val1_4[0], energy_kinetic1_4 - energy_kinetic1_4[0], 'g-', label='Kinetic')
ax_energy_total.legend(frameon=False)
fig_energy_total.tight_layout()
fig_energy_total.savefig('{}/energy_all{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')

# Plot CDFT energy
fig_temp, ax_temp = plt.subplots()
ax_temp.plot(time_start + time_val1_4-time_val1_4[0], temperature1_4, 'k-')
ax_temp.set_xlabel('Time / fs')
ax_temp.set_ylabel('Temperature / K')
ax_temp.set_xlim([x_start, x_end])
# axenergy_total.set_ylim(ylim_4)
# ax_temp.legend(frameon=False)
fig_temp.tight_layout()
fig_temp.savefig('{}/temperature_{}.png'.format(folder_save, run), dpi=300, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
