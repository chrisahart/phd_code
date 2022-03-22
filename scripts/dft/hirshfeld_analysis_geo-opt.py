from __future__ import division, print_function
import numpy as np
from scripts.general import parameters
import shutil
import os
import matplotlib.pyplot as plt
import scipy
import re
import pickle
import pandas as pd
from distutils.dir_util import copy_tree
import copy
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh

""" Plot Hirshfeld from hematite/water MD """


def read_hirsh(folder, filename, num_atoms, filename_brent, filename_mnbrack):
    """
    Read Hirshfeld
    """

    cols_hirsh = ['Atom', 'Element', 'Kind', 'Ref Charge', 'Pop 1', 'Pop 2', 'Spin', 'Charge']
    data_hirsh = pd.read_csv('{}{}'.format(folder, filename), names=cols_hirsh, delim_whitespace=True)
    species = data_hirsh['Element']
    data_hirsh = data_hirsh.apply(pd.to_numeric, errors='coerce')
    num_data = int(np.floor((len(data_hirsh) + 1) / (num_atoms + 2)))
    step = np.linspace(start=0, stop=(num_data - 1), num=num_data, dtype=int)
    brent = np.zeros(num_data)
    mnbrack = np.zeros(num_data)

    if filename_brent:
        cols_brent = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        data_brent = pd.read_csv('{}{}'.format(folder, filename_brent), names=cols_brent, delim_whitespace=True)
        data_mnbrack = pd.read_csv('{}{}'.format(folder, filename_mnbrack), names=cols_brent, delim_whitespace=True)
        brent = data_brent['9']
        mnbrack = data_mnbrack['9']
        num_data = len(brent)
        step = np.linspace(start=0, stop=(num_data - 1), num=num_data, dtype=int)

    return data_hirsh, species, num_data, step, brent, mnbrack


folder = \
    'E:/University/PhD/Programming/dft_ml_md/output/surfin/geo-opt/'
folder_save = '{}/electron'.format(folder)

filename1 = 'electron/analysis/hirshfeld/frozen-water-h-24hr-cg.out'
filename1_brent = 'electron/analysis/brent/frozen-water-h-24hr-cg.out'
filename1_mnbrack = 'electron/analysis/mnbrack/frozen-water-h-24hr-cg.out'

filename2 = 'neutral/analysis/hirshfeld/frozen-water-h-24hr-cg.out'
filename2_brent = 'neutral/analysis/brent/frozen-water-h-24hr-cg.out'
filename2_mnbrack = 'neutral/analysis/mnbrack/frozen-water-h-24hr-cg.out'

# Read number of atoms and labels from .xyz file
skip = 2
num_atoms = 435
file_spec1, species1, num_data1, step1, brent1, mnbrack1 = \
    read_hirsh(folder, filename1, num_atoms, filename1_brent, filename1_mnbrack)
file_spec2, species2, num_data2, step2, brent2, mnbrack2 = \
    read_hirsh(folder, filename2, num_atoms, filename2_brent, filename2_mnbrack)

# Atomic index (221 bulk)
h_all = np.NaN
water = np.NaN
fe_beta = np.array([1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 25, 26, 27, 28, 29, 30, 37, 38, 41, 42, 45, 46]) - 1
fe_alpha = np.array([7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24, 31, 32, 33, 34, 35, 36, 39, 40, 43, 44, 47, 48]) - 1
o_all = np.linspace(start=49, stop=120, num=120-49+1, dtype=int) - 1
num_species = np.array([len(o_all), len(fe_alpha), len(fe_beta)])

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
# fe_polaron = np.array([96, 101]) - 1
fe_polaron = np.array([96, 134]) - 1
fe_alpha = np.concatenate([fe_a, fe_c, fe_e])
fe_beta = np.concatenate([fe_b, fe_d, fe_f])
fe_all = np.concatenate([fe_a, fe_b, fe_c, fe_d, fe_e, fe_f])

# Allocate arrays
mean_fe_alpha_spin = np.zeros(num_data1)
mean_fe_beta_spin = np.zeros(num_data1)
mean_fe_beta_b_spin = np.zeros(num_data1)
mean_fe_beta_d_spin = np.zeros(num_data1)
mean_fe_beta_f_spin = np.zeros(num_data1)
mean_fe_beta_b_charge = np.zeros(num_data1)
mean_fe_beta_d_charge = np.zeros(num_data1)
mean_fe_beta_f_charge = np.zeros(num_data1)
mean_fe_alpha_charge = np.zeros(num_data1)
mean_fe_beta_charge = np.zeros(num_data1)
mean_water_spin = np.zeros(num_data1)
mean_water_charge = np.zeros(num_data1)
mean_o_spin = np.zeros(num_data1)
mean_o_charge = np.zeros(num_data1)
mean_h_spin = np.zeros(num_data1)
mean_h_charge = np.zeros(num_data1)

# Build arrays
k = 0
for j in range(num_data1):
    k = k + brent1[j] + mnbrack1[j] + 1
    i = k - 1

    mean_fe_alpha_spin[j] = np.mean((file_spec1.loc[num_atoms*i+skip*i+fe_alpha, 'Spin']))
    mean_fe_alpha_charge[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_alpha, 'Charge']))
    mean_fe_beta_spin[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_beta, 'Spin']))
    mean_fe_beta_b_spin[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_b, 'Spin']))
    mean_fe_beta_d_spin[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_d, 'Spin']))
    mean_fe_beta_f_spin[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_f, 'Spin']))
    mean_fe_beta_charge[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_beta, 'Charge']))
    mean_fe_beta_b_charge[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_b, 'Charge']))
    mean_fe_beta_d_charge[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_d, 'Charge']))
    mean_fe_beta_f_charge[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_f, 'Charge']))
    mean_o_spin[j] = np.mean((file_spec1.loc[num_atoms*i+skip*i+o_all, 'Spin']))
    mean_o_charge[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + o_all, 'Charge']))
    mean_water_spin[j] = np.mean((file_spec1.loc[num_atoms*i+skip*i+water, 'Spin']))
    mean_water_charge[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + water, 'Charge']))
    mean_h_spin[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + h_all, 'Spin']))
    mean_h_charge[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + h_all, 'Charge']))


# Allocate arrays
mean_fe_alpha_spin2 = np.zeros(num_data1)
mean_fe_beta_spin2 = np.zeros(num_data1)
mean_fe_beta_b_spin2 = np.zeros(num_data1)
mean_fe_beta_d_spin2 = np.zeros(num_data1)
mean_fe_beta_f_spin2 = np.zeros(num_data1)
mean_fe_beta_b_charge2 = np.zeros(num_data1)
mean_fe_beta_d_charge2 = np.zeros(num_data1)
mean_fe_beta_f_charge2 = np.zeros(num_data1)
mean_fe_alpha_charge2 = np.zeros(num_data1)
mean_fe_beta_charge2 = np.zeros(num_data1)
mean_water_spin2 = np.zeros(num_data1)
mean_water_charge2 = np.zeros(num_data1)
mean_o_spin2 = np.zeros(num_data1)
mean_o_charge2 = np.zeros(num_data1)
mean_h_spin2 = np.zeros(num_data1)
mean_h_charge2 = np.zeros(num_data1)

# Build arrays
k = 0
for j in range(num_data1):
    k = k + brent2[j] + mnbrack2[j] + 1
    i = k - 1

    mean_fe_alpha_spin[j] = np.mean((file_spec1.loc[num_atoms*i+skip*i+fe_alpha, 'Spin']))
    mean_fe_alpha_charge[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_alpha, 'Charge']))
    mean_fe_beta_spin[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_beta, 'Spin']))
    mean_fe_beta_b_spin[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_b, 'Spin']))
    mean_fe_beta_d_spin[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_d, 'Spin']))
    mean_fe_beta_f_spin[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_f, 'Spin']))
    mean_fe_beta_charge[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_beta, 'Charge']))
    mean_fe_beta_b_charge[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_b, 'Charge']))
    mean_fe_beta_d_charge[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_d, 'Charge']))
    mean_fe_beta_f_charge[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + fe_f, 'Charge']))
    mean_o_spin[j] = np.mean((file_spec1.loc[num_atoms*i+skip*i+o_all, 'Spin']))
    mean_o_charge[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + o_all, 'Charge']))
    mean_water_spin[j] = np.mean((file_spec1.loc[num_atoms*i+skip*i+water, 'Spin']))
    mean_water_charge[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + water, 'Charge']))
    mean_h_spin[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + h_all, 'Spin']))
    mean_h_charge[j] = np.mean((file_spec1.loc[num_atoms * i + skip * i + h_all, 'Charge']))

# Printing and plotting arrays
time = np.linspace(start=0, stop=0.5*(num_data1-1), num=num_data1)
kinds = ['Hematite: H', 'Hematite: O', 'Hematite: Fe1', 'Hematite: Fe2', 'Water']
data_charge = [mean_h_charge, mean_o_charge, mean_fe_alpha_charge, mean_fe_beta_charge, mean_water_charge]
data_spin = [mean_h_spin, mean_o_spin, mean_fe_alpha_spin, mean_fe_beta_spin, mean_water_spin]
data_neutral_charge = [mean_h_charge2, mean_o_charge2, mean_fe_alpha_charge2, mean_fe_beta_charge2, mean_water_charge2]
data_neutral_spin = [mean_h_spin2, mean_o_spin2, mean_fe_alpha_spin2, mean_fe_beta_spin2, mean_water_spin2]
num_species = np.array([len(h_all), len(o_all), len(fe_alpha), len(fe_beta), len(water)])

# Charged average (res-A-n1-446574211003ee7e3d1d31fc7f9490f2/run-31/)
# data_charge [0.25520557228915663, -0.001389486517498562, -0.13594277108433733, -0.1388433734939759,
#              -0.0014816254264369319]
# data_spin [0.0002577811244979917, 0.006511331038439473, 4.008030371485943, -3.9895373995983934,
#            1.1357256984928968e-05]

# Neutral average (IOHMD-A-prod-22ce9a183b38cedfca608118c1fc99f9/run-41/)
# data_neutral_charge = np.array([0.2570869252873564, 0.007198686371100166, -0.14496048850574716, -0.13569396551724136,
#                                 -0.00014398714621184058])
# data_neutral_spin = np.array([0.00010560344827586253, 0.00026026272577996237, 3.855104166666667, -3.855951149425287,
#                               -1.5449264615004337e-05])

# Plot average iron charge
fig_charge, ax_charge = plt.subplots()
plot_x = step1  # time
x_end = step1[-1]  # time[-1]
ax_charge.plot(plot_x, mean_fe_beta_b_charge, 'rx-', label='Fe b')
ax_charge.plot(plot_x, mean_fe_beta_d_charge, 'gx-', label='Fe d')
ax_charge.plot(plot_x, mean_fe_beta_f_charge, 'bx-', label='Fe f')
# ax_charge.plot(step1, (mean_fe_beta_b_charge+mean_fe_beta_d_charge+mean_fe_beta_f_charge)/3, 'yx-', label='Average')
# ax_charge.plot([0, 35], [data_neutral_charge[3], data_neutral_charge[3]], 'k-', label='Neutral')
ax_charge.plot([0, 1e3], [0.440, 0.440], 'k--', label='Bulk')
# ax_charge.set_xlabel('Time / fs')
ax_charge.set_xlabel('Geometry optimisation step')
ax_charge.set_ylabel('Charge')
ax_charge.legend(frameon=False)
ax_charge.set_xlim([0, x_end])
# ax_charge.set_ylim([0.439, 0.50])
ax_charge.set_ylim(bottom=0.439)
fig_charge.tight_layout()
fig_charge.savefig('{}/analysis/hirshfeld/fe_charge_mean.png'.format(folder_save), dpi=300, bbbox_inches='tight')

# Plot average iron spin
fig_spin, ax_spin = plt.subplots()
ax_spin.plot(plot_x, mean_fe_beta_b_spin, 'rx-', label='Fe b')
ax_spin.plot(plot_x, mean_fe_beta_d_spin, 'gx-', label='Fe d')
ax_spin.plot(plot_x, mean_fe_beta_f_spin, 'bx-', label='Fe f')
# ax_spin.plot(step1, (mean_fe_beta_b_spin+mean_fe_beta_d_spin+mean_fe_beta_f_spin)/3, 'yx-', label='Average')
# ax_spin.plot([0, 35], [data_neutral_spin[3], data_neutral_spin[3]], 'k-', label='Neutral')
ax_spin.plot([0, 35], [-3.9545, -3.9545], 'k--', label='Bulk')
# ax_spin.set_xlabel('Time / fs')
ax_spin.set_xlabel('Geometry optimisation step')
ax_spin.set_ylabel('Spin moment')
ax_spin.legend(frameon=False)
ax_spin.set_xlim([0, x_end])
# ax_spin.set_xlim([0, step1[-1]])
# ax_spin.set_ylim([-4.13, -3.84])
# ax_spin.set_ylim([-4.08, -3.94])
ax_spin.set_ylim(bottom=-4.08)
fig_spin.tight_layout()
fig_spin.savefig('{}/analysis/hirshfeld/fe_spin_mean.png'.format(folder_save), dpi=300, bbbox_inches='tight')

# Plot all iron spin
fig_spin2, ax_spin2 = plt.subplots()
temp1 = np.zeros(num_data1)
temp2 = np.zeros(num_data1)
temp3 = np.zeros(num_data1)
temp4 = np.zeros(num_data1)
for j in range(len(fe_b)):
    k = 0
    for n in range(num_data1):
        k = k + brent1[n] + mnbrack1[n] + 1
        i = k - 1
        temp1[n] = (file_spec1.loc[num_atoms * i + skip * i + fe_b[j], 'Spin'])
        temp2[n] = (file_spec1.loc[num_atoms * i + skip * i + fe_d[j], 'Spin'])
        temp3[n] = (file_spec1.loc[num_atoms * i + skip * i + fe_f[j], 'Spin'])
    ax_spin2.plot(plot_x, temp1, 'rx-')
    ax_spin2.plot(plot_x, temp2, 'gx-')
    ax_spin2.plot(plot_x, temp3, 'bx-')
ax_spin2.plot(step1[0], temp1[0], 'rx-', label='Fe b')
ax_spin2.plot(step1[0], temp2[0], 'gx-', label='Fe d')
ax_spin2.plot(step1[0], temp3[0], 'bx-', label='Fe f')
# for j in range(len(fe_polaron)):
#     k = 0
#     for n in range(num_data1):
#         k = k + brent1[n] + mnbrack1[n] + 1
#         i = k - 1
#         temp4[n] = (file_spec1.loc[num_atoms * i + skip * i + fe_polaron[j], 'Spin'])
#     print(temp4[-1])
#     ax_spin2.plot(step1, temp4, 'kx-')
# ax_spin2.plot(step1[0], temp4[0], 'kx-', label='Fe F e')
ax_spin2.plot([0, 350], [-3.9545, -3.9545], 'k--', label='Bulk')
# ax_spin2.plot([0, 350], [-3.73, -3.73], 'k--', label='Bulk')
# ax_spin2.set_xlabel('Time / fs')
ax_spin2.set_xlabel('Geometry optimisation step')
ax_spin2.set_ylabel('Spin moment')
ax_spin2.legend(frameon=False, loc='upper right')
ax_spin2.set_ylim([-4.13, -3.93])
ax_spin2.set_ylim(bottom=-4.13)
ax_spin2.set_xlim([0, x_end])
# ax_spin2.set_xlim([0, step1[-1]])
fig_spin2.tight_layout()
fig_spin2.savefig('{}/analysis/hirshfeld/fe_spin_all.png'.format(folder_save), dpi=300, bbbox_inches='tight')

# Plot all iron spin change from vertical
fig_spin3, ax_spin3 = plt.subplots()
temp1 = np.zeros(num_data1)
temp2 = np.zeros(num_data1)
temp3 = np.zeros(num_data1)
temp4 = np.zeros(num_data1)
for j in range(len(fe_b)):
    k = 0
    for n in range(num_data1):
        k = k + brent1[n] + mnbrack1[n] + 1
        i = k - 1
        temp1[n] = (file_spec1.loc[num_atoms * i + skip * i + fe_b[j], 'Spin'])
        temp2[n] = (file_spec1.loc[num_atoms * i + skip * i + fe_d[j], 'Spin'])
        temp3[n] = (file_spec1.loc[num_atoms * i + skip * i + fe_f[j], 'Spin'])
    ax_spin3.plot(plot_x, temp1-temp1[0], 'rx-')
    ax_spin3.plot(plot_x, temp2-temp2[0], 'gx-')
    ax_spin3.plot(plot_x, temp3-temp3[0], 'bx-')
ax_spin3.plot(plot_x[0], temp1[0]-temp1[0], 'rx-', label='Fe b')
ax_spin3.plot(plot_x[0], temp2[0]-temp2[0], 'gx-', label='Fe d')
ax_spin3.plot(plot_x[0], temp3[0]-temp3[0], 'bx-', label='Fe f')
for j in range(len(fe_polaron)):
    k = 0
    for n in range(num_data1):
        k = k + brent1[n] + mnbrack1[n] + 1
        i = k - 1
        temp4[n] = (file_spec1.loc[num_atoms * i + skip * i + fe_polaron[j], 'Spin'])
    # print(temp4[-1])
    ax_spin3.plot(plot_x, temp4-temp4[0], 'kx-')
ax_spin3.plot(plot_x[0], temp4[0]-temp4[0], 'kx-', label='Fe F e')
# ax_spin3.plot([0, 35], [-3.9545, -3.9545], 'k--', label='Bulk')
ax_spin3.plot([0, 350], [3.90-3.73, 3.90-3.73], 'k--', label='Bulk')
ax_spin3.set_xlabel('Time / fs')
# ax_spin3.set_xlabel('Geometry optimisation step')
ax_spin3.set_ylabel('Change in spin moment')
ax_spin3.legend(frameon=False, loc='upper right')
# ax_spin3.set_ylim([-4.13, -3.84])
# ax_spin3.set_ylim(bottom=-4.13)
ax_spin3.set_xlim([0, x_end])
# ax_spin3.set_xlim([0, step1[-1]])
fig_spin3.tight_layout()
fig_spin3.savefig('{}/analysis/hirshfeld/fe_spin_all_change-vertical.png'.format(folder_save), dpi=300, bbbox_inches='tight')

# Plot all iron spin change from neutral
# fig_spin4, ax_spin4 = plt.subplots()
# temp1 = np.zeros(num_data1)
# temp2 = np.zeros(num_data1)
# temp3 = np.zeros(num_data1)
# temp4 = np.zeros(num_data1)
# for j in range(len(fe_b)):
#     k = 0
#     for n in range(num_data1):
#         k = k + brent1[n] + mnbrack1[n] + 1
#         i = k - 1
#         temp1[n] = file_spec1.loc[num_atoms * i + skip * i + fe_b[j], 'Spin'] - \
#                    file_spec2.loc[num_atoms * 0 + skip * 0 + fe_b[j], 'Spin']
#         temp2[n] = file_spec1.loc[num_atoms * i + skip * i + fe_d[j], 'Spin'] - \
#                    file_spec2.loc[num_atoms * 0 + skip * 0 + fe_d[j], 'Spin']
#         temp3[n] = file_spec1.loc[num_atoms * i + skip * i + fe_f[j], 'Spin'] - \
#                    file_spec2.loc[num_atoms * 0 + skip * 0 + fe_f[j], 'Spin']
#     ax_spin4.plot(plot_x, temp1, 'rx-')
#     ax_spin4.plot(plot_x, temp2, 'gx-')
#     ax_spin4.plot(plot_x, temp3, 'bx-')
# ax_spin4.plot(plot_x[0], temp1[0], 'rx-', label='Fe b')
# ax_spin4.plot(plot_x[0], temp2[0], 'gx-', label='Fe d')
# ax_spin4.plot(plot_x[0], temp3[0], 'bx-', label='Fe f')
# for j in range(len(fe_polaron)):
#     k = 0
#     for n in range(num_data1):
#         k = k + brent1[n] + mnbrack1[n] + 1
#         i = k - 1
#         temp4[n] = file_spec1.loc[num_atoms * i + skip * i + fe_polaron[j], 'Spin'] - \
#                    file_spec2.loc[num_atoms * 0 + skip * 0 + fe_polaron[j], 'Spin']
#     # print(temp4[-1])
#     ax_spin4.plot(plot_x, temp4, 'kx-')
# ax_spin4.plot(plot_x[0], temp4[0], 'kx-', label='Fe F e')
# # ax_spin4.plot([0, 35], [-3.9545, -3.9545], 'k--', label='Bulk')
# ax_spin4.plot([0, 350], [3.95-3.73, 3.95-3.73], 'k--', label='Bulk')
# ax_spin4.set_xlabel('Time / fs')
# # ax_spin4.set_xlabel('Geometry optimisation step')
# ax_spin4.set_ylabel('Change in spin moment')
# ax_spin4.legend(frameon=False, loc='upper right')
# # ax_spin4.set_ylim([-4.13, -3.84])
# # ax_spin4.set_ylim(bottom=-4.13)
# ax_spin4.set_xlim([0, x_end])
# fig_spin4.tight_layout()
# fig_spin4.savefig('{}/analysis/hirshfeld/fe_spin_all_change-neutral.png'.format(folder_save), dpi=300, bbbox_inches='tight')

# Plot average charge deviation (based on Guido thesis)
fig_charge_deviation, ax_charge_deviation = plt.subplots()
ax_charge_deviation.bar(kinds, data_charge-data_neutral_charge)
ax_charge_deviation.tick_params(axis='x', rotation=90)
ax_charge_deviation.plot([-1, 5], [0, 0], 'r-')
ax_charge_deviation.set_xlim([-0.5, 4.5])
ax_charge_deviation.set_ylabel('Average charge difference from neutral')
fig_charge_deviation.tight_layout()
fig_charge_deviation.savefig('{}/average_charge_deviation.png'.format(folder), dpi=300, bbbox_inches='tight')
#
# # Plot average charge deviation (based on Guido thesis)
# fig_charge_deviation2, ax_charge_deviation2 = plt.subplots()
# ax_charge_deviation2.plot(time, np.abs(temp), 'x-', label='Spin')
# ax_charge_deviation2.bar(kinds, (data_charge-data_neutral_charge)*num_species)
# ax_charge_deviation2.plot([-1, 5], [0, 0], 'r-')
# ax_charge_deviation2.plot([-1, 5], [-1, -1], 'r-')
# ax_charge_deviation2.tick_params(axis='x', rotation=90)
# ax_charge_deviation2.set_ylabel('Average charge difference')
# ax_charge_deviation2.set_ylim([-1.2, 1.2])
# ax_charge_deviation2.set_xlim([-0.5, 4.5])
# fig_charge_deviation2.tight_layout()
# fig_charge_deviation2.savefig('{}/average_charge_deviation2.png'.format(folder), dpi=300, bbbox_inches='tight')
#
# # Plot average spin (based on Guido thesis)
# fig_spin, ax_spin = plt.subplots()
# ax_spin.bar(kinds, data_spin)
# ax_spin.plot([-1, 5], [0, 0], 'r-')
# ax_spin.set_xlim([-0.5, 4.5])
# ax_spin.tick_params(axis='x', rotation=90)
# ax_spin.set_ylabel('Average spin')
# fig_spin.tight_layout()
# fig_spin.savefig('{}/average_spin.png'.format(folder), dpi=300, bbbox_inches='tight')
#
# # Plot average spin deviation (based on Guido thesis)
# fig_spin_deviation, ax_spin_deviation = plt.subplots()
# ax_spin_deviation.plot([-1, 5], [0, 0], 'r-')
# ax_spin_deviation.set_xlim([-0.5, 4.5])
# ax_spin_deviation.bar(kinds, data_spin-data_neutral_spin)
# ax_spin_deviation.tick_params(axis='x', rotation=90)
# ax_spin_deviation.set_ylabel('Average spin difference from neutral')
# fig_spin_deviation.tight_layout()
# fig_spin_deviation.savefig('{}/average_spin_deviation.png'.format(folder), dpi=300, bbbox_inches='tight')
#
# # Plot average spin deviation (based on Guido thesis)
# fig_spin_deviation2, ax_spin_deviation2 = plt.subplots()
# ax_spin_deviation2.bar(kinds, (data_spin-data_neutral_spin)*num_species)
# ax_spin_deviation2.plot([-1, 5], [0, 0], 'r-')
# ax_spin_deviation2.set_xlim([-0.5, 4.5])
# ax_spin_deviation2.tick_params(axis='x', rotation=90)
# ax_spin_deviation2.set_ylabel('Average spin difference')
# ax_spin_deviation2.set_ylim([-4, 4])
# fig_spin_deviation2.tight_layout()
# fig_spin_deviation2.savefig('{}/average_spin_deviation2.png'.format(folder), dpi=300, bbbox_inches='tight')

# fig_spin3, ax_spin3 = plt.subplots()
# temp = np.zeros(num_data1)
# for j in range(len(fe_all)):
#     for i in range(num_data1):
#         temp[i] = (file_spec1.loc[435 * i + skip * i + fe_all[j], 'Charge'])
#     ax_spin3.plot(time, temp, 'x-', label='Spin')
# ax_spin3.set_xlabel('Time / fs')
# ax_spin3.set_ylabel('Absolute value')
# fig_spin3.tight_layout()
# fig_hab_pbe.savefig('{}/hab.png'.format(folder), dpi=300, bbbox_inches='tight')

# fig_water, ax_water = plt.subplots()
# ax_water.plot(time, mean_water_spin-mean_water_spin[0], 'rx-', label='Spin')
# ax_water.plot(time, mean_water_charge-mean_water_charge[0], 'bx-', label='Charge')
# ax_water.set_xlabel('Time / fs')
# ax_water.set_ylabel('Absolute value')
# ax_water.legend(frameon=False, loc='upper left')
# fig_water.tight_layout()
# fig_hab_pbe.savefig('{}/hab.png'.format(folder), dpi=300, bbbox_inches='tight')

# fig_o, ax_o = plt.subplots()
# ax_o.plot(time, mean_o_spin-mean_o_spin[0], 'rx-', label='Spin')
# ax_o.plot(time, mean_o_charge-mean_o_charge[0], 'bx-', label='Charge')
# ax_o.set_xlabel('Time / fs')
# ax_o.set_ylabel('Absolute value')
# ax_o.legend(frameon=False, loc='upper left')
# fig_o.tight_layout()
# fig_hab_pbe.savefig('{}/hab.png'.format(folder), dpi=300, bbbox_inches='tight')

# fig_o_all, ax_o_all = plt.subplots()
# temp = np.zeros(num_data1)
# for j in range(len(o_all)):
#     for i in range(num_data1):
#         temp[i] = np.abs(file_spec1.loc[435 * i + skip * i + o_all[j], 'Spin'])
#     ax_o_all.plot(time, temp, 'x-', label='Spin')
# ax_o_all.set_xlabel('Time / fs')
# ax_o_all.set_ylabel('Absolute value')
# fig_o_all.tight_layout()
# fig_hab_pbe.savefig('{}/hab.png'.format(folder), dpi=300, bbbox_inches='tight')

# fig_o_all, ax_o_all = plt.subplots()
# temp = np.zeros(num_data1)
# for j in range(len(o_all)):
#     for i in range(num_data1):
#         temp[i] = (file_spec1.loc[435 * i + skip * i + o_all[j], 'Charge'])
#     ax_o_all.plot(time, temp, 'x-', label='Spin')
# ax_o_all.set_xlabel('Time / fs')
# ax_o_all.set_ylabel('Absolute value')
# fig_o_all.tight_layout()
# fig_hab_pbe.savefig('{}/hab.png'.format(folder), dpi=300, bbbox_inches='tight')

# fig_o_all2, ax_o_all2 = plt.subplots()
# temp = np.zeros(num_data1)
# for j in range(len(o_a)):
#     for i in range(num_data1):
#         temp[i] = (file_spec1.loc[435 * i + skip * i + o_a[j], 'Spin'])
#     ax_o_all2.plot(time, temp, 'x-', label='Spin')
# ax_o_all2.set_xlabel('Time / fs')
# ax_o_all2.set_ylabel('Absolute value')
# fig_o_all2.tight_layout()
# fig_hab_pbe.savefig('{}/hab.png'.format(folder), dpi=300, bbbox_inches='tight')


if __name__ == "__main__":
    print('Finished.')
    plt.show()
