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
from scripts.dft import load_forces_cg
# from scripts.dft import hirshfeld_analysis_md

""" Plot energy and force from hematite/water MD """


def read_energy(folder, filename):
    """
    Read energy
    """

    cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    data = pd.read_csv('{}{}'.format(folder, filename), names=cols, delim_whitespace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    time = np.linspace(start=0, stop=1 * (len(data) - 1), num=len(data))

    return data, time


def read_force(folder, filename):
    """
    Read energy
    """

    cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    data = pd.read_csv('{}{}'.format(folder, filename), names=cols, delim_whitespace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    time = np.linspace(start=0, stop=1 * (len(data) - 1), num=len(data))

    return data, time


folder1 = \
    'E:/University/PhD/Programming/dft_ml_md/output/surfin/geo-opt/electron/analysis/'
folder2 = \
    'E:/University/PhD/Programming/dft_ml_md/output/surfin/geo-opt/electron-offset-b/analysis/'
folder3 = \
    'E:/University/PhD/Programming/dft_ml_md/output/surfin/geo-opt/electron-offset-d/analysis/'
folder4 = \
    'E:/University/PhD/Programming/dft_ml_md/output/surfin/geo-opt/electron-offset-f/analysis/'
num_atoms = 435
skip = 2
skip2 = 0

# Read energy
energy1_1, energy_time1_1 = read_energy(folder1, 'energy/frozen-water-h-24hr-cg-all.out')
energy1_2, energy_time1_2 = read_energy(folder1, 'energy/frozen-water-h_from-neutral-24hr-cg-all.out')
# energy2_1, energy_time2_1 = read_energy(folder2, 'energy/frozen-water-h-24hr-cg-all.out')
# energy2_2, energy_time2_2 = read_energy(folder2, 'energy/frozen-water-h_from-neutral-24hr-cg.out')
energy3_1, energy_time3_1 = read_energy(folder3, 'energy/frozen-water-h-24hr-cg-all.out')
energy3_2, energy_time3_2 = read_energy(folder3, 'energy/frozen-water-h_from-neutral-24hr-cg.out')
energy4_1, energy_time4_1 = read_energy(folder4, 'energy/frozen-water-h-24hr-cg-all.out')
energy4_2, energy_time4_2 = read_energy(folder4, 'energy/frozen-water-h_from-neutral-24hr-cg-all.out')

# Read RMS force
force_rms1_1, force_rms_time1_1 = read_force(folder1, 'force-rms/frozen-water-h-24hr-cg-all.out')
force_rms1_2, force_rms_time1_2 = read_force(folder1, 'force-rms/frozen-water-h_from-neutral-24hr-cg.out')
# force_rms2_1, force_rms_time2_1 = read_force(folder2, 'force-rms/frozen-water-h-24hr-cg-all.out')
# force_rms2_2, force_rms_time2_2 = read_force(folder2, 'force-rms/frozen-water-h_from-neutral-24hr-cg.out')
force_rms3_1, force_rms_time3_1 = read_force(folder3, 'force-rms/frozen-water-h-24hr-cg-all.out')
force_rms3_2, force_rms_time3_2 = read_force(folder3, 'force-rms/frozen-water-h_from-neutral-24hr-cg.out')
force_rms4_1, force_rms_time4_1 = read_force(folder4, 'force-rms/frozen-water-h-24hr-cg-all.out')
force_rms4_2, force_rms_time4_2 = read_force(folder4, 'force-rms/frozen-water-h_from-neutral-24hr-cg.out')

# Read Max force
force_max1_1, force_max_time1_1 = read_force(folder1, 'force-max/frozen-water-h-24hr-cg-all.out')
force_max1_2, force_max_time1_2 = read_force(folder1, 'force-max/frozen-water-h_from-neutral-24hr-cg.out')
# force_max2_1, force_max_time2_1 = read_force(folder2, 'force-max/frozen-water-h-24hr-cg-all.out')
# force_max2_2, force_max_time2_2 = read_force(folder2, 'force-max/frozen-water-h_from-neutral-24hr-cg.out')
force_max3_1, force_max_time3_1 = read_force(folder3, 'force-max/frozen-water-h-24hr-cg-all.out')
force_max3_2, force_max_time3_2 = read_force(folder3, 'force-max/frozen-water-h_from-neutral-24hr-cg.out')
force_max4_1, force_max_time4_1 = read_force(folder4, 'force-max/frozen-water-h-24hr-cg-all.out')
force_max4_2, force_max_time4_2 = read_force(folder4, 'force-max/frozen-water-h_from-neutral-24hr-cg.out')

# Read all force
# filename1_brent = 'brent/frozen-water-h-24hr-cg.out'
# filename1_mnbrack = 'mnbrack/frozen-water-h-24hr-cg.out'
# file_spec1, num_data1, step1, brent1, mnbrack1 = load_forces_cg.load_file_forces(
#         folder1, 'force-all/frozen-water-h-24hr-cg.out', num_atoms, filename1_brent, filename1_mnbrack)

# Plot energy
xlim = energy_time1_1
fig_energy, ax_energy = plt.subplots()
ax_energy.plot([0, 1e3], [energy1_2['4'][0], energy1_2['4'][0]], 'k--', alpha=0.5)
ax_energy.plot([0, 1e3], [energy1_2['4'][energy_time1_2.shape[0]-1], energy1_2['4'][energy_time1_2.shape[0]-1]],
               'k--', alpha=0.5)
ax_energy.plot(energy_time1_1, energy1_1['4'], 'r--', label='MD structure')
ax_energy.plot(energy_time3_1, energy3_1['4'], 'g--', label='MD structure (offset D)')
ax_energy.plot(energy_time4_1, energy4_1['4'], 'b--', label='MD structure (offset F)')
ax_energy.plot(energy_time1_2, energy1_2['4'], 'r-', label='Neutral structure')
# ax_energy.plot(energy_time3_2, energy3_2['4'], 'g-', label='Neutral structure (offset B)')
ax_energy.plot(energy_time4_2, energy4_2['4'], 'b-', label='Neutral structure (offset F)')
# ax_energy.plot(energy_time2_1, energy2_1['4'], 'r-', label='CG')
# ax_energy.plot([0, 12], [energy3['4'][0], energy3['4'][0]], 'k--', label='Vertical')
ax_energy.set_xlabel('Geometry optimisation step')
ax_energy.set_ylabel('Energy / au')
# ax_energy.set_ylim([-8895.15, -8894.65])
# ax_energy.plot(energy_time3[0:13], energy3['4'][0:13], 'k-')
ax_energy.set_xlim([0, 60])
ax_energy.set_ylim([-8895.295, -8895.275])
fig_energy.tight_layout()
fig_energy.savefig('{}/energy2.png'.format(folder1), dpi=300, bbbox_inches='tight')
ax_energy.legend(frameon=False)
ax_energy.set_ylim([-8895.295, -8894.8])
fig_energy.tight_layout()
fig_energy.savefig('{}/energy.png'.format(folder1), dpi=300, bbbox_inches='tight')


# Plot RMS force_rms
# xlim = force_rms_time1_1+1
# fig_force_rms, ax_force_rms = plt.subplots()
# ax_force_rms.plot(force_rms_time1_1, force_rms1_1['4'], 'r--', label='MD structure')
# ax_force_rms.plot(force_rms_time3_1, force_rms3_1['4'], 'g--', label='MD structure (offset D)')
# ax_force_rms.plot(force_rms_time4_1, force_rms4_1['4'], 'b--', label='MD structure (offset F)')
# ax_force_rms.plot(force_rms_time1_2, force_rms1_2['4'], 'r-', label='Neutral structure')
# ax_force_rms.plot(force_rms_time4_2, force_rms4_2['4'], 'b-', label='Neutral structure (offset F)')
# ax_force_rms.plot([0, 1e3], [3e-4, 3e-4], 'k--')
# ax_force_rms.set_yscale('log')
# ax_force_rms.set_xlabel('Geometry optimisation step')
# ax_force_rms.set_ylabel('RMS force / au')
# ax_force_rms.set_xlim([xlim[0], xlim[-1]])
# ax_force_rms.set_ylim([2e-4, 2e-1])
# ax_force_rms.legend(frameon=False)
# fig_force_rms.tight_layout()
# fig_force_rms.savefig('{}/force_rms.png'.format(folder1), dpi=300, bbbox_inches='tight')

# Plot MAX force_rms
# xlim = force_max_time1_1+1
# fig_force_max, ax_force_max = plt.subplots()
# ax_force_max.plot(force_max_time1_1, force_max1_1['4'], 'r--', label='MD structure')
# ax_force_max.plot(force_max_time3_1, force_max3_1['4'], 'g--', label='MD structure (offset D)')
# ax_force_max.plot(force_max_time4_1, force_max4_1['4'], 'b--', label='MD structure (offset F)')
# ax_force_max.plot(force_max_time1_2, force_max1_2['4'], 'r-', label='Neutral structure')
# ax_force_max.plot(force_max_time4_2, force_max4_2['4'], 'b-', label='Neutral structure (offset F)')
# ax_force_max.plot([0, 1e3], [3e-4, 3e-4], 'k--')
# ax_force_max.set_yscale('log')
# ax_force_max.set_xlabel('Geometry optimisation step')
# ax_force_max.set_ylabel('max force / au')
# ax_force_max.set_xlim([xlim[0], xlim[-1]])
# ax_force_max.set_ylim([2e-4, 2e-1])
# ax_force_max.legend(frameon=False)
# fig_force_max.tight_layout()
# fig_force_max.savefig('{}/force_max.png'.format(folder1), dpi=300, bbbox_inches='tight')

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
# mean_fe_force = np.zeros(num_data1)
# mean_fe_alpha_force = np.zeros(num_data1)
# mean_fe_beta_force = np.zeros(num_data1)
# mean_fe_beta_a_force = np.zeros(num_data1)
# mean_fe_beta_b_force = np.zeros(num_data1)
# mean_fe_beta_c_force = np.zeros(num_data1)
# mean_fe_beta_d_force = np.zeros(num_data1)
# mean_fe_beta_e_force = np.zeros(num_data1)
# mean_fe_beta_f_force = np.zeros(num_data1)
# mean_water_force = np.zeros(num_data1)
# mean_o_force = np.zeros(num_data1)
# mean_h_force = np.zeros(num_data1)

# Build arrays
# k = 0
# force_plot = 'Z'
# for j in range(num_data1):
#     k = k + brent1[j] + 1
#     i = k - 1

    # mean_fe_alpha_force[j] = np.sqrt(np.mean((file_spec1[force_plot][num_atoms * i + skip * i + fe_alpha] ** 2)))
    # mean_fe_beta_force[j] = np.sqrt(np.mean((file_spec1[force_plot][num_atoms * i + skip * i + fe_beta] ** 2)))
    # mean_fe_beta_a_force[j] = np.sqrt(np.mean((file_spec1[force_plot][num_atoms * i + skip * i + fe_a] ** 2)))
    # mean_fe_beta_b_force[j] = np.sqrt(np.mean((file_spec1[force_plot][num_atoms * i + skip * i + fe_b] ** 2)))
    # mean_fe_beta_c_force[j] = np.sqrt(np.mean((file_spec1[force_plot][num_atoms * i + skip * i + fe_c] ** 2)))
    # mean_fe_beta_d_force[j] = np.sqrt(np.mean((file_spec1[force_plot][num_atoms * i + skip * i + fe_d] ** 2)))
    # mean_fe_beta_e_force[j] = np.sqrt(np.mean((file_spec1[force_plot][num_atoms * i + skip * i + fe_e] ** 2)))
    # mean_fe_beta_f_force[j] = np.sqrt(np.mean((file_spec1[force_plot][num_atoms * i + skip * i + fe_f] ** 2)))
    # mean_o_force[j] = np.sqrt(np.mean((file_spec1[force_plot][num_atoms * i + skip * i + o_all] ** 2)))
    # mean_fe_force[j] = np.sqrt(np.mean((file_spec1[force_plot][num_atoms * i + skip * i + fe_all] ** 2)))
    #
    # mean_fe_alpha_force[j] = np.max(np.abs((file_spec1[force_plot][num_atoms * i + skip * i + fe_alpha])))
    # mean_fe_beta_force[j] = np.max(np.abs((file_spec1[force_plot][num_atoms * i + skip * i + fe_beta])))
    # mean_fe_beta_a_force[j] = np.max(np.abs((file_spec1[force_plot][num_atoms * i + skip * i + fe_a])))
    # mean_fe_beta_b_force[j] = np.max(np.abs((file_spec1[force_plot][num_atoms * i + skip * i + fe_b])))
    # mean_fe_beta_c_force[j] = np.max(np.abs((file_spec1[force_plot][num_atoms * i + skip * i + fe_c])))
    # mean_fe_beta_d_force[j] = np.max(np.abs((file_spec1[force_plot][num_atoms * i + skip * i + fe_d])))
    # mean_fe_beta_e_force[j] = np.max(np.abs((file_spec1[force_plot][num_atoms * i + skip * i + fe_e])))
    # mean_fe_beta_f_force[j] = np.max(np.abs((file_spec1[force_plot][num_atoms * i + skip * i + fe_f])))
    # mean_o_force[j] = np.max(np.abs((file_spec1[force_plot][num_atoms * i + skip * i + o_all])))
    # mean_fe_force[j] = np.max(np.abs((file_spec1[force_plot][num_atoms * i + skip * i + fe_all])))

# Plot all forces
# plot_x = force_max_time1_1
# fig_spin, ax_spin = plt.subplots()
# ax_spin.plot(plot_x, (mean_o_force), 'kx-', label='O')
# ax_spin.plot(plot_x, (mean_fe_alpha_force), 'rx-', label='Fe alpha')
# ax_spin.plot(plot_x, (mean_fe_beta_force), 'gx-', label='Fe beta')
# ax_spin.plot(plot_x, (mean_fe_beta_a_force), 'rx-', label='Fe a')
# ax_spin.plot(plot_x, (mean_fe_beta_b_force), 'gx-', label='Fe b')
# ax_spin.plot(plot_x, (mean_fe_beta_c_force), 'bx-', label='Fe c')
# ax_spin.plot(plot_x, (mean_fe_beta_d_force), 'yx-', label='Fe d')
# ax_spin.plot(plot_x, (mean_fe_beta_e_force), 'mx-', label='Fe e')
# ax_spin.plot(plot_x, (mean_fe_beta_f_force), 'x-', color='grey', label='Fe f')
# ax_spin.set_yscale('log')
# ax_spin.plot([0, 1e3], [4.5e-4, 4.5e-4], 'k--')
# # ax_spin.plot([0, 1e3], [3e-4, 3e-4], 'k--')
# ax_spin.set_xlabel('Geometry optimisation step')
# ax_spin.set_ylabel('Max force / au')
# ax_spin.set_xlim([0, plot_x[-1]])
# ax_spin.set_ylim([2e-4, 2e-1])
# ax_spin.legend(frameon=False, loc='upper right')
# fig_spin.tight_layout()
# fig_spin.savefig('{}/force_max_z2.png'.format(folder1), dpi=300, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
