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


"""
    Plot energy and forces for bulk hematite CDFT
"""

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


skip = 2
atoms = 120

folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hematite-conservation/221_supercell_cdft/electron-struct1/final/analysis'
index_fe_1 = np.array([6, 15, 17, 4]) - 1
energy_kinetic1_1, energy_potential1_1, energy_total1_1, temperature1_1, time_val1_1, time_per_step1_1 = load_energy.load_values_energy(folder_1, '/energy/dft-print.out')
energy_kinetic2_1, energy_potential2_1, energy_total2_1, temperature2_1, time_val2_1, time_per_step2_1 = load_energy.load_values_energy(folder_1, '/energy/cdft-1-372_cdft-3e-3-print.out')
energy_kinetic3_1, energy_potential3_1, energy_total3_1, temperature3_1, time_val3_1, time_per_step3_1 = load_energy.load_values_energy(folder_1, '/energy/cdft-744_cdft-3e-3-print.out')
energy_kinetic4_1, energy_potential4_1, energy_total4_1, temperature4_1, time_val4_1, time_per_step4_1 = load_energy.load_values_energy(folder_1, '/energy/cdft-1-372_cdft-3e-3-print-rs.out')
file_spec1_1, species1_1, num_data1_1, step1_1, brent1_1, mnbrack1_1 = read_hirsh(folder_1, '/hirshfeld/dft-print.out', atoms, None, None)
file_spec2_1, species2_1, num_data2_1, step2_1, brent2_1, mnbrack2_1 = read_hirsh(folder_1, '/hirshfeld/cdft-1-372_cdft-3e-3-print.out', atoms, None, None)
file_spec3_1, species3_1, num_data3_1, step3_1, brent3_1, mnbrack3_1 = read_hirsh(folder_1, '/hirshfeld/cdft-744_cdft-3e-3-print.out', atoms, None, None)
force2_1, forces_x2_1, forces_y2_1, forces_z2_1, num_atoms2_1, num_timesteps2_1 = load_forces.load_values_forces(folder_1, 'force/cdft-1-372_cdft-3e-3-print.out')
force3_1, forces_x3_1, forces_y3_1, forces_z3_1, num_atoms3_1, num_timesteps3_1 = load_forces.load_values_forces(folder_1, 'force/cdft-744_cdft-3e-3-print.out')

folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hematite-conservation/221_supercell_cdft/electron-struct2/md/analysis'
index_fe_2 = np.array([2, 13]) - 1
energy_kinetic1_2, energy_potential1_2, energy_total1_2, temperature1_2, time_val1_2, time_per_step1_2 = load_energy.load_values_energy(folder_2, '/energy/dft.out')
energy_kinetic2_2, energy_potential2_2, energy_total2_2, temperature2_2, time_val2_2, time_per_step2_2 = load_energy.load_values_energy(folder_2, '/energy/cdft-1-372_cdft-3e-3-print.out')
energy_kinetic3_2, energy_potential3_2, energy_total3_2, temperature3_2, time_val3_2, time_per_step3_2 = load_energy.load_values_energy(folder_2, '/energy/cdft-2-744_cdft-3e-3-print.out')
file_spec1_2, species1_2, num_data1_2, step1_2, brent1_2, mnbrack1_2 = read_hirsh(folder_2, '/hirshfeld/dft.out', atoms, None, None)
file_spec2_2, species2_2, num_data2_2, step2_2, brent2_2, mnbrack2_2 = read_hirsh(folder_2, '/hirshfeld/cdft-1-372_cdft-3e-3-print.out', atoms, None, None)
file_spec3_2, species3_2, num_data3_2, step3_2, brent3_2, mnbrack3_2 = read_hirsh(folder_2, '/hirshfeld/cdft-2-744_cdft-3e-3-print.out', atoms, None, None)
force1_2, forces_x1_2, forces_y1_2, forces_z1_2, num_atoms1_2, num_timesteps1_2 = load_forces.load_values_forces(folder_2, 'force/dft.out')
force2_2, forces_x2_2, forces_y2_2, forces_z2_2, num_atoms2_2, num_timesteps2_2 = load_forces.load_values_forces(folder_2, 'force/cdft-1-372_cdft-3e-3-print.out')
force3_2, forces_x3_2, forces_y3_2, forces_z3_2, num_atoms3_2, num_timesteps3_2 = load_forces.load_values_forces(folder_2, 'force/cdft-2-744_cdft-3e-3-print.out')
coordinates1_2, coord_x1_2, coord_y1_2, coord_z1_2, _, _, _ = load_coordinates.load_values_coord(folder_2, 'position/dft.out')
coordinates2_2, coord_x2_2, coord_y2_2, coord_z2_2, _, _, _ = load_coordinates.load_values_coord(folder_2, 'position/cdft-1-372_cdft-3e-3-print.out')
coordinates3_2, coord_x3_2, coord_y3_2, coord_z3_2, _, _, _ = load_coordinates.load_values_coord(folder_2, 'position/cdft-2-744_cdft-3e-3-print.out')

folder_3 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hematite-conservation/221_supercell_cdft/hole-struct1/md/analysis'
index_fe_3 = np.array([2, 13]) - 1
energy_kinetic1_3, energy_potential1_3, energy_total1_3, temperature1_3, time_val1_3, time_per_step1_3 = load_energy.load_values_energy(folder_3, '/energy/dft.out')
energy_kinetic2_3, energy_potential2_3, energy_total2_3, temperature2_3, time_val2_3, time_per_step2_3 = load_energy.load_values_energy(folder_3, '/energy/cdft-1-329_cdft-3e-3-print.out')
energy_kinetic3_3, energy_potential3_3, energy_total3_3, temperature3_3, time_val3_3, time_per_step3_3 = load_energy.load_values_energy(folder_3, '/energy/cdft_fe-o-3.17.out')
energy_kinetic4_3, energy_potential4_3, energy_total4_3, temperature4_3, time_val4_3, time_per_step4_3 = load_energy.load_values_energy(folder_3, '/energy/cdft_fe-o-3.17-2.out')
file_spec1_3, species1_3, num_data1_3, step1_3, brent1_3, mnbrack1_3 = read_hirsh(folder_3, '/hirshfeld/dft.out', atoms, None, None)
file_spec2_3, species2_3, num_data2_3, step2_3, brent2_3, mnbrack2_3 = read_hirsh(folder_3, '/hirshfeld/cdft-1-329_cdft-3e-3-print.out', atoms, None, None)
file_spec3_3, species3_3, num_data3_3, step3_3, brent3_3, mnbrack3_3 = read_hirsh(folder_3, '/hirshfeld/cdft_fe-o-3.17.out', atoms, None, None)
file_spec4_3, species4_3, num_data4_3, step4_3, brent4_3, mnbrack4_3 = read_hirsh(folder_3, '/hirshfeld/cdft_fe-o-3.17-2.out', atoms, None, None)
force1_3, forces_x1_3, forces_y1_3, forces_z1_3, num_atoms1_3, num_timesteps1_3 = load_forces.load_values_forces(folder_3, 'force/dft.out')
force2_3, forces_x2_3, forces_y2_3, forces_z2_3, num_atoms2_3, num_timesteps2_3 = load_forces.load_values_forces(folder_3, 'force/cdft-1-329_cdft-3e-3-print.out')
coordinates1_3, coord_x1_3, coord_y1_3, coord_z1_3, _, _, _ = load_coordinates.load_values_coord(folder_3, 'position/dft.out')
coordinates2_3, coord_x2_3, coord_y2_3, coord_z2_3, _, _, _ = load_coordinates.load_values_coord(folder_3, 'position/cdft-1-329_cdft-3e-3-print.out')

# folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hematite-conservation/221_supercell_cdft/electron-struct1/final'
# name_save = '3p72'
# folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hematite-conservation/221_supercell_cdft/electron-struct2/md'
# name_save = '3p72'
# name_save = '7p44'
folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hematite-conservation/221_supercell_cdft/hole-struct1/md'
name_save = '3p29'

# Plot total energy DFT
time_plot = 100
skip = 0
energy_end = time_plot * 2
fig_energy_dft, ax_energy_dft = plt.subplots()
# ax_energy_dft.plot(time_val1_1[skip:] - time_val1_1[skip], (energy_total1_1[skip:] - energy_total1_1[skip]) / atoms, 'k-', label='DFT')
# ax_energy_dft.plot(time_val2_1[skip:]-time_val2_1[skip], (energy_total2_1[skip:]-energy_total2_1[skip])/atoms, 'r-', label='CDFT 3.72 3e-3')
# ax_energy_dft.plot(time_val4_1-time_val2_1[skip], (energy_total4_1-energy_total2_1[skip])/atoms, 'r-')
# ax_energy_dft.plot(time_val1_2[skip:] - time_val1_2[skip], (energy_total1_2[skip:] - energy_total1_2[skip]) / atoms, 'k-', label='DFT')
# ax_energy_dft.plot(time_val2_2[skip:]-time_val2_2[skip], (energy_total2_2[skip:]-energy_total2_2[skip])/atoms, 'r-', label='CDFT 3.72 3e-3')
# ax_energy_dft.plot(time_val3_2[skip:]-time_val3_2[skip], (energy_total3_2[skip:]-energy_total3_2[skip])/atoms, 'r-', label='CDFT 7.44 3e-3')
ax_energy_dft.plot(time_val1_3[skip:] - time_val1_3[skip], (energy_total1_3[skip:] - energy_total1_3[skip]) / atoms, 'k-', label='DFT')
ax_energy_dft.plot(time_val2_3[skip:]-time_val2_3[skip], (energy_total2_3[skip:]-energy_total2_3[skip])/atoms, 'r-', label='CDFT Fe 3e-3')
ax_energy_dft.plot(time_val3_3[skip:]-time_val3_3[skip], (energy_total3_3[skip:]-energy_total3_3[skip])/atoms, 'g-', label='CDFT Fe, O 3e-3')
ax_energy_dft.plot(time_val4_3[skip:]-time_val3_3[skip], (energy_total4_3[skip:]-energy_total3_3[skip])/atoms, 'g-')
ax_energy_dft.set_xlabel('Time / fs')
ax_energy_dft.set_ylabel('Energy change per atom / Ha')
ax_energy_dft.set_xlim([0, time_plot])
ax_energy_dft.set_ylim([-1e-5, 1e-4])
# ax_energy_dft.set_ylim([-0.75e-5, 1.25e-5])
# ax_energy_dft.set_ylim([-6e-6, 6e-5])
# ax_energy_dft.set_ylim([-6e-6, 6e-6])
# ax_energy_dft.set_ylim([-5e-6, 2e-5])
# ax_energy_dft.set_ylim([-8e-6, 2e-5])
ax_energy_dft.legend(frameon=False)
fig_energy_dft.tight_layout()
fig_energy_dft.savefig('{}/energy_cdft_{}_t{}.png'.format(folder_save, name_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot Hirshfeld analysis of selected atoms
# time_plot = 500
skip = 0
skip_line = 2
plot_index = index_fe_1 #index_fe_3
plot_quantity = 'Spin'
fig_hirshfeld, ax_hirshfeld = plt.subplots()
# temp1 = np.zeros(num_data1_1)
# temp2 = np.zeros(num_data1_1)
# temp3 = np.zeros(num_data1_1)
# i = -1
# for n in range(num_data1_1):
#     i = i + 1
#     temp1[n] = (file_spec1_1.loc[atoms * i + skip_line * i + plot_index[0], plot_quantity])
#     temp2[n] = (file_spec1_1.loc[atoms * i + skip_line * i + plot_index[1], plot_quantity])
#     temp3[n] = (file_spec1_1.loc[atoms * i + skip_line * i + plot_index[3], plot_quantity])
# ax_hirshfeld.plot(time_val1_1[skip:]-time_val1_1[skip], temp1[skip:-1], 'r-', label='Fe 1')
# ax_hirshfeld.plot(time_val1_1[skip:]-time_val1_1[skip], temp2[skip:-1], 'g-', label='Fe 2')
# ax_hirshfeld.plot(time_val1_1[skip:]-time_val1_1[skip], temp3[skip:-1], 'b-', label='Fe 3')
# print(np.mean(temp1[skip:]))
# print(np.mean(temp2[skip:]))
# print(np.mean((temp1[skip:]+temp2[skip:])/2))
# ax_hirshfeld.plot(time_val1_1[skip:]-time_val1_1[skip], temp3[skip:], 'k-', alpha=0.4)
# ax_hirshfeld.plot(time_val1_1[skip:] - time_val1_1[skip], (temp1[skip:-1]+temp2[skip:-1])/2, 'k--', label='Fe mean 1-2')
# ax_hirshfeld.plot(time_val1_1[skip:] - time_val1_1[skip], (temp1[skip:-1]+temp3[skip:-1])/2, '--', color='grey', label='Fe mean 1-3')
# temp1 = np.zeros(num_data2_2)
# temp2 = np.zeros(num_data2_2)
# temp3 = np.zeros(num_data2_2)
# i = -1
# for n in range(num_data2_2):
#     i = i + 1
    # temp1[n] = (file_spec2_2.loc[atoms * i + skip_line * i + plot_index[0], 'Spin'])
    # temp2[n] = (file_spec2_2.loc[atoms * i + skip_line * i + plot_index[1], 'Spin'])
    # temp3[n] = (file_spec2_2.loc[atoms * i + skip_line * i + plot_index[2], 'Spin'])
# ax_hirshfeld.plot(time_val2_2[skip:] - time_val2_2[skip], temp1[skip:], 'r-', label='CDFT 3.72 3e-3')
# ax_hirshfeld.plot(time_val2_2[skip:] - time_val2_2[skip], temp2[skip:], 'r-', alpha=0.4)
# ax_hirshfeld.plot(time_val2_2[skip:] - time_val2_2[skip], temp3[skip:], 'r-', alpha=0.4)
# ax_hirshfeld.plot(time_val2_2[skip:] - time_val2_2[skip], (temp1[skip:]+temp2[skip:])/2, 'r--')
# temp1 = np.zeros(num_data3_2)
# temp2 = np.zeros(num_data3_2)
# i = -1
# for n in range(num_data3_2):
#     i = i + 1
#     temp1[n] = (file_spec3_2.loc[atoms * i + skip_line * i + plot_index[0], 'Spin'])
#     temp2[n] = (file_spec3_2.loc[atoms * i + skip_line * i + plot_index[1], 'Spin'])
# ax_hirshfeld.plot(time_val3_2[skip:] - time_val3_2[skip], temp1[skip:-1], 'r-', label='CDFT 7.44 3e-3')
# ax_hirshfeld.plot(time_val3_2[skip:] - time_val3_2[skip], temp2[skip:-1], 'r-', alpha=0.4)
# ax_hirshfeld.plot(time_val3_2[skip:] - time_val3_2[skip], (temp1[skip:-1]+temp2[skip:-1])/2, 'r--')
temp1 = np.zeros(num_data1_3)
i = -1
for n in range(num_data1_3):
    i = i + 1
    temp1[n] = (file_spec1_3.loc[atoms * i + skip_line * i + 12, plot_quantity])
ax_hirshfeld.plot(time_val1_3[skip:] - time_val1_3[skip], temp1[skip:], 'k-', label='DFT')
# ax_hirshfeld.plot([time_val1_3[skip],time_val1_3[-1]], [np.mean(temp1[skip:]), np.mean(temp1[skip:])],'k--', alpha=0.5)
print(np.mean(temp1[skip:]))
temp1 = np.zeros(num_data2_3)
i = -1
for n in range(num_data2_3):
    i = i + 1
    temp1[n] = (file_spec2_3.loc[atoms * i + skip_line * i + 12, 'Spin'])
ax_hirshfeld.plot(time_val2_3[skip:] - time_val2_3[skip], temp1[skip:], 'r-', label='CDFT 3.29 Fe 3e-3')
ax_hirshfeld.set_xlabel('Time / fs')
ax_hirshfeld.set_ylabel('Hirshfeld spin moment')
ax_hirshfeld.set_xlim([0, time_plot])
temp1 = np.zeros(num_data3_3)
i = -1
for n in range(num_data3_3):
    i = i + 1
    temp1[n] = (file_spec3_3.loc[atoms * i + skip_line * i + 12, 'Spin'])
ax_hirshfeld.plot(time_val3_3[skip:] - time_val3_3[skip], temp1[skip:], 'g-', label='CDFT 3.29 Fe, O 3e-3')
temp1 = np.zeros(num_data4_3)
i = -1
for n in range(num_data4_3):
    i = i + 1
    temp1[n] = (file_spec4_3.loc[atoms * i + skip_line * i + 12, 'Spin'])
ax_hirshfeld.plot(time_val4_3[skip:] - time_val3_3[skip], temp1[skip:-1], 'g-')
ax_hirshfeld.set_xlabel('Time / fs')
ax_hirshfeld.set_ylabel('Hirshfeld spin moment')
ax_hirshfeld.set_xlim([0, time_plot])
# ax_hirshfeld.set_ylim([-3.78, -3.67])
ax_hirshfeld.legend(frameon=False)
fig_hirshfeld.tight_layout()
fig_hirshfeld.savefig('{}/spin_{}_t{}.png'.format(folder_save, name_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot Hirshfeld analysis of selected atoms
skip = 0
skip_line = 2
bonds_print = np.array([13, 67, 88, 50, 108, 61, 95])-1
plot_index = atoms
plot_quantity = 'Spin'
fig_hirshfeld2, ax_hirshfeld2 = plt.subplots()
# for j in range(bonds_print.shape[0]):
#     temp1 = np.zeros(num_data2_3)
#     i = -1
#     for n in range(num_data2_3):
#         i = i + 1
#         temp1[n] = (file_spec2_3.loc[atoms * i + skip_line * i + bonds_print[j], 'Spin'])
#     ax_hirshfeld2.plot(time_val2_3[skip:] - time_val2_3[skip], temp1[skip:] - temp1[skip])
for j in range(bonds_print.shape[0]):
    temp1 = np.zeros(num_data3_3)
    temp2 = np.zeros(num_data4_3)
    i = -1
    for n in range(num_data3_3):
        i = i + 1
        temp1[n] = (file_spec3_3.loc[atoms * i + skip_line * i + bonds_print[j], 'Spin'])
    ax_hirshfeld2.plot(time_val3_3[skip:] - time_val3_3[skip], temp1[skip:]-temp1[skip])
    i = -1
    for n in range(num_data4_3):
        i = i + 1
        temp2[n] = (file_spec4_3.loc[atoms * i + skip_line * i + bonds_print[j], 'Spin'])
    ax_hirshfeld2.plot(time_val4_3[skip:] - time_val3_3[skip], temp2[skip:-1] - temp1[skip])
ax_hirshfeld2.set_xlabel('Time / fs')
ax_hirshfeld2.set_ylabel('Change in Hirshfeld spin moment')
ax_hirshfeld2.set_xlim([0, time_plot])
ax_hirshfeld2.set_ylim([-0.15, 0.10])
ax_hirshfeld2.legend(frameon=False)
fig_hirshfeld2.tight_layout()
fig_hirshfeld2.savefig('{}/spin_all_{}_t{}.png'.format(folder_save, name_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot position (hole)
# time_plot = 500
skip = 0
skip_line = 2
bonds_print = index_fe_3
fig_position, ax_position = plt.subplots()
dist1 = functions.calc_distance(coordinates1_2[skip:,0,bonds_print[0]], coordinates1_2[skip:,1,bonds_print[0]], coordinates1_2[skip:,2,bonds_print[0]], coordinates1_2[skip:,0,bonds_print[1]], coordinates1_2[skip:,1,bonds_print[1]], coordinates1_2[skip:,2,bonds_print[1]])
dist2 = functions.calc_distance(coordinates3_2[skip:,0,bonds_print[0]], coordinates3_2[skip:,1,bonds_print[0]], coordinates3_2[skip:,2,bonds_print[0]], coordinates3_2[skip:,0,bonds_print[1]], coordinates3_2[skip:,1,bonds_print[1]], coordinates3_2[skip:,2,bonds_print[1]])
ax_position.plot(time_val1_2[skip:], dist1, 'r-', label='DFT Fe-Fe')
ax_position.plot(time_val3_2[skip:], dist2, 'g-', label='CDFT Fe-Fe')
ax_position.set_xlabel('Time / fs')
ax_position.set_ylabel('Bond length / au')
ax_position.set_xlim([0, time_plot])
# ax_position.set_ylim([-0.25, 0.25])
ax_position.legend(frameon=False)
fig_position.tight_layout()
fig_position.savefig('{}/bond_lengths_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot position (hole)
time_plot = 500
bonds_print = np.array([13, 67, 88, 50, 108, 61, 95])-1
skip = 0
skip_line = 2
a=1
fig_position, ax_position = plt.subplots()
dist1 = functions.calc_distance(coordinates1_3[skip:,0,bonds_print[0]], coordinates1_3[skip:,1,bonds_print[0]], coordinates1_3[skip:,2,bonds_print[0]], coordinates1_3[skip:,0,bonds_print[1]], coordinates1_3[skip:,1,bonds_print[1]], coordinates1_3[skip:,2,bonds_print[1]])
dist2 = functions.calc_distance(coordinates1_3[skip:,0,bonds_print[0]], coordinates1_3[skip:,1,bonds_print[0]], coordinates1_3[skip:,2,bonds_print[0]], coordinates1_3[skip:,0,bonds_print[2]], coordinates1_3[skip:,1,bonds_print[2]], coordinates1_3[skip:,2,bonds_print[2]])
dist3 = functions.calc_distance(coordinates1_3[skip:,0,bonds_print[0]], coordinates1_3[skip:,1,bonds_print[0]], coordinates1_3[skip:,2,bonds_print[0]], coordinates1_3[skip:,0,bonds_print[3]], coordinates1_3[skip:,1,bonds_print[3]], coordinates1_3[skip:,2,bonds_print[3]])
dist4 = functions.calc_distance(coordinates1_3[skip:,0,bonds_print[0]], coordinates1_3[skip:,1,bonds_print[0]], coordinates1_3[skip:,2,bonds_print[0]], coordinates1_3[skip:,0,bonds_print[4]], coordinates1_3[skip:,1,bonds_print[4]], coordinates1_3[skip:,2,bonds_print[4]])
dist5 = functions.calc_distance(coordinates1_3[skip:,0,bonds_print[0]], coordinates1_3[skip:,1,bonds_print[0]], coordinates1_3[skip:,2,bonds_print[0]], coordinates1_3[skip:,0,bonds_print[5]], coordinates1_3[skip:,1,bonds_print[5]], coordinates1_3[skip:,2,bonds_print[5]])
dist6 = functions.calc_distance(coordinates1_3[skip:,0,bonds_print[0]], coordinates1_3[skip:,1,bonds_print[0]], coordinates1_3[skip:,2,bonds_print[0]], coordinates1_3[skip:,0,bonds_print[6]], coordinates1_3[skip:,1,bonds_print[6]], coordinates1_3[skip:,2,bonds_print[6]])
dist_mean = ([dist1 - 2.12 * a, dist2 - 1.96 * a, dist3 - 1.96 * a, dist4 - 1.96 * a, dist5 - 2.12 * a, dist6 - 2.12 * a])
ax_position.plot(time_val1_3[skip:], dist1 - 2.12 * a, 'r-', label='Fe-O 1')
ax_position.plot(time_val1_3[skip:], dist2 - 1.96 * a, 'g-', label='Fe-O 2')
ax_position.plot(time_val1_3[skip:], dist3 - 1.96 * a, 'b-', label='Fe-O 3')
ax_position.plot(time_val1_3[skip:], dist4 - 1.96 * a, 'y-', label='Fe-O 4')
ax_position.plot(time_val1_3[skip:], dist5 - 2.12 * a, 'm-', label='Fe-O 5')
ax_position.plot(time_val1_3[skip:], dist6 - 2.12 * a, '-', color='orange', label='Fe-O 6')
ax_position.plot(time_val1_3[skip:], np.mean(dist_mean, axis=0), 'k-', label='Fe-O avg')
ax_position.set_xlabel('Time / fs')
ax_position.set_ylabel('Bond length / au')
ax_position.set_xlim([0, time_plot])
ax_position.set_ylim([-0.25, 0.25])
ax_position.legend(frameon=False)
fig_position.tight_layout()
fig_position.savefig('{}/position_dft{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot forces
skip = 0
skip_line = 2
plot_index = index_fe_1
fig_forces1, ax_forces1 = plt.subplots()
temp1 = np.zeros(num_data1_3)
temp2 = np.zeros(num_data1_3)
temp3 = np.zeros(num_data1_3)
i = -1
for n in range(num_data1_3):
    i = i + 1
    temp1[n] = np.sum(forces_x1_3[i, :])
    temp2[n] = np.sum(forces_y1_3[i, :])
    temp3[n] = np.sum(forces_z1_3[i, :])
ax_forces1.plot(time_val1_3[skip:], temp1[skip:], 'r-', label='x')
ax_forces1.plot(time_val1_3[skip:], temp2[skip:], 'g-', label='y')
ax_forces1.plot(time_val1_3[skip:], temp3[skip:], 'b-', label='z')
ax_forces1.plot(np.sum([np.abs(temp1[0]), np.abs(temp2[0]), np.abs(temp3[0])]), 'k-', label='sum')
ax_forces1.set_xlabel('Time / fs')
ax_forces1.set_ylabel('Sum of forces on all atoms / au')
ax_forces1.set_xlim([0, time_plot])
ax_forces1.legend(frameon=False)
fig_forces1.tight_layout()
# fig_forces1.savefig('{}/forces_sum_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot forces
# time_plot = 160
skip = 0
skip_line = 2
plot_index = index_fe_3[1]
fig_forces2, ax_forces2 = plt.subplots()
temp1 = np.zeros(num_data1_2)
temp2 = np.zeros(num_data1_2)
temp3 = np.zeros(num_data1_2)
i = -1
for n in range(num_data1_2):
    i = i + 1
    temp1[n] = np.sum(forces_x1_2[i, plot_index])
    temp2[n] = np.sum(forces_y1_2[i, plot_index])
    temp3[n] = np.sum(forces_z1_2[i, plot_index])
ax_forces2.plot(time_val1_2[skip:], temp1[skip:], 'r--', label='DFT x')
ax_forces2.plot(time_val1_2[skip:], temp2[skip:], 'g--', label='DFT y')
ax_forces2.plot(time_val1_2[skip:], temp3[skip:], 'b--', label='DFT z')
temp1 = np.zeros(num_data3_2)
temp2 = np.zeros(num_data3_2)
temp3 = np.zeros(num_data3_2)
i = -1
print(num_data3_2)
print(forces_x3_2.shape)
for n in range(num_data3_2-1):
    i = i + 1
    temp1[n] = np.sum(forces_x3_2[i, plot_index])
    temp2[n] = np.sum(forces_y3_2[i, plot_index])
    temp3[n] = np.sum(forces_z3_2[i, plot_index])
ax_forces2.plot(time_val3_2[skip:], temp1[skip:-1], 'r-', label='CDFT x')
ax_forces2.plot(time_val3_2[skip:], temp2[skip:-1], 'g-', label='CDFT y')
ax_forces2.plot(time_val3_2[skip:], temp3[skip:-1], 'b-', label='CDFT z')
ax_forces2.set_xlabel('Time / fs')
ax_forces2.set_ylabel('Force / au')
ax_forces2.set_xlim([0, time_plot])
# ax_forces2.set_ylim([-0.015, 0.015])
ax_forces2.legend(frameon=False)
fig_forces2.tight_layout()
fig_forces2.savefig('{}/force_cdft_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
