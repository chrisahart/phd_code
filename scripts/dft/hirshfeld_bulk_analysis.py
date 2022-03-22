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
folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/analysis'

folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hematite-conservation/221_supercell_cdft/electron-struct1/final/analysis'
index_fe_1 = np.array([6, 15, 17, 4]) - 1
energy_kinetic1_1, energy_potential1_1, energy_total1_1, temperature1_1, time_val1_1, time_per_step1_1 = load_energy.load_values_energy(folder_1, '/energy/dft-print.out')
file_spec1_1, species1_1, num_data1_1, step1_1, brent1_1, mnbrack1_1 = read_hirsh(folder_1, '/hirshfeld/dft-print.out', atoms, None, None)

folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hematite-conservation/221_supercell_cdft/electron-struct2/md/analysis'
index_fe_2 = np.array([2, 13]) - 1
energy_kinetic1_2, energy_potential1_2, energy_total1_2, temperature1_2, time_val1_2, time_per_step1_2 = load_energy.load_values_energy(folder_2, '/energy/dft.out')
file_spec1_2, species1_2, num_data1_2, step1_2, brent1_2, mnbrack1_2 = read_hirsh(folder_2, '/hirshfeld/dft.out', atoms, None, None)
force1_2, forces_x1_2, forces_y1_2, forces_z1_2, num_atoms1_2, num_timesteps1_2 = load_forces.load_values_forces(folder_2, 'force/dft.out')
coordinates1_2, coord_x1_2, coord_y1_2, coord_z1_2, _, _, _ = load_coordinates.load_values_coord(folder_2, 'position/dft.out')

folder_3 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hematite-conservation/221_supercell_cdft/hole-struct1/md/analysis'
index_fe_3 = np.array([2, 13]) - 1
energy_kinetic1_3, energy_potential1_3, energy_total1_3, temperature1_3, time_val1_3, time_per_step1_3 = load_energy.load_values_energy(folder_3, '/energy/dft.out')
file_spec1_3, species1_3, num_data1_3, step1_3, brent1_3, mnbrack1_3 = read_hirsh(folder_3, '/hirshfeld/dft.out', atoms, None, None)
force1_3, forces_x1_3, forces_y1_3, forces_z1_3, num_atoms1_3, num_timesteps1_3 = load_forces.load_values_forces(folder_3, 'force/dft.out')
coordinates1_3, coord_x1_3, coord_y1_3, coord_z1_3, _, _, _ = load_coordinates.load_values_coord(folder_3, 'position/dft.out')

# folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hematite-conservation/221_supercell_cdft/electron-struct1/final'
# name_save = '3p72'
# folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hematite-conservation/221_supercell_cdft/electron-struct2/md'
# name_save = '3p72'
# name_save = '7p44'
# folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hematite-conservation/221_supercell_cdft/hole-struct1/md'
# name_save = '3p29'

# Atomic index (221 bulk)
h_all = np.NaN
water = np.NaN
fe_beta = np.array([1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 25, 26, 27, 28, 29, 30, 37, 38, 41, 42, 45, 46]) - 1
fe_alpha = np.array([7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24, 31, 32, 33, 34, 35, 36, 39, 40, 43, 44, 47, 48]) - 1
o_all = np.linspace(start=49, stop=120, num=120-49+1, dtype=int) - 1
num_species = np.array([len(o_all), len(fe_alpha), len(fe_beta)])

# Plot Hirshfeld analysis of selected atoms
time_plot = 500
skip = 0
skip_line = 2
plot_index = index_fe_1
plot_quantity = 'Spin'
fig_hirshfeld, ax_hirshfeld = plt.subplots()
temp1 = np.zeros(num_data1_3)
i = -1
for n in range(num_data1_3):
    i = i + 1
    temp1[n] = (file_spec1_3.loc[atoms * i + skip_line * i + 12, plot_quantity])
ax_hirshfeld.plot(time_val1_3[skip:] - time_val1_3[skip], temp1[skip:], 'k-', label='DFT')
ax_hirshfeld.plot([time_val1_3[skip],time_val1_3[-1]], [np.mean(temp1[skip:]), np.mean(temp1[skip:])],'k--', alpha=0.5)
print(np.mean(temp1[skip:]))
ax_hirshfeld.set_xlabel('Time / fs')
ax_hirshfeld.set_ylabel('Hirshfeld spin moment')
ax_hirshfeld.set_xlim([0, time_plot])
# ax_hirshfeld.set_ylim([-3.78, -3.67])
ax_hirshfeld.legend(frameon=False)
fig_hirshfeld.tight_layout()
# fig_hirshfeld.savefig('{}/spin_{}_t{}.png'.format(folder_save, name_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot all iron spin
fig_spin2, ax_spin2 = plt.subplots()
x_end = time_val1_3[-1]-time_val1_3[0]
temp1 = np.zeros(num_data1_3)
temp2 = np.zeros(num_data1_3)
temp3 = np.zeros(num_data1_3)
for j in range(len(fe_beta)):
    for n in range(num_data1_3):
        temp1[n] = (file_spec1_3.loc[0 + atoms * n + 2 * n + fe_beta[j], 'Spin'])
    ax_spin2.plot(time_val1_3-time_val1_3[0], temp1, 'k-')
temp4 = np.zeros(num_data1_3)
for j in range(len(index_fe_2)):
    for n in range(num_data1_3):
        temp4[n] = (file_spec1_3.loc[0 + atoms * n + 2 * n + 12, 'Spin'])
    ax_spin2.plot(time_val1_3-time_val1_3[0], temp4, 'r--')
ax_spin2.plot(time_val1_3[0], temp4[0], 'r--', label='Polaron')
ax_spin2.legend(frameon=True)
ax_spin2.plot([0, x_end], [-3.29, -3.29], '--', label='Bulk', color='grey')
# print(np.mean(temp4))
ax_spin2.set_xlabel('Time / fs')
ax_spin2.set_ylabel('Spin moment')
ax_spin2.set_ylim(-4, -3.1)
ax_spin2.set_xlim([0, x_end])
fig_spin2.tight_layout()
fig_spin2.savefig('{}/spin.png'.format(folder_save), dpi=300, bbbox_inches='tight')

# Plot position (hole)
time_plot = 500
bonds_print = np.array([13, 67, 88, 50, 108, 61, 95])-1
skip = 0
skip_line = 2
a=0
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
# fig_position.savefig('{}/position_dft{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
