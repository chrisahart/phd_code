from __future__ import division, print_function
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scripts.general import parameters

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


atoms = 435
# x_end = 100

folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/hole/archer2/analysis'
folder_save_1 = folder_1
energy_kinetic1_1, energy_potential1_1, energy_total1_1, temperature1_1, time_val1_1, time_per_step1_1 = read_energy(folder_1, '/energy/run-000.out')
energy_kinetic2_1, energy_potential2_1, energy_total2_1, temperature2_1, time_val2_1, time_per_step2_1 = read_energy(folder_1, '/energy/run-001.out')
file_spec1_1, species1_1 = read_hirsh(folder_1, '/hirshfeld/run-000.out')
file_spec2_1, species2_1 = read_hirsh(folder_1, '/hirshfeld/run-001.out')
num_data1_1 = energy_kinetic1_1.shape[0]
num_data2_1 = energy_kinetic2_1.shape[0]
skip_end_1 = 2
index_fe_1 = np.array([129]) - 1

folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/hole/analysis'
run = '00'
folder_save_2 = folder_2
energy_kinetic1_2, energy_potential1_2, energy_total1_2, temperature1_2, time_val1_2, time_per_step1_2 = read_energy(folder_2, '/energy/{}.out'.format(run))
file_spec1_2, species1_2 = read_hirsh(folder_2, '/hirshfeld/{}.out'.format(run))
num_data1_2 = energy_kinetic1_2.shape[0]
skip_start_2 = 3
skip_end_2 = 5
# polaron_index = {'00': [95], '01': [129]}
# index_fe_2 = np.array(polaron_index[run]) - 1

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
fe_alpha = np.concatenate([fe_a, fe_c, fe_e])
fe_beta = np.concatenate([fe_b, fe_d, fe_f])
fe_all = np.concatenate([fe_a, fe_b, fe_c, fe_d, fe_e, fe_f])

# Printing and plotting arrays
# time1 = np.linspace(start=0, stop=0.5*(num_data1_1-1), num=num_data1_1)
kinds = ['Hematite: H', 'Hematite: O', 'Hematite: Fe1', 'Hematite: Fe2', 'Water']

# Plot all iron spin 1
fig_spin1, ax_spin1 = plt.subplots()
x_end = time_val2_1[-1]-time_val1_1[0]
temp1 = np.zeros(num_data1_1)
temp2 = np.zeros(num_data1_1)
temp3 = np.zeros(num_data1_1)
for j in range(len(fe_b)):
    for n in range(num_data1_1):
        temp1[n] = (file_spec1_1.loc[atoms * n + skip_end_1 * n + fe_b[j], 'Spin'])
        temp2[n] = (file_spec1_1.loc[atoms * n + skip_end_1 * n + fe_d[j], 'Spin'])
        temp3[n] = (file_spec1_1.loc[atoms * n + skip_end_1 * n + fe_f[j], 'Spin'])
    ax_spin1.plot(time_val1_1-time_val1_1[0], temp1, 'r-')
    ax_spin1.plot(time_val1_1-time_val1_1[0], temp2, 'g-')
    ax_spin1.plot(time_val1_1-time_val1_1[0], temp3, 'b-')
ax_spin1.plot(time_val1_1[0], temp1[0], 'r-', label='Fe, B')
ax_spin1.plot(time_val1_1[0], temp2[0], 'g-', label='Fe, D')
ax_spin1.plot(time_val1_1[0], temp3[0], 'b-', label='Fe, F')
temp1 = np.zeros(num_data2_1)
temp2 = np.zeros(num_data2_1)
temp3 = np.zeros(num_data2_1)
for j in range(len(fe_b)):
    for n in range(num_data2_1):
        temp1[n] = (file_spec2_1.loc[atoms * n + skip_end_1 * n + fe_b[j], 'Spin'])
        temp2[n] = (file_spec2_1.loc[atoms * n + skip_end_1 * n + fe_d[j], 'Spin'])
        temp3[n] = (file_spec2_1.loc[atoms * n + skip_end_1 * n + fe_f[j], 'Spin'])
    ax_spin1.plot(time_val2_1 - time_val1_1[0], temp1, 'r-')
    ax_spin1.plot(time_val2_1 - time_val1_1[0], temp2, 'g-')
    ax_spin1.plot(time_val2_1 - time_val1_1[0], temp3, 'b-')
temp4 = np.zeros(num_data1_1)
for j in range(len(index_fe_1)):
    for n in range(num_data1_1):
        temp4[n] = (file_spec1_1.loc[atoms * n + skip_end_1 * n + index_fe_1[j], 'Spin'])
    ax_spin1.plot(time_val1_1-time_val1_1[0], temp4, 'k--')
ax_spin1.plot(time_val1_1[0], temp4[0], 'k--', label='Fe, polaron')
temp4 = np.zeros(num_data2_1)
for j in range(len(index_fe_1)):
    for n in range(num_data2_1):
        temp4[n] = (file_spec2_1.loc[atoms * n + skip_end_1 * n + index_fe_1[j], 'Spin'])
    ax_spin1.plot(time_val2_1 - time_val1_1[0], temp4, 'k--')
ax_spin1.legend(frameon=True)
ax_spin1.plot([0, x_end], [-3.29, -3.29], '--', label='Bulk', color='grey')
print(np.mean(temp4))
ax_spin1.set_xlabel('Time / fs')
ax_spin1.set_ylabel('Spin moment')
ax_spin1.set_ylim(-4.1, -3.1)
ax_spin1.set_xlim([0, x_end])
fig_spin1.tight_layout()
fig_spin1.savefig('{}/fe_spin_all.png'.format(folder_save_1), dpi=300, bbbox_inches='tight')

# Plot all iron spin 1
fig_spin2, ax_spin2 = plt.subplots()
x_end = time_val1_2[-1]-time_val1_2[0]
temp1 = np.zeros(num_data1_2)
temp2 = np.zeros(num_data1_2)
temp3 = np.zeros(num_data1_2)
for j in range(len(fe_b)):
    for n in range(num_data1_2):
        temp1[n] = (file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_b[j], 'Spin'])
        temp2[n] = (file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_d[j], 'Spin'])
        temp3[n] = (file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + fe_f[j], 'Spin'])
    ax_spin2.plot(time_val1_2-time_val1_2[0], temp1, 'r-')
    ax_spin2.plot(time_val1_2-time_val1_2[0], temp2, 'g-')
    ax_spin2.plot(time_val1_2-time_val1_2[0], temp3, 'b-')
ax_spin2.plot(time_val1_2[0], temp1[0], 'r-', label='Fe, B')
ax_spin2.plot(time_val1_2[0], temp2[0], 'g-', label='Fe, D')
ax_spin2.plot(time_val1_2[0], temp3[0], 'b-', label='Fe, F')
# temp4 = np.zeros(num_data1_2)
# for j in range(len(index_fe_2)):
#     for n in range(num_data1_2):
#         temp4[n] = (file_spec1_2.loc[skip_start_2 + atoms * n + skip_end_2 * n + index_fe_2[j], 'Spin'])
#     ax_spin2.plot(time_val1_2-time_val1_2[0], temp4, 'k--')
# ax_spin2.plot(time_val1_2[0], temp4[0], 'k--', label='Fe, polaron')
ax_spin2.legend(frameon=True)
ax_spin2.plot([0, x_end], [-3.29, -3.29], '--', label='Bulk', color='grey')
print(np.mean(temp4))
ax_spin2.set_xlabel('Time / fs')
ax_spin2.set_ylabel('Spin moment')
ax_spin2.set_ylim(-4.13, -3.1)
ax_spin2.set_xlim([0, x_end])
fig_spin2.tight_layout()
fig_spin2.savefig('{}/fe_spin_all_{}.png'.format(folder_save_2, run), dpi=300, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
