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
# from scripts.dft import cdft_beta


"""
    Plot energy and forces for hematite interface 
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

atoms = 435
skip = 2

folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/hole/archer2/analysis'
energy_kinetic1_1, energy_potential1_1, energy_total1_1, temperature1_1, time_val1_1, time_per_step1_1 = load_energy.load_values_energy(folder_1, '/energy/run-000.out')
energy_kinetic2_1, energy_potential2_1, energy_total2_1, temperature2_1, time_val2_1, time_per_step2_1 = load_energy.load_values_energy(folder_1, '/energy/run-001.out')
energy_kinetic3_1, energy_potential3_1, energy_total3_1, temperature3_1, time_val3_1, time_per_step3_1 = load_energy.load_values_energy(folder_1, '/energy/run-000-rs-cdft.out')
energy_kinetic4_1, energy_potential4_1, energy_total4_1, temperature4_1, time_val4_1, time_per_step4_1 = load_energy.load_values_energy(folder_1, '/energy/run-000-rs-cdft_dt-001.out')
file_spec1_1, species1_1, num_data1_1, step1_1, brent1_1, mnbrack1_1 = read_hirsh(folder_1, '/hirshfeld/run-000.out', atoms, None, None)
file_spec2_1, species2_1, num_data2_1, step2_1, brent2_1, mnbrack2_1 = read_hirsh(folder_1, '/hirshfeld/run-001.out', atoms, None, None)
file_spec3_1, species3_1, num_data3_1, step3_1, brent3_1, mnbrack3_1 = read_hirsh(folder_1, '/hirshfeld/run-000-rs-cdft.out', atoms, None, None)
file_spec4_1, species4_1, num_data4_1, step4_1, brent4_1, mnbrack4_1 = read_hirsh(folder_1, '/hirshfeld/run-000-rs-cdft_dt-001.out', atoms, None, None)
index_fe_1 = np.array([129]) - 1

index_fe = index_fe_1
folder_save = folder_1

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

# Plot total energy
time_plot = 25
energy_end = time_plot*2
fig_energy, ax_energy = plt.subplots()
# ax_energy.plot(time_val1_1-time_val1_1[0], (energy_kinetic1_1-energy_kinetic1_1[0])/atoms, 'k.-', label='DFT-MD hole (potential)')
# ax_energy.plot(time_val1_1-time_val1_1[0], (energy_potential1_1-energy_potential1_1[0])/atoms, 'k.-', color='grey', label='DFT-MD hole (kinetic)')
# ax_energy.plot(time_val1_1-time_val1_1[0], (energy_total1_1-energy_total1_1[0])/atoms, 'k.-', label='DFT-MD hole')
ax_energy.plot(time_val2_1-time_val2_1[0], (energy_kinetic2_1-energy_kinetic2_1[0])/atoms, 'k.-', label='DFT-MD hole (kinetic)')
ax_energy.plot(time_val2_1-time_val2_1[0], (energy_potential2_1-energy_potential2_1[0])/atoms, 'k.-', color='grey', label='DFT-MD hole (potential)')
# ax_energy.plot(time_val2_1-time_val2_1[0], (energy_total2_1-energy_total2_1[0])/atoms, 'k.-', label='DFT-MD hole')
ax_energy.plot(time_val3_1-time_val3_1[0], (energy_kinetic3_1-energy_kinetic3_1[0])/atoms, 'r.-', label='CDFT-MD hole (kinetic)')
ax_energy.plot(time_val3_1-time_val3_1[0], (energy_potential3_1-energy_potential3_1[0])/atoms, 'r.-', color='orange', label='CDFT-MD hole (potential)')
# ax_energy.plot(time_val3_1-time_val3_1[0], (energy_total3_1-energy_total3_1[0])/atoms, 'r.-', label='CDFT-MD hole')
ax_energy.plot(time_val4_1 - time_val4_1[0], (energy_kinetic4_1 - energy_kinetic4_1[0]) / atoms, 'r.--',label='CDFT-MD hole dt=0.01 (kinetic)')
ax_energy.plot(time_val4_1 - time_val4_1[0], (energy_potential4_1 - energy_potential4_1[0]) / atoms, 'r.--',color='orange', label='CDFT-MD hole dt=0.01 (potential)')
# ax_energy.plot(time_val4_1-time_val4_1[0], (energy_total4_1-energy_total4_1[0])/atoms, 'r.-', label='CDFT-MD hole')
ax_energy.set_xlabel('Time / fs')
ax_energy.set_ylabel('Energy change per atom / Ha')
ax_energy.set_xlim([0, time_plot])
# ax_energy.set_ylim([-5e-6, 5e-4])
ax_energy.legend(frameon=False)
fig_energy.tight_layout()
fig_energy.savefig('{}/energy_all_{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot temperature
time_plot = 20
energy_end = time_plot*2
fig_temperature, ax_temperature = plt.subplots()
ax_temperature.plot(time_val2_1-time_val2_1[0], temperature2_1, 'k.-', label='DFT-MD hole')
ax_temperature.plot(time_val3_1-time_val3_1[0], temperature3_1, 'r.-', label='CDFT-MD hole')
ax_temperature.set_xlabel('Time / fs')
ax_temperature.set_ylabel('Temperature / K')
ax_temperature.set_xlim([0, time_plot])
ax_temperature.set_ylim([300, 1000])
ax_temperature.legend(frameon=False)
fig_temperature.tight_layout()
fig_temperature.savefig('{}/temperature{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot all iron spin DFT vs CDFT
fig_spin_cdft, ax_spin_cdft = plt.subplots()
x_end = 25
temp1 = np.zeros(num_data2_1)
temp2 = np.zeros(num_data2_1)
temp3 = np.zeros(num_data2_1)
for j in range(len(fe_b)):
    k = 0
    for n in range(num_data2_1):
        k = k + brent2_1[n] + mnbrack2_1[n] + 1
        i = k - 1
        temp1[n] = (file_spec2_1.loc[atoms * i + skip * i + fe_b[j], 'Spin'])
        temp2[n] = (file_spec2_1.loc[atoms * i + skip * i + fe_d[j], 'Spin'])
        temp3[n] = (file_spec2_1.loc[atoms * i + skip * i + fe_f[j], 'Spin'])
    ax_spin_cdft.plot(time_val2_1-time_val1_1[0], temp1[:-1], 'r-')
    ax_spin_cdft.plot(time_val2_1-time_val1_1[0], temp2[:-1], 'g-')
    ax_spin_cdft.plot(time_val2_1-time_val1_1[0], temp3[:-1], 'b-')
ax_spin_cdft.plot(time_val2_1[0], temp1[0], 'r-', label='DFT Fe, B')
ax_spin_cdft.plot(time_val2_1[0], temp2[0], 'g-', label='DFT Fe, D')
ax_spin_cdft.plot(time_val2_1[0], temp3[0], 'b-', label='DFT Fe, F')
temp1 = np.zeros(num_data3_1)
temp2 = np.zeros(num_data3_1)
temp3 = np.zeros(num_data3_1)
for j in range(len(fe_b)):
    k = 0
    for n in range(num_data3_1):
        k = k + brent3_1[n] + mnbrack3_1[n] + 1
        i = k - 1
        temp1[n] = (file_spec3_1.loc[atoms * i + skip * i + fe_b[j], 'Spin'])
        temp2[n] = (file_spec3_1.loc[atoms * i + skip * i + fe_d[j], 'Spin'])
        temp3[n] = (file_spec3_1.loc[atoms * i + skip * i + fe_f[j], 'Spin'])
    ax_spin_cdft.plot(time_val3_1 - time_val1_1[0], temp1[:-1], 'r--')
    ax_spin_cdft.plot(time_val3_1 - time_val1_1[0], temp2[:-1], 'g--')
    ax_spin_cdft.plot(time_val3_1 - time_val1_1[0], temp3[:-1], 'b--')
ax_spin_cdft.plot(time_val3_1[0], temp1[0], 'r--', label='CDFT Fe, B')
ax_spin_cdft.plot(time_val3_1[0], temp2[0], 'g--', label='CDFT Fe, D')
ax_spin_cdft.plot(time_val3_1[0], temp3[0], 'b--', label='CDFT Fe, F')
ax_spin_cdft.set_xlabel('Time / fs')
ax_spin_cdft.set_ylabel('Spin moment')
ax_spin_cdft.legend(frameon=True)
ax_spin_cdft.set_ylim(bottom=-4.13)
ax_spin_cdft.set_xlim([time_val2_1[0]-time_val1_1[0], time_val2_1[0]-time_val1_1[0]+x_end])
fig_spin_cdft.tight_layout()
fig_spin_cdft.savefig('{}/fe_spin_all_cdft.png'.format(folder_save), dpi=300, bbbox_inches='tight')

# Plot all iron spin DFT
fig_spin2, ax_spin2 = plt.subplots()
x_end = 450
temp1 = np.zeros(num_data1_1)
temp2 = np.zeros(num_data1_1)
temp3 = np.zeros(num_data1_1)
for j in range(len(fe_b)):
    k = 0
    for n in range(num_data1_1):
        k = k + brent1_1[n] + mnbrack1_1[n] + 1
        i = k - 1
        temp1[n] = (file_spec1_1.loc[atoms * i + skip * i + fe_b[j], 'Spin'])
        temp2[n] = (file_spec1_1.loc[atoms * i + skip * i + fe_d[j], 'Spin'])
        temp3[n] = (file_spec1_1.loc[atoms * i + skip * i + fe_f[j], 'Spin'])
    ax_spin2.plot(time_val1_1-time_val1_1[0], temp1, 'r-')
    ax_spin2.plot(time_val1_1-time_val1_1[0], temp2, 'g-')
    ax_spin2.plot(time_val1_1-time_val1_1[0], temp3, 'b-')
ax_spin2.plot(time_val1_1[0], temp1[0], 'r-', label='Fe, B')
ax_spin2.plot(time_val1_1[0], temp2[0], 'g-', label='Fe, D')
ax_spin2.plot(time_val1_1[0], temp3[0], 'b-', label='Fe, F')
temp1 = np.zeros(num_data2_1)
temp2 = np.zeros(num_data2_1)
temp3 = np.zeros(num_data2_1)
for j in range(len(fe_b)):
    k = 0
    for n in range(num_data2_1):
        k = k + brent2_1[n] + mnbrack2_1[n] + 1
        i = k - 1
        temp1[n] = (file_spec2_1.loc[atoms * i + skip * i + fe_b[j], 'Spin'])
        temp2[n] = (file_spec2_1.loc[atoms * i + skip * i + fe_d[j], 'Spin'])
        temp3[n] = (file_spec2_1.loc[atoms * i + skip * i + fe_f[j], 'Spin'])
    ax_spin2.plot(time_val2_1 - time_val1_1[0], temp1[:-1], 'r-')
    ax_spin2.plot(time_val2_1 - time_val1_1[0], temp2[:-1], 'g-')
    ax_spin2.plot(time_val2_1 - time_val1_1[0], temp3[:-1], 'b-')
temp4 = np.zeros(num_data1_1)
for j in range(len(index_fe)):
    k = 0
    for n in range(num_data1_1):
        k = k + brent1_1[n] + mnbrack1_1[n] + 1
        i = k - 1
        temp4[n] = (file_spec1_1.loc[atoms * i + skip * i + index_fe[j], 'Spin'])
    ax_spin2.plot(time_val1_1-time_val1_1[0], temp4, 'k-')
ax_spin2.plot(time_val1_1[0], temp4[0], 'k-', label='Fe, polaron')
temp4 = np.zeros(num_data2_1)
for j in range(len(index_fe)):
    k = 0
    for n in range(num_data2_1):
        k = k + brent2_1[n] + mnbrack2_1[n] + 1
        i = k - 1
        temp4[n] = (file_spec2_1.loc[atoms * i + skip * i + index_fe[j], 'Spin'])
    ax_spin2.plot(time_val2_1 - time_val1_1[0], temp4[:-1], 'k-')
ax_spin2.plot([0, x_end], [-3.29, -3.29], 'k--', label='Bulk')
print(np.mean(temp4))
ax_spin2.set_xlabel('Time / fs')
ax_spin2.set_ylabel('Spin moment')
ax_spin2.legend(frameon=True)
ax_spin2.set_ylim(bottom=-4.13)
ax_spin2.set_xlim([0, x_end])
fig_spin2.tight_layout()
fig_spin2.savefig('{}/fe_spin_all_dft.png'.format(folder_save), dpi=300, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
