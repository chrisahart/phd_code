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
folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hematite-conservation/221_supercell_cdft'

folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hematite-conservation/221_supercell_cdft/analysis'
energy_kinetic1_1, energy_potential1_1, energy_total1_1, temperature1_1, time_val1_1, time_per_step1_1 = load_energy.load_values_energy(folder_1, '/energy/dft.out')
energy_kinetic2_1, energy_potential2_1, energy_total2_1, temperature2_1, time_val2_1, time_per_step2_1 = load_energy.load_values_energy(folder_1, '/energy/cdft-744.out')
energy_kinetic3_1, energy_potential3_1, energy_total3_1, temperature3_1, time_val3_1, time_per_step3_1 = load_energy.load_values_energy(folder_1, '/energy/cdft-744_cdft-1e-4_atomsmemory-20.out')
energy_kinetic10_1, energy_potential10_1, energy_total10_1, temperature10_1, time_val10_1, time_per_step10_1 = load_energy.load_values_energy(folder_1, '/energy/cdft-744_cdft-1e-2_atomsmemory-20.out')
energy_kinetic12_1, energy_potential12_1, energy_total12_1, temperature12_1, time_val12_1, time_per_step12_1 = load_energy.load_values_energy(folder_1, '/energy/cdft-744_cdft-3e-3_atomsmemory-20.out')
energy_kinetic4_1, energy_potential4_1, energy_total4_1, temperature4_1, time_val4_1, time_per_step4_1 = load_energy.load_values_energy(folder_1, '/energy/cdft-1-372.out')
energy_kinetic5_1, energy_potential5_1, energy_total5_1, temperature5_1, time_val5_1, time_per_step5_1 = load_energy.load_values_energy(folder_1, '/energy/cdft-1-372_cdft-1e-4.out')
energy_kinetic11_1, energy_potential11_1, energy_total11_1, temperature11_1, time_val11_1, time_per_step11_1 = load_energy.load_values_energy(folder_1, '/energy/cdft-1-372_cdft-3e-3.out')
energy_kinetic6_1, energy_potential6_1, energy_total6_1, temperature6_1, time_val6_1, time_per_step6_1 = load_energy.load_values_energy(folder_1, '/energy/cdft-2-0.out')
energy_kinetic7_1, energy_potential7_1, energy_total7_1, temperature7_1, time_val7_1, time_per_step7_1 = load_energy.load_values_energy(folder_1, '/energy/cdft-1-360.out')
energy_kinetic8_1, energy_potential8_1, energy_total8_1, temperature8_1, time_val8_1, time_per_step8_1 = load_energy.load_values_energy(folder_1, '/energy/cdft-1-charge-15p7.out')
energy_kinetic9_1, energy_potential9_1, energy_total9_1, temperature9_1, time_val9_1, time_per_step9_1 = load_energy.load_values_energy(folder_1, '/energy/cdft-1-charge-15p647.out')

cdft_iter2_1 = np.loadtxt('{}/cdft-iter/cdft-744.out'.format(folder_1))
cdft_iter3_1 = np.loadtxt('{}/cdft-iter/cdft-744_cdft-1e-4_atomsmemory-20.out'.format(folder_1))
cdft_iter10_1 = np.loadtxt('{}/cdft-iter/cdft-744_cdft-1e-2_atomsmemory-20.out'.format(folder_1))
cdft_iter12_1 = np.loadtxt('{}/cdft-iter/cdft-744_cdft-3e-3_atomsmemory-20.out'.format(folder_1))
cdft_iter4_1 = np.loadtxt('{}/cdft-iter/cdft-1-372.out'.format(folder_1))
cdft_iter5_1 = np.loadtxt('{}/cdft-iter/cdft-1-372_cdft-1e-4.out'.format(folder_1))
cdft_iter11_1 = np.loadtxt('{}/cdft-iter/cdft-1-372_cdft-3e-3.out'.format(folder_1))
cdft_iter6_1 = np.loadtxt('{}/cdft-iter/cdft-2-0.out'.format(folder_1))
cdft_iter7_1 = np.loadtxt('{}/cdft-iter/cdft-1-360.out'.format(folder_1))
cdft_iter8_1 = np.loadtxt('{}/cdft-iter/cdft-1-charge-15p7.out'.format(folder_1))
cdft_iter9_1 = np.loadtxt('{}/cdft-iter/cdft-1-charge-15p647.out'.format(folder_1))

strength2_1 = np.loadtxt('{}/strength/cdft-744.out'.format(folder_1))
strength3_1 = np.loadtxt('{}/strength/cdft-744_cdft-1e-4_atomsmemory-20.out'.format(folder_1))
strength10_1 = np.loadtxt('{}/strength/cdft-744_cdft-1e-2_atomsmemory-20.out'.format(folder_1))
strength12_1 = np.loadtxt('{}/strength/cdft-744_cdft-3e-3_atomsmemory-20.out'.format(folder_1))
strength4_1 = np.loadtxt('{}/strength/cdft-1-372.out'.format(folder_1))
strength5_1 = np.loadtxt('{}/strength/cdft-1-372_cdft-1e-4.out'.format(folder_1))
strength11_1 = np.loadtxt('{}/strength/cdft-1-372_cdft-3e-3.out'.format(folder_1))
strength6_1 = np.loadtxt('{}/strength/cdft-2-0.out'.format(folder_1))
strength7_1 = np.loadtxt('{}/strength/cdft-1-360.out'.format(folder_1))
strength8_1 = np.loadtxt('{}/strength/cdft-1-charge-15p7.out'.format(folder_1))
strength9_1 = np.loadtxt('{}/strength/cdft-1-charge-15p647.out'.format(folder_1))

file_spec1_1, species1_1, num_data1_1, step1_1, brent1_1, mnbrack1_1 = read_hirsh(folder_1, '/hirshfeld/dft.out', atoms, None, None)
file_spec2_1, species2_1, num_data2_1, step2_1, brent2_1, mnbrack2_1 = read_hirsh(folder_1, '/hirshfeld/cdft-744.out', atoms, None, None)
file_spec3_1, species3_1, num_data3_1, step3_1, brent3_1, mnbrack3_1 = read_hirsh(folder_1, '/hirshfeld/cdft-744_cdft-1e-4_atomsmemory-20.out', atoms, None, None)
file_spec10_1, species10_1, num_data10_1, step10_1, brent10_1, mnbrack10_1 = read_hirsh(folder_1, '/hirshfeld/cdft-744_cdft-1e-2_atomsmemory-20.out', atoms, None, None)
file_spec12_1, species12_1, num_data12_1, step12_1, brent12_1, mnbrack12_1 = read_hirsh(folder_1, '/hirshfeld/cdft-744_cdft-3e-3_atomsmemory-20.out', atoms, None, None)
file_spec4_1, species4_1, num_data4_1, step4_1, brent4_1, mnbrack4_1 = read_hirsh(folder_1, '/hirshfeld/cdft-1-372.out', atoms, None, None)
file_spec5_1, species5_1, num_data5_1, step5_1, brent5_1, mnbrack5_1 = read_hirsh(folder_1, '/hirshfeld/cdft-1-372_cdft-1e-4.out', atoms, None, None)
file_spec11_1, species11_1, num_data11_1, step11_1, brent11_1, mnbrack11_1 = read_hirsh(folder_1, '/hirshfeld/cdft-1-372_cdft-3e-3.out', atoms, None, None)
file_spec6_1, species6_1, num_data6_1, step6_1, brent6_1, mnbrack6_1 = read_hirsh(folder_1, '/hirshfeld/cdft-2-0.out', atoms, None, None)
file_spec7_1, species7_1, num_data7_1, step7_1, brent7_1, mnbrack7_1 = read_hirsh(folder_1, '/hirshfeld/cdft-1-360.out', atoms, None, None)
file_spec8_1, species8_1, num_data8_1, step8_1, brent8_1, mnbrack8_1 = read_hirsh(folder_1, '/hirshfeld/cdft-1-charge-15p7.out', atoms, None, None)
file_spec9_1, species9_1, num_data9_1, step9_1, brent9_1, mnbrack9_1 = read_hirsh(folder_1, '/hirshfeld/cdft-1-charge-15p647.out', atoms, None, None)


force, forces_x, forces_y, forces_z, num_atoms, num_timesteps = load_forces.load_values_forces(folder_1, 'force/cdft-1-372_cdft-3e-3-print.out')

# Plot total energy DFT
time_plot = 100
skip = 0
energy_end = time_plot * 2
fig_energy_dft, ax_energy_dft = plt.subplots()
ax_energy_dft.plot(time_val1_1[skip:] - time_val1_1[skip], (energy_total1_1[skip:] - energy_total1_1[skip]) / atoms, 'k-', label='DFT')
# ax_energy_dft.plot(time_val10_1[skip:]-time_val10_1[skip], (energy_total10_1[skip:]-energy_total10_1[skip])/atoms, 'b-', label='CDFT 7.44 1e-2')
# ax_energy_dft.plot(time_val12_1[skip:]-time_val12_1[skip], (energy_total12_1[skip:]-energy_total12_1[skip])/atoms, 'r-', label='CDFT 7.44 3e-3')
# ax_energy_dft.plot(time_val2_1[skip:]-time_val2_1[skip], (energy_total2_1[skip:]-energy_total2_1[skip])/atoms, 'g-', label='CDFT 7.44 1e-3')
# ax_energy_dft.plot(time_val3_1[skip:]-time_val3_1[skip], (energy_total3_1[skip:]-energy_total3_1[skip])/atoms, 'g-', label='CDFT 7.44 1e-4')
ax_energy_dft.plot(time_val11_1[skip:]-time_val11_1[skip], (energy_total11_1[skip:]-energy_total11_1[skip])/atoms, 'r-', label='CDFT 3.72 3e-3')
ax_energy_dft.plot(time_val4_1[skip:]-time_val4_1[skip], (energy_total4_1[skip:]-energy_total4_1[skip])/atoms, 'g-', label='CDFT 3.72 1e-3')
# ax_energy_dft.plot(time_val5_1[skip:]-time_val5_1[skip], (energy_total5_1[skip:]-energy_total5_1[skip])/atoms, 'b-', label='CDFT 3.72 1e-4')
# ax_energy_dft.plot(time_val6_1[skip:]-time_val6_1[skip], (energy_total6_1[skip:]-energy_total6_1[skip])/atoms, 'g-', label='CDFT diff 0')
# ax_energy_dft.plot(time_val7_1[skip:]-time_val7_1[skip], (energy_total7_1[skip:]-energy_total7_1[skip])/atoms, 'g-', label='CDFT 1 3.60')
# ax_energy_dft.plot(time_val8_1[skip:]-time_val8_1[skip], (energy_total8_1[skip:]-energy_total8_1[skip])/atoms, 'r-', label='CDFT charge 15.7')
# ax_energy_dft.plot(time_val9_1[skip:]-time_val9_1[skip], (energy_total9_1[skip:]-energy_total9_1[skip])/atoms, 'g-', label='CDFT charge 15.647')
ax_energy_dft.set_xlabel('Time / fs')
ax_energy_dft.set_ylabel('Energy change per atom / Ha')
ax_energy_dft.set_xlim([0, time_plot])
# ax_energy_dft.set_ylim([-1e-5, 1e-4])
# ax_energy_dft.set_ylim([-0.75e-5, 1.25e-5])
ax_energy_dft.set_ylim([-6e-6, 6e-6])
ax_energy_dft.legend(frameon=False)
fig_energy_dft.tight_layout()
fig_energy_dft.savefig('{}/energy_charge_3p72_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot strength against time
# fig_strength, ax_strength = plt.subplots()
# i = -1
# cdft_strength2_1 = np.zeros(np.shape(cdft_iter2_1)[0])
# for j in range(np.shape(cdft_iter2_1)[0]):
#     i = int(i + cdft_iter2_1[j])
#     cdft_strength2_1[j] = strength2_1[i]
# i = -1
# cdft_strength3_1 = np.zeros(np.shape(cdft_iter3_1)[0])
# for j in range(np.shape(cdft_iter3_1)[0]):
#     i = int(i + cdft_iter3_1[j])
#     cdft_strength3_1[j] = strength3_1[i]
# i = -1
# cdft_strength4_1 = np.zeros(np.shape(cdft_iter4_1)[0])
# for j in range(np.shape(cdft_iter4_1)[0]):
#     i = int(i + cdft_iter4_1[j])
#     cdft_strength4_1[j] = strength4_1[i]
# i = -1
# cdft_strength5_1 = np.zeros(np.shape(cdft_iter5_1)[0])
# for j in range(np.shape(cdft_iter5_1)[0]):
#     i = int(i + cdft_iter5_1[j])
#     cdft_strength5_1[j] = strength5_1[i]
# i = -1
# cdft_strength6_1 = np.zeros(np.shape(cdft_iter6_1)[0])
# for j in range(np.shape(cdft_iter6_1)[0]):
#     i = int(i + cdft_iter6_1[j])
#     cdft_strength6_1[j] = strength6_1[i]
# i = -1
# cdft_strength7_1 = np.zeros(np.shape(cdft_iter7_1)[0])
# for j in range(np.shape(cdft_iter7_1)[0]):
#     i = int(i + cdft_iter7_1[j])
#     cdft_strength7_1[j] = strength7_1[i]
# i = -1
# cdft_strength8_1 = np.zeros(np.shape(cdft_iter8_1)[0])
# for j in range(np.shape(cdft_iter8_1)[0]):
#     i = int(i + cdft_iter8_1[j])
#     cdft_strength8_1[j] = strength8_1[i]
# i = -1
# cdft_strength9_1 = np.zeros(np.shape(cdft_iter9_1)[0])
# for j in range(np.shape(cdft_iter9_1)[0]):
#     i = int(i + cdft_iter9_1[j])
#     cdft_strength9_1[j] = strength9_1[i]
# ax_strength.plot(time_val2_1[skip:]-time_val2_1[skip], cdft_strength2_1[skip:], 'r-', label='CDFT 7.44 1e-3')
# ax_strength.plot(time_val3_1[skip:-22]-time_val3_1[skip], cdft_strength3_1[skip:], 'g-', label='CDFT 7.44 1e-4')
# ax_strength.plot(time_val4_1[skip:]-time_val4_1[skip], cdft_strength4_1[skip:], 'r-', label='CDFT 3.72 1e-3')
# ax_strength.plot(time_val5_1[skip:-5]-time_val5_1[skip], cdft_strength5_1[skip:], 'g-', label='CDFT 3.762 1e-4')
# ax_strength.plot(time_val6_1[skip:]-time_val6_1[skip], cdft_strength6_1[skip:], 'r-', label='CDFT diff 0')
# ax_strength.plot(time_val7_1[skip:]-time_val7_1[skip], cdft_strength7_1[skip:], 'r-', label='CDFT 1 3.60')
# ax_strength.plot(time_val8_1[skip:]-time_val8_1[skip], cdft_strength8_1[skip:], 'r-', label='CDFT charge 15.7')
# ax_strength.plot(time_val9_1[skip:]-time_val9_1[skip], cdft_strength9_1[skip:], 'g-', label='CDFT charge 15.647')
# ax_strength.set_ylabel('CDFT Lagrange multiplier')
# ax_strength.set_xlabel('Time / fs')
# ax_strength.set_xlim([0, time_plot])
# ax_strength.legend(frameon=False)
# fig_strength.tight_layout()
# fig_strength.savefig('{}/strength_3p72_{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot Hirshfeld analysis (Fe)
index_fe = np.array([6, 15]) - 1
skip = 0
skip_line = 2
plot_index = index_fe
plot_quantity = 'Spin'
fig_hirshfeld, ax_hirshfeld = plt.subplots()
temp1 = np.zeros(num_data1_1)
temp2 = np.zeros(num_data1_1)
i = -1
for n in range(num_data1_1):
    i = i + 1
    temp1[n] = (file_spec1_1.loc[atoms * i + skip_line * i + plot_index[0], plot_quantity])
    temp2[n] = (file_spec1_1.loc[atoms * i + skip_line * i + plot_index[1], plot_quantity])
ax_hirshfeld.plot(time_val1_1[skip:]-time_val1_1[skip], temp1[skip:], 'k-', label='DFT')
ax_hirshfeld.plot(time_val1_1[skip:]-time_val1_1[skip], temp2[skip:], 'k-', alpha=0.2)
temp1 = np.zeros(num_data11_1)
temp2 = np.zeros(num_data11_1)
i = -1
for n in range(num_data11_1):
    i = i + 1
    temp1[n] = (file_spec11_1.loc[atoms * i + skip_line * i + plot_index[0], plot_quantity])
    temp2[n] = (file_spec11_1.loc[atoms * i + skip_line * i + plot_index[1], plot_quantity])
ax_hirshfeld.plot(time_val11_1[skip:] - time_val11_1[skip], temp1[skip:-1], 'r-', label='CDFT 3.72 3e-3')
ax_hirshfeld.plot(time_val11_1[skip:] - time_val11_1[skip], temp2[skip:-1], 'r-', alpha=0.2)
temp1 = np.zeros(num_data4_1)
temp2 = np.zeros(num_data4_1)
i = -1
for n in range(num_data4_1):
    i = i + 1
    temp1[n] = (file_spec4_1.loc[atoms * i + skip_line * i + plot_index[0], plot_quantity])
    temp2[n] = (file_spec4_1.loc[atoms * i + skip_line * i + plot_index[1], plot_quantity])
ax_hirshfeld.plot(time_val4_1[skip:] - time_val4_1[skip], temp1[skip:], 'g-', label='CDFT 3.72 1e-3')
ax_hirshfeld.plot(time_val4_1[skip:] - time_val4_1[skip], temp2[skip:], 'g-', alpha=0.2)
# temp1 = np.zeros(num_data12_1)
# temp2 = np.zeros(num_data12_1)
# i = -1
# for n in range(num_data12_1):
#     i = i + 1
#     temp1[n] = (file_spec12_1.loc[atoms * i + skip_line * i + plot_index[0], 'Spin'])
#     temp2[n] = (file_spec12_1.loc[atoms * i + skip_line * i + plot_index[1], 'Spin'])
# ax_hirshfeld.plot(time_val12_1[skip:] - time_val12_1[skip], temp1[skip:-1], 'r-', label='CDFT 7.44 3e-3')
# ax_hirshfeld.plot(time_val12_1[skip:] - time_val12_1[skip], temp2[skip:-1], 'r-', alpha=1)
# temp1 = np.zeros(num_data2_1)
# temp2 = np.zeros(num_data2_1)
# i = -1
# for n in range(num_data2_1):
#     i = i + 1
#     temp1[n] = (file_spec2_1.loc[atoms * i + skip_line * i + plot_index[0], 'Spin'])
#     temp2[n] = (file_spec2_1.loc[atoms * i + skip_line * i + plot_index[1], 'Spin'])
# ax_hirshfeld.plot(time_val2_1[skip:] - time_val2_1[skip], temp1[skip:], 'g-', label='CDFT 7.44 1e-3')
# ax_hirshfeld.plot(time_val2_1[skip:] - time_val2_1[skip], temp2[skip:], 'g-', alpha=1)
# temp1 = np.zeros(num_data8_1)
# temp2 = np.zeros(num_data8_1)
# i = -1
# for n in range(num_data8_1):
#     i = i + 1
#     temp1[n] = (file_spec8_1.loc[atoms * i + skip_line * i + plot_index[0], 'Spin'])
#     temp2[n] = (file_spec8_1.loc[atoms * i + skip_line * i + plot_index[1], 'Spin'])
# ax_hirshfeld.plot(time_val8_1[skip:] - time_val8_1[skip], temp1[skip:], 'r-', label='CDFT charge 15.7')
# ax_hirshfeld.plot(time_val8_1[skip:] - time_val8_1[skip], temp2[skip:], 'r-', alpha=1)
# temp1 = np.zeros(num_data9_1)
# temp2 = np.zeros(num_data9_1)
# i = -1
# for n in range(num_data9_1):
#     i = i + 1
#     temp1[n] = (file_spec9_1.loc[atoms * i + skip_line * i + plot_index[0], 'Spin'])
#     temp2[n] = (file_spec9_1.loc[atoms * i + skip_line * i + plot_index[1], 'Spin'])
# ax_hirshfeld.plot(time_val9_1[skip:] - time_val9_1[skip], temp1[skip:], 'g-', label='CDFT charge 15.647')
# ax_hirshfeld.plot(time_val9_1[skip:] - time_val9_1[skip], temp2[skip:], 'g-', alpha=1)
ax_hirshfeld.set_xlabel('Time / fs')
ax_hirshfeld.set_ylabel('Hirshfeld spin moment')
ax_hirshfeld.set_xlim([0, time_plot])
ax_hirshfeld.set_ylim([-3.76, -3.68])
# ax_hirshfeld.set_ylim([-3.80, -3.64])
ax_hirshfeld.legend(frameon=False)
fig_hirshfeld.tight_layout()
fig_hirshfeld.savefig('{}/pop2_3p72_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot forces
skip = 0
skip_line = 2
plot_index = index_fe
fig_forces1, ax_forces1 = plt.subplots()
temp1 = np.zeros(num_timesteps)
temp2 = np.zeros(num_timesteps)
temp3 = np.zeros(num_timesteps)
i = -1
for n in range(num_timesteps):
    i = i + 1
    temp1[n] = np.sum(forces_x[i, :])
    temp2[n] = np.sum(forces_y[i, :])
    temp3[n] = np.sum(forces_z[i, :])
ax_forces1.plot(temp1[skip:], 'r-', label='x')
ax_forces1.plot(temp2[skip:], 'g-', label='y')
ax_forces1.plot(temp3[skip:], 'b-', label='z')
ax_forces1.plot(np.sum([np.abs(temp1[0]), np.abs(temp2[0]), np.abs(temp3[0])]), 'k-', label='sum')
# print(temp1[0], temp2[0], temp3[0], np.sum([temp1[0], temp2[0], temp3[0]]), np.sum([np.abs(temp1[0]), np.abs(temp2[0]), np.abs(temp3[0])]))
ax_forces1.set_xlabel('Time / fs')
ax_forces1.set_ylabel('Sum of forces on all atoms / au')
ax_forces1.set_xlim([0, time_plot])
# ax_forces1.set_ylim([-3.80, -3.64])
ax_forces1.legend(frameon=False)
fig_forces1.tight_layout()
fig_forces1.savefig('{}/forces_sum_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot forces
skip = 0
skip_line = 2
plot_index = index_fe
fig_forces2, ax_forces2 = plt.subplots()
temp1 = np.zeros(num_timesteps)
temp2 = np.zeros(num_timesteps)
temp3 = np.zeros(num_timesteps)
i = -1
for n in range(num_timesteps):
    i = i + 1
    temp1[n] = np.sum(forces_x[i, plot_index[0]])
    temp2[n] = np.sum(forces_y[i, plot_index[0]])
    temp3[n] = np.sum(forces_z[i, plot_index[0]])
ax_forces2.plot(temp1[skip:], 'r-', label='x')
ax_forces2.plot(temp2[skip:], 'g-', label='y')
ax_forces2.plot(temp3[skip:], 'b-', label='z')
i = -1
for n in range(num_timesteps):
    i = i + 1
    temp1[n] = np.sum(forces_x[i, plot_index[1]])
    temp2[n] = np.sum(forces_y[i, plot_index[1]])
    temp3[n] = np.sum(forces_z[i, plot_index[1]])
ax_forces2.plot(temp1[skip:], 'r--')
ax_forces2.plot(temp2[skip:], 'g--')
ax_forces2.plot(temp3[skip:], 'b--')
ax_forces2.set_xlabel('Time / fs')
ax_forces2.set_ylabel('Force / au')
ax_forces2.set_xlim([0, time_plot])
# ax_forces2.set_ylim([-3.80, -3.64])
ax_forces2.legend(frameon=False)
fig_forces2.tight_layout()
# fig_forces2.savefig('{}/Force / au {}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
