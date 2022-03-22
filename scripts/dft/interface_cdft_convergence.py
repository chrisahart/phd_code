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

atoms = 435

folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/electron/hawk/hse/nve/dft/analysis'
energy_kinetic1_1, energy_potential1_1, energy_total1_1, temperature1_1, time_val1_1, time_per_step1_1 = load_energy.load_values_energy(folder_1, '/energy/rs-base-subsys-ref.out')
iasd1_1 = np.loadtxt('{}/iasd/rs-base-subsys-ref.out'.format(folder_1))

folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/electron/hawk/hse/nve/cdft-24hr/analysis'
energy_kinetic1_2, energy_potential1_2, energy_total1_2, temperature1_2, time_val1_2, time_per_step1_2 = load_energy.load_values_energy(folder_2, 'energy/552p2591.out')
energy_kinetic2_2, energy_potential2_2, energy_total2_2, temperature2_2, time_val2_2, time_per_step2_2 = load_energy.load_values_energy(folder_2, 'energy/552p2.out')
energy_kinetic3_2, energy_potential3_2, energy_total3_2, temperature3_2, time_val3_2, time_per_step3_2 = load_energy.load_values_energy(folder_2, 'energy/552p3.out')
energy_kinetic4_2, energy_potential4_2, energy_total4_2, temperature4_2, time_val4_2, time_per_step4_2 = load_energy.load_values_energy(folder_2, 'energy/552p4.out')
energy_kinetic5_2, energy_potential5_2, energy_total5_2, temperature5_2, time_val5_2, time_per_step5_2 = load_energy.load_values_energy(folder_2, 'energy/552p1.out')
energy_kinetic6_2, energy_potential6_2, energy_total6_2, temperature6_2, time_val6_2, time_per_step6_2 = load_energy.load_values_energy(folder_2, 'energy/charge-none_spin-plus-1_eps-1e-3.out')
# iasd1_2 = np.loadtxt('{}/iasd/552p2591.out'.format(folder_2))
# iasd2_2 = np.loadtxt('{}/iasd/552p2.out'.format(folder_2))
# iasd3_2 = np.loadtxt('{}/iasd/552p3.out'.format(folder_2))
# iasd4_2 = np.loadtxt('{}/iasd/552p4.out'.format(folder_2))
# iasd5_2 = np.loadtxt('{}/iasd/552p1.out'.format(folder_2))
# iasd6_2 = np.loadtxt('{}/iasd/charge-none_spin-plus-1_eps-1e-3.out'.format(folder_2))
cdft_iter1_2 = np.loadtxt('{}/cdft-iter/552p2591.out'.format(folder_2))
cdft_iter2_2 = np.loadtxt('{}/cdft-iter/552p2.out'.format(folder_2))
cdft_iter3_2 = np.loadtxt('{}/cdft-iter/552p3.out'.format(folder_2))
cdft_iter4_2 = np.loadtxt('{}/cdft-iter/552p4.out'.format(folder_2))
cdft_iter5_2 = np.loadtxt('{}/cdft-iter/552p1.out'.format(folder_2))
cdft_iter6_2 = np.loadtxt('{}/cdft-iter/charge-none_spin-plus-1_eps-1e-3.out'.format(folder_2))
strength1_2 = np.loadtxt('{}/strength/552p2591.out'.format(folder_2))
strength2_2 = np.loadtxt('{}/strength/552p2.out'.format(folder_2))
strength3_2 = np.loadtxt('{}/strength/552p3.out'.format(folder_2))
strength4_2 = np.loadtxt('{}/strength/552p4.out'.format(folder_2))
strength5_2 = np.loadtxt('{}/strength/552p1.out'.format(folder_2))
strength6_2 = np.loadtxt('{}/strength/charge-none_spin-plus-1_eps-1e-3.out'.format(folder_2))

folder_3 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/electron/hawk/hse/nve/cdft/analysis'
energy_kinetic1_3, energy_potential1_3, energy_total1_3, temperature1_3, time_val1_3, time_per_step1_3 = load_energy.load_values_energy(folder_3, 'energy/dt-5_grid-600-40.out')
iasd1_3 = np.loadtxt('{}/iasd/dt-5_grid-600-40.out'.format(folder_3))
cdft_iter1_3 = np.loadtxt('{}/cdft-iter/dt-5_grid-600-40.out'.format(folder_3))
strength1_3 = np.loadtxt('{}/strength/dt-5_grid-600-40.out'.format(folder_3))

folder_4 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/electron/frozen-none/analysis'
energy_kinetic1_4, energy_potential1_4, energy_total1_4, temperature1_4, time_val1_4, time_per_step1_4 = load_energy.load_values_energy(folder_4, 'energy/run-000.out')
iasd1_4 = np.loadtxt('{}/iasd/run-000.out'.format(folder_4))

# Plot total energy
time_plot = 30
energy_end = time_plot*2
fig_energy, ax_energy = plt.subplots()
# ax_energy.plot(time_val1_1-time_val1_1[0], (energy_total1_1-energy_total1_1[0])/atoms, 'k.-', label='DFT')
ax_energy.plot(time_val1_4-time_val1_4[0], (energy_total1_4-energy_total1_4[0])/atoms, 'k.-', label='DFT NPT_F', alpha=0.5)
ax_energy.plot(time_val5_2-time_val5_2[0], (energy_total5_2-energy_total5_2[0])/atoms, 'm.-', label='552.1')
ax_energy.plot(time_val2_2-time_val2_2[0], (energy_total2_2-energy_total2_2[0])/atoms, 'g.-', label='552.2')
ax_energy.plot(time_val3_2-time_val3_2[0], (energy_total3_2-energy_total3_2[0])/atoms, 'b.-', label='552.3')
ax_energy.plot(time_val4_2-time_val4_2[0], (energy_total4_2-energy_total4_2[0])/atoms, 'y.-', label='552.4')
ax_energy.plot(time_val6_2-time_val6_2[0], (energy_total6_2-energy_total6_2[0])/atoms, '.-', color='orange', label='Spin +1')
ax_energy.plot(time_val1_3-time_val1_3[0], (energy_total1_3-energy_total1_3[0])/atoms, '.-', color='grey', label='552.8')
ax_energy.set_xlabel('Time / fs')
ax_energy.set_ylabel('Energy change per atom / Ha')
ax_energy.set_xlim([0, time_plot])
# ax_energy.set_ylim([-2e-6, 2e-6])
# ax_energy.set_ylim([-4e-6, 6e-5])
ax_energy.set_ylim([-4e-6, 5e-5])
ax_energy.legend(frameon=False)
fig_energy.tight_layout()
fig_energy.savefig('{}/energy_{}.png'.format(folder_2, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot iasd
# fig_iasd, ax_iasd = plt.subplots()
# ax_iasd.plot(time_val1_1-time_val1_1[0], iasd1_1, 'k.-', label='DFT')
# ax_iasd.plot(time_val1_4-time_val1_4[0], iasd1_4, 'k.-', label='DFT NPT_F', alpha=0.5)
# ax_iasd.plot(time_val5_2-time_val5_2[0], iasd5_2[:-1], 'm.-', label='552.1')
# ax_iasd.plot(time_val2_2-time_val2_2[0], iasd2_2, 'g.-', label='552.2')
# ax_iasd.plot(time_val3_2-time_val3_2[0], iasd3_2, 'b.-', label='552.3')
# ax_iasd.plot(time_val4_2-time_val4_2[0], iasd4_2, 'y.-', label='552.4')
# ax_iasd.plot(time_val6_2-time_val6_2[0], iasd6_2, '.-',color='orange', label='Spin +1')
# ax_iasd.plot(time_val1_3-time_val1_3[0], iasd1_3, '.-', color='grey', label='552.8')
# ax_iasd.set_xlabel('Time / fs')
# ax_iasd.set_ylabel('Integrated Absolute Spin Density')
# ax_iasd.legend(frameon=False)
# ax_iasd.set_xlim([0, time_plot])
# fig_iasd.tight_layout()
# fig_iasd.savefig('{}/time_iasd_{}.png'.format(folder_2, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot total change iasd against change energy
# fig_cdft1, ax_cdft1 = plt.subplots()
# ax_cdft1.plot([(energy_total5_2[energy_end]-energy_total5_2[0])/atoms], [iasd5_2[0] - iasd1_1[0]], 'm.', label='552.1')
# ax_cdft1.plot([(energy_total2_2[energy_end]-energy_total2_2[0])/atoms], [iasd2_2[0] - iasd1_1[0]], 'g.', label='552.2')
# ax_cdft1.plot([(energy_total3_2[energy_end]-energy_total3_2[0])/atoms], [iasd3_2[0] - iasd1_1[0]], 'b.', label='552.3')
# ax_cdft1.plot([(energy_total4_2[energy_end]-energy_total4_2[0])/atoms], [iasd4_2[0] - iasd1_1[0]], 'y.', label='552.4')
# ax_cdft1.plot([(energy_total1_3[energy_end]-energy_total1_3[0])/atoms], [iasd1_3[0] - iasd1_1[0]], '.-', color='grey', label='552.8')
# ax_cdft1.set_xlabel('Change in energy / Ha')
# ax_cdft1.set_ylabel('Difference in IASD')
# ax_cdft1.legend(frameon=False)
# fig_cdft1.tight_layout()
# fig_cdft1.savefig('{}/energy_iasd_{}.png'.format(folder_2, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot strength against time
fig_cdft2, ax_cdft2 = plt.subplots()
i = -1
cdft_strength5_2 = np.zeros(np.shape(cdft_iter5_2)[0])
for j in range(np.shape(cdft_iter5_2)[0]):
    i = int(i + cdft_iter5_2[j])
    cdft_strength5_2[j] = strength5_2[i]
i = -1
cdft_strength2_2 = np.zeros(np.shape(cdft_iter2_2)[0])
for j in range(np.shape(cdft_iter2_2)[0]):
    i = int(i + cdft_iter2_2[j])
    cdft_strength2_2[j] = strength2_2[i]
i = -1
cdft_strength3_2 = np.zeros(np.shape(cdft_iter3_2)[0])
for j in range(np.shape(cdft_iter3_2)[0]):
    i = int(i + cdft_iter3_2[j])
    cdft_strength3_2[j] = strength3_2[i]
i = -1
cdft_strength4_2 = np.zeros(np.shape(cdft_iter4_2)[0])
for j in range(np.shape(cdft_iter4_2)[0]):
    i = int(i + cdft_iter4_2[j])
    cdft_strength4_2[j] = strength4_2[i]
i = -1
cdft_strength6_2 = np.zeros(np.shape(cdft_iter6_2)[0])
for j in range(np.shape(cdft_iter6_2)[0]):
    i = int(i + cdft_iter6_2[j])
    cdft_strength6_2[j] = strength6_2[i]
i = -1
cdft_strength1_3 = np.zeros(np.shape(cdft_iter1_3)[0])
for j in range(np.shape(cdft_iter1_3)[0]):
    i = int(i + cdft_iter1_3[j])
    cdft_strength1_3[j] = strength1_3[i]
ax_cdft2.plot(time_val5_2-time_val5_2[0], cdft_strength5_2[:-1], 'm.-', label='552.1')
ax_cdft2.plot(time_val2_2[:-1]-time_val2_2[0], cdft_strength2_2, 'g.-', label='552.2')
ax_cdft2.plot(time_val3_2-time_val3_2[0], cdft_strength3_2, 'b.-', label='552.3')
ax_cdft2.plot(time_val4_2-time_val4_2[0], cdft_strength4_2, 'y.-', label='552.4')
ax_cdft2.plot(time_val6_2-time_val6_2[0], cdft_strength6_2, '.-', color='orange', label='Spin +1')
ax_cdft2.plot(time_val1_3-time_val1_3[0], cdft_strength1_3, '.-', color='grey', label='552.8')
ax_cdft2.set_ylabel('CDFT Lagrange multiplier')
ax_cdft2.set_xlabel('Time / fs')
ax_cdft2.set_xlim([0, time_plot])
ax_cdft2.legend(frameon=False)
fig_cdft2.tight_layout()
# fig_cdft2.savefig('{}/time_strength_{}.png'.format(folder_2, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
