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
    Plot energy and forces for water dimer
"""

atoms = 6
folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/water_dimer/md/hawk/md_12-6-6/300k/analysis/energy'
energy_kinetic6_1, energy_potential6_1, energy_total6_1, temperature6_1, time_val6_1, time_per_step6_1 = load_energy.load_values_energy(folder_1, 'eps_1e-2.out')
energy_kinetic7_1, energy_potential7_1, energy_total7_1, temperature7_1, time_val7_1, time_per_step7_1 = load_energy.load_values_energy(folder_1, 'eps_1e-3.out')
energy_kinetic8_1, energy_potential8_1, energy_total8_1, temperature8_1, time_val8_1, time_per_step8_1 = load_energy.load_values_energy(folder_1, 'eps_1e-4.out')
energy_kinetic9_1, energy_potential9_1, energy_total9_1, temperature9_1, time_val9_1, time_per_step9_1 = load_energy.load_values_energy(folder_1, 'eps_1e-5.out')
energy_kinetic10_1, energy_potential10_1, energy_total10_1, temperature10_1, time_val10_1, time_per_step10_1 = load_energy.load_values_energy(folder_1, 'eps_1e-6.out')

# Plot total energy
time_plot = 140
fig_energy, ax_energy = plt.subplots()
# ax_energy.plot(time_val6_1, (energy_total6_1-energy_total6_1[0])/atoms, 'k', label='Hirshfeld 1e-2')
# ax_energy.plot(time_val7_1, (energy_total7_1-energy_total7_1[0])/atoms, 'r', label='Hirshfeld 1e-3')
ax_energy.plot(time_val8_1, (energy_total8_1-energy_total8_1[0])/atoms, 'b', label='Hirshfeld 1e-4')
# ax_energy.plot(time_val9_1, (energy_total9_1-energy_total9_1[0])/atoms, 'y', label='Hirshfeld 1e-5')
ax_energy.plot(time_val10_1, (energy_total10_1-energy_total10_1[0])/atoms, 'm', label='Hirshfeld 1e-6')
# ax_energy.plot(time_val11_1, (energy_total11_1-energy_total11_1[0])/atoms, 'k', label='Hirshfeld 1e-7')
ax_energy.set_xlabel('Time / fs')
ax_energy.set_ylabel('Energy drift per atom / Ha')
ax_energy.set_xlim([0, time_plot])
ax_energy.set_ylim([-2e-6, 8e-6])
# ax_energy.set_ylim([-3e-5, 6e-5])
ax_energy.legend(frameon=False)
fig_energy.tight_layout()
fig_energy.savefig('{}/energy_{}.png'.format(folder_1, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot time taken
time_plot = 1000
fig_time, ax_time = plt.subplots()
data_x = [1e-4, 1e-5, 1e-6]
data_y_1 = np.array([np.mean(time_per_step8_1),
                   np.mean(time_per_step9_1),
                   np.mean(time_per_step10_1)])
ax_time.plot(data_x, data_y_1, 'rx-', label='No PBC')
ax_time.set_xscale('log')
ax_time.set_xlabel('Constraint convergence / e')
ax_time.set_ylabel('Average time per MD step / s')
# ax_time.legend(frameon=False)
fig_time.tight_layout()
fig_time.savefig('{}/time_{}.png'.format(folder_1, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot energy drift log
fig_drift, ax_drift = plt.subplots()
data_x = [1e-4, 1e-5, 1e-6]
data_y_1 = np.array([np.max(np.abs(energy_total8_1[:time_plot]-energy_total8_1[0])/atoms),
                   np.max(np.abs(energy_total9_1[:time_plot]-energy_total9_1[0])/atoms),
                   np.max(np.abs(energy_total10_1[:time_plot]-energy_total10_1[0])/atoms)])
ax_drift.plot([data_x[-1], data_x[0]], [1e-6, 1e-6], 'k--')
ax_drift.plot(data_x, data_y_1, 'rx-', label='No PBC')
ax_drift.set_yscale('log')
ax_drift.set_xscale('log')
ax_drift.set_xlabel('Constraint convergence / e')
ax_drift.set_ylabel('Energy drift per atom / Ha')
ax_drift.set_xlim([1e-6, 1e-3])
ax_drift.set_ylim([1e-7, 1e-4])
# ax_drift.legend(frameon=False)
fig_drift.tight_layout()
fig_drift.savefig('{}/energy_drift_{}.png'.format(folder_1, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Printing
print('Max energy drift', np.max(np.abs(energy_total8_1[:time_plot]-energy_total8_1[0])/atoms))
print('Mean absolute energy drift', np.mean(np.abs(energy_total8_1[:time_plot]-energy_total8_1[0])/atoms))
print('Mean energy drift', np.mean((energy_total8_1[:time_plot]-energy_total8_1[0])/atoms))
print('Final energy drift', (energy_total8_1[-1]-energy_total8_1[0])/atoms)

if __name__ == "__main__":
    print('Finished.')
    plt.show()
