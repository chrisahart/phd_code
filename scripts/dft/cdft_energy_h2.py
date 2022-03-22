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
    Plot energy and forces for h2 dimer
"""

atoms = 2
folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/h2/analysis/final-4h/energy'
energy_kinetic6_1, energy_potential6_1, energy_total6_1, temperature6_1, time_val6_1, time_per_step6_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-2_cell-10A-opt3-md-node2.out')
energy_kinetic7_1, energy_potential7_1, energy_total7_1, temperature7_1, time_val7_1, time_per_step7_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-3_cell-10A-opt3-md-node2.out')
energy_kinetic8_1, energy_potential8_1, energy_total8_1, temperature8_1, time_val8_1, time_per_step8_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-4_cell-10A-opt3-md-node2.out')
energy_kinetic9_1, energy_potential9_1, energy_total9_1, temperature9_1, time_val9_1, time_per_step9_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-5_cell-10A-opt3-md-node2.out')
energy_kinetic10_1, energy_potential10_1, energy_total10_1, temperature10_1, time_val10_1, time_per_step10_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-6_cell-10A-opt3-md-node2.out')
energy_kinetic11_1, energy_potential11_1, energy_total11_1, temperature11_1, time_val11_1, time_per_step11_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-7_cell-10A-opt3-md-node2.out')
energy_kinetic12_1, energy_potential12_1, energy_total12_1, temperature12_1, time_val12_1, time_per_step12_1 = load_energy.load_values_energy(folder_1, 'dft_from_opt.out')
energy_kinetic13_1, energy_potential13_1, energy_total13_1, temperature13_1, time_val13_1, time_per_step13_1 = load_energy.load_values_energy(folder_1, 'dft_from_opt_neutral.out')

folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/h2/analysis/final-4hr-pbc/energy'
energy_kinetic6_2, energy_potential6_2, energy_total6_2, temperature6_2, time_val6_2, time_per_step6_2 = load_energy.load_values_energy(folder_2, 'nve-hirshfeld-charge7_eps-1e-2_cell-10A-opt3-md-node2.out')
energy_kinetic7_2, energy_potential7_2, energy_total7_2, temperature7_2, time_val7_2, time_per_step7_2 = load_energy.load_values_energy(folder_2, 'nve-hirshfeld-charge7_eps-1e-3_cell-10A-opt3-md-node2.out')
energy_kinetic8_2, energy_potential8_2, energy_total8_2, temperature8_2, time_val8_2, time_per_step8_2 = load_energy.load_values_energy(folder_2, 'nve-hirshfeld-charge7_eps-1e-4_cell-10A-opt3-md-node2.out')
energy_kinetic9_2, energy_potential9_2, energy_total9_2, temperature9_2, time_val9_2, time_per_step9_2 = load_energy.load_values_energy(folder_2, 'nve-hirshfeld-charge7_eps-1e-5_cell-10A-opt3-md-node2.out')
energy_kinetic10_2, energy_potential10_2, energy_total10_2, temperature10_2, time_val10_2, time_per_step10_2 = load_energy.load_values_energy(folder_2, 'nve-hirshfeld-charge7_eps-1e-6_cell-10A-opt3-md-node2.out')
energy_kinetic11_2, energy_potential11_2, energy_total11_2, temperature11_2, time_val11_2, time_per_step11_2 = load_energy.load_values_energy(folder_2, 'nve-hirshfeld-charge7_eps-1e-7_cell-10A-opt3-md-node2.out')
energy_kinetic12_2, energy_potential12_2, energy_total12_2, temperature12_2, time_val12_2, time_per_step12_2 = load_energy.load_values_energy(folder_2, 'dft_from_opt.out')
energy_kinetic13_2, energy_potential13_2, energy_total13_2, temperature13_2, time_val13_2, time_per_step13_2 = load_energy.load_values_energy(folder_2, 'dft_from_opt_neutral.out')

# Plot total energy
time_plot = 1000
fig_energy, ax_energy = plt.subplots()
ax_energy.plot(time_val12_2, (energy_total12_2-energy_total12_2[0])/atoms, 'k', label='DFT')
ax_energy.plot(time_val7_2, (energy_total7_2-energy_total7_2[0])/atoms, 'r', label='CDFT 1e-3')
ax_energy.plot(time_val8_2, (energy_total8_2-energy_total8_2[0])/atoms, 'b', label='CDFT 1e-4')
ax_energy.plot(time_val9_2, (energy_total9_2-energy_total9_2[0])/atoms, 'y', label='CDFT 1e-5')
ax_energy.plot(time_val10_2, (energy_total10_2-energy_total10_2[0])/atoms, 'm', label='CDFT 1e-6')
# ax_energy.plot(time_val11_2, (energy_total11_2-energy_total11_2[0])/atoms, color='orange', label='CDFT 1e-7')
ax_energy.set_xlabel('Time / fs')
ax_energy.set_ylabel('Energy drift per atom / Ha')
ax_energy.set_xlim([0, time_plot])
ax_energy.set_ylim([-1e-6, 1e-6])
# ax_energy.set_ylim([-1e-4/3, 3e-4/3])
ax_energy.legend(frameon=False)
fig_energy.tight_layout()
fig_energy.savefig('{}/energy_pbc_{}.png'.format(folder_2, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot time taken
# time_plot = 1000
# fig_time, ax_time = plt.subplots()
# data_x = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
# data_y_1 = np.array([np.mean(time_per_step6_1),
#                    np.mean(time_per_step7_1),
#                    np.mean(time_per_step8_1),
#                    np.mean(time_per_step9_1),
#                    np.mean(time_per_step10_1),
#                    np.mean(time_per_step11_1)])
# data_y_2 = np.array([np.mean(time_per_step6_2),
#                      np.mean(time_per_step7_2),
#                      np.mean(time_per_step8_2),
#                      np.mean(time_per_step9_2),
#                      np.mean(time_per_step10_2),
#                      np.mean(time_per_step11_2)])
# ax_time.plot(data_x, data_y_1, 'rx-', label='No PBC')
# ax_time.plot(data_x, data_y_2, 'gx-', label='PBC')
# ax_time.set_xscale('log')
# ax_time.set_xlabel('Constraint convergence / e')
# ax_time.set_ylabel('Average time per MD step / s')
# ax_time.legend(frameon=False)
# fig_time.tight_layout()
# fig_time.savefig('{}/time_{}.png'.format(folder_2, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot energy drift log
# fig_drift, ax_drift = plt.subplots()
# data_x = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
# data_y_1 = np.array([np.max(np.abs(energy_total6_1[:]-energy_total6_1[0])/atoms),
#                    np.max(np.abs(energy_total7_1[:]-energy_total7_1[0])/atoms),
#                    np.max(np.abs(energy_total8_1[:]-energy_total8_1[0])/atoms),
#                    np.max(np.abs(energy_total9_1[:]-energy_total9_1[0])/atoms),
#                    np.max(np.abs(energy_total10_1[:]-energy_total10_1[0])/atoms),
#                    np.max(np.abs(energy_total11_1[:]-energy_total11_1[0])/atoms)])
# data_y_2 = np.array([np.max(np.abs(energy_total6_2[:] - energy_total6_2[0]) / atoms),
#                      np.max(np.abs(energy_total7_2[:] - energy_total7_2[0]) / atoms),
#                      np.max(np.abs(energy_total8_2[:] - energy_total8_2[0]) / atoms),
#                      np.max(np.abs(energy_total9_2[:] - energy_total9_2[0]) / atoms),
#                      np.max(np.abs(energy_total10_2[:] - energy_total10_2[0]) / atoms),
#                      np.max(np.abs(energy_total11_2[:] - energy_total11_2[0]) / atoms)])
# ax_drift.plot([data_x[-1], data_x[0]], [1e-6, 1e-6], 'k--')
# ax_drift.plot(data_x, data_y_1, 'rx-', label='No PBC')
# ax_drift.plot(data_x, data_y_2, 'gx-', label='PBC')
# ax_drift.set_yscale('log')
# ax_drift.set_xscale('log')
# ax_drift.set_xlabel('Constraint convergence / e')
# ax_drift.set_ylabel('Energy drift per atom / Ha')
# ax_drift.set_xlim([1e-6, 1e-3])
# ax_drift.set_ylim([1e-7, 1e-4])
# ax_drift.legend(frameon=False)
# fig_drift.tight_layout()
# fig_drift.savefig('{}/energy_drift_{}.png'.format(folder_2, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot energy drift log
params = {'font.size': 12,
          'axes.labelsize': 14,
          'lines.markersize': 15,
          'legend.fontsize': 10}
plt.rcParams.update(params)
fig_drift, ax_drift = plt.subplots()
data_x = [1e-3, 1e-4, 1e-5, 1e-6]
dft = (-0.658718021--0.658824319)/2
dft_pbc = (-0.657768859--0.657821197)/2

data_y_1 = np.array([np.max(np.abs(energy_total7_1[:]-energy_total7_1[0])/atoms),
                   np.max(np.abs(energy_total8_1[:]-energy_total8_1[0])/atoms),
                   np.max(np.abs(energy_total9_1[:]-energy_total9_1[0])/atoms),
                   np.max(np.abs(energy_total10_1[:]-energy_total10_1[0])/atoms)])
data_y_2 = np.array([np.max(np.abs(energy_total7_2[:] - energy_total7_2[0]) / atoms),
                     np.max(np.abs(energy_total8_2[:] - energy_total8_2[0]) / atoms),
                     np.max(np.abs(energy_total9_2[:] - energy_total9_2[0]) / atoms),
                     np.max(np.abs(energy_total10_2[:] - energy_total10_2[0]) / atoms)])
# ax_drift.plot([data_x[-1], data_x[0]], [1e-6, 1e-6], 'k--')
ax_drift.plot(data_x, data_y_1, '+-', color='black')
ax_drift.plot(data_x, data_y_2, '+-', color='grey')
# ax_drift.plot([0, 1], [dft, dft], '-', color='grey')
ax_drift.plot([0, 1], [4.0e-5, 4.0e-5], 'r--')
ax_drift.plot([0, 1], [4.3e-5, 4.3e-5], 'g--')
ax_drift.plot([0, 1], [5.5e-5, 5.5e-5], 'b--')
# ax_drift.plot(5e-4, 4.5e-5, 'r+', markersize=20)
ax_drift.plot(5e-4, 4.5e-5, 'rx')
ax_drift.plot(5e-4, 2.6e-5, 'gx')
ax_drift.plot(5e-4, 2.5e-5, 'bx')
ax_drift.set_yscale('log')
ax_drift.set_xscale('log')
ax_drift.set_xlabel('Constraint convergence [e]')
ax_drift.set_ylabel('Energy drift [H/atom/ps]')
ax_drift.set_xlim([8e-7, 1.2e-3])
# ax_drift.set_ylim([5e-7, 1e-4])
ax_drift.legend(frameon=False)
fig_drift.tight_layout()
fig_drift.savefig('{}/energy_drift_{}.png'.format(folder_2, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Printing
print('data_y_1', data_y_1)
print('data_y_2', data_y_2)

if __name__ == "__main__":
    print('Finished.')
    plt.show()
