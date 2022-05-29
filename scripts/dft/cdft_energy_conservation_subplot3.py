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


def metric_1(energy, atoms):
    """ Energy drift as maximum change from starting position """
    metric = np.max(np.abs(energy[:] - energy[0]) / atoms)
    return metric


def metric_2(energy, atoms, interval=50):
    """ Energy drift as peak to peak distance"""
    metric = np.abs(np.max(energy[0:interval])-np.max(energy[-interval:])) / atoms
    return metric


folder_3 = 'F:/Backup/Archer-1/work/cahart/other/cdft/MgO/cpmd_restart/cell-110-96-4nn/md/pbe0-final/analysis'
energy_kinetic6_3, energy_potential6_3, energy_total6_3, temperature6_3, time_val6_3, time_per_step6_3 = load_energy.load_values_energy(folder_3, 'energy/dft.out')
energy_kinetic7_3, energy_potential7_3, energy_total7_3, temperature7_3, time_val7_3, time_per_step7_3 = load_energy.load_values_energy(folder_3, 'energy/cdft_eps-1e-2.out')
energy_kinetic8_3, energy_potential8_3, energy_total8_3, temperature8_3, time_val8_3, time_per_step8_3 = load_energy.load_values_energy(folder_3, 'energy/cdft_eps-1e-3.out')
energy_kinetic9_3, energy_potential9_3, energy_total9_3, temperature9_3, time_val9_3, time_per_step9_3 = load_energy.load_values_energy(folder_3, 'energy/cdft_eps-1e-4.out')
strength7_3 = np.loadtxt('{}/strength/cdft_eps-1e-2.out'.format(folder_3))
strength8_3 = np.loadtxt('{}/strength/cdft_eps-1e-3.out'.format(folder_3))
strength9_3 = np.loadtxt('{}/strength/cdft_eps-1e-4.out'.format(folder_3))

folder_4 = 'F:/Backup/Archer-1/work/cahart/other/cdft/MgO/cpmd_restart/cell-110-96-4nn/md/pbe-final/analysis'
energy_kinetic6_4, energy_potential6_4, energy_total6_4, temperature6_4, time_val6_4, time_per_step6_4 = load_energy.load_values_energy(folder_4, 'energy/dft.out')
energy_kinetic7_4, energy_potential7_4, energy_total7_4, temperature7_4, time_val7_4, time_per_step7_4 = load_energy.load_values_energy(folder_4, 'energy/cdft_eps-1e-2.out')
energy_kinetic8_4, energy_potential8_4, energy_total8_4, temperature8_4, time_val8_4, time_per_step8_4 = load_energy.load_values_energy(folder_4, 'energy/cdft_eps-1e-3.out')
energy_kinetic9_4, energy_potential9_4, energy_total9_4, temperature9_4, time_val9_4, time_per_step9_4 = load_energy.load_values_energy(folder_4, 'energy/cdft_eps-1e-4.out')
strength7_4 = np.loadtxt('{}/strength/cdft_eps-1e-2.out'.format(folder_4))
strength8_4 = np.loadtxt('{}/strength/cdft_eps-1e-3.out'.format(folder_4))
strength9_4 = np.loadtxt('{}/strength/cdft_eps-1e-4.out'.format(folder_4))

folder_51 = 'F:/Backup/Archer-2/other/cdft/iron_oxides/hematite/electron-24hr/analysis'
energy_kinetic6_51, energy_potential6_51, energy_total6_51, temperature6_51, time_val6_51, time_per_step6_51 = load_energy.load_values_energy(folder_51, 'energy/dft.out')
energy_kinetic7_51, energy_potential7_51, energy_total7_51, temperature7_51, time_val7_51, time_per_step7_51 = load_energy.load_values_energy(folder_51, 'energy/cdft_1e-2.out')
energy_kinetic8_51, energy_potential8_51, energy_total8_51, temperature8_51, time_val8_51, time_per_step8_51 = load_energy.load_values_energy(folder_51, 'energy/cdft_1e-3.out')
strength7_51 = np.loadtxt('{}/strength/cdft_1e-2.out'.format(folder_51))
strength8_51 = np.loadtxt('{}/strength/cdft_1e-3.out'.format(folder_51))

folder_52 = 'F:/Backup/Archer-2/other/cdft/iron_oxides/hematite/hole-final/analysis'
energy_kinetic6_52, energy_potential6_52, energy_total6_52, temperature6_52, time_val6_52, time_per_step6_52 = load_energy.load_values_energy(folder_52, 'energy/dft.out')
energy_kinetic7_52, energy_potential7_52, energy_total7_52, temperature7_52, time_val7_52, time_per_step7_52 = load_energy.load_values_energy(folder_52, 'energy/cdft_1e-2.out')
energy_kinetic8_52, energy_potential8_52, energy_total8_52, temperature8_52, time_val8_52, time_per_step8_52 = load_energy.load_values_energy(folder_52, 'energy/cdft_1e-3.out')
strength7_52 = np.loadtxt('{}/strength/cdft_1e-2.out'.format(folder_52))
strength8_52 = np.loadtxt('{}/strength/cdft_1e-3.out'.format(folder_52))

# folder_6 = 'F:/Backup/Archer-2/other/cdft/iron_oxides/lepidocrocite/hole-24hr/analysis'
folder_6 = 'F:/Backup/Archer-2/other/cdft/iron_oxides/lepidocrocite/hole-final/analysis'
energy_kinetic6_6, energy_potential6_6, energy_total6_6, temperature6_6, time_val6_6, time_per_step6_6 = load_energy.load_values_energy(folder_6, 'energy/dft.out')
energy_kinetic7_6, energy_potential7_6, energy_total7_6, temperature7_6, time_val7_6, time_per_step7_6 = load_energy.load_values_energy(folder_6, 'energy/cdft_eps-1e-2.out')
energy_kinetic8_6, energy_potential8_6, energy_total8_6, temperature8_6, time_val8_6, time_per_step8_6 = load_energy.load_values_energy(folder_6, 'energy/cdft_eps-1e-3.out')
energy_kinetic10_6, energy_potential10_6, energy_total10_6, temperature10_6, time_val10_6, time_per_step10_6 = load_energy.load_values_energy(folder_6, 'energy/cdft_eps-3e-3.out')
energy_kinetic11_6, energy_potential11_6, energy_total11_6, temperature11_6, time_val11_6, time_per_step11_6 = load_energy.load_values_energy(folder_6, 'energy/cdft_eps-1e-4.out')
energy_kinetic12_6, energy_potential12_6, energy_total12_6, temperature12_6, time_val12_6, time_per_step12_6 = load_energy.load_values_energy(folder_6, 'energy/cdft_eps-1e-1-cp2k-8.2.out')
energy_kinetic13_6, energy_potential13_6, energy_total13_6, temperature13_6, time_val13_6, time_per_step13_6 = load_energy.load_values_energy(folder_6, 'energy/cdft_eps-1e-2-cp2k-8.2.out')
energy_kinetic14_6, energy_potential14_6, energy_total14_6, temperature14_6, time_val14_6, time_per_step14_6 = load_energy.load_values_energy(folder_6, 'energy/cdft_eps-1e-3-cp2k-8.2.out')
energy_kinetic15_6, energy_potential15_6, energy_total15_6, temperature15_6, time_val15_6, time_per_step15_6 = load_energy.load_values_energy(folder_6, 'energy/cdft_eps-3e-3-cp2k-8.2.out')
energy_kinetic16_6, energy_potential16_6, energy_total16_6, temperature16_6, time_val16_6, time_per_step16_6 = load_energy.load_values_energy(folder_6, 'energy/cdft_eps-1e-4-cp2k-8.2.out')
strength7_6 = np.loadtxt('{}/strength/cdft_eps-1e-2.out'.format(folder_6))
strength8_6 = np.loadtxt('{}/strength/cdft_eps-1e-3.out'.format(folder_6))
strength13_6 = np.loadtxt('{}/strength/cdft_eps-1e-2-cp2k-8.2.out'.format(folder_6))
strength15_6 = np.loadtxt('{}/strength/cdft_eps-3e-3-cp2k-8.2.out'.format(folder_6))

folder_7 = 'F:/Backup/Archer-2/other/cdft/bivo/electron-final/analysis'
energy_kinetic6_7, energy_potential6_7, energy_total6_7, temperature6_7, time_val6_7, time_per_step6_7 = load_energy.load_values_energy(folder_7, 'energy/dft.out')
energy_kinetic7_7, energy_potential7_7, energy_total7_7, temperature7_7, time_val7_7, time_per_step7_7 = load_energy.load_values_energy(folder_7, 'energy/cdft_1e-2.out')
energy_kinetic8_7, energy_potential8_7, energy_total8_7, temperature8_7, time_val8_7, time_per_step8_7 = load_energy.load_values_energy(folder_7, 'energy/cdft_1e-3.out')
strength7_7 = np.loadtxt('{}/strength/cdft_1e-2.out'.format(folder_7))
strength8_7 = np.loadtxt('{}/strength/cdft_1e-3.out'.format(folder_7))

folder_8 = 'F:/Backup/Archer-2/other/cdft/iron_oxides/hematite/hole_25hfx-final-opt/analysis'
energy_kinetic6_8, energy_potential6_8, energy_total6_8, temperature6_8, time_val6_8, time_per_step6_8 = load_energy.load_values_energy(folder_8, 'energy/dft.out')
energy_kinetic7_8, energy_potential7_8, energy_total7_8, temperature7_8, time_val7_8, time_per_step7_8 = load_energy.load_values_energy(folder_8, 'energy/cdft_1e-2.out')
strength7_8 = np.loadtxt('{}/strength/cdft_1e-2.out'.format(folder_8))

folder_9 = 'F:/Backup/Archer-2/other/cdft/iron_oxides/hematite/hole_331/analysis'
energy_kinetic6_9, energy_potential6_9, energy_total6_9, temperature6_9, time_val6_9, time_per_step6_9 = load_energy.load_values_energy(folder_9, 'energy/dft.out')
energy_kinetic7_9, energy_potential7_9, energy_total7_9, temperature7_9, time_val7_9, time_per_step7_9 = load_energy.load_values_energy(folder_9, 'energy/cdft_1e-2.out')
energy_kinetic8_9, energy_potential8_9, energy_total8_9, temperature8_9, time_val8_9, time_per_step8_9 = load_energy.load_values_energy(folder_9, 'energy/cdft_1e-3.out')
strength7_9 = np.loadtxt('{}/strength/cdft_1e-2.out'.format(folder_9))
strength8_9 = np.loadtxt('{}/strength/cdft_1e-3.out'.format(folder_9))

folder_10 = 'F:/Backup/Archer-2/other/cdft/iron_oxides/hematite/hole_50hfx/analysis'
energy_kinetic6_10, energy_potential6_10, energy_total6_10, temperature6_10, time_val6_10, time_per_step6_10 = load_energy.load_values_energy(folder_10, 'energy/dft.out')
energy_kinetic7_10, energy_potential7_10, energy_total7_10, temperature7_10, time_val7_10, time_per_step7_10 = load_energy.load_values_energy(folder_10, 'energy/cdft_1e-2.out')
energy_kinetic8_10, energy_potential8_10, energy_total8_10, temperature8_10, time_val8_10, time_per_step8_10 = load_energy.load_values_energy(folder_10, 'energy/cdft_1e-3.out')
strength7_10 = np.loadtxt('{}/strength/cdft_1e-2.out'.format(folder_10))
strength8_10 = np.loadtxt('{}/strength/cdft_1e-3.out'.format(folder_10))

folder_12 = 'F:/Backup/Archer-2/other/cdft/iron_oxides/hematite/hole-final/analysis'
energy_kinetic6_12, energy_potential6_12, energy_total6_12, temperature6_12, time_val6_12, time_per_step6_12 = load_energy.load_values_energy(folder_12, 'energy/dft.out')
energy_kinetic7_12, energy_potential7_12, energy_total7_12, temperature7_12, time_val7_12, time_per_step7_12 = load_energy.load_values_energy(folder_12, 'energy/cdft_1e-2.out')
energy_kinetic8_12, energy_potential8_12, energy_total8_12, temperature8_12, time_val8_12, time_per_step8_12 = load_energy.load_values_energy(folder_12, 'energy/cdft_1e-3.out')
energy_kinetic10_12, energy_potential10_12, energy_total10_12, temperature10_12, time_val10_12, time_per_step10_12 = load_energy.load_values_energy(folder_12, 'energy/cdft_3e-3.out')
energy_kinetic11_12, energy_potential11_12, energy_total11_12, temperature11_12, time_val11_12, time_per_step11_12 = load_energy.load_values_energy(folder_12, 'energy/cdft_1e-4.out')
energy_kinetic12_12, energy_potential12_12, energy_total12_12, temperature12_12, time_val12_12, time_per_step12_12 = load_energy.load_values_energy(folder_12, 'energy/cdft_1e-1-cp2k-8.2.out')
energy_kinetic13_12, energy_potential13_12, energy_total13_12, temperature13_12, time_val13_12, time_per_step13_12 = load_energy.load_values_energy(folder_12, 'energy/cdft_1e-2-cp2k-8.2.out')
energy_kinetic14_12, energy_potential14_12, energy_total14_12, temperature14_12, time_val14_12, time_per_step14_12 = load_energy.load_values_energy(folder_12, 'energy/cdft_1e-3-cp2k-8.2.out')
energy_kinetic15_12, energy_potential15_12, energy_total15_12, temperature15_12, time_val15_12, time_per_step15_12 = load_energy.load_values_energy(folder_12, 'energy/cdft_3e-3-cp2k-8.2.out')
energy_kinetic16_12, energy_potential16_12, energy_total16_12, temperature16_12, time_val16_12, time_per_step16_12 = load_energy.load_values_energy(folder_12, 'energy/cdft_1e-4-cp2k-8.2.out')
strength7_12 = np.loadtxt('{}/strength/cdft_1e-2.out'.format(folder_12))
strength8_12 = np.loadtxt('{}/strength/cdft_1e-3.out'.format(folder_12))
strength13_12 = np.loadtxt('{}/strength/cdft_1e-2-cp2k-8.2.out'.format(folder_12))
strength15_12 = np.loadtxt('{}/strength/cdft_3e-3-cp2k-8.2.out'.format(folder_12))

folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/energy_conservation'

fig_subplot, ax_subplot = plt.subplots(4, 2, sharex='col', figsize=(9, 9))
time_plot = 100
atoms_h2 = 2
atoms_mgo = 96
atoms_bivo = 192
atoms_hematite_221 = 120
atoms_hematite_331 = 270
atoms_lepidocrocite = 144
y_lim = 10  # micro Hartree

# MgO energy
ax_subplot[0, 0].plot(time_val7_4, 1e6*(energy_total7_4-energy_total7_4[0])/atoms_mgo, 'r', label='CDFT 1e-2')
ax_subplot[0, 0].plot(time_val8_4, 1e6*(energy_total8_4-energy_total8_4[0])/atoms_mgo, 'g', label='CDFT 1e-3')
# ax_subplot[0, 0].plot(time_val9_4, 1e6*(energy_total9_4-energy_total9_4[0])/atoms_mgo, 'b', label='CDFT 1e-4')
ax_subplot[0, 0].plot(time_val6_4, 1e6*(energy_total6_4-energy_total6_4[0])/atoms_mgo, 'k', label='DFT')
ax_subplot[0, 0].set_xlim([0, time_plot])
ax_subplot[0, 0].set_ylim([-y_lim, y_lim])
ax_subplot[0, 0].set_ylabel('Energy / µHa')

# MgO strength
ax_subplot[0, 1].plot(np.arange(0, strength7_4.shape[0]*0.5, 0.5), strength7_4, 'r', label='CDFT 1e-2')
ax_subplot[0, 1].plot(np.arange(0, strength8_4.shape[0]*0.5, 0.5), strength8_4, 'g', label='CDFT 1e-3')
# ax_subplot[0, 1].plot(np.arange(0, strength9_4.shape[0]*0.5, 0.5), strength9_4, 'b', label='CDFT 1e-4')
ax_subplot[0, 1].set_xlim([0, time_plot])
ax_subplot[0, 1].set_ylabel('Lagrange multiplier')

# BiVO4 electron energy
ax_subplot[1, 0].plot(time_val6_7[:200], 1e6*(energy_total6_7[:200]-energy_total6_7[0])/atoms_bivo, 'k', label='DFT')
ax_subplot[1, 0].plot(time_val7_7[:200], 1e6*(energy_total7_7[:200]-energy_total7_7[0])/atoms_bivo, 'r', label='CDFT 1e-2')
ax_subplot[1, 0].plot(time_val8_7[:200], 1e6*(energy_total8_7[:200]-energy_total8_7[0])/atoms_bivo, 'g', label='CDFT 1e-3')
ax_subplot[1, 0].set_xlim([0, time_plot])
ax_subplot[1, 0].set_ylim([-y_lim, y_lim])
ax_subplot[1, 0].set_ylabel('Energy / µHa')

# BiVO4 electron strength
ax_subplot[1, 1].plot(np.arange(0, strength7_7.shape[0]*0.5, 0.5)[:200], strength7_7[:200], 'r', label='CDFT 1e-2')
ax_subplot[1, 1].plot(np.arange(0, strength8_7.shape[0]*0.5, 0.5)[:200], strength8_7[:200], 'g', label='CDFT 1e-3')
ax_subplot[1, 1].set_xlim([0, time_plot])
ax_subplot[1, 1].set_ylabel('Lagrange multiplier')

# Lepidocrocite hole energy
ax_subplot[2, 0].plot(time_val6_6[:200], 1e6*(energy_total6_6[:200]-energy_total6_6[0])/atoms_lepidocrocite, 'k', label='DFT')
ax_subplot[2, 0].plot(time_val13_6[:200], 1e6*(energy_total13_6[:200]-energy_total13_6[0])/atoms_lepidocrocite, 'r', label='CDFT 1e-2')
ax_subplot[2, 0].plot(time_val15_6[:200], 1e6*(energy_total15_6[:200]-energy_total15_6[0])/atoms_lepidocrocite, 'g', label='CDFT 1e-3')
ax_subplot[2, 0].set_xlim([0, time_plot])
ax_subplot[2, 0].set_ylim([-y_lim, y_lim])
ax_subplot[2, 0].set_ylabel('Energy / µHa')

# Lepidocrocite hole strength
ax_subplot[2, 1].plot(np.arange(0, strength13_6.shape[0]*0.5, 0.5)[:200], strength13_6[:200], 'r', label='CDFT 1e-2')
ax_subplot[2, 1].plot(np.arange(0, strength15_6.shape[0]*0.5, 0.5)[:200], strength15_6[:200], 'g', label='CDFT 1e-3')
ax_subplot[2, 1].set_xlim([0, time_plot])
ax_subplot[2, 1].set_ylabel('Lagrange multiplier')
# ax_subplot[2, 1].set_ylim([-0.001, 0.006])

# Hematite hole energy
ax_subplot[3, 0].plot(time_val6_12[:200], 1e6*(energy_total6_12[:200]-energy_total6_12[0])/atoms_hematite_221, 'k', label='DFT')
ax_subplot[3, 0].plot(time_val13_12[:200], 1e6*(energy_total13_12[:200]-energy_total13_12[0])/atoms_hematite_221, 'r', label='CDFT 1e-2')
ax_subplot[3, 0].plot(time_val15_12[:200], 1e6*(energy_total15_12[:200]-energy_total15_12[0])/atoms_hematite_221, 'g', label='CDFT 1e-3')
ax_subplot[3, 0].set_xlim([0, time_plot])
ax_subplot[3, 0].set_ylim([-y_lim, y_lim])
ax_subplot[3, 0].set_ylabel('Energy / µHa')
ax_subplot[3, 0].set_xlabel('Time / fs')

# Hematite hole strength
ax_subplot[3, 1].plot(np.arange(0, strength13_12.shape[0]*0.5, 0.5)[:200], strength13_12[:200], 'r', label='CDFT 1e-2')
ax_subplot[3, 1].plot(np.arange(0, strength15_12.shape[0]*0.5, 0.5)[:200], strength15_12[:200], 'g', label='CDFT 1e-3')
ax_subplot[3, 1].set_xlim([0, time_plot])
ax_subplot[3, 1].set_ylabel('Lagrange multiplier')
ax_subplot[3, 1].set_xlabel('Lagrange multiplier')
# ax_subplot[3, 1].set_ylim([-0.001, 0.006])

fig_subplot.tight_layout()
fig_subplot.savefig('{}/cdft_energy_conservation_strength_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
