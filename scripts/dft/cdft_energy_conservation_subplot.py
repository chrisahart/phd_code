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


folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/h2/analysis/final-4h/energy'
energy_kinetic6_1, energy_potential6_1, energy_total6_1, temperature6_1, time_val6_1, time_per_step6_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-2_cell-10A-opt3-md-node2.out')
energy_kinetic7_1, energy_potential7_1, energy_total7_1, temperature7_1, time_val7_1, time_per_step7_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-3_cell-10A-opt3-md-node2.out')
energy_kinetic8_1, energy_potential8_1, energy_total8_1, temperature8_1, time_val8_1, time_per_step8_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-4_cell-10A-opt3-md-node2.out')
energy_kinetic9_1, energy_potential9_1, energy_total9_1, temperature9_1, time_val9_1, time_per_step9_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-5_cell-10A-opt3-md-node2.out')
energy_kinetic10_1, energy_potential10_1, energy_total10_1, temperature10_1, time_val10_1, time_per_step10_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-6_cell-10A-opt3-md-node2.out')
energy_kinetic11_1, energy_potential11_1, energy_total11_1, temperature11_1, time_val11_1, time_per_step11_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-7_cell-10A-opt3-md-node2.out')
energy_kinetic12_1, energy_potential12_1, energy_total12_1, temperature12_1, time_val12_1, time_per_step12_1 = load_energy.load_values_energy(folder_1, 'dft_from_opt.out')
energy_kinetic13_1, energy_potential13_1, energy_total13_1, temperature13_1, time_val13_1, time_per_step13_1 = load_energy.load_values_energy(folder_1, 'dft_from_opt_neutral.out')

# folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/h2/analysis/final-4hr-pbc/energy'
folder_2 = 'F:/Backup/Archer-2/other/cdft/h2/md/final-4hr-pbc/analysis'
energy_kinetic6_2, energy_potential6_2, energy_total6_2, temperature6_2, time_val6_2, time_per_step6_2 = load_energy.load_values_energy(folder_2, 'energy/nve-hirshfeld-charge7_eps-1e-2_cell-10A-opt3-md-node2.out')
energy_kinetic7_2, energy_potential7_2, energy_total7_2, temperature7_2, time_val7_2, time_per_step7_2 = load_energy.load_values_energy(folder_2, 'energy/nve-hirshfeld-charge7_eps-1e-3_cell-10A-opt3-md-node2.out')
energy_kinetic8_2, energy_potential8_2, energy_total8_2, temperature8_2, time_val8_2, time_per_step8_2 = load_energy.load_values_energy(folder_2, 'energy/nve-hirshfeld-charge7_eps-1e-4_cell-10A-opt3-md-node2.out')
energy_kinetic9_2, energy_potential9_2, energy_total9_2, temperature9_2, time_val9_2, time_per_step9_2 = load_energy.load_values_energy(folder_2, 'energy/nve-hirshfeld-charge7_eps-1e-5_cell-10A-opt3-md-node2.out')
energy_kinetic10_2, energy_potential10_2, energy_total10_2, temperature10_2, time_val10_2, time_per_step10_2 = load_energy.load_values_energy(folder_2, 'energy/nve-hirshfeld-charge7_eps-1e-6_cell-10A-opt3-md-node2.out')
energy_kinetic11_2, energy_potential11_2, energy_total11_2, temperature11_2, time_val11_2, time_per_step11_2 = load_energy.load_values_energy(folder_2, 'energy/nve-hirshfeld-charge7_eps-1e-7_cell-10A-opt3-md-node2.out')
energy_kinetic12_2, energy_potential12_2, energy_total12_2, temperature12_2, time_val12_2, time_per_step12_2 = load_energy.load_values_energy(folder_2, 'energy/dft_from_opt.out')
energy_kinetic13_2, energy_potential13_2, energy_total13_2, temperature13_2, time_val13_2, time_per_step13_2 = load_energy.load_values_energy(folder_2, 'energy/dft_from_opt_neutral.out')
strength6_2 = np.loadtxt('{}/strength/nve-hirshfeld-charge7_eps-1e-2_cell-10A-opt3-md-node2.out'.format(folder_2))
strength7_2 = np.loadtxt('{}/strength/nve-hirshfeld-charge7_eps-1e-3_cell-10A-opt3-md-node2.out'.format(folder_2))
strength8_2 = np.loadtxt('{}/strength/nve-hirshfeld-charge7_eps-1e-4_cell-10A-opt3-md-node2.out'.format(folder_2))
strength9_2 = np.loadtxt('{}/strength/nve-hirshfeld-charge7_eps-1e-5_cell-10A-opt3-md-node2.out'.format(folder_2))
strength10_2 = np.loadtxt('{}/strength/nve-hirshfeld-charge7_eps-1e-6_cell-10A-opt3-md-node2.out'.format(folder_2))
strength11_2 = np.loadtxt('{}/strength/nve-hirshfeld-charge7_eps-1e-7_cell-10A-opt3-md-node2.out'.format(folder_2))

folder_3 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/h2/analysis/final-4hr_dft-eps-scf/energy'
energy_kinetic6_3, energy_potential6_3, energy_total6_3, temperature6_3, time_val6_3, time_per_step6_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-2.out')
energy_kinetic7_3, energy_potential7_3, energy_total7_3, temperature7_3, time_val7_3, time_per_step7_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-3.out')
energy_kinetic8_3, energy_potential8_3, energy_total8_3, temperature8_3, time_val8_3, time_per_step8_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-4.out')
energy_kinetic9_3, energy_potential9_3, energy_total9_3, temperature9_3, time_val9_3, time_per_step9_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-5.out')
energy_kinetic10_3, energy_potential10_3, energy_total10_3, temperature10_3, time_val10_3, time_per_step10_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-6.out')
energy_kinetic11_3, energy_potential11_3, energy_total11_3, temperature11_3, time_val11_3, time_per_step11_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-7.out')
energy_kinetic12_3, energy_potential12_3, energy_total12_3, temperature12_3, time_val12_3, time_per_step12_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-1.out')

folder_4 = 'F:/Backup/Archer-1/work/cahart/other/cdft/MgO/cpmd_restart/cell-110-96-4nn/md/pbe/analysis'
energy_kinetic6_4, energy_potential6_4, energy_total6_4, temperature6_4, time_val6_4, time_per_step6_4 = load_energy.load_values_energy(folder_4, 'energy/dft.out')
energy_kinetic7_4, energy_potential7_4, energy_total7_4, temperature7_4, time_val7_4, time_per_step7_4 = load_energy.load_values_energy(folder_4, 'energy/cdft_eps-1e-2.out')
energy_kinetic8_4, energy_potential8_4, energy_total8_4, temperature8_4, time_val8_4, time_per_step8_4 = load_energy.load_values_energy(folder_4, 'energy/cdft_eps-1e-3.out')
energy_kinetic9_4, energy_potential9_4, energy_total9_4, temperature9_4, time_val9_4, time_per_step9_4 = load_energy.load_values_energy(folder_4, 'energy/cdft_eps-1e-4.out')
energy_kinetic10_4, energy_potential10_4, energy_total10_4, temperature10_4, time_val10_4, time_per_step10_4 = load_energy.load_values_energy(folder_4, 'energy/cdft_eps-1e-5.out')
strength7_4 = np.loadtxt('{}/strength/cdft_eps-1e-2.out'.format(folder_4))
strength8_4 = np.loadtxt('{}/strength/cdft_eps-1e-3.out'.format(folder_4))
strength9_4 = np.loadtxt('{}/strength/cdft_eps-1e-4.out'.format(folder_4))
strength10_4 = np.loadtxt('{}/strength/cdft_eps-1e-5.out'.format(folder_4))

folder_51 = 'F:/Backup/Archer-2/other/cdft/iron_oxides/hematite/electron-24hr/analysis'
energy_kinetic6_51, energy_potential6_51, energy_total6_51, temperature6_51, time_val6_51, time_per_step6_51 = load_energy.load_values_energy(folder_51, 'energy/dft.out')
energy_kinetic7_51, energy_potential7_51, energy_total7_51, temperature7_51, time_val7_51, time_per_step7_51 = load_energy.load_values_energy(folder_51, 'energy/cdft_1e-2.out')
energy_kinetic8_51, energy_potential8_51, energy_total8_51, temperature8_51, time_val8_51, time_per_step8_51 = load_energy.load_values_energy(folder_51, 'energy/cdft_1e-3.out')
strength7_51 = np.loadtxt('{}/strength/cdft_1e-2.out'.format(folder_51))
strength8_51 = np.loadtxt('{}/strength/cdft_1e-3.out'.format(folder_51))

folder_52 = 'F:/Backup/Archer-2/other/cdft/iron_oxides/hematite/hole-24hr/analysis'
energy_kinetic6_52, energy_potential6_52, energy_total6_52, temperature6_52, time_val6_52, time_per_step6_52 = load_energy.load_values_energy(folder_52, 'energy/dft.out')
energy_kinetic7_52, energy_potential7_52, energy_total7_52, temperature7_52, time_val7_52, time_per_step7_52 = load_energy.load_values_energy(folder_52, 'energy/cdft_1e-2.out')
energy_kinetic8_52, energy_potential8_52, energy_total8_52, temperature8_52, time_val8_52, time_per_step8_52 = load_energy.load_values_energy(folder_52, 'energy/cdft_1e-3.out')
strength7_52 = np.loadtxt('{}/strength/cdft_1e-2.out'.format(folder_52))
strength8_52 = np.loadtxt('{}/strength/cdft_1e-3.out'.format(folder_52))

folder_6 = 'F:/Backup/Archer-2/other/cdft/iron_oxides/lepidocrocite/hole-24hr/analysis'
energy_kinetic6_6, energy_potential6_6, energy_total6_6, temperature6_6, time_val6_6, time_per_step6_6 = load_energy.load_values_energy(folder_6, 'energy/dft.out')
energy_kinetic7_6, energy_potential7_6, energy_total7_6, temperature7_6, time_val7_6, time_per_step7_6 = load_energy.load_values_energy(folder_6, 'energy/cdft_eps-1e-2.out')
energy_kinetic8_6, energy_potential8_6, energy_total8_6, temperature8_6, time_val8_6, time_per_step8_6 = load_energy.load_values_energy(folder_6, 'energy/cdft_eps-1e-3.out')
strength7_6 = np.loadtxt('{}/strength/cdft_eps-1e-2.out'.format(folder_6))
strength8_6 = np.loadtxt('{}/strength/cdft_eps-1e-3.out'.format(folder_6))

folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/h2/analysis/final-4hr-pbc/energy'
fig_subplot, ax_subplot = plt.subplots(5, 2, sharex='col', figsize=(9, 9))
time_plot = 1000
atoms_h2 = 2
atoms_mgo = 96
atoms_hematite = 120
atoms_lepidocrocite = 144
y_lim = 20  # micro Hartree
# y_lim = 100  # micro Hartree

# H2 energy
ax_subplot[0, 0].plot(time_val7_2, 1e6*(energy_total7_2-energy_total7_2[0])/atoms_h2, 'g', label='CDFT 1e-3')
ax_subplot[0, 0].plot(time_val8_2, 1e6*(energy_total8_2-energy_total8_2[0])/atoms_h2, 'b', label='CDFT 1e-4')
ax_subplot[0, 0].plot(time_val9_2, 1e6*(energy_total9_2-energy_total9_2[0])/atoms_h2, 'y', label='CDFT 1e-5')
ax_subplot[0, 0].plot(time_val12_2, 1e6*(energy_total12_2-energy_total12_2[0])/atoms_h2, 'k', label='DFT')
ax_subplot[0, 0].set_xlim([0, time_plot])
ax_subplot[0, 0].set_ylim([-y_lim, y_lim])
ax_subplot[0, 0].set_ylabel('Energy / micro Ha')

# H2 strength
ax_subplot[0, 1].plot(time_val7_2, strength7_2, 'g', label='CDFT 1e-3')
ax_subplot[0, 1].plot(time_val8_2, strength8_2, 'b', label='CDFT 1e-4')
ax_subplot[0, 1].plot(time_val9_2, strength9_2, 'y', label='CDFT 1e-5')
ax_subplot[0, 1].set_xlim([0, time_plot])
ax_subplot[0, 1].set_ylabel('Lagrange multiplier')

# MgO energy
ax_subplot[1, 0].plot(time_val7_4, 1e6*(energy_total7_4-energy_total7_4[0])/atoms_mgo, 'r', label='CDFT 1e-2')
ax_subplot[1, 0].plot(time_val8_4, 1e6*(energy_total8_4-energy_total8_4[0])/atoms_mgo, 'g', label='CDFT 1e-3')
ax_subplot[1, 0].plot(time_val9_4, 1e6*(energy_total9_4-energy_total9_4[0])/atoms_mgo, 'b', label='CDFT 1e-4')
ax_subplot[1, 0].plot(time_val10_4, 1e6*(energy_total10_4-energy_total10_4[0])/atoms_mgo, 'y', label='CDFT 1e-5')
ax_subplot[1, 0].plot(time_val6_4, 1e6*(energy_total6_4-energy_total6_4[0])/atoms_mgo, 'k', label='DFT')
ax_subplot[1, 0].set_xlim([0, time_plot])
# ax_subplot[1, 0].set_ylim([-1.2, 1.2])
ax_subplot[1, 0].set_ylim([-y_lim, y_lim])
ax_subplot[1, 0].set_ylabel('Energy / micro Ha')

# MgO strength
ax_subplot[1, 1].plot(time_val7_4, strength7_4, 'r', label='CDFT 1e-2')
ax_subplot[1, 1].plot(time_val8_4, strength8_4, 'g', label='CDFT 1e-3')
ax_subplot[1, 1].plot(time_val9_4, strength9_4, 'b', label='CDFT 1e-4')
ax_subplot[1, 1].plot(time_val10_4, strength10_4, 'y', label='CDFT 1e-5')
ax_subplot[1, 1].set_xlim([0, time_plot])
ax_subplot[1, 1].set_ylabel('Lagrange multiplier')

# Lepidocrocite energy
ax_subplot[2, 0].plot(time_val6_6, 1e6*(energy_total6_6-energy_total6_6[0])/atoms_lepidocrocite, 'k', label='DFT')
ax_subplot[2, 0].plot(time_val7_6, 1e6*(energy_total7_6-energy_total7_6[0])/atoms_lepidocrocite, 'r', label='CDFT 1e-2')
ax_subplot[2, 0].plot(time_val8_6, 1e6*(energy_total8_6-energy_total8_6[0])/atoms_lepidocrocite, 'g', label='CDFT 1e-3')
ax_subplot[2, 0].set_xlim([0, time_plot])
ax_subplot[2, 0].set_ylim([-y_lim, y_lim])
ax_subplot[2, 0].set_ylabel('Energy / micro Ha')

# Lepidocrocite strength
ax_subplot[2, 1].plot(time_val7_6, strength7_6, 'r', label='CDFT 1e-2')
ax_subplot[2, 1].plot(time_val8_6, strength8_6, 'g', label='CDFT 1e-3')
ax_subplot[2, 1].set_xlim([0, time_plot])
ax_subplot[2, 1].set_ylabel('Lagrange multiplier')

# Hematite hole energy
ax_subplot[3, 0].plot(time_val7_52, 1e6*(energy_total7_52-energy_total7_52[0])/atoms_hematite, 'r', label='CDFT 1e-2')
ax_subplot[3, 0].plot(time_val8_52, 1e6*(energy_total8_52-energy_total8_52[0])/atoms_hematite, 'g', label='CDFT 1e-3')
ax_subplot[3, 0].plot(time_val6_52, 1e6*(energy_total6_52-energy_total6_52[0])/atoms_hematite, 'k', label='DFT')
ax_subplot[3, 0].set_xlim([0, time_plot])
ax_subplot[3, 0].set_ylim([-y_lim, y_lim])
ax_subplot[3, 0].set_ylabel('Energy / micro Ha')

# Hematite hole strength
# ax_subplot[3, 1].plot(time_val7_52, strength7_52, 'r', label='CDFT 1e-2')
# ax_subplot[3, 1].plot(time_val8_52, strength8_52, 'g', label='CDFT 1e-3')
ax_subplot[3, 1].plot(np.arange(0, strength7_52.shape[0]*0.5, 0.5), strength7_52, 'r', label='CDFT 1e-2')
ax_subplot[3, 1].plot(np.arange(0, strength8_52.shape[0]*0.5, 0.5), strength8_52, 'g', label='CDFT 1e-3')
ax_subplot[3, 1].set_xlim([0, time_plot])
ax_subplot[3, 1].set_ylabel('Lagrange multiplier')

# Hematite electron energy
ax_subplot[4, 0].plot(time_val7_51, 1e6*(energy_total7_51-energy_total7_51[0])/atoms_mgo, 'r', label='CDFT 1e-2')
ax_subplot[4, 0].plot(time_val8_51, 1e6*(energy_total8_51-energy_total8_51[0])/atoms_mgo, 'g', label='CDFT 1e-3')
ax_subplot[4, 0].plot(time_val6_51, 1e6*(energy_total6_51-energy_total6_51[0])/atoms_mgo, 'k', label='DFT')
ax_subplot[4, 0].set_xlim([0, time_plot])
ax_subplot[4, 0].set_ylim([-y_lim, y_lim])
ax_subplot[4, 0].set_ylabel('Energy / micro Ha')
ax_subplot[4, 1].set_xlabel('Time / fs')

# Hematite electron strength
ax_subplot[4, 1].plot(time_val7_51, strength7_51, 'r', label='CDFT 1e-2')
ax_subplot[4, 1].plot(time_val8_51, strength8_51, 'g', label='CDFT 1e-3')
ax_subplot[4, 1].set_xlim([0, time_plot])
ax_subplot[4, 1].set_ylabel('Lagrange multiplier')
ax_subplot[4, 1].set_xlabel('Time / fs')

fig_subplot.tight_layout()
# fig_subplot.subplots_adjust(hspace=0)
fig_subplot.savefig('{}/cdft_energy_conservation_strength_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')


if __name__ == "__main__":
    print('Finished.')
    plt.show()
