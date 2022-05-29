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


atoms = 96
folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/h2/analysis/final-4h/energy'
energy_kinetic6_1, energy_potential6_1, energy_total6_1, temperature6_1, time_val6_1, time_per_step6_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-2_cell-10A-opt3-md-node2.out')
energy_kinetic7_1, energy_potential7_1, energy_total7_1, temperature7_1, time_val7_1, time_per_step7_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-3_cell-10A-opt3-md-node2.out')
energy_kinetic8_1, energy_potential8_1, energy_total8_1, temperature8_1, time_val8_1, time_per_step8_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-4_cell-10A-opt3-md-node2.out')
energy_kinetic9_1, energy_potential9_1, energy_total9_1, temperature9_1, time_val9_1, time_per_step9_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-5_cell-10A-opt3-md-node2.out')
energy_kinetic10_1, energy_potential10_1, energy_total10_1, temperature10_1, time_val10_1, time_per_step10_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-6_cell-10A-opt3-md-node2.out')
energy_kinetic11_1, energy_potential11_1, energy_total11_1, temperature11_1, time_val11_1, time_per_step11_1 = load_energy.load_values_energy(folder_1, 'nve-hirshfeld-charge7_eps-1e-7_cell-10A-opt3-md-node2.out')
energy_kinetic12_1, energy_potential12_1, energy_total12_1, temperature12_1, time_val12_1, time_per_step12_1 = load_energy.load_values_energy(folder_1, 'dft_from_opt.out')
energy_kinetic13_1, energy_potential13_1, energy_total13_1, temperature13_1, time_val13_1, time_per_step13_1 = load_energy.load_values_energy(folder_1, 'dft_from_opt_neutral.out')

folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/h2/analysis/final-4hr-pbc'
energy_kinetic6_2, energy_potential6_2, energy_total6_2, temperature6_2, time_val6_2, time_per_step6_2 = load_energy.load_values_energy(folder_2, 'energy/nve-hirshfeld-charge7_eps-1e-2_cell-10A-opt3-md-node2.out')
energy_kinetic7_2, energy_potential7_2, energy_total7_2, temperature7_2, time_val7_2, time_per_step7_2 = load_energy.load_values_energy(folder_2, 'energy/nve-hirshfeld-charge7_eps-1e-3_cell-10A-opt3-md-node2.out')
energy_kinetic8_2, energy_potential8_2, energy_total8_2, temperature8_2, time_val8_2, time_per_step8_2 = load_energy.load_values_energy(folder_2, 'energy/nve-hirshfeld-charge7_eps-1e-4_cell-10A-opt3-md-node2.out')
energy_kinetic9_2, energy_potential9_2, energy_total9_2, temperature9_2, time_val9_2, time_per_step9_2 = load_energy.load_values_energy(folder_2, 'energy/nve-hirshfeld-charge7_eps-1e-5_cell-10A-opt3-md-node2.out')
energy_kinetic10_2, energy_potential10_2, energy_total10_2, temperature10_2, time_val10_2, time_per_step10_2 = load_energy.load_values_energy(folder_2, 'energy/nve-hirshfeld-charge7_eps-1e-6_cell-10A-opt3-md-node2.out')
energy_kinetic11_2, energy_potential11_2, energy_total11_2, temperature11_2, time_val11_2, time_per_step11_2 = load_energy.load_values_energy(folder_2, 'energy/nve-hirshfeld-charge7_eps-1e-7_cell-10A-opt3-md-node2.out')
energy_kinetic12_2, energy_potential12_2, energy_total12_2, temperature12_2, time_val12_2, time_per_step12_2 = load_energy.load_values_energy(folder_2, 'energy/dft_from_opt.out')
energy_kinetic13_2, energy_potential13_2, energy_total13_2, temperature13_2, time_val13_2, time_per_step13_2 = load_energy.load_values_energy(folder_2, 'energy/dft_from_opt_neutral.out')
# conserved6_2 = np.loadtxt('{}/energy_conserved/nve-hirshfeld-charge7_eps-1e-2_cell-10A-opt3-md-node2.out'.format(folder_2))
# conserved7_2 = np.loadtxt('{}/energy_conserved/nve-hirshfeld-charge7_eps-1e-3_cell-10A-opt3-md-node2.out'.format(folder_2))
# conserved8_2 = np.loadtxt('{}/energy_conserved/nve-hirshfeld-charge7_eps-1e-4_cell-10A-opt3-md-node2.out'.format(folder_2))
# conserved9_2 = np.loadtxt('{}/energy_conserved/nve-hirshfeld-charge7_eps-1e-5_cell-10A-opt3-md-node2.out'.format(folder_2))
# conserved10_2 = np.loadtxt('{}/energy_conserved/nve-hirshfeld-charge7_eps-1e-6_cell-10A-opt3-md-node2.out'.format(folder_2))
# conserved11_2 = np.loadtxt('{}/energy_conserved/nve-hirshfeld-charge7_eps-1e-7_cell-10A-opt3-md-node2.out'.format(folder_2))
# conserved12_2 = np.loadtxt('{}/energy_conserved/dft_from_opt.out'.format(folder_2))
# conserved13_2 = np.loadtxt('{}/energy_conserved/dft_from_opt_neutral.out'.format(folder_2))

folder_3 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/h2/analysis/final-4hr_dft-eps-scf/energy'
energy_kinetic6_3, energy_potential6_3, energy_total6_3, temperature6_3, time_val6_3, time_per_step6_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-2.out')
energy_kinetic7_3, energy_potential7_3, energy_total7_3, temperature7_3, time_val7_3, time_per_step7_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-3.out')
energy_kinetic8_3, energy_potential8_3, energy_total8_3, temperature8_3, time_val8_3, time_per_step8_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-4.out')
energy_kinetic9_3, energy_potential9_3, energy_total9_3, temperature9_3, time_val9_3, time_per_step9_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-5.out')
energy_kinetic10_3, energy_potential10_3, energy_total10_3, temperature10_3, time_val10_3, time_per_step10_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-6.out')
energy_kinetic11_3, energy_potential11_3, energy_total11_3, temperature11_3, time_val11_3, time_per_step11_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-7.out')
energy_kinetic12_3, energy_potential12_3, energy_total12_3, temperature12_3, time_val12_3, time_per_step12_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-1.out')

folder_41 = 'F:/Backup/Archer-1/work/cahart/other/cdft/MgO/cpmd_restart/cell-110-96-4nn/md/pbe/analysis'
# folder_42 = 'F:/Backup/Archer-1/work/cahart/other/cdft/MgO/cpmd_restart/cell-110-96-4nn/md/pbe-24hr/analysis'
folder_4 = 'F:/Backup/Archer-1/work/cahart/other/cdft/MgO/cpmd_restart/cell-110-96-4nn/md/pbe-final/analysis'
# folder_4 = folder_41
energy_kinetic6_4, energy_potential6_4, energy_total6_4, temperature6_4, time_val6_4, time_per_step6_4 = load_energy.load_values_energy(folder_41, 'energy/dft.out')
energy_kinetic7_4, energy_potential7_4, energy_total7_4, temperature7_4, time_val7_4, time_per_step7_4 = load_energy.load_values_energy(folder_4, 'energy/cdft_eps-1e-2.out')
energy_kinetic8_4, energy_potential8_4, energy_total8_4, temperature8_4, time_val8_4, time_per_step8_4 = load_energy.load_values_energy(folder_4, 'energy/cdft_eps-1e-3.out')
energy_kinetic9_4, energy_potential9_4, energy_total9_4, temperature9_4, time_val9_4, time_per_step9_4 = load_energy.load_values_energy(folder_4, 'energy/cdft_eps-1e-4.out')
conserved6_4 = np.genfromtxt('{}/energy_conserved/dft.out'.format(folder_41), skip_footer=0)
conserved7_4 = np.genfromtxt('{}/energy_conserved/cdft_eps-1e-2.out'.format(folder_4), skip_footer=0)
conserved8_4 = np.genfromtxt('{}/energy_conserved/cdft_eps-1e-3.out'.format(folder_4), skip_footer=0)
conserved9_4 = np.genfromtxt('{}/energy_conserved/cdft_eps-1e-4.out'.format(folder_4), skip_footer=0)

folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/energy_conservation'

# Plot total energy CDFT
time_plot = 100
fig_energy, ax_energy = plt.subplots()
ax_energy.plot(time_val6_4, (energy_total6_4-energy_total6_4[0])/atoms, 'k', label='DFT')
ax_energy.plot(time_val7_4, (energy_total7_4-energy_total7_4[0])/atoms, 'r', label='CDFT 1e-2')
ax_energy.plot(time_val8_4, (energy_total8_4-energy_total8_4[0])/atoms, 'g', label='CDFT 1e-3')
ax_energy.plot(time_val9_4, (energy_total9_4-energy_total9_4[0])/atoms, 'b', label='CDFT 1e-4')
ax_energy.set_xlabel('Time / fs')
ax_energy.set_ylabel('Energy drift per atom / Ha')
ax_energy.set_xlim([0, time_plot])
ax_energy.set_ylim([-1e-6, 1e-6])
# ax_energy.set_ylim([-1e-4/3, 3e-4/3])
ax_energy.legend(frameon=False)
fig_energy.tight_layout()
fig_energy.savefig('{}/energy_cdft_pbc_mgo.png'.format(folder_save), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
