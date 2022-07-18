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
    drift = np.abs(np.max(energy[0:interval])-np.max(energy[-interval:])) / atoms
    norm_factor = 1000 / ((np.shape(energy)[0]*0.5))
    drift_norm = drift * norm_factor
    return drift_norm


def metric_time(t_cdft, t_dft):
    """ Energy drift as peak to peak distance"""
    t_cdft_avg = np.mean(t_cdft[2:])
    t_dft_avg = np.mean(t_dft[2:])
    return t_cdft_avg / t_dft_avg

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

folder_3 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/h2/analysis/final-4hr_dft-eps-scf/energy'
energy_kinetic6_3, energy_potential6_3, energy_total6_3, temperature6_3, time_val6_3, time_per_step6_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-2.out')
energy_kinetic7_3, energy_potential7_3, energy_total7_3, temperature7_3, time_val7_3, time_per_step7_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-3.out')
energy_kinetic8_3, energy_potential8_3, energy_total8_3, temperature8_3, time_val8_3, time_per_step8_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-4.out')
energy_kinetic9_3, energy_potential9_3, energy_total9_3, temperature9_3, time_val9_3, time_per_step9_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-5.out')
energy_kinetic10_3, energy_potential10_3, energy_total10_3, temperature10_3, time_val10_3, time_per_step10_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-6.out')
energy_kinetic11_3, energy_potential11_3, energy_total11_3, temperature11_3, time_val11_3, time_per_step11_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-7.out')
energy_kinetic12_3, energy_potential12_3, energy_total12_3, temperature12_3, time_val12_3, time_per_step12_3 = load_energy.load_values_energy(folder_3, 'eps_dft-1e-1.out')

folder_4 = 'F:/Backup/Archer-1/work/cahart/other/cdft/MgO/cpmd_restart/cell-110-96-4nn/md/pbe-final/analysis/energy'
energy_kinetic6_4, energy_potential6_4, energy_total6_4, temperature6_4, time_val6_4, time_per_step6_4 = load_energy.load_values_energy(folder_4, 'dft.out')
energy_kinetic7_4, energy_potential7_4, energy_total7_4, temperature7_4, time_val7_4, time_per_step7_4 = load_energy.load_values_energy(folder_4, 'cdft_eps-1e-2.out')
energy_kinetic8_4, energy_potential8_4, energy_total8_4, temperature8_4, time_val8_4, time_per_step8_4 = load_energy.load_values_energy(folder_4, 'cdft_eps-1e-3.out')
energy_kinetic9_4, energy_potential9_4, energy_total9_4, temperature9_4, time_val9_4, time_per_step9_4 = load_energy.load_values_energy(folder_4, 'cdft_eps-1e-4.out')

folder_5 = 'F:/Backup/Archer-1/work/cahart/other/cdft/MgO/cpmd_restart/cell-110-96-4nn/md/pbe0-final/analysis/energy'
energy_kinetic6_5, energy_potential6_5, energy_total6_5, temperature6_5, time_val6_5, time_per_step6_5 = load_energy.load_values_energy(folder_5, 'dft.out')
energy_kinetic7_5, energy_potential7_5, energy_total7_5, temperature7_5, time_val7_5, time_per_step7_5 = load_energy.load_values_energy(folder_5, 'cdft_eps-1e-2.out')
energy_kinetic8_5, energy_potential8_5, energy_total8_5, temperature8_5, time_val8_5, time_per_step8_5 = load_energy.load_values_energy(folder_5, 'cdft_eps-1e-3.out')
energy_kinetic9_5, energy_potential9_5, energy_total9_5, temperature9_5, time_val9_5, time_per_step9_5 = load_energy.load_values_energy(folder_5, 'cdft_eps-1e-4.out')

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
strength14_6 = np.loadtxt('{}/strength/cdft_eps-1e-3-cp2k-8.2.out'.format(folder_6))
strength15_6 = np.loadtxt('{}/strength/cdft_eps-3e-3-cp2k-8.2.out'.format(folder_6))

folder_7 = 'F:/Backup/Archer-2/other/cdft/bivo/electron-final/analysis'
energy_kinetic6_7, energy_potential6_7, energy_total6_7, temperature6_7, time_val6_7, time_per_step6_7 = load_energy.load_values_energy(folder_7, 'energy/dft.out')
energy_kinetic7_7, energy_potential7_7, energy_total7_7, temperature7_7, time_val7_7, time_per_step7_7 = load_energy.load_values_energy(folder_7, 'energy/cdft_1e-2.out')
energy_kinetic8_7, energy_potential8_7, energy_total8_7, temperature8_7, time_val8_7, time_per_step8_7 = load_energy.load_values_energy(folder_7, 'energy/cdft_1e-3.out')
energy_kinetic9_7, energy_potential9_7, energy_total9_7, temperature9_7, time_val9_7, time_per_step9_7 = load_energy.load_values_energy(folder_7, 'energy/cdft_1e-2-cp2k-8.2.out')
energy_kinetic10_7, energy_potential10_7, energy_total10_7, temperature10_7, time_val10_7, time_per_step10_7 = load_energy.load_values_energy(folder_7, 'energy/cdft_1e-3.out')
strength7_7 = np.loadtxt('{}/strength/cdft_1e-2.out'.format(folder_7))
strength8_7 = np.loadtxt('{}/strength/cdft_1e-3.out'.format(folder_7))
strength9_7 = np.loadtxt('{}/strength/cdft_1e-2-cp2k-8.2.out'.format(folder_7))
strength10_7 = np.loadtxt('{}/strength/cdft_1e-3.out'.format(folder_7))

folder_8_1 = 'F:/Backup/Archer-2/other/cdft/ru/md/blyp/equilibrated/dft-24h-inverse/analysis/energy'
folder_8_2 = 'F:/Backup/Archer-2/other/cdft/ru/md/blyp/equilibrated/cdft-24h-inverse/analysis/energy'
energy_kinetic1_8, energy_potential1_8, energy_total1_8, temperature1_8, time_val1_8, time_per_step1_8 = load_energy.load_values_energy(folder_8_1, 'initial-timcon-33-rattle-cpmd.out')
energy_kinetic2_8, energy_potential2_8, energy_total2_8, temperature2_8, time_val2_8, time_per_step2_8 = load_energy.load_values_energy(folder_8_2, 'initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out')

folder_9_1 = 'F:/Backup/Archer-2/other/cdft/ru/md/b3lyp/equilibrated/dft-24h-inverse/analysis/energy'
folder_9_2 = 'F:/Backup/Archer-2/other/cdft/ru/md/b3lyp/equilibrated/cdft-24h-inverse/analysis/energy'
energy_kinetic1_9, energy_potential1_9, energy_total1_9, temperature1_9, time_val1_9, time_per_step1_9 = load_energy.load_values_energy(folder_9_1, 'initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out')
energy_kinetic2_9, energy_potential2_9, energy_total2_9, temperature2_9, time_val2_9, time_per_step2_9 = load_energy.load_values_energy(folder_9_2, 'initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out')

folder_10_1 = 'F:/Backup/Archer-2/other/cdft/ru/md/b97x/equilibrated/dft-24h-inverse/analysis/energy'
folder_10_2 = 'F:/Backup/Archer-2/other/cdft/ru/md/b97x/equilibrated/cdft-24h-inverse/analysis/energy'
energy_kinetic1_10, energy_potential1_10, energy_total1_10, temperature1_10, time_val1_10, time_per_step1_10 = load_energy.load_values_energy(folder_10_1, 'initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out')
energy_kinetic2_10, energy_potential2_10, energy_total2_10, temperature2_10, time_val2_10, time_per_step2_10 = load_energy.load_values_energy(folder_10_2, 'initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out')

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
strength14_12 = np.loadtxt('{}/strength/cdft_1e-3-cp2k-8.2.out'.format(folder_12))
strength15_12 = np.loadtxt('{}/strength/cdft_3e-3-cp2k-8.2.out'.format(folder_12))

folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/energy_conservation'
time_plot = 1000

# Plot energy drift log 1
print('Plot energy drift log 1')
params = {'font.size': 12,
          'axes.labelsize': 14,
          'lines.markersize': 15,
          'legend.fontsize': 10}
plt.rcParams.update(params)
fig_drift, ax_drift = plt.subplots()
data_x_1 = [1e-3, 1e-4, 1e-5, 1e-6]
dft = (-0.658718021--0.658824319)/2
dft_pbc = (-0.657768859--0.657821197)/2
atoms_h2 = 2
atoms_ru = 191
atoms_mgo = 96
atoms_bivo = 192
atoms_hematite_221 = 120
atoms_hematite_331 = 270
atoms_lepidocrocite = 144
offset = 50
offset_mgo = 50
end_short = 100 * 2
end_ru = 1000
interval_mgo = 10
data_y_1 = np.array([metric_2(energy_total7_1, atoms),
                     metric_2(energy_total8_1, atoms),
                     metric_2(energy_total9_1, atoms),
                     metric_2(energy_total10_1, atoms)])
data_y_2 = np.array([metric_2(energy_total7_2, atoms),
                     metric_2(energy_total8_2, atoms),
                     metric_2(energy_total9_2, atoms),
                     metric_2(energy_total10_2, atoms)])
data_y_mgo_pbe = np.array([metric_2(energy_total7_4, atoms_mgo, interval_mgo), metric_2(energy_total8_4, atoms_mgo, interval_mgo)])
data_y_mgo_pbe0 = np.array([metric_2(energy_total7_5[offset_mgo:], atoms_mgo, interval_mgo), metric_2(energy_total8_5[offset_mgo:], atoms_mgo, interval_mgo)])
data_y_lepid = np.array([metric_2(energy_total13_6[offset:end_short], atoms_lepidocrocite), metric_2(energy_total14_6[offset:end_short], atoms_lepidocrocite)])
data_y_hem = np.array([metric_2(energy_total13_12[:end_short], atoms_hematite_221, interval_mgo), metric_2(energy_total14_12[:end_short], atoms_hematite_221, interval_mgo)])
data_y_bivo = np.array([metric_2(energy_total9_7[:end_short], atoms_bivo, interval_mgo), metric_2(energy_total10_7[:], atoms_bivo, interval_mgo)])
ax_drift.plot(data_x_1, data_y_1, '+-', color='black', label=r'H$_{2}$ PBE')
ax_drift.plot(data_x_1, data_y_2, '+-', color='grey', label=r'H$_{2}$ PBE PBC')
ax_drift.plot(5e-4, 4.5e-5, 'rx', label=r'Ru$^{2+}$- Ru$^{3+}$ BLYP')
ax_drift.plot(5e-4, 2.6e-5, 'gx', label=r'Ru$^{2+}$- Ru$^{3+}$ B3LYP')
ax_drift.plot(5e-4, 2.5e-5, 'bx', label=r'Ru$^{2+}$- Ru$^{3+}$ ωB97X')
ax_drift.plot([1e-3], data_y_mgo_pbe[1], 'o', color='orange', fillstyle='none', label='MgO PBE')
ax_drift.plot([1e-3], data_y_mgo_pbe0[1], 'o', color='tan', fillstyle='none', label='MgO PBE0')
ax_drift.plot([1e-3], data_y_bivo[1], 'D', color='y', fillstyle='full', label=r'BiVO$_{4}$ HSE(25%)')
ax_drift.plot([1e-3], data_y_lepid[1], '^', color='c', fillstyle='full', label='Lepidocrocite HSE(18%)')
ax_drift.plot([1e-3], data_y_hem[1], 'P', color='m', fillstyle='full', label='Hematite HSE(12%)')
ax_drift.set_yscale('log')
ax_drift.set_xscale('log')
ax_drift.set_xlabel('Constraint convergence [e]')
ax_drift.set_ylabel('Energy drift [H/atom/ps]')
ax_drift.set_xlim([8e-7, 1.3e-3])
ax_drift.set_ylim([0.9e-7, 1.5e-4])
ax_drift.legend(frameon=False)
fig_drift.tight_layout()
fig_drift.savefig('{}/energy_drift_{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot energy drift log 2
print('Plot energy drift log 2')
fig_drift2, ax_drift2 = plt.subplots()
data_y_1 = np.array([metric_2(energy_total7_1, atoms_h2),
                     metric_2(energy_total8_1, atoms_h2),
                     metric_2(energy_total9_1, atoms_h2),
                     metric_2(energy_total10_1, atoms_h2)])
data_y_2 = np.array([metric_2(energy_total7_2, atoms_h2),
                     metric_2(energy_total8_2, atoms_h2),
                     metric_2(energy_total9_2, atoms_h2),
                     metric_2(energy_total10_2, atoms_h2)])
# data_y_ru_blyp = np.array([metric_2(energy_total2_8[:end_ru], atoms_ru)])
# data_y_ru_b3lyp = np.array([metric_2(energy_total2_9[:end_ru], atoms_ru)])
# data_y_ru_b97x = np.array([metric_2(energy_total2_10[:end_ru], atoms_ru)])
data_y_mgo_pbe = np.array([metric_2(energy_total7_4, atoms_mgo, interval_mgo), metric_2(energy_total8_4, atoms_mgo, interval_mgo)])
data_y_mgo_pbe0 = np.array([metric_2(energy_total7_5[offset_mgo:], atoms_mgo, interval_mgo), metric_2(energy_total8_5[offset_mgo:], atoms_mgo, interval_mgo)])
data_y_lepid = np.array([metric_2(energy_total13_6[offset:end_short], atoms_lepidocrocite), metric_2(energy_total14_6[offset:end_short], atoms_lepidocrocite)])
data_y_hem = np.array([metric_2(energy_total13_12[:end_short], atoms_hematite_221, interval_mgo), metric_2(energy_total14_12[:end_short], atoms_hematite_221, interval_mgo)])
data_y_bivo = np.array([metric_2(energy_total9_7[:end_short], atoms_bivo, interval_mgo), metric_2(energy_total10_7[:], atoms_bivo, interval_mgo)])
ax_drift2.plot(data_x_1, data_y_1, '+-', color='black', label=r'H$_{2}$ PBE')
ax_drift2.plot(data_x_1, data_y_2, '+-', color='grey', label=r'H$_{2}$ PBE PBC')
ax_drift2.plot(5e-4, 4.5e-5, 'rx', label=r'Ru$^{2+}$- Ru$^{3+}$ BLYP')
ax_drift2.plot(5e-4, 2.6e-5, 'gx', label=r'Ru$^{2+}$- Ru$^{3+}$ B3LYP')
ax_drift2.plot(5e-4, 2.5e-5, 'bx', label=r'Ru$^{2+}$- Ru$^{3+}$ ωB97X')
# ax_drift2.plot(5e-4, data_y_ru_blyp, 'rx', label=r'Ru$^{2+}$- Ru$^{3+}$ BLYP')
# ax_drift2.plot(5e-4, data_y_ru_b3lyp, 'gx', label=r'Ru$^{2+}$- Ru$^{3+}$ B3LYP')
# ax_drift2.plot(5e-4, data_y_ru_b97x, 'bx', label=r'Ru$^{2+}$- Ru$^{3+}$ ωB97X')
ax_drift2.plot([1e-2, 1e-3], data_y_mgo_pbe, 'o-', color='orange', fillstyle='none', label='MgO PBE')
ax_drift2.plot([1e-2, 1e-3], data_y_mgo_pbe0, 'o-', color='tan', fillstyle='none', label='MgO PBE0')
ax_drift2.plot([1e-2, 1e-3], data_y_bivo, 'D-', color='y', fillstyle='full', label=r'BiVO$_{4}$ HSE(25%)')
ax_drift2.plot([1e-2, 1e-3], data_y_lepid, '^-', color='c', fillstyle='full', label='Lepidocrocite HSE(18%)')
ax_drift2.plot([1e-2, 1e-3], data_y_hem, 'P-', color='m', fillstyle='full', label='Hematite HSE(12%)')
ax_drift2.set_yscale('log')
ax_drift2.set_xscale('log')
ax_drift2.set_xlabel('Constraint convergence [e]')
ax_drift2.set_ylabel('Energy drift [H/atom/ps]')
ax_drift2.set_ylim([0.9e-7, 1.5e-4])
ax_drift2.legend(frameon=False)
fig_drift2.tight_layout()
fig_drift2.savefig('{}/energy_drift2_{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot energy drift DFT
print('Plot energy drift DFT')
fig_drift_dft, ax_drift_dft = plt.subplots()
data_x_1 = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
data_y_1 = np.array([metric_2(energy_total12_3, atoms_h2), metric_2(energy_total6_3, atoms_h2),
                     metric_2(energy_total7_3, atoms_h2), metric_2(energy_total9_3, atoms_h2),
                     metric_2(energy_total8_3, atoms_h2), metric_2(energy_total10_3, atoms_h2),
                     metric_2(energy_total11_3, atoms_h2)])
ax_drift_dft.plot(data_x_1, data_y_1, '+-', color='black')
ax_drift_dft.legend(frameon=False)
ax_drift_dft.set_yscale('log')
ax_drift_dft.set_xscale('log')
ax_drift_dft.set_xlabel('Energy gradient [H]')
ax_drift_dft.set_ylabel('Energy drift [H/atom/ps]')
fig_drift_dft.tight_layout()
fig_drift_dft.savefig('{}/energy_drift_dft_{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot time taken 1
fig_time, ax_time = plt.subplots()
data_x_1 = [1e-5, 1e-6]
data_y_1 = np.array([metric_time(time_per_step9_1, time_per_step12_1),
                     metric_time(time_per_step10_1, time_per_step12_1)])
data_y_2 = np.array([metric_time(time_per_step9_2, time_per_step12_2),
                     metric_time(time_per_step10_2, time_per_step12_2)])
data_y_mgo_pbe = np.array([metric_time(time_per_step7_4, time_per_step6_4),
                          metric_time(time_per_step8_4, time_per_step6_4)])
data_y_mgo_pbe0 = np.array([metric_time(time_per_step7_5, time_per_step6_5),
                            metric_time(time_per_step8_5, time_per_step6_5)])
data_y_ru_blyp = np.array([metric_time(time_per_step2_8, time_per_step1_8)])
data_y_ru_b3lyp = np.array([metric_time(time_per_step2_9, time_per_step1_9)])
data_y_ru_b97x = np.array([metric_time(time_per_step2_10, time_per_step1_10)])
ax_time.plot(data_x_1, data_y_1, '+-', color='black', label=r'H$_{2}$ PBE')
ax_time.plot(data_x_1, data_y_2, '+-', color='grey', label=r'H$_{2}$ PBE PBC')
ax_time.plot([1e-2, 1e-3], data_y_mgo_pbe, 'o-', color='orange', fillstyle='none', label='MgO PBE')
ax_time.plot([1e-2, 1e-3], data_y_mgo_pbe0, 'o-', color='tan', fillstyle='none', label='MgO PBE0')
# ax_time.plot([5e-4], data_y_ru_blyp, 'rx', label=r'Ru$^{2+}$- Ru$^{3+}$ BLYP')
ax_time.plot([5e-4], data_y_ru_b3lyp, 'gx', label=r'Ru$^{2+}$- Ru$^{3+}$ B3LYP')
ax_time.plot([5e-4], data_y_ru_b97x, 'bx', label=r'Ru$^{2+}$- Ru$^{3+}$ ωB97X')
ax_time.set_xscale('log')
ax_time.set_xlabel('Constraint convergence [e]')
ax_time.set_ylabel('Relative time taken per MD step')
ax_time.set_ylim([2, 5])
ax_time.legend(frameon=False)
fig_time.tight_layout()
fig_time.savefig('{}/time_{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot time taken 2
print('Plot time taken 2')
fig_time2, ax_time2 = plt.subplots()
data_x_1 = [1e-5, 1e-6]
data_y_1 = np.array([metric_time(time_per_step9_1, time_per_step12_1),
                     metric_time(time_per_step10_1, time_per_step12_1)])
data_y_2 = np.array([metric_time(time_per_step9_2, time_per_step12_2),
                     metric_time(time_per_step10_2, time_per_step12_2)])
data_y_mgo_pbe = np.array([metric_time(time_per_step7_4, time_per_step6_4),
                          metric_time(time_per_step8_4, time_per_step6_4)])
data_y_mgo_pbe0 = np.array([metric_time(time_per_step7_5, time_per_step6_5),
                            metric_time(time_per_step8_5, time_per_step6_5)])
data_y_lepid = np.array([metric_time(time_per_step13_6, time_per_step6_6), metric_time(time_per_step14_6, time_per_step6_6)])
data_y_hem = np.array([metric_time(time_per_step13_12, time_per_step6_12), metric_time(time_per_step14_12, time_per_step6_12)])
data_y_bivo = np.array([metric_time(time_per_step9_7, time_per_step6_7), metric_time(time_per_step10_7, time_per_step6_7)])
data_y_ru_blyp = np.array([metric_time(time_per_step2_8, time_per_step1_8)])
data_y_ru_b3lyp = np.array([metric_time(time_per_step2_9, time_per_step1_9)])
data_y_ru_b97x = np.array([metric_time(time_per_step2_10, time_per_step1_10)])
ax_time2.plot(data_x_1, data_y_1, '+-', color='black', label=r'H$_{2}$ PBE')
ax_time2.plot(data_x_1, data_y_2, '+-', color='grey', label=r'H$_{2}$ PBE PBC')
ax_time2.plot([5e-4], data_y_ru_blyp, 'rx', label=r'Ru$^{2+}$- Ru$^{3+}$ BLYP')
ax_time2.plot([5e-4], data_y_ru_b3lyp, 'gx', label=r'Ru$^{2+}$- Ru$^{3+}$ B3LYP')
ax_time2.plot([5e-4], data_y_ru_b97x, 'bx', label=r'Ru$^{2+}$- Ru$^{3+}$ ωB97X')
ax_time2.plot([1e-2, 1e-3], data_y_mgo_pbe, 'o-', color='orange', fillstyle='none', label='MgO PBE')
ax_time2.plot([1e-2, 1e-3], data_y_mgo_pbe0, 'o-', color='tan', fillstyle='none', label='MgO PBE0')
ax_time2.plot([1e-2, 1e-3], data_y_bivo, 'D-', color='y', fillstyle='full', label=r'BiVO$_{4}$ HSE(25%)')
ax_time2.plot([1e-2, 1e-3], data_y_lepid, '^-', color='c', fillstyle='full', label='Lepidocrocite HSE(18%)')
ax_time2.plot([1e-2, 1e-3], data_y_hem, 'P-', color='m', fillstyle='full', label='Hematite HSE(12%)')
ax_time2.set_xscale('log')
ax_time2.set_xlabel('Constraint convergence [e]')
ax_time2.set_ylabel('Relative time taken per MD step')
ax_time2.set_ylim([1, 5])
ax_time2.set_ylim([1, 12])
ax_time2.legend(frameon=False)
fig_time2.tight_layout()
fig_time2.savefig('{}/time2_{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
