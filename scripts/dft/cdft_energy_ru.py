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
    Plot energy for ru-ru benchmark 
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
atoms = 191
index_ru1 = np.array([1]) - 1
index_h2o1 = np.array([15, 16, 17, 18, 19, 20, 9, 10, 11, 3, 4, 5, 6, 7, 8, 12, 13, 14]) - 1
index_ru2 = np.array([2]) - 1
index_h2o2 = np.array([24, 25, 26, 21, 22, 23, 36, 37, 38, 33, 34, 35, 27, 28, 29, 30, 31, 32]) - 1
folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/pbe'

folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/pbe/dft/analysis'
energy_kinetic1_1, energy_potential1_1, energy_total1_1, temperature1_1, time_val1_1, time_per_step1_1 = load_energy.load_values_energy(folder_1, '/energy/scf-1e-5.out')
energy_kinetic2_1, energy_potential2_1, energy_total2_1, temperature2_1, time_val2_1, time_per_step2_1 = load_energy.load_values_energy(folder_1, '/energy/scf-1e-6.out')
energy_kinetic3_1, energy_potential3_1, energy_total3_1, temperature3_1, time_val3_1, time_per_step3_1 = load_energy.load_values_energy(folder_1, '/energy/scf-1e-5-inverse.out')

folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/pbe/cdft/analysis'
energy_kinetic1_2, energy_potential1_2, energy_total1_2, temperature1_2, time_val1_2, time_per_step1_2 = load_energy.load_values_energy(folder_2, 'energy/constraint-diff-ru2-water.out')
energy_kinetic2_2, energy_potential2_2, energy_total2_2, temperature2_2, time_val2_2, time_per_step2_2 = load_energy.load_values_energy(folder_2, 'energy/constraint-abs-ru-water_scf-1e-5-inverse.out')
energy_kinetic3_2, energy_potential3_2, energy_total3_2, temperature3_2, time_val3_2, time_per_step3_2 = load_energy.load_values_energy(folder_2, 'energy/constraint-diff-ru2_scf-1e-5-inverse.out')
energy_kinetic4_2, energy_potential4_2, energy_total4_2, temperature4_2, time_val4_2, time_per_step4_2 = load_energy.load_values_energy(folder_2, 'energy/constraint-abs-ru_scf-1e-5-inverse.out')

folder_3 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/pbe/dft-24h/analysis'
energy_kinetic1_3, energy_potential1_3, energy_total1_3, temperature1_3, time_val1_3, time_per_step1_3 = load_energy.load_values_energy(folder_3, '/energy/scf-1e-5.out')
energy_kinetic2_3, energy_potential2_3, energy_total2_3, temperature2_3, time_val2_3, time_per_step2_3 = load_energy.load_values_energy(folder_3, '/energy/scf-1e-6.out')
energy_kinetic3_3, energy_potential3_3, energy_total3_3, temperature3_3, time_val3_3, time_per_step3_3 = load_energy.load_values_energy(folder_3, '/energy/scf-1e-5-inverse.out')
iasd1_3 = np.loadtxt('{}/iasd/scf-1e-5.out'.format(folder_3))
iasd2_3 = np.loadtxt('{}/iasd/scf-1e-6.out'.format(folder_3))
iasd3_3 = np.loadtxt('{}/iasd/scf-1e-5-inverse.out'.format(folder_3))
file_spec1_3, species1_3, num_data1_3, step1_3, brent1_3, mnbrack1_3 = read_hirsh(folder_3, '/hirshfeld/scf-1e-5.out', atoms, None, None)
file_spec2_3, species2_3, num_data2_3, step2_3, brent2_3, mnbrack2_3 = read_hirsh(folder_3, '/hirshfeld/scf-1e-5.out', atoms, None, None)
file_spec3_3, species3_3, num_data3_3, step3_3, brent3_3, mnbrack3_3 = read_hirsh(folder_3, '/hirshfeld/scf-1e-5.out', atoms, None, None)

folder_4 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/pbe/cdft-24/eps-1e-3/scf-1e-5-inverse/analysis'
energy_kinetic1_4, energy_potential1_4, energy_total1_4, temperature1_4, time_val1_4, time_per_step1_4 = load_energy.load_values_energy(folder_4, 'energy/abs-ru.out')
energy_kinetic2_4, energy_potential2_4, energy_total2_4, temperature2_4, time_val2_4, time_per_step2_4 = load_energy.load_values_energy(folder_4, 'energy/abs-ru-water.out')
energy_kinetic3_4, energy_potential3_4, energy_total3_4, temperature3_4, time_val3_4, time_per_step3_4 = load_energy.load_values_energy(folder_4, 'energy/rel-ru.out')
energy_kinetic4_4, energy_potential4_4, energy_total4_4, temperature4_4, time_val4_4, time_per_step4_4 = load_energy.load_values_energy(folder_4, 'energy/rel-ru-water.out')
iasd1_4 = np.loadtxt('{}/iasd/abs-ru.out'.format(folder_4))
iasd2_4 = np.loadtxt('{}/iasd/abs-ru-water.out'.format(folder_4))
iasd3_4 = np.loadtxt('{}/iasd/rel-ru.out'.format(folder_4))
iasd4_4 = np.loadtxt('{}/iasd/rel-ru-water.out'.format(folder_4))
file_spec1_4, species1_4, num_data1_4, step1_4, brent1_4, mnbrack1_4 = read_hirsh(folder_4, '/hirshfeld/abs-ru.out', atoms, None, None)
file_spec2_4, species2_4, num_data2_4, step2_4, brent2_4, mnbrack2_4 = read_hirsh(folder_4, '/hirshfeld/abs-ru-water.out', atoms, None, None)
file_spec3_4, species3_4, num_data3_4, step3_4, brent3_4, mnbrack3_4 = read_hirsh(folder_4, '/hirshfeld/rel-ru.out', atoms, None, None)
file_spec4_4, species4_4, num_data4_4, step4_4, brent4_4, mnbrack4_4 = read_hirsh(folder_4, '/hirshfeld/rel-ru-water.out', atoms, None, None)
force1_4, forces_x1_4, forces_y1_4, forces_z1_4, num_atoms1_4, num_timesteps1_4 = load_forces.load_values_forces(folder_4, 'force/abs-ru.out')
force2_4, forces_x2_4, forces_y2_4, forces_z2_4, num_atoms2_4, num_timesteps2_4 = load_forces.load_values_forces(folder_4, 'force/abs-ru-water.out')
force3_4, forces_x3_4, forces_y3_4, forces_z3_4, num_atoms3_4, num_timesteps3_4 = load_forces.load_values_forces(folder_4, 'force/rel-ru.out')
force4_4, forces_x4_4, forces_y4_4, forces_z4_4, num_atoms4_4, num_timesteps4_4 = load_forces.load_values_forces(folder_4, 'force/rel-ru-water.out')

folder_5 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/pbe/cdft-24/eps-1e-4/scf-1e-5-inverse/analysis'
energy_kinetic1_5, energy_potential1_5, energy_total1_5, temperature1_5, time_val1_5, time_per_step1_5 = load_energy.load_values_energy(folder_5, 'energy/abs-ru.out')
energy_kinetic2_5, energy_potential2_5, energy_total2_5, temperature2_5, time_val2_5, time_per_step2_5 = load_energy.load_values_energy(folder_5, 'energy/abs-ru-water.out')
energy_kinetic3_5, energy_potential3_5, energy_total3_5, temperature3_5, time_val3_5, time_per_step3_5 = load_energy.load_values_energy(folder_5, 'energy/rel-ru.out')
energy_kinetic4_5, energy_potential4_5, energy_total4_5, temperature4_5, time_val4_5, time_per_step4_5 = load_energy.load_values_energy(folder_5, 'energy/rel-ru-water.out')
iasd1_5 = np.loadtxt('{}/iasd/abs-ru.out'.format(folder_5))
iasd2_5 = np.loadtxt('{}/iasd/abs-ru-water.out'.format(folder_5))
iasd3_5 = np.loadtxt('{}/iasd/rel-ru.out'.format(folder_5))
iasd4_5 = np.loadtxt('{}/iasd/rel-ru-water.out'.format(folder_5))

folder_6 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/pbe/cdft-24/eps-1e-3/scf-1e-6-inverse/analysis'
energy_kinetic1_6, energy_potential1_6, energy_total1_6, temperature1_6, time_val1_6, time_per_step1_6 = load_energy.load_values_energy(folder_6, 'energy/abs-ru.out')
energy_kinetic2_6, energy_potential2_6, energy_total2_6, temperature2_6, time_val2_6, time_per_step2_6 = load_energy.load_values_energy(folder_6, 'energy/abs-ru-water.out')
energy_kinetic3_6, energy_potential3_6, energy_total3_6, temperature3_6, time_val3_6, time_per_step3_6 = load_energy.load_values_energy(folder_6, 'energy/rel-ru.out')
energy_kinetic4_6, energy_potential4_6, energy_total4_6, temperature4_6, time_val4_6, time_per_step4_6 = load_energy.load_values_energy(folder_6, 'energy/rel-ru-water.out')
iasd1_6 = np.loadtxt('{}/iasd/abs-ru.out'.format(folder_6))
iasd2_6 = np.loadtxt('{}/iasd/abs-ru-water.out'.format(folder_6))
iasd3_6 = np.loadtxt('{}/iasd/rel-ru.out'.format(folder_6))
iasd4_6 = np.loadtxt('{}/iasd/rel-ru-water.out'.format(folder_6))

# Plot total energy CDFT
time_plot = 1000
skip = 0
energy_end = time_plot*2
fig_energy3, ax_energy3 = plt.subplots()
ax_energy3.plot(time_val3_4[skip:]-time_val3_4[skip], (energy_total3_4[skip:]-energy_total3_4[skip])/atoms, 'r-', label='CDFT REL Ru CDFT 1e-3, DFT 1e-5')
ax_energy3.plot(time_val3_5[skip:]-time_val3_5[skip], (energy_total3_5[skip:]-energy_total3_5[skip])/atoms, 'b-', label='CDFT REL Ru CDFT 1e-4, DFT 1e-5')
ax_energy3.plot(time_val3_6[skip:]-time_val3_6[skip], (energy_total3_6[skip:]-energy_total3_6[skip])/atoms, 'g-', label='CDFT REL Ru CDFT 1e-3, DFT 1e-6')
ax_energy3.set_xlabel('Time / fs')
ax_energy3.set_ylabel('Energy change per atom / Ha')
ax_energy3.set_xlim([0, time_plot])
ax_energy3.set_ylim([-1.5e-5, 1.5e-5])
ax_energy3.legend(frameon=False)
fig_energy3.tight_layout()
# fig_energy3.savefig('{}/energy_cdft_scf-1e-6_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot total energy DFT
time_plot = 200
skip = 30
energy_end = time_plot * 2
fig_energy_dft, ax_energy_dft = plt.subplots()
ax_energy_dft.plot(time_val1_3[skip:] - time_val1_3[skip], (energy_total1_3[skip:] - energy_total1_3[skip]) / atoms, 'k-', label='DFT SCF 1e-5')
ax_energy_dft.plot(time_val2_3[skip:]-time_val2_3[skip], (energy_total2_3[skip:]-energy_total2_3[skip])/atoms, '-', color='grey', label='DFT SCF 1e-6')
ax_energy_dft.set_xlabel('Time / fs')
ax_energy_dft.set_ylabel('Energy change per atom / Ha')
ax_energy_dft.set_xlim([0, time_plot])
ax_energy_dft.set_ylim([-1e-5, 3e-5])
ax_energy_dft.legend(frameon=False)
fig_energy_dft.tight_layout()
# fig_energy_dft.savefig('{}/energy_dft_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot total energy CDFT (DFT SCF 1e-5)
# time_plot = 200
skip = 30
energy_end = time_plot*2
fig_energy, ax_energy = plt.subplots()
# ax_energy.plot(time_val1_3[skip:]-time_val1_3[skip], (energy_total1_3[skip:]-energy_total1_3[skip])/atoms, 'k-', label='DFT SCF 1e-5')
ax_energy.plot(time_val1_3[skip:]-time_val1_3[skip], (energy_total1_3[skip:]-energy_total1_3[skip])/atoms, 'k-', label='DFT')
ax_energy.plot(time_val1_4[skip:]-time_val1_4[skip], (energy_total1_4[skip:]-energy_total1_4[skip])/atoms, 'r-', label='CDFT ABS Ru')
ax_energy.plot(time_val2_4[skip:]-time_val2_4[skip], (energy_total2_4[skip:]-energy_total2_4[skip])/atoms, 'b-', label='CDFT ABS Ru, H2O')
ax_energy.plot(time_val3_4[skip:]-time_val3_4[skip], (energy_total3_4[skip:]-energy_total3_4[skip])/atoms, 'g-', label='CDFT REL Ru')
ax_energy.plot(time_val4_4[skip:]-time_val4_4[skip], (energy_total4_4[skip:]-energy_total4_4[skip])/atoms, 'm-', label='CDFT REL Ru, H2O')
# ax_energy.plot(time_val1_5[skip:]-time_val1_5[skip], (energy_total1_5[skip:]-energy_total1_5[skip])/atoms, 'r-', label='CDFT ABS Ru')
# ax_energy.plot(time_val2_5[skip:]-time_val2_5[skip], (energy_total2_5[skip:]-energy_total2_5[skip])/atoms, 'b-', label='CDFT ABS Ru, H2O')
# ax_energy.plot(time_val3_5[skip:]-time_val3_5[skip], (energy_total3_5[skip:]-energy_total3_5[skip])/atoms, 'g-', label='CDFT REL Ru')
# ax_energy.plot(time_val4_5[skip:]-time_val4_5[skip], (energy_total4_5[skip:]-energy_total4_5[skip])/atoms, 'm-', label='CDFT REL Ru, H2O')
ax_energy.set_xlabel('Time / fs')
ax_energy.set_ylabel('Energy change per atom / Ha')
ax_energy.set_xlim([0, time_plot])
ax_energy.set_ylim([-1e-5, 3e-5])
ax_energy.set_xlim([0, 300])
ax_energy.set_ylim([-1.5e-5, 2e-4])
ax_energy.legend(frameon=False)
fig_energy.tight_layout()
fig_energy.savefig('{}/energy_cdft_scf-1e-5_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot total energy CDFT (DFT SCF 1e-6)
# time_plot = 200
skip = 30
energy_end = time_plot*2
fig_energy2, ax_energy2 = plt.subplots()
ax_energy2.plot(time_val2_3[skip:]-time_val2_3[skip], (energy_total2_3[skip:]-energy_total2_3[skip])/atoms, '-', color='grey', label='DFT SCF 1e-6')
ax_energy2.plot(time_val3_6[skip:]-time_val3_6[skip], (energy_total3_6[skip:]-energy_total3_6[skip])/atoms, 'g-', label='CDFT REL Ru')
ax_energy2.plot(time_val4_6[skip:]-time_val4_6[skip], (energy_total4_6[skip:]-energy_total4_6[skip])/atoms, 'm-', label='CDFT REL Ru, H2O')
ax_energy2.set_xlabel('Time / fs')
ax_energy2.set_ylabel('Energy change per atom / Ha')
ax_energy2.set_xlim([0, time_plot])
ax_energy2.set_ylim([-0.5e-5, 2.2e-5])
ax_energy2.legend(frameon=False)
fig_energy2.tight_layout()
fig_energy2.savefig('{}/energy_cdft_scf-1e-6_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot change in energy difference to DFT
# time_plot = 200
skip = 30
energy_end = time_plot*2
fig_energy3, ax_energy3 = plt.subplots()
# ax_energy3.plot(time_val1_3[skip:]-time_val1_3[skip], (energy_total1_3[skip:]-energy_total1_3[skip])/atoms, 'k-', label='DFT SCF 1e-5')
ax_energy3.plot(time_val1_3[skip:]-time_val1_3[skip], energy_total1_3[skip:]-energy_total1_3[skip], 'k-', label='DFT')
ax_energy3.plot(time_val1_4[skip:]-time_val1_4[skip], energy_total1_4[skip:]-energy_total1_3[skip], 'r-', label='CDFT ABS Ru')
ax_energy3.plot(time_val2_4[skip:]-time_val2_4[skip], energy_total2_4[skip:]-energy_total1_3[skip], 'b-', label='CDFT ABS Ru, H2O')
ax_energy3.plot(time_val3_4[skip:]-time_val3_4[skip], energy_total3_4[skip:]-energy_total1_3[skip], 'g-', label='CDFT REL Ru')
ax_energy3.plot(time_val4_4[skip:]-time_val4_4[skip], energy_total4_4[skip:]-energy_total1_3[skip], 'm-', label='CDFT REL Ru, H2O')
# ax_energy3.plot(time_val1_5[skip:]-time_val1_5[skip], (energy_total1_5[skip:]-energy_total1_5[skip])/atoms, 'r-', label='CDFT ABS Ru')
# ax_energy3.plot(time_val2_5[skip:]-time_val2_5[skip], (energy_total2_5[skip:]-energy_total2_5[skip])/atoms, 'b-', label='CDFT ABS Ru, H2O')
# ax_energy3.plot(time_val3_5[skip:]-time_val3_5[skip], (energy_total3_5[skip:]-energy_total3_5[skip])/atoms, 'g-', label='CDFT REL Ru')
# ax_energy3.plot(time_val4_5[skip:]-time_val4_5[skip], (energy_total4_5[skip:]-energy_total4_5[skip])/atoms, 'm-', label='CDFT REL Ru, H2O')
ax_energy3.set_xlabel('Time / fs')
ax_energy3.set_ylabel('Energy difference to DFT / Ha')
ax_energy3.set_xlim([0, time_plot])
# ax_energy3.set_ylim([-1e-5, 3e-5])
ax_energy3.set_xlim([0, 300])
ax_energy3.set_ylim([0, 0.15])
ax_energy3.legend(frameon=False)
fig_energy3.tight_layout()
fig_energy3.savefig('{}/energy_change_dft_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot temperature
skip = 0
energy_end = time_plot*2
fig_temperature, ax_temperature = plt.subplots()
# ax_temperature.plot(time_val1_3[skip:]-time_val1_3[skip], (temperature1_3[skip:]-temperature1_3[skip])/atoms, 'k-', label='DFT SCF 1e-5')
ax_temperature.plot(time_val1_3[skip:]-time_val1_3[skip], temperature1_3[skip:], 'k-', label='DFT')
ax_temperature.plot(time_val1_4[skip:]-time_val1_4[skip], temperature1_4[skip:], 'r-', label='CDFT ABS Ru')
ax_temperature.plot(time_val2_4[skip:]-time_val2_4[skip], temperature2_4[skip:], 'b-', label='CDFT ABS Ru, H2O')
ax_temperature.plot(time_val3_4[skip:]-time_val3_4[skip], temperature3_4[skip:], 'g-', label='CDFT REL Ru')
ax_temperature.plot(time_val4_4[skip:]-time_val4_4[skip], temperature4_4[skip:], 'm-', label='CDFT REL Ru, H2O')
ax_temperature.set_xlabel('Time / fs')
ax_temperature.set_ylabel('Temperature / K')
ax_temperature.set_xlim([0, 300])
ax_temperature.set_ylim([350, 520])
ax_temperature.legend(frameon=False)
fig_temperature.tight_layout()
fig_temperature.savefig('{}/temperature_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot total iasd
# time_plot = 250
skip = 30
iasd_end = time_plot*2
plot_end = -1
fig_iasd, ax_iasd = plt.subplots()
ax_iasd.plot(time_val1_3[skip:]-time_val1_3[skip], iasd1_3[skip:], 'k-', label='DFT SCF 1e-5')
# ax_iasd.plot(time_val2_3[skip:]-time_val2_3[skip], iasd2_3[skip:], '-', color='grey', label='DFT SCF 1e-6')
ax_iasd.plot(time_val1_4[skip:]-time_val1_4[skip], iasd1_4[skip:], 'r-', alpha=0.5)
# ax_iasd.plot(time_val2_4[skip:]-time_val2_4[skip], iasd2_4[skip:], 'b-', alpha=0.5)
ax_iasd.plot(time_val3_4[skip:]-time_val3_4[skip], iasd3_4[skip:], 'g-', alpha=0.5)
ax_iasd.plot(time_val4_4[skip:]-time_val4_4[skip], iasd4_4[skip:], 'm-', alpha=0.5)
ax_iasd.plot(time_val1_5[skip:]-time_val1_5[skip], iasd1_5[skip:], 'r-', label='CDFT ABS Ru')
ax_iasd.plot(time_val2_5[skip:]-time_val2_5[skip], iasd2_5[skip:], 'b-', label='CDFT ABS Ru, H2O')
ax_iasd.plot(time_val3_5[skip:]-time_val3_5[skip], iasd3_5[skip:], 'g-', label='CDFT REL Ru')
ax_iasd.plot(time_val4_5[skip:]-time_val4_5[skip], iasd4_5[skip:-1], 'm-', label='CDFT REL Ru, H2O')
ax_iasd.set_xlabel('Time / fs')
ax_iasd.set_ylabel('IASD')
ax_iasd.set_xlim([0, time_plot])
ax_iasd.set_ylim([1.065, 1.14])
ax_iasd.legend(frameon=False)
fig_iasd.tight_layout()
fig_iasd.savefig('{}/iasd_{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot Hirshfeld analysis (Ru)
# time_plot = 250
skip = 30
skip_line = 2
plot_index_ru = index_ru1
fig_hirshfeld, ax_hirshfeld = plt.subplots()
temp1 = np.zeros(num_data1_3)
i = -1
for n in range(num_data1_3):
    i = i + 1
    temp1[n] = (file_spec1_3.loc[atoms * i + skip_line * i + plot_index_ru[0], 'Charge'])
ax_hirshfeld.plot(time_val1_3[skip:]-time_val1_3[skip], temp1[skip:], 'k-', label='DFT SCF 1e-5: Ru1')
temp1 = np.zeros(num_data1_4)
i = -1
for n in range(num_data1_4):
    i = i + 1
    temp1[n] = (file_spec1_4.loc[atoms * i + skip_line * i + plot_index_ru[0], 'Charge'])
ax_hirshfeld.plot(time_val1_4[skip:] - time_val1_4[skip], temp1[skip:], 'r-', label='CDFT ABS Ru')
temp1 = np.zeros(num_data2_4)
i = -1
for n in range(num_data2_4):
    i = i + 1
    temp1[n] = (file_spec2_4.loc[atoms * i + skip_line * i + plot_index_ru[0], 'Charge'])
ax_hirshfeld.plot(time_val2_4[skip:] - time_val2_4[skip], temp1[skip:], 'b-', label='CDFT ABS Ru, H2O')
temp1 = np.zeros(num_data3_4)
i = -1
for n in range(num_data3_4):
    i = i + 1
    temp1[n] = (file_spec3_4.loc[atoms * i + skip_line * i + plot_index_ru[0], 'Charge'])
ax_hirshfeld.plot(time_val3_4[skip:] - time_val3_4[skip], temp1[skip:], 'g-', label='CDFT REL Ru')
temp1 = np.zeros(num_data4_4)
i = -1
for n in range(num_data4_4):
    i = i + 1
    temp1[n] = (file_spec4_4.loc[atoms * i + skip_line * i + plot_index_ru[0], 'Charge'])
ax_hirshfeld.plot(time_val4_4[skip:] - time_val4_4[skip], temp1[skip:], 'm-', label='CDFT REL Ru, H2O')
ax_hirshfeld.set_xlabel('Time / fs')
ax_hirshfeld.set_ylabel('Ru Hirshfeld charge')
ax_hirshfeld.set_xlim([0, time_plot])
ax_hirshfeld.set_ylim([-0.05, 1.05])
ax_hirshfeld.legend(frameon=False)
fig_hirshfeld.tight_layout()
fig_hirshfeld.savefig('{}/charge_ru2_{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot Hirshfeld analysis (H2O)
# time_plot = 250
skip = 30
skip_line = 2
plot_index = index_h2o1
# plot_index = np.concatenate([index_ru1,index_h2o1])
fig_hirshfeld2, ax_hirshfeld2 = plt.subplots()
temp1 = np.zeros(num_data1_3)
i = -1
for n in range(num_data1_3):
    i = i + 1
    temp1[n] = temp1[n] + (file_spec1_3.loc[atoms * i + skip_line * i + plot_index_ru[0], 'Charge'])
    for j in range(len(index_h2o1)):
        temp1[n] = temp1[n] + (file_spec1_3.loc[atoms * i + skip_line * i + plot_index[j], 'Charge'])
print(np.mean(temp1[skip:time_plot]))
ax_hirshfeld2.plot(time_val1_3[skip:]-time_val1_3[skip], temp1[skip:], 'k-', label='DFT SCF 1e-5')
temp1 = np.zeros(num_data1_4)
i = -1
for n in range(num_data1_4):
    i = i + 1
    temp1[n] = temp1[n] + (file_spec1_4.loc[atoms * i + skip_line * i + plot_index_ru[0], 'Charge'])
    for j in range(len(index_h2o1)):
        temp1[n] = temp1[n] + (file_spec1_4.loc[atoms * i + skip_line * i + plot_index[j], 'Charge'])
print(np.mean(temp1[skip:time_plot]))
ax_hirshfeld2.plot(time_val1_4[skip:] - time_val1_4[skip], temp1[skip:], 'r-', label='CDFT ABS Ru')
temp1 = np.zeros(num_data2_4)
i = -1
for n in range(num_data2_4):
    i = i + 1
    temp1[n] = temp1[n] + (file_spec2_4.loc[atoms * i + skip_line * i + plot_index_ru[0], 'Charge'])
    for j in range(len(index_h2o1)):
        temp1[n] = temp1[n] + (file_spec2_4.loc[atoms * i + skip_line * i + plot_index[j], 'Charge'])
print(np.mean(temp1[skip:time_plot]))
ax_hirshfeld2.plot(time_val2_4[skip:] - time_val2_4[skip], temp1[skip:], 'b-', label='CDFT ABS Ru, H2O')
temp1 = np.zeros(num_data3_4)
i = -1
for n in range(num_data3_4):
    i = i + 1
    temp1[n] = temp1[n] + (file_spec3_4.loc[atoms * i + skip_line * i + plot_index_ru[0], 'Charge'])
    for j in range(len(index_h2o1)):
        temp1[n] = temp1[n] + (file_spec3_4.loc[atoms * i + skip_line * i + plot_index[j], 'Charge'])
print(np.mean(temp1[skip:time_plot]))
ax_hirshfeld2.plot(time_val3_4[skip:] - time_val3_4[skip], temp1[skip:], 'g-', label='CDFT REL Ru')
temp1 = np.zeros(num_data4_4)
i = -1
for n in range(num_data4_4):
    i = i + 1
    temp1[n] = temp1[n] + (file_spec4_4.loc[atoms * i + skip_line * i + plot_index_ru[0], 'Charge'])
    for j in range(len(index_h2o1)):
        temp1[n] = temp1[n] + (file_spec4_4.loc[atoms * i + skip_line * i + plot_index[j], 'Charge'])
print(np.mean(temp1[skip:time_plot]))
ax_hirshfeld2.plot(time_val4_4[skip:] - time_val4_4[skip], temp1[skip:], 'm-', label='CDFT REL Ru, H2O')
ax_hirshfeld2.set_xlabel('Time / fs')
ax_hirshfeld2.set_ylabel('Ru, H2O Hirshfeld charge')
ax_hirshfeld2.set_xlim([0, time_plot])
# ax_hirshfeld2.set_ylim([0.7, 1.65])
# ax_hirshfeld2.set_ylim([0.3, 1.25])
ax_hirshfeld2.set_ylim([-0.05, 1.25])
ax_hirshfeld2.legend(frameon=False)
fig_hirshfeld2.tight_layout()
fig_hirshfeld2.savefig('{}/charge_water2_{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot forces 1
# skip = 0
time_plot = 1400
skip_line = 2
# plot_index = index_ru1
plot_index = index_h2o1[0]
fig_forces1, ax_forces1 = plt.subplots()
temp7 = np.zeros(num_data2_4)
temp8 = np.zeros(num_data2_4)
temp9 = np.zeros(num_data2_4)
i = -1
for n in range(num_data2_4):
    i = i + 1
    temp7[n] = np.sum(forces_x2_4[i, plot_index])
    temp8[n] = np.sum(forces_y2_4[i, plot_index])
    temp9[n] = np.sum(forces_z2_4[i, plot_index])
ax_forces1.plot(time_val2_4[skip:] - time_val2_4[skip], temp7[skip:], 'r--')
ax_forces1.plot(time_val2_4[skip:] - time_val2_4[skip], temp8[skip:], 'g--')
ax_forces1.plot(time_val2_4[skip:] - time_val2_4[skip], temp9[skip:], 'b--')
temp7 = np.zeros(num_data4_4)
temp8 = np.zeros(num_data4_4)
temp9 = np.zeros(num_data4_4)
i = -1
for n in range(num_data4_4):
    i = i + 1
    temp7[n] = np.sum(forces_x4_4[i, plot_index])
    temp8[n] = np.sum(forces_y4_4[i, plot_index])
    temp9[n] = np.sum(forces_z4_4[i, plot_index])
ax_forces1.plot(time_val4_4[skip:] - time_val4_4[skip], temp7[skip:], 'r-', label='x')
ax_forces1.plot(time_val4_4[skip:] - time_val4_4[skip], temp8[skip:], 'g-', label='y')
ax_forces1.plot(time_val4_4[skip:] - time_val4_4[skip], temp9[skip:], 'b-', label='z')
ax_forces1.set_xlabel('Time / fs')
ax_forces1.set_ylabel('Force / au')
# ax_forces1.set_xlim([0, time_plot])
ax_forces1.set_xlim([0, 100])
# ax_forces1.set_ylim([-0.075, 0.075])
ax_forces1.set_ylim([-0.09, 0.105])
ax_forces1.legend(frameon=False)
fig_forces1.tight_layout()
fig_forces1.savefig('{}/force_cdft_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
