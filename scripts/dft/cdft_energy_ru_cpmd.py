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

"""
    Load .ener
"""

skip = 2
atoms = 191
index_ru1 = np.array([1]) - 1
index_h2o1 = np.array([15, 16, 17, 18, 19, 20, 9, 10, 11, 3, 4, 5, 6, 7, 8, 12, 13, 14]) - 1
index_ru2 = np.array([2]) - 1
index_h2o2 = np.array([24, 25, 26, 21, 22, 23, 36, 37, 38, 33, 34, 35, 27, 28, 29, 30, 31, 32]) - 1
folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/blyp/equilibrated/'
# folder_save = '/home/chris/Storage/DATA/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/blyp/equilibrated'

folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/blyp/equilibrated/dft-24h-inverse/analysis'
# folder_2 = '/home/chris/Storage/DATA/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/blyp/equilibrated/dft-24h-inverse/analysis'
energy_kinetic1_2, energy_potential1_2, energy_total1_2, temperature1_2, time_val1_2, time_per_step1_2 = load_energy.load_values_energy(folder_2, '/energy/initial-timcon-33-rattle-cpmd.out')
energy_kinetic2_2, energy_potential2_2, energy_total2_2, temperature2_2, time_val2_2, time_per_step2_2 = load_energy.load_values_energy(folder_2, '/energy/initial-timcon-33-rattle-cpmd-tight.out')
file_spec1_2, species1_2, num_data1_2, step1_2, brent1_2, mnbrack1_2 = read_hirsh(folder_2, '/hirshfeld/initial-timcon-33-rattle-cpmd.out', atoms, None, None)
file_spec2_2, species2_2, num_data2_2, step2_2, brent2_2, mnbrack2_2 = read_hirsh(folder_2, '/hirshfeld/initial-timcon-33-rattle-cpmd-tight.out', atoms, None, None)

folder_3 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/blyp/equilibrated/cdft-24h-inverse/analysis'
# folder_3 = '/home/chris/Storage/DATA/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/blyp/equilibrated/cdft-24h-inverse/analysis'
energy_kinetic1_3, energy_potential1_3, energy_total1_3, temperature1_3, time_val1_3, time_per_step1_3 = load_energy.load_values_energy(folder_3, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-run-000.out')
energy_kinetic2_3, energy_potential2_3, energy_total2_3, temperature2_3, time_val2_3, time_per_step2_3 = load_energy.load_values_energy(folder_3, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-tight-run-000.out')
energy_kinetic21_3, energy_potential21_3, energy_total21_3, temperature21_3, time_val21_3, time_per_step21_3 = load_energy.load_values_energy(folder_3, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-tight-run-001.out')
energy_kinetic3_3, energy_potential3_3, energy_total3_3, temperature3_3, time_val3_3, time_per_step3_3 = load_energy.load_values_energy(folder_3, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out')
energy_kinetic31_3, energy_potential31_3, energy_total31_3, temperature31_3, time_val31_3, time_per_step31_3 = load_energy.load_values_energy(folder_3, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-run-001.out')
energy_kinetic32_3, energy_potential32_3, energy_total32_3, temperature32_3, time_val32_3, time_per_step32_3 = load_energy.load_values_energy(folder_3, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-run-002.out')
energy_kinetic5_3, energy_potential5_3, energy_total5_3, temperature5_3, time_val5_3, time_per_step5_3 = load_energy.load_values_energy(folder_3, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-tight-run-000.out')
energy_kinetic51_3, energy_potential51_3, energy_total51_3, temperature51_3, time_val51_3, time_per_step51_3 = load_energy.load_values_energy(folder_3, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-tight-run-001.out')
energy_kinetic52_3, energy_potential52_3, energy_total52_3, temperature52_3, time_val52_3, time_per_step52_3 = load_energy.load_values_energy(folder_3, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-tight-run-002.out')
energy_kinetic53_3, energy_potential53_3, energy_total53_3, temperature53_3, time_val53_3, time_per_step53_3 = load_energy.load_values_energy(folder_3, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-tight-run-003.out')
energy_kinetic6_3, energy_potential6_3, energy_total6_3, temperature6_3, time_val6_3, time_per_step6_3 = load_energy.load_values_energy(folder_3, '/energy/initial-timcon-33-rattle-cpmd-abs-ru-water-run-000.out')
energy_kinetic61_3, energy_potential61_3, energy_total61_3, temperature61_3, time_val61_3, time_per_step61_3 = load_energy.load_values_energy(folder_3, '/energy/initial-timcon-33-rattle-cpmd-abs-ru-water-run-001.out')
energy_kinetic7_3, energy_potential7_3, energy_total7_3, temperature7_3, time_val7_3, time_per_step7_3 = load_energy.load_values_energy(folder_3, '/energy/initial-timcon-33-rattle-cpmd-abs-ru-water-tight-run-000.out')
energy_kinetic71_3, energy_potential71_3, energy_total71_3, temperature71_3, time_val71_3, time_per_step71_3 = load_energy.load_values_energy(folder_3, '/energy/initial-timcon-33-rattle-cpmd-abs-ru-water-tight-run-001.out')
energy_kinetic72_3, energy_potential72_3, energy_total72_3, temperature72_3, time_val72_3, time_per_step72_3 = load_energy.load_values_energy(folder_3, '/energy/initial-timcon-33-rattle-cpmd-abs-ru-water-tight-run-002.out')
energy_kinetic8_3, energy_potential8_3, energy_total8_3, temperature8_3, time_val8_3, time_per_step8_3 = load_energy.load_values_energy(folder_3, '/energy/initial-timcon-33-rattle-cpmd-abs-ru-water-tight-run-002_dt-01.out')

file_spec1_3, species1_3, num_data1_3, step1_3, brent1_3, mnbrack1_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-run-000.out', atoms, None, None)
file_spec2_3, species2_3, num_data2_3, step2_3, brent2_3, mnbrack2_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-tight-run-000.out', atoms, None, None)
file_spec21_3, species21_3, num_data21_3, step21_3, brent21_3, mnbrack21_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-tight-run-001.out', atoms, None, None)
file_spec3_3, species3_3, num_data3_3, step3_3, brent3_3, mnbrack3_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out', atoms, None, None)
file_spec31_3, species31_3, num_data31_3, step31_3, brent31_3, mnbrack31_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-run-001.out', atoms, None, None)
file_spec5_3, species5_3, num_data5_3, step5_3, brent5_3, mnbrack5_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-tight-run-000.out', atoms, None, None)
file_spec51_3, species51_3, num_data51_3, step51_3, brent51_3, mnbrack51_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-tight-run-001.out', atoms, None, None)
file_spec6_3, species6_3, num_data6_3, step6_3, brent6_3, mnbrack6_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-abs-ru-water-run-000.out', atoms, None, None)
file_spec61_3, species61_3, num_data61_3, step61_3, brent61_3, mnbrack61_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-abs-ru-water-run-001.out', atoms, None, None)
file_spec7_3, species7_3, num_data7_3, step7_3, brent7_3, mnbrack7_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-abs-ru-water-tight-run-000.out', atoms, None, None)
file_spec71_3, species71_3, num_data71_3, step71_3, brent71_3, mnbrack71_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-abs-ru-water-tight-run-001.out', atoms, None, None)
file_spec72_3, species72_3, num_data72_3, step72_3, brent72_3, mnbrack72_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-abs-ru-water-tight-run-002.out', atoms, None, None)
file_spec8_3, species8_3, num_data8_3, step8_3, brent8_3, mnbrack8_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-abs-ru-water-tight-run-002_dt-01.out', atoms, None, None)

folder_4 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/blyp/lambda'
lambda1_4 = np.loadtxt('{}//initial-timcon-33-rattle-cpmd-rel-ru/analysis/energy.out'.format(folder_4))
lambda2_4 = np.loadtxt('{}//initial-timcon-33-rattle-cpmd-rel-ru-water/analysis/energy.out'.format(folder_4))
lambda3_4 = np.loadtxt('{}//initial-timcon-33-rattle-cpmd-rel-ru-water2/analysis/energy.out'.format(folder_4))

# Plot total energy DFT
cpmd_x = np.linspace(0, 2000, num=1000)
time_plot = 2000
skip = 0
energy_end = time_plot * 2
fig_energy_dft, ax_energy_dft = plt.subplots()
ax_energy_dft.plot(time_val1_2[skip:] - time_val1_2[skip], (energy_total1_2[skip:] - energy_total1_2[skip]) / atoms, 'k-', label='DFT loose')
ax_energy_dft.plot(time_val2_2[skip:]-time_val2_2[skip], (energy_total2_2[skip:]-energy_total2_2[skip])/atoms, '-', color='grey', label='DFT tight')
# ax_energy_dft.plot([skip, time_plot], [9.7e-5, 9.7e-5], 'r--', label='CPMD loose')
# ax_energy_dft.plot([skip, time_plot], [3e-5, 3e-5], '--', color='red', label='CPMD tight')
ax_energy_dft.plot(cpmd_x, 9.7e-5/1e3 * cpmd_x, 'r--', label='CPMD loose')
ax_energy_dft.plot(cpmd_x, 3e-5/1e3 * cpmd_x, '--', color='red', label='CPMD tight')
ax_energy_dft.set_xlabel('Time / fs')
ax_energy_dft.set_ylabel('Energy change per atom / Ha')
ax_energy_dft.set_xlim([0, time_plot])
ax_energy_dft.set_ylim([-1e-5, 9.7e-5*1.5])
ax_energy_dft.legend(frameon=False)
fig_energy_dft.tight_layout()
# fig_energy_dft.savefig('{}/energy_dft_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot time taken per step
# time_plot = 1500
skip = 0
time_end = time_plot * 2
fig_time_dft, ax_time_dft = plt.subplots()
ax_time_dft.plot(time_val1_2[skip:] - time_val1_2[skip], (time_per_step1_2 - time_per_step1_2[skip])/1, 'k-', label='DFT loose')
ax_time_dft.plot(time_val2_2[skip:]-time_val2_2[skip], (time_per_step2_2 - time_per_step2_2[skip])/1, '-', color='grey', label='DFT tight')
ax_time_dft.set_xlabel('Time / fs')
ax_time_dft.set_ylabel('Time taken per MD step / s')
ax_time_dft.set_xlim([0, time_plot])
ax_time_dft.set_ylim([0, 60])
ax_time_dft.legend(frameon=False)
fig_time_dft.tight_layout()
# fig_time_dft.savefig('{}/time_dft_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot total lambda DFT
time_plot = 950
hartree_to_ev = 27.2114
skip = 0
steps1_4 = np.arange(start=1000,stop=1400,step=10)
steps2_4 = np.arange(start=1000,stop=1400,step=10)-741
steps3_4 = np.arange(start=1400,stop=1647,step=10)-741
steps4_4 = np.arange(start=1650,stop=2000,step=10)-1641
index_delete = np.where(steps3_4 == 1440-741)
steps2_4 = steps2_4 #np.delete(steps2_4, index_delete[0][0]-1)
steps3_4 = np.delete(steps3_4, index_delete[0][0])
lambda2_4 = lambda2_4 #np.delete(lambda2_4, index_delete[0][0]-1)
lambda3_4 = np.delete(lambda3_4, index_delete[0][0])
lambda_end = time_plot * 2
fig_lambda, ax_lambda = plt.subplots()
mean1 = np.mean((lambda1_4-energy_potential1_3[steps1_4])*hartree_to_ev)
mean2 = np.mean((lambda2_4-energy_potential31_3[steps2_4])*hartree_to_ev)
mean3 = np.mean( (lambda3_4[:24]-energy_potential31_3[steps3_4[:24]])*hartree_to_ev)
mean4 = np.mean((lambda3_4[25:]-energy_potential32_3[steps4_4])*hartree_to_ev)
mean = 1.50268595
# ax_lambda.plot(time_val1_3[steps1_4]-time_val1_3[steps1_4[0]], (lambda1_4-energy_potential1_3[steps1_4])*hartree_to_ev, 'g-', label='CDFT loose (Ru)')
ax_lambda.plot(time_val31_3[steps2_4]-time_val31_3[steps2_4[0]], (lambda2_4-energy_potential31_3[steps2_4])*hartree_to_ev, 'k-')
ax_lambda.plot(time_val31_3[steps3_4[:24]]-time_val31_3[steps2_4[0]], (lambda3_4[:24]-energy_potential31_3[steps3_4[:24]])*hartree_to_ev, 'k-')
ax_lambda.plot(time_val32_3[steps4_4]-time_val31_3[steps2_4[0]], (lambda3_4[25:]-energy_potential32_3[steps4_4])*hartree_to_ev, 'k-')
# ax_lambda.plot([skip, time_plot], [mean1, mean1], 'g--')
ax_lambda.plot([skip, time_plot], [mean, mean], 'k--')
ax_lambda.plot([skip, time_plot], [1.53, 1.53], 'r--')
print('CDFT loose (Ru)', mean1)
print('CDFT loose (Ru, H2O)', mean, mean2, mean3, mean4)
ax_lambda.set_xlabel('Time / fs')
ax_lambda.set_ylabel('Vertical energy gap / eV')
ax_lambda.set_xlim([0, time_plot])
ax_lambda.set_ylim([0, 4.5])
ax_lambda.legend(frameon=False)
fig_lambda.tight_layout()
# fig_lambda.savefig('{}/lambda_ru_h2o_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot total energy
thickness = 1
time_plot = 2000
# skip = 100
# energy_end = time_plot * 2
fig_energy_1, ax_energy_1 = plt.subplots()
ax_energy_1.plot(time_val1_2[skip:] - time_val1_2[skip], (energy_total1_2[skip:] - energy_total1_2[skip]) / atoms, 'k-', label='DFT loose', linewidth=thickness)
ax_energy_1.plot(time_val1_3[skip:] - time_val1_3[skip], (energy_total1_3[skip:] - energy_total1_3[skip]) / atoms, 'g-', label='CDFT loose (Ru)', linewidth=thickness)
ax_energy_1.plot(time_val3_3[skip:] - time_val3_3[skip], (energy_total3_3[skip:] - energy_total3_3[skip]) / atoms, 'm-', label='CDFT loose (Ru, H2O)', linewidth=thickness)
ax_energy_1.plot(time_val31_3[skip:], (energy_total31_3[skip:] - energy_total3_3[skip]) / atoms, 'm-', linewidth=thickness)
ax_energy_1.plot(time_val32_3[skip:], (energy_total32_3[skip:] - energy_total3_3[skip]) / atoms, 'm-', linewidth=thickness)
# ax_energy_1.plot(time_val6_3[skip:] - time_val6_3[skip], (energy_total6_3[skip:] - energy_total6_3[skip]) / atoms, 'y-', label='CDFT loose ABS (Ru, H2O)')
ax_energy_1.plot(cpmd_x, 9.7e-5/1e3 * cpmd_x, 'r--', color='red', label='CPMD loose')
# ax_energy_1.plot(cpmd_x, 3e-5/1e3 * cpmd_x, '--', color='red', label='CPMD tight')
# print(time_val31_3)
ax_energy_1.set_xlabel('Time / fs')
ax_energy_1.set_ylabel('Energy change per atom / Ha')
ax_energy_1.set_xlim([0, time_plot])
ax_energy_1.set_ylim([-1e-5, 9.7e-5*1.5])
# ax_energy_1.set_xlim([0, 1400])
# ax_energy_1.set_ylim([-1e-5, 4e-3])
ax_energy_1.legend(frameon=False)
fig_energy_1.tight_layout()
# fig_energy_1.savefig('{}/energy_loose_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot total energy
time_plot = 1000
# skip = 100
energy_end = time_plot * 2
fig_energy_2, ax_energy_2 = plt.subplots()
ax_energy_2.plot(time_val2_2[skip:] - time_val2_2[skip], (energy_total2_2[skip:] - energy_total2_2[skip]) / atoms, 'k-', label='DFT tight', linewidth=thickness)
ax_energy_2.plot(time_val2_3[skip:] - time_val2_3[skip], (energy_total2_3[skip:] - energy_total2_3[skip]) / atoms, 'g-', label='CDFT tight (Ru)',  linewidth=thickness)
ax_energy_2.plot(time_val21_3[skip:], (energy_total21_3[skip:] - energy_total2_3[skip]) / atoms, 'g-', linewidth=thickness)
# ax_energy_2.plot(time_val4_3[skip:] - time_val4_3[skip], (energy_total4_3[skip:] - energy_total4_3[skip]) / atoms, 'g-', label='CDFT tight (Ru, H2O)')
ax_energy_2.plot(time_val5_3[skip:] - time_val5_3[skip], (energy_total5_3[skip:] - energy_total5_3[skip]) / atoms, 'm-', label='CDFT tight (Ru, H2O)', linewidth=thickness)
ax_energy_2.plot(time_val51_3[skip:], (energy_total51_3[skip:] - energy_total5_3[skip]) / atoms, 'm-', linewidth=thickness)
ax_energy_2.plot(time_val52_3[skip:], (energy_total52_3[skip:] - energy_total5_3[skip]) / atoms, 'm-', linewidth=thickness)
ax_energy_2.plot(time_val53_3[skip:], (energy_total53_3[skip:] - energy_total5_3[skip]) / atoms, 'm-', linewidth=thickness)
ax_energy_2.plot(cpmd_x, 3e-5/1e3 * cpmd_x, '--', color='red', label='CPMD tight')
ax_energy_2.set_xlabel('Time / fs')
ax_energy_2.set_ylabel('Energy change per atom / Ha')
ax_energy_2.set_xlim([0, time_plot])
ax_energy_2.set_ylim([-1e-5, 3e-5])
# ax_energy_2.set_ylim([-1.25e-5, 0.6e-5])
ax_energy_2.legend(frameon=False)
fig_energy_2.tight_layout()
# fig_energy_2.savefig('{}/energy_tight_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot total energy
thickness = 1
time_plot = 2000
# skip = 100
# energy_end = time_plot * 2
fig_energy_3, ax_energy_3 = plt.subplots()
# ax_energy_3.plot(time_val1_2[skip:] - time_val1_2[skip], (energy_total1_2[skip:] - energy_total1_2[skip]) / atoms, 'k-', label='DFT loose', linewidth=thickness)
# ax_energy_3.plot(time_val1_3[skip:] - time_val1_3[skip], (energy_total1_3[skip:] - energy_total1_3[skip]) / atoms, 'g-', label='CDFT loose (Ru)', linewidth=thickness)
ax_energy_3.plot(time_val3_3[skip:] - time_val3_3[skip], (energy_total3_3[skip:] - energy_total3_3[skip]) / atoms, 'm-', label='CDFT loose REL', linewidth=thickness)
ax_energy_3.plot(time_val31_3[skip:], (energy_total31_3[skip:] - energy_total3_3[skip]) / atoms, 'm-', linewidth=thickness)
ax_energy_3.plot(time_val32_3[skip:], (energy_total32_3[skip:] - energy_total3_3[skip]) / atoms, 'm-', linewidth=thickness)
ax_energy_3.plot(time_val6_3[skip:] - time_val6_3[skip], (energy_total6_3[skip:] - energy_total6_3[skip]) / atoms, 'b-', label='CDFT loose ABS')
ax_energy_3.plot(time_val61_3[skip:] - time_val6_3[skip], (energy_total61_3[skip:] - energy_total6_3[skip]) / atoms, 'b-')
ax_energy_3.plot(time_val7_3[skip:] - time_val7_3[skip], (energy_total7_3[skip:] - energy_total7_3[skip]) / atoms, 'g-', label='CDFT tight ABS')
ax_energy_3.plot(time_val71_3[skip:] - time_val7_3[skip], (energy_total71_3[skip:] - energy_total7_3[skip]) / atoms, 'g-')
ax_energy_3.plot(time_val72_3[skip:] - time_val7_3[skip], (energy_total72_3[skip:] - energy_total7_3[skip]) / atoms, 'g-')
ax_energy_3.plot(time_val8_3[skip:] - time_val7_3[skip], (energy_total8_3[skip:] - energy_total7_3[skip]) / atoms, 'r-', label='CDFT tight ABS dt=0.1')
# ax_energy_3.plot(cpmd_x, 9.7e-5/1e3 * cpmd_x, 'r--', color='red', label='CPMD loose')
# ax_energy_3.plot(cpmd_x, 3e-5/1e3 * cpmd_x, '--', color='red', label='CPMD tight')
# print(time_val31_3)
ax_energy_3.set_xlabel('Time / fs')
ax_energy_3.set_ylabel('Energy change per atom / Ha')
# ax_energy_3.set_xlim([0, time_plot])
# ax_energy_3.set_ylim([-1e-5, 9.7e-5*1.5])
ax_energy_3.set_xlim([0, 1400])
ax_energy_3.set_ylim([-1e-5, 4e-3])
ax_energy_3.legend(frameon=False)
fig_energy_3.tight_layout()
# fig_energy_3.savefig('{}/energy_unhpysical_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot total energy difference to DFT
thickness = 1
time_plot = 2000
# skip = 100
# energy_end = time_plot * 2
fig_energy_4, ax_energy_4 = plt.subplots()
ax_energy_4.plot(time_val1_2[skip:] - time_val1_2[skip], energy_total1_2[skip:] - energy_total1_2[skip], 'k-', label='DFT')
ax_energy_4.plot(time_val6_3[skip:] - time_val6_3[skip], energy_total6_3[skip:] - energy_total1_2[skip], 'b-', label='CDFT ABS (Ru, H2O)')
# ax_energy_4.plot(time_val7_3[skip:] - time_val7_3[skip], energy_total7_3[skip:] - energy_total1_2[skip], 'b-', label='CDFT tight ABS (Ru, H2O)')
ax_energy_4.plot(time_val1_3[skip:] - time_val1_3[skip], energy_total1_3[skip:] - energy_total1_2[skip], 'g-', label='CDFT REL (Ru)')
ax_energy_4.plot(time_val3_3[skip:] - time_val3_3[skip], energy_total3_3[skip:] - energy_total1_2[skip], 'm-', label='CDFT REL (Ru, H2O)')
# ax_energy_4.plot(time_val31_3[skip:], energy_total31_3[skip:] - energy_total3_3[skip], 'm-', linewidth=thickness)
# ax_energy_4.plot(time_val32_3[skip:], energy_total32_3[skip:] - energy_total3_3[skip], 'm-', linewidth=thickness)
# ax_energy_4.plot(time_val6_3[skip:] - time_val6_3[skip], (energy_total6_3[skip:] - energy_total6_3[skip]) / atoms, 'y-', label='CDFT loose ABS (Ru, H2O)')
# ax_energy_4.plot(cpmd_x, 9.7e-5/1e3 * cpmd_x, 'r--', color='red', label='CPMD loose')
# ax_energy_4.plot(cpmd_x, 3e-5/1e3 * cpmd_x, '--', color='red', label='CPMD tight')
# print(time_val31_3)
ax_energy_4.set_xlabel('Time / fs')
ax_energy_4.set_ylabel('Energy difference to DFT / Ha')
ax_energy_4.set_xlim([0, 400])
ax_energy_4.set_ylim([-0.002, 0.15])
# ax_energy_4.set_xlim([0, 1400])
# ax_energy_4.set_ylim([-1e-5, 4e-3])
ax_energy_4.legend(frameon=False)
fig_energy_4.tight_layout()
# fig_energy_4.savefig('{}/energy_change_dft_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot temperature
thickness = 1
time_plot = 2000
# skip = 100
# energy_end = time_plot * 2
fig_temperature_1, ax_temperature_1 = plt.subplots()
# ax_temperature_1.plot(time_val1_2[skip:] - time_val1_2[skip], (temperature1_2[skip:] - temperature1_2[skip]) / atoms, 'k-', label='DFT loose', linewidth=thickness)
# ax_temperature_1.plot(time_val1_3[skip:] - time_val1_3[skip], (temperature1_3[skip:] - temperature1_3[skip]) / atoms, 'g-', label='CDFT loose (Ru)', linewidth=thickness)
ax_temperature_1.plot(time_val3_3[skip:] - time_val3_3[skip], temperature3_3[skip:], 'm-', label='CDFT loose REL', linewidth=thickness)
ax_temperature_1.plot(time_val31_3[skip:], temperature31_3[skip:], 'm-', linewidth=thickness)
ax_temperature_1.plot(time_val32_3[skip:], temperature32_3[skip:], 'm-', linewidth=thickness)
ax_temperature_1.plot(time_val6_3[skip:] - time_val6_3[skip], temperature6_3[skip:], 'b-', label='CDFT loose ABS')
# ax_temperature_1.plot(cpmd_x, 9.7e-5/1e3 * cpmd_x, 'r--', color='red', label='CPMD loose')
# ax_temperature_1.plot(cpmd_x, 3e-5/1e3 * cpmd_x, '--', color='red', label='CPMD tight')
# print(time_val31_3)
ax_temperature_1.set_xlabel('Time / fs')
ax_temperature_1.set_ylabel('Temperature / K')
# ax_temperature_1.set_xlim([0, time_plot])
# ax_temperature_1.set_ylim([-1e-5, 9.7e-5*1.5])
ax_temperature_1.set_xlim([0, 1400])
# ax_temperature_1.set_ylim([-1e-5, 4e-3])
ax_temperature_1.legend(frameon=False)
fig_temperature_1.tight_layout()
# fig_temperature_1.savefig('{}/temperature_unhpysical_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot Hirshfeld analysis
time_plot = 1000
# skip = 30
skip_line = 2
conv_start = 1000
# plot_index = np.concatenate([index_ru1,index_h2o1])
plot_index = index_h2o1
cpmd = 0.52
fig_hirshfeld2, ax_hirshfeld2 = plt.subplots()
temp1 = np.zeros(num_data1_2)
i = -1
for n in range(num_data1_2):
    i = i + 1
    for j in range(len(plot_index)):
        temp1[n] = temp1[n] + (file_spec1_2.loc[atoms * i + skip_line * i + plot_index[j], 'Charge'])
print('Charge DFT', np.mean(temp1[skip:time_plot]))
ax_hirshfeld2.plot(time_val1_2[skip:]-time_val1_2[skip], temp1[skip:], 'k-', label='DFT')
ax_hirshfeld2.plot([skip, time_plot],
                   [np.mean(temp1[skip:time_plot]),np.mean(temp1[skip:time_plot])], 'k--', alpha=0.5)
# temp1 = np.zeros(num_data1_3)
# i = -1
# for n in range(num_data1_3):
#     i = i + 1
#     for j in range(len(plot_index)):
#         temp1[n] = temp1[n] + (file_spec1_3.loc[atoms * i + skip_line * i + plot_index[j], 'Charge'])
# print('Charge CDFT (Ru)', np.mean(temp1[skip:time_plot]))
# ax_hirshfeld2.plot(time_val1_3[skip:] - time_val1_3[skip], temp1[skip:], 'g-', label='CDFT (Ru)')
# ax_hirshfeld2.plot([skip, time_plot],
#                    [np.mean(temp1[skip:time_plot]), np.mean(temp1[skip:time_plot])], 'g--', alpha=0.5)
temp1 = np.zeros(num_data3_3)
i = -1
for n in range(num_data3_3):
    i = i + 1
    for j in range(len(plot_index)):
        temp1[n] = temp1[n] + (file_spec3_3.loc[atoms * i + skip_line * i + plot_index[j], 'Charge'])
ax_hirshfeld2.plot(time_val3_3[skip:] - time_val3_3[skip], temp1[skip:], 'm-', label='CDFT REL (Ru, H2O)')
temp2 = np.zeros(num_data31_3)
i = -1
for n in range(num_data31_3):
    i = i + 1
    for j in range(len(plot_index)):
        temp2[n] = temp2[n] + (file_spec31_3.loc[atoms * i + skip_line * i + plot_index[j], 'Charge'])
temp3 = np.zeros(num_data6_3)
i = -1
for n in range(num_data6_3):
    i = i + 1
    for j in range(len(plot_index)):
        temp3[n] = temp3[n] + (file_spec6_3.loc[atoms * i + skip_line * i + plot_index[j], 'Charge'])
print('Charge DFT', np.mean(temp3[skip:time_plot]))
ax_hirshfeld2.plot(time_val6_3[skip:]-time_val6_3[skip], temp3[skip:], 'b-', label='CDFT ABS (Ru, H2O)')
ax_hirshfeld2.plot([skip, time_plot],
                   [np.mean(temp3[skip:time_plot]),np.mean(temp3[skip:time_plot])], 'b--', alpha=0.5)
# mean1 = np.mean(temp1[:])
# print('Charge CDFT REL (Ru)', mean1)
mean2 = np.mean(temp2[:])
print('Charge CDFT REL (Ru, H2O)', mean2)
mean3 = np.mean(temp3[:])
print('Charge CDFT ABS (Ru, H2O)', mean3)
ax_hirshfeld2.plot(time_val31_3[skip:] - time_val3_3[skip], temp2[skip:-1], 'm-')
ax_hirshfeld2.plot([skip, time_plot],[mean2, mean2], 'm--', alpha=0.5)
# ax_hirshfeld2.plot([skip, time_plot], [cpmd, cpmd], '--', color='red',alpha=1, label='CPMD')
# ax_hirshfeld2.plot([conv_start, conv_start], [0, 1e3], 'k--')
ax_hirshfeld2.set_xlabel('Time / fs')
ax_hirshfeld2.set_ylabel('Total Hirshfeld charge')
ax_hirshfeld2.set_xlim([0, time_plot])
# ax_hirshfeld2.set_ylim([0.8, 1.82])
# ax_hirshfeld2.set_ylim([0.3, 1.4])
# ax_hirshfeld2.set_ylim([-0.05, 1.4])
# ax_hirshfeld2.set_ylim([0.27, 0.57])  # ru1
ax_hirshfeld2.set_ylim([-0.45, 0.88])  # h201
# ax_hirshfeld2.set_ylim([0.45, 0.7])  # ru2
# ax_hirshfeld2.set_ylim([0.3, 1.08])  # h202
ax_hirshfeld2.legend(frameon=False)
fig_hirshfeld2.tight_layout()
# fig_hirshfeld2.savefig('{}/charge_unphysical_h2o2_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
