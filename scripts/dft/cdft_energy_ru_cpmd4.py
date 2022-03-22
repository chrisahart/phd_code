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
    Plot energy for ru-ru benchmark  (BLYP, B3LYP, B97) vertical energy gap
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


def fluctuation(value, value_mean):
    """
    Mean square fluctation
    """

    mean_square_deviation = np.sum((value-value_mean)**2)/value.shape[0]
    root_mean_square_deviation1 = np.sqrt(mean_square_deviation)
    root_mean_square_deviation2 = np.sqrt(np.mean((value-value_mean)**2))

    return root_mean_square_deviation2


"""
    Load .ener
"""

skip = 2
atoms = 191
index_ru1 = np.array([1]) - 1
index_h2o1 = np.array([15, 16, 17, 18, 19, 20, 9, 10, 11, 3, 4, 5, 6, 7, 8, 12, 13, 14]) - 1
index_ru2 = np.array([2]) - 1
index_h2o2 = np.array([24, 25, 26, 21, 22, 23, 36, 37, 38, 33, 34, 35, 27, 28, 29, 30, 31, 32]) - 1
folder_pc = 'E:/University/PhD/Programming/dft_ml_md'
folder_laptop = '/home/chris/Storage/DATA/University/PhD/Programming/dft_ml_md/'
folder = folder_pc
# folder_save = '{}/output/cdft/ru/md/b3lyp/initial/'.format(folder)
folder_save = '{}/output/cdft/ru/md/plotting/'.format(folder)

folder_2 = '{}/output/cdft/ru/md/blyp/equilibrated/dft-24h-inverse/analysis'.format(folder)
# folder_2 = '/home/chris/Storage/DATA/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/blyp/equilibrated/dft-24h-inverse/analysis'
energy_kinetic1_2, energy_potential1_2, energy_total1_2, temperature1_2, time_val1_2, time_per_step1_2 = load_energy.load_values_energy(folder_2, '/energy/initial-timcon-33-rattle-cpmd.out')
energy_kinetic2_2, energy_potential2_2, energy_total2_2, temperature2_2, time_val2_2, time_per_step2_2 = load_energy.load_values_energy(folder_2, '/energy/initial-timcon-33-rattle-cpmd-tight.out')
file_spec1_2, species1_2, num_data1_2, step1_2, brent1_2, mnbrack1_2 = read_hirsh(folder_2, '/hirshfeld/initial-timcon-33-rattle-cpmd.out', atoms, None, None)
file_spec2_2, species2_2, num_data2_2, step2_2, brent2_2, mnbrack2_2 = read_hirsh(folder_2, '/hirshfeld/initial-timcon-33-rattle-cpmd-tight.out', atoms, None, None)

folder_3 = '{}/output/cdft/ru/md/blyp/equilibrated/cdft-24h-inverse/analysis'.format(folder)
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
file_spec32_3, species32_3, num_data32_3, step32_3, brent32_3, mnbrack32_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-run-002.out', atoms, None, None)
file_spec5_3, species5_3, num_data5_3, step5_3, brent5_3, mnbrack5_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-tight-run-000.out', atoms, None, None)
file_spec51_3, species51_3, num_data51_3, step51_3, brent51_3, mnbrack51_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-tight-run-001.out', atoms, None, None)
file_spec6_3, species6_3, num_data6_3, step6_3, brent6_3, mnbrack6_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-abs-ru-water-run-000.out', atoms, None, None)
file_spec61_3, species61_3, num_data61_3, step61_3, brent61_3, mnbrack61_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-abs-ru-water-run-001.out', atoms, None, None)
file_spec7_3, species7_3, num_data7_3, step7_3, brent7_3, mnbrack7_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-abs-ru-water-tight-run-000.out', atoms, None, None)
file_spec71_3, species71_3, num_data71_3, step71_3, brent71_3, mnbrack71_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-abs-ru-water-tight-run-001.out', atoms, None, None)
file_spec72_3, species72_3, num_data72_3, step72_3, brent72_3, mnbrack72_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-abs-ru-water-tight-run-002.out', atoms, None, None)
file_spec8_3, species8_3, num_data8_3, step8_3, brent8_3, mnbrack8_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-abs-ru-water-tight-run-002_dt-01.out', atoms, None, None)

folder_5 = '{}/output/cdft/ru/md/b3lyp/initial/dft-24h-inverse/analysis'.format(folder)
energy_kinetic1_5, energy_potential1_5, energy_total1_5, temperature1_5, time_val1_5, time_per_step1_5 = load_energy.load_values_energy(folder_5, '/energy/initial-timcon-33-rattle-cpmd-tight-truncated-5.out')
file_spec1_5, species1_5, num_data1_5, step1_5, brent1_5, mnbrack1_5 = read_hirsh(folder_5, '/hirshfeld/initial-timcon-33-rattle-cpmd-tight-truncated-5.out', atoms, None, None)

folder_6 = '{}/output/cdft/ru/md/b97x/initial/dft-24h-inverse/analysis'.format(folder)
energy_kinetic1_6, energy_potential1_6, energy_total1_6, temperature1_6, time_val1_6, time_per_step1_6 = load_energy.load_values_energy(folder_6, '/energy/initial-timcon-33-rattle-cpmd-tight-libxc.out')
file_spec1_6, species1_6, num_data1_6, step1_6, brent1_6, mnbrack1_6 = read_hirsh(folder_6, '/hirshfeld/initial-timcon-33-rattle-cpmd-tight-libxc.out', atoms, None, None)

folder_7 = '{}/output/cdft/ru/md/b3lyp/equilibrated/dft-24h-inverse/analysis'.format(folder)
energy_kinetic1_7, energy_potential1_7, energy_total1_7, temperature1_7, time_val1_7, time_per_step1_7 = load_energy.load_values_energy(folder_7, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out')
energy_kinetic2_7, energy_potential2_7, energy_total2_7, temperature2_7, time_val2_7, time_per_step2_7 = load_energy.load_values_energy(folder_7, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-run-001.out')
file_spec1_7, species1_7, num_data1_7, step1_7, brent1_7, mnbrack1_7 = read_hirsh(folder_7, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out', atoms, None, None)
file_spec2_7, species2_7, num_data2_7, step2_7, brent2_7, mnbrack2_7 = read_hirsh(folder_7, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-run-001.out', atoms, None, None)

folder_8 = '{}/output/cdft/ru/md/b97x/equilibrated/dft-24h-inverse/analysis'.format(folder)
energy_kinetic1_8, energy_potential1_8, energy_total1_8, temperature1_8, time_val1_8, time_per_step1_8 = load_energy.load_values_energy(folder_8, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out')
energy_kinetic2_8, energy_potential2_8, energy_total2_8, temperature2_8, time_val2_8, time_per_step2_8 = load_energy.load_values_energy(folder_8, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-run-001.out')
file_spec1_8, species1_8, num_data1_8, step1_8, brent1_8, mnbrack1_8 = read_hirsh(folder_8, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out', atoms, None, None)
file_spec2_8, species2_8, num_data2_8, step2_8, brent2_8, mnbrack2_8 = read_hirsh(folder_8, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-run-001.out', atoms, None, None)

folder_9 = '{}/output/cdft/ru/md/b3lyp/equilibrated/cdft-24h-inverse/analysis'.format(folder)
energy_kinetic1_9, energy_potential1_9, energy_total1_9, temperature1_9, time_val1_9, time_per_step1_9 = load_energy.load_values_energy(folder_9, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out')
energy_kinetic2_9, energy_potential2_9, energy_total2_9, temperature2_9, time_val2_9, time_per_step2_9 = load_energy.load_values_energy(folder_9, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-run-001.out')
energy_kinetic3_9, energy_potential3_9, energy_total3_9, temperature3_9, time_val3_9, time_per_step3_9 = load_energy.load_values_energy(folder_9, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-run-002.out')
file_spec1_9, species1_9, num_data1_9, step1_9, brent1_9, mnbrack1_9 = read_hirsh(folder_9, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out', atoms, None, None)
file_spec2_9, species2_9, num_data2_9, step2_9, brent2_9, mnbrack2_9 = read_hirsh(folder_9, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-run-001.out', atoms, None, None)
file_spec3_9, species3_9, num_data3_9, step3_9, brent3_9, mnbrack3_9 = read_hirsh(folder_9, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-run-002.out', atoms, None, None)

folder_10 = '{}/output/cdft/ru/md/b97x/equilibrated/cdft-24h-inverse/analysis'.format(folder)
energy_kinetic1_10, energy_potential1_10, energy_total1_10, temperature1_10, time_val1_10, time_per_step1_10 = load_energy.load_values_energy(folder_10, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out')
energy_kinetic2_10, energy_potential2_10, energy_total2_10, temperature2_10, time_val2_10, time_per_step2_10 = load_energy.load_values_energy(folder_10, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-run-001.out')
energy_kinetic3_10, energy_potential3_10, energy_total3_10, temperature3_10, time_val3_10, time_per_step3_10 = load_energy.load_values_energy(folder_10, '/energy/initial-timcon-33-rattle-cpmd-rel-ru-water-run-002.out')
file_spec1_10, species1_10, num_data1_10, step1_10, brent1_10, mnbrack1_10 = read_hirsh(folder_10, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out', atoms, None, None)
file_spec2_10, species2_10, num_data2_10, step2_10, brent2_10, mnbrack2_10 = read_hirsh(folder_10, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-run-001.out', atoms, None, None)
file_spec3_10, species3_10, num_data3_10, step3_10, brent3_10, mnbrack3_10 = read_hirsh(folder_10, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-run-002.out', atoms, None, None)

folder_4 = '{}/output/cdft/ru/md/blyp/lambda/'.format(folder)
lambda1_4 = np.loadtxt('{}/blyp/analysis/lambda/initial-timcon-33-rattle-cpmd-rel-ru-water.out'.format(folder_4))
lambda2_4 = np.loadtxt('{}/blyp/analysis/lambda/initial-timcon-33-rattle-cpmd-rel-ru-water-cg-rswfn.out'.format(folder_4))
# lambda3_4 = np.loadtxt('{}/blyp/initial-timcon-33-rattle-cpmd-rel-ru-water2/analysis/energy.out'.format(folder_4))
lambda4_4 = np.loadtxt('{}/b3lyp/analysis/lambda/initial-timcon-33-rattle-cpmd-rel-ru-water-cg-rswfn.out'.format(folder_4))
lambda5_4 = np.loadtxt('{}/b3lyp/analysis/lambda/initial-timcon-33-rattle-cpmd-rel-ru-water_vertical-cg-rswfn.out'.format(folder_4))
lambda6_4 = np.loadtxt('{}/b97x/analysis/lambda/initial-timcon-33-rattle-cpmd-rel-ru-water-cg-rswfn.out'.format(folder_4))
lambda7_4 = np.loadtxt('{}/b97x/analysis/lambda/initial-timcon-33-rattle-cpmd-rel-ru-water_vertical-cg-rswfn.out'.format(folder_4))

folder_5 = '{}/output/cdft/ru/md/b3lyp/lambda/'.format(folder)
lambda1_5 = np.loadtxt('{}/analysis/lambda/initial-timcon-33-rattle-cpmd-rel-ru-water-cg-rswfn.out'.format(folder_5))

folder_6 = '{}/output/cdft/ru/md/b97x/lambda/'.format(folder)
lambda1_6 = np.loadtxt('{}/analysis/lambda/initial-timcon-33-rattle-cpmd-rel-ru-water-cg.out'.format(folder_6))

# Plot total lambda DFT
time_plot = 960
hartree_to_ev = 27.2114
skip = 0
fig_lambda1, ax_lambda1 = plt.subplots()

# BLYP on BLYP structure (old)
# steps2_4 = np.arange(start=1000,stop=1400,step=10)-741
# steps3_4 = np.arange(start=1400,stop=1647,step=10)-741
# steps4_4 = np.arange(start=1650,stop=2010,step=10)-1641
# index_delete = np.where(steps3_4 == 1440-741)
# steps3_4 = np.delete(steps3_4, index_delete[0][0])
# energy_series = np.concatenate((energy_potential31_3[steps2_4], energy_potential31_3[steps3_4[:24]], energy_potential32_3[steps4_4]))
# time_series = (np.concatenate((time_val31_3[steps2_4], time_val31_3[steps3_4[:24]], time_val32_3[steps4_4]))) - time_val31_3[steps2_4[0]]
# lambda1_4 = lambda1_4*hartree_to_ev
# energy_series = energy_series *hartree_to_ev
# lambda_value = lambda1_4-energy_series
# lambda_mean = np.mean(lambda_value)
# ax_lambda1.plot(time_series, lambda_value, '-', color='grey', label='BLYP')
# half = int(lambda2_4.shape[0] / 2)
# error = np.mean((lambda1_4[0:half]-energy_series[0:half])) - np.mean((lambda1_4[half:]-energy_series[half:]))
# mean_square_fluctuation = fluctuation(lambda_value, lambda_mean)
# print('BLYP CDFT', lambda_mean, 'pm', error, 'mean square fluctuation', mean_square_fluctuation)
# ax_lambda1.plot([skip, time_plot], [lambda_mean, lambda_mean], '--', color='grey')

# BLYP on BLYP structure
steps2_4 = np.arange(start=1000,stop=1400,step=10)-741
steps3_4 = np.arange(start=1400,stop=1650,step=10)-741
steps4_4 = np.arange(start=1650,stop=2010,step=10)-1641
time_series = (np.concatenate((time_val31_3[steps2_4], time_val31_3[steps3_4], time_val32_3[steps4_4]))) - time_val31_3[steps2_4[0]]
energy_series = np.concatenate((energy_potential31_3[steps2_4], energy_potential31_3[steps3_4], energy_potential32_3[steps4_4]))
energy_series = energy_series * hartree_to_ev
lambda2_4 = lambda2_4 * hartree_to_ev
lambda_value = lambda2_4 - energy_series
lambda_mean = np.mean(lambda_value)
ax_lambda1.plot(time_series, lambda_value, 'g-')
half = int(lambda2_4.shape[0] / 2)
error = np.mean((lambda2_4[0:half]-energy_series[0:half])) - np.mean((lambda2_4[half:]-energy_series[half:]))
mean_square_fluctuation = fluctuation(lambda_value, lambda_mean)
print('BLYP CDFT', lambda_mean, 'pm', error, 'mean square fluctuation', mean_square_fluctuation)
ax_lambda1.plot([skip, time_plot], [lambda_mean, lambda_mean], 'g--', label='BLYP')

# B3LYP on BLYP structure
# steps4_4 = np.arange(start=1000,stop=2010,step=10)
# delete_val = np.array([1140, 1190, 1240, 1380])
# steps4_4 = np.setdiff1d(steps4_4,delete_val)
# lambda4_4 = lambda4_4 * hartree_to_ev
# lambda5_4 = lambda5_4 * hartree_to_ev
# lambda_value = (lambda4_4-lambda5_4)
# lambda_mean = np.mean(lambda_value)
# ax_lambda1.plot(time_val2_7[steps4_4]-time_val2_7[steps4_4[0]], lambda_value, 'g-', label='B3LYP')
# half = int(lambda4_4.shape[0] / 2)
# error = np.mean((lambda4_4[0:half]-lambda5_4[0:half])) - np.mean((lambda4_4[half:]-lambda5_4[half:]))
# mean_square_fluctuation = fluctuation(lambda_value, lambda_mean)
# print('B3LYP CDFT', lambda_mean, 'pm', error, 'mean square fluctuation', mean_square_fluctuation)
# ax_lambda1.plot([skip, time_plot], [lambda_mean, lambda_mean], 'g--')

# wB97X on BLYP structure
# steps5_4 = np.arange(start=1000,stop=2010,step=10)
# delete_val = np.array([1040])
# steps5_4 = np.setdiff1d(steps5_4,delete_val)
# lambda6_4 = lambda6_4 * hartree_to_ev
# lambda7_4 = lambda7_4 * hartree_to_ev
# lambda_value = (lambda6_4-lambda7_4)
# lambda_mean = np.mean(lambda_value)
# ax_lambda1.plot(time_val2_7[steps5_4]-time_val2_7[steps5_4[0]], lambda_value, 'b-', label='wB97X')
# error = np.mean((lambda6_4[0:half]-lambda7_4[0:half])) - np.mean((lambda6_4[half:]-lambda7_4[half:]))
# mean_square_fluctuation = fluctuation(lambda_value, lambda_mean)
# print('wB97X CDFT', lambda_mean, 'pm', error, 'mean square fluctuation', mean_square_fluctuation)
# ax_lambda1.plot([skip, time_plot], [lambda_mean, lambda_mean], 'b--')

# B3LYP on B3LYP structure
# steps1_5 = np.arange(start=1000, stop=1730, step=10) - 801
# steps2_5 = np.arange(start=1730, stop=2010, step=10) - 1721
# energy_series = np.concatenate((energy_potential2_9[steps1_5], energy_potential3_9[steps2_5]))
# time_series = np.concatenate((time_val2_9[steps1_5], time_val3_9[steps2_5]))
# energy_series = energy_series * hartree_to_ev
# lambda1_5 = lambda1_5 * hartree_to_ev
# lambda_value = lambda1_5 - energy_series
# lambda_mean = np.mean(lambda_value)
# ax_lambda1.plot(time_series-time_series[0], lambda_value, 'g-', label='B3LYP')
# half = int(lambda1_5.shape[0] / 2)
# error = np.mean((lambda1_5[0:half]-energy_series[0:half])) - np.mean((lambda1_5[half:]-energy_series[half:]))
# mean_square_fluctuation = fluctuation(lambda_value, lambda_mean)
# print('B3LYP CDFT', lambda_mean, 'pm', error, 'mean square fluctuation', mean_square_fluctuation)
# ax_lambda1.plot([skip, time_plot], [lambda_mean, lambda_mean], 'g--')

# wB97X on wB97X structure
steps1_6 = np.arange(start=1000, stop=1030, step=10) - 0
steps2_6 = np.arange(start=1030, stop=2010, step=10) - 1021
energy_series = np.concatenate((energy_potential1_10[steps1_6], energy_potential2_10[steps2_6]))
time_series = np.concatenate((time_val1_10[steps1_6], time_val2_10[steps2_6]))
energy_series = energy_series * hartree_to_ev
lambda1_6 = lambda1_6 * hartree_to_ev
lambda_value = lambda1_6 - energy_series
lambda_mean = np.mean(lambda_value)
ax_lambda1.plot(time_series-time_series[0], lambda_value, 'b-')
half = int(lambda1_6.shape[0] / 2)
error = np.mean((lambda1_6[0:half]-energy_series[0:half])) - np.mean((lambda1_6[half:]-energy_series[half:]))
mean_square_fluctuation = fluctuation(lambda_value, lambda_mean)
print('B97X CDFT', lambda_mean, 'pm', error, 'mean square fluctuation', mean_square_fluctuation)
ax_lambda1.plot([skip, time_plot], [lambda_mean, lambda_mean], 'b--', label='wB97X')

ax_lambda1.plot([skip, time_plot], [1.53, 1.53], 'r--', label='BLYP (CPMD)')
ax_lambda1.set_xlabel('Time / fs')
ax_lambda1.set_ylabel('Vertical energy gap / eV')
ax_lambda1.set_xlim([0, time_plot])
# ax_lambda1.set_ylim([0, 5])
ax_lambda1.set_ylim([0.5, 3])
ax_lambda1.legend(frameon=False)
fig_lambda1.tight_layout()
# fig_lambda1.savefig('{}/lambda_blyp_only_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')
# fig_lambda1.savefig('{}/lambda_blyp_all_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')
# fig_lambda1.savefig('{}/lambda_dft_struct_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')
fig_lambda1.savefig('{}/lambda_blyp_b97x_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
