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
file_spec1_3, species1_3, num_data1_3, step1_3, brent1_3, mnbrack1_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-run-000.out', atoms, None, None)
file_spec2_3, species2_3, num_data2_3, step2_3, brent2_3, mnbrack2_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-tight-run-000.out', atoms, None, None)
file_spec21_3, species21_3, num_data21_3, step21_3, brent21_3, mnbrack21_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-tight-run-001.out', atoms, None, None)
file_spec3_3, species3_3, num_data3_3, step3_3, brent3_3, mnbrack3_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out', atoms, None, None)
file_spec31_3, species31_3, num_data31_3, step31_3, brent31_3, mnbrack31_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-run-001.out', atoms, None, None)
file_spec5_3, species5_3, num_data5_3, step5_3, brent5_3, mnbrack5_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-tight-run-000.out', atoms, None, None)
file_spec51_3, species51_3, num_data51_3, step51_3, brent51_3, mnbrack51_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-rel-ru-water-tight-run-001.out', atoms, None, None)
file_spec6_3, species6_3, num_data6_3, step6_3, brent6_3, mnbrack6_3 = read_hirsh(folder_3, '/hirshfeld/initial-timcon-33-rattle-cpmd-abs-ru-water-run-000.out', atoms, None, None)

iasd3_3 = np.loadtxt('{}/{}'.format(folder_3, '/iasd/initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out'))
iasd31_3 = np.loadtxt('{}/{}'.format(folder_3, '/iasd/initial-timcon-33-rattle-cpmd-rel-ru-water-run-001.out'))
iasd32_3 = np.loadtxt('{}/{}'.format(folder_3, '/iasd/initial-timcon-33-rattle-cpmd-rel-ru-water-run-002.out'))
iasd6_3 = np.loadtxt('{}/{}'.format(folder_3, '/iasd/initial-timcon-33-rattle-cpmd-abs-ru-water-run-000.out'))


iter3_3 = np.loadtxt('{}/{}'.format(folder_3, '/cdft-iter/initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out'))
iter31_3 = np.loadtxt('{}/{}'.format(folder_3, '/cdft-iter/initial-timcon-33-rattle-cpmd-rel-ru-water-run-001.out'))
iter32_3 = np.loadtxt('{}/{}'.format(folder_3, '/cdft-iter/initial-timcon-33-rattle-cpmd-rel-ru-water-run-002.out'))
iter6_3 = np.loadtxt('{}/{}'.format(folder_3, '/cdft-iter/initial-timcon-33-rattle-cpmd-abs-ru-water-run-000.out'))

strength3_3 = np.loadtxt('{}/{}'.format(folder_3, '/strength/initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out'))
strength31_3 = np.loadtxt('{}/{}'.format(folder_3, '/strength/initial-timcon-33-rattle-cpmd-rel-ru-water-run-001.out'))
strength32_3 = np.loadtxt('{}/{}'.format(folder_3, '/strength/initial-timcon-33-rattle-cpmd-rel-ru-water-run-002.out'))
strength6_3 = np.loadtxt('{}/{}'.format(folder_3, '/strength/initial-timcon-33-rattle-cpmd-abs-ru-water-run-000.out'))

force1_2, forces_x1_2, forces_y1_2, forces_z1_2, num_atoms1_2, num_timesteps1_2 = load_forces.load_values_forces(folder_2, 'force/initial-timcon-33-rattle-cpmd.out')
force3_3, forces_x3_3, forces_y3_3, forces_z3_3, num_atoms3_3, num_timesteps3_3 = load_forces.load_values_forces(folder_3, 'force/initial-timcon-33-rattle-cpmd-rel-ru-water-run-000.out')
force31_3, forces_x31_3, forces_y31_3, forces_z31_3, num_atoms31_3, num_timesteps31_3 = load_forces.load_values_forces(folder_3, 'force/initial-timcon-33-rattle-cpmd-rel-ru-water-run-001.out')
force6_3, forces_x6_3, forces_y6_3, forces_z6_3, num_atoms6_3, num_timesteps6_3 = load_forces.load_values_forces(folder_3, 'force/initial-timcon-33-rattle-cpmd-abs-ru-water-run-000.out')

print(num_data31_3)
print(iter31_3.shape[0])

# Plot iasd
thickness = 1
skip = 0
fig_iasd, ax_iasd = plt.subplots()
time_plot = 1400
ax_iasd.plot(time_val3_3[skip:] - time_val3_3[skip], iasd3_3, 'm-', label='CDFT loose REL (Ru, H2O)')
ax_iasd.plot(time_val31_3[skip:] - time_val3_3[skip], iasd31_3[:-1], 'm-')
ax_iasd.plot(time_val32_3[skip:] - time_val3_3[skip], iasd32_3[:-1], 'm-')
ax_iasd.plot(time_val6_3[skip:] - time_val6_3[skip], iasd6_3, 'b-', label='CDFT loose ABS (Ru, H2O)')
ax_iasd.set_xlabel('Time / fs')
ax_iasd.set_ylabel('Integrated Absolute Spin Density')
ax_iasd.set_xlim([0, 1400])
ax_iasd.legend(frameon=False)
fig_iasd.tight_layout()
# fig_iasd.savefig('{}/iasd{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot strength
skip = 0
fig_strength, ax_strength = plt.subplots()
time_plot = 1400
temp1 = np.zeros(iter3_3.shape[0])
i = -1
for n in range(iter3_3.shape[0]):
    i = i + int(iter3_3[n])
    temp1[n] = strength3_3[i]
ax_strength.plot(time_val3_3[skip:] - time_val3_3[skip], temp1, 'm-', label='CDFT loose REL (Ru, H2O)')
temp1 = np.zeros(iter31_3.shape[0])
i = -1
for n in range(iter31_3.shape[0]):
    i = i + int(iter31_3[n])
    temp1[n] = strength31_3[i]
ax_strength.plot(time_val31_3[skip:] - time_val3_3[skip], temp1[:-1], 'm-')
temp1 = np.zeros(iter32_3.shape[0])
i = -1
for n in range(iter32_3.shape[0]):
    i = i + int(iter32_3[n])
    temp1[n] = strength32_3[i]
ax_strength.plot(time_val32_3[skip:] - time_val3_3[skip], temp1[:-1], 'm-')
temp1 = np.zeros(iter6_3.shape[0])
i = -1
for n in range(iter6_3.shape[0]):
    i = i + int(iter6_3[n])
    temp1[n] = strength6_3[i]
ax_strength.plot(time_val6_3[skip:] - time_val6_3[skip], temp1, 'b-', label='CDFT loose ABS (Ru, H2O)')
ax_strength.set_xlabel('Time / fs')
ax_strength.set_ylabel('Lagrangian multiplier')
ax_strength.set_xlim([0, 1400])
ax_strength.legend(frameon=False)
fig_strength.tight_layout()
# fig_strength.savefig('{}/strength{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot forces 1
# skip = 0
time_plot = 1400
skip_line = 2
# plot_index = index_ru1
plot_index = index_h2o1[0]
fig_forces1, ax_forces1 = plt.subplots()
# temp1 = np.zeros(num_data1_2)
# temp2 = np.zeros(num_data1_2)
# temp3 = np.zeros(num_data1_2)
# i = -1
# for n in range(num_data1_2):
#     i = i + 1
#     temp1[n] = np.sum(forces_x1_2[i, plot_index])
#     temp2[n] = np.sum(forces_y1_2[i, plot_index])
#     temp3[n] = np.sum(forces_z1_2[i, plot_index])
# ax_forces1.plot(time_val1_2[skip:], temp1[skip:], 'r.-', label='DFT x')
# ax_forces1.plot(time_val1_2[skip:], temp2[skip:], 'g.-')
# ax_forces1.plot(time_val1_2[skip:], temp3[skip:], 'b.-')
temp1 = np.zeros(num_data3_3)
temp2 = np.zeros(num_data3_3)
temp3 = np.zeros(num_data3_3)
i = -1
for n in range(num_data3_3):
    i = i + 1
    temp1[n] = np.sum(forces_x3_3[i, plot_index])
    temp2[n] = np.sum(forces_y3_3[i, plot_index])
    temp3[n] = np.sum(forces_z3_3[i, plot_index])
ax_forces1.plot(time_val3_3[skip:], temp1[skip:], 'r--')
ax_forces1.plot(time_val3_3[skip:], temp2[skip:], 'g--')
ax_forces1.plot(time_val3_3[skip:], temp3[skip:], 'b--')
temp4 = np.zeros(num_data31_3)
temp5 = np.zeros(num_data31_3)
temp6 = np.zeros(num_data31_3)
i = -1
for n in range(num_data31_3-10):
    i = i + 1
    temp4[n] = np.sum(forces_x31_3[i, plot_index])
    temp5[n] = np.sum(forces_y31_3[i, plot_index])
    temp6[n] = np.sum(forces_z31_3[i, plot_index])
ax_forces1.plot(time_val31_3[skip:], temp4[skip:-1], 'r--')
ax_forces1.plot(time_val31_3[skip:], temp5[skip:-1], 'g--')
ax_forces1.plot(time_val31_3[skip:], temp6[skip:-1], 'b--')
temp7 = np.zeros(num_data6_3)
temp8 = np.zeros(num_data6_3)
temp9 = np.zeros(num_data6_3)
i = -1
for n in range(num_data6_3):
    i = i + 1
    temp7[n] = np.sum(forces_x6_3[i, plot_index])
    temp8[n] = np.sum(forces_y6_3[i, plot_index])
    temp9[n] = np.sum(forces_z6_3[i, plot_index])
ax_forces1.plot(time_val6_3[skip:], temp7[skip:], 'r-', label='CDFT loose ABS (Ru, H2O) x')
ax_forces1.plot(time_val6_3[skip:], temp8[skip:], 'g-', label='CDFT loose ABS (Ru, H2O) y')
ax_forces1.plot(time_val6_3[skip:], temp9[skip:], 'b-', label='CDFT loose ABS (Ru, H2O) z')
ax_forces1.set_xlabel('Time / fs')
ax_forces1.set_ylabel('Force / au')
ax_forces1.set_xlim([0, time_plot])
ax_forces1.set_xlim([200, 600])
ax_forces1.set_ylim([-0.075, 0.075])
ax_forces1.set_ylim([-0.06, 0.08])
ax_forces1.legend(frameon=False)
fig_forces1.tight_layout()
# fig_forces1.savefig('{}/force_cdft_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot forces 1 error
# skip = 0
skip_line = 2
fig_force_diff, ax_force_diff = plt.subplots()
ax_force_diff.plot(time_val6_3[:1526], temp7-np.concatenate([temp1, temp4])[:1526], 'r-', label='Diff x')
ax_force_diff.plot(time_val6_3[:1526], temp8-np.concatenate([temp2, temp5])[:1526], 'g-', label='Diff x')
ax_force_diff.plot(time_val6_3[:1526], temp9-np.concatenate([temp3, temp6])[:1526], 'b-', label='Diff x')
ax_force_diff.set_xlabel('Time / fs')
ax_force_diff.set_ylabel('Force / au')
ax_force_diff.set_xlim([0, time_plot])
ax_force_diff.legend(frameon=False)
fig_force_diff.tight_layout()
fig_force_diff.savefig('{}/force_cdft_diff_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot forces 2
skip = 0
fig_forces1, ax_forces1 = plt.subplots()
temp1 = np.zeros(num_data6_3)
temp2 = np.zeros(num_data6_3)
temp3 = np.zeros(num_data6_3)
# plot_index = np.concatenate([index_ru1,index_h2o1])
plot_index = np.arange(start=1, stop=atoms)
for j in range(plot_index.shape[0]):
    i = -1
    for n in range(num_data6_3):
        i = i + 1
        temp1[n] = np.sum(forces_x6_3[i, plot_index[j]])
        temp2[n] = np.sum(forces_y6_3[i, plot_index[j]])
        temp3[n] = np.sum(forces_z6_3[i, plot_index[j]])
    ax_forces1.plot(time_val6_3[skip:], temp1[skip:])
    ax_forces1.plot(time_val6_3[skip:], temp2[skip:])
    ax_forces1.plot(time_val6_3[skip:], temp3[skip:])
ax_forces1.set_xlabel('Time / fs')
ax_forces1.set_ylabel('Force / au')
ax_forces1.set_xlim([0, time_plot])
# ax_forces1.set_ylim([-0.015, 0.015])
ax_forces1.legend(frameon=False)
fig_forces1.tight_layout()
# fig_forces1.savefig('{}/force_cdft_all_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
