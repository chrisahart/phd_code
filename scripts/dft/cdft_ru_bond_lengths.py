from __future__ import division, print_function
import time
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from scripts.general import functions
from scripts.general import parameters
from scripts.formatting import load_coordinates
from scripts.formatting import load_energy
from scripts.formatting import load_forces_out
from scripts.formatting import load_forces


"""
    Plot energy for ru-ru benchmark  (BLYP, B3LYP, B97) vertical energy gap
"""

skip = 2
atoms = 191
box_size = [14.5, 11.35, 11.35, 90, 90, 90]
timestep = 0.96

# folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/blyp/equilibrated'
# folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/blyp/equilibrated/dft-24h-inverse/analysis'
# topology_file_1 = '{}/position/topology.xyz'.format(folder_1)
# trajectory_file_1 = '{}/position/initial-timcon-33-rattle-cpmd.xyz'.format(folder_1)
# folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/blyp/equilibrated/cdft-24h-inverse/analysis'
# topology_file_2 = '{}/position/topology.xyz'.format(folder_2)
# trajectory_file_2 = '{}/position/initial-timcon-33-rattle-cpmd-rel-ru-water-run-merge.xyz'.format(folder_2)

# folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/blyp/equilibrated'
# folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/b3lyp/equilibrated/dft-24h-inverse/analysis'
# topology_file_1 = '{}/position/topology.xyz'.format(folder_1)
# trajectory_file_1 = '{}/position/initial-timcon-33-rattle-cpmd-rel-ru-water-run-merge.xyz'.format(folder_1)
# folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/b3lyp/equilibrated/cdft-24h-inverse/analysis'
# topology_file_2 = '{}/position/topology.xyz'.format(folder_2)
# trajectory_file_2 = '{}/position/initial-timcon-33-rattle-cpmd-rel-ru-water-run-merge.xyz'.format(folder_2)

folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/blyp/equilibrated'
folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/b97x/equilibrated/dft-24h-inverse/analysis'
topology_file_1 = '{}/position/topology.xyz'.format(folder_1)
trajectory_file_1 = '{}/position/initial-timcon-33-rattle-cpmd-rel-ru-water-run-merge.xyz'.format(folder_1)
folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/ru/md/b97x/equilibrated/cdft-24h-inverse/analysis'
topology_file_2 = '{}/position/topology.xyz'.format(folder_2)
trajectory_file_2 = '{}/position/initial-timcon-33-rattle-cpmd-rel-ru-water-run-merge.xyz'.format(folder_2)

# Setup md analysis environment 1
universe_1 = mda.Universe(topology_file_1, trajectory_file_1)
atoms_ru_1 = universe_1.select_atoms('name Ru')
atoms_o_1 = universe_1.select_atoms('name O')
dist_arr_1 = distances.distance_array(atoms_ru_1.positions, atoms_o_1.positions, box=box_size)

# Collect bond lengths over trajectory 1
bond_lengths_time = np.zeros((len(universe_1.trajectory), len(atoms_ru_1), len(atoms_o_1)))
bond_lengths_mean1_1 = np.zeros((len(universe_1.trajectory)))
bond_lengths_mean2_1 = np.zeros((len(universe_1.trajectory)))
for ts in universe_1.trajectory:
    frame = universe_1.trajectory.frame
    bond_lengths_time[frame] = distances.distance_array(atoms_ru_1.positions, atoms_o_1.positions, box=box_size)
    bond_lengths_mean1_1[frame] = np.average(np.sort(bond_lengths_time[frame])[0, 0:6])
    bond_lengths_mean2_1[frame] = np.average(np.sort(bond_lengths_time[frame])[1, 0:6])
bond_lengths_mean1_1_mean = np.average(bond_lengths_mean1_1[1000:])
bond_lengths_mean2_1_mean = np.average(bond_lengths_mean2_1[1000:])
print('np.average(bond_lengths_mean1_2)', bond_lengths_mean1_1_mean)
print('np.average(bond_lengths_mean2_2)', bond_lengths_mean2_1_mean)

# Setup md analysis environment 1
universe_2 = mda.Universe(topology_file_2, trajectory_file_2)
atoms_ru_2 = universe_2.select_atoms('name Ru')
atoms_o_2 = universe_2.select_atoms('name O')
dist_arr_2 = distances.distance_array(atoms_ru_2.positions, atoms_o_2.positions, box=box_size)

# Collect bond lengths over trajectory 1
bond_lengths_time = np.zeros((len(universe_2.trajectory), len(atoms_ru_2), len(atoms_o_2)))
bond_lengths_mean1_2 = np.zeros((len(universe_2.trajectory)))
bond_lengths_mean2_2 = np.zeros((len(universe_2.trajectory)))
for ts in universe_2.trajectory:
    frame = universe_2.trajectory.frame
    bond_lengths_time[frame] = distances.distance_array(atoms_ru_2.positions, atoms_o_2.positions, box=box_size)
    bond_lengths_mean1_2[frame] = np.average(np.sort(bond_lengths_time[frame])[0, 0:6])
    bond_lengths_mean2_2[frame] = np.average(np.sort(bond_lengths_time[frame])[1, 0:6])
bond_lengths_mean1_2_mean = np.average(bond_lengths_mean1_2[1000:])
bond_lengths_mean2_2_mean = np.average(bond_lengths_mean2_2[1000:])
print('np.average(bond_lengths_mean1_2)', bond_lengths_mean1_2_mean)
print('np.average(bond_lengths_mean2_2)', bond_lengths_mean2_2_mean)

# Plot  bond lengths
time_plot_1 = np.linspace(start=0, stop=len(universe_1.trajectory)*timestep, num=len(universe_1.trajectory))
time_plot_2 = np.linspace(start=0, stop=len(universe_2.trajectory)*timestep, num=len(universe_2.trajectory))
fig_bond_lengths, ax_bond_lengths = plt.subplots()
ax_bond_lengths.plot([time_plot_1[0], time_plot_1[-1]], [bond_lengths_mean1_1_mean, bond_lengths_mean1_1_mean], 'r--')
ax_bond_lengths.plot([time_plot_1[0], time_plot_1[-1]], [bond_lengths_mean2_1_mean, bond_lengths_mean2_1_mean], 'g--')
ax_bond_lengths.plot([time_plot_2[0], time_plot_2[-1]], [bond_lengths_mean1_2_mean, bond_lengths_mean1_2_mean], 'b--')
ax_bond_lengths.plot([time_plot_2[0], time_plot_2[-1]], [bond_lengths_mean2_2_mean, bond_lengths_mean2_2_mean], 'm--')
ax_bond_lengths.plot(time_plot_1, bond_lengths_mean1_1, 'r', label='DFT Ru2+')
ax_bond_lengths.plot(time_plot_1, bond_lengths_mean2_1, 'g', label='DFT Ru3+')
ax_bond_lengths.plot(time_plot_2, bond_lengths_mean1_2, 'b', label='CDFT Ru2+')
ax_bond_lengths.plot(time_plot_2, bond_lengths_mean2_2, 'm', label='CDFT Ru3+')
ax_bond_lengths.set_xlabel('Time / s')
ax_bond_lengths.set_ylabel('Average Ru-O bond length / A')
ax_bond_lengths.set_xlim([0, len(universe_2.trajectory)*timestep])
ax_bond_lengths.set_ylim([2.01, 2.26])
ax_bond_lengths.legend(frameon=False)
fig_bond_lengths.tight_layout()
# fig_bond_lengths.savefig('{}/bond_lengths_blyp.png'.format(folder_save), dpi=300, bbbox_inches='tight')
# fig_bond_lengths.savefig('{}/bond_lengths_b3lyp.png'.format(folder_save), dpi=300, bbbox_inches='tight')
fig_bond_lengths.savefig('{}/bond_lengths_wb97x.png'.format(folder_save), dpi=300, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
