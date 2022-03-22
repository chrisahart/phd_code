from __future__ import division, print_function
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scripts.general import parameters
import MDAnalysis as mda
from MDAnalysis.analysis import distances

"""
    Plot Hirshfeld analysis for hematite/water interface hole 
"""

def read_energy(folder, filename):
    """
        Return CP2K MD .ener file as re-structured Numpy array.
    """

    files = ['{}/{}'.format(folder, filename)]
    cols = ['Step', 'Time', 'E_kin', 'Temp', 'E_pot', 'E_tot', 'Time_per_step']
    file_energy = pd.read_csv(files[0], delim_whitespace=True, names=cols, skiprows=[0])

    # Load energy data from Pandas database
    energy_kinetic = file_energy['E_kin'].values
    energy_potential = file_energy['E_pot'].values
    energy_total = file_energy['E_tot'].values
    temperature = file_energy['Temp'].values
    time = file_energy['Time'].values
    time_per_step = file_energy['Time_per_step'].values

    return energy_kinetic, energy_potential, energy_total, temperature, time, time_per_step


def read_hirsh(folder, filename):
    """
    Read Hirshfeld
    """

    cols_hirsh = ['Atom', 'Element', 'Kind', 'Ref Charge', 'Pop 1', 'Pop 2', 'Spin', 'Charge']
    data_hirsh = pd.read_csv('{}{}'.format(folder, filename), names=cols_hirsh, delim_whitespace=True)
    species = data_hirsh['Element']
    data_hirsh = data_hirsh.apply(pd.to_numeric, errors='coerce')

    return data_hirsh, species


def func_metric(a, b, c):
    index = [0, 1, 2, 3, 4, 5]
    metric = np.average(a.flat[index])
    return metric


# Interface Philipp structure
atoms = 435
folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/electron/analysis'
value = 'Spin'
folder_save_2 = folder_2
skip_start_2 = 3
skip_end_2 = 5
polaron_index_fe = {'00': [35-1, 30-1]}
plot_color = 'y', 'm', 'orange', 'hotpink', 'skyblue', 'peru'
box_size = [10.241000, 10.294300, 47.342300,  91.966000, 87.424000, 119.738000]
timestep = 0.5
ylim_1 = [1.6, 3.0]
ylim_2 = [2.150, 1.85]
ylim_3 = [-4.12, -3.5]  # All
ylim_4 = [-0.4, 1.2]
polaron_spin = -3.72

# Atomic index (interface)
h_top = np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]) - 1
h_bot = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) - 1
h_all = np.concatenate([h_top, h_bot])
water = np.linspace(start=157, stop=435, num=435-157+1, dtype=int) - 1
o_a = np.array([42, 45, 48, 75, 78, 81, 108, 111, 114, 141, 144, 147]) - 1
o_b = np.array([49, 52, 55, 82, 85, 88, 115, 118, 121, 148, 151, 154]) - 1
o_c = np.array([41, 44, 47, 74, 77, 80, 107, 110, 113, 140, 143, 146]) - 1
o_d = np.array([51, 54, 57, 84, 87, 90, 117, 120, 123, 150, 153, 156]) - 1
o_e = np.array([40, 43, 46, 73, 76, 79, 106, 109, 112, 139, 142, 145]) - 1
o_f = np.array([50, 53, 56, 83, 86, 89, 116, 119, 122, 149, 152, 155]) - 1
o_g = np.array([25, 26, 27, 58, 59, 60, 91, 92, 93, 124, 125, 126]) - 1
o_all = np.concatenate([o_a, o_b, o_c, o_d, o_e, o_f, o_g])
fe_a = np.array([33, 37, 66, 70, 99, 103, 132, 136]) - 1
fe_b = np.array([29, 34, 62, 67, 95, 100, 128, 133]) - 1
fe_c = np.array([32, 39, 65, 72, 98, 105, 131, 138]) - 1
fe_d = np.array([28, 36, 61, 69, 94, 102, 127, 135]) - 1
fe_e = np.array([31, 38, 64, 71, 97, 104, 130, 137]) - 1
fe_f = np.array([30, 35, 63, 68, 96, 101, 129, 134]) - 1
fe_alpha = np.concatenate([fe_a, fe_c, fe_e])
fe_beta = np.concatenate([fe_b, fe_d, fe_f])
fe_all = np.sort(np.concatenate([fe_a, fe_b, fe_c, fe_d, fe_e, fe_f]))

# Printing and plotting arrays
kinds = ['Hematite: H', 'Hematite: O', 'Hematite: Fe1', 'Hematite: Fe2', 'Water']

# Plot all iron spin 1
rows, cols = 5, 2
run_count = 0
ylim_2 = [2.150, 1.85]
fig_spin2, ax_spin2 = plt.subplots(rows, cols, sharex='col', sharey='row', figsize=(8, 9))
for row in range(rows):
    for col in range(cols):

        run = str(run_count).zfill(2)
        energy_kinetic1_2, energy_potential1_2, energy_total1_2, temperature1_2, time_val1_2, time_per_step1_2 = read_energy(
            folder_2, '/energy/{}.out'.format(run))
        file_spec1_2, species1_2 = read_hirsh(folder_2, '/hirshfeld/{}.out'.format(run))
        num_data1_2 = energy_kinetic1_2.shape[0]
        x_end = time_val1_2[-1] - time_val1_2[0]

        # Setup md analysis environment
        topology_file = '{}/position/topology.xyz'.format(folder_2)
        trajectory_file = '{}/position/{}.xyz'.format(folder_save_2, run)
        universe = mda.Universe(topology_file, trajectory_file)
        atoms_fe = universe.select_atoms('name Fe')
        atoms_o = universe.select_atoms('name O')
        dist_arr = distances.distance_array(atoms_fe.positions, atoms_o.positions, box=box_size)

        # Collect bond lengths over trajectory
        bond_lengths_time = np.zeros((len(universe.trajectory), len(atoms_fe), len(atoms_o)))
        bond_lengths_mean_1 = np.zeros((len(universe.trajectory)))
        bond_lengths_mean_2 = np.zeros((len(universe.trajectory)))
        for ts in universe.trajectory:
            frame = universe.trajectory.frame
            bond_lengths_time[frame] = distances.distance_array(atoms_fe.positions, atoms_o.positions, box=box_size)
            bond_lengths_mean_1[frame] = np.average(np.sort(bond_lengths_time[frame])[:, 0:3])
            bond_lengths_mean_2[frame] = np.average(np.sort(bond_lengths_time[frame])[:, 3:6])

        time_plot = np.linspace(start=0, stop=len(universe.trajectory) * timestep, num=len(universe.trajectory))
        metric = np.zeros((len(universe.trajectory)))
        for i in range(len(atoms_fe)):
            for j in range(len(universe.trajectory)):
                sorted = np.sort(bond_lengths_time[j, i])[0:6]
                metric[j] = func_metric(sorted, bond_lengths_mean_1[j], bond_lengths_mean_2[j])
            ax_spin2[row, col].plot(time_plot, metric, 'k')
        ax_spin2[row, col].set_ylim(ylim_2)
        ax_spin2[row, col].set_xlim([0, len(universe.trajectory)*timestep])
        run_count = run_count + 1

ax_spin2[4, 0].set_xlabel('Time / fs', fontsize=11)
ax_spin2[4, 1].set_xlabel('Time / fs', fontsize=11)
ax_spin2[0, 0].set_ylabel('Fe-O / A', fontsize=11)
ax_spin2[1, 0].set_ylabel('Fe-O / A', fontsize=11)
ax_spin2[2, 0].set_ylabel('Fe-O / A', fontsize=11)
ax_spin2[3, 0].set_ylabel('Fe-O / A', fontsize=11)
ax_spin2[4, 0].set_ylabel('Fe-O / A', fontsize=11)
fig_spin2.tight_layout()
fig_spin2.subplots_adjust(hspace=0)
fig_spin2.savefig('{}/fe_structure_subplot.png'.format(folder_save_2), dpi=300, bbbox_inches='tight')


if __name__ == "__main__":
    print('Finished.')
    plt.show()
