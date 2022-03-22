from __future__ import division, print_function, unicode_literals
import time
import numpy as np
import copy

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)
from matplotlib import pyplot as plt
import load_coordinates
import load_energy
import load_forces
import load_forces_out
import matplotlib.cm as cm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import functions
import parameters as param
from matplotlib.ticker import FormatStrFormatter

"""
    Plotting CP2K outputs
"""


# Read CP2K output files
folder_data = 'data/h2o_2_quantum'
folder_save = 'output/h2o_2_quantum'
pbe_energy_potential = np.loadtxt('{}{}'.format(folder_data, '/pbe_energy.out'), skiprows=1)
coord_x, coord_y, coord_z, num_atoms, num_timesteps = load_coordinates.load_values_coord(folder_data)
force_x, force_y, force_z, _, _ = load_forces.load_values_forces(folder_data)
force_x_pbe, force_y_pbe, force_z_pbe, _, _ = load_forces_out.load_values_forces(folder_data, num_atoms)
energy_kinetic, energy_potential, temperature, time_val, time_per_step = load_energy.load_values_energy(folder_data)
# num_timesteps = 500


# def truncate_func(truncate_list):
#     """
#         Truncate data to desired length
#     """
#     for i in range(0, len(truncate_list)):
#         truncate_list[i] = truncate_list[i][0:num_timesteps]
#     return truncate_list
#
#
# coord_x, coord_y, coord_z, force_x, force_y, force_z, force_x_pbe, force_y_pbe, force_z_pbe, energy_kinetic, \
#     energy_potential, pbe_energy_potential, temperature, time_val = truncate_func([
#         coord_x, coord_y, coord_z, force_x, force_y, force_z, force_x_pbe, force_y_pbe, force_z_pbe, energy_kinetic,
#         energy_potential, pbe_energy_potential, temperature, time_val])


# force_x = force_x - force_x_pbe
# force_y = force_y - force_y_pbe
# force_z = force_z - force_z_pbe

# Plot subfig of all forces
# fig_forces, ax_forces = plt.subplots(4, 3, figsize=(10, 6))
# for col in range(3):
#     ax_forces[0, col].plot(time_val, force_x[:, col], 'r')
#     ax_forces[1, col].plot(time_val, force_y[:, col], 'g')
#     ax_forces[2, col].plot(time_val, force_z[:, col], 'b')
#     ax_forces[3, col].plot(time_val, np.sqrt(force_x[:, col] ** 2 +
#                                              force_y[:, col] ** 2 +
#                                              force_z[:, col] ** 2), 'k')
# ax_forces[0, 0].set_ylabel('Force x')
# ax_forces[1, 0].set_ylabel('Force y')
# ax_forces[2, 0].set_ylabel('Force z')
# ax_forces[3, 0].set_ylabel('Force net')
# ax_forces[3, 0].set_xlabel('O')
# ax_forces[3, 1].set_xlabel('H_1')
# ax_forces[3, 2].set_xlabel('H_2')
# fig_forces.tight_layout()
# fig_forces.savefig('{}{}'.format(folder_save, '/forces_hse.png'), dpi=param.save_dpi, bbbox_inches='tight')

# Plot distances and components between atoms (1 H2O)
# fig_distances, ax_distances = plt.subplots(4, 3, figsize=(10, 6))
# size_triangle = int(((num_atoms ** 2) - num_atoms) / 2)
# matrix_distance_x = np.zeros((num_atoms, num_atoms))
# matrix_distance_y = np.zeros((num_atoms, num_atoms))
# matrix_distance_z = np.zeros((num_atoms, num_atoms))
# matrix_distance_r = np.zeros((num_atoms, num_atoms))
# matrix_distance_r_time = np.zeros((num_timesteps, size_triangle))
# matrix_distance_x_time = np.zeros((num_timesteps, size_triangle))
# matrix_distance_y_time = np.zeros((num_timesteps, size_triangle))
# matrix_distance_z_time = np.zeros((num_timesteps, size_triangle))
# for timestep in range(num_timesteps):
#     for atom_1 in range(num_atoms):
#         for atom_2 in range(num_atoms):
#
#             if atom_1 != atom_2:
#
#                 # Calculate atomic separation
#                 matrix_distance_x[atom_1, atom_2] = np.abs(coord_x[timestep, atom_1] - coord_x[timestep, atom_2])
#                 matrix_distance_y[atom_1, atom_2] = np.abs(coord_y[timestep, atom_1] - coord_y[timestep, atom_2])
#                 matrix_distance_z[atom_1, atom_2] = np.abs(coord_z[timestep, atom_1] - coord_z[timestep, atom_2])
#                 matrix_distance_r[atom_1, atom_2] = np.sqrt(matrix_distance_x[atom_1, atom_2] ** 2.0 +
#                                                             matrix_distance_y[atom_1, atom_2] ** 2.0 +
#                                                             matrix_distance_z[atom_1, atom_2] ** 2.0)
#     matrix_distance_r_time[timestep, :] = functions.upper_tri(matrix_distance_r)
#     matrix_distance_x_time[timestep, :] = functions.upper_tri(matrix_distance_x)
#     matrix_distance_y_time[timestep, :] = functions.upper_tri(matrix_distance_y)
#     matrix_distance_z_time[timestep, :] = functions.upper_tri(matrix_distance_z)
# for col in range(3):
#     ax_distances[0, col].plot(time_val, matrix_distance_x_time[:, col], 'r')
#     ax_distances[1, col].plot(time_val, matrix_distance_y_time[:, col], 'g')
#     ax_distances[2, col].plot(time_val, matrix_distance_z_time[:, col], 'b')
#     ax_distances[3, col].plot(time_val, matrix_distance_r_time[:, col], 'k')
# ax_distances[0, 0].set_ylabel('Distance x / $\AA$')
# ax_distances[1, 0].set_ylabel('Distance y / $\AA$')
# ax_distances[2, 0].set_ylabel('Distance z / $\AA$')
# ax_distances[3, 0].set_ylabel('Distance net / $\AA$')
# ax_distances[3, 0].set_xlabel('O - H')
# ax_distances[3, 1].set_xlabel('O - H')
# ax_distances[3, 2].set_xlabel('H - H')
# fig_distances.tight_layout()
# fig_distances.savefig('{}{}'.format(folder_save, '/distances.png'), dpi=param.save_dpi, bbbox_inches='tight')

# Plot distances between atoms (2 H20)
# fig_dimer, ax_dimer = plt.subplots(3, 6, figsize=(10, 6))
# distance = functions.calculate_distances(coord_x, coord_y, coord_z, num_atoms, num_timesteps)
# for col in range(6):
#     ax_dimer[0, col].plot(time_val, distance[:, 0, col], 'r')
#     ax_dimer[1, col].plot(time_val, distance[:, 1, col], 'g')
#     ax_dimer[2, col].plot(time_val, distance[:, 2, col], 'b')
# ax_dimer[0, 0].set_ylabel('O1')
# ax_dimer[1, 0].set_ylabel('H1')
# ax_dimer[2, 0].set_ylabel('H2')
# ax_dimer[2, 0].set_xlabel('O1')
# ax_dimer[2, 1].set_xlabel('H1')
# ax_dimer[2, 2].set_xlabel('H2')
# ax_dimer[2, 3].set_xlabel('O2')
# ax_dimer[2, 4].set_xlabel('H3')
# ax_dimer[2, 5].set_xlabel('H4')
# fig_dimer.tight_layout()
# fig_dimer.savefig('{}{}'.format(folder_save, '/coordinates.png'), dpi=param.save_dpi, bbbox_inches='tight')

# Plot intramolecular distances between atoms (2 H20)
fig_dimer_intra, ax_dimer_intra = plt.subplots(2, 3, figsize=(10, 6))
distance = functions.calculate_distances(coord_x, coord_y, coord_z, num_atoms, num_timesteps)
ax_dimer_intra[0, 0].plot(time_val, distance[:, 0, 1], 'r')
ax_dimer_intra[0, 1].plot(time_val, distance[:, 0, 2], 'r')
ax_dimer_intra[0, 2].plot(time_val, distance[:, 1, 2], 'r')
ax_dimer_intra[1, 0].plot(time_val, distance[:, 3, 4], 'g')
ax_dimer_intra[1, 1].plot(time_val, distance[:, 3, 5], 'g')
ax_dimer_intra[1, 2].plot(time_val, distance[:, 4, 5], 'g')
ax_dimer_intra[0, 0].set_xlabel('O1-H1')
ax_dimer_intra[0, 1].set_xlabel('O1-H2')
ax_dimer_intra[0, 2].set_xlabel('H1-H2')
ax_dimer_intra[1, 0].set_xlabel('O2-H3')
ax_dimer_intra[1, 1].set_xlabel('O2-H4')
ax_dimer_intra[1, 2].set_xlabel('H3-H4')
fig_dimer_intra.tight_layout()
fig_dimer_intra.savefig('{}{}'.format(folder_save, '/presentation2/intramolecular.png'), dpi=param.save_dpi,
                        bbbox_inches='tight')

# Plot intermolecular distances between atoms (2 H20)
fig_dimer_inter, ax_dimer_inter = plt.subplots(3, 3, figsize=(10, 6))
distance = functions.calculate_distances(coord_x, coord_y, coord_z, num_atoms, num_timesteps)
for col in range(3, 6):
    ax_dimer_inter[0, col-3].plot(time_val, distance[:, 0, col], 'r')
    ax_dimer_inter[1, col-3].plot(time_val, distance[:, 1, col], 'g')
    ax_dimer_inter[2, col-3].plot(time_val, distance[:, 2, col], 'b')
ax_dimer_inter[0, 0].set_ylabel('O1')
ax_dimer_inter[1, 0].set_ylabel('H1')
ax_dimer_inter[2, 0].set_ylabel('H2')
ax_dimer_inter[2, 0].set_xlabel('O2')
ax_dimer_inter[2, 1].set_xlabel('H3')
ax_dimer_inter[2, 2].set_xlabel('H4')
fig_dimer_inter.tight_layout()
fig_dimer_inter.savefig('{}{}'.format(folder_save, '/presentation2/intermolecular.png'), dpi=param.save_dpi,
                        bbbox_inches='tight')

fig_inter, ax_inter = plt.subplots(1, 1)
ax_inter.plot(time_val, (distance[:, 0, 3] + distance[:, 0, 4] + distance[:, 0, 5]), 'k')
ax_inter.set_xlabel('O intermolecular')
fig_inter.savefig('{}{}'.format(folder_save, '/presentation2/O_intermolecular.png'), dpi=param.save_dpi,
                  bbbox_inches='tight')

fig_intra, ax_intra = plt.subplots(1, 1)
ax_intra.plot(time_val, (distance[:, 0, 1] + distance[:, 0, 2]), 'k')
ax_intra.set_xlabel('O intramolecular')
fig_intra.savefig('{}{}'.format(folder_save, '/presentation2/O_intramolecular.png'), dpi=param.save_dpi,
                  bbbox_inches='tight')

# fig_representation2, ax_representation2 = plt.subplots()
# ax_representation2.plot(time_val, 1/np.sum(distance[:, 0, 1:], axis=1))  # LR
# ax_representation2.plot(time_val, np.sum(distance[:, 0, 1:3], axis=1))  # SR

# Plot distances between atoms (2 H20)
# fig_distances, ax_distances = plt.subplots(2, 3, figsize=(10, 6))
# size_triangle = int(((num_atoms ** 2) - num_atoms) / 2)
# matrix_distance_x = np.zeros((num_atoms, num_atoms))
# matrix_distance_y = np.zeros((num_atoms, num_atoms))
# matrix_distance_z = np.zeros((num_atoms, num_atoms))
# matrix_distance_r = np.zeros((num_atoms, num_atoms))
# matrix_distance_r_time = np.zeros((num_timesteps, size_triangle))
# for timestep in range(num_timesteps):
#     for atom_1 in range(num_atoms):
#         for atom_2 in range(num_atoms):
#
#             if atom_1 != atom_2:
#
#                 # Calculate atomic separation
#                 matrix_distance_x[atom_1, atom_2] = np.abs(coord_x[timestep, atom_1] - coord_x[timestep, atom_2])
#                 matrix_distance_y[atom_1, atom_2] = np.abs(coord_y[timestep, atom_1] - coord_y[timestep, atom_2])
#                 matrix_distance_z[atom_1, atom_2] = np.abs(coord_z[timestep, atom_1] - coord_z[timestep, atom_2])
#                 matrix_distance_r[atom_1, atom_2] = np.sqrt(matrix_distance_x[atom_1, atom_2] ** 2.0 +
#                                                             matrix_distance_y[atom_1, atom_2] ** 2.0 +
#                                                             matrix_distance_z[atom_1, atom_2] ** 2.0)
#         matrix_distance_r_time = matrix_distance_r_time + np.sin(timestep/10) * 1e-3
#     matrix_distance_r_time[timestep, :] = functions.upper_tri(matrix_distance_r)
# ax_distances[0, 0].plot(time_val, matrix_distance_r_time[:, 0], 'r')
# ax_distances[0, 1].plot(time_val, matrix_distance_r_time[:, 1], 'r')
# ax_distances[0, 2].plot(time_val, matrix_distance_r_time[:, 2], 'r')
# ax_distances[1, 0].plot(time_val, matrix_distance_r_time[:, 3], 'b')
# ax_distances[1, 1].plot(time_val, matrix_distance_r_time[:, 4], 'b')
# ax_distances[1, 2].plot(time_val, matrix_distance_r_time[:, 5], 'b')
# fig_distances.tight_layout()
# fig_distances.savefig('{}{}'.format(folder_save, '/distances_inter.png'), dpi=param.save_dpi, bbbox_inches='tight')

# Plot subfig of all coordinates
# fig_coordinates, ax_coordinates = plt.subplots(3, 3, figsize=(10, 6))
# for col in range(3):
#     ax_coordinates[0, col].plot(time_val, coord_x[:, col], 'r')
#     ax_coordinates[1, col].plot(time_val, coord_y[:, col], 'g')
#     ax_coordinates[2, col].plot(time_val, coord_z[:, col], 'b')
# ax_coordinates[0, 0].set_ylabel('Coordinate x / $\AA$')
# ax_coordinates[1, 0].set_ylabel('Coordinate y/  $\AA$')
# ax_coordinates[2, 0].set_ylabel('Coordinate z / $\AA$')
# ax_coordinates[2, 0].set_xlabel('O')
# ax_coordinates[2, 1].set_xlabel('H_1')
# ax_coordinates[2, 2].set_xlabel('H_2')
# fig_coordinates.tight_layout()
# fig_coordinates.savefig('{}{}'.format(folder_save, '/coordinates.png'), dpi=param.save_dpi, bbbox_inches='tight')

# Plot three body terms
# fig_three_body, ax_three_body = plt.subplots(1, 3, figsize=(9, 3))
# ax_three_body[0].plot(time_val, matrix_distance_r_time[:, 0] - matrix_distance_r_time[:, 1], 'r')
# ax_three_body[1].plot(time_val, matrix_distance_r_time[:, 1] - matrix_distance_r_time[:, 2], 'g')
# ax_three_body[2].plot(time_val, matrix_distance_r_time[:, 2] - matrix_distance_r_time[:, 0], 'b')
# ax_three_body[0].set_ylabel('Distance OH1 - OH2 / $\AA$')
# ax_three_body[1].set_ylabel('Distance OH2 - H1H2 / $\AA$')
# ax_three_body[2].set_ylabel('Distance H1H2 - OH1 / $\AA$')
# fig_three_body.tight_layout()
# fig_three_body.savefig('{}{}'.format(folder_save, '/three_body_minus.png'), dpi=param.save_dpi, bbbox_inches='tight')

# Plot subfig of all energies
# fig_energy, ax_energy = plt.subplots(1, 3, figsize=(9, 3))
# ax_energy[0].plot(time_val, pbe_energy_potential, 'r')
# ax_energy[1].plot(time_val, energy_potential, 'g')
# ax_energy[2].plot(time_val, energy_potential - pbe_energy_potential, 'b')
# ax_energy[0].set_ylabel('PBE energy / Eh')
# ax_energy[1].set_ylabel('HSE energy / Eh')
# ax_energy[2].set_ylabel('HSE - PBE energy / Eh')
# fig_energy.tight_layout()
# fig_energy.savefig('{}{}'.format(folder_save, '/energy.png'), dpi=param.save_dpi, bbbox_inches='tight')


if __name__ == "__main__":
    print('Finished.')
    plt.show()
