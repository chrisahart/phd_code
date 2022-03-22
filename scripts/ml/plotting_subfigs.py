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

"""
    Plotting
"""


# Read CP2K output files
coord_x, coord_y, coord_z, num_atoms, num_timesteps = load_coordinates.load_values_coord('data/h2o_classical')
hse_force_x, hse_force_y, hse_force_z, _, _ = load_forces.load_values_forces('data/h2o_classical')
hse_energy_kinetic, hse_energy_potential, temperature, time_val, time_per_step = \
    load_energy.load_values_energy('data/h2o_classical')
# pbe_energy_potential = np.loadtxt('data/h2o_qm/pbe_energy.out', skiprows=1)
# pbe_forces_x, pbe_forces_y, pbe_forces_z, _, _ = load_forces_out.load_values_forces('data/h2o_classical')

# Plot subfig of all forces
fig_forces, ax_forces = plt.subplots(4, 3)
for col in range(3):
    ax_forces[0, col].plot(time_val, hse_force_x[:, col], 'r')
    ax_forces[1, col].plot(time_val, hse_force_y[:, col], 'g')
    ax_forces[2, col].plot(time_val, hse_force_z[:, col], 'b')
    ax_forces[3, col].plot(time_val, np.sqrt(hse_force_x[:, col] ** 2 +
                                             hse_force_y[:, col] ** 2 +
                                             hse_force_z[:, col] ** 2), 'k')

fig_forces.suptitle('Forces')
ax_forces[3, 0].set_xlabel('O')
ax_forces[3, 1].set_xlabel('H_1')
ax_forces[3, 2].set_xlabel('H_2')
fig_forces.tight_layout()
fig_forces.savefig('{}'.format('output/forces.png'), dpi=param.save_dpi, bbbox_inches='tight')

# Plot absolute distances between atoms
matrix_distance_x = np.zeros((num_atoms, num_atoms))
matrix_distance_y = np.zeros((num_atoms, num_atoms))
matrix_distance_z = np.zeros((num_atoms, num_atoms))
matrix_distance_r = np.zeros((num_atoms, num_atoms))
matrix_distance_r_time = np.zeros((num_timesteps, 3))
matrix_distance_x_time = np.zeros((num_timesteps, 3))
matrix_distance_y_time = np.zeros((num_timesteps, 3))
matrix_distance_z_time = np.zeros((num_timesteps, 3))

for timestep in range(num_timesteps):
    for atom_1 in range(num_atoms):
        for atom_2 in range(num_atoms):

            if atom_1 != atom_2:

                # Calculate atomic separation
                matrix_distance_x[atom_1, atom_2] = np.abs(coord_x[timestep, atom_1] - coord_x[timestep, atom_2])
                matrix_distance_y[atom_1, atom_2] = np.abs(coord_y[timestep, atom_1] - coord_y[timestep, atom_2])
                matrix_distance_z[atom_1, atom_2] = np.abs(coord_z[timestep, atom_1] - coord_z[timestep, atom_2])
                matrix_distance_r[atom_1, atom_2] = np.sqrt(matrix_distance_x[atom_1, atom_2] ** 2.0 +
                                                            matrix_distance_y[atom_1, atom_2] ** 2.0 +
                                                            matrix_distance_z[atom_1, atom_2] ** 2.0)

    matrix_distance_r_time[timestep, :] = functions.upper_tri(matrix_distance_r)
    matrix_distance_x_time[timestep, :] = functions.upper_tri(matrix_distance_x)
    matrix_distance_y_time[timestep, :] = functions.upper_tri(matrix_distance_y)
    matrix_distance_z_time[timestep, :] = functions.upper_tri(matrix_distance_z)

fig_distances, ax_distances = plt.subplots(4, 3)
for col in range(3):
    ax_distances[0, col].plot(time_val, matrix_distance_x_time[:, col], 'r')
    ax_distances[1, col].plot(time_val, matrix_distance_y_time[:, col], 'g')
    ax_distances[2, col].plot(time_val, matrix_distance_z_time[:, col], 'b')
    ax_distances[3, col].plot(time_val, matrix_distance_r_time[:, col], 'k')

fig_distances.suptitle('Absolute distances')
ax_distances[3, 0].set_xlabel('O - H_1')
ax_distances[3, 1].set_xlabel('O - H_2')
ax_distances[3, 2].set_xlabel('H - H')
fig_distances.tight_layout()
fig_distances.savefig('{}'.format('output/distances.png'), dpi=param.save_dpi, bbox_inches='tight')

# Plot subfig of all coordinates
fig_coordinates, ax_coordinates = plt.subplots(3, 3)
for col in range(3):
    ax_coordinates[0, col].plot(time_val, coord_x[:, col], 'r')
    ax_coordinates[1, col].plot(time_val, coord_y[:, col], 'g')
    ax_coordinates[2, col].plot(time_val, coord_z[:, col], 'b')

fig_coordinates.suptitle('Coordinates')
ax_coordinates[2, 0].set_xlabel('O')
ax_coordinates[2, 1].set_xlabel('H_1')
ax_coordinates[2, 2].set_xlabel('H_2')
fig_coordinates.tight_layout()
fig_coordinates.savefig('{}'.format('output/coordinates.png'), dpi=param.save_dpi, bbox_inches='tight')

# Plot subfig of all energies
# fig_energy, ax_energy = plt.subplots(1, 3)
# ax_energy[0].plot(time_val, pbe_energy_potential, 'r')
# ax_energy[1].plot(time_val, hse_energy_potential, 'g')
# ax_energy[2].plot(time_val, hse_energy_potential - pbe_energy_potential, 'b')
#
# fig_energy.suptitle('Energies')
# ax_energy[0].set_xlabel('PBE energy')
# ax_energy[1].set_xlabel('HSE energy')
# ax_energy[2].set_xlabel('HSE - PBE energy')
# fig_energy.savefig('{}'.format('output/energies.png'), dpi=param.save_dpi, bbox_inches='tight')


if __name__ == "__main__":
    print('Finished.')
    plt.show()
