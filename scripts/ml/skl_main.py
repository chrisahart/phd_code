from __future__ import division, print_function
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
import matplotlib.cm as cm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from scripts.general import parameters 
from scripts.formatting import load_coordinates
from scripts.formatting import load_energy
from scripts.formatting import load_forces
from scripts.formatting import load_forces_out
from scripts.general import functions
from scripts.general import parameters
from scripts.ml import parameters_sklearn
from scripts.ml import representations

"""
    SKL script
"""

# Read CP2K output files
folder_data = '../../data/h2o_1_quantum'
folder_save = '../../output/h2o_1_quantum'
coord, coord_x, coord_y, coord_z, species, num_atoms, num_timesteps = \
    load_coordinates.load_values_coord(folder_data, 'hse_coordinates.xyz')
forces, force_x, force_y, force_z, _, _ = load_forces.load_values_forces(folder_data)
force_x_pbe, force_y_pbe, force_z_pbe, _, _ = load_forces_out.load_values_forces(folder_data, num_atoms)
force_net = functions.calculate_force_net(forces, num_timesteps, num_atoms)
energy_kinetic, energy_potential, temperature, time_val, time_per_step = load_energy.load_values_energy(folder_data)
pbe_energy_potential = np.loadtxt('{}{}'.format(folder_data, '/pbe_energy.out'), skiprows=1)

# Edit energy and forces for delta ML
print('Data size', pbe_energy_potential.shape[0], '. Training size', parameters_sklearn.training_size)
energy_potential = energy_potential #- pbe_energy_potential
force_x = force_x - force_x_pbe
force_y = force_y - force_y_pbe
force_z = force_z - force_z_pbe

# Copy ML from parameter file
krr = copy.copy(parameters_sklearn.krr)
gpr = copy.copy(parameters_sklearn.gpr)
nn = copy.copy(parameters_sklearn.nn)

# Training data for ML (energy only), this should be [num_timesteps, n] n>1 for sci-kit learn
target = np.zeros((num_timesteps, 2))
target[:, 0] = energy_potential

# Calculate time dependent Coulomb matrix
matrix_coulomb_time = representations.calculate_coulomb_matrix(coord_x, coord_y, coord_z, num_atoms, num_timesteps)
# matrix_coulomb_time = functions.representation_botu(coord_x, coord_y, coord_z, num_atoms, num_timesteps,
#                                                     parameters.eta_array, parameters.distance_cutoff)
print('matrix_coulomb_time', matrix_coulomb_time.shape)

# Calculate training set convergence
# basis_set_size = np.linspace(12, num_timesteps, num=5, dtype=int)
# print('basis_set_size', basis_set_size)
# training_krr, training_gpr, time_krr, time_gpr\
#     = functions.training_set_convergence(matrix_coulomb_time, basis_set_size, target)
#
# # Plot error convergence
# fig_training_energy1, ax_training_energy1 = plt.subplots()
# # ax_training_energy1.plot(basis_set_size, hse_pbe_error_energy * np.ones(training_krr.shape[0]), 'k', label='PBE')
# ax_training_energy1.plot(basis_set_size, training_krr, 'go')
# ax_training_energy1.plot(basis_set_size, training_krr, 'g', label='ML(krr)')
# ax_training_energy1.plot(basis_set_size, training_gpr, 'b+')
# ax_training_energy1.plot(basis_set_size, training_gpr, 'b', label='ML(gpr)')
# ax_training_energy1.set_xlabel('Training set size')
# ax_training_energy1.set_ylabel('Mean average error')
# # ax_training_energy1.set_ylim([0, 3e-3])
# ax_training_energy1.legend()
# fig_training_energy1.tight_layout()
# fig_training_energy1.savefig('{}{}'.format(folder_save, '/convergence_error_coulomb_force.png'),
#                              dpi=parameters.save_dpi, bbbox_inches='tight')
#
# # Plot error convergence zoomed in
# fig_training_energy2, ax_training_energy2 = plt.subplots()
# # ax_training_energy2.plot(basis_set_size, hse_pbe_error_energy * np.ones(training_krr.shape[0]), 'k', label='PBE')
# ax_training_energy2.plot(basis_set_size, training_krr, 'ro')
# ax_training_energy2.plot(basis_set_size, training_krr, 'r', label='ML(krr)')
# ax_training_energy2.plot(basis_set_size, training_gpr, 'g+')
# ax_training_energy2.plot(basis_set_size, training_gpr, 'g', label='ML(gpr)')
# ax_training_energy2.plot(basis_set_size, training_nn, 'bx')
# ax_training_energy2.plot(basis_set_size, training_nn, 'b', label='ML(nn)')
# ax_training_energy2.set_xlabel('Training set size')
# ax_training_energy2.set_ylabel('Mean average error')
# ax_training_energy2.set_ylim([0, 4e-3])
# ax_training_energy2.legend()
# fig_training_energy2.tight_layout()
# fig_training_energy2.savefig('{}{}'.format(folder_save, '/convergence_error_coulomb_force_zoom.png'),
#                              dpi=parameters.save_dpi, bbbox_inches='tight')
#
# # Plot training_set_convergence
# fig_training_time, ax_training_time = plt.subplots()
# ax_training_time.plot(basis_set_size, time_krr, 'ro')
# ax_training_time.plot(basis_set_size, time_krr, 'r', label='ML(krr)')
# ax_training_time.plot(basis_set_size, time_gpr, 'g+')
# ax_training_time.plot(basis_set_size, time_gpr, 'g', label='ML(gpr)')
# ax_training_time.plot(basis_set_size, time_nn, 'bx')
# ax_training_time.plot(basis_set_size, time_nn, 'b', label='ML(nn)')
# ax_training_time.set_xlabel('Training set size')
# ax_training_time.set_ylabel('Time / s')
# ax_training_time.legend()
# fig_training_time.tight_layout()
# fig_training_time.savefig('{}{}'.format(folder_save, '/convergence_time_coulomb_force2.png'),
#                           dpi=parameters.save_dpi, bbbox_inches='tight')

# Standardise target data
# scaler_energy = StandardScaler()
# scaler_energy.fit(target)
# target = scaler_energy.transform(target)

# Standardise Coulomb matrix

# matrix_coulomb_time_old = matrix_coulomb_time
#
scaler_coulomb = StandardScaler()
scaler_coulomb.fit(matrix_coulomb_time)
matrix_coulomb_time = scaler_coulomb.transform(matrix_coulomb_time)

# Reduce magnitude of intermolecular bonds
# matrix_coulomb_time_new = scaler_coulomb.transform(matrix_coulomb_time)
# matrix_coulomb_time = matrix_coulomb_time_new/10
# matrix_coulomb_time = matrix_coulomb_time_new / matrix_coulomb_time_old[0, :] ** 2
# print('matrix_coulomb_time_old[0, :]', matrix_coulomb_time_old[0, :])
# for test in [0, 1, 5, 12, 13, 14]:
#     matrix_coulomb_time[:, test] = matrix_coulomb_time_new[:, test]

# Train ML using Coulomb matrix as input and energy difference as output
krr.fit(matrix_coulomb_time[:parameters_sklearn.training_size, :], target[:parameters_sklearn.training_size])
gpr.fit(matrix_coulomb_time[:parameters_sklearn.training_size, :], target[:parameters_sklearn.training_size])
nn.fit(matrix_coulomb_time[:parameters_sklearn.training_size, :], target[:parameters_sklearn.training_size])

# Predict ML output data
krr_results = krr.predict(matrix_coulomb_time)
gpr_results = gpr.predict(matrix_coulomb_time)
nn_results = nn.predict(matrix_coulomb_time)

# Perform back transformation of standardisation
# target = (scaler_energy.inverse_transform(target))
# krr_results = (scaler_energy.inverse_transform(krr_results))
# gpr_results = (scaler_energy.inverse_transform(gpr_results))
# nn_results = (scaler_energy.inverse_transform(nn_results))

# energy_potential *= parameters.hartree_to_ev
# pbe_energy_potential *= parameters.hartree_to_ev
# force_x_pbe *= parameters.hartree_per_bohr_to_ev_per_angstrom
# force_y_pbe *= parameters.hartree_per_bohr_to_ev_per_angstrom
# force_z_pbe *= parameters.hartree_per_bohr_to_ev_per_angstrom
# force_x *= parameters.hartree_per_bohr_to_ev_per_angstrom
# force_y *= parameters.hartree_per_bohr_to_ev_per_angstrom
# force_z *= parameters.hartree_per_bohr_to_ev_per_angstrom
#
# hse_pbe_error_energy = np.sum(energy_potential - pbe_energy_potential) / (1 * num_timesteps)
# hse_pbe_error_force = np.sum((np.abs(force_x - force_x_pbe),
#                               np.abs(force_y - force_y_pbe),
#                               np.abs(force_z - force_z_pbe))) / (num_atoms * 3 * num_timesteps)

# Calculate mean error of prediction and print as ['HSE - PBE', 'KRR', 'GPR', 'NN']
# mean_error_pbe = np.reshape((np.sum((np.abs(energy_potential - pbe_energy_potential))) / (num_timesteps * 1)), (1, 1))
# mean_error_krr = np.reshape((np.sum((np.abs(krr_results[:, 0] - target[:, 0]))) / (num_timesteps * 1)), (1, 1))
# mean_error_gpr = np.reshape((np.sum((np.abs(gpr_results[:, 0] - target[:, 0]))) / (num_timesteps * 1)), (1, 1))
# mean_error = np.hstack((mean_error_krr, mean_error_gpr))
# # np.savetxt('{}{}'.format(folder_save, '/presentation2/error_coulomb_power6_inter_train_500.txt'), mean_error)
# print('mean_error', mean_error)

# Calculate mean error of prediction and print as ['HSE - PBE', 'KRR', 'GPR', 'NN']
# start = 1
# end = num_atoms*3
# mean_error_pbe = np.reshape(hse_pbe_error_force, (1, 1))
# mean_error_krr = np.reshape((np.sum((np.abs(krr_results[:, 1:] - target[:, 1:]))) /
#                              (num_timesteps * num_atoms * 3)), (1, 1))
# mean_error_gpr = np.reshape((np.sum((np.abs(gpr_results[:, 1:] - target[:, 1:]))) /
#                              (num_timesteps * num_atoms * 3)), (1, 1))
# mean_error_nn = np.reshape((np.sum((np.abs(nn_results[:, 1:] - target[:, 1:]))) /
#                             (num_timesteps * num_atoms * 3)), (1, 1))
# mean_error = np.hstack((mean_error_pbe, mean_error_krr, mean_error_gpr, mean_error_nn))
# print('mean_error', mean_error)
# np.savetxt('{}{}'.format(folder_save, '/energy_coulomb_energy_only.out'), mean_error)

# Plot machined learned results from first column of training data
fig_results1, ax_results1 = plt.subplots(figsize=(10, 3))
# fig_results1, ax_results1 = plt.subplots()
# ax_results1.plot(time_val, force_x_pbe[:, 0], 'k-', label='PBE', alpha=0.5)

ax_results1.plot(time_val, target[:, 0], 'k', label='HSE')
ax_results1.plot(time_val[0:parameters_sklearn.training_size], krr_results[0:parameters_sklearn.training_size, 0], 'g', alpha=0.5)
ax_results1.plot(time_val[parameters_sklearn.training_size:], krr_results[parameters_sklearn.training_size:, 0], 'g',
                 label='ML(krr)', alpha=1)
ax_results1.plot(time_val[0:parameters_sklearn.training_size], gpr_results[0:parameters_sklearn.training_size, 0], 'b', alpha=0.5)
ax_results1.plot(time_val[parameters_sklearn.training_size:], gpr_results[parameters_sklearn.training_size:, 0], 'b',
                 label='ML(gpr)', alpha=1)

# ax_results1.plot(time_val, pbe_energy_potential +target[:, 0], 'k', label='HSE')
# ax_results1.plot(time_val[0:parameters_sklearn.training_size], pbe_energy_potential[0:parameters_sklearn.training_size]+krr_results[0:parameters_sklearn.training_size, 0], 'g', alpha=0.5)
# ax_results1.plot(time_val[parameters_sklearn.training_size:], pbe_energy_potential[parameters_sklearn.training_size:]+krr_results[parameters_sklearn.training_size:, 0], 'g',
#                  label='ML(krr)', alpha=1)
# ax_results1.plot(time_val[0:parameters_sklearn.training_size], pbe_energy_potential[0:parameters_sklearn.training_size]+gpr_results[0:parameters_sklearn.training_size, 0], 'b', alpha=0.5)
# ax_results1.plot(time_val[parameters_sklearn.training_size:], pbe_energy_potential[parameters_sklearn.training_size:]+gpr_results[parameters_sklearn.training_size:, 0], 'b',
#                  label='ML(gpr)', alpha=1)

# ax_results1.plot(time_val, force_x_pbe[:, 0] +target[:, 0], 'k', label='HSE')
# ax_results1.plot(time_val[0:parameters_sklearn.training_size], force_x_pbe[0:parameters_sklearn.training_size, 0]+krr_results[0:parameters_sklearn.training_size, 0], 'g', alpha=0.5)
# ax_results1.plot(time_val[parameters_sklearn.training_size:], force_x_pbe[parameters_sklearn.training_size:, 0]+krr_results[parameters_sklearn.training_size:, 0], 'g',
#                  label='ML(krr)', alpha=1)
# ax_results1.plot(time_val[0:parameters_sklearn.training_size], force_x_pbe[0:parameters_sklearn.training_size, 0]+gpr_results[0:parameters_sklearn.training_size, 0], 'b', alpha=0.5)
# ax_results1.plot(time_val[parameters_sklearn.training_size:], force_x_pbe[parameters_sklearn.training_size:, 0]+gpr_results[parameters_sklearn.training_size:, 0], 'b',
#                  label='ML(gpr)', alpha=1)

ax_results1.set_xlabel('Time / fs')
ax_results1.set_ylabel('Energy / a.u.')
ax_results1.legend()
fig_results1.tight_layout()
# fig_results1.savefig('{}{}'.format(folder_save, '/presentation2/force_Ox_prediction_diff.png'),
#                      dpi=parameters.save_dpi, bbbox_inches='tight')
#
# # Plot machined learned results from second column of training data
# fig_results2, ax_results2 = plt.subplots()
# # ax_results2.plot(time_val, force_x_pbe[:, 0], 'k-', label='PBE', alpha=0.5)
# ax_results2.plot(time_val, target[:, 1], 'k', label='Exact')
# ax_results2.plot(time_val, krr_results[:, 1], c='g', label='ML(krr)')
# ax_results2.plot(time_val, gpr_results[:, 1], c='b', label='ML(gpr)')
# ax_results2.plot(time_val, nn_results[:, 1], c='y', label='ML(nn)')
# ax_results2.set_xlabel('Time / fs')
# ax_results2.set_ylabel(' / a.u.')
# ax_results2.legend()
# fig_results2.tight_layout()
# fig_results2.savefig('{}{}'.format(folder_save, '/results_2.png'),
#                      dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot values of representation against time
# fig_representation, ax_representation = plt.subplots(figsize=(10, 6))
fig_representation, ax_representation = plt.subplots(figsize=(10, 3))
ax_representation.plot(time_val, matrix_coulomb_time[:, 0], label='O-H1')
ax_representation.plot(time_val, matrix_coulomb_time[:, 1], label='O-H2')
ax_representation.plot(time_val, matrix_coulomb_time[:, 2], label='H1-H2')
#
# ax_representation.plot(time_val, matrix_coulomb_time[:, 0], label='O-H1')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 1], label='O-H2')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 2], label='H1-H2')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 3], label='O2-H3')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 4], label='O2-H4')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 5], label='H3-H4')
#
# ax_representation.plot(time_val, matrix_coulomb_time[:, 0], label='O1-H1')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 1], label='O1-H2')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 2], label='O1-O2')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 3], label='O1-H3')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 4], label='O1-H4')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 5], label='H1-H2')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 6], label='H1-O2')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 7], label='H1-H3')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 8], label='H1-H4')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 9], label='H2-O1')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 10], label='H2-H3')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 11], label='H2-H4')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 12], label='O2-H3')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 13], label='O2-H4')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 14], label='H3-H4')

ax_representation.set_xlabel('Time / fs')
ax_representation.set_ylabel('Representation')
ax_representation.legend()
fig_representation.tight_layout()
# fig_representation.savefig('{}{}'.format(folder_save, '/presentation2/rep_coulomb_power6_inter_train_500.png'),
#                            dpi=parameters.save_dpi, bbbox_inches='tight')


# eta_array = 1.89 * np.logspace(-1, 2, num=12) #np.array([np.logspace(-1, 2, num=12)[i] for i in (0, 1, 2)])
# representation = functions.representation_botu(coord_x, coord_y, coord_z, num_atoms, num_timesteps, eta_array)

# Plot eta dependency of time independent representation
# for value in range(0, 3):
#     ax_representation.plot(eta_array, matrix_coulomb_time[0, value*eta_array.shape[0]:
# (value*eta_array.shape[0])+eta_array.shape[0]])
#     ax_representation.plot(eta_array, matrix_coulomb_time[0, value*eta_array.shape[0]:
# (value*eta_array.shape[0])+eta_array.shape[0]], 'x')

# Plot eta dependence of vector representation
# labels = ['Ox', 'H1x', 'H2x', 'Oy', 'H1y', 'H2y', 'Oz', 'H1z', 'H2z']
# colors = ['r', 'g', 'b', 'r', 'g', 'b', 'r', 'g', 'b']
# for value in range(num_atoms*3):
#
#     ax_representation.plot(eta_array, matrix_coulomb_time[-1, value * eta_array.shape[0]:(value * eta_array.shape[0])
#                                                           + eta_array.shape[0]], colors[value], label=labels[value])
#     ax_representation.plot(eta_array, matrix_coulomb_time[-1, value * eta_array.shape[0]:(value * eta_array.shape[0])
#                                                           + eta_array.shape[0]], '{}{}'.format(colors[value], 'x'))

# Plot eta dependence of scaler representation
# labels = ['O', 'H1', 'H2']
# colors = ['r', 'g', 'b']
# fig_representation, ax_representation = plt.subplots()
# labels = ['O1', 'H1', 'H2', 'O2', 'H3', 'H4']
# colors = ['r', 'g', 'b', 'c', 'm', 'y']
# for value in range(num_atoms):
#     ax_representation.plot(eta_array, matrix_coulomb_time[-1, value * eta_array.shape[0]:(value * eta_array.shape[0])
#                                                           + eta_array.shape[0]], colors[value], label=labels[value])
#     ax_representation.plot(eta_array, matrix_coulomb_time[-1, value * eta_array.shape[0]:(value * eta_array.shape[0])
#                                                           + eta_array.shape[0]], '{}{}'.format(colors[value], 'x'))
# ax_representation.legend()
# ax_representation.set_xlabel('Eta')
# ax_representation.set_ylabel('Representation / a.u.')
# fig_representation.tight_layout()
# fig_representation.savefig('{}{}'.format(folder_save, '/presentation2/eta_cutoff1.7.png'),
#                            dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot damping function
# x_values = np.linspace(0, 10, num=100)
# y_values = np.zeros(100)
# for a in range(0, 100):
#     y_values[a] = functions.damping_function(x_values[a], 10)
# ax_representation.plot(x_values, y_values)
# ax_representation.set_xlabel('Distance / a.u')
# ax_representation.set_ylabel('Cutoff function')
# fig_representation.savefig('{}{}'.format(folder_save, '/presentation2/cutoff_function.png'),
#                            dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot eta dependency of crystal representation
# ax_representation.plot(eta_array, matrix_coulomb_time[0, :], 'r')
# ax_representation.plot(eta_array, matrix_coulomb_time[0, :], 'rx')

# Plot time dependency of representation
# for value in range(0, eta_array.shape[0]):
#     ax_representation.plot(time_val, matrix_coulomb_time[:, (eta_array.shape[0]*0)+value], 'x')

# for value in range(0, matrix_coulomb_time.shape[1]):
#     ax_representation.plot(time_val, matrix_coulomb_time[:, value], label=str(value))

# ax_representation.plot(time_val, matrix_coulomb_time[:, 0], 'rx', label='O-H1x')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 1], 'gx', label='O-H1y')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 2], 'bx', label='O-H1z')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 3], 'rx', label='O-H2x')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 4], 'gx', label='O-H2y')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 5], 'bx', label='O-H2z')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 6], 'rx', label='H-Hx')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 7], 'gx', label='H-Hy')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 8], 'bx', label='H-Hz')

# ax_representation.plot(time_val, matrix_coulomb_time[:, 0], 'rx', label='O')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 1], 'gx', label='H1')
# ax_representation.plot(time_val, matrix_coulomb_time[:, 2], 'bx', label='H2')
#
# for value in range(0, matrix_coulomb_time.shape[1]):
#     ax_representation.plot(time_val, matrix_coulomb_time[:, value], 'x')

# for coulomb in range(0, int(matrix_coulomb_time.shape[1])):
#     ax_representation.plot(time_val, matrix_coulomb_time[:, coulomb], label=str(coulomb))
# ax_representation.set_xlabel('Time / fs')
# ax_representation.set_ylabel('Standardised representation')

# fig_representation.tight_layout()
# fig_representation.savefig('{}{}'.format(folder_save, '/representation/botu_representation.png'),
#                            dpi=parameters.save_dpi, bbbox_inches='tight')

# fig_representation2, ax_representation2 = plt.subplots()
# for value in range(0, eta_array.shape[0]):
#     ax_representation2.plot(time_val, matrix_coulomb_time[:, (eta_array.shape[0]*1)+value])
# ax_representation2.set_xlabel('Time / fs')
# ax_representation2.set_ylabel('Representation / a.u.')
# fig_representation2.tight_layout()
# fig_representation2.savefig('{}{}'.format(folder_save, '/presentation2/eta_time_H1_cutoff10.png'),
#                            dpi=parameters.save_dpi, bbbox_inches='tight')


# Print results of grid search
# print('KRR grid search parameters', krr.best_params_)
# print('GPR grid search parameters', gpr.best_params_)
# print('NN grid search parameters', nn.best_params_, '\n')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
