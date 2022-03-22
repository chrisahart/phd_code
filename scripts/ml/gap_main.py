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
import quippy
import time
import ase
from ase.visualize import view
import subprocess
import qlab
import patrick_analysis
import plots_generic

# Folder names
folder_hse_data = '/scratch/cahart/work/personal_files/dft_ml_md/data/h2o_1_quantum'
folder_data = '/scratch/cahart/work/personal_files/dft_ml_md/data/h2o_1_quantum/gap/'
folder_save = '/scratch/cahart/work/personal_files/dft_ml_md/output/h2o_1_quantum/gap/'

# Input filenames
filename_coordinates = '{}{}'.format(folder_data, 'hse_coordinates.xyz')
filename_forces = '{}{}'.format(folder_data, 'hse_forces.xyz')

# Output filenames
filename_gap_model = '{}{}'.format(folder_data, 'gap.xml')
filename_training_data = '{}{}'.format(folder_data, 'gap_training_data.xyz')
filename_validation_data = '{}{}'.format(folder_data, 'gap_validation_data.xyz')
filename_training_output = '{}{}'.format(folder_data, 'gap_training_output.out')
filename_validation_output = '{}{}'.format(folder_data, 'gap_validation_output.out')

# Import HSE data todo tidy these
energy_kinetic, energy_potential, temperature, time_val, time_per_step = load_energy.load_values_energy(folder_hse_data)
force, _, _, _, _, _ = load_forces.load_values_forces(folder_hse_data)
coordinates, _, _, _, _, num_atoms, num_timesteps = load_coordinates.load_values_coord(folder_hse_data,
                                                                                       '*coordinates.xyz')

# todo reduce size of training and validation set to 50 evenly spaced samples from trajectory
# todo look into convergence of training and validation data
# todo get lammps MD working, look into divergence from DFT trajectory with time
# num_timesteps = parameters_sklearn.training_size + 50

# # Make data subfolder if doesn't exist
# if not os.path.exists('{}{}'.format(folder_data, folder_sub)):
#     os.makedirs('{}{}'.format(folder_data, folder_sub))
#
# # Make output subfolder if doesn't exist
# if not os.path.exists(folder_save):
#     os.makedirs(folder_save)
#
# # Start timer
# start = timer()
#
# # Print GAP compatible training and validation data sets
# cp2k_to_gap.print_gap(filename_coordinates, filename_forces, filename_training_data,
#                       0, parameters.training_size)
# cp2k_to_gap.print_gap(filename_coordinates, filename_forces, filename_validation_data,
#                       parameters.training_size, parameters.training_size+50)
#
# # Print time taken
# time_prep = timer()
# print('\n GAP file preparation:', time_prep - start, 's \n')
#
# # Call GAP teach_sparse as subprocess call
# teach_sparse_command = ["/scratch/cahart/software/QUIP/bin/teach_sparse",
#                         "default_sigma={0.05 0.01 0.0 0.0}",
#                         "gap={distance_2b cutoff=4.0 covariance_type=ard_se delta=0.5 "
#                         "theta_uniform=1.0 sparse_method=uniform add_species=T n_sparse=10 "
#                         ": angle_3b cutoff=4.0 n_sparse=200 covariance_type=ard_se delta=3.663 theta_uniform=1.0}",
#                         "{}{}".format('gp_file=', filename_gap_model),
#                         "{}{}".format('at_file=', filename_training_data),
#                         "energy_parameter_name=energy",
#                         "force_parameter_name=force"]
# subprocess.call(teach_sparse_command)
#
# # Print GAP teach_sparse command to file for reference
# with open('{}{}{}'.format(folder_data, folder_sub, 'teach_sparse_command.txt'), "w") as text_file:
#     for item in teach_sparse_command:
#         text_file.write('{}{}'.format(item, ' '))
#
# # Print time taken
# time_gap = timer()
# print('GAP teach_sparse:', time_prep - start)
#
# # Evaluate GAP model using QUIP for training and validation data
# functions.evaluate_gap_model(filename_gap_model, filename_training_data, filename_training_output,
#                              parameters.training_size, num_atoms)
# functions.evaluate_gap_model(filename_gap_model, filename_validation_data, filename_validation_output,
#                              50, num_atoms)

# Load GAP evaluated energy and forces, concatenating training and validation data
training_output = np.loadtxt(filename_training_output, delimiter=',')
validation_output = np.loadtxt(filename_validation_output, delimiter=',')
energy_gap = np.concatenate((training_output[:, 0], validation_output[:, 0]), axis=0)
force_gap = np.concatenate((training_output[:, 1:], validation_output[:, 1:]), axis=0).reshape(num_timesteps, 3, 3)

# Initialise SKL GPR
gpr_energy = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), alpha=1e-9, normalize_y=True)
gpr_force = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), alpha=1e-5, normalize_y=True)
gpr_energy_force = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), alpha=1e-5, normalize_y=True)

# Format SKL training data
# skl_training_data = functions.skl_training_data(num_timesteps, num_atoms, energy_potential, force_hse)
skl_training_data = np.zeros((num_timesteps, 2))
skl_training_data[:, 0] = energy_potential

# Calculate SKL 2b descriptor
matrix_coulomb_time = functions.calculate_descriptor_2b(coordinates, num_atoms, num_timesteps)

# Fit SKL GPR
gpr_energy.fit(matrix_coulomb_time[:parameters.training_size, :], skl_training_data[:parameters.training_size, 0])
gpr_force.fit(matrix_coulomb_time[:parameters.training_size, :], skl_training_data[:parameters.training_size, 1:-1])
gpr_energy_force.fit(matrix_coulomb_time[:parameters.training_size, :], skl_training_data[:parameters.training_size])

# Evaluate SKL GPR
gpr_energy_results = gpr_energy.predict(matrix_coulomb_time)
gpr_force_results = gpr_force.predict(matrix_coulomb_time)
gpr_energy_force_results = gpr_energy_force.predict(matrix_coulomb_time)

# Calculate net force on each atom
force_skl_energy_force = functions.skl_transform_force(gpr_energy_force_results, num_timesteps, num_atoms)
force_skl_force = functions.skl_transform_force(gpr_force_results, num_timesteps, num_atoms)
force_skl_net_energy_force = functions.calculate_force_net(force_skl_energy_force, num_timesteps, num_atoms)
force_skl_net_force = functions.calculate_force_net(force_skl_force, num_timesteps, num_atoms)
force_hse_net = functions.calculate_force_net(force_hse, num_timesteps, num_atoms)
force_gap_net = functions.calculate_force_net(force_gap, num_timesteps, num_atoms)

# Plotting variables
truncate = parameters.training_size + 50 #num_timesteps
opacity_cutoff = parameters.training_size

# Subtract mean HSE energy for plotting purposes (simply translates all plots, doesn't affect fitting)
# translate_energy = np.mean(energy_potential)
# energy_potential = energy_potential - translate_energy
# energy_gap = energy_gap - translate_energy
# gpr_energy_results = gpr_energy_results - translate_energy
# gpr_energy_force_results[:, 0] = gpr_energy_force_results[:, 0] - translate_energy

# Calculate mean errors
energy_error_gap = np.sum(np.abs(energy_potential[:truncate] - energy_gap[:truncate]))/truncate
energy_error_skl = np.sum(np.abs(energy_potential[:truncate] - gpr_energy_results[:truncate]))/truncate

force_error_gap = np.sum(np.abs(force_hse[:truncate, 0, 0] - force_gap[:truncate, 0, 0]))/truncate
force_error_skl = np.sum(np.abs(force_hse[:truncate, 0, 0] - force_skl_force[:truncate, 0, 0]))/truncate

print('energy_error_gap', energy_error_gap*27.2114*1e3)
print('energy_error_skl', energy_error_skl*27.2114*1e3)

print('force_error_gap', force_error_gap*27.2114*1e3)
print('force_error_skl', force_error_skl*27.2114*1e3)

# Plot energy time dependency
fig_energy, ax_energy = plt.subplots(figsize=parameters.time_figsize)
plots_generic.time_plot(time_val[:truncate],
                        [energy_potential[:truncate], energy_gap[:truncate], gpr_energy_force_results[:truncate, 0],
                         gpr_energy_results[:truncate]],
                        ['HSE', 'GAP', 'SKL(E, F)', 'SKL(E)', ], 'energy.png',
                        fig_energy, ax_energy, 'Time / fs', 'Energy / au', opacity_cutoff, folder_save, True)

# plots_generic.time_plot(time_val[:truncate],
#                         [energy_gap[:truncate]],
#                         ['HSE', 'GAP', 'SKL(E, F)', 'SKL(E)', ], 'energy.png',
#                         fig_energy, ax_energy, 'Time / fs', 'Energy / au', opacity_cutoff, folder_save, True)


# Plot force time dependency (Ox)
fig_force, ax_force = plt.subplots(figsize=parameters.time_figsize)
plots_generic.time_plot(time_val[:truncate], [force_hse[:truncate, 0, 0], force_gap[:truncate, 0, 0],
                                              force_skl_force[:truncate, 0, 0]],
                        ['HSE', 'GAP', 'SKL(F)'], 'force_Ox.png',
                        fig_force, ax_force, 'Time / fs', 'Force Ox / au', opacity_cutoff, folder_save, True)

# Plot total force time dependency (O)
fig_force_total, ax_force_total = plt.subplots(figsize=parameters.time_figsize)
plots_generic.time_plot(time_val[:truncate], [force_hse_net[:truncate, 0], force_gap_net[:truncate, 0],
                                              force_skl_net_force[:truncate, 0]],
                        ['HSE', 'GAP', 'SKL(F)'], 'force_total_O.png',
                        fig_force_total, ax_force_total, 'Time / fs', 'Total force O / au',
                        opacity_cutoff, folder_save, True)

# Plot energy parity
fig_parity_energy, ax_parity_energy = plt.subplots(figsize=parameters.parity_figsize)
plots_generic.parity_plot(energy_potential[:truncate], [energy_gap[:truncate],
                                                        gpr_energy_results[:truncate]],
                          ['GAP', 'SKL(E)', ], 'parity_energy.png',
                          fig_parity_energy, ax_parity_energy, 'Energy / au', 'GAP  energy / au',
                          opacity_cutoff, folder_save)

# Plot force parity (Ox)
fig_parity_force, ax_parity_force = plt.subplots(figsize=parameters.parity_figsize)
plots_generic.parity_plot(force_hse[:truncate, 0, 0],
                          [force_gap[:truncate, 0, 0],
                           force_skl_force[:truncate, 0, 0]],
                          ['GAP', 'SKL(F)'], 'parity_force.png',
                          fig_parity_force, ax_parity_force,
                          'DFT force Ox / au', 'ML force Ox / au', opacity_cutoff, folder_save)

# Plot total force parity (O)
fig_parity_force_total, ax_parity_force_total = plt.subplots(figsize=parameters.parity_figsize)
plots_generic.parity_plot(force_hse_net[:truncate, 0],
                          [force_gap_net[:truncate, 0],
                           force_skl_net_force[:truncate, 0]],
                          ['GAP', 'SKL'], 'parity_force_total.png',
                          fig_parity_force_total, ax_parity_force_total,
                          'Total force O / au', 'GAP total force O / au')
fig_parity_force_total.savefig('{}{}'.format(folder_save, 'parity_force_total.png'),
                          dpi=param.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
