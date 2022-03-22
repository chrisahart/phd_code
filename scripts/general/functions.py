from __future__ import division, print_function
import time
import numpy as np
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
import copy
from sklearn.preprocessing import StandardScaler
import random


"""
    General functions
"""


def damping_function(distance, cutoff):
    if distance < cutoff:
        damp = 0.5 * (np.cos((np.pi * distance / cutoff)) + 1)

    else:
        damp = 0

    return damp


def random_sign():
    if random.random() < 0.5:
        return 1
    else:
        return -1


def training_set_convergence(matrix_coulomb_time, basis_set_size, data):
    """
        Training set size convergence
    """

    # Assign variables
    num_timesteps = matrix_coulomb_time.shape[0]
    num_steps = basis_set_size.shape[0]

    # Allocate arrays
    krr_error = np.zeros(num_steps)
    gpr_error = np.zeros(num_steps)
    nn_error = np.zeros(num_steps)
    krr_time = np.zeros(num_steps)
    gpr_time = np.zeros(num_steps)
    nn_time = np.zeros(num_steps)

    # Loop over number of steps
    for step in range(num_steps):
        # Reset ML
        krr = copy.copy(param.krr)
        gpr = copy.copy(param.gpr)

        print('Loop number', (step + 1), '/', num_steps)
        start = time.time()

        # Calculate KRR
        krr.fit(matrix_coulomb_time[:basis_set_size[step], :], data_standardised[:basis_set_size[step]])
        krr_values = krr.predict(matrix_coulomb_time)
        krr_values = (scaler.inverse_transform(krr_values))
        krr_error[step] = (np.sum((np.abs(krr_values[:, 0] - data[:, 0]))) / (num_timesteps * 1))

        krr_time[step] = time.time() - start

        # Calculate GPR
        gpr.fit(matrix_coulomb_time[:basis_set_size[step], :], data_standardised[:basis_set_size[step]])
        gpr_values = gpr.predict(matrix_coulomb_time)
        gpr_values = (scaler.inverse_transform(gpr_values))
        gpr_error[step] = (np.sum((np.abs(gpr_values[:, 0] - data[:, 0]))) / (num_timesteps * 1))
        gpr_time[step] = time.time() - start

        # Print results of grid search
        # print('KRR grid search parameters', krr.best_params_)
        # print('GPR grid search parameters', gpr.best_params_)

    return krr_error, gpr_error, krr_time, gpr_time


# def upper_tri(a):
#     m = a.shape[0]
#     r = np.arange(m)
#     mask = r[:, None] < r
#     return a[mask]


def upper_tri(a):
    a = a[np.triu_indices(a.shape[0], k = 1)]
    return a


def distances_md(coord_x, coord_y, coord_z, num_atoms, num_timesteps):

    # Calculate distances between atoms for MD trajectory
    matrix_distance_x = np.zeros((num_atoms, num_atoms))
    matrix_distance_y = np.zeros((num_atoms, num_atoms))
    matrix_distance_z = np.zeros((num_atoms, num_atoms))
    matrix_distance_r = np.zeros((num_atoms, num_atoms))
    matrix_distance_r_time = np.zeros((num_timesteps, int(num_atoms*(num_atoms-1)/2)))
    matrix_distance_x_time = np.zeros((num_timesteps, int(num_atoms*(num_atoms-1)/2)))
    matrix_distance_y_time = np.zeros((num_timesteps, int(num_atoms*(num_atoms-1)/2)))
    matrix_distance_z_time = np.zeros((num_timesteps, int(num_atoms*(num_atoms-1)/2)))

    for timestep in range(num_timesteps):
        for atom_1 in range(num_atoms):
            for atom_2 in range(num_atoms):

                if atom_1 != atom_2:
                    # Calculate atomic separation
                    matrix_distance_x[atom_1, atom_2] = (coord_x[timestep, atom_1] - coord_x[timestep, atom_2])
                    matrix_distance_y[atom_1, atom_2] = (coord_y[timestep, atom_1] - coord_y[timestep, atom_2])
                    matrix_distance_z[atom_1, atom_2] = (coord_z[timestep, atom_1] - coord_z[timestep, atom_2])
                    matrix_distance_r[atom_1, atom_2] = np.sqrt(matrix_distance_x[atom_1, atom_2] ** 2.0 +
                                                                matrix_distance_y[atom_1, atom_2] ** 2.0 +
                                                                matrix_distance_z[atom_1, atom_2] ** 2.0)

        matrix_distance_r_time[timestep, :] = upper_tri(matrix_distance_r)
        matrix_distance_x_time[timestep, :] = upper_tri(matrix_distance_x)
        matrix_distance_y_time[timestep, :] = upper_tri(matrix_distance_y)
        matrix_distance_z_time[timestep, :] = upper_tri(matrix_distance_z)

    return matrix_distance_r_time, matrix_distance_x_time, matrix_distance_y_time, matrix_distance_z_time


def calculate_force_net(forces, num_timesteps, num_atoms):
    force_net = np.zeros((num_timesteps, num_atoms))
    for atom in range(num_atoms):
        force_net[:, atom] = np.sqrt(forces[0:num_timesteps, atom, 0] ** 2 +
                                     forces[0:num_timesteps, atom, 1] ** 2 +
                                     forces[0:num_timesteps, atom, 2] ** 2)
    return force_net


def calculate_force_net_skl(target, num_timesteps, num_atoms):
    force_net = np.zeros((num_timesteps, num_atoms))
    for atom in range(num_atoms):
        force_net[:, atom] = np.sqrt(target[0:num_timesteps, (atom * 3) + 1] ** 2 \
                                     + target[0:num_timesteps, (atom * 3) + 2] ** 2 \
                                     + target[0:num_timesteps, (atom * 3) + 3] ** 2)

    return force_net


def skl_transform_force(target, num_timesteps, num_atoms):
    start = target.shape[1] - (3 * num_atoms)
    force = np.zeros((num_timesteps, num_atoms, 3))
    for atom in range(num_atoms):
        for direction in range(3):
            force[:, atom, direction] = target[0:num_timesteps, (atom * 3) + start + direction]

    return force


# def calculate_force_net(force_x, force_y, force_z, num_timesteps, num_atoms):
#     force_net = np.zeros((num_timesteps, num_atoms))
#     for atom in range(num_atoms):
#         force_net[:, atom] = np.sqrt(force_x[0:num_timesteps, atom] ** 2 +
#                                      force_y[0:num_timesteps, atom] ** 2 +
#                                      force_z[0:num_timesteps, atom] ** 2)
#     return force_net


def calc_distance(x1, y1, z1, x2, y2, z2):
    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    return distance


def truncate_func(truncate_list):
    """
        Truncate data to desired length
    """

    for i in range(0, len(truncate_list)):
        truncate_list[i] = truncate_list[i][0:num_timesteps]
    return truncate_list


def evaluate_gap_model(filename_gap, filename_data, filename_output, simulation_length, num_atoms):
    """ Evaluate GAP model using QUIP """

    # Initialize QUIP interatomic potential with GAP file as input
    potential = qlab.Potential('IP GAP', param_filename=filename_gap)

    # Initialize arrays
    forces = np.zeros((simulation_length, 3, num_atoms))
    energy = np.zeros(simulation_length)
    output_energy = np.zeros(1)
    output_force = np.zeros((3, num_atoms), order='F')

    # Loop over number of timesteps
    for i in range(simulation_length):
        print('Evaluating timestep', i, '/', simulation_length - 1)

        # Read input data into AtomsList
        input_data = qlab.AtomsList(filename_data)[i]

        # Evaluate energy and force using QUIP
        potential.calc(input_data, energy=output_energy, force=output_force)

        # Save force and energies
        forces[i, :] = output_force
        energy[i] = output_energy

    # Concatonate energy and force data, save to file
    data = np.concatenate((energy.reshape(energy.shape[0], 1), forces.reshape(forces.shape[0], 3 * num_atoms)), axis=1)
    np.savetxt(filename_output, data, delimiter=',')

    print('Finished evaluating GAP model. \n')


def skl_training_data(num_timesteps, num_atoms, energy_potential, force):
    """" Format training data for scikit-learn"""

    target = np.zeros((num_timesteps, 1 + num_atoms * 3))
    target[:, 0] = energy_potential
    for atom in range(num_atoms):
        target[:, (atom * 3) + 1] = force[:, 0, atom]
        target[:, (atom * 3) + 2] = force[:, 1, atom]
        target[:, (atom * 3) + 3] = force[:, 2, atom]

    return target


def calc_bond_lengths(coordinates, num_atoms):
    """" Calculates bond lengths of coordinates, returning NxN array"""

    # Construct NxN matrix of bond lengths
    bond_lengths = np.zeros((num_atoms, num_atoms))
    for a in range(num_atoms):
        for b in range(num_atoms):
            bond_lengths[a, b] = calc_distance(coordinates[0, 0, a], coordinates[0, 1, a], coordinates[0, 2, a],
                                               coordinates[0, 0, b], coordinates[0, 1, b], coordinates[0, 2, b])

    return bond_lengths


def layer_identifier(coordinates, species, decimal_places, axis, atom_type):
    """" Takes coordinates, atom label and axis and returns array of layer value for each atom of type specified"""

    # Detect unique layers
    layers, layers_indices = np.unique(coordinates[0, axis, :], return_inverse=True)

    # Number of atoms
    num_atoms = coordinates.shape[2]

    # Extract locations of iron atoms
    iron_layers = np.zeros(num_atoms)
    count = 0
    for atom in range(num_atoms):

        if species.iloc[atom] == atom_type:
            print(atom_type, atom, layers[layers_indices[atom]])
            iron_layers[count] = layers[layers_indices[atom]]
            count = count + 1

    iron_layers = iron_layers[0:count]
    iron_layer, layer_indices = np.unique(iron_layers, return_inverse=True)
    layer_diff = np.zeros(iron_layer.shape[0] - 1)
    iron_layer_truncated = np.copy(iron_layer)

    # If layers are within tolerance set coordinate to bottom of layer
    for i in range(0, iron_layer.shape[0] - 1):

        layer_diff[i] = np.round(iron_layer[i + 1] - iron_layer[i], decimal_places)

        if layer_diff[i] < 1:
            iron_layer_truncated[i + 1] = iron_layer_truncated[i]

    iron_layer_final = iron_layer_truncated[layer_indices]
    unique_layer, unique_index = np.unique(iron_layer_final, return_inverse=True)

    return unique_index


def extract_cp2k_log(filename, search_string):
    """"" Extract from CP2k log file"""""

    temp = []

    # Search for string and return numbers from each line
    for line in open(filename).read().splitlines():

        if search_string in line:
            line_numeric = float(line.split()[-1])
            temp.append(line_numeric)

    return np.array(temp)