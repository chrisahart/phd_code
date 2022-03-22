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
import time
import copy
from scripts.general import parameters
from scripts.general import functions


"""
    Representations
"""


def calculate_distances(coord_x, coord_y, coord_z, num_atoms, num_timesteps):
    """
        Calculate original Coulomb matrix.
    """

    # Initialise matrices
    matrix_coulomb = np.zeros((num_atoms, num_atoms))
    size_triangle = int(((num_atoms ** 2) - num_atoms) / 2)
    matrix_coulomb_time = np.zeros((num_timesteps, num_atoms, num_atoms))

    # folder_data = 'data/h2o_1_quantum'
    # pbe_energy_potential = np.loadtxt('{}{}'.format(folder_data, '/pbe_energy.out'), skiprows=1)
    # pbe_forces_x, pbe_forces_y, pbe_forces_z, _, _ = load_forces_out.load_values_forces(folder_data)
    #
    # pbe_energy_potential *= parameters.hartree_to_ev
    # pbe_forces_x *= parameters.hartree_per_bohr_to_ev_per_angstrom
    # pbe_forces_y *= parameters.hartree_per_bohr_to_ev_per_angstrom
    # pbe_forces_z *= parameters.hartree_per_bohr_to_ev_per_angstrom

    # Loop over number of timesteps and atoms
    for timestep in range(num_timesteps):
        for atom_1 in range(num_atoms):
            for atom_2 in range(num_atoms):

                if atom_1 == atom_2:

                    # Self-interaction (atomic units)
                    matrix_coulomb[atom_1, atom_1] = 0.5 * (parameters.atom_charge[atom_1] ** 2.4)

                else:

                    # Calculate atomic separation
                    x_dist = coord_x[timestep, atom_1] - coord_x[timestep, atom_2]
                    y_dist = coord_y[timestep, atom_1] - coord_y[timestep, atom_2]
                    z_dist = coord_z[timestep, atom_1] - coord_z[timestep, atom_2]
                    distance = np.sqrt(x_dist ** 2.0 + y_dist ** 2.0 + z_dist ** 2.0)

                    # Coulomb repulsion (atomic units)
                    # matrix_coulomb[atom_1, atom_2] = distance
                    # matrix_coulomb[atom_1, atom_2] = (parameters.atom_charge[atom_1] * parameters.atom_charge[atom_2]) / \
                    #                                  ((distance ** 2) / parameters.bohr)
                    # matrix_coulomb[atom_1, atom_2] = 1 / (distance ** 6)
                    matrix_coulomb[atom_1, atom_2] = distance

        # Store upper triangle of instantaneous Coulomb matrix
        # matrix_coulomb_time = matrix_coulomb_time + np.sin(timestep / 10) * 1e-3
        # matrix_coulomb_time[timestep, :] = np.ravel(matrix_coulomb)
        matrix_coulomb_time[timestep, :, :] = matrix_coulomb

    return matrix_coulomb_time


def calculate_coulomb_matrix(coord_x, coord_y, coord_z, num_atoms, num_timesteps):
    """
        Calculate original Coulomb matrix.
    """

    # Initialise matrices
    matrix_coulomb = np.zeros((num_atoms, num_atoms))
    size_triangle = int(((num_atoms ** 2) - num_atoms) / 2)
    matrix_coulomb_time = np.zeros((num_timesteps, size_triangle))

    # folder_data = 'data/h2o_1_quantum'
    # pbe_energy_potential = np.loadtxt('{}{}'.format(folder_data, '/pbe_energy.out'), skiprows=1)
    # pbe_forces_x, pbe_forces_y, pbe_forces_z, _, _ = load_forces_out.load_values_forces(folder_data)
    #
    # pbe_energy_potential *= parameters.hartree_to_ev
    # pbe_forces_x *= parameters.hartree_per_bohr_to_ev_per_angstrom
    # pbe_forces_y *= parameters.hartree_per_bohr_to_ev_per_angstrom
    # pbe_forces_z *= parameters.hartree_per_bohr_to_ev_per_angstrom

    # Loop over number of timesteps and atoms
    for timestep in range(num_timesteps):
        for atom_1 in range(num_atoms):
            for atom_2 in range(num_atoms):

                if atom_1 == atom_2:

                    # Self-interaction (atomic units)
                    matrix_coulomb[atom_1, atom_1] = 0.5 * (parameters.atom_charge[atom_1] ** 2.4)

                else:

                    # Calculate atomic separation
                    x_dist = coord_x[timestep, atom_1] - coord_x[timestep, atom_2]
                    y_dist = coord_y[timestep, atom_1] - coord_y[timestep, atom_2]
                    z_dist = coord_z[timestep, atom_1] - coord_z[timestep, atom_2]
                    distance = np.sqrt(x_dist ** 2.0 + y_dist ** 2.0 + z_dist ** 2.0)

                    # Coulomb repulsion (atomic units)
                    # matrix_coulomb[atom_1, atom_2] = distance
                    # matrix_coulomb[atom_1, atom_2] = (parameters.atom_charge[atom_1] * parameters.atom_charge[atom_2]) / \
                    #                                  ((distance ** 2) / parameters.bohr)
                    matrix_coulomb[atom_1, atom_2] = (parameters.atom_charge[atom_1] * parameters.atom_charge[atom_2]) / distance
                    # matrix_coulomb[atom_1, atom_2] = 1 / (distance ** 1)
                    # matrix_coulomb[atom_1, atom_2] = 1 / distance
                    # matrix_coulomb[atom_1, atom_2] = distance

                    # matrix_coulomb[atom_1, atom_2] = 1 / (distance ** 6)
                    # matrix_coulomb[atom_1, atom_2] = distance

        # Store upper triangle of instantaneous Coulomb matrix
        # matrix_coulomb_time = matrix_coulomb_time + np.sin(timestep / 10) * 1e-3
        # matrix_coulomb_time[timestep, :] = np.ravel(matrix_coulomb)
        # vals = functions.upper_tri(matrix_coulomb)
        # matrix_coulomb_time[timestep, 0] = np.abs(vals[0]+vals[1])
        # matrix_coulomb_time[timestep, 1] = np.abs(vals[0]-vals[1])
        # matrix_coulomb_time[timestep, 2] = vals[2]
        # matrix_coulomb_time[timestep, :] = np.sort(functions.upper_tri(matrix_coulomb))
        matrix_coulomb_time[timestep, :] = functions.upper_tri(matrix_coulomb)
        # matrix_coulomb_time[timestep, :] = [functions.upper_tri(matrix_coulomb)[i] for i in (0, 1, 5, 12, 13, 14)]
        # matrix_coulomb_time[timestep, :] = [functions.upper_tri(matrix_coulomb)[i] for i in (0, 1, 5, 12, 13, 14)]
        # matrix_coulomb_time[timestep, :] = np.hstack(([functions.upper_tri(matrix_coulomb)[i] for i in (0, 1, 5, 12, 13, 14)],
        #                                                   timestep + np.sin(timestep) * 1e2))

        # matrix_coulomb_time[timestep, :] = np.hstack((pbe_energy_potential[timestep],
        #                                               functions.upper_tri(matrix_coulomb)))

        # three_body_1 = [functions.upper_tri(matrix_coulomb)[0] + functions.upper_tri(matrix_coulomb)[1],
        #                 (functions.upper_tri(matrix_coulomb)[0] - functions.upper_tri(matrix_coulomb)[1]) ** 2,
        #                 functions.upper_tri(matrix_coulomb)[2]]
        #
        # three_body_2 = [functions.upper_tri(matrix_coulomb)[1] + functions.upper_tri(matrix_coulomb)[2],
        #                 (functions.upper_tri(matrix_coulomb)[1] - functions.upper_tri(matrix_coulomb)[2]) ** 2,
        #                 functions.upper_tri(matrix_coulomb)[0]]
        #
        # three_body_3 = [functions.upper_tri(matrix_coulomb)[2] + functions.upper_tri(matrix_coulomb)[0],
        #                 (functions.upper_tri(matrix_coulomb)[2] - functions.upper_tri(matrix_coulomb)[0]) ** 2,
        #                 functions.upper_tri(matrix_coulomb)[1]]

        # matrix_coulomb_time[timestep, :] = np.hstack((pbe_energy_potential[timestep],
        #                                               functions.upper_tri(matrix_coulomb)))

        # matrix_coulomb_time[timestep, :] = np.hstack((pbe_energy_potential[timestep],
        #                                               pbe_forces_x[timestep, 0],
        #                                               pbe_forces_x[timestep, 1],
        #                                               pbe_forces_x[timestep, 2],
        #                                               pbe_forces_y[timestep, 0],
        #                                               pbe_forces_y[timestep, 1],
        #                                               pbe_forces_y[timestep, 2],
        #                                               pbe_forces_z[timestep, 0],
        #                                               pbe_forces_z[timestep, 1],
        #                                               pbe_forces_z[timestep, 2],
        #                                               three_body_1, three_body_2, three_body_3,
        #                                               functions.upper_tri(matrix_coulomb)))

        # matrix_coulomb_time[timestep, :] = np.hstack((three_body_1, three_body_2, three_body_3,
        #                                               functions.upper_tri(matrix_coulomb)))

        # matrix_coulomb_time[timestep, :] = np.hstack((pbe_energy_potential[timestep],
        #                                               pbe_forces_x[timestep, 0],
        #                                               pbe_forces_x[timestep, 1],
        #                                               pbe_forces_x[timestep, 2],
        #                                               pbe_forces_y[timestep, 0],
        #                                               pbe_forces_y[timestep, 1],
        #                                               pbe_forces_y[timestep, 2],
        #                                               pbe_forces_z[timestep, 0],
        #                                               pbe_forces_z[timestep, 1],
        #                                               pbe_forces_z[timestep, 2],
        #                                               functions.upper_tri(matrix_coulomb)))

    return matrix_coulomb_time


def calculate_descriptor_2b(coordinates, num_atoms, num_timesteps):
    """
        Calculate original Coulomb matrix.
    """

    # Initialise matrices
    matrix_coulomb = np.zeros((num_atoms, num_atoms))
    size_triangle = int(((num_atoms ** 2) - num_atoms) / 2)
    matrix_coulomb_time = np.zeros((num_timesteps, size_triangle))

    # Loop over number of timesteps and atoms
    for timestep in range(num_timesteps):
        for atom_1 in range(num_atoms):
            for atom_2 in range(num_atoms):

                if atom_1 == atom_2:

                    # Self-interaction (atomic units)
                    matrix_coulomb[atom_1, atom_1] = 0.5 * (parameters.atom_charge[atom_1] ** 2.4)

                else:

                    # Calculate atomic separation
                    x_dist = coordinates[timestep, 0, atom_1] - coordinates[timestep, 0, atom_2]
                    y_dist = coordinates[timestep, 1, atom_1] - coordinates[timestep, 1, atom_2]
                    z_dist = coordinates[timestep, 2, atom_1] - coordinates[timestep, 2, atom_2]
                    distance = np.sqrt(x_dist ** 2.0 + y_dist ** 2.0 + z_dist ** 2.0)
                    matrix_coulomb[atom_1, atom_2] = 1 / distance

        matrix_coulomb_time[timestep, :] = functions.upper_tri(matrix_coulomb)

    return matrix_coulomb_time


def representation_scaler(coord_x, coord_y, coord_z, num_atoms, num_timesteps,
                          energy_potential, force_x, force_y, force_z):
    """
        Calculate original Coulomb matrix.
    """

    # Initialise matrices
    matrix_coulomb = np.zeros((num_atoms, num_atoms))
    size_triangle = int(((num_atoms ** 2) - num_atoms) / 2)
    size_triangle_3 = int(((size_triangle ** 2) - size_triangle) / 2)
    matrix_coulomb_time = np.zeros((num_timesteps, size_triangle))
    # matrix_coulomb_time = np.zeros((num_timesteps, size_triangle+1+size_triangle_3*2))
    three_body_p = np.zeros((size_triangle, size_triangle))
    three_body_m = np.zeros((size_triangle, size_triangle))

    # Loop over number of timesteps and atoms
    for timestep in range(num_timesteps):

        # to calculate two body interactions
        for atom_1 in range(num_atoms):
            for atom_2 in range(num_atoms):

                if atom_1 == atom_2:

                    # Self-interaction (atomic units)
                    matrix_coulomb[atom_1, atom_1] = 0.5 * (parameters.atom_charge[atom_1] ** 2.4)

                else:

                    # Calculate atomic separation
                    x_dist = coord_x[timestep, atom_1] - coord_x[timestep, atom_2]
                    y_dist = coord_y[timestep, atom_1] - coord_y[timestep, atom_2]
                    z_dist = coord_z[timestep, atom_1] - coord_z[timestep, atom_2]
                    distance = np.sqrt(x_dist ** 2.0 + y_dist ** 2.0 + z_dist ** 2.0)
                    matrix_coulomb[atom_1, atom_2] = distance

        distances = functions.upper_tri(matrix_coulomb)

        # to calculate three body interactions
        for distance_1 in range(size_triangle):
            for distance_2 in range(size_triangle):
                if distance_1 != distance_2:
                    # calculate sum of distances
                    three_body_p[distance_1, distance_2] = distances[distance_1] + distances[distance_2]
                    three_body_m[distance_1, distance_2] = (distances[distance_1] - distances[distance_2]) ** 2

        # Store upper triangle of instantaneous Coulomb matrix
        matrix_coulomb_time[timestep, :] = functions.upper_tri(matrix_coulomb)
        # matrix_coulomb_time[timestep, :] = [functions.upper_tri(matrix_coulomb)[i] for i in (0, 1, 5, 12, 13, 14)]

        # matrix_coulomb_time[timestep, :] = np.hstack((1 * energy_potential[timestep],
        #                                               1 * functions.upper_tri(matrix_coulomb),
        #                                               1 * functions.upper_tri(three_body_p),
        #                                               1 * functions.upper_tri(three_body_m)))

        # matrix_coulomb_time[timestep, :] = pbe_energy_potential[timestep]

        # three_body_1 = [functions.upper_tri(matrix_coulomb)[0] + functions.upper_tri(matrix_coulomb)[1],
        #                 (functions.upper_tri(matrix_coulomb)[0] - functions.upper_tri(matrix_coulomb)[1]) ** 2,
        #                 functions.upper_tri(matrix_coulomb)[2]]
        #
        # three_body_2 = [functions.upper_tri(matrix_coulomb)[1] + functions.upper_tri(matrix_coulomb)[2],
        #                 (functions.upper_tri(matrix_coulomb)[1] - functions.upper_tri(matrix_coulomb)[2]) ** 2,
        #                 functions.upper_tri(matrix_coulomb)[0]]
        #
        # three_body_3 = [functions.upper_tri(matrix_coulomb)[2] + functions.upper_tri(matrix_coulomb)[0],
        #                 (functions.upper_tri(matrix_coulomb)[2] - functions.upper_tri(matrix_coulomb)[0]) ** 2,
        #                 functions.upper_tri(matrix_coulomb)[1]]
        #
        # matrix_coulomb_time[timestep, :] = np.hstack((three_body_1, three_body_2, three_body_3,
        #                                               functions.upper_tri(matrix_coulomb)))

        # Energy, forces, representation
        # target = np.zeros(1 + num_atoms * 3 + size_triangle)
        # target[0] = energy_potential[timestep]
        # for atom in range(num_atoms):
        #     target[(atom * 3) + 1] = force_x[timestep, atom]
        #     target[(atom * 3) + 2] = force_y[timestep, atom]
        #     target[(atom * 3) + 3] = force_z[timestep, atom]
        # target[target.shape[0]-size_triangle:] = functions.upper_tri(matrix_coulomb)
        # matrix_coulomb_time[timestep, :] = target

    return matrix_coulomb_time


def calculate_feature_vector(coord_x, coord_y, coord_z, num_atoms, num_timesteps):
    """
        Calculate feature vector using internal vectors of molecule
    """

    # Initialise matrices
    matrix_distance_x = np.zeros((num_atoms, num_atoms))
    matrix_distance_y = np.zeros((num_atoms, num_atoms))
    matrix_distance_z = np.zeros((num_atoms, num_atoms))
    size_triangle = int(((num_atoms ** 2) - num_atoms) / 2)
    matrix_coulomb_time = np.zeros((num_timesteps, 6 * size_triangle))

    # Loop over each timesteps and all atoms
    for timestep in range(num_timesteps):
        for atom_1 in range(num_atoms):
            for atom_2 in range(num_atoms):
                matrix_distance_x[atom_1, atom_2] = coord_x[timestep, atom_1] - coord_x[timestep, atom_2]
                matrix_distance_y[atom_1, atom_2] = coord_y[timestep, atom_1] - coord_y[timestep, atom_2]
                matrix_distance_z[atom_1, atom_2] = coord_z[timestep, atom_1] - coord_z[timestep, atom_2]

        # Store upper triangle of matrix
        matrix_coulomb_time[timestep, :] = np.concatenate((functions.upper_tri(matrix_distance_x),
                                                           functions.upper_tri(np.transpose(matrix_distance_x)),
                                                           functions.upper_tri(matrix_distance_y),
                                                           functions.upper_tri(np.transpose(matrix_distance_y)),
                                                           functions.upper_tri(matrix_distance_z),
                                                           functions.upper_tri(np.transpose(matrix_distance_z))), axis=0)

        # matrix_coulomb_time[timestep, :] = np.hstack(((coord_x[timestep, 0]),
        #                                               (coord_x[timestep, 1]),
        #                                               (coord_x[timestep, 2]),
        #                                               (coord_y[timestep, 0]),
        #                                               (coord_y[timestep, 1]),
        #                                               (coord_y[timestep, 2]),
        #                                               (coord_z[timestep, 0]),
        #                                               (coord_z[timestep, 1]),
        #                                               (coord_z[timestep, 2])))

    return matrix_coulomb_time


def representation_botu(coord_x, coord_y, coord_z, num_atoms, num_timesteps, eta_array, cutoff):
    # Initialise matrices
    matrix_coulomb = np.zeros((num_atoms, num_atoms))
    x_distances = np.zeros((num_atoms, num_atoms))
    y_distances = np.zeros((num_atoms, num_atoms))
    z_distances = np.zeros((num_atoms, num_atoms))
    size_triangle = int(((num_atoms ** 2) - num_atoms) / 2)
    matrix_coulomb_time = np.zeros((num_timesteps, size_triangle))
    x_distances_time = np.zeros((num_timesteps, size_triangle))
    y_distances_time = np.zeros((num_timesteps, size_triangle))
    z_distances_time = np.zeros((num_timesteps, size_triangle))

    # Loop over number of timesteps and atoms
    for timestep in range(num_timesteps):

        # to calculate two body interactions
        for atom_1 in range(num_atoms):
            for atom_2 in range(num_atoms):
                if atom_1 == atom_2:

                    # Self-interaction (atomic units)
                    matrix_coulomb[atom_1, atom_1] = 0.5 * (parameters.atom_charge[atom_1] ** 2.4)

                else:

                    # Calculate atomic separation
                    x_dist = coord_x[timestep, atom_1] - coord_x[timestep, atom_2]
                    y_dist = coord_y[timestep, atom_1] - coord_y[timestep, atom_2]
                    z_dist = coord_z[timestep, atom_1] - coord_z[timestep, atom_2]
                    distance = np.sqrt(x_dist ** 2.0 + y_dist ** 2.0 + z_dist ** 2.0)

                    matrix_coulomb[atom_1, atom_2] = distance
                    x_distances[atom_1, atom_2] = x_dist
                    y_distances[atom_1, atom_2] = y_dist
                    z_distances[atom_1, atom_2] = z_dist

        matrix_coulomb_time[timestep, :] = functions.upper_tri(matrix_coulomb)
        x_distances_time[timestep, :] = functions.upper_tri(x_distances)
        y_distances_time[timestep, :] = functions.upper_tri(y_distances)
        z_distances_time[timestep, :] = functions.upper_tri(z_distances)

    distances_array = matrix_coulomb_time
    representation = np.zeros((num_timesteps, (eta_array.shape[0] * size_triangle)))
    representation_x = np.zeros((num_timesteps, (eta_array.shape[0] * size_triangle)))
    representation_y = np.zeros((num_timesteps, (eta_array.shape[0] * size_triangle)))
    representation_z = np.zeros((num_timesteps, (eta_array.shape[0] * size_triangle)))

    # Calculate eta resolved Botu fingerprint from atomic distances
    # for distance in range(0, num_atoms):
    #     for timestep in range(num_timesteps):
    #         for eta in range(0, eta_array.shape[0]):
    #
    #             representation[timestep, (distance*eta_array.shape[0])+eta] = \
    #                 np.exp((-(distances_array[timestep, distance] / eta_array[eta]) ** 2)) * \
    #                 1 #damping_function(distances_array[timestep, distance], 8)

    # Calculate eta resolved Botu fingerprint from atomic distances (many body) SCALER FINGERPRINT
    representation = np.zeros((num_timesteps, eta_array.shape[0] * num_atoms))
    for atom in range(0, num_atoms):
        for timestep in range(num_timesteps):
            for neighbour in range(0, num_atoms):

                if atom != neighbour:

                    # Calculate distance between atom and neighbour
                    x_dist = coord_x[timestep, atom] - coord_x[timestep, neighbour]
                    y_dist = coord_y[timestep, atom] - coord_y[timestep, neighbour]
                    z_dist = coord_z[timestep, atom] - coord_z[timestep, neighbour]
                    distance = np.sqrt(x_dist ** 2.0 + y_dist ** 2.0 + z_dist ** 2.0)

                    for eta in range(0, eta_array.shape[0]):
                        # Calculate representation for particular atom, timestep, neighbour and eta value
                        representation[timestep, (atom * eta_array.shape[0]) + eta] += \
                            np.exp((-(distance / eta_array[eta]) ** 2)) * \
                            damping_function(distance, parameters.distance_cutoff)
    # representation = representation / np.max(representation)

    # Calculate eta summed Botu fingerprint from atomic distances
    # representation = np.zeros((num_timesteps, size_triangle))
    # for distance in range(0, num_atoms):
    #     for timestep in range(num_timesteps):
    #         for eta in range(0, eta_array.shape[0]):
    #
    #             representation[timestep, distance] += \
    #                 np.exp((-(distances_array[timestep, distance] / eta_array[eta]) ** 2)) * \
    #                 damping_function(distances_array[timestep, distance], 5)

    # Calculate eta summed Botu fingerprint from atomic distances (many body)
    # representation = np.zeros((num_timesteps, num_atoms))
    # for atom in range(0, num_atoms):
    #     for timestep in range(num_timesteps):
    #         for neighbour in range(0, num_atoms):
    #
    #             if atom != neighbour:
    #
    #                 # Calculate distance between atom and neighbour
    #                 x_dist = coord_x[timestep, atom] - coord_x[timestep, neighbour]
    #                 y_dist = coord_y[timestep, atom] - coord_y[timestep, neighbour]
    #                 z_dist = coord_z[timestep, atom] - coord_z[timestep, neighbour]
    #                 distance = np.sqrt(x_dist ** 2.0 + y_dist ** 2.0 + z_dist ** 2.0)
    #
    #                 for eta in range(0, eta_array.shape[0]):
    #
    #                     # Calculate representation for particular atom, timestep, neighbour and eta value
    #                     representation[timestep, atom] += \
    #                         np.exp((-(distance / eta_array[eta]) ** 2)) * \
    #                         damping_function(distance, 5)
    #                     print('distance', distance)
    #                     print('((-(distance / eta_array[eta]) ** 2)) ', ((-(distance / eta_array[eta]) ** 2)) )
    #                     print('representation', representation[timestep])
    #
    # representation = representation / (eta_array.shape[0] * num_atoms)

    # Calculate pairwise eta summed Botu fingerprint from atomic distances
    # representation = np.zeros((num_timesteps, size_triangle * 3))
    # for distance in range(0, num_atoms):
    #     for timestep in range(num_timesteps):
    #         for eta in range(0, eta_array.shape[0]):
    #
    #             representation[timestep, distance] += \
    #                 (x_distances_time[timestep, distance] / distances_array[timestep, distance]) * \
    #                 np.exp((-(distances_array[timestep, distance] / eta_array[eta]) ** 2)) * \
    #                 damping_function(distances_array[timestep, distance], 5)
    #
    #             representation[timestep, distance + num_atoms] += \
    #                 (y_distances_time[timestep, distance] / distances_array[timestep, distance]) * \
    #                 np.exp((-(distances_array[timestep, distance] / eta_array[eta]) ** 2)) * \
    #                 damping_function(distances_array[timestep, distance], 5)
    #
    #             representation[timestep, distance+num_atoms*2] += \
    #                 (z_distances_time[timestep, distance] / distances_array[timestep, distance]) * \
    #                 np.exp((-(distances_array[timestep, distance] / eta_array[eta]) ** 2)) * \
    #                 damping_function(distances_array[timestep, distance], 5)

    # Calculate many body eta summed Botu fingerprint from atomic distances
    # representation = np.zeros((num_timesteps, num_atoms*3))
    # for atom in range(0, num_atoms):
    #     for timestep in range(num_timesteps):
    #         for neighbour in range(0, num_atoms):
    #
    #             if atom != neighbour:
    #
    #                 # Calculate distance between atom and neighbour
    #                 x_dist = coord_x[timestep, atom] - coord_x[timestep, neighbour]
    #                 y_dist = coord_y[timestep, atom] - coord_y[timestep, neighbour]
    #                 z_dist = coord_z[timestep, atom] - coord_z[timestep, neighbour]
    #                 distance = np.sqrt(x_dist ** 2.0 + y_dist ** 2.0 + z_dist ** 2.0)
    #
    #                 for eta in range(0, eta_array.shape[0]):
    #
    #                     # Calculate representation for particular atom, timestep, neighbour and eta value
    #                     representation[timestep, atom] += \
    #                         (x_dist / distance) * \
    #                         np.exp((-(distance / eta_array[eta]) ** 2)) * \
    #                         damping_function(distance, 5)
    #
    #                     representation[timestep, atom + num_atoms] += \
    #                         (y_dist / distance) * \
    #                         np.exp((-(distance / eta_array[eta]) ** 2)) * \
    #                         damping_function(distance, 5)
    #
    #                     representation[timestep, atom + num_atoms * 2] += \
    #                         (z_dist / distance) * \
    #                         np.exp((-(distance / eta_array[eta]) ** 2)) * \
    #                         damping_function(distance, 5)
    #
    # representation = representation / (eta_array.shape[0] * num_atoms)

    # Calculate eta resolved direction resolved Botu fingerprint from atomic distances (many body) VECTOR FINGERPRINT
    # representation = np.zeros((num_timesteps, eta_array.shape[0] * num_atoms * 3))
    # for atom in range(0, num_atoms):
    #     for timestep in range(num_timesteps):
    #         for neighbour in range(0, num_atoms):
    #
    #             if atom != neighbour:
    #
    #                 # Calculate distance between atom and neighbour
    #                 x_dist = coord_x[timestep, atom] - coord_x[timestep, neighbour]
    #                 y_dist = coord_y[timestep, atom] - coord_y[timestep, neighbour]
    #                 z_dist = coord_z[timestep, atom] - coord_z[timestep, neighbour]
    #                 distance = np.sqrt(x_dist ** 2.0 + y_dist ** 2.0 + z_dist ** 2.0)
    #
    #                 for eta in range(0, eta_array.shape[0]):
    #
    #                     representation[timestep, (atom * eta_array.shape[0] + eta)*1] = \
    #                         (x_dist / distance) * \
    #                         np.exp((-(distance / eta_array[eta]) ** 2)) * \
    #                         damping_function(distance, cutoff)
    #
    #                     representation[timestep, (atom * eta_array.shape[0] + eta)+eta_array.shape[0] * num_atoms * 1] = \
    #                         (y_dist / distance) * \
    #                         np.exp((-(distance / eta_array[eta]) ** 2)) * \
    #                         damping_function(distance, cutoff)
    #
    #                     representation[timestep, (atom * eta_array.shape[0]+ eta)+eta_array.shape[0] * num_atoms * 2] = \
    #                         (z_dist / distance) * \
    #                         np.exp((-(distance / eta_array[eta]) ** 2)) * \
    #                         damping_function(distance, cutoff)

    # Calculate total eta summed Botu fingerprint from atomic distances
    # representation = np.zeros((num_timesteps, 1))
    # for distance in range(0, num_atoms):
    #     for timestep in range(num_timesteps):
    #         for eta in range(0, eta_array.shape[0]):
    #
    #             representation[timestep] += \
    #                 np.exp((-(distances_array[timestep, distance] / eta_array[eta]) ** 2)) * \
    #                 damping_function(distances_array[timestep, distance], 8)
    # representation = representation / num_atoms

    # Calculate total eta summed Botu fingerprint from atomic distances (many body)
    # representation = np.zeros((num_timesteps, 1))
    # for atom in range(0, num_atoms):
    #     for timestep in range(num_timesteps):
    #         for neighbour in range(0, num_atoms):
    #
    #             # Calculate distance between atom and neighbour
    #             x_dist = coord_x[timestep, atom] - coord_x[timestep, neighbour]
    #             y_dist = coord_y[timestep, atom] - coord_y[timestep, neighbour]
    #             z_dist = coord_z[timestep, atom] - coord_z[timestep, neighbour]
    #             distance = np.sqrt(x_dist ** 2.0 + y_dist ** 2.0 + z_dist ** 2.0)
    #
    #             for eta in range(0, eta_array.shape[0]):
    #
    #                 # Calculate representation for particular atom, timestep, neighbour and eta value
    #                 representation[timestep] += \
    #                     np.exp((-(distance / eta_array[eta]) ** 2)) * \
    #                     damping_function(distance, 5)
    # representation = representation / (num_atoms ** 2 * eta_array.shape[0])

    # Calculate total eta resolved Botu fingerprint from atomic distances (many body) MOLECULAR FINGERPRINT
    # representation = np.zeros((num_timesteps, eta_array.shape[0]))
    # for eta in range(0, eta_array.shape[0]):
    #     for atom in range(0, num_atoms):
    #         for timestep in range(num_timesteps):
    #             for neighbour in range(0, num_atoms):
    #
    #                 if atom != neighbour:
    #
    #                     # Calculate distance between atom and neighbour
    #                     x_dist = coord_x[timestep, atom] - coord_x[timestep, neighbour]
    #                     y_dist = coord_y[timestep, atom] - coord_y[timestep, neighbour]
    #                     z_dist = coord_z[timestep, atom] - coord_z[timestep, neighbour]
    #                     distance = np.sqrt(x_dist ** 2.0 + y_dist ** 2.0 + z_dist ** 2.0)
    #
    #                     # Calculate representation for particular atom, timestep, neighbour and eta value
    #                     representation[timestep, eta] += \
    #                         np.exp((-(distance / eta_array[eta]) ** 2)) * \
    #                         damping_function(distance, cutoff)
    # representation = representation / np.max(representation)

    print('representation \n', representation[0])
    return representation
