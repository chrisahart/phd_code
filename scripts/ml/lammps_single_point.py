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
import quippy
import time
import ase
from ase.visualize import view
import subprocess
import qlab
import cp2k_to_gap
import patrick_analysis
import plots_generic
import os
import pandas as pd
import re
from shutil import copyfile
import lammps_to_cp2k
from timeit import default_timer as timer

"""
    LAMMPS analysis scripts.
"""

# Folder paths
folder = \
    '/scratch/cahart/work/personal_files/dft_ml_md/data/h2o_1_quantum/gap/lammps_testing/water_testing/single_points2/'
data_gap = '/gap/data_gap'
in_gap = '/gap/in.gap'
hse_coordinates = 'hse_coordinates.xyz'

# Number of timesteps
num_timesteps = 2000

# Start timer
start = timer()

# Create file structure using: data_gap, in_gap and hse_coordinates
# for timestep in range(num_timesteps):
#
#     if not os.path.exists('{}{}'.format(folder, timestep)):
#         os.makedirs('{}{}'.format(folder, timestep))
#
#     copyfile('{}{}'.format(folder, data_gap), '{}{}{}'.format(folder, timestep, '/data_gap'))
#     copyfile('{}{}'.format(folder, in_gap), '{}{}{}'.format(folder, timestep, '/in.gap'))
#
#     # Read data_gap
#     cols = ['a', 'b', 'X', 'Y', 'Z', 'f', 'g']
#     file_coord = pd.read_csv('{}{}'.format(folder, data_gap), names=cols,
#                              delim_whitespace=True, skip_blank_lines=False)
#
#     file_coord['X'][3] = ''
#     file_coord['b'][3] = 'atom types'
#
#     file_coord['Y'][4, 5, 6] = ''
#     file_coord['X'][4] = 'xlo xhi'
#     file_coord['X'][5] = 'ylo yhi'
#     file_coord['X'][6] = 'zlo zhi'
#
#     # Read HSE coordinates
#     cols = ['Species', 'X', 'Y', 'Z']
#
#     # Read as csv file with whitespace delimiter
#     file_hse = pd.read_csv('{}{}'.format(folder, hse_coordinates), names=cols, delim_whitespace=True)
#     num_atoms = int(file_hse['Species'][0])
#     file_hse = file_hse.apply(pd.to_numeric, errors='coerce')
#     file_hse = file_hse.dropna(axis='rows', thresh=2)
#     file_hse = file_hse.dropna(axis='columns', thresh=1)
#     file_hse = file_hse.reset_index(drop=True)
#
#     # Replace data_gap with HSE_coordinates
#     for atom in range(num_atoms):
#         file_coord['X'][15 + atom] = file_hse['X'][0 + atom + timestep * 3]
#         file_coord['Y'][15 + atom] = file_hse['Y'][0 + atom + timestep * 3]
#         file_coord['Z'][15 + atom] = file_hse['Z'][0 + atom + timestep * 3]
#
#     with open('{}{}{}'.format(folder, timestep, '/data_gap'), 'w+') as f:
#         file_coord.to_csv(f, sep='\t', index=False, header=False)
#
# # Print time taken
# time_folder = timer()
# print('Folder construction:', time_folder - start)
#
# # Call bash script to loop over all folders and call LAMMPS (faster than Python loop)
# os.system('/scratch/cahart/work/personal_files/dft_ml_md/data/h2o_1_quantum/gap/'
#           'lammps_testing/water_testing/single_points2/run_single_points.sh')
#
# # Print time taken
# time_lammps = timer()
# print('LAMMPS:', time_lammps - time_folder)
#
# # Read all LAMMPS files and print to single file
# energy = np.zeros(num_timesteps)
# with open('{}{}'.format(folder, '/forces.xyz'), "a") as force_file:
#     with open('{}{}'.format(folder, '/coordinates.lammps'), "a") as coordinates_file:
#         for timestep in range(num_timesteps):
#             # Save forces
#             forces = lammps_to_cp2k.read_dump('{}{}{}'.format(folder, timestep, '/forces.lammps'),
#                                               '{}{}{}'.format(folder, timestep, '/forces.xyz'))
#             forces.to_csv(force_file, sep=' ', index=False, header=False)
#
#             # Save energy
#             energy[timestep], _ = lammps_to_cp2k.read_energy('{}{}{}'.format(folder, timestep, '/log.lammps'),
#                                                              '{}{}{}'.format(folder, timestep,
#                                                                              '/file_energy_lammps.lammps'))
#
#             # Save coordinates
#             coordinates = lammps_to_cp2k.read_dump('{}{}{}'.format(folder, timestep, '/coordinates.lammps'),
#                                                    '{}{}{}'.format(folder, timestep, '/coordinates.xyz'))
#             coordinates.to_csv(coordinates_file, sep=' ', index=False, header=False)
#
# # Save energy
# np.savetxt('{}{}'.format(folder, '/energy.lammps'), energy, delimiter=',')
#
# # Print time taken
# time_concat = timer()
# print('Concatenating files:', time_concat - time_lammps)

# Read energy, forces and coordinates from files
energy = np.loadtxt('{}{}'.format(folder, '/energy.lammps'), delimiter=',')
forces, _, _, _, _, _ = load_forces.load_values_forces(folder)
coord, coord_x, coord_y, coord_z, species, num_atoms, num_timesteps = \
    load_coordinates.load_values_coord(folder, 'coordinates.lammps')

# Create time array
time_array = np.linspace(0, num_timesteps * 0.25, num_timesteps)

# Plot bond lengths against time
matrix_coulomb_time = functions.calculate_coulomb_matrix(coord_x, coord_y, coord_z, num_atoms, num_timesteps)
fig_bond, ax_bond = plt.subplots(figsize=param.time_figsize)
plots_generic.time_plot(time_array, [matrix_coulomb_time[:, 0], matrix_coulomb_time[:, 1], matrix_coulomb_time[:, 2]],
                        ['OH', 'OH', 'HH'], 'filename', fig_bond, ax_bond, 'Time / ', 'Bond length / ',
                        0, 'folder_save', 'print')

# Plot energy against time
fig_energy, ax_energy = plt.subplots(figsize=param.time_figsize)
plots_generic.time_plot(time_array, [energy],
                        [''], 'filename', fig_energy, ax_energy, 'Time / ', 'Energy / ',
                        0, 'folder_save', 'print')

# Plot force time dependency (Ox)
fig_force, ax_force = plt.subplots(figsize=param.time_figsize)
plots_generic.time_plot(time_array, [forces[:, 0, 0]],
                        [''], 'force_Ox.png',
                        fig_force, ax_force, 'Time / fs', 'Force Ox / au', 0, 'folder_save', 'print')

# Print time taken
# time_plot = timer()
# print('Plotting:', time_plot - time_concat)

plt.show()
