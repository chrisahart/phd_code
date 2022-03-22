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
import lammps_to_cp2k

"""
    LAMMPS analysis scripts.
"""

# Read log.lammps and write to energy.ener
# Read coordinates.lammps and write to coordinates.xyz (CP2K/VMD readable)
# Read forces.lammps and write to forces.xyz

file_log_lammps = \
    '/scratch/cahart/work/personal_files/dft_ml_md/data/h2o_1_quantum/gap/lammps_testing/water_testing/log.lammps'

file_energy_lammps = \
    '/scratch/cahart/work/personal_files/dft_ml_md/data/h2o_1_quantum/gap/lammps_testing/water_testing/energy.lammps'

file_coordinates_lammps = \
    '/scratch/cahart/work/personal_files/dft_ml_md/data/h2o_1_quantum/gap/lammps_testing/water_testing/coordinates.lammps'

file_coordinates = \
    '/scratch/cahart/work/personal_files/dft_ml_md/data/h2o_1_quantum/gap/lammps_testing/water_testing/coordinates.xyz'

file_forces_lammps = \
    '/scratch/cahart/work/personal_files/dft_ml_md/data/h2o_1_quantum/gap/lammps_testing/water_testing/forces.lammps'

file_forces = \
    '/scratch/cahart/work/personal_files/dft_ml_md/data/h2o_1_quantum/gap/lammps_testing/water_testing/forces.xyz'

folder_data = '/scratch/cahart/work/personal_files/dft_ml_md/data/h2o_1_quantum/gap/lammps_testing/water_testing/'

# Remove Pandas warning
pd.options.mode.chained_assignment = None

# # Call bash script to call lammps
# os.system('/scratch/cahart/work/personal_files/dft_ml_md/data/h2o_1_quantum/gap/lammps_testing/water_testing/lammps.sh')

# Read LAMMPS energy
energy, num_timesteps = lammps_to_cp2k.read_energy(file_log_lammps, file_energy_lammps)

# Read LAMMPS coordinates and forces
forces = lammps_to_cp2k.read_dump(file_forces_lammps, file_forces)
coordinates = lammps_to_cp2k.read_dump(file_coordinates_lammps, file_coordinates)
coord, coord_x, coord_y, coord_z, species, num_atoms, num_timesteps = \
    load_coordinates.load_values_coord(folder_data, 'coordinates.xyz')

# Create time array
num_timesteps = 2001
time_array = np.linspace(0, num_timesteps * 0.25, num_timesteps)

print('forces', forces.shape)
forces, _, _, _, _, _ = load_forces.load_values_forces(folder_data)
print('forces', forces.shape)

# Plot bond lengths against time
matrix_coulomb_time = functions.calculate_coulomb_matrix(coord_x, coord_y, coord_z, num_atoms, num_timesteps)
fig_bond, ax_bond = plt.subplots(figsize=param.parity_figsize)
plots_generic.time_plot(time_array, [matrix_coulomb_time[:, 0], matrix_coulomb_time[:, 1], matrix_coulomb_time[:, 2]],
                        ['OH', 'OH', 'HH'], 'filename', fig_bond, ax_bond, 'Time / ', 'Bond length / ',
                        0, 'folder_save', 'print')

# Plot energy against time
fig_energy, ax_energy = plt.subplots(figsize=param.parity_figsize)
plots_generic.time_plot(time_array, [energy],
                        [''], 'filename', fig_energy, ax_energy, 'Time / ', 'Energy / ',
                        0, 'folder_save', 'print')

# Plot force time dependency (Ox)
fig_force, ax_force = plt.subplots(figsize=param.parity_figsize)
plots_generic.time_plot(time_array, [forces[:, 0, 0]],
                        [''], 'force_Ox.png',
                        fig_force, ax_force, 'Time / ', 'Forces / ',
                        0, 'folder_save', 'print')

plt.show()
