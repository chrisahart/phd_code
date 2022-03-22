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

"""
    Functions for reformatting LAMMPS to CP2K 
"""


def get_lines_between(infile):
    """ Extract result between in and out """

    with open(infile) as fp:
        result = re.search('Step Temp E_pair E_mol TotEng Press (.*)Loop time of ', fp.read(), re.S)

    return result.group(1)


def read_energy(file_log_lammps, file_energy_lammps):
    """ Extract energies from file_log_lammps to file_energy_lammps, then return energy and timesteps"""

    test = get_lines_between(file_log_lammps)
    with open(file_energy_lammps, "w") as text_file:
        text_file.write(test)

    # Read number of atoms and labels from .xyz file
    cols = ['Step', 'Temperature', 'Energy', 'E_mol', 'TotEng', 'Press']

    # Read coordinates and energies
    file_coord = pd.read_csv(file_energy_lammps, names=cols, delim_whitespace=True)
    energy = file_coord['Energy']
    num_timesteps = int(file_coord.shape[0])

    return energy, num_timesteps


def read_dump(file_coordinates_lammps, file_coordinates):
    """ Extract energies from file_coordinates_lammps to file_coordinates, then return energy and timesteps"""

    # Read number of atoms and labels from .xyz file
    cols = ['ID', 'TYPE', 'X', 'Y', 'Z']

    # Read coordinates and energies
    file_coordinates_lammps = pd.read_csv(file_coordinates_lammps, names=cols, delim_whitespace=True)

    # Read number of atoms
    num_atoms = int(file_coordinates_lammps['ID'][3])

    # Force database to numeric, assigning any non-numeric as NaN
    file_coordinates_lammps = file_coordinates_lammps.apply(pd.to_numeric, errors='coerce')

    # Filter rows with two or more NaN and columns with one of more NaN, leaving only coordinate data
    file_coordinates_lammps = file_coordinates_lammps.dropna(axis='rows', thresh=3, inplace=False)
    file_coordinates_lammps = file_coordinates_lammps.reset_index(drop=True)
    file_coordinates_lammps = file_coordinates_lammps.drop(['ID'], 1)

    # Calculate number of timesteps
    num_timesteps = file_coordinates_lammps.shape[0]/num_atoms

    # Replace LAMMPS atom type with atom species
    for i in range(file_coordinates_lammps.shape[0]):

        if file_coordinates_lammps['TYPE'][i] == 1.0:
            file_coordinates_lammps['TYPE'][i] = 'O'

        elif file_coordinates_lammps['TYPE'][i] == 2.0:
            file_coordinates_lammps['TYPE'][i] = 'H'

    # Add number of atoms
    for i in range(num_timesteps):
        file_coordinates_lammps = pd.DataFrame(np.insert(file_coordinates_lammps.values, 3*i+0+2*i,
                                                         values=['', '3', '', ''], axis=0))
        file_coordinates_lammps = pd.DataFrame(np.insert(file_coordinates_lammps.values, 3*i+1+2*i,
                                                         values=['', '', '', ''], axis=0))

    # Save forces
    with open(file_coordinates, 'a') as f:
        file_coordinates_lammps.to_csv(f, sep='\t', index=False, header=False)

    return file_coordinates_lammps



