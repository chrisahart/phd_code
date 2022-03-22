from __future__ import division, print_function, unicode_literals
import pandas as pd
import numpy as np
import glob

"""
    Load velocity
"""


def load_file_velocity(folder):
    """
        Return CP2K MD .XYZ velocity file as Pandas database.
    """

    # Search for all files with path "data/*velocity.xyz"
    files = []
    for file in glob.glob('{}{}'.format(folder, "/*velocity.xyz")):
        files.append(file)

    if not files:
        print('No files were found, causing program to crash.')

    # Assign column identities
    cols = ['Species', 'X', 'Y', 'Z']

    # Read as csv file with whitespace delimiter
    file_velocity = pd.read_csv(files[0], names=cols, delim_whitespace=True)

    # Force database to numeric, assigning any non-numeric as NaN
    file_velocity = file_velocity.apply(pd.to_numeric, errors='coerce')

    # Determine number of atoms
    num_atoms = int(file_velocity['Species'][0])

    # Filter rows with two or more NaN and columns with one of more NaN, leaving only velocity data
    file_velocity = file_velocity.dropna(axis='rows', thresh=2)
    file_velocity = file_velocity.dropna(axis='columns', thresh=1)

    return file_velocity, num_atoms


def load_values_velocity(folder):
    """
        Return CP2K MD .XYZ velocity file as re-structured Numpy array.
    """

    # Load velocity data from Pandas database
    db_velocity, num_atoms = load_file_velocity(folder)
    velocity_pandas_x = db_velocity['X'].values
    velocity_pandas_y = db_velocity['Y'].values
    velocity_pandas_z = db_velocity['Z'].values

    # Assign variables
    num_timesteps = int(velocity_pandas_x.shape[0] / num_atoms)

    # Initialise arrays
    velocity_x = np.zeros((num_timesteps, num_atoms))
    velocity_y = np.zeros((num_timesteps, num_atoms))
    velocity_z = np.zeros((num_timesteps, num_atoms))

    # Loop over each timestep and atoms
    for timestep in range(num_timesteps):
        for atom in range(num_atoms):

            # Re-structure velocity arrays
            velocity_x[timestep, atom] = velocity_pandas_x[atom + timestep * num_atoms]
            velocity_y[timestep, atom] = velocity_pandas_y[atom + timestep * num_atoms]
            velocity_z[timestep, atom] = velocity_pandas_z[atom + timestep * num_atoms]

    return velocity_x, velocity_y, velocity_z, num_atoms, num_timesteps
