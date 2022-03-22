from __future__ import division, print_function, unicode_literals
import pandas as pd
import numpy as np
import glob

"""
    Load forces
"""


def load_file_forces(folder, filename):
    """
        Return CP2K MD .XYZ forces file as Pandas database.
    """

    # Search for all files with path "data/*forces.xyz"
    # files = []
    # for file in glob.glob('{}{}'.format(folder, "/*frc-1.xyz")):
    #     files.append(file)
    #
    # if not files:
    #     print('No files were found, causing program to crash.')

    # Single file
    files = ['{}/{}'.format(folder, filename)]

    # Assign column identities
    cols = ['Species', 'X', 'Y', 'Z']

    # Read as csv file with whitespace delimiter
    file_forces = pd.read_csv(files[0], names=cols, delim_whitespace=True)

    # Force database to numeric, assigning any non-numeric as NaN
    file_forces = file_forces.apply(pd.to_numeric, errors='coerce')

    # Determine number of atoms
    num_atoms = int(file_forces['Species'][0])

    # Filter rows with two or more NaN and columns with one of more NaN, leaving only forces data
    file_forces = file_forces.dropna(axis='rows', thresh=2)
    file_forces = file_forces.dropna(axis='columns', thresh=1)

    return file_forces, num_atoms


def load_values_forces(folder, filename):
    """
        Return CP2K MD .XYZ forces file as re-structured Numpy array.
    """

    # Load forces data from Pandas database
    db_forces, num_atoms = load_file_forces(folder, filename)
    forces_pandas_x = db_forces['X'].values
    forces_pandas_y = db_forces['Y'].values
    forces_pandas_z = db_forces['Z'].values

    # Assign variables
    num_timesteps = int(forces_pandas_x.shape[0] / num_atoms)

    # Initialise arrays
    forces_x = np.zeros((num_timesteps, num_atoms))
    forces_y = np.zeros((num_timesteps, num_atoms))
    forces_z = np.zeros((num_timesteps, num_atoms))
    force = np.zeros((num_timesteps, 3, num_atoms))

    # Loop over each timestep and atoms
    for timestep in range(num_timesteps):
        for atom in range(num_atoms):

            # Re-structure forces arrays
            forces_x[timestep, atom] = forces_pandas_x[atom + timestep * num_atoms]
            forces_y[timestep, atom] = forces_pandas_y[atom + timestep * num_atoms]
            forces_z[timestep, atom] = forces_pandas_z[atom + timestep * num_atoms]

            force[timestep, 0, atom] = forces_pandas_x[atom + timestep * num_atoms]
            force[timestep, 1, atom] = forces_pandas_y[atom + timestep * num_atoms]
            force[timestep, 2, atom] = forces_pandas_z[atom + timestep * num_atoms]

    return force, forces_x, forces_y, forces_z, num_atoms, num_timesteps
