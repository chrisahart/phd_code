from __future__ import division, print_function, unicode_literals
import pandas as pd
import numpy as np
import glob

"""
    Load forces out
"""


def load_file_forces(folder, num_atoms):
    """
        Return CP2K FORCE EVAL custom .out forces file as Pandas database.
        todo modify batch script to read specify number of atoms at start of file, read in pandas
    """

    # Search for all files with path "data/*forces.xyz"
    files = []
    for file in glob.glob('{}{}'.format(folder, "/*forces.out")):
        files.append(file)

    if not files:
        print('No files were found, causing program to crash.')

    # Assign column identities
    cols = ['a', 'b', 'c', 'X', 'Y', 'Z', 'd']

    # Read as csv file with whitespace delimiter
    file_forces = pd.read_csv(files[0], names=cols, delim_whitespace=True, skiprows=2)

    # Force database to numeric, assigning any non-numeric as NaN
    file_forces = file_forces.apply(pd.to_numeric, errors='coerce')

    # Determine number of atoms
    # num_atoms = int(file_forces['a'][3])

    # Filter rows with two or more NaN and columns with one of more NaN, leaving only forces data
    file_forces = file_forces.dropna(axis='rows', thresh=2)
    file_forces = file_forces.dropna(axis='columns', thresh=1)
    file_forces = file_forces.drop(["a", "b"], axis=1)

    return file_forces, num_atoms


def load_values_forces(folder, num_atoms):
    """
        Return CP2K MD .XYZ forces file as re-structured Numpy array.
    """

    # Load forces data from Pandas database
    db_forces, num_atoms = load_file_forces(folder, num_atoms)
    forces_pandas_x = db_forces['X'].values
    forces_pandas_y = db_forces['Y'].values
    forces_pandas_z = db_forces['Z'].values

    # Assign variables
    num_timesteps = int(forces_pandas_x.shape[0] / num_atoms)

    # Initialise arrays
    forces_x = np.zeros((num_timesteps, num_atoms))
    forces_y = np.zeros((num_timesteps, num_atoms))
    forces_z = np.zeros((num_timesteps, num_atoms))

    # Loop over each timestep and atoms
    for timestep in range(num_timesteps):
        for atom in range(num_atoms):

            # Re-structure forces arrays
            forces_x[timestep, atom] = forces_pandas_x[atom + timestep * num_atoms]
            forces_y[timestep, atom] = forces_pandas_y[atom + timestep * num_atoms]
            forces_z[timestep, atom] = forces_pandas_z[atom + timestep * num_atoms]

    return forces_x, forces_y, forces_z, num_atoms, num_timesteps
