from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob

"""
    Load coordinates.
    Load .xyz coordinates file in given folder with given filename.
"""


def load_file_coord(folder, filename):
    """
        Return CP2K MD .XYZ coordinate file as Pandas database.
    """

    # Search for all files with path "data/*coordinates.xyz"
    # files = []
    # for file in glob.glob('{}{}{}'.format(folder, '/', filename)):
    #     files.append(file)
    #
    # if not files:
    #     print('\n No files were found, causing program to crash. \n')

    # Single file
    files = ['{}/{}'.format(folder, filename)]

    # Assign column identities
    cols = ['Species', 'X', 'Y', 'Z']

    # Read as csv file with whitespace delimiter
    file_coord = pd.read_csv(files[0], names=cols, delim_whitespace=True)

    # Determine number of atoms
    num_atoms = int(float(file_coord['Species'][0]))

    # Save species as separate Series
    # species = file_coord['Species'][1:num_atoms + 1]
    species = file_coord['Species'][2:num_atoms + 2]

    # Force database to numeric, assigning any non-numeric as NaN
    file_coord = file_coord.apply(pd.to_numeric, errors='coerce')

    # Filter rows with two or more NaN and columns with one of more NaN, leaving only coordinate data
    file_coord = file_coord.dropna(axis='rows', thresh=2)
    file_coord = file_coord.dropna(axis='columns', thresh=1)

    return file_coord, num_atoms, species


def load_values_coord(folder, filename):
    """
        Return CP2K MD .XYZ file as Numpy array.
    """

    # Load coordinate data from Pandas database
    db_coord, num_atoms, species = load_file_coord(folder, filename)
    coord_pandas_x = db_coord['X'].values
    coord_pandas_y = db_coord['Y'].values
    coord_pandas_z = db_coord['Z'].values

    # Assign variables
    num_timesteps = int(coord_pandas_x.shape[0] / num_atoms)

    # Initialise arrays
    coord_x = np.zeros((num_timesteps, num_atoms))
    coord_y = np.zeros((num_timesteps, num_atoms))
    coord_z = np.zeros((num_timesteps, num_atoms))
    coord = np.zeros((num_timesteps, 3, num_atoms))

    # Loop over each timestep and atoms
    for timestep in range(num_timesteps):
        for atom in range(num_atoms):

            # Re-structure coordinate arrays
            coord_x[timestep, atom] = coord_pandas_x[atom + timestep * num_atoms]
            coord_y[timestep, atom] = coord_pandas_y[atom + timestep * num_atoms]
            coord_z[timestep, atom] = coord_pandas_z[atom + timestep * num_atoms]

            coord[timestep, 0, atom] = coord_pandas_x[atom + timestep * num_atoms]
            coord[timestep, 1, atom] = coord_pandas_y[atom + timestep * num_atoms]
            coord[timestep, 2, atom] = coord_pandas_z[atom + timestep * num_atoms]

    return coord, coord_x, coord_y, coord_z, species, num_atoms, num_timesteps
