from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
from scripts.general import parameters 

"""
    Load .cube
    Load .cube coordinates file in given folder with given filename
    Allows for direct analysis of .cube file as .xyz without converting file
"""


def load_file_coord(folder, filename):
    """
        Return CP2K MD .XYZ coordinate file as Pandas database.
    """

    # Search for all files with path "data/*coordinates.xyz"
    files = []
    for file in glob.glob('{}{}{}'.format(folder, '/', filename)):
        files.append(file)

    if not files:
        print('\n No files were found, causing program to crash. \n')

    # Assign column identities
    cols = ['a', 'b', 'X', 'Y', 'Z']

    # Read as csv file with whitespace delimiter
    file_coord = pd.read_csv(files[0], names=cols, delim_whitespace=True)

    # Read number of atoms from header
    num_atoms = int(file_coord['a'][2])

    # Read displacement from header
    disp_x = float(file_coord['b'][2])
    disp_y = float(file_coord['X'][2])
    disp_z = float(file_coord['Y'][2])

    # Displace atoms
    file_coord['X'] = file_coord['X'] - disp_x
    file_coord['Y'] = file_coord['Y'] - disp_y
    file_coord['Z'] = file_coord['Z'] - disp_z

    # Drop data related to orbital density
    file_coord = file_coord[6:num_atoms+6]
    file_coord = file_coord.drop(['a', 'b'], 1)

    # Convert from Bohr (default in cube files) to Angstrom
    file_coord = file_coord / parameters.angstrom_to_bohr

    # Force database to numeric, assigning any non-numeric as NaN
    file_coord = file_coord.apply(pd.to_numeric, errors='coerce')

    # Filter rows with two or more NaN and columns with one of more NaN, leaving only coordinate data
    file_coord = file_coord.dropna(axis='rows', thresh=2)
    file_coord = file_coord.dropna(axis='columns', thresh=1)

    return file_coord, num_atoms


def load_values_coord(folder, filename):
    """
        Return CP2K MD .XYZ file as re-structured Numpy array.
    """

    # Load coordinate data from Pandas database
    db_coord, num_atoms = load_file_coord(folder, filename)
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

    return coord, coord_x, coord_y, coord_z, num_atoms, num_timesteps
