from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
import csv

""" Print xyz. 
    Functions for printing .xyz given different inputs. """


def print_from_pandas(coord_xyz, num_atoms, filename_output):
    """ Print xyz from pandas dataframe of species and coordinates, adding header of number of atoms. """

    # Add number of atoms to header with blank line
    num_atoms = int(num_atoms)
    coord_xyz.loc[-1] = [None, None, None, None]
    coord_xyz.index = coord_xyz.index + 1
    coord_xyz = coord_xyz.sort_index()
    coord_xyz.loc[-1] = [str(num_atoms), None, None, None]  # Use string as Pandas would convert int to float
    coord_xyz.index = coord_xyz.index + 1
    coord_xyz = coord_xyz.sort_index()
    coord_xyz.to_csv(filename_output, index=False,header=False, quoting=csv.QUOTE_NONE, sep=" ")


def print_from_cube_xyz(filename_xyz, filename_cube, filename_output):
    """ Print xyz from species from .xyz and coordinates from .cube (not very useful). """

    # Read number of atoms and labels from .xyz file
    cols = ['Species', 'X', 'Y', 'Z']
    file_spec = pd.read_csv(filename_xyz, names=cols, delim_whitespace=True)
    num_atoms = int(file_spec['Species'][0])
    species = file_spec['Species'][1:num_atoms+1]

    # Read coordinates from .cube file
    cols = ['a', 'b', 'X', 'Y', 'Z']
    file_coord = pd.read_csv(filename_cube, names=cols, delim_whitespace=True)
    num_atoms = int(file_coord['a'][2])
    file_coord = file_coord[6:num_atoms + 6]
    file_coord = file_coord.drop(['a', 'b'], 1)
    file_coord = file_coord / param.angstrom_to_bohr
    file_coord = file_coord.apply(pd.to_numeric, errors='coerce')
    file_coord = file_coord.dropna(axis='rows', thresh=2)
    file_coord = file_coord.dropna(axis='columns', thresh=1)

    # Insert atom labels to database of coordinates
    file_coord.insert(loc=0, column='A', value=pd.Series(species).values)

    # Print atom labels and coordinates to file
    file_coord.to_csv(filename_output, sep=' ', index=False, header=False)

