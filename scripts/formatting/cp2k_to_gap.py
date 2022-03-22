from __future__ import division, print_function
import time
import numpy as np
import pandas as pd
import os

"""
    Functions for re-formatting CP2K to GAP.
"""


def print_gap(filename_coordinates, filename_forces, filename_output, start, end):
    """ Read CP2K MD output files and print to GAP compatible .xyz file """

    # Read number of atoms and labels from .xyz file
    cols = ['Species', 'X', 'Y', 'Z', 'a', 'Time', 'c', 'd', 'Energy']

    # Read coordinates and energies
    file_coord = pd.read_csv(filename_coordinates, names=cols, delim_whitespace=True)
    file_coord = file_coord.drop(['a', 'c', 'd'], 1)

    # Read forces
    file_force = pd.read_csv(filename_forces, names=cols, delim_whitespace=True)
    file_force = file_force.drop(['a', 'c', 'd'], 1)

    # Read number of atoms and timesteps
    num_atoms = int(file_coord['Species'][0])
    num_timesteps = int(file_coord.shape[0]/(num_atoms+2))

    # Remove output file if present
    if os.path.isfile(filename_output):
        os.remove(filename_output)

    # Join two pandas databases, add GAP header and append to csv (could be optimised)
    with open(filename_output, 'a') as f:
        for timestep in range(start, end):

            # Current value
            val = timestep * num_atoms + timestep * 2

            # Concatonate pandas databases
            file_coord_n = file_coord[2 + val:2 + num_atoms + val].drop(['Time', 'Energy'], 1)
            file_force_f = file_force[2 + val:2 + num_atoms + val].drop(['Species', 'Time', 'Energy'], 1)
            result = pd.concat([file_coord_n, file_force_f], axis=1, sort=False)

            # Add header of number of atoms, followed by Lattice parameters, Properties, energy and PBC boolean
            text = '{}{}{}{}{}'.format('Lattice="9.0 0.0 0.0 0.0 9.0 0.0 0.0 0.0 9.0"',
                                       ' Properties=species:S:1:pos:R:3:force:R:3',
                                       ' energy=', file_coord['Energy'][1 + val],
                                       ' pbc="T T T"')

            df = pd.DataFrame([[num_atoms, '', '', '', '', '', ''],
                               [text, '', '', '', '']],
                              columns=['Species', 'X', 'Y', 'Z', 'X', 'Y', 'Z'])

            # Concatonate header to pandas database
            result = df.append(result, ignore_index=True, sort=False)

            # Append to .csv file
            result.to_csv(f, sep='\t', index=False, header=False, quotechar=' ')
