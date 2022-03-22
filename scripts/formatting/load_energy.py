from __future__ import division, print_function, unicode_literals
import pandas as pd
import glob

"""
    Load .ener
"""


def load_file_energy(folder, filename):
    """
        Return CP2K MD .ener file as Pandas database.
    """

    # # Search for all files with path "data/*.ener"
    # files = []
    # for file in glob.glob('{}{}'.format(folder, "/*.ener")):
    #     files.append(file)
    #
    # if not files:
    #     print('No files were found, causing program to crash.')

    files = ['{}/{}'.format(folder, filename)]

    # Assign column identities
    cols = ['Step', 'Time', 'E_kin', 'Temp', 'E_pot', 'E_tot', 'Time_per_step']

    # Read as csv file with whitespace delimiter
    file_energy = pd.read_csv(files[0], delim_whitespace=True, names=cols, skiprows=[0])

    return file_energy


def load_values_energy(folder, filename):
    """
        Return CP2K MD .ener file as re-structured Numpy array.
    """

    # Load energy data from Pandas database
    db_energy = load_file_energy(folder, filename)
    energy_kinetic = db_energy['E_kin'].values
    energy_potential = db_energy['E_pot'].values
    energy_total = db_energy['E_tot'].values
    temperature = db_energy['Temp'].values
    time = db_energy['Time'].values
    time_per_step = db_energy['Time_per_step'].values

    return energy_kinetic, energy_potential, energy_total, temperature, time, time_per_step
