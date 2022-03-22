from __future__ import division, print_function, unicode_literals
import pandas as pd
import numpy as np
import glob

"""
    Load forces
"""


def load_file_forces(folder, filename, num_atoms, filename_brent, filename_mnbrack):
    """
        Return CP2K MD .XYZ forces file as Pandas database.
    """

    cols = ['1', '2', 'X', 'Y', 'Z']
    data = pd.read_csv('{}{}'.format(folder, filename), names=cols, delim_whitespace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    num_data = int(np.floor((len(data) + 1) / (num_atoms + 2)))
    step = np.linspace(start=0, stop=(num_data - 1), num=num_data, dtype=int)
    brent = np.zeros(num_data)
    mnbrack = np.zeros(num_data)

    if filename_brent:
        cols_brent = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        data_brent = pd.read_csv('{}{}'.format(folder, filename_brent), names=cols_brent, delim_whitespace=True)
        data_mnbrack = pd.read_csv('{}{}'.format(folder, filename_mnbrack), names=cols_brent, delim_whitespace=True)
        brent = data_brent['9']
        mnbrack = data_mnbrack['9']
        num_data = len(brent)
        step = np.linspace(start=0, stop=(num_data - 1), num=num_data, dtype=int)

    return data, num_data, step, brent, mnbrack

