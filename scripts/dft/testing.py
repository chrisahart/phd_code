from __future__ import division, print_function
import pandas as pd
import numpy as np

"""
    Test.
"""

def read_hirsh(folder, filename):
    """
    Read Hirshfeld
    """

    cols_hirsh = ['Atom', 'Element', 'Kind', 'Ref Charge', 'Pop 1', 'Pop 2', 'Spin', 'Charge']
    data_hirsh = pd.read_csv('{}{}'.format(folder, filename), names=cols_hirsh, delim_whitespace=True)
    species = data_hirsh['Element']
    return data_hirsh, species

file = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/cdft/bulk/cdft/prevent-crossing/400K/extrap-0/constraint-bd/analysis/hirshfeld/step-3000_cdft-newton-63_eps-0.2_dft-cg.out'
