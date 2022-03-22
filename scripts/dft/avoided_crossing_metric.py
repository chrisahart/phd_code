from __future__ import division, print_function
import pandas as pd
import numpy as np

"""
    Avoided crossing strength
"""

def read_hirsh(folder, filename):
    """
    Read Hirshfeld charges
    """

    cols_hirsh = ['Atom', 'Element', 'Kind', 'Ref Charge', 'Pop 1', 'Pop 2', 'Spin', 'Charge']
    data_hirsh = pd.read_csv('{}{}'.format(folder, filename), names=cols_hirsh, delim_whitespace=True)
    species = data_hirsh['Element']
    return data_hirsh, species


def calc_strength(diff):
    """
    Calculate CDFT strength (Lagrange multiplier)
    """
    metric = max_strength * np.tanh(diff**2)
    return metric


folder = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/cdft/bulk/cdft/prevent-crossing/400K/extrap-0/constraint-bd/analysis/hirshfeld'
hirshfeld, species = read_hirsh(folder, '//step-3000_cdft-newton-63_eps-0.2_dft-cg.out')

atoms = 120
skip_start_2 = 3
skip_end_2 = 5
max_strength = 0.002
num_trajectories = int(len(hirshfeld)/(120+5))

fe_b = np.array([14, 16, 18, 42, 27, 45, 25, 29]) - 1
fe_d = np.array([6, 2, 13, 17, 38, 4, 15, 41]) - 1
fe_f = np.array([46, 28, 5, 1, 30, 26, 37, 3]) - 1

hirshfeld_fe_b = np.zeros(8)
hirshfeld_fe_d = np.zeros(8)
hirshfeld_fe_f = np.zeros(8)
value = num_trajectories - 1
for j in range(len(fe_f)):
    hirshfeld_fe_b[j] = (hirshfeld.loc[skip_start_2 + atoms * value + skip_end_2 * value + fe_b[j], 'Spin'])
    hirshfeld_fe_d[j] = (hirshfeld.loc[skip_start_2 + atoms * value + skip_end_2 * value + fe_b[j], 'Spin'])
    hirshfeld_fe_f[j] = (hirshfeld.loc[skip_start_2 + atoms * value + skip_end_2 * value + fe_b[j], 'Spin'])

mean_spin = -62.97
current_spin = np.sum(hirshfeld_fe_b)+np.sum(hirshfeld_fe_d)
diff = current_spin - mean_spin

strength = 0
if diff > 0:
    strength = calc_strength(diff)

# print('current_spin', current_spin)
# print('diff', diff)
print('strength', strength)
