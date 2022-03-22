from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh
import matplotlib.pyplot as plt


""" xyz average bonds. 
    Calculates average bond lengths for each bulk iron oxide. """


def rmse(x):
    return np.sqrt(np.mean((x-np.mean(x))**2))


# Filename
# folder = '/scratch/cahart/work/personal_files/feIV_bulk/geothite/final/hf18/316_supercell/neutral_cubes'
# folder = '/scratch/cahart/work/personal_files/feIV_bulk/hematite/convergence/multigrid_tight/331/exp_vesta_m700'
# folder = '/scratch/cahart/work/personal_files/fe_bulk/hematite/221_supercell/scan/neutral_scan_custom_pseudo_m1200'
# input_filename = 'geom.xyz'

# folder = 'E:/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/'
# input_filename = 'hematite_221.xyz'

# folder = 'E:/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/221_supercell/neutral'
# input_filename = 'neutral.xyz'

folder = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/geo-opt/neutral/density/lumo_s1'
input_filename = 'hematite-WFN_01021_1-1_0.xyz'

# Read coordinates
coordinates, coord_x, coord_y, coord_z, species, num_atoms, num_timesteps = \
    load_coordinates.load_values_coord(folder, input_filename)
species = species.reset_index(drop=True)

# Parameters
neighbours = {'Fe': 6, 'Fe_a': 6, 'Fe_b': 6, 'O': 3, 'H': 1}
decimal_places = 5

# Detect unique atoms
atom_types, atom_indices = np.unique(species, return_inverse=True)

# Calculate bond lengths
bond_lengths = functions.calc_bond_lengths(coordinates, num_atoms)
bond_lengths[bond_lengths == 0] = 1E6

# Calculate indexes of different elements
H_index = np.array([i for i, e in enumerate(species) if e == 'H'])
O_index = [i for i, e in enumerate(species) if e == 'O']
Fe_index = [i for i, e in enumerate(species) if e == 'Fe_a' or e == 'Fe_b' or e == 'Fe']
Mg_index = [i for i, e in enumerate(species) if e == 'Mg']

# Separate bond lengths matrix into individual species
Mg_bonds = bond_lengths[Mg_index, :]
Fe_bonds = bond_lengths[Fe_index, :]
O_bonds = bond_lengths[O_index, :]
# H_bonds = bond_lengths[H_index, :]

# Truncate bond lengths to bonded atoms
Mg_bonds_trunc = np.sort(Mg_bonds)[:, 0:neighbours['Fe']]
Mg_bonds_trunc[Mg_bonds_trunc > 2.3] = np.nan
Fe_bonds_trunc = np.sort(Fe_bonds)[:, 0:neighbours['Fe']]
Fe_bonds_trunc[Fe_bonds_trunc > 2.3] = np.nan
O_bonds_trunc = np.sort(O_bonds)[:, 0:neighbours['O']]
O_bonds_trunc[O_bonds_trunc > 2.5] = np.nan
# H_bonds_trunc = np.sort(H_bonds)[:, 0:neighbours['H']]

# Truncate to fully coordinated atoms
Fe_bonds_trunc_all = Fe_bonds_trunc[~np.isnan(Fe_bonds_trunc).any(axis=1)]
Mg_bonds_trunc_all = Mg_bonds_trunc[~np.isnan(Mg_bonds_trunc).any(axis=1)]

# Printing (hematite)
# hematite_exp = np.array([1.94401, 2.11409])
# print(np.mean(Fe_bonds_trunc_all[:, 0:3]) -Fe_bonds_trunc_all[:, 0:3].ravel())
print(Fe_bonds_trunc_all[:, 0:3].ravel())
# print(np.mean(Fe_bonds_trunc_all[:, 3:5]))
print(Fe_bonds_trunc_all[:, 3:5].ravel())
# print(np.mean(Fe_bonds_trunc_all[:, 3:5]) -Fe_bonds_trunc_all[:, 3:5].ravel())

# print('\nExperimental', hematite_exp)
print('Average Fe-O (A) (fully coordinated)', np.mean(Fe_bonds_trunc_all[:, 0:3]))
print('Range Fe-O (A) (fully coordinated)', np.ptp(Fe_bonds_trunc_all[:, 0:3]))
print('RMSD Fe-O (A) (fully coordinated)',
      np.sqrt(1/(Fe_bonds_trunc_all[:, 0:3].ravel().shape[0]) * np.sum((np.mean(Fe_bonds_trunc_all[:, 0:3]) -
                                                                        Fe_bonds_trunc_all[:, 0:3].ravel()) ** 2)))

print('\nAverage Fe-O (B) (fully coordinated)', np.mean(Fe_bonds_trunc_all[:, 3:5]))
print('Range Fe-O (B) (fully coordinated)', np.ptp(Fe_bonds_trunc_all[:, 3:5]))
print('RMSD Fe-O (B) (fully coordinated)',
      np.sqrt(1/(Fe_bonds_trunc_all[:, 3:5].ravel().shape[0]) * np.sum((np.mean(Fe_bonds_trunc_all[:, 3:5]) -
                                                                        Fe_bonds_trunc_all[:, 3:5].ravel()) ** 2)))
# print('All', np.mean(Fe_bonds_trunc_all[:, 0:3]),
#       np.sqrt(1 / (Fe_bonds_trunc_all[:, 0:3].ravel().shape[0]) * np.sum((np.mean(Fe_bonds_trunc_all[:, 0:3]) -
#                                                                           Fe_bonds_trunc_all[:, 0:3].ravel()) ** 2)),
#       np.mean(Fe_bonds_trunc_all[:, 3:5]),
#       np.sqrt(1 / (Fe_bonds_trunc_all[:, 3:5].ravel().shape[0]) * np.sum((np.mean(Fe_bonds_trunc_all[:, 3:5]) -
#                                                                           Fe_bonds_trunc_all[:, 3:5].ravel()) ** 2))
#       )

# Printing (goethite)
# print('Average Fe-O (1i) (fully coordinated)', np.mean(Fe_bonds_trunc_all[:, 0]))
# print('RMSD Fe-O (1i) (fully coordinated)',
#       np.sqrt(1 / (Fe_bonds_trunc_all[:, 0].ravel().shape[0]) * np.sum((np.mean(Fe_bonds_trunc_all[:, 0]) -
#                                                                         Fe_bonds_trunc_all[:, 0].ravel()) ** 2)))
# print('\nAverage Fe-O (1ii) (fully coordinated)', np.mean(Fe_bonds_trunc_all[:, 1:3]))
# print('RMSD Fe-O (1i) (fully coordinated)',
#       np.sqrt(1 / (Fe_bonds_trunc_all[:, 1:3].ravel().shape[0]) * np.sum((np.mean(Fe_bonds_trunc_all[:, 1:3]) -
#                                                                         Fe_bonds_trunc_all[:, 1:3].ravel()) ** 2)))
# print('\nAverage Fe-O (2iii) (fully coordinated)', np.mean(Fe_bonds_trunc_all[:, 3]))
# print('RMSD Fe-O (1i) (fully coordinated)',
#       np.sqrt(1 / (Fe_bonds_trunc_all[:, 3].ravel().shape[0]) * np.sum((np.mean(Fe_bonds_trunc_all[:, 3]) -
#                                                                         Fe_bonds_trunc_all[:, 3].ravel()) ** 2)))
# print('\nAverage Fe-O (2iv) (fully coordinated)', np.mean(Fe_bonds_trunc_all[:, 4:5]))
# print('RMSD Fe-O (1i) (fully coordinated)',
#       np.sqrt(1 / (Fe_bonds_trunc_all[:, 4:5].ravel().shape[0]) * np.sum((np.mean(Fe_bonds_trunc_all[:, 4:5]) -
#                                                                         Fe_bonds_trunc_all[:, 4:5].ravel()) ** 2)))
# print('\nAverage O-H', np.mean(H_bonds_trunc))

# Printing (lepidocrocite)
# print('Average Fe-O (1i) (fully coordinated)', np.mean(Fe_bonds_trunc_all[:, 0]))
# print('RMSD Fe-O (1i) (fully coordinated)',
#       np.sqrt(1/(Fe_bonds_trunc_all[:, 0].ravel().shape[0]) * np.sum((np.mean(Fe_bonds_trunc_all[:, 0]) -
#                                                                         Fe_bonds_trunc_all[:, 0].ravel()) ** 2)))
# print('\nAverage Fe-O (1i) (fully coordinated)', np.mean(Fe_bonds_trunc_all[:, 1]))
# print('RMSD Fe-O (1ii) (fully coordinated)',
#       np.sqrt(1/(Fe_bonds_trunc_all[:, 1].ravel().shape[0]) * np.sum((np.mean(Fe_bonds_trunc_all[:, 1]) -
#                                                                         Fe_bonds_trunc_all[:, 1].ravel()) ** 2)))
# print('\nAverage Fe-O (1i) (fully coordinated)', np.mean(Fe_bonds_trunc_all[:, 2:3]))
# print('RMSD Fe-O (1i) (fully coordinated)',
#       np.sqrt(1/(Fe_bonds_trunc_all[:, 2:3].ravel().shape[0]) * np.sum((np.mean(Fe_bonds_trunc_all[:, 2:3]) -
#                                                                         Fe_bonds_trunc_all[:, 2:3].ravel()) ** 2)))
# print('\nAverage Fe-O (1i) (fully coordinated)', np.mean(Fe_bonds_trunc_all[:, 4:5]))
# print('RMSD Fe-O (1i) (fully coordinated)',
#       np.sqrt(1/(Fe_bonds_trunc_all[:, 4:5].ravel().shape[0]) * np.sum((np.mean(Fe_bonds_trunc_all[:, 4:5]) -
#                                                                         Fe_bonds_trunc_all[:, 4:5].ravel()) ** 2)))
# print('\nAverage O-H', np.mean(H_bonds_trunc))

# Printing (brucite)
# print('Average Mg-O (1i) (fully coordinated)', np.mean(Mg_bonds_trunc[:, 0]))
# print('Average Mg-O (1ii) (fully coordinated)', np.mean(Mg_bonds_trunc[:, 1:3]))
# print('Average Mg-O (2iii) (fully coordinated)', np.mean(Mg_bonds_trunc[:, 3]))
# print('Average Mg-O (2iv) (fully coordinated)', np.mean(Mg_bonds_trunc[:, 4:5]))
# print('Average O-H', np.mean(H_bonds))
# print('Average H-H', np.mean(np.sort(H_bonds)[:, 1]))

# Printing (white rust)
# print('Average Fe-O (1i) (fully coordinated)', np.mean(Fe_bonds_trunc_all[:, 0:4]))
# print('Average Fe-O (1i) (fully coordinated)', np.mean(Fe_bonds_trunc_all[:, 4:5]))
# print('Average O-H', np.mean(H_bonds_trunc))
# print('Average H-H', np.mean(np.sort(H_bonds)[:, 1]))

# Printing (other)
# print('All bonds \n', Fe_bonds_trunc_all)
# atom = 125 - 1
# atom_bond_length = np.sort(bond_lengths[atom, :])[1:1+neighbours[str(species[atom])]]
# print('coord', atom+1,  species[atom], '\ncoordinates', coordinates[0, :, atom], '\nbond lengths', atom_bond_length)
