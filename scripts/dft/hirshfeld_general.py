from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh
import matplotlib.pyplot as plt

"""
    General Hirshfeld analysis. 
    Prints total iron spin moment.
"""

# Filenames
# filename2 = '/scratch/cahart/work/personal_files/feIV_bulk/hematite/final/221_supercell/exp_vesta_neutral_cubes_s4/hirsh.out'
filename1 = '/scratch/cahart/work/personal_files/feIV_bulk/hematite/final/221_supercell/exp_vesta_neutral_cubes_s4_hole/hirsh_init.out'

# filename1 = '/scratch/cahart/work/personal_files/feIV_bulk/geothite/final/hf18/313_supercell/positive_17_0_tight_neutral_cubes/hirsh_neut.out'
# filename1 = '/scratch/cahart/work/personal_files/feIV_bulk/geothite/final/hf18/313_supercell/positive_17_0_tight_neutral_cubes_hole/hirsh_init.out'

# filename1 = '/scratch/cahart/work/personal_files/feIV_bulk/lepidocrocite/final/313_supercell/neutral_18_cubes_symm_tight/hirsh_neut.out'
# filename1 = '/scratch/cahart/work/personal_files/feIV_bulk/lepidocrocite/final/313_supercell/neutral_18_cubes_symm_tight_hole/hirsh_final.out'

# Read files
Fe_db1, O_db1, H_db1, file_spec1 = cp2k_hirsh.read_hirsh(filename1)

# Calculations
pop1_max = np.sum(file_spec1['Pop 1'])
pop2_max = np.sum(file_spec1['Pop 2'])
spin_total = np.sum(file_spec1['Spin'])
charge_total = np.sum(file_spec1['Charge'])

# Printing
# print('file_spec1 \n', file_spec1)
# print('Fe_db1 \n', Fe_db1)
print('Total population spin 1', pop1_max)
print('Total population spin 2', pop2_max)
print('Fe average spin moment', np.average(np.abs(Fe_db1['Spin'])))
print('O average spin moment', np.average(np.abs(O_db1['Spin'])))
print('H average spin moment', np.average(np.abs(H_db1['Spin'])))
print('Fe average charge', np.average(np.abs(Fe_db1['Charge'])))

print('\nAverage Fe + spin moment', np.average([x for x in Fe_db1['Spin'] if x > 0]))
print('Average Fe - spin moment', np.average([x for x in Fe_db1['Spin'] if x < 0]))
print('Average O + spin moment', np.average([x for x in O_db1['Spin'] if x > 0]))
print('Average O - spin moment', np.average([x for x in O_db1['Spin'] if x < 0]))
print('Average H + spin moment', np.average([x for x in H_db1['Spin'] if x > 0]))
print('Average H - spin moment', np.average([x for x in H_db1['Spin'] if x < 0]))

print('\nTotal Fe + spin moment', np.sum(x for x in Fe_db1['Spin'] if x > 0))
print('Total Fe - spin moment', np.sum(x for x in Fe_db1['Spin'] if x < 0))
print('Total O + spin moment', np.sum(x for x in O_db1['Spin'] if x > 0))
print('Total O - spin moment', np.sum(x for x in O_db1['Spin'] if x < 0))
print('Total H + spin moment', np.sum(x for x in H_db1['Spin'] if x > 0))
print('Total H - spin moment', np.sum(x for x in H_db1['Spin'] if x < 0))


print('\nSpin total', np.sum(file_spec1['Spin']))
print('Total Fe spin moment', np.sum(Fe_db1['Spin']))
print('Total O spin moment', np.sum(O_db1['Spin']))
print('Total H spin moment', np.sum(H_db1['Spin']))

print('\nTotal Fe spin moment / spin total', np.sum(Fe_db1['Spin'])/np.sum(file_spec1['Spin']))
print('Total O spin moment / spin total', np.sum(O_db1['Spin'])/np.sum(file_spec1['Spin']))
print('Total H spin moment / spin total', np.sum(H_db1['Spin'])/np.sum(file_spec1['Spin']))

print('\nTotal Fe/(Fe+O) spin moment / spin total', np.abs(np.sum(Fe_db1['Spin']))/
      (np.abs(np.sum(Fe_db1['Spin']))+np.abs(np.sum(O_db1['Spin']))))
print('Total O/(Fe+O) spin moment / spin total', np.abs(np.sum(O_db1['Spin']))/
      (np.abs(np.sum(Fe_db1['Spin']))+np.abs(np.sum(O_db1['Spin']))))

print('\nABS Total Fe spin moment / spin total', np.abs(np.sum(Fe_db1['Spin']))/
      (np.abs(np.sum(Fe_db1['Spin']))+np.abs(np.sum(O_db1['Spin']))+np.abs(np.sum(H_db1['Spin']))))
print('ABS Total O spin moment / spin total', np.abs(np.sum(O_db1['Spin']))/
      (np.abs(np.sum(Fe_db1['Spin']))+np.abs(np.sum(O_db1['Spin']))+np.abs(np.sum(H_db1['Spin']))))
print('ABS Total H spin moment / spin total', np.abs(np.sum(H_db1['Spin']))/
      (np.abs(np.sum(Fe_db1['Spin']))+np.abs(np.sum(O_db1['Spin']))+np.abs(np.sum(H_db1['Spin']))))

# Printing charge
print('\nCharge total', np.sum(file_spec1['Charge']))
print('Fe charge', np.sum(Fe_db1['Charge']))
print('O charge', np.sum(O_db1['Charge']))
print('H charge', np.sum(H_db1['Charge']))

print('\nFe charge / charge total', np.sum(Fe_db1['Charge'])/np.sum(file_spec1['Charge']))
print('O charge / charge total', np.sum(O_db1['Charge'])/np.sum(file_spec1['Charge']))
print('H charge / charge total', np.sum(H_db1['Charge'])/np.sum(file_spec1['Charge']))
