from __future__ import division, print_function
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
import scipy
import re
import pickle
import pandas as pd
from distutils.dir_util import copy_tree

""" Write force constants. 
    Copy folder contents to new folder, modifying input and submit files.  """

# Files
# file_read = '/scratch/cahart/work/personal_files/fe_bulk/hematite/final/221_supercell/exp_vesta_0_cubes/cp2k_log.log'
# file_read = '/scratch/cahart/work/personal_files/fe_bulk/hematite/final/221_supercell/exp_vesta_neutral_cubes_s4/hem_221.o564992'
# file_read = '/scratch/cahart/work/personal_files/fe_bulk/hematite/final/221_supercell/exp_vesta_neutral_cubes_s4_elec/cp2k_log.log'
# file_read = '/scratch/cahart/work/personal_files/fe_bulk/hematite/final/221_supercell/exp_vesta_neutral_cubes_s4_hole/hem_221.o565241'
# file_read = '/scratch/cahart/work/personal_files/fe_bulk/hematite/221_supercell/scan/neutral_scan_custom_pseudo_m1200_hole/cp2k_log.log'
# file_read = '/scratch/cahart/work/personal_files/fe_bulk/hematite/final/331_supercell/neutral_vesta/cp2k_log.log'
file_read = '//scratch/cahart/work/personal_files/fe_bulk/lepidocrocite/final/613_supercell/hf_x25/neutral_cubes/cp2k_log.log'


# Search strings
# search_string = ' OT DIIS     0.80E-01 '
search_string1 = ' OT DIIS     0.80E-01 '
search_string2 = 'outer SCF loop converged in'
store1 = []
store2 = []

# Search for string and return numbers from each line
for line in open(file_read).read().splitlines():

    if search_string1 in line:
        # Time taken
        store1.append(line.split()[4])

    if search_string2 in line:
        # Time taken
        store2.append(line.split()[8])

    # Return final value
average_vals1 = np.mean((np.array(store1, dtype=np.float32)))
average_vals2 = np.mean((np.array(store2, dtype=np.float32)))

print('average_vals\n', average_vals1)
print('average_vals\n', average_vals2)
print('product\n', average_vals1 * average_vals2)


# Hematite 221 HSE06 m400 (n=16)
# exp_vesta_0_cubes: 47.859993, 6.923077 steps (331, as a result of schwartz 1e-6)
# exp_vesta_neutral_cubes_s4: 7.9973335, 21.774193 steps (174)
# exp_vesta_neutral_cubes_s4_elec: 5.7657523, 35.233334 steps (203)
# exp_vesta_neutral_cubes_s4_hole: 7.2164645, 29.117647 steps (210)

# Hematite 221 HSE06 m400 25%
# exp_vesta_neutral_cubes_s4  9.541818,13.75 steps (131)
# exp_vesta_neutral_cubes_s4_cubes   10.3058815,  12.75 (131)
# exp_vesta_neutral_cubes_s4_elec  6.4994464, 24.636364 (160)
# exp_vesta_neutral_cubes_s4_cubes_hole 5.772573,   34.255814 (197)

# Hematite 221 HSE06 m400 50% (n=16)
# exp_vesta_m400_50 10.504117,  12.142858 (127)

# Hematite 331 HSE06 m400 (n=24)
# neutral_vesta_cubes 23.617647, 12.75 steps (301)

# Hematite 221 SCAN
# neutral_scan_m1200, 3.1193452, 5.6949153 steps
# neutral_scan_m1200_elec, 3.1484668, 18.844444 steps
# neutral_scan_m1200_hole, 3.2801528, 23.818182 steps
# neutral_scan_custom_pseudo_m1200 3.0796332, 7.220588 steps
# neutral_scan_custom_pseudo_m1200_elec 3.126294, 15.580646 steps
# neutral_scan_custom_pseudo_m1200_hole 3.2951636, 18.314285 steps
