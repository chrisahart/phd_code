from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.general import parameters
from scripts.formatting import cp2k_hirsh
import matplotlib.pyplot as plt
import scipy

""" Testing file """

x = np.linspace(37, 72, num=36, dtype='int')
print(x)
# index = [147, 148, 126, 125, 139, 127, 131, 140, 132, 141, 133, 138, 150, 149, 137, 142, 136, 121, 122, 135, 143, 146, 144, 145, 129, 128, 134, 130, 123, 124]
# x = np.delete(x[:], index)
print(x*0-1)

folder_1 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/electron/delete/analysis'
folder_save_1 = folder_1
energy_kinetic1_1, energy_potential1_1, energy_total1_1, temperature1_1, time_val1_1, time_per_step1_1 = load_energy.load_values_energy(folder_1, '/energy/frozen-none.out')
file_spec1_1, species1_1, num_data1_1, step1_1, brent1_1, mnbrack1_1 = read_hirsh(folder_1, '/hirshfeld/frozen-none.out', atoms, None, None)
index_fe_1 = np.array([96, 134]) - 1

folder_2 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/philipp-share/electron/analysis'
folder_save_2 = folder_2
energy_kinetic1_2, energy_potential1_2, energy_total1_2, temperature1_2, time_val1_2, time_per_step1_2 = load_energy.load_values_energy(folder_2, '/energy/00.out')
file_spec1_2, species1_2, num_data1_2, step1_2, brent1_2, mnbrack1_2 = read_hirsh(folder_2, '/hirshfeld/00.out', atoms, None, None)
index_fe_2 = np.array([129, 30]) - 1