from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
import random
from numpy import nan as Nan
import matplotlib.pyplot as plt
import scipy
from scripts.general import parameters
from scripts.general import functions
from scripts.formatting import load_coordinates
from scripts.formatting import load_cube

"""
    Plotting of HW force profile data.
"""


# General
folder_input = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hw_forces/he2'
labels = ['+ve', '-ve', 'Average']

# Data
cols = ['Filename', 'A', 'B', 'C', 'D', 'E', 'F', 'Time']
he_hw_couplings = pd.read_csv('{}/he2_hw_cutoff_couplings.out'.format(folder_input), names=cols, delim_whitespace=True)
he_hw_couplings = he_hw_couplings.drop(['A', 'B', 'C', 'D', 'E', 'Time'], axis=1)
he_hw_cutoff_time = pd.read_csv('{}/he2_hw_cutoff_time.out'.format(folder_input), names=cols, delim_whitespace=True)
he_hw_cutoff_time = he_hw_cutoff_time.drop(['A', 'B', 'C', 'D', 'E', 'F'], axis=1)
time_average = (he_hw_cutoff_time['Time'].values[0::2]+he_hw_cutoff_time['Time'].values[1::2])/2

# Specific
samples = 5
cores = np.array([10, 12, 16, 6, 8], dtype='int')

# Plot time
fig_hw_coupling_time, ax_hw_coupling_time = plt.subplots()
ax_hw_coupling_time.plot(cores, time_average, 'ko', label='Average')
ax_hw_coupling_time.plot(cores, he_hw_cutoff_time['Time'].values[0::2], 'ro', label='+1 charge')
ax_hw_coupling_time.plot(cores, he_hw_cutoff_time['Time'].values[1::2], 'go', label='-1 charge')
ax_hw_coupling_time.set_xlabel('-log_10(eps)')
ax_hw_coupling_time.set_ylabel('Time / s')
ax_hw_coupling_time.legend(frameon=True)
fig_hw_coupling_time.tight_layout()
# fig_hw_coupling_time.savefig('{}/plotted/{}_time.png'.format(folder_input, name), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot time
fig_hw_coupling, ax_hw_coupling = plt.subplots()
ax_hw_coupling.plot(cores, he_hw_couplings['F'].values[0::2]*parameters.hartree_to_ev/1e3, 'ko', label='')
ax_hw_coupling.set_xlabel('-log_10(eps)')
ax_hw_coupling.set_ylabel('Hab / meV')
ax_hw_coupling.legend(frameon=True)
fig_hw_coupling.tight_layout()


if __name__ == "__main__":
    print('Finished.')
    plt.show()
