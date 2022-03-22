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

"""
    .xyz
"""

folder_data = 'E:/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/benchmarks/archer/data'
folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/benchmarks/archer/plots'

# plot used time
cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
cols2 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
cores_archer = np.array([192, 384, 576, 768, 960, 1536])
cores_archer2 = np.array([128, 256, 384, 512, 640, 768, 1024, 1280, 1536])

# Plotting
trunc_221_1 = [0, 1, 2, 3]
trunc_221_2 = [0, 1, 2, 3, 5]

trunc_331_1 = [0, 1, 2, 4, 5]
trunc_331_2 = [0, 1, 2, 3, 5, 7, 8]

trunc_441_1 = [0, 1, 2, 4, 5]
trunc_441_2 = [0, 2, 4, 6, 8]

# Total used time
archer_221_used = pd.read_csv('{}/archer1/221_used_time.out'.format(folder_data), names=cols, delim_whitespace=True)
archer_331_used = pd.read_csv('{}/archer1/331_used_time.out'.format(folder_data), names=cols, delim_whitespace=True)
archer_441_used = pd.read_csv('{}/archer1/441_used_time.out'.format(folder_data), names=cols, delim_whitespace=True)
archer2_221_used = pd.read_csv('{}/archer2/221_used_time.out'.format(folder_data), names=cols, delim_whitespace=True)
archer2_331_used = pd.read_csv('{}/archer2/331_used_time.out'.format(folder_data), names=cols, delim_whitespace=True)
archer2_441_used = pd.read_csv('{}/archer2/441_used_time.out'.format(folder_data), names=cols, delim_whitespace=True)

# SCF time
archer_221_scf = pd.read_csv('{}/archer1/221_scf_time.out'.format(folder_data), names=cols2, delim_whitespace=True)
archer_331_scf = pd.read_csv('{}/archer1/331_scf_time.out'.format(folder_data), names=cols2, delim_whitespace=True)
archer_441_scf = pd.read_csv('{}/archer1/441_scf_time.out'.format(folder_data), names=cols2, delim_whitespace=True)
archer2_221_scf = pd.read_csv('{}/archer2/221_scf_time.out'.format(folder_data), names=cols2, delim_whitespace=True)
archer2_331_scf = pd.read_csv('{}/archer2/331_scf_time.out'.format(folder_data), names=cols2, delim_whitespace=True)
archer2_441_scf = pd.read_csv('{}/archer2/441_scf_time.out'.format(folder_data), names=cols2, delim_whitespace=True)

# Collect average SCF time
archer_221_scf_avg = np.zeros(np.shape(cores_archer)[0])
archer_331_scf_avg = np.zeros(np.shape(cores_archer)[0])
archer_441_scf_avg = np.zeros(np.shape(cores_archer)[0])
archer2_221_scf_avg = np.zeros(np.shape(cores_archer2)[0])
archer2_331_scf_avg = np.zeros(np.shape(cores_archer2)[0])
archer2_441_scf_avg = np.zeros(np.shape(cores_archer2)[0])

for i in range(0, np.shape(cores_archer)[0]):
    archer_221_scf_avg[i] = np.average(np.sort(archer_221_scf['F'][i*43:43*(i+1)])[0:-2])
    archer_331_scf_avg[i] = np.average(np.sort(archer_331_scf['F'][i*41:41*(i+1)])[0:-2])
    archer_441_scf_avg[i] = np.average(np.sort(archer_441_scf['F'][i*40:40*(i+1)])[0:-2])

for i in range(0, np.shape(cores_archer2)[0]):
    archer2_221_scf_avg[i] = np.average(np.sort(archer2_221_scf['F'][i*43:43* (i + 1)])[0:-2])
    archer2_331_scf_avg[i] = np.average(np.sort(archer2_331_scf['F'][i*41:41* (i + 1)])[0:-2])
    archer2_441_scf_avg[i] = np.average(np.sort(archer2_441_scf['F'][i * 40:40 * (i + 1)])[0:-2])

# Linear
linear_221_x = np.linspace(cores_archer2[trunc_221_2[0]], 800, num=800)
linear_331_x = np.linspace(cores_archer2[trunc_331_2[0]], 1600, num=1600)
linear_441_x = np.linspace(cores_archer2[trunc_441_2[0]], 1600, num=1600)
linear_221_y=np.zeros(800)
linear_331_y=np.zeros(1600)
linear_441_y=np.zeros(1600)
for i in range(0, 800):
    linear_221_y[i]=60*60/archer2_221_scf_avg[trunc_221_2[0]]*(linear_221_x[i]/linear_221_x[0])

for i in range(0, 1600):
    linear_331_y[i]=60*60/archer2_331_scf_avg[trunc_331_2[0]]*(linear_331_x[i]/linear_331_x[0])
    linear_441_y[i]=60*60/archer2_441_scf_avg[trunc_441_2[0]]*(linear_441_x[i]/linear_441_x[0])

# Plot total used time
fig_time_total, ax_time_total = plt.subplots(figsize=[6, 5])
ax_time_total.plot(cores_archer[trunc_221_1], archer_221_used['H'][trunc_221_1], 'rx-', label='221 Archer')
ax_time_total.plot(cores_archer2[trunc_221_2], archer2_221_used['H'][trunc_221_2], 'ro--', label='221 Archer 2')
ax_time_total.plot(cores_archer[trunc_331_1], archer_331_used['H'][trunc_331_1], 'gx-', label='331 Archer')
ax_time_total.plot(cores_archer2[trunc_331_2], archer2_331_used['H'][trunc_331_2], 'go--', label='331 Archer 2')
ax_time_total.set_xlabel('Number of cores')
ax_time_total.set_ylabel('Total time / s')
ax_time_total.legend(frameon=True, loc='upper right')
fig_time_total.tight_layout()
fig_time_total.savefig('{}/time_total.png'.format(folder_save), dpi=300, bbbox_inches='tight')

# Plot scf steps per hour
fig_time_scf, ax_time_scf = plt.subplots(nrows=1, ncols=3, figsize=[20, 5])
ax_time_scf[0].plot(linear_221_x, linear_221_y, '-', color='grey', label='Linear')
ax_time_scf[0].plot(cores_archer[trunc_221_1], 60*60/archer_221_scf_avg[trunc_221_1], 'o-', color='blue', label='Archer')
ax_time_scf[0].plot(cores_archer2[trunc_221_2], 60*60/archer2_221_scf_avg[trunc_221_2], 'o-', color='darkorange', label='Archer2')
ax_time_scf[0].set_xlabel('Number of cores')
ax_time_scf[0].set_ylabel('SCF steps per hour / h')
ax_time_scf[0].set_ylim([600,  1850])
ax_time_scf[0].legend(frameon=True, loc='upper left')
ax_time_scf[1].plot(linear_331_x, linear_331_y, '-', color='grey', label='Linear')
ax_time_scf[1].plot(cores_archer[trunc_331_1], 60*60/archer_331_scf_avg[trunc_331_1], 'o-', color='blue', label='Archer')
ax_time_scf[1].plot(cores_archer2[trunc_331_2], 60*60/archer2_331_scf_avg[trunc_331_2], 'o-', color='darkorange', label='Archer2')
ax_time_scf[1].set_xlabel('Number of cores')
ax_time_scf[1].set_ylabel('SCF steps per hour / h')
ax_time_scf[1].set_ylim([200,  950])
ax_time_scf[1].legend(frameon=True, loc='upper left')
ax_time_scf[2].plot(linear_441_x, linear_441_y, '-', color='grey', label='Linear')
ax_time_scf[2].plot(cores_archer[trunc_441_1], 60*60/archer_441_scf_avg[trunc_441_1], 'o-', color='blue', label='Archer')
ax_time_scf[2].plot(cores_archer2[trunc_441_2], 60*60/archer2_441_scf_avg[trunc_441_2], 'o-', color='darkorange', label='Archer2')
ax_time_scf[2].set_xlabel('Number of cores')
ax_time_scf[2].set_ylabel('SCF steps per hour / h')
# ax_time_scf[2].plot([640, 640], [0, 1e3], '--', color='grey')
ax_time_scf[2].set_ylim([100,  500])
ax_time_scf[2].legend(frameon=True, loc='upper left')
fig_time_scf.savefig('{}/scf_per_hour_linear.png'.format(folder_save), dpi=300, bbbox_inches='tight')

# Plot scf steps per hour 221
fig_time_scf_221, ax_time_scf_221 = plt.subplots(figsize=[6, 5])
ax_time_scf_221.plot(linear_221_x, linear_221_y, '-', color='grey', label='Linear')
ax_time_scf_221.plot(cores_archer[trunc_221_1], 60*60/archer_221_scf_avg[trunc_221_1], 'o-', color='blue', label='Archer')
ax_time_scf_221.plot(cores_archer2[trunc_221_2], 60*60/archer2_221_scf_avg[trunc_221_2], 'o-', color='darkorange', label='Archer2')
ax_time_scf_221.set_xlabel('Number of cores')
ax_time_scf_221.set_ylabel('SCF steps per hour / h')
ax_time_scf_221.legend(frameon=True, loc='upper left')
ax_time_scf_221.set_ylim([600,  1850])
fig_time_scf_221.tight_layout()
fig_time_scf_221.savefig('{}/scf_per_hour_221_linear.png'.format(folder_save), dpi=300, bbbox_inches='tight')

# Plot scf steps per hour 331
fig_time_scf_331, ax_time_scf_331 = plt.subplots(figsize=[6, 5])
ax_time_scf_331.plot(linear_331_x, linear_331_y, '-', color='grey', label='Linear')
ax_time_scf_331.plot(cores_archer[trunc_331_1], 60*60/archer_331_scf_avg[trunc_331_1], 'o-', color='blue', label='Archer')
ax_time_scf_331.plot(cores_archer2[trunc_331_2], 60*60/archer2_331_scf_avg[trunc_331_2], 'o-', color='darkorange', label='Archer2')
ax_time_scf_331.set_xlabel('Number of cores')
ax_time_scf_331.set_ylabel('SCF steps per hour / h')
ax_time_scf_331.legend(frameon=True, loc='upper left')
ax_time_scf_331.set_ylim([200,  950])
fig_time_scf_331.tight_layout()
fig_time_scf_331.savefig('{}/scf_per_hour_331_linear.png'.format(folder_save), dpi=300, bbbox_inches='tight')

# Plot scf steps per hour 441
fig_time_scf_441, ax_time_scf_441 = plt.subplots(figsize=[6, 5])
# ax_time_scf_441.plot([640, 640], [0, 1e3], '--', color='grey')
ax_time_scf_441.plot(linear_441_x, linear_441_y, '-', color='grey', label='Linear')
ax_time_scf_441.plot(cores_archer[trunc_441_1], 60*60/archer_441_scf_avg[trunc_441_1], 'o-', color='blue', label='Archer')
ax_time_scf_441.plot(cores_archer2[trunc_441_2], 60*60/archer2_441_scf_avg[trunc_441_2], 'o-', color='darkorange', label='Archer2')
ax_time_scf_441.set_xlabel('Number of cores')
ax_time_scf_441.set_ylabel('SCF steps per hour / h')
ax_time_scf_441.legend(frameon=True, loc='upper left')
ax_time_scf_441.set_ylim([100,  500])
fig_time_scf_441.tight_layout()
fig_time_scf_441.savefig('{}/scf_per_hour_441_linear.png'.format(folder_save), dpi=300, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
