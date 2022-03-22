from __future__ import division, print_function
import time
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from scripts.general import functions
from scripts.general import parameters
from scripts.formatting import load_coordinates
from scripts.formatting import load_energy
from scripts.formatting import load_forces_out
from scripts.formatting import load_forces
# from scripts.dft import cdft_beta


"""
    Plot energy and forces for hematite interface 
"""


skip = 2
num_atoms = 435

folder1 = 'E:/University/PhD/Programming/dft_ml_md/output/surfin/dft-md/data/benchmarking'
iter1_1 = np.loadtxt('{}/{}'.format(folder1, 'neutral/iter/node-8.out'))
iter1_2 = np.loadtxt('{}/{}'.format(folder1, 'neutral/iter/node-12.out'))
iter1_3 = np.loadtxt('{}/{}'.format(folder1, 'neutral/iter/node-16.out'))
iter1_4 = np.loadtxt('{}/{}'.format(folder1, 'neutral/iter/node-20.out'))
iter1_5 = np.loadtxt('{}/{}'.format(folder1, 'neutral/iter/node-4.out'))
iter2_1 = np.loadtxt('{}/{}'.format(folder1, 'hole/iter/node-8.out'))
iter2_2 = np.loadtxt('{}/{}'.format(folder1, 'hole/iter/node-12.out'))
iter2_3 = np.loadtxt('{}/{}'.format(folder1, 'hole/iter/node-16.out'))
iter2_4 = np.loadtxt('{}/{}'.format(folder1, 'hole/iter/node-20.out'))
iter2_5 = np.loadtxt('{}/{}'.format(folder1, 'hole/iter/node-4.out'))
iter3_1 = np.loadtxt('{}/{}'.format(folder1, 'electron/iter/node-8.out'))
iter3_2 = np.loadtxt('{}/{}'.format(folder1, 'electron/iter/node-12.out'))
iter3_3 = np.loadtxt('{}/{}'.format(folder1, 'electron/iter/node-16.out'))
iter3_4 = np.loadtxt('{}/{}'.format(folder1, 'electron/iter/node-20.out'))
iter3_5 = np.loadtxt('{}/{}'.format(folder1, 'electron/iter/node-4.out'))

time1_1 = np.loadtxt('{}/{}'.format(folder1, 'neutral/time-per-step/node-8.out'))
time1_2 = np.loadtxt('{}/{}'.format(folder1, 'neutral/time-per-step/node-12.out'))
time1_3 = np.loadtxt('{}/{}'.format(folder1, 'neutral/time-per-step/node-16.out'))
time1_4 = np.loadtxt('{}/{}'.format(folder1, 'neutral/time-per-step/node-20.out'))
time1_5 = np.loadtxt('{}/{}'.format(folder1, 'neutral/time-per-step/node-4.out'))
time2_1 = np.loadtxt('{}/{}'.format(folder1, 'hole/time-per-step/node-8.out'))
time2_2 = np.loadtxt('{}/{}'.format(folder1, 'hole/time-per-step/node-12.out'))
time2_3 = np.loadtxt('{}/{}'.format(folder1, 'hole/time-per-step/node-16.out'))
time2_4 = np.loadtxt('{}/{}'.format(folder1, 'hole/time-per-step/node-20.out'))
time2_5 = np.loadtxt('{}/{}'.format(folder1, 'hole/time-per-step/node-4.out'))
time3_1 = np.loadtxt('{}/{}'.format(folder1, 'hole/time-per-step/node-8.out'))
time3_2 = np.loadtxt('{}/{}'.format(folder1, 'hole/time-per-step/node-12.out'))
time3_3 = np.loadtxt('{}/{}'.format(folder1, 'hole/time-per-step/node-16.out'))
time3_4 = np.loadtxt('{}/{}'.format(folder1, 'hole/time-per-step/node-20.out'))
time3_5 = np.loadtxt('{}/{}'.format(folder1, 'hole/time-per-step/node-4.out'))

# Plot steps
# fig_steps, ax_steps = plt.subplots()
# ax_steps.plot(iter1_1, 'rx-', label='Neutral 8 nodes')
# ax_steps.plot(iter2_1, 'gx-', label='Hole 8 nodes')
# ax_steps.plot(iter3_1, 'bx-', label='Electron 8 nodes')
# ax_steps.set_xlabel('MD step')
# ax_steps.set_ylabel('SCF steps')
# ax_steps.legend(frameon=False)
# ax_steps.set_xlim([0, 5])
# fig_steps.tight_layout()
# fig_steps.savefig('{}/steps.png'.format(folder1), dpi=300, bbbox_inches='tight')

# Plot time taken
nodes = np.array([4, 8, 12, 16, 20])
ignore_first = 1
fig_spin2, ax_spin2 = plt.subplots()
temp1 = np.zeros(iter1_5.shape[0])
temp2 = np.zeros(iter1_5.shape[0])
temp3 = np.zeros(iter1_5.shape[0])
temp4 = np.zeros(iter1_5.shape[0])
temp5 = np.zeros(iter1_5.shape[0])
i = 0
j = 0
for n in range(iter1_5.shape[0]):
    i = i + int(iter1_1[n])
    temp1[n] = np.mean(time1_1[j+ignore_first:i])
    temp2[n] = np.mean(time1_2[j + ignore_first:i])
    temp3[n] = np.mean(time1_3[j + ignore_first:i])
    temp4[n] = np.mean(time1_4[j + ignore_first:i])
    print(time1_3[j + ignore_first:i + 1])
    print(temp3[n - 1])
    temp5[n] = np.mean(time1_5[j + ignore_first:i])
    j = i
print(temp3)
neutral_scaling = np.array([np.mean(temp5), np.mean(temp1), np.mean(temp2), np.mean(temp3), np.mean(temp4)])
ax_spin2.plot(nodes, neutral_scaling, 'rx-', label='Neutral')
temp1 = np.zeros(iter3_5.shape[0])
temp2 = np.zeros(iter3_5.shape[0])
temp3 = np.zeros(iter3_5.shape[0])
temp4 = np.zeros(iter3_5.shape[0])
temp5 = np.zeros(iter3_5.shape[0])
i = 0
j = 0
for n in range(iter3_5.shape[0]):
    i = i + int(iter3_1[n])
    temp1[n] = np.mean(time3_1[j + ignore_first:i])
    temp2[n] = np.mean(time3_2[j + ignore_first:i])
    temp3[n] = np.mean(time3_3[j + ignore_first:i])
    temp4[n] = np.mean(time3_4[j + ignore_first:i])
    print(time3_3[j + ignore_first:i + 1])
    print(temp3[n - 1])
    temp5[n] = np.mean(time3_5[j + ignore_first:i])
    j = i
print(temp3)
electron_scaling = np.array([np.mean(temp5), np.mean(temp1), np.mean(temp2), np.mean(temp3), np.mean(temp4)])
ax_spin2.plot(nodes, electron_scaling, 'bx-', label='Electron')
temp1 = np.zeros(iter2_4.shape[0]-1)
temp2 = np.zeros(iter2_4.shape[0]-1)
temp3 = np.zeros(iter2_4.shape[0]-1)
temp4 = np.zeros(iter2_4.shape[0]-1)
temp5 = np.zeros(iter2_4.shape[0]-1)
i = 0
j = 0
print('\nHole')
for n in range(1, iter2_4.shape[0]):
    i = i + int(iter2_1[n])
    temp1[n-1] = np.mean(time2_1[j+ignore_first:i+1])
    temp2[n-1] = np.mean(time2_2[j + ignore_first:i+1])
    temp3[n-1] = np.mean(time2_3[j + ignore_first:i+1])
    print(time2_3[j + ignore_first:i+1])
    print(temp3[n-1])
    temp4[n-1] = np.mean(time2_4[j + ignore_first:i+1])
    temp5[n-1] = np.mean(time2_5[j + ignore_first:i+1])
    j = i
print(temp3)
hole_scaling = np.array([np.mean(temp5), np.mean(temp1), np.mean(temp2), np.mean(temp3), np.mean(temp4)])
ax_spin2.plot(nodes, hole_scaling, 'gx-', label='Hole')
ax_spin2.set_xlabel('Node')
ax_spin2.set_ylabel('Time per SCF step / s')
ax_spin2.set_xlim([4-0.5, 16+0.5])
ax_spin2.set_ylim([4, 12])
ax_spin2.legend(frameon=False)
fig_spin2.tight_layout()
fig_spin2.savefig('{}/node_scaling.png'.format(folder1), dpi=300, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
