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
    Test.
"""

distance = np.array([2.90, 2.95, 3.00, 3.05, 3.10])   # Distance in Bohr
folder = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/he2/forces'

bw_energy = np.array([-4.816700624867496, -4.819453280540432, -4.822036508049324, -4.824461141004925, -4.826742981348088])
bw_force_1 = np.array([-0.05682237, -0.05330009,  -0.04999567, -0.04704956, -0.04431479])
bw_force_2 = np.array([0.05686733, 0.05329755, 0.04999157, 0.04703509, 0.04430313])
bw_calc_force = np.zeros(3)
bw_calc_force[0] = -(bw_energy[2] - bw_energy[0]) / (2 * (distance[2] - distance[0]))
bw_calc_force[1] = -(bw_energy[3] - bw_energy[1]) / (2 * (distance[3] - distance[1]))
bw_calc_force[2] = -(bw_energy[4] - bw_energy[2]) / (2 * (distance[4] - distance[2]))

hw_energy = np.array([-4.817525688406072, -4.821059521705444, -4.824358963857581,   -4.827441274904142, -4.830325217903233])
hw_force_1 = np.array([-0.07312637, -0.06826401, -0.06381219,  -0.05962329, -0.05583031])
hw_force_2 = np.array([0.07316412, 0.06825703, 0.06380665, 0.05960875, 0.05582371])
hw_calc_force = np.zeros(3)
hw_calc_force[0] = -(hw_energy[2] - hw_energy[0]) / (2 * (distance[2] - distance[0]))
hw_calc_force[1] = -(hw_energy[3] - hw_energy[1]) / (2 * (distance[3] - distance[1]))
hw_calc_force[2] = -(hw_energy[4] - hw_energy[2]) / (2 * (distance[4] - distance[2]))

fig_bw_force, ax_bw_force = plt.subplots()
ax_bw_force.plot(distance, bw_force_2, 'ko-', label='Analytic')
ax_bw_force.plot(distance[1:4], 2*bw_calc_force, 'ro-', label='Centred diff.')
ax_bw_force.set_xlabel('Distance / a.u')
ax_bw_force.set_ylabel('Force / a.u')
ax_bw_force.legend(frameon=True)
fig_bw_force.tight_layout()
# fig_bw_force.savefig('{}/bw.png'.format(folder), dpi=parameters.save_dpi, bbbox_inches='tight')

print('\nBW average error:', (2*bw_calc_force-bw_force_2[1:4]))
print('BW error:', np.mean(2*bw_calc_force-bw_force_2[1:4]))

fig_hw_force, ax_hw_force = plt.subplots()
ax_hw_force.plot(distance, hw_force_2, 'ko-', label='Analytic')
ax_hw_force.plot(distance[1:4], 2*hw_calc_force, 'ro-', label='Centred diff.')
ax_hw_force.set_xlabel('Distance / a.u')
ax_hw_force.set_ylabel('Force / a.u')
ax_hw_force.legend(frameon=True)
fig_hw_force.tight_layout()
# fig_hw_force.savefig('{}/hw.png'.format(folder), dpi=parameters.save_dpi, bbbox_inches='tight')

print('\nHW error:', (2*hw_calc_force-hw_force_2[1:4]))
print('HW average error:', np.mean(2*hw_calc_force-hw_force_2[1:4]))

if __name__ == "__main__":
    print('Finished.')
    plt.show()
