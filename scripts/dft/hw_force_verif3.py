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

hw_dft_energy = np.array([-4.817525688406072, -4.821059521705444, -4.824358963857581,   -4.827441274904142, -4.830325217903233])
hw_cdft_energy = np.array([-4.991773287137395, -4.990867726580115,  -4.990006461364701, -4.989190980184610, -4.988423778199512])
hw_energy = (hw_dft_energy - hw_cdft_energy)
hw_force_dft = np.array([0.94277969182926957, 0.93772687042364511, 0.93475479401600003, 0.92983859179262995, 0.926794040407717730])
hw_force_cft_1 = np.array([-0.07312637, -0.06826401, -0.06381219, -0.05962329, -0.05583031])
hw_force_cft_2 = np.array([0.07316412, 0.06825703, 0.06380665, 0.05960875, 0.05582372])
hw_force_only_1 = np.array([ 0.16452682806330227, 0.15406225758507597, 0.14465054924683060, 0.13535800697619677, 0.12697231791648808])
# hw_force_only_1 = np.array([0.14921550487905319, 0.14002532103705129, 0.13174051892252409, 0.12352003889851203, 0.11608564563978761])
hw_force_only_2 = np.array([0.14921550487905319, 0.14002532103705129, 0.13174051892252409, 0.12352003889851203, 0.11608564563978761])
hw_force_1 = hw_force_only_1
hw_calc_force = np.zeros(3)
hw_calc_force[0] = -(hw_energy[2] - hw_energy[0]) / (2 * (distance[2] - distance[0]))
hw_calc_force[1] = -(hw_energy[3] - hw_energy[1]) / (2 * (distance[3] - distance[1]))
hw_calc_force[2] = -(hw_energy[4] - hw_energy[2]) / (2 * (distance[4] - distance[2]))

fig_hw_force, ax_hw_force = plt.subplots()
# ax_hw_force.plot(distance, (np.abs(hw_force_1)+np.abs(hw_force_1))/2, 'ko-', label='Analytic')
ax_hw_force.plot(distance, hw_force_1, 'ko-', label='Analytic')
ax_hw_force.plot(distance[1:4], 4*hw_calc_force, 'ro-', label='Centred diff.')
ax_hw_force.set_xlabel('Distance / a.u')
ax_hw_force.set_ylabel('Force / a.u')
ax_hw_force.legend(frameon=True)
fig_hw_force.tight_layout()
fig_hw_force.savefig('{}/hw-cdft-dev.png'.format(folder), dpi=parameters.save_dpi, bbbox_inches='tight')

print('\nBW average error:', (4*hw_calc_force-hw_force_1[1:4]))
print('BW error:', np.mean(4*hw_calc_force-hw_force_1[1:4]))

if __name__ == "__main__":
    print('Finished.')
    plt.show()
