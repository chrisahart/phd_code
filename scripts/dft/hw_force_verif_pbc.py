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

bw_energy = np.array([-4.879790480618770,  -4.882542067900578, -4.885123344542297,  -4.887547573989776, -4.889827921005030])
bw_force_1 = np.array([0.05680021, 0.05327763, 0.04999479, 0.04702441, 0.04427716])
bw_calc_force = np.zeros(3)
bw_calc_force[0] = -(bw_energy[2] - bw_energy[0]) / (2 * (distance[2] - distance[0]))
bw_calc_force[1] = -(bw_energy[3] - bw_energy[1]) / (2 * (distance[3] - distance[1]))
bw_calc_force[2] = -(bw_energy[4] - bw_energy[2]) / (2 * (distance[4] - distance[2]))

hw_energy = np.array([-4.880612457427011, -4.884144978662786, -4.887441139091886, -4.890521781939398, -4.893403667962914])
hw_force_1 = np.array([0.07308179, 0.06814562,  0.06372376, 0.05958534, 0.05579364])
hw_calc_force = np.zeros(3)
hw_calc_force[0] = -(hw_energy[2] - hw_energy[0]) / (2 * (distance[2] - distance[0]))
hw_calc_force[1] = -(hw_energy[3] - hw_energy[1]) / (2 * (distance[3] - distance[1]))
hw_calc_force[2] = -(hw_energy[4] - hw_energy[2]) / (2 * (distance[4] - distance[2]))

fig_hw_force, ax_hw_force = plt.subplots()
ax_hw_force.plot(distance, hw_force_1, 'ko-', label='Analytic')
ax_hw_force.plot(distance[1:4], 2*hw_calc_force, 'ro-', label='Centred diff.')
ax_hw_force.set_xlabel('Distance / a.u')
ax_hw_force.set_ylabel('Force / a.u')
ax_hw_force.legend(frameon=True)
fig_hw_force.tight_layout()
# fig_hw_force.savefig('{}/hw_pbc.png'.format(folder), dpi=parameters.save_dpi, bbbox_inches='tight')

print('\nHW average error:', (2*hw_calc_force-hw_force_1[1:4]))
print('HW error:', np.mean(2*hw_calc_force-hw_force_1[1:4]))

fig_bw_force, ax_bw_force = plt.subplots()
ax_bw_force.plot(distance, bw_force_1, 'ko-', label='Analytic')
ax_bw_force.plot(distance[1:4], 2*bw_calc_force, 'ro-', label='Centred diff.')
ax_bw_force.set_xlabel('Distance / a.u')
ax_bw_force.set_ylabel('Force / a.u')
ax_bw_force.legend(frameon=True)
fig_bw_force.tight_layout()
# fig_bw_force.savefig('{}/bw_pbc.png'.format(folder), dpi=parameters.save_dpi, bbbox_inches='tight')

print('\nBW average error:', (2*bw_calc_force-bw_force_1[1:4]))
print('BW error:', np.mean(2*bw_calc_force-bw_force_1[1:4]))


if __name__ == "__main__":
    print('Finished.')
    plt.show()
