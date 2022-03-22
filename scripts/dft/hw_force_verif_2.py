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

folder = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/he2/forces'

# distance = np.array([5.683015 - 5.828865, 5.683015 - 5.838865,  5.683015 - 5.848865,  5.683015 - 5.858865])\
#            * parameters.angstrom_to_bohr  # Distance in Bohr
distance = np.array([5.683015 - 5.828865, 5.683015 - 5.838865,  5.683015 - 5.848865,  5.683015 - 5.858865, 5.683015 - 5.868865, 5.683015 - 5.878865])\
           * parameters.angstrom_to_bohr  # Distance in Bohr
hw_energy = np.array([-34.041282988475537,  -34.041784154409662, -34.042113628415990,  -34.042281305150759, -34.042297354673160, -34.042172350567945])
hw_force_1 = np.array([-0.02967639, -0.02063227, -0.01220066, -0.00434713,   0.00296328,  0.00976378 ])
hw_force_2 = np.array([0.01499172,  0.00587027, -0.0027335, -0.01084041,  -0.01843973, -0.02549161])
# hw_force_1 = hw_force_1 + hw_force_2
# hw_force_1 = np.array([0.00097200, 0.00080647,  0.00050265, 0.00020681])
# hw_force_2 = np.array([0.14921550487905319,  0.14002532103705129,  0.13174051892252409,  0.12352003889851203,  0.11608564563978761])
hw_calc_force = np.zeros(4)
hw_calc_force[0] = -(hw_energy[2] - hw_energy[0]) / (2 * (distance[2] - distance[0]))
hw_calc_force[1] = -(hw_energy[3] - hw_energy[1]) / (2 * (distance[3] - distance[1]))
hw_calc_force[2] = -(hw_energy[4] - hw_energy[2]) / (2 * (distance[4] - distance[2]))
hw_calc_force[3] = -(hw_energy[5] - hw_energy[3]) / (2 * (distance[5] - distance[3]))

fig_hw_force, ax_hw_force = plt.subplots()
ax_hw_force.plot(distance, hw_force_1, 'ko-', label='Analytic')
ax_hw_force.plot(distance[1:5], 2*hw_calc_force, 'ro-', label='Centred diff.')
ax_hw_force.set_xlabel('Distance / a.u')
ax_hw_force.set_ylabel('Force / a.u')
ax_hw_force.legend(frameon=True)
fig_hw_force.tight_layout()
# fig_hw_force.savefig('{}/hw-cdft.png'.format(folder), dpi=parameters.save_dpi, bbbox_inches='tight')

print('\nHW error:', (2*hw_calc_force-hw_force_1[1:5]))
print('HW average error:', np.mean(2*hw_calc_force-hw_force_1[1:5]))


if __name__ == "__main__":
    print('Finished.')
    plt.show()
