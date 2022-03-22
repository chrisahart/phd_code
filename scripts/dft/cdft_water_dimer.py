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


"""
    Plot energy and forces for water dimer
"""

def angle(v1, v2, acute):
# v1 is your firsr vector
# v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if (acute == True):
        return angle
    else:
        return 2 * np.pi - angle

# Data 1
start = -900
# folder1 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/water_dimer/md/archer2/dimer-small-pbc/analysis/all/nve-hirshfeld-charge7_eps-1e-5'
folder1 = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/water_dimer/md/archer2/dimer-small-frozen/analysis/all/nve-hirshfeld-charge7_eps-1e-4'
coord1, coord_x1, coord_y1, coord_z1, _, num_atoms1, _ = load_coordinates.load_values_coord(folder1, 'water_dimer-pos-1.xyz')
energy_kinetic1, energy_potential1, _, _, time_val1, time_per_step1 = load_energy.load_values_energy(folder1, 'water_dimer-1.ener')

# Calculate distances and forces
distances1, distances1_x, distances1_y, distances1_z = functions.distances_md(coord_x1, coord_y1, coord_z1, num_atoms1, len(time_per_step1))
print('O1-H1', np.mean(distances1[start:-1, 0]))
print('O1-H2', np.mean(distances1[start:-1, 1]))
print('O2-H1', np.mean(distances1[start:-1, 12]))
print('O2-H2', np.mean(distances1[start:-1, 13]))

# Calculate angles
angle1 = np.zeros( len(time_per_step1))
angle2 = np.zeros( len(time_per_step1))
for i in range(0, len(time_per_step1)):
    vector_1_1 = np.array([distances1_x[i, 0], distances1_y[i, 0], distances1_z[i, 0]])
    vector_1_2 = np.array([distances1_x[i, 1], distances1_y[i, 1], distances1_z[i, 1]])
    vector_2_1 = np.array([distances1_x[i, 12], distances1_y[i, 12], distances1_z[i, 12]])
    vector_2_2 = np.array([distances1_x[i, 13], distances1_y[i, 13], distances1_z[i, 13]])
    unit_vector_1_1 = vector_1_1 / np.linalg.norm(vector_1_1)
    unit_vector_1_2 = vector_1_2 / np.linalg.norm(vector_1_2)
    unit_vector_2_1 = vector_2_1 / np.linalg.norm(vector_2_1)
    unit_vector_2_2 = vector_2_2 / np.linalg.norm(vector_2_2)
    angle1[i] = np.degrees(np.arccos(np.dot(unit_vector_1_1, unit_vector_1_2)))
    angle2[i] = np.degrees(np.arccos(np.dot(unit_vector_2_1, unit_vector_2_2)))
print('angle1', np.mean(angle1[start:-1]))
print('angle2', np.mean(angle2[start:-1]))

# Compare distances
fig_distance1, ax_distance1 = plt.subplots()
ax_distance1.plot(time_val1, distances1[:, 0], 'r', alpha=0.4)
ax_distance1.plot([time_val1[0], time_val1[-1]], [np.mean(distances1[start:-1, 0]), np.mean(distances1[start:-1, 0])], 'r', label='O1-H1')
ax_distance1.plot(time_val1, distances1[:, 1], 'g', alpha=0.4)
ax_distance1.plot([time_val1[0], time_val1[-1]], [np.mean(distances1[start:-1, 1]), np.mean(distances1[start:-1, 1])], 'g', label='O1-H2')
ax_distance1.plot(time_val1, distances1[:, 12], 'b', alpha=0.4)
ax_distance1.plot([time_val1[0], time_val1[-1]], [np.mean(distances1[start:-1, 12]), np.mean(distances1[start:-1, 12])], 'b', label='O2-H3')
ax_distance1.plot(time_val1, distances1[:, 13], 'y', alpha=0.4)
ax_distance1.plot([time_val1[0], time_val1[-1]], [np.mean(distances1[start:-1, 13]), np.mean(distances1[start:-1, 13])], 'y', label='O2-H4')
ax_distance1.set_ylabel(r'Bond length$ \ / \ \mathrm{\AA}$')
ax_distance1.set_xlabel('Time / fs')
ax_distance1.legend(frameon=True)
fig_distance1.tight_layout()
fig_distance1.savefig('{}/distances.png'.format(folder1), dpi=parameters.save_dpi, bbbox_inches='tight')

# Compare angles
fig_angle1, ax_angle1 = plt.subplots()
ax_angle1.plot(time_val1, angle1, 'r', alpha=0.4)
ax_angle1.plot([time_val1[0], time_val1[-1]], [np.mean(angle1[start:-1]), np.mean(angle1[start:-1])], 'r', label='H1-O1-H2')
ax_angle1.plot(time_val1, angle2, 'g', alpha=0.4)
ax_angle1.plot([time_val1[0], time_val1[-1]], [np.mean(angle2[start:-1]), np.mean(angle2[start:-1])], 'g', label='H3-O2-H4')
ax_angle1.set_ylabel('Angle')
ax_angle1.set_xlabel('Time / fs')
ax_angle1.legend(frameon=True)
fig_angle1.tight_layout()
fig_angle1.savefig('{}/angles.png'.format(folder1), dpi=parameters.save_dpi, bbbox_inches='tight')


if __name__ == "__main__":
    print('Finished.')
    plt.show()