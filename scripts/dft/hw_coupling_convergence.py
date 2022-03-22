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

folder = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/he2/cutoff'

# NEWTON-LS
# name = 'newton-ls'
# value = np.array([4, 6, 8, 10, 12, 14, 16, 18])
# ac_coupling = np.array([4.407100835797, 4.384042185498, 4.367624140907, 4.265461663692, 4.326457180281, 4.354598444393, 4.469076075943])
# ac_energy = np.array([-4.879308879628371, -4.879310619265427,  -4.879310590683215, -4.879310590167445, -4.879310590186392, -4.879310590192225, -4.879310590208077])
# ec_coupling = np.array([4.407100835797, 4.384042185498, 4.367624140907, 4.265461663692, 4.326457180281, 4.354598444393, 4.469076075943])
# ec_energy = np.array([-4.879308879628371, -4.879310619265427,  -4.879310590683215, -4.879310590167445, -4.879310590186392, -4.879310590192225, -4.879310590208077])
# ag_ec_coupling = np.array([4.406160271846, 4.362320571225, 4.554313969764, 4.413282554878, 4.465397114205, 4.380396279538, 4.352800700720, 4.342561817432])
# ag_ec_energy = np.array([-4.879308091402445, -4.879310587229468, -4.879310590204842, -4.879310590200122, -4.879310590171550, -4.879310590186124,  -4.879310590182885, -4.879310590188113])
# ag_ec_strength = np.array([0.280153647861, 0.280148823913, 0.280148817871, 0.280148817864, 0.280154866825, 0.280148817865, 0.280148817864, 0.280148817864])

# DIIS
# name = 'diis'
# value = np.array([8, 10, 12, 14, 16, 18, 24])
# ag_ec_coupling = np.array([6.552309072234, 6.591358739647,  6.574003109394, 6.595738911852, 6.537916890811, 6.621984878973, 6.590992070771])
# ag_ec_energy = np.array([-4.879309336078620, -4.879309341356729, -4.879309341365603,  -4.879309341382571, -4.879309341353696,  -4.879309341394831, -4.879309341384650])
# ag_ec_strength = np.array([0.500002843806, 0.500002838547, 0.500002838541, 0.500002838557, 0.500002838531, 0.500002838570,  0.500002838538])
# ag_ec_time = np.array([425.656, 450.671, 401.280, 432.218, 430.911, 439.415, 420.017])

# Ethylene DIIS
name = 'ethylene_diis'
value = np.array([4, 8, 12, 14, 16, 18, 24, 30])
ag_ec_coupling = np.array([22.302025294462,  22.248648039547, 22.248646452435, 22.230630520143, 22.231168646410,  22.248648281048, 22.226085560753, 22.226085560753])
ag_ec_coupling2 = np.array([22.302025294462, 22.248648039548, 22.248646452435,  22.230620310417, 22.231158524669, 22.248648281048, 22.226076028316, 22.226076028316])
ag_ec_energy = np.array([-27.052932159618180,  -27.053300830499776, -27.053300830334400,  -27.053300830331690, -27.053300830333974, -27.053300830331683,  -27.053300830333239, -27.053300830333239])
ag_ec_energy2 = np.array([-27.052932159890158,  -27.053300830495111, -27.053300830326659, -27.053300829995610,  -27.053388294944067,  -27.053300830326926, -27.053301760878959, -27.053301760878959])
ag_ec_strength = np.array([0.231571235524, 0.230992672104,   0.230992670916,  0.230992671354, 0.230992671078, 0.230992671354, 0.230992671260,  0.230992671260 ])
ag_ec_strength2 = np.array([-0.231571249167, -0.230992673783, -0.230992672854,-0.228802605056, -0.228812386745, -0.230992673075,  -0.228779599434, -0.228779599434])
ag_ec_time = np.array([663.281, 597.220, 534.796, 622.988,  537.127, 565.622, 687.280, 695.479])
ag_ec_time2 = np.array([590.635, 512.359, 457.575,  687.339, 973.341,  488.876, 646.277,  651.277])
extract_values = [3, 4, 6, 7]

# Ethylene DIIS
# name = 'ethylene_diis_atomic'
# value = np.array([10, 12, 14, 16, 18, 24, 30])
# ag_ec_coupling = np.array([22.248654278918, 22.231299764614, 22.247681426023, 22.237663887522, 22.237663887522, 22.247700324260, 22.237671887233])
# ag_ec_coupling2 = np.array([22.248654278918, 22.231289508429,  22.247681427298, 22.237663888808, 22.237663888808, 22.247700325494, 22.237671888529])
# ag_ec_time = np.array([612.551, 552.545, 558.404, 608.186, 608.277, 561.434, 696.014])
# ag_ec_time2 = np.array([524.738, 1267.638, 512.106, 512.092, 511.909, 521.025, 517.026])

# Coupling against cutoff
fig_coupling, ax_coupling = plt.subplots()
ax_coupling.plot(value[:], ag_ec_coupling[:], 'ko-')
ax_coupling.plot(value[:], ag_ec_coupling2[:], 'ko--')
# ax_coupling.plot(value[extract_values], ag_ec_coupling[extract_values], 'ro')
# ax_coupling.plot(value[extract_values], ag_ec_coupling2[extract_values], 'ro')
ax_coupling.set_xlabel('Cutoff')
ax_coupling.set_ylabel('Coupling / a.u')
ax_coupling.set_xlim([value[0]-0.4, value[-1]+0.4])
fig_coupling.tight_layout()
# fig_coupling.savefig('{}/coupling_{}.png'.format(folder, name), dpi=parameters.save_dpi, bbbox_inches='tight')

# Energy against cutoff
fig_energy, ax_energy = plt.subplots()
ax_energy.plot(value[:], ag_ec_energy[:], 'ko-')
ax_energy.plot(value[:], ag_ec_energy2[:], 'ko--')
ax_energy.plot(value[extract_values], ag_ec_energy[extract_values], 'ro')
ax_energy.plot(value[extract_values], ag_ec_energy2[extract_values], 'ro')
ax_energy.set_xlabel('Cutoff')
ax_energy.set_ylabel('Energy / a.u')
ax_energy.set_xlim([value[0]-0.4, value[-1]+0.4])
fig_energy.tight_layout()
# fig_energy.savefig('{}/energy_{}.png'.format(folder, name), dpi=parameters.save_dpi, bbbox_inches='tight')

# Strength against cutoff
fig_strength, ax_strength = plt.subplots()
ax_strength.plot(value[:], np.abs(ag_ec_strength[:]), 'ko-')
ax_strength.plot(value[:], np.abs(ag_ec_strength2[:]), 'ko-')
ax_strength.plot(value[extract_values], np.abs(ag_ec_strength[extract_values]), 'ro')
ax_strength.plot(value[extract_values], np.abs(ag_ec_strength2[extract_values]), 'ro')
ax_strength.set_xlabel('Cutoff')
ax_strength.set_ylabel('Strength / a.u')
ax_strength.set_xlim([value[0]-0.4, value[-1]+0.4])
fig_strength.tight_layout()
# fig_strength.savefig('{}/strength_{}.png'.format(folder, name), dpi=parameters.save_dpi, bbbox_inches='tight')

# time against cutoff
fig_time, ax_time = plt.subplots()
ax_time.plot(value[:], ag_ec_time[:], 'ko-')
ax_time.plot(value[:], ag_ec_time2[:], 'ko--')
# ax_time.plot(value[extract_values], ag_ec_time[extract_values], 'ro')
# ax_time.plot(value[extract_values], ag_ec_time2[extract_values], 'ro')
ax_time.set_xlabel('Cutoff')
ax_time.set_ylabel('time / a.u')
ax_coupling.set_xlim([value[0]-0.4, value[-1]+0.4])
fig_time.tight_layout()
fig_time.savefig('{}/time_{}.png'.format(folder, name), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
