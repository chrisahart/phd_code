from __future__ import division, print_function
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
import scipy
import re
import pickle
import pandas as pd
from distutils.dir_util import copy_tree
import copy
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh

""" Calculate beta value """


def calc_lambda(r, a, b):
    """" Lambda according to Marcus theory """

    # return a - b * np.exp(-zeta * r)
    return a * (1 - b/r)


folder = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/MgO/plotting'

# MgO
d_hab = np.array([3.5, 4.0, 4.5, 5.0])
hab_pbe_hab11 = np.array([408.9, 274.6, 134.4, 63.8])
hab_ref_hab11 = np.array([382.3, 176.8, 78.4, 32.5])
hab_pbe_h = np.array([638.9, 228.8, 98.0, 40.2])

hab_pbe_hab7 = np.array([207.3, 105.0, 61.6, 36.7])
hab_ref_hab7 = np.array([411.0, 198.0, 92.4, 41.0])
hab_pbe_e = np.array([708.3, 250.9, 111.2, 49.3])
pbe_values = [1, 2, 3]
d_fine = np.linspace(0, 100, num=int(1e3))

hab_pbe_hab11_fit = np.polyfit(d_hab[pbe_values], np.log(hab_pbe_hab11[pbe_values]), 1)
hab_ref_hab11_fit = np.polyfit(d_hab[pbe_values], np.log(hab_ref_hab11[pbe_values]), 1)
hab_pbe_hab7_fit = np.polyfit(d_hab[pbe_values], np.log(hab_pbe_hab7[pbe_values]), 1)

hab_pbe_ref_fit = np.polyfit(d_hab[pbe_values], np.log(hab_ref_hab7[pbe_values]), 1)
hab_pbe_h_fit = np.polyfit(d_hab[pbe_values], np.log(hab_pbe_h[pbe_values]), 1)
hab_pbe_e_fit = np.polyfit(d_hab[pbe_values], np.log(hab_pbe_e[pbe_values]), 1)

print('\nPBE HAB11 beta', 2 * -hab_pbe_hab11_fit[0], 2.7 + 2*hab_pbe_hab11_fit[0])
print('Ref HAB11 beta', 2 * -hab_ref_hab11_fit[0], 2.7 + 2*hab_ref_hab11_fit[0])
print('PBE h beta', 2 * -hab_pbe_h_fit[0], 2.7 + 2*hab_pbe_h_fit[0])

print('\nPBE HAB7 beta', 2 * -hab_pbe_hab7_fit[0], 2.7 + 2*hab_pbe_hab7_fit[0])
print('Ref HAB7 beta', 2 * -hab_pbe_ref_fit[0], 2.7 + 2*hab_pbe_ref_fit[0])
print('PBE e beta', 2 * -hab_pbe_e_fit[0], 2.7 + 2*hab_pbe_e_fit[0])

# Plot Hab (BW, HW, CPMD)
fig_hab_pbe, ax_hab_pbe = plt.subplots()
# ax_hab_pbe.plot(d_fine, np.exp(d_fine*hab_pbe_hab11_fit[0] + hab_pbe_hab11_fit[1]), 'r-')
# ax_hab_pbe.plot(d_fine, np.exp(d_fine*hab_pbe_hab7_fit[0] + hab_pbe_hab7_fit[1]), 'k-')
# ax_hab_pbe.plot(d_fine, np.exp(d_hab*hab_pbe0_hw_fit2[0] + hab_pbe0_hw_fit2[1]), 'b-', label='CP2K')
# ax_hab_pbe.plot(d_fine, np.exp(d_fine*hab_pbe_h_fit[0] + hab_pbe_h_fit[1]), 'r-', label='CP2K')
# ax_hab_pbe.plot(d_hab, np.exp(d_fine*hab_pbe_e_fit[0] + hab_pbe_e_fit[1]), 'k-', label='CPMD')
# ax_hab_pbe.plot(d_hab, hab_pbe_hab11, 'ro', fillstyle='none')
# ax_hab_pbe.plot(d_hab, hab_pbe_h, 'ko', fillstyle='none')
ax_hab_pbe.set_ylim([1e1,  1e3])
ax_hab_pbe.set_xlim([3.4, 5.2])
ax_hab_pbe.set_yscale('log')
# ax_hab_pbe.set_xlabel(r'Distance / $\mathrm{\AA}$')
# ax_hab_pbe.set_ylabel(r'|H$_\mathrm{ab}$| / meV')
# ax_hab_pbe.legend(frameon=False, loc='upper left')
fig_hab_pbe.tight_layout()
# fig_hab_pbe.savefig('{}/hab.png'.format(folder), dpi=300, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
