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
from scripts.general import parameters as param
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh

""" Calculate beta value """

folder = 'E:/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/331_supercell_cdft/plots'

energy1 = np.array([-15942.78926130987929, -15942.78913498106340,
                   -15942.78408495839903, -15942.78401070803920,  -15942.78404870857958])
energy2 = -15942.8034639966
error = np.array([1.061E+00, -4.517E-01,  -1.664E-01,  -4.289E-02, -2.469E-02])

# Plot energy against error
# fig_hab_pbe, ax_hab_pbe = plt.subplots()
# ax_hab_pbe.plot(np.abs(energy2-energy1[1:])*param.hartree_to_ev, error[1:], 'kx-')
# ax_hab_pbe.set_xlabel('Lambda / eV')
# ax_hab_pbe.set_ylabel('Deviation from target')
# fig_hab_pbe.tight_layout()
# # fig_hab_pbe.savefig('{}/lambda_convergence.png'.format(folder), dpi=300, bbbox_inches='tight')
# 
# print(np.abs(energy2-energy1[1:])*param.hartree_to_ev)
# print(error[1:])

distance = np.array([2.97, 5.05, 5.87, 7.72])
coupling = np.array([[0.25, 0.23, 0.15, 0.12],
                     [0.18, 0.18, 0.09, 0.10],
                     [0.22, 0.22, 0.09, 0.13],
                     [0.28, 0.28, 0.16, 0.17]])

# Plot Hab
fig_hab, ax_hab = plt.subplots()
ax_hab.plot(distance, coupling[:, 1], 'kx-', label='Fe (abs)')
ax_hab.plot(distance, coupling[:, 0], 'rx', label='Fe (diff)')
ax_hab.plot(distance, coupling[:, 3], 'bx', label='FeO (abs)')
ax_hab.plot(distance, coupling[:, 2], 'gx', label='FeO (diff)')
ax_hab.set_xlim([2, 9])
ax_hab.set_xlabel(r'Distance / $\mathrm{\AA}$')
ax_hab.set_ylabel('Reorgansiation energy / eV')
ax_hab.legend(frameon=True, loc='lower right')
fig_hab.tight_layout()
fig_hab.savefig('{}/lambda.png'.format(folder), dpi=300, bbbox_inches='tight')


if __name__ == "__main__":
    print('Finished.')
    plt.show()
