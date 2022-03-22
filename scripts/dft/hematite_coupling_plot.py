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

""" Plot hematite hole couplings 
     """

folder = 'E:/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/331_supercell_cdft/plots'

distance = np.array([2.97, 5.05, 5.87, 7.72])
coupling = np.array([[68.3, 56.5, 76.7, 63.5],
                     [1.7, 2.5, 4.0, 0.9],
                     [8.3, 5.4, 17.7, 11.6],
                     [40.8, 27.3, 63.8, 50.9]])

# Plot Hab
fig_hab, ax_hab = plt.subplots()
ax_hab.plot(distance, coupling[:, 1], 'kx-', label='Fe (abs)')
ax_hab.plot(distance, coupling[:, 0], 'rx', label='Fe (diff)')
ax_hab.plot(distance, coupling[:, 3], 'bx', label='FeO (abs)')
ax_hab.plot(distance, coupling[:, 2], 'gx', label='FeO (diff)')
ax_hab.set_xlim([2, 9])
ax_hab.set_xlabel(r'Distance / $\mathrm{\AA}$')
ax_hab.set_ylabel(r'|H$_\mathrm{ab}$| / meV')
ax_hab.legend(frameon=True, loc='lower left')
fig_hab.tight_layout()
fig_hab.savefig('{}/hab.png'.format(folder), dpi=300, bbbox_inches='tight')

# Plot Hab log 
fig_hab_log, ax_hab_log = plt.subplots()
ax_hab_log.plot(distance, coupling[:, 1], 'kx-', label='Fe (abs)')
ax_hab_log.plot(distance, coupling[:, 0], 'rx', label='Fe (diff)')
ax_hab_log.plot(distance, coupling[:, 3], 'bx', label='FeO (abs)')
ax_hab_log.plot(distance, coupling[:, 2], 'gx', label='FeO (diff)')
ax_hab_log.set_ylim([0.7,  1e2])
ax_hab_log.set_xlim([2, 9])
ax_hab_log.set_yscale('log')
ax_hab_log.set_xlabel(r'Distance / $\mathrm{\AA}$')
ax_hab_log.set_ylabel(r'|H$_\mathrm{ab}$| / meV')
ax_hab_log.legend(frameon=True, loc='lower left')
fig_hab_log.tight_layout()
fig_hab_log.savefig('{}/hab_log.png'.format(folder), dpi=300, bbbox_inches='tight')


if __name__ == "__main__":
    print('Finished.')
    plt.show()
