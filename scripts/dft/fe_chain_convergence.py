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
from scripts.general import parameters
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh

""" Calculate convergence for 1D hematite chain """

folder = 'E:/University/PhD/Programming/dft_ml_md/output/fe_chain/convergence'

# Shwarz convergence
schwarz = np.array([6, 8, 10, 12])

hf_dupuis_neutral = np.array([-1322.502767900930394, -1322.503590539827201, -1322.503598383957296, -1322.503598450040727])
hf_dupuis_electron = np.array([-1322.892561916700515, -1322.893470795454959,  -1322.893477361293208, -1322.893477423059267])
hf_dupuis_ea = hf_dupuis_electron - hf_dupuis_neutral

hf_neutral = np.array([-1323.257737551242826, -1323.258562699226331, -1323.258569615200713, -1323.258569674059572])
hf_electron = np.array([ -1323.573436203685105, -1323.574344259974396, -1323.574349775992005, -1323.574349833658289])
hf_ea = hf_electron - hf_neutral

hf_dupuis_time = np.array([34.767, 41.243, 48.501, 57.042])
hf_normal_time = np.array([286.316, 379.509, 440.389, 518.269])

fig_dupuis, ax_dupuis = plt.subplots()
ax_dupuis.plot(schwarz, hf_dupuis_neutral, 'kx-')
ax_dupuis.set_ylabel('Energy / au')
ax_dupuis.set_xlabel('Schwarz screening')
fig_dupuis.tight_layout()
fig_dupuis.savefig('{}/hf-dupuis_neutral.png'.format(folder), dpi=parameters.save_dpi, bbbox_inches='tight')

fig_normal, ax_normal = plt.subplots()
ax_normal.plot(schwarz, hf_neutral, 'kx-')
ax_normal.set_ylabel('Energy / au')
ax_normal.set_xlabel('Schwarz screening')
fig_normal.tight_layout()
fig_normal.savefig('{}/hf_neutral.png'.format(folder), dpi=parameters.save_dpi, bbbox_inches='tight')

# fig_ea, ax_ea = plt.subplots()
# ax_ea.plot(schwarz, hf_ea, 'rx-')
# ax_ea.plot(schwarz, hf_dupuis_ea, 'bx-')
# ax_ea.set_ylabel('Energy / au')
# ax_ea.set_xlabel('Schwarz screening')
# fig_ea.tight_layout()

fig_time, ax_time = plt.subplots()
ax_time.plot(schwarz, hf_normal_time, 'rx-', label='HF')
ax_time.plot(schwarz, hf_dupuis_time, 'gx-', label='HF (Dupuis)')
ax_time.set_ylabel('Time / s')
ax_time.set_xlabel('Schwarz screening')
ax_time.legend(frameon=False)
fig_time.tight_layout()
fig_time.savefig('{}/time.png'.format(folder), dpi=parameters.save_dpi, bbbox_inches='tight')


if __name__ == "__main__":
    print('Finished.')
    plt.show()
