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


""" Plot bond lengths of Fe chain """

folder = 'E:/University/PhD/Programming/dft_ml_md/output/fe_chain/bond_lengths'

unit_7_hf_dupuis = np.array([1.94996, 2.02194, 1.91086, 2.11604, 2.11510, 1.91084, 2.02202, 1.94990, 1.99286, 1.97059, 1.98114, 1.98104, 1.97065])
unit_7_hf_dupuis = np.roll(unit_7_hf_dupuis, 2)

unit_7_hf = np.array([1.96849, 1.97791, 1.95014, 2.00511, 1.91020, 2.09082, 2.08748, 1.91110,  2.00539, 1.95007, 1.97780, 1.96874, 1.97174])
unit_7_hf = np.roll(unit_7_hf, 0)

unit_7_pbe = np.array([1.97347, 2.02625, 1.97662, 2.02488, 1.98306, 1.99613, 1.97404, 2.02329, 1.98285, 1.99545, 1.97453, 2.02279, 1.98349])
unit_7_pbe = np.roll(unit_7_pbe, 2)

unit_7_hse = np.array([1.98960, 1.97051, 1.96174, 1.99690, 1.96430, 1.99233, 1.95092, 2.01065, 1.97457, 1.99479, 1.95961, 2.00654, 1.98478])
unit_7_hse = np.roll(unit_7_hse, 2)

unit_7 = np.linspace(-3, 3, num=np.shape(unit_7_hf_dupuis)[0])

unit_9_hf_dupuis = np.array([1.96762, 1.99766, 1.94670, 2.02691, 1.90881, 2.11661, 2.11655, 1.90884, 2.02706, 1.94659, 1.99747, 1.96752, 1.98780, 1.97637, 1.98192, 1.98214, 1.97652])
unit_9_hf_dupuis = np.roll(unit_9_hf_dupuis, 2)

unit_9_hf = np.array([])
# unit_9_hf = np.roll(unit_9_hf, 2)

unit_9 = np.linspace(-4, 4, num=np.shape(unit_9_hf_dupuis)[0])
unit = np.linspace(-10, 10, num=np.shape(unit_9_hf_dupuis)[0])
neutral_hf_dupuis = np.ones(np.shape(unit_9_hf_dupuis)[0])*1.98059
neutral_pbe = np.ones(np.shape(unit_9_hf_dupuis)[0])*1.98059
neutral_hse = np.ones(np.shape(unit_9_hf_dupuis)[0])*1.98059

fig_feoh, ax_feoh = plt.subplots()

# ax_feoh.plot(unit, neutral_hf_dupuis, 'grey', label='Neutral (Dupuis)', alpha=0.6)
# ax_feoh.plot(unit_7, unit_7_hf_dupuis, 'bo--', fillstyle='full', label='7 units')
# ax_feoh.plot(unit_9, unit_9_hf_dupuis, 'go--', fillstyle='full', label='9 units')

# ax_feoh.plot(unit, neutral_hf_dupuis, 'grey', label='Neutral (Dupuis)', alpha=0.6)
# ax_feoh.plot(unit_7, unit_7_hf_dupuis, 'bo--', fillstyle='full', label='7 units (HF Dupuis)')
# ax_feoh.plot(unit_7, unit_7_hf, 'ko--', fillstyle='full', label='7 units (HF MOLOPT)')
ax_feoh.plot(unit_9, unit_9_hf_dupuis, 'go--', fillstyle='full', label='9 units (HF Dupuis)')
ax_feoh.plot(unit_9, unit_9_hf, 'ko--', fillstyle='full', label='9 units (HF MOLOPT)')

# ax_feoh.plot(unit_7, unit_7_pbe, 'ro--', fillstyle='full', label='7 units (PBE)')
# ax_feoh.plot(unit_7, unit_7_hse, 'go--', fillstyle='full', label='7 units (HSE)')
# ax_feoh.plot(unit_7, unit_7_hf, 'ko--', fillstyle='full', label='7 units (HF MOLOPT)')
# ax_feoh.plot(unit_7, unit_7_hf_dupuis, 'ko--', fillstyle='full', label='7 units (HF Dupuis)')

# ax_feoh.set_ylabel(r'Fe-OH distance / $\mathrm{\AA}$')
# ax_feoh.set_xlabel('Bond number')
ax_feoh.set_xlim([-5, 5])
ax_feoh.set_ylim([1.9, 2.2])  # Dupuis
# ax_feoh.set_ylim([1.85, 2.15])  # Rosso
ax_feoh.legend(frameon=False)
fig_feoh.tight_layout()
# fig_feoh.savefig('{}/hf_simple_feoh_dupuis_transparent.png'.format(folder),
#                  transparent=True, dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
