from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh
import matplotlib.pyplot as plt
from scripts.general import parameters

"""
    Convergence multigrid. 
    Plot convergence of energy and energy difference with multigrid.
"""

# Files
filename_save = '/scratch/cahart/work/personal_files/dft_ml_md/output/fe_bulk/convergence'

# Data
supercell_221_bandgap_pbe = np.array([[2.086693, 2.121542, 2.109360, 2.110723, 2.110785],
                                      [2.090142, 2.121966, 2.109663, 2.111074, 2.111540],
                                      [1.940041, 2.103275, 2.100139, 2.133501, 2.117503],
                                      [1.964306, 2.103208, 2.096219, 2.134382, 2.117346]])

supercell_221_bandgap_pade = np.array([[2.138960, 2.157082, 2.143721, 2.144306, 2.145232],
                                       [2.139075, 2.157481, 2.143999, 2.144671, 2.146776],
                                       [2.080838, 2.138137, 2.150063, 2.163270, 2.149276],
                                       [2.080124, 2.138049, 2.149942, 2.159523, 2.148991]])

supercell_221_bandgap_scan = np.array([[2.100822, 2.120723, 2.107489, 2.108035, 2.108254],
                                       [2.101000, 2.121092, 2.107883, 2.108245, 2.108543],
                                       [2.073657, 2.100650, 2.085092, 2.114618, 2.114426],
                                       [2.056411, 2.099524, 2.085309, 2.115166, 2.114491]])

supercell_221_pbe_time=np.array([34.35264516, 47.0166676,  30.82666418, 17.76440662, 36.61232224])
supercell_221_scan_time=np.array([32.12451881, 22.83749914, 27.69999743, 22.23676343, 66.25297966])

multigrid_neutral = np.array([600, 800, 1000, 1200, 1400])

# 221 supercell band gap experimental

fig_221_bandgap_exp, ax_221_bandgap_exp = plt.subplots()
ax_221_bandgap_exp.plot(multigrid_neutral, supercell_221_bandgap_pade[0, :], 'k+-', label='PADE s1')
ax_221_bandgap_exp.plot(multigrid_neutral, supercell_221_bandgap_pade[1, :], 'k--', label='PADE s2')
ax_221_bandgap_exp.plot(multigrid_neutral, supercell_221_bandgap_pbe[0, :], 'b+-', label='PBE s1')
ax_221_bandgap_exp.plot(multigrid_neutral, supercell_221_bandgap_pbe[1, :], 'b--', label='PBE s2')
ax_221_bandgap_exp.plot(multigrid_neutral, supercell_221_bandgap_scan[0, :], 'r+-', label='SCAN s1')
ax_221_bandgap_exp.plot(multigrid_neutral, supercell_221_bandgap_scan[1, :], 'r--', label='SCAN s2')
ax_221_bandgap_exp.set_xlabel('Multigrid cutoff')
ax_221_bandgap_exp.set_ylabel('Band gap / eV')
ax_221_bandgap_exp.set_title('Convergence of experimental band gap')
ax_221_bandgap_exp.legend(frameon=True)
fig_221_bandgap_exp.tight_layout()
fig_221_bandgap_exp.savefig('{}/221_scan/bandgap_exp'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# 221 supercell band gap
fig_221_bandgap, ax_221_bandgap = plt.subplots()
ax_221_bandgap.plot(multigrid_neutral, supercell_221_bandgap_pade[2, :], 'k+-', label='PADE s1')
ax_221_bandgap.plot(multigrid_neutral, supercell_221_bandgap_pade[3, :], 'k--', label='PADE s2')
ax_221_bandgap.plot(multigrid_neutral, supercell_221_bandgap_pbe[2, :], 'b+-', label='PBE s1')
ax_221_bandgap.plot(multigrid_neutral, supercell_221_bandgap_pbe[3, :], 'b--', label='PBE s2')
ax_221_bandgap.plot(multigrid_neutral, supercell_221_bandgap_scan[2, :], 'r+-', label='SCAN s1')
ax_221_bandgap.plot(multigrid_neutral, supercell_221_bandgap_scan[3, :], 'r--', label='SCAN s2')
ax_221_bandgap.set_xlabel('Multigrid cutoff')
ax_221_bandgap.set_ylabel('Band gap / eV')
ax_221_bandgap.set_title('Convergence of optimised band gap')
ax_221_bandgap.legend(frameon=True)
fig_221_bandgap.tight_layout()
fig_221_bandgap.savefig('{}/221_scan/bandgap'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# 221 supercell time taken
fig_221_t, ax_221_t = plt.subplots()
ax_221_t.plot(multigrid_neutral, supercell_221_pbe_time, 'b+-', label='PBE')
ax_221_t.plot(multigrid_neutral, supercell_221_scan_time, 'r+-', label='SCAN')
ax_221_t.set_xlabel('Multigrid cutoff')
ax_221_t.set_ylabel('Time on 16 nodes / s')
ax_221_t.set_title('221 time per SCF cycle')
ax_221_t.legend(frameon=True)
fig_221_t.tight_layout()
fig_221_t.savefig('{}/221_scan/time'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')


if __name__ == "__main__":
    print('Finished.')
    plt.show()
