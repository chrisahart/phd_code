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
    Plot energy for ru-ru benchmark 
"""

folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/'

iasd_ru_pbe = np.array([1.09, 1.09, 1.09, 1.09])
iasd_ru_blyp = np.array([1.08, 1.08, 1.08])
iasd_h2 = np.array([1])
iasd_h2o = np.array([1.075175311])
iasd_pyrene = np.array([1.450032059, 1.498048961, 1.460101756, 1.203279553, 1.23154839, 1.213224122, 1.272419551, 1.243528423])
iasd_pentacene = np.array([1.565107347, 1.34136084, 1.401492948])
iasd_pentacene2 = np.array([1.335425959])
iasd_mgo_pbe = np.array([1.044639076, 1.052647814, 1.042107635, 1.039153038, 1.040044798])
iasd_mgo_pbe0 = np.array([1.044685611, 1.063187843, 1.046291942, 1.042885271, 1.051400614])

energy_ru_pbe = np.array([0.11, 0.046, 0.093, 0.046])
energy_ru_blyp = np.array([0.11, 0.092, 0.031])
energy_h2 = np.array([0.0669872835])
energy_h2o = np.array([0.069164406])
energy_pyrene = np.array([0.0368315064, 0.0657122439, 0.0495980536, 0.0063766282, 0.014386712, 0.0097220911, 0.0372923312, 0.02748587612])
energy_pentacene = np.array([0.0296757795, 0.0058603222, 0.0220866699])
energy_pentacene2 = np.array([0.0122370681])
energy_mgo_pbe = np.array([0.0567932074, 0.05354758728, 0.04037970259, 0.05611235527, 0.03083513794])
energy_mgo_pbe0 = np.array([0.03606613397, 0.04897881259, 0.04136302962, 0.04667665773, 0.02841257439])

# Plot total energy difference to DFT
# fig_metric, ax_metric = plt.subplots()
# ax_metric.plot(iasd_ru_pbe[1:], energy_ru_pbe[1:], 'kx', label='Ru (PBE)')
# ax_metric.plot(iasd_ru_pbe[0], energy_ru_pbe[0], 'rx')
# ax_metric.plot(iasd_ru_blyp[1:], energy_ru_blyp[1:], 'k+', label='Ru (BLYP)')
# ax_metric.plot(iasd_ru_blyp[0], energy_ru_blyp[0], 'r+')
# ax_metric.plot(iasd_h2, energy_h2, 'ko', label='H2 (PBE)')
# ax_metric.plot(iasd_h2o, energy_h2o, 'ks', label='H2O (PBE)')
# ax_metric.plot(iasd_pyrene, energy_pyrene, 'rD', label='Pyrene (PBE)')
# ax_metric.plot(iasd_pentacene, energy_pentacene, 'r^', label='Pentacene (PBE)')
# ax_metric.plot(iasd_mgo_pbe, energy_mgo_pbe, 'k*', label='MgO (PBE)')
# ax_metric.plot(iasd_mgo_pbe0, energy_mgo_pbe0, 'kP', label='MgO (PBE0)')
# ax_metric.set_xlabel('Integrated Absolute Spin Density')
# ax_metric.set_ylabel('Energy / Ha')
# ax_metric.set_xlim([0.98, 1.6])
# ax_metric.set_ylim([0, 0.12])
# ax_metric.legend(frameon=True)
# fig_metric.tight_layout()
# fig_metric.savefig('{}/energy_change_dft_t{}.png'.format(folder_save, time_plot), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot total energy difference to DFT
fig_metric2, ax_metric2 = plt.subplots()
ax_metric2.plot(energy_ru_pbe[1:], iasd_ru_pbe[1:], 'kP', label=r'Ru$^{2+}$-Ru$^{3+}$')
ax_metric2.plot(energy_ru_pbe[0], iasd_ru_pbe[0], 'rP')
# ax_metric2.plot(energy_ru_blyp[1:], iasd_ru_blyp[1:], 'kx')
# ax_metric2.plot(energy_ru_blyp[0], iasd_ru_blyp[0], 'rx')
ax_metric2.plot(energy_mgo_pbe, iasd_mgo_pbe, 'k*', label='MgO')
ax_metric2.plot(energy_mgo_pbe0, iasd_mgo_pbe0, 'k*')
ax_metric2.plot(energy_pyrene, iasd_pyrene, 'rD', label='Pyrene-COF')
ax_metric2.plot(energy_pentacene, iasd_pentacene, 'r^', label='Pentacene')
ax_metric2.plot(energy_h2, iasd_h2, 'go', label=r'H$_{2}$', fillstyle='none')
ax_metric2.plot(energy_h2o, iasd_h2o, 'gs', label=r'H$_{2}$O', fillstyle='none')
ax_metric2.plot(energy_pentacene2, iasd_pentacene2, 'g^', label='Pentacene', fillstyle='none')
ax_metric2.set_ylabel('Integrated Absolute Spin Density')
ax_metric2.set_xlabel('CDFT-DFT energy difference / Ha')
ax_metric2.set_ylim([0.98, 1.6])
ax_metric2.set_xlim([0, 0.12])
# ax_metric2.set_xlim([0, 1400])
# ax_metric2.set_ylim([-1e-5, 4e-3])
ax_metric2.legend(frameon=True)
fig_metric2.tight_layout()
fig_metric2.savefig('{}/metric.png'.format(folder_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot total energy difference to DFT
# fig_metric2, ax_metric2 = plt.subplots()
# ax_metric2.plot(energy_h2,iasd_h2,  'go', label='H2 (PBE)', fillstyle='none')
# ax_metric2.plot(energy_h2o, iasd_h2o, 'gs', label='H2O (PBE)', fillstyle='none')
# ax_metric2.plot(energy_pentacene2,iasd_pentacene2,  'g^', label='Pentacene (PBE)', fillstyle='none')
# ax_metric2.plot(energy_ru_pbe[1:], iasd_ru_pbe[1:], 'kx', label='Ru (PBE)')
# ax_metric2.plot(energy_ru_pbe[0], iasd_ru_pbe[0], 'rx')
# ax_metric2.plot(energy_ru_blyp[1:], iasd_ru_blyp[1:], 'k+', label='Ru (BLYP)')
# ax_metric2.plot(energy_ru_blyp[0], iasd_ru_blyp[0], 'r+')
# ax_metric2.plot( energy_mgo_pbe,iasd_mgo_pbe, 'k*', label='MgO (PBE)')
# ax_metric2.plot(energy_mgo_pbe0, iasd_mgo_pbe0,  'kP', label='MgO (PBE0)')
# ax_metric2.plot(energy_pyrene,iasd_pyrene,  'rD', label='Pyrene (PBE)')
# ax_metric2.plot(energy_pentacene,iasd_pentacene,  'r^', label='Pentacene (PBE)')
# ax_metric2.set_ylabel('Integrated Absolute Spin Density')
# ax_metric2.set_xlabel('CDFT-DFT energy difference / Ha')
# ax_metric2.set_ylim([0.98, 1.6])
# ax_metric2.set_xlim([0, 0.12])
# ax_metric2.set_xlim([0, 1400])
# ax_metric2.set_ylim([-1e-5, 4e-3])
# ax_metric2.legend(frameon=True)
# fig_metric2.tight_layout()
# fig_metric2.savefig('{}/metric.png'.format(folder_save), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
