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

""" Calculate beta value """


def slater(r, a, zeta):
    """" Lambda according to Marcus continuum theory """

    return a * np.exp(-zeta * r)


def voorhis(r):
    """" Lambda according to Marcus continuum theory """

    return 1/4 * (r**2 - r**4)


folder = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/MgO/density_matrix'

# Zn
end = 10
hab_zn_alpha = np.array([0.12, 0.06, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
hab_zn_beta = np.array([0.97, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])
y_alpha = np.linspace(1, end, np.shape(hab_zn_alpha)[0])
y_beta = np.linspace(1, end, np.shape(hab_zn_beta)[0])
fig_zn, ax_zn = plt.subplots()
ax_zn.plot(y_alpha, hab_zn_alpha, 'r.--', label=r'|$\mathrm{\lambda^{(\alpha)}}$|')
ax_zn.plot(y_beta, hab_zn_beta, 'g.--', label=r'|$\mathrm{\lambda^{(\beta)}}$|')
ax_zn.set_xlim([0.5, 10.5])
ax_zn.set_ylabel(r'|$\mathrm{\lambda}$|')
ax_zn.legend(frameon=False)
fig_zn.tight_layout()
# fig_zn.savefig('{}/plotted/{}.png'.format(folder, 'zn'), dpi=parameters.save_dpi, bbbox_inches='tight')

# Hab11
end = 10
hab_ethylene_alpha = np.array([0.11, 0.09, 0.09, 0.07, 0.06, 0.04, 0.04, 0.04, 0.04, 0.03, 0.03, 0.02, 0.01])
hab_ethylene_beta = np.array([0.62, 0.11, 0.07, 0.06, 0.06, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01])
y_alpha = np.linspace(1, end, np.shape(hab_ethylene_alpha)[0])
y_beta = np.linspace(1, end, np.shape(hab_ethylene_beta)[0])
fig_ethylene, ax_ethylene = plt.subplots()
ax_ethylene.plot(y_alpha, hab_ethylene_alpha, 'r.--', label=r'|$\mathrm{\lambda^{(\alpha)}}$|')
ax_ethylene.plot(y_beta, hab_ethylene_beta, 'g.--', label=r'|$\mathrm{\lambda^{(\beta)}}$|')
ax_ethylene.set_xlim([0.5, 10.5])
ax_ethylene.set_ylabel(r'|$\mathrm{\lambda}$|')
ax_ethylene.legend(frameon=False)
fig_ethylene.tight_layout()
# fig_ethylene.savefig('{}/plotted/{}.png'.format(folder, 'ethylene'), dpi=parameters.save_dpi, bbbox_inches='tight')

# MgO
cols1 = ['Value']
name = 'cell-111-288-14A'
end = 200
print(name)

hab_mgo_alpha = pd.read_csv('{}/{}_alpha.out'.format(folder, name), names=cols1, delim_whitespace=True)
hab_mgo_alpha = -hab_mgo_alpha.to_numpy()[:end, 0]
y_alpha = np.linspace(1, end, np.shape(hab_mgo_alpha)[0])
hab_mgo_beta = pd.read_csv('{}/{}_beta.out'.format(folder, name), names=cols1, delim_whitespace=True)
hab_mgo_beta = -hab_mgo_beta.to_numpy()[:end, 0]
y_beta = np.linspace(1, end, np.shape(hab_mgo_beta)[0])

fig_mgo, ax_mgo = plt.subplots()
ax_mgo.plot(y_alpha, hab_mgo_alpha, 'r.--', label=r'|$\mathrm{\lambda^{(\alpha)}}$|')
ax_mgo.plot(y_beta, hab_mgo_beta, 'g.--', label=r'|$\mathrm{\lambda^{(\beta)}}$|')
ax_mgo.set_xlim([0.5, 10.5])
ax_mgo.set_ylabel(r'|$\mathrm{\lambda}$|')
ax_mgo.legend(frameon=False)
fig_mgo.tight_layout()
# fig_mgo.savefig('{}/plotted/{}.png'.format(folder, name), dpi=parameters.save_dpi, bbbox_inches='tight')

# MgO comparison
cols1 = ['Value']
name = 'cell-100-192-12A'
end = 200
print(name)

hab_mgo_alpha2 = pd.read_csv('{}/{}_alpha.out'.format(folder, name), names=cols1, delim_whitespace=True)
hab_mgo_alpha2 = -hab_mgo_alpha2.to_numpy()[:end, 0]
y_alpha2 = np.linspace(1, end, np.shape(hab_mgo_alpha2)[0])
hab_mgo_beta2 = pd.read_csv('{}/{}_beta.out'.format(folder, name), names=cols1, delim_whitespace=True)
hab_mgo_beta2 = -hab_mgo_beta2.to_numpy()[:end, 0]
y_beta2 = np.linspace(1, end, np.shape(hab_mgo_beta2)[0])

fig_mgo2a, ax_mgo2a = plt.subplots()
ax_mgo2a.plot(y_alpha, hab_mgo_alpha, 'r.--', label=r'|$\mathrm{\lambda^{(\alpha)}}$| cell-111-288-24')
ax_mgo2a.plot(y_beta, hab_mgo_beta, 'g.--', label=r'|$\mathrm{\lambda^{(\beta)}}$| cell-111-288-24')
ax_mgo2a.plot(y_alpha2, hab_mgo_alpha2, 'm.--', label=r'|$\mathrm{\lambda^{(\alpha)}}$| cell-100-192-18')
ax_mgo2a.plot(y_beta2, hab_mgo_beta2, 'c.--', label=r'|$\mathrm{\lambda^{(\beta)}}$| cell-100-192-18')
ax_mgo2a.set_xlim([0.5, 10.5])
ax_mgo2a.set_ylabel(r'|$\mathrm{\lambda}$|')
ax_mgo2a.legend(frameon=False)
fig_mgo2a.tight_layout()
# fig_mgo2a.savefig('{}/plotted/{}.png'.format(folder, 'comparison_211-240-3_111-288-24'), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
