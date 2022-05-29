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


def calc_lambda(r, a, b):
    """" Lambda according to Marcus theory """

    # return a - b * np.exp(-zeta * r)
    return a * (1 - b/r)


folder = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/MgO/plotting'

# MgO
d_hab = np.array([5.21, 6.02, 6.73, 7.37, 8.51, 9.03, 12.76, 14.74])
d_lambda = np.array([5.21, 6.02, 7.37, 9.03, 12.76, 14.74])
d_lambda_fine = np.linspace(d_lambda[0], d_lambda[-1], num=50)

# Hab values
hab_pbe_cpmd = np.array([641.7, 899.2, 353.9, 485.5, 264.1, 440.8, 1447.5, 1507.9])
hab_pbe_bw = np.array([914.6, 1332.1, 536.4, 882.4, 663.5, 874.4, 2245.1, 2246.5])
hab_pbe_hw = np.array([797.9, 1186.2, 457.2, 705.1, 459.1, 618.6, 1897.5, 1930.8])
pbe_values = [0, 1, 2, 3, 4, 5]

hab_pbe0_cpmd = np.array([432.1, 537.0, 161.3, 183.7, 81.0, 142.5, 44.6, 16.6])
# hab_pbe0_cpmd = np.array([418.6, 503.3, 139.5, 112.9, 64.6, 126.0, 28.2, 10.8])  # size corrected
hab_pbe0_bw = np.array([613.4, 733.4, 173.6, 239.8, 92.6, 148.2, 145.3, 85.7])
hab_pbe0_hw = np.array([577.1, 677.9, 197.2, 238.6, 92.9, 150.4, 95.8, 52.2])

hab_pbe_cpmd_fit, covariance_pbe_cpmd = np.polyfit(d_hab[pbe_values], np.log(hab_pbe_cpmd[pbe_values]), 1, cov=True)
hab_pbe_bw_fit, covariance_pbe_bw = np.polyfit(d_hab[pbe_values], np.log(hab_pbe_bw[pbe_values]), 1, cov=True)
hab_pbe_hw_fit, covariance_pbe_hw = np.polyfit(d_hab[pbe_values], np.log(hab_pbe_hw[pbe_values]), 1, cov=True)

print('\nPBE CPMD beta', 2 * -hab_pbe_cpmd_fit[0], 2.7 + 2*hab_pbe_cpmd_fit[0], 'pm', np.sqrt(np.diag(covariance_pbe_cpmd)))
print('PBE BW beta', 2 * -hab_pbe_bw_fit[0], 2.7 + 2*hab_pbe_bw_fit[0], 'pm', np.sqrt(np.diag(covariance_pbe_bw)))
print('PBE HW beta', 2 * -hab_pbe_hw_fit[0], 2.7 + 2*hab_pbe_hw_fit[0], 'pm', np.sqrt(np.diag(covariance_pbe_hw)))

hab_pbe0_cpmd_fit, covariance_pbe0_cpmd = np.polyfit(d_hab, np.log(hab_pbe0_cpmd), 1, cov=True)
hab_pbe0_bw_fit, covariance_pbe0_bw = np.polyfit(d_hab, np.log(hab_pbe0_bw), 1,  cov=True)
hab_pbe0_hw_fit, covariance_pbe0_hw = np.polyfit(d_hab, np.log(hab_pbe0_hw), 1, cov=True)
# hab_pbe0_hw_fit2 = np.polyfit(d_hab[:-2], np.log(hab_pbe0_hw[:-2]), 1)

print('\nPBE0 CPMD beta', 2 * -hab_pbe0_cpmd_fit[0], 2.7 + 2*hab_pbe0_cpmd_fit[0], 'pm', np.sqrt(np.diag(covariance_pbe0_cpmd)))
print('PBE0 BW beta', 2 * -hab_pbe0_bw_fit[0], 2.7 + 2*hab_pbe0_bw_fit[0], 'pm', np.sqrt(np.diag(covariance_pbe0_bw)))
print('PBE0 HW beta', 2 * -hab_pbe0_hw_fit[0], 2.7 + 2*hab_pbe0_hw_fit[0], 'pm', np.sqrt(np.diag(covariance_pbe0_hw)))
# print('PBE0 HW beta 2', 2 * -hab_pbe0_hw_fit2[0], 2.7 + 2*hab_pbe0_hw_fit2[0])

# Lambda values
lambda_pbe_cpmd = np.array([0.76, 0.79, 0.98, 0.92, 1.12, 1.11])
lambda_pbe_bw = np.array([0.94, 1.07, 1.27, 1.25, 1.44, 1.37])
lambda_pbe_hw = np.array([0.89, 0.96, 1.18, 1.13, 1.35, 1.34])
lambda_pbe_hw_opt = np.array([0.93, 1.01, 1.19, 1.14, 1.26, 1.32])

lambda_pbe0_cpmd = np.array([0.86, 0.98, 1.33, 1.26, 1.66, 1.63])
lambda_pbe0_bw = np.array([1.07, 1.34, 1.69, 1.69, 1.97, 1.87])
lambda_pbe0_bw2 = np.array([1.07, 1.34, 1.69, 1.69, 1.97, 1.97])
lambda_pbe0_hw = np.array([1.00, 1.19, 1.56, 1.52, 1.89, 1.85])
lambda_pbe0_hw_opt = np.array([1.06, 1.29, 1.63, 1.59, 1.85, 1.87])

print('hab_pbe_hw/hab_pbe_cpmd', hab_pbe_hw/hab_pbe_cpmd)
print('hab_pbe0_hw/hab_pbe0_cpmd', hab_pbe0_hw/hab_pbe0_cpmd)
print('mean hab_pbe0_hw/hab_pbe0_cpmd', np.average(hab_pbe0_hw/hab_pbe0_cpmd))
print('mean hab_pbe0_hw[:5]/hab_pbe0_cpmd[:5]', np.average(hab_pbe0_hw[:5]/hab_pbe0_cpmd[:5]))
print('mean hab_pbe0_hw[:4]/hab_pbe0_cpmd[:4]', np.average(hab_pbe0_hw[:4]/hab_pbe0_cpmd[:4]))
print('mean hab_pbe0_hw[:4]/hab_pbe0_cpmd[:4]', np.average(hab_pbe0_hw[4]/hab_pbe0_cpmd[4]))
print('mean hab_pbe0_hw[:4]/hab_pbe0_cpmd[:4]', np.average(hab_pbe0_hw[5]/hab_pbe0_cpmd[5]))

print('\nlambda_pbe_hw/lambda_pbe_cpmd', lambda_pbe_hw/lambda_pbe_cpmd)
print('lambda_pbe_hw_opt/lambda_pbe_cpmd', lambda_pbe_hw_opt/lambda_pbe_cpmd)
print('lambda_pbe0_hw/lambda_pbe0_cpmd', lambda_pbe0_hw/lambda_pbe0_cpmd)
print('lambda_pbe0_hw_opt/lambda_pbe0_cpmd', lambda_pbe0_hw_opt/lambda_pbe0_cpmd)

print('mean lambda_pbe0_hw_opt/lambda_pbe0_cpmd', np.average(lambda_pbe0_hw_opt/lambda_pbe0_cpmd))
print('mean lambda_pbe0_hw_opt[:5]/lambda_pbe0_cpmd[:5]', np.average(lambda_pbe0_hw_opt[:5]/lambda_pbe0_cpmd[:5]))
print('mean lambda_pbe0_hw_opt[:4]/lambda_pbe0_cpmd[:4]', np.average(lambda_pbe0_hw_opt[:4]/lambda_pbe0_cpmd[:4]))

print('mean lambda_pbe0_hw_opt/lambda_pbe0_hw', np.average(lambda_pbe0_hw_opt/lambda_pbe0_hw))

lambda_pbe_cpmd_fit, _ = scipy.optimize.curve_fit(calc_lambda, d_lambda, lambda_pbe_cpmd)
lambda_pbe_bw_fit, _ = scipy.optimize.curve_fit(calc_lambda, d_lambda, lambda_pbe_bw)
lambda_pbe_hw_fit, _ = scipy.optimize.curve_fit(calc_lambda, d_lambda, lambda_pbe_hw)
lambda_pbe_hw_opt_fit, _ = scipy.optimize.curve_fit(calc_lambda, d_lambda, lambda_pbe_hw_opt)

lambda_pbe0_cpmd_fit, _ = scipy.optimize.curve_fit(calc_lambda, d_lambda, lambda_pbe0_cpmd)
lambda_pbe0_bw_fit, _ = scipy.optimize.curve_fit(calc_lambda, d_lambda, lambda_pbe0_bw2)
lambda_pbe0_hw_fit, _ = scipy.optimize.curve_fit(calc_lambda, d_lambda, lambda_pbe0_hw)
lambda_pbe0_hw_opt_fit, _ = scipy.optimize.curve_fit(calc_lambda, d_lambda, lambda_pbe0_hw_opt)

# Plot Hab (BW, HW, CPMD)
fig_hab_pbe, ax_hab_pbe = plt.subplots()
ax_hab_pbe.plot(d_hab[pbe_values], np.exp(d_hab[pbe_values]*hab_pbe_hw_fit[0] + hab_pbe_hw_fit[1]), 'r-')
ax_hab_pbe.plot(d_hab[pbe_values], np.exp(d_hab[pbe_values]*hab_pbe_cpmd_fit[0] + hab_pbe_cpmd_fit[1]), 'k-')
# ax_hab_pbe.plot(d_hab, np.exp(d_hab*hab_pbe0_hw_fit2[0] + hab_pbe0_hw_fit2[1]), 'b-', label='CP2K')
ax_hab_pbe.plot(d_hab, np.exp(d_hab*hab_pbe0_hw_fit[0] + hab_pbe0_hw_fit[1]), 'r-', label='CP2K')
ax_hab_pbe.plot(d_hab, np.exp(d_hab*hab_pbe0_cpmd_fit[0] + hab_pbe0_cpmd_fit[1]), 'k-', label='CPMD')
ax_hab_pbe.plot(d_hab, hab_pbe_hw, 'ro', fillstyle='none')
ax_hab_pbe.plot(d_hab, hab_pbe_cpmd, 'ko', fillstyle='none')
ax_hab_pbe.plot(d_hab, hab_pbe0_hw, 'rs', fillstyle='none')
ax_hab_pbe.plot(d_hab, hab_pbe0_cpmd, 'ks', fillstyle='none')
ax_hab_pbe.set_ylim([1e1,  1e4])
ax_hab_pbe.set_xlim([4, 16])
ax_hab_pbe.set_yscale('log')
ax_hab_pbe.set_xlabel(r'Distance / $\mathrm{\AA}$')
ax_hab_pbe.set_ylabel(r'$\frac{1}{2}|H^{pbc}_\mathrm{ab}$| / meV')
ax_hab_pbe.legend(frameon=False, loc='upper left')
fig_hab_pbe.tight_layout()
# fig_hab_pbe.savefig('{}/hab.png'.format(folder), dpi=300, bbbox_inches='tight')

# Plot Hab (BW, HW, CPMD)
fig_hab_pbe0, ax_hab_pbe0 = plt.subplots()
ax_hab_pbe0.plot(d_hab[pbe_values], np.exp(d_hab[pbe_values]*hab_pbe_bw_fit[0] + hab_pbe_bw_fit[1]), 'g-')
ax_hab_pbe0.plot(d_hab[pbe_values], np.exp(d_hab[pbe_values]*hab_pbe_hw_fit[0] + hab_pbe_hw_fit[1]), 'b-')
ax_hab_pbe0.plot(d_hab[pbe_values], np.exp(d_hab[pbe_values]*hab_pbe_cpmd_fit[0] + hab_pbe_cpmd_fit[1]), 'r-')
ax_hab_pbe0.plot(d_hab, np.exp(d_hab*hab_pbe0_bw_fit[0] + hab_pbe0_bw_fit[1]), 'g-', label='CP2K BW')
ax_hab_pbe0.plot(d_hab, np.exp(d_hab*hab_pbe0_hw_fit[0] + hab_pbe0_hw_fit[1]), 'b-', label='CP2K HW')
ax_hab_pbe0.plot(d_hab, np.exp(d_hab*hab_pbe0_cpmd_fit[0] + hab_pbe0_cpmd_fit[1]), 'r-', label='CPMD')
ax_hab_pbe0.plot(d_hab, hab_pbe_bw, 'go', fillstyle='none')
ax_hab_pbe0.plot(d_hab, hab_pbe_hw, 'bo', fillstyle='none')
ax_hab_pbe0.plot(d_hab, hab_pbe_cpmd, 'ro', fillstyle='none')
ax_hab_pbe0.plot(d_hab, hab_pbe0_bw, 'gs', fillstyle='none')
ax_hab_pbe0.plot(d_hab, hab_pbe0_hw, 'bs', fillstyle='none')
ax_hab_pbe0.plot(d_hab, hab_pbe0_cpmd, 'rs', fillstyle='none')
ax_hab_pbe0.set_ylim([1e1,  1e4])
ax_hab_pbe0.set_xlim([4, 16])
ax_hab_pbe0.set_yscale('log')
ax_hab_pbe0.set_xlabel(r'Distance / $\mathrm{\AA}$')
ax_hab_pbe0.set_ylabel(r'|H$_\mathrm{ab}$| / meV')
ax_hab_pbe0.legend(frameon=False, loc='upper left')
fig_hab_pbe0.tight_layout()
# fig_hab_pbe0.savefig('{}/hab_all.png'.format(folder), dpi=300, bbbox_inches='tight')

# Plot lambda PBE
fig_lambda_pbe, ax_lambda_pbe = plt.subplots()
ax_lambda_pbe.plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe_bw_fit[0], lambda_pbe_bw_fit[1]), 'g--', label='CP2K BW (CPMD struct)')
ax_lambda_pbe.plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe_hw_fit[0], lambda_pbe_hw_fit[1]), 'b--', label='CP2K HW (CPMD struct)')
ax_lambda_pbe.plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe_hw_opt_fit[0], lambda_pbe_hw_opt_fit[1]), '-', c='mediumblue', label='CP2K HW')
ax_lambda_pbe.plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe_cpmd_fit[0], lambda_pbe_cpmd_fit[1]), 'r-', label='CPMD')
ax_lambda_pbe.plot(d_lambda, lambda_pbe_bw, 'go', fillstyle='none')
ax_lambda_pbe.plot(d_lambda, lambda_pbe_hw, 'bo', fillstyle='none')
ax_lambda_pbe.plot(d_lambda, lambda_pbe_hw_opt, 'o', c='mediumblue', fillstyle='none')
ax_lambda_pbe.plot(d_lambda, lambda_pbe_cpmd, 'ro', fillstyle='none')
ax_lambda_pbe.set_xlabel(r'Distance / $\mathrm{\AA}$')
ax_lambda_pbe.set_ylabel(r'$\lambda$ / eV')
ax_lambda_pbe.set_ylim([0.6, 2.1])
ax_lambda_pbe.legend(frameon=False, loc='upper left')
fig_lambda_pbe.tight_layout()
# fig_lambda_pbe.savefig('{}/lambda_all_pbe.png'.format(folder), dpi=300, bbbox_inches='tight')

# Plot lambda PBE0
fig_lambda_pbe0, ax_lambda_pbe0 = plt.subplots()
ax_lambda_pbe0.plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe0_bw_fit[0], lambda_pbe0_bw_fit[1]), 'g--', label='CP2K BW (CPMD struct)')
ax_lambda_pbe0.plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe0_hw_fit[0], lambda_pbe0_hw_fit[1]), 'b--', label='CP2K HW (CPMD struct)')
ax_lambda_pbe0.plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe0_hw_opt_fit[0], lambda_pbe0_hw_opt_fit[1]), '-', c='mediumblue', label='CP2K HW')
ax_lambda_pbe0.plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe0_cpmd_fit[0], lambda_pbe0_cpmd_fit[1]), 'r-', label='CPMD')
ax_lambda_pbe0.plot(d_lambda, lambda_pbe0_bw, 'gs', fillstyle='none')
ax_lambda_pbe0.plot(d_lambda, lambda_pbe0_hw, 'bs', fillstyle='none')
ax_lambda_pbe0.plot(d_lambda, lambda_pbe0_hw_opt, 's', c='mediumblue', fillstyle='none')
ax_lambda_pbe0.plot(d_lambda, lambda_pbe0_cpmd, 'rs', fillstyle='none')
ax_lambda_pbe0.set_xlabel(r'Distance / $\mathrm{\AA}$')
ax_lambda_pbe0.set_ylabel(r'$\lambda$ / eV')
ax_lambda_pbe0.set_ylim([0.6, 2.1])
ax_lambda_pbe0.legend(frameon=False, loc='upper left')
fig_lambda_pbe0.tight_layout()
# fig_lambda_pbe0.savefig('{}/lambda_all_pbe0.png'.format(folder), dpi=300, bbbox_inches='tight')

# Plot lambda PBE0
fig_lambda, ax_lambda = plt.subplots()
ax_lambda.plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe_hw_fit[0], lambda_pbe_hw_fit[1]), '--', c='deeppink')
ax_lambda.plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe_hw_opt_fit[0], lambda_pbe_hw_opt_fit[1]), 'r-')
ax_lambda.plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe_cpmd_fit[0], lambda_pbe_cpmd_fit[1]), 'k-')
ax_lambda.plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe0_hw_fit[0], lambda_pbe0_hw_fit[1]), '--', c='deeppink', label='CP2K (CPMD struct)')
ax_lambda.plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe0_hw_opt_fit[0], lambda_pbe0_hw_opt_fit[1]), 'r-', label='CP2K')
ax_lambda.plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe0_cpmd_fit[0], lambda_pbe0_cpmd_fit[1]), 'k-', label='CPMD')
ax_lambda.plot(d_lambda, lambda_pbe_hw, 'o', c='deeppink', fillstyle='none')
ax_lambda.plot(d_lambda, lambda_pbe_hw_opt,  'ro', fillstyle='none')
ax_lambda.plot(d_lambda, lambda_pbe_cpmd, 'ko', fillstyle='none')
ax_lambda.plot(d_lambda, lambda_pbe0_hw, 's', c='deeppink', fillstyle='none')
ax_lambda.plot(d_lambda, lambda_pbe0_hw_opt, 'rs', fillstyle='none')
ax_lambda.plot(d_lambda, lambda_pbe0_cpmd, 'ks', fillstyle='none')
ax_lambda.set_xlabel(r'Distance / $\mathrm{\AA}$')
ax_lambda.set_ylabel(r'$\lambda$ / eV')
ax_lambda.set_ylim([0.6, 2.1])
ax_lambda.legend(frameon=False, loc='upper left')
fig_lambda.tight_layout()
# fig_lambda.savefig('{}/lambda.png'.format(folder), dpi=300, bbbox_inches='tight')


params = {'font.size': 12,
          'axes.labelsize': 14,
          'legend.fontsize': 10}
plt.rcParams.update(params)

fig, axs = plt.subplots(2, 1, sharex='col', figsize=(6,8))
# plt.tick_params(axis='x', which='major', labelsize=12.5)
# plt.tick_params(axis='y', which='major', labelsize=12.5)
axs[1].set_yscale('log')
axs[1].plot(d_hab[pbe_values], np.exp(d_hab[pbe_values]*hab_pbe_hw_fit[0] + hab_pbe_hw_fit[1]), 'r-')
axs[1].plot(d_hab[pbe_values], np.exp(d_hab[pbe_values]*hab_pbe_cpmd_fit[0] + hab_pbe_cpmd_fit[1]), 'k-')
axs[1].plot(d_hab, np.exp(d_hab*hab_pbe0_hw_fit[0] + hab_pbe0_hw_fit[1]), 'r-', label='CP2K')
axs[1].plot(d_hab, np.exp(d_hab*hab_pbe0_cpmd_fit[0] + hab_pbe0_cpmd_fit[1]), 'k-', label='CPMD')
axs[1].plot(d_hab, hab_pbe_hw, 'ro', fillstyle='none')
axs[1].plot(d_hab, hab_pbe_cpmd, 'ko', fillstyle='none')
axs[1].plot(d_hab, hab_pbe0_hw, 'rs', fillstyle='none')
axs[1].plot(d_hab, hab_pbe0_cpmd, 'ks', fillstyle='none')
axs[1].set_ylim([1e1,  1e4])
# axs[1].set_ylabel(r'|H$_\mathrm{ab}$| / meV')
axs[1].set_ylabel(r'$\frac{1}{2}|H^{pbc}_\mathrm{ab}$| / meV')
axs[0].plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe_hw_fit[0], lambda_pbe_hw_fit[1]), '--', c='deeppink')
axs[0].plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe_hw_opt_fit[0], lambda_pbe_hw_opt_fit[1]), 'r-')
axs[0].plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe_cpmd_fit[0], lambda_pbe_cpmd_fit[1]), 'k-')
axs[0].plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe0_cpmd_fit[0], lambda_pbe0_cpmd_fit[1]), 'k-', label='CPMD')
axs[0].plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe0_hw_fit[0], lambda_pbe0_hw_fit[1]), 'r-', label='CP2K (CPMD structure)')
axs[0].plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe0_hw_opt_fit[0], lambda_pbe0_hw_opt_fit[1]), '--', c='deeppink', label='CP2K')


# axs[0].plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe0_hw_opt_fit[0], lambda_pbe0_hw_opt_fit[1]), 'r-', label='CP2K')
# axs[0].plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe0_hw_opt_fit[0], lambda_pbe0_hw_opt_fit[1]), '--', c='deeppink', label='CP2K')
# axs[0].plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe0_hw_fit[0], lambda_pbe0_hw_fit[1]), '--', c='deeppink', label='CP2K (CPMD structure)')
# axs[0].plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe0_hw_fit[0], lambda_pbe0_hw_fit[1]), 'r-', label='CP2K (CPMD structure)')
# axs[0].plot(d_lambda_fine, calc_lambda(d_lambda_fine, lambda_pbe0_hw_opt_fit[0], lambda_pbe0_hw_opt_fit[1]), '--', c='deeppink', label='CP2K (re-optimised structure)')
axs[0].plot(d_lambda, lambda_pbe_hw, 'o', c='deeppink', fillstyle='none')
axs[0].plot(d_lambda, lambda_pbe_hw_opt,  'ro', fillstyle='none')
axs[0].plot(d_lambda, lambda_pbe_cpmd, 'ko', fillstyle='none')
axs[0].plot(d_lambda, lambda_pbe0_hw, 's', c='deeppink', fillstyle='none')
axs[0].plot(d_lambda, lambda_pbe0_hw_opt, 'rs', fillstyle='none')
axs[0].plot(d_lambda, lambda_pbe0_cpmd, 'ks', fillstyle='none')
axs[1].set_xlabel(r'Distance / $\mathrm{\AA}$')
axs[0].set_ylabel(r'$\lambda$ / eV')
axs[0].legend(frameon=False, loc='lower right')
# axs[0].tick_params(axis='x', labelsize=50)
axs[0].set_xlim([4.8, 15.2])
axs[0].set_ylim([0.7, 2.0])
fig.tight_layout()
fig.subplots_adjust(hspace=0)
fig.savefig('{}/lambda_coupling.png'.format(folder), dpi=300, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
