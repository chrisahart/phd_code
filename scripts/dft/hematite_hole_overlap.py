from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.general import parameters
from scripts.formatting import cp2k_hirsh
import matplotlib.pyplot as plt
import scipy

""" Plot hematite overlap. """


def fit_overlap(x, m, c):
    """" Fit overlap to straight line y = mx + c"""
    return m * x + c


def fit_gmh(x, m, c):
    """" Fit GMH to straight line y = mx + c"""
    return m * x + c


def calc_energy_spencer(lambda_tot, v_ab):
    """ Calculate activation energy using Jochen preferred method (Spencer 2016). """
    return (lambda_tot / 4) - (v_ab - ((v_ab**2)/lambda_tot))


folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/plots'

overlap_221_1 = np.array([0.38, 0.47, 0.71])
# overlap_441_t_1 = np.array([0.40, 0.48, 0.65, ])
# overlap_441_t_2 = np.array([0.12, 0.09, 0.17, 0.10, 0.29, 0.15])
# overlap_441_t_3 = np.array([0.04, 0.13, 0.36])
overlap_441_t_1 = np.array([0.39, 0.48, 0.62, 0.64, 0.76])
overlap_441_t_2 = np.array([0.14, 0.07, 0.15, 0.30, 0.28, 0.17])
overlap_441_t_3 = np.array([0.03, 0.09, 0.37])
overlap_1 = np.concatenate([overlap_441_t_1])
overlap_23 = np.concatenate([overlap_441_t_2, overlap_441_t_3])
overlap_all = np.concatenate([overlap_441_t_1, overlap_441_t_2, overlap_441_t_3])

gmh_441_t_1 = np.array([93, 104, 137, 139, 183])
gmh_441_t_2 = np.array([38, 18, 25, 55, 53, 48])
gmh_441_t_3 = np.array([24, 11, 52])
gmh_441_e = np.array([82])
gmh_1 = np.concatenate([gmh_441_t_1])
gmh_23 = np.concatenate([gmh_441_t_2, gmh_441_t_3])
gmh_all = np.concatenate([gmh_441_t_1, gmh_441_t_2, gmh_441_t_3])

coupling_221_1 = np.array([35.9, 51.4, 165.9])
# coupling_441_t_1 = np.array([40.6, 53.6, 120.6])
# coupling_441_t_2 = np.array([12.3, 8.7, 16.4, 8.4, 28.3, 14.1])
# coupling_441_t_3 = np.array([4.7, 13.3, 43.3])
coupling_441_t_1 = np.array([39, 53, 101, 110, 203])
coupling_441_t_2 = np.array([15, 8, 15, 30, 28, 16])
coupling_441_t_3 = np.array([3, 9, 45])
coupling_441_e = np.array([57])
coupling_1 = np.concatenate([coupling_441_t_1])
coupling_23 = np.concatenate([coupling_441_t_2, coupling_441_t_3])
coupling_all = np.concatenate([coupling_441_t_1, coupling_441_t_2, coupling_441_t_3])

lambda_221_1 = np.array([199, 223, 331])
lambda_441_t_1 = np.array([241, 258, 318])
lambda_441_t_2 = np.array([239, 152, 150, 240, 260, 239])
lambda_441_t_3 = np.array([252, 188, 319])

energy_221 = np.array([-7085.913178])
energy_441 = np.array([-28342.53906])
energy_221_1 = np.array([-7085.90574, -7085.90575, -7085.90851])
energy_441_t_1 = np.array([-28342.53324, -28342.53347, -28342.53462])
energy_441_t_2 = np.array([-28342.53165, -28342.53079, -28342.53094, -28342.53164, -28342.53205, -28342.53180])
energy_441_t_3 = np.array([-28342.53120, -28342.53066, -28342.53259])

# Fit straight lines through 441 data, 1st nearest neighbour and 2/3rd together
overlap_array = np.linspace(0, 10, num=int(1e3))
energy_array = np.linspace(100, 300, num=int(1e3))
optimised_1, _ = scipy.optimize.curve_fit(fit_overlap, overlap_1/(1-overlap_1**2), coupling_1)
optimised_23, _ = scipy.optimize.curve_fit(fit_overlap, overlap_23/(1-overlap_23**2), coupling_23)
optimised_all, _ = scipy.optimize.curve_fit(fit_overlap, overlap_all/(1-overlap_all**2), coupling_all)

gmh_array = np.linspace(0, 200, num=int(1e3))
optimised_gmh_1, _ = scipy.optimize.curve_fit(fit_gmh, gmh_1, coupling_1)
optimised_gmh_23, _ = scipy.optimize.curve_fit(fit_gmh, gmh_23, coupling_23)
optimised_gmh_all, _ = scipy.optimize.curve_fit(fit_gmh, gmh_all, coupling_all)

# optimised_lambda_221_1, _ = scipy.optimize.curve_fit(fit_overlap, (energy_221_1-energy_221)*parameters.hartree_to_ev*1e3, lambda_221_1)
# optimised_lambda_441_1, _ = scipy.optimize.curve_fit(fit_overlap, (energy_441_t_1-energy_441)*parameters.hartree_to_ev*1e3, lambda_441_t_1)
# optimised_lambda_441_2, _ = scipy.optimize.curve_fit(fit_overlap, (energy_441_t_2-energy_441)*parameters.hartree_to_ev*1e3, lambda_441_t_2)
# optimised_lambda_441_3, _ = scipy.optimize.curve_fit(fit_overlap, (energy_441_t_3-energy_441)*parameters.hartree_to_ev*1e3, lambda_441_t_3)
#
# optimised_energy_221_1, _ = scipy.optimize.curve_fit(fit_overlap, (energy_221_1-energy_221)*parameters.hartree_to_ev*1e3, calc_energy_spencer(lambda_221_1, coupling_221_1))
# optimised_energy_441_1, _ = scipy.optimize.curve_fit(fit_overlap, (energy_441_t_1-energy_441)*parameters.hartree_to_ev*1e3, calc_energy_spencer(lambda_441_t_1, coupling_441_t_1))
# optimised_energy_441_2, _ = scipy.optimize.curve_fit(fit_overlap, (energy_441_t_2-energy_441)*parameters.hartree_to_ev*1e3, calc_energy_spencer(lambda_441_t_2, coupling_441_t_2))
# optimised_energy_441_3, _ = scipy.optimize.curve_fit(fit_overlap, (energy_441_t_3-energy_441)*parameters.hartree_to_ev*1e3, calc_energy_spencer(lambda_441_t_3, coupling_441_t_3))

# fig_lambda1, ax_lambda1 = plt.subplots()
# ax_lambda1.plot(energy_array, fit_overlap(energy_array, *optimised_energy_221_1), 'r')
# ax_lambda1.plot(energy_array, fit_overlap(energy_array, *optimised_energy_441_1), 'g')
# ax_lambda1.plot((energy_221_1-energy_221)*parameters.hartree_to_ev*1e3, calc_energy_spencer(lambda_221_1, coupling_221_1), 'ro', fillstyle='full', label='221 nn-1')
# ax_lambda1.plot((energy_441_t_1-energy_441)*parameters.hartree_to_ev*1e3, calc_energy_spencer(lambda_441_t_1, coupling_441_t_1), 'go', fillstyle='full', label='441 nn-1')
# ax_lambda1.set_xlabel(r'$E^{TS}_{\mathrm{DFT}} - E^{GS}_{\mathrm{DFT}}$ / meV')
# ax_lambda1.set_ylabel(r'$\frac{\lambda}{4} - (V-\frac{V^2}{\lambda})$ / meV')
# ax_lambda1.legend(frameon=True)
# ax_lambda1.set_xlim([100, 220])
# ax_lambda1.set_ylim([-2, 30])
# fig_lambda1.tight_layout()
# fig_lambda1.savefig('{}/activation_energy1.png'.format(folder_save), dpi=parameters.save_dpi)

# fig_lambda2, ax_lambda2 = plt.subplots()
# ax_lambda2.plot(energy_array, fit_overlap(energy_array, *optimised_energy_441_2), 'k')
# ax_lambda2.plot(energy_array, fit_overlap(energy_array, *optimised_energy_441_3), 'k')
# ax_lambda2.plot((energy_441_t_2-energy_441)*parameters.hartree_to_ev*1e3, calc_energy_spencer(lambda_441_t_2, coupling_441_t_2), 'go', fillstyle='none', label='441 nn-2')
# ax_lambda2.plot((energy_441_t_3-energy_441)*parameters.hartree_to_ev*1e3, calc_energy_spencer(lambda_441_t_3, coupling_441_t_3), 'g+', label='441 nn-3')
# ax_lambda2.set_xlabel(r'$E^{TS}_{\mathrm{DFT}} - E^{GS}_{\mathrm{DFT}}$ / meV')
# ax_lambda2.set_ylabel(r'$\frac{\lambda}{4} - (V-\frac{V^2}{\lambda})$ / meV')
# ax_lambda2.legend(frameon=True)
# ax_lambda2.set_xlim([170, 240])
# ax_lambda2.set_ylim([20, 60])
# fig_lambda2.tight_layout()
# fig_lambda2.savefig('{}/activation_energy2.png'.format(folder_save), dpi=parameters.save_dpi)

# fig_lambda3, ax_lambda3 = plt.subplots()
# ax_lambda3.plot(lambda_441_t_2/4, calc_energy_spencer(lambda_441_t_2, coupling_441_t_2), 'go', fillstyle='none', label='441 nn-2')
# ax_lambda3.plot(lambda_441_t_3/4, calc_energy_spencer(lambda_441_t_3, coupling_441_t_3), 'g+', label='441 nn-3')
# ax_lambda3.set_xlabel(r'$E^{TS}_{\mathrm{DFT}} - E^{GS}_{\mathrm{DFT}}$ / meV')
# ax_lambda3.set_ylabel(r'$\frac{\lambda}{4} - (V-\frac{V^2}{\lambda})$ / meV')
# ax_lambda3.legend(frameon=True)
# ax_lambda3.set_xlim([170, 240])
# ax_lambda3.set_ylim([20, 60])
# fig_lambda3.tight_layout()
# fig_lambda3.savefig('{}/activation_energy2.png'.format(folder_save), dpi=parameters.save_dpi)

# fig_energy1, ax_energy1 = plt.subplots()
# ax_energy1.plot(energy_array, fit_overlap(energy_array, *optimised_lambda_221_1), 'r')
# ax_energy1.plot(energy_array, fit_overlap(energy_array, *optimised_lambda_441_1), 'g')
# ax_energy1.plot(energy_array, fit_overlap(energy_array, *optimised_lambda_441_2), 'k')
# ax_energy1.plot(energy_array, fit_overlap(energy_array, *optimised_lambda_441_3), 'k')
# ax_energy1.plot((energy_221_1-energy_221)*parameters.hartree_to_ev*1e3, lambda_221_1, 'ro', fillstyle='full', label='221 nn-1')
# ax_energy1.plot((energy_441_t_1-energy_441)*parameters.hartree_to_ev*1e3, lambda_441_t_1, 'go', fillstyle='full', label='441 nn-1')
# ax_energy1.plot((energy_441_t_2-energy_441)*parameters.hartree_to_ev*1e3, lambda_441_t_2, 'go', fillstyle='none', label='441 nn-2')
# ax_energy1.plot((energy_441_t_3-energy_441)*parameters.hartree_to_ev*1e3, lambda_441_t_3, 'g+', label='441 nn-3')
# ax_energy1.set_xlabel(r'$E^{TS}_{\mathrm{DFT}} - E^{GS}_{\mathrm{DFT}}$ / meV')
# ax_energy1.set_ylabel(r'$\lambda$ / meV')
# ax_energy1.legend(frameon=True)
# ax_energy1.set_xlim([100, 220])
# ax_energy1.set_ylim([175, 350])
# fig_energy1.tight_layout()
# fig_energy1.savefig('{}/lambda_energy1.png'.format(folder_save), dpi=parameters.save_dpi)

# fig_energy2, ax_energy2 = plt.subplots()
# ax_energy2.plot(energy_array, fit_overlap(energy_array, *optimised_lambda_221_1), 'r')
# ax_energy2.plot(energy_array, fit_overlap(energy_array, *optimised_lambda_441_1), 'k')
# ax_energy2.plot(energy_array, fit_overlap(energy_array, *optimised_lambda_441_2), 'k')
# ax_energy2.plot(energy_array, fit_overlap(energy_array, *optimised_lambda_441_3), 'k')
# ax_energy2.plot((energy_221_1-energy_221)*parameters.hartree_to_ev*1e3, lambda_221_1, 'ro', fillstyle='full', label='221 nn-1')
# ax_energy2.plot((energy_441_t_1-energy_441)*parameters.hartree_to_ev*1e3, lambda_441_t_1, 'go', fillstyle='full', label='441 nn-1')
# ax_energy2.plot((energy_441_t_2-energy_441)*parameters.hartree_to_ev*1e3, lambda_441_t_2, 'go', fillstyle='none', label='441 nn-2')
# ax_energy2.plot((energy_441_t_3-energy_441)*parameters.hartree_to_ev*1e3, lambda_441_t_3, 'g+', label='441 nn-3')
# ax_energy2.set_xlabel(r'$E^{TS}_{\mathrm{DFT}} - E^{GS}_{\mathrm{DFT}}$ / meV')
# ax_energy2.set_ylabel(r'$\lambda$ / meV')
# ax_energy2.legend(frameon=True)
# ax_energy2.set_xlim([170, 240])
# ax_energy2.set_ylim([125, 325])
# fig_energy2.tight_layout()
# fig_energy2.savefig('{}/lambda_energy2.png'.format(folder_save), dpi=parameters.save_dpi)

# fig_overlap, ax_overlap = plt.subplots()
# ax_overlap.plot(overlap_221_1, coupling_221_1, 'ro', fillstyle='full', label='221 nn-1')
# ax_overlap.plot(overlap_441_t_1, coupling_441_t_1, 'go', fillstyle='full', label='441 nn-1')
# ax_overlap.plot(overlap_441_t_2, coupling_441_t_2, 'go', fillstyle='none', label='441 nn-2')
# ax_overlap.plot(overlap_441_t_3, coupling_441_t_3, 'g+', label='441 nn-3')
# ax_overlap.set_xlabel(r'$|\mathrm{S_{AB}}|$')
# ax_overlap.set_ylabel(r'$|\mathrm{H_{ab}^{pbc}}|$ / meV')
# ax_overlap.legend(frameon=True)
# ax_overlap.set_xlim([0, 0.8])
# ax_overlap.set_ylim([0, 180])
# fig_overlap.tight_layout()
# fig_overlap.savefig('{}/coupling_overlap.png'.format(folder_save), dpi=parameters.save_dpi)

fig_overlap_log_1, ax_overlap_log_1 = plt.subplots()
ax_overlap_log_1.plot(overlap_array, fit_overlap(overlap_array, *optimised_1), 'k')
# ax_overlap_log_1.plot(overlap_221_1/(1-overlap_221_1**2), coupling_221_1, 'ro', fillstyle='full', label='221 1st')
ax_overlap_log_1.plot(overlap_441_t_1/(1-overlap_441_t_1**2), coupling_441_t_1, 'go', fillstyle='full', label='441 1st')
ax_overlap_log_1.set_xlabel(r'$|\mathrm{S_{AB}}| / (1-|\mathrm{S_{AB}}|^2)$')
ax_overlap_log_1.set_ylabel(r'$|\mathrm{H_{ab}^{pbc}}|$ / meV')
ax_overlap_log_1.legend(frameon=True)
ax_overlap_log_1.set_xlim([0.2, 1.9])
ax_overlap_log_1.set_ylim([0, 220])
fig_overlap_log_1.tight_layout()
fig_overlap_log_1.savefig('{}/coupling_overlap_log_1.png'.format(folder_save), dpi=parameters.save_dpi)

fig_overlap_log_2, ax_overlap_log_2 = plt.subplots()
ax_overlap_log_2.plot(overlap_array, fit_overlap(overlap_array, *optimised_23), 'k')
ax_overlap_log_2.plot(overlap_441_t_2/(1-overlap_441_t_2**2), coupling_441_t_2, 'go', fillstyle='none', label='441 2nd')
ax_overlap_log_2.plot(overlap_441_t_3/(1-overlap_441_t_3**2), coupling_441_t_3, 'g+', label='441 3rd')
ax_overlap_log_2.set_xlabel(r'$|\mathrm{S_{AB}}| / (1-|\mathrm{S_{AB}}|^2)$')
ax_overlap_log_2.set_ylabel(r'$|\mathrm{H_{ab}^{pbc}}|$ / meV')
ax_overlap_log_2.legend(frameon=True)
ax_overlap_log_2.set_xlim([0, 0.5])
ax_overlap_log_2.set_ylim([0, 55])
fig_overlap_log_2.tight_layout()
fig_overlap_log_2.savefig('{}/coupling_overlap_log_2.png'.format(folder_save), dpi=parameters.save_dpi)

# fig_overlap_log_3, ax_overlap_log_3 = plt.subplots()
# ax_overlap_log_3.plot(overlap_array, fit_overlap(overlap_array, *optimised_all), 'k')
# ax_overlap_log_3.plot(overlap_221_1/(1-overlap_221_1**2), coupling_221_1, 'ro', fillstyle='full', label='221 1st')
# ax_overlap_log_3.plot(overlap_441_t_1/(1-overlap_441_t_1**2), coupling_441_t_1, 'go', fillstyle='full', label='441 1st')
# ax_overlap_log_3.plot(overlap_441_t_2/(1-overlap_441_t_2**2), coupling_441_t_2, 'go', fillstyle='none', label='441 2nd')
# ax_overlap_log_3.plot(overlap_441_t_3/(1-overlap_441_t_3**2), coupling_441_t_3, 'g+', label='441 3rd')
# ax_overlap_log_3.set_xlabel(r'$|\mathrm{S_{AB}}| / (1-|\mathrm{S_{AB}}|^2)$')
# ax_overlap_log_3.set_ylabel(r'$|\mathrm{H_{ab}^{pbc}}|$ / meV')
# ax_overlap_log_3.legend(frameon=True)
# ax_overlap_log_3.set_xlim([0, 1.6])
# ax_overlap_log_3.set_ylim([0, 180])
# fig_overlap_log_3.tight_layout()
# fig_overlap_log_3.savefig('{}/coupling_overlap_log_3.png'.format(folder_save), dpi=parameters.save_dpi)

# 24/05/2021
fig_overlap_log_3, ax_overlap_log_3 = plt.subplots()
ax_overlap_log_3.plot(overlap_array, fit_overlap(overlap_array, *optimised_all), 'k')
ax_overlap_log_3.plot(overlap_441_t_1/(1-overlap_441_t_1**2), coupling_441_t_1, 'go', fillstyle='full', label='441 1st')
ax_overlap_log_3.plot(overlap_441_t_2/(1-overlap_441_t_2**2), coupling_441_t_2, 'go', fillstyle='none', label='441 2nd')
ax_overlap_log_3.plot(overlap_441_t_3/(1-overlap_441_t_3**2), coupling_441_t_3, 'g+', label='441 3rd')
ax_overlap_log_3.set_xlabel(r'$|\mathrm{S_{AB}}| / (1-|\mathrm{S_{AB}}|^2)$')
ax_overlap_log_3.set_ylabel(r'$|\mathrm{H_{ab}^{pbc}}|$ / meV')
ax_overlap_log_3.legend(frameon=True)
ax_overlap_log_3.set_xlim([0, 1.9])
ax_overlap_log_3.set_ylim([0, 220])
fig_overlap_log_3.tight_layout()
fig_overlap_log_3.savefig('{}/coupling_overlap_log_3.png'.format(folder_save), dpi=parameters.save_dpi)

# 24/05/2021
fig_gmh, ax_gmh = plt.subplots()
ax_gmh.plot(gmh_array, fit_overlap(gmh_array, *optimised_gmh_all), 'k')
ax_gmh.plot(gmh_441_t_1, coupling_441_t_1, 'go', fillstyle='full', label='441 1st')
ax_gmh.plot(gmh_441_t_2, coupling_441_t_2, 'go', fillstyle='none', label='441 2nd')
ax_gmh.plot(gmh_441_t_3, coupling_441_t_3, 'g+', label='441 3rd')
ax_gmh.set_xlabel(r'$|\mathrm{S_{AB}}| / (1-|\mathrm{S_{AB}}|^2)$')
ax_gmh.set_ylabel(r'$|\mathrm{H_{ab}^{pbc}}|$ / meV')
ax_gmh.legend(frameon=True)
# ax_gmh.set_xlim([0, 1.9])
# ax_gmh.set_ylim([0, 220])
fig_gmh.tight_layout()
# fig_gmh.savefig('{}/coupling_gmh.png'.format(folder_save), dpi=parameters.save_dpi)

fig_gmh2, ax_gmh2 = plt.subplots()
ax_gmh2.plot(gmh_array, fit_overlap(gmh_array, *optimised_gmh_1), 'k')
print('optimised_gmh_1', optimised_gmh_1)
ax_gmh2.plot(gmh_441_t_1, coupling_441_t_1, 'go', fillstyle='full', label='441 1st')
ax_gmh2.plot(gmh_441_e, coupling_441_e, 'ro', fillstyle='full', label='441 electron')
# ax_gmh2.plot(gmh_441_t_3[-1], coupling_441_t_3[-1], 'go', fillstyle='full', label='441 3rd')
ax_gmh2.set_xlabel(r'$|\mathrm{H_{ab}^{GMH}}|$ / meV')
ax_gmh2.set_ylabel(r'$|\mathrm{H_{ab}^{CDFT}}|$ / meV')
ax_gmh2.legend(frameon=True)
# ax_gmh2.set_xlim([90, 200])
ax_gmh2.set_xlim([80, 200])
ax_gmh2.set_ylim([30, 225])
fig_gmh2.tight_layout()
fig_gmh2.savefig('{}/coupling_gmh2_electron.png'.format(folder_save), dpi=parameters.save_dpi)

fig_gmh3, ax_gmh3 = plt.subplots()
ax_gmh3.plot(gmh_array, fit_overlap(gmh_array, *optimised_gmh_23), 'k')
ax_gmh3.plot(gmh_441_t_2, coupling_441_t_2, 'go', fillstyle='none', label='441 2nd')
ax_gmh3.plot(gmh_441_t_3, coupling_441_t_3, 'g+', label='441 3rd')
ax_gmh3.set_xlabel(r'$|\mathrm{H_{ab}^{GMH}}|$ / meV')
ax_gmh3.set_ylabel(r'$|\mathrm{H_{ab}^{CDFT}}|$ / meV')
ax_gmh3.legend(frameon=True)
ax_gmh3.set_xlim([5, 70])
ax_gmh3.set_ylim([0, 48])
fig_gmh3.tight_layout()
# fig_gmh3.savefig('{}/coupling_gmh3.png'.format(folder_save), dpi=parameters.save_dpi)


if __name__ == "__main__":
    print('Finished.')
    plt.show()