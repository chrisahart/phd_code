from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
import random
from numpy import nan as Nan
import matplotlib.pyplot as plt
import scipy
from scripts.general import parameters
from scripts.general import functions
from scripts.formatting import load_coordinates
from scripts.formatting import load_cube


"""
    Finite size correction.
    Script used to calculate finite size corrections.
"""


def calc_lambda(length, polaron_size):
    """" Lambda according to Marcus continuum theory """

    return 0.5 * ((1 / dielectric_optical) - (1 / dielectric_static)) * ((1 / polaron_size[0]) + (madelung / length))


def calc_lambda_fit_polaron(x, polaron_size):
    """" Lambda according to Marcus continuum theory with fitting polaron size only"""

    return 0.47/parameters.hartree_to_ev + \
           0.5 * ((1 / dielectric_optical) - (1 / dielectric_static)) * ((1 / polaron_size) + (madelung / x))


def calc_lambda_fit_polaron_inner(x, inner_reorganisation, polaron_size):
    """" Lambda according to Marcus continuum theory with fitting polaron size only"""

    return inner_reorganisation + 0.5 * ((1 / dielectric_optical) - (1 / dielectric_static)) * \
           ((1 / polaron_size) + (madelung / x))


def calc_lambda_fit_polaron_inner_prefactor(x, inner_reorganisation, polaron_size, prefactor):
    """" Lambda according to Marcus continuum theory with fitting polaron size only"""

    return  inner_reorganisation + 0.5 * prefactor * ((1 / dielectric_optical) - (1 / dielectric_static)) * \
           ((1 / polaron_size) + (madelung / x))


# Parameters
madelung = -2.837  # Madelung constant for simple cubic
# dielectric_optical = np.average([6.7, 7.0])  # Phys. Rev. B 1977, taking average or not causes no significant change
# dielectric_static = np.average([20.6, 24.1])  # Phys. Rev. B 1977
dielectric_optical = np.array(9.0)  # Rosso et al.
dielectric_static = np.array(25.0)  # Rosso et al.

# Hematite hole
hematite_hole_length = np.array([np.cbrt(np.product((10.071, 10.071, 13.7471))),
                                 np.cbrt(np.product((15.1065, 15.1065, 13.7471)))]) * parameters.angstrom_to_bohr
hematite_hole_energy = np.array([0.0046 + 0.017, 0.0063 + 0.02])/2
hematite_hole_energy_neut = np.array([0.017, 0.02])
hematite_hole_energy_charged = np.array([0.0046, 0.0063])
hematite_hole_polaron = np.array([2.08986, 3.0]) * parameters.angstrom_to_bohr
hematite_hole_polaron_fitted = np.array(2.2440183309191375) * parameters.angstrom_to_bohr
# hematite_hole_inner = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.49]) / parameters.hartree_to_ev
hematite_hole_inner = np.linspace(0, 0.55, num=1000)/2 / parameters.hartree_to_ev
hematite_hole_inner_fitted = np.array(0.4897267397467392) / parameters.hartree_to_ev

# Hematite electron
hematite_electron_length = np.array([np.cbrt(np.product((10.071, 10.071, 13.7471))),
                                 np.cbrt(np.product((15.1065, 15.1065, 13.7471)))]) * parameters.angstrom_to_bohr
hematite_electron_energy = np.array([0.0038 + 0.0101, 0.0051 + 0.0108]) / 2
hematite_electron_polaron = np.array([2.9, 6.0]) * parameters.angstrom_to_bohr
hematite_electron_polaron_fitted = np.array(3.204209444118321) * parameters.angstrom_to_bohr
# hematite_electron_inner = np.array([0.00, 0.10, 0.20, 0.30, 0.34]) / parameters.hartree_to_ev
hematite_electron_inner = np.linspace(0, 0.35, num=1000) / parameters.hartree_to_ev
hematite_electron_inner_fitted = np.array(0.34112113473961664) / parameters.hartree_to_ev

# Lepidicrocite hole
lepidocrocite_hole_length = np.array([np.cbrt(np.product((9.216, 12.516, 11.619))),
                                 np.cbrt(np.product((18.432, 12.516, 11.619)))]) * parameters.angstrom_to_bohr
lepidocrocite_hole_energy = np.array([0.0111 + 0.0266, 0.0161 + 0.0269])/2
lepidocrocite_hole_polaron = np.array([2.08986, 3.0]) * parameters.angstrom_to_bohr
lepidocrocite_hole_inner = (np.linspace(0, 0.93/2, num=1000) / parameters.hartree_to_ev)
lepidocrocite_hole_polaron_fitted = np.array(2.20814675820659) * parameters.angstrom_to_bohr
lepidocrocite_hole_inner_fitted = np.array(0.8925421677499845) / parameters.hartree_to_ev

# Lepidicrocite electron
lepidocrocite_electron_length = np.array([np.cbrt(np.product((18.432, 12.516, 11.619)))]) * parameters.angstrom_to_bohr
lepidocrocite_electron_energy = np.array([0.0157 + 0.0109]) / 2
lepidocrocite_electron_polaron = np.array([3.5, 3.5 + 1e-6]) * parameters.angstrom_to_bohr
lepidocrocite_electron_polaron_fitted = np.array(2.20814675820659) * parameters.angstrom_to_bohr
# lepidocrocite_electron_inner = np.array([0.00, 0.20, 0.4, 0.6, 0.8]) / parameters.hartree_to_ev
lepidocrocite_electron_inner_fitted = np.array(0.8925421677499845) / 2 / parameters.hartree_to_ev
lepidocrocite_electron_inner = np.linspace(0, 0.35/2, num=100) / parameters.hartree_to_ev

# Goethite electron
goethite_electron_length = np.array([np.cbrt(np.product((13.7937, 9.9510, 18.1068)))]) * parameters.angstrom_to_bohr
goethite_electron_energy = np.array([0.01694+0.00842])/2
goethite_electron_polaron = np.array([3.8, 3.8 + 1e-6]) * parameters.angstrom_to_bohr
goethite_electron_polaron_fitted = np.array(3.2020992191804947) * parameters.angstrom_to_bohr
# goethite_electron_inner = np.array([0.00, 0.20, 0.4, 0.6, 0.8]) / parameters.hartree_to_ev
goethite_electron_inner_fitted = np.array(0.5372185010227378) / parameters.hartree_to_ev
goethite_electron_inner = np.linspace(0, 0.33, num=100) / 2 / parameters.hartree_to_ev

# Setup
folder_save = '../../output//feIV_bulk/finite_size_corrections/'
filename = 'hematite_electron_fit_half.png'
filename_rmse = 'hematite_electron_rmse.png'
filename_radius = 'hematite_electron_radius.png'
filename_energy = 'hematite_electron_energy.png'
polaron = hematite_electron_polaron
polaron_fitted = hematite_electron_polaron_fitted
energy = hematite_electron_energy
inner = hematite_electron_inner
length = hematite_electron_length
length_array = np.linspace(1, 100, num=100)
lambda_values = calc_lambda(length_array, polaron)

# Calculate and plot best fit for inner reorganisation and polaron size
optimised, covariance = scipy.optimize.curve_fit(calc_lambda_fit_polaron_inner, length, energy,
                                                 bounds=[(0, polaron[0],),
                                                         (energy[0], polaron[1],)])
# optimised, covariance = scipy.optimize.curve_fit(calc_lambda_fit_polaron_inner, length, energy,
#                                                  bounds=[(0.48/parameters.hartree_to_ev,
#                                                           polaron[0],),
#                                                          (0.48001/parameters.hartree_to_ev,
#                                                           polaron[1],)])

# Plotting
fig_fit, ax_fit = plt.subplots(figsize=(6, 4))
ax_fit.plot(length_array / parameters.angstrom_to_bohr, calc_lambda_fit_polaron_inner(length_array, *optimised)
            * parameters.hartree_to_ev, 'k')
ax_fit.plot(length / parameters.angstrom_to_bohr, energy * parameters.hartree_to_ev, 'kx')
ax_fit.set_xlim([0, 40])
ax_fit.set_ylim([0,  calc_lambda_fit_polaron_inner(1e10, *optimised) * parameters.hartree_to_ev])
ax_fit.set_xlabel(r'Unit cell length / $\mathrm{\AA}$')
ax_fit.set_ylabel('Reorganisation energy / eV')
fig_fit.tight_layout()

# Print values
print('Cell', length)
print('Energy', energy*parameters.hartree_to_ev)
print('Inner', 'fitted', [0, energy[0] * parameters.hartree_to_ev], optimised[0] * parameters.hartree_to_ev)
print('External', (calc_lambda_fit_polaron_inner(1e10, *optimised) - optimised[0]) * parameters.hartree_to_ev)
print('Polaron_size (A)', polaron / parameters.angstrom_to_bohr, 'fitted', optimised[1] / parameters.angstrom_to_bohr)
print('Infinite unit cell unfitted (eV)', calc_lambda(1e10, polaron) * parameters.hartree_to_ev)
print('Infinite unit cell fitted (eV)', calc_lambda_fit_polaron_inner(1e10, *optimised) * parameters.hartree_to_ev)
print('Increase %', calc_lambda_fit_polaron_inner(1e10, *optimised) / np.max(energy))

# Calculate RMSE
RMSE_fit = np.sqrt(np.average((energy-calc_lambda_fit_polaron_inner(length, *optimised))**2))
print('RMSE_fit', (energy-calc_lambda_fit_polaron_inner(length, *optimised))*parameters.hartree_to_ev,
      RMSE_fit*parameters.hartree_to_ev)

# Plot series of inner reorganisation values
fig_fit1, ax_fit1 = plt.subplots(figsize=(6, 4))
fig_fit2, ax_fit2 = plt.subplots(figsize=(6, 4))
fig_fit3, ax_fit3 = plt.subplots(figsize=(6, 4))
for i in range(inner.shape[0]):

    optimised2, covariance = scipy.optimize.curve_fit(calc_lambda_fit_polaron_inner, length, energy,
                                                     bounds=[(inner[i] - 1e-10, 0,), (inner[i], polaron[1],)])
    RMSE = np.sqrt(np.average((energy - calc_lambda_fit_polaron_inner(length, *optimised2)) ** 2))
    ax_fit1.plot(optimised2[1] / parameters.angstrom_to_bohr, inner[i] * parameters.hartree_to_ev, 'kx')
    ax_fit3.plot(optimised2[1] / parameters.angstrom_to_bohr,
                 2*calc_lambda_fit_polaron_inner(1e10, *optimised2) * parameters.hartree_to_ev, 'kx')
    ax_fit2.plot(optimised2[1] / parameters.angstrom_to_bohr, (RMSE / RMSE_fit), 'kx')

    # print('\nInner', 'fitted', inner[i] * parameters.hartree_to_ev, optimised2[0] * parameters.hartree_to_ev)
    # print('Polaron_size (A)', polaron / parameters.angstrom_to_bohr, 'fitted', optimised2[1] / parameters.angstrom_to_bohr)
    # print('Infinite unit cell fitted (eV)', calc_lambda_fit_polaron_inner(1e10, *optimised2) * parameters.hartree_to_ev)
    # print('RMSE', (energy - calc_lambda_fit_polaron_inner(length, *optimised2)) * parameters.hartree_to_ev,
    #       RMSE * parameters.hartree_to_ev)
    # print('RMSE / RMSE_fit', RMSE / RMSE_fit)

# ax_fit1.set_xlim([2.1, 2.6])  # Lepidicrocite hole
# ax_fit1.set_ylim([0.37, 0.44])  # Lepidicrocite hole
# ax_fit1.set_xlim([2.1, 2.6])  # Hematite hole
# ax_fit1.set_ylim([0.14, 0.21])  # Hematite hole
ax_fit1.set_xlim([3.0, 3.7])  # Hematite electron
ax_fit1.set_ylim([0.12, 0.17])  # Hematite electron
ax_fit1.set_xlabel(r'Cavity radius / $\mathrm{\AA}$')
ax_fit1.set_ylabel('Internal reorganisation energy / eV')
fig_fit1.tight_layout()

ax_fit2.set_xlim([3.0, 3.7])   # Electron
ax_fit3.set_xlim([3.0, 3.7])   # Electron
# ax_fit2.set_xlim([2.1, 2.6])  # Hole
# ax_fit3.set_xlim([2.1, 2.6])  # Hole
ax_fit2.set_ylim([1-0.5*1e-6, 1+4*1e-6])
ax_fit2.set_xlabel(r'Cavity radius / $\mathrm{\AA}$')
ax_fit2.set_ylabel('Normalised RMSE / eV')
fig_fit2.tight_layout()

ax_fit3.set_ylim([np.average(2*calc_lambda_fit_polaron_inner(1e10, *optimised) * parameters.hartree_to_ev),
                  np.average(2*calc_lambda_fit_polaron_inner(1e10, *optimised) * parameters.hartree_to_ev)+1e-5])
ax_fit3.set_xlabel(r'Cavity radius / $\mathrm{\AA}$')
ax_fit3.set_ylabel('Total reorgansiation energy / eV')
fig_fit3.tight_layout()

# Save all
# fig_fit.savefig('{}{}'.format(folder_save, filename), dpi=parameters.save_dpi, bbbox_inches='tight')
# fig_fit1.savefig('{}{}'.format(folder_save, filename_radius), dpi=parameters.save_dpi, bbbox_inches='tight')
# fig_fit2.savefig('{}{}'.format(folder_save, filename_rmse), dpi=parameters.save_dpi, bbbox_inches='tight')
# fig_fit3.savefig('{}{}'.format(folder_save, filename_energy), dpi=parameters.save_dpi, bbbox_inches='tight')

# Hematite hole:
# Inner fitted [0.   0.1  0.2  0.3  0.4  0.49] 0.4897267397467392
# Polaron_size (A) [2.08986 3.     ] fitted 2.2416714913301155
# Infinite unit cell unfitted (eV) 0.3487935618848698
# Infinite unit cell fitted (eV) 0.8148991461410724
# Error 0.0015443769461181672

# Hematite hole (Kevin dielectrics):
# Inner fitted [0, 0.5877662400000001] 0.5311484805407302
# Polaron_size (A) [2.08986 3.     ] fitted 2.1775310559908205
# Infinite unit cell unfitted (eV) 0.24498659711848067
# Infinite unit cell fitted (eV) 0.766271505960452

# Hematite hole with prefactor:
# Inner fitted [0, 0.5877662400000001] 0.2820196601073209
# Polaron_size (A) [2.08986 3.     ] fitted 2.5142553648241295
# Prefactor [0.5, 2] 2.917095808742224
# Infinite unit cell unfitted (eV) 0.3487935618848698
# Infinite unit cell fitted (eV) 1.1277403616030082

# Hematite electron:
# Inner fitted [0, 0.37823846] 0.34112113473961664
# Polaron_size (A) [2.9 6. ] fitted 3.204209444118321
# Infinite unit cell unfitted (eV) 0.25135507342211
# Infinite unit cell fitted (eV) 0.5686124162921592

# Hematite electron with prefactor:
# Inner fitted [0, 0.37823846] 0.36426129347253833
# Polaron_size (A) [2.9 6. ] fitted 3.7149719847485785
# Prefactor [0.5, 2] 1.2464729751702033
# Infinite unit cell unfitted (eV) 0.25135507342211
# Infinite unit cell fitted (eV) 0.6088368087904552

# Lepidicrocite hole:
# Inner fitted [0, 1.02586978] 0.8925421677499845
# Polaron_size (A) [2.08986 3.     ] fitted 2.20814675820659
# Infinite unit cell unfitted (eV) 0.3487935618848698
# Infinite unit cell fitted (eV) 1.222651437202034

# Lepidicrocite electron:
# Inner fitted [0, 0.6421890400000001] 0.563377071746177
# Polaron_size (A) [3.2        3.20420944] fitted 3.20209940645328
# Infinite unit cell unfitted (eV) 0.22779053525215057
# Infinite unit cell fitted (eV) 0.7910182596921319

# Goethite electron:
# Inner fitted [0, 0.6122565] 0.5372185010227378
# Polaron_size (A) [3.2        3.20420944] fitted 3.2020992191804947
# Infinite unit cell unfitted (eV) 0.22779053525215057
# Infinite unit cell fitted (eV) 0.7648597022821463

if __name__ == "__main__":
    plt.show()


