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


def fit_straight(x, m, c):
    """" Fit overlap to straight line y = mx + c"""
    return m * x + c


def fit_exp(x, a, b):
    """" Fit overlap to straight line y = mx + c"""
    return a/x * np.exp(-b / x)


def fit_log(x, a, b):
    """" Fit overlap to straight line y = mx + c"""
    return np.log10(a/x) + np.log10(np.exp(-b / x))


def fit_inverse_t(x, a, b, c):
    """" Fit overlap to straight line y = mx + c"""
    return a - (b / (c * x))


folder_save = 'E:/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/mobility/plots'

# Data
neel = 955  # K Lu2010a
# morin = 263  # K Lu2010a
morin = 100  # K Lu2010a
temperature = np.array([200, 300, 600, 800, 1000, 1200, 1500])  # Kelvin
temperature_array = np.linspace(temperature[0], temperature[-1], num=int(1e3))
lit_hall_mobility = np.array([2e-2, 1e-1])
lit_hall_temperature = np.array([960, 1500])
lit_exp_mobility = np.array([9.1e-1, 2.8e-1, 4e-2, 4.6e-1, 2e-2])
lit_exp_temperature = np.array([1000, 1000, 290, 290, 780])
# lit_calc_mobility = np.array([6.2e-2, 5.6e-4, 1.7e-4, 9.0e-3])
# lit_calc_mobility = np.array([2.6e-3, 2.9e-4, 8.3e-5, 1.4e-3, 6.2e-4])  # Re-calculated
lit_calc_mobility = np.array([6.2e-2, 5.6e-4, 1.7e-4, 9.0e-3, 8.6e-4])  # Literature
lit_calc_temperature = np.array([298, 300, 300, 300])
# hole_mobility = np.array([9.28343816923639e-05, 0.001778382047, 0.02406613388, 0.04191229685, 0.05367764286, 0.06067841544, 0.06511361651])
# hole_mobility = np.array([9.07E-05, 0.00172687, 0.02310558, 0.04020849, 0.05136504, 0.05796623, 0.06210813])
# hole_mobility = np.array([8.59E-04, 5.43E-03, 3.00E-02, 4.37E-02, 5.27E-02, 5.79E-02, 6.09E-02])
hole_mobility = np.array([1.20E-02, 3.08E-02, 5.94E-02, 6.24E-02, 6.11E-02, 5.82E-02, 5.32E-02])
# electron_mobility = np.copy(hole_mobility)
# electron_mobility = np.array([4.76E-02, 1.36E-01, 2.83E-01, 2.96E-01, 2.85E-01, 2.66E-01, 2.39E-01]) * (4/6)
electron_mobility2 = np.array([3.86e-2, 1.17E-01, 2.54e-1, 2.69e-1, 2.62e-1, 2.47e-1, 2.21e-1]) * (4/6)  # NN-2
electron_mobility1 = np.array([9.07E-03, 1.97E-02, 2.91E-02, 2.74E-02, 2.46E-02, 2.19E-02, 1.83E-02])  # NN-1
electron_mobility = electron_mobility1 + electron_mobility2
hole_mobility_log = np.log10(hole_mobility)
electron_mobility_log = np.log10(hole_mobility)

# Fitting
mobility_hole_fit, _ = scipy.optimize.curve_fit(fit_exp, temperature, hole_mobility)
mobility_electron_fit, _ = scipy.optimize.curve_fit(fit_exp, temperature, electron_mobility)
mobility_hole_log_fit2, _ = scipy.optimize.curve_fit(fit_log, temperature, np.log10(hole_mobility))
mobility_electron_log_fit2, _ = scipy.optimize.curve_fit(fit_log, temperature, np.log10(electron_mobility))

# Plot mobility
fig_plot_mobility, ax_plot_mobility = plt.subplots()
ax_plot_mobility.plot(np.ones(int(1e3))*neel, np.linspace(-100, 100, num=int(1e3)), 'k--', fillstyle='full', alpha=0.3)
ax_plot_mobility.plot(np.ones(int(1e3))*morin, np.linspace(-100, 100, num=int(1e3)), 'k-.', fillstyle='full', alpha=0.3)
ax_plot_mobility.plot(temperature_array, fit_exp(temperature_array, *mobility_electron_fit), 'b')
ax_plot_mobility.plot(temperature_array, fit_exp(temperature_array, *mobility_hole_fit), 'r')
ax_plot_mobility.plot(temperature, electron_mobility, 'b*', fillstyle='full', label=r'e$^-$ This work')
ax_plot_mobility.plot(temperature, hole_mobility, 'r*', fillstyle='full', label=r'h$^+$ This work')
ax_plot_mobility.plot(lit_calc_temperature[2], (lit_calc_mobility[2]), 'ro', label=r'h$^+$ Iordanova')
ax_plot_mobility.plot(lit_calc_temperature[1], (lit_calc_mobility[1]), 'bo', label=r'e$^-$ Iordanova ')
ax_plot_mobility.plot(lit_calc_temperature[3], (lit_calc_mobility[3]), 'bs', label=r'e$^-$ Adelstein ')
# ax_plot_mobility.plot(lit_hall_temperature[0], (lit_hall_mobility[0]), 'b^', label=r'e$^-$ (Nb, Zr) VanDaal', fillstyle='none')
# ax_plot_mobility.plot(lit_hall_temperature[1], (lit_hall_mobility[1]), 'bv', label=r'e$^-$ (Nb, Zr) VanDaal', fillstyle='none')
ax_plot_mobility.plot(lit_exp_temperature[0], (lit_exp_mobility[0]), 'rD', label=r'h$^+$ (Mg) Warnes', fillstyle='none')
ax_plot_mobility.plot(lit_exp_temperature[1], (lit_exp_mobility[1]), 'bD', label=r'e$^-$ (Ti) Warnes', fillstyle='none')
ax_plot_mobility.plot(lit_exp_temperature[3], (lit_exp_mobility[3]), 'b^', label=r'e$^-$ (2% Ti) Gharibi', fillstyle='none')
ax_plot_mobility.plot(lit_exp_temperature[2], (lit_exp_mobility[2]), 'bX', label=r'e$^-$ (3% Ti) Zhao', fillstyle='none')
ax_plot_mobility.plot(lit_exp_temperature[3], (lit_exp_mobility[3]), 'bP', label=r'e$^-$ (5% Ti) Zhao', fillstyle='none')
ax_plot_mobility.set_xlabel('Temperature (K)')
ax_plot_mobility.set_ylabel(r'Mobility (cm$^2$/Vs)')
ax_plot_mobility.set_xlim([170, 1600])
# ax_plot_mobility.set_ylim([-0.002, 0.07])
ax_plot_mobility.set_ylim([-0.01, 0.31])
# ax_plot_mobility.set_ylim([-0.002, 0.105])
# ax_plot_mobility.legend(loc="lower right", frameon=True)
fig_plot_mobility.tight_layout()
# fig_plot_mobility.savefig('{}/mobility.png'.format(folder_save), dpi=parameters.save_dpi)
fig_plot_mobility.savefig('{}/mobility_lit.pdf'.format(folder_save), dpi=parameters.save_dpi)

# Plot log10 mobility vs T
fig_plot_mobility_log, ax_plot_mobility_log = plt.subplots()
ax_plot_mobility_log.plot(np.ones(int(1e3))*neel, np.linspace(-100, 100, num=int(1e3)), 'k--', fillstyle='full', alpha=0.3)
ax_plot_mobility_log.plot(np.ones(int(1e3))*morin, np.linspace(-100, 100, num=int(1e3)), 'k-.', fillstyle='full', alpha=0.3)
ax_plot_mobility_log.plot(temperature_array, fit_log(temperature_array, *mobility_hole_log_fit2), 'r')
ax_plot_mobility_log.plot(temperature_array, fit_log(temperature_array, *mobility_electron_log_fit2), 'b')
ax_plot_mobility_log.plot(temperature, np.log10(hole_mobility), 'r*', fillstyle='full', label=r'h$^+$ This work')
ax_plot_mobility_log.plot(temperature, np.log10(electron_mobility), 'b*', fillstyle='full', label=r'e$^-$ This work')
ax_plot_mobility_log.plot(lit_calc_temperature[0], np.log10(lit_calc_mobility[0]), 'bv', label=r'e$^-$ Rosso')
ax_plot_mobility_log.plot(lit_calc_temperature[2], np.log10(lit_calc_mobility[2]), 'ro', label=r'h$^+$ Iordanova')
ax_plot_mobility_log.plot(lit_calc_temperature[1], np.log10(lit_calc_mobility[1]), 'bo', label=r'e$^-$ Iordanova ')
ax_plot_mobility_log.plot(lit_calc_temperature[3], np.log10(lit_calc_mobility[3]), 'bs', label=r'e$^-$ Adelstein ')
# ax_plot_mobility_log.plot(lit_hall_temperature[0], np.log10(lit_hall_mobility[0]), 'b^', label=r'e$^-$ (Nb, Zr) VanDaal', fillstyle='none')
# ax_plot_mobility_log.plot(lit_hall_temperature[1], np.log10(lit_hall_mobility[1]), 'bv', label=r'e$^-$ (Nb, Zr) VanDaal', fillstyle='none')
ax_plot_mobility_log.plot(lit_exp_temperature[0], np.log10(lit_exp_mobility[0]), 'rD', label=r'h$^+$ (Mg) Warnes', fillstyle='none')
ax_plot_mobility_log.plot(lit_exp_temperature[1], np.log10(lit_exp_mobility[1]), 'bD', label=r'e$^-$ (Ti) Warnes', fillstyle='none')
ax_plot_mobility_log.plot(lit_exp_temperature[4], (lit_exp_mobility[4]), 'b^', label=r'e$^-$ (2% Ti) Gharibi', fillstyle='none')
ax_plot_mobility_log.plot(lit_exp_temperature[2], np.log10(lit_exp_mobility[2]), 'bX', label=r'e$^-$ (3% Ti) Zhao', fillstyle='none')
# ax_plot_mobility_log.plot(lit_exp_temperature[3], np.log10(lit_exp_mobility[3]), 'bP', label=r'e$^-$ (5% Ti) Zhao', fillstyle='none')
ax_plot_mobility_log.set_xlabel('Temperature (K)')
ax_plot_mobility_log.set_ylabel('Log Mobility')
ax_plot_mobility_log.set_ylabel(r'Log [mobility (cm$^2$/Vs)]')
ax_plot_mobility_log.set_xlim([170, 1600])
ax_plot_mobility_log.set_ylim([-4.2, 0.2])
ax_plot_mobility_log.legend(loc="lower right", frameon=True)
fig_plot_mobility_log.tight_layout()
# fig_plot_mobility_log.savefig('{}/mobility_log.png'.format(folder_save), dpi=parameters.save_dpi)
fig_plot_mobility_log.savefig('{}/mobility_log_lit.pdf'.format(folder_save), dpi=parameters.save_dpi)

# Plot log10 mobility vs 1 / T
# fig_plot_mobility_log, ax_plot_mobility_log = plt.subplots()
# ax_plot_mobility_log.plot(1e3/(np.ones(int(1e3))*neel), np.linspace(-100, 100, num=int(1e3)), 'k-.', fillstyle='full', alpha=0.5)
# ax_plot_mobility_log.plot(1e3/(np.ones(int(1e3))*morin), np.linspace(-100, 100, num=int(1e3)), 'k--', fillstyle='full', alpha=0.5)
# ax_plot_mobility_log.plot(1e3/temperature_array, fit_exp2(1e3/temperature_array, *mobility_log_fit), 'k')
# ax_plot_mobility_log.plot(1e3/temperature, np.log10(hole_mobility), 'ko', fillstyle='full')
# ax_plot_mobility_log.set_xlabel(r'$10^3$ / T (1 / K)')
# ax_plot_mobility_log.set_ylabel('Log Mobility')
# ax_plot_mobility_log.set_xlim([0.5, 3.5])
# ax_plot_mobility_log.set_ylim([-6.5, -2.5])
# fig_plot_mobility_log.tight_layout()
# fig_plot_mobility_log.savefig('{}/mobility_log.png'.format(folder_save), dpi=parameters.save_dpi)

# Plot conductivity
# fig_plot_mobility_log, ax_plot_mobility_log = plt.subplots()
# ax_plot_mobility_log.plot(1e3/temperature_array, 1.6e-19*1e20*fit_exp2(1e3/temperature_array, *mobility_log_fit), 'k')
# ax_plot_mobility_log.plot(1e3/temperature, 1.6e-19*1e20*np.log(hole_mobility), 'ko', fillstyle='full')
# ax_plot_mobility_log.set_xlabel(r'$10^3$ / T (1 / K)')
# ax_plot_mobility_log.set_ylabel('Log Conductivity')
# fig_plot_mobility_log.tight_layout()
# fig_plot_mobility_log.savefig('{}/conductivity_log.png'.format(folder_save), dpi=parameters.save_dpi)

if __name__ == "__main__":
    print('Finished.')
    plt.show()
