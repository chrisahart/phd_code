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
    Script used to calculate finite size corrections for charging an ion in a dielectric in pbc.
"""


# def calc_ip(length, born_radius):
#     """" Lambda according to Marcus continuum theory """
#
#     return 0.5 * ((madelung / (dielectric_static * length)) +
#                   ((4 * np.pi * (dielectric_static - 1) * (born_radius ** 2)) / (3 * dielectric_static * length ** 3)))


def calc_ip(length, born_radius):
    """" Lambda according to Marcus continuum theory """

    return (np.abs(madelung) / (2 * length)) * ((1/dielectric_static) - 1) + \
                  ((2 * np.pi * (dielectric_static - 1) * (born_radius ** 2)) / (3 * dielectric_static * length ** 3))


# Parameters
madelung = -2.837  # Madelung constant for simple cubic
dielectric_optical = np.average([6.7, 7.0])  # Phys. Rev. B 1977
dielectric_static = np.average([20.6, 24.1])  # Phys. Rev. B 1977
# dielectric_optical = np.array(9.0)  # Rosso et al.
# dielectric_static = np.array(25.0)  # Rosso et al.

# Hematite hole
hematite_hole_length = np.array([np.cbrt(np.product((10.071, 10.071, 13.7471))),
                                 np.cbrt(np.product((15.1065, 15.1065, 13.7471)))]) * parameters.angstrom_to_bohr
hematite_hole_ip = np.array([-0.28158, -0.28129])

# Lepidocrocite hole
lepidicrocite_hole_length = np.array([np.cbrt(np.product((10.071, 10.071, 13.7471))),
                                 np.cbrt(np.product((15.1065, 15.1065, 13.7471)))]) * parameters.angstrom_to_bohr
lepidicrocite_hole_ip = np.array([-0.17049, -0.16910])

length_array = np.linspace(1, 100, num=1000)

print('(np.abs(madelung) / (2 * length)) * ((1/dielectric_static) - 1)',
      (np.abs(madelung) / (2 * hematite_hole_length[0])) * ((1/dielectric_static - 1)))

print( ((2 * np.pi * (dielectric_static - 1) * (4 ** 2)) / (3 * dielectric_static * hematite_hole_length[0] ** 3)))

print('calc_ip(length, born_radius)', hematite_hole_ip[0] + calc_ip(hematite_hole_length[0], 4))

fig_fit, ax_fit = plt.subplots(figsize=(6, 4))
ax_fit.plot(length_array, hematite_hole_ip[1] + calc_ip(length_array, 4))
ax_fit.plot(hematite_hole_length, hematite_hole_ip, 'x')
# ax_fit.set_xlim([0, 40])
# ax_fit.set_ylim([0,  calc_lambda_fit_polaron_inner(1e10, *optimised) * parameters.hartree_to_ev])
# ax_fit.set_xlabel(r'Unit cell length / $\mathrm{\AA}$')
# ax_fit.set_ylabel('Energy / eV')
fig_fit.tight_layout()

if __name__ == "__main__":
    plt.show()
