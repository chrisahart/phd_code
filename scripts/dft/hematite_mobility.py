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

""" Calculate hematite mobility. All units are au unless otherwise specified """


def calc_adiabaticity(vn, kb_t, planck, lambda_tot, v_ab):
    """ Calculate adiabaticity parameter (2 pi gamma). """
    return (np.pi**(3/2)*v_ab**2) / (planck*vn*(lambda_tot*kb_t)**(1/2))


def calc_probability(vn, kb_t, planck, lambda_tot, v_ab):
    """ Calculate Landau-Zener transition probability (P_LZ). """
    return 1-np.exp(-calc_adiabaticity(vn, kb_t, planck, lambda_tot, v_ab))


def calc_transmission(vn, kb_t, planck, lambda_tot, v_ab):
    """ Calculate electronic transmission coefficient (k_el). """
    p_lz = calc_probability(vn, kb_t, planck, lambda_tot, v_ab)
    return (2 * p_lz) / (1 + p_lz)


def calc_energy_rosso(lambda_tot, v_ab):
    """ Calculate activation energy using Rosso method [Iordanova 2005]. """
    return -lambda_tot/4 + (lambda_tot**2+4*v_ab**2)**(1/2)/2 - v_ab


def calc_energy_spencer(lambda_tot, v_ab):
    """ Calculate activation energy using Jochen preferred method [Spencer 2016]. """
    return lambda_tot / 4 - (v_ab - 1/lambda_tot * v_ab**2)


def calc_energy_na_spencer(lambda_tot):
    """ Calculate non-adiabatic activation energy using Jochen preferred method [Spencer 2016]. """
    return lambda_tot / 4


def calc_rate(vn, kb_t, k_el, energy):
    """ Calculate electron transfer rate constant [Spencer 2016]. """
    return vn * k_el * np.exp(-energy/kb_t)


def calc_rate_ad(vn, kb_t, v_ab, lambda_tot, energy):
    """ Calculate non-adiabatic electron transfer rate constant [Spencer 2016]. """
    factor = 2*np.pi * v_ab**2 * (4*np.pi*lambda_tot*kb_t)**(-1/2)
    return vn * factor * np.exp(-energy/kb_t)


def calc_diffusion(multiplicity, r, k):
    """ Calculate diffusion coefficient. """
    return ((r*angstrom_to_cm)**2 * multiplicity * k) / 2


def calc_mobility(diffusion, kb_t):
    """ Calculate mobility. """
    return diffusion / kb_t


# Parameters
temp = 400  # K
multiplicity = 2  # Site multiplicity
vn = 1.85e13  # 1.85e13, 1.72e13, 1.8e14 s-1
# vn = 616.666666667 * (3e8*100)  # cm -1
# print('vn:', vn, 's-1 or ', vn/(3e8*100), 'cm-1')

# Constants
kb_t_au = 8.617333262145E-5 * temp  # KbT in eV
kb_t = 1.38e-23 * temp  # KbT in SI units
planck = 6.63e-34  # Planck constant in SI units
planck_au = 2 * np.pi  # Planck constant in SI units
angstrom_to_cm = 1e-8
ev_to_joules = 1.60218e-19

# Results
# r_fe = np.array([2.97])
# vab_441 = np.array([0.2])  # Rosso results
# lambda_441 = np.array([1.59])  # Rosso results

# r_fe = np.array([2.97])
# vab_441 = np.array([40]) / 1e3  # Dupuis results
# lambda_441 = np.array([800]) / 1e3  # Dupuis results
# vab_441 = np.array([200]) / 1e3  # Rosso results
# lambda_441 = np.array([1200]) / 1e3  # Rosso results
# vab_441 = np.array([41]) / 1e3  # Adelstein2014 results
# lambda_441 = np.array([674]) / 1e3  # Adelstein2014 results

# r_fe = np.array([2.97] * 4)
# vab_441 = np.array([39, 53, 203, 110]) / 1e3
# lambda_441 = np.array([887, 870, 657, 819]) / 1e3

# Hole Boltzmann factor at 300 K
# r_fe = np.array([2.97])
# vab_441 = np.array([147]) / 1e3
# lambda_441 = np.array([752]) / 1e3

# Hole Boltzmann factor at 330 K
# r_fe = np.array([2.97])
# vab_441 = np.array([143]) / 1e3
# lambda_441 = np.array([759]) / 1e3

# Hole Boltzmann factor at 400 K
# r_fe = np.array([2.97])
# vab_441 = np.array([135]) / 1e3
# lambda_441 = np.array([771]) / 1e3

# Electron nn-1
# r_fe = np.array([2.97])
# vab_441 = np.array([26]) / 1e3
# lambda_441 = np.array([363]) / 1e3

# Electron nn-2
r_fe = np.array([5.04])
vab_441 = np.array([57]) / 1e3
lambda_441 = np.array([522]) / 1e3

# r_fe = [2.97] * 5 + [5.04] * 6 + [5.85] * 3
# vab_441 = np.array([39, 53, 101, 110, 203, 15, 8, 15, 30, 28, 16, 3, 9, 45]) / 1e3
# lambda_441 = np.array([881, 865, 784, 814, 652, 1050, 1050, 1028, 1016, 1022, 1034, 1087, 1106, 1026]) / 1e3


for i in range(0, np.shape(vab_441)[0]):
    adiabaticity_parameter = calc_adiabaticity(vn, kb_t, planck, lambda_441[i]*ev_to_joules, vab_441[i]*ev_to_joules)
    lz_probability = calc_probability(vn, kb_t, planck, lambda_441[i]*ev_to_joules, vab_441[i]*ev_to_joules)
    transmission_coefficient = calc_transmission(vn, kb_t, planck, lambda_441[i]*ev_to_joules, vab_441[i]*ev_to_joules)

    energy_rosso = calc_energy_rosso(lambda_441[i], vab_441[i])
    rate_rosso = calc_rate(vn, kb_t_au, 1, energy_rosso)
    diffusion_rosso = calc_diffusion(multiplicity, r_fe, rate_rosso)
    mobility_rosso = calc_mobility(diffusion_rosso, kb_t_au)

    print("\nRosso Activation energy (delta G*): {0:.2} eV".format(energy_rosso))
    print("Rosso Activation energy (delta G*): {} meV".format(energy_rosso*1e3))
    print("Rosso Electron transfer rate constant (k_et): {0:.2E} s-1".format(rate_rosso))
    print("Rosso Electron transfer rate constant (i k_et): {0:.2E} s-1".format(multiplicity * rate_rosso))
    print("Rosso Mobility: {0:.2E} cm2/V".format(float(mobility_rosso)))

    energy_spencer = calc_energy_spencer(lambda_441[i], vab_441[i])
    rate_spencer_ad = calc_rate(vn, kb_t_au, 1, energy_spencer)
    diffusion_spencer_ad = calc_diffusion(multiplicity, r_fe[i], rate_spencer_ad)
    mobility_spencer_ad = calc_mobility(diffusion_spencer_ad, kb_t_au)

    energy_na_spencer = calc_energy_na_spencer(lambda_441[i])
    rate_spencer_na = calc_rate_ad(vn, kb_t_au, vab_441[i], lambda_441[i], energy_na_spencer)
    diffusion_spencer_na = calc_diffusion(multiplicity, r_fe[i], rate_spencer_na)
    mobility_spencer_na = calc_mobility(diffusion_spencer_na, kb_t_au)

    rate_spencer = calc_rate(vn, kb_t_au, transmission_coefficient, energy_spencer)
    diffusion_spencer = calc_diffusion(multiplicity, r_fe[i], rate_spencer)
    mobility_spencer = calc_mobility(diffusion_spencer, kb_t_au)

    print("-------------------------------------------") 
    print("Distance: {} A".format(r_fe[i]))
    print("Coupling (v_ab): {} meV".format(int(np.round(vab_441[i]*1e3))))
    print("Reorganisation energy energy (lambda): {} meV".format(int(np.round(lambda_441[i]*1e3))))

    print("\nAdiabaticity parameter (2 pi gamma): {}".format(adiabaticity_parameter))
    # print("Landau-Zener transition probability (P_LZ): {0:.2}".format(lz_probability))
    print("Electronic transmission coefficient (k_el): {0:.1}".format(transmission_coefficient))

    print("\nActivation energy (delta G*): {} meV".format(int(np.round(energy_spencer*1e3))))
    print("Electron transfer rate constant (k_et): {0:.1E} s-1".format(rate_spencer))
    print("1/Electron transfer rate constant (1/k_et): {} fs".format((1/rate_spencer)*1e15))
    # print("Mobility: {0:.2} cm2/V".format(mobility_spencer))
    print("Mobility: {0:.1E} cm2/V".format(mobility_spencer))

    # print("\nAdiabatic electron transfer rate constant (k_et): {0:.1E} s-1".format(rate_spencer_ad))
    # print("Adiabatic mobility: {0:.2} cm2/V".format(mobility_spencer_ad))

    # print("\nNon-adiabatic activation energy (sigma A dagger na): {} meV".format(int(np.round(energy_na_spencer*1e3))))
    # print("Non-adiabatic electron transfer rate constant (k_et): {0:.1E} s-1".format(rate_spencer_na))
    # print("Non-adiabatic mobility: {0:.2} cm2/V".format(mobility_spencer_na))

