from __future__ import division, print_function
import numpy as np
import shutil
import os
from distutils.dir_util import copy_tree
import matplotlib.pyplot as plt
import scipy
from scripts.general import parameters
from scripts.general import functions

""" Force constants. 
    Calculates force constants for goethite and lepidicrocite. """


def quadratic_fit(r, force_constant):
    """" Potential energy E = 1/2 kr^2 for r change in length from equilibrium"""

    return 0.5 * force_constant * (r ** 2)


# Lepidicrocite_FeOFe_b data (au)
# energy = np.array(
#     [-5623.840248327367590, -5623.840101330619291, -5623.839716156126087, -5623.839224297682449, -5623.838763653761816,
#      -5623.838086442232452])  # Neutral
# energy = np.array(
#     [-5624.010739729903435, -5624.010739786029262, -5624.010784847030664, -5624.010755558481833, -5624.010733843162598,
#      -5624.010686736112802, -5624.010573304478385, -5624.010370135965786, -5624.010219944986602,
#      -5624.009927014014465])  # Hole
# pos_Fe = np.array([6.14397, 8.44692, 8.63134]) * parameters.angstrom_to_bohr
# pos_O = np.array([7.68005,  9.86102,  8.63564]) * parameters.angstrom_to_bohr
# filename_parabola = 'lepidicrocite/neutral/lepidicrocite_FeOFe_b'
# filename_parabola = 'lepidicrocite/hole/lepidicrocite_FeOFe_b_hole'
# fit_start = 4

# Lepidicrocite_FeOFe_c data (au)
# energy = np.array(
#     [-5623.840248327367590, -5623.840061670346586, -5623.839653998822541, -5623.839110433659698, -5623.838324672278759,
#      -5623.837137795949275])  # Neutral
# energy = np.array(
#     [-5624.010739729903435, -5624.010742714069238, -5624.010756244998447, -5624.010861565934647, -5624.010838362950381,
#      -5624.010823992655787, -5624.010827098360096, -5624.010718277543674, -5624.010506692246054,
#      -5624.009990060033488])  # Hole
# pos_Fe = np.array([6.14397, 8.44692, 8.63134]) * parameters.angstrom_to_bohr
# pos_O = np.array([6.14403, 8.91279, 6.69937]) * parameters.angstrom_to_bohr
# filename_parabola = 'lepidicrocite/neutral/lepidicrocite_FeOFe_c'
# filename_parabola = 'lepidicrocite/hole/lepidicrocite_FeOFe_c_hole'
# fit_start = 6

# Lepidicrocite_FeOH data (au)
# energy = np.array(
#     [-5623.840248327367590, -5623.840094822602623, -5623.839626486118505, -5623.838987927630114, -5623.838092545418476,
#      -5623.836837701149307])
# energy = np.array(
#     [-5624.010739729903435, -5624.010750056737379, -5624.010710990007283, -5624.010698896451686, -5624.010670247938833,
#      -5624.010569152751486, -5624.010415799771181, -5624.009976246563383, -5624.009374140709042,
#      -5624.008403817088947])  # Hole
# pos_Fe = np.array([6.14397, 8.44692, 8.63134]) * parameters.angstrom_to_bohr
# pos_O = np.array([7.67995,  7.12934,  8.63155]) * parameters.angstrom_to_bohr
# filename_parabola = 'lepidicrocite/neutral/lepidicrocite_FeOH'
# filename_parabola = 'lepidicrocite/hole/lepidicrocite_FeOH_hole'
# fit_start = 0

# Lepidicrocite_FeOH_both data (au)
# energy = np.array(
#     [-5623.840248327367590, -5623.840153769136123, -5623.839849999374565, -5623.839493056178981, -5623.838986409436984
#      , -5623.838237264922100])
# energy = np.array(
#     [-5624.010739729903435, -5624.010728462114457, -5624.010713657309680, -5624.010659364679668, -5624.010717456176280,
#      -5624.010591980309982, -5624.010554502659033, -5624.010339197875510, -5624.010075360502924,
#      -5624.009542840924951])  # Hole
# pos_Fe = np.array([6.14397, 8.44692, 8.63134]) * parameters.angstrom_to_bohr
# pos_O = np.array([7.67995,  7.12934,  8.63155]) * parameters.angstrom_to_bohr
# filename_parabola = 'lepidicrocite/neutral/lepidicrocite_FeOH_both'
# filename_parabola = 'lepidicrocite/hole/lepidicrocite_FeOH_both_hole'
# fit_start = 0

# Goethite FeOFe_b data (au)
# energy = np.array(
#     [-5624.237304782903266, -5624.237118551849562, -5624.236730020212235, -5624.236186507361708, -5624.235339279915934,
#      -5624.234264163997068])
# energy = np.array(
#     [-5624.441419951936950, -5624.441382182615598, -5624.441369018401019, -5624.441332557684291, -5624.441272789466893,
#      -5624.441097237677241, -5624.440931711069425, -5624.440444992646917, -5624.439665225635508, -5624.438674908547910])
# pos_Fe = np.array([2.53061,  6.47301,  8.29892]) * parameters.angstrom_to_bohr
# pos_O = np.array([1.38342,  8.03502,  8.29895]) * parameters.angstrom_to_bohr
# filename_parabola = 'goethite/neutral/goethite_FeOFe_b'
# filename_parabola = 'goethite/hole/goethite_FeOFe_b_hole'
# fit_start = 0

# Goethite FeOFe_a data (au)
# energy = np.array(
#     [-5624.237304782903266, -5624.237101763766077, -5624.236644052567499, -5624.236022409476391, -5624.235185641377029,
#      -5624.234069847570026])
# energy = np.array(
#     [-5624.441419951936950, -5624.441403042334059, -5624.441382093848915, -5624.441359407504024, -5624.441297043926170,
#      -5624.441146115573247, -5624.440934803621531, -5624.440432045447778, -5624.439762302926283, -5624.438842252192444])
# pos_Fe = np.array([2.53061,  6.47301,  8.29892]) * parameters.angstrom_to_bohr
# pos_O = np.array([3.68270,  6.97418,  6.79004]) * parameters.angstrom_to_bohr
# filename_parabola = 'goethite/neutral/goethite_FeOFe_a'
# filename_parabola = 'goethite/hole/goethite_FeOFe_a_hole'
# fit_start = 0

# Goethite FeOH_a data (au)
# energy = np.array(
#     [-5624.237304782903266, -5624.237060417912289, -5624.236569710546974, -5624.235880270187408, -5624.234824157740150,
#      -5624.233530306401917])
# energy = np.array(
#     [-5624.441419951936950, -5624.441391085360920, -5624.441327828777503, -5624.441225167193807, -5624.441234292825357,
#      -5624.440790308307442, -5624.440809716951662, -5624.440198482117012, -5624.439227874116114, -5624.438041110972335])
# pos_Fe = np.array([2.53061,  6.47301,  8.29892]) * parameters.angstrom_to_bohr
# pos_O = np.array([1.38266,  5.55573,  6.79006]) * parameters.angstrom_to_bohr
# filename_parabola = 'goethite/neutral/goethite_FeOFe_a'
# filename_parabola = 'goethite/hole/goethite_FeOH_a_hole'
# fit_start = 0

# Goethite FeOH_a_both data (au)
# energy = np.array(
#     [-5624.237304782903266, -5624.237114087055488, -5624.236792094492557, -5624.236403084116318, -5624.235767096566633,
#      -5624.235024289464491])
energy = np.array(
    [-5624.441419951936950, -5624.441391971828125, -5624.441344070175546, -5624.441258909966564, -5624.441276898683100,
     -5624.440938883916715, -5624.441011270105264, -5624.440685896376635, -5624.440120735066557, -5624.439468423900507])
pos_Fe = np.array([2.53061,  6.47301,  8.29892 ]) * parameters.angstrom_to_bohr
pos_O = np.array([1.38266,  5.55573,  6.79006]) * parameters.angstrom_to_bohr
# filename_parabola = 'goethite/neutral/goethite_FeOFe_a_both'
filename_parabola = 'goethite/hole/goethite_FeOH_a_hole_both'
fit_start = 0

# Goethite FeOH_b data (au)
# energy = np.array(
#     [-5624.237304782903266, -5624.236993148584588, -5624.236256859228888, -5624.235266975656486, -5624.233892892567383,
#      -5624.231904634708371])
# energy = np.array(
#     [-5624.441419951936950, -5624.441350526279166, -5624.441117541062340, -5624.440787807823654, -5624.441124112981925,
#      -5624.439262400712323, -5624.440404504111029, -5624.439432712334565, -5624.438085898394093, -5624.436129785654884])
# pos_Fe = np.array([2.53061,  6.47301,  8.29892]) * parameters.angstrom_to_bohr
# pos_O = np.array([3.21517,  4.47859,  8.29893]) * parameters.angstrom_to_bohr
# filename_parabola = 'goethite/neutral/goethite_FeOFe_a'
# filename_parabola = 'goethite/hole/goethite_FeOH_b_hole'
# fit_start = 0

# Goethite FeOH_b_both data (au)
# energy = np.array(
#     [-5624.237304782903266, -5624.237303308193987, -5624.237256771284592, -5624.237198122199516, -5624.237117395961832,
#      -5624.236952860528618, -5624.236759547941801, -5624.236380988229030, -5624.235851935176470, -5624.234925402673070])
# energy = np.array(
#     [-5624.441419951936950, -5624.441421270382307, -5624.441377775157889, -5624.441322678348115, -5624.441246687046259,
#      - 5624.441087500604226, -5624.440904437319659, -5624.440534495633074, -5624.440025059903746, -5624.439121597337362])
# pos_Fe = np.array([2.53061,  6.47301,  8.29892]) * parameters.angstrom_to_bohr
# pos_O = np.array([3.21517,  4.47859,  8.29893]) * parameters.angstrom_to_bohr
# filename_parabola = 'goethite/neutral/goethite_FeOFe_b_both'
# filename_parabola = 'goethite/hole/goethite_FeOH_b_both_hole'
# fit_start = 0

# Saving
folder_save = '/scratch/cahart/work/personal_files/dft_ml_md/output/feIV_bulk/force_constants/'

# Calculate change in length from equilibrium
percent_range = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5])
values = np.shape(percent_range)[0]
energy = energy[0:values] - energy[0]
pos_length_change = np.zeros((values, 3))
length_change = np.zeros(values)

for i in range(0, values):
        pos_length_change[i, :] = (pos_Fe - pos_O) * percent_range[i]/100  # (i/100)
        length_change[i] = np.linalg.norm(pos_length_change[i, :])

# Fit parabola
energy_fit = energy[:values] - energy[fit_start]
length_change_fit = length_change - length_change[fit_start]
optimised, covariance = scipy.optimize.curve_fit(quadratic_fit, length_change_fit[fit_start:], energy_fit[fit_start:])
print('Force constant (au)', optimised[0])
print('Force constant (mdyn/A)', optimised[0]*15.569141)

# Calculate RMSE
RMSE_fit = np.sqrt(np.average((energy-quadratic_fit(length_change, *optimised))**2))
print('RMSE (eV)', RMSE_fit*parameters.hartree_to_ev)

# Plotting
fig_fit1, ax_fit1 = plt.subplots(figsize=(6, 4))
ax_fit1.plot(length_change / parameters.angstrom_to_bohr, energy * parameters.hartree_to_ev, 'kx')
ax_fit1.plot((length_change[fit_start]+length_change_fit[fit_start:]) / parameters.angstrom_to_bohr,
             (quadratic_fit(length_change_fit[fit_start:], *optimised)+energy[fit_start]) * parameters.hartree_to_ev, 'k')
ax_fit1.set_xlabel(r'Change in bond length / $\mathrm{\AA}$')
ax_fit1.set_ylabel('Change in energy / eV')
fig_fit1.tight_layout()
# fig_fit1.savefig('{}{}'.format(folder_save, filename_parabola), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    plt.show()