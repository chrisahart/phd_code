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
    Four point diagram.
    Script used to plot four point diagram.
"""

hematite_ip = 7.00
folder_save = '../../output//feIV_bulk/four_point_scheme/'

# Hematite hole
hematite_hole_ip = np.array([-0.28158, -0.28129]) * parameters.hartree_to_ev
hematite_hole_ea = np.array([0.30320, 0.30320]) * parameters.hartree_to_ev
hematite_hole_lambda_charged = np.array([0.0046, 0.0063]) * parameters.hartree_to_ev + (0.98-0.71)/2
hematite_hole_lambda_neutral = np.array([0.0170, 0.020]) * parameters.hartree_to_ev + (0.98-0.71)/2
shift_hole = hematite_hole_ip[1] - hematite_ip

# Lepidocrocite hole
lepidicrocite_hole_ip = np.array([-0.17049, -0.1690]) * parameters.hartree_to_ev
lepidicrocite_hole_ea = np.array([0.20816, 0.21208]) * parameters.hartree_to_ev
lepidicrocite_hole_lambda_charged = np.array([0.0111, 0.0161]) * parameters.hartree_to_ev + (1.43-1.17)/2
lepidicrocite_hole_lambda_neutral = np.array([0.0266, 0.0269]) * parameters.hartree_to_ev + (1.43-1.17)/2

# Goethite hole
goethite_hole_ip = np.array([0, -0.20303]) * parameters.hartree_to_ev
goethite_hole_ea = np.array([0, -0.20303]) * parameters.hartree_to_ev
goethite_hole_lambda_charged = np.array([0, 0]) * parameters.hartree_to_ev
goethite_hole_lambda_neutral = np.array([0, 0]) * parameters.hartree_to_ev

# Hematite electron
hematite_electron_ip = np.array([0.35820, 0.35983]) * parameters.hartree_to_ev
hematite_electron_ea = np.array([-0.34434, -0.34403]) * parameters.hartree_to_ev
hematite_electron_lambda_charged = np.array([0.0038, 0.0051]) * parameters.hartree_to_ev + (0.73-0.43)/2
hematite_electron_lambda_neutral = np.array([0.0101, 0.0108]) * parameters.hartree_to_ev + (0.73-0.43)/2
shift_electron = -(hematite_hole_ip[1] - hematite_ip)

# Lepidocrocite electron
lepidicrocite_electron_ip = np.array([0.25236, 0.25322]) * parameters.hartree_to_ev
lepidicrocite_electron_ea = np.array([0, -0.22963]) * parameters.hartree_to_ev
lepidicrocite_electron_lambda_charged = np.array([0, 0.0157]) * parameters.hartree_to_ev + (1.02-0.72)/2
lepidicrocite_electron_lambda_neutral = np.array([0, 0.0109]) * parameters.hartree_to_ev + (1.02-0.72)/2

# Goethite electron
goethite_electron_ip = np.array([0.29197, 0.29344]) * parameters.hartree_to_ev
goethite_electron_ea = np.array([0, -0.26808]) * parameters.hartree_to_ev
goethite_electron_lambda_charged = np.array([0, 0.0142]) * parameters.hartree_to_ev + (1.00-0.69)/2
goethite_electron_lambda_neutral = np.array([0, 0.0083]) * parameters.hartree_to_ev + (1.00-0.69)/2

# Plot hole four point scheme
color = ['r', 'g', 'b']
thickness = 2
offset = 0.1
fig_hole, ax_hole = plt.subplots(figsize=(6, 4))

# Lepidocrocite
ax_hole.plot([0, 0.4], np.ones(2) * (lepidicrocite_hole_ip[1] - shift_hole), color[1], linewidth=thickness)
ax_hole.plot([0.6, 1], np.ones(2) * ((lepidicrocite_hole_ip[1] - shift_hole) - lepidicrocite_hole_lambda_charged[1]),
             color[1], linewidth=thickness)
ax_hole.plot([0.6, 1], np.ones(2) * (lepidicrocite_hole_lambda_neutral[1]), color[1])

# Goethite
ax_hole.plot([0, 0.4], np.ones(2) * 0, color[2],  linewidth=thickness)
ax_hole.plot([0, 0.4], np.ones(2) * (goethite_hole_ip[1] - shift_hole), color[2], linewidth=thickness)
ax_hole.plot([0.6, 1], np.ones(2) * ((goethite_hole_ip[1] - shift_hole) - goethite_hole_lambda_charged[1]), color[2], linewidth=thickness)
ax_hole.plot([0.6, 1], np.ones(2) * (goethite_hole_lambda_neutral[1]), color[2], linewidth=thickness)

# Hematite
ax_hole.plot([0, 0.4], np.ones(2) * 0, color[0],  linewidth=thickness)
ax_hole.plot([0, 0.4], np.ones(2) * (hematite_hole_ip[1] - shift_hole), color[0], linewidth=thickness)
ax_hole.plot([0.6, 1], np.ones(2) * ((hematite_hole_ip[1] - shift_hole) - hematite_hole_lambda_charged[1]), color[0], linewidth=thickness)
ax_hole.plot([0.6, 1], np.ones(2) * (hematite_hole_lambda_neutral[1]), color[0], linewidth=thickness)

# IP arrow
ax_hole.annotate('', xy=(0.2, (hematite_hole_ip[1] - shift_hole)-offset),
                 xytext=(0.2, offset),
                 arrowprops=dict(headlength=10, headwidth=10, color=color[0], width=1.5))
ax_hole.annotate('IP', fontsize=15, xy=(0.22, (hematite_hole_ip[1] - shift_hole) / 2))

# Lambda o arrow
ax_hole.annotate('', xy=(0.58, (hematite_hole_ip[1] - shift_hole) - hematite_hole_lambda_charged[1]),
                 xytext=(0.42, (hematite_hole_ip[1] - shift_hole)),
                 arrowprops=dict(headlength=10, headwidth=10, color=color[0], width=1.5))
ax_hole.annotate('$\lambda_o$', fontsize=15, xy=((0.58 + 0.42) / 2, (hematite_hole_ip[1] - shift_hole) + 0.3))

# EA arrow
ax_hole.annotate('', xy=(0.8, hematite_hole_lambda_neutral[1] + offset),
                 xytext=(0.8, (hematite_hole_ip[1] - shift_hole) - hematite_hole_lambda_charged[1] - offset),
                 arrowprops=dict(headlength=10, headwidth=10, color=color[0], width=1.5))
ax_hole.annotate('EA', fontsize=15, xy=(0.82, ((hematite_hole_ip[1] - shift_hole) - hematite_hole_lambda_charged[1]) / 2))

# Lambda r arrow
ax_hole.annotate('', xy=(0.42, 0),
                 xytext=(0.58, hematite_hole_lambda_neutral[1]),
                 arrowprops=dict(headlength=10, headwidth=10, color=color[0], width=1.5))
ax_hole.annotate('$\lambda_r$', fontsize=15, xy=((0.58 + 0.42) / 2, (hematite_hole_lambda_neutral[1]) + 0.2))

ax_hole.set_xlim([-0.1, 1.1])
ax_hole.spines['right'].set_visible(False)
ax_hole.spines['top'].set_visible(False)
ax_hole.spines['bottom'].set_visible(False)
ax_hole.set_xticks([])
ax_hole.set_ylabel('Energy / eV')
fig_hole.tight_layout()
fig_hole.savefig('{}{}'.format(folder_save, 'hole.png'), dpi=parameters.save_dpi, bbbox_inches='tight')

# Plot electron four point scheme
fig_electron, ax_electron = plt.subplots(figsize=(6, 4))

# Lepidocrocite
ax_electron.plot([0, 0.4], np.ones(2) * (lepidicrocite_electron_ip[1] - shift_electron), color[1], linewidth=thickness)
ax_electron.plot([0.6, 1], np.ones(2) * (
            (lepidicrocite_electron_ip[1] - shift_electron) - lepidicrocite_electron_lambda_charged[1]), color[1], linewidth=thickness)
ax_electron.plot([0.6, 1], np.ones(2) * (lepidicrocite_electron_lambda_neutral[1]), color[1], linewidth=thickness)

# Goethite
ax_electron.plot([0, 0.4], np.ones(2) * (goethite_electron_ip[1] - shift_electron), color[2], linewidth=thickness)
ax_electron.plot([0.6, 1],
                 np.ones(2) * ((goethite_electron_ip[1] - shift_electron) - goethite_electron_lambda_charged[1]),
                 color[2], linewidth=thickness)
ax_electron.plot([0.6, 1], np.ones(2) * (goethite_electron_lambda_neutral[1]), color[2], linewidth=thickness)

# Hematite
ax_electron.plot([0, 0.4], np.ones(2) * 0, color[0], linewidth=thickness)
ax_electron.plot([0, 0.4], np.ones(2) * (hematite_electron_ip[1] - shift_electron), color[0], linewidth=thickness)
ax_electron.plot([0.6, 1],
                 np.ones(2) * ((hematite_electron_ip[1] - shift_electron) - hematite_electron_lambda_charged[1]),
                 color[0], linewidth=thickness)
ax_electron.plot([0.6, 1], np.ones(2) * (hematite_electron_lambda_neutral[1]), color[0], linewidth=thickness)


# IP arrow
ax_electron.annotate('', xy=(0.2, (hematite_electron_ip[1] - shift_electron) + offset),
                     xytext=(0.2, -offset),
                     arrowprops=dict(headlength=10, headwidth=10, color=color[0], width=1.5))
ax_electron.annotate('EA', fontsize=15, xy=(0.22, (hematite_electron_ip[1] - shift_electron) / 2))

# Lambda r arrow
ax_electron.annotate('', xy=(0.58, (hematite_electron_ip[1] - shift_electron) - hematite_electron_lambda_charged[1]),
                     xytext=(0.42, (hematite_electron_ip[1] - shift_electron)),
                     arrowprops=dict(headlength=10, headwidth=10, color=color[0], width=1.5))
ax_electron.annotate('$\lambda_r$', fontsize=15, xy=(
    (0.58 + 0.42) / 2, -0.3 + (hematite_electron_ip[1] - shift_electron) - hematite_electron_lambda_charged[1]))

# EA arrow
ax_electron.annotate('', xy=(0.8, hematite_electron_lambda_neutral[1] - offset),
                     xytext=(0.8, offset + (hematite_electron_ip[1] - shift_electron) - hematite_electron_lambda_charged[1]),
                     arrowprops=dict(headlength=10, headwidth=10, color=color[0], width=1.5))
ax_electron.annotate('IP', fontsize=15,
                     xy=(0.82, ((hematite_electron_ip[1] - shift_electron) - hematite_electron_lambda_charged[1]) / 2))

# Lambda o arrow
ax_electron.annotate('', xy=(0.42, 0),
                     xytext=(0.58, hematite_electron_lambda_neutral[1]),
                     arrowprops=dict(headlength=10, headwidth=10, color=color[0], width=1.5))
ax_electron.annotate('$\lambda_o$', fontsize=15, xy=((0.58 + 0.42) / 2, -0.1))

ax_electron.set_xlim([-0.1, 1.1])
ax_electron.spines['right'].set_visible(False)
ax_electron.spines['top'].set_visible(False)
ax_electron.spines['bottom'].set_visible(False)
ax_electron.invert_yaxis()
ax_electron.set_xticks([])
ax_electron.set_ylabel('Energy / eV')
fig_electron.tight_layout()
fig_electron.savefig('{}{}'.format(folder_save, 'electron.png'), dpi=parameters.save_dpi, bbbox_inches='tight')


if __name__ == "__main__":
    plt.show()

