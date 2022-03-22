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

hematite_ip = 4.88+2.2
folder_save = '../../output//feIV_bulk/four_point_scheme/'

# Hematite hole
hematite_hole_ip = np.array([-0.28158, -0.28129]) * parameters.hartree_to_ev
hematite_hole_ea = np.array([0.30320, 0.30755]) * parameters.hartree_to_ev
hematite_hole_lambda_charged = np.array([0.0046, 0.0063]) * parameters.hartree_to_ev + (0.49-0.36)/2
hematite_hole_lambda_neutral = np.array([0.0170, 0.020]) * parameters.hartree_to_ev + (0.49-0.36)/2
shift_hole = hematite_hole_ip[1] - hematite_ip
print('shift_hole', shift_hole)

# Lepidocrocite hole
lepidicrocite_hole_ip = np.array([-0.17049, -0.1690]) * parameters.hartree_to_ev
lepidicrocite_hole_ea = np.array([0.20816, 0.21208]) * parameters.hartree_to_ev
lepidicrocite_hole_lambda_charged = np.array([0.0111, 0.0161]) * parameters.hartree_to_ev + (0.72-0.59)/2
lepidicrocite_hole_lambda_neutral = np.array([0.0266, 0.0269]) * parameters.hartree_to_ev + (0.72-0.59)/2

# Goethite hole
goethite_hole_ip = np.array([0, -0.20303]) * parameters.hartree_to_ev
goethite_hole_ea = np.array([0, -0.20303]) * parameters.hartree_to_ev
goethite_hole_lambda_charged = np.array([0, 0]) * parameters.hartree_to_ev
goethite_hole_lambda_neutral = np.array([0, 0]) * parameters.hartree_to_ev

# Hematite electron
hematite_electron_ip = np.array([0.35820, 0.35983]) * parameters.hartree_to_ev
hematite_electron_ea = np.array([-0.34434, -0.34403]) * parameters.hartree_to_ev
hematite_electron_lambda_charged = np.array([0.0038, 0.0051]) * parameters.hartree_to_ev + (0.37-0.22)/2
hematite_electron_lambda_neutral = np.array([0.0101, 0.0108]) * parameters.hartree_to_ev + (0.37-0.22)/2
shift_electron = -(hematite_hole_ip[1] - hematite_ip)

# Lepidocrocite electron
lepidicrocite_electron_ip = np.array([0.25236, 0.25322]) * parameters.hartree_to_ev
lepidicrocite_electron_ea = np.array([0, -0.22963]) * parameters.hartree_to_ev
lepidicrocite_electron_lambda_charged = np.array([0, 0.0157]) * parameters.hartree_to_ev + (0.51-0.36)/2
lepidicrocite_electron_lambda_neutral = np.array([0, 0.0109]) * parameters.hartree_to_ev + (0.51-0.36)/2

# Goethite electron
goethite_electron_ip = np.array([0.29197, 0.29344]) * parameters.hartree_to_ev
goethite_electron_ea = np.array([0, -0.26808]) * parameters.hartree_to_ev
goethite_electron_lambda_charged = np.array([0, 0.0142]) * parameters.hartree_to_ev + (0.50-0.35)/2
goethite_electron_lambda_neutral = np.array([0, 0.0083]) * parameters.hartree_to_ev + (0.50-0.35)/2

print('hematite_hole_ip', hematite_hole_ip)
print('hematite_electron_ip', hematite_electron_ip)
print('shift_hole', shift_hole)
print('(hematite_hole_ip[1] - shift_hole)', (hematite_hole_ip[1] - shift_hole))
print('(hematite_electron_ip[1] - shift_hole)', (hematite_electron_ip[1] + shift_hole))

print('\nlepidicrocite_hole_ip', lepidicrocite_hole_ip)
print('lepidicrocite_electron_ip', lepidicrocite_electron_ip)
print('shift_hole', shift_hole)
print('(lepidicrocite_hole_ip[1] - shift_hole)', (lepidicrocite_hole_ip[1] - shift_hole))
print('(lepidicrocite_electron_ip[1] - shift_hole)', (lepidicrocite_electron_ip[1] + shift_hole))

print('\ngoethite_hole_ip', goethite_hole_ip)
print('goethite_electron_ip', goethite_electron_ip)
print('shift_hole', shift_hole)
print('(goethite_hole_ip[1] - shift_hole)', (goethite_hole_ip[1] - shift_hole))
print('(goethite_electron_ip[1] - shift_hole)', (goethite_electron_ip[1] + shift_hole))

# Plot hole four point scheme
color = ['r', 'g', 'b']
thickness = 3.5
offset = 0.3
length = 0.1
font = 13
fig_hole, ax_hole = plt.subplots(figsize=(6, 4))

# Hematite
ax_hole.plot([0, length], -1*np.ones(2) * (hematite_hole_ip[1] - shift_hole), color[0], linewidth=thickness)
ax_hole.plot([0, length], -1*np.ones(2) * (hematite_hole_ip[1] - shift_hole - hematite_hole_lambda_charged[1]), 'r--', linewidth=thickness/2)
ax_hole.plot([0, length], 1*np.ones(2) * (hematite_electron_ip[1] + shift_hole), color[0], linewidth=thickness)
ax_hole.plot([0, length], 1*np.ones(2) * (hematite_electron_ip[1] + shift_hole - hematite_electron_lambda_charged[1]), 'r--', linewidth=thickness/2)
ax_hole.plot([0, 0], [-1 * (hematite_hole_ip[1] - shift_hole), (hematite_electron_ip[1] + shift_hole)], 'grey', linewidth=thickness)

# Hematite labels
ax_hole.annotate(r'-IP$_{\rm v}$', fontsize=font, xy=(length+0.01, -(hematite_hole_ip[1] - shift_hole)-0.1))
ax_hole.annotate(r'-IP$_{\rm ad}$', fontsize=font, xy=(length+0.01, -(hematite_hole_ip[1] - shift_hole - hematite_hole_lambda_charged[1])+0.1))
ax_hole.annotate(r'-EA$_{\rm v}$', fontsize=font, xy=(length+0.01, (hematite_electron_ip[1] + shift_hole)))
ax_hole.annotate(r'-EA$_{\rm ad}$', fontsize=font, xy=(length+0.01, (hematite_electron_ip[1] + shift_hole - hematite_electron_lambda_charged[1])-0.2))

# Lepidocrocite
ax_hole.plot([0+offset, length+offset], -1*np.ones(2) * (lepidicrocite_hole_ip[1] - shift_hole), color[1], linewidth=thickness)
ax_hole.plot([0+offset, length+offset], -1*np.ones(2) * (lepidicrocite_hole_ip[1] - shift_hole - lepidicrocite_hole_lambda_charged[1]), 'g--', linewidth=thickness/2)
ax_hole.plot([0+offset, length+offset], 1*np.ones(2) * (lepidicrocite_electron_ip[1] + shift_hole), color[1], linewidth=thickness)
ax_hole.plot([0+offset, length+offset], 1*np.ones(2) * (lepidicrocite_electron_ip[1] + shift_hole - lepidicrocite_electron_lambda_charged[1]), 'g--', linewidth=thickness/2)
ax_hole.plot([1*offset, 1*offset], [-1 * (lepidicrocite_hole_ip[1] - shift_hole), (lepidicrocite_electron_ip[1] + shift_hole)], 'grey', linewidth=thickness)

# Lepidocrocite labels
ax_hole.annotate(r'-IP$_{\rm v}$', fontsize=font, xy=(1*offset+length+0.01, -(lepidicrocite_hole_ip[1] - shift_hole)-0.1))
ax_hole.annotate(r'-IP$_{\rm ad}$', fontsize=font, xy=(1*offset+length+0.01, -(lepidicrocite_hole_ip[1] - shift_hole - lepidicrocite_hole_lambda_charged[1])+0.0))
ax_hole.annotate(r'-EA$_{\rm v}$', fontsize=font, xy=(1*offset+length+0.01, (lepidicrocite_electron_ip[1] + shift_hole)))
ax_hole.annotate(r'-EA$_{\rm ad}$', fontsize=font, xy=(1*offset+length+0.01, (lepidicrocite_electron_ip[1] + shift_hole - lepidicrocite_electron_lambda_charged[1])-0.0))

# # Goethtite
ax_hole.plot([0+2*offset, length+2*offset], -1*np.ones(2) * (goethite_hole_ip[1] - shift_hole), color[2], linewidth=thickness)
ax_hole.plot([0+2*offset, length+2*offset], -1*np.ones(2) * (goethite_hole_ip[1] - shift_hole - goethite_hole_lambda_charged[1]), 'b--', linewidth=thickness/2)
ax_hole.plot([0+2*offset, length+2*offset], 1*np.ones(2) * (goethite_electron_ip[1] + shift_hole), color[2], linewidth=thickness)
ax_hole.plot([0+2*offset, length+2*offset], 1*np.ones(2) * (goethite_electron_ip[1] + shift_hole - goethite_electron_lambda_charged[1]), 'b--', linewidth=thickness/2)
ax_hole.plot([2*offset, 2*offset], [-1 * (goethite_hole_ip[1] - shift_hole), (goethite_electron_ip[1] + shift_hole)], 'grey', linewidth=thickness)

# Goethite labels
ax_hole.annotate(r'-IP$_{\rm v}$', fontsize=font, xy=(2*offset+length+0.01, -(goethite_hole_ip[1] - shift_hole)-0.1))
ax_hole.annotate(r'-EA$_{\rm v}$', fontsize=font, xy=(2*offset+length+0.01, (goethite_electron_ip[1] + shift_hole)))
ax_hole.annotate(r'-EA$_{\rm ad}$', fontsize=font, xy=(2*offset+length+0.01, (goethite_electron_ip[1] + shift_hole - goethite_electron_lambda_charged[1])-0.2))

# Hydrogen
ax_hole.plot([-10, 10], [-4.4, -4.4], 'k--', linewidth=thickness/4)
ax_hole.plot([-10, 10], [-4.4-1.23, -4.4-1.23], 'k--', linewidth=thickness/4)

# Hydrogen labels
ax_hole.annotate(r'$\rm{H_{2}/H^{+}}$ pH = 0', fontsize=font, xy=(0.4, -4.4+0.1))
ax_hole.annotate(r'$\rm{O_{2}/H_{2}O}$', fontsize=font, xy=(0.4, -4.4-1.23+0.1))

# Vacuum
ax_hole.set_ylim([-10.5, -3])
ax_hole.set_xlim([-0.05, 2*offset + length + 0.15])
ax_hole.set_xticks([])
ax_hole.set_ylabel('Energy wrt vacuum (eV)')

# Hydrogen
ax2 = ax_hole.twinx()
mn, mx = ax_hole.get_ylim()
ax2.set_ylim(-(mn+4.4), -(mx+4.4))
ax2.set_ylabel('Redox potential wrt NHE (V)')

# Disable axis
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax_hole.spines['right'].set_visible(False)
ax_hole.spines['top'].set_visible(False)
ax_hole.spines['bottom'].set_visible(False)

fig_hole.tight_layout()
fig_hole.savefig('{}{}'.format(folder_save, 'band_edge.png'), dpi=parameters.save_dpi, bbbox_inches='tight')


if __name__ == "__main__":
    plt.show()

