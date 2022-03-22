from __future__ import division, print_function
import pandas as pd
import numpy as np
import glob
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh
import matplotlib.pyplot as plt
from scripts.general import parameters

"""
    Convergence multigrid. 
    Plot convergence of energy and energy difference with multigrid.
"""

# Folder
folder = '/media/chris/DATA/Storage/University/PhD/Programming/dft_ml_md/output/cdft/MgO/data'
folder_save = '/media/chris/DATA/Storage/University/PhD/Programming/dft_ml_md/output/cdft/MgO/convergence/single'

# Data geometry optimisation partial (2 steps)
band_gap = np.loadtxt('{}/band_gap.out'.format(folder))
band_gap_opt2 = np.loadtxt('{}/band_gap_opt2.out'.format(folder))
energy = np.loadtxt('{}/energy.out'.format(folder)) * parameters.hartree_to_ev
energy_opt2 = np.loadtxt('{}/energy_opt2.out'.format(folder)) * parameters.hartree_to_ev
force = np.loadtxt('{}/force.out'.format(folder))
force_opt2 = np.loadtxt('{}/force_opt2.out'.format(folder))
time = np.loadtxt('{}/time.out'.format(folder))
charge_density = np.loadtxt('{}/charge_density.out'.format(folder))

# Grid
cutoff = np.array([600, 1200, 1800, 2400, 3000, 4000])
rel_cutoff = np.array([60, 140, 200, 250])
cutoff_rel_cutoff = []
pstart = 1
prel = rel_cutoff.shape[0]
ptotal = cutoff.shape[0]*rel_cutoff.shape[0]
pshape = [s + 'x--' for s in parameters.plotting_colors]
pshape2 = [s + 'x-.' for s in parameters.plotting_colors]
for i in range(cutoff.shape[0]):
    for j in range(rel_cutoff.shape[0]):
        cutoff_rel_cutoff.append('{}_{}'.format(cutoff[i], rel_cutoff[j]))
cutoff_rel_cutoff = np.array(cutoff_rel_cutoff)

# Band gap
fig_bandgap, ax_bandgap = plt.subplots()
for i in range(prel):
    ax_bandgap.plot(cutoff[pstart:], band_gap[pstart*prel+i:ptotal:prel], pshape[i], label=rel_cutoff[i])
ax_bandgap.set_xlabel('Multigrid cutoff')
ax_bandgap.set_ylabel('Band gap / eV')
ax_bandgap.legend(frameon=True)
fig_bandgap.tight_layout()
fig_bandgap.savefig('{}/bandgap.png'.format(folder_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# Band gap
fig_bandgap2, ax_bandgap2 = plt.subplots()
for i in range(prel):
    ax_bandgap2.plot(cutoff[pstart:], band_gap_opt2[pstart*prel+i:ptotal:prel], pshape[i], label=rel_cutoff[i])
ax_bandgap2.set_xlabel('Multigrid cutoff')
ax_bandgap2.set_ylabel('Band gap / eV')
ax_bandgap2.legend(frameon=True)
fig_bandgap2.tight_layout()
fig_bandgap2.savefig('{}/bandgap2.png'.format(folder_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# Total energy
fig_energy, ax_energy = plt.subplots()
for i in range(prel):
    ax_energy.plot(cutoff[pstart:], energy[pstart*prel+i:ptotal:prel], pshape[i], label=rel_cutoff[i])
ax_energy.set_xlabel('Multigrid cutoff')
ax_energy.set_ylabel('Energy / eV')
ax_energy.legend(frameon=True)
fig_energy.tight_layout()
fig_energy.savefig('{}/energy.png'.format(folder_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# Force
fig_force, ax_force = plt.subplots()
for i in range(prel):
    ax_force.plot(cutoff[pstart:], force[pstart*prel+i:ptotal:prel], pshape[i], label=rel_cutoff[i])
    # ax_force.plot(cutoff[pstart:], force_opt2[pstart * prel + i:ptotal:prel], pshape[i])
ax_force.set_xlabel('Multigrid cutoff')
ax_force.set_ylabel('Force / au')
ax_force.legend(frameon=True)
fig_force.tight_layout()
fig_force.savefig('{}/force.png'.format(folder_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# Time
fig_time, ax_time = plt.subplots()
for i in range(prel):
    ax_time.plot(cutoff[pstart:], time[pstart*prel+i:ptotal:prel], pshape[i], label=rel_cutoff[i])
ax_time.set_xlabel('Multigrid cutoff')
ax_time.set_ylabel('Time / s')
ax_time.legend(frameon=True)
fig_time.tight_layout()
fig_time.savefig('{}/time.png'.format(folder_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# Charge density
fig_charge_density, ax_charge_density = plt.subplots()
for i in range(prel):
    ax_charge_density.plot(cutoff[pstart:], charge_density[pstart*prel+i:ptotal:prel], pshape[i], label=rel_cutoff[i])
ax_charge_density.set_xlabel('Multigrid cutoff')
ax_charge_density.set_ylabel('Charge density on grid')
ax_charge_density.legend(frameon=True)
fig_charge_density.tight_layout()
fig_charge_density.savefig('{}/charge_density.png'.format(folder_save), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()