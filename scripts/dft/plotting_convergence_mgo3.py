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
folder = '/media/chris/DATA/Storage/University/PhD/Programming/dft_ml_md/output/cdft/MgO/data/convergence'
folder_save = '/media/chris/DATA/Storage/University/PhD/Programming/dft_ml_md/output/cdft/MgO/convergence/opt_defect'

# Plotting
pshape = [s + 'x-' for s in parameters.plotting_colors]
pshape2 = [s + 'x--' for s in parameters.plotting_colors]

# Data geometry optimisation full
file_log = '{}/neutral_2400_140_opt_defect/cp2k_log.log'.format(folder)
bandgap_2400_140 = functions.extract_cp2k_log(file_log, ' HOMO - LUMO gap [eV] : ')
energy_2400_140 = functions.extract_cp2k_log(file_log, 'Total F') * parameters.hartree_to_ev
force_2400_140 = functions.extract_cp2k_log(file_log, 'Max. gradient')
time_2400_140 = functions.extract_cp2k_log(file_log, 'CP2K                                 1  1.0 ')

file_log = '{}/neutral_3000_140_opt_defect/cp2k_log.log'.format(folder)
bandgap_3000_140 = functions.extract_cp2k_log(file_log, ' HOMO - LUMO gap [eV] : ')
energy_3000_140 = functions.extract_cp2k_log(file_log, 'Total F') * parameters.hartree_to_ev
force_3000_140 = functions.extract_cp2k_log(file_log, 'Max. gradient')
time_3000_140 = functions.extract_cp2k_log(file_log, 'CP2K                                 1  1.0 ')

# Band gap
fig_bandgap, ax_bandgap = plt.subplots()
ax_bandgap.plot(bandgap_2400_140[::2], pshape[0], label='2400')
ax_bandgap.plot(bandgap_3000_140[::2], pshape[1], label='3000')
ax_bandgap.set_xlabel('Geometry optimisation step')
ax_bandgap.set_ylabel('Band gap / eV')
ax_bandgap.legend(frameon=True)
fig_bandgap.tight_layout()
fig_bandgap.savefig('{}/geoopt_bandgap.png'.format(folder_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# Energy
fig_energy, ax_energy = plt.subplots()
ax_energy.plot(energy_2400_140[::2], pshape[0], label='2400')
ax_energy.plot(energy_3000_140[::2], pshape[1], label='3000')
ax_energy.set_xlabel('Geometry optimisation step')
ax_energy.set_ylabel('Energy / eV')
ax_energy.legend(frameon=True)
fig_energy.tight_layout()
fig_energy.savefig('{}/geoopt_energy.png'.format(folder_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# Force
fig_force, ax_force = plt.subplots()
ax_force.plot(force_2400_140[::2], pshape[0], label='2400')
ax_force.plot(force_3000_140[::2], pshape[1], label='3000')
ax_force.set_xlabel('Geometry optimisation step')
ax_force.set_ylabel('force / eV')
ax_force.legend(frameon=True)
fig_force.tight_layout()
fig_force.savefig('{}/geoopt_force.png'.format(folder_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# Time
fig_time, ax_time = plt.subplots()
ax_time.plot(time_2400_140[::2], pshape[0], label='2400')
ax_time.plot(time_3000_140[::2], pshape[1], label='3000')
ax_time.set_xlabel('Geometry optimisation step')
ax_time.set_ylabel('time / eV')
ax_time.legend(frameon=True)
fig_time.tight_layout()
fig_time.savefig('{}/geoopt_time.png'.format(folder_save), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()
