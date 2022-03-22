from __future__ import division, print_function, unicode_literals
import numpy as np
from matplotlib import pyplot as plt
import load_coordinates
import load_energy
import load_forces

"""
    Plotting
"""

# Plotting variables
save_dpi = 200

# Read CP2K output files
hse_energy_kinetic, hse_energy_potential, temperature, time_val, time_per_step = load_energy.load_values_energy()
pbe_energy_potential = np.loadtxt('data/pbe_energy.out', skiprows=1)

# Plot HSE potential energy
fig_hse_energy_potential = plt.figure()
ax_hse_energy_potential = fig_hse_energy_potential.add_subplot(111)
ax_hse_energy_potential.plot(time_val, hse_energy_potential)
ax_hse_energy_potential.set_xlabel('Time / fs')
ax_hse_energy_potential.set_ylabel('HSE potential energy / Ha')
fig_hse_energy_potential.savefig('{}'.format('output/hse_potential_energy.png'), dpi=save_dpi, bbox_inches='tight')

# Plot HSE potential energy
fig_pbe_energy_potential = plt.figure()
ax_pbe_energy_potential = fig_pbe_energy_potential.add_subplot(111)
ax_pbe_energy_potential.plot(time_val, pbe_energy_potential)
ax_pbe_energy_potential.set_xlabel('Time / fs')
ax_pbe_energy_potential.set_ylabel('PBE potential energy / Ha')
fig_pbe_energy_potential.savefig('{}'.format('output/pbe_potential_energy.png'), dpi=save_dpi, bbox_inches='tight')

# Plot HSE and PBE potential energy
fig_energy_potential = plt.figure()
ax_energy_potential = fig_energy_potential.add_subplot(111)
ax_energy_potential.plot(time_val, hse_energy_potential, label='HSE')
ax_energy_potential.plot(time_val, pbe_energy_potential, label='PBE')
ax_energy_potential.legend()
ax_energy_potential.set_xlabel('Time / fs')
ax_energy_potential.set_ylabel('Potential energy / Ha')
fig_energy_potential.savefig('{}'.format('output/potential_energy.png'), dpi=save_dpi, bbox_inches='tight')

if __name__ == "__main__":
    plt.show()
