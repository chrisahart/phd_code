from __future__ import division, print_function
import time
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

"""
    General parameters
"""

# Change matplotlib defaults to large title labels and standard form
params = {'axes.formatter.limits': [-4, 4],
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'lines.markersize': '8',
          }
plt.rcParams.update(params)

# Plotting parameters
save_dpi = 500  # DPI to save figure
time_figsize = (10, 3)  # Figure size for time dependency plots
parity_figsize = (6.4, 4.8)  # Figure size for parity plots
plotting_colors = ['r', 'g', 'b', 'm', 'grey']
plotting_opacity = 0.3

# Conversions
hartree_to_ev = 27.2114  # Energy conversion
angstrom_to_bohr = 1.8897259886  # Distance conversion
hartree_per_bohr_to_ev_per_angstrom = hartree_to_ev / angstrom_to_bohr  # Force conversion
bohr = 0.529  # Conversion Angstrom to Bohr

# Universal parameters
sfc_converge = 1E-5  # Convergence target
atom_charge = [6.0, 1.0, 1.0, 6.0, 1.0, 1.0]  # Atomic charges OHHOHH
