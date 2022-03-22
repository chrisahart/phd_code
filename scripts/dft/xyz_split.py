from __future__ import division, print_function
import time
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from scripts.general import functions
from scripts.general import parameters
from scripts.formatting import load_coordinates
from scripts.formatting import load_energy
from scripts.formatting import load_forces_out
from scripts.formatting import load_forces
from scripts.formatting import print_xyz


"""
    Split xyz file into individual files for other use
"""

folder_in = '/media/chris/Elements/Backup/Archer-2/other/cdft/ru/md/b97x/equilibrated/cdft-24h-inverse/analysis/position'
filename_in = 'initial-timcon-33-rattle-cpmd-rel-ru-water-run-001.out'
folder_out = '/media/chris/Elements/Backup/Archer-2/other/cdft/ru/md/b97x/lambda/initial-timcon-33-rattle-cpmd-rel-ru-water'
# start_step = 801
start_step = 1021

file_path = '{}/xyz'.format(folder_out)
if not os.path.exists(file_path):
    os.makedirs(file_path)

coordinates, coord_x, coord_y, coord_z, species, num_atoms, num_timesteps = load_coordinates.load_values_coord(folder_in, filename_in)
print_steps = np.arange(start=1020,stop=2010,step=10)
print(print_steps)

for i in range(print_steps.shape[0]):
    coord = np.column_stack((coord_x[print_steps[i]-start_step].ravel(), coord_y[print_steps[i]-start_step].ravel(), coord_z[print_steps[i]-start_step].ravel()))
    coord_xyz = pd.DataFrame(data=coord)
    coord_xyz.insert(loc=0, column='A', value=pd.Series(species).values)
    print_xyz.print_from_pandas(coord_xyz, num_atoms, '{}/step-{}'.format(file_path, print_steps[i]))
