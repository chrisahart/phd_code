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
    Test.
"""


def boltzmann(x, kbt):
    return np.exp(-(x)/(kbt))

# constant = 1.38064852e-23 * 300
temp = 300
constant = 8.617e-5 * 1000 * temp

# nn-1
# val_lambda = np.array([652, 814, 814, 784, 784, 865, 865, 881, 881]) / 4
# val_hab = np.square(np.array([203, 110, 110, 101, 101, 53, 53, 39, 39]))  # CDFT
# val_hab = np.square(np.array([44, 39, 39, 38, 38, 32, 32, 31, 31]))  # Energy gap 1
# val_hab = np.square(np.array([183, 139, 139, 137, 137, 104, 104, 93, 93]))  # Energy gap 1

# nn-2
val_lambda = np.array([1050, 1050, 1028, 1016, 1022, 1034]) / 4
val_hab = np.square(np.array([15, 8, 15, 30, 28, 16]))  # CDFT

# nn-3
# val_lambda = np.array([1087, 1106, 1026]) / 4
# val_hab = np.square(np.array([3, 9, 45]))  # CDFT

val = val_hab
weight = val_lambda
mean = np.mean(val)
rms = np.sqrt(np.mean(val**2))
mean_boltzmann = 0
for i in range(val.shape[0]):
    mean_boltzmann +=boltzmann(weight[i], constant)*val[i]
mean_boltzmann = mean_boltzmann / (np.sum(boltzmann(weight, constant)))

print('Coupling hab')
# print(np.sqrt(mean))
# print(np.sqrt(rms))
print(np.sqrt(mean_boltzmann))

val = val_lambda
weight = val_lambda
mean = np.mean(val)
rms = np.sqrt(np.mean(val**2))
mean_boltzmann = 0
for i in range(val.shape[0]):
    mean_boltzmann +=boltzmann(weight[i], constant)*val[i]
mean_boltzmann = mean_boltzmann / (np.sum(boltzmann(weight, constant)))

print('Reorganisation energy lambda')
# print(mean * 4)
# print(rms * 4)
print(mean_boltzmann * 4)
