from __future__ import division, print_function
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
import scipy
import re
import pickle
import pandas as pd
from distutils.dir_util import copy_tree
import copy
from scripts.formatting import load_coordinates
from scripts.general import functions
from scripts.formatting import print_xyz
from scripts.formatting import cp2k_hirsh

""" Calculate beta value """


def func(x):
    return [x[0]*np.exp(-x[1]*5/2)+x[0]*np.exp(-x[1]*10/2) - 0.0013645861425814689,
            x[0]*np.exp(-x[1]*5.9/2)+x[0]*np.exp(-x[1]*10.5/2) - 0.001065925578462248]


# print(func([0.005, 0.6]))

root = scipy.optimize.fsolve(func, [1, 1])
print(root)
print(np.isclose(func(root), [0.0, 0.0]))
