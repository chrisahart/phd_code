from __future__ import division, print_function, unicode_literals
import time
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)
from matplotlib import pyplot as plt
import load_coordinates
import load_energy
import load_forces
import matplotlib.cm as cm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import functions

"""
    General parameters
"""