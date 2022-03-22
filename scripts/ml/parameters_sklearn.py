from __future__ import division, print_function
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
import matplotlib.cm as cm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

"""
    Parameters for scikit-learn
"""

# Parameters
distance_shortest = 0.5  # Shortest distance for eta grid
distance_eta = 5  # Cutoff for eta grid (less than or equal to distance_cutoff
distance_cutoff = 10  # Cutoff for damping function and eta grid
eta_grid = 5  # Size of eta grid
eta_array = np.logspace(np.log10(distance_shortest), np.log10(distance_eta), num=eta_grid)  # Eta grid for AGNI ACSF
training_size = 50  # 0  # Training set size
folds = 2  # Number of cross validation steps for ML (2 is adequate)

# Kernel ridge regression (2 hyperparameter grid search)
# Gamma is proportional to curvature of function
# Alpha is inversely proportional to penalty for missing point
# krr_init = KernelRidge(kernel='rbf')
# krr = GridSearchCV(krr_init, cv=folds, param_grid={"alpha": np.logspace(-7, 2, num=15),
#                                                    "gamma": np.logspace(-5, 1, num=10)})
# krr = GridSearchCV(krr_init, cv=folds, param_grid={"alpha": np.logspace(-3, 3, num=10),
#                                                    "gamma": np.logspace(-6, 2, num=10)})

# Gaussian process regression (grid search for alpha value, measure of noise in data)
# gpr_init = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b')
# gpr = GridSearchCV(gpr_init, cv=folds, param_grid={'alpha': [1e-4]})
# gpr = GridSearchCV(gpr_init, cv=folds, param_grid={'alpha': np.logspace(1e-1, 1e1, num=10)})

# gpr = GaussianProcessRegressor()
# 1e-4 for for force and energy, 1e-6 for force only, 1e-3 for energy only? odd trend, perhaps energy is smoother
# for coulomb matrix lower value of 1e-10 is used, though gpr = GaussianProcessRegressor() is best which is odd
# for coulomb with force and energy 1e0 is best, for energy with R, F, E 1e-2?

# Neural network (1 hyperparameter grid search)
# Alpha is inversely proportional to penalty for missing point
# nn_init = MLPRegressor(max_iter=500, tol=1e-8, activation='logistic', solver='lbfgs')
# nn = GridSearchCV(nn_init, cv=folds, param_grid={'alpha': np.logspace(-8, 2, num=10)})
# nn = GridSearchCV(nn_init, cv=folds, param_grid={'alpha': np.logspace(-4, 2, num=10)})
# for internal vectors or coordinates np.logspace(-8, -4, num=10) works reasonably

# ML parameter grid
alpha_values = np.logspace(-7, 1, num=10)  # Inversely proportional to penalty for missing point
gamma_values = np.logspace(-7, 1, num=10)  # Proportional to curvature of function
gpr_alpha = np.logspace(-3, 2, num=10)  # GPR requires higher lower limit for stability

# Kernel ridge regression
# krr_init = KernelRidge(kernel='rbf')
# krr = GridSearchCV(krr_init, cv=folds, param_grid={"alpha": alpha_values, "gamma": gamma_values})

# Gaussian process regression
# gpr_init = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b')
# gpr = GridSearchCV(gpr_init, cv=folds, param_grid={'alpha': [1e-2]})
# gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e-10)

# kernel = 1.0 * RBF(1.0)
# gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True)

# kernel = DotProduct() ** 2
# gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
# alpha=1e-3 For energy calculation with Coulomb matrix and with PBE energy (representation or difference)
# alpha=1e0 For energy calculation with Coulomb matrix only, or for force and energy calculation
# alpha=1e-2 For energy and force calculation for classical single molecule from distances
# alpha=1e-4 For energy and force calculation for classical single molecule from direction resolved distances

# Neural network
# nn_init = MLPRegressor(max_iter=500, tol=1e-8, activation='logistic', solver='lbfgs')
# nn = GridSearchCV(nn_init, cv=folds, param_grid={'alpha': [1e-1]})
# nn = MLPRegressor(max_iter=500, tol=1e-8, activation='logistic', solver='lbfgs', alpha=1e-2)
# alpha=1e-6 For energy calculation with Coulomb matrix and with PBE energy (representation or difference)
# alpha=1e-1 For energy calculation with Coulomb matrix only, or for force/energy calculation
# alpha=1e-2 For energy and force calculation for classical single molecule

# Without standardisation (force and energy with coulomb quantum)
# krr_init = KernelRidge(kernel='rbf')
# krr = GridSearchCV(krr_init, cv=folds, param_grid={"alpha": np.logspace(-4, 1, num=10),
#                                                    "gamma": np.logspace(-4, 1, num=10)})
#
# gpr_init = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b')
# gpr = GridSearchCV(gpr_init, cv=folds, param_grid={'alpha': [1e-5]})
#
# nn_init = MLPRegressor(max_iter=500, tol=1e-8, activation='logistic', solver='lbfgs')
# nn = GridSearchCV(nn_init, cv=folds, param_grid={'alpha': np.logspace(-2, 6, num=10)})

# Without standardisation (energy only with coulomb quantum)
# krr_init = KernelRidge(kernel='rbf')
# krr = GridSearchCV(krr_init, cv=folds, param_grid={"alpha": np.logspace(-4, 1, num=10),
#                                                    "gamma": np.logspace(-4, 1, num=10)})
#
# gpr_init = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b')
# gpr = GridSearchCV(gpr_init, cv=folds, param_grid={'alpha': [1e-8]})
#
# nn_init = MLPRegressor(max_iter=500, tol=1e-8, activation='logistic', solver='lbfgs')
# nn = GridSearchCV(nn_init, cv=folds, param_grid={'alpha': np.logspace(-2, 6, num=10)})


# two water molecule distance + distance2 with bond distances
# krr_init = KernelRidge(kernel='rbf')
# krr = GridSearchCV(krr_init, cv=folds, param_grid={"alpha": np.logspace(-6, -1, num=10),
#                                                    "gamma": np.logspace(-6, -4, num=10)})
# gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e-6)
# nn_init = MLPRegressor(max_iter=500, tol=1e-8, activation='logistic', solver='lbfgs')
# nn = GridSearchCV(nn_init, cv=folds, param_grid={'alpha': [1e-2]})

# two water molecule dimer
# krr_init = KernelRidge(kernel='rbf')
# krr = GridSearchCV(krr_init, cv=folds, param_grid={"alpha": np.logspace(-6, 4, num=10),
#                                                    "gamma": np.logspace(-6, 4, num=10)})
# # gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e-1)
# gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e-3)
# # for training set size of 100  1e-4 works, for 1000 0.7e-6 is best (really need a better convergence method)
# nn_init = MLPRegressor(max_iter=500, tol=1e-8, activation='logistic', solver='lbfgs')
# nn = GridSearchCV(nn_init, cv=folds, param_grid={'alpha': [1e-2]})

# two water molecule dimer 1/r6 representation not standardised
# krr_init = KernelRidge(kernel='rbf')
# krr = GridSearchCV(krr_init, cv=folds, param_grid={"alpha": np.logspace(-6, 4, num=10),
#                                                    "gamma": np.logspace(-6, 4, num=10)})
# gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e0)
# # for training set size of 100  1e-4 works, for 1000 0.7e-6 is best (really need a better convergence method)
# nn_init = MLPRegressor(max_iter=500, tol=1e-8, activation='logistic', solver='lbfgs')
# nn = GridSearchCV(nn_init, cv=folds, param_grid={'alpha': [1e-2]})

# two water molecule dimer Botu representation
# krr_init = KernelRidge(kernel='rbf')
# krr = GridSearchCV(krr_init, cv=folds, param_grid={"alpha": np.logspace(-6, 4, num=10),
#                                                    "gamma": np.logspace(-6, 4, num=10)})
# gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e-3)
# # for training set size of 100  1e-4 works, for 1000 0.7e-6 is best (really need a better convergence method)
# nn_init = MLPRegressor(max_iter=500, tol=1e-8, activation='logistic', solver='lbfgs')
# nn = GridSearchCV(nn_init, cv=folds, param_grid={'alpha': [1e-7]})

# single water molecule force (failed)
# krr_init = KernelRidge(kernel='rbf')
# krr = GridSearchCV(krr_init, cv=folds, param_grid={"alpha": np.logspace(-6, -4, num=10),
#                                                    "gamma": np.logspace(-6, -4, num=10)})
# # gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e-1)
# gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e-3)
# # for training set size of 100  1e-4 works, for 1000 0.7e-6 is best (really need a better convergence method)
# nn_init = MLPRegressor(max_iter=500, tol=1e-8, activation='logistic', solver='lbfgs')
# nn = GridSearchCV(nn_init, cv=folds, param_grid={'alpha': [1e-2]})

# two water molecule dimer botu
# krr_init = KernelRidge(kernel='rbf')
# krr = GridSearchCV(krr_init, cv=folds, param_grid={"alpha": np.logspace(-8, 6, num=10),
#                                                    "gamma": np.logspace(-8, 6, num=10)})
# # gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e-1)
# gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e-1)
# # for training set size of 100  1e-4 works, for 1000 0.7e-6 is best (really need a better convergence method)
# nn_init = MLPRegressor(max_iter=500, tol=1e-8, activation='logistic', solver='lbfgs')
# nn = GridSearchCV(nn_init, cv=folds, param_grid={'alpha': [1e-2]})

# single water molecule coulomb un-standardised (symmetrised or not) .2e-8
krr_init = KernelRidge(kernel='rbf')
krr = GridSearchCV(krr_init, cv=folds, param_grid={"alpha": np.logspace(-8, 6, num=20),
                                                   "gamma": np.logspace(-8, 6, num=20)})
# gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e-1)
gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=0.2e-8)
# for training set size of 100  1e-4 works, for 1000 0.7e-6 is best (really need a better convergence method)
nn_init = MLPRegressor(max_iter=500, tol=1e-8, activation='logistic', solver='lbfgs')
nn = GridSearchCV(nn_init, cv=folds, param_grid={'alpha': [1e-2]})

# single water molecule coulomb un-standardised (symmetrised or not) force prediction
# krr_init = KernelRidge(kernel='rbf')
# krr = GridSearchCV(krr_init, cv=folds, param_grid={"alpha": np.logspace(-4, 6, num=20),
#                                                    "gamma": np.logspace(-4, -2, num=20)})
# # gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e-1)
# gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e-5)
# # for training set size of 100  1e-4 works, for 1000 0.7e-6 is best (really need a better convergence method)
# nn_init = MLPRegressor(max_iter=500, tol=1e-8, activation='logistic', solver='lbfgs')
# nn = GridSearchCV(nn_init, cv=folds, param_grid={'alpha': [1e-2]})

# single water molecule coulomb standardised
# krr_init = KernelRidge(kernel='rbf')
# krr = GridSearchCV(krr_init, cv=folds, param_grid={"alpha": np.logspace(-8, 6, num=20),
#                                                    "gamma": np.logspace(-8, 6, num=20)})
# # gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e-1)
# gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e0)
# # for training set size of 100  1e-4 works, for 1000 0.7e-6 is best (really need a better convergence method)
# nn_init = MLPRegressor(max_iter=500, tol=1e-8, activation='logistic', solver='lbfgs')
# nn = GridSearchCV(nn_init, cv=folds, param_grid={'alpha': [1e-2]})

# single water molecule coulomb standardised (symmetrised) (standardised works for everything as expected)
# krr_init = KernelRidge(kernel='rbf')
# krr = GridSearchCV(krr_init, cv=folds, param_grid={"alpha": np.logspace(-8, 6, num=20),
#                                                    "gamma": np.logspace(-8, 6, num=20)})
# # gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e-1)
# gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e0)
# # for training set size of 100  1e-4 works, for 1000 0.7e-6 is best (really need a better convergence method)
# nn_init = MLPRegressor(max_iter=500, tol=1e-8, activation='logistic', solver='lbfgs')
# nn = GridSearchCV(nn_init, cv=folds, param_grid={'alpha': [1e-2]})

# dimer coulomb (no charges, power 1) 1e-8 for power 1 and 2, 3e-8 for power 3, 1e-7 for power 4
# 0.5e-7 for power 1 intramolecular only
# krr_init = KernelRidge(kernel='rbf')
# krr = GridSearchCV(krr_init, cv=folds, param_grid={"alpha": np.logspace(-8, 6, num=20),
#                                                    "gamma": np.logspace(-8, 6, num=20)})
# # gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e-1)
# gpr = GaussianProcessRegressor(kernel=1.0 * RBF(1.0), normalize_y=False, optimizer='fmin_l_bfgs_b', alpha=1e-8)
# # for training set size of 100  1e-4 works, for 1000 0.7e-6 is best (really need a better convergence method)
# nn_init = MLPRegressor(max_iter=500, tol=1e-8, activation='logistic', solver='lbfgs')
# nn = GridSearchCV(nn_init, cv=folds, param_grid={'alpha': [1e-2]})
