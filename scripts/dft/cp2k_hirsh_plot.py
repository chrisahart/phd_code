from __future__ import division, print_function
import time
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

""" 
Plot Hirshfeld atomic density

"""


def gaussian(r_grid, z, a):
    """ Return normalised radial component of wavefunction R_nl(r)"""

    return a * np.exp(- z * (r_grid ** 2))


# Variables
folder = 'E:/University/PhD/Programming/dft_ml_md/output/cdft/hw_forces/atom_fit/C'
cp2k_density = np.loadtxt('{}/density.out'.format(folder))
cp2k_grid = np.loadtxt('{}/grid.out'.format(folder))
zeta = np.loadtxt('{}/zeta.out'.format(folder))
coef = np.loadtxt('{}/coef.out'.format(folder))
N = cp2k_grid.shape[0]
r_end = 5
r_grid = np.copy(cp2k_grid)

#  Plot each Gaussian
fig_gaussian, ax_gaussian = plt.subplots()
gto = np.ones((zeta.shape[0], N))
for i in range(0, zeta.shape[0]):
    gto[i, :] = gaussian(r_grid, zeta[i], coef[i])
    ax_gaussian.plot(r_grid, gto[i, :], '-', label=i)
ax_gaussian.set_xlabel('r')
ax_gaussian.set_ylabel('')
# ax_gaussian.set_title('Fitting atomic density')
ax_gaussian.legend(frameon=True)
ax_gaussian.set_xlim([0, r_end])
fig_gaussian.tight_layout()
fig_gaussian.savefig('{}{}'.format(folder, '/gaussians.png'), dpi=200, bbbox_inches='tight')

# Calculate
gto_norm = np.sum(gto, axis=0)/np.sum(np.sum(gto, axis=0))
cp2k_density_norm = cp2k_density/np.sum(cp2k_density)

#  Plot sum of Gaussians
fig_radial_sum, ax_radial_sum = plt.subplots()
ax_radial_sum.plot(r_grid, np.sum(gto, axis=0), 'k-', label='Sum')
# ax_radial_sum.plot(cp2k_grid, cp2k_density/cp2k_density[-1], 'k-', label='Sum')
ax_radial_sum.set_xlabel('r')
ax_radial_sum.set_ylabel('')
# ax_radial_sum.set_title('Summed atomic density')
# ax_radial_sum.legend(frameon=True)
ax_radial_sum.set_xlim([0, r_end])
fig_radial_sum.tight_layout()
fig_radial_sum.savefig('{}{}'.format(folder, '/gaussians_sum.png'), dpi=200, bbbox_inches='tight')

#  Plot CP2K density
fig_cp2k, ax_cp2k = plt.subplots()
ax_cp2k.plot(cp2k_grid, cp2k_density, 'k-', label='Sum')
ax_cp2k.set_xlabel('r')
ax_cp2k.set_ylabel('')
# ax_cp2k.set_title('Atomic density')
# ax_cp2k.legend(frameon=True)
ax_cp2k.set_xlim([0, r_end])
fig_cp2k.tight_layout()
fig_cp2k.savefig('{}{}'.format(folder, '/density.png'), dpi=200, bbbox_inches='tight')

#  sum of Gaussians against CP2K density
fig_radial_compare, ax_radial_compare = plt.subplots()
ax_radial_compare.plot(r_grid, np.sum(gto, axis=0)/np.sum(np.sum(gto, axis=0)), 'r-', label='Fitted')
ax_radial_compare.plot(cp2k_grid, cp2k_density/np.sum(cp2k_density), 'g-', label='CP2K')
ax_radial_compare.set_xlabel('r')
ax_radial_compare.set_ylabel('')
# ax_radial_compare.set_title('Fitting atomic density')
ax_radial_compare.legend(frameon=True)
ax_radial_compare.set_xlim([0, r_end])
fig_radial_compare.tight_layout()
fig_radial_compare.savefig('{}{}'.format(folder, '/normalised_comparison.png'), dpi=200, bbbox_inches='tight')

print('Average fitting error:', np.mean(gto_norm - cp2k_density_norm))
print('Sum fitting error:', np.sum(gto_norm - cp2k_density_norm))

print('coef: \n', coef)
print('zeta: \n', zeta)


if __name__ == "__main__":
    plt.show()
