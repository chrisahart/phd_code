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

# Files
filename_save = '/scratch/cahart/work/personal_files/dft_ml_md/output/feIV_bulk/convergence/preliminary'

# exp_vesta_neutral_cubes_s4m500,
supercell_221_neutral = np.array([[-7085.644606634845331, -7085.645402533588822],
                                  [-7085.623364993864016, -7085.628627945527114],
                                  [-7085.619604201035145, -7085.624369181680777]])

supercell_221_hole = np.array([[-7085.926980369409648, -7085.931600850739414, -7085.628400755995244],
                                  [-7085.910383610077588, -7085.918549931996495, -7085.610797439412636],
                                  [-7085.905663951793940, -7085.912710354124101, -7085.606786661145634]])

supercell_221_electron = np.array([[-7085.287203167203188, -7085.290988088857375, -7085.635330753179915],
                                  [-7085.270068629676643, -7085.275519984501443, -7085.618387961229928],
                                  [-7085.265316327051551, -7085.271397156385319, -7085.614958382081568]])

multigrid = np.array([400, 500, 600])

# 221 supercell neutral
fig_221_neutral, ax_221_neutral = plt.subplots()
ax_221_neutral.plot(multigrid, supercell_221_neutral[:, 1] * parameters.hartree_to_ev, 'k+-', label='Neutral')
ax_221_neutral.plot(multigrid, supercell_221_hole[:, 2] * parameters.hartree_to_ev, 'b+-', label='Vertical (hole)')
ax_221_neutral.plot(multigrid, supercell_221_electron[:, 2] * parameters.hartree_to_ev, 'r+-', label='Vertical (electron)')
ax_221_neutral.set_xlabel('Multigrid cutoff')
ax_221_neutral.set_ylabel('Energy / eV')
ax_221_neutral.set_title('221 neutral energy convergence')
ax_221_neutral.legend(frameon=True)
fig_221_neutral.tight_layout()
fig_221_neutral.savefig('{}/221_neutral'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# 221 supercell hole
fig_221_hole, ax_221_hole = plt.subplots()
ax_221_hole.plot(multigrid, supercell_221_hole[:, 0] * parameters.hartree_to_ev, 'k+-', label='Vertical')
ax_221_hole.plot(multigrid, supercell_221_hole[:, 1] * parameters.hartree_to_ev, 'b+-', label='Relaxed')
ax_221_hole.set_xlabel('Multigrid cutoff')
ax_221_hole.set_ylabel('Energy / eV')
ax_221_hole.set_title('221 hole energy convergence')
ax_221_hole.legend(frameon=True)
fig_221_hole.tight_layout()
fig_221_hole.savefig('{}/221_hole'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# 221 supercell hole lambda
fig_221_hole_lambda, ax_221_hole_lambda = plt.subplots()
ax_221_hole_lambda.plot(multigrid, (supercell_221_hole[:, 1]- supercell_221_hole[:, 0])
                        * parameters.hartree_to_ev, 'r+-', label='Charged lambda')
ax_221_hole_lambda.plot(multigrid, (supercell_221_neutral[:, 1]- supercell_221_hole[:, 2])
                        * parameters.hartree_to_ev, 'b+-', label='Neutral lambda')
ax_221_hole_lambda.plot(multigrid, ((supercell_221_neutral[:, 1]- supercell_221_hole[:, 2]) +
                                        (supercell_221_hole[:, 1]- supercell_221_hole[:, 0])) / 2
                            * parameters.hartree_to_ev, 'k+-', label='Reorganisation energy')
ax_221_hole_lambda.set_xlabel('Multigrid cutoff')
ax_221_hole_lambda.set_ylabel('Energy / eV')
ax_221_hole_lambda.set_title('221 hole lambda convergence')
ax_221_hole_lambda.legend(frameon=True)
fig_221_hole_lambda.tight_layout()
fig_221_hole_lambda.savefig('{}/221_hole_lambda'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# 221 supercell electron
fig_221_electron, ax_221_electron = plt.subplots()
ax_221_electron.plot(multigrid, supercell_221_electron[:, 0] * parameters.hartree_to_ev, 'k+-', label='Vertical')
ax_221_electron.plot(multigrid, supercell_221_electron[:, 1] * parameters.hartree_to_ev, 'b+-', label='Relaxed')
ax_221_electron.set_xlabel('Multigrid cutoff')
ax_221_electron.set_ylabel('Energy / eV')
ax_221_electron.set_title('221 electron energy convergence')
ax_221_electron.legend(frameon=True)
fig_221_electron.tight_layout()
fig_221_electron.savefig('{}/221_electron'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# 221 supercell electron lambda
fig_221_electron_lambda, ax_221_electron_lambda = plt.subplots()
ax_221_electron_lambda.plot(multigrid, (supercell_221_electron[:, 1]- supercell_221_electron[:, 0])
                        * parameters.hartree_to_ev, 'r+-', label='Charged lambda')
ax_221_electron_lambda.plot(multigrid, (supercell_221_neutral[:, 1]- supercell_221_electron[:, 2])
                        * parameters.hartree_to_ev, 'b+-', label='Neutral lambda')
ax_221_electron_lambda.plot(multigrid, ((supercell_221_neutral[:, 1]- supercell_221_electron[:, 2]) +
                                        (supercell_221_electron[:, 1]- supercell_221_electron[:, 0])) / 2
                            * parameters.hartree_to_ev, 'k+-', label='Reorganisation energy')
ax_221_electron_lambda.set_xlabel('Multigrid cutoff')
ax_221_electron_lambda.set_ylabel('Energy / eV')
ax_221_electron_lambda.set_title('221 electron lambda convergence')
ax_221_electron_lambda.legend(frameon=True)
fig_221_electron_lambda.tight_layout()
fig_221_electron_lambda.savefig('{}/221_electron_lambda'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# Energy difference, vertical - neutral
fig_221_hole_diff, ax_221_hole_diff = plt.subplots()
ax_221_hole_diff.plot(multigrid, (supercell_221_hole[:, 0]- supercell_221_neutral[:, 1]) * parameters.hartree_to_ev, 'r+-', label='IP')
ax_221_hole_diff.set_xlabel('Multigrid cutoff')
ax_221_hole_diff.set_ylabel('Energy / eV')
ax_221_hole_diff.set_title('IP convergence (hole)')
# ax_221_hole_diff.legend(frameon=True)
fig_221_hole_diff.tight_layout()
fig_221_hole_diff.savefig('{}/221_ip_hole'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# Energy difference, vertical - neutral
fig_221_electron_diff, ax_221_electron_diff = plt.subplots()
ax_221_electron_diff.plot(multigrid, (supercell_221_electron[:, 0]- supercell_221_neutral[:, 1]) * parameters.hartree_to_ev, 'r+-', label='EA')
ax_221_electron_diff.set_xlabel('Multigrid cutoff')
ax_221_electron_diff.set_ylabel('Energy / eV')
ax_221_electron_diff.set_title('EA convergence (electron)')
# ax_221_electron_diff.legend(frameon=True)
fig_221_electron_diff.tight_layout()
fig_221_electron_diff.savefig('{}/221_ea_hole'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')

if __name__ == "__main__":
    print('Finished.')
    plt.show()