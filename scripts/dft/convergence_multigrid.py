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
filename_save = '/scratch/cahart/work/personal_files/dft_ml_md/output/fe_bulk/convergence'

# Data
supercell_221_neutral = np.array([[-7085.672045085312675, -7085.706491345680661, -7085.706491345680661],
                                  [-7085.639759701011826, -7085.648723839279228, -7085.648723839279228],
                                  [-7085.638833901497492, -7085.644499177828038, -7085.644738686124583],
                                  [-7085.632022970109574, -7085.636310290869005, -7085.636423914200350],
                                  [-7085.625529743905645, -7085.627619473984851, -7085.627636413047185],
                                  [-7085.616075780535539, -7085.620136577578705, -7085.620200865993866],
                                  [-7085.621110530991245, -7085.624441026986460, -7085.624423261647280],
                                  [-7085.619065716417936, -7085.623245292585125, -7085.623348577771139],
                                  [-7085.618176053963907, -7085.623332745365587, -7085.623485859656284],
                                  [-7085.622761324027124, -7085.628208203431313, -7085.628380450139048]
                                  ])

supercell_331_neutral = np.array([[-15942.736793974405373, -15942.764707864060256, -15942.764707864060256],
                                  [-15942.548430669477966, -15942.556279947562871, -15942.556279947562871],
                                  [-15942.524658735292178, -15942.542537479435850, -15942.543110816999615],
                                  [-15942.530507526462316, -15942.538950833941271, -15942.539259023886189],
                                  [-15942.495938787686100, -15942.501496989032603, -15942.502140823640730],
                                  [-15942.495518423482281, -15942.504455366350157, -15942.504678106732172],
                                  [-15942.506692844683130, -15942.512128201646192, -15942.512546852780360],
                                  [-15942.502010198026255, -15942.507272214154000, -15942.507638106948434],
                                  [-15942.500037614841858, -15942.509416740265806, -15942.509799754177948],
                                  [-15942.502737881324720, -15942.515524271642789, -15942.515993868108126]
                                  ])

supercell_221_force = np.array([0.0025526340,
                                0.0022814971,
                                0.0064570603,
                                0.0036374460,
                                0.0033042412,
                                0.0025500007,
                                0.0020023232,
                                0.0026416058,
                                0.0029678801,
                                0.0026892698])

supercell_221_bandgap = np.array([[2.1773231,  2.17727804, 2.17680812, 2.17663193, 2.17587209, 2.17604709, 2.17625594, 2.17581606],
                                  [2.1753509,  2.18475008, 2.17894602, 2.18146491, 2.21481895, 2.19419003, 2.21853495, 2.20675611]])


supercell_221_time = np.array([112.75385055, 114.63000268, 109.54374254, 115.05832398, 119.56667736, 122.78889185, 119.74446317, 118.90997124])

supercell_221_tight_time = np.array([150.94482037, 150.22500372, 146.86665916, 155.66668764, 151.26667075, 157.56664562, 161.43335172, 165.29999138])

supercell_221_electron_time = np.array([212.13565394, 160.78447767, 156.79123915, 193.37393018, 210.25865035, 195.02161001, 210.20000777, 264.32501972])

supercell_331_time = np.array([177.50218541, 175.8889122, 188.46668579, 186.58888951, 167.5600082, 176.09999631, 170.80833106])

supercell_331_tight_time = np.array([225.87503099, 230.44997692, 247.90002441, 254.13331016, 254.36668396, 244.33332825, 229.49999428])

supercell_331_electron_time = np.array([297.92857681, 294.02230668, 374.89565934, 289.20289764, 288.72641426, 365.23999329, 335.6332258])


# supercell_221_time = np.array([740.559021, 737.72454834, 730.93890381, 741.69885254, 741.94140625,
#                                753.70074463, 742.64111328, 744.22192383])
#
# supercell_221_electron_time = np.array([853.62438965, 795.85821533, 792.46679688, 830.82537842, 850.05053711,
#                                         831.14129639, 851.99725342, 901.1595459])
#
# supercell_221_tight_time = np.array([788.00750732, 799.72967529, 795.26098633, 807.43499756, 803.96496582,
#                                      822.97650146, 814.82897949, 822.69152832])
#
# supercell_331_time = np.array([944.14019775, 941.63311768, 960.20343018, 937.26989746, 954.25915527, 938.07659912,
#                                949.1081543,  934.8293457])
#
# supercell_331_tight_time = np.array([1027.2130127, 1021.75335693, 1068.17041016, 1073.58300781, 1071.47497559,
#                                      1080.17895508, 1051.29699707, 1014.0569458])
#
# supercell_331_electron_time = np.array([1104.6574707,  1052.50488281, 1117.71166992, 0, 1069.81494141, 1064.13110352,
#                                         1201.60498047, 1109.60583496])

supercell_221_electron = np.array([[-7085.286031858751812, -7085.291378280396202, -7085.635958369518448],
                                  [-7085.277922388087973, -7085.281935095672452, -7085.625361610032996],
                                  [-7085.269669217248520, -7085.275687717530673, -7085.617791094221502],
                                  [-7085.261997864858131, -7085.272160027562677, -7085.615661790206104],
                                  [-7085.265466493250642, -7085.271396274511972, -7085.614638303428364],
                                  [-7085.264395911094653, -7085.269398561731578, -7085.613470653074728],
                                  [-7085.264430641064791, -7085.269313395556310, -7085.612373336538440],
                                  [-7085.269463720390377, -7085.272555200966963, -7085.620598188532313]])

supercell_331_electron = np.array([[-15942.183234740017724, -15942.183746366439664],
                                  [-15942.178996652208298, -15942.184258880213747],
                                  [-15942.142375046076268, -15942.158624214891461],
                                  [-15942.152351956467101, -15942.160439513725578],
                                  [-15942.152351956467101, -15942.160439513725578],
                                  [-15942.147227587227462, -15942.154978002212374],
                                  [-15942.149200937530622, -15942.149634708597659],
                                  [-15942.155648387728434, -15942.156039868048538]])

# Rel cutoff 60
supercell_221_r60_neutral = np.array([[-7085.672080027662560, -7085.707046407898815],
                                      [-7085.639736114364496, -7085.643307087239009],
                                      [-7085.638679572397450, -7085.644606777492299],
                                      [-7085.632069259324453, -7085.636375505616343],
                                      [-7085.625557128802939, -7085.627551277755629],
                                      [-7085.616077071784275, -7085.620161705599457],
                                      [-7085.621114042137378, -7085.624439097960021],
                                      [-7085.619075363467346, -7085.622846918196046],
                                      [-7085.618399455854160, -7085.623574329304574],
                                      [-7085.622625359163067, -7085.627953700467515]
                                      ])

supercell_331_r60_neutral = np.array([[-15942.736871021661500, -15942.764485704386971],
                                       [-15942.548538267947151, -15942.556409640532365],
                                       [-15942.524260984766443, -15942.542178776937362],
                                       [-15942.530507526462316, -15942.538950833941271],
                                       [-15942.496001158113359, -15942.501561087501614],
                                       [-15942.495290628641669, -15942.504223655574606],
                                       [-15942.506656574414592, -15942.512071760749677],
                                       [-15942.502060447190161, -15942.507335647689615],
                                       [-15942.500541681649338, -15942.509953446255167],
                                       [-15942.502440592625135, -15942.515230200502629]])

# Single point from m400 optimised
supercell_221_single = np.array([-7085.645409505156749,
                                 -7085.630485659489750,
                                 -7085.622177345549972,
                                 -7085.618292708670197,
                                 -7085.617604815780396,
                                 -7085.617225042421524,
                                 -7085.616363126043325,
                                 -7085.619807074192067])

# Single point from m400 optimised
supercell_221_bonds = np.array([[1.9384616246587487, 0.008741971621989494, 2.118476308478875, 0.006529534875900799],
                                [1.9373812439963116, 0.0036732888861774103, 2.122156126513385, 0.00196123325280483],
                                [1.9392216518990546, 0.004226075369628334, 2.120285928525526, 0.006652815743220908],
                                [1.939699326461922, 0.003317558041189344, 2.121279569173847,  0.005615948425109012],
                                [1.9331640857795716, 0.00032538026580064734, 2.1307278647843355, 0.0005157879683408705],
                                [1.9345724663165622, 0.005843963051024139, 2.12899037528686, 0.009546048033432564],
                                [1.9345174455393008, 0.0021720407861454665, 2.127016658032275, 0.007042625632839846],
                                [1.9349085417870366, 3.3499717494425e-05, 2.128206698546635, 4.009803362330284e-05]
                                ])
# Single point from m400 optimised
supercell_331_bonds = np.array([[1.9376313848501647, 0.007069975113329027, 2.11624454325678, 0.00823245781827039],
                                [1.9381758656655912, 0.0032426683066656076, 2.120782660543153, 0.004241297086829618],
                                [1.9435775615063384, 2.2473036222238747e-05, 2.1143358786774846, 5.31455654125201e-06],
                                [1.9399874724581567, 0.004148150734980521, 2.120690384875138, 0.006922291257722041],
                                [1.936878648805834, 0.0007837539799496303, 2.124478086112056, 0.00033616766364439715],
                                [1.939929976400732, 0.0034479663318554366, 2.121406273738956, 0.0032140472699605047],
                                [1.9356607224282796, 0.0022583715200126322, 2.1252432121883027, 0.0072276294167367494],
                                [1.935789928886471, 0.004100371774428307, 2.1252253024393215, 0.003161225474504795]
                                ])

multigrid = np.linspace(300, 750, num=supercell_221_neutral.shape[0])
multigrid_electron = np.linspace(400, 750, num=supercell_221_electron.shape[0])

# 221 supercell band gap experimental
list_f400 = [0, 1, 2, 4, 5, 6]
list_f300 = [2, 3, 4, 6, 7, 8]
fig_221_bandgap_exp, ax_221_bandgap_exp = plt.subplots()
ax_221_bandgap_exp.plot(multigrid[2:], supercell_221_bandgap[0, :], 'k+-', label='Experimental')
ax_221_bandgap_exp.set_xlabel('Multigrid cutoff')
ax_221_bandgap_exp.set_ylabel('Band gap / eV')
ax_221_bandgap_exp.set_title('Convergence of band gap')
ax_221_bandgap_exp.legend(frameon=True)
# ax_221_bandgap_exp.set_xlim([380, 720])
fig_221_bandgap_exp.tight_layout()
# fig_221_bandgap_exp.savefig('{}/221/bandgap_exp'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# 221 supercell band gap
fig_221_bandgap, ax_221_bandgap = plt.subplots()
print(supercell_221_bandgap.shape)
ax_221_bandgap.plot(multigrid[2:], supercell_221_bandgap[0, :], 'k+-', label='Experimental')
ax_221_bandgap.plot(multigrid[2:], supercell_221_bandgap[1, :], 'b+-', label='Optimised')
ax_221_bandgap.set_xlabel('Multigrid cutoff')
ax_221_bandgap.set_ylabel('Band gap / eV')
ax_221_bandgap.set_title('Convergence of band gap')
ax_221_bandgap.legend(frameon=True)
# ax_221_bandgap.set_xlim([380, 720])
fig_221_bandgap.tight_layout()
# fig_221_bandgap.savefig('{}/221/bandgap'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# 221 supercell bond lengths 1
hematite_exp = np.array([1.94401, 2.11409])
hematite_guido = np.array([1.961, 2.137])
fig_221_bonds1, ax_221_bonds1 = plt.subplots()
ax_221_bonds1.errorbar(multigrid[list_f300], supercell_221_bonds[list_f400, 0], fmt='k+-',
                       yerr=supercell_221_bonds[list_f400, 1], label='221')
# ax_221_bonds1.errorbar(multigrid[list_f300], supercell_331_bonds[list_f400, 0], fmt='r+-',
#                        yerr=supercell_331_bonds[list_f400, 1], label='331')
ax_221_bonds1.plot([300, 800], [hematite_exp[0], hematite_exp[0]], 'k--')
ax_221_bonds1.annotate('Experimental', xy=(600, hematite_exp[0]+0.0003))
ax_221_bonds1.set_xlabel('Multigrid cutoff')
ax_221_bonds1.set_ylabel(r'Bond length / $\rm \AA$')
ax_221_bonds1.set_title('Convergence of bond lengths')
ax_221_bonds1.set_xlim([380, 720])
# ax_221_bonds1.legend(frameon=True)
fig_221_bonds1.tight_layout()
# fig_221_bonds1.savefig('{}/221/bonds1'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# 221 supercell bond lengths 2
fig_221_bonds2, ax_221_bonds2 = plt.subplots()
ax_221_bonds2.errorbar(multigrid[list_f300], supercell_221_bonds[list_f400, 2], fmt='k+-',
                       yerr=supercell_221_bonds[list_f400, 3], label='221')
# ax_221_bonds2.errorbar(multigrid[list_f300], supercell_331_bonds[list_f400, 2], fmt='r+-',
#                        yerr=supercell_331_bonds[list_f400, 3], label='331')
ax_221_bonds2.plot([300, 800], [hematite_exp[1], hematite_exp[1]], 'k--')
ax_221_bonds2.annotate('Experimental', xy=(550, hematite_exp[1]+0.0003))
# ax_221_bonds2.plot([300, 800], [hematite_guido[1], hematite_guido[1]], 'k--')
# ax_221_bonds2.annotate('Guido 221', xy=(550, hematite_guido[1]+0.0003))
ax_221_bonds2.set_xlabel('Multigrid cutoff')
ax_221_bonds2.set_ylabel(r'Bond length / $\rm \AA$')
ax_221_bonds2.set_title('Convergence of bond lengths')
ax_221_bonds2.set_xlim([380, 720])
# ax_221_bonds2.legend(frameon=True)
fig_221_bonds2.tight_layout()
# fig_221_bonds2.savefig('{}/221/bonds2'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# 221 supercell time taken
list_f400 = [0, 1, 2, 4, 5, 6]
list_f300 = [2, 3, 4, 6, 7, 8]
fig_221_t, ax_221_t = plt.subplots()
ax_221_t.plot(multigrid[list_f300], supercell_221_time[list_f400], 'k+-', label='Neutral SCF 1e-5')
ax_221_t.plot(multigrid[list_f300], supercell_221_tight_time[list_f400], 'k+--', label='Neutral')
ax_221_t.plot(multigrid[list_f300], supercell_221_electron_time[list_f400], 'b+-', label='Excess electron')
ax_221_t.set_xlabel('Multigrid cutoff')
ax_221_t.set_ylabel('Time on 16 nodes / s')
ax_221_t.set_title('221 time per SCF cycle')
ax_221_t.legend(frameon=True)
fig_221_t.tight_layout()
# fig_221_t.savefig('{}/221/time'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# 331 supercell time taken
list_f400 = [0, 1, 2, 4, 5, 6]
list_f300 = [2, 3, 4, 6, 7, 8]
fig_331_t, ax_331_t = plt.subplots()
ax_331_t.plot(multigrid[list_f300], supercell_331_time[list_f400], 'k+-', label='Neutral SCF 1e-5')
ax_331_t.plot(multigrid[list_f300], supercell_331_tight_time[list_f400], 'k+--', label='Neutral SCF')
ax_331_t.plot(multigrid[list_f300], supercell_331_electron_time[list_f400], 'b+-', label='Excess electron')
ax_331_t.set_xlabel('Multigrid cutoff')
ax_331_t.set_ylabel('Time on 24 nodes / s')
ax_331_t.set_title('331 time per SCF cycle')
ax_331_t.legend(frameon=True)
fig_331_t.tight_layout()
# fig_331_t.savefig('{}/331/time'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# 221 supercell neutral
list_f400 = [0, 1, 2, 4, 5, 6]
list_f300 = [2, 3, 4, 6, 7, 8]
fig_221, ax_221 = plt.subplots()
ax_221.plot(multigrid[list_f300], supercell_221_neutral[list_f300, 0] * parameters.hartree_to_ev, 'k+-', label='Experimental (40)')
ax_221.plot(multigrid[list_f300], supercell_221_neutral[list_f300, 1] * parameters.hartree_to_ev, 'b+-', label='Optimised (40)')
ax_221.plot(multigrid[list_f300], supercell_221_r60_neutral[list_f300, 0] * parameters.hartree_to_ev, 'k+--', label='Experimental (60)')
ax_221.plot(multigrid[list_f300], supercell_221_r60_neutral[list_f300, 1] * parameters.hartree_to_ev, 'b+--', label='Optimised (60)')
ax_221.set_xlabel('Multigrid cutoff')
ax_221.set_ylabel('Energy / eV')
ax_221.set_title('221 energy convergence')
ax_221.legend(frameon=True)
fig_221.tight_layout()
# fig_221.savefig('{}/221/neutral'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')
#
# # 221 supercell neutral forces
fig_221_f, ax_221_f = plt.subplots()
ax_221_f.plot(multigrid[list_f300], supercell_221_force[list_f300], 'k+-')
ax_221_f.set_xlabel('Multigrid cutoff')
ax_221_f.set_ylabel('Energy / eV')
ax_221_f.set_title('221 neutral forces')
ax_221_f.legend(frameon=True)
fig_221_f.tight_layout()
# fig_221_f.savefig('{}/221/neutral_forces'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')
#
# # 221 supercell neutral difference
# # fig_221_diff, ax_221_diff = plt.subplots()
# # ax_221_diff.plot(multigrid[list_f300], (supercell_221_neutral[list_f300, 0] - supercell_221_neutral[list_f300, 1]) * parameters.hartree_to_ev,
# #                  'k+-')
# # ax_221_diff.plot(multigrid[list_f300], (supercell_221_r60_neutral[list_f300, 0] - supercell_221_r60_neutral[list_f300, 1]) * parameters.hartree_to_ev,
# #                  'k+--')
# # ax_221_diff.set_xlabel('Multigrid cutoff')
# # ax_221_diff.set_ylabel('Energy / eV')
# # ax_221_diff.set_title('221 exp - opt convergence')
# # fig_221_diff.tight_layout()
# # fig_221_diff.savefig('{}/221/neutral_diff'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')
#
# # 221 supercell single point neutral
# # fig_221_single, ax_221_single = plt.subplots()
# # ax_221_single.plot(multigrid_electron[list_f400], supercell_221_neutral[list_f300, 0] * parameters.hartree_to_ev, 'b+-', label='Experimental')
# #
# # ax_221_single.plot(multigrid_electron[list_f400], supercell_221_single[list_f400] * parameters.hartree_to_ev, 'r+-', label='Single point from m400')
# # ax_221_single.set_xlabel('Multigrid cutoff')
# # ax_221_single.set_ylabel('Energy / eV')
# # ax_221_single.set_title('221 energy convergence')
# # ax_221_single.legend(frameon=True)
# # fig_221_single.tight_layout()
# # fig_221_single.savefig('{}/221/neutral_single point'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')
#
# # 221 supercell electron
# fig_221_e, ax_221_e = plt.subplots()
# ax_221_e.plot(multigrid_electron[list_f400], supercell_221_electron[list_f400, 0] * parameters.hartree_to_ev, 'k+-', label='Vertical')
# ax_221_e.plot(multigrid_electron[list_f400], supercell_221_electron[list_f400, 1] * parameters.hartree_to_ev, 'b+-', label='Relaxed')
# ax_221_e.set_xlabel('Multigrid cutoff')
# ax_221_e.set_ylabel('Energy / eV')
# ax_221_e.set_title('221 electron energy convergence')
# ax_221_e.legend(frameon=True)
# fig_221_e.tight_layout()
# fig_221_e.savefig('{}/221/electron'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')
#
# # 221 supercell electron lambda
# fig_221_e_lambda, ax_221_e_lambda = plt.subplots()
# ax_221_e_lambda.plot(multigrid_electron[list_f400], (supercell_221_electron[list_f400, 0]-supercell_221_electron[list_f400, 1])
#                      * parameters.hartree_to_ev, 'r+-', label='Charged lambda')
# ax_221_e_lambda.plot(multigrid_electron[list_f400], (supercell_221_electron[list_f400, 2]-supercell_221_neutral[list_f300, 1])
#                      * parameters.hartree_to_ev, 'b+-', label='Neutral lambda')
# ax_221_e_lambda.plot(multigrid_electron[list_f400], ((supercell_221_electron[list_f400, 2]-supercell_221_neutral[list_f300, 1]) +
#                                           (supercell_221_electron[list_f400, 0]-supercell_221_electron[list_f400, 1])) / 2
#                      * parameters.hartree_to_ev, 'k+-', label='Reorganisation energy')
# ax_221_e_lambda.set_xlabel('Multigrid cutoff')
# ax_221_e_lambda.set_ylabel('Energy / eV')
# ax_221_e_lambda.set_title('221 electron lambda convergence')
# ax_221_e.legend(frameon=True)
# fig_221_e_lambda.tight_layout()
# fig_221_e_lambda.savefig('{}/221/electron_lambda'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')
#
# # 221 supercell electron affinity
# fig_221_e_ea, ax_221_e_ea = plt.subplots()
# ax_221_e_ea.plot(multigrid_electron[list_f400], (supercell_221_electron[list_f400, 0]-supercell_221_neutral[list_f300, 1])
#                  * parameters.hartree_to_ev, 'k+-', label='Vertical')
# ax_221_e_ea.set_xlabel('Multigrid cutoff')
# ax_221_e_ea.set_ylabel('Energy / eV')
# ax_221_e_ea.set_title('221 EA convergence (electron)')
# fig_221_e_ea.tight_layout()
# fig_221_e_ea.savefig('{}/221/electron_ea'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')
#
# # # 331 supercell neutral
# fig_331, ax_331 = plt.subplots()
# ax_331.plot(multigrid[list_f300], supercell_331_neutral[list_f300, 0] * parameters.hartree_to_ev, 'k+-', label='Experimental (40)')
# ax_331.plot(multigrid[list_f300], supercell_331_neutral[list_f300, 1] * parameters.hartree_to_ev, 'b+-', label='Optimised (40)')
# ax_331.plot(multigrid[list_f300], supercell_331_r60_neutral[list_f300, 0] * parameters.hartree_to_ev, 'k+--', label='Experimental (60)')
# ax_331.plot(multigrid[list_f300], supercell_331_r60_neutral[list_f300, 1] * parameters.hartree_to_ev, 'b+--', label='Optimised (60)')
# ax_331.set_xlabel('Multigrid cutoff')
# ax_331.set_ylabel('Energy / eV')
# ax_331.set_title('331 energy convergence')
# ax_331.legend(frameon=True)
# fig_331.tight_layout()
# fig_331.savefig('{}/331/neutral'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')
#
# # 331 supercell neutral difference
# # fig_331_diff, ax_331_diff = plt.subplots()
# # ax_331_diff.plot(multigrid, (supercell_331_neutral[:, 0] - supercell_331_neutral[:, 1]) * parameters.hartree_to_ev,
# #                  'k+-')
# # ax_331_diff.plot(multigrid, (supercell_331_r60_neutral[:, 0] - supercell_331_r60_neutral[:, 1]) * parameters.hartree_to_ev,
# #                  'k+--')
# # ax_331_diff.set_xlabel('Multigrid cutoff')
# # ax_331_diff.set_ylabel('Energy / eV')
# # ax_331_diff.set_title('331 exp - opt convergence')
# # fig_331_diff.tight_layout()
# # fig_331_diff.savefig('{}/331/neutral_diff'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')
#
# # 221, 331 supercell neutral difference
# # fig_diff, ax_diff = plt.subplots()
# # ax_diff.plot(multigrid, supercell_331_neutral[:, 1] - supercell_221_neutral[:, 1] * parameters.hartree_to_ev, 'k+-')
# # ax_diff.plot(multigrid, supercell_331_r60_neutral[:, 1] - supercell_221_r60_neutral[:, 1] * parameters.hartree_to_ev, 'k+--')
# # ax_diff.set_xlabel('Multigrid cutoff')
# # ax_diff.set_ylabel('Energy / eV')
# # ax_diff.set_title('331 - 221 energy convergence')
# # fig_diff.tight_layout()
# # fig_diff.savefig('{}/221/221_331_neutral_diff'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')
#
# # 331 supercell electron
# fig_331_e, ax_331_e = plt.subplots()
# ax_331_e.plot(multigrid_electron[list_f400], supercell_331_electron[list_f400, 0] * parameters.hartree_to_ev, 'k+-', label='Vertical')
# ax_331_e.set_xlabel('Multigrid cutoff')
# ax_331_e.set_ylabel('Energy / eV')
# ax_331_e.set_title('331 electron energy convergence')
# ax_331_e.legend(frameon=True)
# fig_331_e.tight_layout()
# fig_331_e.savefig('{}/331/electron'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')
#
# # 331 supercell electron lambda
# fig_331_e_ea, ax_331_e_ea = plt.subplots()
# ax_331_e_ea.plot(multigrid_electron[list_f400], (supercell_331_electron[list_f400, 0]-supercell_331_neutral[list_f300, 1])
#                  * parameters.hartree_to_ev, 'k+-')
# ax_331_e_ea.set_xlabel('Multigrid cutoff')
# ax_331_e_ea.set_ylabel('Energy / eV')
# ax_331_e_ea.set_title('331 EA convergence (electron)')
# fig_331_e_ea.tight_layout()
# fig_331_e_ea.savefig('{}/331/electron_ea'.format(filename_save), dpi=parameters.save_dpi, bbbox_inches='tight')

# Time per geo_opt step


if __name__ == "__main__":
    print('Finished.')
    plt.show()
