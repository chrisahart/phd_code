import re
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import math
from math import factorial
from scipy.optimize import curve_fit
from scipy import stats
from collections import OrderedDict

def calculate_correction_barrier(reorganization, free_energy, couplings):
    if couplings > reorganization/2:
        return reorganization/4
    else:
        if free_energy == 0.0:
            return couplings - couplings**2/reorganization
        else:
            print "Free energy != O not implemented"
            raise SystemExit

def calculate_factor(coupling, reorganization, free_energy, temperature, frequency):
    exposent = (np.pi**1.5 * coupling**2) / ( 2*np.pi * np.sqrt(reorganization*temperature) * frequency)
    Plz = 1 - np.exp(-exposent)
    #print "Hab", coupling
    #print "exposent", exposent
    #print "plz", Plz
    #print 2*Plz / (1 + Plz)
    if free_energy >= - reorganization:
        return 2*Plz / (1 + Plz)

def calculate_rate(coupling, reorganization, free_energy, temperature, method, frequency = 1):
    #Convert everything in atomic units
    coupling = coupling  / 27211.399
    reorganization = reorganization  / 27211.399
    free_energy = free_energy  /     27211.399
    temperature = temperature / 315777.09
    frequency = frequency * 0.0000046 / (2*np.pi)

    barrier = np.square( ( reorganization + free_energy ) ) / ( 4 * reorganization)
    if method == 'NA':
        factor = (2 * np.pi) * np.square(coupling) / np.sqrt( 4 * np.pi * reorganization * temperature)
    elif method == 'AD':
        barrier += - calculate_correction_barrier(reorganization, free_energy, coupling)
        factor = frequency
    elif 'TOT' in method:
        barrier += - calculate_correction_barrier(reorganization, free_energy, coupling)
        factor = frequency * calculate_factor(coupling, reorganization, free_energy, temperature, frequency)
        #print frequency
        #print factor
    else:
        print "method should be NA or AD"
   
    if barrier < 0:
        barrier = 0.0
    #print barrier
    expo = np.exp( - barrier / temperature)
    #print "expo", expo
    rate = expo*factor / 0.024188  # 1au = 0.02418884 fs
    # rate = 0.0189
    return rate # in fs-1

import math
import numpy as np


def calculate_rate_from_spectral_overlap(coupling, spectral_overlap):
    """
    formula: K = 2\pi/\bar *V^2 *J
    Inputs:
        - Coupling [meV]
        - spectral_overlap [eV^-1]
     """
    hbar  = 6.58211928E-16 # eV*s
    
    #Convert everything in eV
    coupling = coupling*1e-3
    
    rate_s   = 2*math.pi*(spectral_overlap*coupling**2)/hbar # s^-1
    rate = rate_s*1e-15 # convert in fs-1
    return rate


def calculate_rate_MLJ(coupling, lambS, lambI, free_energy, W0, temperature):
    """ This calculated the quantized Marcus-Levitch-Jortner rate according to Cupellini 2017 paper
        Inputs:
          - Coupling [meV]
          - lambS [meV] : Classical reorganization energy (called also solvent reorganization)
          - lambI [meV]: Quantum reorganization energy (that could be derived from the 4 point scheme)
          - free_energy [meV]
          - W0 [s-1]: which is the angular frequency of the quantized mode
          - temperature [K]

        Outputs:
          - rate in s-1

    """

    hbar  = 6.58211928E-16 # eV*s
    kboltz = 8.617333262145E-5  # eV

    #Convert everything in eV
    coupling = coupling*1e-3
    lambS = lambS*1e-3
    lambI = lambI*1e-3
    free_energy = free_energy*1e-3
    
    # Huang-Rhys factor for the quantum mode
    S = lambI/(hbar*W0)
    KT = temperature*kboltz

    FCS = 0.0
    control = 0.0
    j = 0
    while True:
        FCS += math.exp(-S)*S**j/(math.factorial(j))*math.exp(-(free_energy +lambS+j*hbar*W0)**2/(4*lambS*KT))
        control += math.exp(-S)*S**j/(math.factorial(j))
        #print j,control,FCS
        if 1.0 - control < 1e-14: break
        j += 1
    
    
    J = math.sqrt(1/(4*math.pi*lambS*KT))*FCS
    rate_s   = 2*math.pi*(J*coupling**2)/hbar # s^-1
    #print 'k(M.L.J.) =', rate_s, 's^-1'
    
    rate = rate_s*1e-15 # convert in fs-1
    return rate


def get_eigen(matrix):
    # we use np.linalg.eigh to deal with simmetric matrix
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    idx = eigenvalues.argsort()
    eigenvalues = np.array(eigenvalues[idx])
    eigenvectors = np.array(eigenvectors[:, idx])
    #print "all_eigen", eigenvectors
    return eigenvalues, eigenvectors # eigenvectors[0, :], eigenvectors[:, 0], eigenvectors[:, 1]

def time_evolution(n, Kinetic_matrix, tstep, nstep, starting_pos):
    #find eigenvalues and eigenvectors
    L, U = get_eigen(Kinetic_matrix)
    print "DIAGONALIZATION FINISHED"

    #Define initial condition
    P0 = np.zeros(n)
    P0[starting_pos] = 1.0
    print('starting_pos')
    print(starting_pos)
    
    #Define time step serie
    t = np.arange(nstep+1)*tstep

    Uinv = np.linalg.inv(U)

    # Find y(0)
    y0 = np.dot(Uinv,P0)
    # Solving P(t) = P(O)*U*exp(Lt)*U-1
    exps = np.exp(np.einsum('i,j->ij',L,t))
    Pt  = np.dot(U,np.einsum('ij,i->ij',exps,y0))
    return t, Pt

def read_coordinates(filename):
    com = []
    with open(filename, "r") as f:
        for line in f.readlines():
            com.append([float(i) for i in line.split()])
    return com
    
def read_connect(filename):
    f = open(filename, "r")
    file_ = f.readlines()
    natom = int(file_[0])
    connectivity_list = []
    for line in file_[1:]:
        list_line = [int(i) for i in line.split()]
        connectivity_list.append(list_line)
    return natom, connectivity_list

def isclose(a, b, rel_tol=1e-09, abs_tol=0.1):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def check_connect_from_distance(msd_length, com):
    connectivity = []
    #start looping over one mol
    count1 = 0
    for point1 in com:
        count1 += 1
        count2 = count1
        for point2 in com[count1:]:
            count2 += 1
            distance = abs(np.array(point2)-np.array(point1))
            distance =  np.linalg.norm(distance)
            # loop over list of interaction to do the check: if there are two equivalent interactions we need to do something else
            for iter_3, check_dist in enumerate(msd_length):
                diff = isclose(distance, check_dist)
                if diff:
                    # connectivity.append([count1, count2, iter_3 + 1])

                    # Get largest y value atom
                    if np.maximum(point1[1], point2[1]) in point1:
                        largest = point1
                        smallest = point2
                    else:
                        largest = point2
                        smallest = point1

                    # Calculate angle with x and y axis
                    angle_x = arctan((largest[1] - smallest[1]) / ((largest[0] - smallest[0])+1e-6)) * 180 / np.pi
                    angle_y = arctan((largest[0] - smallest[0]) / ((largest[1] - smallest[1])+1e-6)) * 180 / np.pi

                    # Second nearest neighbour
                    # if isclose(distance, msd_length[0]):
                    #
                    #     # +a if 0 degrees to x axis and 90 degrees to y axis
                    #     if abs(np.round(angle_x)) == 0 and abs(np.round(angle_y)) == 90:
                    #         # pass
                    #         connectivity.append([count1, count2, 1])
                    #
                    #     # +b if 60 degrees to x axis and 30 degrees to y axis
                    #     elif np.round(angle_x) == 60 and np.round(angle_y) == 30:
                    #         # pass
                    #         connectivity.append([count1, count2, 2])
                    #
                    #     # If -60 to x axis and -30 to y axis
                    #     elif np.round(angle_x) == -60 and np.round(angle_y) == -30:
                    #         # pass
                    #         connectivity.append([count1, count2, 3])
                    #
                    #     else:
                    #         print('Found atom not included in classification for msd_length[1]')
                    #         print(angle_x)
                    #         print(angle_y)

                    # First nearest neighbour
                    if isclose(distance, msd_length[0]):

                        # +a if -30 degrees to x axis and -60 degrees to y axis
                        if np.round(angle_x) == -30 and np.round(angle_y) == -60:
                            connectivity.append([count1, count2, 1])

                        # -a if 30 degrees to x axis and 60 degrees to y axis
                        elif np.round(angle_x) == 30 and np.round(angle_y) == 60:
                            connectivity.append([count1, count2, 2])

                        # +ab if 0 degrees to x axis and 90 degrees to y axis
                        elif abs(np.round(angle_x)) == 90 and abs(np.round(angle_y)) == 0:
                            connectivity.append([count1, count2, 3])
                        else:
                            print('Found atom not included in classification for msd_length[0]')
                            print(angle_x)
                            print(angle_y)

                    # Second nearest neighbour
                    # elif isclose(distance, msd_length[1]):
                    #
                    #     # +a if 0 degrees to x axis and 90 degrees to y axis
                    #     if abs(np.round(angle_x)) == 0 and abs(np.round(angle_y)) == 90:
                    #         # pass
                    #         connectivity.append([count1, count2, 4])
                    #
                    #     # +b if 60 degrees to x axis and 30 degrees to y axis
                    #     elif np.round(angle_x) == 60 and np.round(angle_y) == 30:
                    #         # pass
                    #         connectivity.append([count1, count2, 5])
                    #
                    #     # If -60 to x axis and -30 to y axis
                    #     elif np.round(angle_x) == -60 and np.round(angle_y) == -30:
                    #         # pass
                    #         connectivity.append([count1, count2, 6])
                    #
                    #     else:
                    #         print('Found atom not included in classification for msd_length[1]')
                    #         print(angle_x)
                    #         print(angle_y)
                    #
                    # # Third nearest neighbour
                    # elif isclose(distance, msd_length[2]):
                    #
                    #     # +a if -30 degrees to x axis and -60 degrees to y axis
                    #     if np.round(angle_x) == 30 and np.round(angle_y) == 60:
                    #         connectivity.append([count1, count2, 7])
                    #
                    #     # -a if 30 degrees to x axis and 60 degrees to y axis
                    #     elif np.round(angle_x) == -30 and np.round(angle_y) == -60:
                    #         connectivity.append([count1, count2, 8])
                    #
                    #     # +ab if 0 degrees to x axis and 90 degrees to y axis
                    #     elif abs(np.round(angle_x)) == 90 and abs(np.round(angle_y)) == 0:
                    #         connectivity.append([count1, count2, 9])
                    #
                    #     else:
                    #         print('Found atom not included in classification for msd_length[2]')
                    #         print(angle_x)
                    #         print(angle_y)

    return connectivity


def build_kinetic_matrix(natom, connectivity_list, interaction_dict):
    #build kinetic matrix able to deal with both PBC and no PBC depending if their are present or not in connectivity.dat 
    k_matrix = np.zeros((natom,natom))
    
    #loop over the connectivity instead of a double loop over the kinetic matrix is faster
    for il in connectivity_list:
        #the interactions are 1-based but python counts from zero
        k = il[0]-1
        l = il[1]-1
        type_inter = str(il[2])
        mean = interaction_dict[type_inter]
        # deal with PBC in connectivity file
        if k_matrix[k,l] == 0.0:
            k_matrix[k,l] = mean
            k_matrix[l,k] = k_matrix[k,l]
        else:
            k_matrix[k,l] += mean
            k_matrix[l,k] = k_matrix[k,l]

    # construct the diagonal
    for diag in range(natom):
        # sum the column elements to keep detailed balance
        k_matrix[diag,diag] = -sum(k_matrix[diag,:])
    #print "\n"
    #print "Kmatrix \n"
    #print('\n'.join([''.join(['{:20}'.format(item) for item in row]) 
    #  for row in k_matrix]))
    return k_matrix


#def build_kinetic_matrix_NOPBC(natom, connectivity_list, interaction_dict):
#    #build kinetic matrix no PBC 
#    k_matrix = np.zeros((natom,natom))
#    
#    #loop over the connectivity instead of a double loop over the kinetic matrix is faster
#    for il in connectivity_list:
#        #the interactions are 1-based but python counts from zero
#        k = il[0]-1
#        l = il[1]-1
#        type_inter = str(il[2])
#        mean = interaction_dict[type_inter]
#    
#        k_matrix[k,l] = mean
#        k_matrix[l,k] = k_matrix[k,l]
#    
#    # construct the diagonal
#    for diag in range(natom):
#        # sum the column elements to keep detailed balance
#        k_matrix[diag,diag] = -sum(k_matrix[diag,:])
#    #print "Kmatrix", k_matrix
#    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
#      for row in k_matrix]))
#    return k_matrix

#<L^2> = \sum_i Pi*(iL)^2 #correct expression
def _calc_3D_msd3(Pt, com_chain):
    populations = Pt.T
    com = np.array(com_chain)
    results = []
    vect0 = np.array([0.0, 0.0, 0.0])
    for (x,y) in zip(populations[0], com):
        vect0 +=  x *y

    for serie in populations:
        msd = np.zeros(9)
        for alpha in range(3):
            for beta in range(3):
                for (x, y) in zip(serie, com):
                    index = beta * 3 + alpha
                    msd[index] += x * (y[alpha] - vect0[alpha]) * (y[beta] - vect0[beta])
        results.append(list(msd))
    return results

#check if the matrix is symmetric and use this info for the diagonalization
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def calc_mobility_tensor(mobility_array):
    """"
    Calculate mobility tensor to get mobility along the eigendirection of system (Chris Ahart code)
    """

    eig_vals, eig_vectors = np.linalg.eig(mobility_array[0:2,0:2])
    eig_vals2, eig_vectors2 = np.linalg.eig(mobility_array[0:2,0:2])
    norm = np.linalg.norm(mobility_array[0:2,0:2])

    print('2x2')
    print('mobility_array')
    print(mobility_array[0:2,0:2])
    print('\neig_vectors (np.linalg.eig)')
    print(eig_vectors)
    print('eig_vals (np.linalg.eig)')
    print(eig_vals)
    print('norm (np.linalg.norm)')
    print(norm)

    # print('\neig_vectors (np.linalg.eigh)')
    # print(eig_vectors2)
    # print('eig_vals (np.linalg.eigh)')
    # print(eig_vals2)

    # print('\nmobility_array\n', mobility_array)
    # print('np.diag\n', np.diag(mobility_array))
    # print('np.linalg.eig\n', np.linalg.eig(mobility_array))
    # print('np.linalg.norm\n', np.linalg.norm(mobility_array))
    # return results

############################   INPUT   ######################################################################

if __name__ == '__main__':

    """This code can solve the Master equations for a charge travelling in a given plane or chain depending 
      on the topology. The available rates are Non-adiabatic Marcus rate, Adiabatic Marcus rate (see Blumberger 2015) """



    # general_path = "./"
    general_path = 'E:/University/PhD/Programming/dft_ml_md/output/fe_bulk/hematite/mobility/2d_plane/'
    
    select_list_dir_2_plot = {
        # name of the system  
        'RUB_AOM':{

            # 1D chain (+ab)
            # 'npairs': 2,
            # 'Coupling': [120.6, 120.6],
            # 'msd_length': [2.97, 2.97],
            # 'Reorg energy': [807, 807],
            # 'free_energy_bias': np.zeros(2),
            # 'frequency': 616.667 * np.ones(2),

            # 2D chain (+ab)
            # 'npairs': 2,
            # 'Coupling': [120.6, 120.6],
            # 'msd_length': [2.97, 2.97],
            # 'Reorg energy': [807, 807],
            # 'free_energy_bias': np.zeros(2),
            # 'frequency': 616.667 * np.ones(2),

            # 2D plane (+ab, +ab, 0)
            # 'npairs': 3,
            # 'Coupling': [120.6, 120.6, 0.1],
            # 'msd_length': [2.97, 2.97, 2.97],
            # 'Reorg energy': [807, 807, 10000],
            # 'free_energy_bias': np.zeros(3),
            # 'frequency': 616.667 * np.ones(3),

            # 2D plane (+ab, +ab, +ab)
            # 'npairs': 3,
            # 'Coupling': [120.6, 120.6, 120.6],
            # 'msd_length': [2.97, 2.97, 2.97],
            # 'Reorg energy': [807, 807, 807],
            # 'free_energy_bias': np.zeros(3),
            # 'frequency': 616.667 * np.ones(3),

            # 2D plane (+a, -a, 0)
            # 'npairs': 3,
            # 'Coupling': [40.6, 53.6, 0.1],
            # 'msd_length': [2.97, 2.97, 2.97],
            # 'Reorg energy': [883, 873, 10000],
            # 'free_energy_bias': np.zeros(3),
            # 'frequency': 616.667 * np.ones(3),

            # 2D plane (nn-1)
            # 'npairs': 3,
            # 'Coupling': [40.6, 53.6, 120.6],
            # 'msd_length': [2.97, 2.97, 2.97],
            # 'Reorg energy': [241, 258, 318],
            # 'Reorg energy': [883, 873, 807],
            # 'free_energy_bias': np.zeros(3),
            # 'frequency': 616.667*np.ones(3),

            # 2D plane (nn-1) new
            # 'npairs': 3,
            # 'Coupling': [39, 53, 203],
            # 'msd_length': [2.97, 2.97, 2.97],
            # 'Reorg energy': [881, 865, 652],
            # 'free_energy_bias': np.zeros(3),
            # 'frequency': 616.667 * np.ones(3),

            # 2D plane (nn-1, 2) new
            # 'npairs': 6,
            # 'Coupling': np.array([39, 53, 203, (15+8)/2, (15+30)/2, (28+16)/2]),
            # 'msd_length': [2.97] * 3 + [5.04] * 3,
            # 'Reorg energy': np.array([881, 865, 652, (1050+1050)/2, (1028+1016)/2, (1022 + 1034)/2]),
            # 'free_energy_bias': np.zeros(6),
            # 'frequency': 616.667 * np.ones(6),

            # 2D plane (nn-1, 2, 3) new
            # 'npairs': 9,
            # 'Coupling': np.array([39, 53, 203, (15+8)/2, (15+30)/2, (28+16)/2, 3, 9, 45]),
            # 'msd_length': [2.97] * 3 + [5.04] * 3 + [5.85] * 3,
            # 'Reorg energy': np.array([881, 865, 652, (1050+1050)/2, (1028+1016)/2, (1022 + 1034)/2, 1087, 1106, 1026]),
            # 'free_energy_bias': np.zeros(9),
            # 'frequency': 616.667 * np.ones(9),

            # 2D plane (nn-1) 1 on axis
            # 'npairs': 3,
            # 'Coupling': [106, 106, 106],
            # 'msd_length': [2.97, 2.97, 2.97],
            # 'Reorg energy': [799, 799, 799],
            # 'free_energy_bias': np.zeros(3),
            # 'frequency': 616.667 * np.ones(3),

            # 2D plane (nn-1) RMS
            # 'npairs': 3,
            # 'Coupling': [102, 102, 102],
            # 'msd_length': [2.97, 2.97, 2.97],
            # 'Reorg energy': [818, 818, 818],
            # 'free_energy_bias': np.zeros(3),
            # 'frequency': 616.667 * np.ones(3),

            # 2D plane (nn-1) Boltzmann
            # 'npairs': 3,
            # 'Coupling': [147, 147, 147],
            # 'msd_length': [2.97, 2.97, 2.97],
            # 'Reorg energy': [752, 752, 752],
            # 'free_energy_bias': np.zeros(3),
            # 'frequency': 616.667 * np.ones(3),

            # 2D plane (nn-1, 2, 3) Boltzmann
            # 'npairs': 9,
            # 'Coupling': np.array([147, 147, 147, 21, 21, 21, 32, 32, 32]),
            # 'msd_length': [2.97] * 3 + [5.04] * 3 + [5.85] * 3,
            # 'Reorg energy': np.array([752, 752, 752, 1032, 1032, 1032, 1061, 1061, 1061]),
            # 'free_energy_bias': np.zeros(9),
            # 'frequency': 616.667 * np.ones(9),

            # 2D plane (nn-1,2) RMS MAX
            # 'npairs': 6,
            # 'Coupling': np.array([102, 102, 102, 30, 30, 30]),
            # 'msd_length': [2.97] * 3 + [5.04] * 3,
            # 'Reorg energy': np.array([818, 818, 818, 1016, 1016, 1016]),
            # 'free_energy_bias': np.zeros(6),
            # 'frequency': 616.667 * np.ones(6),

            # 2D plane (nn-1,2,3) RMS MAX MAX
            # 'npairs': 9,
            # 'Coupling': np.array([102, 102, 102, 30, 30, 30, 45, 45, 45]),
            # 'msd_length': [2.97] * 3 + [5.04] * 3 + [5.85] * 3,
            # 'Reorg energy': np.array([818, 818, 818, 1016, 1016, 1016, 1026, 1026, 1026]),
            # 'free_energy_bias': np.zeros(9),
            # 'frequency': 616.667 * np.ones(9),

            # 2D plane (nn-1,2) RMS RMS
            # 'npairs': 6,
            # 'Coupling': np.array([102, 102, 102, 20, 20, 20]),
            # 'msd_length': [2.97] * 3 + [5.04] * 3,
            # 'Reorg energy': np.array([818, 818, 818, 1033, 1033, 1033]),
            # 'free_energy_bias': np.zeros(6),
            # 'frequency': 616.667 * np.ones(6),

            # 2D plane (nn-1,2,3) RMS RMS RMS
            # 'npairs': 9,
            # 'Coupling': np.array([102, 102, 102, 20, 20, 20, 27, 27, 27]),
            # 'msd_length': [2.97] * 3 + [5.04] * 3 + [5.85] * 3,
            # 'Reorg energy': np.array([818, 818, 818, 1033, 1033, 1033, 1074, 1074, 1074]),
            # 'free_energy_bias': np.zeros(9),
            # 'frequency': 616.667 * np.ones(9),

            # 2D plane (nn-1) mean
            # 'npairs': 3,
            # 'Coupling': [101, 101, 101],
            # 'msd_length': [2.97, 2.97, 2.97],
            # 'Reorg energy': [799, 799, 799],
            # 'free_energy_bias': np.zeros(3),
            # 'frequency': 616.667 * np.ones(3),

            # 2D plane (nn-1) electron
            'npairs': 3,
            'Coupling': [26, 26, 26],
            'msd_length': [2.97, 2.97, 2.97],
            'Reorg energy': [363, 363, 363],
            'free_energy_bias': np.zeros(3),
            'frequency': 616.667 * np.ones(3),

            # 2D plane (nn-1, 2) electron
            # 'npairs': 6,
            # 'Coupling': np.array([26, 26, 26, 57, 57, 57]),
            # 'msd_length': [2.97] * 3 + [5.04] * 3,
            # 'Reorg energy': np.array([363, 363, 363, 522, 522, 522]),
            # 'free_energy_bias': np.zeros(6),
            # 'frequency': 616.667 * np.ones(6),

            # 2D plane (nn-2) electron
            # 'npairs': 3,
            # 'Coupling': np.array([57, 57, 57]),
            # 'msd_length': [5.04] * 3,
            # 'Reorg energy': np.array([522, 522, 522]),
            # 'free_energy_bias': np.zeros(3),
            # 'frequency': 616.667 * np.ones(3),

            # 2D plane (nn-1, 2: +-a, 0, 0)
            # 'npairs': 6,
            # 'Coupling': [40.6, 53.6, 120.6, (12.3 + 8.7) / 2, 0.1, 0.1],
            # 'msd_length': [2.97, 2.97, 2.97, 5.05, 5.05, 5.05],
            # 'Reorg energy': [883, 873, 807, (1061+1058)/2, 10000, 10000],
            # 'free_energy_bias': np.zeros(6),
            # 'frequency': 616.667 * np.ones(6),

            # 2D plane (nn-1, 2)
            # 'npairs': 6,
            # 'Coupling': [40.6, 53.6, 120.6, (12.3+8.7)/2, (16.4+8.4)/2, (28.3+14.1)/2],
            # 'msd_length': [2.97, 2.97, 2.97, 5.05, 5.05, 5.05],
            # 'Reorg energy': [883, 873, 807, (1061+1058)/2, (1039+1054)/2, (1029+1035)/2],
            # 'free_energy_bias': np.zeros(6),
            # 'frequency': 616.667 * np.ones(6),

            # 2D plane (nn-1, nn-3: 0, 0, +ab)
            # 'npairs': 6,
            # 'Coupling': [40.6, 53.6, 120.6, 0.1, 0.1, 43.3],
            # 'msd_length': [2.97, 2.97, 2.97, 5.87, 5.87, 5.87],
            # 'Reorg energy': [883, 873, 807, 10000, 10000, 1029],
            # 'free_energy_bias': np.zeros(6),
            # 'frequency': 616.667 * np.ones(6),

            # 2D plane (nn-1, 3)
            # 'npairs': 6,
            # 'Coupling': [40.6, 53.6, 120.6, 4.7, 13.3, 43.3],
            # 'msd_length': [2.97, 2.97, 2.97, 5.87, 5.87, 5.87],
            # 'Reorg energy': [883, 873, 807, 1114, 1108, 1029],
            # 'free_energy_bias': np.zeros(6),
            # 'frequency': 616.667 * np.ones(6),

            # 2D plane (nn-1, 2, 3)
            # 'npairs': 9,
            # 'Coupling': [40.6, 53.6, 120.6, (12.3+8.7)/2, (16.4+8.4)/2, (28.3+14.1)/2, 4.7, 13.3, 43.3],
            # 'msd_length': [2.97, 2.97, 2.97, 5.05, 5.05, 5.05, 5.87, 5.87, 5.87],
            # 'Reorg energy': [883, 873, 807, (1061+1058)/2, (1039+1054)/2, (1029+1035)/2, 1114, 1108, 1029],
            # 'free_energy_bias': np.zeros(9),
            # 'frequency': 616.667 * np.ones(9),

            # # 2D plane (nn-1) Literature
            # 'Coupling': 40*np.ones(3),
            # 'Reorg energy': 800*np.ones(3),
            # 'Coupling': [35, 35, 41],
            # 'Reorg energy': [733, 733, 674],
            # 'npairs': 3,
            # 'msd_length': [2.97, 2.97, 2.97],
            # 'free_energy_bias': np.zeros(3),
            # 'frequency': 616.667*np.ones(3),

             'Temperature' : 600, #Kelvin
            'connectivity' : general_path + "connectivity.dat",
            'coordinates' : general_path + "40_40_1_supercell_2d_fe.dat",
            'Read_connectivity' : False, # if true, you mast supply connectivity file created manually (with or without PBC)
             'connectivity_create' : general_path + "connectivity_created.dat" ,
            'start_position' : 115-1, # 0-based stating position
            'number of steps' : 10000, #tot number of steps,
            'range linear fit' : slice(10000-500, 10000), #range for the linear fit depending on the number of total steps
            'Time step' : 0.01 , #fs
             'Method rate calc' : 'TOT', #this can be Marcus = "NA", Jacob mod = "TOT", "MLJ_rate" for quantized rate, "SO" for spectral overlap
            # below stuff for MLJ_rate only
              "lambda_classical"  : [124.7, 124.7, 124.7 ], #meV
              "lambda_quantum"    : [436.0,436.0,436.0 ], # meV
              'frequency_quantum' : [2.79e14, 2.79e14, 2.79e14],  # angualar freq in s-1
            # if SO provided
              "Spectral_overlap" : [0.35, 0.35, 0.35 ]  # in eV^-1 (only useful if MLJ is used)
        },

    }
    
    # save_name = 'electron-rms_nn-1,2_temp-800_n-3200_timestep-0p01_steps-10000'
    save_name = 'test'
    # Multiple systems can be done one after the other by providing the appropriate parameters
    systems_list = [ 'RUB_AOM']
    
    #############################################  MAIN  CODE ############################################


    params = {
       'axes.labelsize': 30,
       'font.size': 30,
       'legend.fontsize': 30,
       'xtick.labelsize': 30,
       'ytick.labelsize': 30,
       'text.usetex': False,
       'figure.figsize': [7,7],
        'axes.linewidth' : 3
       }
    rcParams.update(params)
    
    #hardcoded
    #free_energy = 0.0
    
    for system in systems_list:
        print "SYSTEM:", system
        
        n_pairs = select_list_dir_2_plot[system]['npairs']
        msd_length = select_list_dir_2_plot[system]['msd_length']
        hab = select_list_dir_2_plot[system]['Coupling']
        lambda_ = select_list_dir_2_plot[system]['Reorg energy']
        frequency = select_list_dir_2_plot[system]['frequency']
        temperature = select_list_dir_2_plot[system]['Temperature']
        nstep = select_list_dir_2_plot[system]['number of steps']
        range_ = select_list_dir_2_plot[system]['range linear fit']
        tstep = select_list_dir_2_plot[system]['Time step']
        pbc =  select_list_dir_2_plot[system]['Read_connectivity']
        created_conn =  select_list_dir_2_plot[system]['connectivity_create']
        method =  select_list_dir_2_plot[system]['Method rate calc']
        free_energy = select_list_dir_2_plot[system]['free_energy_bias']

        print('temperature')
        print(temperature)
        print('number of steps')
        print(nstep)
        print('Time step')
        print(tstep)
        print('Coupling')
        print(hab)
        print('Reorg energy')
        print(lambda_)


        coord_file = select_list_dir_2_plot[system]['coordinates']
        starting_pos = select_list_dir_2_plot[system]['start_position']

        if (method=="MLJ_rate"):
            lambS = select_list_dir_2_plot[system]["lambda_classical"]
            lambI =  select_list_dir_2_plot[system]["lambda_quantum"]
            W0 =    select_list_dir_2_plot[system]['frequency_quantum']
        elif (method=="SO"):
            spectral_overlap= select_list_dir_2_plot[system]["Spectral_overlap"]    
        
        # check if you want to use PBC or not
        if pbc:
            print "CONNECTIVITY IS READ FROM EXTERNAL FILE"
            connectivity = select_list_dir_2_plot[system]['connectivity']
            nmol, connectivity_list = read_connect(connectivity)
            nmol = int(nmol)
            print "number of interactions :", len(connectivity_list)
        else:
            print "CONNECTIVITY IS RECONSTRUCTED FROM DISTANCES AND PBC NOT USED"
            coord_ = read_coordinates(coord_file)
            print('coord[starting_pos]')
            print(coord_[starting_pos])
            nmol = int(len(coord_))
            #chech for duplicate 
            msd_length = list(OrderedDict.fromkeys(msd_length))
            ##### here we check if there are duplicated interaction that we cannot handle using connect from distance for now!!!
            print "adjusted number of interactions :", msd_length
            print "NMOL:", nmol
            connectivity_list = check_connect_from_distance(msd_length, coord_)
            # write constructed connect to an external file
            with open(created_conn, 'w') as f:
                f.write("%s \n" %nmol)
                for item in connectivity_list:
                    f.write("%s  %s  %s\n" % (item[0],item[1],item[2]) )
        
        # write interaction list
        interaction_dict = {}
        for i in range(n_pairs):
            #v = hrr[i] # get coupling
            #below is the line actually implemented in the matlab code from where I took this
            #w =(v*v/6.5821192569654e-16)*((3.1415926/(e1[i]*0.026))**0.5)*np.exp(-e1[i]/(4*temperature*8.617333*1e-5))    
            #you can just give more line as input my rate matches the published
            if (method=="MLJ_rate"):
                # calculate the quantum MLJ rate
                rate = calculate_rate_MLJ(coupling=hab[i], lambS=lambS[i], lambI=lambI[i], 
                                                         free_energy=free_energy[i], W0=W0[i], temperature=temperature)
            elif (method=="SO"):
                # use provided spectral overlap to calculate rate
                rate = calculate_rate_from_spectral_overlap(coupling=hab[i], spectral_overlap=spectral_overlap[i])

            else:
                # evaluate rate as specified in the input 
                rate = calculate_rate(coupling=hab[i], reorganization=lambda_[i], 
                                                              free_energy=free_energy[i], temperature=temperature,
                               method=method, frequency=frequency[i])###*1e15 #convertion to second-1
            print "rate %s: %s fs-1" %(str(i+1), rate)
            
            interaction_dict.update({str(i+1) : rate})
    
            
        # for now the number of interactions is just determined by the connectivity file
        k_matrix = build_kinetic_matrix(nmol, connectivity_list, interaction_dict)
        #print k_matrix
        simmetric = check_symmetric(k_matrix, rtol=1e-07, atol=1e-10)
        print "SIMMETRY", simmetric
    
        # calculate time evolution
        t, Pt = time_evolution(nmol, k_matrix, tstep, nstep, starting_pos)
        
        # plot population evolution for n states
        #for i in range(5):
        #    plt.plot(t, Pt[i])
        #plt.show()
        
        # read coordinates of the center of masses    
        com_chain = read_coordinates(coord_file)
        #print "COM", com_chain
        
        # calculate msd list (direction) of list(snaph): [[xx,xy,xz,yx,yy,yz, zx,zy,zz],[xx,xy,xz,yx,yy...]] 
        msd_list = _calc_3D_msd3(Pt, com_chain)
        # transpose to have time snapshots for each direction
        msd_snap = np.array(msd_list).T

        mobility_array = np.ones((3, 3))
        
        #fitting msd <L^2> xx
        fit = np.polyfit( t[range_], msd_snap[0][range_], 1 )
        #plot msd <L^2> xx
        fit_fn = np.poly1d(fit)
        plt.plot(t[range_], fit_fn(t[range_]), color = "k")
        plt.plot(t, msd_snap[0], label = "<L^2> dir. XX", linewidth=5, color="r")
        plt.plot(t[range_], fit_fn(t[range_]), color = "k")
        #mobility calculation
        mobility = fit[0] * 10**15 * 10**(-16) / (2 * 0.0000861728 * float(temperature) )
        mobility_array[0, 0] = mobility
        print "Linear mob. from <L^2> in xx = \sum_i Pi*(iL)^2 is: %.5f cm2.(V.s)-1" % mobility
        
        #fitting msd <L^2> xy
        fit = np.polyfit( t[range_], msd_snap[1][range_], 1 )
        #plot msd <L^2> xxy
        fit_fn = np.poly1d(fit)
        plt.plot(t[range_], fit_fn(t[range_]), color = "k")
        plt.plot(t, msd_snap[1], label = "<L^2> dir. XY", linewidth=5, color="y")
        plt.plot(t[range_], fit_fn(t[range_]), color = "k")
        #mobility calculation
        mobility = fit[0] * 10**15 * 10**(-16) / (2 * 0.0000861728 * float(temperature) )
        mobility_array[0, 1] = mobility
        mobility_array[1, 0] = mobility
        print "Linear mob. from <L^2> in xy = \sum_i Pi*(iL)^2 is: %.5f cm2.(V.s)-1" % mobility
        
        #fitting msd <L^2> xz
        fit = np.polyfit( t[range_], msd_snap[2][range_], 1 )
        #plot msd <L^2> xz
        fit_fn = np.poly1d(fit)
        plt.plot(t[range_], fit_fn(t[range_]), color = "k")
        plt.plot(t, msd_snap[2], label = "<L^2> dir. XZ", linewidth=5, color="k")
        plt.plot(t[range_], fit_fn(t[range_]), color = "k")
        #mobility calculation
        mobility = fit[0] * 10**15 * 10**(-16) / (2 * 0.0000861728 * float(temperature) )
        mobility_array[0, 2] = mobility
        mobility_array[2, 0] = mobility
        print "Linear mob. from <L^2> in xz = \sum_i Pi*(iL)^2 is: %.5f cm2.(V.s)-1" % mobility
        
        #fitting msd <L^2> yy
        fit = np.polyfit( t[range_], msd_snap[4][range_], 1 )
        #plot msd <L^2> yy
        fit_fn = np.poly1d(fit)
        plt.plot(t[range_], fit_fn(t[range_]), color = "k")
        plt.plot(t, msd_snap[4], label = "<L^2> dir. YY", linewidth=5, color="g")
        plt.plot(t[range_], fit_fn(t[range_]), color = "k")
        #mobility calculation
        mobility = fit[0] * 10**15 * 10**(-16) / (2 * 0.0000861728 * float(temperature) )
        mobility_array[1, 1] = mobility
        print "Linear mob. from <L^2> in yy = \sum_i Pi*(iL)^2 is: %.5f cm2.(V.s)-1" % mobility
        
    
        #fitting msd <L^2> yz
        fit = np.polyfit( t[range_], msd_snap[5][range_], 1 )
        #plot msd <L^2> yz
        fit_fn = np.poly1d(fit)
        plt.plot(t[range_], fit_fn(t[range_]), color = "k")
        plt.plot(t, msd_snap[5], label = "<L^2> dir. YZ", linewidth=5, color="m")
        plt.plot(t[range_], fit_fn(t[range_]), color = "k")
        #mobility calculation
        mobility = fit[0] * 10**15 * 10**(-16) / (2 * 0.0000861728 * float(temperature) )
        mobility_array[1, 2] = mobility
        mobility_array[2, 1] = mobility
        print "Linear mob. from <L^2> in yz = \sum_i Pi*(iL)^2 is: %.5f cm2.(V.s)-1" % mobility
    
    
        #fitting msd <L^2> zz
        fit = np.polyfit( t[range_], msd_snap[8][range_], 1 )
        #plot msd <L^2> zz
        fit_fn = np.poly1d(fit)
        plt.plot(t[range_], fit_fn(t[range_]), color = "k")
        plt.plot(t, msd_snap[8], label = "<L^2> dir. ZZ", linewidth=5, color="c")
        plt.plot(t[range_], fit_fn(t[range_]), color = "k")
        #mobility calculation
        mobility = fit[0] * 10**15 * 10**(-16) / (2 * 0.0000861728 * float(temperature) )
        mobility_array[2, 2] = mobility
        print "Linear mob. from <L^2> in zz = \sum_i Pi*(iL)^2 is: %.5f cm2.(V.s)-1" % mobility
    
        # plt.title(system)
        plt.xlabel('Time (fs)')
        plt.ylabel( r"MSD ($\AA^2$)")
        lgd = plt.legend(frameon=False, bbox_to_anchor=(1.1, 1.0))
        #plt.tight_layout()
        plt.savefig('MSD-%s' % system, bbox_inches='tight', bbox_extra_artists=(lgd,))
        plt.savefig('MSD-%s' % save_name, bbox_inches='tight', bbox_extra_artists=(lgd,))
        #plt.show()

        # Calculate mobility tensor to get mobility along the eigendirection of system (Chris Ahart code)
        calc_mobility_tensor(mobility_array)
        
        

            
