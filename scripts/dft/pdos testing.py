from __future__ import division, print_function
import time
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from scripts.general import functions
from scripts.general import parameters
from scripts.general.pdos import *
# from scripts.general.pdos_org import *
import sys

# infilename = sys.argv[1]
infilename = '/scratch/cahart/work/personal_files/feIV_bulk/pdos/hematite/hematite-ALPHA_k1-1.pdos'


alpha = pdos(infilename)
npts = len(alpha.e)
alpha_smeared = alpha.smearing(npts, 0.1)
eigenvalues = np.linspace(min(alpha.e), max(alpha.e), npts)

# g = open('smeared.dat', 'w')
# for i, j in zip(eigenvalues, alpha_smeared):
#     t = str(i).ljust(15) + '     ' + str(j).ljust(15) + '\n'
#     g.write(t)

print(eigenvalues)
print(alpha_smeared)
plt.plot(eigenvalues, alpha_smeared)

if __name__ == "__main__":
    print('Finished.')
    plt.show()
