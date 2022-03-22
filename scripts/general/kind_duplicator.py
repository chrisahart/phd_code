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


""" Generate %KIND input file. 
    Take template and duplicate 100 times, changing &KIND number.  """

# Files
file_in = "/scratch/cahart/work/personal_files/fe_bulk/pdos/neutral/hematite_hse_pdos_fine_all/template/kind_oh_template.inp"
file_out = "/scratch/cahart/work/personal_files/fe_bulk/pdos/neutral/hematite_hse_pdos_fine_all/template/kind_oh.inp"

number = 0
number_end = 1000

with open(file_in) as f:
    with open(file_out, "w") as f1:

        contents = f.readlines()
        contents_new = copy.copy(contents)

        for i in range(1, number_end):

            contents_new[0] = '{}_{}\n'.format(contents[0][:-1], i)
            print(contents_new)

            for line in contents_new:
                    f1.write(line)

print('Finished.')
