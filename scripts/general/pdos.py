from __future__ import division, print_function
from math import pi, sqrt
import numpy as np


class pdos:

    def __init__(self, infilename):

        input_file = open(infilename, 'r')

        firstline = input_file.readline().strip().split()
        secondline = input_file.readline().strip().split()

        # Kind of atom
        self.atom = firstline[6]
        # iterationstep
        self.iterstep = int(firstline[12][:-1])  # [:-1] delete ","
        # Energy of the Fermi level
        self.efermi = float(firstline[15])

        # it keeps just the orbital names
        secondline[0:5] = []
        self.orbitals = secondline

        lines = input_file.readlines()

        eigenvalue = []
        self.occupation = []
        data = []
        self.pdos = []
        for index, line in enumerate(lines):
            data.append(line.split())
            data[index].pop(0)
            eigenvalue.append(float(data[index].pop(0)))
            self.occupation.append(int(float(data[index].pop(0))))
            self.pdos.append([float(i) for i in data[index]])

        self.e = [(x - self.efermi) * 27.211384523 for x in eigenvalue]

        print('pdos\n', pdos)

        self.tpdos = []
        for i in self.pdos:
            self.tpdos.append(sum(i))

    def __add__(self, other):
        """Return the sum of two PDOS objects"""

        sumtpdos = [i + j for i, j in zip(self.tpdos, other.tpdos)]
        return sumtpdos

    def delta(self, emin, emax, npts, energy, width):

        energies = np.linspace(emin, emax, npts)
        x = -((energies - energy) / width) ** 2
        # print('np.exp(x) / (sqrt(pi) * width)', np.exp(x) / (sqrt(pi) * width))

        return np.exp(x) / (sqrt(pi) * width)

    def smearing(self, npts, width, ):
        """Return a gaussian smeared DOS"""

        d = np.zeros(npts)
        print('d \n', d)
        print('self.e \n', self.e)
        print('tpdos \n', self.tpdos)

        emin = min(self.e)
        emax = max(self.e)
        for i in range(npts):
            d += self.tpdos[i] * self.delta(emin, emax, npts, self.e[i], width)
            print('d\n', d)

        # for e, pd in zip(self.e, self.tpdos):
        #     d += pd * self.delta(emin, emax, npts, e, width)
            # print('\n pd\n', pd)  # Convolved PDOS

        # print('\n pd\n', pd)  # Convolved PDOS
        # print('\n d\n', d)  # Convolved PDOS
        # print('self.e\n', self.e) # Eigenvalues
        # print('tpdos\n', self.tpdos) # Sum of each row in PDOS

        return d
