import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PDB_small, PDB_closed
from MDAnalysis.analysis import distances
import matplotlib.pyplot as plt

u1 = mda.Universe(PDB_small)   # open AdK
u2 = mda.Universe(PDB_closed)  # closed AdK

ca1 = u1.select_atoms('name CA')
ca2 = u2.select_atoms('name CA')
resids1, resids2, dist = distances.dist(ca1, ca2, offset=0)  # for residue numbers
plt.plot(resids1, dist)
plt.ylabel('Ca distance (Angstrom)')
plt.axvspan(122, 159, zorder=0, alpha=0.2, color='orange', label='LID')
plt.axvspan(30, 59, zorder=0, alpha=0.2, color='green', label='NMP')
plt.legend()


if __name__ == "__main__":
    print('Finished.')
    plt.show()
