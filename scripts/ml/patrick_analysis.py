#!/usr/bin/python
import qlab as qp
import matplotlib.pyplot as plt
import numpy as np


def get_force_diff_config(in_fn, ff_pot, step):
    in_conf = qp.AtomsList(in_fn)[step + 1]
    ff_conf = in_conf.copy()

    ff_energy = qp.farray(0.0)
    ff_pot.calc(ff_conf, energy=ff_energy, force=True)

    return np.array(in_conf.force), np.array(ff_conf.force), np.array(in_conf.energy), np.array(ff_energy)


def get_range_f_diff(in_fn, ff_pot, n_conf):
    myforces = []
    myenergies = []
    for i in range(n_step):
        result = get_force_diff_config(in_fn, ff_pot, i)
        myforces.append((result[0], result[1]))
        myenergies.append((result[2], result[3]))
    return myforces, myenergies


def get_force_errors(my_forces_tuple, dim):
    force_err = []
    for i in range(0, len(my_forces_tuple[:])):
        force_err.append(np.array(my_forces_tuple[i][0][:][dim, :]
                                  - my_forces_tuple[i][1][:][dim, :]))
    return force_err


def get_3d_force_errors(my_forces_tuple):
    force_err = []
    for i in range(3):
        force_err.append(get_force_errors(my_forces_tuple, i))
    return force_err


def get_energy_errors(my_energy_tuple, n_steps):
    energy_err = []
    for i in range(0, len(my_energy_tuple[:])):
        energy_err.append(np.array(my_energy_tuple[i][0] -
                                   my_energy_tuple[i][1]))
    return energy_err


def plot_forces(my_forces_tuple, n_step, save=False, subplots=False):
    markers = ['r.', 'r.', 'k.']
    if subplots == False:
        fig2, axes2 = plt.subplots(1, 1)
        for step in my_forces_tuple:
            axes2.plot(step[0], step[1],
                       markers[0], alpha=0.6)

        axes2.plot(np.arange(-50, 50, 1), np.arange(-50, 50, 1), 'k', linestyle='dashed')
        minim = -30
        maxim = 30
        plt.xlim(minim, maxim)
        plt.ylim(minim, maxim)
        # plt.xlim(-15, 15)
        # plt.ylim(-15, 15)
        plt.legend(prop={'size': 8}, loc='best')
    elif subplots == True:
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
        for dim in range(3):
            for i in range(n_step):
                axes[dim].plot(my_forces_tuple[i][0][:][dim, :],
                               my_forces_tuple[i][1][:][dim, :],
                               markers[dim], alpha=0.3)

    plt.xlabel(r"DFT Force $eV \quad \AA^{-1}$")
    plt.ylabel(r"Predicted Force $eV \quad \AA^{-1}$")
    font = {'size': '20'}
    plt.rc('font', **font)
    plt.tight_layout()
    if save == True:
        plt.savefig("forces.png", format='png', dpi=300)
    # plt.show()


def plot_force_errors(my_forces_tuple, f_errors, n_step, save=False):
    outp = open("Force_Errors.dat", 'w')
    stddev = []
    avg = []
    markers = ['r.', 'r.', 'k.']
    x_labels = ['Force x ($eV \quad {\AA}^{-1}$)',
                'Force y ($eV \quad {\AA}^{-1}$)',
                'Force z ($eV \quad {\AA}^{-1}$)']
    fig, axes = plt.subplots(1, 1, sharey=True)
    for dim in range(3):
        forces = []
        for i in range(n_step):
            forces.append(my_forces_tuple[i][0][:][dim, :])

        if dim == 0:
            axes.plot(forces,
                      np.absolute(f_errors[dim][:]),
                      markers[dim], alpha=0.6)  # , label='In Plane')
        elif dim == 1:
            axes.plot(forces,
                      np.absolute(f_errors[dim][:]),
                      markers[dim], alpha=0.6)
        elif dim == 2:
            axes.plot(forces,
                      np.absolute(f_errors[dim][:]),
                      markers[dim], alpha=0.6)  # , label='Out of Plane')

        axes.set_xlabel("DFT Force")
        axes.set_ylabel("Force Error vs DFT $\Delta$f $(eV \quad {\AA}^{-1})$")

        stddev.append(np.std(np.absolute(f_errors[dim][:])))
        avg.append(np.average(np.absolute(f_errors[dim][:])))
        outp.write(str(avg[dim]) + "\t" + str(stddev[dim]) + '\n')

    axes.plot(np.array(['NaN']), np.array(['NaN']), markers[0], label='In Plane Forces')
    axes.plot(np.array(['NaN']), np.array(['NaN']), markers[2], label='Out of Plane Forces')
    # plt.xlim(-8, 8)
    # plt.ylim(0, 12)
    font = {'size': '20'}
    plt.rc('font', **font)
    plt.tight_layout()
    plt.legend(prop={'size': 8}, loc='best')
    if save == True:
        plt.savefig("force_errors.png", format='png', dpi=300)


def plot_energy_errors(my_energy_tuple, e_errors, n_step, save=True):
    outp = open("Force_Errors.dat", 'w')
    energies = []
    for i in range(n_step):
        energies.append(my_energy_tuple[i][0])
    mae = np.sum(np.absolute(energies
                             - np.array(e_errors))) / (len(e_errors))
    outp.write(str(mae) + '\n')
    plt.figure(3)
    plt.plot(energies, e_errors, 'r+', alpha=0.5)
    font = {'size': '20'}
    plt.rc('font', **font)
    plt.tight_layout()
    print energies
    print e_errors

    if save == True:
        plt.savefig("Energy_Errors.png", format='png', dpi=300)


def new_plot_forces(force_diffs, n_step):
    for step in force_diffs:
        plt.plot(step[0], step[1])

    plt.show()


n_step = 10