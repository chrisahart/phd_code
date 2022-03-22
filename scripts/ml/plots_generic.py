from __future__ import division, print_function, unicode_literals
import time
import numpy as np
import copy
from matplotlib import pyplot as plt
import matplotlib.cm as cm


"""
    Plotting functions
"""


def time_plot(data_x, data_y, labels, filename, fig, ax, xlabel, ylabel, low_opacity, folder_save, prints):
    """ Time dependency plot for data_x and list of arrays data_y """

    # Loop over each y array
    for i in range(len(data_y)):
        ax.plot(data_x[0:low_opacity], data_y[i][0:low_opacity], str(param.plotting_colors[i]),
                alpha=param.plotting_opacity)
        ax.plot(data_x[low_opacity:], data_y[i][low_opacity:], str(param.plotting_colors[i]), label=labels[i])

    # Set limits
    offset = 0.01
    data_y_all = np.concatenate(data_y, axis=0)
    ax.set_xlim([-1, np.max(data_x) * (1 + offset)])
    # ax.set_ylim([np.min(data_y_all) * (1 - offset), np.max(data_y_all) * (1 + offset)])

    # Set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Misc
    ax.legend(frameon=True)
    fig.tight_layout()

    # if print:
    #     fig.savefig('{}{}'.format(folder_save, filename), dpi=param.save_dpi, bbbox_inches='tight')


def parity_plot(data_x, data_y, labels, filename, fig, ax, xlabel, ylabel, low_opacity, folder_save):
    """ Parity plot for data_x and list of arrays data_y """

    # Loop over each y array
    for i in range(len(data_y)):
        ax.scatter(data_x[0:low_opacity], data_y[i][0:low_opacity], s=6, c=param.plotting_colors[i+1],
                   alpha=param.plotting_opacity)
        ax.scatter(data_x[low_opacity:], data_y[i][low_opacity:], s=6, c=param.plotting_colors[i+1], label=labels[i])

    # Set limits
    data = np.concatenate((data_x, np.concatenate(data_y, axis=0)), axis=0)
    ax.set_xlim([np.min(data), np.max(data)])
    ax.set_ylim([np.min(data), np.max(data)])
    ax.plot(np.linspace(np.min(data), np.max(data), num=2),
            np.linspace(np.min(data), np.max(data), num=2), 'k')

    # Set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Misc
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig('{}{}'.format(folder_save, filename), dpi=param.save_dpi, bbbox_inches='tight')
