import os
from itertools import product
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker

X_MAJOR_LOCATOR_1D = 0.2
Y_MAJOR_LOCATOR_1D = 0.1

ALPHA_2D_PLOT = 0.75
AXIS_RANGE_SLACK = 0.2
NUM_SURFACES = 32
Z_MAJOR_LOCATOR_2D = 10
COLORBAR_ASPECT = 5
COLORBAR_SHRINK = 0.5


def visualize_1d_critic(model, xrange, step, cuda=False):
    # Create the axis on which the dualnet will be evaluated.
    xrange = np.arange(xrange[0], xrange[1], step=step)
    inputs = xrange[..., None]

    # Evaluate the critic at those points and reshape on a grid.
    if cuda:
        outs = model.forward(Variable(torch.from_numpy(inputs).float()).cuda())
        outs = outs.data.cpu().numpy().flatten()
    else:
        outs = model.forward(Variable(torch.from_numpy(inputs).float()))
        outs = outs.data.numpy().flatten()

    # Plot the critic landscape.
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(xrange, outs, label="approximation")

    ax.set_xlabel("input")
    ax.set_ylabel("output")
    ax.grid()

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(X_MAJOR_LOCATOR_1D))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(Y_MAJOR_LOCATOR_1D))

    return ax


def visualize_2d_critic(model, xrange, yrange, step, fig_type, cuda=False):
    assert fig_type in ["contour", "contourf", "plot_surface"], "Requested 3d plot type not supported. "
    # Form the coordinates at which the critic will be evaluated.
    xrange = np.arange(xrange[0], xrange[1], step=step)
    yrange = np.arange(yrange[0], yrange[1], step=step)
    xv, yv = np.meshgrid(xrange, yrange)
    full_coords = np.concatenate((xv[None, :], yv[None, :]), axis=0).reshape(2, -1).T

    # Evaluate the critic at those points and reshape on a grid.
    if cuda:
        critic_vals = model.forward(Variable(torch.from_numpy(full_coords).float()).cuda())
        landscape = critic_vals.data.cpu().numpy().reshape(xv.shape)
    else:
        critic_vals = model.forward(Variable(torch.from_numpy(full_coords).float()))
        landscape = critic_vals.data.numpy().reshape(xv.shape)

    # Plot the critic landscape.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if fig_type == "plot_surface":
        surf = ax.plot_surface(xv, yv, landscape, cmap=cm.coolwarm, linewidth=0, antialiased=False,
                               alpha=ALPHA_2D_PLOT)
    elif fig_type == "contourf":
        surf = ax.contourf(xv, yv, landscape, NUM_SURFACES, cmap=cm.coolwarm, linewidth=0, antialiased=False,
                           alpha=ALPHA_2D_PLOT)
    elif fig_type == "contour":
        surf = ax.contour(xv, yv, landscape, NUM_SURFACES, cmap=cm.coolwarm, linewidth=0, antialiased=False,
                          alpha=ALPHA_2D_PLOT)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Customize the z axis.
    landscape_range = np.max(landscape) - np.min(landscape)

    ax.set_zlim(np.min(landscape) - AXIS_RANGE_SLACK * landscape_range,
                np.max(landscape) + AXIS_RANGE_SLACK * landscape_range)

    ax.zaxis.set_major_locator(LinearLocator(Z_MAJOR_LOCATOR_2D))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=COLORBAR_SHRINK, aspect=COLORBAR_ASPECT)

    return ax


def save_2d_dualnet_visualizations(model, figures_dir, config, epoch=None, loss=None, after_training=False):
    for fig_type in config.visualize_2d.fig_types:
        ax = visualize_2d_critic(model, config.visualize_2d.xrange, config.visualize_2d.yrange,
                                 config.visualize_2d.step, fig_type=fig_type, cuda=config.cuda)
        for elev, azim in product(config.visualize_2d.elev, config.visualize_2d.azim):
            ax.view_init(elev=elev, azim=azim)

            if after_training:
                title_text = "Model: {} - Activation: {}".format(config.model.name, config.model.activation)
                save_path = os.path.join(figures_dir, "visualize_2d_" + fig_type +
                                         "_elev_{}_azim{}_best_model".format(elev, azim) + ".png")
            else:
                title_text = "Model: {} - Activation: {}\nEpoch: {} - Loss: {}".format(config.model.name,
                                                                                   config.model.activation,
                                                                                   epoch,
                                                                                   loss)
                save_path = os.path.join(figures_dir, "epoch_{:04}_visualize_2d_".format(epoch) + fig_type +
                                         "_elev_{}_azim{}".format(elev, azim) + ".png")

            plt.title(title_text, x=0.5, y=1.0)
            plt.tight_layout()
            plt.savefig(save_path)
    plt.close('all')


def save_1d_dualnet_visualizations(model, figures_dir, config, epoch=None, loss=None, after_training=False):
    visualize_1d_critic(model, config.visualize_1d.xrange, config.visualize_1d.step, config.cuda)

    if after_training:
        title_text = "Model: {} - Activation: {}".format(config.model.name, config.model.activation)
        save_path = os.path.join(figures_dir, "visualize_1d_best_model.png")
    else:
        title_text = "Model: {} - Activation: {}\nEpoch: {} - Loss: {}".format(config.model.name,
                                                                               config.model.activation,
                                                                               epoch,
                                                                               loss)
        save_path = os.path.join(figures_dir, "epoch_{:04}_visualize_1d_".format(epoch))

    plt.title(title_text, x=0.5, y=1.0)
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path)
    # plt.show(block=True)
    plt.close('all')
