import matplotlib.pyplot as plt
import extract
from air import *

def residuals(Cr,Xr,Yr,Er):
    """
    Plot the residuals all together
    :param Cr: continuity residual (list)
    :param Xr: X-momentum residual (list)
    :param Yr: Y-momentum residual (list)
    :param Er: energy residual (list)
    """
    parameters = {'axes.labelsize': 20,
                  'axes.titlesize': 20,
                  'legend.fontsize': 20,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15}
    plt.rcParams.update(parameters)
    plt.plot(Cr,label='Continuity residual')
    plt.plot(Xr, label='X-momentum residual')
    plt.plot(Yr, label='Y-momentum residual')
    plt.plot(Er, label='Energy residual')
    plt.legend()
    plt.semilogy()
    plt.xlabel('Number of iterations')
    plt.show()

def physical_data(Q,req,x_V,y_V,colormap='jet',levels=20):
    """
    Plot all the data required by the user
    :param Q: state vector (3D-array)
    :param req: (list) of (str) defining the required plots
    :param x_V: x coordinates of the cell centers (2D-array)
    :param y_V: y coordinates of the cell centers (2D-array)
    :param colormap: colormap used on the plot (str)
    :param levels: number of level lines required on the plots (int)
    """
    parameters = {'axes.labelsize': 20,
                  'axes.titlesize': 20,
                  'legend.fontsize': 20,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15}
    plt.rcParams.update(parameters)
    for x in req:
        plots = x.split(';')
        for p in plots:
            if p == 'rho':
                plt.pcolormesh(x_V, y_V, Q[:, :, 0], cmap=colormap)
                #plt.contourf(x_V, y_V, Q[:, :, 0], levels=levels, cmap=colormap)
                plt.colorbar()
            elif p == 'rhou':
                plt.contourf(x_V, y_V, Q[:, :, 1], levels=levels, cmap=colormap)
                plt.colorbar()
            elif p == 'rhov':
                plt.contourf(x_V, y_V, Q[:, :, 2], levels=levels, cmap=colormap)
                plt.colorbar()
            elif p == 'rhoet':
                plt.contourf(x_V, y_V, Q[:, :, 3], levels=levels, cmap=colormap)
                plt.colorbar()
            elif p == 'u':
                plt.contourf(x_V, y_V, extract.u(Q), levels=levels, cmap=colormap)
                plt.colorbar()
            elif p == 'v':
                plt.contourf(x_V, y_V, extract.v(Q), levels=levels, cmap=colormap)
                plt.colorbar()
            elif p == 'et':
                plt.contourf(x_V, y_V, extract.et(Q), levels=levels, cmap=colormap)
                plt.colorbar()
            elif p == 'P':
                # can be converted in hpa or be given relative to the initial value
                plt.contourf(x_V, y_V, extract.P(Q), levels=levels, cmap=colormap)
                plt.colorbar()
            elif p == 'P0':
                # can be converted in hpa or be given relative to the initial value
                plt.contourf(x_V, y_V, extract.P0(Q), levels=levels, cmap=colormap)
                plt.colorbar()
            elif p == 'c':
                plt.contourf(x_V, y_V, extract.c(gamma, extract.T(Q)), levels=levels, cmap=colormap)
                plt.colorbar()
            elif p == 'T':
                plt.contourf(x_V, y_V, extract.T(Q), levels=levels, cmap=colormap)
                plt.colorbar()
            elif p == 'ht':
                plt.contourf(x_V, y_V, extract.ht(Q), levels=levels, cmap=colormap)
                plt.colorbar()
            elif p == 'M':
                plt.contourf(x_V, y_V, extract.M(Q), levels=levels, cmap=colormap)
                plt.colorbar()
            elif p == 'quiver':
                # this plot does not look nice and could be improved
                plt.quiver(x_V, y_V, extract.u(Q), extract.v(Q))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(plots)
        plt.show()