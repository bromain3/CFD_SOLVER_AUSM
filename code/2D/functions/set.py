import numpy as np
from initial_conditions import *
import extract

def initial_conditions(nx,ny,bottom_surfaces):
    """
    sets all the interior points at the initial condition given in the project assignment
    :param nx, ny: (int) size of the augmented grid which is not the number of cells
    :param bottom_surfaces: (3D array) matrix containing the top and bottom surface vectors
    :return:(3D array) Q
    """
    Q = np.zeros((ny-1,nx-1,4))

    Q[:, :, 0] += rhoini
    Q[:, :, 1] += rhoini * Mini * extract.c(gamma, Tini)
    Q[:, :, 2] += 0
    Q[:, :, 3] += Pini / (gamma-1) + rhoini * (Mini * extract.c(gamma, Tini)) ** 2 / 2

    return(Q)

def bc(X,nx_eta,ny_eta):
    # inlet conditions
    X[:, 0, 0] = rhoini
    X[:, 0, 1] = rhoini * Mini * extract.c(gamma, Tini)
    X[:, 0, 2] = 0
    X[:, 0, 3] = Pini / (gamma - 1) + rhoini * (Mini * extract.c(gamma, Tini)) ** 2 / 2



    X[:, -1, :] = X[:, -2, :]  # setting all the hallo cells on the right to the value of the last column of computed cells

    # slip wall condition #
    # top wall
    #this approximation is fine as the top wall is flat, but in a second time it could be verified more rigorously
    X[-1, 1:-1, 1] = X[-1, 1:-1, 0] * X[-2, 1:-1,1] / X[-2, 1:-1,0]  # slip wall (the boudary condition is actualy computed only over the computational domaine)
    X[-1, 1:-1, 2] = - X[-1, 1:-1, 0] * X[-2, 1:-1,2] / X[-2, 1:-1,0] # slip wall (the boudary condition is actualy computed only over the computational domaine)

    # bottom wall
    X[0,1:-1,1] = X[0, 1:-1, 0]*(
                extract.u(X)[1, 1:-1] * (ny_eta[1, :] ** 2 - nx_eta[1, :] ** 2) - 2 * nx_eta[1, :] * ny_eta[1, :] * extract.v(X)[1, 1:-1])
    X[0, 1:-1, 2] = X[0, 1:-1, 0] * (
                extract.v(X)[1, 1:-1] * (nx_eta[1, :] ** 2 - ny_eta[1, :] ** 2) - 2 * nx_eta[1, :] * ny_eta[1, :] * extract.u(X)[1, 1:-1])

    # adiabatic condition #
    # top wall
    X[-1, :, 3] = r * X[-1, :, 0] / (gamma - 1) * ((gamma - 1) * (X[-2, :, 3] - 1 / 2 * (X[-2, :, 1] ** 2 + X[-2, :, 2] ** 2) / X[-2, :, 0]) / r / X[-2, :, 0]) + 1 / 2 * (X[-1, :, 1] ** 2 + X[-1, :, 2] ** 2) / X[-1, :, 0]
    # bottom wall
    X[0, :, 3] = r * X[0, :, 0] / (gamma - 1) * ((gamma - 1) * (X[1, :, 3] - 1 / 2 * (X[1, :, 1] ** 2 + X[1, :, 2] ** 2) / X[1, :, 0]) / r / X[1, :, 0]) + 1 / 2 * (X[0, :, 1] ** 2 + X[0, :, 2] ** 2) / X[0, :, 0]

    # 0 pressure gradient at walls #
    # bottom wall
    X[0, -1:1, 3] = (X[1, -1:1, 3] - 0.5 * (X[1, -1:1, 1] ** 2 + X[1, -1:1, 2] ** 2) / X[1, -1:1, 0]) + 0.5 * (X[0, -1:1, 1] ** 2 + X[0, -1:1, 2] ** 2) / X[0, -1:1, 0]
    # top wall
    X[-1, -1:1, 3] = (X[-2, -1:1, 3] - 0.5 * (X[-2, -1:1, 1] ** 2 + X[-2, -1:1, 2] ** 2) / X[-2, -1:1, 0]) + 0.5 * (X[-1, -1:1, 1] ** 2 + X[-1, -1:1, 2] ** 2) / X[-1, -1:1, 0]

    return(X)