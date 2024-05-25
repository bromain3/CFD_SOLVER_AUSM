import numpy as np
import matplotlib.pyplot as plt

import readgrid
import cellproperties

######################### change boundary conditions with the new functions made(T and P)

######################################################################
###################  air thermo data definition  #####################
######################################################################

######################################################################
#################  initial conditions definition  ####################
######################################################################
from initial_conditions import *

######################################################################
####################  physical data extraction  ######################
######################################################################
import extract

######################################################################
##########################  MUSCL scheme  ############################
######################################################################
import interpolation

######################################################################
################  Initialisation and boundaries  #####################
######################################################################
import set


######################################################################
########################  2nd order MUSCL  ###########################
######################################################################

def M_plus(M):
    """
    Computes the value of M+ as given in the AUSM scheme
    :param M: mach number (2D-array)
    :return: M+ (2D-array)
    """
    return((M+np.abs(M))/2)

def M_minus(M):
    """
    Computes the value of M- as given in the AUSM scheme
    :param M: mach number (2D-array)
    :return: M- (2D-array)
    """
    return ( (M - np.abs(M)) / 2)

def P_plus(P,M):
    """
    Computes the value of P+ at the first order as given in the AUSM scheme
    :param P: pressure (2D-array)
    :return: P+ at first order (2D-array)
    """
    return(P/2*(1+np.sign(M)))


def P_minus(P,M):
    """
    Computes the value of P- at the first order as given in the AUSM scheme
    :param P: pressure (2D-array)
    :return: P- at first order (2D-array)
    """
    return ((P / 2 * (1 - np.sign(M))))


def phi(X):
    """
    Computes the vector called phi in the AUSM schem divided by Mach 1/2
    :param X: state vector (3D-array)
    :return: phi/M_half (3D-array)
    """
    a = extract.c(gamma, extract.T(X))
    res = np.zeros(np.shape(X))
    rho = X[:,:,0]
    rho_u = X [:,:, 1]
    rho_v = X[:,:,2]
    ht = extract.ht(X)
    res[:,:,0] = rho*a
    res[:,:,1] = rho_u*a
    res[:,:,2] = rho_v*a
    res[:,:,3] = rho*ht*a
    return(res)


def one_step(X,deltat,V,Sb,Sl,x_V,y_V,nx_xi,ny_xi,nx_eta,ny_eta,plot,iter):
    """
    Updates the array of state vectors after one time step of length deltat
    :param X: array of state vectors (3D-arrays)
    :param deltat: time step (float)
    :param V: array of cells volume (2D-array)
    :param Sb: array of surface vectors of surfaces parallel to xi (3D-array)
    :param Sl: array of surface vectors of surfaces parallel to eta (3D-array)
    :param x_V: x coordinates of cells center (2D-array)
    :param y_V: y coordinates of cells center (2D-array)
    :param nx_xi,ny_xi,nx_eta,ny_eta: direction vector of the surfaces (2D-array)
    :param plot: flag set to 1 if we want a plot of the grid after the computed time step (bool)
    :param iter: the iteration number of the plot made (int)
    :return: updated array of state vectors X (3D-arrays)
    """

    # extrapolate Q on cell edges with MUSCL scheme #
    Q_l_x_plus_half,Q_r_x_plus_half,Q_l_y_plus_half,Q_r_y_plus_half = interpolation.MUSCL(X, eps=1, k=-1, phi=interpolation.minmod)

    # Use extrapolated Q to compute E and F #
    # E_xi
    u_l = extract.u(Q_l_x_plus_half)
    v_l = extract.v(Q_l_x_plus_half)
    u_xi_l = nx_xi * u_l + ny_xi * v_l
    P_xi_l = extract.P(Q_l_x_plus_half)
    E_xi_l = np.zeros(np.shape(Q_l_x_plus_half))
    E_xi_l[:,:,0] = Q_l_x_plus_half[:,:,0]*u_xi_l
    E_xi_l[:,:,1] = Q_l_x_plus_half[:,:,1]*u_xi_l + nx_xi*P_xi_l
    E_xi_l[:, :, 2] = Q_l_x_plus_half[:, :, 2] * u_xi_l + ny_xi * P_xi_l
    E_xi_l[:, :, 3] = Q_l_x_plus_half[:, :, 0] * extract.ht(Q_l_x_plus_half) * u_xi_l

    u_r = extract.u(Q_r_x_plus_half)
    v_r = extract.v(Q_r_x_plus_half)
    u_xi_r = nx_xi * u_r + ny_xi * v_r
    P_xi_r = extract.P(Q_r_x_plus_half)
    E_xi_r = np.zeros(np.shape(Q_r_x_plus_half))
    E_xi_r[:, :, 0] = Q_r_x_plus_half[:, :, 0] * u_xi_r
    E_xi_r[:, :, 1] = Q_r_x_plus_half[:, :, 1] * u_xi_r + nx_xi * P_xi_r
    E_xi_r[:, :, 2] = Q_r_x_plus_half[:, :, 2] * u_xi_r + ny_xi * P_xi_r
    E_xi_r[:, :, 3] = Q_r_x_plus_half[:, :, 0] * extract.ht(Q_r_x_plus_half) * u_xi_r

    ########## under that, we can select l or r state in function of the direction of u_xi as in ausm
    E_xi = 0.5 * (E_xi_l + E_xi_r)

    # F_eta
    u_l = extract.u(Q_l_y_plus_half)
    v_l = extract.v(Q_l_y_plus_half)
    v_eta_l = nx_eta * u_l + ny_eta * v_l
    P_eta_l = extract.P(Q_l_y_plus_half)
    F_eta_l = np.zeros(np.shape(Q_l_y_plus_half))
    F_eta_l[:, :, 0] = Q_l_y_plus_half[:, :, 0] * v_eta_l
    F_eta_l[:, :, 1] = Q_l_y_plus_half[:, :, 1] * v_eta_l + nx_eta * P_eta_l
    F_eta_l[:, :, 2] = Q_l_y_plus_half[:, :, 2] * v_eta_l + ny_eta * P_eta_l
    F_eta_l[:, :, 3] = Q_l_y_plus_half[:, :, 0] * extract.ht(Q_l_y_plus_half) * v_eta_l

    u_r = extract.u(Q_r_y_plus_half)
    v_r = extract.v(Q_r_y_plus_half)
    v_eta_r = nx_eta * u_r + ny_eta * v_r
    P_eta_r = extract.P(Q_r_y_plus_half)
    F_eta_r = np.zeros(np.shape(Q_r_y_plus_half))
    F_eta_r[:, :, 0] = Q_r_y_plus_half[:, :, 0] * v_eta_r
    F_eta_r[:, :, 1] = Q_r_y_plus_half[:, :, 1] * v_eta_r + nx_eta * P_eta_r
    F_eta_r[:, :, 2] = Q_r_y_plus_half[:, :, 2] * v_eta_r + ny_eta * P_eta_r
    F_eta_r[:, :, 3] = Q_r_y_plus_half[:, :, 0] * extract.ht(Q_r_y_plus_half) * v_eta_r

    ########## under that, we can select l or r state in function of the direction of v_eta as in ausm
    F_eta = 0.5 * (F_eta_l + F_eta_r)

    # computing the step of the iteration with a Roe scheme #
    i_variation = E_xi[:,1:,:]*np.linalg.norm(Sl[1:-1,2:-1,:],axis=2)[:,:,np.newaxis] - E_xi[:,:-1,:]*np.linalg.norm(Sl[1:-1,1:-2,:],axis=2)[:,:,np.newaxis]
    j_variation = F_eta[1:,:,:]*np.linalg.norm(Sb[2:-1,1:-1,:],axis=2)[:,:,np.newaxis] - F_eta[:-1,:,:]*np.linalg.norm(Sb[1:-2,1:-1,:],axis=2)[:,:,np.newaxis]
    step = deltat*(i_variation)#+j_variation)

    # apply the step #
    X[1:-1, 1:-1, :] = X[1:-1, 1:-1, :] - step/V[1:-1,1:-1,np.newaxis]

    # reset the boundary conditions #
    X = set.bc(X, nx_eta, ny_eta)

    # if required, plot intermediate states #
    if plot:
        plt.contourf(x_V, y_V, X[:, :, 0], levels=20, cmap='jet')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Density')
        #plt.savefig('C:/Users/bapti/OneDrive/fichiers ancien pc/OneDrive/Documents/prepa/GeorgiaTech/sur place/spring 2023/CFD/Project/data/output/step_1_AUSM/gif_images/'+str(iter),dpi=500)
        #plt.clf()
        plt.show()

    return(X)

if __name__ == '__main__':
    # reading the grid #
    nx, ny, x, y = readgrid.readGrid('../../grids', 'g65x49u_augmented.dat')
    Volumes, x_V, y_V, bottom_Surfaces, x_Sb, y_Sb, left_Surfaces, x_Sl, y_Sl = cellproperties.cellsCharacteristics('../../grids', 'g65x49u_augmented.dat')
    # intitialise the computation parameters #
    Q = set.initial_conditions(nx, ny, bottom_Surfaces)
    Q = set.bc(Q, bottom_Surfaces)