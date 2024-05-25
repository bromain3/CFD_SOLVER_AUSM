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
##########################  AUSM scheme  #############################
######################################################################

def M_plus(M):
    """
    Computes the value of M+ as given in the AUSM scheme
    :param M: mach number (2D-array)
    :return: M+ (2D-array)
    """
    return((np.abs(M)<=1)*(M+1)**2/4 + (np.abs(M)>1)*(M+np.abs(M))/2)
    y, x = np.shape(M)
    res = np.zeros((y, x))
    for j in range(y):
        for i in range(x):
            if np.abs(M[j, i]) <= 1:
                res[j, i] = ((M[j, i] + 1) ** 2) / 4
            else:
                res[j, i] = (M[j, i] + np.abs(M[j, i])) / 2
    return (res)

def M_minus(M):
    """
    Computes the value of M- as given in the AUSM scheme
    :param M: mach number (2D-array)
    :return: M- (2D-array)
    """
    return ((np.abs(M) <= 1) * (-(M - 1)**2) / 4 + (np.abs(M) > 1) * (M - np.abs(M)) / 2)
    y, x = np.shape(M)
    res = np.zeros((y, x))
    for j in range(y):
        for i in range(x):
            if np.abs(M[j, i]) <= 1:
                res[j, i] = (-(M[j,i]-1)**2)/4
            else:
                res[j, i] = (M[j, i]-np.abs(M[j,i])) / 2
    return (res)

def P_plus(P,M):
    """
    Computes the value of P+ at the first order as given in the AUSM scheme
    :param P: pressure (2D-array)
    :return: P+ at first order (2D-array)
    """
    return((np.abs(M)>1)*(P/2*(1+np.sign(M))) + (np.abs(M)<=1)*(P/2*(1+M)))
    y,x = np.shape(P)
    res = np.zeros((y,x))
    for j in range(y):
        for i in range(x):
            if np.abs(M[j,i]) <= 1:
                res[j,i] = P[j,i]/2*(1+M[j,i])
            else:
                res[j,i] = P[j,i]/2*(M[j,i]+np.abs(M[j,i]))/M[j,i]
    return(res)

def P_minus(P,M):
    """
    Computes the value of P- at the first order as given in the AUSM scheme
    :param P: pressure (2D-array)
    :return: P- at first order (2D-array)
    """
    return ((np.abs(M) > 1) * (P / 2 * (1 - np.sign(M))) + (np.abs(M) <= 1) * (P / 2 * (1 - M)))
    y, x = np.shape(P)
    res = np.zeros((y, x))
    for j in range(y):
        for i in range(x):
            if np.abs(M[j, i]) <= 1:
                res[j, i] = -P[j, i] / 2 * (1 - M[j, i])
            else:
                res[j, i] = P[j, i] / 2 * (M[j, i] - np.abs(M[j, i])) / M[j, i]
    return (res)

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
    Q_l_x_plus_half,Q_r_x_plus_half,Q_l_y_plus_half,Q_r_y_plus_half = interpolation.MUSCL(X)  # we can use the function MUSCL_V1 to have faster computations

    # Use extrapolated Q to compute E and F as given in the AUSM scheme #
        # E_xi
            # compute Mxi1/2
    u_l = extract.u(Q_l_x_plus_half)
    v_l = extract.v(Q_l_x_plus_half)
    u_xi_l = nx_xi*u_l + ny_xi*v_l
    a_l_x = extract.c(gamma, extract.T(Q_l_x_plus_half))
    M_xi_l_plus = M_plus(u_xi_l/a_l_x)

    u_r = extract.u(Q_r_x_plus_half)
    v_r = extract.v(Q_r_x_plus_half)
    u_xi_r = nx_xi * u_r + ny_xi * v_r
    a_r_x = extract.c(gamma, extract.T(Q_r_x_plus_half))
    M_xi_r_minus = M_minus(u_xi_r / a_r_x)

    M_xi_half = M_xi_l_plus + M_xi_r_minus

    #advection_E = ((np.abs(M_xi_half) >= 0)[:,:,np.newaxis]*phi(Q_l_x_plus_half) + (np.abs(M_xi_half) < 0)[:,:,np.newaxis]*phi(Q_r_x_plus_half))  ########multiply with M_half
    advection_E = M_xi_half[:,:,np.newaxis]*0.5*(phi(Q_l_x_plus_half)+phi(Q_r_x_plus_half))-0.5*np.abs(M_xi_half)[:,:,np.newaxis]*(phi(Q_r_x_plus_half)-phi(Q_l_x_plus_half))

    # compute the pressure part
    P_l = extract.P(Q_l_x_plus_half)
    P_l_plus = P_plus(P_l,u_xi_l/a_l_x)
    #P_l_plus = P_plus(P_l, M_xi_half)
    P_r = extract.P(Q_r_x_plus_half)
    P_r_minus = P_minus(P_r,u_xi_r / a_r_x)
    #P_r_minus = P_minus(P_r, M_xi_half)

    pressure = np.zeros(np.shape(Q_l_x_plus_half))
    pressure[:,:,1] = nx_xi*(P_l_plus+P_r_minus)  # as here the computation averages at cell face, we use extrapolated direction vector at cell face
    pressure[:,:,2] = ny_xi*(P_l_plus+P_r_minus)  # as here the computation averages at cell face, we use extrapolated direction vector at cell face

    E_xi = advection_E+pressure

        # F_eta
            # compute Meta1/2
    u_l = extract.u(Q_l_y_plus_half)
    v_l = extract.v(Q_l_y_plus_half)
    v_eta_l = nx_eta * u_l + ny_eta * v_l
    a_l_y = extract.c(gamma, extract.T(Q_l_y_plus_half))
    M_eta_l_plus = M_plus(v_eta_l / a_l_y)

    u_r = extract.u(Q_r_y_plus_half)
    v_r = extract.v(Q_r_y_plus_half)
    v_eta_r = nx_eta * u_r + ny_eta * v_r
    a_r_y = extract.c(gamma, extract.T(Q_r_y_plus_half))
    M_eta_r_minus = M_minus(v_eta_r / a_r_y)

    M_eta_half = M_eta_l_plus + M_eta_r_minus

    #advection_F = ((np.abs(M_eta_half) >= 0)[:, :, np.newaxis] * phi(Q_l_y_plus_half) + (np.abs(M_eta_half) < 0)[:, :,np.newaxis] * phi(Q_r_y_plus_half))  ##### when we forget to multiply by M_half, it works but when we multiply it doesn't work
    advection_F = 0.5*M_eta_half[:,:,np.newaxis]*(phi(Q_l_y_plus_half)+phi(Q_r_y_plus_half))-0.5*np.abs(M_eta_half)[:,:,np.newaxis]*(phi(Q_r_y_plus_half)-phi(Q_l_y_plus_half))

    # compute the pressure part
    P_l = extract.P(Q_l_y_plus_half)
    P_l_plus = P_plus(P_l, v_eta_l / a_l_y)  ############## vérifier le mach number à utiliser ici
    #P_l_plus = P_plus(P_l, M_eta_half)
    P_r = extract.P(Q_r_y_plus_half)
    P_r_minus = P_minus(P_r, v_eta_r / a_r_y)  ############## vérifier le mach number à utiliser ici
    #P_r_minus = P_minus(P_r, M_eta_half)

    pressure = np.zeros(np.shape(Q_l_y_plus_half))
    pressure[:, :, 1] = nx_eta * (P_l_plus + P_r_minus)  # as here the computation averages at cell face, we use extrapolated direction vector at cell face
    pressure[:, :, 2] = ny_eta * (P_l_plus + P_r_minus)  # as here the computation averages at cell face, we use extrapolated direction vector at cell face

    F_eta = advection_F + pressure # the flux for u or v is certainly too big to have any convergence

    # computing the step of the iteration #
    i_variation = E_xi[:,1:,:]*np.linalg.norm(Sl[1:-1,2:-1,:],axis=2)[:,:,np.newaxis] - E_xi[:,:-1,:]*np.linalg.norm(Sl[1:-1,1:-2,:],axis=2)[:,:,np.newaxis]
    j_variation = F_eta[1:,:,:]*np.linalg.norm(Sb[2:-1,1:-1,:],axis=2)[:,:,np.newaxis] - F_eta[:-1,:,:]*np.linalg.norm(Sb[1:-2,1:-1,:],axis=2)[:,:,np.newaxis]
    step = deltat*(i_variation+j_variation)

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
        plt.quiver(x_V, y_V, extract.u(X), extract.v(X))
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