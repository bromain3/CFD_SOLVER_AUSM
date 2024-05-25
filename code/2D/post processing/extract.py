import numpy as np
import matplotlib.pyplot as plt
import cellproperties
import plotgrid
import readgrid
from air import *


"""
PROBABLE SOURCE DE L'ERREUR DU CODE : LES FONCTIONS POUR DONNER LES DIRECTION VECTORS DONNE TOUT DANS UNE SEULE DIRECTION. C'EST SUREMENT PROBLéMATIQUE
"""





######################################################################
####################  physical data extraction  ######################
######################################################################

def c(g,T):
    """
    Returns the value of speed of sound
    :param g: gamma at the points considered (float or 2D-array)
    :param T: temperature at a given point (float or 2D-array)
    :return: speed of sound (float or 2D-array)
    """
    return(np.sqrt(g*r*T))

def u(Q):
    """
    Extract the value of u form Q
    :param Q: array of state vectors (3D-array)
    :return: array containing the value of u (2D-array)
    """
    return(Q[:,:,1]/Q[:,:,0])

def v(Q):
    """
    Extract the value of v form Q
    :param Q: array of state vectors (3D-array)
    :return: array containing the value of v (2D-array)
    """
    return(Q[:,:,2]/Q[:,:,0])

def et(Q):
    """
    Extract the value of et form Q
    :param Q: array of state vectors (3D-array)
    :return: array containing the value of et (2D-array)
    """
    return (Q[:, :, 3] / Q[:, :, 0])

def P(Q):
    """
    Extract the value of P form Q
    :param Q: array of state vectors (3D-array)
    :return: array containing the value of P (2D-array)
    """
    Et = et(Q)
    U = u(Q)
    V = v(Q)
    return((gamma-1)*Q[:,:,0]*(Et-0.5*(U**2+V**2)))

def ht(Q):
    """
    Extract the value of ht form Q
    :param Q: array of state vectors (3D-array)
    :return: array containing the value of ht (2D-array)
    """
    Et = et(Q)
    p = P(Q)
    return(Et+p/Q[:, :, 0])

def T(Q):
    """
    Extract the value of T form Q
    :param Q: array of state vectors (3D-array)
    :return: array containing the value of T (2D-array)
    """
    return(((gamma - 1) * (Q[:, :, 3] - 1 / 2 * (Q[:, :, 1] ** 2 + Q[:, :, 2] ** 2) / Q[:, :, 0])) / r / Q[:, :, 0])

def M(Q):
    U = u(Q)
    V = v(Q)
    a = c(gamma,T(Q))
    return np.sqrt(U**2+V**2)/a

def P0(Q):
    p = P(Q)
    m = M(Q)
    return (p*(1+(gamma-1)/2*m**2)**(gamma/(gamma-1)))

######################################################################
###################  geometric data extraction  ######################
######################################################################

def direction_vectors(Sb,Sl):
    """
    Extrapolate the value of xi and eta in the middle of the cell
    :param Sb: Surface vectors of surfaces parallel to xi of the augmented grid (3D-array)
    :param Sl: Surface vectors of surfaces parallel to eta of the augmented grid (3D-array)
    :return: interpolated direction vectors components (2D-array)
    """
    # interpolation of surface vectors on centers of surfaces along i #
    Sb_mean = (Sb[:-1,:,:]+Sb[1:,:,:])/2
    Sb_mean_norm = np.linalg.norm(Sb_mean,axis=2)
    nx_xi = -Sb_mean[:,:,1]/Sb_mean_norm
    ny_xi = Sb_mean[:,:,0]/Sb_mean_norm

    Sl_mean = (Sl[:,0:-1,:]+Sl[:,1:,:])/2
    Sl_norm = np.linalg.norm(Sl_mean,axis = 2)
    nx_eta = Sl_mean[:,:,1]/Sl_norm
    ny_eta = -Sl_mean[:,:,0]/Sl_norm
    return(nx_xi,ny_xi,nx_eta,ny_eta)

def direction_vectors_i(Sb,Sl):
    """
    Extrapolate the value of xi and eta in the middle of vertical surfaces
    :param Sb: Surface vectors of surfaces parallel to xi of the augmented grid (3D-array)
    :param Sl: Surface vectors of surfaces parallel to eta of the augmented grid (3D-array)
    :return: interpolated direction vectors components (2D-array)
    """
    # interpolation of surface vectors on centers of surfaces along i #
    Sl_norm = np.linalg.norm(Sl[1:-1,1:-1,:],axis = 2)
    nx_xi = -Sl[1:-1,1:-1,0] / Sl_norm
    ny_xi = -Sl[1:-1,1:-1,1] / Sl_norm

    return(nx_xi,ny_xi)

def direction_vectors_j(Sb,Sl):
    """
    Extrapolate the value of xi and eta in the middle of horizontal surfaces
    :param Sb: Surface vectors of surfaces parallel to xi of the augmented grid (3D-array)
    :param Sl: Surface vectors of surfaces parallel to eta of the augmented grid (3D-array)
    :return: interpolated direction vectors components (2D-array)
    """
    # interpolation of surface vectors on centers of surfaces along j #
    Sb_norm = np.linalg.norm(Sb[1:-1,1:-1,:],axis = 2)
    nx_eta = -Sb[1:-1,1:-1,0] / Sb_norm
    ny_eta = -Sb[1:-1, 1:-1, 1] / Sb_norm

    return (nx_eta, ny_eta)

######################################################################
######################  residuals extraction  ########################
######################################################################

def residuals_L2(X,X0,Xini):
    """
    Computes the residuals in norm 2
    :param X: actual state vector (3D-array)
    :param X0: previous state vector (3D-array)
    :param Xini: initial state vector used for normalization (3D-array)
    :return: the value of residuals for each component (float)
    """
    rho_v=np.sqrt(Xini[:,:,1]**2+Xini[:,:,2]**2)
    cr = np.sqrt(np.sum(((X[:, :, 0] - X0[:, :, 0])/Xini[:,:,0]) ** 2))
    xr = np.sqrt(np.sum(((X[:, :, 1] - X0[:, :, 1])/rho_v) ** 2))
    yr = np.sqrt(np.sum(((X[:, :, 2] - X0[:, :, 2])/rho_v) ** 2))
    er = np.sqrt(np.sum(((X[:, :, 3] - X0[:, :, 3])/Xini[:,:,3]) ** 2))
    return(cr, xr, yr, er)

def residuals_Linf(X,X0,Xini):
    """
    Computes the residuals in norm inf
    :param X: actual state vector (3D-array)
    :param X0: previous state vector (3D-array)
    :param Xini: initial state vector used for normalization (3D-array)
    :return: the value of residuals for each component (float)
    """
    rho_v = np.sqrt(Xini[:, :, 1] ** 2 + Xini[:, :, 2] ** 2)
    cr = np.max((X[:, :, 0] - X0[:, :, 0])/Xini[:,:,0])
    xr = np.max((X[:, :, 1] - X0[:, :, 1])/rho_v)
    yr = np.max((X[:, :, 2] - X0[:, :, 2])/rho_v)
    er = np.max((X[:, :, 3] - X0[:, :, 3])/Xini[:,:,3])
    return(cr, xr, yr, er)

if __name__ == "__main__":
    nx, ny, x, y = readgrid.readGrid('../../grids', 'g33x25u_augmented.dat')
    Volumes, x_V, y_V, bottom_Surfaces, x_Sb, y_Sb, left_Surfaces, x_Sl, y_Sl = cellproperties.cellsCharacteristics(
        '../../grids', 'g33x25u_augmented.dat')
    nx_xi, ny_xi = direction_vectors_i(bottom_Surfaces, left_Surfaces)
    plt.xlabel('x')
    plt.ylabel('y')
    print(np.all((nx_xi ** 2 + ny_xi ** 2)==1))  # testing that vectors are of norm 1 (they are plus or minus computation errors)
    plt.quiver(x_Sl[1:-1, 1:-1], y_Sl[1:-1, 1:-1], nx_xi, ny_xi)
    plotgrid.gridePlotting('../../grids', 'g33x25u_augmented.dat')
    plt.show()

    nx_eta, ny_eta = direction_vectors_j(bottom_Surfaces, left_Surfaces)
    plt.xlabel('x')
    plt.ylabel('y')
    print(np.all((nx_eta ** 2 + ny_eta ** 2)==1))  # testing that vectors are of norm 1 (they are plus or minus computation errors)
    plt.quiver(x_Sb[1:-1, 1:-1], y_Sb[1:-1, 1:-1], nx_eta, ny_eta)
    plotgrid.gridePlotting('../../grids', 'g33x25u_augmented.dat')
    plt.show()
