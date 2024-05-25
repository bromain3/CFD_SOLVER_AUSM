######################################################################
########################  flux limiters  #############################
######################################################################
import numpy as np

def minmod(r):
    min= 1*(r>=1) + (r<1)*r
    max = 0*(min<=0) + (min>0)*min
    return(max)

def van_albada_1_flux_limiter(r):
    return((r**2+r)/(r**2+1))

def van_albada_2_flux_limiter(r):
    return(2*r/(r**2+1))

def lax_wendroff(r):
    return 1

def beam_warming(r):
    return r


def MUSCL(X,eps=0,k=-1,phi=lax_wendroff):
    """
    Return the interpolation of cells state on cell faces in the computational domain (boundary included)
    :param X: matrix storing the state vectors (3D-array)
    :param eps: epsilon parameter of the MUSCL scheme as described in the lecture notes
    :param k: k parameter of the MUSCL scheme as described in the lecture notes
    :return: interpolated right and left state vectors (3D-arrays)
    """
    # for the moment, only first order is working
    # reading of the neighbor cells #
    bw_x = X[1:-1,1:-1,:]-X[1:-1,:-2,:]
    fw_x = X[1:-1,2:,:]-X[1:-1,1:-1,:]
    bw_x = (bw_x!=0)*bw_x + (bw_x==0)*1e-16
    fw_x = (fw_x!=0)*fw_x + (fw_x==0)*1e-16
    Q_l_x_plus_half = X[1:-1,1:-1,:] + eps/4*((1-k)*phi(fw_x/bw_x)*(X[1:-1,1:-1,:]-X[1:-1,:-2,:])+(1+k)*phi(bw_x/fw_x)*(X[1:-1,2:,:]-X[1:-1,1:-1,:]))  # forgetting the leftmost cells
    Q_l_x_plus_half = np.hstack(((X[1:-1,0,:])[:,np.newaxis,:],Q_l_x_plus_half))
    Q_r_x_plus_half = X[1:-1,1:-1,:] - eps/4*((1+k)*phi(fw_x/bw_x)*(X[1:-1,1:-1,:]-X[1:-1,:-2,:])+(1-k)*phi(bw_x/fw_x)*(X[1:-1,2:,:]-X[1:-1,1:-1,:]))  # forgetting the rightmost cells
    Q_r_x_plus_half = np.hstack((Q_r_x_plus_half,(X[1:-1,-1,:])[:,np.newaxis,:]))

    bw_y = X[1:-1,1:-1,:]-X[:-2,1:-1,:]
    fw_y = X[2:,1:-1,:]-X[1:-1,1:-1,:]
    bw_y = (bw_y != 0) * bw_y + (bw_y == 0) * 1e-16
    fw_y = (fw_y != 0) * fw_y + (fw_y == 0) * 1e-16
    Q_l_y_plus_half = X[1:-1,1:-1,:] + eps/4*((1-k)*phi(fw_y/bw_y)*(X[1:-1,1:-1,:]-X[:-2,1:-1,:])+(1+k)*phi(bw_y/fw_y)*(X[2:,1:-1,:]-X[1:-1,1:-1,:]))  # forgetting the bottom cells
    Q_l_y_plus_half = np.vstack(((X[0,1:-1,:])[np.newaxis,:,:],Q_l_y_plus_half))
    Q_r_y_plus_half = X[1:-1,1:-1,:] - eps/4*((1+k)*phi(fw_y/bw_y)*(X[1:-1,1:-1,:]-X[:-2,1:-1,:])+(1+k)*phi(bw_y/fw_y)*(X[2:,1:-1,:]-X[1:-1,1:-1,:]))  # forgetting the top cells
    Q_r_y_plus_half = np.vstack((Q_r_y_plus_half,(X[-1,1:-1,:])[np.newaxis,:,:]))


    return(Q_l_x_plus_half,Q_r_x_plus_half,Q_l_y_plus_half,Q_r_y_plus_half)


def MUSCL_V1(X,eps=0,k=-1):
    """
    Return the interpolation of cells state on cell faces in the computational domain (boundary included)
    :param X: matrix storing the state vectors (3D-array)
    :param eps: epsilon parameter of the MUSCL scheme as described in the lecture notes
    :param k: k parameter of the MUSCL scheme as described in the lecture notes
    :return: interpolated right and left state vectors (3D-arrays)
    """
    # for the moment, only first order is working
    # reading of the neighbor cells #

    Q_l_x_plus_half = X[1:-1,:-1,:]
    Q_r_x_plus_half = X[1:-1,1:,:]

    Q_l_y_plus_half = X[:-1,1:-1,:]
    Q_r_y_plus_half = X[1:,1:-1,:]

    return(Q_l_x_plus_half,Q_r_x_plus_half,Q_l_y_plus_half,Q_r_y_plus_half)