import numpy as np
import matplotlib.pyplot as plt
import initialisation
import cellproperties
import plotgrid
import readgrid

######################### change boundary conditions with the new functions made(T and P)
######################### at some point we have to operate the transformation physical coordinates -> computational coord and vice versa
######################### what i think at the moment : Q = MUSCLE(Q) -> Qcomp = transfo(Q) -> Qcomp = SW(Qcomp) -> Q = transfo(Qcomp)


######################################################################
####################  data definition  ###############################
######################################################################

# constant thermodynamic data of air
R = 8.31446261815324
MW = 28.96e-3
r = R/MW
gamma = 1.400  # can use NIST data to compute
Cp = 1005  # J/kg/K , can use NIST data to compute

# initial conditions
Tini = 300.0
Pini = 101325.0
Mini = 2.0
rhoini = Pini/r/Tini

######################################################################
###############  extraction function definition  #####################
######################################################################

def c(g,T):
    return(np.sqrt(g*r*T))

def extract_u(Q):
    return(Q[:,:,1]/Q[:,:,0])

def extract_v(Q):
    return(Q[:,:,2]/Q[:,:,0])

def extract_T(Q):
    return(((gamma - 1) * (Q[:, :, 3] - 1 / 2 * (Q[:, :, 1] ** 2 + Q[:, :, 2] ** 2) / Q[:, :, 0])) / r / Q[:, :, 0])


######################################################################
########################  flux limiters  #############################
######################################################################

def van_albada_1_flux_limiter(r):
    return((r**2+r)/(r**2+1))

def van_albada_2_flux_limiter(r):
    return(2*r/(r**2+1))

def lax_wendroff(r):
    return 1

def beam_warming(r):
    return r

######################################################################
#################  left ad right state averaging  ####################
######################################################################

def f(left,right):
    return((left+right)/2)

######################################################################
########################### MUSCL scheme #############################
######################################################################


def MUSCL(X,eps=0,k=-1):
    """
    return the interpolation of cells state on cell faces in the computational domaine (boundary included)
    :param X:
    :param eps:
    :param k:
    :return:
    """
    # for the moment, only first order is working
    # reading of the neighbor cells #
    Q_comp = X[1:-1,1:-1,:]
    Q_x_minus_1 = X[1:-1,0:-2,:]
    Q_x_plus_1 = X[1:-1, 2:, :]
    Q_y_minus_1 = X[0:-2,1:-1,:]
    Q_y_plus_1 = X[2:,1:-1,:]

    Q_l_x_plus_half = X[1:-1,:-1,:]
    Q_r_x_plus_half = X[1:-1,1:,:]
    Q_x_half = f(Q_l_x_plus_half,Q_r_x_plus_half)

    Q_l_y_plus_half = X[:-1,1:-1,:]
    Q_r_y_plus_half = X[1:,1:-1,:]
    Q_y_half = f(Q_l_y_plus_half,Q_r_y_plus_half)

    return(Q_x_half,Q_y_half)


######################################################################
####################  Steger warming scheme  #########################
######################################################################


def FII(X,k1,k2,l1,l3,l4,c,u,v):
    k1tilde = k1/(k1**2+k2**2)**(1/2)
    k2tilde = k2 / (k1 ** 2 + k2 ** 2) ** (1 / 2)
    Fii =np.zeros(np.shape(X))
    wii = (3 - initialisation.gamma) * (l3 + l4) * c ** 2 / (2 * (initialisation.gamma - 1))
    Fii[:, :, 0] = X[:, :, 0] / 2 / initialisation.gamma * (2 * (initialisation.gamma - 1) * l1 + l3 + l4)
    Fii[:, :, 1] = X[:, :, 0] / 2 / initialisation.gamma * (2 * (initialisation.gamma - 1) * l1 * u + l3 * (u + c * k1tilde) + l4 * (u - c * k1tilde))
    Fii[:, :, 2] = X[:, :, 0] / 2 / initialisation.gamma * (2 * (initialisation.gamma - 1) * l1 * v + l3 * (v + c * k2tilde) + l4 * (v - c * k2tilde))
    Fii[:, :, 3] = X[:, :, 0] / 2 / initialisation.gamma * ((initialisation.gamma - 1) * l1 * (u ** 2 + v ** 2) + l3 / 2 * ((u + c * k1tilde) ** 2 + (v + c * k2tilde) ** 2) + l4 / 2 * ((u - c * k1tilde) ** 2 + (v - c * k2tilde) ** 2) + wii)
    return(Fii)

def E_plus(X):
    u = extract_u(X)
    v = extract_v(X)
    T = extract_T(X)
    #T = T - (T<0)*T  # sets negative temperature to 0
    a = c(gamma, T)
    k1 = 1
    k2 = 0
    l1 = k1 * u + k2 * v
    l3 = l1 + a * (k1 ** 2 + k2 ** 2) ** (1 / 2)
    l4 = l1 - a * (k1 ** 2 + k2 ** 2) ** (1 / 2)
    l1_plus = (l1+np.abs(l1))/2
    l3_plus = (l3 + np.abs(l3)) / 2
    l4_plus = (l4 + np.abs(l4)) / 2
    return (FII(X, k1, k2, l1_plus, l3_plus, l4_plus, a, u, v))

def E_minus(X):
    u = extract_u(X)
    v = extract_v(X)
    T = extract_T(X)
    #T = T - (T < 0) * T  # sets negative temperature to 0
    a = c(gamma, T)
    k1 = 1
    k2 = 0
    l1 = k1 * u + k2 * v
    l3 = l1 + a * (k1 ** 2 + k2 ** 2) ** (1 / 2)
    l4 = l1 - a * (k1 ** 2 + k2 ** 2) ** (1 / 2)
    l1_minus = (l1 - np.abs(l1)) / 2
    l3_minus = (l3 - np.abs(l3)) / 2
    l4_minus = (l4 - np.abs(l4)) / 2
    return (FII(X, k1, k2, l1_minus, l3_minus, l4_minus, a, u, v))

def F_plus(X):
    u = extract_u(X)
    v = extract_v(X)
    T = extract_T(X)
    #T = T - (T < 0) * T  # sets negative temperature to 0
    a = c(gamma, T)
    k1 = 0
    k2 = 1
    l1 = k1 * u + k2 * v
    l3 = l1 + a * (k1 ** 2 + k2 ** 2) ** (1 / 2)
    l4 = l1 - a * (k1 ** 2 + k2 ** 2) ** (1 / 2)
    l1_plus = (l1 + np.abs(l1)) / 2
    l3_plus = (l3 + np.abs(l3)) / 2
    l4_plus = (l4 + np.abs(l4)) / 2
    return (FII(X, k1, k2, l1_plus, l3_plus, l4_plus, a, u, v))

def F_minus(X):
    u = extract_u(X)
    v = extract_v(X)
    T = extract_T(X)
    #T = T - (T < 0) * T  # sets negative temperature to 0
    a = c(gamma, T)
    k1 = 0
    k2 = 1
    l1 = k1 * u + k2 * v
    l3 = l1 + a * (k1 ** 2 + k2 ** 2) ** (1 / 2)
    l4 = l1 - a * (k1 ** 2 + k2 ** 2) ** (1 / 2)
    l1_minus = (l1 - np.abs(l1)) / 2
    l3_minus = (l3 - np.abs(l3)) / 2
    l4_minus = (l4 - np.abs(l4)) / 2
    return (FII(X, k1, k2, l1_minus, l3_minus, l4_minus, a, u, v))


######################################################################
################  Initialisation and boundaries  #####################
######################################################################

def initialise(nx,ny,bottom_surfaces):
    """
    sets all the interior points at the initial condition given in the project assignment
    :param nx, ny: (int) size of the augmented grid which is not the number of cells
    :param bottom_surfaces: (3D array) matrix containing the top and bottom surface vectors
    :return:(3D array) Q
    """
    ########## when changing initial conditions with rho/2 and P/2, does not eveniterate 1 time, the computation of c instatly gives a nan, this could be the source of the error

    Q = np.zeros((ny-1,nx-1,4))  # initialasing Q vector
    S_mean = (bottom_surfaces[:-1,:,:]+bottom_surfaces[1:,:,:])/2 # computing the interpolation of surface directions at the cells center
    S_norm = np.linalg.norm(S_mean,axis=2)
    #  initial set in contravariant velocity
    Q[:, :, 0] += rhoini
    Q[:, :, 1] += rhoini*Mini*c(gamma,Tini)#-rhoini*Mini*c(gamma,Tini)*S_mean[:,:,1]/S_norm  # speed vector times xi direction
    Q[:, :, 2] += 0#rhoini*Mini*c(gamma,Tini)*(S_mean[:,:,0])/S_norm  # speed vector times eta direction
    Q[:, :, 3] += Pini/(gamma-1) + rhoini*(Mini*c(gamma,Tini))**2/2

    return(Q)


def bc(X,bottom_surfaces):
    X[:, 0, 0] = rhoini
    X[:, 0, 1] = rhoini * Mini * c(gamma, Tini)
    X[:, 0, 2] = 0
    X[:, 0, 3] = Pini / (gamma - 1) + rhoini * (Mini * c(gamma, Tini)) ** 2 / 2


    S_norm = np.linalg.norm(bottom_surfaces, axis=2)

    X[:, -1, :] = X[:, -2, :]  # setting all the hallo cells on the right to the value of the last column of computed cells

    # slip wall condition #
    # top wall
    #this approximation is fine as the top wall is flat, but in a second time it could be verified mor rigorously
    X[-1, 1:-1, 1] = X[-1, 1:-1, 0] * X[-2, 1:-1,1] / X[-2, 1:-1,0]  # slip wall (the boudary condition is actualy computed only over the computational domaine)
    X[-1, 1:-1, 2] = - X[-1, 1:-1, 0] * X[-2, 1:-1,2] / X[-2, 1:-1,0] # slip wall (the boudary condition is actualy computed only over the computational domaine)

    # bottom wall
    #the two following conditions com from the lesson. removing them for having u vector paralel to the boundary seems to give slightly better results
    X[0, 1:-1, 1] = X[0, 1:-1, 0]/(S_norm[1,1:-1]**2)* ( (X[1, 1:-1, 1] / X[1, 1:-1, 0])*(bottom_surfaces[1,1:-1,1]**2-bottom_surfaces[1,1:-1,0]**2) - 2*bottom_surfaces[1,1:-1,1]*bottom_surfaces[1,1:-1,0]*(X[1, 1:-1, 2] / X[1, 1:-1, 0]) )
    X[0, 1:-1, 2] = X[0, 1:-1, 0] / (S_norm[1, 1:-1] ** 2) * ((X[1, 1:-1, 2] / X[1, 1:-1, 0]) * (bottom_surfaces[1, 1:-1, 0] ** 2 - bottom_surfaces[1, 1:-1, 1] ** 2) - 2 * bottom_surfaces[1, 1:-1,1] * bottom_surfaces[1, 1:-1,0] * (X[1, 1:-1, 1] / X[1, 1:-1, 0]))

    # adiabatic condition # (to have adiabaticity, we need delta T = 0 as stated by fourier's law
    ############################### if we change the boundary condition for isoenergetic, the result change completly, so that has an impact somewhere
    # top wall
    X[-1, :, 3] = r * X[-1, :, 0] / (gamma - 1) * ((gamma - 1) * (X[-2, :, 3] - 1 / 2 * (X[-2, :, 1] ** 2 + X[-2, :, 2] ** 2) / X[-2, :, 0]) / r / X[-2, :, 0]) + 1 / 2 * (X[-1, :, 1] ** 2 + X[-1, :, 2] ** 2) / X[-1, :, 0]
    # bottom wall
    X[0, :, 3] = r * X[0, :, 0] / (gamma - 1) * ((gamma - 1) * (X[1, :, 3] - 1 / 2 * (X[1, :, 1] ** 2 + X[1, :, 2] ** 2) / X[1, :, 0]) / r / X[1, :, 0]) + 1 / 2 * (X[0, :, 1] ** 2 + X[0, :, 2] ** 2) / X[0, :, 0]

    # 0 pressure gradient at walls #
    # bottom wall
    X[0, -1:1, 3] = (X[1, -1:1, 3] - 0.5 * (X[1, -1:1, 1] ** 2 + X[1, -1:1, 2] ** 2) / X[1, -1:1, 0]) + 0.5 * (
                X[0, -1:1, 1] ** 2 + X[0, -1:1, 2] ** 2) / X[0, -1:1, 0]
    # bottom wall
    X[-1, -1:1, 3] = (X[-2, -1:1, 3] - 0.5 * (X[-2, -1:1, 1] ** 2 + X[-2, -1:1, 2] ** 2) / X[-2, -1:1, 0]) + 0.5 * (
                X[-1, -1:1, 1] ** 2 + X[-1, -1:1, 2] ** 2) / X[-1, -1:1, 0]

    return(X)


def time_step(X,deltat,V,Sb,Sl,plot):
    ############################## steger warming est défini en carthésien donc on doit transformer Q en carthésien (computational) avant de le donner à S-W

    ###############################################################
    ##################  schéma à adopter:  ########################
    ###############################################################
    #SUIVRE LES NOTES 26.0 et les combiner avec le rho scheme
    #pour évaluer Ahat en Qhat, il faut Qhat. pour ça il faut utiliser le rho average

    # COMMENCER AVEC LE RHO SCHEME
    # Qleft,Qright = MUSCL(Q)
    # calculer Ahat en Qhat
    # en déduire E

    # compute the surfaces areas to use in the fluxes calculation (we use the projections on x and y, this could be confirmed) #
    # extrapolate Q on cell edges with MUSCL scheme #
    Q_x_half,Q_y_half = MUSCL(X)
    # Use extrapolated Q to compute E and F as given in the Steger-Warming paper #
    # no transformation in the computational domaine is made here

    E_minus_x_plus_half = E_minus(Q_x_half[:,1:,:])
    E_plus_x_plus_half = E_plus(Q_x_half[:,1:,:])
    F_minus_x_plus_half = F_minus(Q_x_half[:,1:,:])
    F_plus_x_plus_half = F_plus(Q_x_half[:, 1:, :])

    E_minus_x_minus_half = E_minus(Q_x_half[:,:-1,:])
    E_plus_x_minus_half = E_plus(Q_x_half[:,:-1,:])
    F_minus_x_minus_half = F_minus(Q_x_half[:,:-1,:])
    F_plus_x_minus_half = F_plus(Q_x_half[:,:-1,:])

    F_minus_y_plus_half = F_minus(Q_y_half[1:,:,:])
    F_plus_y_plus_half = F_plus(Q_y_half[1:,:,:])
    E_minus_y_plus_half = E_minus(Q_y_half[1:,:,:])
    E_plus_y_plus_half = E_plus(Q_y_half[1:,:,:])

    F_minus_y_minus_half = F_minus(Q_y_half[:-1,:,:])
    F_plus_y_minus_half = F_plus(Q_y_half[:-1,:,:])
    E_minus_y_minus_half = E_minus(Q_y_half[1:,:,:])
    E_plus_y_minus_half = E_plus(Q_y_half[1:,:,:])

    #computing the step of the iteration with a steger warming scheme
    #we have to be carefull on how surfaces are computed. We write the code as it would be one paper by taking the scalar product with surfaces
    x_variation = (E_plus_x_plus_half+E_minus_x_plus_half)*(-Sl[1:-1,2:-1,0])[:,:,np.newaxis] + (E_plus_x_minus_half+E_minus_x_minus_half)*(Sl[1:-1,1:-2,0])[:,:,np.newaxis] + (E_plus_y_plus_half+E_minus_y_plus_half)*(-Sb[2:-1,1:-1,0])[:,:,np.newaxis] + (E_plus_y_minus_half+E_minus_y_minus_half)*(Sb[1:-2,1:-1,0])[:,:,np.newaxis]
    y_variation = (F_plus_x_plus_half+F_minus_x_plus_half)*(-Sl[1:-1,2:-1,1])[:,:,np.newaxis] + (F_plus_x_minus_half+F_minus_x_minus_half)*(Sl[1:-1,1:-2,1])[:,:,np.newaxis] + (F_plus_y_plus_half + F_minus_y_plus_half)*(-Sb[2:-1,1:-1,1])[:,:,np.newaxis] + + (F_plus_y_minus_half+F_minus_y_minus_half)*(Sb[1:-2,1:-1,1])[:,:,np.newaxis]
    #x_variation = (E_plus_x_plus_half+E_minus_x_plus_half)*Sx_plus_half[:,:,np.newaxis]-(E_plus_x_minus_half+E_minus_x_minus_half)*Sx_minus_half[:,:,np.newaxis]
    #y_variation = (F_plus_y_plus_half + F_minus_y_plus_half)*Sy_plus_half[:,:,np.newaxis] - (F_plus_y_minus_half + F_minus_y_minus_half)*Sy_minus_half[:,:,np.newaxis]
    step = deltat*(x_variation+y_variation)/V[1:-1,1:-1,np.newaxis]
    #apply the step
    X[1:-1, 1:-1, :] = X[1:-1, 1:-1, :] - step
    #reset the boundary conditions
    X = bc(X, bottom_Surfaces)
    if plot:
        plt.contourf(x_V, y_V, X[:, :, 0], levels=20, cmap='jet')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Density')
        plt.show()
    return(X)

def solve(nx,ny,V,Sb,Sl):
    ########### by varying delta t we can see that the simulation diverges allways at the exact same moment so the problem, i guess, is not that there is an instability growing due to numerical scheme but that ther is a reall problem in the formulas
    nb_iter = 0
    plot = 0
    Q = initialise(nx,ny,Sb)
    Q = bc(Q, Sb)
    Q0 = np.copy(Q)
    Q = time_step(Q, 0.000001,V,Sb,Sl,plot)
    continuity_residual = np.sqrt(np.sum((Q[:,:,0]-Q0[:,:,0])**2))
    print(continuity_residual)
    while continuity_residual > 1e-3:  # the right condition is here to automaticaly stop in case of divergence
        Q0 = np.copy(Q)
        Q = time_step(Q, 0.000001,V,Sb,Sl,plot)
        continuity_residual = np.sqrt(np.sum((Q[:, :, 0] - Q0[:, :, 0]) ** 2))
        print(continuity_residual)
        nb_iter += 1
        if nb_iter > 1000:
            plot = 1
            nb_iter = 0
        else:
            plot = 0
    plt.contourf(x_V, y_V, Q[:, :, 0], levels=20, cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Density')
    plt.show()
    plt.contourf(x_V, y_V, (initialisation.gamma - 1) * (Q[:, :, 3] - Q[:, :, 1] ** 2 / Q[:, :, 0] / 2), levels=20, cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.quiver(x_V, y_V, Q[:, :, 1] / Q[:, :, 0], Q[:, :, 2] / Q[:, :, 0])
    plotgrid.gridePlotting('../../../grids','g33x25u_augmented.dat')
    plt.title('Pressure')
    plt.show()
    plt.contourf(x_V, y_V, Q[:, :, 1], levels=20, cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Density*u')
    plt.show()
    plt.contourf(x_V, y_V, Q[:, :, 2], levels=20, cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Density*v')
    plt.show()
    plt.contourf(x_V, y_V, Q[:, :, 3], levels=20, cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Density*et')
    plt.show()
    plt.contourf(x_V, y_V, ((initialisation.gamma - 1) * (Q[:, :, 3] - 1 / 2 * (Q[:, :, 1] ** 2 + Q[:, :, 2] ** 2) / Q[:, :, 0])) / initialisation.r / Q[:, :, 0], levels=20, cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('T')
    plt.show()


if __name__ == '__main__':
    nx, ny, x, y = readgrid.readGrid('../../../grids', 'g33x25u_augmented.dat')
    Volumes, x_V, y_V, bottom_Surfaces, x_Sb, y_Sb, left_Surfaces, x_Sl, y_Sl = cellproperties.cellsCharacteristics('../../../grids', 'g33x25u_augmented.dat')
    #Q = initialisation.initialise_computational_space(nx, ny, bottom_Surfaces)
    #time_step(Q,0.0001,x_Sl,y_Sb,bottom_Surfaces)
    solve(nx,ny,Volumes,bottom_Surfaces,left_Surfaces)