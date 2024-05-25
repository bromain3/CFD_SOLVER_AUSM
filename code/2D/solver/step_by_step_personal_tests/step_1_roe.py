import numpy as np
import matplotlib.pyplot as plt
import initialisation
import cellproperties
import plotgrid
import readgrid

######################### change boundary conditions with the new functions made(T and P)
######################### at some point we have to operate the transformation physical coordinates -> computational coord and vice versa
######################### what i think at the moment : Q = MUSCLE(Q) -> Qcomp = transfo(Q) -> Qcomp = SW(Qcomp) -> Q = transfo(Qcomp)
######################### no propagation in eta direction

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

def extract_et(Q):
    return (Q[:, :, 3] / Q[:, :, 0])

def extract_P(Q):
    et = extract_et(Q)
    u = extract_u(Q)
    v = extract_v(Q)
    return((gamma-1)*Q[:,:,0]*(et-0.5*(u**2+v**2)))

def extract_ht(Q):
    et = extract_et(Q)
    P = extract_P(Q)
    return(et+P/Q[:, :, 0])

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

    Q_l_x_plus_half = X[1:-1,:-1,:]
    Q_r_x_plus_half = X[1:-1,1:,:]
    #Q_x_half = f(Q_l_x_plus_half,Q_r_x_plus_half)

    Q_l_y_plus_half = X[:-1,1:-1,:]
    Q_r_y_plus_half = X[1:,1:-1,:]
    #Q_y_half = f(Q_l_y_plus_half,Q_r_y_plus_half)

    return(Q_l_x_plus_half,Q_r_x_plus_half,Q_l_y_plus_half,Q_r_y_plus_half)


######################################################################
##########################  Roe scheme  ##############################
######################################################################

def R_eig(u,v,u_xi,a,nx,ny,ht):
    res = np.array([
        [1,1,1,0],
        [u - a * nx, u, u + a * nx, ny],
        [v - a * ny, v, v + a * ny, -nx],
        [ht - a * u_xi, 0.5 * (u ** 2 + v ** 2), ht + a * u_xi, u * ny - v * nx]
    ])
    return(res)

def L_eig(u,v,u_xi,a,nx,ny):
    pass
    #phi = 0.5*(gamma-1)*(u**2+v**2)
    #res = np.array([
    #    [(phi + a * u_xi) / 2 / a ** 2, ((1 - gamma) * u - a * nx) / 2 / a ** 2, ((1 - gamma) * v - a * ny) / 2 / a ** 2, (gamma - 1) / 2 / a ** 2],
    #    [(a ** 2 - phi) / a ** 2, ((gamma - 1) * u) / a ** 2, ((gamma - 1) * v) / a ** 2, (1 - gamma) / 2 * a ** 2],
    #    [(phi - a * u_xi) / 2 / a ** 2, ((1 - gamma) * u + a * nx) / 2 / a ** 2, ((1 - gamma) * v + a * ny) / 2 / a ** 2, (gamma - 1) / 2 / a ** 2],
    #    [(v-u_xi*ny)/nx,ny,(ny**2-1)/nx,0]
    #])
    #return(res)

def A(u,v,u_xi,nx,ny,et,V):
    phi = (gamma-1)/2*(u**2+v**2)
    res = 1/V * np.array([
        [0,nx,ny,0],
        [-u*u_xi+phi*nx,u_xi+(2-gamma)*u*nx,u*ny-(gamma-1)*v*nx,(gamma-1)*nx],
        [-v*u_xi+phi*ny,v*nx-(gamma-1)*u*ny,u_xi+(2-gamma)*v*ny,(gamma-1)*ny],
        [(2*phi-gamma*et)*u_xi,(gamma*et-phi)*nx-(gamma-1)*u*u_xi,(gamma*et-phi)*ny-(gamma-1)*v*u_xi,gamma*u_xi]
    ])
    return(res)

def abs_A(X,nx,ny,V):
    y,x,z = np.shape(X)
    jac = np.zeros((y,x,z,z))
    u = extract_u(X)
    v = extract_v(X)
    u_xi = nx*u+ny*v
    ht = extract_ht(X)
    et = extract_et(X)
    T = extract_T(X)
    a = c(gamma,T)
    for j in range(y):
        for i in range(x):
            # computer diagonalisation
            w,R = np.linalg.eig(A(u[j,i],v[j,i],u_xi[j,i],nx[j,i],ny[j,i],et[j,i],V[j,i]))
            D = np.diag(np.abs(w))
            jac[j, i, :, :] = np.linalg.inv(R) @ D @ R
            # hand made diagonalisation
            #D = np.diag([np.abs(u_xi[j,i]-a[j,i]),np.abs(u_xi[j,i]),np.abs(u_xi[j,i]+a[j,i]),np.abs(u_xi[j,i])])
            #jac[j,i,:,:] = np.linalg.inv(R_eig(u[j,i],v[j,i],u_xi[j,i],a[j,i],nx[j,i],ny[j,i],ht[j,i])) @ D @ R_eig(u[j,i],v[j,i],u_xi[j,i],a[j,i],nx[j,i],ny[j,i],ht[j,i]) / V[j,i]
    return(jac)

def roe_avg(Xr,Xl):
    avg = np.zeros(np.shape(Xr))
    ur = extract_u(Xr)
    ul = extract_u(Xl)
    vr = extract_v(Xr)
    vl = extract_v(Xl)
    rhol = Xl[:,:,0]
    rhor = Xr[:,:,0]
    htr = extract_ht(Xr)
    htl = extract_ht(Xl)
    rho_hat = np.sqrt(rhol*rhor)
    u_hat = (np.sqrt(rhor)*ur+np.sqrt(rhol)*ul)/(np.sqrt(rhor)+np.sqrt(rhol))
    v_hat = (np.sqrt(rhor) * vr + np.sqrt(rhol) * vl) / (np.sqrt(rhor) + np.sqrt(rhol))
    ht_hat = (np.sqrt(rhor) * htr + np.sqrt(rhol) * htl) / (np.sqrt(rhor) + np.sqrt(rhol))
    et_hat = (ht_hat+(gamma-1)/2*(u_hat**2+v_hat**2))/gamma
    avg[:,:,0] = rho_hat
    avg[:,:,1] = rho_hat*u_hat
    avg[:, :, 2] = rho_hat * v_hat
    avg[:, :, 3] = rho_hat * et_hat
    return(avg)

def flux_hat(X,nx,ny):
    res = np.zeros(np.shape(X))
    q1 = X[:,:,0]
    q2 = X[:,:,1]
    q3 = X[:, :, 2]
    q4 = X[:, :, 3]
    q22_q1 = q2*q2/q1
    q2q3_q1 = q2*q3/q1
    q32_q1 = q3*q3/q1
    q2_q1 = q2/q1
    q3_q1 = q3/q1
    res[:,:,0] = q2*nx+q3*ny
    res[:,:,1] = q22_q1*nx+q2q3_q1*ny+(gamma-1)*(q4-0.5*(q22_q1+q32_q1))*nx
    res[:,:,2] = q2q3_q1*nx + q32_q1*ny + (gamma-1)*(q4-0.5*(q22_q1+q32_q1))*ny
    res[:,:,3] = (gamma*q4 - 0.5*(gamma-1)*(q22_q1+q32_q1))*(q2_q1*nx+q3_q1*ny)
    return(res)

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

def direction_vectors_i(Sb,Sl):
    # works with raw Sl and Sb on the augmented grid
    # interpolation of surface vectors on centers of surfaces along i #
    ######## seems that the xi vector is a little bit off
    Sb_mean = (Sb[1:-2,:,:]+Sb[2:-1,:,:])/2
    Sb_mean = (Sb_mean[:,:-1,:] + Sb_mean[:,1:,:])/2
    Sb_mean_norm = np.linalg.norm(Sb_mean,axis=2)
    nx_xi = -Sb_mean[:,:,1]/Sb_mean_norm
    ny_xi = Sb_mean[:,:,0]/Sb_mean_norm

    Sl_norm = np.linalg.norm(Sl[1:-1,1:-1],axis = 2)
    nx_eta = Sl[1:-1,1:-1,1]/Sl_norm
    ny_eta = -Sl[1:-1, 1:-1, 0]/Sl_norm
    return(nx_xi,ny_xi,nx_eta,ny_eta)

def direction_vectors_j(Sb,Sl):
    # works with raw Sl and Sb on the augmented grid
    # interpolation of surface vectors on centers of surfaces along i #
    ######## seems that the xi vector is a little bit off
    Sb_norm = np.linalg.norm(Sb[1:-1,1:-1],axis=2)
    nx_xi = -Sb[1:-1,1:-1,1]/Sb_norm
    ny_xi = Sb[1:-1,1:-1,0]/Sb_norm

    Sl_mean = (Sl[:,1:-2,:]+Sl[:,2:-1,:])/2
    Sl_mean = (Sl_mean[:-1,:,:]+Sl_mean[1:,:,:])/2
    Sl_mean_norm = np.linalg.norm(Sl_mean,axis=2)
    nx_eta = Sl_mean[:,:,1]/Sl_mean_norm
    ny_eta = -Sl_mean[:,:,0]/Sl_mean_norm
    return (nx_xi, ny_xi, nx_eta, ny_eta)

def V_average_i(V):
    return((V[1:-1,:-1]+V[1:-1,1:])/2)

def V_average_j(V):
    return ((V[:-1, 1:-1] + V[1:, 1:-1]) / 2)


def time_step(X,deltat,V,Sb,Sl,plot):
    # extrapolate Q on cell edges with MUSCL scheme #
    Q_l_x_plus_half,Q_r_x_plus_half,Q_l_y_plus_half,Q_r_y_plus_half = MUSCL(X)

    # Use extrapolated Q to compute E and F as given in the Roe scheme #
    # E_xi
    nx_xi,ny_xi,nx_eta,ny_eta = direction_vectors_i(Sb,Sl)
    Q_roe_avg = roe_avg(Q_r_x_plus_half,Q_l_x_plus_half)
    jac = abs_A(Q_roe_avg,nx_xi,ny_xi,V_average_i(V))
        # matrix multiplication
    Q_dif = Q_r_x_plus_half-Q_l_x_plus_half
    y,x,z = np.shape(Q_dif)
    res = np.zeros((y,x,z))
    for j in range(y):
        for i in range(x):
            res[j,i] = jac[j,i] @ Q_dif[j,i]
    E_xi_plus_half = 0.5*(flux_hat(Q_l_x_plus_half,nx_xi,ny_xi) + flux_hat(Q_r_x_plus_half,nx_xi,ny_xi)) - 0.5*res

    # F_eta
    nx_xi, ny_xi, nx_eta, ny_eta = direction_vectors_j(Sb, Sl)
    Q_roe_avg = roe_avg(Q_r_y_plus_half, Q_l_y_plus_half)
    jac = abs_A(Q_roe_avg, nx_eta, ny_eta, V_average_j(V))
    # matrix multiplication
    Q_dif = Q_r_y_plus_half-Q_l_y_plus_half
    y,x,z = np.shape(Q_dif)
    res = np.zeros((y,x,z))
    for j in range(y):
        for i in range(x):
            res[j,i] = jac[j,i] @ Q_dif[j,i]
    F_eta_plus_half = 0.5*(flux_hat(Q_l_y_plus_half,nx_eta,ny_eta) + flux_hat(Q_r_y_plus_half,nx_eta,ny_eta)) - 0.5*res

    # computing the step of the iteration with a Roe scheme #
    i_variation = E_xi_plus_half[:,1:]*np.linalg.norm(Sl[1:-1,2:-1,:],axis=2)[:,:,np.newaxis] - E_xi_plus_half[:,:-1]*np.linalg.norm(Sl[1:-1,1:-2,:],axis=2)[:,:,np.newaxis]
    j_variation = F_eta_plus_half[1:,:]*np.linalg.norm(Sb[2:-1,1:-1,:],axis=2)[:,:,np.newaxis] - F_eta_plus_half[:-1,:]*np.linalg.norm(Sb[1:-2,1:-1,:],axis=2)[:,:,np.newaxis]
    # seems that there is a sign mistake somewher because it shows non physical behaviors if we use the conventional signes and it is not TVD (it is as usual associated with F flux)
    step = -deltat*(i_variation+j_variation)

    # apply the step #
    X[1:-1, 1:-1, :] = X[1:-1, 1:-1, :] - step/V[1:-1,1:-1,np.newaxis]

    # reset the boundary conditions #
    X = bc(X, bottom_Surfaces)

    # if required, plot intermediate states #
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
    Q = time_step(Q, 0.00000000001,V,Sb,Sl,plot)
    continuity_residual = np.sqrt(np.sum((Q[:,:,0]-Q0[:,:,0])**2))
    print(continuity_residual)
    while continuity_residual > 1e-3:  # the right condition is here to automaticaly stop in case of divergence
        Q0 = np.copy(Q)
        Q = time_step(Q, 0.00000000001,V,Sb,Sl,plot)
        continuity_residual = np.sqrt(np.sum((Q[:, :, 0] - Q0[:, :, 0]) ** 2))
        print(continuity_residual)
        nb_iter += 1
        if nb_iter > 100:
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
    # reading the grid #
    nx, ny, x, y = readgrid.readGrid('../../../grids', 'g33x25u_augmented.dat')
    Volumes, x_V, y_V, bottom_Surfaces, x_Sb, y_Sb, left_Surfaces, x_Sl, y_Sl = cellproperties.cellsCharacteristics('../../../grids', 'g33x25u_augmented.dat')
    # intitialise the computation parameters #
    Q = initialise(nx, ny, bottom_Surfaces)
    Q = bc(Q, bottom_Surfaces)
    # test of the direction vectors #
    nx_xi,ny_xi,nx_eta,ny_eta = direction_vectors_i(bottom_Surfaces,left_Surfaces)
    plt.xlabel('x')
    plt.ylabel('y')
    print(np.mean((nx_xi**2+ny_xi**2))==1) # testing that vectors are of norm 1
    print(np.mean((nx_eta ** 2 + ny_eta ** 2)) == 1)  # testing that vectors are of norm 1
    print(np.sum(nx_xi==0)) # = 0 so we can use the matrix R1 of AIAA-2001-2609
    plt.quiver(x_Sl[1:-1,1:-1], y_Sl[1:-1,1:-1],nx_xi,ny_xi)
    plt.quiver(x_Sl[1:-1, 1:-1], y_Sl[1:-1, 1:-1], nx_eta, ny_eta)
    plotgrid.gridePlotting('../../../grids', 'g33x25u_augmented.dat')
    plt.title('Pressure')
    plt.show()
    nx_xi,ny_xi,nx_eta,ny_eta = direction_vectors_j(bottom_Surfaces,left_Surfaces)
    plt.xlabel('x')
    plt.ylabel('y')
    print(np.mean((nx_xi**2+ny_xi**2))==1) # testing that vectors are of norm 1
    print(np.mean((nx_eta ** 2 + ny_eta ** 2)) == 1)  # testing that vectors are of norm 1
    print(np.sum(ny_eta == 0)) # = 0 so we can use the matrix R2 of AIAA-2001-2609
    plt.quiver(x_Sb[1:-1,1:-1], y_Sb[1:-1,1:-1],nx_xi,ny_xi)
    plt.quiver(x_Sb[1:-1, 1:-1], y_Sb[1:-1, 1:-1], nx_eta, ny_eta)
    plotgrid.gridePlotting('../../../grids', 'g33x25u_augmented.dat')
    plt.title('Pressure')
    plt.show()
    # test of the roe scheme matrix #
    #abs_A(Q, nx, ny, Volumes)
    # test of the computation loop #
    time_step(Q,0.0001,Volumes,bottom_Surfaces,left_Surfaces,0)
    solve(nx,ny,Volumes,bottom_Surfaces,left_Surfaces)