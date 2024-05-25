import numpy as np
import time

import cellproperties
import extract
import plot
import readgrid
import set
import step_2_second_order

def solve(nx,ny,V,Sb,Sl,x_V,y_V):
    """
    Computational loop used to find Q with a given scheme on the chosen grid
    It returns all parameters used for data saving
    :param nx, ny: grid size (int)
    :param V: list of cells volume (2D-array)
    :param Sb: list of bottom surface vectors (3D-array)
    :param Sl: list of left surface vectors (3D-array)
    :return:
    """
    # initialize simulation parameters #
    deltat = 0.000001 # for unstretched grids,0.00001 is fine. for stretched mesh, 0.000001 is fine
    nb_iter = 0
    plot_iter = 0
    plot = 0
    nx_xi, ny_xi = extract.direction_vectors_i(Sb, Sl)
    nx_eta, ny_eta = extract.direction_vectors_j(Sb, Sl)
    # initialize data to be saved #
    Cr = []
    Xr = []
    Yr = []
    Er = []
    T = [0]
    tini = time.time()
    dateini = time.ctime()
    converged = False

    # Initialise computational space #
    Q = set.initial_conditions(nx, ny, Sb)
    Q = set.bc(Q, nx_eta, ny_eta)
    Qini = np.copy(Q)
    Q0 = np.copy(Q)
    Q = step_2_second_order.one_step(Q, deltat, V, Sb, Sl, x_V, y_V, nx_xi, ny_xi, nx_eta, ny_eta, plot, nb_iter)
    T.append(T[-1]+deltat)
    nb_iter += 1
    plot_iter += 1
    cr,xr,yr,er = extract.residuals_L2(Q, Q0, Qini)
    Cr.append(cr)
    Xr.append(xr)
    Yr.append(yr)
    Er.append(er)
    print(cr)

    # solving loop #
    while cr > 1e-16:  # for the moment the residual are not relative to initial conditions so in the end we should look at max(cr,xr,yr,er) but for the moment as rhoini~1, we use this one to set convergence
        # update Q #
        Q0 = np.copy(Q)
        Q = step_2_second_order.one_step(Q, deltat, V, Sb, Sl, x_V, y_V, nx_xi, ny_xi, nx_eta, ny_eta, plot, nb_iter)

        # compute residuals #
        cr, xr, yr, er = extract.residuals_L2(Q, Q0, Qini)
        Cr.append(cr)
        Xr.append(xr)
        Yr.append(yr)
        Er.append(er)
        print(cr)

        # update simulation data #
        nb_iter += 1
        plot_iter += 1
        T.append(T[-1]+deltat)

        # set plot flag #
        if plot_iter > 1000:
            plot = 1
            plot_iter = 0
        else:
            plot = 0
    print(T[-1])
    converged = True
    simulation_time = time.time() - tini
    date_final = time.ctime()
    return(Q,T,Cr,Xr,Yr,Er,nb_iter,dateini,date_final,simulation_time,converged)

if __name__ == '__main__':
    # reading the grid #
    nx, ny, x, y = readgrid.readGrid('../../grids', 'g65x49u_augmented.dat')
    Volumes, x_V, y_V, bottom_Surfaces, x_Sb, y_Sb, left_Surfaces, x_Sl, y_Sl = cellproperties.cellsCharacteristics('../../grids', 'g65x49u_augmented.dat')
    # solve the grid #
    Q,T,Cr,Xr,Yr,Er,nb_iter,dateini,date_final,simulation_time,converged = solve(nx,ny,Volumes,bottom_Surfaces,left_Surfaces,x_V, y_V)
    # save data #
    #save.log('../../data/output/step_2','log',nb_iter,dateini,date_final,simulation_time,converged)
    #save.residuals('../../data/output/step_2','residuals',Cr,Xr,Yr,Er)
    #save.computational_time('../../data/output/step_2','stats',T)
    #save.states('../../data/output/step_2','states',Q,nx,ny)
    # plot data #
    plot.residuals(Cr, Xr, Yr, Er)
    plot.physical_data(Q, ['rho', 'P;quiver', 'u', 'v', 'T', 'c'], x_V, y_V, levels=100)