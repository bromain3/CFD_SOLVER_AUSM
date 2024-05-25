import numpy as np
import time
import  matplotlib.pyplot as plt
import cellproperties
import extract
import plot
import readgrid
import save
import set
import step_1_AUSM

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
    deltat = 0.00001 # for unstretched grids,0.00001 is fine. for stretched mesh, 0.000001 is fine
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
    Q = step_1_AUSM.one_step(Q, deltat, V, Sb, Sl, x_V, y_V, nx_xi, ny_xi, nx_eta, ny_eta, plot, nb_iter)
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
        Q = step_1_AUSM.one_step(Q, deltat, V, Sb, Sl, x_V, y_V, nx_xi, ny_xi, nx_eta, ny_eta, plot, nb_iter)

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
        if plot_iter > 10000:
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
    save.log('../../../data/2D/output/step_1_AUSM', 'log', nb_iter, dateini, date_final, simulation_time, converged)
    save.residuals('../../../data/2D/output/step_1_AUSM', 'residuals', Cr, Xr, Yr, Er)
    save.computational_time('../../../data/2D/output/step_1_AUSM', 'stats', T)
    save.states('../../../data/2D/output/step_1_AUSM', 'states', Q, nx, ny)
    # plot data #
    plot.residuals(Cr, Xr, Yr, Er)
    plot.physical_data(Q, ['rho', 'P', 'P0', 'T', 'M'], x_V, y_V, levels=100)
    parameters = {'axes.labelsize': 20,
                  'axes.titlesize': 20,
                  'legend.fontsize': 20,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15}
    plt.rcParams.update(parameters)
    plt.plot(x_Sb[1,:], extract.P(Q)[1, :])
    plt.xlabel('x')
    plt.ylabel('P')
    plt.show()
    print(Q[7,32,0],Q[1,0,0],Q[7,32,0]/Q[1,0,0])  # rho
    print(extract.P(Q)[7, 32], extract.P(Q)[1, 0], extract.P(Q)[7, 32] / extract.P(Q)[1, 0])  # P
    print(extract.T(Q)[7, 32], extract.T(Q)[1, 0], extract.T(Q)[7, 32] / extract.T(Q)[1, 0])  # T
    print(extract.P0(Q)[7, 32], extract.P0(Q)[1, 0], extract.P0(Q)[7, 32] / extract.P0(Q)[1, 0])  # P0
    print(extract.M(Q)[7, 32])  # M