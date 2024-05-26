import sys
sys.path.append(r'..\solver')
sys.path.append(r'..\grid_managment')
sys.path.append(r'..\code_readability_functions')
import os
import step_1_np as solver
import matplotlib.pyplot as plt
import grid_np as grid
import time
import numpy as np
from slices import *


class user_interface(object):
    plot_parameters = { 'axes.labelsize': 20,
                        'axes.titlesize': 20,
                        'legend.fontsize': 20,
                        'xtick.labelsize': 15,
                        'ytick.labelsize': 15}
    colormap = 'jet'
    plotting_levels = 1000
    fps = 100

    def __init__(self,path,grid_name):
        self.grid = grid.computational_grid(path, grid_name)
        self.solver = solver.solver_np(self.grid)
        self.converged = False

    def initialize_computation(self,dt):
        self.Q_ini = self.grid.Q.copy()
        self.Q_minus_1 = self.grid.Q.copy()
        self.nb_iter = 0
        self.Cr = []
        self.Xr = []
        self.Yr = []
        self.Er = []
        self.T = [0]
        self.tini = time.time()
        print('Simulation initialized at',time.ctime(self.tini))
        self.dt = dt

    def run_computation(self,tol,residuals_printing_interval,saving_interval,save_path,case_name,save_video=False):
        if save_video:
            if not os.path.exists(os.path.join(save_path, case_name)):
                os.mkdir(os.path.join(save_path, case_name))
        self.solver.one_step_AUSM(self.dt)
        self.T.append(self.T[-1] + self.dt)
        self.nb_iter += 1
        cr, xr, yr, er = self.residuals_L2()
        self.Cr.append(cr)
        self.Xr.append(xr)
        self.Yr.append(yr)
        self.Er.append(er)
        print('residuals:\tcontinuity|%.8E\tX-momentum|%.8E\tY-momentum|%.8E\tenergy|%.8E' % (cr,xr,yr,er))

        # solving loop #
        while cr > tol: # machine precision : 1e-16
            # update Q #
            self.Q_minus_1 = self.grid.Q.copy()
            self.solver.one_step_AUSM(self.dt)

            # compute residuals #
            cr, xr, yr, er = self.residuals_L2()
            self.Cr.append(cr)
            self.Xr.append(xr)
            self.Yr.append(yr)
            self.Er.append(er)
            if self.nb_iter % residuals_printing_interval == 0:
                print('residuals:\tcontinuity|%.8E\tX-momentum|%.8E\tY-momentum|%.8E\tenergy|%.8E' % (cr, xr, yr, er))


            # update simulation data #
            self.nb_iter += 1
            self.T.append(self.T[-1] + self.dt)

            if self.nb_iter % saving_interval == 0:
                self.save_states(save_path,case_name)
                self.save_residuals_and_physical_time(save_path, case_name)
                self.save_computation_log(save_path, case_name)
                print(self.grid)

            if save_video:
                """
                #!!! certainly problematic if dt is too high and fps too low (or it will juste save every frame, idk)
                if self.T[-1] * user_interface.fps >= img_saved:
                    plt.contourf(x_V, y_V, self.grid.get_cells_physical_value('rho'), levels=user_interface.plotting_levels,cmap=user_interface.colormap)
                    plt.colorbar()
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title(r'$\rho$')
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path,case_name,str(img_saved))+'.png')
                    plt.clf()
                    img_saved += 1
                """
                pass

        print('Seconds simulated :', self.T[-1], 'seconds')
        self.converged = True
        self.end_time = time.time()
        print('End of the simulation on', self.end_time - self.tini, 'seconds')


    def residuals_L2(self):
        speed = (self.Q_ini[rho_u] ** 2 + self.Q_ini[rho_v] ** 2) ** (1 / 2)
        cr = np.linalg.norm((self.grid.Q[rho] - self.Q_minus_1[rho]) / self.Q_ini[rho])
        xr = np.linalg.norm((self.grid.Q[rho_u] - self.Q_minus_1[rho_u]) / speed)
        yr = np.linalg.norm((self.grid.Q[rho_v] - self.Q_minus_1[rho_v]) / speed)
        er = np.linalg.norm((self.grid.Q[rho_et] - self.Q_minus_1[rho_et]) / self.Q_ini[rho_et])
        return(cr, xr, yr, er)


    def residuals_Linf(self):
        pass

    def plot_residuals(self):
        plt.rcParams.update(user_interface.plot_parameters)
        plt.plot(self.Cr, label='Continuity residual')
        plt.plot(self.Xr, label='X-momentum residual')
        plt.plot(self.Yr, label='Y-momentum residual')
        plt.plot(self.Er, label='Energy residual')
        plt.legend()
        plt.semilogy()
        plt.xlabel('Number of iterations')
        plt.show()

    def plot_physical_data(self, req):
        plt.rcParams.update(user_interface.plot_parameters)
        for r in req:
            plots = r.split(';')
            for p in plots:
                if p == 'rho':
                    # plt.pcolormesh(self.grid.centroids[x], self.grid.centroids[y], self.grid.Q[rho], cmap=user_interface.colormap)
                    plt.contourf(self.grid.centroids[x], self.grid.centroids[y], self.grid.Q[rho], levels=user_interface.plotting_levels,cmap=user_interface.colormap)
                    plt.colorbar()
                elif p == 'rho_u':
                    # plt.pcolormesh(self.grid.centroids[x], self.grid.centroids[y], self.grid.Q[rho_u], cmap=user_interface.colormap)
                    plt.contourf(self.grid.centroids[x], self.grid.centroids[y], self.grid.Q[rho_u], levels=user_interface.plotting_levels,cmap=user_interface.colormap)
                    plt.colorbar()
                elif p == 'rho_u':
                    # plt.pcolormesh(self.grid.centroids[x], self.grid.centroids[y], self.grid.Q[rho_v], cmap=user_interface.colormap)
                    plt.contourf(self.grid.centroids[x], self.grid.centroids[y], self.grid.Q[rho_v], levels=user_interface.plotting_levels,cmap=user_interface.colormap)
                    plt.colorbar()
                elif p == 'rho_et':
                    # plt.pcolormesh(self.grid.centroids[x], self.grid.centroids[y], self.grid.Q[rho_et], cmap=user_interface.colormap)
                    plt.contourf(self.grid.centroids[x], self.grid.centroids[y], self.grid.Q[rho_et], levels=user_interface.plotting_levels,cmap=user_interface.colormap)
                    plt.colorbar()
                elif p == 'quiver':
                    # this plot does not look nice and could be improved
                    plt.quiver(self.grid.centroids[x], self.grid.centroids[y], grid.u(self.grid.Q),grid.v(self.grid.Q))
                else:
                    try:
                        # plt.pcolormesh(self.grid.centroids[x], self.grid.centroids[y], getattr(grid,p)(self.grid.Q), cmap=user_interface.colormap)
                        plt.contourf(self.grid.centroids[x], self.grid.centroids[y], getattr(grid,p)(self.grid.Q),levels=user_interface.plotting_levels, cmap=user_interface.colormap)
                        plt.colorbar()
                    except:
                        print('The value ' + p + ' connot be computed because ther is no function defined to do so')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(plots)
            plt.show()

    def save_computation_log(self, path, case_name):
        with open(os.path.join(path,case_name+'.log'), 'w') as fout:
            fout.write('Number of iterations: ' + str(self.nb_iter) + '\n')
            fout.write('Beginning of simulation: ' + time.ctime(self.tini) + '\n')
            fout.write('log written on: ' + time.ctime(time.time()) + '\n')
            fout.write('Time of simulation (seconds): ' + str(time.time() - self.tini) + '\n')
            fout.write('Simulation convergence: ' + str(self.converged) + '\n')

    def save_residuals_and_physical_time(self, path, case_name):
        # we shouod append only the values at the given saving interval to save storage
        with open(os.path.join(path,case_name+'.residuals'), 'w') as fout:
            fout.write("physical time;continuity residual;x momentum residual;y momentum residual;energy residual" + '\n')
            T = self.T[1:]
            for i in range(len(self.Cr)):
                fout.write(str(T[i]) + ';' + str(self.Cr[i]) + ';' + str(self.Xr[i]) + ';'  + str(self.Yr[i]) + ';' + str(self.Er[i]) + '\n')

    def save_states(self, path, case_name):
        with open(os.path.join(path, case_name + '.state'), "w") as fout:
            fout.write("rho;rho*u;roh*v;rho*et" + '\n')
            for y in range(self.grid.ny_cells):
                for x in range(self.grid.nx_cells):
                    fout.write(str(self.grid.Q[rho][y,x]) + ';' + str(self.grid.Q[rho_u][y,x]) + ';' + str(self.grid.Q[rho_v][y,x]) + ';' + str(self.grid.Q[rho_et][y,x]) + '\n')

    def load_calculation(self,storage,case_name):
        # !!! add log reading so that we can start an unachived computation (maybe not mandatory, but still, it is better)
        log = os.path.join(storage, case_name + '.log')
        res = os.path.join(storage, case_name + '.residuals')
        state = os.path.join(storage, case_name + '.state')
        with open(res,'r') as f:
            res = f.readlines()
        self.T = [float(i.split(';')[0].strip()) for i in res[1:]]
        self.Cr = [float(i.split(';')[1].strip()) for i in res[1:]]
        self.Xr = [float(i.split(';')[2].strip()) for i in res[1:]]
        self.Yr = [float(i.split(';')[3].strip()) for i in res[1:]]
        self.Er = [float(i.split(';')[4].strip()) for i in res[1:]]

        with open(state,'r') as f:
            state = f.readlines()
        r = np.zeros(self.grid.nx_cells * self.grid.ny_cells)  # preallocating array size
        r_u = np.zeros(self.grid.nx_cells * self.grid.ny_cells)  # preallocating array size
        r_v = np.zeros(self.grid.nx_cells * self.grid.ny_cells)  # preallocating array size
        r_et = np.zeros(self.grid.nx_cells * self.grid.ny_cells)  # preallocating array size

        line_count = 0
        for line in state[1:]:
            r[line_count] = float(line.strip("\n").split(";")[0])  # reading data points
            r_u[line_count] = float(line.strip("\n").split(";")[1])  # reading data points
            r_v[line_count] = float(line.strip("\n").split(";")[2])  # reading data points
            r_et[line_count] = float(line.strip("\n").split(";")[3])  # reading data points
            line_count += 1
        r = r.reshape((self.grid.ny_cells, self.grid.nx_cells))  # reshaping the flat array in a 2D array
        r_u = r_u.reshape((self.grid.ny_cells, self.grid.nx_cells))  # reshaping the flat array in a 2D array
        r_v = r_v.reshape((self.grid.ny_cells, self.grid.nx_cells))  # reshaping the flat array in a 2D array
        r_et = r_et.reshape((self.grid.ny_cells, self.grid.nx_cells))  # reshaping the flat array in a 2D array
        Q = np.zeros((self.grid.ny_cells, self.grid.nx_cells,4))
        Q[rho] = r
        Q[rho_u] = r_u
        Q[rho_v] = r_v
        Q[rho_et] = r_et
        self.grid.initialize_cells_with_save(Q)

if __name__ == '__main__':
    path = r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\grids\2D'
    save_path = r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\data\2D\output\new_scheme'
    grid_name = 'g65x49u.dat'
    case_name = 'step_1_AUSM_M_2'

    dt = 0.000001
    tol = 1e-16 #3.842363063744487e-05
    # new calculation
    ui = user_interface(path,grid_name)
    ui.initialize_computation(dt)
    user_interface.fps = 10000
    ui.run_computation(tol,1000,10000,save_path,case_name,save_video=False)
    ui.plot_residuals()
    print(ui.grid)
    ui.save_computation_log(save_path,case_name)
    ui.save_residuals_and_physical_time(save_path,case_name)
    ui.save_states(save_path,case_name)

    """
    # Load calculation
    ui = user_interface(path,grid_name)
    ui.load_calculation(save_path,case_name)
    #ui.plot_residuals()
    ui.plot_physical_data(['M','P','rho','T','P0'])
    """