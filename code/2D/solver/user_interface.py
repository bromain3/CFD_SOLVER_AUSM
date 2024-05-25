import os

import step_1_AUSM_POO as solver
import matplotlib.pyplot as plt
import grid as grid
import time
import numpy as np


class user_interface(object):
    plot_parameters = {'axes.labelsize': 20,
                  'axes.titlesize': 20,
                  'legend.fontsize': 20,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15}
    colormap = 'jet'
    plotting_levels = 1000

    def __init__(self,path,grid_name):
        self.grid = grid.computational_grid(path, grid_name)
        self.solver = solver.solver(self.grid)
        self.converged = False

    def initialize_computation(self,dt):
        self.g_ini = self.extrac_grid_values()
        self.g_minus_1 = self.extrac_grid_values()
        self.nb_iter = 0
        self.Cr = []
        self.Xr = []
        self.Yr = []
        self.Er = []
        self.T = [0]
        self.tini = time.time()
        self.dt = dt

    def run_computation(self,tol,residuals_printing_interval,saving_interval,save_path,case_name):
        self.solver.one_step_AUSM(self.dt)
        self.T.append(self.T[-1] + self.dt)
        self.nb_iter += 1
        cr, xr, yr, er = self.residuals_L2()
        self.Cr.append(cr)
        self.Xr.append(xr)
        self.Yr.append(yr)
        self.Er.append(er)
        print(cr)

        # solving loop #
        while cr > tol: # machine precision : 1e-16
            # update Q #
            self.g_minus_1 = self.extrac_grid_values()
            self.solver.one_step_AUSM(self.dt)

            # compute residuals #
            cr, xr, yr, er = self.residuals_L2()
            self.Cr.append(cr)
            self.Xr.append(xr)
            self.Yr.append(yr)
            self.Er.append(er)
            if self.nb_iter % residuals_printing_interval == 0:
                print('residuals:\tcontinuity|' + str(cr)+'\tX-momentum|'+str(xr)+'\tY-momentum|'+str(yr)+'\tenergy|'+str(er))
                print(self.grid)

            # update simulation data #
            self.nb_iter += 1
            self.T.append(self.T[-1] + self.dt)

            if self.nb_iter % saving_interval == 0:
                self.save_states(save_path,case_name)
                self.save_residuals_and_physical_time(save_path, case_name)
                self.save_computation_log(save_path, case_name)

        print('Seconds simulated :', self.T[-1], 'seconds')
        self.converged = True
        self.end_time = time.time()
        print('End of the simulation in ', self.end_time - self.tini, 'seconds')


    def residuals_L2(self):
        cr = 0
        xr = 0
        yr = 0
        er = 0
        for y in range(self.grid.ny_cells):
            for x in range(self.grid.nx_cells):
                speed = (self.g_ini[y][x][1] ** 2 + self.g_ini[y][x][2] ** 2) ** (1/2)
                cr += ((self.grid.cells[y][x].rho - self.g_minus_1[y][x][0]) / self.g_ini[y][x][0])**2
                xr += ((self.grid.cells[y][x].rho_u - self.g_minus_1[y][x][1]) / speed)**2
                yr += ((self.grid.cells[y][x].rho_v - self.g_minus_1[y][x][2]) / speed)**2
                er += ((self.grid.cells[y][x].rho_et - self.g_minus_1[y][x][3]) / self.g_ini[y][x][3])**2
        cr = cr ** (1/2)
        xr = xr ** (1 / 2)
        yr = yr ** (1 / 2)
        er = er ** (1 / 2)
        return(cr, xr, yr, er)


    def residuals_Linf(self):
        eps = 1e-16
        cr = []
        xr = []
        yr = []
        er = []
        for y in range(self.grid.ny_cells):
            for x in range(self.grid.nx_cells):
                cr.append((self.grid.cells[y][x].rho - self.g_minus_1[y][x][0]) / self.g_ini[y][x][0])
                xr.append((self.grid.cells[y][x].rho_u - self.g_minus_1[y][x][1]) / self.g_ini[y][x][1])
                yr.append((self.grid.cells[y][x].rho_v - self.g_minus_1[y][x][2]) / (self.g_ini[y][x][2]+eps))
                er.append((self.grid.cells[y][x].rho_et - self.g_minus_1[y][x][3]) / self.g_ini[y][x][3])
        cr = max(cr)
        xr = max(xr)
        yr = max(yr)
        er = max(er)
        return(cr, xr, yr, er)


    def extrac_grid_values(self):
        res = [[0 for x in range(self.grid.nx_cells)] for y in range(self.grid.ny_cells)]
        for y in range(self.grid.ny_cells):
            for x in range(self.grid.nx_cells):
                state_vect = [self.grid.cells[y][x].rho,self.grid.cells[y][x].rho_u,self.grid.cells[y][x].rho_v,self.grid.cells[y][x].rho_et]
                res[y][x] = state_vect
        return res

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
        x_V,y_V = self.grid.get_centroids()
        for x in req:
            plots = x.split(';')
            for p in plots:
                if p == 'rho' or p == 'rho_u' or p == 'rho_v' or p == 'rho_et':
                    #plt.pcolormesh(x_V, y_V, self.grid.get_cells_physical_value(p), cmap=user_interface.colormap)
                    plt.contourf(x_V, y_V, self.grid.get_cells_physical_value(p), levels=user_interface.plotting_levels, cmap=user_interface.colormap)
                    plt.colorbar()
                elif p == 'quiver':
                    # this plot does not look nice and could be improved
                    plt.quiver(x_V, y_V, self.grid.get_cells_computed_physical_value('u'), self.grid.get_cells_computed_physical_value('v'))
                else:
                    plt.contourf(x_V, y_V, self.grid.get_cells_computed_physical_value(p), levels=user_interface.plotting_levels, cmap=user_interface.colormap)
                    plt.colorbar()
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
        with open(os.path.join(path,case_name+'.residuals'), 'w') as fout:
            fout.write("physical time;continuity residual;x momentum residual;y momentum residual;energy residual" + '\n')
            T = self.T[1:]
            for i in range(len(self.Cr)):
                fout.write(str(T[i]) + ';' + str(self.Cr[i]) + ';' + str(self.Xr[i]) + ';'  + str(self.Yr[i]) + ';' + str(self.Er[i]) + '\n')

    def save_states(self, path, case_name):
        with open(os.path.join(path,case_name+'.state'), "w") as fout:
            fout.write("rho;rho*u;roh*v;rho*et" + '\n')
            for y in range(self.grid.ny_cells):
                for x in range(self.grid.nx_cells):
                    fout.write(str(self.grid.cells[y][x].rho) + ';' + str(self.grid.cells[y][x].rho_u) + ';' + str(self.grid.cells[y][x].rho_v) + ';' + str(self.grid.cells[y][x].rho_et) + '\n')

    def load_calculation(self,storage,case_name):
        # !!! add log reading so that we can start an unachived computation (maybe not mandatory, but still, it is better)
        log = os.path.join(storage, case_name + '.log')
        res = os.path.join(storage, case_name + '.residuals')
        state = os.path.join(storage, case_name + '.state')
        with open(res,'r') as f:
            res = f.readlines()
        self.T = [i.split(';')[0].strip() for i in res[1:]]
        self.Cr = [i.split(';')[1].strip() for i in res[1:]]
        self.Xr = [i.split(';')[2].strip() for i in res[1:]]
        self.Yr = [i.split(';')[3].strip() for i in res[1:]]
        self.Er = [i.split(';')[4].strip() for i in res[1:]]

        with open(state,'r') as f:
            state = f.readlines()
        rho = np.zeros(self.grid.nx_cells * self.grid.ny_cells)  # preallocating array size
        rho_u = np.zeros(self.grid.nx_cells * self.grid.ny_cells)  # preallocating array size
        rho_v = np.zeros(self.grid.nx_cells * self.grid.ny_cells)  # preallocating array size
        rho_et = np.zeros(self.grid.nx_cells * self.grid.ny_cells)  # preallocating array size

        line_count = 0
        for line in state[1:]:
            rho[line_count] = float(line.strip("\n").split(";")[0])  # reading data points
            rho_u[line_count] = float(line.strip("\n").split(";")[1])  # reading data points
            rho_v[line_count] = float(line.strip("\n").split(";")[2])  # reading data points
            rho_et[line_count] = float(line.strip("\n").split(";")[3])  # reading data points
            line_count += 1
        rho = rho.reshape((self.grid.ny_cells, self.grid.nx_cells))  # reshaping the flat array in a 2D array
        rho_u = rho_u.reshape((self.grid.ny_cells, self.grid.nx_cells))  # reshaping the flat array in a 2D array
        rho_v = rho_v.reshape((self.grid.ny_cells, self.grid.nx_cells))  # reshaping the flat array in a 2D array
        rho_et = rho_et.reshape((self.grid.ny_cells, self.grid.nx_cells))  # reshaping the flat array in a 2D array
        self.grid.initialize_cells_with_save(rho,rho_u,rho_v,rho_et)


if __name__ == '__main__':
    path = r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\grids\2D'
    save_path = r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\data\2D\output\new_scheme'
    grid_name = 'g65x49u.dat'
    case_name = 'step_1_AUSM'

    dt = 0.000001
    tol = 1e-16 #3.842363063744487e-05

    # new calculation
    ui = user_interface(path,grid_name)
    ui.initialize_computation(dt)
    ui.run_computation(tol,1000,1000,save_path,case_name)
    ui.plot_residuals()
    print(ui.grid)
    ui.save_computation_log(save_path,case_name)
    ui.save_residuals_and_physical_time(save_path,case_name)
    ui.save_states(save_path,case_name)

    """
    # Load calculation
    ui = user_interface(path,grid_name)
    ui.load_calculation(save_path,case_name)
    ui.plot_residuals()
    ui.plot_physical_data(['M','P','rho','T','P0'])
    """