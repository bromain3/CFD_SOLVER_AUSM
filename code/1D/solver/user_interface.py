import step_1_AUSM_one_step_new_scheme as solver
import matplotlib.pyplot as plt
import grid1D as grid
import time


class user_interface(object):
    plot_parameters = {'axes.labelsize': 20,
                  'axes.titlesize': 20,
                  'legend.fontsize': 20,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15}
    colormap = 'jet'

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
        self.Er = []
        self.T = [0]
        self.tini = time.time()
        self.dt = dt

    def run_computation(self,tol,residuals_printing_interval,saving_interval):
        self.solver.one_step_AUSM(self.dt)
        self.T.append(self.T[-1] + self.dt)
        self.nb_iter += 1
        cr, xr, er = self.residuals_L2()
        self.Cr.append(cr)
        self.Xr.append(xr)
        self.Er.append(er)
        print(cr)

        # solving loop #
        while cr > tol: # machine precision : 1e-16
            # update Q #
            self.g_minus_1 = self.extrac_grid_values()
            self.solver.one_step_AUSM(self.dt)

            # compute residuals #
            cr, xr, er = self.residuals_L2()
            self.Cr.append(cr)
            self.Xr.append(xr)
            self.Er.append(er)
            if self.nb_iter % residuals_printing_interval == 0:
                print(cr)
                print(self.grid)

            # update simulation data #
            self.nb_iter += 1
            self.T.append(self.T[-1] + self.dt)

            if self.nb_iter % saving_interval == 0:
                pass

        print('Seconds simulated :', self.T[-1], 'seconds')
        self.converged = True
        self.end_time = time.time()
        print('End of the simulation in ', self.end_time - self.tini, 'seconds')


    def residuals_L2(self):
        cr = 0
        xr = 0
        er = 0
        for x in range(self.grid.nx_cells):
            cr += ((self.grid.cells[x].rho - self.g_minus_1[x][0]) / self.g_ini[x][0])**2
            xr += ((self.grid.cells[x].rho_u - self.g_minus_1[x][1]) / self.g_ini[x][1])**2
            er += ((self.grid.cells[x].rho_et - self.g_minus_1[x][2]) / self.g_ini[x][2])**2
        cr = cr ** (1/2)
        xr = xr ** (1 / 2)
        er = er ** (1 / 2)
        return(cr, xr, er)


    def residuals_Linf(self):
        cr = []
        xr = []
        er = []
        for x in range(self.grid.nx_cells):
            cr.append((self.grid.cells[x].rho - self.g_minus_1[x][0]) / self.g_ini[x][0])
            xr.append((self.grid.cells[x].rho_u - self.g_minus_1[x][1]) / self.g_ini[x][1])
            er.append((self.grid.cells[x].rho_et - self.g_minus_1[x][2]) / self.g_ini[x][2])
        cr = max(cr)
        xr = max(xr)
        er = max(er)
        return(cr, xr, er)


    def extrac_grid_values(self):
        res = []
        for i in self.grid.cells:
            res.append([i.rho,i.rho_u,i.rho_et])
        return res

    def plot_residuals(self):
        plt.rcParams.update(user_interface.plot_parameters)
        plt.plot(self.Cr, label='Continuity residual')
        plt.plot(self.Xr, label='X-momentum residual')
        plt.plot(self.Er, label='Energy residual')
        plt.legend()
        plt.semilogy()
        plt.xlabel('Number of iterations')
        plt.show()

    def plot_physical_data(self, req):
        plt.rcParams.update(user_interface.plot_parameters)
        for x in req:
            plots = x.split(';')
            for p in plots:
                if p == 'rho' or p == 'rho_u' or p == 'rho_et':
                    plt.plot(self.grid.get_centroids(), self.grid.get_cells_physical_value(p))
                elif p == 'u' or p == 'et' or p == 'P' or p == 'P0' or p == 'c' or p == 'T' or p == 'ht' or p == 'M':
                    plt.plot(self.grid.get_centroids(), self.grid.get_cells_computed_physical_value(p))
                elif p == 'V':
                    plt.plot(self.grid.get_centroids(), self.grid.get_volumes())

            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(plots)
            plt.show()

    def save_computation_log(self, path, file):
        with open(path + '/' + file, 'w') as fout:
            fout.write('Number of iterations: ' + str(self.nb_iter) + '\n')
            fout.write('Beginning of simulation: ' + time.ctime(self.tini) + '\n')
            fout.write('End of simulation: ' + time.ctime(self.end_time) + '\n')
            fout.write('Time of simulation (seconds): ' + str(self.end_time - self.tini) + '\n')
            fout.write('Simulation convergence: ' + str(self.converged) + '\n')

    def save_residuals_and_physical_time(self, path, file):
        with open(path + '/' + file, 'w') as fout:
            fout.write("physical time;continuity residual;x momentum residual;energy residual" + '\n')
            T = self.T[1:]
            for i in range(len(self.Cr)):
                fout.write(str(T[i]) + ';' + str(self.Cr[i]) + ';' + str(self.Xr[i]) + ';' + str(self.Er[i]) + '\n')

    def save_states(self, path, file):
        with open(path + '/' + file, "w") as fout:
            fout.write("rho;rho*u;rho*et" + '\n')
            for i in range(self.grid.nx_cells):
                fout.write(str(self.grid.cells[i].rho) + ';' + str(self.grid.cells[i].rho_u) + ';' + str(self.grid.cells[i].rho_et) + '\n')

if __name__ == '__main__':
    path = r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\grids\1D'
    grid_name = 'diamond_airfoil.dat'

    dt = 0.000001
    tol = 1e-16

    ui = user_interface(path,grid_name)
    ui.initialize_computation(dt)
    ui.run_computation(tol,10000,1000000000000000)
    print(ui.grid)
    ui.plot_residuals()
    ui.save_computation_log(r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\data\1D\output','step_1_AUSM.log')
    ui.save_residuals_and_physical_time(r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\data\1D\output','step_1_AUSM.residuals')
    ui.save_states(r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\data\1D\output','step_1_AUSM.state')