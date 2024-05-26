import sys
sys.path.append(r'..\..\data\2D\input')
sys.path.append(r'..\code_readability_functions')
import numpy as np
import matplotlib.pyplot as plt
import os
from slices import *
from initial_conditions import *
from air import *

# !!! can be optimised by having nodes as a state var instead of x and y so that we do not have to recompute the dstack everytime

colorScheme = 'jet'
levels = 50

class grid_by_nodes(object):
    def __init__(self,grid_directory, grid_name):
        self.grid_directory = grid_directory
        self.grid_name = grid_name
        self.nx,self.ny,self.x,self.y = self.read_grid(grid_directory, grid_name)

    def read_grid(self,grid_directory, grid_name):
        """
        read grid file to have the position of the nodes and grid size
        :param file: path to the grid file (str)
        :return: nx,ny : number of points in the file (int)
                 x,y : coordinates of the points as it would be done by the meshgrid function (np.ndarray 2d)
        """

        # open file and read line by line to assign to array
        line_count = 0
        with open(os.path.join(grid_directory,grid_name), "r") as fp:
            [nx, ny] = [int(m) for m in fp.readline().strip("\n").split(", ")]  # reading grid dimensions
            x = np.zeros(nx * ny)  # preallocating array size
            y = np.zeros(nx * ny)  # preallocating array size
            for line in fp:
                x[line_count] = float(line.strip("\n").split(", ")[0])  # reading data points
                y[line_count] = float(line.strip("\n").split(", ")[1])  # reading data points
                line_count += 1
        x = x.reshape((ny, nx))  # reshaping the flat array in a 2D array
        y = y.reshape((ny, nx))  # reshaping the flat array in a 2D array
        return (nx, ny, x, y)

    def augment_grid(self):
        """
        augment a given grid with hallo cells
        :param file: path to the grid file (str)
        :return: augmented grid file
        """

        # new grid definition #
        # definition of new grid size
        new_nx = self.nx + 2
        new_ny = self.ny + 2
        # halo cells size computation (same thickness as adjacent cells)
        # all the dy and dx are computed to be positive
        dy_bottom = self.y[1, :] - self.y[0, :]
        dy_top = self.y[-1, :] - self.y[-2, :]
        dx_left = self.x[:, 1] - self.x[:, 0]
        dx_right = self.x[:, -1] - self.x[:, -2]
        # computation of the augmented coordinates matrix
        new_x = np.hstack([np.hstack([(self.x[:, 0] - dx_left).reshape((1, np.size(self.x[:, 0]))).T, self.x]),(self.x[:, -1] + dx_right).reshape((1, np.size(self.x[:, -1]))).T])
        new_x = np.vstack([np.vstack([new_x[0, :], new_x]), new_x[-1, :]])
        new_y = np.vstack([np.vstack([self.y[0, :] - dy_bottom, self.y]), self.y[-1, :] + dy_top])
        new_y = np.hstack([np.hstack([new_y[:, 0].reshape((1, np.size(new_y[:, 0]))).T, new_y]),
                           new_y[:, -1].reshape((1, np.size(new_y[:, -1]))).T])
        # converting 2D arrays in 1D arrays to use the writing code provided
        new_x = new_x.flatten()
        new_y = new_y.flatten()
        # new grid saving
        with open(os.path.join(self.grid_directory,self.grid_name.split('.')[0] + '_augmented.' + self.grid_name.split('.')[1]),'w') as new_grid: # creation of the new augmented file
            new_grid.write(" %d, %d\n" % (new_nx, new_ny,))
            for (xx, yy) in zip(new_x, new_y):
                new_grid.write("%s, %s\n" % (xx, yy,))
        self.augmented_grid = grid_by_cells(self.grid_directory,self.grid_name.split('.')[0] + '_augmented.' + self.grid_name.split('.')[1])
        return(self.augmented_grid)

    def plot_grid(self,color='red'):
        """
        function to plot the grid and the x,y,eta,xi axis. It actually does not show the grid, and it is necessary to call plt.plot() afterwards
        :param file: path to the grid file (str)
        :return: None
        """
        # mesh ploting
        for i in range(self.ny):  # plot of the line 'parallel' to x
            connecting_lines_x_coord = self.x[i, :].reshape((1, self.nx))[0]
            connecting_lines_y_coord = self.y[i, :].reshape((1, self.nx))[0]
            plt.plot(connecting_lines_x_coord, connecting_lines_y_coord, color=color, linewidth=0.5)
        for i in range(self.nx):  # plot of the line 'parallel' to y
            connecting_lines_x_coord = self.x[:, i].reshape((1, self.ny))[0]
            connecting_lines_y_coord = self.y[:, i].reshape((1, self.ny))[0]
            plt.plot(connecting_lines_x_coord, connecting_lines_y_coord, color=color, linewidth=0.5)
        # Axis plotting
        plt.arrow(-0.5, 0, 0.25, 0, head_width=0.02)  # xi
        plt.arrow(-0.5, 0, 0, 0.25, head_width=0.02)  # eta
        plt.arrow(0, 0, 1, 0, head_width=0.02)  # x
        plt.arrow(0, 0, 0, 0.5, head_width=0.02)  # y


class grid_by_cells(grid_by_nodes):
    #!!! can be optimised by having nodes as a state var instead of x and y so that we do not have to recompute the dstack everytime
    def __init__(self,grid_directory, grid_name):
        grid_by_nodes.__init__(self,grid_directory, grid_name)
        self.nx_cells = self.nx - 1
        self.ny_cells = self.ny - 1
        self.nodes = np.dstack((self.x, self.y))
        self.V = self.get_volumes()
        self.centroids = self.get_centroids()
        self.S_xi = self.get_S_xi()
        self.S_eta = self.get_S_eta()
        self.n_xi = self.get_n_xi()
        self.n_eta = self.get_n_eta()
        del self.nodes # not usefull anymore

    def get_volumes(self):
        r13 = self.nodes[upper_right] - self.nodes[lower_left]
        r24 = self.nodes[upper_left] - self.nodes[lower_right]
        matrix = np.stack((r13,r24),axis=3)
        res = 0.5 * np.linalg.det(matrix)
        return(res)

    def get_centroids(self):
        # the following computation is an approximation
        c = (self.nodes[upper_right] + self.nodes[lower_left] + self.nodes[upper_left] + self.nodes[lower_right]) / 4
        return(c)

    def get_S_xi(self):
        S_xi_beginig_points = (self.nodes[lower_right] + self.nodes[upper_right]) / 2
        r23 = self.nodes[upper_right] - self.nodes[lower_right]  # computing cells' sides lengths
        r23[x],r23[y] = r23[y],- r23[x]
        res = np.dstack((S_xi_beginig_points,r23))
        return(res)

    def get_S_eta(self):
        S_xi_beginig_points = (self.nodes[upper_left] + self.nodes[upper_right]) / 2
        r34 = self.nodes[upper_left] - self.nodes[upper_right]  # computing cells' sides lengths
        r34[x], r34[y] = r34[y], - r34[x]
        res = np.dstack((S_xi_beginig_points, r34))
        return (res)

    def get_n_xi(self):
        res = np.zeros(np.shape(self.S_xi))
        res[origin] = self.S_xi[origin]
        res[length] = self.S_xi[length] / np.linalg.norm(self.S_xi[length],axis=2)[:,:,np.newaxis]
        return(res)

    def get_n_eta(self):
        res = np.zeros(np.shape(self.S_eta))
        res[origin] = self.S_eta[origin]
        res[length] = self.S_eta[length] / np.linalg.norm(self.S_eta[length],axis=2)[:,:,np.newaxis]
        return(res)

    def plot_surface_vectors(self):
        plt.quiver(self.S_xi[origin][x], self.S_xi[origin][y], self.S_xi[length][x], self.S_xi[length][y], scale=5, headwidth=2, headlength=5)
        plt.quiver(self.S_eta[origin][x], self.S_eta[origin][y], self.S_eta[length][x], self.S_eta[length][y], scale=5,headwidth=2, headlength=5)

    def plot_direction_vectors(self):
        plt.quiver(self.n_xi[origin][x], self.n_xi[origin][y], self.n_xi[length][x], self.n_xi[length][y], scale=50,headwidth=2, headlength=5)
        plt.quiver(self.n_eta[origin][x], self.n_eta[origin][y], self.n_eta[length][x], self.n_eta[length][y], scale=50,headwidth=2, headlength=5)


class computational_grid(grid_by_cells):
    def __init__(self,grid_directory,grid_name):
        grid_by_cells.__init__(self,grid_directory,grid_name)
        if not hasattr(self,'augmented_grid'):
            self.augment_grid()
        self.augmented_grid.Q = np.zeros((self.augmented_grid.ny_cells,self.augmented_grid.nx_cells,4))
        self.intricate_base_to_augmented_grid()

    def intricate_base_to_augmented_grid(self):
        self.Q = self.augmented_grid.Q[interior]

    def initialize_cells(self):
        self.augmented_grid.Q[rho] = rhoini
        self.augmented_grid.Q[rho_u] = rhoini*Mini*c(self.augmented_grid.Q,Tini)
        self.augmented_grid.Q[rho_v] = 0
        self.augmented_grid.Q[rho_et] = Pini/(gamma-1) + rhoini*(Mini*c(self.augmented_grid.Q,Tini))**2/2

    def initialize_cells_with_save(self,rho,rho_u,rho_v,rho_et):
        pass

    def set_boundary_conditions(self):

        # we have to define gamma = gam(Q) if we want to implement non constant gamma
        # we can also assign V_2 = (self.Q[rho_u][top_wall] ** 2 + self.Q[rho_v][top_wall] ** 2) and this kind of variables for the sake of readability (be be carfull of not computing to much useless cells
        # the following can be written in less lines by appying the same function to the variables top_wall and bottom_wall. it will also make the code more consistant

        # set inlet conditions
        self.augmented_grid.Q[rho][inlet] = rhoini
        self.augmented_grid.Q[rho_u][inlet] = rhoini*Mini*c(self.augmented_grid.Q,Tini) # can be optimised by giving only the x = 0 column of Q to the c function and not computing the whole array of c and thene taking what we want
        self.augmented_grid.Q[rho_v][inlet] = 0
        self.augmented_grid.Q[rho_et][inlet] = Pini/(gamma-1) + rhoini*(Mini*c(self.augmented_grid.Q,Tini))**2/2

        # setting all the hallo cells on the right to the value of the last column of computed cells
        self.augmented_grid.Q[outlet] = self.augmented_grid.Q[just_before_outlet] # to improve performance, we only have to make the -1 and -2 column point on the same memory adresse once in the init and we don't have to compute it every iteration

        # slip wall condition #
            # top wall
            # this approximation is fine as the top wall is flat, but in a second time it could be verified more rigorously
        self.augmented_grid.Q[rho_u][top_wall] = self.augmented_grid.Q[rho][top_wall] * u(self.augmented_grid.Q)[just_before_top_wall]  # can be optimised by giving only the x = 0 column of Q to the c function and not computing the whole array of c and thene taking what we want
        self.augmented_grid.Q[rho_v][top_wall] = self.augmented_grid.Q[rho][top_wall] * v(self.augmented_grid.Q)[just_before_top_wall]  # can be optimised by giving only the x = 0 column of Q to the c function and not computing the whole array of c and thene taking what we want
            # bottom wall
        self.augmented_grid.Q[rho_u][bottom_wall] = self.augmented_grid.Q[rho][bottom_wall] * (u(self.augmented_grid.Q)[just_after_bottom_wall] * (self.augmented_grid.n_eta[length][y][bottom_wall] ** 2 - self.augmented_grid.n_eta[length][x][bottom_wall] ** 2) - 2 * self.augmented_grid.n_eta[length][x][bottom_wall] * self.augmented_grid.n_eta[length][y][bottom_wall] * v(self.augmented_grid.Q)[just_after_bottom_wall])
        self.augmented_grid.Q[rho_v][bottom_wall] = self.augmented_grid.Q[rho][bottom_wall] * (v(self.augmented_grid.Q)[just_after_bottom_wall] * (self.augmented_grid.n_eta[length][x][bottom_wall] ** 2 - self.augmented_grid.n_eta[length][y][bottom_wall] ** 2) - 2 * self.augmented_grid.n_eta[length][x][bottom_wall] * self.augmented_grid.n_eta[length][y][bottom_wall] * u(self.augmented_grid.Q)[just_after_bottom_wall])

        # adiabatic condition #
            # top wall
        self.augmented_grid.Q[rho_et][top_wall] = r * self.augmented_grid.Q[rho][top_wall] / (gamma - 1) * ((gamma - 1) * (self.augmented_grid.Q[rho_et][just_before_top_wall] - 1 / 2 * (self.augmented_grid.Q[rho_u][just_before_top_wall] ** 2 + self.augmented_grid.Q[rho_v][just_before_top_wall] ** 2) /self.augmented_grid.Q[rho][just_before_top_wall]) / r / self.augmented_grid.Q[rho][just_before_top_wall]) + 1 / 2 * (self.augmented_grid.Q[rho_u][top_wall] ** 2 + self.augmented_grid.Q[rho_v][top_wall] ** 2) / self.augmented_grid.Q[rho][top_wall]
            # bottom wall
        self.augmented_grid.Q[rho_et][bottom_wall] = r * self.augmented_grid.Q[rho][bottom_wall] / (gamma - 1) * ((gamma - 1) * (self.augmented_grid.Q[rho_et][just_after_bottom_wall] - 1 / 2 * (self.augmented_grid.Q[rho_u][just_after_bottom_wall] ** 2 + self.augmented_grid.Q[rho_v][just_after_bottom_wall] ** 2) / self.augmented_grid.Q[rho][just_after_bottom_wall]) / r /self.augmented_grid.Q[rho][just_after_bottom_wall]) + 1 / 2 * (self.augmented_grid.Q[rho_u][bottom_wall] ** 2 + self.augmented_grid.Q[rho_v][bottom_wall] ** 2) / self.augmented_grid.Q[rho][bottom_wall]

        """
            # 0 pressure gradient at walls #
            # bottom wall
            X[0, -1:1, 3] = (X[1, -1:1, 3] - 0.5 * (X[1, -1:1, 1] ** 2 + X[1, -1:1, 2] ** 2) / X[1, -1:1, 0]) + 0.5 * (
                        X[0, -1:1, 1] ** 2 + X[0, -1:1, 2] ** 2) / X[0, -1:1, 0]
            # top wall
            X[-1, -1:1, 3] = (X[-2, -1:1, 3] - 0.5 * (X[-2, -1:1, 1] ** 2 + X[-2, -1:1, 2] ** 2) / X[-2, -1:1,
                                                                                                   0]) + 0.5 * (
                                         X[-1, -1:1, 1] ** 2 + X[-1, -1:1, 2] ** 2) / X[-1, -1:1, 0]"""

    def __str__(self):
        x_centroid, y_centroid = self.centroids[x],self.centroids[y]
        r = self.Q[rho]
        vel_u = u(self.Q)
        vel_v = v(self.Q)
        r_et = self.Q[rho_et]
        plt.contourf(x_centroid, y_centroid, r, levels=levels, cmap=colorScheme)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$\rho$')
        plt.show()
        plt.contourf(x_centroid, y_centroid, vel_u, levels=levels, cmap=colorScheme)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$u$')
        plt.show()
        plt.contourf(x_centroid, y_centroid, vel_v, levels=levels, cmap=colorScheme)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$v$')
        plt.show()
        plt.contourf(x_centroid, y_centroid, r_et, levels=levels, cmap=colorScheme)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$\rho e_{t}$')
        plt.show()
        return('')

######################################################################
####################  physical data extraction  ######################
######################################################################

def gam(Q):
    return(gamma)

def c(Q, Temp='not_given'):
    if Temp == 'not_given':
        Temp = T(Q)
    return (np.sqrt(gam(Q) * r * Temp))

def u(Q):
    return (Q[rho_u] / Q[rho])

def v(Q):
    return (Q[rho_v] / Q[rho])

def et(Q):
    return (Q[rho_et] / Q[rho])

def P(Q):
    return ((gam(Q) - 1) * Q[rho] * (et(Q) - 0.5 * (u(Q) ** 2 + v(Q) ** 2)))

def ht(Q):
    return (et(Q) + P(Q) / Q[rho])

def T(Q):
    return max((((gam(Q) - 1) * (Q[rho_et] - 1 / 2 * (Q[rho_u] ** 2) / Q[rho])) / r / Q[rho]),100) # by taking T values no smaller than 100K, We avoid too large values of M (because small c), so it helps convergence a lot. In a second time, we could use this formula for transiant, and after enough step, go back to the normal formula

def M(Q):
    return np.sqrt(u(Q) ** 2 + v(Q) ** 2) / c(Q)

def P0(Q):
    return (P(Q) * (1 + (gam(Q) - 1) / 2 * M(Q) ** 2) ** (gam(Q) / (gam(Q) - 1)))



if __name__ == '__main__':
    path = r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\grids\2D'
    grid = 'g33x25u.dat'
    g = grid_by_cells(path,grid)
    aug_g = g.augment_grid()
    g.plot_grid()
    g.plot_direction_vectors()
    plt.show()

    g = computational_grid(path,grid)
    g.initialize_cells()
    g.set_boundary_conditions()
    print(g)
