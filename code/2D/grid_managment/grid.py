import numpy as np
import matplotlib.pyplot as plt
import os
import cell
import geometry
from initial_conditions import *

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
    def __init__(self,grid_directory, grid_name):
        grid_by_nodes.__init__(self,grid_directory, grid_name)
        self.cells = self.convert_nodes_to_cells()
        self.nx_cells = self.nx - 1
        self.ny_cells = self.ny - 1

    def convert_nodes_to_cells(self):
        cells = [[0 for i in range(self.nx-1)] for j in range(self.ny-1)]
        for y in range(self.ny-1):
            for x in range(self.nx-1):
                lower_left_node = geometry.node(self.x[y, x], self.y[y, x])
                lower_right_node = geometry.node(self.x[y, x + 1], self.y[y, x + 1])
                upper_left_node = geometry.node(self.x[y + 1, x], self.y[y + 1, x])
                upper_right_node = geometry.node(self.x[y + 1, x + 1], self.y[y + 1, x + 1])
                cells[y][x] = cell.cell(lower_left_node, lower_right_node, upper_left_node, upper_right_node)
        return(cells)

    def get_centroids(self):
        x_centroid = np.zeros((self.ny_cells, self.nx_cells))
        y_centroid = np.zeros((self.ny_cells, self.nx_cells))
        for y in range(self.ny_cells):
            for x in range(self.nx_cells):
                x_centroid[y,x] = self.cells[y][x].centroid.x
                y_centroid[y, x] = self.cells[y][x].centroid.y
        return(x_centroid,y_centroid)

    def get_Volumes(self):
        x_centroid = np.zeros((self.ny_cells,self.nx_cells))
        y_centroid = np.zeros((self.ny_cells,self.nx_cells))
        V = np.zeros((self.ny_cells,self.nx_cells))
        for y in range(self.ny_cells):
            for x in range(self.nx_cells):
                x_centroid[y,x] = self.cells[y][x].centroid.x
                y_centroid[y, x] = self.cells[y][x].centroid.y
                V[y,x] = self.cells[y][x].V
        return(V,x_centroid,y_centroid)

    def get_surfaces(self,surf):
        x_surface = np.zeros((self.ny_cells, self.nx_cells))
        y_surface = np.zeros((self.ny_cells, self.nx_cells))
        S = np.zeros((self.ny_cells, self.nx_cells, 2))
        for y in range(self.ny_cells):
            for x in range(self.nx_cells):
                x_surface[y, x] = getattr(self.cells[y][x],surf).initial_node.x
                y_surface[y, x] = getattr(self.cells[y][x],surf).initial_node.y
                S[y, x] = getattr(self.cells[y][x],surf).vector_representation
        return (S, x_surface, y_surface)

    def get_vect(self,vect):
        x_surface = np.zeros((self.ny_cells, self.nx_cells))
        y_surface = np.zeros((self.ny_cells, self.nx_cells))
        S = np.zeros((self.ny_cells, self.nx_cells, 2))
        for y in range(self.ny_cells):
            for x in range(self.nx_cells):
                x_surface[y, x] = getattr(self.cells[y][x],vect)().initial_node.x
                y_surface[y, x] = getattr(self.cells[y][x],vect)().initial_node.y
                S[y, x] = getattr(self.cells[y][x],vect)().vector_representation
        return (S, x_surface, y_surface)

    def plot_direction_vectors(self):
        x_start = []
        y_start = []
        vect_x = []
        vect_y = []
        attr = ['n_xi_plus_half','n_xi_minus_half','n_eta_minus_half','n_eta_plus_half']
        for y in range(self.ny_cells):
            for x in range(self.nx_cells):
                for at in attr:
                    n = getattr(self.cells[y][x],at)()
                    x_start.append(n.initial_node.x)
                    y_start.append(n.initial_node.y)
                    vect_x.append(n.x)
                    vect_y.append(n.y)
                    print(n.x,n.y)
        plt.quiver(x_start, y_start, vect_x, vect_y,scale=50,headwidth=2,headlength=5)

    def plot_surface_vectors(self):
        x_start = []
        y_start = []
        vect_x = []
        vect_y = []
        attr = ['Sbottom','Sleft','Sright','Stop']
        for y in range(self.ny_cells):
            for x in range(self.nx_cells):
                for at in attr:
                    n = getattr(self.cells[y][x],at)
                    x_start.append(n.initial_node.x)
                    y_start.append(n.initial_node.y)
                    vect_x.append(n.x)
                    vect_y.append(n.y)
        plt.quiver(x_start, y_start, vect_x, vect_y)


class computational_grid(grid_by_cells):
    def __init__(self,grid_directory,grid_name):
        grid_by_cells.__init__(self,grid_directory,grid_name)
        if not hasattr(self,'augmented_grid'):
            self.augment_grid()
        self.convert_geometric_cells_to_computational_cells()
        self.intricate_base_to_augmented_grid()

    def get_cells_physical_value(self,value):
        val = np.zeros((self.ny_cells, self.nx_cells))
        for y in range(self.ny_cells):
            for x in range(self.nx_cells):
                val[y, x] = getattr(self.cells[y][x],value)
        return (val)

    def get_cells_computed_physical_value(self,value):
        val = np.zeros((self.ny_cells, self.nx_cells))
        for y in range(self.ny_cells):
            for x in range(self.nx_cells):
                val[y, x] = getattr(self.cells[y][x],value)()
        return (val)

    def convert_geometric_cells_to_computational_cells(self):
        for y in range(self.augmented_grid.ny_cells):
            for x in range(self.augmented_grid.nx_cells):
                self.augmented_grid.cells[y][x] = cell.computational_cell(self.augmented_grid.cells[y][x])

    def intricate_base_to_augmented_grid(self):
        for y in range(self.ny_cells):
            for x in range(self.nx_cells):
                self.cells[y][x] = self.augmented_grid.cells[y + 1][x + 1]

    def initialize_cells(self):
        """
        sets all the interior points at the initial condition given in the project assignment
        :param nx, ny: (int) size of the augmented grid which is not the number of cells
        :param bottom_surfaces: (3D array) matrix containing the top and bottom surface vectors
        :return:(3D array) Q
        """
        for y in range(self.augmented_grid.ny_cells):
            for x in range(self.augmented_grid.nx_cells):
                self.augmented_grid.cells[y][x].rho = rhoini
                self.augmented_grid.cells[y][x].rho_u = rhoini*Mini*self.augmented_grid.cells[y][x].c(Tini)
                self.augmented_grid.cells[y][x].rho_v = 0
                self.augmented_grid.cells[y][x].rho_et = Pini/(gamma-1) + rhoini*(Mini*self.augmented_grid.cells[y][x].c(Tini))**2/2

    def initialize_cells_with_save(self,rho,rho_u,rho_v,rho_et):
        for y in range(self.ny_cells):
            for x in range(self.nx_cells):
                self.cells[y][x].rho = rho[y][x]
                self.cells[y][x].rho_u = rho_u[y][x]
                self.cells[y][x].rho_v = rho_v[y][x]
                self.cells[y][x].rho_et = rho_et[y][x]


    def set_boundary_conditions(self):
        # set inlet conditions
        for y in range(self.augmented_grid.ny_cells):
            # set inlet conditions
            self.augmented_grid.cells[y][0].rho = rhoini
            self.augmented_grid.cells[y][0].rho_u = rhoini * Mini * self.augmented_grid.cells[y][0].c(Tini)
            self.augmented_grid.cells[y][0].rho_v = 0
            self.augmented_grid.cells[y][0].rho_et = Pini / (gamma - 1) + rhoini * (Mini * self.augmented_grid.cells[y][0].c(Tini)) ** 2 / 2

            # setting all the hallo cells on the right to the value of the last column of computed cells
            self.augmented_grid.cells[y][-1].Q = self.augmented_grid.cells[y][-2].Q

        for x in range(self.nx_cells):
            # slip wall condition #
            # top wall
            # this approximation is fine as the top wall is flat, but in a second time it could be verified more rigorously
            self.augmented_grid.cells[-1][x].rho_u = self.augmented_grid.cells[-1][x].rho * self.augmented_grid.cells[-2][x].u()
            self.augmented_grid.cells[-1][x].rho_v = self.augmented_grid.cells[-1][x].rho * self.augmented_grid.cells[-2][x].v()

            # bottom wall
            n_eta = self.augmented_grid.cells[0][x].Stop.get_normalized()
            self.augmented_grid.cells[0][x].rho_u = self.augmented_grid.cells[0][x].rho * (self.augmented_grid.cells[1][x].u()*(n_eta.y**2-n_eta.x**2)-2*n_eta.x*n_eta.y*self.augmented_grid.cells[1][x].v())
            self.augmented_grid.cells[0][x].rho_v = self.augmented_grid.cells[0][x].rho * (self.augmented_grid.cells[1][x].v() * (n_eta.x ** 2 - n_eta.y ** 2) - 2 * n_eta.x * n_eta.y *self.augmented_grid.cells[1][x].u())

            # adiabatic condition #
            # top wall
            self.augmented_grid.cells[-1][x].rho_et = r*self.augmented_grid.cells[-1][x].rho / (gamma-1) * ((gamma - 1) * (self.augmented_grid.cells[-2][x].rho_et - 1/2 * (self.augmented_grid.cells[-2][x].rho_u**2+self.augmented_grid.cells[-2][x].rho_v**2)/self.augmented_grid.cells[-2][x].rho)/r/self.augmented_grid.cells[-2][x].rho) +1/2*(self.augmented_grid.cells[-1][x].rho_u**2+self.augmented_grid.cells[-1][x].rho_v**2)/self.augmented_grid.cells[-1][x].rho
            # bottom wall
            self.augmented_grid.cells[0][x].rho_et = r * self.augmented_grid.cells[0][x].rho / (gamma - 1) * ((gamma - 1) * (self.augmented_grid.cells[1][x].rho_et - 1 / 2 * (self.augmented_grid.cells[1][x].rho_u ** 2 + self.augmented_grid.cells[1][x].rho_v ** 2) /self.augmented_grid.cells[1][x].rho) / r / self.augmented_grid.cells[1][x].rho) + 1 / 2 * (self.augmented_grid.cells[0][x].rho_u ** 2 + self.augmented_grid.cells[0][x].rho_v ** 2) / self.augmented_grid.cells[0][x].rho

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
        x_centroid, y_centroid = self.get_centroids()
        rho = self.get_cells_physical_value('rho')
        u = self.get_cells_computed_physical_value('u')
        v = self.get_cells_computed_physical_value('v')
        rho_et = self.get_cells_physical_value('rho_et')
        plt.contourf(x_centroid, y_centroid, rho, levels=levels, cmap=colorScheme)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$\rho$')
        plt.show()
        plt.contourf(x_centroid, y_centroid, u, levels=levels, cmap=colorScheme)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$u$')
        plt.show()
        plt.contourf(x_centroid, y_centroid, v, levels=levels, cmap=colorScheme)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$v$')
        plt.show()
        plt.contourf(x_centroid, y_centroid, rho_et, levels=levels, cmap=colorScheme)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$\rho e_{t}$')
        plt.show()
        return('')


if __name__=='__main__':
    g = grid_by_cells(r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\grids', 'g33x25u.dat')
    aug_g = g.augment_grid()
    aug_g.plot_grid(color='k')
    g.plot_grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Base grid overlayed on augmented grid')
    plt.show()