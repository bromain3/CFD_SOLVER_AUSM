import os
import matplotlib.pyplot as plt
from initial_conditions import *
from air import *
import cell1D as cell

colorScheme = 'jet'

class grid_by_node(object):
    def __init__(self, grid_directory, grid_name):
        """
        Initialisation function of the grid_by_node class
        :param grid_directory (str): directory in which is stored the grid file
        :param grid_name (str): name of the grid file
        """
        self.grid_directory = grid_directory
        self.grid_name = grid_name
        self.nx, self.x, self.S = self.read_grid()

    def read_grid(self):
        """
        Reads the grid file and creates python variables that represents it
        :return nx (int): number of nodes in the grid
        :return x (list): list of coordinates of nodes
        :return S (list): list of the surface associated to each node
        """
        with open(os.path.join(self.grid_directory, self.grid_name), 'r') as grid:
            grid.readline()
            val = grid.readlines()
        nx = len(val)
        x = []
        S = []
        for line in val:
            l = line.split(('\t'))
            x.append(float(l[0]))
            S.append(float(l[1]))
        return(nx,x,S)

    def augment_grid(self,save=False):
        """
        Adds shadow nodes to the base grid and can save it if wanted
        :param save (bool): Flag to say if you want to save the augmented grid or not
        :return augmented_grid (grid_by_cell): object representing the augmented grid
        """
        if type(self) == computational_grid:
            print('WARNING, the computational grid has been augmented. There is no problem if this message appeares at grid creation. However, this will creat bugs if you augmented it by hand')
        new_x = [2*self.x[0]-self.x[1]] + self.x + [2*self.x[-1]+self.x[-1]]
        new_S = [self.S[0]] + self.S + [self.S[-1]]
        with open(os.path.join(self.grid_directory, self.grid_name.split('.')[0]+'_augmented'+self.grid_name.split('.')[1]), 'w') as grid:
            grid.write('nodes_x\tsurface\n')
            for i in range(len(new_x) - 1):
                grid.write(str(new_x[i]) + '\t' + str(new_S[i]) + '\n')
        self.augmented_grid = grid_by_cell(self.grid_directory,self.grid_name.split('.')[0]+'_augmented'+self.grid_name.split('.')[1])
        if not save:
            os.remove(os.path.join(self.grid_directory,self.grid_name.split('.')[0]+'_augmented'+self.grid_name.split('.')[1]))
        return(self.augmented_grid)

    def plot_grid(self,ax,show_cell=False):
        """
        Plots the grid on a given axis with or without showing the cells
        :param ax (matplotlib.pyplot.ax): axes on which to plot the grid
        :param show_cell (bool): flag to say if you want to show cell limits or not
        :return: None
        """
        upper_line = []
        lower_line = []
        cell_limits = []
        for i in range(len(self.x)):
            upper_line.append(self.S[i] / 2)
            lower_line.append( - self.S[i] / 2)
            cell_limits.append([[self.x[i],self.x[i]],[self.S[i] / 2,-self.S[i] / 2]])
        ref = ax.plot(self.x,upper_line,color='k')
        ax.plot(self.x,lower_line,color=ref[0].get_color(),linewidth=ref[0].get_linewidth())
        if show_cell:
            for i in cell_limits:
                ax.plot(i[0],i[1],color=ref[0].get_color(),linewidth=ref[0].get_linewidth()/2)


class grid_by_cell(grid_by_node):
    def __init__(self,grid_directory, grid_name):
        """
        Initialisation function of the grid_by_cell class
        :param grid_directory (str): directory in which is stored the grid file
        :param grid_name (str): name of the grid file
        """
        grid_by_node.__init__(self,grid_directory, grid_name)
        self.nx_cells = self.nx - 1
        self.cells = self.convert_nodes_to_cells()

    def convert_nodes_to_cells(self):
        """
        Converts the grid defined by nodes to a grid defined by cells
        :return cells (list of cell1D.cell): return the list of every cell object of the grid
        """
        cells = []
        for i in range(self.nx_cells):
            cells.append(cell.cell(self.x[i],self.x[i+1],self.S[i],self.S[i+1]))
        return(cells)

    def get_centroids(self):
        """
        extract the centroids of every cell of the grid
        :return (list): centroid of each cell
        """
        centroids = []
        for c in self.cells:
            centroids.append(c.centroid)
        return(centroids)

    def get_volumes(self):
        """
        extract the volumes of every cell of the grid
        :return (list): volume of each cell
        """
        volumes = []
        for c in self.cells:
            volumes.append(c.V)
        return(volumes)

class computational_grid(grid_by_cell):
    def __init__(self,grid_directory,grid_name):
        """
        Initialisation function of the grid by node class
        :param grid_directory (str): directory in which is stored the grid file
        :param grid_name (str): name of the grid file
        """
        grid_by_cell.__init__(self,grid_directory,grid_name)
        if not hasattr(self,'augmented_grid'):
            self.augment_grid()
        self.convert_geometric_cells_to_computational_cells()
        self.intricate_base_to_augmented_grid()

    def convert_geometric_cells_to_computational_cells(self):
        """
        convert the list of cells to a list of computational_cell (for the augmented grid)
        :return: None
        """
        for x in range(self.augmented_grid.nx_cells):
            self.augmented_grid.cells[x] = cell.computational_cell(self.augmented_grid.cells[x])

    def intricate_base_to_augmented_grid(self):
        """
        Make every cell in the base grid point to the same computational cell as the augmented grid
        :return: None
        """
        for x in range(self.nx_cells):
            self.cells[x] = self.augmented_grid.cells[x + 1]

    def get_cells_physical_value(self,value):
        """
        Returns the value of the parameter given in input. Only works for state variables (rho, rho_u and rho_et)
        :param value (str): value that you want to extract (rho,rho_u and rho_et)
        :return values (list): value that you asked for, for each node
        """
        val = []
        for x in range(self.nx_cells):
            val.append(getattr(self.cells[x],value))
        return (val)

    def get_cells_computed_physical_value(self,value):
        """
        Returns the value of the parameter given in input. Only works for variable derived from state variables (P, c, T, ...)
        :param value (str): value that you want to extract (P, c, T, ...)
        :return values (list): value that you asked for, for each node
        """
        val = []
        for x in range(self.nx_cells):
            val.append(getattr(self.cells[x],value)())
        return (val)

    def initialize_cells(self):
        """
        Set the initial value for each cell
        :return: None
        """
        for x in range(self.augmented_grid.nx_cells):
            """
            # Initial values for Rieman problem
            if self.augmented_grid.cells[x].centroid < x0:
                self.augmented_grid.cells[x].rho = rho_l
                self.augmented_grid.cells[x].rho_u = rho_l * u_l
                self.augmented_grid.cells[x].rho_et = p_l/(gamma-1) + 1/2*(rho_l*u_l)**2/rho_l
            else:
                self.augmented_grid.cells[x].rho = rho_r
                self.augmented_grid.cells[x].rho_u = rho_r * u_r
                self.augmented_grid.cells[x].rho_et = p_r / (gamma - 1) + 1 / 2 * (rho_r * u_r) ** 2 / rho_r
            """
            self.augmented_grid.cells[x].rho = rhoini
            self.augmented_grid.cells[x].rho_u = rhoini * Mini * self.augmented_grid.cells[x].c(Tini)
            self.augmented_grid.cells[x].rho_et = Pini / (gamma - 1) + rhoini * (Mini * self.augmented_grid.cells[x].c(Tini)) ** 2 / 2

    def set_boundary_conditions(self):
        """
        Set boundary conditions for halo cells
        :return: None
        """
        """
        # Bondary condititons for Rieman problem
        # set inlet conditions
        self.augmented_grid.cells[0].rho = self.augmented_grid.cells[1].rho
        self.augmented_grid.cells[0].rho_u = self.augmented_grid.cells[1].rho_u
        self.augmented_grid.cells[0].rho_et = self.augmented_grid.cells[1].rho_et
        # set outlet conditions
        self.augmented_grid.cells[-1].rho = self.augmented_grid.cells[-2].rho
        self.augmented_grid.cells[-1].rho_u = self.augmented_grid.cells[-2].rho_u
        self.augmented_grid.cells[-1].rho_et = self.augmented_grid.cells[-2].rho_et
        """
        # set inlet conditions
        self.augmented_grid.cells[0].rho = rhoini
        self.augmented_grid.cells[0].rho_u = rhoini * Mini * self.augmented_grid.cells[0].c(Tini)
        self.augmented_grid.cells[0].rho_et = Pini / (gamma - 1) + rhoini * (Mini * self.augmented_grid.cells[0].c(Tini)) ** 2 / 2
        # set outlet conditions
        self.augmented_grid.cells[-1].rho = self.augmented_grid.cells[-2].rho
        self.augmented_grid.cells[-1].rho_u = self.augmented_grid.cells[-2].rho_u
        self.augmented_grid.cells[-1].rho_et = self.augmented_grid.cells[-2].rho_et

    def __str__(self):
        """
        Overload of the string function. Allows to plot state variables for the whole grid
        :return Constant (str)
        """
        centroids= self.get_centroids()
        rho = self.get_cells_physical_value('rho')
        u = self.get_cells_computed_physical_value('u')
        rho_et = self.get_cells_physical_value('rho_et')

        fig, ax = plt.subplots(2)

        self.plot_grid(ax[0], True)
        ax[0].scatter(centroids, [0 for i in centroids], marker='x', color=ax[0].lines[0].get_color())
        ax[0].set_title('Graphical representation')

        l1 = ax[1].plot(centroids, rho,label=r'$\rho$')
        ax[1].spines.left.set_color(l1[0].get_color())
        ax[1].tick_params(axis='y', colors=ax[1].lines[0].get_color())

        nax = ax[1].twinx()
        l2 = nax.plot(centroids,u,label=r'$u$',color='g')
        nax.spines.right.set_color(nax.lines[0].get_color())
        nax.tick_params(axis='y', colors=nax.lines[0].get_color())

        nnax = ax[1].twinx()
        nnax.spines.right.set_position(("axes", 1.1))
        l3 = nnax.plot(centroids, rho_et,label=r'$\rho e_{t}$',color='r')
        nnax.spines['right'].set_color(nnax.lines[0].get_color())
        nnax.tick_params(axis='y', colors=nnax.lines[0].get_color())

        ax[1].set_title('Physical properties')
        ax[1].set_xlim(ax[0].get_xlim())
        lns = l1 + l2 + l3
        labs = [l.get_label() for l in lns]
        ax[1].legend(lns, labs)

        plt.tight_layout()
        plt.show()

        return('The main grid properties have been succesfully plotted')

if __name__ == '__main__':
    path = r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\grids\1D'
    grid_name = 'diamond_airfoil_coarse.dat'
    g = computational_grid(path,grid_name)
    fig,ax = plt.subplots()
    g.plot_grid(ax,True)
    plt.show()
    g.initialize_cells()
    print(g)