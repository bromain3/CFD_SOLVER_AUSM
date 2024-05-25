import grid1D as grid
#!!! We have huge oscilations in transient, but we manage to avoid explosion by forcing T to be above 100K and dt to be small enough

def Mach_split(M):
    """
    Define the logic for mach splitting as it is defined in the AUSM scheme
    :param M (float) : Mach number to be split
    :return: M_plus,M_minus (float,float) : Split mach number at each cell face
    """
    if M <= -1:
        M_plus = 0
        M_minus = M
    elif M < 1:
        M_plus = 0.25 * (M+1)**2
        M_minus = - 0.25 * (M - 1)**2
    else:
        M_plus = M
        M_minus = 0
    return (M_plus, M_minus)


def Pressure_split(P, M):
    """
    Define the logic for Pressure splitting as it is defined in the AUSM scheme
    :param M (float) : Mach number used to compute the pressure split
    :param P (float) : Pressure to be split
    :return: P_plus,P_minus (float,float) : Split Pressure at each cell face
    """
    if M <= -1:
        P_plus = 0
        P_minus = P
    elif M < 1:
        P_plus = 1/2 * (1 + M) * P
        P_minus = 0.5 * (1 - M) * P
    else:
        P_plus = P
        P_minus = 0
    return (P_plus, P_minus)


def convective_flux(cell):
    """
    Gives the convective flux of a given cell
    :param cell (cell1D.computational_cell): cell that we want to extract the convective flux of
    :return: E (list): convective flux
    """
    E = [0, 0, 0]
    E[0] = cell.rho * cell.c()
    E[1] = cell.rho_u * cell.c()
    E[2] = cell.rho * cell.ht() * cell.c()
    return (E)

def flux_split(cell):
    """
    Define the logic for flux splitting as it is defined in the AUSM scheme
    :param cell (cell1D.computational_cell): cell that we want to extract the splited flux of
    :return: flux_p,flux_m (float, float): right and left splited flux
    """
    #compute Mach
    M = cell.u()/cell.c()
    #compute pressure
    P = cell.P()
    # compute mache and pressure split and convective flux
    M_p, M_m = Mach_split(M)
    P_p, P_m = Pressure_split(P, M)
    conv_flux = convective_flux(cell)
    # split the flux

    return(conv_flux,P_p, P_m,M_p, M_m)


class solver(object):
    def __init__(self,grid):
        """
        Constructor of the solver class. It takes a grid as input and initialize it
        :param grid (grid1D.computational_grid): Grid that we want to make computations on
        """
        self.grid = grid
        self.grid.initialize_cells()
        self.grid.set_boundary_conditions()

    def one_step_AUSM(self, dt):
        """
        This function computes the spatial evolution at the next time step of a size dt
        :param dt (float): time step used for computation
        :return : None
        """
        # it seems that this scheme is not physicaly representative with low speed conditions
        # it also shows big oscilations in trans-sonic flows (but it still works (but is surely inacurate) thanks to the condition on minimal temperature)
        # Compute fluxes #
        st = [0 for i in range(self.grid.augmented_grid.nx_cells)]
        for x in range(1, self.grid.augmented_grid.nx_cells - 1):
            # Computing fluxes with extrapolated values #
            conv_flux_x ,P_p_x, P_m_x, M_p_x, M_m_x = flux_split(self.grid.augmented_grid.cells[x])
            conv_flux_x_plus_1 ,P_p_x_plus_1, P_m_x_plus_1, M_p_x_plus_1, M_m_x_plus_1 = flux_split(self.grid.augmented_grid.cells[x + 1])
            conv_flux_x_minus_1 ,P_p_x_minus_1, P_m_x_minus_1, M_p_x_minus_1, M_m_x_minus_1 = flux_split(self.grid.augmented_grid.cells[x - 1])

            # computing the step of the iteration #
            var_x = [0, 0, 0]
            var_x[0] = (max(0,M_p_x+M_m_x_plus_1) * conv_flux_x[0] + min(0,M_p_x+M_m_x_plus_1) * conv_flux_x_plus_1[0]) * self.grid.augmented_grid.cells[x].S1 - \
                       (max(0,M_p_x_minus_1+M_m_x) * conv_flux_x_minus_1[0] + min(0,M_p_x_minus_1+M_m_x) * conv_flux_x[0]) * self.grid.augmented_grid.cells[x].S0
            var_x[1] = (max(0,M_p_x+M_m_x_plus_1) * conv_flux_x[1] + min(0,M_p_x+M_m_x_plus_1) * conv_flux_x_plus_1[1] + P_p_x + P_m_x_plus_1) * self.grid.augmented_grid.cells[x].S1 - \
                       (max(0,M_p_x_minus_1+M_m_x) * conv_flux_x_minus_1[1] + min(0,M_p_x_minus_1+M_m_x) * conv_flux_x[1] + P_p_x_minus_1 + P_m_x) * self.grid.augmented_grid.cells[x].S0
            var_x[2] = (max(0,M_p_x+M_m_x_plus_1) * conv_flux_x[2] + min(0,M_p_x+M_m_x_plus_1) * conv_flux_x_plus_1[2]) * self.grid.augmented_grid.cells[x].S1 - \
                       (max(0,M_p_x_minus_1+M_m_x) * conv_flux_x_minus_1[2] + min(0,M_p_x_minus_1+M_m_x) * conv_flux_x[2]) * self.grid.augmented_grid.cells[x].S0
            st[x] = var_x

        # apply the step #
        for x in range(1, self.grid.augmented_grid.nx_cells - 1):
            self.grid.augmented_grid.cells[x].rho = self.grid.augmented_grid.cells[x].rho - dt * st[x][0] / self.grid.augmented_grid.cells[x].V
            self.grid.augmented_grid.cells[x].rho_u = self.grid.augmented_grid.cells[x].rho_u - dt * st[x][1] / self.grid.augmented_grid.cells[x].V
            self.grid.augmented_grid.cells[x].rho_et = self.grid.augmented_grid.cells[x].rho_et - dt * st[x][2] / self.grid.augmented_grid.cells[x].V

            # reset the boundary conditions #
            self.grid.set_boundary_conditions()


if __name__ == '__main__':
    dt = 0.000001
    T = 0
    path = r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\grids\1D'
    grid_name = 'diamond_airfoil_coarse.dat'
    g = grid.computational_grid(path, grid_name)
    calcul = solver(g)
    for i in range(100000):
        T = i*dt
        calcul.one_step_AUSM(dt)
        if i % 10000 == 0:
            print(g)
            print('T =',T)