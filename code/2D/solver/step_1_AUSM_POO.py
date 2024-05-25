import numpy as np
import matplotlib.pyplot as plt
import grid


def Mach_split(M):
    if M <= -1:
        M_plus = 0
        M_minus = 0.5 * (M - abs(M))
    elif M >= 1:
        M_plus = 0.5 * (M + abs(M))
        M_minus = 0
    else:
        M_plus = 0.25 * (M + 1) ** 2
        M_minus = -0.25 * (M - 1) ** 2
    return (M_plus, M_minus)


def Pressure_split(P, M):
    if M <= -1:
        P_plus = 0
        P_minus = (P / 2) * (M - abs(M)) / M
    elif M >= 1:
        P_plus = (P / 2) * (M + abs(M)) / M
        P_minus = 0
    else:
        P_plus = (P/4) * (M+1)**2 * (2-M)
        P_minus = (P / 4) * (M - 1) ** 2 * (2 + M)
        # P_plus = (P / 2) * (1 + M)
        # P_minus = (P / 2) * (1 - M)
    return (P_plus, P_minus)


def convective_flux(cell):
    E = [0, 0, 0, 0]
    E[0] = cell.rho * cell.c()
    E[1] = cell.rho_u * cell.c()
    E[2] = cell.rho_v * cell.c()
    E[3] = cell.rho * cell.ht() * cell.c()
    return (E)

def flux_split_xi(cell):

    # first try with contravariant velocity extrapolated at cell center
    n_p = cell.n_xi_plus_half()
    n_m = cell.n_xi_minus_half()
    n_mean_x = (n_p.x-n_m.x)/2
    n_mean_y = (n_p.y - n_m.y) / 2
    u_contra = n_mean_x * cell.u() + n_mean_y * cell.v()
    #compute Mach
    M = u_contra/cell.c()
    #compute pressure
    P = cell.P()
    # compute mache and pressure split and convective flux
    M_p, M_m = Mach_split(M)
    P_p, P_m = Pressure_split(P, M)
    conv_flux = convective_flux(cell)
    # split the flux
    flux_p_xi = [M_p * conv_flux[0], M_p * conv_flux[1] + n_mean_x * P_p, M_p * conv_flux[2] + n_mean_y * P_p, M_p * conv_flux[3]]
    flux_m_xi = [M_m * conv_flux[0], M_m * conv_flux[1] + n_mean_x * P_m, M_m * conv_flux[2] + n_mean_y * P_m, M_m * conv_flux[3]]
    return(flux_p_xi,flux_m_xi)


def flux_split_eta(cell):
    # first try with contravariant velocity extrapolated at cell center
    n_p = cell.n_eta_plus_half()
    n_m = cell.n_eta_minus_half()
    n_mean_x = (n_p.x - n_m.x) / 2
    n_mean_y = (n_p.y - n_m.y) / 2
    v_contra = n_mean_x * cell.u() + n_mean_y * cell.v()
    #compute Mach
    M = v_contra/cell.c()
    #compute pressure
    P = cell.P()
    # compute mache and pressure split and convective flux
    M_p, M_m = Mach_split(M)
    P_p, P_m = Pressure_split(P, M)
    conv_flux = convective_flux(cell)
    # split the flux
    flux_p_eta = [M_p * conv_flux[0], M_p * conv_flux[1] + n_mean_x * P_p, M_p * conv_flux[2] + n_mean_y * P_p, M_p * conv_flux[3]]
    flux_m_eta = [M_m * conv_flux[0], M_m * conv_flux[1] + n_mean_x * P_m, M_m * conv_flux[2] + n_mean_y * P_m, M_m * conv_flux[3]]
    return(flux_p_eta,flux_m_eta)

class solver(object):
    def __init__(self,grid):
        self.grid = grid
        self.grid.initialize_cells()
        self.grid.set_boundary_conditions()

    def one_step_AUSM(self,dt):
        # Compute fluxes #
        st = [[0 for i in range(self.grid.augmented_grid.nx_cells)] for j in range(self.grid.augmented_grid.ny_cells)]
        for y in range(1, self.grid.augmented_grid.ny_cells - 1):
            for x in range(1, self.grid.augmented_grid.nx_cells - 1):
                # Computing fluxes with extrapolated values #
                flux_p_xi, flux_m_xi = flux_split_xi(self.grid.augmented_grid.cells[y][x])
                flux_p_xi_plus_1, flux_m_xi_plus_1 = flux_split_xi(self.grid.augmented_grid.cells[y][x + 1])
                flux_p_xi_minus_1, flux_m_xi_minus_1 = flux_split_xi(self.grid.augmented_grid.cells[y][x - 1])
                flux_p_eta, flux_m_eta = flux_split_eta(self.grid.augmented_grid.cells[y][x])
                flux_p_eta_plus_1, flux_m_eta_plus_1 = flux_split_eta(self.grid.augmented_grid.cells[y + 1][x])
                flux_p_eta_minus_1, flux_m_eta_minus_1 = flux_split_eta(self.grid.augmented_grid.cells[y - 1][x])

                # computing the step of the iteration #
                var_x = [0, 0, 0, 0]
                var_y = [0, 0, 0, 0]
                var_tot = [0, 0, 0, 0]
                var_x[0] = (flux_p_xi[0] + flux_m_xi_plus_1[0]) * self.grid.augmented_grid.cells[y][x].Sright.norm() - (flux_m_xi[0] + flux_p_xi_minus_1[0]) * self.grid.augmented_grid.cells[y][x].Sleft.norm()
                var_x[1] = (flux_p_xi[1] + flux_m_xi_plus_1[1]) * self.grid.augmented_grid.cells[y][x].Sright.norm() - (flux_m_xi[1] + flux_p_xi_minus_1[1]) * self.grid.augmented_grid.cells[y][x].Sleft.norm()
                var_x[2] = (flux_p_xi[2] + flux_m_xi_plus_1[2]) * self.grid.augmented_grid.cells[y][x].Sright.norm() - (flux_m_xi[2] + flux_p_xi_minus_1[2]) * self.grid.augmented_grid.cells[y][x].Sleft.norm()
                var_x[3] = (flux_p_xi[3] + flux_m_xi_plus_1[3]) * self.grid.augmented_grid.cells[y][x].Sright.norm() - (flux_m_xi[3] + flux_p_xi_minus_1[3]) * self.grid.augmented_grid.cells[y][x].Sleft.norm()
                var_y[0] = (flux_p_eta[0] + flux_m_eta_plus_1[0]) * self.grid.augmented_grid.cells[y][x].Stop.norm() - (flux_m_eta[0] + flux_p_eta_minus_1[0]) * self.grid.augmented_grid.cells[y][x].Sbottom.norm()
                var_y[1] = (flux_p_eta[1] + flux_m_eta_plus_1[1]) * self.grid.augmented_grid.cells[y][x].Stop.norm() - (flux_m_eta[1] + flux_p_eta_minus_1[1]) * self.grid.augmented_grid.cells[y][x].Sbottom.norm()
                var_y[2] = (flux_p_eta[2] + flux_m_eta_plus_1[2]) * self.grid.augmented_grid.cells[y][x].Stop.norm() - (flux_m_eta[2] + flux_p_eta_minus_1[2]) * self.grid.augmented_grid.cells[y][x].Sbottom.norm()
                var_y[3] = (flux_p_eta[3] + flux_m_eta_plus_1[3]) * self.grid.augmented_grid.cells[y][x].Stop.norm() - (flux_m_eta[3] + flux_p_eta_minus_1[3]) * self.grid.augmented_grid.cells[y][x].Sbottom.norm()
                var_tot[0] = var_x[0] + var_y[0]
                var_tot[1] = var_x[1] + var_y[1]
                var_tot[2] = var_x[2] + var_y[2]
                var_tot[3] = var_x[3] + var_y[3]
                st[y][x] = var_tot

        # apply the step #
        for y in range(1, self.grid.augmented_grid.ny_cells - 1):
            for x in range(1, self.grid.augmented_grid.nx_cells - 1):
                self.grid.augmented_grid.cells[y][x].rho = self.grid.augmented_grid.cells[y][x].rho - dt * st[y][x][0] / self.grid.augmented_grid.cells[y][x].V
                self.grid.augmented_grid.cells[y][x].rho_u = self.grid.augmented_grid.cells[y][x].rho_u - dt * st[y][x][1] / self.grid.augmented_grid.cells[y][x].V
                self.grid.augmented_grid.cells[y][x].rho_v = self.grid.augmented_grid.cells[y][x].rho_v - dt * st[y][x][2] / self.grid.augmented_grid.cells[y][x].V
                self.grid.augmented_grid.cells[y][x].rho_et = self.grid.augmented_grid.cells[y][x].rho_et - dt * st[y][x][3] / self.grid.augmented_grid.cells[y][x].V

        # reset the boundary conditions #
        self.grid.set_boundary_conditions()

if __name__ == '__main__':
    dt = 0.00001
    colorScheme = 'jet'
    g = grid.computational_grid(r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\grids\2D', 'g33x25u.dat')
    x_centroid, y_centroid = g.get_centroids()
    calcul = solver(g)
    for i in range(10000):
        T = i * dt
        calcul.one_step_AUSM(dt)
        if i % 1000 == 0:
            print('T =', T)
            rho = g.get_cells_physical_value('rho')
            plt.contourf(x_centroid, y_centroid, rho, levels=20, cmap=colorScheme)
            plt.colorbar()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(r'$\rho$')
            plt.show()