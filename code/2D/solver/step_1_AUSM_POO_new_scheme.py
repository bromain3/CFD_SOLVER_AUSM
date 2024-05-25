import numpy as np
import matplotlib.pyplot as plt
import grid


def Mach_split(M):
    if M <= -1:
        M_plus = 0
        M_minus = M
    elif M < 1:
        M_plus = 0.25 * (M + 1) ** 2
        M_minus = - 0.25 * (M - 1) ** 2
    else:
        M_plus = M
        M_minus = 0
    return (M_plus, M_minus)


def Pressure_split(P, M):
    if M <= -1:
        P_plus = 0
        P_minus = P
    elif M < 1:
        P_plus = 1 / 2 * (1 + M) * P
        P_minus = 0.5 * (1 - M) * P
    else:
        P_plus = P
        P_minus = 0
    return (P_plus, P_minus)


def convective_flux(cell):
    E = [0, 0, 0, 0]
    E[0] = cell.rho * cell.c()
    E[1] = cell.rho_u * cell.c()
    E[2] = cell.rho_v * cell.c()
    E[3] = cell.rho * cell.ht() * cell.c()
    return (E)

def flux_split_xi(cell):
    """
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
    """
    # second try with contravariant velocity computed at xi+1/2 for flux_p and xi-1/2 for the other (this should be the best methode as the teacher at GT told "you don't extrapolate geometrical values")
    n_p = cell.n_xi_plus_half()
    n_m = cell.n_xi_minus_half()
    u_contra_p = n_p.x * cell.u() + n_p.y * cell.v()
    u_contra_m = -n_m.x * cell.u() - n_m.y * cell.v()
    # compute Mach
    M_m = u_contra_m / cell.c()
    M_p = u_contra_p / cell.c()
    # compute pressure
    P = cell.P()
    # compute mache and pressure split and convective flux
    M_p, _ = Mach_split(M_p)
    _, M_m = Mach_split(M_m)
    P_p, _ = Pressure_split(P, M_p)
    _, P_m = Pressure_split(P, M_m)
    conv_flux = convective_flux(cell)
    return(conv_flux,P_p, P_m,M_p, M_m,n_p,n_m)


def flux_split_eta(cell):
    """
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
    """
    # second try with contravariant velocity computed at xi+1/2 for flux_p and xi-1/2 for the other (this should be the best methode as the teacher at GT told "you don't extrapolate geometrical values")
    n_p = cell.n_eta_plus_half()
    n_m = cell.n_eta_minus_half()
    u_contra_p = n_p.x * cell.u() + n_p.y * cell.v()
    u_contra_m = -n_m.x * cell.u() - n_m.y * cell.v()
    # compute Mach
    M_m = u_contra_m / cell.c()
    M_p = u_contra_p / cell.c()
    # compute pressure
    P = cell.P()
    # compute mache and pressure split and convective flux
    M_p, _ = Mach_split(M_p)
    _, M_m = Mach_split(M_m)
    P_p, _ = Pressure_split(P, M_p)
    _, P_m = Pressure_split(P, M_m)
    conv_flux = convective_flux(cell)
    return(conv_flux,P_p, P_m,M_p, M_m,n_p,n_m)

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
                conv_flux_xi ,P_p_xi, P_m_xi, M_p_xi, M_m_xi,n_xi_p,n_xi_m = flux_split_xi(self.grid.augmented_grid.cells[y][x])
                conv_flux_xi_plus_1 ,P_p_xi_plus_1, P_m_xi_plus_1, M_p_xi_plus_1, M_m_xi_plus_1,n_xi_plus_1_p,n_xi_plus_1_m = flux_split_xi(self.grid.augmented_grid.cells[y][x + 1])
                conv_flux_xi_minus_1 ,P_p_xi_minus_1, P_m_xi_minus_1, M_p_xi_minus_1, M_m_xi_minus_1,n_xi_minus_1_p,n_xi_minus_1_m = flux_split_xi(self.grid.augmented_grid.cells[y][x - 1])
                conv_flux_eta ,P_p_eta, P_m_eta, M_p_eta, M_m_eta,n_eta_p,n_eta_m = flux_split_eta(self.grid.augmented_grid.cells[y][x])
                conv_flux_eta_plus_1 ,P_p_eta_plus_1, P_m_eta_plus_1, M_p_eta_plus_1, M_m_eta_plus_1,n_eta_plus_1_p,n_eta_plus_1_m = flux_split_eta(self.grid.augmented_grid.cells[y + 1][x])
                conv_flux_eta_minus_1 ,P_p_eta_minus_1, P_m_eta_minus_1, M_p_eta_minus_1, M_m_eta_minus_1,n_eta_minus_1_p,n_eta_minus_1_m = flux_split_eta(self.grid.augmented_grid.cells[y - 1][x])

                # computing the step of the iteration #
                var_x = [0, 0, 0, 0]
                var_y = [0, 0, 0, 0]
                var_tot = [0, 0, 0, 0]
                var_x[0] = (max(0,M_p_xi+M_m_xi_plus_1) * conv_flux_xi[0] + min(0,M_p_xi+M_m_xi_plus_1) * conv_flux_xi_plus_1[0]) * self.grid.augmented_grid.cells[y][x].Sright.norm() - \
                           (max(0,M_p_xi_minus_1+M_m_xi) * conv_flux_xi_minus_1[0] + min(0,M_p_xi_minus_1+M_m_xi) * conv_flux_xi[0]) * self.grid.augmented_grid.cells[y][x].Sleft.norm()
                var_x[1] = (max(0,M_p_xi+M_m_xi_plus_1) * conv_flux_xi[1] + min(0,M_p_xi+M_m_xi_plus_1) * conv_flux_xi_plus_1[1] + n_xi_p.x * (P_p_xi + P_m_xi_plus_1)) * self.grid.augmented_grid.cells[y][x].Sright.norm() - \
                           (max(0,M_p_xi_minus_1+M_m_xi) * conv_flux_xi_minus_1[1] + min(0,M_p_xi_minus_1+M_m_xi) * conv_flux_xi[1] + n_xi_minus_1_p.x * (P_p_xi_minus_1 + P_m_xi)) * self.grid.augmented_grid.cells[y][x].Sleft.norm()
                var_x[2] = (max(0,M_p_xi+M_m_xi_plus_1) * conv_flux_xi[2] + min(0,M_p_xi+M_m_xi_plus_1) * conv_flux_xi_plus_1[2] + n_xi_p.y * (P_p_xi + P_m_xi_plus_1)) * self.grid.augmented_grid.cells[y][x].Sright.norm() - \
                           (max(0,M_p_xi_minus_1+M_m_xi) * conv_flux_xi_minus_1[2] + min(0,M_p_xi_minus_1+M_m_xi) * conv_flux_xi[2] + n_xi_minus_1_p.y * (P_p_xi_minus_1 + P_m_xi)) * self.grid.augmented_grid.cells[y][x].Sleft.norm()
                var_x[3] = (max(0,M_p_xi+M_m_xi_plus_1) * conv_flux_xi[3] + min(0,M_p_xi+M_m_xi_plus_1) * conv_flux_xi_plus_1[3]) * self.grid.augmented_grid.cells[y][x].Sright.norm() - \
                           (max(0,M_p_xi_minus_1+M_m_xi) * conv_flux_xi_minus_1[3] + min(0,M_p_xi_minus_1+M_m_xi) * conv_flux_xi[3]) * self.grid.augmented_grid.cells[y][x].Sleft.norm()
                var_y[0] = (max(0,M_p_eta+M_m_eta_plus_1) * conv_flux_eta[0] + min(0,M_p_eta+M_m_eta_plus_1) * conv_flux_eta_plus_1[0]) * self.grid.augmented_grid.cells[y][x].Stop.norm() - \
                           (max(0,M_p_eta_minus_1+M_m_eta) * conv_flux_eta_minus_1[0] + min(0,M_p_eta_minus_1+M_m_eta) * conv_flux_eta[0]) * self.grid.augmented_grid.cells[y][x].Sbottom.norm()
                var_y[1] = (max(0,M_p_eta+M_m_eta_plus_1) * conv_flux_eta[1] + min(0,M_p_eta+M_m_eta_plus_1) * conv_flux_eta_plus_1[1] + n_eta_p.x * (P_p_eta + P_m_eta_plus_1)) * self.grid.augmented_grid.cells[y][x].Stop.norm() - \
                           (max(0,M_p_eta_minus_1+M_m_eta) * conv_flux_eta_minus_1[1] + min(0,M_p_eta_minus_1+M_m_eta) * conv_flux_eta[1] + n_eta_minus_1_p.x * (P_p_eta_minus_1 + P_m_eta)) * self.grid.augmented_grid.cells[y][x].Sbottom.norm()
                var_y[2] = (max(0,M_p_eta+M_m_eta_plus_1) * conv_flux_eta[2] + min(0,M_p_eta+M_m_eta_plus_1) * conv_flux_eta_plus_1[2] + n_eta_p.y * (P_p_eta + P_m_eta_plus_1)) * self.grid.augmented_grid.cells[y][x].Stop.norm() - \
                           (max(0,M_p_eta_minus_1+M_m_eta) * conv_flux_eta_minus_1[2] + min(0,M_p_eta_minus_1+M_m_eta) * conv_flux_eta[2] + n_eta_minus_1_p.y * (P_p_eta_minus_1 + P_m_eta)) * self.grid.augmented_grid.cells[y][x].Sbottom.norm()
                var_y[3] = (max(0,M_p_eta+M_m_eta_plus_1) * conv_flux_eta[3] + min(0,M_p_eta+M_m_eta_plus_1) * conv_flux_eta_plus_1[3]) * self.grid.augmented_grid.cells[y][x].Stop.norm() - \
                           (max(0,M_p_eta_minus_1+M_m_eta) * conv_flux_eta_minus_1[3] + min(0,M_p_eta_minus_1+M_m_eta) * conv_flux_eta[3]) * self.grid.augmented_grid.cells[y][x].Sbottom.norm()
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