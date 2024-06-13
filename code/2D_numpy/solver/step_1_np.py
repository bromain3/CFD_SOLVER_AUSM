import sys
sys.path.append(r'..\code_readability_functions')
sys.path.append(r'..\grid_managment')
import grid_np
import numpy as np
from slices import *


def Mach_split(M):
    """
    Defines the logic for Mach splitting as it is defined in the AUSM scheme
    :param M: Mach number of each cell (computed with contravariant velocities) (2D-array)
    :return M_plus, M_minus: Split Mach number (2D-array, 2D-array)
    """
    condition_1 = (M <= -1)
    condition_2 = (M >= 1)
    condition_3 = (M > -1) * (M < 1)
    M_plus = condition_2 * 0.5 * (M + np.abs(M)) + condition_3 * (0.25 * (M + 1) ** 2) # maybe that perf could be improved if M_plusand M_minus were created ony once at the begining of the simulation and thene only modified
    M_minus = condition_1 * 0.5 * (M - np.abs(M)) - condition_3 * (0.25 * (M - 1) ** 2)
    return (M_plus, M_minus)


def Pressure_split(P, M):
    """
    Defines the logic for Pressure splitting as it is defined in the AUSM scheme
    :param M: Mach number of each cell (computed with contravariant velocities) (2D-array)
    :param P: Pressure of each cell (2D-array)
    :return P_plus, P_minus: Split Pressure (2D-array, 2D-array)
    """
    condition_1 = (M <= -1) # performance could be improved by not recomputing those conditions, but by reusing those in Mach_split
    condition_2 = (M >= 1)
    condition_3 = (M > -1) * (M < 1)
    P_plus = condition_2 * ((P/2) * (1 + np.sign(M))) + condition_3 * ((P/4) * (M+1)**2 * (2-M))  # maybe that perf could be improved if M_plusand M_minus were created ony once at the begining of the simulation and thene only modified
    P_minus = condition_1 * ((P/2) * (1 - np.sign(M))) + condition_3 * ((P/4) * (M-1)**2 * (2+M))
    return (P_plus, P_minus)


def convective_flux(Q):
    """
    Returns the convective flux divided by Mach number
    :param Q: Array of state vectors (3D-array)
    :return E: Array of convective fluxes (3D-array)
    """
    E = np.zeros(np.shape(Q))
    E[rho] = Q[rho] * grid_np.c(Q)
    E[rho_u] = Q[rho_u] * grid_np.c(Q)
    E[rho_v] = Q[rho_v] * grid_np.c(Q)
    E[rho_et] = Q[rho] * grid_np.ht(Q) * grid_np.c(Q)
    return (E)

def flux_split_xi(Q,n_xi):
    """
    Split the flux for each cell and returns the interface flux for left and right interface
    :param Q: Array of state vectors (3D-array)
    :param n_xi: Array of unit vectors normal to vertical surfaces (3D-array)
    :return flux_p_xi, flux_m_xi: Array of left and right split fluxes (3D-array)
    """
    # first try with contravariant velocity extrapolated at cell center
    n_p = n_xi[xi_plus]
    n_m = n_xi[xi_minus]
    n_mean = ((n_p + n_m)/2)[length]
    u_contra = n_mean[x] * grid_np.u(Q) + n_mean[y] * grid_np.v(Q)
    #compute Mach
    M = u_contra/grid_np.c(Q)
    #compute pressure
    P = grid_np.P(Q)
    # compute mache and pressure split and convective flux
    M_p, M_m = Mach_split(M)
    P_p, P_m = Pressure_split(P, M)
    conv_flux = convective_flux(Q)
    # split the flux
    flux_p_xi = M_p[:,:,np.newaxis] * conv_flux
    flux_p_xi[rho_u] += n_mean[x] * P_p
    flux_p_xi[rho_v] += n_mean[y] * P_p
    flux_m_xi = M_m[:,:,np.newaxis] * conv_flux
    flux_m_xi[rho_u] += n_mean[x] * P_m
    flux_m_xi[rho_v] += n_mean[y] * P_m
    return(flux_p_xi,flux_m_xi)

def flux_split_eta(Q,n_eta):
    """
    Split the flux for each cell and returns the interface flux for top and bottom interface
    :param Q: Array of state vectors (3D-array)
    :param n_eta: Array of unit vectors normal to horizontal surfaces (3D-array)
    :return flux_p_eta, flux_m_eta: Array of top and bottom split fluxes (3D-array)
    """
    # first try with contravariant velocity extrapolated at cell center
    n_p = n_eta[eta_plus]
    n_m = n_eta[eta_minus]
    n_mean = ((n_p + n_m)/2)[length]
    v_contra = n_mean[x] * grid_np.u(Q) + n_mean[y] * grid_np.v(Q)
    #compute Mach
    M = v_contra/grid_np.c(Q)
    #compute pressure
    P = grid_np.P(Q)
    # compute mache and pressure split and convective flux
    M_p, M_m = Mach_split(M)
    P_p, P_m = Pressure_split(P, M)
    conv_flux = convective_flux(Q)
    # split the flux
    flux_p_eta = M_p[:,:,np.newaxis] * conv_flux
    flux_p_eta[rho_u] += n_mean[x] * P_p
    flux_p_eta[rho_v] += n_mean[y] * P_p
    flux_m_eta = M_m[:,:,np.newaxis] * conv_flux
    flux_m_eta[rho_u] += n_mean[x] * P_m
    flux_m_eta[rho_v] += n_mean[y] * P_m
    return(flux_p_eta,flux_m_eta)



class solver_np(object):
    def __init__(self,grid):
        """
        Initialisation function of the solver_np class
        :param grid: grid to be used (grid_np.computational_grid)
        """
        self.grid = grid
        self.grid.initialize_cells()
        self.grid.set_boundary_conditions()

    def one_step_AUSM(self,dt):
        """
        Function to compute one time step using AUSM flux splitting methode
        :param dt: time step used for integration (float)
        :return: None
        """
        tot_f_xi_p, tot_f_xi_m = flux_split_xi(self.grid.augmented_grid.Q,self.grid.augmented_grid.n_xi)
        flux_p_xi, flux_m_xi = tot_f_xi_p[interior], tot_f_xi_m[interior]
        flux_p_xi_plus_1, flux_m_xi_plus_1 = tot_f_xi_p[xi_plus_aug][1:-1,:], tot_f_xi_m[xi_plus_aug][1:-1,:]
        flux_p_xi_minus_1, flux_m_xi_minus_1 = tot_f_xi_p[xi_minus_aug][1:-1,:], tot_f_xi_m[xi_minus_aug][1:-1,:]

        tot_f_eta_p, tot_f_eta_m = flux_split_eta(self.grid.augmented_grid.Q, self.grid.augmented_grid.n_eta)
        flux_p_eta, flux_m_eta = tot_f_eta_p[interior], tot_f_eta_m[interior]
        flux_p_eta_plus_1, flux_m_eta_plus_1 = tot_f_eta_p[eta_plus_aug][:, 1:-1], tot_f_eta_m[eta_plus_aug][:, 1:-1]
        flux_p_eta_minus_1, flux_m_eta_minus_1 = tot_f_eta_p[eta_minus_aug][:, 1:-1], tot_f_eta_m[eta_minus_aug][:, 1:-1]

        var_x = (flux_p_xi + flux_m_xi_plus_1) * np.linalg.norm(self.grid.S_xi[length],axis=2)[xi_plus][:,:,np.newaxis] - (flux_m_xi + flux_p_xi_minus_1) * np.linalg.norm(self.grid.S_xi[length],axis=2)[xi_minus][:,:,np.newaxis]
        var_y = (flux_p_eta + flux_m_eta_plus_1) * np.linalg.norm(self.grid.S_eta[length],axis=2)[eta_plus][:,:,np.newaxis] - (flux_m_eta + flux_p_eta_minus_1) * np.linalg.norm(self.grid.S_eta[length],axis=2)[eta_minus][:,:,np.newaxis]
        step = var_x + var_y
        self.grid.Q -= dt * step / self.grid.V[:,:,np.newaxis]

        # reset the boundary conditions #
        self.grid.set_boundary_conditions()


if __name__ == '__main__':
    path = r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\grids\2D'
    grid_name = 'g33x50u_pipe.dat'
    dt = 0.00001
    colorScheme = 'jet'
    g = grid_np.computational_grid(path, grid_name)
    calcul = solver_np(g)
    print('commencé')
    for i in range(10000):
        calcul.one_step_AUSM(dt)
    print('fini')