import sys
sys.path.append(r'..\code\2D_numpy\solver')
sys.path.append(r'..\code\2D_numpy\code_readability_functions')
sys.path.append(r'..\code\2D\solver')
import numpy as np
from slices import *
import grid
import grid_np
import step_1_np
import step_1_AUSM_POO

precision = 0.99
def find_equality(X,Y):
    eps = 1000
    while np.allclose(X,Y,atol=eps,rtol=0):
        eps *= precision
        if eps < 1e-50:
            return(eps*1/precision)
    return(eps*1/precision)


if __name__ == '__main__':
    path = r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\grids\2D'
    name = 'g33x25u.dat'
    g = grid.computational_grid(path, name)  # grid using pur python synthax
    g_np = grid_np.computational_grid(path, name)  # grid using numpy
    solver = step_1_AUSM_POO.solver(g)
    solver_np = step_1_np.solver_np(g_np)

    ###############################
    ## Mach splitting comparison ##
    ###############################
    M_p_np, M_m_np = step_1_np.Mach_split(grid_np.M(solver_np.grid.Q))
    M_p = np.zeros(np.shape(M_p_np))
    M_m = np.zeros(np.shape(M_m_np))
    for y in range(g.ny_cells):
        for x in range(g.nx_cells):
            M_p[y, x],M_m[y, x] = step_1_AUSM_POO.Mach_split(g.cells[y][x].M())

    print('The maximum difference between Mach plus is:', find_equality(M_p, M_p_np))
    print('The maximum difference between Mach minus is:', find_equality(M_m, M_m_np))

    ###################################
    ## Pressure splitting comparison ##
    ###################################
    P_p_np, P_m_np = step_1_np.Pressure_split(grid_np.P(solver_np.grid.Q),grid_np.M(solver_np.grid.Q))
    P_p = np.zeros(np.shape(P_p_np))
    P_m = np.zeros(np.shape(P_m_np))
    for y in range(g.ny_cells):
        for x in range(g.nx_cells):
            P_p[y, x],P_m[y, x] = step_1_AUSM_POO.Pressure_split(g.cells[y][x].P(),g.cells[y][x].M())

    print('The maximum difference between P plus is:', find_equality(P_p, P_p_np))
    print('The maximum difference between P minus is:', find_equality(P_m, P_m_np))

    ##################################
    ## convective fluxes comparison ##
    ##################################
    E_np = step_1_np.convective_flux(solver_np.grid.Q)
    E = np.zeros(np.shape(E_np))
    for y in range(g.ny_cells):
        for x in range(g.nx_cells):
            E[y, x] = step_1_AUSM_POO.convective_flux(g.cells[y][x])

    print('The maximum difference between convective fluxes is:', find_equality(E, E_np))

    ###################################
    ## xi fluxes spliting comparison ##
    ###################################
    E_p_xi_np,E_m_xi_np = step_1_np.flux_split_xi(solver_np.grid.augmented_grid.Q,solver_np.grid.augmented_grid.n_xi)
    E_p_xi = np.zeros(np.shape(E_p_xi_np))
    E_m_xi = np.zeros(np.shape(E_m_xi_np))
    for y in range(g.augmented_grid.ny_cells):
        for x in range(g.augmented_grid.nx_cells):
            E_p_xi[y, x],E_m_xi[y, x] = step_1_AUSM_POO.flux_split_xi(g.augmented_grid.cells[y][x])

    print('The maximum difference between xi plus splited fluxes is:', find_equality(E_p_xi, E_p_xi_np))
    print('The maximum difference between xi minus splited fluxes is:', find_equality(E_m_xi, E_m_xi_np))

    ####################################
    ## eta fluxes spliting comparison ##
    ####################################
    E_p_eta_np,E_m_eta_np = step_1_np.flux_split_eta(solver_np.grid.augmented_grid.Q,solver_np.grid.augmented_grid.n_eta)
    E_p_eta = np.zeros(np.shape(E_p_eta_np))
    E_m_eta = np.zeros(np.shape(E_m_eta_np))
    for y in range(g.augmented_grid.ny_cells):
        for x in range(g.augmented_grid.nx_cells):
            E_p_eta[y, x],E_m_eta[y, x] = step_1_AUSM_POO.flux_split_eta(g.augmented_grid.cells[y][x])

    print('The maximum difference between eta plus splited fluxes is:', find_equality(E_p_eta, E_p_eta_np))
    print('The maximum difference between eta minus splited fluxes is:', find_equality(E_m_eta, E_m_eta_np))

    ##############################################
    ## one step comparison (the most important) ##
    ##############################################
    dt = 0.00001
    solver.one_step_AUSM(dt)
    solver_np.one_step_AUSM(dt)
    Q = np.zeros((g.augmented_grid.ny_cells, g.augmented_grid.nx_cells, 4))
    for y in range(g.augmented_grid.ny_cells):
        for x in range(g.augmented_grid.nx_cells):
            Q[y, x] = g.augmented_grid.cells[y][x].Q
    print('The maximum difference between a single step with both scheme is:', find_equality(g_np.augmented_grid.Q, Q))

