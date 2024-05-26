import sys
sys.path.append(r'..\code\2D_numpy\grid_managment')
sys.path.append(r'..\code\2D_numpy\code_readability_functions')
sys.path.append(r'..\code\2D\grid_managment')
import numpy as np
from slices import *
import grid_np
import grid

precision = 0.99
def find_equality(X,Y):
    eps = 1000
    while np.allclose(X,Y,atol=eps,rtol=0):
        eps *= precision
        if eps < 1e-50:
            return(eps*1/precision)
    return(eps*1/precision)




if __name__ == '__main__':
    # grid used for comparison
    path = r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\grids\2D'
    name = 'g33x25u.dat'

    g = grid.grid_by_cells(path,name) # grid using pur python synthax
    g_np = grid_np.grid_by_cells(path,name) # grid using numpy

    ########################
    ## Volumes comparison ##
    ########################
    V = g.get_Volumes()[0]
    V_np = g_np.V
    print('The maximum difference between cells volume is:',find_equality(V,V_np))

    ##########################
    ## centroids comparison ##
    ##########################
    _,x_c,y_c = g.get_Volumes()
    c = g_np.centroids
    print('The maximum difference between x componnent of cells centroids is:', find_equality(x_c, c[x]))
    print('The maximum difference between y componnent of cells centroids is:', find_equality(y_c, c[y]))

    #####################
    ## S_xi comparison ##
    #####################
    S_xi, x_S_xi, y_S_xi = g.get_surfaces('Sright')
    S_xi_np = g_np.S_xi
    print('The maximum difference between x componnent of cells S_xi_right is:', find_equality(S_xi[x], S_xi_np[length][x]))
    print('The maximum difference between y componnent of cells S_xi_right is:', find_equality(S_xi[y], S_xi_np[length][y]))
    print('The maximum difference between x componnent of cells S_xi_right origin is:',find_equality(x_S_xi, S_xi_np[origin][x]))
    print('The maximum difference between y componnent of cells S_xi_right origin is:',find_equality(y_S_xi, S_xi_np[origin][y]))

    ######################
    ## S_eta comparison ##
    ######################
    S_eta, x_S_eta, y_S_eta = g.get_surfaces('Stop')
    S_eta_np = g_np.S_eta
    print('The maximum difference between x componnent of cells S_eta_right is:',find_equality(S_eta[x], S_eta_np[length][x]))
    print('The maximum difference between y componnent of cells S_eta_right is:',find_equality(S_eta[y], S_eta_np[length][y]))
    print('The maximum difference between x componnent of cells S_eta_right origin is:',find_equality(x_S_eta, S_eta_np[origin][x]))
    print('The maximum difference between y componnent of cells S_eta_right origin is:',find_equality(y_S_eta, S_eta_np[origin][y]))

    ###########################
    ## n_xi/n_eta comparison ##
    ###########################
    n_xi, x_n_xi, y_n_xi = g.get_vect('n_xi_plus_half')
    n_xi_np = g_np.n_xi
    print('The maximum difference between x componnent of cells n_xi_right is:',find_equality(n_xi[x], n_xi_np[length][x]))
    print('The maximum difference between y componnent of cells n_xi_right is:',find_equality(n_xi[y], n_xi_np[length][y]))
    print('The maximum difference between x componnent of cells n_xi_right origin is:',find_equality(x_n_xi, n_xi_np[origin][x]))
    print('The maximum difference between y componnent of cells n_xi_right origin is:',find_equality(y_n_xi, n_xi_np[origin][y]))
    n_eta, x_n_eta, y_n_eta = g.get_vect('n_eta_plus_half')
    n_eta_np = g_np.n_eta
    print('The maximum difference between x componnent of cells n_eta_top is:',find_equality(n_eta[x], n_eta_np[length][x]))
    print('The maximum difference between y componnent of cells n_eta_top is:',find_equality(n_eta[y], n_eta_np[length][y]))
    print('The maximum difference between x componnent of cells n_eta_top origin is:',find_equality(x_n_eta, n_eta_np[origin][x]))
    print('The maximum difference between y componnent of cells n_eta_top origin is:',find_equality(y_n_eta, n_eta_np[origin][y]))

    #############################
    ## check great intrication ##
    #############################
    g = grid.computational_grid(path,name)
    g_np = grid_np.computational_grid(path,name)

    aug_g_np = g_np.augmented_grid
    print('The maximum difference between augmented and base grid is:',find_equality(aug_g_np.Q[interior], g_np.Q))
    g_np.Q[2,3] = 100000
    print('If we change a parameter of the base grid, the difference is:',find_equality(aug_g_np.Q[interior], g_np.Q))
    aug_g_np.Q[interior][18, 6] = 100000
    print('If we change a parameter of the augmented grid, the difference is:', find_equality(aug_g_np.Q[interior], g_np.Q))


    ################################
    ## check cells initialization ##
    ################################
    g = grid.computational_grid(path, name)
    g.initialize_cells()
    Q = np.zeros((g.augmented_grid.ny_cells, g.augmented_grid.nx_cells, 4))
    for y in range(g.augmented_grid.ny_cells):
        for x in range(g.augmented_grid.nx_cells):
            Q[y, x] = g.augmented_grid.cells[y][x].Q
    g_np = grid_np.computational_grid(path, name)
    g_np.initialize_cells()
    print('The maximum difference between Q vectors after initializing the grids is:',find_equality(Q, g_np.augmented_grid.Q))

    ###############################
    ## check boundary conditions ##
    ###############################
    g.set_boundary_conditions()
    Q = np.zeros((g.augmented_grid.ny_cells,g.augmented_grid.nx_cells,4))
    for y in range(g.augmented_grid.ny_cells):
        for x in range(g.augmented_grid.nx_cells):
            Q[y,x] = g.augmented_grid.cells[y][x].Q
    g_np.set_boundary_conditions()
    print('The maximum difference between rho vectors after setting boundary conditions is:', find_equality(Q[rho], g_np.augmented_grid.Q[rho]))
    print('The maximum difference between rho_u vectors after setting boundary conditions is:',find_equality(Q[rho_u], g_np.augmented_grid.Q[rho_u]))
    print('The maximum difference between rho_v vectors after setting boundary conditions is:',find_equality(Q[rho_v], g_np.augmented_grid.Q[rho_v]))
    print('The maximum difference between rho_et vectors after setting boundary conditions is:',find_equality(Q[rho_et], g_np.augmented_grid.Q[rho_et]))
