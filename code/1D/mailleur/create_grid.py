import os
import numpy as np

def linear_spaced_grid(path,grid_name,nodes,sections,n_nodes):
    if len(nodes) != len(sections):
        raise ValueError('The length of nodes and sections lists are not the same')
    else:
        with open(os.path.join(path,grid_name),'w') as grid:
            grid.write('nodes_x\tsurface\n')
            for i in range(len(nodes)-1):
                node_list = np.linspace(nodes[i], nodes[i+1], n_nodes[i])
                S_list = np.linspace(sections[i], sections[i+1], n_nodes[i])
                for j in range(len(node_list)):
                    if i == 0:
                        grid.write(str(node_list[j])+'\t'+str(S_list[j])+'\n')
                    elif j != 0:
                        grid.write(str(node_list[j]) + '\t' + str(S_list[j]) + '\n')

if __name__ == '__main__':
    path = r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\grids\1D'
    grid_name = 'diamond_airfoil_ultra_detailed.dat'
    nodes = [-0.5,0,0.5,1,1.5]
    sections = [1.5,1.5,1.413175911,1.5,1.5]
    n_nodes = [100,100,100,100]
    """
    grid_name = 'Rieman.dat'
    nodes = [0,1]
    sections = [1,1]
    n_nodes = [20]
    """
    linear_spaced_grid(path,grid_name,nodes,sections,n_nodes)

