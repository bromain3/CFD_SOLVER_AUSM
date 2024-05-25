import grid as gm
import matplotlib.pyplot as plt
import numpy as np

file = 'g65x49u.dat'
colorScheme = 'jet'

# augmenting the grid with hallo cells #
g = gm.grid_by_cells(r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\grids', file)
aug_g = g.augment_grid()
# overlaying the plots of augmented and unaugmented grids to see the difference #
aug_g.plot_grid(color='blue')
g.plot_grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Base grid overlayed on augmented grid')
plt.show()

# show unitary vectors of cell faces #
g.plot_direction_vectors()
g.plot_grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Base grid with direction vectors')
plt.show()

# plotting cells properties #
# Cells volume
V,x_centroid,y_centroid = g.get_Volumes()
plt.contourf(x_centroid,y_centroid,V,levels=20,cmap=colorScheme)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cells volume')
plt.show()
bottom_Surfaces,x_Sb,y_Sb = g.get_surfaces('Sbottom')
left_Surfaces,x_Sl,y_Sl = g.get_surfaces('Sleft')
# Cells surfaces 'parallel' to x projected on x
plt.contourf(x_Sb,y_Sb,np.absolute(bottom_Surfaces[:,:,0]),levels=20,cmap=colorScheme)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('projection on x of the surface vectors of cells parallel to \u03BE')
plt.show()
# Cells surfaces 'parallel' to x projected on y
plt.contourf(x_Sb,y_Sb,np.absolute(bottom_Surfaces[:,:,1]),levels=20,cmap=colorScheme)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('projection on y of the surface vectors of cells parallel to \u03BE')
plt.show()
# Cells surfaces 'parallel' to y projected on x
plt.contourf(x_Sl,y_Sl,np.absolute(left_Surfaces[:,:,0]),levels=20,cmap=colorScheme)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('projection on x of the surface vectors of cells parallel to $\eta$')
plt.show()
# Cells surfaces 'parallel' to y projected on y
plt.contourf(x_Sl,y_Sl,np.absolute(left_Surfaces[:,:,1]),levels=20,cmap=colorScheme)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('projection on y of the surface vectors of cells parallel to $\eta$')
plt.show()