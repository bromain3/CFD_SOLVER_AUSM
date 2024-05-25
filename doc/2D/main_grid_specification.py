import plotgrid
import gridaugmentation
import cellproperties
import matplotlib.pyplot as plt

grid_directory = 'grids'
grid_name = 'g33x25u.dat'
colorScheme = 'jet'

# augmenting the grid with hallo cells #
gridaugmentation.gridAugmentation(grid_directory,grid_name)
augmented_name = grid_name.split('.')[0]+'_augmented.'+grid_name.split('.')[1]
# overlaying the plots of augmented and unaugmented grids to see the difference #
plotgrid.gridePlotting(grid_directory,augmented_name, 'blue')
plotgrid.gridePlotting(grid_directory,grid_name)
plt.show()
# plotting cells properties #
# Cells volume
Volumes, x_V, y_V, bottom_Surfaces, x_Sb, y_Sb, left_Surfaces, x_Sl, y_Sl = cellproperties.cellsCharacteristics(grid_directory,augmented_name)
plt.contourf(x_V,y_V,Volumes,levels=20,cmap=colorScheme)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cells volume')
plt.show()
# Cells surfaces 'parallel' to x projected on x
plt.contourf(x_Sb,y_Sb,bottom_Surfaces[:,:,0],levels=20,cmap=colorScheme)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('projection on x of the surface vectors of cells parallel to \u03BE')
plt.show()
# Cells surfaces 'parallel' to x projected on y
plt.contourf(x_Sb,y_Sb,bottom_Surfaces[:,:,1],levels=20,cmap=colorScheme)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('projection on y of the surface vectors of cells parallel to \u03BE')
plt.show()
# Cells surfaces 'parallel' to y projected on x
plt.contourf(x_Sl,y_Sl,left_Surfaces[:,:,0],levels=20,cmap=colorScheme)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('projection on x of the surface vectors of cells parallel to $\eta$')
plt.show()
# Cells surfaces 'parallel' to y projected on y
plt.contourf(x_Sl,y_Sl,left_Surfaces[:,:,1],levels=20,cmap=colorScheme)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('projection on y of the surface vectors of cells parallel to $\eta$')
plt.show()