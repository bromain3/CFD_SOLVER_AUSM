import grid1D as gm
import matplotlib.pyplot as plt

path = r'C:\Users\bapti\OneDrive\Documents\python\reprise_code_cfd\grids\1D'
grid_name = 'diamond_airfoil_coarse.dat'
g = gm.grid_by_cell(path, grid_name)
g.augment_grid()

centroids = g.get_centroids()
volumes = g.get_volumes()
fig, ax = plt.subplots(2)

g.plot_grid(ax[0],True)
ax[0].scatter(centroids,[0 for i in centroids],marker='x',color=ax[0].lines[0].get_color())
ax[0].set_title('Graphical representation')

ax[1].plot(centroids,volumes)
ax[1].set_title('Volumes')
ax[1].set_xlim(ax[0].get_xlim())

plt.tight_layout()
plt.show()