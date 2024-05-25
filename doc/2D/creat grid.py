import numpy as np

X = np.linspace(-0.5,1.5,33)
Y = np.linspace(0,1.5,25)
nx = len(X)
ny = len(Y)
X,Y = np.meshgrid(X,Y)
X = X.flatten()
Y = Y.flatten()
print(nx,ny)
with open("../../grids/2D/g33x25u_free.dat", "w") as fout:
    fout.write("%d, %d\n" % (nx, ny,))
    for (xx, yy) in zip(X, Y):
        fout.write("%s, %s\n" % (xx, yy,))