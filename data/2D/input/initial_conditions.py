from air import *
# initial conditions
Tini = 300.0
Pini = 101325.0
#Mini = 0.98 # representative of a chocked nozzle
Mini = 2.0 # representative of a diamond airfoil
#Mini = 5.0 # representative of a diamond airfoil in super fast flow
#Mini = 0.8 # representative of compressible flow
#Mini = 0.02 # representative of incompressible flow, so the scheme has some dificulties and is not physicaly representative
rhoini = Pini/r/Tini