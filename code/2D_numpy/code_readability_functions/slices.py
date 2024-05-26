# Nodes positions in a 2D array
upper_right = (slice(1,None),slice(1,None))
lower_left = (slice(None,-1),slice(None,-1))
upper_left = (slice(1,None),slice(None,-1))
lower_right = (slice(None,-1),slice(1,None))

# x and y coordinates of a 3D array
x = (slice(None),slice(None),0)
y = (slice(None),slice(None),1)

# distinction between origine coordinates and length values for a 3D array containing vectors
origin = (slice(None),slice(None),slice(None,2))
length = (slice(None),slice(None),slice(2,None))

# interior cells of an augmented grid
interior = (slice(1,-1),slice(1,-1))

# pysical values of Q
rho = (slice(None),slice(None),0)
rho_u = (slice(None),slice(None),1)
rho_v = (slice(None),slice(None),2)
rho_et = (slice(None),slice(None),3)

# boundaries for Q
inlet = (slice(None),0)
just_after_inlet = (slice(None),1)
outlet = (slice(None),-1)
just_before_outlet = (slice(None),-2)
top_wall = (-1,slice(None))
just_before_top_wall = (-2,slice(None))
bottom_wall = (0,slice(None))
just_after_bottom_wall = (1,slice(None))

# plus and minus states in the solver
xi_plus = (slice(None),slice(1,None))
xi_minus = (slice(None),slice(None,-1))
eta_plus = (slice(1,None),slice(None))
eta_minus = (slice(None,-1),slice(None))

# plus and minus states in the solver (used on augmented_grid when added to the interior grid)
xi_plus_aug = (slice(None),slice(2,None))
xi_minus_aug = (slice(None),slice(None,-2))
eta_plus_aug = (slice(2,None),slice(None))
eta_minus_aug = (slice(None,-2),slice(None))