import sympy as sp

# Assign symbols
x, y, z, rho, sigma, beta = sp.symbols('x y z rho sigma beta')

sigma0 = 10.
beta0 = 8. / 3.
rho0 = 24.06

# Define vector fields
F = sp.Matrix([sigma * (y - x), x * (rho - z) - y, x*y - beta * z])

# Get Jacobian
J = F.jacobian((x, y, z))

# Get roots
roots = sp.solve(F, (x, y, z))
C0, C1, C2 = roots

# Evaluate Jacobian at fixed point at origin
J0 = J.subs([(x, C0[0]), (y, C0[1]), (z, C0[2])])

# Get spectrum of J0
spec0, spec1, spec2 = J0.eigenvects()
eigVal = (spec0[0], spec1[0], spec2[0])
eigVal = sp.simplify(eigVal)
eigVec = (spec0[2][0], spec1[2][0], spec2[2][0])
eigVec = sp.simplify(eigVec)

subs0 = [(rho, rho0), (sigma, sigma0), (beta, beta0)]
w = eigVal.subs(subs0)
v = eigVec.subs(subs0)
