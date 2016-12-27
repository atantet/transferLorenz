"""
Continuation of the Hopf fixed points and periodic orbit
"""
import numpy as np

def fieldPolar(x, p):
    """Vector field of the Hopf normal form in polar coordinates.
    Input arguments
    x --- The radius and the angle of the current state
    p --- A tuple or list with the parameters mu, gamma and beta
    """
    
    fx = np.empty(x.shape, x.dtype)
    fx[0] = x[0] - p[0] * x[0]**3
    fx[1] = p[1] - p[2] * x[0]**2

    return fx

def JacobianPolar(x, p):
    """Jacobian of the Hopf normal form at the fixed point.
    Input arguments
    x --- The radius and the angle of the current state
    p --- A tuple or list with the parameters mu, gamma and beta
    """

    J = np.zeros(x.shape, x.dtype)
    J[0, 0] = x[0] - p[0] * x[0]**3
    J[1, 0] = p[1] - p[2] * x[0]**2

    return J

def fieldCart(x, p):
    """Vector field of the Hopf normal form in Cartesian coordinates.
    Input arguments
    x --- (x, y) coordinates of the current state
    p --- A tuple or list with the parameters mu, gamma and beta
    """

    fx = np.empty(x.shape, x.dtype)
    r2 = x[0]**2 + x[1]**2
    fx[0] = (p[0] - r2) * x[0] - (p[1] - p[2] * r2) * x[1]
    fx[1] = (p[1] - p[2] * r2) * x[0] + (p[0] - r2) * x[1]

    return fx

def JacobianCart(x, p):
    """Jacobian of the Hopf normal form at the fixed point.
    Input arguments
    x --- (x, y) coordinates of the current state
    p --- A tuple or list with the parameters mu, gamma and beta
    """

    r2 = x[0]**2 + x[1]**2
    J = np.array([[(p[0] - r2) - 2*x[0]**2 + 2*p[2]*x[0]*x[1],
                   -(p[1] - p[2]*r2) + 2*p[2]*x[1]**2 - 2*x[0]*x[1]],
                  [(p[1] - p[2]*r2) - 2*p[2]*x[0]**2 - 2*x[0]*x[1],
                   (p[0] - r2) - 2*x[1]**2 - 2*p[2]*x[0]*x[1]]])

    return J

def PoincareXLine(traj, idx):
    ''' Poincare section on the x axis.'''
    if (traj[idx][1] >= 0) \
       & (traj[idx-1][1] < 0):
        return True
    else:
        return False


# Phase space dimension
dim = 2
# Parameters (mu, gamma, beta)
p = np.array([1., 100., 0.1])
omega = p[1] - p[2] * p[0]

# Plot
fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot(111)
#ax.plot(X[:, 0], X[:, 1])
t = np.linspace(0., 2*np.pi, 1000)
Sx = np.sqrt(p[0]) * np.cos(t)
Sy = np.sqrt(p[0]) * np.sin(t)
plt.plot(Sx, Sy, '--k')
xlim = 2*np.array([-np.sqrt(p[0]), np.sqrt(p[0])])
ylim = xlim.copy()
ax.set_xlim(xlim)
ax.set_ylim(ylim)


# Define Poincare section
Poincare = PoincareXLine
plt.plot(np.linspace(xlim[0], xlim[1], 1000), np.zeros((1000,)), '--k')

# Time step
dt = 1.e-6
# Distance before to stop and maximum number of iterations
eps = np.sqrt(p[0]) * 0.001
maxIter = 10
# Initial state
x0 = np.array([10., 10.])
msize = 60
plt.scatter(x0[0], x0[1], s=msize, c='b', marker='+')

# niter = 20
print 'x0 = ', x0
xn = x0
dist = 1.e27
#for k in np.arange(niter):
nIter = 0
while (dist > eps) and (nIter < maxIter):
    # Integrate solution Euler forward
    x0 = xn
    print '---Iteration ', nIter, '---'
    print 'x0 = ', x0
    plt.scatter(x0[0], x0[1], s=msize, c='k', marker='+')
    X = np.empty((2, dim))
    X[0] = x0.copy()
    X[1] = X[0].copy()
    nt = len(X)
    M = [np.empty((dim, dim)), np.eye(dim)]
    # Distance has started to decrease again
    dec = False
    T = 0.
    while (not dec) or (np.sum((X[1] - x0)**2) < np.sum((X[0] - x0)**2)):
        if np.sum((X[1] - x0)**2) < np.sum((X[0] - x0)**2):
            dec = True
        # Step solution forward
        X[0] = X[1]
        X[1] = X[0] + dt * fieldCart(X[0], p)
        # Step fundamental forward
        M[0] = M[1]
        M[1] = M[0] + dt * np.dot(JacobianCart(X[0], p), M[0])
        T += dt
    xT = X[-1]
    MT = M[-1]
    dist = np.sqrt(np.sum((xT - x0)**2))
    print 'xT = ', xT
    print 'd(xT, x0) = ', dist
    print 'T = ', T
    
    plt.scatter(xT[0], xT[1], s=msize, c='r', marker='x')

    # Newton step
    xn = np.dot(np.linalg.inv(np.eye(dim) - MT), (xT - np.dot(MT, x0)))
    
    nIter += 1

# Get Floquet multipliers
(multipliers, eigVec) = np.linalg.eig(MT)
exponents = np.log(multipliers) / T
print '---Floquet---'
print 'Floquet multipliers = ', multipliers
print 'Floquet exponents = ', exponents

print '---Convergence---'
if nIter == maxIter:
    print 'Did not converge!'
else:
    print 'Converged.'

    
