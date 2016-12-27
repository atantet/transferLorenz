"""
Continuation of the Hopf fixed points and periodic orbit
"""
import numpy as np

def fieldPolar(x, p):
    """Vector field of the Hopf normal form in polar coordinates.
    Input arguments
    x --- The radius and the angle of the current state
    p --- A tuple or list with the parameters mu, omega and beta
    """
    
    fx = np.empty(state.shape, state.dtype)
    fx[0] = x[0] - p[0] * x[0]**3
    fx[1] = p[1] - p[2] * x[0]**2

    return fx

def JacobianFPPolar(x, p):
    """Jacobian of the Hopf normal form at the fixed point.
    Input arguments
    x --- The radius and the angle of the current state
    p --- A tuple or list with the parameters mu, omega and beta
    """

    J = np.empty(state.shape, state.dtype)
    fx[0] = x[0] - p[0] * x[0]**3
    fx[1] = p[1] - p[2] * x[0]**2

    return fx

