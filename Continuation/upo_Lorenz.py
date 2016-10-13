"""
Continuation of the Lorenz fixed points and periodic orbit
"""
import numpy as np
from scipy import optimize, linealg
import pylibconfig2
import sympy as sp

configFile = '../cfg/Lorenz63.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)
readCount = int(np.round(cfg.model.dim * cfg.simulation.LCut \
                         / cfg.simulation.printStep / 4))

delayName = ""
if hasattr(cfg.model, 'delaysDays'):
    for d in np.arange(len(cfg.model.delaysDays)):
        delayName = "%s_d%d" % (delayName, cfg.model.delaysDays[d])

L = cfg.simulation.LCut + cfg.simulation.spinup
printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt + 0.1)
caseName = cfg.model.caseName
if (hasattr(cfg.model, 'rho') & hasattr(cfg.model, 'sigma') \
    & hasattr(cfg.model, 'beta')):
    caseName = "%s_rho%d_sigma%d_beta%d" \
               % (caseName, (int) (cfg.model.rho * 1000),
                  (int) (cfg.model.sigma * 1000),
                  (int) (cfg.model.beta * 1000))
srcPostfix = "_%s%s_L%d_spinup%d_dt%d_samp%d" \
             % (caseName, delayName, L, cfg.simulation.spinup,
                -np.round(np.log10(cfg.simulation.dt)), printStepNum)

def field(x, par):
    """Vector field of the Lorenz flow.
    Input arguments
    x --- The state
    par --- The parameters rho, sigma and beta
    """
    
    fx = np.array([par[1] * (x[1] - x[0]),
                   x[0] * (par[0] - x[2]) - x[1],
                   x[0] * x[1] - par[2] * x[2]])

    return fx

def Jacobian(x, par):
    """Jacobian of the Lorenz flow.
    Input arguments
    x --- The state
    par --- The parameters rho, sigma and beta
    """

    J = np.array([[-par[1], par[1], 0],
                  [par[0] - x[2], -1., -x[0]],
                  [x[1], x[0], -par[2]]])

    return J

def getFP(par):
    x0 = np.array([0., 0., 0.])
    xp = np.array([np.sqrt(par[2] * (par[0] - 1.)),
                   np.sqrt(par[2] * (par[0] - 1.)),
                   par[0] - 1.])
    xm = np.array([-xp[0], -xp[1], xp[2]])
    
    return (x0, xp, xm)

# Get Monodromy matrix symbolicly
x, y, z = sp.symbols('x, y, z')
rho, sigma, beta = sp.symbols('rho, sigma, beta', positive=True)
Fx = sp.Matrix([[sigma * (y - x)],
                [x * (rho - z) - y],
                [x * y - beta * z]])
J = Fx.jacobian([x, y, z])


# #p0 = np.array([0.5, 10., 8. / 3])
# #p0 = np.array([1.5, 10., 8. / 3])
# p0 = np.array([28, 10., 8. / 3])
# x0 = np.array([1., 1., 1.])
# #x0 = np.array([0.1, 0.1, 0.1])

# # method = 'hybr'
# # tol = 1.e-8
# # # maxiter = 50

# # xf = optimize.root(field, x0, jac=Jacobian, args=(p0,), method=method,
# #                    tol=tol)

# p = np.array([24.5, 10., 8. / 3])
# x0, xp, xm =  getFP(p)
# eigVal, eigVec = np.linalg.eig(Jacobian(xp, p))
# print np.max(eigVal.real)


# Read time series
simFile = '%s/simulation/sim%s.%s' \
          % (cfg.general.resDir, srcPostfix, cfg.simulation.file_format)
print 'Reading time series from ' + simFile
if cfg.simulation.file_format == 'bin':
    X = np.fromfile(simFile, dtype=float, count=readCount)
else:
    X = np.loadtxt(simFile, dtype=float)
X = X.reshape(-1, cfg.model.dim)
time = np.arange(0., cfg.simulation.LCut, cfg.simulation.printStep)


# Define Poincare section
par0 = np.array([cfg.model.rho, cfg.model.sigma, cfg.model.beta])
pMax = 5
delta = 5.e-1
#z = rho - 1
cross = (X[:-1, 2] >= par0[0] - 1.) & (X[1:, 2] < par0[0] - 1.)
idxCross = np.nonzero(cross)[0]
pairs = []
dist = []
# Order pairs by number of crossing
for ip in np.arange(pMax):
    p = ip + 1
    x0 = X[idxCross[:-p]]
    xf = X[idxCross[p:]]
    periods = time[idxCross[p:]] - time[idxCross[:-p]]
    d = np.sqrt(np.sum((x0 - xf)**2, 1))
    isort = np.argsort(d)
    x0 = x0[isort]
    xf = xf[isort]
    periods = periods[isort]
    d = d[isort]
    pairs.append((x0, xf, periods))
    dist.append(d)

# Apply Newton
Jp = J.subs(rho, par0[0]).subs(sigma, par0[1]).subs(beta, par0[2])
for ip in np.arange(pMax):
    p = ip + 1
    x0 = pairs[ip][0][0]
    xf = pairs[ip][1][0]
    period = pairs[ip][2][0]
    Jx0 = np.matrix(Jp.subs(x, x0[0]).subs(y, x0[1]).subs(z, x0[2]).tolist(),
                    dtype=float)
    mono = linalg.expm(Jx0 * period)
    xf = optimize.root(field, x0, jac=Jacobian, args=(p0,), method=method,
                       tol=tol)
