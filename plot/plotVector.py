import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylibconfig2

def field(x, p):
    rho, sigma, beta = p
    f = np.array([sigma * (x[1] - x[0]),
                  x[0] * (rho - x[2]) - x[1],
                  x[0] * x[1] - beta * x[2]])
    return f

def Jacobian(x, p):
    rho, sigma, beta = p
    J = np.array([[-sigma, sigma, 0.],
                  [rho - x[2], -1., -x[0]],
                  [x[1], x[0], -beta]])
    return J

def propagateEuler(x0, field, p, dt, nt):
    '''Propagate solution of ODE according to the vector field field \
    with Euler scheme from x0 for nt time steps of size dt.'''
    xt = np.empty((nt, x0.shape[0]))
    xt[0] = x0.copy()
    for t in np.arange(1, nt):
        # Step solution forward
        xt[t] = xt[t-1] + dt * field(xt[t-1], p)

    return xt

def propagateRK4(x0, field, p, dt, nt):
    '''Propagate solution of ODE according to the vector field field \
    with Euler scheme from x0 for nt time steps of size dt.'''
    xt = np.empty((nt, x0.shape[0]))
    xt[0] = x0.copy()
    for t in np.arange(1, nt):
        # Step solution forward
        k1 = field(xt[t-1], p) * dt
        tmp = k1 * 0.5 + xt[t-1]

        k2 = field(tmp, p) * dt
        tmp = k2 * 0.5 + xt[t-1]

        k3 = field(tmp, p) * dt
        tmp = k3 + xt[t-1]

        k4 = field(tmp, p) * dt
        tmp = (k1 + 2*k2 + 2*k3 + k4) / 6
        
        xt[t] = xt[t-1] + tmp

    return xt

fs_latex = 'xx-large'
fs_xticklabels = 'large'
fs_yticklabels = fs_xticklabels

configFile = '../cfg/Lorenz63.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)

dim = cfg.model.dim
L = cfg.simulation.LCut + cfg.simulation.spinup
printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt + 0.1)
caseName = cfg.model.caseName
fileFormat = cfg.general.fileFormat

delayName = ""
if (hasattr(cfg.model, 'delaysDays')):
    for d in np.arange(len(cfg.model.delaysDays)):
        delayName = "%s_d%d" % (delayName, cfg.model.delaysDays[d])

# List of continuations to plot
initContRng = [[7.956126, 7.956126, 24.737477, 24.5, 0.652822]]
#contStepRng = [0.01]
contStepRng = [-0.001]
dtRng = [1.e-5, 1.e-5]
nCont = len(initContRng)

srcPostfix = "_%s%s" % (caseName, delayName)
resDir = '../results/'
contDir = '%s/continuation' % resDir
plotDir = '%s/plot/' % resDir

k = 0
initCont = initContRng[k]
contStep = contStepRng[k]

contAbs = sqrt(contStep*contStep)
sign = contStep / contAbs
exp = np.log10(contAbs)
mantis = sign * np.exp(np.log(contAbs) / exp)
dstPostfix = "%s_cont%04d_contStep%de%d_dt%d_numShoot%d" \
             % (srcPostfix, int(initCont[dim] * 1000 + 0.1),
                int(mantis*1.01), (int(exp*1.01)),
                -np.round(np.log10(dtRng[k])), cfg.continuation.numShoot)
poFileName = '%s/poCont%s.%s' % (contDir, dstPostfix, fileFormat)
FloquetExpFileName = '%s/poExpCont%s.%s' % (contDir, dstPostfix, fileFormat)
FloquetVecFileName = '%s/poVecCont%s.%s' % (contDir, dstPostfix, fileFormat)


if (fileFormat == 'bin'):
    # Read fixed point and cont
    state = np.fromfile(poFileName)
    # Read FloquetExpenvalues
    FloquetExp = np.fromfile(FloquetExpFileName)
    # Read fundamental matrices
    FloquetVec = np.fromfile(FloquetVecFileName)
else:
    # Read fixed point and cont
    state = np.loadtxt(poFileName)
    # Read FloquetExpenvalues
    FloquetExp = np.loadtxt(FloquetExpFileName)
    # Read fundamental matrices
    FloquetVec = np.loadtxt(FloquetVecFileName)
state = state.reshape(-1, dim+2)
FloquetExp = FloquetExp.reshape(-1, 2)
FloquetExp = (FloquetExp[:, 0] + 1j * FloquetExp[:, 1]).reshape(-1, dim)
FloquetVec = FloquetVec.reshape(-1, 2)
FloquetVecReal = FloquetVec[:, 0].reshape(-1, dim, dim)
FloquetVecImag = FloquetVec[:, 1].reshape(-1, dim, dim)

po = state[:, :dim]
TRng = state[:, dim+1]
contRng = state[:, dim]


# # Reorder Floquet exp
# for t in np.arange(1, contRng.shape[0]):
#     tmp = FloquetExp[t].tolist()
#     for exp in np.arange(dim):
#         idx = np.argmin(np.abs(tmp - FloquetExp[t-1, exp]))
#         FloquetExp[t, exp] = tmp[idx]
#         tmp.pop(idx)

#contSel = 24.09
contSel = 14.
idx = np.argmin((contRng - contSel)**2)
cont = contRng[idx]
poSel = po[idx]
T = TRng[idx]
FV = FloquetVecReal[idx]
FE = FloquetExp[idx]
    
nt = int(np.ceil(T / dtRng[k]))
# propagate
p = [cont, cfg.model.sigma, cfg.model.beta]
xt = propagateRK4(poSel, field, p, dtRng[k]*10, nt/10)
dstPostfixPlot = "%s_cont%04d_contStep%de%d_dt%d_numShoot%d" \
                 % (srcPostfix, int(cont * 1000 + 0.1),
                    int(mantis*1.01), (int(exp*1.01)),
                    -np.round(np.log10(dtRng[k])), cfg.continuation.numShoot)

# Plot
LyapExpNames = ['+', '0', '-']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xt[:, 0], xt[:, 1], xt[:, 2], linestyle='--', linewidth=2)
scale = 3
for d in np.arange(dim):
    if FE[d] > 1.e-6:
        label = '+'
    elif FE[d] < -1.e-6:
        label = '-'
    else:
        label = '0'
        
    ax.plot([poSel[0], poSel[0] + scale*FV[0, d]],
            [poSel[1], poSel[1] + scale*FV[1, d]],
            [poSel[2], poSel[2] + scale*FV[2, d]], linestyle='-', linewidth=2,
            label=r'$v^{%s}$' % label)
ax.legend(loc='lower right', fontsize=fs_latex)
ax.set_xlabel(r'$x$', fontsize=fs_latex)
ax.set_ylabel(r'$y$', fontsize=fs_latex)
ax.set_zlabel(r'$z$', fontsize=fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)
plt.setp(ax.get_zticklabels(), fontsize=fs_xticklabels)
plt.savefig('%s/continuation/FloquetVec%s.eps' % (plotDir, dstPostfixPlot),
           dpi=300, bbox_inches='tight')
