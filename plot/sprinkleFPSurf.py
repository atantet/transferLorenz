import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylibconfig2
import matplotlib.tri as mtri
from scipy.integrate import ode

def field(t, x, p):
    rho, sigma, beta = p
    f = np.array([sigma * (x[1] - x[0]),
                  x[0] * (rho - x[2]) - x[1],
                  x[0] * x[1] - beta * x[2]])
    return f

def Jacobian(t, x, p):
    rho, sigma, beta = p
    J = np.array([[-sigma, sigma, 0.],
                  [rho - x[2], -1., -x[0]],
                  [x[1], x[0], -beta]])
    return J

def fieldJac(t, dx, p):
    rho, sigma, beta, x = p
    return np.dot(Jacobian(t, x, p[:-1]), dx)

fs_latex = 'xx-large'
fs_xticklabels = 'large'
fs_yticklabels = fs_xticklabels

configFile = '../cfg/Lorenz63.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)


dim = cfg.model.dim
rho = cfg.model.rho
sigma = cfg.model.sigma
beta = cfg.model.beta
L = cfg.simulation.LCut + cfg.simulation.spinup
printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt + 0.1)
caseName = cfg.model.caseName
fileFormat = cfg.general.fileFormat

delayName = ""
if (hasattr(cfg.model, 'delaysDays')):
    for d in np.arange(len(cfg.model.delaysDays)):
        delayName = "%s_d%d" % (delayName, cfg.model.delaysDays[d])

srcPostfix = "_%s%s" % (caseName, delayName)
resDir = '../results/'
contDir = '%s/continuation' % resDir
plotDir = '%s/plot/' % resDir


# List of continuations for Periodic orbits
initContRng = [[7.956126, 7.956126, 24.737477, 24.5, 0.652822]]
contStepRng = [0.01]
dtRng = [1.e-5, 1.e-5]
nCont = len(initContRng)

poL = np.empty((1, dim))
FloquetExpL = np.empty((1, dim), dtype=complex)
FloquetVecRealL = np.empty((1, dim, dim))
contL = np.empty((1,))
TRngL = np.empty((1,))
for k in np.arange(nCont):
    initCont = initContRng[k]
    contStep = contStepRng[k]
    
    contAbs = np.sqrt(contStep*contStep)
    sign = contStep / contAbs
    exp = np.log10(contAbs)
    mantis = sign * np.exp(np.log(contAbs) / exp)
    dstPostfix = "%s_cont%04d_contStep%de%d_dt%d_numShoot%d" \
                 % (srcPostfix, int(initCont[dim] * 1000 + 0.1), int(mantis*1.01),
                    (int(exp*1.01)), -np.round(np.log10(dtRng[k])),
                    cfg.continuation.numShoot)
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


    # Reorder Floquet exp
    for t in np.arange(1, contRng.shape[0]):
        tmp = FloquetExp[t].tolist()
        for exp in np.arange(dim):
            idx = np.argmin(np.abs(tmp - FloquetExp[t-1, exp]))
            FloquetExp[t, exp] = tmp[idx]
            tmp.pop(idx)

    poL = np.concatenate((poL, po), axis=0)
    FloquetExpL = np.concatenate((FloquetExpL, FloquetExp), axis=0)
    FloquetVecRealL = np.concatenate((FloquetVecRealL, FloquetVecReal), axis=0)
    contL = np.concatenate((contL, contRng), axis=0)
    TRngL = np.concatenate((TRngL, TRng), axis=0)

poL = poL[1:]
FloquetExpL = FloquetExpL[1:]
FloquetVecRealL = FloquetVecRealL[1:]
contL = contL[1:]
TRngL = TRngL[1:]


# Ensemble
Ns = 2
L = 5.e1
dt = 1.e-3
eps = 1.e-8

#contSel = 24.047927
contSel = 20.
idx = np.argmin((contL - contSel)**2)
cont = contL[idx]
poSel = poL[idx]
TSel = TRngL[idx]
FE = FloquetExpL[idx]
FV = FloquetVecRealL[idx]
p = [cont, sigma, beta]

# Eigenvectors and eigenvalues
eigVal = np.array([-beta,
                   (-sigma - np.sqrt(4*cont*sigma + sigma**2 - 2*sigma + 1) - 1)/2,
                   (-sigma + np.sqrt(4*cont*sigma + sigma**2 - 2*sigma + 1) - 1)/2])
v0 = np.array([0., 0., 1.])
v1 = np.array([2.*sigma \
               / (sigma - np.sqrt(4*cont*sigma + sigma**2 - 2*sigma + 1.) - 1),
               1., 0.])
v2 = np.array([2.*sigma \
               / (sigma + np.sqrt(4*cont*sigma + sigma**2 - 2*sigma + 1.) - 1),
               1., 0.])
v1 = v1 / np.sqrt(np.dot(v1, v1))
v2 = v2 / np.sqrt(np.dot(v1, v1))
eigVec = [v0, v1, v2]

C0 = np.array([0., 0., 0.])
C1 = np.array([np.sqrt(beta*(cont - 1)),
               np.sqrt(beta*(cont - 1)),
               cont - 1])
C2 = np.array([-np.sqrt(beta*(cont - 1)),
               -np.sqrt(beta*(cont - 1)),
               cont - 1])

# Set integrator
r = ode(field).set_integrator('dopri5')
r.set_f_params(p)
# Set linear integrator
rlin = ode(fieldJac).set_integrator('dopri5')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot unstable manifold of fixed point
nt = int(L / dt)
idEig = 2
x0 = C0 + eps * eigVal[idEig] * eigVec[idEig]
print 'Integrating from x0 = ', x0
r.set_initial_value(x0, 0.)
xt = np.zeros((nt, dim))
xt[0] = x0
for k in np.arange(1, nt):
    xt[k] = r.integrate(r.t + dt)
print 'Plot...'
ax.plot(xt[:, 0], xt[:, 1], xt[:, 2])
# Plot symmetric
#ax.plot(-xt[:, 0], -xt[:, 1], xt[:, 2])

# # Plot other orbit
# nt = int(L / dt)
# x0L = [np.array([5., 0., 0.]), np.array([0., 5., 0.])]
# lsL = ['-', '--']
# for s in np.arange(Ns):
#     print 's = ', s, ' / ', (Ns - 1)
#     x0 = x0L[s]
#     print 'x0 = ', x0
#     xt = propagateRK4(x0, field, p, dt, nt)
#     ax.plot(xt[:, 0], xt[:, 1], xt[:, 2], linestyle=lsL[s])

# Get periodic orbit
ntPO = int(np.ceil(TSel / dt) + 0.1)
x0 = poSel
print 'Integrating periodic orbit from x0 = ', x0
r.set_initial_value(x0, 0.)
xtPO = np.zeros((ntPO, dim))
timePO = np.zeros((ntPO,))
xtPO[0] = x0
for k in np.arange(1, ntPO):
    xtPO[k] = r.integrate(r.t + dt)
    timePO[k] = r.t + dt

# Plot stable manifold of periodic orbit
nt = int(L / dt / 50)
idx = np.argmin(FE.real)
dmax = 10.
xtL = np.empty((1, dim))
uvL = np.empty((1, 2))
px = [p[0], p[1], p[2], np.empty((dim,))]
rlin.set_initial_value(FV[:, idx], 0.)
sampPlot = int(20 * 1.e-3 / dt + 0.1)
sampLeaf = 10
for t in np.arange(0, ntPO):
    # Propagate Floquet Vector
    px[3] = xtPO[t]
    rlin.set_f_params(px)
    rlin.integrate(rlin.t + dt)
    # Renormalize
    #rlin.set_initial_value(rlin.y * np.exp(-FE[idx].real*dt), rlin.t)
    rlin.set_initial_value(rlin.y / np.sqrt(np.dot(rlin.y, rlin.y)), rlin.t)

    if (t%sampPlot == 0):  
        print 'time = ', timePO[t], ' / ', timePO[-1]
        # Plot symmetric
        #x0 = xtPO[t] + 10. * rlin.y
        x0 = xtPO[t] + 0.1 * FV[:, idx]
        print 'Integrating from x0 = ', x0
        r.set_initial_value(x0, 0.)
        xt = np.zeros((nt, dim))
        xt[0] = x0
        uv = np.zeros((nt, 2))
        dist = np.sqrt(np.min(np.sum((xt[0] - xtPO)**2, 1)))
        uv[0] = [t*dt/TSel*2*np.pi, dist]
        for k in np.arange(1, nt):
            x = r.integrate(r.t-dt)
            dist = np.sqrt(np.min(np.sum((x - xtPO)**2, 1)))
            if dist > dmax:
                xt[k] = np.nan
                uv[k] = np.nan
            else:
                xt[k] = x
                # Save phase and distance
                uv[k] = [t*dt/TSel*2*np.pi, dist]
        ax.plot(-xt[:, 0], -xt[:, 1], xt[:, 2], '--k', linewidth=1./2)
        #ax.plot([-xtPO[t, 0], -x0[0]], [-xtPO[t, 1], -x0[1]],
        #        [xtPO[t, 2], x0[2]], '-r', linewidth=1)
        xtL = np.concatenate((xtL, xt[::sampLeaf]), axis=0)
        uvL = np.concatenate((uvL, uv[::sampLeaf]), axis=0)
#     x0 = xtPO[t] - 1.e-1 * FV[:, idx]
#     print 'Integrating from x0 = ', x0
#     r.set_initial_value(x0, 0.)
#     xt = np.zeros((nt, dim))
#     xt[0] = x0
#     for k in np.arange(1, nt):
#         xt[k] = r.integrate(r.t-dt)
#     for s in np.arange(xt.shape[0]):
#         if np.sqrt(np.min(np.sum((xt[s] - xtPO)**2, 1))) > dmax:
#             xt[s] = np.nan
#     ax.plot(-xt[:, 0], -xt[:, 1], xt[:, 2], '--k', linewidth=1./2)
#     xtL = np.concatenate((xtL, xt), axis=0)

# Plot surface
xtL = xtL[1:]
xtL = xtL[~np.any(np.isnan(xtL), 1)]
uvL = uvL[1:]
uvL = uvL[~np.any(np.isnan(uvL), 1)]
tri = mtri.Triangulation(uvL[:, 0], uvL[:, 1])
ax.plot_trisurf(-xtL[:, 0], -xtL[:, 1], triangles=tri.triangles, Z=xtL[:, 2],
                cmap=matplotlib.cm.Spectral, linewidth=0.)

# Plot periodic orbit
# ax.plot(xtPO[:, 0], xtPO[:, 1], xtPO[:, 2], '-k', linewidth=2)
# plot symmetric
ax.plot(-xtPO[:, 0], -xtPO[:, 1], xtPO[:, 2], '-k', linewidth=2)


# Plot fixed points
FPSize = 42
ax.scatter(C0[0], C0[1], C0[2], s=56, c='k', marker='+')
# ax.scatter(C1[0], C1[1], C1[2], s=40, c='k', marker='o', edgecolors='none')
ax.scatter(C2[0], C2[1], C2[2], s=40, c='k', marker='o', edgecolors='none')

ax.set_xlabel(r'$x$', fontsize=fs_latex)
ax.set_ylabel(r'$y$', fontsize=fs_latex)
ax.set_zlabel(r'$z$', fontsize=fs_latex)
# ax.set_xlim(-2.8, 2.8)
# ax.set_ylim(1.2, 2.8)
plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)

# plt.savefig('%s/continuation/chaoticOrbit_rho%04d%s.%s' \
#             % (plotDir, cont*1000, dstPostfix, figFormat),
#             dpi=300, bbox_inches='tight')
        
