import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
import ergoPlot
import fileinput

figFormat = 'eps'

def field(x, p, rec):
    rec[0] = p[1] * (x[1] - x[0])
    rec[1] = x[0] * (p[0] - x[2]) - x[1]
    rec[2] = x[0] * x[1] - p[2] * x[2]

def Jacobian(x, p):
    rho, sigma, beta = p
    J = np.array([[-sigma, sigma, 0.],
                  [rho - x[2], -1., -x[0]],
                  [x[1], x[0], -beta]])
    return J

def propagateRK4(x0, field, p, rec, dt, nt, samp=1):
    '''Propagate solution of ODE according to the vector field field \
    with Euler scheme from x0 for nt time steps of size dt.'''
    rec[0] = x0.copy()
    xt = x0.copy()
    k1 = np.empty((dim,))
    k2 = np.empty((dim,))
    k3 = np.empty((dim,))
    k4 = np.empty((dim,))
    tmp = np.empty((dim,))
    t = 1
    while t < nt:
        # Step solution forward
        field(xt, p, k1)
        k1 *= dt
        tmp[:] = k1 * 0.5 + xt

        field(tmp, p, k2)
        k2 *= dt
        tmp[:] = k2 * 0.5 + xt

        field(tmp, p, k3)
        k3 *= dt
        tmp[:] = k3 + xt

        field(tmp, p, k4)
        k4 *= dt
        tmp[:] = (k1 + 2*k2 + 2*k3 + k4) / 6
        
        xt[:] += tmp

        if (t % samp == 0):
            rec[t/samp] = xt
        t += 1

    return rec

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

delayName = ""
if (hasattr(cfg.model, 'delaysDays')):
    for d in np.arange(len(cfg.model.delaysDays)):
        delayName = "%s_d%d" % (delayName, cfg.model.delaysDays[d])

# List of continuations to plot
initCont = [7.956126, 7.956126, 24.737477, 24.5, 0.652822]
contStep = 0.01
dt = 1.e-5

srcPostfix = "_%s%s" % (caseName, delayName)
resDir = '../results/'
contDir = '%s/continuation' % resDir
plotDir = '%s/plot/' % resDir

contAbs = np.sqrt(contStep*contStep)
sign = contStep / contAbs
exp = np.log10(contAbs)
mantis = sign * np.exp(np.log(contAbs) / exp)
dstPostfix = "%s_cont%04d_contStep%de%d_dt%d_numShoot%d" \
             % (srcPostfix, int(initCont[dim] * 1000 + 0.1), int(mantis*1.01),
                (int(exp*1.01)), -np.round(np.log10(dt)),
                cfg.continuation.numShoot)
poFileName = '%s/poCont%s.txt' % (contDir, dstPostfix)
FloquetExpFileName = '%s/poExpCont%s.txt' % (contDir, dstPostfix)

# Read fixed point and cont
state = np.loadtxt(poFileName).reshape(-1, dim+2)
# Read FloquetExpenvalues
FloquetExp = np.loadtxt(FloquetExpFileName)
FloquetExp = (FloquetExp[:, 0] + 1j * FloquetExp[:, 1]).reshape(-1, dim)
# Remove last
state = state[:-1]
FloquetExp = FloquetExp[:-1]

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

isStable = np.max(FloquetExp.real, 1) < 1.e-6

#contSelRng = np.arange(24.1, 24.751, 0.05)
contSelRng = np.arange(24.3, 24.751, 0.05)
contSelRng[-1] = contRng.max()
dt = 1.e-3

distRng = np.empty((contSelRng.shape[0],))
L = 1000000
samp = 10
#L = 100
#samp = 1
f = open('%s/continuation/distChaosPeriodic%s.txt' \
         % (plotDir, dstPostfix), 'a')

for icont in np.arange(contSelRng.shape[0]):
    cont = contSelRng[icont]
    t = np.argmin((cont - contRng)**2)
    cont = contRng[t]
    T = TRng[t] * L
    print 'Propagating attractor orbit for ', T, ' at rho = ', cont, \
        ' from x(0) = ', po[t]

    nt = int(np.ceil(T / dt))
    # Propagate aperiodic orbit
    p = [cont, cfg.model.sigma, cfg.model.beta]
    xtA = np.empty((nt / samp + 1, dim))
    propagateRK4(po[t] + 5, field, p, xtA, dt, nt, samp=samp)    

    # Remove spinup of a tenth
    xtA = xtA[nt/samp/ 10:-1]
    
    # Propagate periodic orbit
    T = TRng[t]
    print 'Propagating orbit of period ', T, ' at cont = ', cont, \
        ' from x(0) = ', po[t]
    nt = int(np.ceil(T / dt))
    p = [cont, cfg.model.sigma, cfg.model.beta]
    xt = np.empty((nt + 1, dim))
    propagateRK4(po[t], field, p, xt, dt, nt)
    xt = xt[:-1]

    # Find minimum distance between two orbits
    print 'Getting minimum distance...'
    minDist = np.empty((xt.shape[0],))
    for k in np.arange(xt.shape[0]):
        x = np.tile(xt[k], (xtA.shape[0], 1))
        minDist[k] = np.min(np.sqrt(np.sum((xtA - x)**2, 1)))
    distRng[icont] = np.min(minDist)
    f.write(str(distRng[icont]) + '\n')
    f.flush()
f.close()

xticks = np.arange(24.1, 24.71, 0.1)
(xticks[0], xticks[-1]) = (24.06, 24.74)
xticklabels = []
for lab in np.arange(xticks.shape[0]):
    xticklabels.append(str(xticks[lab]))
xticklabels[0] = r'$\rho_A$'
xticklabels[-1] = r'$\rho_{\mathrm{Hopf}}$'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(contSelRng, distRng, '-k', linewidth=2)
ax.set_xlabel(r'$\rho$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$d(\Lambda, \Gamma^{\pm})$',
              fontsize=ergoPlot.fs_latex)
ax.set_xticks(xticks)
ax.set_xlim(xticks[0], xticks[-1])
ax.set_ylim(0., 4.)
ax.set_xticklabels(xticklabels)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
plt.savefig('%s/continuation/distChaosPeriodic%s.%s' \
            % (plotDir, dstPostfix, figFormat),
            dpi=300, bbox_inches='tight')
        
