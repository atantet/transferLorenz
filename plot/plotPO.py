import numpy as np
import matplotlib.pyplot as plt
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
initContRng = [[7.956126, 7.956126, 24.737477, 24.5, 0.652822],
               [10.33683, 6.022949, 23.479173, 15.477484, 1.415303]]
contStepRng = [0.01, -0.001, -0.0001]
dtRng = [1.e-5]*3
nCont = len(initContRng)

srcPostfix = "_%s%s" % (caseName, delayName)
resDir = '../results/'
contDir = '%s/continuation' % resDir
plotDir = '%s/plot/' % resDir

# Prepare plot
plotImag = False
#plotImag = True
if plotImag:
    fig = plt.figure(figsize=(6, 9))
else:
    fig = plt.figure(figsize=(6, 6))
ax = []
if plotImag:
    nPan = 100*(1+2*1) + 10 + 1
else:
    nPan = 100*(1+1) + 10 + 1
ax.append(fig.add_subplot(nPan))
#for k in np.arange(nCont):
for k in np.arange(1):
    nPan += 1
    ax.append(fig.add_subplot(nPan))
    if plotImag:
        nPan += 1
        ax.append(fig.add_subplot(nPan))

poL = []
FloquetExpL = []
contL = []
TRngL = []
contLim = np.empty((nCont, 2))
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

    if (fileFormat == 'bin'):
        # Read fixed point and cont
        state = np.fromfile(poFileName)
        # Read FloquetExpenvalues
        FloquetExp = np.fromfile(FloquetExpFileName)
    else:
        # Read fixed point and cont
        state = np.loadtxt(poFileName)
        # Read FloquetExpenvalues
        FloquetExp = np.loadtxt(FloquetExpFileName)
        
    state = state.reshape(-1, dim+2)
    FloquetExp = FloquetExp.reshape(-1, 2)
    FloquetExp = (FloquetExp[:, 0] + 1j * FloquetExp[:, 1]).reshape(-1, dim)

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

    asort = np.argsort(contRng)
    poL.append(po[asort])
    FloquetExpL.append(FloquetExp[asort])
    contL.append(contRng[asort])
    TRngL.append(TRng[asort])
    contLim[k, 0] = contRng[asort][0]
    contLim[k, 1] = contRng[asort][-1]
    
    isStable = np.max(FloquetExp.real, 1) < 1.e-6

    # Plot period
    ax[0].plot(contRng[isStable], TRng[isStable], '-k', linewidth=2)
    ax[0].plot(contRng[~isStable], TRng[~isStable], '--k', linewidth=2)

    # Plot real parts
    k = 0
    ax[1+2*k].plot(contRng, np.zeros((contRng.shape[0],)), '--k')
    ax[1+2*k].plot(contRng, FloquetExp.real, linewidth=2)
    ax[1+2*k].set_ylabel(r'$\Re(\lambda_i)$', fontsize=fs_latex)
    plt.setp(ax[1+2*k].get_xticklabels(), fontsize=fs_xticklabels)
    plt.setp(ax[1+2*k].get_yticklabels(), fontsize=fs_yticklabels)

    # Plot imaginary parts
    if plotImag:
        ax[1+2*k+1].plot(contRng, FloquetExp.imag, linewidth=2)
        ax[1+2*k+1].set_ylabel(r'$\Im(\lambda_i)$', fontsize=fs_latex)
        plt.setp(ax[1+2*k+1].get_xticklabels(), fontsize=fs_xticklabels)
        plt.setp(ax[1+2*k+1].get_yticklabels(), fontsize=fs_yticklabels)
ax[0].set_ylabel(r'$T$', fontsize=fs_latex)
plt.setp(ax[0].get_xticklabels(), fontsize=fs_xticklabels)
plt.setp(ax[0].get_yticklabels(), fontsize=fs_yticklabels)
ax[-1].set_xlabel(r'$\rho$', fontsize=fs_latex)
for k in np.arange(len(ax)):
    ax[k].set_xlim(np.min(contLim[:, 0]), np.max(contLim[:, 1]))

fig.savefig('%s/continuation/poCont%s.eps' % (plotDir, dstPostfix),
            dpi=300, bbox_inches='tight')


# Fixed point
initContRngFP = [[0., 0., 0., 0., 0.],
                 [1.6329, 1.6329, 1., 2.],
                 [1.6329, 1.6329, 1., 2.]]
contStepRngFP = [0.001, 0.001, -0.001]

fig = plt.figure()
ax = fig.add_subplot(111)
dist = []
for k in np.arange(len(initContRngFP)):
    initContFP = initContRngFP[k]
    contStepFP = contStepRngFP[k]
    contAbs = np.sqrt(contStepFP*contStepFP)
    sign = contStepFP / contAbs
    exp = np.log10(contAbs)
    mantis = sign * np.exp(np.log(contAbs) / exp)
    dstPostfixFP = "%s_cont%04d_contStep%de%d" \
                   % (srcPostfix, int(initContFP[dim] * 1000 + 0.1),
                      int(mantis*1.01), (int(exp*1.01)))
    fpFileName = '%s/fpCont%s.txt' % (contDir, dstPostfixFP)
    eigFileName = '%s/fpEigCont%s.txt' % (contDir, dstPostfixFP)

    # Read fixed point and cont
    state = np.loadtxt(fpFileName).reshape(-1, dim+1)
    fp = state[:, :dim]
    contRngFP = state[:, dim]
    eig = np.loadtxt(eigFileName)
    eig = (eig[:, 0] + 1j * eig[:, 1]).reshape(-1, dim)

    isStable = np.max(eig.real, 1) < 0

    ax.plot(fp[isStable, 0] + fp[isStable, 1], fp[isStable, 2], '-k',
            linewidth=2)
    ax.plot(fp[~isStable, 0] + fp[~isStable, 1], fp[~isStable, 2], '--k',
            linewidth=2)

sampOrbitRng = [150, 4000]
sampInit = [0, 0]
for k in np.arange(nCont-1, -1, -1):
    sampOrbit = sampOrbitRng[k]
    po = poL[k]
    FloquetExp = FloquetExpL[k]
    contRng = contL[k]
    TRng = TRngL[k]
    isStable = np.max(FloquetExp.real, 1) < 1.e-6
    
    for t in np.arange(sampInit[k], contRng.shape[0], sampOrbit):
        cont = contRng[t]
        T = TRng[t]
        print 'Propagating orbit of period ', T, ' at cont = ', cont, \
            ' from x(0) = ', po[t]
        print 'Floquet = ', FloquetExp[t]

        nt = int(np.ceil(T / dtRng[k]))
        # propagate
        p = [cont, cfg.model.sigma, cfg.model.beta]
        xt = propagateRK4(po[t], field, p, dtRng[k]*10, nt/10)
        if isStable[t]:
            ls = '-'
        else:
            ls = '--'
        #ax.plot(xt[:, 0] + xt[:, 2], xt[:, 1], xt[:, 3],
        #         linestyle=ls, linewidth=2)
        ax.plot(xt[:, 0] + xt[:, 1], xt[:, 2], linestyle=ls, linewidth=2,
                label=r'$\rho = %.1f$' % cont)
        

# Homoclinic orbit
print 'Propagating homoclinic orbit.'
cont = contL[1][0]
Th = TRngL[1][0]
nt = int(np.ceil(Th / cfg.simulation.dt * 3.))
# propagate
p = [cont-0.1791603, cfg.model.sigma, cfg.model.beta]
#x0 = xt[-1]
#x0[2] += 0.1
#x0 = np.array([ 0.152511, 0.25898653, 0.49468841])
x0 = np.array([ 0.152511, 0.25898653, 0.49468841]) / 10000
xt = propagateRK4(x0, field, p, cfg.simulation.dt*10, nt/10)
spinup = int(Th*6 / cfg.simulation.dt / 10)
line, = ax.plot(xt[:, 0] + xt[:, 1], xt[:, 2], '--k', linewidth=1)
ax.legend(loc='upper left', fontsize='x-large', ncol=2)
ax.set_xlabel(r'$x + y$', fontsize=fs_latex)
ax.set_ylabel(r'$z$', fontsize=fs_latex)
# ax.set_xlim(-2.8, 2.8)
ax.set_ylim(0., 36.)
plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)
fig.savefig('%s/continuation/poContOrbits%s.eps' % (plotDir, dstPostfix),
            dpi=300, bbox_inches='tight')
