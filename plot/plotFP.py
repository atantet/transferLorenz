import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2

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
initContRng = [[0., 0., 0., 0.],
               [1.6329, 1.6329, 1., 2.],
               [1.6329, 1.6329, 1., 2.]]
contStepRng = [0.001, 0.001, -0.001]
nCont = len(initContRng)

srcPostfix = "_%s%s" % (caseName, delayName)
resDir = '../results/'
contDir = '%s/continuation' % resDir
plotDir = '%s/plot/' % resDir

# Prepare plot
fig = plt.figure(figsize=(8, 10))
xmin = cfg.continuation.contMin
xmax = cfg.continuation.contMax
ax = []
nPan = 100*(1+2*nCont) + 10 + 1
ax.append(fig.add_subplot(nPan))
for k in np.arange(nCont):
    nPan += 1
    ax.append(fig.add_subplot(nPan))
    nPan += 1
    ax.append(fig.add_subplot(nPan))

fpL = []
eigL = []
contL = []
for k in np.arange(nCont):
    initCont = initContRng[k]
    contStep = contStepRng[k]
    
    contAbs = sqrt(contStep*contStep)
    sign = contStep / contAbs
    exp = np.log10(contAbs)
    mantis = sign * np.exp(np.log(contAbs) / exp)
    dstPostfix = "%s_cont%04d_contStep%de%d" \
                 % (srcPostfix, int(initCont[dim] * 1000 + 0.1), int(mantis*1.01),
                    (int(exp*1.01)))
    fpFileName = '%s/fpCont%s.txt' % (contDir, dstPostfix)
    eigFileName = '%s/fpEigCont%s.txt' % (contDir, dstPostfix)

    # Read fixed point and cont
    state = np.loadtxt(fpFileName).reshape(-1, dim+1)
    fp = state[:, :dim]
    contRng = state[:, dim]
    # Read eigenvalues
    eig = np.loadtxt(eigFileName)
    eig = (eig[:, 0] + 1j * eig[:, 1]).reshape(-1, dim)

    # # Bound
    # isig = contRng < 1.
    # contRng = contRng[isig]
    # fp = fp[isig]
    # eig = eig[isig]

    fpL.append(fp)
    eigL.append(eig)
    contL.append(contRng)
    
    isStable = np.max(eig.real, 1) < 0
    change = np.nonzero(~isStable)[0][0]
    print 'Change of stability at cont = ', contRng[change]
    print 'Fixed point at change of instability: ', fp[change]
    print 'Characteristic exponents at instability: ', eig[change]

    # # Save branches
    # np.savetxt('%s/continuation/contBranch%d.txt' % (plotDir, k), contRng)
    # np.savetxt('%s/continuation/fpBranch%d.txt' % (plotDir, k), fp)
    # np.savetxt('%s/continuation/eigRealBranch%d.txt' % (plotDir, k), eig.real)
    # np.savetxt('%s/continuation/eigImagBranch%d.txt' % (plotDir, k), eig.imag)

    # Plot diagram
    ax[0].plot(contRng[isStable], fp[:, 0][isStable], '-k', linewidth=2)
    ax[0].plot(contRng[~isStable], fp[:, 0][~isStable], '--k', linewidth=2)

    # Plot real parts
    ax[1+2*k].plot(contRng, np.zeros((contRng.shape[0],)), '--k')
    ax[1+2*k].plot(contRng, eig.real, linewidth=2)
    ax[1+2*k].set_ylabel(r'$\Re(\lambda_i)$', fontsize=fs_latex)
    plt.setp(ax[1+2*k].get_xticklabels(), fontsize=fs_xticklabels)
    plt.setp(ax[1+2*k].get_yticklabels(), fontsize=fs_yticklabels)
    ax[1+2*k].set_xlim(xmin, xmax)

    # Plot imaginary parts
    ax[1+2*k+1].plot(contRng, eig.imag, linewidth=2)
    ax[1+2*k+1].set_ylabel(r'$\Im(\lambda_i)$', fontsize=fs_latex)
    plt.setp(ax[1+2*k+1].get_xticklabels(), fontsize=fs_xticklabels)
    plt.setp(ax[1+2*k+1].get_yticklabels(), fontsize=fs_yticklabels)
    ax[1+2*k+1].set_xlim(xmin, xmax)
ax[0].set_ylabel(r'$x$', fontsize=fs_latex)
ax[0].set_xlim(xmin, xmax)
plt.setp(ax[0].get_xticklabels(), fontsize=fs_xticklabels)
plt.setp(ax[0].get_yticklabels(), fontsize=fs_yticklabels)
ax[-1].set_xlabel(r'$\rho$', fontsize=fs_latex)

plt.savefig('%s/continuation/fpCont%s.eps' % (plotDir, dstPostfix),
            dpi=300, bbox_inches='tight')

