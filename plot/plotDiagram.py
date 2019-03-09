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
fig = plt.figure()
xmin = cfg.continuation.contMin
xmax = cfg.continuation.contMax
ax = fig.add_subplot(111)

for k in np.arange(nCont):
    initCont = initContRng[k]
    contStep = contStepRng[k]
    
    contAbs = np.sqrt(contStep*contStep)
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

    isStable = np.max(eig.real, 1) < 0
    change = np.nonzero(~isStable)[0][0]
    print 'Change of stability at cont = ', contRng[change]
    print 'Fixed point at change of instability: ', fp[change]
    print 'Characteristic exponents at instability: ', eig[change]

    # Plot diagram
    ax.plot(contRng[isStable], fp[:, 0][isStable], '-k', linewidth=2)
    ax.plot(contRng[~isStable], fp[:, 0][~isStable], '--k', linewidth=2)

ax.annotate('', xy=(contRng[isStable][-1], fp[isStable, 0][-1]),
            xycoords='data', xytext=(15., fp[isStable, 0][-1]),
            textcoords='data', color='k',
            arrowprops=dict(arrowstyle='simple, tail_width= 0.08',
                            fc='0.4', ec="none",
                            connectionstyle="arc3,rad=-0.3"))

ax.set_ylabel(r'$x$', fontsize=fs_latex)
ax.set_xlim(xmin, xmax)
plt.setp(ax.get_xticklabels(), fontsize=fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=fs_yticklabels)
ax.set_xlabel(r'$\rho$', fontsize=fs_latex)

plt.savefig('%s/continuation/diagram%s.%s' \
            % (plotDir, dstPostfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

