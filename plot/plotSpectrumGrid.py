import os
import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
import ergoPlot

#ergoPlot.dpi = 2000

configFile = '../cfg/Lorenz63.cfg'
compName1 = 'x'
compName2 = 'y'
compName3 = 'z'

cfg = pylibconfig2.Config()
cfg.read_file(configFile)

# Transition lag
if (hasattr(cfg.stat, 'tauPlot')):
    tau = cfg.stat.tauPlot
else:
    tau = cfg.transfer.tauRng[0]

delayName = ""
if (hasattr(cfg.model, 'delaysDays')):
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

embedding = (np.array(cfg.observable.embeddingDays) / 365 \
             / cfg.simulation.printStep).astype(int)
dimObs = len(cfg.observable.components)
obsName = ""
for d in np.arange(dimObs):
    obsName = "%s_c%d_e%d" % (obsName, cfg.observable.components[d],
                              cfg.observable.embeddingDays[d])

nLags = len(cfg.transfer.tauRng)
ev_xlabel = r'$%s$' % compName1
if dimObs > 1:
    ev_ylabel = r'$%s$' % compName2
if dimObs > 2:
    ev_zlabel = r'$%s$' % compName3
corrLabel = r'$C_{%s, %s}(t)$' % (compName1[0], compName1[0])
powerLabel = r'$S_{%s, %s}(\omega)$' % (compName1[0], compName1[0])
xlabelCorr = r'$t$'

xmineigVal = -cfg.stat.rateMax
ymineigVal = -cfg.stat.angFreqMax
xlimEig = [xmineigVal, -xmineigVal/100]
ylimEig = [ymineigVal, -ymineigVal]
zlimEig = [cfg.stat.powerMin, cfg.stat.powerMax]
xticks = None
yticksPos = np.arange(0, ylimEig[1], 5.)
yticksNeg = np.arange(0, ylimEig[0], -5.)[::-1]
yticks = np.concatenate((yticksNeg, yticksPos))
zticks = np.logspace(np.log10(zlimEig[0]), np.log10(zlimEig[1]),
                    int(np.round(np.log10(zlimEig[1]/zlimEig[0]) + 1)))
zticks = np.logspace(np.log10(zlimEig[0]), np.log10(zlimEig[1]),
                     int(np.round(np.log10(zlimEig[1]/zlimEig[0])/2 + 1)))



# Read grid
nx0Rng = np.arange(100, 501, 100)
#nx0Rng = np.arange(100, 501, 50)
nGrids = nx0Rng.shape[0]
eigValGen = np.empty((nGrids, cfg.spectrum.nev), dtype=complex)
for k in np.arange(nGrids):
    nx0 = nx0Rng[k]
    N = nx0**cfg.model.dim
    gridPostfix = ""
    for d in np.arange(dimObs):
        if (hasattr(cfg.grid, 'nSTDLow') & hasattr(cfg.grid, 'nSTDHigh')):
            gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, nx0,
                                            cfg.grid.nSTDLow[d],
                                            cfg.grid.nSTDHigh[d])
        else:
            gridPostfix = "%s_n%dminmax" % (gridPostfix, nx0)
    gridPostfix = "%s%s%s" % (srcPostfix, obsName, gridPostfix)

    # Define file names
    postfix = "%s_tau%03d" % (gridPostfix, tau * 1000)
    eigValForwardFile = '%s/eigval/eigvalForward_nev%d%s.%s' \
                        % (cfg.general.specDir, cfg.spectrum.nev, postfix,
                           cfg.general.fileFormat)

    # Read transfer operator spectrum from file and create a bi-orthonormal basis
    # of eigenvectors and backward eigenvectors:
    print 'Readig spectrum for nx0 = %d...' % nx0
    (eigValForward,) = ergoPlot.readSpectrum(eigValForwardFile, 
                                             fileFormat=cfg.general.fileFormat)
    isort = np.argsort(-np.abs(eigValForward))
    eigValGen[k] = np.log(eigValForward[isort][:cfg.spectrum.nev]) / tau


# Plot
nevPlot = 50
lw = 1
fig = plt.figure()
ax = fig.add_subplot(111)
for ev in np.arange(nevPlot):
    plt.plot(nx0Rng, eigValGen[:, ev].real, linewidth=lw)
ax.set_xlabel(r'$n_d$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Re(\lambda_k)$', fontsize=ergoPlot.fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
ax.set_xlim(nx0Rng[0], nx0Rng[-1])
ax.set_ylim(-3, 0.03)
plt.savefig('%s/spectrum/eigval/eigValGridCvg%s.%s'
            % (cfg.general.plotDir, postfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)


