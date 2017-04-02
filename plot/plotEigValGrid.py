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
#readSpec = ergoPlot.readSpectrum
readSpec = ergoPlot.readSpectrumCompressed

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

embedding = (np.array(cfg.observable.embeddingDays) / 365 \
             / cfg.simulation.printStep).astype(int)
rho = cfg.model.rho
dim = cfg.model.dim
dimObs = dim
nProc = ''
if (hasattr(cfg.sprinkle, 'nProc')):
    nProc = '_nProc' + str(cfg.sprinkle.nProc)

# Read grid
nx0Rng = np.arange(100, 501, 100)
#nx0Rng = np.arange(100, 501, 50)
nGrids = nx0Rng.shape[0]
eigValGen = np.zeros((nGrids, cfg.spectrum.nev), dtype=complex)
aspect = np.array([1, 2, 1], dtype=int)
for k in np.arange(nGrids):
    nx0 = nx0Rng[k]
    N = nx0**cfg.model.dim
    gridPostfix = ""
    for d in np.arange(dimObs):
        if (hasattr(cfg.grid, 'nSTDLow') & hasattr(cfg.grid, 'nSTDHigh')):
            gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, nx0 / aspect[d],
                                            cfg.grid.nSTDLow[d],
                                            cfg.grid.nSTDHigh[d])
        else:
            gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, nx0 / aspect[d],
                                            cfg.sprinkle.minInitState[d],
                                            cfg.sprinkle.maxInitState[d])
    gridPostfix = "_%s%s" % (caseName, gridPostfix)
    srcPostfixSim = "%s_rho%04d_L%d_dt%d_nTraj%d%s" \
                    % (gridPostfix, int(rho * 100 + 0.1),
                       int(tau * 1000 + 0.1),
                       -np.round(np.log10(cfg.simulation.dt)),
                       cfg.sprinkle.nTraj, nProc)

    # Define file names
    postfix = "%s" % (srcPostfixSim,)
    eigValForwardFile = '%s/eigval/eigvalForward_nev%d%s.%s' \
                        % (cfg.general.specDir, cfg.spectrum.nev, postfix,
                           cfg.general.fileFormat)

    # Read transfer spectrum from file and create a bi-orthonormal basis
    # of eigenvectors and backward eigenvectors:
    print 'Readig spectrum for nx0 = %d...' % nx0
    (eigValForward,) = readSpec(eigValForwardFile)
    isort = np.argsort(-np.abs(eigValForward))
    nevk = eigValForward.shape[0]
    if nevk > cfg.spectrum.nev:
        nevk = cfg.spectrum.nev
    eigValGen[k, :nevk] = np.log(eigValForward[isort][:nevk]) / tau


# Plot
nevPlot = 5
lw = 2
fig = plt.figure()
ax = fig.add_subplot(111)
ls = ['-', '-']
xticks = nx0Rng
yticks = np.arange(-0.2, 0.01, 0.02)
for ev in np.arange(nevPlot):
    plt.plot(nx0Rng, eigValGen[:, ev].real, linewidth=lw, linestyle=ls[ev%2])
ax.set_xlabel(r'$n_d$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Re(\lambda_k)$', fontsize=ergoPlot.fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
ax.set_xticks(xticks)
ax.set_xlim(nx0Rng[0], nx0Rng[-1])
ax.set_yticks(yticks)
ax.set_ylim(yticks[0], yticks[-1] + 0.002)
plt.savefig('%s/spectrum/eigval/eigValGridCvg%s.%s'
            % (cfg.general.plotDir, postfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)


