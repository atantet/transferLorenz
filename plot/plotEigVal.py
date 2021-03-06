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

L = cfg.simulation.LCut + cfg.simulation.spinup
spinup = cfg.simulation.spinup
tau = cfg.transfer.tauRng[0]
printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt + 0.1)
caseName = cfg.model.caseName
rho = cfg.model.rho
dim = cfg.model.dim
dimObs = dim
nProc = ''
if (hasattr(cfg.sprinkle, 'nProc')):
    nProc = '_nProc' + str(cfg.sprinkle.nProc)

N = np.prod(np.array(cfg.grid.nx))
gridPostfix = ""
for d in np.arange(dimObs):
    if (hasattr(cfg.grid, 'nSTDLow') & hasattr(cfg.grid, 'nSTDHigh')):
        gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, cfg.grid.nx[d],
                                        cfg.grid.nSTDLow[d],
                                        cfg.grid.nSTDHigh[d])
    else:
        gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, cfg.grid.nx[d],
                                        cfg.sprinkle.minInitState[d],
                                        cfg.sprinkle.maxInitState[d])
readSpec = ergoPlot.readSpectrumCompressed
srcPostfixSim = "_%s%s_rho%04d_L%d_dt%d_nTraj%d%s" \
                % (caseName, gridPostfix, int(rho * 100 + 0.1),
                   int(tau * 1000 + 0.1),
                   -np.round(np.log10(cfg.simulation.dt)),
                   cfg.sprinkle.nTraj, nProc)
postfix = "%s" % (srcPostfixSim,)
# readSpec = ergoPlot.readSpectrum
# srcPostfixSim = "_%s_rho%04d_L%d_spinup%d_dt%d_samp%d_nTraj%d" \
#                 % (caseName, int(rho * 100 + 0.1), int(L + 0.1),
#                    int(spinup + 0.1), -np.round(np.log10(cfg.simulation.dt)),
#                    int(printStepNum), cfg.sprinkle.nTraj)
# postfix = "%s%s_tau%03d" % (srcPostfixSim, gridPostfix, int(tau * 1000 + 0.1))

xmineigVal = -cfg.stat.rateMax
ymineigVal = -cfg.stat.angFreqMax
xlimEig = [xmineigVal, -xmineigVal/100]
ylimEig = [ymineigVal, -ymineigVal]
xticks = None
yticksPos = np.arange(0, ylimEig[1], 5.)
yticksNeg = np.arange(0, ylimEig[0], -5.)[::-1]
yticks = np.concatenate((yticksNeg, yticksPos))


# Define file names
eigValForwardFile = '%s/eigval/eigvalForward_nev%d%s.%s' \
                    % (cfg.general.specDir, cfg.spectrum.nev, postfix,
                       cfg.general.fileFormat)

# Read transfer operator spectrum from file and create a bi-orthonormal basis
# of eigenvectors and backward eigenvectors:
print 'Readig spectrum for tau = %.3f...' % tau
(eigValForward,) = readSpec(eigValForwardFile)

# Get generator eigenvalues (using the complex logarithm)
eigValGen = np.log(eigValForward) / tau

realLabel = r'$\Re(\lambda_k)$'
imagLabel = r'$\Im(\lambda_k)$'

ergoPlot.plotEig(eigValGen, xlabel=realLabel, ylabel=imagLabel,
                 xlim=xlimEig, ylim=ylimEig)
plt.text(xlimEig[0]*0.2, ylimEig[1]*1.05, r'$\rho = %.1f$' % rho,
         fontsize=ergoPlot.fs_latex)
plt.savefig('%s/spectrum/eigVal/eigVal%s.%s'\
            % (cfg.general.plotDir, postfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

