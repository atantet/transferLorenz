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

N = np.prod(np.array(cfg.grid.nx))
gridPostfix = ""
for d in np.arange(dimObs):
    if (hasattr(cfg.grid, 'nSTDLow') & hasattr(cfg.grid, 'nSTDHigh')):
        gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, cfg.grid.nx[d],
                                        cfg.grid.nSTDLow[d],
                                        cfg.grid.nSTDHigh[d])
    else:
        gridPostfix = "%s_n%dminmax" % (gridPostfix, cfg.grid.nx[d])
gridPostfix = "%s%s%s" % (srcPostfix, obsName, gridPostfix)
gridFile = '%s/grid/grid%s.txt' % (cfg.general.resDir, gridPostfix)

nLags = len(cfg.transfer.tauRng)
ev_xlabel = r'$%s$' % compName1
if dimObs > 1:
    ev_ylabel = r'$%s$' % compName2
if dimObs > 2:
    ev_zlabel = r'$%s$' % compName3
corrLabel = r'$C_{%s, %s}(t)$' % (compName1[0], compName1[0])
powerLabel = r'$S_{%s, %s}(\omega)$' % (compName1[0], compName1[0])
xlabelCorr = r'$t$'

# Define file names
postfix = "%s_tau%03d" % (gridPostfix, tau * 1000)
eigValForwardFile = '%s/eigval/eigvalForward_nev%d%s.txt' \
                    % (cfg.general.specDir, cfg.spectrum.nev, postfix)

# Read transfer operator spectrum from file and create a bi-orthonormal basis
# of eigenvectors and backward eigenvectors:
print 'Readig spectrum for tau = %.3f...' % tau
eigValForward = np.loadtxt(eigValForwardFile)
eigValForward = eigValForward[:, 0] + 1j * eigValForward[:, 1]

# Get generator eigenvalues (using the complex logarithm)
eigValGen = np.log(eigValForward) / tau

idEV = 1
eigVal = eigValGen[idEV]

maxHarm = 20
# eigHarm = np.empty((maxHarm,), dtype=complex)
# idHarm = np.empty((maxHarm,), dtype=int)
# for k in np.arange(int(maxHarm / 2)):
#     idHarm[k] = np.argmin((eigVal.imag * (k+1) - eigValGen.imag)**2)
#     eigHarm[k] = eigValGen[idHarm[k]]
#     idHarm[int(maxHarm/2)+k] = np.argmin((eigVal.imag * (k+1) \
#                                           + eigValGen.imag)**2)
#     eigHarm[int(maxHarm/2)+k] = eigValGen[idHarm[int(maxHarm/2)+k]]

eps = 0.08
idHarm = np.minimum(np.abs(np.mod(eigValGen.imag, eigVal.imag)),
                    np.abs(np.mod(-eigValGen.imag, eigVal.imag))) \
                    < eps * np.abs(eigVal.imag)
# idHarmMinus = np.minimum(np.abs(np.mod(eigValGen.imag, -eigVal.imag)),
#                          np.abs(np.mod(-eigValGen.imag, -eigVal.imag))) \
#                          < eps * np.abs(eigVal.imag)
# idHarm = np.concatenate((idHarmPlus, idHarmMinus))

eigHarm = eigValGen[idHarm]

# Plot config
markersize=20
realLabel = r'$\Re(\lambda_k)$'
imagLabel = r'$\Im(\lambda_k)$'
nHarmPlot = 5
stepHarm = np.abs(eigVal.imag)
xmineigVal = -cfg.stat.rateMax
ymineigVal = -cfg.stat.angFreqMax
xlimEig = [xmineigVal, -xmineigVal/100]
ylimEig = [ymineigVal, -ymineigVal]
yticksPos = np.arange(0, ylimEig[1], 5.)
yticksNeg = np.arange(0, ylimEig[0], -5.)[::-1]
yticks = np.concatenate((yticksNeg, yticksPos))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(eigHarm.real, eigHarm.imag,
           s=markersize, marker='o', edgecolors='face', c='k')
ax.set_xlabel(realLabel, fontsize=ergoPlot.fs_default)
#ax.set_xticks(xticks)
ax.set_xlim(xlimEig)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
ax.set_ylabel(imagLabel, fontsize=ergoPlot.fs_default)
ax.set_yticks(yticks)
ax.set_ylim(ylimEig)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
ax.grid()
plt.savefig('%s/spectrum/harm/harm%d%s.%s'\
            % (cfg.general.plotDir, idEV,
               postfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)


