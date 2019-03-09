import os
import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
import ergoPlot

configFile = '../cfg/Lorenz63.cfg'
compName1 = 'x'
compName2 = 'y'
compName3 = 'z'
#readSpec = ergoPlot.readSpectrum
readSpec = ergoPlot.readSpectrumCompressed

cfg = pylibconfig2.Config()
cfg.read_file(configFile)

L = cfg.simulation.LCut + cfg.simulation.spinup
tau = cfg.transfer.tauRng[0]
printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt + 0.1)
caseName = cfg.model.caseName
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
gridPostfix = "_%s%s" % (caseName, gridPostfix)
rhoRng = np.arange(22., 26.01, 0.5)
rhoRng = np.concatenate((rhoRng, np.arange(23.1, 24.91, 0.1)))
rhoRng = np.unique((rhoRng * 10 + 0.1).astype(int)) * 1. / 10
rhoRng = np.concatenate(([20, 21.], rhoRng, [27., 28.]))
nTrajRng = np.ones((rhoRng.shape[0],), dtype=int) * 500000000
nTrajRng[(rhoRng > 23.09) & (rhoRng < 23.99)] = 100000000
nTrajRng[np.argmin(np.abs(rhoRng - 23.5))] = 500000000
# gapRng = np.empty((rhoRng.shape[0],))
eigValRng = np.empty((rhoRng.shape[0], cfg.spectrum.nev))
for irho in np.arange(rhoRng.shape[0]):
    rho = rhoRng[irho]
    simPostfix = "_L%d_dt%d_nTraj%d%s" \
                 % (int(tau * 1000 + 0.1),
                    -np.round(np.log10(cfg.simulation.dt)),
                    nTrajRng[irho], nProc)
    srcPostfixSim = "%s_rho%04d%s" % (gridPostfix, int(rho * 100 + 0.1),
                                      simPostfix)

    # Define file names
    postfix = srcPostfixSim
    eigValForwardFile = '%s/eigval/eigvalForward_nev%d%s.%s' \
                        % (cfg.general.specDir, cfg.spectrum.nev, postfix,
                           cfg.general.fileFormat)

    # Read transfer operator spectrum from file
    # and create a bi-orthonormal basis
    # of eigenvectors and backward eigenvectors:
    print 'Readig spectrum for tau = %.3f...' % tau
    (eigValForward,) = readSpec(eigValForwardFile)

    # Get generator eigenvalues (using the complex logarithm)
    eigValGen = np.log(eigValForward) / tau
    isort = np.argsort(-eigValGen.real)
    eigValGen = eigValGen[isort]

    # Record spectral gap
    eigValRng[irho] = eigValGen[:cfg.spectrum.nev].real
    # Remove spurious positive eigenvalues (where do they come from?)
    eigValRng[irho, eigValRng[irho] > 1.e-4] = np.nan
    isort = np.argsort(-eigValRng[irho])
    eigValRng[irho] = eigValRng[irho, isort]


# Plot
dstPostfix = "%s_rhomin%04d_rhomax%04d%s" \
             % (gridPostfix, int(rhoRng[0] * 100 + 0.1),
                int(rhoRng[-1] * 100 + 0.1), simPostfix)
nevPlot = 7
ls = '-'
mk = '+'
lw = 2
markers = ['o', '+', 'x']
msizes = [8, 8, 6]
colors = ['k', 'k', 'k']
fig = plt.figure()
ax = fig.add_subplot(111)
linestyles = ['-', '--']
for ev in np.arange(nevPlot):
    ax.plot(rhoRng, eigValRng[:, ev].real, linestyle=linestyles[ev%2],
            linewidth=lw)
ax.set_xlim(rhoRng[0], rhoRng[-1])
ax.set_ylim(-0.12, 0.001)
ax.set_xlabel(r'$\rho$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Re(\lambda_i)$', fontsize=ergoPlot.fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
plt.savefig('%s/spectrum/eigVal/eigVSRho%s.%s'\
            % (cfg.general.plotDir, dstPostfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# selRho = (rhoRng >= 21.) & (rhoRng <= 25.)
# for ev in np.arange(nevPlot):
#     ax.plot(rhoRng[selRho], eigValRng[selRho, ev].real,
#             linestyle=linestyles[ev%d], linewidth=lw)
# ax.set_xlim(rhoRng[selRho][0], rhoRng[selRho][-1])
# ax.set_ylim(-0.06, 0.002)
# ax.set_xticks(np.arange(21, 25.1, 1))
# ax.set_xlabel(r'$\rho$', fontsize=ergoPlot.fs_latex)
# ax.set_ylabel(r'$\Re(\lambda_i)$', fontsize=ergoPlot.fs_latex)
# plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
# plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
# plt.savefig('%s/spectrum/eigVal/eigVSRhoZoom%s.%s'\
#             % (cfg.general.plotDir, dstPostfix, ergoPlot.figFormat),
#             dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(rhoRng, np.zeros((rhoRng.shape[0],)), '-k', linewidth=0.5)
# for ev in np.arange(nevPlot):
#     ax.plot(rhoRng, eigValRng[:, ev].real, marker=markers[ev],
#             color=colors[ev], markersize=msizes[ev],
#             linestyle='none', fillstyle='none')
# ax.set_xlim(rhoRng[0], rhoRng[-1])
# ax.set_xlabel(r'$\rho$', fontsize=ergoPlot.fs_latex)
# ax.set_ylabel(r'$\Re(\lambda_i)$', fontsize=ergoPlot.fs_latex)
# plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
# plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
# plt.savefig('%s/spectrum/eigVal/eigVSRho%s.%s'\
#             % (cfg.general.plotDir, dstPostfix, ergoPlot.figFormat),
#             dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# selRho = (rhoRng >= 21.) & (rhoRng <= 25.)
# ax.plot(rhoRng[selRho], np.zeros((rhoRng[selRho].shape[0],)), '-k',
#         linewidth=0.5)
# for ev in np.arange(nevPlot):
#     ax.plot(rhoRng[selRho], eigValRng[selRho, ev].real, marker=markers[ev],
#             color=colors[ev], markersize=msizes[ev],
#             linestyle='none', fillstyle='none')
# ax.set_xlim(rhoRng[selRho][0], rhoRng[selRho][-1])
# ax.set_ylim(-0.06, 0.002)
# ax.set_xticks(np.arange(21, 25.1, 1))
# ax.set_xlabel(r'$\rho$', fontsize=ergoPlot.fs_latex)
# ax.set_ylabel(r'$\Re(\lambda_i)$', fontsize=ergoPlot.fs_latex)
# plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
# plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
# plt.savefig('%s/spectrum/eigVal/eigVSRhoZoom%s.%s'\
#             % (cfg.general.plotDir, dstPostfix, ergoPlot.figFormat),
#             dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)
