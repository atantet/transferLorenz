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
tau = cfg.transfer.tauRng[0]
printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt + 0.1)
caseName = cfg.model.caseName
dim = cfg.model.dim
dimObs = dim

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
simPostfix = "_L%d_dt%d_nTraj%d" \
             % (int(tau * 1000 + 0.1),
                -np.round(np.log10(cfg.simulation.dt)),
                cfg.sprinkle.nTraj)

rhoRng = np.array([0., 1., 2., 5., 7.5, 10., 11., 12., 13., 13.5, 13.75,
                   14., 15., 16., 18., 19., 19.5, 20., 20.5, 21., 21.25,
                   21.5, 21.75, 22., 22.25, 22.5, 22.75, 23., 23.25, 23.5,
                   23.75, 24., 24.25, 24.5, 24.75, 25., 26.])
# gapRng = np.empty((rhoRng.shape[0],))
eigValRng = np.empty((rhoRng.shape[0], cfg.spectrum.nev))
dstPostfix = "%s_rhomin%04d_rhomax%04d%s" \
             % (gridPostfix, int(rhoRng[0] * 100 + 0.1),
                int(rhoRng[-1] * 100 + 0.1), simPostfix)
for irho in np.arange(rhoRng.shape[0]):
    rho = rhoRng[irho]
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
    (eigValForward,) \
        = ergoPlot.readSpectrum(eigValForwardFile,
                                fileFormat=cfg.general.fileFormat)

    # Get generator eigenvalues (using the complex logarithm)
    eigValGen = np.log(eigValForward) / tau
    isort = np.argsort(-eigValGen.real)
    eigValGen = eigValGen[isort]

    # Record spectral gap
    eigValRng[irho] = eigValGen[:cfg.spectrum.nev]


# Plot
nevPlot = 3
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
ax.set_xlabel(r'$\rho$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Re(\lambda_i)$', fontsize=ergoPlot.fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
plt.savefig('%s/spectrum/eigVal/eigVSRho%s.%s'\
            % (cfg.general.plotDir, dstPostfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

fig = plt.figure()
ax = fig.add_subplot(111)
for ev in np.arange(nevPlot):
    ax.plot(rhoRng[selRho], eigValRng[selRho, ev].real,
            linestyle=linestyles[ev%d], linewidth=lw)
ax.set_xlim(rhoRng[selRho][0], rhoRng[selRho][-1])
ax.set_ylim(-0.06, 0.002)
ax.set_xticks(np.arange(21, 25.1, 1))
ax.set_xlabel(r'$\rho$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\Re(\lambda_i)$', fontsize=ergoPlot.fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
plt.savefig('%s/spectrum/eigVal/eigVSRhoZoom%s.%s'\
            % (cfg.general.plotDir, dstPostfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

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
