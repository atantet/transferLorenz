import os
import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
import ergoPlot

#ergoPlot.dpi = 2000

configFile = '../cfg/Lorenz63.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)
compNames = ('x', 'y', 'z')
realLabel = r'$\Re(\lambda_k)$'
imagLabel = r'$\Im(\lambda_k)$'
xmineigVal = -cfg.stat.rateMax
ymineigVal = -cfg.stat.angFreqMax
xlimEig = [xmineigVal, -xmineigVal/100]
ylimEig = [ymineigVal, -ymineigVal]
(ev_xlabel, ev_ylabel, ev_zlabel) = compNames


def sphere2Cart(x, p):
    rho, sigma, beta = p

    x[:, 0] *= rho + sigma

    x[:, 0] = x[:, 0] * np.sin(x[:, 1]) * np.cos(x[:, 2])
    x[:, 1] = x[:, 0] * np.sin(x[:, 1]) * np.sin(x[:, 2]);
    x[:, 2] = x[:, 0] * np.cos(x[:, 1]) + rho + sigma;

def sphere2CartXYZ(X, Y, Z, p):
    rho, sigma, beta = p
    r = X.copy()
    theta = Y.copy()
    phi = Z.copy()

    r *= rho + sigma

    X[:] = r * np.sin(theta) * np.cos(phi)
    Y[:] = r * np.sin(theta) * np.sin(phi);
    Z[:] = r * np.cos(theta) + rho + sigma;

L = cfg.simulation.LCut + cfg.simulation.spinup
spinup = cfg.simulation.spinup
tau = cfg.transfer.tauRng[0]
printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt + 0.1)
caseName = cfg.model.caseName
p = (cfg.model.rho, cfg.model.sigma, cfg.model.beta)
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
gridPostfix = '_%s%s' % (caseName, gridPostfix)
srcPostfixSim = "%s_rho%04d_L%d_dt%d_nTraj%d%s" \
                % (gridPostfix, int(p[0] * 100 + 0.1),
                   int(tau * 1000 + 0.1),
                   -np.round(np.log10(cfg.simulation.dt)),
                   cfg.sprinkle.nTraj, nProc)
postfix = "%s" % (srcPostfixSim,)

# Read grid
sub = 1
gridFile = '%s/grid/grid%s.txt' % (cfg.general.resDir, gridPostfix)
coord = ergoPlot.readGrid(gridFile, dimObs)
X, Y, Z = np.meshgrid(coord[0], coord[1], coord[2], indexing='ij')
# Convert spherical coordinates to Cartesian coordinates
sphere2CartXYZ(X, Y, Z, p)

# Define file names
eigValForwardFile = '%s/eigval/eigvalForward_nev%d%s.%s' \
                    % (cfg.general.specDir, cfg.spectrum.nev, postfix,
                       cfg.general.fileFormat)
eigVecForwardFile = '%s/eigvec/eigvecForward_nev%d%s.mm' \
                    % (cfg.general.specDir, cfg.spectrum.nev, postfix)

# Read transfer operator spectrum from file and create a bi-orthonormal basis
# of eigenvectors and backward eigenvectors:
print 'Readig spectrum for tau = %.3f...' % tau
(eigValForward, eigVecForward) \
    = ergoPlot.readSpectrumCompressed(eigValForwardFile,
                                      eigVecForwardFile=eigVecForwardFile)
nev = eigValForward.shape[0]
eigValGen = np.log(eigValForward) / tau

ergoPlot.plotEig(eigValGen, xlabel=realLabel, ylabel=imagLabel,
                 xlim=xlimEig, ylim=ylimEig)
# plt.savefig('%s/spectrum/eigVal/eigVal%s.%s'\
#             % (cfg.general.plotDir, postfix, ergoPlot.figFormat),
#             dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)


# Plot eigenvectors of transfer operator
alpha = 0.0
ss = 6
os.system('mkdir %s/spectrum/eigvec 2> /dev/null' % cfg.general.plotDir)
nevPlot = 3
eigVecForward[:, 0]  /= eigVecForward[:, 0].sum()
for ev in np.arange(nevPlot):
    vec = eigVecForward[:, ev].real.reshape(X.shape)
    print 'Plotting real part of eigenvector %d...' % (ev + 1,)
    ergoPlot.plot3D(X[::sub, ::sub, ::sub], Y[::sub, ::sub, ::sub],
                    Z[::sub, ::sub, ::sub], vectOrig=vec[::sub, ::sub, ::sub],
                    xlabel=ev_xlabel, ylabel=ev_ylabel, zlabel=ev_zlabel,
                    alpha=alpha, scattersize=ss)
    dstFile = '%s/spectrum/eigvec/eigvecForwardReal_ev%03d%s.%s' \
              % (cfg.general.plotDir, ev + 1, postfix, ergoPlot.figFormat)
    plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
