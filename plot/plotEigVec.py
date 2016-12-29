import os
import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
import ergoPlot

#ergoPlot.dpi = 2000

def sphere2Cart(x, p):
    rho, sigma, beta = p
    r, theta, phi = x

    r *= rho + sigma

    x[0] = r * np.sin(theta) * np.cos(phi)
    x[1] = r * np.sin(theta) * np.sin(phi);
    x[2] = r * np.cos(theta) + rho + sigma;

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
rho = cfg.model.rho
sigma = cfg.model.sigma
beta = cfg.model.beta
p = (rho, sigma, beta)
dim = cfg.model.dim
dimObs = dim

xmineigVal = -cfg.stat.rateMax
ymineigVal = -cfg.stat.angFreqMax
xlimEig = [xmineigVal, -xmineigVal/100]
ylimEig = [ymineigVal, -ymineigVal]
xticks = None
yticksPos = np.arange(0, ylimEig[1], 5.)
yticksNeg = np.arange(0, ylimEig[0], -5.)[::-1]
yticks = np.concatenate((yticksNeg, yticksPos))


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
srcPostfixSim = "%s_rho%04d_L%d_dt%d_nTraj%d" \
                % (gridPostfix, int(rho * 100 + 0.1), int(tau * 1000 + 0.1),
                   -np.round(np.log10(cfg.simulation.dt)), cfg.sprinkle.nTraj)

# Read grid
gridFile = '%s/grid/grid%s.txt' % (cfg.general.resDir, gridPostfix)
coord = ergoPlot.readGrid(gridFile, dimObs)
X, Y, Z = np.meshgrid(coord[0], coord[1], coord[2], indexing='ij')
coord = np.empty((N, dim))
coord[:, 0] = X.flatten()
coord[:, 1] = Y.flatten()
coord[:, 2] = Z.flatten()

# Convert spherical coordinates to Cartesian coordinates
for k in np.arange(N):
    sphere2Cart(coord[k], p)

# Define file names
postfix = "%s" % (srcPostfixSim,)
eigValForwardFile = '%s/eigval/eigvalForward_nev%d%s.%s' \
                    % (cfg.general.specDir, cfg.spectrum.nev, postfix,
                       cfg.general.fileFormat)
eigVecForwardFile = '%s/eigvec/eigvecForward_nev%d%s.%s' \
                    % (cfg.general.specDir, cfg.spectrum.nev, postfix,
                    cfg.general.fileFormat)

# Read transfer operator spectrum from file and create a bi-orthonormal basis
# of eigenvectors and backward eigenvectors:
print 'Readig spectrum for tau = %.3f...' % tau
(eigValForward, eigVecForward) \
    = ergoPlot.readSpectrum(eigValForwardFile, eigVecForwardFile,
                            fileFormat=cfg.general.fileFormat)

# Get generator eigenvalues (using the complex logarithm)
eigValGen = np.log(eigValForward) / tau

realLabel = r'$\Re(\lambda_k)$'
imagLabel = r'$\Im(\lambda_k)$'

ergoPlot.plotEig(eigValGen, xlabel=realLabel, ylabel=imagLabel,
                 xlim=xlimEig, ylim=ylimEig)
plt.savefig('%s/spectrum/eigVal/eigVal%s.%s'\
            % (cfg.general.plotDir, postfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)


# Plot eigenvectors of transfer operator
alpha = 0.0
os.system('mkdir %s/spectrum/eigvec 2> /dev/null' % cfg.general.plotDir)
for ev in np.arange(cfg.spectrum.nEigVecPlot):
    print 'Plotting real part of eigenvector %d...' % (ev + 1,)
    ergoPlot.plot3D(coord[0], coord[1], coord[2], eigVecForward[ev].real,
                    ev_xlabel, ev_ylabel, ev_zlabel, alpha)
    dstFile = '%s/spectrum/eigvec/eigvecForwardReal_ev%03d%s.%s' \
              % (cfg.general.plotDir, ev + 1, postfix, ergoPlot.figFormat)
    plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
    
    if cfg.spectrum.plotImag & (eigValForward[ev].imag != 0):
        print 'Plotting imaginary  part of eigenvector %d...' % (ev + 1,)
        ergoPlot.plot3D(coord[0], coord[1], coord[2], eigVecForward[ev].imag,
                        ev_xlabel, ev_ylabel, ev_zlabel, alpha)
        dstFile = '%s/spectrum/eigvec/eigvecForwardImag_ev%03d%s.%s' \
                  % (cfg.general.plotDir, ev + 1, postfix, ergoPlot.figFormat)
        plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches,
                    dpi=ergoPlot.dpi)
