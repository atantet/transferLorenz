# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import pylibconfig2
# import ergoPlot

# #ergoPlot.dpi = 2000

# def sphere2Cart(x, p):
#     rho, sigma, beta = p
#     r, theta, phi = x

#     r *= rho + sigma

#     x[0] = r * np.sin(theta) * np.cos(phi)
#     x[1] = r * np.sin(theta) * np.sin(phi);
#     x[2] = r * np.cos(theta) + rho + sigma;

# configFile = '../cfg/Lorenz63.cfg'
# compName1 = 'x'
# compName2 = 'y'
# compName3 = 'z'

# cfg = pylibconfig2.Config()
# cfg.read_file(configFile)

# L = cfg.simulation.LCut + cfg.simulation.spinup
# tau = cfg.transfer.tauRng[0]
# printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt + 0.1)
# caseName = cfg.model.caseName
# rho = cfg.model.rho
# sigma = cfg.model.sigma
# beta = cfg.model.beta
# p = (rho, sigma, beta)
# dim = cfg.model.dim
# dimObs = dim

# N = np.prod(np.array(cfg.grid.nx))
# gridPostfix = ""
# for d in np.arange(dimObs):
#     if (hasattr(cfg.grid, 'nSTDLow') & hasattr(cfg.grid, 'nSTDHigh')):
#         gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, cfg.grid.nx[d],
#                                         cfg.grid.nSTDLow[d],
#                                         cfg.grid.nSTDHigh[d])
#     else:
#         gridPostfix = "%s_n%dl%dh%d" % (gridPostfix, cfg.grid.nx[d],
#                                         cfg.sprinkle.minInitState[d],
#                                         cfg.sprinkle.maxInitState[d])
# gridPostfix = "_%s%s" % (caseName, gridPostfix)
# srcPostfixSim = "%s_rho%04d_L%d_dt%d_nTraj%d" \
#                 % (gridPostfix, int(rho * 100 + 0.1), int(tau * 1000 + 0.1),
#                    -np.round(np.log10(cfg.simulation.dt)), cfg.sprinkle.nTraj)

# # Read grid
# print 'Reading spherical grid...'
# gridFile = '%s/grid/grid%s.txt' % (cfg.general.resDir, gridPostfix)
# coord = ergoPlot.readGrid(gridFile, dimObs)
# X, Y, Z = np.meshgrid(coord[0], coord[1], coord[2], indexing='ij')
# coord = np.empty((dim, N))
# coord[0] = X.flatten()
# coord[1] = Y.flatten()
# coord[2] = Z.flatten()

# # Convert spherical coordinates to Cartesian coordinates
# print 'Converting to Cartesian coordinates...'
# for k in np.arange(N):
#     sphere2Cart(coord[:, k], p)

# coord_full = coord.copy()
# coord = coord[:, ::100]

print 'Plotting grid...'
scattersize=1
os.system('mkdir %s/grid/ 2> /dev/null' % cfg.general.plotDir)
ergoPlot.plot3D(coord[0], coord[1], coord[2],
                xlabel=compName1, ylabel=compName2, zlabel=compName3,
                scattersize=scattersize)
dstFile = '%s/grid/grid%s.%s' \
          % (cfg.general.plotDir, gridPostfix, ergoPlot.figFormat)
plt.savefig(dstFile, bbox_inches=ergoPlot.bbox_inches, dpi=ergoPlot.dpi)
