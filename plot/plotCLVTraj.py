import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylibconfig2
import ergoPlot

def field(x, p):
    rho, sigma, beta = p
    f = np.array([sigma * (x[1] - x[0]),
                  x[0] * (rho - x[2]) - x[1],
                  x[0] * x[1] - beta * x[2]])
    return f

def getAnglesCLVs(CLV):
    # Calculate angles between CLVs (1, 2), (1, 3) and (2, 3)
    ntSamp, dim, dim = CLV.shape
    angle = np.empty((ntSamp, dim, dim))
    norm = np.empty((ntSamp, dim))
    for d in np.arange(dim):
        norm[:, d] = np.sqrt(np.sum(CLV[:, :, d] * CLV[:, :, d], 1))
    for d in np.arange(dim):
        for dd in np.arange(dim):
            angle[:, d, dd] = np.arccos(np.sum(CLV[:, :, d] \
                                               * CLV[:, :, dd], 1) \
                                        / (norm[:, d] * norm[:, dd]))
    return (angle, norm)

configFile = '../cfg/Lorenz63.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)
fileFormat = cfg.general.fileFormat

dim = cfg.model.dim
spinup = cfg.simulation.spinup
L = cfg.simulation.LCut + spinup
dt = cfg.simulation.dt
printStep = cfg.simulation.printStep
ntSamp = int(cfg.simulation.LCut / printStep + 0.1)
time = np.arange(0., cfg.simulation.LCut, printStep)
printStepNum = int(printStep / dt + 0.1)
caseName = cfg.model.caseName

delayName = ""
if (hasattr(cfg.model, 'delaysDays')):
    for d in np.arange(len(cfg.model.delaysDays)):
        delayName = "%s_d%d" % (delayName, cfg.model.delaysDays[d])

resDir = '../results/'
contDir = '%s/continuation' % resDir
plotDir = '%s/plot/' % resDir

nWin = 10
LWin = 100
win = int(100 / printStep)
timeWin = time[nWin*win:(nWin+1)*win]
iobs = 0
labelsAngle = [r'$\theta_{+0}$', r'$\theta_{+-}$', r'$\theta_{0-}$']
labelsObs = [r'$x$', r'$y$', r'$z$']

rho = cfg.model.rho
caseNameModel = "%s_rho%04d" \
                % (caseName, int(rho * 100 + 0.1));
srcPostfixModel = "_%s%s" % (caseNameModel, delayName)
dstPostfix = "%s_L%d_spinup%d_dt%d_samp%d" \
             % (srcPostfixModel, int(L), int(spinup),
                int(round(-log10(dt)) + 0.1), int(printStepNum))

# Read Lyapunov exponents, stretching rates and CLVs
LyapExpFile = "%s/CLV/LyapExp%s.%s" \
              % (resDir, dstPostfix, cfg.general.fileFormat)
stretchRateFile = "%s/CLV/stretchRate%s.%s" \
                  % (resDir, dstPostfix, cfg.general.fileFormat)
CLVFile = "%s/CLV/CLV%s.%s" \
          % (resDir, dstPostfix, cfg.general.fileFormat)
tsFile = "%s/CLV/ts%s.%s" \
         % (resDir, dstPostfix, cfg.general.fileFormat)
if fileFormat == 'bin':
    readFile = np.fromfile
else:
    readFile = np.loadtxt
print 'Reading Lyapunov exponents...'
LyapExp = readFile(LyapExpFile)
print 'Reading covariant Lyapunov vectors...'
CLV = readFile(CLVFile)
print 'Reading time series...'
ts = readFile(tsFile)
# # Reshape
CLV = CLV.reshape(ntSamp, dim, dim)
ts = ts.reshape(ntSamp, dim)

# Angle with flow
print 'Calculating angle with flow...'
angleFlow = np.empty((ntSamp,))
for k in np.arange(ntSamp):
    vec = CLV[k, :, 1]
    vecf = field(ts[k], (rho, cfg.model.sigma, cfg.model.beta))
    angleFlow[k] = np.arccos(np.dot(vec, vecf) \
                             / (np.sqrt(np.dot(vec, vec)) \
                                * np.sqrt(np.dot(vecf, vecf))))

# Plot
scale = 8
msize = 3
sampPlot = 10
LTraj = 3.
ntTraj = int(LTraj / printStep)
xlim = (-20., 20.)
ylim = (-30., 30.)
zlim = (0., 50.)
tsWin = ts[nWin*win:(nWin+1)*win]
for k in np.arange(0, win/10, sampPlot):
    print 'Plotting figure %d of %d...' % (k/sampPlot, win/10/sampPlot - 1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    k0 = np.max([0, (k+1)-ntTraj])
    ax.plot(tsWin[k0:(k+1), 0], tsWin[k0:(k+1), 1], tsWin[k0:(k+1), 2],
            '-k', linewidth=1)
    ax.plot([tsWin[k, 0]], [tsWin[k, 1]], [tsWin[k, 2]], 'ok',
            markeredgecolor='k', markersize=msize)
    ax.plot([tsWin[k0, 0]], [tsWin[k0, 1]], [tsWin[k0, 2]], 'sk',
            markeredgecolor='k', markersize=msize)
    for d in np.arange(dim):
        if LyapExp[d] > 1.e-6:
            label = '+'
        elif LyapExp[d] < -1.e-6:
            label = '-'
        else:
            label = '0'        
        ax.plot([tsWin[k, 0], tsWin[k, 0] + scale*CLV[k, 0, d]],
                [tsWin[k, 1], tsWin[k, 1] + scale*CLV[k, 1, d]],
                [tsWin[k, 2], tsWin[k, 2] + scale*CLV[k, 2, d]],
                linestyle='-', linewidth=2, label=r'$v^{%s}$' % label)
    ax.set_xlabel(r'$x$', fontsize=ergoPlot.fs_latex)
    ax.set_ylabel(r'$y$', fontsize=ergoPlot.fs_latex)
    ax.set_zlabel(r'$z$', fontsize=ergoPlot.fs_latex)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    plt.setp(ax.get_zticklabels(), fontsize=ergoPlot.fs_xticklabels)
    fig.savefig('%s/CLV/traj/trajCLV_k%06d%s.eps' % (plotDir, k, dstPostfix),
                dpi=300, bbox_inches='tight')
