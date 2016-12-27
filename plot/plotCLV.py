import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylibconfig2
import ergoPlot

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
caseNameModel = "%s_rho%d_sigma%d_beta%d" % (caseName,
                                             int(cfg.model.rho * 1000 + 0.1),
                                             int(cfg.model.sigma * 1000 + 0.1),
		                             int(cfg.model.beta * 1000 + 0.1));

delayName = ""
if (hasattr(cfg.model, 'delaysDays')):
    for d in np.arange(len(cfg.model.delaysDays)):
        delayName = "%s_d%d" % (delayName, cfg.model.delaysDays[d])

srcPostfixModel = "_%s%s" % (caseNameModel, delayName)
resDir = '../results/'
contDir = '%s/continuation' % resDir
plotDir = '%s/plot/' % resDir

dstPostfix = "%s_L%d_spinup%d_dt%d_samp%d" \
             % (srcPostfixModel, int(L), int(spinup),
                int(round(-log10(dt)) + 0.1), int(printStepNum))

# Read Lyapunov exponents, stretching rates and CLVs
LyapExpFile = "%s/CLV/LyapExp%s.%s" % (resDir, dstPostfix, cfg.general.fileFormat)
stretchRateFile = "%s/CLV/stretchRate%s.%s" \
                  % (resDir, dstPostfix, cfg.general.fileFormat)
CLVFile = "%s/CLV/CLV%s.%s" \
                  % (resDir, dstPostfix, cfg.general.fileFormat)
if fileFormat == 'bin':
    print 'Reading Lyapunov exponents...'
    LyapExp = np.fromfile(LyapExpFile)
    print 'Reading stretching rates...'
    stretchRate = np.fromfile(stretchRateFile)
    print 'Reading covariant Lyapunov vectors...'
    CLV = np.fromfile(CLVFile)
else:
    print 'Reading Lyapunov exponents...'
    LyapExp = np.loadtxt(LyapExpFile)
    print 'Reading stretching rates...'
    stretchRate = np.loadtxt(stretchRateFile)
    print 'Reading covariant Lyapunov vectors...'
    CLV = np.loadtxt(CLVFile)
# Reshape
stretchRate = stretchRate.reshape(ntSamp, dim)
CLV = CLV.reshape(ntSamp, dim, dim)


# Calculate angles between CLVs (1, 2), (1, 3) and (2, 3)
angle = np.empty((ntSamp, 3))
norm = np.empty((ntSamp, 3))
norm[:, 0] = np.sqrt(np.sum(CLV[:, :, 0] * CLV[:, :, 0], 1))
norm[:, 1] = np.sqrt(np.sum(CLV[:, :, 1] * CLV[:, :, 1], 1))
norm[:, 2] = np.sqrt(np.sum(CLV[:, :, 2] * CLV[:, :, 2], 1))
angle[:, 0] = np.arccos(np.sum(CLV[:, :, 0] * CLV[:, :, 1], 1) / (norm[:, 0] * norm[:, 1]))
angle[:, 1] = np.arccos(np.sum(CLV[:, :, 0] * CLV[:, :, 2], 1) / (norm[:, 0] * norm[:, 2]))
angle[:, 2] = np.arccos(np.sum(CLV[:, :, 1] * CLV[:, :, 2], 1) / (norm[:, 1] * norm[:, 2]))

# Calculate the determinant
detCLV = np.linalg.det(CLV)

nWin = 10
win = int(100 / printStep)
timeWin = time[nWin*win:(nWin+1)*win]

# Plot stretching rates
print 'Plotting stretching rates...'
fig = plt.figure(figsize=(8, 10))
colorsStretch = ['r', 'g', 'b']
LyapExpNames = ['+', '0', '-']
xminStretch = -10.
xmaxStretch = -xminStretch
ax = []
for d in np.arange(dim):
    ax.append(fig.add_subplot(310 + d + 1))
    ax[d].plot(timeWin, stretchRate[nWin*win:(nWin+1)*win, d],
            linestyle='-', color=colorsStretch[d], linewidth=1)
    ax[d].plot(timeWin, np.ones((win,)) * LyapExp[d],
            linestyle='--', color=colorsStretch[d], linewidth=2)
    ax[d].set_ylabel(r'$\xi_{%s}(t)$' % LyapExpNames[d], fontsize=ergoPlot.fs_latex)
    ax[d].set_xlim(timeWin[0], timeWin[-1])
    plt.setp(ax[d].get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
    plt.setp(ax[d].get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    ax[d].set_ylim(LyapExp[d] + xminStretch, LyapExp[d] + xmaxStretch)
ax[d].set_xlabel(r'$t$', fontsize=ergoPlot.fs_latex)
# ax.set_ylim(1.2, 2.8)
fig.savefig('%s/CLV/stretchRate%s.eps' % (plotDir, dstPostfix),
            dpi=300, bbox_inches='tight')
    
# Plot angles
print 'Plotting CLV angles...'
fig = plt.figure(figsize=(8, 10))
colorsAngle = ['r', 'g', 'b']
ax = []
angleNames = ['+0', '+-', '0-']
for d in np.arange(3):
    ax.append(fig.add_subplot(310 + d + 1))
    ax[d].plot(timeWin, angle[nWin*win:(nWin+1)*win, d],
               linestyle='-', color=colorsAngle[d], linewidth=1)
    ax[d].set_ylabel(r'$\theta_{%s}(t)$' % angleNames[d], fontsize=ergoPlot.fs_latex)
    ax[d].set_ylim(0, np.pi)
    plt.setp(ax[d].get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
    plt.setp(ax[d].get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    ax[d].set_xlim(timeWin[0], timeWin[-1])
    ax[d].grid('on')
# d = 3
# ax.append(fig.add_subplot(410 + d + 1))
# ax[d].plot(timeWin, detCLV[nWin*win:(nWin+1)*win],
#            '-k', linewidth=2)
# ax[d].set_ylabel(r'$\Delta(t)$', fontsize=ergoPlot.fs_latex)
# #ax[d].set_ylim(0, np.pi)
# plt.setp(ax[d].get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
# plt.setp(ax[d].get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
# ax[d].set_xlim(timeWin[0], timeWin[-1])
# ax[d].grid('on')
ax[d].set_xlabel(r'$t$', fontsize=ergoPlot.fs_latex)
fig.savefig('%s/CLV/angle%s.eps' % (plotDir, dstPostfix),
            dpi=300, bbox_inches='tight')
