import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylibconfig2
import ergoPlot

def getAnglesCLVs(CLV, abs=False):
    # Calculate angles between CLVs (1, 2), (1, 3) and (2, 3)
    ntSamp, dim, dim = CLV.shape
    angle = np.zeros((ntSamp, dim, dim))
    norm = np.empty((ntSamp, dim))
    for d in np.arange(dim):
        norm[:, d] = np.sqrt(np.sum(CLV[:, :, d] * CLV[:, :, d], 1))
    for d in np.arange(dim):
        for dd in np.arange(d+1, dim):
            dotprod = np.sum(CLV[:, :, d] * CLV[:, :, dd], 1)
            if abs:
                dotprod = np.abs(dotprod)
            angle[:, d, dd] = np.arccos(dotprod / (norm[:, d] * norm[:, dd]))
                
    return (angle, norm)

configFile = '../cfg/Lorenz63.cfg'
cfg = pylibconfig2.Config()
cfg.read_file(configFile)
fileFormat = cfg.general.fileFormat
rhoRng = np.arange(24., 28.1, 1.)
#rhoRng = np.array([28.])

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
win = int(100 / printStep)
timeWin = time[nWin*win:(nWin+1)*win]
iobs = 0
labelsAngle = [r'$\theta_{+0}$', r'$\theta_{+-}$', r'$\theta_{0-}$']
labelsStretch = [r'$\xi_{+}$', r'$\xi_{0}$', r'$\xi_{-}$']
labelsObs = [r'$x$', r'$y$', r'$z$']
ticksAngle = np.arange(0., np.pi/2+0.1, np.pi/4)
ticklabelsAngle = [r'$0$', r'$\pi/4$', r'$\pi/2$']
nBins = 500

plotCLV = False
#plotCLV = True
plotStretch = True
LyapExpRng = np.empty((rhoRng.shape[0], dim))
meanAngleRng = np.empty((rhoRng.shape[0], dim, dim))
minAngleRng = np.empty((rhoRng.shape[0], dim, dim))
meanDist2SingRng = np.empty((rhoRng.shape[0],))
volAnaRng = np.empty((rhoRng.shape[0],))
for irho in np.arange(rhoRng.shape[0]):
    rho = rhoRng[irho]
    caseNameModel = "%s_rho%04d" \
                    % (caseName, int(rho * 100 + 0.1));
    srcPostfixModel = "_%s%s" % (caseNameModel, delayName)
    dstPostfix = "%s_L%d_spinup%d_dt%d_samp%d" \
                 % (srcPostfixModel, int(L), int(spinup),
                    int(round(-log10(dt)) + 0.1), int(printStepNum))

    # Get volume contraction rate analytically
    volAnaRng[irho] = -1. - cfg.model.sigma - cfg.model.beta

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
    if plotStretch:
        print 'Reading stretching rates...'
        stretchRate = readFile(stretchRateFile)
        stretchRate = stretchRate.reshape(ntSamp, dim)
    if plotCLV:
        print 'Reading covariant Lyapunov vectors...'
        CLV = readFile(CLVFile)
        print 'Reading time series...'
        ts = readFile(tsFile)
        CLV = CLV.reshape(ntSamp, dim, dim)
        ts = ts.reshape(ntSamp, dim)

        # Get angles
        print 'Calculating angles...'
        angle, norm = getAnglesCLVs(CLV, abs=True)
        meanAngleRng[irho] = np.mean(angle, 0)
        minAngleRng[irho] = np.min(angle, 0)
        dist2Sing = sqrt(np.sum(ts**2, 1))
        meanDist2SingRng[irho] = np.mean(dist2Sing)

    if plotStretch:
        bins = 500
        for d in np.arange(dim):
            # Plot histogram
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(stretchRate[:, d], bins, color='k', normed=True)
            ax.set_xlabel(labelsStretch[k], fontsize=ergoPlot.fs_latex)
            #ax.set_xlim(bins[0], bins[-1])
            #ax.set_ylim(0., ylimHist[k])
            #ax.set_xticks(ticksAngle)
            #ax.set_xticklabels(ticklabelsAngle)
            plt.setp(ax.get_xticklabels(),
                     fontsize=ergoPlot.fs_xticklabels)
            plt.setp(ax.get_yticklabels(),
                     fontsize=ergoPlot.fs_yticklabels)
            fig.savefig('%s/CLV/stretchHist/stretchHist%d%s.eps' \
                        % (plotDir, d, dstPostfix),
                        dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)


    if plotCLV:
        print 'Plotting angles...'
        angleWin = angle[nWin*win:(nWin+1)*win]
        tsWin = ts[nWin*win:(nWin+1)*win]
        dist2SingWin = dist2Sing[nWin*win:(nWin+1)*win]
        inLobePlus = (tsWin[:, iobs] > 0.)
        ylimAngle = [0., np.pi/2]
        ylimHist = [2., 2.1, 3.7]
        bins = np.linspace(ylimAngle[0], ylimAngle[1], nBins)
        ylimTS = np.array([np.min(tsWin[:, iobs]),
                           np.max(tsWin[:, iobs])])
        ylimDist = np.array([0., 50.])
        xlimTime = np.array([timeWin[0], timeWin[-1]])
        alphaTS = 0.75
        # Plot CLV angles time series
        k = 0
        for d in np.arange(dim):
            for dd in np.arange(d+1, dim):
                # Plot time series
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.fill_between(timeWin, ylimAngle[0], ylimAngle[1],
                                where=inLobePlus,
                                color=[alphaTS, 1., alphaTS])
                ax.fill_between(timeWin, ylimAngle[0], ylimAngle[1],
                                where=~inLobePlus,
                                color=[1., alphaTS, alphaTS])
                ax.plot(timeWin, angleWin[:, d, dd], linestyle='-', color='k',
                        linewidth=1)
                ax.set_ylabel(labelsAngle[k], fontsize=ergoPlot.fs_latex)
                ax.set_ylim(ylimAngle[0], ylimAngle[1])
                ax.set_yticks(ticksAngle)
                ax.set_yticklabels(ticklabelsAngle)
                ax.set_xlim(xlimTime[0], xlimTime[1])
                ax.set_xlabel(r't', fontsize=ergoPlot.fs_latex)
                plt.setp(ax.get_xticklabels(),
                         fontsize=ergoPlot.fs_xticklabels)
                plt.setp(ax.get_yticklabels(),
                         fontsize=ergoPlot.fs_yticklabels)
                fig.savefig('%s/CLV/angle/angle%d%d%s.eps' \
                            % (plotDir, d, dd, dstPostfix),
                            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

                # Plot histogram
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.hist(angle[:, d, dd], bins, color='k', normed=True)
                ax.set_xlabel(labelsAngle[k], fontsize=ergoPlot.fs_latex)
                ax.set_xlim(bins[0], bins[-1])
                ax.set_ylim(0., ylimHist[k])
                ax.set_xticks(ticksAngle)
                ax.set_xticklabels(ticklabelsAngle)
                plt.setp(ax.get_xticklabels(),
                         fontsize=ergoPlot.fs_xticklabels)
                plt.setp(ax.get_yticklabels(),
                         fontsize=ergoPlot.fs_yticklabels)
                fig.savefig('%s/CLV/angleHist/angleHist%d%d%s.eps' \
                            % (plotDir, d, dd, dstPostfix),
                            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)
                k +=1

        # Plot state time series
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.fill_between(timeWin, ylimTS[0], ylimTS[1],
                        where=inLobePlus,
                        color=[alphaTS, 1., alphaTS])
        ax.fill_between(timeWin, ylimTS[0], ylimTS[1],
                        where=~inLobePlus,
                        color=[1., alphaTS, alphaTS])
        ax.plot(timeWin, tsWin[:, iobs],
                linestyle='-', color='k', linewidth=1)
        ax.set_ylabel(labelsObs[iobs], fontsize=ergoPlot.fs_latex)
        ax.set_xlim(xlimTime[0], xlimTime[1])
        ax.set_ylim(ylimTS[0], ylimTS[1])
        ax.set_xlabel(r't', fontsize=ergoPlot.fs_latex)
        plt.setp(ax.get_xticklabels(),
                 fontsize=ergoPlot.fs_xticklabels)
        plt.setp(ax.get_yticklabels(),
                 fontsize=ergoPlot.fs_yticklabels)
        fig.savefig('%s/CLV/ts/ts%s.eps' % (plotDir, dstPostfix),
                    dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)
    
        # Plot distance to singularity
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.fill_between(timeWin, ylimDist[0], ylimDist[1],
                        where=inLobePlus,
                        color=[alphaTS, 1., alphaTS])
        ax.fill_between(timeWin, ylimDist[0], ylimDist[1],
                        where=~inLobePlus,
                        color=[1., alphaTS, alphaTS])
        ax.plot(timeWin, dist2SingWin, linestyle='-', color='k',
                linewidth=1)
        ax.set_ylabel(r'$\sqrt{x^2 + y^2 + z^2}$', fontsize=ergoPlot.fs_latex)
        ax.set_xlim(xlimTime[0], xlimTime[1])
        ax.set_ylim(ylimDist[0], ylimDist[1])
        ax.set_xlabel(r't', fontsize=ergoPlot.fs_latex)
        plt.setp(ax.get_xticklabels(),
                 fontsize=ergoPlot.fs_xticklabels)
        plt.setp(ax.get_yticklabels(),
                 fontsize=ergoPlot.fs_yticklabels)
        fig.savefig('%s/CLV/ts/tsDist2Sing%s.eps' % (plotDir, dstPostfix),
                    dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)
    
    # Record
    LyapExpRng[irho] = LyapExp    
    
# Plot Lyapunov exponents
print 'Plotting Lyapunov exponents...'
fig = plt.figure()
ax = fig.add_subplot(111)
labels = [r'$\lambda_+$', r'$\lambda_0$', r'$\lambda_-$']
fact = np.array([1., 1., 10.])
for d in np.arange(dim):
    ax.plot(rhoRng, LyapExpRng[:, d] / fact[d], linestyle='-', linewidth=2,
            label=labels[d])
ax.set_xticks(rhoRng)
ax.set_xlim(rhoRng[0], rhoRng[-1])
ax.legend(fontsize=ergoPlot.fs_latex, loc=(0.72, 0.1))
ax.set_xlabel(r'$\rho$', fontsize=ergoPlot.fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/CLV/LyapExp%s.eps' % (plotDir, dstPostfix),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# Plot volume
print 'Plotting volume...'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(rhoRng, LyapExpRng.sum(1), color='k', linestyle='-', linewidth=2,
        label=r'$\lambda_{+} + \lambda_{0} + \lambda_{-}$')
ax.plot(rhoRng, volAnaRng, color='k', linestyle='--', linewidth=2,
        label=r'$-1 -\sigma -\beta$')
ax.set_xticks(rhoRng)
ax.set_xlim(rhoRng[0], rhoRng[-1])
ax.set_ylim(-15., 0.)
ax.set_xlabel(r'$\rho$', fontsize=ergoPlot.fs_latex)
ax.set_ylabel(r'$\lambda$', fontsize=ergoPlot.fs_latex)
plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
fig.savefig('%s/CLV/vol/vol%s.eps' % (plotDir, dstPostfix),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

if plotCLV:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels = [r'$\theta_{+0}$', r'$\theta_{+-}$', r'$\theta_{0-}$']
    k = 0
    for d in np.arange(dim):
        for dd in np.arange(d+1, dim):
            ax.plot(rhoRng, meanAngleRng[:, d, dd], linestyle='-',
                    linewidth=2, label=(r'$\mathrm{mean}~$' + labels[k]))
            k += 1
    ax.legend(fontsize=ergoPlot.fs_latex)
    ax.set_xlim(rhoRng[0], rhoRng[-1])
    ax.set_xticks(rhoRng)
    ax.set_xlabel(r'$\rho$', fontsize=ergoPlot.fs_latex)
    #ax.set_ylabel(r'$\$', fontsize=ergoPlot.fs_latex)
    plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    fig.savefig('%s/CLV/meanAngle%s.eps' % (plotDir, dstPostfix),
                dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels = [r'$\theta_{+0}$', r'$\theta_{+-}$', r'$\theta_{0-}$']
    k = 0
    for d in np.arange(dim):
        for dd in np.arange(d+1, dim):
            ax.plot(rhoRng, minAngleRng[:, d, dd], linestyle='-', linewidth=2,
                    label=(r'$\mathrm{min}~$' + labels[k]))
            k += 1
    ax.set_xlim(rhoRng[0], rhoRng[-1])
    ax.legend(fontsize=ergoPlot.fs_latex)
    ax.set_xticks(rhoRng)
    ax.set_xlabel(r'$\rho$', fontsize=ergoPlot.fs_latex)
    plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    fig.savefig('%s/CLV/minAngle%s.eps' % (plotDir, dstPostfix),
                dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rhoRng, meanDist2SingRng, linestyle='-', linewidth=2)
    ax.set_xlim(rhoRng[0], rhoRng[-1])
    ax.legend(fontsize=ergoPlot.fs_latex)
    ax.set_xticks(rhoRng)
    ax.set_xlabel(r'$\rho$', fontsize=ergoPlot.fs_latex)
    ax.set_ylabel(r'$\mathrm{mean}~\sqrt{x^2 + y^2 + z^2}$',
                  fontsize=ergoPlot.fs_latex)
    plt.setp(ax.get_xticklabels(), fontsize=ergoPlot.fs_xticklabels)
    plt.setp(ax.get_yticklabels(), fontsize=ergoPlot.fs_yticklabels)
    fig.savefig('%s/CLV/meanDist2Sing%s.eps' % (plotDir, dstPostfix),
                dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)
