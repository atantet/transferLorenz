import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pylibconfig2
import ergoPlot

configFile = '../cfg/Lorenz63.cfg'
compName = ['x', 'y', 'z']
cfg = pylibconfig2.Config()
cfg.read_file(configFile)

delayName = ""
if (hasattr(cfg.model, 'delaysDays')):
    for d in np.arange(len(cfg.model.delaysDays)):
        delayName = "%s_d%d" % (delayName, cfg.model.delaysDays[d])

#L = cfg.simulation.LCut + cfg.simulation.spinup
#dt = cfg.simulation.dt
#printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt + 0.1)
L = 101000
dt = 1.e-5
printStepNum = 1000
caseName = cfg.model.caseName
if (hasattr(cfg.model, 'rho') & hasattr(cfg.model, 'sigma') \
    & hasattr(cfg.model, 'beta')):
    caseName = "%s_rho%d_sigma%d_beta%d" \
               % (caseName, (int) (cfg.model.rho * 1000),
                  (int) (cfg.model.sigma * 1000),
                  (int) (cfg.model.beta * 1000))
srcPostfix = "_%s%s_L%d_spinup%d_dt%d_samp%d" \
             % (caseName, delayName, L, cfg.simulation.spinup,
                -np.round(np.log10(dt)), printStepNum)
sampFreq = 1. / cfg.simulation.printStep
lagMaxNum = int(np.round(cfg.stat.lagMax / cfg.simulation.printStep))
lags = np.arange(-cfg.stat.lagMax,
                 cfg.stat.lagMax + 0.999 * cfg.simulation.printStep,
                 cfg.simulation.printStep)
corrName = 'C%d%d' % (cfg.stat.idxf, cfg.stat.idxg)
corrLabel = r'$C_{%s, %s}(t)$' % (compName[cfg.stat.idxf],
                                      compName[cfg.stat.idxg])
powerName = 'S%d%d' % (cfg.stat.idxf, cfg.stat.idxg)
powerLabel = r'$S_{%s, %s}(\omega)$' % (compName[cfg.stat.idxf],
                                            compName[cfg.stat.idxg])

# Read ccf
print 'Reading correlation function and periodogram...'
ccf = np.loadtxt('%s/correlation/%s_lag%d%s.txt'\
                 % (cfg.general.resDir, corrName, int(cfg.stat.lagMax),
                    srcPostfix))
lags = np.loadtxt('%s/correlation/lags_lag%d%s.txt'\
                  % (cfg.general.resDir, int(cfg.stat.lagMax),
                     srcPostfix))
perio = np.loadtxt('%s/power/%s_chunk%d%s.txt'\
                   % (cfg.general.resDir, powerName, int(cfg.stat.chunkWidth),
                      srcPostfix))
perioSTD = np.loadtxt('%s/power/%sSTD_chunk%d%s.txt' \
                      % (cfg.general.resDir, powerName,
                         int(cfg.stat.chunkWidth), srcPostfix))
freq = np.loadtxt('%s/power/freq_chunk%d%s.txt' \
                  % (cfg.general.resDir, cfg.stat.chunkWidth,
                     srcPostfix))
        
# Plot CCF
print 'Plotting correlation function...'
os.system('mkdir -p %s/correlation 2> /dev/null' % cfg.general.plotDir)
os.system('mkdir -p %s/power 2> /dev/null' % cfg.general.plotDir)
(fig, ax) = ergoPlot.plotCCF(ccf, lags, ylabel=corrLabel, plotPositive=True)
plt.savefig('%s/plot/correlation/%s_lag%d%s.%s'\
           % (cfg.general.resDir, corrName, int(cfg.stat.lagMax),
              srcPostfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

# Plot perio
print 'Plotting periodogram...'
angFreq = freq * 2 * np.pi
(fig, ax) = ergoPlot.plotPerio(perio, perioSTD=perioSTD, freq=angFreq,
                               ylabel=powerLabel, plotPositive=True,
                               absUnit='', yscale='log',
                               xlim=(0, cfg.stat.angFreqMax),
                               ylim=(cfg.stat.powerMin, cfg.stat.powerMax))
plt.savefig('%s/plot/power/%s_chunk%d%s.%s'\
            % (cfg.general.resDir, powerName, int(cfg.stat.chunkWidth),
               srcPostfix, ergoPlot.figFormat),
            dpi=ergoPlot.dpi, bbox_inches=ergoPlot.bbox_inches)

