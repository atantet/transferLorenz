# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import pylibconfig2

# figFormat = 'png'

# def field(x, p):
#     rho, sigma, beta = p
#     f = np.array([sigma * (x[1] - x[0]),
#                   x[0] * (rho - x[2]) - x[1],
#                   x[0] * x[1] - beta * x[2]])
#     return f

# def Jacobian(x, p):
#     rho, sigma, beta = p
#     J = np.array([[-sigma, sigma, 0.],
#                   [rho - x[2], -1., -x[0]],
#                   [x[1], x[0], -beta]])
#     return J

# def propagateRK4(x0, field, p, dt, nt, samp=1):
#     '''Propagate solution of ODE according to the vector field field \
#     with Euler scheme from x0 for nt time steps of size dt.'''
#     xtRec = np.empty((nt / samp + 1, x0.shape[0]))
#     xt = x0.copy()
#     xtRec[0] = x0.copy()
#     for t in np.arange(1, nt):
#         # Step solution forward
#         k1 = field(xt, p) * dt
#         tmp = k1 * 0.5 + xt

#         k2 = field(tmp, p) * dt
#         tmp = k2 * 0.5 + xt

#         k3 = field(tmp, p) * dt
#         tmp = k3 + xt

#         k4 = field(tmp, p) * dt
#         tmp = (k1 + 2*k2 + 2*k3 + k4) / 6
        
#         xt += tmp

#         if (t % samp == 0):
#             xtRec[t/samp] = xt

#     return xtRec

# fs_latex = 'xx-large'
# fs_xticklabels = 'large'
# fs_yticklabels = fs_xticklabels

# configFile = '../cfg/Lorenz63.cfg'
# cfg = pylibconfig2.Config()
# cfg.read_file(configFile)

# dim = cfg.model.dim
# L = cfg.simulation.LCut + cfg.simulation.spinup
# printStepNum = int(cfg.simulation.printStep / cfg.simulation.dt + 0.1)
# caseName = cfg.model.caseName

# delayName = ""
# if (hasattr(cfg.model, 'delaysDays')):
#     for d in np.arange(len(cfg.model.delaysDays)):
#         delayName = "%s_d%d" % (delayName, cfg.model.delaysDays[d])

# # List of continuations to plot
# initCont = [7.956126, 7.956126, 24.737477, 24.5, 0.652822]
# contStep = 0.01
# dt = 1.e-5

# srcPostfix = "_%s%s" % (caseName, delayName)
# resDir = '../results/'
# contDir = '%s/continuation' % resDir
# plotDir = '%s/plot/' % resDir

# contAbs = np.sqrt(contStep*contStep)
# sign = contStep / contAbs
# exp = np.log10(contAbs)
# mantis = sign * np.exp(np.log(contAbs) / exp)
# dstPostfix = "%s_cont%04d_contStep%de%d_dt%d_numShoot%d" \
#              % (srcPostfix, int(initCont[dim] * 1000 + 0.1), int(mantis*1.01),
#                 (int(exp*1.01)), -np.round(np.log10(dt)),
#                 cfg.continuation.numShoot)
# poFileName = '%s/poCont%s.txt' % (contDir, dstPostfix)
# FloquetExpFileName = '%s/poExpCont%s.txt' % (contDir, dstPostfix)

# # Read fixed point and cont
# state = np.loadtxt(poFileName).reshape(-1, dim+2)
# # Read FloquetExpenvalues
# FloquetExp = np.loadtxt(FloquetExpFileName)
# FloquetExp = (FloquetExp[:, 0] + 1j * FloquetExp[:, 1]).reshape(-1, dim)
# # Remove last
# state = state[:-1]
# FloquetExp = FloquetExp[:-1]

# po = state[:, :dim]
# TRng = state[:, dim+1]
# contRng = state[:, dim]


# # Reorder Floquet exp
# for t in np.arange(1, contRng.shape[0]):
#     tmp = FloquetExp[t].tolist()
#     for exp in np.arange(dim):
#         idx = np.argmin(np.abs(tmp - FloquetExp[t-1, exp]))
#         FloquetExp[t, exp] = tmp[idx]
#         tmp.pop(idx)

# isStable = np.max(FloquetExp.real, 1) < 1.e-6

# #cont = 24.1
# #cont = 24.4
# cont = 24.8
# dt = 1.e-3
# samp = 20

# t = np.argmin((cont - contRng)**2)
# cont = contRng[t]
# T = TRng[t] * 1000000
# print 'Propagating attractor orbit for ', T, ' at rho = ', cont, \
#     ' from x(0) = ', po[t]

# nt = int(np.ceil(T / dt))
# # propagate
# p = [cont, cfg.model.sigma, cfg.model.beta]
# xtA = propagateRK4(po[t] + 5, field, p, dt, nt, samp=samp)
# xtA = xtA[nt / samp / 10:-1]

nBlock = 100
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Lb = xtA.shape[0] / nBlock

for b in np.arange(nBlock):
    ax.plot(xtA[b*Lb:(b+1)*Lb, 0], xtA[b*Lb:(b+1)*Lb, 1], xtA[b*Lb:(b+1)*Lb, 2], '-b')

#T = TRng[t]
#print 'Propagating orbit of period ', T, ' at cont = ', cont, \
#    ' from x(0) = ', po[t]
#nt = int(np.ceil(T / dt))
## propagate
#p = [cont, cfg.model.sigma, cfg.model.beta]
#xt = propagateRK4(po[t], field, p, dt, nt)
#xt = xt[:-1]

ax.plot(xt[:, 0], xt[:, 1], xt[:, 2], '-k', linewidth=2)


plt.savefig('%s/continuation/chaoticOrbit_rho%04d%s.%s' \
            % (plotDir, cont*1000, dstPostfix, figFormat), dpi=300, bbox_inches='tight')
        
