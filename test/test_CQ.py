# script to test the CQ methods
# last modified: January 2, 2017

import numpy as np
import ConvolutionQuadrature as CQ
import geometry
import helmholtz
import waves
import CalderonCalculusTest as CCT
import sys

# command line arguments
N = int(sys.argv[1])
M = int(sys.argv[2])

# geometry
g = geometry.kite(N,0)
gp = geometry.kite(N,1./6)
gm = geometry.kite(N,-1./6)

# signal
T = 5
kappa = float(T)/M
direction = np.array([1,1])/np.sqrt(2)
tlag = 1.5
signal = lambda t: np.sin(2*t)**9*(t>=0)
signalp = lambda t: 18*np.sin(2*t)**8*np.cos(2*t)*(t>=0)
uincp = waves.planewave(signal,signalp,direction,tlag,gp['midpt'],gp['normal'],T,M,0)
uincm = waves.planewave(signal,signalp,direction,tlag,gm['midpt'],gm['normal'],T,M,0)
beta0 = 0.5*uincp+0.5*uincm

MATS = helmholtz.CalderonCalculusHelmholtz(g,gp,gm)
V = MATS[0]

eta = CQ.CQequation(V,beta0,kappa)
obs = np.array([[0, -1.5],[1, 0],[1.5, 0.5]])

POTS = helmholtz.HelmholtzPotentials(g,obs)
S = POTS[0]

uh = CQ.CQforward(S,eta,kappa)

uexact = waves.planewave(signal,signalp,direction,tlag,obs,[],T,M,0)

error = np.max(np.abs(uh[:,M]-uexact[:,M]))

print(error)


