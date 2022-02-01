# script to test the CQ methods
# last modified: January 2, 2017

import numpy as np
import ConvolutionQuadrature as CQ
import geometry
import helmholtz
import waves
import CalderonCalculusTest as CCT
import CalderonCalculusMatrices as CCM
import sys
from dask.distributed import Client

if __name__ == "__main__":

    client = Client(n_workers=4)

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
    uincp, graduincp = waves.planewave(signal,signalp,direction,tlag,gp['midpt'],gp['normal'],T,M,True)
    uincm, graduincm = waves.planewave(signal,signalp,direction,tlag,gm['midpt'],gm['normal'],T,M,True)
    beta0 = 0.5*uincp + 0.5*uincm

    MATS = helmholtz.CalderonCalculusHelmholtz(g,gp,gm)
    V = MATS[0]

    eta = CQ.CQequation(V,beta0,kappa)
    obs = np.array([[0, -1.5],[1, 0],[1.5, 0.5]])

    POTS = helmholtz.HelmholtzPotentials(g,obs)
    S = POTS[0]

    uh = CQ.CQforward(S,eta,kappa)

    uexact = waves.planewave(signal,signalp,direction,tlag,obs,[],T,M,False)

    error = np.max(np.abs(uh[:,M]-uexact[:,M]))

    print(f"Single Layer error: {error}")
    
    '''
    W = MATS[3]
    ccm = CCM.CalderonCalculusMatrices(g)
    Q = ccm[0]
    beta1 = Q @ (0.5*graduincp + 0.5*graduincm)
    eta = CQ.CQequation(W,-beta1,kappa)
    obs = np.array([[0, -1.5],[1, 0],[1.5, 0.5]])

    POTS = helmholtz.HelmholtzPotentials(g,obs)
    D = POTS[1]

    uh = CQ.CQforward(D, Q @ eta,kappa)

    error = np.max(np.abs(uh[:,M]-uexact[:,M]))

    print(f"Double Layer error: {error}")
    '''
