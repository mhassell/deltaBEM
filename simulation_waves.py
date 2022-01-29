# make a time domain simulation of a scattering problem
# last modified: January 29, 2022

import numpy as np
import ConvolutionQuadrature as CQ
import geometry
import helmholtz
import waves
import CalderonCalculusTest as CCT
import sys
import matplotlib.pyplot as plt
from matplotlib import animation

if __name__ == "__main__":
    try:
        import dask
        from dask.distributed import Client
        
        client = Client(n_workers=4)
    except: 
        pass

    # command line arguments
    N = int(sys.argv[1])
    M = int(sys.argv[2])

    # geometry
    g = geometry.kite(N,0)
    gp = geometry.kite(N,1./6)
    gm = geometry.kite(N,-1./6)

    # signal
    T = 10
    kappa = float(T)/M
    direction = np.array([1,0])
    tlag = 2
    signal = lambda t: np.sin(2*t)**9*(t>=0)
    signalp = lambda t: 18*np.sin(2*t)**8*np.cos(2*t)*(t>=0)
    uincp = waves.planewave(signal,signalp,direction,tlag,gp['midpt'],gp['normal'],T,M,0)
    uincm = waves.planewave(signal,signalp,direction,tlag,gm['midpt'],gm['normal'],T,M,0)
    beta0 = -(0.5*uincp+0.5*uincm)

    MATS = helmholtz.CalderonCalculusHelmholtz(g,gp,gm)
    V = MATS[0]

    eta = CQ.CQequation(V,beta0,kappa)

    # postprocessing
    x = np.linspace(-4,4,100)
    y = np.linspace(-4,4,100)

    X,Y = np.meshgrid(x,y)

    obs = np.array([X.flatten(), Y.flatten()]).T

    POTS = helmholtz.HelmholtzPotentials(g,obs)

    S = POTS[0]

    uscat = CQ.CQforward(S,eta,kappa)

    uscat = uscat.reshape(100,100,M+1)
    uinc = waves.planewave(signal,signalp,direction,tlag,obs,[],T,M,0)
    uinc = uinc.reshape(100,100,M+1)

    utotal = uinc + uscat

    fig = plt.figure()
    ax = plt.axes()

    def animate(i):
        ax.plot(g['midpt'][:,0], g['midpt'][:,1],'k', linewidth=2)
        z = utotal[:,:,i]
        cont = plt.pcolormesh(X,Y,z)
        #cont = plt.contourf(X,Y,z)
        return cont

    anim = animation.FuncAnimation(fig,animate,frames=M,repeat=False)
    plt.show()
