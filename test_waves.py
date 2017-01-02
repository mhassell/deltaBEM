# a script to test planewave and cylindrical wave
# last modified: January 2, 2017

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import waves
import sys
import time

N = int(sys.argv[1])
M = int(sys.argv[2])
opt = int(sys.argv[3])  # 0 for planewave, 1 for cylindrical wave

T = 4
if opt==0: 
    # planewave params
    f = lambda t: np.sin(2*np.pi*t)*(t>=0)
    fp = lambda t: 2*np.pi*np.cos(2*np.pi*t)*(t>=0)
    d = np.array([1,1])/np.sqrt(2)
    tlag = 0.5
    obsgrid = np.linspace(0,3,N)
    xv, yv = np.meshgrid(obsgrid,obsgrid)
    obs = np.array([xv.reshape(N**2),yv.reshape(N**2)]).T

    pw = waves.planewave(f,fp,d,tlag,obs,[],T,M,0)
    pw = pw.reshape(N,N,M+1)

    fig = plt.figure()
    ax = plt.axes()

    def animate(i):
        z = pw[:,:,i]
        #cont = plt.pcolormesh(xv,yv,z)
        cont = plt.contourf(xv,yv,z)
        return cont

    anim = animation.FuncAnimation(fig,animate,frames=M)
    plt.show()

if opt==1:
    # cylindrical wave params
    f = lambda t: np.sin(2*np.pi*t)*(t>=0)
    src = np.array([3.5, 3.5])
    obsgrid = np.linspace(0,3,N)
    xv, yv = np.meshgrid(obsgrid,obsgrid)
    obs = np.array([xv.reshape(N**2),yv.reshape(N**2)]).T
    cw = waves.cylindricalwave(f,src,obs,[],T,M,0)
    cw = cw.reshape(N,N,M+1)
    
    fig = plt.figure()
    ax = plt.axes()

    def animate(i):
        z = cw[:,:,i]
        # cont = plt.pcolormesh(xv,yv,z)
        cont = plt.contourf(xv,yv,z)
        print i
        return cont

    anim = animation.FuncAnimation(fig,animate,frames=M)
    plt.show()
    
