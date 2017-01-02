"""
A module to generate time domain readings of cylindrical or plane waves
Last modified: January 1, 2017
"""

import numpy as np
import scipy.special
import ConvolutionQuadrature as CQ

def planewave(f,fp,d,tlag,x,normal,T,M,flag):
    """
    Input:
    f:      lambda of one variable
    fp:     derivative of f
    d:      1x2 unit vector (direction of travel)
    tlag:   lag time
    x:      observation points (N x 2)
    normal: normal vectors (N x 2)
    T:      final time
    M:      number of time steps
    flag:   compute only the wave (0) or it's normal deriv too (1)

    Output:
    u:      planewave (N x M+1)
    un:     normal derivative 
    """

    t = np.linspace(0,T,M+1)
    direction = x[:,0]*d[0]+x[:,1]*d[1]
    u = f((t-tlag)-direction[:,np.newaxis])

    if flag:
        nor = normal[:,0]*d[0]+normal[:,1]*d[1]
        dnu = fp((t-tlag)-direction[:,np.newaxis])
        dnu = -dnu*nor[:,np.newaxis]
        return (u,dnu)
    
    return u
    
def cylindricalwave(f,src,obs,normal,T,M,flag):
    """
    Input:
    f:     lambda of one variable
    src:   1 x 2 array of source point
    obs:   N x 2 array of obs points
    normal: N x 2 array of vectors for normal obs
    T:     Final time
    M:     number of time steps
    flag:  0 compute only the wave, 1 compute wave and normal deriv
    
    Output:
    u:     cylindrical wave observed on obs (N x M+1 array)
    dnu:   normal deriv of u (N x M+1 array)
    """

    N = obs.shape[0]
    t = np.linspace(0,T,M+1)
    signal = f(t)
    signal = signal[np.newaxis,:]
    diff = obs - src[np.newaxis,:]
    dist = np.sqrt(diff[:,0]**2 + diff[:,1]**2)
    kernel = lambda s: 1j/4*scipy.special.hankel1(0,1j*s*dist)
    u = CQ.CQforward(kernel,signal,float(T)/M)

    if flag:
        dipole = np.sum(diff*normal,1)/dist
        kernel = lambda s: s/4*scipy.special.hankel1(1,1j*s*dist)*dipole
        dnu = CQ.CQforward(kernel,f,float(T)/M)
        return u, dnu

    return u
