"""
A module to generate time domain readings of cylindrical or plane waves
Last modified: January 1, 2017
"""

import numpy as np

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
        nor = normal[:,0]*d[:,0]+normal[:,1]*d[:,1]
        dnu = fp((t-tlag)-direction[:,np.newaxis])
        dnu = -dnu*nor
        return (u,dnu)
    
    return u
    
def cylindricalwave():
    pass
