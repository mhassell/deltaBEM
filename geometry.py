# various deltaBEM geometries
# last modified: December 20, 2016

import numpy as np

def kite(N, ep):
    """
    Input:
    N      Number of space intervals
    ep     epsilon parameter

    Output:
    g      discrete sampled geometry for a kite domain

    last modified: December 20, 2016
    """

    h = 1.0/N
    t = h*np.linspace(0,N-1,N)
    t = t + ep*h

    g = {}
    g['midpt'] = np.array([np.cos(2*np.pi*t)+np.cos(4*np.pi*t),
                        2*np.sin(2*np.pi*t)]).T
    g['brkpt'] = np.array([np.cos(2*np.pi*(t-0.5*h))+np.cos(4*np.pi*(t-0.5*h)),
                         2*np.sin(2*np.pi*(t-0.5*h))]).T
    g['xp']    = np.array([-2*np.pi*np.sin(2*np.pi*t)-4*np.pi*np.sin(4*np.pi*t),
                        4*np.pi*np.cos(2*np.pi*t)]).T
    g['normal']= h*np.array([g['xp'][:,1], -g['xp'][:,0]])
    g['next'] = np.append([np.linspace(2,N,N-1)],1)
    g['comp'] = np.array([1])

    return g
                        
