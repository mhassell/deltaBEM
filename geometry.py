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

def starshape(N,ep,r,rp):
    """
    Input:
    N     number of space intervals
    ep    epsilon parameter
    r     2-pi periodic radius function (lambda)
    rp    first derivative of r

    Output:
    g     discrete geometry

    last modified: December 21, 2016
    """

    h = 1.0/N
    t = h*np.linspace(0,N-1,N)
    t = t + ep*h
    t = 2*np.pi*t
    tau = 2*np.pi*(t-0.5*h)
    cost = np.cos(t)
    sint = np.sin(t)
    rt = r(t)
    rpt = rp(t)

    g = {}
    g['midpt'] = np.array([cost, sint])*rt
    g['brkpt'] = np.array([np.cos(tau), np.sin(tau)])*r(tau)
    g['xp']    = 2*np.pi*np.array([(rpt*cost - rt*sint),
                           rpt*sint+rt*cost])
    g['normal'] = h*np.array([g['xp'][:,1], -g['xp'][:,0]])
    g['next'] = np.append([np.linspace(2,N,N-1)],1)
    g['comp'] = np.array([1])

    return g

def tvshape(N,ep):
    """
    Input:
    N     Number of space intervals
    ep    epsilon parameter

    Output:
    g     sampling of a smoothed square

    last modified: December 21, 2016
    """

    h = 1.0/N
    t = h*np.linspace(0,N-1,N)
    t = t + ep*h
    tau = t - 0.5*h

    g['midpt'] = np.array([(1+np.cos(2*np.pi*t))**2*np.cos(2*np.pi*t),
                           (1+np.sin(2*np.pi*t))**2*np.sin(2*np.pi*t)]).T
    g['brkpt'] = np.array([(1+np.cos(2*np.pi*tau))**2*np.cos(2*np.pi*tau),
                           (1+np.sin(2*np.pi*tau))**2*np.sin(2*np.pi*tau)]).T
    R = np.array([[np.cos(np.pi/4), np.sin(np.pi/4)],
                 [-np.sin(np.pi/4), np.cos(np.pi/4)]])
    g['midpt'] = np.dot(g['midpt'],R)
    g['brkpt'] = np.dot(g['brkpt'],R)

    g['xp'] = np.array([6*np.pi*np.sin(2*np.pi*t)**3 - 8*np.pi*np.sin(2*np.pi*t),
                   2*np.pi*(4*np.cos(2*np.pi*t)- 3*np.cos(2*np.pi*t)**3)]).T
    g['xp'] = np.dot(g['xp'], R)
    g['normal'] = h*np.array([g['xp'][:,1], -g['xp'][:,0]])
    g['next'] = np.append([np.linspace(2,N,N-1)],1)
    g['comp'] = np.array([1])

    return g
                           
