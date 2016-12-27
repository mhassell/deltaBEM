# various deltaBEM geometries
# last modified: December 27, 2016

import numpy as np

def kite(N, ep):
    """
    Input:
    N      Number of space intervals
    ep     epsilon parameter

    Output:
    g      discrete sampled geometry for a kite domain
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
    g['normal']= h*np.array([g['xp'][:,1], -g['xp'][:,0]]).T
    g['next'] = np.append([range(1,N)],0)
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
    """

    h = 1.0/N
    t = h*np.linspace(0,N-1,N)
    t.reshape(N,1)
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
    g['normal'] = h*np.array([g['xp'][:,1], -g['xp'][:,0]]).T
    g['next'] = np.append([range(1,N)],0)
    g['comp'] = np.array([1])

    return g

def tvshape(N,ep):
    """
    Input:
    N     Number of space intervals
    ep    epsilon parameter

    Output:
    g     sampling of a smoothed square
    """

    h = 1.0/N
    t = h*np.linspace(0,N-1,N)
    t = t + ep*h
    tau = t - 0.5*h

    g = {}

    g['midpt'] = np.array([(1+np.cos(2*np.pi*t)**2)*np.cos(2*np.pi*t),
                           (1+np.sin(2*np.pi*t)**2)*np.sin(2*np.pi*t)]).T
    g['brkpt'] = np.array([(1+np.cos(2*np.pi*tau)**2)*np.cos(2*np.pi*tau),
                           (1+np.sin(2*np.pi*tau)**2)*np.sin(2*np.pi*tau)]).T
    R = np.array([[np.cos(np.pi/4), np.sin(np.pi/4)],
                 [-np.sin(np.pi/4), np.cos(np.pi/4)]])
    g['midpt'] = np.dot(g['midpt'],R)
    g['brkpt'] = np.dot(g['brkpt'],R)

    g['xp'] = np.array([6*np.pi*np.sin(2*np.pi*t)**3 - 8*np.pi*np.sin(2*np.pi*t),
                   2*np.pi*(4*np.cos(2*np.pi*t)- 3*np.cos(2*np.pi*t)**3)]).T
    g['xp'] = np.dot(g['xp'], R)
    g['normal'] = h*np.array([g['xp'][:,1], -g['xp'][:,0]]).T
    g['next'] = np.append([range(1,N)],0)
    g['comp'] = np.array([1])

    return g

def ellipse(N,ep,R,c):
    """
    Input:
    N     : number of intervals
    ep    : epsilon parameter
    R     : [a,b] semiaxes  (np array)
    c     : [cx, cy] center (np array)

    Output:
    g     : discrete geometry
    """

    h = 1.0/N
    t = h*np.linspace(0,N-1,N)
    t = t + ep*h
    cost = np.cos(2*np.pi*t)
    sint = np.sin(2*np.pi*t)
    costau = np.cos(2*np.pi*(t-0.5*h))
    sintau = np.sin(2*np.pi*(t-0.5*h))

    g['midpt'] = np.array([c[0]+R[0]*cost,
                           c[1]+R[1]*sint]).T
    g['brkpt'] = np.array([c[0]+R[0]*costau,
                           c[1]+R[1]*sintau]).T
    g['xp']    = np.array([-R[0]*2*np.pi*sint,
                           R[1]*2*np.pi*cost])
    g['normal']= h*np.array([g['xp'][:,1], -g['xp'][:,0]]).T
    g['next'] = np.append([range(1,N)],0)
    g['comp'] = np.array([1])

    return g
