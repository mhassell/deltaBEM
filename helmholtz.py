import numpy as np
import scipy.special
import CalderonCalculusMatrices as CCM

"""
A module for the Helmholtz Calderon Calculus
Last modified: December 29, 2016
"""

def HelmholtzPotentials(g,obs):
    """
    Input:
    g:     principle geometry
    obs:   obs points (K x 2 array)

    Output:
    SL(s): lambda for the SL pot
    DL(s): lambda for the DL pot
    """

    # SL Pot
    RX = obs[:,0][:,np.newaxis] - g['midpt'][:,0]
    RY = obs[:,1][:,np.newaxis] - g['midpt'][:,1]
    R = np.sqrt(RX**2 + RY**2)
    SL = lambda s: 1j/4*scipy.special.hankel1(0,1j*s*R)

    # DL Pot
    RN = RX*g['normal'][:,0][np.newaxis,:] + RY*g['normal'][:,1][np.newaxis,:]
    RN = RN/R
    DL = lambda s: -s/4*scipy.special.hankel1(1,1j*s*R)*RN

    return (SL,DL)
    
def _CalderonCalculusHelmholtzHalf(g,gp):
    """
    Input:
    g:     principle geometry
    gp:    companion geometry

    Output:
    V:     SL op
    K:     DL op
    J:     transpose of K
    Wp:    Singular part of W
    Vn:    regular part of W
    """

    N = g['midpt'].shape[0]

    # SL op and regular part of W
    DX = gp['midpt'][:,0][:,np.newaxis] - g['midpt'][:,0]
    DY = gp['midpt'][:,1][:,np.newaxis] - g['midpt'][:,1]
    D = np.sqrt(DX**2 + DY**2)
    H = (gp['normal'][:,0]*g['normal'][:,0][np.newaxis,:].T+
         gp['normal'][:,1]*g['normal'][:,1][np.newaxis,:].T)
    V = lambda s: 1j/4*scipy.special.hankel1(0,1j*s*D)
    Vn = lambda s: s**2*H*1j/4*scipy.special.hankel1(0,1j*s*D)

    # singular part of dubyah
    DX = gp['brkpt'][:,0][:,np.newaxis] - g['brkpt'][:,0]
    DY = gp['brkpt'][:,1][:,np.newaxis] - g['brkpt'][:,1]
    D1 = np.sqrt(DX**2 + DY**2)
    D2 = D1[np.ix_(gp['next'],np.arange(0,N))]
    D3 = D1[np.ix_(np.arange(0,N),g['next'])]
    D4 = D1[np.ix_(gp['next'],g['next'])]
    Wp = (lambda s: 1j/4* scipy.special.hankel1(0,1j*s*D1)
          -1j/4*scipy.special.hankel1(0,1j*s*D2)
          -1j/4*scipy.special.hankel1(0,1j*s*D3)
          +1j/4*scipy.special.hankel1(0,1j*s*D4))

    # DL op
    DX = gp['midpt'][:,0][:,np.newaxis] - g['midpt'][:,0]
    DY = gp['midpt'][:,1][:,np.newaxis] - g['midpt'][:,1]
    D = np.sqrt(DX**2 + DY**2)
    N = DX*g['normal'][:,0][np.newaxis,:] + DY*g['normal'][:,1][np.newaxis,:]
    N = N/D
    K = lambda s: -s/4*scipy.special.hankel1(1,1j*s*D)*N

    # transposed DL op
    N = DX*gp['normal'][:,0][:,np.newaxis] + DY*gp['normal'][:,1][:,np.newaxis]
    N = N/D
    J = lambda s: s/4*scipy.special.hankel1(1,1j*s*D)*N

    return (V,K,J,Wp,Vn)

def CalderonCalculusHelmholtz(g,gp,gm,fork=0):
    """
    Input:
    g:     principle geometry
    gp,gm: companion geometries
    fork:  fork (1 or 0)

    Output:
    V:     SL op (lambda of the parameter s)
    K:     DL op (lambda of the parameter s)
    J:     transposed DL op (lambda of the parameter s)
    W:     hypersingular op (lambda of the parameter s)
    """

    Hp = _CalderonCalculusHelmholtzHalf(g,gp)
    Hm = _CalderonCalculusHelmholtzHalf(g,gm)

    Vp = Hp[0]
    Vm = Hm[0]

    Kp = Hp[1]
    Km = Hm[1]

    Jp = Hp[2]
    Jm = Hm[2]

    Wpp = Hp[3]
    Wpm = Hm[3]

    Vnp = Hp[4]
    Vnm = Hm[4]

    MATS = CCM.CalderonCalculusMatrices(g,fork)

    Q = MATS[0]
    Pp = MATS[2]
    Pm = MATS[3]

    V = lambda s: Pp*Vp(s) + Pm*Vm(s)
    K = lambda s: Q.T.dot((Pp*Kp(s) + Pm*Km(s)).T).T
    J = lambda s: Q.dot(Pp*Jp(s) + Pm*Jm(s))
    W = lambda s: Pp*Wpp(s) + Pm*Wpm(s) + Q.dot((Q.T.dot(Pp*Vnp(s) + Pm*Vnm(s)).T).T)

    return (V,K,J,W)
