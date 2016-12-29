# operators and potentials for Laplace's equation
# last modified: December 29, 2016

import numpy as np
import CalderonCalculusMatrices as CCP
import scipy.sparse

def LaplacePotentials(g,z):
    """
    Input:
    g:     geometry
    z:     observation points (k x 2 array)

    Output:
    S:     SL matrix
    D:     DL matrix
    """

    # SL Pot 
    DX = z[:,0][:,np.newaxis] - g['midpt'][:,0]
    DY = z[:,1][:,np.newaxis] - g['midpt'][:,1]
    D = np.sqrt(DX**2 + DY**2)
    SL = -1.0/(2*np.pi)*np.log(D)

    # DL Pot
    N = DX*g['normal'][:,0][np.newaxis,:] + DY*g['normal'][:,1][np.newaxis,:]
    DL = (1.0/(2*np.pi))*N/(D*D)

    return (SL,DL)

def _CalderonCalculusLaplaceHalf(g,gp):
    """
    Private function to split up the computation on plus and minus geometries
    
    Input:
    g:     principle geometry
    gp:    companion geometry

    Output:
    V:     single layer operator
    K:     double layer operator
    J:     tranposed double layer operator
    W:     hypersingular operator
    C:     rank Ncomp perturbation
    """
    # V - SL operator

    DX = gp['midpt'][:,0][:,np.newaxis] - g['midpt'][:,0]
    DY = gp['midpt'][:,1][:,np.newaxis] - g['midpt'][:,1]
    D = np.sqrt(DX**2 + DY**2)
    V = -1.0/(2*np.pi)*np.log(D)

    # K - DL op
    
    N = DX*g['normal'][:,0][np.newaxis,:] + DY*g['normal'][:,1][np.newaxis,:]
    K = 1.0/(2*np.pi)*N/(D**2)

    # J - transposed DL op
    N = gp['normal'][:,0][:,np.newaxis]*DX + gp['normal'][:,1][:,np.newaxis]*DY
    J = -1.0/(2*np.pi)*N/D**2

    # W - hypersingular op
    DX = gp['brkpt'][:,0][:,np.newaxis] - g['brkpt'][:,0]
    DY = gp['brkpt'][:,1][:,np.newaxis] - g['brkpt'][:,1]
    D = np.sqrt(DX**2 + DY**2)

    N = g['midpt'].shape[0]
    
    W = -1.0/(2*np.pi)*(np.log(D[np.ix_(gp['next'],g['next'])]) + np.log(D)
                        - np.log(D[np.ix_(gp['next'],np.arange(0,N))])
                        - np.log(D[np.ix_(np.arange(0,N),g['next'])]))

    # C perturbation matrix
    lengths = np.sum(g['normal']**2,1)
    lengthsp = np.sum(gp['normal']**2,1)

    C = np.zeros((N,N))
    if g['comp'].shape[0]>1:
        print "Not yet implemented for multiple components"
    else:
        C = lengthsp[:,np.newaxis]*lengths[np.newaxis,:]

    return (V,K,J,W,C)

def CalderonCalculusLaplace(g,gp,gm,fork=0):
    """
    Input:
    g:     principle geometry
    gp:    plus geometry
    gm:    minus geometry

   Output:
    V:     single layer operator
    K:     double layer operator
    J:     tranposed double layer operator
    W:     hypersingular operator
    C:     rank Ncomp perturbation
    """

    Lp = _CalderonCalculusLaplaceHalf(g,gp)
    Lm = _CalderonCalculusLaplaceHalf(g,gm)

    MATS = CCP.CalderonCalculusMatrices(g,fork)

    # CC matrices
    Q  = MATS[0]
    Pp = MATS[2]
    Pm = MATS[3]

    # SL operator
    Vp = Lp[0]
    Vm = Lm[0]
    V = np.dot(Pp,Vp) + np.dot(Pm,Vm)

    # DL operator
    Kp = Lp[1]
    Km = Lm[1]
    # numpy not yet aware of sparse arrays (this wastes memory!!)
    tmp = np.dot(Pp,Kp)+np.dot(Pm,Km)
    K = (Q.T.dot(tmp.T)).T    

    # adjoint DL op
    Jp = Lp[2]
    Jm = Lm[2]
    tmp = np.dot(Pp,Kp)+np.dot(Pm,Km)  # FIX ME, I WASTE MEMORY!
    J = Q.dot(np.dot(Pp,Jp)+np.dot(Pm,Jm))

    # hypersingular op
    Wp = Lp[3]
    Wm = Lm[3]
    W = np.dot(Pp,Wp)+np.dot(Pm,Wm)

    # C
    Cp = Lp[4]
    Cm = Lm[4]
    C = (Q.T.dot((Q.dot(np.dot(Pp,Cp)+np.dot(Pm,Cm))).T)).T

    return (V,K,J,W,C)
    
