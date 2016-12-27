# operators and potentials for Laplace's equation
# last modified: December 26, 2016

import numpy as np

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
    DX = z[:,0] - g['midpt'][:,0].T
    DY = z[:,1] - g['midpt'][:,1].T
    D = np.sqrt(DX**2 + DY**2)
    S = -1.0/(2*np.pi)*np.log(D)

    # DL Pot
    N = DX*g['normal'][:,0].T + DY*g['normal'][:,1].T
    DL = 1.0/(2*np.pi)*N/(D**2)

    return (S, D)

def _CalderonCalculusLaplaceHalf(g,gp):
    """
    Internal function to split up the computation on plus and minus geometries
    
    Input:
    g:     principle geometry
    gp:    companion geometry

    Output:
    V:     single layer operator
    K:     double layer operator
    J:     tranposed double layer operator
    W:     hypersingular operator
     // To be added C:     rank Ncomp perturbation
    """
    # V - SL operator

    DX = gp['midpt'][np.newaxis,0].T - g['midpt'][:,0]
    DY = gp['midpt'][np.newaxis,1].T - g['midpt'][:,1]
    D = np.sqrt(DX**2 + DY**2)
    V = -1.0/(2*np.pi)*np.log(D)

    # K - DL op
    N = DX*g['normal'][np.newaxis,0].T + DY*g['normal'][:,1].T
    K = 1.0/(2*np.pi)*N/(D**2)

    # J - transposed DL op
    N = gp['normal'][np.newaxis,0].T*DX + gp['normal'][np.newaxis,1].T*DY
    J = -1.0/(2*np.pi)*N/D**2

    # W - hypersingular op
    DX = gp['brkpt'][np.newaxis,0] - g['brkpt'][np.newaxis,0].T
    DY = gp['brkpt'][np.newaxis,1] - g['brkpt'][np.newaxis,1].T
    D = np.sqrt(DX**2 + DY**2)
    W = -1.0/(2*np.pi)*(np.log(D[gp['next'],g['next'])) + np.log(D)
                        -np.log(D(gp['next'],:)) - np.log(D(:,g['next'])))

    return (V,K,J,W)

def CalderonCalculusLaplace(g,gp,gm):
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
     // To be added C:     rank Ncomp perturbation
    """

    LP = _CalderonCalculusLaplaceHalf(g,gp)
    LM = _CalderonCalculusLaplaceHalf(g,gm)

    
