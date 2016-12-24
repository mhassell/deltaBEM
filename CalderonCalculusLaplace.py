import numpy as np

def _CalderonCalculusLaplaceHalf(g,gp):
    """
    Input:
    g:     principle geometry
    gp:    companion geometry

    Output:
    V:     single layer operator
    K:     double layer operator
    J:     tranposed double layer operator
    W:     hypersingular operator
     // To be added C:     rank Ncomp perturbation

    last modified: December 24, 2016
    """
    # V - SL operator

    DX = gp['midpt'][:,0].T - g['midpt'][:,0]
    DY = gp['midpt'][:,1].T - g['midpt'][:,1]
    D = np.sqrt(DX**2 + DY**2)
    V = -1.0/(2*np.pi)*np.log(D)

    # K - DL op
    N = DX*g['normal'][:,0].T + DY*g['normal'][:,1].T
    K = 1.0/(2*np.pi)*N/(D**2)

    # J - transposed DL op
    N = gp['normal'][:,0].T*DX + gp['normal'][:,1].T*DY
    J = -1.0/(2*np.pi)*N/D**2

    # W - hypersingular op
    DX = gp['brkpt'][:,0] - g['brkpt'][:,0].T
    DY = gp['brkpt'][:,1] - g['brkpt'][:,1].T
    D = np.sqrt(DX**2 + DY**2)
    W = -1.0/(2*np.pi)*(np.log(D[gp['next'],g['next'])) + np.log(D)
                        -np.log(D(gp['next'],:)) - np.log(D(:,g['next'])))

    return (V,K,J,W)

def CalderonCalculusLaplace(g,gp,gm):
    """
    """
    pass
