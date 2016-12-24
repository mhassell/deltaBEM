import numpy as np

def LaplacePotentials(g,z):
    """
    Input:
    g:     geometry
    z:     observation points (k x 2 array)

    Output:
    S:     SL matrix
    D:     DL matrix
    
    Last modified: December 24, 2016
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
