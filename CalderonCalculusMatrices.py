# Quadrature, mass, and mixing matrices
# Last modified: December 27, 2016

import numpy as np
import scipy.sparse as sp

def CalderonCalculusMatrices(g,fork=0):
    
    """
    Input:
    g:     Geometry
    fork:  (optional)

    Output:
    Q:     Quadrature matrix
    M:     Mass Matrix
    Pp,Pm: Mixing matrices

    last modified: December 26, 2016
    """
    
    
    if fork:
        alpha = 5.0/6
    else:
        alpha = 1.0

    N = g['midpt'].shape[0]

    indicesZeroToN = np.linspace(0,N-1,N)

    # quadrature matrix
    Q = (sp.csr_matrix((22.0/24*np.ones(N),
                       (indicesZeroToN,indicesZeroToN)),shape=(N,N)) + 
    sp.csr_matrix((1./24*np.ones(N),(g['next'],indicesZeroToN)),shape=(N,N)) + 
    sp.csr_matrix((1./24*np.ones(N),(indicesZeroToN,g['next'])),shape=(N,N)))

    # mass matrix
    M = (sp.csr_matrix(((4.0+12*alpha)/18*np.ones(N),
                       (indicesZeroToN,indicesZeroToN)),shape=(N,N))+
    sp.csr_matrix(((7.0-6*alpha)/18*np.ones(N),
                   (g['next'],indicesZeroToN)),shape=(N,N))+
    sp.csr_matrix(((7.0-6*alpha)/18*np.ones(N),
                   (indicesZeroToN,g['next'])),shape=(N,N)))

    # mixing/averaging matrices
    if fork:
        Pp = (sp.csr_matrix((alpha/2*np.ones(N),
                             (indicesZeroToN,indicesZeroToN)),shape=(N,N))+
        sp.csr_matrix(((1-alpha)/2*np.ones(N), (g['next'],indicesZeroToN)),shape=(N,N)))
        Pm = Pp.T

    else:
        Pp = 0.5
        Pm = 0.5

    return(Q,M,Pp,Pm)
    
    

    
    



