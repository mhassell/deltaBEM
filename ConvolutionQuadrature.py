"""
Convolution Quadrature routines
Last modified: January 2, 2017
"""

import numpy as np
import sys
#from joblib import Parallel, delayed
#import multiprocessing

def CQforward(F,g,kappa):
    """
    Input:
    F:     lambda of the complex parameter s, (d1 x d2)
    g:     RHS array (d2 x M+1)
    kappa: Time step size
    p:     (optional) lambda representing a time stepper (default is bdf 2)

    Output:
    h:     the convolution F(\partial_kappa)g (d1 x M+1)
    """
    # need to figure out varargin!
    p = lambda z : (1-z) + 0.5*(1-z)**2

    eps = np.finfo(float).eps
    d1 = F(1).shape[0]
    M  = g.shape[1]-1

    omega = np.exp(2*np.pi*1j/(M+1))
    R = eps**(0.5/(M+1))

    # scale
    h = g*(R**(np.arange(0,M+1)))
    h = np.fft.fft(h)
    u = np.zeros((d1,M+1))+1j*np.zeros((d1,M+1))

    """
    # set up parallel stuff
    num_cores = multiprocessing.cpu_count()-1

    def mult(l):
        MAT = F(p(R*omega**(-l))/kappa)
        if len(MAT.shape)==1:
            MAT = MAT[:,np.newaxis] 
        return np.dot(MAT,h[:,l])

    u = Parallel(n_jobs=num_cores)(delayed(mult)(l) for l in range(M+1))
    """
    
    for l in range(0,M+1):
        MAT = F(p(R*omega**(-l))/kappa)
        if len(MAT.shape)==1:
            MAT = MAT[:,np.newaxis]
            
        u[:,l] = np.dot(MAT,h[:,l])
    
    u = np.real(np.fft.ifft(u))
    u = u*(R**(-np.arange(0,M+1)))

    return u

def CQequation(F,g,kappa):
    """
    Input:
    F:     lambda of the complex parameter s, (d x d)
    g:     RHS array (d x M+1)
    kappa: Time step size
    p:     (optional) lambda representing a time stepper (default is bdf 2)

    Output:
    h:     the convolution F(\partial_kappa)^-1 g (d x M+1)
    """

    p = lambda z : (1-z) + 0.5*(1-z)**2

    eps = np.finfo(float).eps
    d = F(1).shape[0]
    M  = g.shape[1]-1

    omega = np.exp(2*np.pi*1j/(M+1))
    R = eps**(0.5/(M+1))

    # scale
    h = g*(R**(np.arange(0,M+1)))
    h = np.fft.fft(h)
    u = np.zeros((d,M+1))+1j*np.zeros((d,M+1))

    """
    # set up parallel stuff
    num_cores = multiprocessing.cpu_count()-1
    def solve(l):
        return np.linalg.solve(F(p(R*omega**(-l))/kappa),h[:,l])
    
    u = Parallel(n_jobs=num_cores)(delayed(solve)(l) for l in range(M+1))
    """
    
    for l in range(0,M+1):
        u[:,l] = np.linalg.solve(F(p(R*omega**(-l))/kappa),h[:,l])

    u = np.real(np.fft.ifft(u))
    u = u*(R**(-np.arange(0,M+1)))

    return u
