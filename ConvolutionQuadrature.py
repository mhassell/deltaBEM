"""
Convolution Quadrature routines
Last modified: January 29 2022
"""

import numpy as np
import config

if config.dask_flag:
    import dask
    from dask import delayed
else:
    pass
    
bdf2 = lambda z : (1-z) + 0.5*(1-z)**2

build_matrix = lambda F,g,kappa,p,l,R,omega : F(p(R*omega**(-l))/kappa)

if config.dask_flag:
    @delayed
    def build_and_mult(F,g,kappa,p,l,R,omega,b):
        A = build_matrix(F,g,kappa,p,l,R,omega)
        return np.dot(A,b)
      
    @delayed  
    def build_and_solve(F,g,kappa,p,l,R,omega,b):
        A = build_matrix(F,g,kappa,p,l,R,omega)
        return np.linalg.solve(A, b)
else: 
    def mult(A,b):
        return np.dot(A,b)
      
    def solve(A,b):
        return np.linalg.solve(A, b)

def CQforward(F, g, kappa, p=bdf2):
    """
    Input:
    F:     lambda of the complex parameter s, (d1 x d2)
    g:     RHS array (d2 x M+1)
    kappa: Time step size
    p:     (optional) lambda representing a time stepper (default is bdf 2)

    Output:
    h:     the convolution F(\partial_kappa)g (d1 x M+1)
    """

    eps = np.finfo(float).eps
    d1 = F(1).shape[0]
    M  = g.shape[1]-1
    
    omega = np.exp(2*np.pi*1j/(M+1))
    R = eps**(0.5/(M+1))

    # scale
    h = g*(R**(np.arange(0,M+1)))
    h = np.fft.fft(h)
    u = np.zeros((d1,M+1))+1j*np.zeros((d1,M+1))
      
    results = []   
    for l in range(0,M+1):
        if config.dask_flag:
            results.append(build_and_mult(F,g,kappa,p,l,R,omega,h[:,l]))
        else:
            MAT = F(p(R*omega**(-l))/kappa)
            if len(MAT.shape)==1:
                MAT = MAT[:,np.newaxis]
                
            u[:,l] = np.dot(MAT,h[:,l])
    
    if config.dask_flag:
        u = dask.compute(*results)
        u = np.array(u)
        u = u.T
    
    u = np.real(np.fft.ifft(u))
    u = u*(R**(-np.arange(0,M+1)))

    return u

def CQequation(F,g,kappa,p=bdf2):
    """
    Input:
    F:     lambda of the complex parameter s, (d x d)
    g:     RHS array (d x M+1)
    kappa: Time step size
    p:     (optional) lambda representing a time stepper (default is bdf 2)

    Output:
    h:     the convolution F(\partial_kappa)^-1 g (d x M+1)
    """

    eps = np.finfo(float).eps
    d = F(1).shape[0]
    M  = g.shape[1]-1
    
    omega = np.exp(2*np.pi*1j/(M+1))
    R = eps**(0.5/(M+1))

    # scale
    h = g*(R**(np.arange(0,M+1)))
    h = np.fft.fft(h)
    u = np.zeros((d,M+1))+1j*np.zeros((d,M+1))
    
    results = []
    for l in range(0,M+1):
        
        if config.dask_flag:
            results.append(build_and_solve(F,g,kappa,p,l,R,omega,h[:,l]))
        else:
            u[:,l] = np.linalg.solve(F(p(R*omega**(-l))/kappa),h[:,l])
        
    if config.dask_flag:
        u = dask.compute(*results)
        u = np.array(u)
        u = u.T

    u = np.real(np.fft.ifft(u))
    u = u*(R**(-np.arange(0,M+1)))

    return u
