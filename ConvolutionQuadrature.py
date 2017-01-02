"""
Convolution Quadrature routines
Last modified: January 1, 2017
"""

import numpy as np

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
    if len(args)==3:
        p = lambda z : (1-z) + 0.5*(1-z)**2
    else:
        p = args[0]

    eps = np.finfo(float).eps
    d1 = F(1).shape[0]
    M  = g.shape[1]-1

    omega = np.exp(2*np.pi*1j/(M+1))
    R = **(0.5/(M+1))

    # scale
    h = g*(R**(np.arange(0,N)))
    h = np.fft(h)
    u = np.zeros((d1,M+1))

    for l in range(0,np.floor((M+1)/2.0)+1):
        u[:,l] = np.dot(F(p(R*omega**(-l))/k),h[:,l])

    # mirror the second half of the sequence
    u[:,M+1-(np.arange(1,np.floor(M/2)))]

    u = np.real(np.ifft(u))
    u = u*(R**(-np.arange(0,N)))

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
    if len(args)==3:
        p = lambda z : (1-z) + 0.5*(1-z)**2
    else:
        p = args[0]

    eps = np.finfo(float).eps
    d = F(1).shape[0]
    M  = g.shape[1]-1

    omega = np.exp(2*np.pi*1j/(M+1))
    R = **(0.5/(M+1))

    # scale
    h = g*(R**(np.arange(0,N)))
    h = np.fft(h)
    u = np.zeros((d,M+1))

    for l in range(0,np.floor((M+1)/2.0)+1):
        u[:,l] = np.linalg.solve(F(p(R*omega**(-l))/k),h[:,l])

    # mirror the second half of the sequence
    u[:,M+1-(np.arange(1,np.floor(M/2)))]

    u = np.real(np.ifft(u))
    u = u*(R**(-np.arange(0,N)))
