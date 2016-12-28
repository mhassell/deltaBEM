# function for testing right-hand-sides for BIEs
# Last modified: December 27, 2016

import numpy as np
import CalderonCalculusMatrices as CCM

def test(u, gradu, gp, gm, fork=0):
    """
    Input:
    u:     function handle of two variables
    gradu: function handle of two variables
    gp,gm: companion geometries
    fork:  averaged or mixed method

    Output:
    beta0:     testing of dirichlet BCs
    beta1:     testing of neumann BCs
    """
    #print u
    #print gradu
    
    beta0p = u(gp['midpt'][:,0],gp['midpt'][:,1])
    
    ux = gradu[0]
    uy = gradu[1]

    uxp = ux(gp['midpt'][:,0],gp['midpt'][:,1])
    uyp = uy(gp['midpt'][:,0],gp['midpt'][:,1])

    beta1p = uxp*gp['normal'][:,0] + uyp*gp['normal'][:,1]

    beta0m = u(gm['midpt'][:,0],gm['midpt'][:,1])

    uxm = ux(gm['midpt'][:,0],gm['midpt'][:,1])
    uym = uy(gm['midpt'][:,0],gm['midpt'][:,1])

    beta1m = uxm*gm['normal'][:,0] + uym*gm['normal'][:,1]

    MATS = CCM.CalderonCalculusMatrices(gp,fork)

    Q  = MATS[0]
    Pp = MATS[2]
    Pm = MATS[3]

    beta0 = np.dot(Pp,beta0p)+np.dot(Pm,beta0m)
    beta1 = Q.dot(np.dot(Pp,beta1p)+np.dot(Pm,beta1m))

    return (beta0, beta1)

    
