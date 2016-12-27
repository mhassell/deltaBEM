# a script to test the Laplace potentials and operators
# last modified: December 27, 2016

import geometry
import laplace
import numpy as np

N = 10000

g  = geometry.kite(N,0)
gp = geometry.kite(N,1.0/6)
gm = geometry.kite(N,1.0/6)

# Laplacian has a kernel on bounded domains,
# use Lagrange multipliers to keep everything
# uniquely solvable
BIOs = laplace.CalderonCalculusLaplace(g,gp,gm)

V = np.ones((N+1,N+1))
V[0:N,0:N]=BIOs[0]
V[-1,-1] = 0

# harmonic polynomial for the solution
f = lambda x,y : x**2 - y**2
RHS = np.zeros((N+1,1))
RHS[0:N,:] = f(g['midpt'][:,0], g['midpt'][:,1])[:,np.newaxis]


# observation points in the domain
obs = np.array([[0.5, 0], [1., 0.5], [0, -1.25]])

POTs = laplace.LaplacePotentials(g,obs)

S = POTs[0]

# discrete solution
Lambda = np.linalg.solve(V, RHS)

uh = np.dot(S,Lambda[0:N,0])
print uh

uexact = f(obs[:,0],obs[:,1])
print uexact

err = np.max(np.abs(uexact - uh))

print err



