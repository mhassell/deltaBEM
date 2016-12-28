# a script to test the Laplace potentials and operators
# last modified: December 27, 2016

import geometry
import laplace
import numpy as np
import CalderonCalculusTest as CCT

N = 80

#g = geometry.kite(N,0)
#gp = geometry.kite(N,1./6)
#gm = geometry.kite(N,-1./6)

g  = geometry.ellipse(N,0,[1,1],[0,0])
gp = geometry.ellipse(N,1./6,[1,1],[0,0])
gm = geometry.ellipse(N,-1./6,[1,1],[0,0])

# Laplacian has a kernel, so we use Lagrange multipliers
# to keep everything uniquely solvable
BIOs = laplace.CalderonCalculusLaplace(g,gp,gm)

V = np.ones((N+1,N+1))
V[0:N,0:N]=BIOs[0]
V[-1,-1] = 0

# exact solution
f = lambda x,y : x/(x**2+y**2)
fx = lambda x,y : 1/(x**2+y**2) - 2*x**2/(x**2+y**2)**2
fy = lambda x,y : -2*x*y/(x**2+y**2)**2
gradf = lambda x,y : np.array([fx, fy]).T 

# testing the RHS
RHS = CCT.test(f,[fx,fy],gp,gm)
beta0 = np.zeros((N+1,))
beta1 = np.zeros((N+1,))

beta0[0:N] = RHS[0]
beta1[0:N] = RHS[1]

# observation points outside the domain
obs = np.array([[3.5, 0], [2., 2.5], [1, -2.25]])

POTs = laplace.LaplacePotentials(g,obs)

S = POTs[0]


# discrete solution
Lambda = np.linalg.solve(V, beta0)

uh = np.dot(S,Lambda[0:N,])
print uh

uexact = f(obs[:,0],obs[:,1])
print uexact

err = np.max(np.abs(uexact - uh))

print err



