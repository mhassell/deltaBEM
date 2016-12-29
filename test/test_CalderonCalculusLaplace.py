# a script to test the Laplace potentials and operators
# last modified: December 29, 2016

import geometry
import laplace
import numpy as np
import CalderonCalculusTest as CCT
import CalderonCalculusMatrices as CCM
import scipy.sparse.linalg
import sys

N = int(sys.argv[1])

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
K = BIOs[1]
J = BIOs[2]
W = BIOs[3]
C = BIOs[4]
V[-1,-1] = 0

MATS = CCM.CalderonCalculusMatrices(g)

Q = MATS[0]
M = MATS[1]

# exact solution
f = lambda x,y : x/(x**2+y**2)
fx = lambda x,y : 1./(x**2+y**2) - 2*x**2/(x**2+y**2)**2
fy = lambda x,y : -2.*x*y/(x**2+y**2)**2

# testing the RHS
RHS = CCT.test(f,[fx,fy],gp,gm)
beta0 = np.zeros((N+1,))
beta1 = np.zeros((N,))

beta0[0:N] = RHS[0]
beta1 = RHS[1]

# observation points outside the domain
obs = np.array([[0.,4.],[4.,0.],[-4.,2.],[2.,-4.]])

POTs = laplace.LaplacePotentials(g,obs)

S = POTs[0]
D = POTs[1]

# First experiment
eta = np.linalg.solve(V, beta0)
eta = eta[0:N,]
uh = np.dot(S,eta)

phi = scipy.sparse.linalg.spsolve(M,beta0[0:N,])
phi = Q.dot(phi)

Lambda = -0.5*eta + scipy.sparse.linalg.spsolve(M,np.dot(J,eta))

# exact solution
uexact = f(obs[:,0],obs[:,1])
beta0exact = f(g['midpt'][:,0],g['midpt'][:,1])
beta1exact = (fx(g['midpt'][:,0],g['midpt'][:,1])*g['normal'][:,0]
              +fy(g['midpt'][:,0],g['midpt'][:,1])*g['normal'][:,1])

erru = np.max(np.abs(uexact - uh))
errLambda = np.max(np.abs(beta1exact-Lambda))*N
errPhi    = np.max(np.abs(beta0exact-phi))

errors = np.array([erru, errPhi, errLambda])

# second experiment
psi = -np.linalg.solve((W+C),beta1)
phi = 0.5*psi+scipy.sparse.linalg.spsolve(M,np.dot(K,psi))
phi = Q.dot(phi)
Lambda = scipy.sparse.linalg.spsolve(M,beta1)
psi = Q.dot(psi)
uh = np.dot(D,psi)

erru = np.max(np.abs(uexact - uh))
errLambda = np.max(np.abs(beta1exact-Lambda))*N
errPhi    = np.max(np.abs(beta0exact-phi))

errors = np.array([errors,[erru, errLambda, errPhi]])

print errors



