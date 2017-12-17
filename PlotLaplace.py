# a script to test the Laplace potentials and operators
# last modified: December 15, 2017

import geometry
import laplace
import numpy as np
import CalderonCalculusTest as CCT
import CalderonCalculusMatrices as CCM
import scipy.sparse.linalg
import sys
import matplotlib.pyplot as plt

N = int(sys.argv[1])

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

# postprocessing
x = np.linspace(-4,4,100)
y = np.linspace(-4,4,100)

X,Y = np.meshgrid(x,y)

obs = np.array([X.flatten(), Y.flatten()]).T

POTs = laplace.LaplacePotentials(g,obs)

S = POTs[0]

# First experiment
eta = np.linalg.solve(V, beta0)
eta = eta[0:N,]
uh = np.dot(S,eta)

uh = uh.reshape(N,N)

fig = plt.figure()
ax = plt.axes()
cont = plt.pcolormesh(X,Y,uh)
plt.show()
