# a script to test the Laplace potentials and operators
# last modified: December 24, 2016

import geometry
# maybe pull all the Laplace ops and pots into one module
# construct each op as a separate function?  This will save computation
# time if we only need one op or pot and not all of them.
# It also makes the naming less awful
import CalderonCalculusLaplace
import LaplacePotentials

N = 100

g  = geometry.kite(N,0)
gp = geometry.kite(N,1.0/6)
gm = geometry.kite(N,1.0/6)

# Laplacian has a kernel on bounded domains
# use Lagrange multipliers to keep everything solvable

BIOs = CalderonCalculusLaplace(g,gp,gm)


V = np.append([[BIOs[0],np.ones(N,1)]
                 [np.ones(1,N), 0]])

# harmonic polynomial for the solution
f = lambda x,y : x**2 - y**2 
RHS = f(g['midpt'][:,0], g['midpt'][:,1])

# observation points in the domain

obs = np.array([])

Lambda = np.linalg.solve(SLOp, RHS)



