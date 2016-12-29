import helmholtz
import geometry
import sys
import CalderonCalculusTest as CCT
import CalderonCalculusMatrices as CCM
import numpy as np
import scipy.sparse.linalg
import scipy.special

N = int(sys.argv[1])

g = geometry.ellipse(N,0,[1,1],[0,0])
gp = geometry.ellipse(N,1./6,[1,1],[0,0])
gm = geometry.ellipse(N,-1./6,[1,1],[0,0])

s = -2j

# matrices and such
BIOs = helmholtz.CalderonCalculusHelmholtz(g,gp,gm)

V = BIOs[0]
K = BIOs[1]
J = BIOs[2]
W = BIOs[3]

MATS = CCM.CalderonCalculusMatrices(g)

Q = MATS[0]
M = MATS[1]

# exact solution
r = lambda x,y: np.sqrt(x**2+y**2)
rx = lambda x,y: x/np.sqrt(x**2+y**2)
ry = lambda x,y: y/np.sqrt(x**2+y**2)
u = lambda x,y: scipy.special.hankel1(0,1,1j*s*r(x,y))
ux = lambda x,y: -s*scipy.special.hankel1(1,1,1j*s*r(x,y))*rx(x,y)
uy = lambda x,y: -s*scipy.special.hankel1(1,1,1j*s*r(x,y))*ry(x,y)

# test the RHS
RHS = CCT.test(u,[ux,uy],gp,gm)
beta0 = RHS[0]
beta1 = RHS[1]

# observation points outside the domain
obs = np.array([[0.,4.],[4.,0.],[-4.,2.],[2.,-4.]])

POTs = helmholtz.HelmholtzPotentials(g,obs)

S = POTs[0]
D = POTs[1]

S = S(s)
D = D(s)

# dirichlet problem & DtN operator
V = V(s)
eta = np.linalg.solve(V,beta0)
uh = np.dot(S,eta)

phi = scipy.sparse.linalg.spsolve(M,beta0.real)+1j*scipy.sparse.linalg.spsolve(M,beta0.imag)

phi = Q.dot(phi)
J = J(s)
Lambda = -0.5*eta + scipy.sparse.linalg.spsolve(M,np.dot(J,eta))

uexact = u(obs[:,0],obs[:,1])
beta0exact = u(g['midpt'][:,0],g['midpt'][:,1])
beta1exact = (ux(g['midpt'][:,0],g['midpt'][:,1])*g['normal'][:,0]
              +uy(g['midpt'][:,0],g['midpt'][:,1])*g['normal'][:,1])

erru = np.max(np.abs(uexact - uh))
errLambda = np.max(np.abs(beta1exact-Lambda))*N
errPhi    = np.max(np.abs(beta0exact-phi))

errors = np.array([erru, errPhi, errLambda])

# neumann problem & NtD operator

W = W(s)
K = K(s)

psi = -np.linalg.solve(W,beta1)
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
