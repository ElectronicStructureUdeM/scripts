from pyscf import gto,scf,dft
import numpy as np
import matplotlib.pyplot as plt
import scipy
from pyscf import lib
from calc_rhoru import calc_rhoru
from scipy.optimize import brentq
from compute_energies import compute_ex_exact
def calcFx4(alpha,beta,xi,gamma,u):
    #todo cythonize  fx4
    return (-1./2.+alpha*u**2+beta*u**4+xi*u**6)*np.exp(-gamma*u**2)

def calcIntRhoRUFX(power, gamma,rhoRU,ux,uwei):
    return np.sum(ux**power*rhoRU*uwei*np.exp(-gamma*ux**2))

def calcIntRhoXPlus(power,alpha,beta,xi,gamma,rhoRU,ux,uwei):
    #todo cythonize fx4, vectorize is slow
    vfunc = np.vectorize(calcFx4)
    fx = vfunc(alpha,beta,xi,gamma,ux)
    fxPlus = (fx>0)*fx
    return np.sum(ux**power*rhoRU*uwei*fxPlus)

def findGamma(gamma,norm,rhoRU,ux,uwei):
    return -2.*np.pi*calcIntRhoRUFX(2, gamma,rhoRU,ux,uwei)-norm
def calcGamma(norm,rhoRU,ux,uwei):
    return brentq(findGamma,-1e-3,1000,args=(norm,rhoRU,ux,uwei))

def calcAlphaBetaXi(gamma,norm,epsilonX,Q,lap,rho,rhoRU,ux,uwei):
    alpha = (-Q+(1./12.)*lap)/(2.*rho)-gamma/2
    f1 = calcIntRhoRUFX(1, gamma,rhoRU,ux,uwei)
    f2 = calcIntRhoRUFX(2, gamma,rhoRU,ux,uwei)
    f3 = calcIntRhoRUFX(3, gamma,rhoRU,ux,uwei)
    f4 = calcIntRhoRUFX(4, gamma,rhoRU,ux,uwei)
    f5 = calcIntRhoRUFX(5, gamma,rhoRU,ux,uwei)
    f6 = calcIntRhoRUFX(6, gamma,rhoRU,ux,uwei)
    f7 = calcIntRhoRUFX(7, gamma,rhoRU,ux,uwei)
    f8 = calcIntRhoRUFX(8, gamma,rhoRU,ux,uwei)
    a_data = np.array([[f6,f8],
                        [f5,f7]])
    b_data = np.array([norm/(4.*np.pi)+f2/2.-alpha*f4,
                        epsilonX/(2.*np.pi)+f1/2.-alpha*f3])
    beta,xi = np.linalg.solve(a_data,b_data)
    return alpha,beta,xi

#calculate PBE
lib.num_threads(1)
mol = gto.Mole()
mol.atom='He'
mol.cart=True
mol.spin=0
mol.basis = '6-311+g2dp.nw'
mol.build()
mf = scf.KS(mol)
mf.small_rho_cutoff = 1e-8
mf.xc='pbe'
mf.kernel()

#grid
grids = mf.grids
ux,uwei,rhoRU = calc_rhoru(mol,mf,grids)
np.save(mol.atom,[ux,uwei,rhoRU])
#ux,uwei,rhoRU = np.load(mol.atom+".npy",allow_pickle=True)
nGrid=np.shape(grids.coords)[0]
R = np.linalg.norm(grids.coords,axis=1)
#densities
ao_value = dft.numint.eval_ao(mol, grids.coords, deriv=2)
dm = mf.make_rdm1()
rho,dRhoX,dRhoY,dRhoZ,lap,tau = dft.numint.eval_rho(mol, ao_value, dm, xctype="MGGA")
gradSquared = dRhoX**2+dRhoY**2+dRhoZ**2
Q = 1./12.*(lap-4.*(tau-(1./8.)*gradSquared/rho))
#pbetest
epsilonX,vx = dft.libxc.eval_xc("pbe,", np.array([rho,dRhoX,dRhoY,dRhoZ]))[:2]
#parameters calculation
norm=-1
rhoXIntPlusPBE=np.zeros(nGrid)
rhoXIntPlusExact=np.zeros(nGrid)
for gridID in range(nGrid):
    gamma=calcGamma(norm,rhoRU[gridID],ux,uwei)
    alpha,beta,xi=calcAlphaBetaXi(gamma,norm,epsilonX[gridID],Q[gridID],
                               lap[gridID],rho[gridID],
                               rhoRU[gridID],ux,uwei)
    rhoXIntPlusPBE[gridID]=4.*np.pi*calcIntRhoXPlus(2,alpha,beta,xi,gamma,rhoRU[gridID],ux,uwei)
    epsilonXExact = compute_ex_exact(mol,ao_value[0,gridID,:],
                                    dm,grids.coords[gridID])/(2.*rho[gridID])
    alpha,beta,xi=calcAlphaBetaXi(gamma,norm,epsilonXExact,Q[gridID],
                               lap[gridID],rho[gridID],
                               rhoRU[gridID],ux,uwei)
    rhoXIntPlusExact[gridID]=4.*np.pi*calcIntRhoXPlus(2,alpha,beta,xi,gamma,rhoRU[gridID],ux,uwei)
plt.scatter(R,rhoXIntPlusPBE,label="PBE")
plt.scatter(R,rhoXIntPlusExact,label=r'$Ex^{KS}$')
plt.title(mol.atom)
plt.xlabel(r'R(Bohr)')
plt.ylabel(r'$4 \pi \int_0^\infty du u^2 \rho_x^{fx4+}$')
plt.legend()
plt.show()