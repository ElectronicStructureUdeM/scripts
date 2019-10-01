from pyscf import gto,scf,dft,lib
import numpy as np
import matplotlib.pyplot as plt
import scipy
from compute_energies import compute_ex_exact
import BRx
from numba import vectorize, float64

@vectorize([float64(float64, float64,float64,float64,float64)])
def calcFx4Plus(alpha,beta,xi,gamma,u):
    """
    Calculate the 4 parameters exchange factor if its positive:
        fx = -1/2+alpha*u**2+beta*u**4+xi*u**6)*np.exp(-gamma*u**2)
    Input:
        alpha: float
        beta: float
        xi: float
        gamma: float
        u: float 
            sphere radius
    Return:
        fx4: float
    """
    fx = (-1./2.+alpha*u**2+beta*u**4+xi*u**6)*np.exp(-gamma*u**2)
    if fx>0:
        return fx
    else:
        return 0.

def calcIntRhoRUFX(power, gamma,rhoRU,ux,uwei):
    """
    Calculate the following integral:
        int_0^umax u**power * rho(r,u)*exp^(-gamma*u**2) du
    Input:
        power:int
        gamma:float
        rhoRU: array of float
            spherically averaged non local density
        ux: array of float
            all the u values
        uwei: array of float
            the weights for the integration
    Return:
        integral value: float
    """

    return np.einsum("i,i,i,i->",ux**power,rhoRU,uwei,np.exp(-gamma*ux**2))

def calcIntRhoXPlus(power,alpha,beta,xi,gamma,rhoRU,ux,uwei):
    """ 
    Integrate the moment of the positive part of the exchange hole:
        int_0^maxu u**power * fx4 * rho(r,u) du
    Input:
        power:int
        alpha:float
        beta:float
        xi:float
        gamma:float
        rho(r,u): array of float
            spherically averaged non local density
        ux: array of float
        uwei: array of float
    Returns:
        integral value (float)
    """
    fxPlus= calcFx4Plus(alpha,beta,xi,gamma,ux)
    return np.einsum("i,i,i,i->",ux**power,rhoRU,uwei,fxPlus)

def findGamma(gamma,norm,rhoRU,ux,uwei):
    """
    The target function to calculate the parameter gamma, which is the normalisation condition
    while all the other parameters are set to 0 to calculate the range
        0=1+int_0^maxu rho(r,u)*u**2*exp(-gamma*u**2)du
    
    Input:
        gamma:float
        norm: float
            the desired normalization value, normaly -1
        rhoRU: array of float
        ux
        uwei
    Output
        1+int_0^maxu rho(r,u)*u**2*exp(-gamma*u**2)du
    """
    return -2.*np.pi*calcIntRhoRUFX(2, gamma,rhoRU,ux,uwei)-norm

def calcGamma(norm,rhoRU,ux,uwei):
    """
    To optimize gamma using the function findGamma with Brent's algorithm to solve the root

    Input:
        norm: float
            the desired normalization value, normaly -1
        rhoru: array of float
        ux: float
        uwei:float
    return:
        gamma: float
    """

    return scipy.optimize.brentq(findGamma,-1e-1,1000,args=(norm,rhoRU,ux,uwei))

def calcAlphaBetaXi(gamma,norm,epsilonX,Q,lap,rho,rhoRU,ux,uwei):
    """
    Calculate alpha, beta,xi parameters of fx4
    Alpha is obtained from the exact curvature condition
    beta is obtained by normalizing the hole to norm
    xi is obtained by reproducing an exchange energy density

    Input:
        gamma: float
        norm:float
            the desired normalisation value, usualy -1
        epsilonX:float
            exchange energy density to reproduce (pbe, exact, etc.)
        Q:float
            For the curvature condition
        lap:float
            laplacian of the electronic density
        rho:float
            local electronic density
        rhoru:array of float
            spherically averaged non local density
        ux:array of float
        uwei: array of float
    Return:
        alpha:float
        beta:float
        xi:float
    """
    alpha = (-Q+(1./12.)*lap)/(2.*rho)-gamma/2
    #calculate all the moments
    f1 = calcIntRhoRUFX(1, gamma,rhoRU,ux,uwei)
    f2 = calcIntRhoRUFX(2, gamma,rhoRU,ux,uwei)
    f3 = calcIntRhoRUFX(3, gamma,rhoRU,ux,uwei)
    f4 = calcIntRhoRUFX(4, gamma,rhoRU,ux,uwei)
    f5 = calcIntRhoRUFX(5, gamma,rhoRU,ux,uwei)
    f6 = calcIntRhoRUFX(6, gamma,rhoRU,ux,uwei)
    f7 = calcIntRhoRUFX(7, gamma,rhoRU,ux,uwei)
    f8 = calcIntRhoRUFX(8, gamma,rhoRU,ux,uwei)
    #solve the system of linear equations
    a_data = np.array([[f6,f8],
                        [f5,f7]])
    b_data = np.array([norm/(4.*np.pi)+f2/2.-alpha*f4,
                        epsilonX/(2.*np.pi)+f1/2.-alpha*f3])
    beta,xi = np.linalg.solve(a_data,b_data)
    return alpha,beta,xi


def calcFx4Params(norm,epsilonX,Q,lap,rho,rhoRU,ux,uwei):
    """Calculate all the parameters of fx4
    See the previous function for details
    """
    gamma = calcGamma(norm,rhoRU,ux,uwei)
    alpha,beta,xi=calcAlphaBetaXi(gamma,norm,epsilonX,Q,lap,rho,
                               rhoRU,ux,uwei)
    return gamma,alpha,beta,xi
def calcRhoXPlus(ao_value,dm,grids,rhoRU):
    """
    To calculate the integral of the positive part of rhox
    """
    rho,dRhoX,dRhoY,dRhoZ,lap,tau = 2*dft.numint.eval_rho(mol, ao_value, dm, xctype="MGGA") #WARNING we use 2*rho_sigma 
                                                                                            #because fx4 is spin unpolarized version
    gradSquared = dRhoX**2+dRhoY**2+dRhoZ**2
    Q = 1./12.*(lap-4.*(tau-(1./8.)*gradSquared/rho))
    #pbe and exact exchange enrgy density
    epsilonX,vx = dft.libxc.eval_xc("pbe,", np.array([rho,dRhoX,dRhoY,dRhoZ]))[:2]
    nGrid = np.shape(grids.coords)[0]
    epsilonXExact = np.array([compute_ex_exact(mol,ao_value[0,gridID,:],dm,grids.coords[gridID])/(rho[gridID]/2.)#for this we dont want 2*rho
                    for gridID in range(nGrid)])
    rhoXIntPlusPBE=np.zeros(nGrid)
    rhoXIntPlusExact=np.zeros(nGrid)
    ux2 = ux**2
    
    for gridID in range(nGrid):
        a,b,c,n=BRx.brhparam([rho[gridID]/2.,dRhoX[gridID]/2.,dRhoY[gridID]/2.,dRhoZ[gridID]/2.,
                            lap[gridID]/2.,tau[gridID]/2.],epsilonXExact[gridID])
        norm=-n
        gamma,alpha,beta,xi =  calcFx4Params(norm,epsilonX[gridID],Q[gridID],lap[gridID],rho[gridID],rhoRU[gridID],ux,uwei)
        rhoXIntPlusPBE[gridID]=4.*np.pi*calcIntRhoXPlus(2,alpha,beta,xi,gamma,rhoRU[gridID],ux,uwei)
        alpha,beta,xi=calcAlphaBetaXi(gamma,norm,epsilonXExact[gridID],Q[gridID],
                                lap[gridID],rho[gridID],
                                rhoRU[gridID],ux,uwei)
        rhoXIntPlusExact[gridID]=4.*np.pi*calcIntRhoXPlus(2,alpha,beta,xi,gamma,rhoRU[gridID],ux,uwei)
        print(4.*np.pi*np.einsum("i,i,i->",ux2,uwei,rhoRU[gridID]/2.))
    return rhoXIntPlusPBE,rhoXIntPlusExact

#calculate PBE
lib.num_threads(1)
mol = gto.Mole()
mol.atom='Ar'
mol.cart=True
mol.spin=0
mol.basis = '6-311+g2dp.nw'
mol.build()
mf = scf.KS(mol)
mf.small_rho_cutoff = 1e-12
mf.xc='pbe'
mf.grids.radi_method=dft.radi.delley
mf.kernel()

#grid and rhoru
grids = mf.grids
if mol.spin ==0:
    ux,uwei,rhoRUA = np.load(mol.atom+".npy",allow_pickle=True)
else:
    ux,uwei,rhoRUA,rhoRUB = np.load(mol.atom+".npy",allow_pickle=True)
R = np.linalg.norm(grids.coords,axis=1)

#densities
ao_value = dft.numint.eval_ao(mol, grids.coords, deriv=2)
if mol.spin==0:
    dmA = mf.make_rdm1(mo_occ=mf.mo_occ/2)
    rhoXIntPlusPBEA,rhoXIntPlusExactA=calcRhoXPlus(ao_value,dmA,grids,2*rhoRUA)
    dmB=dmA
    rhoXIntPlusPBEB=rhoXIntPlusPBEA
    rhoXIntPlusExactB=rhoXIntPlusExactA
else:
    dmA = mf.make_rdm1()[0]
    rhoXIntPlusPBEA,rhoXIntPlusExactA=calcRhoXPlus(ao_value,dmA,grids,2*rhoRUA)
    dmB=mf.make_rdm1()[1]
    if mol.nelectron>1:
        rhoXIntPlusPBEB,rhoXIntPlusExactB=calcRhoXPlus(ao_value,dmB,grids,2*rhoRUB)
    else:
        rhoXIntPlusPBEB=np.zeros(np.shape(grids.coords)[0])
        rhoXIntPlusExactB=np.zeros(np.shape(grids.coords)[0])

#to calculate rhoX plus



plt.scatter(R,rhoXIntPlusPBEA,label=r'$PBE_\alpha$')
plt.scatter(R,rhoXIntPlusPBEB,label=r'$PBE_\beta$')
plt.scatter(R,rhoXIntPlusExactA,label=r'$Ex_\alpha^{KS}$')
plt.scatter(R,rhoXIntPlusExactB,label=r'$Ex_\beta^{KS}$')
plt.title(mol.atom)
plt.xlabel(r'R(Bohr)')
plt.ylabel(r'$4 \pi \int_0^\infty du u^2 \rho_x^{fx4+}$')
plt.legend()
plt.savefig(mol.atom+".png")
#plt.show()