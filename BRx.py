import numpy as np
from pyscf import gto, dft
from pyscf.dft import numint
from pyscf import lib
import matplotlib.pyplot as plt
import pylab
import scipy
from scipy.integrate import quad

def compute_ex_exact(mol,ao_value,dm,coords):
    with mol.with_rinv_origin((coords[0],coords[1],coords[2])):
       A = mol.intor('int1e_rinv')
    F = np.dot(DMA,ao_value)
    return -.5*np.einsum('i,j,ij',F,F,A)

def quadrat(x,a):
    return x**2+a

def findxbr(x,Qp):
    lhs = (x-2)/(x**2) * (np.exp(x) -1 - x/2)
    rhs = -3/(2* np.pi) * Qp
    return lhs-rhs

def brhole(u,a,b,c):
    return -c/(2*a**2*b*u)* ((a*np.abs(b-u)+1)*np.exp(-a*np.abs(b-u)) - (a*np.abs(b+u)+1)*np.exp(-a*np.abs(b+u)))

def brhpot(u,a,b,c):
    return u*brhole(u,a,b,c)

def brhparam(rhoa,epsx):
     grad = rhoa[1] + rhoa[2] + rhoa[3]
     rgrad = grad/rhoa[0]
     D = 2*rhoa[5] - rgrad/4
     Q = (rhoa[4] -2*D)/6
     #print(rhoa[4,iG])
     #print(D)
     rhor = rhoa[0]
     Qp = Q/(rhor**2) * epsx
     sol = scipy.optimize.root_scalar(findxbr, args=(Qp), xtol=1e-16, bracket=[1e-8,1000] , method='brentq')
     sol = sol.root
     #print(sol)

     a = 6*Q/rhor * sol/(sol-2)
     #print(a)
     a = np.sqrt(a)
     b = sol/a
     c = rhor*np.exp(sol)
     n = 8*np.pi*c/(a**3)
   #print(n)
     return a,b,c



spin=0
mol = gto.Mole()
mol.atom='Ne'
mol.cart=True
mol.spin=spin
mol.charge= 0
mol.basis='sto-3g'
mol.build()

mf = dft.RKS(mol)
mf.xc='PBE0'
mf.conv_tol=1e-8
mf.small_rho_cutoff = 1e-12
mf.kernel()

lib.num_threads(1)

DM = mf.make_rdm1()
DMA = DM/2

# Use default mesh grids and weights
coords = mf.grids.coords
weights = mf.grids.weights
ao_value = numint.eval_ao(mol, coords,deriv=2)
rhoa = numint.eval_rho(mol,ao_value,DMA,xctype='MGGA')

Ngrid = weights.size 

epsx = np.zeros(Ngrid)
epsbr = np.zeros(Ngrid)

for iG in range(0,Ngrid):

   if rhoa[0,iG] > 1e-8:
    epsx[iG] = compute_ex_exact(mol,ao_value[0,iG,:],DMA,coords[iG])
    a,b,c = brhparam(rhoa[:,iG], epsx[iG])
    epsbr[iG] = quad(brhpot,0,np.inf, args=(a,b,c), epsabs=1e-16)[0]
    epsbr[iG] = epsbr[iG] *2*np.pi

   #print(epsbr)
   #print(epsx)

Exbr = np.einsum('i,i', epsbr, weights)
Exx = np.einsum('i,i', epsx, weights)

print(Exbr)
print(Exx)