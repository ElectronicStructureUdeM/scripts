import numpy as np
from pyscf import gto, dft
from pyscf.dft import numint
from pyscf import lib
import matplotlib.pyplot as plt
import pylab
import scipy
from scipy.integrate import quad

def findxbr(x,Qp):
    lhs = (x-2)/(x**2) * (np.exp(x) -1 - x/2)
    rhs = -3/(2* np.pi) * Qp
    return lhs-rhs

def brhole(u,a,b,c):
    return -c/(2*a**2*b*u)* ((a*np.abs(b-u)+1)*np.exp(-a*np.abs(b-u)) - (a*np.abs(b+u)+1)*np.exp(-a*np.abs(b+u)))


def brholetot(u,zeta,aa,ab,ac,ba,bb,bc):
    return (0.5*(1+zeta))*brhole(u,aa,ab,ac) + (0.5*(1-zeta))*brhole(u,ba,bb,bc)

def brholed(u,a,b,c,d):
    np.seterr(invalid='raise')
    try:
        return 1/(1+d*u**4) * -c/(2*a**2*b*u)* ((a*np.abs(b-u)+1)*np.exp(-a*np.abs(b-u)) - (a*np.abs(b+u)+1)*np.exp(-a*np.abs(b+u)))
    except FloatingPointError:#a, b,c or u are zeros, zeros are returned
        return np.zeros(np.shape(u)[0])

def brholedtot(u,zeta,aa,ab,ac,ad,ba,bb,bc,bd):
    if zeta>1:
        zeta=1.
    return (0.25*(1+zeta)**2)*brholed(((0.5*(1+zeta))**(1.0/3.0))*u,aa,ab,ac,ad) + (0.25*(1-zeta)**2)*brholed(((0.5*(1-zeta))**(1.0/3.0))*u,ba,bb,bc,bd)

def m1brd(u,rhot,d):
    kf = (3.0*np.pi**2*rhot)**(1.0/3.0)
    a = 0.96*kf
    b = -np.log(rhot*8*np.pi/(a**3))/a
    c = (a**3.0)/(8.0*np.pi)
    return brholed(u,a,b,c,d)*u

def m1brholed(u,a,b,c,d):
    return brholed(u,a,b,c,d)*u
    

def brxchole(u,a,b):
    return -a/(16*np.pi*b*u)* ((a*np.abs(b-u)+1)*np.exp(-a*np.abs(b-u)) - (a*np.abs(b+u)+1)*np.exp(-a*np.abs(b+u)))

def findabrxc(a,rhot,epsxc):
    b = -np.log(rhot*8*np.pi/(a**3))/a
    x = b*a
    rhs = (0.5*(1-np.exp(-x)-0.5*x*np.exp(-x)))/b
    return epsxc + rhs

def finddbrxn(d,rho,epsx):
    
    holeint = scipy.integrate.quad(m1brd,1e-10,np.inf,args=(rho,d))[0]
    holeint = holeint * 2.0*np.pi
    #print('findd',holeint,epsx)
    return holeint - epsx

def brhpot(u,a,b,c):
    return u*brhole(u,a,b,c)

def brhparam(Q,rho,epsx):
    size = np.shape(Q)[0]
    a = np.ones(size)
    b = np.ones(size)
    c = np.ones(size)
    n = np.ones(size)
    for gridID in range(size):
        if rho[gridID] > 6e-10:
         Qp = Q[gridID]/(rho[gridID]**2) * epsx[gridID]
         sol = scipy.optimize.root_scalar(findxbr, args=(Qp), xtol=1e-10, bracket=[1e-8,1000] , method='brentq')
         sol = sol.root

         a[gridID] = 6*Q[gridID]/rho[gridID] * sol/(sol-2)
         a[gridID] = np.sqrt(a[gridID])
         b[gridID] = sol/a[gridID]
         c[gridID] = rho[gridID]*np.exp(sol)
         n[gridID] = 8*np.pi*c[gridID]/(a[gridID]**3)
         if n[gridID]>2: n[gridID]=2
        
    return a,b,c,n

def brxcparam(rhot,epsxc):
    sol = scipy.optimize.root_scalar(findabrxc, args=(rhot,epsxc), xtol=1e-16, bracket=[1e-5,1000] , method='brentq')
    a = sol.root
    b = -np.log(rhot*8*np.pi/(a**3))/a
    return a,b 

def brxnparam(rhot,epsx):
  
    alim = 2.0/((3.0*np.pi)**(1.0/3.0))

    sol = scipy.optimize.root_scalar(findabrxc, args=(rhot,epsx), xtol=1e-10, bracket=[1e-10,1000] , method='brentq')

    a = sol.root 
    if a !=0.0:
     b = -np.log(rhot*8*np.pi/(a**3))/a
     c = (a**3.0)/(8.0*np.pi)
     d = 0.0  
    else :
     b = 0.0 
     c =0.0
     d = 0.0 


      

    kf = (3.0*rhot*np.pi**2.0)**(1.0/3.0)
    #if (a<0 or b<0):
    #    exit("a or b<0 %.2f %.2f %.2f %.2f"%(epsx,rhot,a/kf,b))
    if a/kf <= alim:
      #print('finddbrn',a/kf)
      a = 0.96*kf
      b = -np.log(rhot*8*np.pi/(a**3))/a
      c = (a**3.0)/(8.0*np.pi)
      sold = scipy.optimize.root_scalar(finddbrxn, args=(rhot,epsx), bracket=[0,1000] , method='brentq')
      d = sold.root

    return a,b,c,d 

