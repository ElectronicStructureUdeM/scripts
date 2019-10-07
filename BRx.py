import scipy
import numpy as np



def findxbr(x,Qp):
    lhs = (x-2)/(x**2) * (np.exp(x) -1 - x/2)
    rhs = -3/(2* np.pi) * Qp
    return lhs-rhs

def brhole(u,a,b,c):
    return -c/(2*a**2*b*u)* ((a*np.abs(b-u)+1)*np.exp(-a*np.abs(b-u)) - (a*np.abs(b+u)+1)*np.exp(-a*np.abs(b+u)))

def brhpot(u,a,b,c):
    return u*brhole(u,a,b,c)

def brhparam(Q,rho,epsx):
    size = np.shape(Q)[0]
    a = np.zeros(size)
    b = np.zeros(size)
    c = np.zeros(size)
    n = np.zeros(size)
    for gridID in range(size):
        Qp = Q[gridID]/(rho[gridID]**2) * epsx[gridID]
        sol = scipy.optimize.root_scalar(findxbr, args=(Qp), xtol=1e-16, bracket=[1e-10,100] , method='brentq')
        sol = sol.root

        a[gridID] = 6*Q[gridID]/rho[gridID] * sol/(sol-2)
        a[gridID] = np.sqrt(a[gridID])
        b[gridID] = sol/a[gridID]
        c[gridID] = rho[gridID]*np.exp(sol)
        n[gridID] = 8*np.pi*c[gridID]/(a[gridID]**3)
        if n[gridID]>2: n[gridID]=2
        
    return a,b,c,n