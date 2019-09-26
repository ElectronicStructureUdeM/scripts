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

def brhparam(rhoa,epsx):
     grad = rhoa[1]**2 + rhoa[2]**2 + rhoa[3]**2
     rgrad = grad/rhoa[0]
     D = 2*rhoa[5] - rgrad/4
     Q = (rhoa[4] -2*D)/6
     #print(rhoa[4,iG])
     #print(D)
     rhor = rhoa[0]
     Qp = Q/(rhor**2) * epsx
     sol = scipy.optimize.root_scalar(findxbr, args=(Qp), xtol=1e-16, bracket=[1e-10,100] , method='brentq')
     sol = sol.root
     #print(sol)

     a = 6*Q/rhor * sol/(sol-2)
     #print(a)
     a = np.sqrt(a)
     b = sol/a
     c = rhor*np.exp(sol)
     n = 8*np.pi*c/(a**3)
   #print(n)
     if n>2: n=2
     return a,b,c,n

