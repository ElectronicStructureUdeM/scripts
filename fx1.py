from modelXC import ModelXC
import numpy as np
import scipy
from numba import vectorize, float64
class fx1(ModelXC):
    """
    Simple one parameters exchange factor: fx=np.exp(-alpha*u**2)
    where alpha is found by reproducing epsilon_x.
    """
    def __init__(self,molecule,positions,spin,functional='pbe,pbe',approx='pbe,pbe',basis='6-311+g2dp.nw',num_threads=1):
        super().__init__(molecule,positions,spin,approx,basis,num_threads)
        self.functional=functional
        self.calc_eps_xks_post_approx()#for exks
        self.calc_eps_xc_post_approx(self.functional)#for epsx
        if self.mol.spin==0:
            self.ux,self.uwei,self.rhoRUA = np.load(self.mol_name+".npy",allow_pickle=True)
            self.rhoRUB=self.rhoRUA
        else:
            self.ux,self.uwei,self.rhoRUA,self.rhoRUB = np.load(self.mol_name+".npy",allow_pickle=True)
        self.ux_pow = {1:self.ux,2:self.ux**2,3:self.ux**3,4:self.ux**4,
                        5:self.ux**5,6:self.ux**6,7:self.ux**7,8:self.ux**8}#all the important power of ux
    
    @vectorize([float64(float64, float64)])
    def calc_fx1(alpha,u):
        """
        Calculate -np(-alpha*u)
        input:
            alpha:float
                parameter
            u:float
                sphere radius, which could be to some power
        """
        return -np.exp(-alpha*u)

    def find_alpha(self,alpha,epsilonX,rhoRU):
        """
        Target function to find alpha by reproducing some epsilon_x using a root finding algorithm
        Input:
            alpha:float
                parameter
            epsilonX: float
                exchange energy density to reproduce
            rhoRU: array of float
                spherically non local averaged density
        return:
            0 = int_0^maxu 2*pi*u*rho(r,u)*fx1 du - epsilonX
        """
        fx1 = self.calc_fx1(alpha,self.ux_pow[2])
        return 2.*np.pi*np.einsum("i,i,i,i->",self.uwei,self.ux_pow[1],rhoRU,fx1)-epsilonX
    
    def calc_alpha(self,epsilonX,rhoRU):
        """
        To calculate alpha using a root finding algorithm to reproduce an exchange energy density
        input:
            epsilonX:float
                exchange energy density to reproduce
            rhoRU: array of float
                spherically averaged energy density
        Returns:
            alpha:float
                parameter of fx1
        """
        return scipy.optimize.brentq(self.find_alpha,0,1000,args=(epsilonX,rhoRU))
    
    