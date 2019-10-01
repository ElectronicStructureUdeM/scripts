from modelXC import ModelXC
import numpy as np
import scipy
from numba import vectorize, float64
class fx1(ModelXC):
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
        return -np.exp(-alpha*u)

    def find_alpha(self,alpha,epsilonX,rhoRU):
        fx1 = self.calc_fx1(alpha,self.ux_pow[2])
        return 2.*np.pi*np.einsum("i,i,i,i->",self.uwei,self.ux_pow[1],rhoRU,fx1)-epsilonX
    
    def calc_alpha(self,epsilonX,rhoRU):
        return scipy.optimize.brentq(self.find_alpha,0,1000,args=(epsilonX,rhoRU))
    
    def test(self):
        for gridID in range(self.n_grid):
            alpha = self.calc_alpha(self.eps_x_up[gridID],self.rhoRUA[gridID])
            fx1 = self.calc_fx1(alpha,self.ux_pow[2])
            print(2.*np.pi*np.einsum("i,i,i,i->",self.uwei,self.ux_pow[1],self.rhoRUA[gridID],fx1)-self.eps_x_up[gridID])
test = fx1("Ar",[[0,0,0]],0,"pbe,pbe")

test.test()