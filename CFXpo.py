import numpy as np
from pyscf import gto, dft
from pyscf.dft import numint
from pyscf import lib
from matplotlib import pyplot as plt
import sys
import re
from modelXC import ModelXC
from locEs import *
from BRx import *

class CFXN(ModelXC):
    def __init__(self,molecule,positions,spin,approx='pbe,pbe',basis='6-311+g2dp.nw',num_threads=1,ASE=False):
        super().__init__(molecule,positions,spin,approx,basis,num_threads,ASE)
        # to obtain exks energy densities
        self.calc_eps_xks_post_approx() 

        # to obtain LSD energy densities
        self.calc_eps_xc_post_approx('LDA,PW_mod')
        self.eps_x_LSD_up = self.eps_x_up
        self.eps_x_LSD_down = self.eps_x_down
        self.eps_c_LSD = self.eps_c
        self.eps_xc_LSD = (self.eps_x_LSD_up*self.rho_up+self.eps_x_LSD_down*
                                self.rho_down)/self.rho_tot + self.eps_c_LSD

        # to obtain LSD energy densities
        self.calc_eps_xc_post_approx('pbe,pbe')
        self.eps_x_PBE_up = self.eps_x_up
        self.eps_x_PBE_down = self.eps_x_down
        self.eps_c_PBE = self.eps_c
        self.eps_xc_PBE = (self.eps_x_PBE_up*self.rho_up+self.eps_x_PBE_down*
                                self.rho_down)/self.rho_tot + self.eps_c_PBE
        #for the integration grid
        min_y = 1e-10
        max_y = 20.0
        gauleg_deg = 2000
        y_grid, self.y_weights =  np.polynomial.legendre.leggauss(gauleg_deg)
        self.y_values = 0.5*(y_grid + 1)*(max_y - min_y) + min_y
        self.y_weights = self.y_weights * 0.5*(max_y - min_y)

        self.y_values_power = {1:self.y_values,2:self.y_values**2,3:self.y_values**3,
                              4:self.y_values**4,5:self.y_values**5,6:self.y_values**6}

    
    def calc_BRXN_params(self,rho,eps_x):
        a,b,c,d = brxnparam(rho,eps_x)
        kf = (3.0*(np.pi**2.0) *rho)**(1.0/3.0)
        a = a/kf
        b = b*kf
        c = c/rho
        d = d/kf
        return a,b,c,d
    
    def calc_jx_approx(self,gridID,eps_x_up,eps_x_down):
        br_a_up,br_b_up,br_c_up,br_d_up = self.calc_BRXN_params(self.rho_up[gridID],
                                                                    eps_x_up)
        if (self.mol.spin>0 and self.mol.nelectron>1):
            br_a_down,br_b_down,br_c_down,br_d_down = self.calc_BRXN_params(self.rho_down[gridID],
                                                                eps_x_down)
        elif self.mol.nelectron==1:#for hydrogen
            br_a_down,br_b_down,br_c_down,br_d_down=(0,0,0,0)
        else:
            br_a_down,br_b_down,br_c_down,br_d_down=(br_a_up,br_b_up,br_c_up,br_d_up)

        return  brholedtot(self.y_values,self.zeta[gridID],br_a_up,br_b_up,br_c_up,br_d_up,
                                br_a_down,br_b_down,br_c_down,br_d_down)

    def calc_A(self,rs,zeta):
        """
        Calculate A which is used to reproduce the on-top value of rho_xc of LSD
        Input:
            Rs:float
                wigner radius
            zeta:
                spin polarisation
        """
        mu=0.193
        nu=0.525
       #Calculate A (eq.38)
        return ((1-zeta**2)*0.5*(1.0+mu*rs)/(1.0+nu*rs+mu*nu*rs**2)-1.0)*(-2./(1.+zeta**2))

    def calc_B(self,rs,zeta):
        """
        Calculate B which is used to reproduce the cusp value of rho_xc of LSD
        Input:
            Rs:float
                wigner radius
            zeta:
                spin polarisation
        """
        a=0.193
        b=0.525
        u=0.33825
        v=0.89679
        t=0.10134
        kappa=(4.0/(3.0*np.pi))*(9.0*np.pi/4.0)**(1.0/3.0)
        H = (1.0+u*rs)/(2.0+v*rs+t*rs**2)
        return (-4.0/(3.0*np.pi*kappa))*rs*H*((1.0-zeta**2)/(zeta**2+1.0))*((1.0+a*rs)/(1.0+b*rs+a*b*rs**2))
    
    def moment_JX_E(self,E,n,JX):
        return np.sum(self.y_weights*self.y_values_power[n]*JX*np.exp(-E*self.y_values_power[2]))

    def calc_C(self,E):
        #compute the moments and C
        m1 = self.moment_JX_E(E,1,self.JX_LSD)
        m2 = self.moment_JX_E(E,2,self.JX_LSD)
        m3 = self.moment_JX_E(E,3,self.JX_LSD)
        m4 = self.moment_JX_E(E,4,self.JX_LSD)
        return m1,m2,m3,m4,-(3.*np.pi/4.+m2*self.A+m3*self.B)/m4
    
    def find_E(self,E,rho,kf,epsilonXC):
        m1,m2,m3,m4,self.C = self.calc_C(E)
        xcint = self.A*m1+self.B*m2+self.C*m3
        self.eps_xc_LSD_calc = 2*np.pi*rho/kf**2*xcint
        return self.eps_xc_LSD_calc-epsilonXC
    
    def calc_E(self,rho,kf,epsilonXC):
        sol = scipy.optimize.root_scalar(self.find_E,bracket=[0.0,1000], 
                                    method = 'brentq',args=(rho,kf,epsilonXC)) 
        return sol.root

    def calc_CD(self,rho,kf,eps_xc):
        m1 = self.moment_JX_E(self.E,1,self.JX_PBE)
        m2 = self.moment_JX_E(self.E,2,self.JX_PBE)
        m3 = self.moment_JX_E(self.E,3,self.JX_PBE)
        m4 = self.moment_JX_E(self.E,4,self.JX_PBE)
        m5 = self.moment_JX_E(self.E,5,self.JX_PBE)
        m6 = self.moment_JX_E(self.E,6,self.JX_PBE)
        a=np.array([[m4,m6],
                    [m3,m5]])
        b=np.array([-(3.*np.pi/4.+self.A*m2+self.B*m3),
                    eps_xc*kf**2/(2.*np.pi*rho)-self.A*m1-self.B*m2])
        C,D = np.linalg.solve(a, b)
        self.eps_xc_PBE_calc=2.*np.pi*rho/kf**2*(self.A*m1+self.B*m2+C*m3+D*m5)
        return C,D

    def calc_eps_xc_cfxn(self,gridID):
        # for jx lsd
        self.JX_LSD = self.calc_jx_approx(gridID,self.eps_x_LSD_up[gridID],self.eps_x_LSD_down[gridID])
        # for fc lsd
        self.A = self.calc_A(self.rs[gridID],self.zeta[gridID])
        self.B = self.calc_B(self.rs[gridID],self.zeta[gridID])
        self.E=self.calc_E(self.rho_tot[gridID],self.kf[gridID],self.eps_xc_LSD[gridID])

        #for jx pbe
        self.JX_PBE = self.calc_jx_approx(gridID,self.eps_x_PBE_up[gridID],self.eps_x_PBE_down[gridID])

        #for C,D pbe
        self.C,self.D=self.calc_CD(self.rho_tot[gridID],
                                                self.kf[gridID],self.eps_xc_PBE[gridID])
        
        #for jx exact
        self.JX_Exact = self.calc_jx_approx(gridID,self.eps_x_exact_up[gridID],
                                                  self.eps_x_exact_down[gridID])
        #to calculate C and energy 
        m1 = self.moment_JX_E(self.E,1,self.JX_Exact)
        m2 = self.moment_JX_E(self.E,2,self.JX_Exact)
        m3 = self.moment_JX_E(self.E,3,self.JX_Exact)
        m4 = self.moment_JX_E(self.E,4,self.JX_Exact)
        m5 = self.moment_JX_E(self.E,5,self.JX_Exact)
        m6 = self.moment_JX_E(self.E,6,self.JX_Exact)
        self.C = -(3.*np.pi/4.+m2*self.A+m3*self.B+m6*self.D)/m4
        self.eps_xc_calc = 2.*np.pi*self.rho_tot[gridID]/self.kf[gridID]**2*(self.A*m1+
                                        self.B*m2+self.C*m3+self.D*m5)

        return self.eps_xc_calc-(self.eps_x_exact_up[gridID]*self.rho_up[gridID]+self.eps_x_exact_down[gridID]*self.rho_down[gridID])/self.rho_tot[gridID]
   
    def calc_Exc_cfxn(self):
        sum=0.
        for gridID in range(self.n_grid):
            sum+=self.weights[gridID]*self.rho_tot[gridID]*self.calc_eps_xc_cfxn(gridID)
        print(sum)




test = CFXN('Ar',[],0,basis = 'cc-pvtz',ASE=False)
test.calc_Exc_cfxn()



