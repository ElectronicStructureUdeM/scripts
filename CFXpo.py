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

class CF(ModelXC):
    def __init__(self,molecule,positions,spin,method,approx='pbe,pbe',
                    basis='6-311+g2dp.nw',num_threads=1,ASE=False):
        super().__init__(molecule,positions,spin,approx,basis,num_threads,ASE)
        # to obtain exks energy densities
        self.calc_eps_xks_post_approx() 
        self.method=method
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
        min_y = 0.
        max_y = 100.0
        gauleg_deg = 2000
        y_grid, self.y_weights =  np.polynomial.legendre.leggauss(gauleg_deg)
        self.y_values = 0.5*(y_grid + 1)*(max_y - min_y) + min_y
        self.y_weights = self.y_weights * 0.5*(max_y - min_y)

        self.y_values_power = {1:self.y_values,2:self.y_values**2,3:self.y_values**3,
                              4:self.y_values**4,5:self.y_values**5,6:self.y_values**6}

    
    def calc_BRXN_params_spin(self,rho,eps_x):
        """
        To calculate the parameters of the becke roussel normalized echange
        hole which reproduces an energy density from an approximation

        Input:
            rho: local electronic density
            eps_x: exchange energy density per particle to reproduce
        returns:
            a,b,c,d: Becke-Roussel model parameters
        """
        a,b,c,d = brxnparam(rho,eps_x)
        kf = (3.0*(np.pi**2.0) *rho)**(1.0/3.0)
        a = a/kf
        b = b*kf
        c = c/rho
        d = d/kf**4
        return a,b,c,d
    
    def calc_BRXN_params(self,gridID,eps_x_up,eps_x_down):
        """
        Calculate the total reduced exchange hole BR exchange hole which reproduces
        the exchange energy density per particle of an approximation
        This function calculate it for both spins

        Input:
            gridID:current grid point ID
            eps_x_up:up spin epsilon_x to reproduce
            eps_x_down: down epsilon_x to reproduce
        Return:
            jx: total reduced exchange hole

        """
        br_a_up,br_b_up,br_c_up,br_d_up = self.calc_BRXN_params_spin(self.rho_up[gridID],
                                                                    eps_x_up)
        
        if (self.mol.spin>0 and self.mol.nelectron>1):
            br_a_down,br_b_down,br_c_down,br_d_down = self.calc_BRXN_params_spin(self.rho_down[gridID],
                                                                eps_x_down)
        elif self.mol.nelectron==1:#for hydrogen or one electron system
            br_a_down,br_b_down,br_c_down,br_d_down=(0.,0.,0.,0.)
        else:
            br_a_down,br_b_down,br_c_down,br_d_down=(br_a_up,br_b_up,br_c_up,br_d_up)
        return {"up":[br_a_up,br_b_up,br_c_up,br_d_up],"down":[br_a_down,br_b_down,br_c_down,br_d_down]}

    def calc_jx_method(self,gridID,y_values):
                 #For cfxN begin
        if self.method=="cfxn" or self.method=="cfxav" or self.method=="cfxav_sicA":
            self.JX_Exact = brholedtot(y_values,self.zeta[gridID],self.brxn_params_EXACT["up"][0],
                                                self.brxn_params_EXACT["up"][1],self.brxn_params_EXACT["up"][2],
                                                self.brxn_params_EXACT["up"][3],self.brxn_params_EXACT["down"][0],
                                                self.brxn_params_EXACT["down"][1],self.brxn_params_EXACT["down"][2],
                                                self.brxn_params_EXACT["down"][3])
        # for cfxn end
        if self.method=="cfx" or self.method=="cf3" or self.method=="cfxav" or self.method=="cfxav_sicA":
            #for cfx begin
            kfa = (3.0*(np.pi**2.0) *self.rho_up[gridID])**(1.0/3.0)
            kfb = (3.0*(np.pi**2.0) *self.rho_down[gridID])**(1.0/3.0)
            br03_a_up = self.br_a_up[gridID]/kfa
            br03_b_up = self.br_b_up[gridID]*kfa
            br03_c_up = self.br_c_up[gridID]/self.rho_up[gridID]
            if self.mol.nelectron>1:
                br03_a_down = self.br_a_down[gridID]/kfb
                br03_b_down = self.br_b_down[gridID]*kfb
                br03_c_down = self.br_c_down[gridID]/self.rho_down[gridID]
            else:
                br03_a_down=0.
                br03_b_down=0.
                br03_c_down=0.
            if self.method=="cfx" or self.method=="cf3":
             self.JX_Exact = brholedtot(y_values,self.zeta[gridID],br03_a_up,
                                            br03_b_up,br03_c_up,0.,
                                            br03_a_down,br03_b_down,
                                            br03_c_down,0)
            if self.method=="cfxav" or self.method=="cfxav_sicA":
                self.lambd=0.201
                self.JX_Exact = self.lambd*self.JX_Exact
                self.JX_Exact = self.JX_Exact + (1.0-self.lambd)*brholedtot(y_values,self.zeta[gridID],br03_a_up,
                                            br03_b_up,br03_c_up,0.,
                                            br03_a_down,br03_b_down,
                                            br03_c_down,0)

    def calc_A(self,rs,zeta):
        """
        Calculate A which is used to reproduce the on-top value of rho_xc of LSD
        Input:
            Rs:float
                wigner radius
            zeta:
                spin polarisation
        Return:
            A
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
        Return:
            B
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
        """
        Calculate the following integral:
        int_miny^maxy dy y^n * exp(-E*y^2)*jx(y)

        Input:
            E: parameter
            n: power of y
            JX: reduced exchange hole
        return:
            integral value
        """
        return np.sum(self.y_weights*self.y_values_power[n]*JX*np.exp(-E*self.y_values_power[2]))

    def calc_C(self,E):
        """
        Calculate the parameter C, which is used to normalize the XC hole
        Input:
            E:parameter of fc
        return:
            C
        """

        #compute the moments and C
        m1 = self.moment_JX_E(E,1,self.JX_LSD)
        m2 = self.moment_JX_E(E,2,self.JX_LSD)
        m3 = self.moment_JX_E(E,3,self.JX_LSD)
        m4 = self.moment_JX_E(E,4,self.JX_LSD)
        return m1,m2,m3,m4,-(3.*np.pi/4.+m2*self.A+m3*self.B)/m4
    
    def find_E(self,E,rho,kf,epsilonXC):
        """
        Target function to find E, by reproducing and epsilon_XC from an approximation
        
        Input:
            E: parameter of fc
            rho: local electronic density
            kf: fermi wavevector
            epsilonXC: epsilonXC to reproduce
        Return:
            calculate epsilonXC - target epsilonXC
        """
        m1,m2,m3,m4,self.C = self.calc_C(E)
        xcint = self.A*m1+self.B*m2+self.C*m3
        self.eps_xc_LSD_calc = 2*np.pi*rho/kf**2*xcint
        return self.eps_xc_LSD_calc-epsilonXC
    
    def calc_E(self,rho,kf,epsilonXC):
        """
        To calculate the E parameter of fc by using a root finding algorithm to reproduce an epsilonXC

        Input:
            rho:electronic density
            kf: fermi wavevector
            epsilonXC: espilonXC to reproduce
        Return:
            E
        """
        sol = scipy.optimize.root_scalar(self.find_E,bracket=[0.0,1000], 
                                    method = 'brentq',args=(rho,kf,epsilonXC)) 
        return sol.root

    def calc_CD(self,rho,kf,eps_xc):
        """
        Calculate the C and D parameter by solving the system of linear equation
        by normalizing the XC hole and reproducing and epsilonxc from an approximation

        Input:
            rho: electronic density local
            kf: fermi wave vector
            eps_xc: epsilonXC to reproduce
        Return:
            C,D
        """

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

    def calc_eps_xc_cf(self,gridID):
        """
        To calculate the XC energy density per particle for CF.

        Input:
            gridID: current grid point
        Return:
            epsilonXC calculated
        """
        self.A = self.calc_A(self.rs[gridID],self.zeta[gridID])
        self.B = self.calc_B(self.rs[gridID],self.zeta[gridID])
        if self.method=="cfx" or self.method=="cfxn" or self.method=="cfxav" or self.method=="cfxav_sicA":
         # for jx lsd
         self.brxn_params_lsd = self.calc_BRXN_params(gridID,self.eps_x_LSD_up[gridID],self.eps_x_LSD_down[gridID])
         self.JX_LSD = brholedtot(self.y_values,self.zeta[gridID],self.brxn_params_lsd["up"][0],
                                            self.brxn_params_lsd["up"][1],self.brxn_params_lsd["up"][2],
                                            self.brxn_params_lsd["up"][3],self.brxn_params_lsd["down"][0],
                                            self.brxn_params_lsd["down"][1],self.brxn_params_lsd["down"][2],
                                            self.brxn_params_lsd["down"][3])
         # for fc lsd
         self.E=self.calc_E(self.rho_tot[gridID],self.kf[gridID],self.eps_xc_LSD[gridID])

         #for jx pbe
         self.brxn_params_PBE = self.calc_BRXN_params(gridID,self.eps_x_PBE_up[gridID],self.eps_x_PBE_down[gridID])
         self.JX_PBE = brholedtot(self.y_values,self.zeta[gridID],self.brxn_params_PBE["up"][0],
                                            self.brxn_params_PBE["up"][1],self.brxn_params_PBE["up"][2],
                                            self.brxn_params_PBE["up"][3],self.brxn_params_PBE["down"][0],
                                            self.brxn_params_PBE["down"][1],self.brxn_params_PBE["down"][2],
                                            self.brxn_params_PBE["down"][3])
         #for C,D pbe
         self.C,self.D=self.calc_CD(self.rho_tot[gridID],
                                                self.kf[gridID],self.eps_xc_PBE[gridID])
        
         #for jx of the models which reproduces exact exchange
        if self.method=="cfxn" or self.method=="cfxav" or self.method=="cfxav_sicA":
            self.brxn_params_EXACT = self.calc_BRXN_params(gridID,self.eps_x_exact_up[gridID],self.eps_x_exact_down[gridID])
        self.calc_jx_method(gridID,self.y_values)
        
        # correction to paramters of fc
        if self.method=="cfxav_sicA":
            self.a_bryn=-0.30
            self.a_bryc=0.30
            self.a_avg = self.lambd*self.a_bryn+(1.-self.lambd)*self.a_bryc
            self.b_sic = 1.-self.a_avg
            self.sicA = self.a_avg+self.b_sic*self.zeta[gridID]**2*self.tauratio[gridID]**2
            self.D=self.D*(1.-self.sicA)
            self.E=self.E*(1.-self.sicA)

        if self.method=='cf3':
       	 self.E=0.0
         self.D=0.0
        
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
                                        

        return self.eps_xc_calc
   
    def calc_Exc_cf(self):
        """
        To calculate the total XC energy.

        Return:
            Exc
        """
        sum=0.
        for gridID in range(self.n_grid):
            if self.rho_tot[gridID]>1e-8:
                sum+=self.weights[gridID]*self.rho_tot[gridID]*self.calc_eps_xc_cf(gridID)
        return sum

    def calc_Etot_cf(self):
        """
        To calculate the total energy

        Return
        """
        Exc = self.calc_Exc_cf()
        Etot = self.mf.e_tot-self.approx_Exc+Exc
        #print(Etot)
        return Etot






