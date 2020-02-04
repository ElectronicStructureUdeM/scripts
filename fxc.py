from modelXC import ModelXC
import numpy as np
import scipy
from numba import vectorize, float64
import matplotlib.pyplot as plt
from scipy import integrate
import h5py
class Fxc(ModelXC):
    """
    Simple one parameters exchange factor: fx=np.exp(-gamma*u**2)
    where gamma is found by reproducing epsilon_x exact.
    4 parameter for fc: A+Bu+Cu^2
    Where A is found from on-top of lsd
    B from cusp of LSD
    C by normalising the xc hole

    """
    def __init__(self,molecule,positions,spin,approx='tpss,tpss',basis='6-311+g2dp.nw',num_threads=1):
        super().__init__(molecule,positions,spin,approx,basis,num_threads)
        self.calc_eps_xks_post_approx()#for exks
        self.ux = np.array(self.f.get('ux'))
        self.uwei = np.array(self.f.get('uwei'))
        self.rhoRUA = np.array(self.f.get('rhoRUA'))
        if self.mol.spin==0:
            self.rhoRUB=self.rhoRUA
        else:
            self.rhoRUB = np.array(self.f.get('rhoRUB'))
        self.f.close()
        self.ux_pow = {1:self.ux,2:self.ux**2,3:self.ux**3,4:self.ux**4,
                        5:self.ux**5,6:self.ux**6,7:self.ux**7,8:self.ux**8}#all the important power of ux
        self.calc_eps_xc_post_approx(approx)
        self.eps_xc_post_approx = self.exc_post_approx/self.rho_tot


    @vectorize([float64(float64,float64)])
    def calc_fx1(gamma,u2):
        """
        Calculate fx1:-exp(-gamma*u**2)


        """
        return -np.exp(-gamma*u2)

    def find_gamma(self,gamma,epsilonX,rhoRU):
        """
        Target function to find gamma by reproducing epsilon_x 
        Input:
            gamma:parameter
            epsilonX: float
                exchange energy density to reproduce

            rhoru: array of float
                all the rho(r,u) for a r
        return:
            array with:
               int_0^maxu 2*pi*u*rho(r,u)*fx1 du - epsilonX

        """
        fx1 = self.calc_fx1(gamma,self.ux_pow[2])
        return 2.*np.pi*np.einsum("i,i,i->",self.uwei,self.ux_pow[1],fx1*rhoRU)-epsilonX


    def calc_fx(self,epsilonX,Q,rhoR,lap,rhoRU):
        """
        To calculate gamma using a root finding algorithm to reproduce exact exchange energy density
        input:
            epsilonX:float
                exchange energy density to reproduce
            norm:float
                normalisation to reproduce
            Q:float
                for the curvature condition
            rhoRU: array of float
                spherically averaged energy density
        output:
            fx:array of float
                the exchange factor for each u
        """
        gamma= scipy.optimize.brentq(self.find_gamma,-1e-3,1000,args=(epsilonX,rhoRU))
        fx1 = self.calc_fx1(gamma,self.ux_pow[2])
        return fx1

    ####for correlation######################
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

    def calc_fc5(self,A,B,C,D,E):
        """
        To calculate the correlation factor for each u
        fc = (A+B*u+C*u**2+D*u**4)*exp(-E*u**2)
        """
        return (A+B*self.ux_pow[1]+C*self.ux_pow[2]+D*self.ux_pow[4])*np.exp(-E*self.ux_pow[2])

    def calc_exc_fxc(self,gridID):
        """
        To calculate exc for the model for a grid point
        It starts by calculating A and B of the correlation factor,
         calculates the parameters of fx, normalize the xc hole then calculate exc.
        """
        #for correlation factor
        A = self.calc_A(self.rs[gridID],self.zeta[gridID])
        B = self.calc_B(self.rs[gridID],self.zeta[gridID])*self.kf[gridID] # because we have a function of u and not y
        #for exact exchange
        self.fx_up=self.calc_fx(self.eps_x_exact_up[gridID],self.Q_up[gridID],self.rho_up[gridID],
                                                                self.lap_up[gridID],self.rhoRUA[gridID])

        if self.mol.spin>0:
            if self.mol.nelectron==1:
                self.fx_down=0.
            else:
                self.fx_down = self.calc_fx(self.eps_x_exact_down[gridID],self.Q_down[gridID],
                                    self.rho_down[gridID],self.lap_down[gridID],self.rhoRUB[gridID])
        else:
            self.fx_down=self.fx_up
        if self.mol.nelectron>1:
            f_b03_up = (1. - self.br_n_up[gridID])/self.br_n_down[gridID]
            f_b03_down = (1. - self.br_n_down[gridID])/self.br_n_up[gridID]
            f_b03 = np.min([f_b03_up,f_b03_down,1.])
        else:
            f_b03=0.
        self.rho_x_up = self.fx_up*self.rhoRUA[gridID]+f_b03*self.fx_down*self.rhoRUB[gridID]
        self.rho_x_down = self.fx_down*self.rhoRUB[gridID]+f_b03*self.fx_up*self.rhoRUA[gridID]
        self.rho_x = 1./2.*(1.+self.zeta[gridID])*self.rho_x_up+\
                        1./2.*(1.-self.zeta[gridID])*self.rho_x_down
        #renormalize
        m1 = np.einsum("i,i,i->",self.uwei,self.ux_pow[1],self.rho_x)
        m2 = np.einsum("i,i,i->",self.uwei,self.ux_pow[2],self.rho_x)
        m3 = np.einsum("i,i,i->",self.uwei,self.ux_pow[3],self.rho_x)
        m4 = np.einsum("i,i,i->",self.uwei,self.ux_pow[4],self.rho_x)
        
        C = (-1/(4.*np.pi)-A*m2-B*m3)/m4        
        
        eps_xc = 2.*np.pi*(m1*A+B*m2+C*m3)
        return  eps_xc*self.rho_tot[gridID]


    def calc_Etot_fxc(self):
        """
        to calculate the total Exchange-correlation energy of the model
        """
        sum=0
        for gridID in range(self.n_grid):
            sum+=self.calc_exc_fxc(gridID)*self.weights[gridID]
        self.Exc = sum
        self.E_tot_model = self.approx_E_tot-self.approx_Exc+self.Exc
        return self.E_tot_model




