from modelXC import ModelXC
import numpy as np
import scipy
from numba import vectorize, float64

class Fxc(ModelXC):
    """
    Simple one parameters exchange factor: fx=np.exp(-gamma*u**2)
    where gamma is found by reproducing epsilon_x exact.
    4 parameter for fc: A+Bu+Cu^2
    Where A is found from on-top of lsd
    B from cusp of LSD
    C by normalising the xc hole
    
    """
    def __init__(self,molecule,positions,spin,approx='pbe,pbe',basis='6-311+g2dp.nw',num_threads=1):
        super().__init__(molecule,positions,spin,approx,basis,num_threads)
        self.calc_eps_xks_post_approx()#for exks
        if self.mol.spin==0:
            self.ux,self.uwei,self.rhoRUA = np.load("/media/etienne/LACIE_SHARE/phd/rhoru/"+\
                                                        self.mol_name+".npy",allow_pickle=True)
            self.rhoRUB=self.rhoRUA
        else:
            self.ux,self.uwei,self.rhoRUA,self.rhoRUB = np.load("/media/etienne/LACIE_SHARE/phd/rhoru/"+\
                                                                self.mol_name+".npy",allow_pickle=True)
        self.ux_pow = {1:self.ux,2:self.ux**2,3:self.ux**3,4:self.ux**4,
                        5:self.ux**5,6:self.ux**6,7:self.ux**7,8:self.ux**8}#all the important power of ux
    
    @vectorize([float64(float64, float64,float64,float64)])
    def calc_fx2(alpha,gamma,u_alpha,u_gamma):
        """
        Calculate -np(-gamma*u)
        input:
            alpha:float
                parameter
            gamma:float
                parameter
            u_alpha:float
                sphere radius, which could be to some power (by default u**2)
            u_gamma:float
                sphere radius, which could be to some power (by default u**2)

        """
        return (-1+alpha*u_alpha)*np.exp(-gamma*u_gamma)

    def find_gamma(self,gamma,epsilonX,rhoRU):
        """
        Target function to find gamma by reproducing epsilon_x ks using a root finding algorithm
        Input:
            gamma:float
                parameter
            epsilonX: float
                exchange energy density to reproduce
            rhoRU: array of float
                spherically non local averaged density
        return:
            0 = int_0^maxu 2*pi*u*rho(r,u)*fx1 du - epsilonX
        """
        fx2 = self.calc_fx2(0.,gamma,0.,self.ux_pow[2])
        return 4.*np.pi*np.einsum("i,i,i,i->",self.uwei,self.ux_pow[2],rhoRU,fx2)+1
    
    def calc_gamma_alpha(self,epsilonX,rhoRU):
        """
        To calculate gamma using a root finding algorithm to reproduce exact exchange energy density
        input:
            epsilonX:float
                exchange energy density to reproduce
            rhoRU: array of float
                spherically averaged energy density
        output:
            fx:array of float
                the exchange factor for each u
        """
        gamma= scipy.optimize.brentq(self.find_gamma,-1e-2,100,args=(epsilonX,rhoRU))
        f1 = np.einsum("i,i,i,i->",self.ux_pow[1],self.uwei,rhoRU,np.exp(-gamma*self.ux_pow[2]))
        f3 = np.einsum("i,i,i,i->",self.ux_pow[3],self.uwei,rhoRU,np.exp(-gamma*self.ux_pow[2]))
        alpha = (epsilonX/(2.*np.pi)+f1)/f3
        self.fx = self.calc_fx2(alpha,gamma,self.ux_pow[2],self.ux_pow[2])

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
    
    def calc_fc4(self):
        """
        To calculate the correlation factor for each u
        fc = A+B*u+C*u**2
        """
        return (self.A+self.B*self.ux_pow[1]+self.C*self.ux_pow[2])

    def calc_C(self):
        """
        TO calculate C, which is found by normalizing the exchange-correlation hole
        to -1
        """
        f2 = np.einsum("i,i,i->",self.rho_x,self.uwei,self.ux_pow[2])
        f3 = np.einsum("i,i,i->",self.rho_x,self.uwei,self.ux_pow[3])
        f4 = np.einsum("i,i,i->",self.rho_x,self.uwei,self.ux_pow[4])
        self.C=(-1./(4.*np.pi)-self.A*f2-self.B*f3)/f4
    
    def calc_exc_fxc(self,gridID):
        """
        To calculate exc for the model for a grid point
        It starts by calculating A and B of the correlation factor,
         calculates the parameters of fx, normalize the xc hole then calculate exc.
        """
        #for correlation factor
        self.A = self.calc_A(self.rs[gridID],self.zeta[gridID])
        self.B = self.calc_B(self.rs[gridID],self.zeta[gridID])*self.kf[gridID] # because we have a function of u and not y

        #for exact exchange
        self.calc_gamma_alpha(self.eps_x_exact_up[gridID],self.rhoRUA[gridID])
        self.fx_up = self.fx
        if self.mol.nelectron==1:
            self.fx_down=self.fx*0
        else:
            self.calc_gamma_alpha(self.eps_x_exact_down[gridID],self.rhoRUB[gridID])
            self.fx_down = self.fx
        self.rho_x = 1./2.*(1.+self.zeta[gridID])*self.fx_up*self.rhoRUA[gridID]+\
                        1./2.*(1.-self.zeta[gridID])*self.fx_down*self.rhoRUB[gridID]
        #renormalize
        self.calc_C()
        self.fc = self.calc_fc4()
        
        #to calculate energy
        eps_xc = 2.*np.pi*np.einsum("i,i,i,i->",self.fc,self.rho_x,self.ux_pow[1],self.uwei)
        return  eps_xc*self.rho_tot[gridID]

    def calc_Etot_fxc(self):
        """
        to calculate the total Exchange-correlation energy of the model
        """
        sum=0
        for gridID in range(self.n_grid):
            sum+=self.calc_exc_fxc(gridID)*self.weights[gridID]
        self.Exc = sum
        self.Etot = self.mf.e_tot-self.approx_Exc+self.Exc
        return self.Etot



    