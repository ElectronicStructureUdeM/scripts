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
    
    @vectorize([float64(float64,float64,float64, float64,float64,float64,float64,float64)])
    def calc_fx(alpha,beta,chi,gamma,u_alpha,u_beta,u_chi,u_gamma):
        """
        Calculate (-1+alpha*u_alpha+beta*u_beta)*exp(-gamma*u_gamma)
        input:
            alpha:float
                parameter
            beta:float
                parameter
            chi:float
                parameter
            gamma:float
                parameter
            u_alpha:float
                sphere radius, which could be to some power (usually u**2)
            u_beta:float
                sphere radius, which could be to some power (usually u**4)
            u_chi:float
                sphere radius, which could be to some power (usually u**6)            
            u_gamma:float
                sphere radius, which could be to some power (usually u**2)

        """
        return (-1.+alpha*u_alpha+beta*u_beta+chi*u_chi)*np.exp(-gamma*u_gamma)

    def find_gamma(self,gamma,epsilonX,rhoRU):
        """
        Target function to find gamma by reproducing epsilon_x ks or norm using a root finding algorithm
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
        fx2 = self.calc_fx(0.,0.,0.,gamma,0.,0.,0.,self.ux_pow[2])
        return 2.*np.pi*np.einsum("i,i,i,i->",self.uwei,self.ux_pow[1],rhoRU,fx2)-epsilonX
    
    def calc_gamma_alpha_beta_chi(self,epsilonX,rhoRU,Q,lap,rho):
        """
        To calculate gamma using a root finding algorithm to reproduce exact exchange energy density
        or normalisation
        input:
            epsilonX:float
                exchange energy density to reproduce
            rhoRU: array of float
                spherically averaged energy density
            Q: float
                -curvature
            lap:float
                laplacian of the density
            rho:float
                density
        output:
            fx:array of float
                the exchange factor for each u
        """
        gamma= scipy.optimize.brentq(self.find_gamma,-1e-2,100,args=(epsilonX,rhoRU))
        alpha = (-Q+(1./6.)*lap)/(4*rho)-gamma
        f1 = np.einsum("i,i,i,i->",self.ux_pow[1],self.uwei,rhoRU,np.exp(-gamma*self.ux_pow[2]))
        f2 = np.einsum("i,i,i,i->",self.ux_pow[2],self.uwei,rhoRU,np.exp(-gamma*self.ux_pow[2]))
        f3 = np.einsum("i,i,i,i->",self.ux_pow[3],self.uwei,rhoRU,np.exp(-gamma*self.ux_pow[2]))
        f4 = np.einsum("i,i,i,i->",self.ux_pow[4],self.uwei,rhoRU,np.exp(-gamma*self.ux_pow[2]))
        f5 = np.einsum("i,i,i,i->",self.ux_pow[5],self.uwei,rhoRU,np.exp(-gamma*self.ux_pow[2]))
        f6 = np.einsum("i,i,i,i->",self.ux_pow[6],self.uwei,rhoRU,np.exp(-gamma*self.ux_pow[2]))
        f7 = np.einsum("i,i,i,i->",self.ux_pow[7],self.uwei,rhoRU,np.exp(-gamma*self.ux_pow[2]))
        f8 = np.einsum("i,i,i,i->",self.ux_pow[8],self.uwei,rhoRU,np.exp(-gamma*self.ux_pow[2]))        
        a_data = np.array([[f6,f8],
                          [f5,f7]])
        b_data = np.array([-1./(4.*np.pi)+f2-alpha*f4,
                        epsilonX/(2.*np.pi)+f1-alpha*f3])
        beta,chi = np.linalg.solve(a_data,b_data)
        return self.calc_fx(alpha,beta,chi,gamma,self.ux_pow[2],self.ux_pow[4],
                                    self.ux_pow[6],self.ux_pow[2])

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
        self.fx_up=self.calc_gamma_alpha_beta_chi(self.eps_x_exact_up[gridID],self.rhoRUA[gridID],
                                self.Q_up[gridID],self.lap_up[gridID],self.rho_up[gridID])
        if self.mol.nelectron==1:
            self.fx_down=self.fx_up*0
        else:
            self.fx_down = self.calc_gamma_alpha_beta_chi(self.eps_x_exact_down[gridID],self.rhoRUB[gridID],
                                        self.Q_down[gridID],self.lap_down[gridID],self.rho_down[gridID])
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



    