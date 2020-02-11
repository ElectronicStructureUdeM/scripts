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
    def __init__(self,molecule,positions,spin,approx='pbe,pbe',basis='6-311+g2dp.nw',num_threads=1):
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
        self.eps_xc_calc = np.zeros(self.n_grid)


    @vectorize([float64(float64,float64,float64,float64,float64,float64,float64)])
    def calc_fx4(alpha,beta,xi,gamma,u2,u4,u6):
        """
        Calculate fx4:(-1+alpha*u**2+beta*u**4+xi*u**4)*exp(-gamma*u**2)
        input:
            alpha,beta,xi,gamma:
            parameters
        u2,u4,u6:
            power of u

        """
        return (-1.+alpha*u2+beta*u4+xi*u6)*np.exp(-gamma*u2)

    def find_gamma(self,gamma,norm,rhoRU):
        """
        Target function to find gamma by normalising the exchange hole,
        with all the other parameters set to 0.
        Input:
            gamma:parameter
            norm: float
                desired normalisation

            rhoru: array of float
                all the rho(r,u) for a r
        return:
            array with:
               int_0^maxu 4*pi*u**2*rho(r,u)*fx4 du - norm

        """
        fx4 = self.calc_fx4(0.,0.,0.,gamma,self.ux_pow[2],0.,0.)
        
        return 4.*np.pi*np.einsum("i,i,i->",self.uwei,self.ux_pow[2],fx4*rhoRU)-norm


    def calc_fx(self,norm,epsilonX,Q,rhoR,lap,rhoRU):
        """
        To calculate gamma by nomalizing rho_x while the other parameters are set to 0.
        alpha is for the curvature
        beta, xi are to normalize and reproduce an energy density
        input:
            norm: float
                normalisation to reproduce
            epsilonX:float
                exchange energy density to reproduce
            Q:float
                for the curvature condition
            rhoRU: array of float
                spherically averaged energy density
        output:
            fx:array of float
                the exchange factor for each u
        """
        #gamma
        gamma= scipy.optimize.brentq(self.find_gamma,-1e-3,1000,args=(norm,rhoRU))
        #alpha
        alpha = (-Q+(1./6.)*lap)/(2.*rhoR)-gamma
        #beta and xi
        m1 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[1],rhoRU,np.exp(-gamma*self.ux_pow[2]))
        m2 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[2],rhoRU,np.exp(-gamma*self.ux_pow[2]))
        m3 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[3],rhoRU,np.exp(-gamma*self.ux_pow[2]))
        m4 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[4],rhoRU,np.exp(-gamma*self.ux_pow[2]))
        m5 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[5],rhoRU,np.exp(-gamma*self.ux_pow[2]))
        m6 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[6],rhoRU,np.exp(-gamma*self.ux_pow[2]))
        m7 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[7],rhoRU,np.exp(-gamma*self.ux_pow[2]))
        m8 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[8],rhoRU,np.exp(-gamma*self.ux_pow[2]))
        #
        a_data = [[m6,m8],
                  [m5,m7]]
        b_data = [norm/(4.*np.pi)+m2-alpha*m4,
                    epsilonX/(2.*np.pi)+m1-alpha*m3]
        beta,xi = np.linalg.solve(a_data, b_data)
        fx4 = self.calc_fx4(alpha,beta,xi,gamma,self.ux_pow[2],
                            self.ux_pow[4],self.ux_pow[6])
        return fx4

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
        norm_up=-1.
        norm_down=-1.
        self.fx_up=self.calc_fx(norm_up,self.eps_x_exact_up[gridID],self.Q_up[gridID],self.rho_up[gridID],
                                                                self.lap_up[gridID],self.rhoRUA[gridID])

        if self.mol.spin>0:
            if self.mol.nelectron==1:
                self.fx_down=0.
            else:
                self.fx_down = self.calc_fx(norm_down,self.eps_x_exact_down[gridID],self.Q_down[gridID],
                                    self.rho_down[gridID],self.lap_down[gridID],self.rhoRUB[gridID])
        else:
            self.fx_down=self.fx_up

        self.rho_x = 1./2.*(1.+self.zeta[gridID])*self.fx_up*self.rhoRUA[gridID]+\
                    1./2.*(1.-self.zeta[gridID])*self.fx_down*self.rhoRUB[gridID]
        #renormalize
        m1 = np.einsum("i,i,i->",self.uwei,self.ux_pow[1],self.rho_x)
        m2 = np.einsum("i,i,i->",self.uwei,self.ux_pow[2],self.rho_x)
        m3 = np.einsum("i,i,i->",self.uwei,self.ux_pow[3],self.rho_x)
        m4 = np.einsum("i,i,i->",self.uwei,self.ux_pow[4],self.rho_x)
        C = (-1/(4.*np.pi)-A*m2-B*m3)/m4
        self.eps_xc_calc[gridID] = 2.*np.pi*(m1*A+B*m2+C*m3)
        #print(eps_xc*self.rho_tot[gridID],self.eps_x_exact_up[gridID]*self.rho_tot[gridID])       

        #eps_xc = 2.*np.pi*m1
        return  self.eps_xc_calc[gridID]*self.rho_tot[gridID]


    def calc_Etot_fxc(self):
        """
        to calculate the total Exchange-correlation energy of the model
        """
        sum=0
        for gridID in range(self.n_grid):
            sum+=self.calc_exc_fxc(gridID)*self.weights[gridID]
        self.Exc = sum
        self.E_tot_model = self.approx_E_tot-self.approx_Exc+self.Exc
        plt.scatter(self.rho_tot,self.eps_xc_calc)
        plt.scatter(self.rho_tot,self.eps_x_exact_up*2)
        plt.show()
        return self.E_tot_model




