from modelXC import ModelXC
import numpy as np
import scipy
from numba import vectorize, float64
import matplotlib.pyplot as plt
from scipy import integrate
import h5py
from CFXpo import CF
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
        #self.eps_xc_calc=np.zeros(self.n_grid)
                #for correlation factor



    @vectorize([float64(float64,float64,float64,float64,float64,float64,float64,float64,float64)])
    def calc_fx4(alpha,beta,xi,gamma,u1,u2,u4,u6,umax):
        """
        Calculate fx4
        input:
            alpha,beta,xi,gamma:
            parameters
        u1,u2,u4,u6:
            power of u, precalculated to save time
        umax:
            u corresponding to the max value of rho(r,u)
        """
        return np.exp(-gamma*u2)*(-1.+alpha*u2)+(beta*u4+xi*u6)*np.exp(-gamma*(u1-umax)**2)

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
        fx4 = self.calc_fx4(0.,0.,0.,gamma,0.,self.ux_pow[2],0.,0.,0.)
        #print(gamma, 4.*np.pi*np.einsum("i,i,i->",self.uwei,self.ux_pow[2],fx4*rhoRU),norm)
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
        gamma= scipy.optimize.brentq(self.find_gamma,-1e-1,1000,args=(norm,rhoRU))
        #alpha
        alpha = (-Q+(1./6.)*lap)/(2.*rhoR)-gamma
        #beta
        umax_ind = np.argmax(rhoRU)
        umax =0.# self.ux[umax_ind]
        m1 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[1],rhoRU,np.exp(-gamma*self.ux_pow[2]))
        m2 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[2],rhoRU,np.exp(-gamma*self.ux_pow[2]))
        m3 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[3],rhoRU,np.exp(-gamma*self.ux_pow[2]))
        m4 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[4],rhoRU,np.exp(-gamma*self.ux_pow[2]))
        m5_umax = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[5],rhoRU,np.exp(-gamma*(self.ux_pow[1]-umax)**2))
        m6_umax = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[6],rhoRU,np.exp(-gamma*(self.ux_pow[1]-umax)**2))
        m7_umax = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[7],rhoRU,np.exp(-gamma*(self.ux_pow[1]-umax)**2))
        m8_umax = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[8],rhoRU,np.exp(-gamma*(self.ux_pow[1]-umax)**2))
        a_data=[[m6_umax,m8_umax],
                [m5_umax,m7_umax]]
        b_data=[norm/(4.*np.pi)+m2-alpha*m4,
                epsilonX/(2.*np.pi)+m1-alpha*m3]
        beta,xi = np.linalg.solve(a_data,b_data)
        fx4 = self.calc_fx4(alpha,beta,xi,gamma,self.ux_pow[1],self.ux_pow[2],
                            self.ux_pow[4],self.ux_pow[6],umax)
        #print(alpha,beta,gamma)
        return fx4,gamma

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
    

    def calc_rho_x(self,gridID):
        self.rho_x_stat_up = self.fx_up*self.rhoRUA[gridID]+self.f_b03_up[gridID]*self.fx_down*self.rhoRUB[gridID]
        self.rho_x_stat_down = self.fx_down*self.rhoRUB[gridID]+self.f_b03_down[gridID]*self.fx_up*self.rhoRUA[gridID]
        rho_x_stat = 1./2.*(1.+self.zeta[gridID])*self.rho_x_stat_up+\
                    1./2.*(1.-self.zeta[gridID])*self.rho_x_stat_down
        norm_rho_x_stat = lambda gamma2: 4.*np.pi*np.sum(self.uwei*self.ux_pow[2]*
                                                    rho_x_stat*np.exp(-gamma2*self.ux_pow[2]))+1.
        norm_0 = norm_rho_x_stat(0)
        if np.abs(norm_0)<1e-10:#if the error is already small, skip root finding
            gamma2=0.
            self.rho_x = rho_x_stat
        else:
            gamma2 = scipy.optimize.brentq(norm_rho_x_stat,0,1000)
            self.rho_x = rho_x_stat*np.exp(-gamma2*self.ux_pow[2])
        
    def calc_zeta_eff(self):
        """
        To calculate an effective polarisation,
        from calculating the on-top value of the hole using a procedure 
        similar to B03
        """
        self.br_n_up[self.br_n_up>1.]=1. #issues if normalisation is below <-1
        self.br_n_down[self.br_n_down>1.]=1.
        if self.mol.nelectron>1:
            self.f_b03_up = (1.-self.br_n_up)/self.br_n_down
            self.f_b03_down = (1.-self.br_n_down)/self.br_n_up
            ones = np.ones(self.n_grid)
            f_b03 = np.minimum(self.f_b03_up,self.f_b03_down)
            f_b03 = np.minimum(f_b03,ones)
            self.f_b03_up=f_b03
            self.f_b03_down=f_b03
        else:
            self.f_b03_up = np.zeros(self.n_grid)
            self.f_b03_down= np.zeros(self.n_grid)
        ontop_rho_x_stat = 0.5*(1.+self.zeta)*(self.rho_up+self.f_b03_up*self.rho_down)+\
                      0.5*(1.-self.zeta)*(self.rho_down+self.f_b03_down*self.rho_up)
        self.zeta_eff = np.sqrt(2.*ontop_rho_x_stat/self.rho_tot-1.)
    
    def calc_rho_eff(self):
        plus=(1.+self.zeta_eff)/2.
        minus=(1.-self.zeta_eff)/2.

        self.rho_up_eff = plus*self.rho_tot
        self.rho_down_eff = minus*self.rho_tot
        #grad
        dx_tot = (self.dx_rho_up+self.dx_rho_down)
        self.dx_rho_up_eff = plus*dx_tot
        self.dx_rho_down_eff = minus*dx_tot

        dy_tot=(self.dy_rho_up+self.dy_rho_down)
        self.dy_rho_up_eff = plus*dy_tot
        self.dy_rho_down_eff = minus*dy_tot   

        dz_tot=(self.dz_rho_up+self.dz_rho_down)
        self.dz_rho_up_eff = plus*dz_tot
        self.dz_rho_down_eff = minus*dz_tot 

        #tau
        tau_tot = (self.tau_up+self.tau_down)
        self.tau_up_eff = plus*tau_tot
        self.tau_down_eff = minus*tau_tot

    
    def calc_exc_fxc(self,gridID):
        """
        To calculate exc for the model for a grid point
        It starts by calculating A and B of the correlation factor,
         calculates the parameters of fx, normalize the xc hole then calculate exc.
        """

        #for exact exchange
        norm_up=-1.#-self.br_n_up[gridID]
        norm_down=-1.#-self.br_n_down[gridID]

        self.fx_up,self.gamma_up=self.calc_fx(norm_up,self.eps_x_exact_up[gridID],self.Q_up[gridID],self.rho_up[gridID],
                                                                self.lap_up[gridID],self.rhoRUA[gridID])

        if self.mol.spin>0:
            if self.mol.nelectron==1:
                self.fx_down=0.
                self.gamma_down=0.
            else:
                self.fx_down,self.gamma_down = self.calc_fx(norm_down,self.eps_x_exact_down[gridID],self.Q_down[gridID],
                                    self.rho_down[gridID],self.lap_down[gridID],self.rhoRUB[gridID])
        else:
            self.fx_down=self.fx_up
            self.gamma_down=self.gamma_up


        self.calc_rho_x(gridID)
#
        ##renormalize
#
        self.cfx.calc_eps_xc_cf(gridID)
        self.A=self.cfx.A
        self.B=self.cfx.B*self.kf[gridID]
        self.D=self.cfx.D*self.kf[gridID]**4
        self.E=self.cfx.E*self.kf[gridID]**2
        
        m1 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[1],self.rho_x,np.exp(-self.E*self.ux_pow[2]))
        m2 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[2],self.rho_x,np.exp(-self.E*self.ux_pow[2]))
        m3 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[3],self.rho_x,np.exp(-self.E*self.ux_pow[2]))
        m4 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[4],self.rho_x,np.exp(-self.E*self.ux_pow[2]))
        m5 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[5],self.rho_x,np.exp(-self.E*self.ux_pow[2]))
        m6 = np.einsum("i,i,i,i->",self.uwei,self.ux_pow[6],self.rho_x,np.exp(-self.E*self.ux_pow[2]))
        self.C = (-1/(4.*np.pi)-self.A*m2-self.B*m3-self.D*m6)/m4

        self.eps_xc_calc= 2.*np.pi*(m1*self.A+self.B*m2+self.C*m3+self.D*m5)
        #self.eps_xc_calc = 
        return  self.eps_xc_calc*self.rho_tot[gridID]


    def calc_Etot_fxc(self):
        """
        to calculate the total Exchange-correlation energy of the model
        """
        self.calc_zeta_eff()
        self.calc_rho_eff()
        self.cfx = CF(self.mol,method="cfxav_sicA",rho={"up":self.rho_up_eff,"down":self.rho_down_eff,
                "tot":self.rho_up_eff+self.rho_down_eff},
            grad={"up":[self.dx_rho_up_eff,self.dy_rho_up_eff,self.dz_rho_up_eff],
            "down":[self.dx_rho_down_eff,self.dy_rho_down_eff,self.dz_rho_down_eff]},
            tau={"up":self.tau_up_eff,"down":self.tau_down_eff},
            eps_x_exact={"up":self.eps_x_exact_up,"down":self.eps_x_exact_down},
            br03_params={"up":{"a":self.br_a_up,"b":self.br_b_up,"c":self.br_c_up,"n":self.br_n_up},
            "down":{"a":self.br_a_down,"b":self.br_b_down,"c":self.br_c_down,"n":self.br_n_down}})
        sum=0.
        for gridID in range(self.n_grid):
            if self.rho_tot[gridID]>1e-8:
                sum+=self.calc_exc_fxc(gridID)*self.weights[gridID]
        self.Exc = sum
        self.E_tot_model = self.approx_E_tot-self.approx_Exc+self.Exc
        return self.E_tot_model




