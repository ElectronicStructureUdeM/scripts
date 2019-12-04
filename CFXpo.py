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

def factor(u,A,B,C,D,E):
    return (A+B*u+C*u**2+D*u**4)*np.exp(-E*u**2)

#def Mnx(u,n,E,zeta,Abra,Abrb,Abrc,Bbra,Bbrb,Bbrc):
#    return (u**n)*brholetot(u,zeta,Abra,Abrb,Abrc,Bbra,Bbrb,Bbrc)*np.exp(-E*u**2)


class CFX:

    def __init__(self,rs,rhot,zeta,Abra,Abrb,Abrc,Abrd,Bbra,Bbrb,Bbrc,Bbrd):
        self.rs = rs
        self.rhot = rhot
        self.kf = (3.0*(np.pi**2.0) *self.rhot)**(1.0/3.0)
        self.zeta = zeta
        self.Abra = Abra
        self.Abrb = Abrb
        self.Abrc = Abrc
        self.Abrd = Abrd
        self.Bbra = Bbra
        self.Bbrb = Bbrb
        self.Bbrc = Bbrc
        self.Bbrd = Bbrd

        self.lav = 1.0

        self.A = self.LSDOT()*(-2.0/(self.zeta**2 +1.0))
        self.B = self.LSDCSP()*(-2.0/(self.zeta**2 +1.0))
        self.C = 0.0
        self.D = 0.0
        self.E = 0.0 

        self.xcLSD = 0.0
        self.xcPBE = 0.0


    def Jx(self,u):
        return brholedtot(u,self.zeta,self.Abra,self.Abrb,self.Abrc,self.Abrd,self.Bbra,self.Bbrb,self.Bbrc,self.Bbrd)
#    def Jxav(self,u):
#        return brholedtot(u,self.zeta,self.Abra,self.Abrb,self.Abrc,self.Abrd,self.Bbra,self.Bbrb,self.Bbrc,self.Bbrd)
    def Jxav(self,u):
        return brholedtot(u,self.zeta,self.Abra0,self.Abrb0,self.Abrc0,self.Abrd0,self.Bbra0,self.Bbrb0,self.Bbrc0,self.Bbrd0)

    
#    def Jxav(self,u):
#        return self.lav*brholedtot(u,self.zeta,self.Abra,self.Abrb,self.Abrc,self.Abrd,self.Bbra,self.Bbrb,self.Bbrc,self.Bbrd) + (1-self.lav)*brholedtot(u,self.zeta,self.Abra0,self.Abrb0,self.Abrc0,self.Abrd0,self.Bbra0,self.Bbrb0,self.Bbrc0,self.Bbrd0)

    def Mnxc(self,u,A,B,C,D,E,n):
     return (u**n)*self.Jx(u)*factor(u,A,B,C,D,E)

    def Mnxcav(self,u,A,B,C,D,E,n):
     return (u**n)*self.Jxav(u)*factor(u,A,B,C,D,E)

    def Mnx(self,u,E,n):
     return (u**n)*self.Jx(u)*np.exp(-E*u**2)
    
    def Cnumintegrand(self,u,E):
        A = self.A
        B = self.B
        D = self.D
        return u**2*self.Jx(u)*(A+B*u+D*u**4)*np.exp(-E*u**2)

    def Cdenintegrand(self,u,E):
        return u**4*self.Jx(u)*np.exp(-E*u**2)

    def Cnumav(self,u):
        A = self.A
        B = self.B
        D = self.D
        E = self.E
        return u**2*self.Jxav(u)*(A+B*u+D*u**4)*np.exp(-E*u**2)

    def Cdenav(self,u):
        E = self.E
        return u**4*self.Jxav(u)*np.exp(-E*u**2)
    def Get_C(self,E):
        #print(E)
        #numint =  scipy.integrate.quad(self.Cnumintegrand,1e-10,np.inf,args=(E))[0]
        #print('numint',numint)
        #denint =  scipy.integrate.quad(self.Cdenintegrand,1e-10,np.inf,args=(E))[0]

        #unif
        #ugrid = np.arange(1e-10,10,0.03)

        #gauleg

        ugrid = self.ugrid
        uwei = self.uwei

        num = self.Cnumintegrand(ugrid,E)
        den = self.Cdenintegrand(ugrid,E)

        numint = np.einsum('i,i', num, uwei)
        denint = np.einsum('i,i', den, uwei)

        C = (-1.0-4.0/(3.0*np.pi) * numint)/(4.0/(3.0*np.pi) *denint)
        return C

    def Get_CAV(self):

        ugrid = self.ugrid
        uwei = self.uwei

        num = self.Cnumav(ugrid)
        den = self.Cdenav(ugrid)

        numint = np.einsum('i,i', num, uwei)
        denint = np.einsum('i,i', den, uwei)

        C = (-1.0-4.0/(3.0*np.pi) * numint)/(4.0/(3.0*np.pi) *denint)
        return C
    def findE(self,E):
        #print(E)
        rhot = self.rhot
        kf = (3.0*(np.pi**2.0) *rhot)**(1.0/3.0)
        A = self.A
        B = self.B
        D = 0.0
        C = self.Get_C(E)

        #ugrid = np.arange(1e-10,10,0.03)
        ugrid = self.ugrid
        uwei = self.uwei
        xc = self.Mnxc(ugrid,A,B,C,D,E,1)

        #xcint = scipy.integrate.quad(self.Mnxc,1e-10,np.inf,args=(A,B,C,D,E,1))[0]
        #xcint = scipy.integrate.trapz(xc,ugrid)
        #xcint = np.sum(0.03*xc)
        xcint = np.einsum('i,i',xc,uwei)

        return 2*np.pi*rhot/(kf**2)*xcint - self.xcLSD
    
    def Get_E(self):
      sol = scipy.optimize.root_scalar(self.findE,bracket=[0.0,1000], method = 'brentq') 
      self.E = sol.root
      self.C = self.Get_C(self.E)
      #(sol.root)
      return sol.root

    
    def Momentx(self,E,n):
     ugrid = self.ugrid
     uwei = self.uwei

     mnx = self.Mnx(ugrid,E,n)
     moment = np.einsum('i,i',mnx,uwei)

     return moment

    def LSDOT(self):
      mu=0.193
      nu=0.525
      zeta = self.zeta
      rs = self.rs

      #Calculate A (eq.38)
      return ((1-zeta**2)*0.5*(1.0+mu*rs)/(1.0+nu*rs+mu*nu*rs**2)-1.0)
      
    def LSDCSP(self):
        rs = self.rs
        zeta = self.zeta
        a=0.193 
        b=0.525
        u=0.33825 
        v=0.89679
        t=0.10134
        Pi = np.pi
        kappa=(4.0/(3.0*Pi))*(9.0*Pi/4.0)**(1.0/3.0)
        H = (1.0+u*rs)/(2.0+v*rs+t*rs**2)
        lsdot = self.LSDOT()
        return (4.0/(3.0*Pi*kappa))*rs*H*(lsdot+1.0)
    @profile
    def Get_CD(self,eps):
        A = self.A
        B = self.B
        E = self.E
        Pi = np.pi
        M1 = self.Momentx(E,1)
        M2 = self.Momentx(E,2)
        M3 = self.Momentx(E,3)
        M4 = self.Momentx(E,4)
        M5 = self.Momentx(E,5)
        M6 = self.Momentx(E,6)

        Z1 = 3.0*Pi/4.0
        Z2 = -(eps*self.kf**2)/(2*np.pi*self.rhot)

        C = (-A*M1*M6 + A*M2*M5 - B*M2*M6 + B*M3*M5 + M5*Z1 - M6*Z2)/(M3*M6 - M4*M5)
        self.C = C

        D =  (-M3*(A*M2 + B*M3 + Z1) + M4*(A*M1 + B*M2 + Z2))/(M3*M6 - M4*M5)
        self.D = D

mod = ModelXC('He',[],0,basis = 'ccpvtz',ASE=False)


def CFX_XC(mod):
 Ngrid = mod.n_grid
 
 ModelXC.calc_eps_xc_post_approx(mod,'LDA,VWN')
 LSDx_up = mod.eps_x_up
 LSDx_down = mod.eps_x_down
 LSDc = mod.eps_c
 #print(LSDc)
 #print(LSDx_up)
 
 ModelXC.calc_eps_xc_post_approx(mod,'PBE,PBE')
 PBEx_up = mod.eps_x_up
 PBEx_down = mod.eps_x_down
 PBEc = mod.eps_c
 
 
 ua = 1e-10
 ub = 20.0
 deg = 2000
 ugrid, uwei =  np.polynomial.legendre.leggauss(deg)
 t = 0.5*(ugrid + 1)*(ub - ua) + ua
 uwei = uwei * 0.5*(ub - ua)
 
 Ec = 0.0
 
 
 for iG in range(0,Ngrid):
     if mod.rho_tot[iG] > 1e-08:
      rho_up = mod.rho_up[iG]
      rho_down = mod.rho_down[iG]
      rhot = mod.rho_tot[iG]
      rs = (3.0/(4.0*np.pi*rhot))**(1.0/3.0)
      zeta = mod.zeta[iG]
      kf = (3.0*(np.pi**2.0) *rhot)**(1.0/3.0)
      kfa = (3.0*(np.pi**2.0) *rho_up)**(1.0/3.0)
      kfb = (3.0*(np.pi**2.0) *rho_down)**(1.0/3.0)
  
      epsxa = LSDx_up[iG]
      epsxb = LSDx_down[iG]
      epsc = LSDc[iG]
  
      if rho_up != 0.0 :
       Abra,Abrb,Abrc,Abrd = brxnparam(rho_up,epsxa)
       Abra = Abra/kfa
       Abrb = Abrb*kfa
       Abrc = Abrc/rho_up
       Abrd = Abrd/kfa
      else:
       Abra,Abrb,Abrc,Abrd = 0.0,0.0,0.0,0.0
 
      #holeinta = scipy.integrate.quad(m1brholed,1e-10,np.inf,args=(Abra,Abrb,Abrc,Abrd))[0]
      #holeinta = holeinta*2*np.pi
      if rho_down != 0.0 :
       Bbra,Bbrb,Bbrc,Bbrd = brxnparam(rho_down,epsxb)
       Bbra = Bbra/kfb
       Bbrb = Bbrb*kfb
       Bbrc = Bbrc/rho_down
       Bbrd = Bbrd/kfb
      else :
       Bbra,Bbrb,Bbrc,Bbrd = 0.0,0.0,0.0,0.0
 
      CF = CFX(rs,rhot,zeta,Abra,Abrb,Abrc,Abrd,Bbra,Bbrb,Bbrc,Bbrd)
      CF.xcLSD = (epsxa*rho_up + epsxb*rho_down)/rhot + epsc
      CF.ugrid = t
      CF.uwei = uwei 
      #for E in range(0,10):
      #print(CF.findE(0.01))
      EE= CF.Get_E()
      #print('EE',EE,CF.findE(EE), epsc)
      #xint = scipy.integrate.quad(CF.Mnxc,1e-10,np.inf,args=(CF.A,CF.B,CF.C,0.0,CF.E,2))[0]
      #xint = xint * 4/(np.pi*3)
      #print(xint)
 
      #xint = xint * 2*np.pi*rhot/(kf**2)
      #print(xint, epsxa)
      #print(CF.A,CF.B)
      #print(CF.Jx(0.00001))
 
      ugrid = t
      xc = CF.Mnxc(ugrid,CF.A,CF.B,CF.C,CF.D,CF.E,1)
      xcint = np.einsum('i,i',xc,uwei)
      xcint = 2*np.pi*rhot/(kf**2)*xcint
      #print('Xcint cf1',xcint, CF.xcLSD)
 
 
 
      epsxa = PBEx_up[iG]
      epsxb = PBEx_down[iG]
      epsc = PBEc[iG]

      if rho_down != 0.0 :
       Abra,Abrb,Abrc,Abrd = brxnparam(rho_up,epsxa)
       Abra = Abra/kfa
       Abrb = Abrb*kfa
       Abrc = Abrc/rho_up
       Abrd = Abrd/kfa
      else:
       Abra,Abrb,Abrc,Abrd = 0.0,0.0,0.0,0.0
 
      #holeinta = scipy.integrate.quad(m1brholed,1e-10,np.inf,args=(Abra,Abrb,Abrc,Abrd))[0]
      #holeinta = holeinta*2*np.pi
      if rho_down != 0.0 :
       Bbra,Bbrb,Bbrc,Bbrd = brxnparam(rho_down,epsxb)
       Bbra = Bbra/kfb
       Bbrb = Bbrb*kfb
       Bbrc = Bbrc/rho_down
       Bbrd = Bbrd/kfb
      else :
       Bbra,Bbrb,Bbrc,Bbrd = 0.0,0.0,0.0,0.0
 
      CF2 = CFX(rs,rhot,zeta,Abra,Abrb,Abrc,Abrd,Bbra,Bbrb,Bbrc,Bbrd)
      CF2.xcPBE = (epsxa*rho_up + epsxb*rho_down)/rhot + epsc
      CF2.ugrid = t
      CF2.uwei = uwei 
      CF2.E = CF.E 
      CF2.Get_CD(CF2.xcPBE)
      #xint = scipy.integrate.quad(CF2.Mnxc,1e-10,np.inf,args=(CF2.A,CF2.B,CF2.C,CF2.D,CF2.E,2))[0]
      #xint = xint * 4/(np.pi*3)
      #print(xint)
 
      ugrid = t
      xc = CF2.Mnxc(ugrid,CF2.A,CF2.B,CF2.C,CF2.D,CF2.E,1)
      xcint = np.einsum('i,i',xc,uwei)
      #xcint = 4/(3*np.pi)*xcint
      xcint = 2*np.pi*rhot/(kf**2)*xcint
      #print('Xcint cf2',xcint, CF2.xcPBE)
 
      #print('exx',mod.eps_x_exact_up)
      if rho_up != 0.0:
       epsxa = mod.eps_x_exact_up[iG]
      else:
       epsxa =0.0
      if rho_down !=0.0:
       epsxb = mod.eps_x_exact_down[iG]
      else:
       epsxb=0.0
 
      if rho_up != 0.0 :
       Abra,Abrb,Abrc,Abrd = brxnparam(rho_up,epsxa)
       if Abra/kfa <= 2.0/((3.0*np.pi)**(1.0/3.0)) :
       #print('find in here')
#       Abra,Abrb,Abrc =  brhparam(mod.Q_up[iG],mod.rho_up[iG],mod.eps_x_exact_up[iG])
        Abra = mod.br_a_up[iG]
        Abrb = mod.br_b_up[iG]
        Abrc = mod.br_c_up[iG]
        Abrd = 0.0

       Abra = Abra/kfa
       Abrb = Abrb*kfa
       Abrc = Abrc/rho_up
       Abrd = Abrd/kfa
      else :
        Abra,Abrb,Abrc,Abrd = 0.0,0.0,0.0,0.0
 
      #holeinta = scipy.integrate.quad(m1brholed,1e-10,np.inf,args=(Abra,Abrb,Abrc,Abrd))[0]
      #holeinta = holeinta*2*np.pi
      #print(holeinta,epsxa)
      
      if rho_down != 0.0 :
       Bbra,Bbrb,Bbrc,Bbrd = brxnparam(rho_down,epsxb)
       if Bbra/kfb <= 2.0/((3.0*np.pi)**(1.0/3.0)) :
#       Bbra,Bbrb,Bbrc=  brhparam(mod.Q_down[iG],mod.rho_down[iG],mod.eps_x_exact_down[iG])
        Bbra = mod.br_a_up[iG]
        Bbrb = mod.br_b_up[iG]
        Bbrc = mod.br_c_up[iG]
        Bbrd = 0.0
       Bbra = Bbra/kfb
       Bbrb = Bbrb*kfb
       Bbrc = Bbrc/rho_down
       Bbrd = Bbrd/kfb
      else :
       Bbra,Bbrb,Bbrc,Bbrd = 0.0,0.0,0.0,0.0
 
      CF3 = CFX(rs,rhot,zeta,Abra,Abrb,Abrc,Abrd,Bbra,Bbrb,Bbrc,Bbrd)
      CF3.ugrid = t
      CF3.uwei = uwei 
 
      CF3.E = CF2.E 
      CF3.D = CF2.D
      
      if rho_up != 0.0 :
       CF3.Abra0 = mod.br_a_up[iG]/kfa
       CF3.Abrb0 = mod.br_b_up[iG]*kfa
       CF3.Abrc0 = mod.br_c_up[iG]/rho_up
       CF3.Abrd0 = 0.0
      else :
        CF3.Abra0,CF3.Abrb0,CF3.Abrc0,CF3.Abrd0 = 0.0,0.0,0.0,0.0
      if rho_down != 0.0 :
       CF3.Bbra0 = mod.br_a_down[iG]/kfb
       CF3.Bbrb0 = mod.br_b_down[iG]*kfb
       CF3.Bbrc0 = mod.br_c_down[iG]/rho_down
       CF3.Bbrd0 = 0.0
      else :
       CF3.Bbra0,CF3.Bbrb0,CF3.Bbrc0,CF3.Bbrd0 = 0.0,0.0,0.0,0.0
 
      CF3.C = CF3.Get_CAV()
 
 
      ugrid = t
      xc = CF3.Mnxcav(ugrid,CF3.A,CF3.B,CF3.C,CF3.D,CF3.E,1)
      #xc = CF3.Mnxcav(ugrid,1.0,0.0,0.0,0.0,0.0,2)
      #print(CF3.Mnxcav(0.01,1.0,0.0,0.0,0.0,0.0,0))
      xcint = np.einsum('i,i',xc,uwei)
      #xcint = 4/(3*np.pi)*xcint
      #print(xcint, mod.br_n_up[iG],mod.br_n_down[iG])
      xcint = 2*np.pi*rhot/(kf**2)*xcint
      #print('xint',xcint, (epsxa*rho_up + epsxb*rho_down)/rhot )
      epscmod = (xcint - (epsxa*rho_up + epsxb*rho_down)/rhot )
      #epscmod = np.abs(xcint)
      Ec = Ec + epscmod*rhot*mod.weights[iG]

 return Ec 
 
print(CFX_XC(mod))


