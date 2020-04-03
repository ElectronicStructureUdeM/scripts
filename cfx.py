import numpy as np
from pyscf import dft,lib,scf
import scipy

import kernel
from modelxc import ModelXC
from ksex import ExKS
from dfa import DFA
from cf import CF
from xhole import XHole
from B03_xhole import B03
from BRN_xhole import BRN

class CFX(CF):
    def __init__(self, mol, kskernel):
        super().__init__(mol, kskernel)

        self.mol = mol
        self.kskernel = kskernel

        self.aovalues = kskernel.GetAOValues()
        self.coords = kskernel.GetCoords()
        self.weights = kskernel.GetWeights()
        self.ngridpoints = self.weights.shape[0]
        self.params_up, self.params_down = kskernel.GetParams()


    def CalculateA(self, rs, zeta):
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
        zeta2 = zeta * zeta

        #Calculate A (eq.38)
        return ((1.0 - zeta2) * 0.5 * (1.0 + mu * rs) / (1.0 + nu * rs + mu * nu * rs * rs) - 1.0) * (-2.0 / (1.0 + zeta2))

    def CalculateB(self, rs, zeta):
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
        kappa = (4.0 / (3.0 * np.pi)) * (9.0 * np.pi / 4.0) ** (1.0 / 3.0)
        rs2 = rs * rs
        zeta2 = zeta * zeta
        H = (1.0 + u * rs) / (2.0 + v * rs + t * rs2)

        return (-4.0 / (3.0 * np.pi * kappa)) * rs * H *((1.0 - zeta2) / (zeta2 + 1.0)) * ((1.0 + a * rs) /(1.0 + b * rs + a * b * rs2))


    def CalculateMomentJXE(self, n, JX, E,):
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
        return np.sum(self.y_weights * self.y_values_power[n] * JX * np.exp(-E * self.y_values_power[2]))

    def CalculateC(self, JX, A, B, E):
        """
        Calculate the parameter C, which is used to normalize the XC hole
        Input:
            E:parameter of fc
        return:
            C
        """

        #compute the moments and C
        m1 = self.CalculateMomentJXE(1, JX, E)
        m2 = self.CalculateMomentJXE(2, JX, E)
        m3 = self.CalculateMomentJXE(3, JX, E)
        m4 = self.CalculateMomentJXE(4, JX, E)
        return m1, m2, m3, m4, -(3. * np.pi / 4. + m2 * A + m3 * B) / m4

    def CalculateCD(self, rho, kf, eps_xc, JX, A, B, E):
        """
        Calculate the parameters C and D by solving the system of linear equation
        by normalizing the XC hole and reproducing and epsilonxc from an approximation

        Input:
            rho: density
            kf: fermi wave vector
            eps_xc: epsilonXC to reproduce
            JX
            A
            B
            E
        Return:
            C,D
        """

        m1 = self.CalculateMomentJXE(1, JX, E)
        m2 = self.CalculateMomentJXE(2, JX, E)
        m3 = self.CalculateMomentJXE(3, JX, E)
        m4 = self.CalculateMomentJXE(4, JX, E)
        m5 = self.CalculateMomentJXE(5, JX, E)
        m6 = self.CalculateMomentJXE(6, JX, E)


        a = np.array([[m4,m6],
                      [m3,m5]])
        b = np.array([-(3. * np.pi / 4. + A * m2 + B * m3), eps_xc * kf ** 2 / 
                (2. * np.pi * rho) - A * m1 - B * m2])
        C,D = np.linalg.solve(a, b)
        return C , D

    def SolveE(self, E, rho , kf, epsilonXC, JX, A, B):
        """
        Target function to find E, by reproducing and epsilon_XC from an approximation
        
        Input:
            rho: local electronic density
            kf: fermi wavevector
            epsilonXC: epsilonXC to reproduce
            A
            B
            E: parameter of fc            
        Return:
            calculate epsilonXC - target epsilonXC
        """
        m1, m2, m3, m4, C = self.CalculateC(JX, A, B, E)
        xcint = A * m1 + B * m2 + C * m3
        eps_xc_LSD_calc = 2.0 * np.pi * rho / kf ** 2 * xcint

        return eps_xc_LSD_calc - epsilonXC
    
    def CalculateE(self, rho, kf, epsilonXC, JX, A, B):
        """
        Calculate the parameter E of fc by using a root finding algorithm to reproduce an epsilonXC

        Input:
            rho: density
            kf: fermi wavevector
            epsilonXC: espilonXC to reproduce
        Return:
            E
        """
        sol = scipy.optimize.root_scalar(self.SolveE,bracket=[0.0, 10000], method = 'brentq',args=(rho, kf, epsilonXC, JX, A, B)) 
        return sol.root        
        # try:
        #     sol = scipy.optimize.root_scalar(self.SolveE,bracket=[1.0e-10,100000], method = 'brentq',args=(rho, kf, epsilonXC, JX, A, B)) 
        #     return sol.root
        # except:
        #     print("********************************* ", rho, kf, epsilonXC, JX, A, B)
        #     return 0.0

    def CalculateTotalXC(self, params_up = None, params_down = None):

        if params_up is None and params_down is None:
            params_up = self.params_up
            params_down = self.params_down

        rho_up_arr              = params_up[0]
        rho_down_arr            = params_down[0]
        rho_arr                  = rho_up_arr + rho_down_arr

        cfx_eps_xc          = np.zeros(self.kskernel.ngrid)
        cfx_xc             = 0.0

        # 1 A, 
        # 2 LSD
        # 2.1 LSD x
        # 2.2 LSD xc
        # 3 PBE
        # 3.1 PBE x
        # 3.2 PBE xc
        # 4 eps_x_exact

        # for now, we store this after we've processed kskernel.
        exks = ExKS(self.mol, self.kskernel, 'exks,')
        self.kskernel.exact_eps_x_up, self.kskernel.exact_eps_x_down = exks.CalculateEpsilonX()

        lsd = DFA(self.mol, self.kskernel, 'LDA,PW_MOD')
        lsd_eps_x_up_arr, lsd_eps_x_down_arr, lsd_eps_c_arr = lsd.CalculateEpsilonXC()
        lsd_eps_x_arr = lsd_eps_x_up_arr + lsd_eps_x_down_arr

        pbe = DFA(self.mol, self.kskernel, 'PBE,PBE')
        pbe_eps_x_up_arr, pbe_eps_x_down_arr, pbe_eps_c_arr = pbe.CalculateEpsilonXC()
        pbe_eps_x_arr = pbe_eps_x_up_arr + pbe_eps_x_down_arr        

        b03_xhole = B03(self.mol, self.kskernel)
        brn_xhole = BRN(self.mol, self.kskernel)        

        for gridID in range(self.kskernel.ngrid):

            weight = self.kskernel.weights[gridID]
            rho_up = rho_up_arr[gridID]
            rho_down = rho_down_arr[gridID]
            rho = rho_arr[gridID]
            rs = self.kskernel.rs[gridID] 
            kf = self.kskernel.kf[gridID]
            kf_up = (3. * np.pi**2 * rho_up)**(1. / 3.)
            kf_down = (3. * np.pi**2 * rho_down)**(1. / 3.)

            zeta = self.kskernel.zeta[gridID]
            Q_up = self.kskernel.Q_up[gridID]
            Q_down = self.kskernel.Q_down[gridID]

            # print('Parameters: ru = {:.12e}\trd = {:.12e}\tqu = {:.12e}\tqd = {:.12e}'.format(rho_up, rho_down, Q_up, Q_down))

            lsd_eps_x_up = lsd_eps_x_up_arr[gridID]
            lsd_eps_x_down = lsd_eps_x_down_arr[gridID]
            lsd_eps_c = lsd_eps_c_arr[gridID]
            lsd_eps_xc = (rho_up * lsd_eps_x_up + rho_down * lsd_eps_x_down)/rho + lsd_eps_c
            
            pbe_eps_x_up = pbe_eps_x_up_arr[gridID]
            pbe_eps_x_down = pbe_eps_x_down_arr[gridID]
            pbe_eps_c = pbe_eps_c_arr[gridID]
            pbe_eps_xc = (rho_up * pbe_eps_x_up + rho_down * pbe_eps_x_down)/rho + pbe_eps_c

            exact_eps_x_up = self.kskernel.exact_eps_x_up[gridID]
            exact_eps_x_down = self.kskernel.exact_eps_x_down[gridID]
            
            if rho > 1.0e-8:
                A = self.CalculateA(rs, zeta)
        
                B = self.CalculateB(rs, zeta)

                # LSD CF
                # print('Parameters: r = {:.12e}\tkf = {:.12e}\txc_LSD = {:.12e}'.format(rho, kf, lsd_eps_xc))
                LSD_JXBRN = brn_xhole.CalculateJX(rho_up, rho_down, lsd_eps_x_up, lsd_eps_x_down)
                # print(LSD_JXBRN)
                E = self.CalculateE(rho, kf, lsd_eps_xc, LSD_JXBRN, A, B)

                # PBE CF
                PBE_JXBRN = brn_xhole.CalculateJX(rho_up, rho_down, pbe_eps_x_up, pbe_eps_x_down)            
                C, D = self.CalculateCD(rho, kf, pbe_eps_xc, PBE_JXBRN, A, B, E)
                
                # print('Parameters: A = {:.12e}\tB = {:.12e}\tC = {:.12e}\tD = {:.12e}\tE = {:.12e}'.format(A, B, C, D, E))

                # return 0.0
                # CFX
                EXACT_JXB03 = b03_xhole.CalculateJX(rho_up, rho_down, Q_up, Q_down, exact_eps_x_up, exact_eps_x_down)

                #to calculate C and energy 
                m1 = self.CalculateMomentJXE(1, EXACT_JXB03, E)
                m2 = self.CalculateMomentJXE(2, EXACT_JXB03, E)
                m3 = self.CalculateMomentJXE(3, EXACT_JXB03, E)
                m4 = self.CalculateMomentJXE(4, EXACT_JXB03, E)
                m5 = self.CalculateMomentJXE(5, EXACT_JXB03, E)
                m6 = self.CalculateMomentJXE(6, EXACT_JXB03, E)
                C = -(3. * np.pi / 4. + m2 * A + m3 * B + m6 * D) / m4
                cfx_eps_xc[gridID] = 2. * np.pi * rho / kf ** 2 *(A * m1 + B * m2 + C * m3 + D * m5)
                # cfx_xc += weight * rho * cfx_eps_xc[gridID]
                print('Parameters: A = {:.12e}\tB = {:.12e}\tC = {:.12e}\tD = {:.12e}\tE = {:.12e}'.format(A, B, C, D, E))
        cfx_xc = np.sum(self.kskernel.weights * rho_arr * cfx_eps_xc)
        
        return cfx_xc

    def CalculateTotalEnergy(self, params_up = None, params_down = None):
        """
        To calculate the total energies of a functional
        with post-approx densities

        Input:
            functional:string
                functional name in pyscf format
        """

        if params_up is None and params_down is None:
            params_up = self.params_up
            params_down = self.params_down

        xc = self.CalculateTotalXC(params_up, params_down)

        return self.kskernel.mf.e_tot - self.kskernel.approx_xc + xc