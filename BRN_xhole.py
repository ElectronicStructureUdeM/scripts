import numpy as np
import scipy

from xhole import XHole
import kernel

class BRN(XHole):

    def __init__(self, mol, KSKernel):
        super().__init__()
        self.mol = mol
        self.kskernel = KSKernel
        self.root_up = 0.0   # store the solutions on the grid of the spin up xhole
        self.root_down = 0.0 # store the solutions on the grid of the spin down xhole
        self.JXBRN_up = 0.0
        self.JXBRN_down = 0.0

    def EquationX(self, x, rho, eps_x_exact):
        """
        BRN's equation to solve with scipy
        Input: x is the value given by the rootfind method, rho and eps_x_exact are the density and the exchange energy per particle
        for a given spin at a given point
        Ouput: value of the equation
        """
        x3 = x**3
        b = -np.log(rho * 8.0 * np.pi / (x3)) / x
        a = b * x
        expma = np.exp(-a)
        rhs = (0.5 * (1.0 - expma - 0.5 * a * expma)) / b
        return eps_x_exact + rhs

    def brholed(self, u,a,b,c,d):
        np.seterr(invalid='raise')
        try:
            return 1/(1+d*u**4) * -c/(2*a**2*b*u)* ((a*np.abs(b-u)+1)*np.exp(-a*np.abs(b-u)) - (a*np.abs(b+u)+1)*np.exp(-a*np.abs(b+u)))
        except FloatingPointError:#a, b,c or u are zeros, zeros are returned
            return np.zeros(np.shape(u)[0])

    def m1brd(self, u, rho, d):
        kf = (3.0*np.pi**2*rho)**(1.0/3.0)
        a = 0.96*kf
        b = -np.log(rho*8*np.pi/(a**3))/a
        c = (a**3.0)/(8.0*np.pi)
        return brholed(u,a,b,c,d)*u

    def finddbrxn(self, d, rho, epsx):

        holeint = scipy.integrate.quad(m1brd,1e-10,np.inf,args=(rho,d))[0]
        holeint = holeint * 2.0*np.pi
        #print('findd',holeint,epsx)
        return holeint - epsx

    def SolveSigma(self, rho, eps_x_exact):
        """
            Input: rho, Q and exact exchange per particle
            Output: root of the BRN's equation for a spin coordinate
            Description: Finds the root of the equation XXX developed in ref XXX for a given spin coordinate
        """
        alim = 2.0 / ((3.0 * np.pi) ** (1.0 / 3.0))
        kf = (3.0 * (np.pi * np.pi) * rho) ** (1.0/3.0)

        solobj = scipy.optimize.root_scalar(self.EquationX, args=(rho, eps_x_exact), xtol=1e-10, bracket=[1e-8,1000] , method='brentq')
        root = solobj.root
        if root / kf <= alim:
            print('fuck')
            solobj = scipy.optimize.root_scalar(self.finddbrxn, args=(rho, eps_x_exact), xtol=1e-10, bracket=[0,1000] , method='brentq')
            root = solobj.root

        return root

    def GetParamsSigma(self, rho, root):
        """
            Input: rho, Q, and the root for a given point of space
            Output: the parameters that solves the exchange hole model for a given point in space
            Description: With a given solution of the exchange hole model the function obtains its parameters
        """
        a       = root
        a3      = a**3        
        b       = 0.0 
        c       = 0.0
        d       = 0.0        
        alim    = 2.0 / ((3.0 * np.pi) ** (1.0 / 3.0))
        kf = (3.0 * (np.pi * np.pi) * rho) ** (1.0/3.0)
        kf4     = kf * kf * kf * kf

        if a != 0.0:
            b = -np.log(rho * 8.0 * np.pi / a3) / a
            c = a3 / (8.0 * np.pi)

        if a / kf <= alim:
            a = 0.96 * kf
            b = -np.log(rho * 8.0 * np.pi / a3 ) / a
            c = a3 / (8.0 * np.pi)
            d = root

        a = a / kf
        b = b * kf
        c = c / rho
        d = d / kf4

        return a, b, c, d         

    def TotalHoleSigma(self, u, a, b, c, d):
        """
            Input: interelectronic distance, and BRN's parameters a, b and c
            Output: the value of BRN at this position u for a given tuple of parameters
            Description: Calculates BRN for a given spin density
        """
        # return 1.0 / (1.0 + d * u ** 4) * -c / (2.0 * a **2 * b * u) * ((a * np.abs(b - u) + 1.0) * np.exp(-a * np.abs(b - u)) - (a * np.abs(b + u) + 1.0) * np.exp(-a * np.abs(b + u)))
        np.seterr(invalid='raise')
        try:
            return 1/(1+d*u**4) * -c/(2*a**2*b*u)* ((a*np.abs(b-u)+1)*np.exp(-a*np.abs(b-u)) - (a*np.abs(b+u)+1)*np.exp(-a*np.abs(b+u)))
        except FloatingPointError:#a, b,c or u are zeros, zeros are returned
            return np.zeros(np.shape(u)[0])        
        
    def TotalHole(self, u, zeta, aa, ab, ac, ad, ba, bb, bc, bd):
        # print('TotalHoleSigma alpha spin: ', (0.25 * (1.0 + zeta) ** 2) * self.TotalHoleSigma(((0.5 * (1.0 + zeta)) ** (1.0/3.0)) * u, aa, ab, ac, ad))
        # print('TotalHoleSigma beta spin: ', (0.25 * (1.0 - zeta) ** 2) * self.TotalHoleSigma(((0.5 * (1.0 - zeta)) ** (1.0/3.0)) * u, ba, bb, bc, bd))
        # return (0.25 * (1.0 + zeta) ** 2) * self.TotalHoleSigma(((0.5 * (1.0 + zeta)) ** (1.0/3.0)) * u, aa, ab, ac, ad) + (0.25 * (1.0 - zeta) ** 2) * self.TotalHoleSigma(((0.5 * (1.0 - zeta)) ** (1.0/3.0)) * u, ba, bb, bc, bd)
        return (0.25*(1+zeta)**2)*self.TotalHoleSigma(((0.5*(1+zeta))**(1.0/3.0))*u,aa,ab,ac,ad) + (0.25*(1-zeta)**2)*self.TotalHoleSigma(((0.5*(1-zeta))**(1.0/3.0))*u,ba,bb,bc,bd)
    def SolveOnGrid(self):
        """
            Input:
            Output: solutions of the up and down spin xholes
            Description: The function solves the BRN for each spin coordinate in the grid, 
            stores the solutions in the object's and returns the solutions
        """

        self.root_up = np.zeros(self.kskernel.ngrid)
        self.root_down = np.zeros(self.kskernel.ngrid)
        self.JXBRN_up = np.zeros(len(self.y_values))
        self.JXBRN_down = np.zeros(len(self.y_values))
        self.JXBRN = np.zeros(self.kskernel.ngrid)
        self.exed_up = np.zeros(self.kskernel.ngrid)
        self.exed_down = np.zeros(self.kskernel.ngrid)
        self.exed = np.zeros(self.kskernel.ngrid)

        # unpack all data needed
        params_up, params_down = self.kskernel.GetParams()
        rho_up = params_up[0]
        rho_down = params_down[0]
        rho = rho_up + rho_down
        zeta = self.kskernel.zeta
        kf_up = (3. * np.pi**2 * rho_up)**(1. / 3.)
        kf_down = (3. * np.pi**2 * rho_down)**(1. / 3.)
        eps_x_exact_up = self.kskernel.exact_eps_x_up
        eps_x_exact_down = self.kskernel.exact_eps_x_down

        aa = ab = ac = ad = 0.0
        ba = bb = bc = bd = 0.0

        for gridID in range(self.kskernel.ngrid):

            # Solve the alpha's xhole
            self.root_up[gridID] = self.SolveSigma(rho_up[gridID], eps_x_exact_up[gridID])
            aa, ab, ac, ad = self.GetParamsSigma(rho_up[gridID], self.root_up[gridID])
            print(aa, ab, ac, ad)
            self.JXBRN_up = 2.0 * np.pi * rho_up[gridID] / (kf_up[gridID]**2) * self.TotalHoleSigma(self.y_values, aa, ab, ac, ad)
            self.exed_up[gridID] = np.sum(self.y_weights * self.y_values_power[1] * self.JXBRN_up) * rho_up[gridID]

            # Solve the beta's xhole
            if self.mol.spin == 0:
                self.root_down[gridID] = self.root_up[gridID]
                ba, bb, bc, bd = aa, ab, ac, ad
                self.JXBRN_down = self.JXBRN_up
                self.exed_down[gridID] = self.exed_up[gridID]

            elif self.mol.nelectron > 1:
                self.root_down[gridID] = self.SolveSigma(rho_down[gridID], eps_x_exact_down[gridID])
                ba, bb, bc, bd = self.GetParamsSigma(rho_down[gridID], self.root_down[gridID])
                self.JXBRN_down = 2.0 * np.pi * rho_down[gridID] / (kf_down[gridID]**2) * self.TotalHoleSigma(self.y_values, ba, bb, bc, bd)
                self.exed_down[gridID] = np.sum(self.y_weights * self.y_values_power[1] * self.JXBRN_down) * rho_down[gridID]

            # calculate total JX
            self.JXBRN = (0.25 * (1.0 + zeta[gridID]) ** 2) * self.JXBRN_up + (0.25 * (1.0 - zeta[gridID]) ** 2) * self.JXBRN_down

            # exchange energy density
            # self.exed[gridID] = np.sum(self.y_weights * self.y_values_power[1] * self.JXBRN)
            
            self.exed[gridID] = self.exed_up[gridID] + self.exed_down[gridID]

        return

    def CalculateJX(self, rho_up, rho_down, eps_x_exact_up, eps_x_exact_down):

        rho = rho_up + rho_down
        zeta = (rho_up - rho_down) / rho
        kf_up = (3. * np.pi**2 * rho_up)**(1. / 3.)
        kf_down = (3. * np.pi**2 * rho_down)**(1. / 3.)

        JXBRN_up = 0.0
        JXBRN_down = 0.0
        JXBRN = 0.0
        root_up = 0.0
        root_down = 0.0        
        aa = ab = ac = ad = 0.0
        ba = bb = bc = bd = 0.0
        
        # Solve the alpha's xhole
        if rho_up > 1.0e-8:
            root_up = self.SolveSigma(rho_up, eps_x_exact_up)
            # print('*********************************: ', root_up)
            aa, ab, ac, ad = self.GetParamsSigma(rho_up, root_up)
            # print('*********************************: ', aa,ab,ac,ad)
            # used to get the energy
            # JXBRN_up = 2.0 * np.pi * rho_up / (kf_up ** 2) * self.TotalHoleSigma(self.y_values, aa, ab, ac, ad)
            JXBRN_up = self.TotalHoleSigma(self.y_values, aa, ab, ac, ad)

        # Solve the beta's xhole
        if self.mol.spin == 0:
            root_down = root_up
            ba, bb, bc, bd = aa, ab, ac, ad
            JXBRN_down = JXBRN_up

        elif self.mol.spin > 0 and self.mol.nelectron > 1:
            if rho_down > 1.0e-8:
                root_down = self.SolveSigma(rho_down, eps_x_exact_down)
                # print('*********************************: ', root_down)
                ba, bb, bc, bd = self.GetParamsSigma(rho_down, root_down)
                # print('*********************************: ', ba,bb,bc,bd)
                # used to get the energy
                # JXBRN_down = 2.0 * np.pi * rho_down / (kf_down ** 2) * self.TotalHoleSigma(self.y_values, ba, bb, bc, bd)
                JXBRN_down = self.TotalHoleSigma(self.y_values, ba, bb, bc, bd)

        # calculate total JX
        # JXBRN = (0.25 * (1.0 + zeta) ** 2) * JXBRN_up + (0.25 * (1.0 - zeta) ** 2) * JXBRN_down
        JXBRN = self.TotalHole(self.y_values, zeta, aa, ab, ac, ad, ba, bb, bc, bd)

        return JXBRN

    def CalculateTotalX(self):
        """
        Description: Function to compute the total exchange energy with exact exchange exchange KS.
        Output: the total exchange energy
        """
        self.SolveOnGrid()
        # return np.sum(self.kskernel.weights * self.kskernel.rho * self.exed)
        return np.sum(self.kskernel.weights * self.exed)
