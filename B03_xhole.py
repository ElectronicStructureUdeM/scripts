import numpy as np
import scipy

from xhole import XHole
import kernel

class B03(XHole):

    def __init__(self, mol, KSKernel):
        self.mol = mol
        self.kskernel = KSKernel
        self.root_up = 0.0   # store the solutions on the grid of the spin up xhole
        self.root_down = 0.0 # store the solutions on the grid of the spin down xhole
        self.JXB03_up = 0.0
        self.JXB03_down = 0.0

        #for the integration grid
        min_y = 0.
        max_y = 100.0
        gauleg_deg = 2000
        y_grid, self.y_weights =  np.polynomial.legendre.leggauss(gauleg_deg)
        self.y_values = 0.5*(y_grid + 1)*(max_y - min_y) + min_y
        self.y_weights = self.y_weights * 0.5*(max_y - min_y)

        self.y_values_power = {1:self.y_values,2:self.y_values**2,3:self.y_values**3,
                              4:self.y_values**4,5:self.y_values**5,6:self.y_values**6}

    def EquationX(self, x, Qp):
        return (x - 2.0) / (x * x) * (np.exp(x) - 1.0 - x / 2.0) - (-3.0 / (2.0 * np.pi) * Qp)

    def SolveSigma(self, rho, Q, eps_x_exact):
        """
            Input: rho, Q and exact exchange per particle
            Output: root of the B03's equation for a spin coordinate
            Description: Finds the root of the equation XXX developed in ref XXX for a given spin coordinate
        """

        rho2 = rho * rho
        Qp = Q / rho2 * eps_x_exact
        solobj = scipy.optimize.root_scalar(self.EquationX, args=(Qp), xtol=1e-10, bracket=[1e-8,1000] , method='brentq')
        root = solobj.root

        return root

    def GetParamsSigma(self, rho, Q, root):
        """
            Input: rho, Q, and the root for a given point of space
            Output: the parameters that solves the exchange hole model for a given point in space
            Description: With a given solution of the exchange hole model the function obtains its parameters
        """

        a = np.sqrt(6.0 * Q / rho * root / (root - 2.0))
        b = root / a
        c = rho * np.exp(root)
        n = 8.0 * np.pi * c / (a ** 3)        
        kf = (3.0 * (np.pi * np.pi) * rho) ** (1.0/3.0)
        a = a / kf
        b = b * kf
        c = c / rho
        return a, b, c, n

    def TotalHoleSigma(self, u, a, b, c, d):
        """
            Input: interelectronic distance, and B03's parameters a, b and c
            Output: the value of B03 at this position u for a given tuple of parameters
            Description: Calculates B03 for a given spin density
        """
        return 1.0 / (1.0 + d * u ** 4) * -c / (2.0 * a **2 * b * u) * ((a * np.abs(b - u) + 1.0) * np.exp(-a * np.abs(b - u)) - (a * np.abs(b + u) + 1.0) * np.exp(-a * np.abs(b + u)))

    def TotalHole(self, u, zeta, aa, ab, ac, ad, ba, bb, bc, bd):
        # print('TotalHoleSigma alpha spin: ', (0.25 * (1.0 + zeta) ** 2) * self.TotalHoleSigma(((0.5 * (1.0 + zeta)) ** (1.0/3.0)) * u, aa, ab, ac, ad))
        # print('TotalHoleSigma beta spin: ', (0.25 * (1.0 - zeta) ** 2) * self.TotalHoleSigma(((0.5 * (1.0 - zeta)) ** (1.0/3.0)) * u, ba, bb, bc, bd))
        return (0.25 * (1.0 + zeta) ** 2) * self.TotalHoleSigma(((0.5 * (1.0 + zeta)) ** (1.0/3.0)) * u, aa, ab, ac, ad) + (0.25 * (1.0 - zeta) ** 2) * self.TotalHoleSigma(((0.5 * (1.0 - zeta)) ** (1.0/3.0)) * u, ba, bb, bc, bd)

    def SolveOnGrid(self):
        """
            Input:
            Output: solutions of the up and down spin xholes
            Description: The function solves the B03 for each spin coordinate in the grid, 
            stores the solutions in the object's and returns the solutions
        """

        self.root_up = np.zeros(self.kskernel.ngrid)
        self.root_down = np.zeros(self.kskernel.ngrid)
        self.JXB03_up = np.zeros(len(self.y_values))
        self.JXB03_down = np.zeros(len(self.y_values))
        self.JXB03 = np.zeros(self.kskernel.ngrid)
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
        Q_up = self.kskernel.Q_up
        Q_down = self.kskernel.Q_up
        eps_x_exact_up = self.kskernel.eps_x_exact_up
        eps_x_exact_down = self.kskernel.eps_x_exact_down

        aa = ab = ac = ad = 0.0
        ba = bb = bc = bd = 0.0

        for gridID in range(self.kskernel.ngrid):

            # Solve the alpha's xhole
            self.root_up[gridID] = self.SolveSigma(rho_up[gridID], Q_up[gridID], eps_x_exact_up[gridID])
            aa, ab, ac, ad = self.GetParamsSigma(rho_up[gridID], Q_up[gridID], self.root_up[gridID])
            self.JXB03_up = 2.0 * np.pi * rho_up[gridID] / (kf_up[gridID]**2) * self.TotalHoleSigma(self.y_values, aa, ab, ac, 0.0)
            self.exed_up[gridID] = np.sum(self.y_weights * self.y_values_power[1] * self.JXB03_up) * rho_up[gridID]

            # Solve the beta's xhole
            if self.mol.spin == 0:
                self.root_down[gridID] = self.root_up[gridID]
                ba, bb, bc, bd = aa, ab, ac, ad
                self.JXB03_down = self.JXB03_up
                self.exed_down[gridID] = self.exed_up[gridID]

            elif self.mol.nelectron > 1:
                self.root_down[gridID] = self.SolveSigma(rho_down[gridID], Q_down[gridID], eps_x_exact_down[gridID])
                ba, bb, bc, bd = self.GetParamsSigma(rho_down[gridID], Q_down[gridID], self.root_down[gridID])
                self.JXB03_down = 2.0 * np.pi * rho_down[gridID] / (kf_down[gridID]**2) * self.TotalHoleSigma(self.y_values, ba, bb, bc, 0.0)
                self.exed_down[gridID] = np.sum(self.y_weights * self.y_values_power[1] * self.JXB03_down) * rho_down[gridID]

            # calculate total JX
            self.JXB03 = (0.25 * (1.0 + zeta[gridID]) ** 2) * self.JXB03_up + (0.25 * (1.0 - zeta[gridID]) ** 2) * self.JXB03_down

            # exchange energy density
            # self.exed[gridID] = np.sum(self.y_weights * self.y_values_power[1] * self.JXB03)
            
            self.exed[gridID] = self.exed_up[gridID] + self.exed_down[gridID]

        return

    def CalculateTotalX(self):
        """
        Description: Function to compute the total exchange energy with exact exchange exchange KS.        
        """
        self.SolveOnGrid()
        # return np.sum(self.kskernel.weights * self.kskernel.rho * self.exed)
        return np.sum(self.kskernel.weights * self.exed)
