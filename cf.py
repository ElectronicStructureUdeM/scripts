from pyscf import dft,lib,scf
import numpy as np
import kernel

class CF():
    def __init__(self, mol, kskernel):

        #for the integration grid
        self.min_y = 0.0
        self.max_y = 100.0
        self.gauleg_deg = 2000
        self.y_grid, self.y_weights =  np.polynomial.legendre.leggauss(self.gauleg_deg)
        self.y_values = 0.5 * (self.y_grid + 1) * (self.max_y - self.min_y) + self.min_y
        self.y_weights = self.y_weights * 0.5*(self.max_y - self.min_y)

        self.y_values_power = {1:self.y_values, 2:self.y_values**2, 3:self.y_values**3,
                              4:self.y_values**4, 5:self.y_values**5, 6:self.y_values**6}

    @property
    def CalculateMomentJXE(self, n, JX, E):
        return NotImplementedError('Subclass specialization')

    @property
    def CalculateA(self, rs, zeta):
        return NotImplementedError('Subclass specialization')
    @property
    def CalculateB(self, rs, zeta):
        return NotImplementedError('Subclass specialization')
    @property
    def CalculateC(self, JX, A, B, E):
        return NotImplementedError('Subclass specialization')
    @property
    def SolveE(self, E, rho , kf, epsilonXC, JX, A, B):
        return NotImplementedError('Subclass specialization')
    @property
    def CalculateE(self, rho, kf, epsilonXC, JX, A, B):
        return NotImplementedError('Subclass specialization')
    @property
    def CalculateCD(self, rho, kf, eps_xc, JX, A, B, E):
        return NotImplementedError('Subclass specialization')        
    @property
    def CalculateTotalXC(self, params_up = None, params_down = None):
        return NotImplementedError('Subclass specialization')