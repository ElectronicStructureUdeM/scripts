import numpy as np
from pyscf import gto # to be deleted

import kernel
from modelxc import ModelXC

class ExKS(ModelXC):

    def __init__(self, mol, KSKernel, functional='exks,'):
        super().__init__(mol, KSKernel, functional)
        
        self.ex_exact_up    = 0.0
        self.ex_exact_down  = 0.0
        self.eps_x_exact_up = 0.0
        self.eps_x_exact_down = 0.0

    def compute_ex_exact(self, ao_value, dm, coord):
        """
        Function to compute the exact kohn sham exchange energy density
        for a grid point.
        See the appendix of https://doi.org/10.1063/1.5083840 for details.
        Input:
            ao_value: array
                ao values for a grid point
            dm: array
                density matrix
            coord:array 
                x,y,z coordinates
        Returns:
            ex:float
                ex^ks
        """
        with self.mol.with_rinv_origin((coord[0], coord[1], coord[2])):
            A = self.mol.intor('int1e_rinv')
        F = np.dot(dm, ao_value)

        return -np.einsum('i,j,ij',F,F,A)/2.

    def CalculateEpsilonX(self, params_up=None, params_down=None):
        """
        To calculate the exact exchange energy density on the grid
        """

        if params_up is None and params_down is None:
            params_up = self.params_up
            params_down = self.params_down

        rho_up = params_up[0]
        rho_down = params_down[0]
        eps_x_exact_up = np.zeros(self.kskernel.ngrid)
        eps_x_exact_down = np.zeros(self.kskernel.ngrid)
        ex_exact_up = np.zeros(self.kskernel.ngrid)
        ex_exact_down = np.zeros(self.kskernel.ngrid)
        
        for gridID in range(self.kskernel.ngrid):
            ex_exact_up[gridID] = self.compute_ex_exact(self.kskernel.aovalues[0, gridID,:], self.kskernel.dm_up, self.kskernel.coords[gridID])
        eps_x_exact_up = ex_exact_up / rho_up
        
        if self.mol.spin == 0: 
            ex_exact_down = ex_exact_up
            eps_x_exact_down = eps_x_exact_up
        elif self.mol.nelectron > 1:
            for gridID in range(self.kskernel.ngrid):
                ex_exact_down[gridID] = self.compute_ex_exact(self.kskernel.aovalues[0, gridID,:], self.kskernel.dm_down, self.kskernel.coords[gridID])
            eps_x_exact_down = ex_exact_down / rho_down

        return eps_x_exact_up, eps_x_exact_down
        
    def CalculateTotalX(self, params_up=None, params_down=None):
        """
        Description: Function to compute the total exchange energy with 
        exact exchange exchange KS.
        """

        if params_up is None and params_down is None:
            params_up = self.params_up
            params_down = self.params_down            

        rho_up = params_up[0]
        rho_down = params_down[0]
        rho = rho_up + rho_down

        eps_x_exact_up, eps_x_exact_down = self.CalculateEpsilonX(params_up, params_down)

        ex_exact_up = rho_up * eps_x_exact_up
        ex_exact_down = rho_down * eps_x_exact_down

        return np.einsum('i,i->', ex_exact_up + ex_exact_down, self.kskernel.weights)

    def CalculateEpsilonC(self, params_up = None, params_down = None):
        return np.zeros(self.kskernel.ngrid)

    def CalculateTotalC(self, params_up = None, params_down = None):
        return 0.0

    def CalculateTotalXC(self, params_up = None, params_down = None):

        if params_up is None and params_down is None:
            params_up = self.params_up
            params_down = self.params_down            

        return self.CalculateTotalX(params_up, params_down)
    
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