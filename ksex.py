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
        with mol.with_rinv_origin((coord[0], coord[1], coord[2])):
            A = mol.intor('int1e_rinv')
        F = np.dot(dm, ao_value)
        return -np.einsum('i,j,ij',F,F,A)/2.
        
    def CalculateEpsilonX(self, functional, params_up, params_down):
        """
        To calculate the exact exchange energy density on the grid
        """
        rho_up = params_up[0]
        rho_down = params_down[0]
        ex_exact_up = np.zeros(self.ngridpoints)
        ex_exact_down = np.zeros(self.ngridpoints)

        if self.mol.spin == 0:
            #EX exact
            for gridID in range(self.ngridpoints):
                ex_exact_up[gridID] = self.compute_ex_exact(self.ao_values[0,gridID,:], self.dm_up,self.coords[gridID])
                
            ex_exact_down = self.ex_exact_up
        else:# for spin polarized molecule
            for gridID in range(self.ngridpoints):
                ex_exact_up[gridID] = self.compute_ex_exact(self.ao_values[0,gridID,:], self.dm_up,self.coords[gridID])
                ex_exact_down[gridID] = self.compute_ex_exact(self.ao_values[0,gridID,:], self.dm_down,self.coords[gridID])

        eps_x_exact_up = ex_exact_up / rho_up
        eps_x_exact_down = ex_exact_down / rho_down

        return eps_x_exact_up, eps_x_exact_down
        
    def CalculateTotalX(self):
        """
        Function to compute the total exchange energy of a molecule with 
        exact exchange exchange KS.
        The energies are are calculated post-approx (not self-consitent).
        """
        eps_x_exact_up, eps_x_exact_down = self.CalculateEpsilonX(functional, params_up, params_down)
        return np.einsum('i,i->', ex_exact_up + ex_exact_down, weights)

    def CalculateTotalXC(self):
        eps_x_exact_up, eps_x_exact_down = self.CalculateEpsilonX(functional, params_up, params_down)
        return np.einsum('i,i->', ex_exact_up + ex_exact_down, weights)