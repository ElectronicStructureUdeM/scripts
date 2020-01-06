
from pyscf import gto # to be deleted
import numpy as np

import kernel
from modelxc import ModelXC

class ExKS(ModelXC):

    def __init__(self, mol, aovalues, dm, coords, weights):
        super().__init__(mol, aovalues, dm, coords, weights)
        
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
        F = np.dot(dm,ao_value)
        return -np.einsum('i,j,ij',F,F,A)/2.
        
    def CalculateEpsilonX(self, functional, params_up, params_down):
        """
        To calculate the exact exchange energy density on the grid
        """
        npoints = self.weights.shape[0]
        rho_up = params_up[0]
        rho_down = params_down[0]
        ex_exact_up=np.zeros(npoints)
        ex_exact_down=np.zeros(npoints)

        if self.mol.spin == 0:
            #EX exact
            for gridID in range(npoints):
                ex_exact_up[gridID] = self.compute_ex_exact(self.ao_values[0,gridID,:], self.dm_up,self.coords[gridID])
                
            ex_exact_down = self.ex_exact_up
        else:# for spin polarized molecule
            for gridID in range(npoints):
                ex_exact_up[gridID] = self.compute_ex_exact(self.ao_values[0,gridID,:], self.dm_up,self.coords[gridID])
                ex_exact_down[gridID] = self.compute_ex_exact(self.ao_values[0,gridID,:], self.dm_down,self.coords[gridID])

        eps_x_exact_up = ex_exact_up / rho_up
        eps_x_exact_down = ex_exact_down / rho_down

        return eps_x_exact_up, eps_x_exact_down
        
    def CalculateTotalX(self, functional, params_up, params_down):
        """
        Function to compute the total exchange energy of a molecule with 
        exact exchange exchange KS.
        The energies are are calculated post-approx (not self-consitent).
        """

        return np.einsum('i,i->', ex_exact_up + ex_exact_down, weights)

def main():


    functionals = ['LDA,PW_MOD', 'PBE,PBE']
    kskernel = kernel.KSKernel()

    mol = gto.Mole()
    mol.atom = 'Ar 0 0 0'
    mol.basis = 'cc-pvtz'
    mol.spin = 0
    mol.charge = 0
    mol.build()

    kskernel.CalculateKSKernel(mol, functionals[0])
    aovalues = kskernel.GetAOValues()
    dm = kskernel.GetDM()
    coords = kskernel.GetCoords()
    weights = kskernel.GetWeights()
    params_up, params_down = kskernel.GetParams()
    
    model = ExKS(mol, aovalues, dm, coords, weights)
    model.CalculateTotalX('', params_up, params_down)

    return

if __name__ == "__main__":
    main()