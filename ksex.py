import numpy as np

class ExKS:

    def __init__(self):
        
        self.ex_exact_up    = 0.0
        self.ex_exact_down  = 0.0
        self.eps_x_exact_up = 0.0
        self.eps_x_exact_down = 0.0

    def compute_ex_exact(self,ao_value,dm,coord):
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
        with self.mol.with_rinv_origin((coord[0],coord[1],coord[2])):
            A = self.mol.intor('int1e_rinv')
        F = np.dot(dm,ao_value)
        return -np.einsum('i,j,ij',F,F,A)/2.
        
    def calc_eps_xks_post_approx(self):
        """
        To calculate the exact exchange energy density on the grid
        """
        self.ex_exact_up=np.zeros(self.n_grid)
        self.ex_exact_down=np.zeros(self.n_grid)
        if self.mol.spin==0:
            #EX exact
            for gridID in range(self.n_grid):
                self.ex_exact_up[gridID] = self.compute_ex_exact(self.ao_values[0,gridID,:],
                                        self.dm_up,self.coords[gridID])
            self.ex_exact_down = self.ex_exact_up
        else:# for spin polarized molecule
            for gridID in range(self.n_grid):
                self.ex_exact_up[gridID] = self.compute_ex_exact(self.ao_values[0,gridID,:],
                                                self.dm_up,self.coords[gridID])
                self.ex_exact_down[gridID] = self.compute_ex_exact(self.ao_values[0,gridID,:],
                                                self.dm_down,self.coords[gridID])
        self.eps_x_exact_up = self.ex_exact_up/self.rho_up
        self.eps_x_exact_down = self.ex_exact_down/self.rho_down
        
    def calc_Exks_post_approx(self):
        """
        Function to compute the total exchange energy of a molecule with 
        exact exchange exchange KS.
        The energies are are calculated post-approx (not self-consitent).
        """

        self.Ex_KS_tot= np.einsum('i,i->', self.ex_exact_up+self.ex_exact_down, 
                                            self.weights)
        return self.Ex_KS_tot
    
    def calc_total_energy_Ex_ks(self):
        """
        To return the total energy using exact KS exchange
        """
        try: 
            return self.mf.e_tot-self.approx_Exc+self.Ex_KS_tot
        except AttributeError:#if it was never calculated before
            self.calc_Exks_post_approx()
            return self.mf.e_tot-self.approx_Exc+self.Ex_KS_tot
