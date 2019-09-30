from pyscf import dft,gto,lib,scf
import re
import numpy as np
class ModelXC:
    def __init__(self,molecule,positions,spin,approx='pbe,pbe',basis='6-311+g2dp.nw',num_threads=1):
        """
        In the init, the pyscf Mole object and scf.ks object will be created
        """
        lib.num_threads(num_threads)
        self.approx=approx
        self.mol = gto.Mole()
        atoms = re.findall('[A-Z][^A-Z]*', molecule)
        molecule =[]
        nAtom=0
        for atom in atoms:
            atom_pos = positions[nAtom]
            molecule.append([atom,(atom_pos[0],atom_pos[1],atom_pos[2])])
            nAtom=nAtom+1
        self.mol.atom=molecule
        self.mol.verbose=0
        self.mol.spin=spin
        self.mol.basis = basis # downloaded from BSE
        self.mol.build()
        self.mf = scf.KS(self.mol)
        self.mf.small_rho_cutoff = 1e-12
        self.mf.grids.radi_method=dft.radi.delley
        self.mf.xc=self.approx
        self.mf.kernel()
        self.approx_Exc = self.mf.get_veff().exc

        #for stuff related to grid
        self.coords = self.mf.grids.coords
        self.weights = self.mf.grids.weights
        self.n_grid = np.shape(self.coords)[0]
        self.ao_values = dft.numint.eval_ao(self.mol, self.coords, deriv=2) #deriv=2, since we get every info(tau,lap,etc)
        if self.mol.spin==0:
            self.dm_up = self.mf.make_rdm1(mo_occ=self.mf.mo_occ/2)
            self.dm_down = self.dm_up
            self.rho_up,self.dx_rho_up,self.dy_rho_up,self.dz_rho_up,self.lap_up,self.tau_up = \
                                    dft.numint.eval_rho(self.mol, self.ao_values, self.dm_up, xctype="MGGA")
            self.rho_down=self.rho_up
            self.dx_rho_down=self.dx_rho_up
            self.dy_rho_down=self.dy_rho_up
            self.dz_rho_down=self.dz_rho_up
            self.lap_down= self.lap_up
            self.tau_down = self.tau_up
            
        else:
            dm = self.mf.make_rdm1()
            self.dm_up = dm[0]
            self.dm_down=dm[1]
            self.rho_up,self.dx_rho_up,self.dy_rho_up,self.dz_rho_up,self.lap_up,self.tau_up = \
                        dft.numint.eval_rho(self.mol, self.ao_values, self.dm_up, xctype="MGGA")

            self.rho_down,self.dx_rho_down,self.dy_rho_down,self.dz_rho_down,self.lap_down,self.tau_down = \
                        dft.numint.eval_rho(self.mol, self.ao_values, self.dm_down, xctype="MGGA")
        self.rho_tot = self.rho_up+self.rho_down
    def compute_ex_exact(self,ao_value,dm,coord):
        """
        Function to compute the exact kohn sham exchange energy density
        for a grid point.
        See the appendix of https://doi.org/10.1063/1.5083840 for details.
        Input:
            ao_value(array): ao values for a grid point
            dm(array): density matrix
            coord(array): x,y,z coordinates
        Returns:
            ex(float):ex^ks
        """
        with self.mol.with_rinv_origin((coord[0],coord[1],coord[2])):
            A = self.mol.intor('int1e_rinv')
        F = np.dot(dm,ao_value)
        return -np.einsum('i,j,ij',F,F,A)/2.

    def calc_Exks(self):
        """
        Function to compute the total energy of a molecule with 
        exchange exchange KS.
        The energies are are calculated post-approx (not self-consi).
        TODO:
            This could is quite fast, but could probably be made faster if 
            the calculation over the grid was compiled but using directly libcint
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

        self.Ex_KS_tot= np.einsum('i,i->', self.ex_exact_up+self.ex_exact_down, 
                                            self.weights)
        return self.Ex_KS_tot
    
    def calc_total_energy_Ex_ks(self):
        """
        To return the total energy using exact KS exchange
        with post-approx orbitls
        """
        try: 
            return self.mf.e_tot-self.approx_Exc+self.Ex_KS_tot
        except AttributeError:#if it was never calculated before
            self.calc_Exks()
            return self.mf.e_tot-self.approx_Exc+self.Ex_KS_tot
            
    def calc_eps_xc_post_approx(self,functional):
        """
        This function calculates the exchange-correlation energy densities
        from a functional in with converged self-consistant approx. densities
        Warning: both exchange and correlation must be specified ie. pbe,pbe and not pbe
        TODO:
            For spin unpolarized, the calculation are done uselessly for down spin
        """
        #here spin is defined as greater than one so we can exact up and down energies densites
        exchange_functional,correlation_functional = functional.split(",")
        zeros=np.zeros(self.n_grid) # also used so we can exact up and down energies densites
        if dft.libxc.is_meta_gga(functional):
            mgga_up = [self.rho_up,self.dx_rho_up,self.dy_rho_up,self.dz_rho_up,self.lap_up,self.tau_up]
            mgga_down = [self.rho_down,self.dx_rho_down,self.dy_rho_down,self.dz_rho_down,self.lap_down,self.tau_down]
            self.eps_x_up,vx_up = dft.libxc.eval_xc(exchange_functional+",", [mgga_up,[zeros,zeros,zeros,zeros,zeros,zeros]],spin=5)[:2]
            self.eps_x_down,vx_down = dft.libxc.eval_xc(exchange_functional+",", [[zeros,zeros,zeros,zeros,zeros,zeros],mgga_down],spin=5)[:2]
            self.eps_c,vc = dft.libxc.eval_xc(","+correlation_functional,[mgga_up,mgga_down],spin=5)[:2]
        elif dft.libxc.is_gga(functional):
            gga_up = [self.rho_up,self.dx_rho_up,self.dy_rho_up,self.dz_rho_up]
            gga_down = [self.rho_down,self.dx_rho_down,self.dy_rho_down,self.dz_rho_down]
            self.eps_x_up,vx_up = dft.libxc.eval_xc(exchange_functional+",", [gga_up,[zeros,zeros,zeros,zeros]],spin=5)[:2]
            self.eps_x_down,vx_down = dft.libxc.eval_xc(exchange_functional+",", [[zeros,zeros,zeros,zeros],gga_down],spin=5)[:2]
            self.eps_c,vc = dft.libxc.eval_xc(","+correlation_functional,[gga_up,gga_down],spin=5)[:2]
        else:
            self.eps_x_up,vx_up = dft.libxc.eval_xc(exchange_functional+",", [self.rho_up,zeros],spin=5)[:2]
            self.eps_x_down,vx_down = dft.libxc.eval_xc(exchange_functional+",", [zeros,self.rho_down],spin=5)[:2]
            self.eps_c,vc = dft.libxc.eval_xc(","+correlation_functional,[self.rho_up,self.rho_down],spin=5)[:2]

    def calc_Exc_post_approx(self,functional):
        if self.approx==functional:
            return self.approx_Exc
        else:
            try: 
                Ex_up = np.einsum("i,i,i->",self.eps_x_up,self.rho_up,self.weights)
                Ex_down = np.einsum("i,i,i->",self.eps_x_down,self.rho_down,self.weights)
                Ec = np.einsum("i,i,i->",self.eps_c,self.rho_tot,self.weights)
                self.Exc_post_approx = Ex_up+Ex_down+Ec
            except AttributeError:#if it was never calculated before
                self.calc_eps_xc_post_approx(functional)
                Ex_up = np.einsum("i,i,i->",self.eps_x_up,self.rho_up,self.weights)
                Ex_down = np.einsum("i,i,i->",self.eps_x_down,self.rho_down,self.weights)
                Ec = np.einsum("i,i,i->",self.eps_c,self.rho_tot,self.weights)
                self.Exc_post_approx = Ex_up+Ex_down+Ec
            finally:
                return self.Exc_post_approx

    def calc_Etot_post_approx(self,functional):
        if self.approx==functional:
            return self.mf.e_tot
        else:
            try:
                self.Etot_post_approx=self.mf.e_tot-self.approx_Exc+self.Exc_post_approx
            except AttributeError:
                self.calc_Exc_post_approx(functional)
                self.Etot_post_approx=self.mf.e_tot-self.approx_Exc+self.Exc_post_approx
            finally:
                return self.Etot_post_approx
fxc = ModelXC("Ar",[[0,0,0]],0,approx='pbe,pbe')
print(fxc.calc_Etot_post_approx("lda,pw_mod"))