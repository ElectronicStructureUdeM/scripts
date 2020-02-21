from pyscf import dft,gto,lib,scf
import numpy as np

class KSKernel:

    def __init__(self, num_threads=1):
        """
        In the init, the pyscf Mole object and scf.ks object will be created
        Input:
            molecule:string
                the string for the molecule where all the element symbol are present i.e.: CHHHH and not Ch4
            positions:list of list
                The positions for each atoms in angstromg
            spin:int
                total spin of the molecule
            approx:string
                the functional in pyscf format
            basis:string
                the basis set in pyscf format
            num_threads:int
                the number of threads for pyscf
        """

        lib.num_threads(num_threads)

        # this is an object by itself so we initialize it with None
        self.mf = None

        # Coordinates, weights, number of grid points and atomic orbital values (hopefully calculated by pyscf at every grid point)
        self.coords = 0.0
        self.weights = 0.0
        self.ngrid = 0
        self.aovalues = 0.0

        # density matrices
        self.dm = 0.0
        self.dm_up = 0.0
        self.dm_down = 0.0

        # densities
        self.rho_up = 0.0
        self.rho_down = 0.0
        self.rho_tot = 0.0

        # density derivarive's components
        self.dx_rho_up = 0.0
        self.dy_rho_up = 0.0
        self.dz_rho_up = 0.0
        self.dx_rho_down = 0.0
        self.dy_rho_down = 0.0
        self.dz_rho_down = 0.0        

        # gradient of the density
        self.gradrho_up = 0.0
        self.gradrho_down = 0.0
        # laplacian of the density
        self.laprho_up = 0.0
        self.laprho_down = 0.0
        # kinetic energy density
        self.tau_up = 0.0
        self.tau_down = 0.0

        # additional parameters
        self.D_up = 0.0
        self.D_down = 0.0

        # curvature of the exact exchange hole
        self.Q_up = 0.0
        self.Q_down = 0.0

        eps_x_exact_up = 0.0
        eps_x_exact_down = 0.0

    def CalculateKSKernel(self, mol):

        self.mf = scf.KS(mol)
        self.mf.small_rho_cutoff = 1e-12
        self.mf.grids.radi_method = dft.radi.delley
        self.mf.xc = 'pbe,pbe'
        # self.mf.verbose = 0
        self.mf.kernel()
        self.approx_xc = self.mf.get_veff().exc

        #for stuff related to grid
        self.coords = self.mf.grids.coords
        self.weights = self.mf.grids.weights
        self.ngrid = np.shape(self.coords)[0]
        self.aovalues = dft.numint.eval_ao(mol, self.coords, deriv=2) #deriv=2, since we get every info(tau,lap,etc)
        self.dm_up = np.zeros(self.ngrid)
        self.dm_down = np.zeros(self.ngrid)
        self.dm = self.mf.make_rdm1()
        self.dm_up = self.dm[0]

        if mol.spin == 0:
            self.dm_up = self.mf.make_rdm1(mo_occ=self.mf.mo_occ/2)
            self.dm_down = self.dm_up
            self.rho_up,self.dx_rho_up,self.dy_rho_up,self.dz_rho_up,self.laprho_up,self.tau_up = \
                                    dft.numint.eval_rho(mol, self.aovalues, self.dm_up, xctype="MGGA")
            self.gradrho_up = self.dx_rho_up**2+self.dy_rho_up**2+self.dz_rho_up**2
            self.D_up = self.tau_up*2-(1./4.)*self.gradrho_up/self.rho_up
            self.Q_up = 1./6.*(self.laprho_up-2.*self.D_up)
            self.rho_down=self.rho_up
            self.dx_rho_down=self.dx_rho_up
            self.dy_rho_down=self.dy_rho_up
            self.dz_rho_down=self.dz_rho_up
            self.laprho_down= self.laprho_up
            self.tau_down = self.tau_up
            self.D_down = self.D_up
            self.Q_down = self.Q_up
            
        else:

            self.rho_up,self.dx_rho_up,self.dy_rho_up,self.dz_rho_up,self.laprho_up,self.tau_up = \
                        dft.numint.eval_rho(mol, self.aovalues, self.dm_up, xctype="MGGA")
            self.gradrho_up = self.dx_rho_up**2+self.dy_rho_up**2+self.dz_rho_up**2
            self.D_up = self.tau_up*2-(1./4.)*self.gradrho_up/self.rho_up
            self.Q_up = 1./6.*(self.laprho_up-2.*self.D_up) 

            self.rho_down = np.zeros(self.ngrid)
            self.dx_rho_down = np.zeros(self.ngrid)
            self.dy_rho_down = np.zeros(self.ngrid)
            self.dz_rho_down = np.zeros(self.ngrid)
            self.gradrho_down = np.zeros(self.ngrid)
            self.laprho_down = np.zeros(self.ngrid)
            self.tau_down = np.zeros(self.ngrid)
            self.D_down = np.zeros(self.ngrid)
            self.Q_down = np.zeros(self.ngrid)

            if mol.nelectron > 1:
                self.dm_down = self.dm[1]
                self.rho_down, self.dx_rho_down, self.dy_rho_down, self.dz_rho_down, self.laprho_down, self.tau_down = \
                            dft.numint.eval_rho(mol, self.aovalues, self.dm_down, xctype="MGGA")
                self.gradrho_down = self.dx_rho_down**2+self.dy_rho_down**2+self.dz_rho_down**2
                self.D_down = self.tau_down*2-(1./4.)*self.gradrho_down/self.rho_down
                self.Q_down = 1./6.*(self.laprho_down-2.*self.D_down)

        # store some useful quantities
        self.rho = self.rho_up + self.rho_down
        self.zeta = (self.rho_up - self.rho_down) / self.rho
        self.kf = (3. * np.pi**2 * self.rho)**(1. / 3.)
        self.rs = (3. / (4. * np.pi * self.rho))**(1. / 3.)

        # future storage the exact exchange
        eps_x_exact_up = np.zeros(self.ngrid)
        eps_x_exact_down = np.zeros(self.ngrid)


    def GetDM(self):
        return self.dm

    def GetAOValues(self):
        return self.aovalues

    def GetCoords(self):
        return self.coords

    def GetWeights(self):
        return self.weights

    def GetParams(self):
        params_up = [self.rho_up, self.dx_rho_up, self.dy_rho_up, self.dz_rho_up, self.laprho_up, self.tau_up]
        params_down = [self.rho_down, self.dx_rho_down, self.dy_rho_down, self.dz_rho_down, self.laprho_down, self.tau_down]
        return params_up, params_down

    def GetParamUP(self):
        return [self.rho_up, self.dx_rho_up, self.dy_rho_up, self.dz_rho_up, self.laprho_up, self.tau_up]

    def GetParamDown(self):
        return [self.rho_down, self.dx_rho_down, self.dy_rho_down, self.dz_rho_down, self.laprho_down, self.tau_down]
