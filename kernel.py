from pyscf import dft,gto,lib,scf
import numpy as np
from numba import vectorize,float64

from scipy.integrate import quad
from scipy import interpolate

import time

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

    def CalculateKSKernel(self, mol, functional):

        self.mf = scf.UKS(mol)
        self.mf.small_rho_cutoff = 1e-12
        self.mf.grids.radi_method = dft.radi.delley
        self.mf.xc = functional
        self.mf.kernel()

        self.approx_Exc = self.mf.get_veff().exc

        #for stuff related to grid
        self.coords = self.mf.grids.coords
        self.weights = self.mf.grids.weights
        self.ngrid = np.shape(self.coords)[0]
        self.aovalues = dft.numint.eval_ao(mol, self.coords, deriv=2) #deriv=2, since we get every info(tau,lap,etc)

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

            dm = self.mf.make_rdm1()

            self.dm_up = dm[0]
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
                self.dm_down = dm[1]
                self.rho_down, self.dx_rho_down, self.dy_rho_down, self.dz_rho_down, self.laprho_down, self.tau_down = \
                            dft.numint.eval_rho(mol, self.aovalues, self.dm_down, xctype="MGGA")
                self.gradrho_down = self.dx_rho_down**2+self.dy_rho_down**2+self.dz_rho_down**2
                self.D_down = self.tau_down*2-(1./4.)*self.gradrho_down/self.rho_down
                self.Q_down = 1./6.*(self.laprho_down-2.*self.D_down)

        self.rho_tot = self.rho_up + self.rho_down
        self.zeta = (self.rho_up - self.rho_down) / self.rho_tot
        self.kf = (3. * np.pi**2 * self.rho_tot)**(1. / 3.)
        self.rs = (3. / (4. * np.pi * self.rho_tot))**(1. / 3.)

    def GetCoords(self):
        return self.coords

    def GetWeights(self):
        return self.weights

    def GetRho(self):
        return (self.rho_up + self.rho_down)

    def GetParams(self):
        mgga_up = [self.rho_up, self.dx_rho_up, self.dy_rho_up, self.dz_rho_up, self.laprho_up, self.tau_up]
        mgga_down = [self.rho_down, self.dx_rho_down, self.dy_rho_down, self.dz_rho_down, self.laprho_down, self.tau_down]
        return mgga_up, mgga_down

    def ScaleParams(self, value):
        self.rho_up     = value**-3.0 * self.rho_up
        self.rho_down   = value**-3.0 * self.rho_down

    def CalculateEpsilonC(self, functional, params):

        params_up, params_down = params
        eps_c = 0.0
        vc = 0.0
        eps_c, vc = dft.libxc.eval_xc("," + functional, [params_up, params_down], spin=5)[:2]
        return eps_c, vc

    def CalculateEpsilonX(self, functional, params):

        eps_x = 0.0
        vx = 0.0
        zeros = np.zeros(self.ngrid) # also used so we can exact up and down energies densites
        eps_x, vx = dft.libxc.eval_xc(functional + ",", [params, [zeros, zeros, zeros, zeros, zeros, zeros]], spin=5)[:2]
        return eps_x, vx

    def CalculateEpsilonXC(self, functional, params_up, params_down):
        """
        This function calculates the exchange-correlation energy densities
        from a functional with converged self-consistant  densities
        Warning: both exchange and correlation must be specified ie. pbe,pbe and not pbe
        Input:
            functional:string
                functional name in pyscf format
        TODO:
            For spin unpolarized, the calculation are done uselessly for down spin
        """
        #here spin is defined as greater than one so we can exact up and down energies densites
        exchange_functional, correlation_functional = functional.split(",")
        zeros = np.zeros(self.ngrid) # also used so we can exact up and down energies densites

        eps_x_up, vx_up = self.CalculateEpsilonX(exchange_functional, params_up)
        # eps_x_up,vx_up = dft.libxc.eval_xc(exchange_functional+",", [mgga_up,[zeros,zeros,zeros,zeros,zeros,zeros]],spin=5)[:2]

        eps_x_down, vx_down = self.CalculateEpsilonX(exchange_functional, params_down)
        # eps_x_down,vx_down = dft.libxc.eval_xc(exchange_functional+",", [[zeros,zeros,zeros,zeros,zeros,zeros],mgga_down],spin=5)[:2]
        
        eps_c,vc = dft.libxc.eval_xc("," + correlation_functional, [params_up, params_down], spin = 5)[:2]

        return eps_x_up, vx_up, eps_x_down, vx_down, eps_c, vc


    def CalculateTotalXC(self, functional, params_up, params_down):
        """
        To calculate the total exchange-correlation energy for a functional
        in a post-approx manner
        Input:
            functional:string
                functional in pyscf format
        """
        Ex_up = 0.0
        Ex_down = 0.0
        Ec = 0.0

        eps_x_up, vx_up, eps_x_down, vx_down, eps_c, vc = self.CalculateEpsilonXC(functional, params_up, params_down)
        Ex_up   = np.einsum("i,i,i->", eps_x_up, self.rho_up, self.weights)

        if np.all(self.rho_down) > 0.0:
            Ex_down = np.einsum("i,i,i->", eps_x_down, self.rho_down, self.weights)
        
        Ec = np.einsum("i,i,i->", eps_c, self.rho_tot, self.weights)
        print(Ex_up, Ex_down, Ec)
        return (Ex_up + Ex_down + Ec)

    def CalculateTotalEnergy(self, functional, params_up, params_down):
        """
        To calculate the total energies of a functional
        with post-approx densities

        Input:
            functional:string
                functional name in pyscf format
        """
        
        Exc = self.CalculateTotalXC(functional, params_up, params_down)
        return self.mf.e_tot + Exc

def ScaledXC(lm):

    params_up, params_down = kernel.GetParams()
    rho_up = params_up[0]
    rho_down = params_down[0]

    x_up = kernel.CalculateEpsilonX('LDA', params_up)[0]
    if np.all(rho_down) > 0.0:
        x_down = kernel.CalculateEpsilonX('LDA', params_down)[0]
    c = kernel.CalculateEpsilonC('PW_mod', [params_up, params_down])[0]

    xpp_up = x_up / rho_up
    xpp_down = np.zeros(xpp_up.shape[0])
    if np.all(rho_down) > 0.0:
        xpp_down = x_down / rho_down
    cpp = c / (rho_up + rho_down)
    return (xpp_up + xpp_down + cpp)

mol = gto.Mole()
mol.atom = 'H 0 0 0'
mol.basis = 'cc-pvtz'
mol.spin = 1
mol.charge = 0
mol.build()

kernel = KSKernel()
kernel.CalculateKSKernel(mol, 'pbe,pbe')

params_up, params_down = kernel.GetParams()
weights = kernel.GetWeights()
rho = kernel.GetRho()

TotalXC = kernel.CalculateTotalXC('LDA,PW_mod', params_up, params_down)
print('XC = {:.12e}'.format(TotalXC))

###############################################################################
def ScaleXC(lm):
    kernel.ScaleParams(lm)
    params_up, params_down = kernel.GetParams()
    rho_up = params_up[0]
    rho_down = params_down[0]
    x_up = kernel.CalculateEpsilonX('LDA', params_up)[0]
    if np.all(rho_down) > 0.0:
        x_down = kernel.CalculateEpsilonX('LDA', params_down)[0]
    c = kernel.CalculateEpsilonC('PW_mod', [params_up, params_down])[0]
    xpp_up = x_up / rho_up
    xpp_down = np.zeros(xpp_up.shape[0])
    if np.all(rho_down) > 0.0:
        xpp_down = x_down / rho_down
    cpp = c / (rho_up + rho_down)
    xcpp = (xpp_up + xpp_down + cpp)
    return xcpp
###############################################################################

# prepare scaling parameters
lmbd = np.arange(0.0, 1.0, 0.1)
npoints = len(lmbd)
lmbd[0] = 1.0e-3
delta = 1.0e-4

# prepare matrix for vectorization ( weights, rho, excpp )
data = np.ndarray(shape=(weights.shape[0], 3), dtype=float)
data[:,0] = weights.T 
data[:,1] = rho.T
data[:,2] = np.zeros(rho.shape[0])

# weight, rho, xcpp = data # unpack the data from the ith-row

def XCPPLM():
  
    xcpplm = np.ndarray(shape=(weights.shape[0], npoints), dtype=float) # an array of xc per particle lambda dependent with shape n,npoints
                                                                        # where n is the number of grid points
    # Calculate the xc per particle for every point
    for i, lm in enumerate(lmbd): # ToDo check if enumerate is fast enough with arange

        # This block is for f(l + d)
        lmpd = lm + delta
        xc_lmpd =  lmpd * lmpd * ScaledXC(lmpd)

        # This block is for f(l - d)
        lmmd = lm - delta
        xc_lmmd = lmmd * lmmd * ScaledXC(lmmd)

        # store the xc per particle lambda dependent in the array xcpplm
        xcpplm[:,i] = (xc_lmpd - xc_lmmd) / (2.0 * delta)

    return xcpplm 

def XCPP(xcpplm):

    # Calculate the interpolation
    tck = interpolate.splrep(lmbd, xcpplm, k=3, s=0.) # k is the degree
    xnew = np.linspace(lmbd[0], 1.0, num=10000) # prepare x-axis
    yinterp = interpolate.splev(xnew, tck, der=0) # interpolate the ac
    xcavg = interpolate.splint(1.0e-8, 1.0e0, tck) # integrate over lambda
    return xcavg

times = np.zeros(shape=(10,1))
for i in enumerate(arange(0, 9)):
    start_time = time.time()
    xcpplm_arr = XCPPLM()
    times((time.time() - start_time))

eps_xcavg = np.apply_along_axis(XCPP, 1, xcpplm_arr)

# eps_xcavg = np.array(
#     [XCPP(row) for row in xcpplm_arr]
# )

print(eps_xcavg)
# xclm = np.sum(weights * rho * eps_xcavg)
# print('XC = {:.12e}'.format(xclm))

