import numpy as np
from pyscf import dft,lib,scf

import kernel
from modelxc import ModelXC

class DFA(ModelXC):
    
    def __init__(self, mol, KSKernel, functional):
        super().__init__(mol, KSKernel, functional)

    def CalculateEpsilonC(self, params_up, params_down):

        if params_up is None and params_down is None:
            params_up, params_down = self.kskernel.GetParams()

        eps_c = 0.0
        vc = 0.0

        eps_c, vc = dft.libxc.eval_xc("," + self.correlation_functional, [params_up, params_down], spin=5)[:2]
        return eps_c, vc

    def CalculateEpsilonX(self, params_up, params_down):

        if params_up is None and params_down is None:
            params_up, params_down = self.kskernel.GetParams()

        eps_x_up = 0.0
        eps_x_down = 0.0
        vx_up = 0.0
        vx_down = 0.0
        zeros = np.zeros(self.ngridpoints)
        eps_x_up, vx_up = dft.libxc.eval_xc(self.exchange_functional + ",", [params_up, [zeros, zeros, zeros, zeros, zeros, zeros]], spin=5)[:2]

        eps_x_down, vx_down = dft.libxc.eval_xc(self.exchange_functional + ",", [params_down, [zeros, zeros, zeros, zeros, zeros, zeros]], spin=5)[:2]
        return eps_x_up, vx_up, eps_x_down, vx_down

    def CalculateEpsilonXC(self, params_up, params_down):
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
        if params_up is None and params_down is None:
            params_up, params_down = self.kskernel.GetParams()

        #here spin is defined as greater than one so we can exact up and down energies densites
        zeros = np.zeros(self.ngridpoints) # also used so we can exact up and down energies densites

        eps_x_up, vx_up, eps_x_down, vx_down = self.CalculateEpsilonX(self.exchange_functional, params_up)
        
        eps_c,vc = dft.libxc.eval_xc("," + self.correlation_functional, [params_up, params_down], spin = 5)[:2]

        return eps_x_up, vx_up, eps_x_down, vx_down, eps_c, vc

    def CalculateTotalX(self, params_up, params_down):

        if params_up is None and params_down is None:
            params_up, params_down = self.kskernel.GetParams()

        Ex_up = 0.0
        Ex_down = 0.0

        rho_up = params_up[0]
        rho_down = params_down[0]

        eps_x_up, vx_up, eps_x_down, vx_down = self.CalculateEpsilonX(params_up, params_down)
        Ex_up = np.einsum("i,i,i->", eps_x_up, rho_up, self.weights)

        if np.all(rho_down) > 0.0:
            Ex_down = np.einsum("i,i,i->", eps_x_down, rho_down, self.weights)

        return Ex_up + Ex_down

    def CalculateTotalC(self, params_up, params_down):

        if params_up is None and params_down is None:
            params_up, params_down = self.kskernel.GetParams()

        rho_up = params_up[0]
        rho_down = params_down[0]
        rho = rho_up + rho_down

        eps_c,vc = self.CalculateEpsilonC(params_up, params_down)
        return np.einsum("i,i,i->", eps_c, rho, self.weights)

    def CalculateTotalXC(self, params_up, params_down):
        """
        To calculate the total exchange-correlation energy for a functional
        in a post-approx manner
        Input:
            functional:string
                functional in pyscf format
        """

        if params_up is None and params_down is None:
            params_up, params_down = self.kskernel.GetParams()

        Ex = 0.0
        Ec = 0.0

        Ex = self.CalculateTotalX(params_up, params_down)
        Ec = self.CalculateTotalC(params_up, params_down)

        return (Ex + Ec)

    def CalculateTotalEnergy(self, params_up, params_down):
        """
        To calculate the total energies of a functional
        with post-approx densities

        Input:
            functional:string
                functional name in pyscf format
        """

        if params_up is None and params_down is None:
            params_up, params_down = self.kskernel.GetParams()

        Exc = self.CalculateTotalXC(params_up, params_down)
        return self.mf.e_tot + Exc