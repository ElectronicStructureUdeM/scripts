from pyscf import dft,lib,scf

class ModelXC:
    def __init__(self, mol, aovalues, dm, coords, weights):
        self.mol = mol
        self.aovalues = aovalues
        self.dm = dm
        self.coords = coords
        self.weights = weights

    def SetMol(self, mol):
        self.mol = mol
    def SetAOValues(self, aovalues):
        self.aovalues = aovalues
    def SetDM(self, dm):
        self.dm = dm
    def SetCoords(self, coords):
        self.coords = coords
    def SetWeights(self, weights):
        self.weights = weights

    def CalculateEpsilonC(self, functional, params_up, params_down):

        eps_c = 0.0
        vc = 0.0
        eps_c, vc = dft.libxc.eval_xc("," + functional, [params_up, params_down], spin=5)[:2]
        return eps_c, vc

    def CalculateEpsilonX(self, functional, params):

        eps_x = 0.0
        vx = 0.0
        zeros = np.zeros(self.ngrid)
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


    def CalculateTotalX(self, functional, params_up, params_down):

        Ex_up = 0.0
        Ex_down = 0.0
        
        rho_up = params_up[0]
        rho_down = params_down[0]

        eps_x_up, vx_up = self.CalculateEpsilonX(functional, params_up)
        Ex_up = np.einsum("i,i,i->", eps_x_up, rho_up, self.weights)

        if np.all(self.rho_down) > 0.0:
            eps_x_down, vx_down = self.CalculateEpsilonX(functional, params_down)
            Ex_down = np.einsum("i,i,i->", eps_x_down, rho_down, self.weights)

        return Ex_up + Ex_down

    def CalculateTotalC(self, functional, params_up, params_down):
        
        rho_up = params_up[0]
        rho_down = params_down[0]
        rho = rho_up + rho_down

        eps_c,vc = self.CalculateEpsilonC(functional, params_up, params_down)
        return np.einsum("i,i,i->", eps_c, rho, self.weights)

    def CalculateTotalXC(self, functional, params_up, params_down):
        """
        To calculate the total exchange-correlation energy for a functional
        in a post-approx manner
        Input:
            functional:string
                functional in pyscf format
        """
        exchange_functional, correlation_functional = functional.split(",")
        Ex = 0.0
        Ec = 0.0

        Ex = self.CalculateTotalX(exchange_functional, params_up, params_down)
        Ec = self.CalculateTotalC(correlation_functional, params_up, params_down)

        return (Ex + Ec)

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
