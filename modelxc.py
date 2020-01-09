from pyscf import dft,lib,scf

import kernel

class ModelXC():
    def __init__(self, mol, kskernel, functional):

        self.mol = mol
        self.kskernel = kskernel

        self.aovalues = kskernel.GetAOValues()
        self.dm = kskernel.GetDM()
        self.coords = kskernel.GetCoords()
        self.weights = kskernel.GetWeights()
        self.ngridpoints = self.weights.shape[0]
        self.params_up, self.params_down = kskernel.GetParams()

        self.functional = functional
        self.exchange_functional, self.correlation_functional = functional.split(",")

    @property
    def CalculateEpsilonC(self, functional, params_up, params_down):
        return NotImplementedError('Subclass specialization')

    @property
    def CalculateEpsilonX(self, params_up, params_down):
        return NotImplementedError('Subclass specialization')

    @property
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
        return NotImplementedError('Subclass specialization')

    @property
    def CalculateTotalX(self, functional, params_up, params_down):
        return NotImplementedError('Subclass specialization')

    @property
    def CalculateTotalC(self, functional, params_up, params_down):
        return NotImplementedError('Subclass specialization')        

    @property
    def CalculateTotalXC(self, functional, params_up, params_down):
        """
        To calculate the total exchange-correlation energy for a functional
        in a post-approx manner
        Input:
            functional:string
                functional in pyscf format
        """
        return NotImplementedError('Subclass specialization')

    @property
    def CalculateTotalEnergy(self, functional, params_up, params_down):
        """
        To calculate the total energies of a functional
        with post-approx densities

        Input:
            functional:string
                functional name in pyscf format
        """
        return NotImplementedError('Subclass specialization')        