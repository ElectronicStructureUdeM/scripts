from pyscf import dft,lib,scf
import kernel

class CF():
    def __init__(self, mol, kskernel):

        self.mol = mol
        self.kskernel = kskernel

        self.aovalues = kskernel.GetAOValues()
        self.coords = kskernel.GetCoords()
        self.weights = kskernel.GetWeights()
        self.ngridpoints = self.weights.shape[0]
        self.params_up, self.params_down = kskernel.GetParams()

    @property
    def CalculateA(self, rs, zeta):
        return NotImplementedError('Subclass specialization')
    @property
    def CalculateB(self, rs, zeta):
        return NotImplementedError('Subclass specialization')
    @property
    def CalculateTotalXC(self, params_up = None, params_down = None):
        return NotImplementedError('Subclass specialization')