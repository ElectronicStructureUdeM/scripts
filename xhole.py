
class XHole():

    @property
    def SolveSigma(self):
        return NotImplementedError('Subclass specialization')

    @property
    def SolveOnGrid(self):
        return NotImplementedError('Subclass specialization')

    @property
    def CalculateTotalX(self):
        return NotImplementedError('Subclass specialization')