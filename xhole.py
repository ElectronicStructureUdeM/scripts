import numpy as np

class XHole():
    """
    Base class for exchange hole models
    """
    def __init__(self):

        #for the integration grid
        self.min_y = 0.0
        self.max_y = 100.0
        self.gauleg_deg = 2000
        self.y_grid, self.y_weights =  np.polynomial.legendre.leggauss(self.gauleg_deg)
        self.y_values = 0.5 * (self.y_grid + 1) * (self.max_y - self.min_y) + self.min_y
        self.y_weights = self.y_weights * 0.5*(self.max_y - self.min_y)

        self.y_values_power = {1:self.y_values, 2:self.y_values**2, 3:self.y_values**3,
                              4:self.y_values**4, 5:self.y_values**5, 6:self.y_values**6}

    @property
    def SolveSigma(self):
        return NotImplementedError('Subclass specialization')

    @property
    def SolveOnGrid(self):
        return NotImplementedError('Subclass specialization')

    @property
    def CalculateTotalX(self):
        return NotImplementedError('Subclass specialization')