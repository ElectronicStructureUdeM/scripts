
    class XHole():
        def __init__(self, KSKernel, eps_x_exact_up, eps_x_exact_down):
            self.kskernel = KSKernel
            self.eps_x_exact_up
            self.eps_x_exact_down

        def Solve(self):
            return NotImplementedError('Subclass specialization')

    class BRXN():
        def __init__(self, KSKernel, eps_x_exact_up, eps_x_exact_down):
            super().__init__(KSKernel, eps_x_exact_up, eps_x_exact_down)

        def Solve(self, params_up=None, params_down=None):
        """
        To calculate the parameters of the becke roussel normalized echange
        hole which reproduces an energy density from an approximation

        Input:
            rho: local electronic density
            eps_x: exchange energy density per particle to reproduce
        returns:
            a,b,c,d: Becke-Roussel model parameters
        """

        rho_up = params_up[0]
        rho_down = params_down[0]

        a,b,c,d = brxnparam(rho_up, eps_x_exact_up)
        kf = (3.0*(np.pi**2.0) *rho)**(1.0/3.0)
        a = a/kf
        b = b*kf
        c = c/rho
        d = d/kf**4

        return a,b,c,d