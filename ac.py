import os
import time

import numpy as np
from scipy.integrate import quad
from scipy import interpolate
from pyscf import gto
import matplotlib.pyplot as plt

import kernel
from modelxc import ModelXC


class AC:
    def __init__(self, df, approximation):
        self.df = df
        self.kskernel = df.kskernel # pointer to kskernel
        self.approximation = approximation

        # prepare scaling parameters
        self.lmbd = np.arange(0.0, 0.2, 0.02)
        self.lmbd = np.append(self.lmbd, np.arange(0.2, 1.05, 0.05))
        self.npoints = len(self.lmbd)

        self.lmbd[0] = 1.0e-10
        self.delta = 1.0e-11

    ###############################################################################
    def ScaleXC(self, lm):
        """
            This function scales all parameters (rho, gradrho, laprho, tau, ... ) for functionals
            and computes the exchange-correlation energy per particle
            Input: lambda (scaling) value
            Output: xc energy per particle
        """
        params_up, params_down = self.kskernel.GetParams()

        rho_up, dx_rho_up, dy_rho_up, dz_rho_up, laprho_up, tau_up = params_up
        rho_down, dx_rho_down, dy_rho_down, dz_rho_down, laprho_down, tau_down = params_down
        
        # scale the parameters with spin alpha
        scaled_rho_up = lm**(-3.0) * rho_up
        scaled_dx_rho_up = lm**(-8.0) * dx_rho_up
        scaled_dy_rho_up = lm**(-8.0) * dy_rho_up
        scaled_dz_rho_up = lm**(-8.0) * dz_rho_up
        scaled_laprho_up = lm**(-5.0) * laprho_up
        scaled_tau_up = lm**(-5.0) * tau_up

        # scale the parameters with spin beta
        scaled_rho_down = lm**(-3.0) * rho_down
        scaled_dx_rho_down = lm**(-5.0) * dx_rho_down
        scaled_dy_rho_down = lm**(-5.0) * dy_rho_down
        scaled_dz_rho_down = lm**(-5.0) * dz_rho_down
        scaled_laprho_down = lm**(-5.0) * laprho_down
        scaled_tau_down = lm**(-5.0) * tau_down

        # put the scaled paramters back into the packages for calculation
        params_up = scaled_rho_up, scaled_dx_rho_up, scaled_dy_rho_up, scaled_dz_rho_up, scaled_laprho_up, scaled_tau_up
        params_down = scaled_rho_down, scaled_dx_rho_down, scaled_dy_rho_down, scaled_dz_rho_down, scaled_laprho_down, scaled_tau_down

        # x_up = self.df.CalculateEpsilonX(params_up, params_down)[0]
        eps_x_up, eps_x_down = self.df.CalculateEpsilonX(params_up, params_down)
            
        eps_c = self.df.CalculateEpsilonC(params_up, params_down)
        
        xc = (eps_x_up * rho_up + eps_x_down * rho_down) / (rho_up + rho_down) + eps_c

        return xc

    ###############################################################################
    def XCPPLM(self, ngridpoints, npoints):

        xcpplm = np.zeros(shape=(ngridpoints, npoints), dtype=float) # an array of xc per particle lambda dependent with shape n,npoints
                                                                            # where n is the number of grid points
                                                                            
        # ToDo can this be split over multiple cores and nodes ?
        # Calculate the xc per particle for every point
        for i, lm in enumerate(self.lmbd): # ToDo check if enumerate is fast enough with arange

            # This block is for f(l + d)
            lmpd = lm + self.delta
            xc_lmpd = lmpd * lmpd * self.ScaleXC(lmpd)            

            if i == 0:
                xc_lm = lm * lm * self.ScaleXC(lm)
                xcpplm[:,i] = (xc_lmpd - xc_lm) / (self.delta)
            else:
                # This block is for f(l - d)
                lmmd = lm - self.delta
                xc_lmmd = lmmd * lmmd * self.ScaleXC(lmmd)
                # store the xc per particle lambda dependent in the array xcpplm
                xcpplm[:,i] = (xc_lmpd - xc_lmmd) / (2.0 * self.delta)
        
        return xcpplm 
    ###############################################################################
    def XCPP(self, xcpplm):
        # Calculate the interpolation
        tck = interpolate.splrep(self.lmbd, xcpplm, k=3, s=0.) # k is the degree
        xnew = np.linspace(self.lmbd[0], 1.0, num=10000) # prepare x-axis
        yinterp = interpolate.splev(xnew, tck, der=0) # interpolate the ac
        # if np.abs(xcpplm[-1]) > 800.0:
        # print(xcpplm, lmbd[-1])
        # plt.plot(xnew, yinterp, label = "CFX")
        # plt.show(block = False)
        # plt.pause(0.000001)
        xcavg = interpolate.splint(self.lmbd[0], 1.0, tck) # integrate over lambda
        return xcavg
    ###############################################################################        

    def CalculateTotalXC(self):
        
        params_up, params_down = self.kskernel.GetParams()

        # TotalX = self.df.CalculateTotalX(params_up, params_down)
        # TotalC = self.df.CalculateTotalC(params_up, params_down)
        # TotalXC = self.df.CalculateTotalXC(params_up, params_down)

        # print('X = {:.12e}\tC = {:.12e}\tXC = {:.12e}'.format(TotalX, TotalC, TotalXC))

        xcpplm_arr = self.XCPPLM(self.df.kskernel.weights.shape[0], self.npoints) # ToDo change access to ngridpoints

        eps_xcavg = np.array(
            [self.XCPP(row) for row in xcpplm_arr]
        )

        # times = np.zeros(shape=(10,1))
        # for i, j in enumerate(np.arange(0, 9)):
        #     start_time = time.time()
        #     # eps_xcavg = np.apply_along_axis(XCPP, 1, xcpplm_arr)
        #     # eps_xcavg = np.array(
        #     #     [XCPP(row) for row in xcpplm_arr]
        #     # )    
        #     times[i] = ((time.time() - start_time))
        #     print(times[i])
        # print(times.sum()/10.0)

        xclm = np.sum(self.df.kskernel.weights * self.df.kskernel.rho * eps_xcavg)

        return xclm