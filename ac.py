import os
import time

import numpy as np
from scipy.integrate import quad
from scipy import interpolate
from pyscf import gto
import matplotlib.pyplot as plt

import kernel


class AC:
    def __init__(self, mol, kskernel, dfa, approximation):


###############################################################################
def ScaleXC(functional, lm):
    """
        This function scales all parameters (rho, gradrho, laprho, tau, ... ) for functionals
        and computes the exchange-correlation energy per particle
        Input: lambda (scaling) value
        Output: xc energy per particle
    """
    exchange_functional, correlation_functional = functional.split(",")
    params_up, params_down = kskernel.GetParams()

    rho_up, dx_rho_up, dy_rho_up, dz_rho_up, laprho_up, tau_up = params_up
    rho_down, dx_rho_down, dy_rho_down, dz_rho_down, laprho_down, tau_down = params_down
    
    scaled_rho_up = lm**(-3.0) * rho_up
    scaled_dx_rho_up = lm**(-8.0) * dx_rho_up
    scaled_dy_rho_up = lm**(-8.0) * dy_rho_up
    scaled_dz_rho_up = lm**(-8.0) * dz_rho_up
    scaled_laprho_up = lm**(-5.0) * laprho_up
    scaled_tau_up = lm**(-5.0) * tau_up

    scaled_rho_down = lm**(-3.0) * rho_down
    scaled_dx_rho_down = lm**(-5.0) * dx_rho_down
    scaled_dx_rho_down = lm**(-5.0) * dx_rho_down
    scaled_dx_rho_down = lm**(-5.0) * dx_rho_down
    scaled_laprho_down = lm**(-5.0) * laprho_down
    scaled_tau_down = lm**(-5.0) * tau_down

    params_up = scaled_rho_up, scaled_dx_rho_up, scaled_dy_rho_up, scaled_dz_rho_up, scaled_laprho_up, scaled_tau_up
    params_down = scaled_rho_down, scaled_dx_rho_down, scaled_dy_rho_down, scaled_dz_rho_down, scaled_laprho_down, scaled_tau_down

    x_up = kskernel.CalculateEpsilonX(exchange_functional, params_up)[0]
    # xpp_up = x_up / scaled_rho_up

    x_down = np.zeros(x_up.shape[0])
    if np.all(rho_down) > 0.0:
        x_down = kskernel.CalculateEpsilonX(exchange_functional, params_down)[0]
        # xpp_down = x_down / scaled_rho_down
        
    c = kskernel.CalculateEpsilonC(correlation_functional, params_up, params_down)[0]
    
    xc = (x_up * rho_up + x_down * rho_down) / (rho_up + rho_down) + c

    return xc
###############################################################################
def XCPPLM(ngridpoints, npoints):

    xcpplm = np.zeros(shape=(ngridpoints, npoints), dtype=float) # an array of xc per particle lambda dependent with shape n,npoints
                                                                        # where n is the number of grid points
    # Calculate the xc per particle for every point
    for i, lm in enumerate(lmbd): # ToDo check if enumerate is fast enough with arange

        # This block is for f(l + d)
        lmpd = lm + delta
        xc_lmpd =  lmpd * lmpd * ScaleXC(lmpd)            

        if i == 0:
            xc_lm = lm * lm * ScaleXC(lm)
            xcpplm[:,i] = (xc_lmpd - xc_lm) / (delta)
        else:
            # This block is for f(l - d)
            lmmd = lm - delta
            xc_lmmd = lmmd * lmmd * ScaleXC(lmmd)
            # store the xc per particle lambda dependent in the array xcpplm
            xcpplm[:,i] = (xc_lmpd - xc_lmmd) / (2.0 * delta)
    
    return xcpplm 
###############################################################################
def XCPP(xcpplm):
    # Calculate the interpolation
    tck = interpolate.splrep(lmbd, xcpplm, k=3, s=0.) # k is the degree
    xnew = np.linspace(lmbd[0], 1.0, num=10000) # prepare x-axis
    yinterp = interpolate.splev(xnew, tck, der=0) # interpolate the ac
    # if np.abs(xcpplm[-1]) > 800.0:
    # print(xcpplm, lmbd[-1])
    # plt.plot(xnew, yinterp, label = "CFX")
    # plt.show(block = False)
    # plt.pause(0.000001)
    xcavg = interpolate.splint(lmbd[0], 1.0, tck) # integrate over lambda
    return xcavg
###############################################################################        

def CalculateTotalXC(self, functional, params_up, params_down):
    
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    
    TotalX = kskernel.CalculateTotalX('LDA', params_up, params_down)
    TotalC = kskernel.CalculateTotalC('LDA_C_PW_MOD', params_up, params_down)
    TotalXC = kskernel.CalculateTotalXC('LDA,LDA_C_PW_MOD', params_up, params_down)

    print('X = {:.12e}\tC = {:.12e}\tXC = {:.12e}'.format(TotalX, TotalC, TotalXC))

    # prepare scaling parameters
    lmbd = np.arange(0.0, 0.2, 0.02)
    lmbd = np.append(lmbd, np.arange(0.2, 1.05, 0.05))
    npoints = len(lmbd)

    lmbd[0] = 1.0e-10
    delta = 1.0e-11

    xcpplm_arr = XCPPLM(weights.shape[0], npoints)

    eps_xcavg = np.array(
        [XCPP(row) for row in xcpplm_arr]
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

    xclm = np.sum(weights * rho * eps_xcavg)
    print('AC XC = {:.12e} AVG XC = {:.12e}, Error = {:.12e}'.format(xclm, TotalXC, (TotalXC - xclm)))

    return xclm

def main():

    functionals = ['LDA,PW_MOD', 'PBE,PBE']
    kskernel = kernel.KSKernel()

    mol = gto.Mole()
    mol.atom = 'Ar 0 0 0'
    mol.basis = 'cc-pvtz'
    mol.spin = 0
    mol.charge = 0
    mol.build()

    kskernel.CalculateKSKernel(mol, functional)

    params_up, params_down = kskernel.GetParams()
    weights = kskernel.GetWeights()
    rho = kskernel.GetRho()

    return

if __name__ == "__main__":
    main()