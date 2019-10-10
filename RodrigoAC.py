import numpy as np
from pyscf import gto, dft
from pyscf.dft import numint
from pyscf import lib
from matplotlib import pyplot as plt
import sys
import re
from modelXC import ModelXC
from locEs import *
from mp2 import *
from BRx import *

#molecule = 'Li'

#mol = gto.Mole()
#mol.atoms = molecule
#model = ModelXC(molecule, None, 1, ASE = False, approx='PBE0',basis = 'ccpvtz')

def RodrigoAC(model):

    ex_up = np.zeros((model.n_grid))
    ex_down = np.zeros((model.n_grid))
    ec = np.zeros((model.n_grid))

    Ex = 0.0
    Ec = 0.0
    if model.mol.spin == 0 :
        GAMMP2 = myrmp2(model.mf,model.mol)
    else:
        GAMMP2 = GAMUMP2(model.mf,model.mol)

    for iG in range (0,model.n_grid):


        if model.rho_tot[iG] > 1e-8:
            ex_up[iG] = compute_ex_exact(model.mol,model.ao_values[0,iG,:],model.dm_up,model.coords[iG])
            ex_down[iG] = compute_ex_exact(model.mol,model.ao_values[0,iG,:],model.dm_down,model.coords[iG]) 
            ec[iG] ,mp2ot = calclocmp2(model.mol,model.ao_values[0,iG,:],GAMMP2,model.coords[iG])

            Ex += (ex_up[iG] + ex_down[iG]) * model.weights[iG]
            Ec += ec[iG] * model.weights[iG]

            rhoa = model.rho_up[iG]
            rhob = model.rho_down[iG]
            rhot = model.rho_tot[iG]

            zeta = (rhoa -rhob)/ rhot
            rs = (3.0/(4.0*np.pi*rhot))**(1/3.0)
    
    return ex_up,ex_down, ec



#print(CF3mod(model))
