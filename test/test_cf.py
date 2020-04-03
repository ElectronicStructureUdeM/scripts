import sys
from timeit import default_timer as timer
# add code's directory
sys.path.insert(1, '../')

import numpy as np
from pyscf import gto

import kernel
from ksex import ExKS
from cfx import CFX

def main():
    
    systems = {
            # "H" : { 
            #     'name'         : 'Hydrogen',
            #     'symbol'       : 'H',
            #     'spin'         : 1,
            #     'positions'    : [[ 0., 0., 0.]]},
            "Li" : {
                'name'         : 'Lithium',
                'symbol'       : 'Li',
                'spin'         : 1,
                'positions'    : [[ 0., 0., 0.]]},
            "O" : {
                'name'         : 'Oxygen',
                'symbol'       : 'O',
                'spin'         : 2,
                'positions'    : [[ 0., 0., 0.]]},
            "Ar" : {
                'name'         : 'Argon',
                'symbol'       : 'Ar',
                'spin'         : 0,
                'positions'    : [[ 0., 0., 0.]]}
    }

    kskernel = kernel.KSKernel()

    for key in systems:

        system = systems[key]

        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(system['name'] + ' 0.0 0.0 0.0')
        coords = system['symbol'] + ' 0.0 0.0 0.0'

        mol = gto.Mole()
        mol.atom = coords
        mol.basis = 'cc-pvtz'
        mol.cart = True
        mol.spin = system['spin']
        mol.charge = 0
        mol.build()

        kskernel.CalculateKSKernel(mol)
        print('E = {:.12e}\tX = {:.12e}\tC = {:.12e}\tXC = {:.12e}'.format(kskernel.mf.e_tot, 0.0, 0.0, kskernel.mf.get_veff().exc))

        exks = ExKS(mol, kskernel, 'exks,')
        # x = exks.CalculateTotalX()
        # c = exks.CalculateTotalC()
        # xc = exks.CalculateTotalXC()
        # e = exks.CalculateTotalEnergy()
        # print('KS E = {:.12e}\tX = {:.12e}\tC = {:.12e}\tXC = {:.12e}'.format(e, x, c, xc))
        start = timer()
        # for now, we store this after we've processed kskernel.
        kskernel.exact_eps_x_up, kskernel.exact_eps_x_down = exks.CalculateEpsilonX()

        cfx = CFX(mol, kskernel)
        x = 0.0
        c = 0.0
        xc = 0.0
        # xc = cfx.CalculateTotalXC()
        e = cfx.CalculateTotalEnergy()
        end = timer()
        print('CFX E = {:.12e}\tX = {:.12e}\tC = {:.12e}\tXC = {:.12e} in '.format(e, x, c, xc, (end - start)))                
if __name__ == "__main__":
    main()
