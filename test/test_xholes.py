import sys
# add code's directory
sys.path.insert(1, '../')

import numpy as np
from pyscf import gto

import kernel
from modelxc import ModelXC
from ksex import ExKS
from dfa import DFA
from ac import AC

from xhole import XHole
from B03_xhole import B03
from BRN_xhole import BRN

def main():

    functionals = ['LDA,PW_MOD', 'PBE,PBE']
    
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
        mol.basis = '../basis/6-311+g2dp.nw'
        mol.cart = True
        mol.spin = system['spin']
        mol.charge = 0
        mol.build()

        kskernel.CalculateKSKernel(mol)
        print('E = {:.12e}\tX = {:.12e}\tC = {:.12e}\tXC = {:.12e}'.format(kskernel.mf.e_tot, 0.0, 0.0, kskernel.mf.get_veff().exc))

        exks = ExKS(mol, kskernel, 'exks,')
        x = exks.CalculateTotalX()
        c = exks.CalculateTotalC()
        xc = exks.CalculateTotalXC()
        e = exks.CalculateTotalEnergy()
        print('KS E = {:.12e}\tX = {:.12e}\tC = {:.12e}\tXC = {:.12e}'.format(e, x, c, xc))

        # for now, we store this after we've processed kskernel.
        kskernel.exact_eps_x_up, kskernel.exact_eps_x_down = exks.CalculateEpsilonX()

        b03_xhole = B03(mol, kskernel)
        x = b03_xhole.CalculateTotalX()
        c = 0.0
        xc = x
        e = 0.0
        print('B03 E = {:.12e}\tX = {:.12e}\tC = {:.12e}\tXC = {:.12e}'.format(e, x, c, xc))

        brn_xhole = BRN(mol, kskernel)
        x = brn_xhole.CalculateTotalX()
        c = 0.0
        xc = x
        e = 0.0
        print('BRN E = {:.12e}\tX = {:.12e}\tC = {:.12e}\tXC = {:.12e}'.format(e, x, c, xc))
if __name__ == "__main__":
    main()
