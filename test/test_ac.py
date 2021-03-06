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

def main():

    systems = {
            "H" : { 
                'name'         : 'Hydrogen',
                'symbol'       : 'H',
                'spin'         : 1,
                'positions'    : [[ 0., 0., 0.]]},
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
        mol.spin = system['spin']
        mol.charge = 0
        mol.build()

        kskernel.CalculateKSKernel(mol)

        print('PBE:')
        pbe = DFA(mol, kskernel, 'PBE,PBE')
        xc = pbe.CalculateTotalXC()
        pbepade3p = AC(pbe, 'pade3p')
        ac_xc = pbepade3p.CalculateTotalXC()
        print('AC XC = {:.12e} AVG XC = {:.12e}, Error = {:.12e}'.format(ac_xc, xc, (xc - ac_xc)))

        print('Exact KS:')
        exks = ExKS(mol, kskernel, 'exks,')
        xc = exks.CalculateTotalXC()
        exks_ac = AC(exks, 'pade3p')
        ac_xc = exks_ac.CalculateTotalXC()
        print('AC XC = {:.12e} AVG XC = {:.12e}, Error = {:.12e}'.format(ac_xc, xc, (xc - ac_xc)))


    return

if __name__ == "__main__":
    main()
