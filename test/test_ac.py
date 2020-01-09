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

        print(system['name'] + ' 0.0 0.0 0.0')
        coords = system['symbol'] + ' 0.0 0.0 0.0'

        mol = gto.Mole()
        mol.atom = coords
        mol.basis = 'cc-pvtz'
        mol.spin = system['spin']
        mol.charge = 0
        mol.build()

        kskernel.CalculateKSKernel(mol)

        pbe = DFA(mol, kskernel, 'PBE,PBE')
        pbepade3p = AC(pbe, 'pade3p')
        pbepade3p_xc = pbepade3p.CalculateTotalXC()

        exks = ExKS(mol, kskernel)
        exks_ac = AC(exks, 'pade3p')
        exks_xc = exks_ac.CalculateTotalXC()

        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    return

if __name__ == "__main__":
    main()