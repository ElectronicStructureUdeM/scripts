import numpy as np
import sys
# add code's directory
sys.path.insert(1, '../')

import kernel
from modelxc import ModelXC
from exks import ExKS
from dfa import DFA

def main():

    functionals = ['LDA,PW_MOD', 'PBE,PBE']
    
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

        ex = ExKS(mol, kskernel, 'exks')
        exks = ex.CalculateEpsilonX()

        lsd = DFA(mol, kskernel, 'LDA,PW_MOD')
        lsd_xc = lsd.CalculateEpsilonXC()

        pbe = DFA(mol, kskernel, 'PBE,PBE')
        pbe_xc = pbe.CalculateEpsilonXC()

        cfx = CF('cfx', mol, kernel)
        cfx_xc = cfx.CalculateEpsilonXC()

    return

if __name__ == "__main__":
    main()