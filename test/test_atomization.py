import sys
import re
# add code's directory
sys.path.insert(1, '../')

import numpy as np
from pyscf import gto

import kernel
from modelxc import ModelXC
from ksex import ExKS
from dfa import DFA
from ac import AC

MGAtomEnergie = [-0.500, -7.478, -14.667, -37.845,-54.589, -75.067, -99.734, -341.259, -460.184]

# 109.000
# 58.000
# 419.000
# 297.000
# 232.000
# 141.000
# 24.000
# 139.000
# 3.000
# 405.000
# 563.000
# 312.000
# 259.000
# 229.000
# 153.000
# 121.000
# 39.000
# 117.000
# 58.000

molecules = {
        "H2" : { 
            'name'          : 'Hydrogen Molecule',
            'symbol'        : 'HH',
            'spin'          : 0,
            'energy'        : 0.0,
            'atomization'   : 109.000,
            'positions'     : [
                                [0, 0, -0.0259603084], 
                                [0, 0, 0.7259603084]]
                },
        
        "H2O" : {
            'name'          : 'Water Molecule',
            'symbol'        : 'HOH',
            'spin'          : 0,
            'energy'        : 0.0,
            'atomization'   : 232.000,
            'positions'     : [
                                [-0.7435290312,-0.0862560218,-0.2491318075],
                                [0.2269625234,-0.0687025898,-0.2099668601],
                                [ -1.0265534922,0.2938386117,0.5988786675]]
                },
        "O2" : {
            'name'          : 'Oxygen Molecule',
            'symbol'        : 'OO',
            'spin'          : 2,
            'energy'        : 0.0,
            'atomization'   : 121.000,
            'positions'     : [
                                [0, 0, -0.0114390797],
                                [0, 0, 1.2114390797]]
                }            
}

AtomsSpin = {
            "X"     :   -1, 
            "H"     :   1,
            "He"    :   0,
            "Li"    :   1,
            "Be"    :   0,
            "B"     :   1,
            "C"     :   2,
            "N"     :   3,
            "O"     :   2,
            "F"     :   1,
            "Ne"    :   0,
            "Na"    :   1,
            "Ne"    :   0,
            "Na"    :   1,
            "Mg"    :   0,
            "Al"    :   1,
            "Si"    :   2,
            "P"     :   3,
            "S"     :   2,
            "Cl"    :   1,
            "Ar"    :   0}


def PrepareSystems(molecules):

    systems = {}

    # prepare systems for calculation
    for key in molecules:

        molecule = molecules[key]

        # add molecule to dictionary system
        systems[key] = molecule
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # print(molecule['symbol'])
        # print(system['name'] + ' 0.0 0.0 0.0')
        # coords = system['symbol'] + ' 0.0 0.0 0.0'

        atoms = re.findall('[A-Z][^A-Z]*', molecule['symbol']) # to split at each uppercase using regular expression

        # add atoms to the dictionary system
        for atom in atoms:
            systems[atom] =  {
                'name'          : 'TTT',
                'symbol'        : atom,
                'spin'          : AtomsSpin[atom],
                'energy'        : 0.0,
                'atomization'   : 0.0,
                'positions'     : [
                                    [0.0, 0.0, 0.0]]
    
                        }
    return systems

def MakeMoleculePositions(molecule, positions):

    atoms = re.findall('[A-Z][^A-Z]*', molecule)
    molecule = []
    nAtom = 0
    for atom in atoms:
        atom_position = positions[nAtom]
        molecule.append([atom, (atom_position[0], atom_position[1], atom_position[2])])
        nAtom=nAtom+1

    return molecule

def CalculateTotalEnergies(systems, functionals, pval):
        
    systems_names = []
    for key in systems:
        system = systems[key]
        systems_names.append(key)

    systems_energies = []

    # ar_es:     array of total energies
    ar_es = np.ndarray((len(systems_names), len(functionals)), dtype=object)
    ar_es[:,0] = systems_names

    kskernel = kernel.KSKernel()
    for key in systems:

        system = systems[key]

        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(system)
        
        mol = gto.Mole()
        print(MakeMoleculePositions(system['symbol'], system['positions']))
        mol.atom = MakeMoleculePositions(system['symbol'], system['positions'])
        mol.basis = '../basis/6-311+g2dp.nw'
        mol.cart = True
        mol.spin = system['spin']
        mol.charge = 0
        mol.build()

        kskernel.CalculateKSKernel(mol)
        print('E = {:.12e}\tX = {:.12e}\tC = {:.12e}\tXC = {:.12e}'.format(kskernel.mf.e_tot, 0.0, 0.0, kskernel.mf.get_veff().exc))

        # exks = ExKS(mol, kskernel, 'exks,')
        # Ex = exks.CalculateTotalX()
        # Ec = exks.CalculateTotalC()
        # Exc = exks.CalculateTotalXC()
        # E = exks.CalculateTotalEnergy()
        # print('E = {:.12e}\tX = {:.12e}\tC = {:.12e}\tXC = {:.12e}'.format(E, Ex, Ec, Exc))
        
        # lsd = DFA(mol, kskernel, 'LDA,PW_MOD')
        # x = lsd.CalculateTotalX()
        # c = lsd.CalculateTotalC()
        # xc = lsd.CalculateTotalXC()
        # e = lsd.CalculateTotalEnergy()
        # print('E = {:.12e}\tX = {:.12e}\tC = {:.12e}\tXC = {:.12e}'.format(e, x, c, xc))
        
        pbe = DFA(mol, kskernel, 'PBE,PBE')
        Ex = pbe.CalculateTotalX()
        Ec = pbe.CalculateTotalC()
        Exc = pbe.CalculateTotalXC()
        E = pbe.CalculateTotalEnergy()
        print('E = {:.12e}\tX = {:.12e}\tC = {:.12e}\tXC = {:.12e}'.format(E, Ex, Ec, Exc))

        # cfx = CF('cfx', mol, kernel)
        # cfx_xc = cfx.CalculateEpsilonXC()
        system['energy'] = E
        systems_energies.append(E)

    ar_es[:,1] = systems_energies

    return ar_es

def CalculateAtomizationEnergies(systems, functionals, pval):
    
    es = CalculateTotalEnergies(systems, functionals, pval)
    aes = []
    molecule_names = []
    # iterate over all systems, get the molecules and atomize them
    for key in systems:
        system = systems[key]
        atoms = re.findall('[A-Z][^A-Z]*', system['symbol'])
        if len(atoms) > 1:
            molecule_names.append(system['symbol'])
            eatoms = 0.0
            for atom in atoms:                
                system_a = systems[atom]
                eatoms += system_a['energy']
            ae = (eatoms - system['energy']) * 627.509
            aes.append(ae)

    ar_aes = np.ndarray((len(aes), 3), dtype=object)
    ar_aes[:,0] = molecule_names
    ar_aes[:,1] = aes

    return ar_aes

def MakeExactAtomizationEnergies(systems):
    """
        Creates a list with exact atomization energies of the systems in the dictionary
        Input: system's directory
        Output: list of AE
    """
    exact_aes = []

    for key in systems:
        system = systems[key]
        atoms = re.findall('[A-Z][^A-Z]*', system['symbol'])
        if len(atoms) > 1:        
            exact_aes.append(system['atomization'])

    return exact_aes

def CalculateMAE(systems, functionals, pval):
    """
        This function calculates the MAE of atomization energies with a given parameter value pval
        input: parameters values
        output: mae
    """

    mae = 0.0
    pval = 1.0 # functional parameter's value
    # Calculate Atomization Energies
    aes = CalculateAtomizationEnergies(systems, 'PBE,PBE', pval)

    exact_aes = MakeExactAtomizationEnergies(systems)
    aes[:,2] = exact_aes
    print(aes)
    # mae = aes.sum() / len(aes)

    return mae

def main():

    functionals = ['LDA,PW_MOD', 'PBE,PBE']

    # extract atoms that form the molecules in the dictonary
    # create new entries for each of them for calculations
    systems = PrepareSystems(molecules)

    # minimize MAE with 10 steps
    pval = 0.0
    for step in range(0, 10):
        
        mae = CalculateMAE(systems, functionals, pval)

        if step == 0:
            old_mae = mae
        elif mae > old_mae:
            pval += 0.1
        elif mae < old_mae:
            old_mae = mae
            break

if __name__ == "__main__":
    main()
