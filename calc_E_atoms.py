from pyscf import gto,scf,dft
from pyscf import lib
import numpy as np
import re
"""
Description:
    This code can be used to compute the total energies of various atoms from H to Ar
    with PBE/6-311+g(2d,p)
    To use it, it is important to download the basis set from https://www.basissetexchange.org
    and name it 6-311+g2dp.nw.

Usage: python3 calc_E_atom.py
    It will create a file named E_atoms.txt with all the energies.

TODO:
    Make it more flexible to chose different functionnals and basis sets
    A prettier format for E_atom.txt
"""

def calc_energy_pbe(atom,spin):
    """
    Calculate the total energies of an atom with pbe/6-311+g(2d,p).
    Input:
        atom(string): the element symbol
        spin(int): The total spin
    returns:
        energy(float): the total energy
    """
    mol = gto.Mole()
    mol.atom=atom
    mol.verbose=0
    mol.spin=spin
    mol.basis = '6-311+g2dp.nw'
    mol.build()
    mf = scf.KS(mol)
    mf.small_rho_cutoff = 1e-12
    mf.xc='pbe'
    mf.kernel()
    return mf.e_tot
    

lib.num_threads(1)# pySCF will only use 1 thread
#Dictionary with the atoms and it's spin
atoms={"H":1,"He":0,"Li":1,"Be":0,"B":1,"C":2,"N":3,
        "O":2,"F":1,"Ne":0,"Na":1,"Ne":0,"Na":1,
        "Mg":0,"Al":1,"Si":2,"P":3,"S":2,"Cl":1,"Ar":0}
f=open("E_atom.txt","w")
f.write("Atom PBE\n")
for atom in atoms:
    f.write(atom +" %.8f\n"%calc_energy_pbe(atom,atoms[atom]))
f.close()