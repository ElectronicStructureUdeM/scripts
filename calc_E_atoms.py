from pyscf import gto,scf,dft
from pyscf import lib
from pyscf.dft import numint
from compute_energies import compute_ex_exact,calc_energy_Exks,calc_energy_dft
import numpy as np
import sys
"""
Description:
    This code can be used to compute the total energies of various atoms from H to Ar
    with functional/6-311+g(2d,p) or Kohn-Sham exact exchange energy density post-PBE.
    To use it, it is important to download the basis set from https://www.basissetexchange.org
    and name it 6-311+g2dp.nw.
    For the implementation of kohn-Sham exact exchange energy density, see the appendix of
    https://doi.org/10.1063/1.5083840 .

Usage: python3 calc_E_atom.py functional  (functional is a function from pySCF or EXKS)
    It will create a file named E_atoms.txt with all the energies.

TODO:
    Make it more flexible to chose different basis sets
    A prettier format for E_atom.txt
"""

lib.num_threads(1)# pySCF will only use 1 thread
#Dictionary with the atoms and it's total spin
atoms={"H":1,"He":0,"Li":1,"Be":0,"B":1,"C":2,"N":3,
        "O":2,"F":1,"Ne":0,"Na":1,"Ne":0,"Na":1,
        "Mg":0,"Al":1,"Si":2,"P":3,"S":2,"Cl":1,"Ar":0}
        
functional = sys.argv[1]
#file stuff
f=open("E_atom.txt","w")
f.write("Atom "+functional+"\n")
for atom in atoms:
    if functional=="EXKS":
        f.write(atom +" %.8f\n"%calc_energy_Exks(atom,[[0,0,0]],atoms[atom]))
    else:
        f.write(atom +" %.8f\n"%calc_energy_dft(atom,[[0,0,0]],atoms[atom],functional))
f.close()
