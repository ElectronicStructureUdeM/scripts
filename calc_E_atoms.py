from modelXC import ModelXC
import numpy as np
import sys
"""
Description:
    This code can be used to compute the total energies of various atoms from H to Ar
    with a fonctional or Kohn-Sham exact exchange energy density post-pbe.
    To use it, it is important to download the basis set from https://www.basissetexchange.org
    and name it 6-311+g2dp.nw, since it is the one used by default.
    For the implementation of kohn-Sham exact exchange energy density, see the appendix of
    https://doi.org/10.1063/1.5083840 .

Usage: python3 calc_E_atom.py functional  (functional is a function from pySCF or EXKS)
    It will create a file named E_atoms.txt with all the energies.

TODO:
    A prettier format for E_atom.txt
"""

#Dictionary with the atoms and it's total spin
atoms={"H":1,"He":0,"Li":1,"Be":0,"B":1,"C":2,"N":3,
        "O":2,"F":1,"Ne":0,"Na":1,"Ne":0,"Na":1,
        "Mg":0,"Al":1,"Si":2,"P":3,"S":2,"Cl":1,"Ar":0}
        
functional = sys.argv[1]
#file stuff
f=open("E_atom.txt","w")
f.write("Atom "+functional+"\n")
for atom in atoms:
    post_pbe  = ModelXC(atom,[[0,0,0]],atoms[atom],approx='pbe,pbe')
    if functional=="EXKS":
        f.write(atom +" %.8f\n"%post_pbe.calc_total_energy_Ex_ks())
    else:
        f.write(atom +" %.8f\n"%post_pbe.calc_Etot_post_approx(functional))
f.close()
