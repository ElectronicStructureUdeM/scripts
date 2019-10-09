from modelXC import ModelXC
from CF3 import *
import numpy as np
import sys
from multiprocessing import Process , current_process
import os

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

def Run_atmz(atom):
    process_id = os.getpid()
    print("Procid" + str(process_id) )
    model  = ModelXC(atom,[[0,0,0]],atoms[atom],approx='PBE0',basis='ccpvtz')
    print(atom +" %.8f\n"%model.calc_total_energy_Ex_ks())

#Dictionary with the atoms and it's total spin
atoms={"H":1,"He":0,"Li":1,"Be":0,"B":1,"C":2,"N":3,
        "O":2,"F":1,"Ne":0,"Na":1,"Ne":0,"Na":1,
        "Mg":0,"Al":1,"Si":2,"P":3,"S":2,"Cl":1,"Ar":0}

processes = []
for atom in atoms:
     process = Process(target = Run_atmz, args=(atom,))
     processes.append(process)
     process.start()
