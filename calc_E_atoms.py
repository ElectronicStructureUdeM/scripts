from modelXC import ModelXC
import numpy as np
import sys
from fxc import Fxc
import multiprocess as mp

#Dictionary with the atoms and it's total spin
atoms={"H":1,"He":0,"Li":1,
        "Be":0,"B":1,"C":2,"N":3,
        "O":2,"F":1,"Ne":0,"Na":1,"Ne":0,"Na":1,
        "Mg":0,"Al":1,"Si":2,"P":3,"S":2,"Cl":1,
        "Ar":0}

def calc_E(atom,positions,spin,functional):
    """
    To calculate the total energies a functional implemented in pyscf or our model
    with converged pbe densities
    Input:
        atom:string
            atomic symbol
        positions:array
            positions of the atom
        spin:int
            the spin of the atom
        functional:string
            functional name 
    Return:
        model_dict:dict
            atom_name:energy
    """
    if functional == "EXKS":
        post_pbe  = ModelXC(atom,positions,atoms[atom],approx='pbe,pbe')
        E=post_pbe.calc_total_energy_Ex_ks()
    elif functional == "fxc":
        fxc = Fxc(atom,positions,atoms[atom],approx='pbe,pbe')
        E = fxc.calc_Etot_fxc()
    else:
        post_pbe  = ModelXC(atom,positions,atoms[atom],approx='pbe,pbe')
        E=post_pbe.calc_Etot_post_approx(functional)
    model_dict = {atom:E}
    print(model_dict)
    return model_dict

num_proc = sys.argv[1]
functional = sys.argv[2]
pool = mp.Pool(int(num_proc))

#Calculate for the models
results = pool.map(lambda atom:calc_E(atom,[[0,0,0]],atoms[atom],functional),
                                                    [atom for atom in atoms])

#to convert the list of dictionaries to a dictionary
results_dict = {}
for sub_dict in results:
    results_dict.update(sub_dict)

#creation of the file
f=open("E_atom.txt","w")
f.write("Atom "+functional+"\n")
for atom in results_dict:
    f.write(atom +" %.8f\n"%results_dict[atom])
f.close()
