from modelXC import ModelXC
import numpy as np
import sys
from CFXpo import CF

#Dictionary with the atoms and it's total spin
atoms={#"H":1,"He":0,"Li":1,
        #"Be":0,"B":1,
         #"C":2,"N":3,
        #"O":2,
        "F":1,"Ne":0,"Na":1,"Ne":0,"Na":1,
        "Mg":0,"Al":1,"Si":2,"P":3,"S":2,"Cl":1,
        "Ar":0}

def calc_E_EXKS(atom,positions,spin):
    """
    To calculate the total energies using exact KS exchange 
    with converged pbe densities
    Input:
        atom:string
            atomic symbol
        positions:array
            positions of the atom
        spin:int
            the spin of the atom
    Return:
        model_dict:dict
            atom_name:energy
    """
    post_pbe  = ModelXC(atom,positions,atoms[atom],approx='pbe,pbe')
    E=post_pbe.calc_total_energy_Ex_ks()
    model_dict = {atom:E}
    print(model_dict)
    return model_dict

def calc_E_cf(atom,positions,spin,method):
    """
    To calculate the total energies using the correlation factor model
    with converged pbe densities
    Input:
        atom:string
            atomic symbol
        positions:array
            positions of the atom
        spin:int
            the spin of the atom
    Return:
        model_dict:dict
            atom_name:energy
    """
    cf = CF(atom,positions,atoms[atom],method,approx='pbe,pbe',basis="6-311+g2dp.nw")
    E = cf.calc_Etot_cf()
    model_dict = {atom:E}
    print(model_dict)
    return model_dict

def calc_E_postpbe(atom,positions,spin,functional):
    """
    To calculate the total energies a functional implemented in pyscf
    with converged pbe densities
    Input:
        atom:string
            atomic symbol
        positions:array
            positions of the atom
        spin:int
            the spin of the atom
        functional:string
            functional name in pyscf format
    Return:
        model_dict:dict
            atom_name:energy
    """
    post_pbe  = ModelXC(atom,positions,atoms[atom],approx='pbe,pbe')
    E=post_pbe.calc_Etot_post_approx(functional)
    model_dict = {atom:E}
    print(model_dict)
    return model_dict

functional = sys.argv[1]

#Calculate for the models
if functional=="EXKS":
    results = [calc_E_EXKS(atom,[[0,0,0]],atoms[atom]) for atom in atoms]
elif functional.startswith("cf"):
    results = [calc_E_cf(atom,[[0,0,0]],atoms[atom],functional) for atom in atoms]
else:
    results = [calc_E_postpbe(atom,[[0,0,0]],atoms[atom],functional) for atom in atoms]

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
