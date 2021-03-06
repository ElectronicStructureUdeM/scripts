import numpy as np
import sys
from modelXC import ModelXC
import re
from CFXpo import CF
"""
Script to compute the atomization energies of functional/6-311+g(2d,p) for a small dataset. 
The energies are calculated in a post-PBE way, thus not self-consistent

Dependencies:
    calc_E_atoms.py
Usage:
    (if not already done) python3 calc_E_atoms functional
    python3 calc_atomization.py functional
Output:
    atomization.txt(file): molecule total energy(Ha) atomization energy (kcal/mol)
TODO
    automatic way to check if calc_E_atoms.py was already done
    Prettier format for atomization.txt
"""

    
def open_E_atoms():
    """
    Function to open a file with atomization energy.
    The file must be named E_atom.txt.
    See calc_E_atoms.py
    Return:
        atom_E(dictionary): atom(string),E(float)
    """
    filename = 'E_atom.txt'
    # todo calculate for each atom
    atom_E = {}
    with open(filename) as fh:
        next(fh) # ignore header
        for line in fh:
            atom, E = line.split()
            atom_E[atom] = float(E)
    fh.close()
    return atom_E

def atomization(mol,positions,spin,functional):
    """
    Function to calculate the atomization energy of a molecule with a functional from PBE converged densities.
    Will read the total energy of atoms from open_E_atoms.
    Warning: if atomization.txt already exists, it will continue to the next molecule
    Args:
        mol(string): The molecule name, each atom must be specified individually(ex:CHHHH and not CH4)
        positions(array): positions of each atoms
        spin(int): total spin
        functional(string): a functional implemented in pySCF
    Output:
        atomization.txt
    """
    print("Begin calculate atomization energy for "+mol+"\n")
    convert_kcalmol = 627.509474
    E_atoms = open_E_atoms() # total energy of atoms
    out = open("atomization.txt","a+")
    out.seek(0)# return to begining of file
    fl = out.readlines()
    if fl == []:
        out.write("Mol  E_tot_"+functional+" Atomization_"+functional+"\n")
    mol_exist=False
    # if the calculation is already done for the molecule, it won't calculate again
    for line in fl:
        if line.split()[0] == mol:
            mol_exist=True
    if mol_exist == False:       
        if functional=="EXKS":
            post_pbe  = ModelXC(mol,positions,spin,approx='pbe,pbe')
            E_mol =post_pbe.calc_total_energy_Ex_ks()
        elif functional.startswith("cf"):
            cf = CF(mol,positions,spin,functional,approx='pbe,pbe',basis='cc-pvtz',ASE=True)
            E_mol = cf.calc_Etot_cf()
        else:
            post_pbe  = ModelXC(mol,positions,spin,approx='pbe,pbe')
            E_mol=post_pbe.calc_Etot_post_approx(functional)
        tot_E_atoms=0
        atoms = re.findall('[A-Z][^A-Z]*', mol) # to split at each uppercase using regular expression
        for atom in atoms:
            tot_E_atoms+=E_atoms[atom]
        atomi = (tot_E_atoms-E_mol)*convert_kcalmol
        out.write(mol+"\t"+str(E_mol)+"\t"+str(atomi)+"\n")
    print("End calculate atomization energy for "+mol+"\n")

    out.close()
# All the geometries were optimized at the PBE/6-311+g(2d,p) level
functional = sys.argv[1]
#H2
mol="HH"
spin=0
positions = [[0, 0, -0.0259603084],
                [0, 0, 0.7259603084]]
atomization(mol,positions,spin,functional)
#LiH
mol="LiH"
spin=0
positions= [[0, 0, 0.3962543501],
                [0, 0, 2.0037456499]]
atomization(mol,positions,spin,functional)
#CH4
mol = "CHHHH"
spin=0
positions = [[-0.0000016290,0.00000,0.0000078502],
                  [-0.0000022937,0.00000,1.0970267936],
                  [1.0342803963,0.00000,-0.3656807611],
                  [-0.5171382367,-0.8957112847,-0.3656769413],
                  [-0.5171382368,0.8957112847,-0.3656769413]]
atomization(mol,positions,spin,functional )
#NH3
mol = "NHHH"
spin=0
positions = [[-0.7080703847,0.5736644371,-0.2056610779],
                  [0.3140478690,0.6090902876,-0.2439925162],
                  [ -1.0241861213,0.3280701680,-1.1475765240],
                  [-1.0241913630,1.5321151073,-0.0355598819]]
atomization(mol,positions,spin,functional )
#H2O
mol = "OHH"
spin=0
positions = [[-0.7435290312,-0.0862560218,-0.2491318075],
                  [0.2269625234,-0.0687025898,-0.2099668601],
                  [ -1.0265534922,0.2938386117,0.5988786675]]
atomization(mol,positions,spin,functional )
#HF
mol="FH"
spin=0
positions= [[0, 0, -0.0161104113],[0, 0, 0.9161104113]]
atomization(mol,positions,spin,functional )
#Li2
mol="LiLi"
spin=0
positions= [[0, 0, -0.0155360351],[0, 0, 2.7155360351]]
atomization(mol,positions,spin,functional )
#LiF
mol="LiF"
spin=0
positions= [[0, 0, 0.0578619642],[0, 0, 1.6421380358]]
atomization(mol,positions,spin,functional )
#Be2
mol="BeBe"
spin=0
positions= [[0, 0, 0.0085515554],[0, 0, 2.4414484446]]
atomization(mol,positions,spin,functional )
#C2H2
mol = "CHCH"
spin=0
positions = [[ -7.5637480678 ,-4.0853657900,0.00000000],
                  [-8.6353642657,-4.0853657900,0.00000000],
                  [-6.3570037122,-4.0853657900,0.00000000],
                  [-5.2853875143,-4.0853657900,0.00000000]]
atomization(mol,positions,spin,functional )
#C2H4
mol = "CCHHHH"
spin=0
positions = [[-4.5194036917,0.9995360751,-0.0000241325],
                  [-3.1861963083,0.9995360751,-0.0000241325],
                  [-5.0929778983,0.1325558381,-0.3377273553],
                  [-5.0929780326,1.8664879090,0.3377519084],
                  [ -2.6126221017,0.1325558381,-0.3377273553],
                  [-2.6126219674,1.8664879090,0.3377519084]]
atomization(mol,positions,spin,functional )
#HCN
mol = "HCN"
spin=0
positions = [[ -2.1652707291,0.9995300000,0.0000000000],
                  [-3.2423025370,0.9995300000,0.0000000000],
                  [-4.4007967339,0.9995300000,0.0000000000]]
atomization(mol,positions,spin,functional )
#CO
mol="CO"
spin=0
positions= [[0, 0, -0.0185570711],
                [0, 0, 1.1185570711]]
atomization(mol,positions,spin,functional )
#N2
mol="NN"
spin=0
positions= [[0, 0, -0.0017036831],
                [0, 0, 1.1017036831]]
atomization(mol,positions,spin,functional )
#NO
mol="NO"
spin=1
positions= [[0, 0, -0.0797720915],
                [0, 0, 1.0797720915]]
atomization(mol,positions,spin,functional )
#triplet O2
mol="OO"
spin=2
positions= [[0, 0, -0.0114390797],
                [0, 0, 1.2114390797]]
atomization(mol,positions,spin,functional )
#F2
mol="FF"
spin=0
positions= [[0, 0, -0.0083068123],
                [0, 0, 1.4083068123]]
atomization(mol,positions,spin,functional )
#P2
mol="PP"
spin=0
positions= [[0, 0, -0.0063578484],
                [0, 0, 1.9063578484]]
atomization(mol,positions,spin,functional )
#Cl2
mol="ClCl"
spin=0
positions= [[0, 0, -0.0645570711],
                [0, 0, 1.9645570711]]
atomization(mol,positions,spin,functional )