from pyscf import gto,scf,dft
import numpy as np
import re
"""
Script to compute the atomization energies of PBE/6-311+g(2d,p) for a small dataset. 

Dependencies:
    calc_E_atoms
Usage:
    (if not already done) python3 calc_E_atoms
    python3 calc_atomization.py
Output:
    atomization.txt(file): molecule total energy(Ha) atomization energy (kcal/mol)
TODO
    Making the code more flexible to chose other methods.
    Prettier format for atommization.txt
"""
def calc_energy_mol(molec,positions,spin):
    """
    Function to compute the total energy of a molecule with 
    PBE/6-311+g(2d,p). The basis set file 6-311+g2dp.nw was
    downloaded from https://www.basissetexchange.org/.

    Input:
        Molec(string): a string with each atoms in a molecule.
                        Each atom must be specified.
                        ex: "CHHHH" and not "CH4"
        positions(list): a list with the positions of each atom
                        ex:[[x1,y1,z1],[x2,y2,z2]]
        spin(int): the total spin of the molecule
    output:
        total energy
    """
    mol = gto.Mole()
    atoms = re.findall('[A-Z][^A-Z]*', molec)
    molecule =[]
    nAtom=0
    for atom in atoms:
        atom_pos = positions[nAtom]
        molecule.append([atom,(atom_pos[0],atom_pos[1],atom_pos[2])])
        nAtom=nAtom+1
    mol.atom=molecule
    mol.verbose=0
    mol.spin=spin
    mol.basis = '6-311+g2dp.nw' # downloaded from BSE
    mol.build()
    mf = scf.KS(mol)
    mf.small_rho_cutoff = 1e-12
    mf.xc='pbe'
    mf.kernel()
    return mf.e_tot

def open_E_atoms():
    """
    Function to open a file with atomization energy from pbe.
    The file must be named E_atom.txt.
    Return:
        atom_pbe_E(dictionary): atom(string),E_pbe(float)
    """
    filename = 'E_atom.txt'
    # todo calculate for each atom
    atom_pbe_E = {}
    with open(filename) as fh:
        next(fh) # ignore header
        for line in fh:
            atom, E_pbe = line.split()
            atom_pbe_E[atom] = float(E_pbe)
    fh.close()
    return atom_pbe_E
def atomization(mol,positions,spin):
    """
    Function to calculate the atomization energy of a molecule with  the pbe.
    Will read the total energy of atoms from open_E_atoms.
    Warning: if atomization.txt already exists, it will start for the next molecule
    Args:
        mol(string): The molecule name, each atom must be specified individually(ex:CHHHH and not CH4)
        positions(array): positions of each atoms
        spin(int): total spin
    Output:
        atomization.txt
    """
    print("Begin calculate atomization energy for "+mol+"\n")
    convert_kcalmol = 627.509474
    E_pbe = open_E_atoms() # total energy of atoms
    out = open("atomization.txt","a+")
    out.seek(0)# return to begining of file
    fl = out.readlines()
    if fl == []:
        out.write("Mol  E_tot_pbe Atomization_pbe\n")
    mol_exist=False
    # if the calculation is already done for the molecule, it won't calculate again
    for line in fl:
        if line.split()[0] == mol:
            mol_exist=True
    if mol_exist == False: 
        E_pbe_mol =calc_energy_mol(mol,positions,spin)
        atoms_E_pbe=0
        atoms = re.findall('[A-Z][^A-Z]*', mol) # to split at each uppercase using regular expression
        for atom in atoms:
            atoms_E_pbe+=E_pbe[atom]
        atomi_pbe = (atoms_E_pbe-E_pbe_mol)*convert_kcalmol
        out.write(mol+"\t"+str(E_pbe_mol)+"\t"+str(atomi_pbe)+"\n")
    print("End calculate atomization energy for "+mol+"\n")

    out.close()
# All the geometries were optimized at the PBE/6-311+g(2d,p) level
#H2
mol="HH"
spin=0
positions = [[0, 0, -0.0259603084],
                [0, 0, 0.7259603084]]
atomization(mol,positions,spin )
#LiH
mol="LiH"
spin=0
positions= [[0, 0, 0.3962543501],
                [0, 0, 2.0037456499]]
atomization(mol,positions,spin )
#CH4
mol = "CHHHH"
spin=0
positions = [[-0.0000016290,0.00000,0.0000078502],
                  [-0.0000022937,0.00000,1.0970267936],
                  [1.0342803963,0.00000,-0.3656807611],
                  [-0.5171382367,-0.8957112847,-0.3656769413],
                  [-0.5171382368,0.8957112847,-0.3656769413]]
atomization(mol,positions,spin )
#NH3
mol = "NHHH"
spin=0
positions = [[-0.7080703847,0.5736644371,-0.2056610779],
                  [0.3140478690,0.6090902876,-0.2439925162],
                  [ -1.0241861213,0.3280701680,-1.1475765240],
                  [-1.0241913630,1.5321151073,-0.0355598819]]
atomization(mol,positions,spin )
#H2O
mol = "OHH"
spin=0
positions = [[-0.7435290312,-0.0862560218,-0.2491318075],
                  [0.2269625234,-0.0687025898,-0.2099668601],
                  [ -1.0265534922,0.2938386117,0.5988786675]]
atomization(mol,positions,spin )
#HF
mol="FH"
spin=0
positions= [[0, 0, -0.0161104113],[0, 0, 0.9161104113]]
atomization(mol,positions,spin )
#Li2
mol="LiLi"
spin=0
positions= [[0, 0, -0.0155360351],[0, 0, 2.7155360351]]
atomization(mol,positions,spin )
#LiF
mol="LiF"
spin=0
positions= [[0, 0, 0.0578619642],[0, 0, 1.6421380358]]
atomization(mol,positions,spin )
#Be2
mol="BeBe"
spin=0
positions= [[0, 0, 0.0085515554],[0, 0, 2.4414484446]]
atomization(mol,positions,spin )
#C2H2
mol = "CHCH"
spin=0
positions = [[ -7.5637480678 ,-4.0853657900,0.00000000],
                  [-8.6353642657,-4.0853657900,0.00000000],
                  [-6.3570037122,-4.0853657900,0.00000000],
                  [-5.2853875143,-4.0853657900,0.00000000]]
atomization(mol,positions,spin )
#C2H4
mol = "CCHHHH"
spin=0
positions = [[-4.5194036917,0.9995360751,-0.0000241325],
                  [-3.1861963083,0.9995360751,-0.0000241325],
                  [-5.0929778983,0.1325558381,-0.3377273553],
                  [-5.0929780326,1.8664879090,0.3377519084],
                  [ -2.6126221017,0.1325558381,-0.3377273553],
                  [-2.6126219674,1.8664879090,0.3377519084]]
atomization(mol,positions,spin )
#HCN
mol = "HCN"
spin=0
positions = [[ -2.1652707291,0.9995300000,0.0000000000],
                  [-3.2423025370,0.9995300000,0.0000000000],
                  [-4.4007967339,0.9995300000,0.0000000000]]
atomization(mol,positions,spin )
#CO
mol="CO"
spin=0
positions= [[0, 0, -0.0185570711],
                [0, 0, 1.1185570711]]
atomization(mol,positions,spin )
#N2
mol="NN"
spin=0
positions= [[0, 0, -0.0017036831],
                [0, 0, 1.1017036831]]
atomization(mol,positions,spin )
#NO
mol="NO"
spin=1
positions= [[0, 0, -0.0797720915],
                [0, 0, 1.0797720915]]
atomization(mol,positions,spin )
#triplet O2
mol="OO"
spin=2
positions= [[0, 0, -0.0114390797],
                [0, 0, 1.2114390797]]
atomization(mol,positions,spin )
#F2
mol="FF"
spin=0
positions= [[0, 0, -0.0083068123],
                [0, 0, 1.4083068123]]
atomization(mol,positions,spin )
#P2
mol="PP"
spin=0
positions= [[0, 0, -0.0063578484],
                [0, 0, 1.9063578484]]
atomization(mol,positions,spin )
#Cl2
mol="ClCl"
spin=0
positions= [[0, 0, -0.0645570711],
                [0, 0, 1.9645570711]]
atomization(mol,positions,spin )