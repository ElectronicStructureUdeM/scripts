from pyscf import gto,scf,dft
from pyscf import lib
from pyscf.dft import numint
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

def calc_energy_dft(atom,spin,functional):
    """
    Calculate the total energies of an atom with functional/6-311+g(2d,p).
    The energies are computed from PBE converged density (post-PBE method)
    Input:
        atom(string): the element symbol
        spin(int): The total spin
        functional(string): functional name in pyscf format
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
    mf.xc=functional
    mf.kernel()
    pbe_exc=mf.get_veff().exc
    return mf.e_tot


def compute_ex_exact(mol,ao_value,dm,coord):
    """
    Function to compute the exact kohn sham exchange energy density.
    See the appendix of https://doi.org/10.1063/1.5083840 for details.
    Input:
        mol(mol): molecule object from pyscf
        ao_values(array): ao values for a grid point
        dm(array): density matrix
        coord(array): x,y,z coordinates
    Returns:
        ex(float):ex^ks
    """
    with mol.with_rinv_origin((coord[0],coord[1],coord[2])):
        A = mol.intor('int1e_rinv')
    F = np.dot(dm,ao_value)
    return -np.einsum('i,j,ij',F,F,A)/2.

def calc_energy_exks(atom,spin):
    """
    Calculate the total energy when using kohn sham exact exchange post PBE

    Input:
        atom(string): element symbol
        spin(int): Spin of the atom
    TODO

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
    pbe_exc=mf.get_veff().exc
    grids = mf.grids
    #pbe
    ao_value = numint.eval_ao(mol, grids.coords, deriv=1)
    dm = mf.make_rdm1()
    if spin==0:
        #EX exact
        nGrid = np.shape(grids.coords)[0]
        exExact=np.zeros(nGrid)
        for iG in range(nGrid):
            exExact[iG] = compute_ex_exact(mol,ao_value[0,iG,:],
                                    dm,grids.coords[iG])/2.
        Extot= np.einsum('i,i->', exExact,grids.weights)
    else:# for spin polarized molecule
        #EX exact
        nGrid = np.shape(grids.coords)[0]
        exExactA=np.zeros(nGrid)
        exExactB=np.zeros(nGrid)
        for iG in range(nGrid):
            exExactA[iG] = compute_ex_exact(mol,ao_value[0,iG,:],
                                    dm[0],grids.coords[iG])
            exExactB[iG] = compute_ex_exact(mol,ao_value[0,iG,:],
                                    dm[1],grids.coords[iG])
        Extot= np.einsum('i,i->', exExactA+exExactB, 
                                         grids.weights)
    #ex exact
    ETotalExKS = mf.e_tot-pbe_exc+Extot
    return ETotalExKS

lib.num_threads(1)# pySCF will only use 1 thread
#Dictionary with the atoms and it's spin
atoms={"H":1,"He":0,"Li":1,"Be":0,"B":1,"C":2,"N":3,
        "O":2,"F":1,"Ne":0,"Na":1,"Ne":0,"Na":1,
        "Mg":0,"Al":1,"Si":2,"P":3,"S":2,"Cl":1,"Ar":0}
        
functional = sys.argv[1]
#file stuff
f=open("E_atom.txt","w")
f.write("Atom "+functional+"\n")
for atom in atoms:
    if functional=="EXKS":
        f.write(atom +" %.8f\n"%calc_energy_exks(atom,atoms[atom]))
    else:
        f.write(atom +" %.8f\n"%calc_energy_dft(atom,atoms[atom],functional))
f.close()