from pyscf import gto,scf,dft
from pyscf import lib
from pyscf.dft import numint
import numpy as np
import sys
"""
Description:
    This code can be used to compute the total energies of various atoms from H to Ar
    with PBE/6-311+g(2d,p) or Kohn-Sham exact exchange energy density post-PBE.
    To use it, it is important to download the basis set from https://www.basissetexchange.org
    and name it 6-311+g2dp.nw.

Usage: python3 calc_E_atom.py method  (methos id PBE or EXKS)
    It will create a file named E_atoms.txt with all the energies.

TODO:
    Make it more flexible to chose different functionnals and basis sets
    A better way to compute total exchange energy density by not having to calculate Exc of PBE
        since useless calculation is done
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


def compute_ex_exact(mol,ao_value,dm,coord):
    """
    Function to compute the exact kohn sham exchange energy density.
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
    Calculate the total energy when using kohn sham exact exchange

    Input:
        atom(string): element symbol
        spin(int): Spin of the atom
    TODO
        Finding a way in PYSCF to extract all the contribution to the total energy
        so Exc PBE won't have to be recalculated
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
    grids = mf.grids
    #pbe
    ao_value = numint.eval_ao(mol, grids.coords, deriv=1)
    dm = mf.make_rdm1()
    if spin==0:
        rho = numint.eval_rho(mol, ao_value, dm, xctype='GGA')
        ex, vx = dft.libxc.eval_xc('pbe,', rho)[:2]
        ec, vc = dft.libxc.eval_xc(',pbe', rho)[:2]
        Exc = np.einsum('i,i,i->', ex+ec, rho[0],grids.weights)
        #EX exact
        nGrid = np.shape(grids.coords)[0]
        exExact=np.zeros(nGrid)
        for iG in range(nGrid):
            exExact[iG] = compute_ex_exact(mol,ao_value[0,iG,:],
                                    dm,grids.coords[iG])/2.
        Extot= np.einsum('i,i->', exExact,grids.weights)
    else:# for spin polarized molecule
        #pbe
        rhoA = numint.eval_rho(mol, ao_value, dm[0], xctype='GGA')
        zeros=np.zeros(np.shape(rhoA))
        rhoB = numint.eval_rho(mol, ao_value, dm[1], xctype='GGA')
        exA, vx = dft.libxc.eval_xc('pbe,', [rhoA,zeros],spin=spin)[:2]
        exB, vx = dft.libxc.eval_xc('pbe,', [zeros,rhoB],spin=spin)[:2]
        ec, vc = dft.libxc.eval_xc(',pbe', [rhoA,rhoB],spin=spin)[:2]
        ExA = np.einsum('i,i,i->', exA, rhoA[0],grids.weights)
        ExB = np.einsum('i,i,i->', exB, rhoB[0],grids.weights)
        Ec = np.einsum('i,i,i->', ec, rhoA[0]+rhoB[0],grids.weights)
        Exc = ExA+ExB+Ec
        #EX
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
    ETotalExKS = mf.e_tot-Exc+Extot
    return ETotalExKS

lib.num_threads(1)# pySCF will only use 1 thread
#Dictionary with the atoms and it's spin
atoms={"H":1,"He":0,"Li":1,"Be":0,"B":1,"C":2,"N":3,
        "O":2,"F":1,"Ne":0,"Na":1,"Ne":0,"Na":1,
        "Mg":0,"Al":1,"Si":2,"P":3,"S":2,"Cl":1,"Ar":0}
        
implemented_method=["PBE","EXKS"]
method = sys.argv[1]
assert method in implemented_method,"Method must be PBE or EXKS"
if method=="PBE": energy=calc_energy_pbe
if method=="EXKS":energy = calc_energy_exks

#file stuff
f=open("E_atom.txt","w")
f.write("Atom "+method+"\n")
for atom in atoms:
    f.write(atom +" %.8f\n"%energy(atom,atoms[atom]))
f.close()