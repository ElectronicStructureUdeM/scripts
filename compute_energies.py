
from pyscf.dft import numint
import numpy as np
from pyscf import gto,scf,dft
import re

def compute_ex_exact(mol,ao_value,dm,coord):
    """
    Function to compute the exact kohn sham exchange energy density
    for a grid point.
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

def calc_energy_Exks(molec,positions,spin):
    """
    Function to compute the total energy of a molecule with 
    exchange exchange KS/6-311+g(2d,p). The basis set file 6-311+g2dp.nw was
    downloaded from https://www.basissetexchange.org/.
    The energies are are calculated post-PBE.

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
    mf.grids.radi_method=dft.radi.delley
    mf.xc='pbe'
    mf.kernel()
    Exc_pbe=mf.get_veff().exc
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
    ETotalExKS = mf.e_tot-Exc_pbe+Extot
    return ETotalExKS

def calc_energy_dft(molec,positions,spin,functional):
    """
    Function to compute the total energy of a molecule with 
    functional/6-311+g(2d,p) in a post-PBE manner. The basis set file 6-311+g2dp.nw was
    downloaded from https://www.basissetexchange.org/.

    Input:
        Molec(string): a string with each atoms in a molecule.
                        Each atom must be specified.
                        ex: "CHHHH" and not "CH4"
        positions(list): a list with the positions of each atom
                        ex:[[x1,y1,z1],[x2,y2,z2]]
        spin(int): the total spin of the molecule
        functional(string): a dft functional implemented in pyXCF
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
    mf.xc="pbe"
    mf.kernel()
    pbe_exc=mf.get_veff().exc
    if functional=="pbe":
        return mf.e_tot
    else:
        if dft.libxc.is_meta_gga(functional):
            deriv=2
            xctype="MGGA"
        elif dft.libxc.is_gga(functional):
            deriv=1
            xctype="GGA"
        else:
            deriv=0
            xctype="LDA"
        grids = mf.grids
        ao_value = numint.eval_ao(mol, grids.coords, deriv=deriv)
        dm = mf.make_rdm1()
        if spin ==0:
            rho = numint.eval_rho(mol, ao_value, dm, xctype=xctype)
            exc,vxc = dft.libxc.eval_xc(functional, rho)[:2]
            if xctype=="LDA":
                Exc = np.einsum('i,i,i->', exc, rho,grids.weights)
            else:
                Exc = np.einsum('i,i,i->', exc, rho[0],grids.weights)
            return mf.e_tot-pbe_exc+Exc
        else:
            rhoA = numint.eval_rho(mol, ao_value, dm[0], xctype=xctype)
            rhoB = numint.eval_rho(mol, ao_value, dm[1], xctype=xctype)
            exc, vxc= dft.libxc.eval_xc(functional, [rhoA,rhoB],spin=spin)[:2]
            if xctype=="LDA":
                Exc = np.einsum('i,i,i->', exc, rhoA+rhoB,grids.weights)
            else:
                Exc = np.einsum('i,i,i->', exc, rhoA[0]+rhoB[0],grids.weights)
            return mf.e_tot-pbe_exc+Exc
