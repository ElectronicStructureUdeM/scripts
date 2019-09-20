from pyscf import gto,scf,dft
import numpy as np
import matplotlib.pyplot as plt
"""
This script is used to show the self interaction error of PBE
by stretching an H2+ molecule

Usage:
    python3 stretched_h2+.py
Output:
    a maplotlib graph of the energy as a fonction of the bond length
TODO:
    Different bimolecular molecules than H2+
    Different methods than PBE
"""
def calc_energy_h2plus(mol,z):
    """
    Calculate the energy for H2+ at various bond length

    Input:
        mol: molecule object of pySCF
        z(float): z distance between the atoms
    returns:
        total energy(float)
    """
    mol.atom=[["H",(0,0,0)],["H",(0,0,z)]]
    mol.charge=1
    mol.spin=1
    mol.basis = '6-311+g2dp.nw'#from basis set exchange
    mol.build()
    mf = scf.UKS(mol)
    mf.small_rho_cutoff = 1e-12
    mf.xc='pbe'
    mf.kernel()
    return mf.e_tot

#parameters for the molecule
mol = gto.Mole()
mol.charge=1
mol.spin=1
mol.basis = '6-311+g2dp.nw'#from basis set exchange

#initialization of arrays
positions = np.arange(0.7,5,0.1)
size = np.shape(positions)[0]
energies=np.zeros(size)
#loop
for i in range(size):
    energies[i] = calc_energy_h2plus(mol,positions[i])
#for the graph
plt.plot(positions,energies)
plt.title(r'Stretched H$_2^+$ with PBE')
plt.xlabel(r'H-H length ($10^{-10}$m)')
plt.ylabel(r'total energy')
plt.show()
