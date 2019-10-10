import numpy as np
def calclocmp2(mol,ao_value,GAMMP2,coords):
   nao=mol.nao_nr()

   with mol.with_rinv_origin((coords[0],coords[1],coords[2])):
          A = mol.intor('int1e_rinv')

   F=np.zeros(nao,nao)
   F = np.einsum('i,j,ijkl->kl',ao_value,ao_value,GAMMP2)
   mp2ot = np.einsum('ij,i,j',F,ao_value,ao_value)
   eps = 0.5* np.einsum('kl,kl',F,A)
   return eps , mp2ot

def compute_ex_exact(mol,ao_value,DMA,coords):
    with mol.with_rinv_origin((coords[0],coords[1],coords[2])):
       A = mol.intor('int1e_rinv')
    F = np.dot(DMA,ao_value)
    return -.5*np.einsum('i,j,ij',F,F,A)
    
