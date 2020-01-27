from CFXpo import CF
import numpy as np
import dataset
import sys
from mpi4py import MPI
from pyscf import dft,gto,lib,scf
#mpistuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
total_fractions=20
fractions_energies=None
perrank=total_fractions//size
if rank==0:
    fractions = np.linspace(0,1.,total_fractions)
    energies = np.zeros(total_fractions)
    fractions_energies = np.column_stack((fractions,energies))
recvbuf = np.empty((perrank,2), dtype='d')
comm.Scatter(fractions_energies, recvbuf, root=0)
fractions_energies_rank = recvbuf
num_fractions = np.shape(fractions_energies_rank)[0]
#positions= [[0,0,0]]
model=sys.argv[1]
data=sys.argv[2]
if data=="atoms":
    molecules=dataset.atoms
elif data=="molecules":
    molecules=dataset.molecules
    charge=0
elif data=="cations":
    molecules=dataset.IP13_cations
    charge=1
elif data=="neutrals":
    molecules=dataset.IP13_neutrals
    charge=0
elif data=="reactants":
    molecules=dataset.BH6_reactants
    charge=0
elif data=="TS":
    molecules=dataset.BH6_TS
    charge=0
for molecule in molecules:
#for molecule in dataset.atoms:
    comm.barrier()
    for i in range(num_fractions):
        frac = fractions_energies_rank[i,0]
        print(str(frac)+"*HF+"+str(1.-frac)+"*pbe")
        if data=="atoms":
            cf = CF(molecule,[[0,0,0]],dataset.atoms[molecule],
                    model,approx=str(frac)+"*HF+"+str(1.-frac)+"*pbe,pbe",basis="cc-pvtz",num_threads=1)
        else:
            cf = CF(molecule,molecules[molecule][1],molecules[molecule][0],
                    model,approx=str(frac)+"*HF+"+str(1.-frac)+"*pbe,pbe",basis="cc-pvtz",num_threads=1,charge=charge)
        fractions_energies_rank[i,1]=cf.calc_Etot_cf()
        print(fractions_energies_rank[i,1])
    comm.barrier()
    if rank==0:
        recvbuf = np.empty((total_fractions,2),dtype='d')
    comm.Gather(fractions_energies_rank,recvbuf,root=0)
    if rank==0:
        np.savetxt(molecule+"_"+model+"_"+data+".txt",recvbuf)
    
