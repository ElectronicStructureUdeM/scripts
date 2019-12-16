from CFXpo import CF
import numpy as np
import dataset
fractions = np.arange(0,1.1,0.1)
energies = np.zeros(np.shape(fractions))
positions= [[0,0,0]]
model="cfx"
for atom in dataset.atoms:
    j=0
    for i in fractions:
        print(str(i)+"*HF+"+str(1.-i)+"*pbe")
        cf = CF(atom,positions,dataset.atoms[atom],
                    model,approx=str(i)+"*HF+"+str(1.-i)+"*pbe,pbe",basis="cc-pvtz")
        energies[j]=cf.calc_Etot_cf()
        print(energies[j])
        j+=1
    np.savetxt(atom+"_"+model+".txt",(fractions,energies))