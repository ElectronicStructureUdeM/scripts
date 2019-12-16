from CFXpo import CF
import numpy as np

fractions = np.arange(0,1.1,0.1)
energies = np.zeros(np.shape(fractions))
j=0
mol="H"
spin=1
positions= [[0,0,0]]
for i in fractions:
    print(str(i)+"*HF+"+str(1.-i)+"*pbe")
    test = CF(mol,positions,spin,"cfx",approx=str(i)+"*HF+"+str(1.-i)+"*pbe,pbe",basis="cc-pvtz")
    energies[j]=test.calc_Etot_cf()
    print(energies[j])
    j+=1
np.savetxt(mol+".txt",(fractions,energies))