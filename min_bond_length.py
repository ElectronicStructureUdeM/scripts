from CFXpo import CF
import numpy as np
import dataset
import sys
model=sys.argv[1]

molecule="ClCl"
spin=0
begin_length=2.045
z=0.
while z <=0.1:
    cf = CF(molecule,[[0,0,0],[0,0,begin_length+z]],spin,
                model,approx="pbe,pbe",basis="cc-pvtz",num_threads=1)
    E=cf.calc_Etot_cf()
    print(z+begin_length,E)
    z+=0.001