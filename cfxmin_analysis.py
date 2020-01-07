import glob
import numpy as np

files = glob.glob("*.txt")

for f in files:
    mol = f.split("_")[0]
    fractions_energies = np.loadtxt(f)
    arg_min = np.argmin(fractions_energies[:,1])
    print(mol,fractions_energies[arg_min,0],fractions_energies[arg_min,1])