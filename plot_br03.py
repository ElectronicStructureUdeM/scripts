import numpy as np
import matplotlib.pyplot as plt
from modelXC import ModelXC
import Dicts

for atom in Dicts.atoms:
    spin=Dicts.atoms[atom]
    model = ModelXC(atom,[[0,0,0]],spin,approx='pbe,pbe')
    distance = np.linalg.norm(model.coords,axis=1)
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(20,10))
    avg_n_up = np.average(model.br_n_up)
    avg_n_down = np.average(model.br_n_down)
    plt.scatter(distance,model.br_n_up,alpha=0.5,label=r'$\alpha$ spin,  average= %.2f' %avg_n_up)
    plt.scatter(distance,model.br_n_down,alpha=0.5,label=r'$\beta$ spin,  average= %.2f' %avg_n_down)
    plt.title(r'$N^{B03}$ for %s'%atom)
    plt.xlabel(r'Distance (Bohr)')
    plt.ylabel(r'$N^{B03}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(atom+"_nb03.png")