import numpy as np
from ase.atoms import Atoms
import ase.data.g2 as g2

class Data:

    def __init__(self):

        self.AtomSpin = {"X": -1, 
        "H":1,"He":0,"Li":1,"Be":0,"B":1,"C":2,"N":3,
        "O":2,"F":1,"Ne":0,"Na":1,"Ne":0,"Na":1,
        "Mg":0,"Al":1,"Si":2,"P":3,"S":2,"Cl":1,"Ar":0}

        self.MGAtoms = [ "H", "Li", "Be", "C", "N", "O", "F", "O", "F", "P", "Cl"]
        self.MGAtomEnergie = [-0.500, -7.478, -14.667, -37.845,-54.589, -75.067, -99.734, -341.259, -460.184]

        self.AE6MoleculeName = ['SiH4', 'S2', 'SiO', 'C3H4_C3v', 'OCHCHO', 'cyclobutane']
        self.AE6 = {}

    def GetAtoms(self):
        return self.MGAtoms

    def GetMolecule(self, name):
        return g2.data[name]

    def GetMoleculeEnergy(self, name):
        data = g2.data[name]
        return data['enthalpy']

    def GetMoleculeZPE(self, name):
        data = g2.data[name]
        return data['ZPE']

    def GetMoleculeGeometry(self, name):
        data = g2.data[name]
        geometry = Atoms(symbols=data['symbols'], positions=data['positions'])
        return geometry

    def CreateAE6(self):
        names = ['SiH4', 'S2', 'SiO', 'C3H4_C3v', 'OCHCHO', 'cyclobutane']
        for name in names:
            self.AE6.update(g2.data[name])

    def GetAE6Names(self):
        return self.AE6MoleculeName

    def GetAE6Geometries(self):
        geometries = []
        for name in self.AE6MoleculeName:
            data = g2.data[name]
            geometries.append(Atoms(symbols=data['symbols'], positions=data['positions']))
        return geometries

    def GetAE6Energies(self):
        energies = []
        for name in self.AE6MoleculeName:
            data = g2.data[name]
            energies.append(data['enthalpy'])
        return energies