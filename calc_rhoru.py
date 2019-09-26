from pyscf import gto,scf,dft,lib
import numpy as np
import scipy
import calcrhoru
def gto_cart_norm(nx,ny,nz,alpha):
    """
    Compute the normalization constant for gaussian basis set.
    For details: https://theochem.github.io/horton/2.0.2/tech_ref_gaussian_basis.html
    Input:
        nx(int):     power of x
        nx(int):     power of y
        nx(int):     power of z
        alpha(float):exponant
    Return:
        N(float):    normalization constant
    """
    #convert to int, otherwise scipy will fail
    nx = int(nx)
    ny = int(ny)
    nz = int(nz)
    return np.sqrt((2.*alpha/np.pi)**(1.5)*(4*alpha)**(nx+ny+nz)/(
                    scipy.special.factorial2(2*nx-1)*
                    scipy.special.factorial2(2*ny-1)*
                    scipy.special.factorial2(2*nz-1)))

def get_aoCoef_pgInfo(mol):
    """
    Description:
        Function to get all the informations about the gaussian basis set
        to be used to compute the spherically averaged non local density
    
    Input:
        mol(Mole Object): Mole Object from pyscf (atoms+basis set with cart=true)
    Return:
        aoCoef(numpy array of float): number of cartesian basis by 12
            the first element is the pointer to the position of the basis in pginfo
            the second element is the pointer to the end position in pginfo
            the rest are the contraction coefficient
        pgInfo(numpy array of float): total number of primitives, 8
            exponent of the gaussian
            Rx,Ry,Rz (position of the gaussian)
            l,m,n power for the polynomial in cartesian basis
            N: normalization constant
    """
    #number of contracted  cartesian GTOs basis
    nBasCart = mol.nao_cart()
    # number of shells
    nShells = mol.nbas
    # number of primitives
    nPrims = mol.npgto_nr()

    #The first and second element are the pointer to begining and end position
    #The rests are the contraintion coefficient of the gaussian primitive
    aoCoef = np.zeros((nBasCart,12))
    #in this order: exponent, position (x,y,z), power in cartesian cordonaites(l,m,n), normalization constant
    pgInfo = np.zeros((nPrims,8))
    nBas=0# index of bas
    iPrims=0 # index of primitive
    iBasBgn=1 #index for the psotion of bas
    #loop for each shell
    for shell in range(nShells):
        l = mol.bas_angular(shell)
        cartPosPow=0# position in pginfo to change the power of the polynomial in cartesian
        #loop for each cartesian basis for the shell (1 for s, 3 for p, 6 for d)
        for i in range(mol.bas_len_cart(shell)):
            aoCoef[nBas,0]=iBasBgn
            aoCoef[nBas,1]=iBasBgn+mol.bas_nprim(shell)-1
            ctrCoeff = mol.bas_ctr_coeff(shell).T[0]
            aoCoef[nBas,2:2+len(ctrCoeff)]=ctrCoeff
            #loop over each primitive
            for j in range(mol.bas_nprim(shell)):
                pgInfo[iPrims,0]=mol.bas_exp(shell)[j]
                pgInfo[iPrims,1:4]=mol.bas_coord(shell).T
                if (l==1):# for p orbital
                    pgInfo[iPrims,4+cartPosPow]=1
                if (l==2):#for d orbitals, the order is the same as pyscf
                    if(cartPosPow==0):#xx
                        pgInfo[iPrims,4]=2
                    if(cartPosPow==1):#xy
                        pgInfo[iPrims,4]=1
                        pgInfo[iPrims,5]=1   
                    if(cartPosPow==2):#xz
                        pgInfo[iPrims,4]=1
                        pgInfo[iPrims,6]=1                
                    if(cartPosPow==3):#yy
                        pgInfo[iPrims,5]=2
                    if(cartPosPow==4):#yz
                        pgInfo[iPrims,5]=1
                        pgInfo[iPrims,6]=1
                    if(cartPosPow==5):#zz
                        pgInfo[iPrims,6]=2
                if (l==2):
                    pgInfo[iPrims,7]=mol.gto_norm(l,pgInfo[iPrims,0]) # for some reason we
                                                                    #need to normalization
                                                                    #of only the radiall part
                else:
                    pgInfo[iPrims,7]=gto_cart_norm(pgInfo[iPrims,4],pgInfo[iPrims,5],
                                                pgInfo[iPrims,6],pgInfo[iPrims,0])
                iPrims=iPrims+1
                iBasBgn=iBasBgn+1
            cartPosPow=cartPosPow+1
            nBas = nBas+1
    return aoCoef,pgInfo

def calc_rhoru(mol,mf,grids):
    aoCoef,pgInfo = get_aoCoef_pgInfo(mol)
# generate all the u point
    nU = 2500
    smallU = 1e-3
    maxU= 50.0
    ux, uwei = calcrhoru.azwghtuni(smallU,nU,maxU)
    uwei[0]=uwei[0]+smallU/2.
    #calculate rhoru for a all grid
    nGrid = np.shape(grids.coords)[0]
    rhoruA=np.zeros((nGrid,nU))
    if (mol.nelectron==1):
        NMOA = np.count_nonzero(mf.mo_occ[0])
    if (mol.spin==0 and mol.nelectron>1):
        NMOA = np.count_nonzero(mf.mo_occ)
    if (mol.spin>0 and mol.nelectron>1):
        rhoruB=np.zeros((nGrid,nU))
        NMOA = np.count_nonzero(mf.mo_occ[0])
        NMOB = np.count_nonzero(mf.mo_occ[1])
    if (mol.nelectron==1):
        rhoruA = calcrhoru.calcrhoru(NMOA,aoCoef,pgInfo,grids.coords,mf.mo_coeff[0],ux)
    if (mol.spin==0 and mol.nelectron>1):
        rhoruA = calcrhoru.calcrhoru(NMOA,aoCoef,pgInfo,grids.coords,mf.mo_coeff,ux)
    if (mol.spin>0 and mol.nelectron>1):
        #alpha
        rhoruA,rhoruB = calcrhoru.calcrhorupol(NMOA,NMOB,aoCoef,pgInfo,grids.coords,
                        mf.mo_coeff[0],mf.mo_coeff[1],ux)
    if mol.spin==0 or mol.nelectron==1:return ux,uwei,rhoruA,rhoruA
    else:return ux,uwei,rhoruA,rhoruB

def output_rhoRU_atoms():
    atoms={"H":1,"He":0,"Li":1,"Be":0,"B":1,"C":2,"N":3,
        "O":2,"F":1,"Ne":0,"Na":1,"Ne":0,"Na":1,
        "Mg":0,"Al":1,"Si":2,"P":3,"S":2,"Cl":1,"Ar":0}
    for atom in atoms:
        print("Begin atom: "+atom)
        mol = gto.Mole()
        mol.atom=atom
        mol.cart=True
        mol.spin=atoms[atom]
        mol.basis = '6-311+g2dp.nw'
        mol.build()
        mf = scf.KS(mol)
        mf.small_rho_cutoff = 1e-12
        mf.xc='pbe'
        mf.grids.radi_method=dft.radi.delley
        mf.kernel()

        #grid
        grids = mf.grids
        ux,uwei,rhoRUA,rhoRUB= calc_rhoru(mol,mf,grids)
        np.save(atom,[ux,uwei,rhoRUA,rhoRUB])
def output_rhoRU_mol(molec,positions,spin):
    print("Begin mol: "+molec)
    mol = gto.Mole()
    atoms = re.findall('[A-Z][^A-Z]*', molec)
    molecule =[]
    nAtom=0
    for atom in atoms:
        atom_pos = positions[nAtom]
        molecule.append([atom,(atom_pos[0],atom_pos[1],atom_pos[2])])
        nAtom=nAtom+1
    mol.atom=molecule
    mol.cart=True
    mol.spin=spin
    mol.basis = '6-311+g2dp.nw'
    mol.build()
    mf = scf.KS(mol)
    mf.small_rho_cutoff = 1e-12
    mf.xc='pbe'
    mf.grids.radi_method=dft.radi.delley
    mf.kernel()

    #grid
    grids = mf.grids
    ux,uwei,rhoRUA,rhoRUB= calc_rhoru(mol,mf,grids)
    if spin==0:
        np.save(molec,[ux,uwei,rhoRUA])
    else:
        np.save(molec,[ux,uwei,rhoRUA,rhoRUB])

lib.num_threads(8)

#H2
mol="HH"
spin=0
positions = [[0, 0, -0.0259603084],
                [0, 0, 0.7259603084]]
output_rhoRU_mol(mol,positions,spin)
#LiH
mol="LiH"
spin=0
positions= [[0, 0, 0.3962543501],
                [0, 0, 2.0037456499]]
output_rhoRU_mol(mol,positions,spin)
#CH4
mol = "CHHHH"
spin=0
positions = [[-0.0000016290,0.00000,0.0000078502],
                  [-0.0000022937,0.00000,1.0970267936],
                  [1.0342803963,0.00000,-0.3656807611],
                  [-0.5171382367,-0.8957112847,-0.3656769413],
                  [-0.5171382368,0.8957112847,-0.3656769413]]
output_rhoRU_mol(mol,positions,spin )
#NH3
mol = "NHHH"
spin=0
positions = [[-0.7080703847,0.5736644371,-0.2056610779],
                  [0.3140478690,0.6090902876,-0.2439925162],
                  [ -1.0241861213,0.3280701680,-1.1475765240],
                  [-1.0241913630,1.5321151073,-0.0355598819]]
output_rhoRU_mol(mol,positions,spin )
#H2O
mol = "OHH"
spin=0
positions = [[-0.7435290312,-0.0862560218,-0.2491318075],
                  [0.2269625234,-0.0687025898,-0.2099668601],
                  [ -1.0265534922,0.2938386117,0.5988786675]]
output_rhoRU_mol(mol,positions,spin )
#HF
mol="FH"
spin=0
positions= [[0, 0, -0.0161104113],[0, 0, 0.9161104113]]
output_rhoRU_mol(mol,positions,spin )
#Li2
mol="LiLi"
spin=0
positions= [[0, 0, -0.0155360351],[0, 0, 2.7155360351]]
output_rhoRU_mol(mol,positions,spin )
#LiF
mol="LiF"
spin=0
positions= [[0, 0, 0.0578619642],[0, 0, 1.6421380358]]
output_rhoRU_mol(mol,positions,spin )
#Be2
mol="BeBe"
spin=0
positions= [[0, 0, 0.0085515554],[0, 0, 2.4414484446]]
output_rhoRU_mol(mol,positions,spin )
#C2H2
mol = "CHCH"
spin=0
positions = [[ -7.5637480678 ,-4.0853657900,0.00000000],
                  [-8.6353642657,-4.0853657900,0.00000000],
                  [-6.3570037122,-4.0853657900,0.00000000],
                  [-5.2853875143,-4.0853657900,0.00000000]]
output_rhoRU_mol(mol,positions,spin )
#C2H4
mol = "CCHHHH"
spin=0
positions = [[-4.5194036917,0.9995360751,-0.0000241325],
                  [-3.1861963083,0.9995360751,-0.0000241325],
                  [-5.0929778983,0.1325558381,-0.3377273553],
                  [-5.0929780326,1.8664879090,0.3377519084],
                  [ -2.6126221017,0.1325558381,-0.3377273553],
                  [-2.6126219674,1.8664879090,0.3377519084]]
output_rhoRU_mol(mol,positions,spin )
#HCN
mol = "HCN"
spin=0
positions = [[ -2.1652707291,0.9995300000,0.0000000000],
                  [-3.2423025370,0.9995300000,0.0000000000],
                  [-4.4007967339,0.9995300000,0.0000000000]]
output_rhoRU_mol(mol,positions,spin )
#CO
mol="CO"
spin=0
positions= [[0, 0, -0.0185570711],
                [0, 0, 1.1185570711]]
output_rhoRU_mol(mol,positions,spin )
#N2
mol="NN"
spin=0
positions= [[0, 0, -0.0017036831],
                [0, 0, 1.1017036831]]
output_rhoRU_mol(mol,positions,spin )
#NO
mol="NO"
spin=1
positions= [[0, 0, -0.0797720915],
                [0, 0, 1.0797720915]]
output_rhoRU_mol(mol,positions,spin )
#triplet O2
mol="OO"
spin=2
positions= [[0, 0, -0.0114390797],
                [0, 0, 1.2114390797]]
output_rhoRU_mol(mol,positions,spin )
#F2
mol="FF"
spin=0
positions= [[0, 0, -0.0083068123],
                [0, 0, 1.4083068123]]
output_rhoRU_mol(mol,positions,spin )
#P2
mol="PP"
spin=0
positions= [[0, 0, -0.0063578484],
                [0, 0, 1.9063578484]]
output_rhoRU_mol(mol,positions,spin )
#Cl2
mol="ClCl"
spin=0
positions= [[0, 0, -0.0645570711],
                [0, 0, 1.9645570711]]
output_rhoRU_mol(mol,positions,spin )