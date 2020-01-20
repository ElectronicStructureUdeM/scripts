atoms={"H":1,"He":0,"Li":1,"Be":0,"B":1,"C":2,"N":3,
        "O":2,
         "F":1,"Ne":0,"Na":1,"Ne":0,"Na":1,
        "Mg":0,"Al":1,"Si":2,"P":3,"S":2,"Cl":1,"Ar":0}

molecules={'HH': [0, [[0, 0, -0.0259603084], [0, 0, 0.7259603084]]],
'LiH': [0, [[0, 0, 0.3962543501], [0, 0, 2.0037456499]]],
'CHHHH': [0, [[-1.629e-06, 0.0, 7.8502e-06], [-2.2937e-06, 0.0, 1.0970267936], [1.0342803963, 0.0, -0.3656807611], [-0.5171382367, -0.8957112847, -0.3656769413], [-0.5171382368, 0.8957112847, -0.3656769413]]], 'NHHH': [0, [[-0.7080703847, 0.5736644371, -0.2056610779], [0.314047869, 0.6090902876, -0.2439925162], [-1.0241861213, 0.328070168, -1.147576524], [-1.024191363, 1.5321151073, -0.0355598819]]],
'OHH': [0, [[-0.7435290312, -0.0862560218, -0.2491318075], [0.2269625234, -0.0687025898, -0.2099668601], [-1.0265534922, 0.2938386117, 0.5988786675]]],
'FH': [0, [[0, 0, -0.0161104113], [0, 0, 0.9161104113]]],
'LiLi': [0, [[0, 0, -0.0155360351], [0, 0, 2.7155360351]]],
 'LiF': [0, [[0, 0, 0.0578619642], [0, 0, 1.6421380358]]],
 'BeBe': [0, [[0, 0, 0.0085515554], [0, 0, 2.4414484446]]],
 'CHCH': [0, [[-7.5637480678, -4.08536579, 0.0], [-8.6353642657, -4.08536579, 0.0], [-6.3570037122, -4.08536579, 0.0], [-5.2853875143, -4.08536579, 0.0]]],
 'CCHHHH': [0, [[-4.5194036917, 0.9995360751, -2.41325e-05], [-3.1861963083, 0.9995360751, -2.41325e-05], [-5.0929778983, 0.1325558381, -0.3377273553], [-5.0929780326, 1.866487909, 0.3377519084], [-2.6126221017, 0.1325558381, -0.3377273553], [-2.6126219674, 1.866487909, 0.3377519084]]], 'HCN': [0, [[-2.1652707291, 0.99953, 0.0], [-3.242302537, 0.99953, 0.0], [-4.4007967339, 0.99953, 0.0]]], 'CO': [0, [[0, 0, -0.0185570711], [0, 0, 1.1185570711]]], 'NN': [0, [[0, 0, -0.0017036831], [0, 0, 1.1017036831]]], 'NO': [1, [[0, 0, -0.0797720915], [0, 0, 1.0797720915]]], 'OO': [2, [[0, 0, -0.0114390797], [0, 0, 1.2114390797]]], 'FF': [0, [[0, 0, -0.0083068123], [0, 0, 1.4083068123]]], 'PP': [0, [[0, 0, -0.0063578484], [0, 0, 1.9063578484]]],
 'ClCl': [0, [[0, 0, -0.0645570711], [0, 0, 1.9645570711]]]}

#optimized with pbe/cc-pvtz
cations_atomisation={'HH': [1, [[0, 0, -0.16549127], [0, 0, 0.96549095]]],
           'HeHe':[1,[[0, 0,0.16204284] , [0, 0, 1.33795656]]],
           'BB':[1,[[0, 0,-0.15411969] , [0, 0, 1.65411909]]],
           'NeNe':[1,[[0, 0,0.03090072] , [0, 0,1.96909848 ]]],
           'ArAr':[1,[[0, 0,-0.28261396] , [0, 0, 2.28261317]]]}

#obtained from  https://comp.chem.umn.edu/db/dbs/ip21.html
IP13_neutrals={'C': [2, [[0, 0, 0.]]],
              'S': [2, [[0, 0, 0.]]],
              'SH':[1,[[0,0,0],[0,0,1.3402]]],
              'Cl': [1, [[0, 0, 0.]]],
              'ClCl':[0,[[0,0,0],[0,0,2.00783]]],
              'OH':[1,[[0,0,0],[0,0,0.96890]]],
              'O': [2, [[0, 0, 0.]]],
              'OO':[2,[[0,0,0],[0,0,1.20132]]],
              'P': [3, [[0, 0, 0.]]],
              'PH':[2,[[0,0,0],[0,0,1.42202]]],
              'PHH':[1,[[0,0,-0.11566],[1.02013,0,0.86743],[-1.02013 ,0,0.86743]]],
              'SS':[2,[[0,0,0],[0,0,1.89259]]],
              'Si':[2,[[0,0,0]]]
              }

IP13_cations={'C': [1, [[0, 0, 0.]]],
              'S': [3, [[0, 0, 0.]]],
              'SH':[2,[[0,0,0],[0,0,1.36126]]],
              'Cl': [2, [[0, 0, 0.]]],
              'ClCl':[1,[[0,0,0],[0,0,1.90094]]],
              'OH':[2,[[0,0,0],[0,0,1.02658]]],
              'O': [3, [[0, 0, 0.]]],
              'OO':[1,[[0,0,0],[0,0,1.10999]]],
              'P': [2, [[0, 0, 0.]]],
              'PH':[1,[[0,0,0],[0,0,1.42258]]],
              'PHH':[0,[[0,0,0],[0,0,1.41743],[1.41577 ,0,-0.06860]]],
              'SS':[1,[[0,0,0],[0,0,1.82018843]]],
              'Si':[1,[[0,0,0]]]}

#BH6 data set, from https://comp.chem.umn.edu/db/dbs/htbh38.html
BH6_reactants={
              'OH':[1,[[0,0,0],[0,0,0.96890]]],
              'CHHHH':[0,[[0,0,0],[0,0,1.08744],[1.02525,0.,-0.36248],[-0.51263,0.88789,-0.36248],[-0.51263,-0.88789,-0.36248]]],
              'OHH':[0,[[0,0,0],[0,0,0.95691],[0.92636,0.,-0.23987]]],
              'CHHH':[1,[[0,0,0],[0,0,1.07732],[0.93298,0.,-0.53866],[-0.93298,0,-0.53866]]],
              'H':[1,[[0,0,0]]],
              'HH':[0,[[0,0,0],[0,0,0.74188]]],
              'O':[2,[[0,0,0]]],
              'SHH':[0,[[0,0,0.10252],[0,0.96625,-0.82015],[0,-0.96625,-0.82015]]],
              'SH':[1,[[0,0,0],[0,0,1.34020]]]
              }

BH6_TS={'COHHHHH':[1,[[-1.211487,0.007968,0.000407],[1.293965,-0.108694,0.000133],
                  [0.009476,-0.118020,0.002799],[-1.525529,-0.233250,1.010070],
                  [-1.430665,1.033233,-0.278082],[-1.552710,-0.710114,-0.737702],
                  [1.416636,0.849894,-0.000591]]],
        'HOH':[2,[[0,0,-0.860287],[0,0,0.329024],[0,0,-1.771905]]],
        'HSHH':[1,[[1.262097,-0.220097,0.],[0,0.223153,0.],[-0.500576,-1.115445,0.],[-0.761521,-2.234913,0.]]]}

