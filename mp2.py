def myrmp2(mf,mol):
    import numpy
    from pyscf import ao2mo

    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    o = mo_coeff[:,mo_occ>0]
    v = mo_coeff[:,mo_occ==0]
    eo = mo_energy[mo_occ>0] 
    ev = mo_energy[mo_occ==0]
    no = o.shape[1]
    nv = v.shape[1]
    noa = sum(mo_occ>0)
    nva = sum(mo_occ==0)
    eri = ao2mo.general(mf.mol, (o,v,o,v)).reshape(no,nv,no,nv)

    g = 4*eri - 2*eri.transpose(0,3,2,1)
    eov = eo.reshape(-1,1) - ev.reshape(-1)
    de = 1/(eov.reshape(-1,1) + eov.reshape(-1)).reshape(g.shape)
    emp2 = 0.5 * numpy.einsum('iajb,iajb,iajb->', g, eri, de)

    GV = numpy.einsum('iajb,iajb->iajb', g, de)
    
    GV = numpy.einsum('ab,bcde->acde', o, GV)    # i
    GV = numpy.einsum('abcd,eb->aecd', GV, v)    # j
    GV = numpy.einsum('ad,bcde->bcae', o, GV)    # k
    GV = numpy.einsum('abcd,ed->abce', GV, v)    # l 

#    eri2 = mol.intor('int2e')
#    EMP2 = 0.5*numpy.einsum('ijkl,ijkl->',gamma,eri2)
    
    print(str(emp2))
#    print(str(emp2p))
#    print(str(EMP2))
    return GV

def oppspin(mf,mol,case):
    import numpy
    from pyscf import ao2mo

    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff

    if case ==0:
       s1=0
       s2=1
    if case ==1:
       s1=1
       s2=0

#   Calculate alpha-beta constribution 

    o1 = mo_coeff[s1][:,mo_occ[s1]>0] 
    v1 = mo_coeff[s1][:,mo_occ[s1]==0]
    o2 = mo_coeff[s2][:,mo_occ[s2]>0] 
    v2 = mo_coeff[s2][:,mo_occ[s2]==0]
    
    eo1 = mo_energy[s1][mo_occ[s1]>0]
    ev1 = mo_energy[s1][mo_occ[s1]==0]
    eo2 = mo_energy[s2][mo_occ[s2]>0]
    ev2 = mo_energy[s2][mo_occ[s2]==0]

    no1 = o1.shape[1]
    nv1 = v1.shape[1]
    no2 = o2.shape[1]
    nv2 = v2.shape[1]

    noa = sum(mo_occ[0]>0)
    nva = sum(mo_occ[0]==0)

    eri = ao2mo.general(mf.mol, (o1,v1,o2,v2)).reshape(no1,nv1,no2,nv2)
    g = eri
    eov1 = eo1.reshape(-1,1) - ev1.reshape(-1)
    eov2 = eo2.reshape(-1,1) - ev2.reshape(-1)
    de = 1/(eov1.reshape(-1,1) + eov2.reshape(-1)).reshape(eri.shape)
    emp2 =  numpy.einsum('iajb,iajb,iajb->', eri, eri, de)

    GV = numpy.einsum('iajb,iajb->iajb', g, de)
#    gamma = gamma - gamma.transpose(0,3,2,1)
#    emp2p = 0.5 * numpy.einsum('iajb,iajb->', gamma, eri)
    
    GV = numpy.einsum('ab,bcde->acde', o1, GV)    # i
    GV = numpy.einsum('abcd,eb->aecd', GV, v1)    # j
    GV = numpy.einsum('ad,bcde->bcae', o2, GV)    # k
    GV = numpy.einsum('abcd,ed->abce', GV, v2)    # l 

#    eri2 = mol.intor('int2e')
#    EMP2 = 0.5*numpy.einsum('ijkl,ijkl->',gamma,eri2)
    
    print(str(emp2))
#    print(str(EMP2))
    return GV

def samespin(mf,mol,case):
    import numpy
    from pyscf import ao2mo
    
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff

    if case ==0:
       s=0
    if case ==1:
       s=1

    o = mo_coeff[s][:,mo_occ[s]>0] 
    v = mo_coeff[s][:,mo_occ[s]==0]

    eo = mo_energy[s][mo_occ[s]>0]
    ev = mo_energy[s][mo_occ[s]==0]

    no = o.shape[1]
    nv = v.shape[1]

    noa = sum(mo_occ[s]>0)
    nva = sum(mo_occ[s]==0)

    eri = ao2mo.general(mf.mol, (o,v,o,v)).reshape(no,nv,no,nv)
    g = eri - eri.transpose(0,3,2,1)
    eov = eo.reshape(-1,1) - ev.reshape(-1)
    de = 1/(eov.reshape(-1,1) + eov.reshape(-1)).reshape(eri.shape)
    emp2 =  numpy.einsum('iajb,iajb,iajb->', g, g, de)

    gamma = numpy.einsum('iajb,iajb->iajb', g, de)
    gamma = gamma - gamma.transpose(0,3,2,1)
    emp2p =  numpy.einsum('iajb,iajb->', gamma, eri)
    GV =  gamma
    
    GV = numpy.einsum('ab,bcde->acde', o, GV)    # i
    GV = numpy.einsum('abcd,eb->aecd', GV, v)    # j
    GV = numpy.einsum('ad,bcde->bcae', o, GV)    # k
    GV = numpy.einsum('abcd,ed->abce', GV, v)    # l 

#    eri2 = mol.intor('int2e')
    #print(eri2.shape)
#    EMP2 = 0.5*numpy.einsum('ijkl,ijkl->',gamma,eri2)
    
#    print(str(EMP2))
    return GV


def formGAMUMP2(mf,mol):
   GAMMP2 = oppspin(mf,mol,0)
   GAMMP2 += oppspin(mf,mol,1)
   GAMMP2 += 0.5*samespin(mf,mol,0)
   GAMMP2 += 0.5*samespin(mf,mol,1)
   return GAMMP2

