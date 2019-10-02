*Deck CalcRhoRU
      Subroutine CalcRhoRU(Ngrid,NBasCart,NMO,NPrims,AOCoef,PGInfo,
     $ GridTot,CMO,RhoRU,U,NU)
      Implicit Real*8(A-H,O-Z)
      integer OMP_GET_THREAD_NUM
Cf2py intent(out) RhoRU
C-----------------------------------------------------------------------
C     Calculate Rho(R,u)= Sum_i Rho_i(R,u), where Rho_i(r,u) is 
C     calculated from the subroutine azCalRMOU in azCalRMOU.F
C     It is calculated for all  the grid point
C-----------------------------------------------------------------------
C     Input:
C       NGrid:  number of grid point 
C       NBasCart: Number of cartesian basis functions
C       NMO:    Number of MOs
C       Nprism: Number of primite gaussian
C       AOcoef: Atomic orbital contraction coefficient
C       PGInfo: Primitive Gaussian informations
C       grid:   grid(x,y,z)
C       CMO:    molecular orbital coefficient
C       U1:     sphere radius
C       NU:     number of u
C     Output:
C       RhoRU:  Rho(R,u)
C-----------------------------------------------------------------------
C     Version 1.00, given by Yongxi on 10/10/18
C     Version 1.01, for only one grid point by EC 04/12/18
C     Version 2.00, adapted for pyscf by EC 13/09/19
C
      Dimension AOCoef(NBasCart,12),PGInfo(NPrims,8),GridTot(Ngrid,3),
     $ CMO(NBasCart,*),u(nu),rhoRU(ngrid,nu),RAOU(NBasCart,NBasCart),
     $ itresh(NBasCart,NBasCart),grid(3)
C
C-----------------------------------------------------------------------
C     Calculate rho(R,u) 
      ! clear the tresh matrice to wich will have 0 if the MO coeff are too smal
      CThr = 1.D-12
!$OMP PARALLEL PRIVATE(iGrid,itresh,RAOU,grid) shared(rhoru)
!$OMP DO 
      do iGrid=1,ngrid
        grid(1)=gridTot(igrid,1)
        grid(2)=gridTot(igrid,2)
        grid(3)=gridTot(igrid,3)
        do i=1,NBasCart
          do j=i,NBasCart
            itresh(i,j)=0
          enddo
        enddo
        !check if the coefficient are small, increament itresh if they are
        do imo=1,NMO
          do i=1,NBasCart
            do j=i,NBasCart
              If((dabs(cmo(i,imo)).gt.CThr).and.(dabs(cmo(j,imo)).gt.
     $       CThr)) itresh(i,j)=itresh(i,j)+1
            enddo
          enddo
        enddo
        !calculate rho(r,u)
        do iu=1, nU
          rhoru(iGrid,iu)=0.D0
          do i=1,NBasCart
            do j=i,NBasCart
              if (itresh(i,j).gt.0) then
                Call azCalRAOU(NBasCart,AOCoef,PGInfo,Grid,
     $ RAOU(i,j),NPrims,U(iu),i,j) 
              Else
                raou(i,j)=0.D0
              endif
            enddo
          enddo
          Do iMO=1,NMO
            do i=1,NBasCart
              do j=i,NBasCart
                rhoru(iGrid,iu)=rhoru(iGrid,iu)+
     $          cmo(i,imo)*cmo(j,imo)*RAOU(i,j)
              enddo
            enddo
          EndDo
          if (rhoru(iGrid,iu).lt.0)rhoru(iGrid,iu)=0.D0
        enddo !u
      enddo !grid
!$OMP END DO
!$OMP END PARALLEL
      Return
      End
*Deck CalcRhoRU
      Subroutine CalcRhoRUPol(Ngrid,NBasCart,NMOA,NMOB,NPrims,AOCoef,
     $ PGInfo,GridTot,CMOA,CMOB,RhoRUA,RhoRUB,U,NU)
      Implicit Real*8(A-H,O-Z)
Cf2py intent(out) RhoRUA
Cf2py intent(out) RhoRUB
C-----------------------------------------------------------------------
C     Calculate Rho(R,u)= Sum_i Rho_i(R,u), where Rho_i(r,u) is 
C     calculated from the subroutine azCalRMOU in azCalRMOU.F
C     It is calculated for all  the grid point, for spin polarized sytem
C     TODO: this is a repetition of code from the previous function
C           it should be made simpler
C-----------------------------------------------------------------------
C     Input:
C       NGrid:  number of grid point 
C       NBasCart: Number of cartesian basis functions
C       NMOA:    Number of MOs of alpha spin
C       NMOB:    Number of MOs of alpha spin
C       Nprism: Number of primite gaussian
C       AOcoef: Atomic orbital contraction coefficient
C       PGInfo: Primitive Gaussian informations
C       grid:   gridtot(x,y,z) for all gridpoint
C       CMO:    molecular orbital coefficient
C       U1:     sphere radius
C       NU:     number of u
C     Output:
C       RhoRUA:  Rhoalpha(R,u)
C       rhoruB: rhobeta(r,u)
C-----------------------------------------------------------------------
C     Version 1.00, given by Yongxi on 10/10/18
C     Version 1.01, for only one grid point by EC 04/12/18
C     Version 2.00, adapted for pyscf by EC 13/09/19
C
      Dimension AOCoef(NBasCart,12),PGInfo(NPrims,8),GridTot(Ngrid,3),
     $ CMOA(NBasCart,*),CMOB(NBasCart,*),u(nu),rhoRUA(ngrid,nu),
     $ rhoRUB(ngrid,nu),RAOU(NBasCart,NBasCart),
     $itresh(NBasCart,NBasCart),grid(3)
C
C-----------------------------------------------------------------------
C     Calculate rho(R,u) 
C clear the tresh matrice to wich will have 0 if the MO coeff are too smal
      CThr = 1.D-12
!$OMP PARALLEL PRIVATE(iGrid,itresh,RAOU,grid) SHARED(rhoruA,rhoruB)
!$OMP DO 
      do iGrid=1,ngrid
        grid(1)=gridTot(igrid,1)
        grid(2)=gridTot(igrid,2)
        grid(3)=gridTot(igrid,3)
        do i=1,NBasCart
          do j=i,NBasCart
            itresh(i,j)=0
          enddo
        enddo
        !check if the coefficient are small, increament itresh if they are
        do imo=1,NMOA 
          do i=1,NBasCart
            do j=i,NBasCart
              If((dabs(cmoA(i,imo)).gt.CThr).and.(dabs(cmoA(j,imo)).gt.
     $       CThr)) itresh(i,j)=itresh(i,j)+1
            enddo
          enddo
        enddo
        do imo=1,NMOB
          do i=1,NBasCart
            do j=i,NBasCart
              If((dabs(cmoB(i,imo)).gt.CThr).and.(dabs(cmoB(j,imo)).gt.
     $       CThr)) itresh(i,j)=itresh(i,j)+1
            enddo
          enddo
        enddo
        !calculate rho(r,u)
        do iu=1, nU
          rhoruA(iGrid,iu)=0.D0
          rhoruB(iGrid,iu)=0.D0
          do i=1,NBasCart
            do j=i,NBasCart
              if (itresh(i,j).gt.0) then
                Call azCalRAOU(NBasCart,AOCoef,PGInfo,Grid,
     $ RAOU(i,j),NPrims,U(iu),i,j) 
              Else
                raou(i,j)=0.D0
              endif
            enddo
          enddo
          Do iMO=1,NMOA
            do i=1,NBasCart
              do j=i,NBasCart
                rhoruA(iGrid,iu)=rhoruA(iGrid,iu)+
     $          cmoA(i,imo)*cmoA(j,imo)*RAOU(i,j)
              enddo
            enddo
          EndDo
          Do iMO=1,NMOB
            do i=1,NBasCart
              do j=i,NBasCart
                rhoruB(iGrid,iu)=rhoruB(iGrid,iu)+
     $          cmoB(i,imo)*cmoB(j,imo)*RAOU(i,j)
              enddo
            enddo
          EndDo
          if (rhoruA(iGrid,iu).lt.0)rhoruA(iGrid,iu)=0.D0
          if (rhoruB(iGrid,iu).lt.0)rhoruB(iGrid,iu)=0.D0
        enddo !u
      enddo !grid
!$OMP END DO
!$OMP END PARALLEL
      Return
      End
      
      Subroutine azCalRAOU(NBasis,AOCoef,PGInfo,Grid,RAO1dU,
     $    NPrims,U,i,j)
      Implicit Real*8(A-H,O-Z)
Cf2py intent(out) RMO1dU
C
C     Evaluate Rho_i(R,u)
C
      Dimension Grid(3)
      Dimension AOCoef(NBasis,12),PGInfo(NPrims,8),Rmu(3),Kmu(3),Rnu(3),
     $ Knu(3)
C
C     initiate it
      CThr=1.D-12
C
!       Do i=1,NBasis
      RAO1dU=0.D0
       NiPG = AOCoef(i,2) - AOCoef(i,1) + 1
       iStPos = AOCoef(i,1)
!      Do j=i,NBasis !EC for symmetric sum, we do not want to calculate i=j
       NjPG = AOCoef(j,2) - AOCoef(j,1) + 1
       jStPos = AOCoef(j,1)
C       If((dabs(Calpha).gt.CThr).and.(dabs(Cbeta).gt.CThr)) then
       Do mu=1,NiPG   ! p in the manuscript
       Do nu=1,NjPG   ! q in the manuscript
       Dmu = AOCoef(i,2+mu)
       Dnu = AOCoef(j,2+nu)
       muPosPG = iStPos + mu - 1
       nuPosPG = jStPos + nu - 1
       ! mu info: exponent, R, K, Nmu
       Amu = PGInfo(muPosPG,1)
       Rmu(1) = PGInfo(muPosPG,2)
       Rmu(2) = PGInfo(muPosPG,3)
       Rmu(3) = PGInfo(muPosPG,4)
       Kmu(1) = Int(PGInfo(muPosPG,5))
       Kmu(2) = Int(PGInfo(muPosPG,6))
       Kmu(3) = Int(PGInfo(muPosPG,7))
       dNmu = PGInfo(muPosPG,8)
       ! nu info: exponent, R, K, Nnu
       Anu = PGInfo(nuPosPG,1)
       Rnu(1) = PGInfo(nuPosPG,2)
       Rnu(2) = PGInfo(nuPosPG,3)
       Rnu(3) = PGInfo(nuPosPG,4)
       Knu(1) = Int(PGInfo(nuPosPG,5))
       Knu(2) = Int(PGInfo(nuPosPG,6))
       Knu(3) = Int(PGInfo(nuPosPG,7))
       dNnu = PGInfo(nuPosPG,8)
       Call azCalChiUIntPrd(Amu,Anu,Rmu,Rnu,Kmu,Knu,U,
     $ ChiU,Grid,dNmu,dNnu)
          if (i.ne.j)RAO1dU=RAO1dU+Dmu*Dnu*ChiU !ECmultipled by two when i is not equal to J for symmetric sum
          RAO1dU=RAO1dU+Dmu*Dnu*ChiU
        EndDo  !p
        EndDo  !q
C        Endif

!      EndDo  !j
!      EndDo  !i 
        If(RAO1dU.lt.0.D0) then
           RMO1dU=0.D0
         Endif
       Return
       End


*Deck azCalChiUIntPrd
      Subroutine azCalChiUIntPrd(Amu,Anu,Rmu,Rnu,Kmu,Knu,U,
     $ ChiU,Grid,dNmu,dNnu)
      Implicit Real*8(A-H,O-Z)
Cf2py intent(out) ChiU
C
C     Calculate Chi(R,u) from the product of two Chi. ChiU is spherically 
C     integrated over the angular part. Formulas for different values of 
C     Kmu and Knu are built from Mathematica.
C     Input:
C        Amu, Rmu, Kmu - alpha, center, momentum of 1st Chi
C        Anu, Rnu, Knu - alpha, center, momentum of 2nd Chi
C        U - sphere radius
C     Output:
C        ChiU   - Spherically integrated Chi1*Chi2
C
      Dimension Grid(3),Rmu(3),Rnu(3),Kmu(3),Knu(3)
      Dimension RA(3),RB(3),TA(3),TB(3),KA(3),KB(3)
C
      Drv=0.D0
C
        NAngA=Kmu(1)+Kmu(2)+Kmu(3)
        NAngB=Knu(1)+Knu(2)+Knu(3)
        MaxAngA=Max0(Kmu(1),Kmu(2),Kmu(3))
        MaxAngB=Max0(Knu(1),Knu(2),Knu(3))
        If (NAngA.le.NAngB.and.MaxAngA.le.MaxAngB) then
          Do k=1,3
            EA = Amu
            EB = Anu
            RA(k) = Grid(k) - Rmu(k)
            RB(k) = Grid(k) - Rnu(k)
            TA(k) = -2.D0*(EA*RA(k)+EB*RB(k)) 
            TB(k) = TA(k)
            KA(k) = Kmu(k)
            KB(k) = Knu(k)
          Enddo
        Else !switch A B
          Do k=1,3
            EA = Anu
            EB = Amu
            RA(k) = Grid(k) - Rnu(k)
            RB(k) = Grid(k) - Rmu(k)
            TA(k) = -2.D0*(EA*RA(k)+EB*RB(k)) 
            TB(k) = TA(k)
            KA(k) = Knu(k)
            KB(k) = Kmu(k)
          Enddo
        Endif
        Call CalcDrvAI(KA,KB,EA,EB,RA,RB,TA,TB,U,Drv) 
        ChiU=dNmu*dNnu*Drv
      Return
      End

*Deck SortAngP
      Subroutine GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
      Implicit Real*8(A-H,O-Z)
C
C     Prepare arguments for spherical integral
C
      Dimension RA(3),RB(3),TA(3),TB(3)
      ExpFac = -EA*(RA(1)**2 + RA(2)**2 + RA(3)**2 + U**2)
     $  -EB*(RB(1)**2 + RB(2)**2 + RB(3)**2 + U**2)
      TAx = TA(1)
      TAy = TA(2)
      TAz = TA(3)
      TBx = TB(1)
      TBy = TB(2)
      TBz = TB(3)
      RAx = RA(1)
      RAy = RA(2)
      RAz = RA(3)
      RBx = RB(1)
      RBy = RB(2)
      RBz = RB(3)
      WSqr = TAx**2 + TAy**2 + TAz**2
      W = dSqrt(WSqr)
      Tmp=U*W
      !If (Tmp.gt.1.D1) then  !1.D-2
      If (Tmp.gt.1.D-1) then  !1.D-2
        Sinhuw = (DExp(Tmp+ExpFac)-DExp(-Tmp+ExpFac))/2.D0
        Coshuw = (DExp(Tmp+ExpFac)+DExp(-Tmp+ExpFac))/2.D0
      Else
        Sinhuw = Dsinh(Tmp)*DExp(ExpFac)
        Coshuw = Dcosh(Tmp)*DExp(ExpFac)
        !Sinhuw = (SmlExp(Tmp)-SmlExp(-Tmp))/2.D0*DExp(ExpFac)
        !Coshuw = ((SmlExp(Tmp)+SmlExp(-Tmp))/2.D0+1.D0)*DExp(ExpFac)
      Endif
      Return
      END

*Deck SmlExp
      Real*8 Function SmlExp(x)
      Implicit Real*8(A-H,O-Z)
C     x -> 0.D0, only keep the decimal digit
      SmlExp=x + x**2D0/2D0 + x**3D0/6D0 + x**4D0/24D0 + x**5D0/120D0
      Return
      End

*Deck SortAngP
      Subroutine SortAngP(iPos,KA,RA,TA)
      Implicit Real*8(A-H,O-Z)
C
C     Sort the x,y,z to apply symmetry (1,0,0)/(0,1,0)
C
      Dimension KA(3),RA(3),TA(3)
      Do i=1,3
        If (KA(i).eq.1) then
          Temp=RA(iPos)
          RA(iPos)=RA(i)
          RA(i)=Temp
          Temp=TA(iPos)
          TA(iPos)=TA(i)
          TA(i)=Temp
          Goto 10
        Endif
      Enddo
10    Continue
      Return
      End

*Deck SortAngD1
      Subroutine SortAngD1(iPos,KA,RA,TA)
      Implicit Real*8(A-H,O-Z)
C
C     Sort the x,y,z to apply symmetry (1,1,0)/(1,0,1)---iPos:3/2
C
      Dimension KA(3),RA(3),TA(3)
      Do i=1,3
        If (KA(i).eq.0) then
           Temp=RA(iPos)
           RA(iPos)=RA(i)
           RA(i)=Temp
           Temp=TA(iPos)
           TA(iPos)=TA(i)
           TA(i)=Temp
           Goto 30
         Endif
      Enddo
30    Continue
      Return
      End

*Deck SwitchRATA
      Subroutine SwitchRT(iPo,iPt,RA,TA)
      Implicit Real*8(A-H,O-Z)
C
C     Switch positions
C
      Dimension RA(3),TA(3)
      Temp=RA(iPt)
      RA(iPt)=RA(iPo)
      RA(iPo)=Temp
      Temp=TA(iPt)
      TA(iPt)=TA(iPo)
      TA(iPo)=Temp
      Return
      End

*Deck SwitchK
      Subroutine SwitchK(iPo,iPt,KA,KB)
      Implicit Real*8(A-H,O-Z)
C
C     Switch positions
C
      Dimension KA(3),KB(3)
      iTemp=KA(iPt)
      KA(iPt)=KA(iPo)
      KA(iPo)=iTemp
      iTemp=KB(iPt)
      KB(iPt)=KB(iPo)
      KB(iPo)=iTemp
      Return
      End

*Deck SortAngD2
      Subroutine SortAngD2(iPos,KA,RA,TA)
      Implicit Real*8(A-H,O-Z)
C
C     Sort the x,y,z to apply symmetry (2,0,0)/(0,2,0)
C
      Dimension KA(3),RA(3),TA(3)
      Do i=1,3
        If (KA(i).eq.2) then
          Temp=RA(iPos)
          RA(iPos)=RA(i)
          RA(i)=Temp
          Temp=TA(iPos)
          TA(iPos)=TA(i)
          TA(i)=Temp
          Goto 20
        Endif
      Enddo
20    Continue
      Return
      End

*Deck CalcDrvAI
      SUBROUTINE CalcDrvAI(KA,KB,EA,EB,RA,RB,TA,TB,U,Drv)
      Implicit Real*8(A-H,O-Z)
C
C     6 dimension derivative, spherical integral part
C
      Dimension RA(3),RB(3),TA(3),TB(3),KA(3),KB(3)
      Real*8 TAx,TAy,TAz,TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,
     $ WSqr,W,Sinhuw,Coshuw
      Data iTgtP1,iTgtP2/1,2/,iTgtD1a,iTgtD1b/3,2/,iTgtD2a,iTgtD2b/1,2/
C
      NAngA=KA(1)+KA(2)+KA(3)
      NAngB=KB(1)+KB(2)+KB(3)
      MxA=Max0(KA(1),KA(2),KA(3))
      MxB=Max0(KB(1),KB(2),KB(3))
C     Do the case      
      If(NAngA.eq.0.and.NAngB.eq.0) then   
        !(*SvsS [000D0,000D0]*)
        Call GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
        Drv=Sinhuw/(U*dSqrt(WSqr))
      Else If(NAngA.eq.0.and.NAngB.eq.1) then 
        !(*SvsP [000D0,100D0]*)
        Call SortAngP(iTgtP1,KB,RB,TB)
        Call GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
        Drv=(-(Sinhuw*TBx) + Coshuw*TBx*U*dSqrt(WSqr) + 
     $ RBx*Sinhuw*WSqr)/(U*WSqr**1.5D0)
      Else If(NAngA.eq.0.and.NAngB.eq.2.and.MxB.eq.1) then 
        !(*SvsD1D0 [000D0,110D0]*)
        Call SortAngD1(iTgtD1a,KB,RB,TB)
        Call GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
        Drv=(Coshuw*U*DSqrt(WSqr)*(-3D0*TBx*TBy + RBy*TBx*WSqr + 
     $ RBx*TBy*WSqr) + Sinhuw*(RBx*WSqr*(-TBy + RBy*WSqr) + 
     $ TBx*(-(RBy*WSqr) + TBy*(3D0 + U**2D0*WSqr))))/(U*WSqr**2.5D0)
      Else If(NAngA.eq.0.and.NAngB.eq.2.and.MxB.eq.2) then 
        !(*SvsD2D0 [000D0,200D0]*)
        Call SortAngD2(iTgtD2a,KB,RB,TB)
        Call GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
        Drv=(Coshuw*U*DSqrt(WSqr)*(-3D0*TBx**2D0 + WSqr + 
     $ 2D0*RBx*TBx*WSqr) + Sinhuw*(-2D0*RBx*TBx*WSqr + WSqr*(-1D0 + 
     $ RBx**2D0*WSqr) + TBx**2D0*(3D0 + U**2D0*WSqr)))/(U*WSqr**2.5D0)
      Else If(NAngA.eq.1.and.NAngB.eq.1) then 
        If(KA(1).eq.KB(1).and.KA(2).eq.KB(2).and.KA(3).eq.KB(3)) then
        !(*PvsP-1D0 [100D0,100D0]*)
          Call SortAngP(iTgtP1,KA,RA,TA)
          Call SortAngP(iTgtP1,KB,RB,TB)
          Call GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
          Drv=(Coshuw*U*DSqrt(WSqr)*(-3D0*TAx*TBx+WSqr+RBx*TAx*WSqr + 
     $ RAx*TBx*WSqr) + Sinhuw*(3D0*TAx*TBx - (1D0 + RBx*TAx + RAx*TBx - 
     $ TAx*TBx*U**2D0)*WSqr + RAx*RBx*WSqr**2D0))/(U*WSqr**2.5D0)
        Else
        !(*PvsP-2D0 [100D0,010D0]*)
          Call SortAngP(iTgtP1,KA,RA,TA)
          Call SortAngP(iTgtP2,KB,RB,TB)
          Call GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
          Drv=(Coshuw*U*DSqrt(WSqr)*(-3D0*TAx*TBy + RBy*TAx*WSqr + 
     $ RAx*TBy*WSqr) + Sinhuw*(RAx*WSqr*(-TBy + RBy*WSqr) + 
     $ TAx*(-(RBy*WSqr) + TBy*(3D0 + U**2D0*WSqr))))/(U*WSqr**2.5D0)
        Endif
      Else If(NAngA.eq.1.and.NAngB.eq.2.and.MxB.eq.1) then 
        If(KA(1).ne.KB(1).and.KA(2).ne.KB(2).and.KA(3).ne.KB(3)) then
        !(*PvsD1D0-1D0 [010D0,101D0]*)
          Call SortAngP(iTgtP2,KA,RA,TA)
          Call SortAngD1(iTgtD1b,KB,RB,TB)
          Call GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
          Drv=(Coshuw*U*DSqrt(WSqr)*(RAy*WSqr*(-3D0*TBx*TBz +
     $ RBz*TBx*WSqr + RBx*TBz*WSqr)+TAy*(RBx*WSqr*(-3D0*TBz + RBz*WSqr)
     $ + TBx*(-3D0*RBz*WSqr + TBz*(15D0 + U**2D0*WSqr)))) + 
     $ Sinhuw*(RAy*WSqr*(RBx*WSqr*(-TBz + RBz*WSqr) + 
     $ TBx*(-(RBz*WSqr) + TBz*(3D0 + U**2D0*WSqr))) + 
     $ TAy*(RBx*WSqr*(-(RBz*WSqr) + TBz*(3D0 + U**2D0*WSqr)) + 
     $ TBx*(RBz*WSqr*(3D0 + U**2D0*WSqr) - 3D0*TBz*(5D0 + 
     $ 2D0*U**2D0*WSqr)))))/
     $ (U*WSqr**3.5D0)
        Else
        !(*PvsD1D0-2D0 [100D0,110D0]*)
          Do is=1,3
            If((KA(is).eq.1).and.(KA(is).eq.KB(is))) iSame=is
          Enddo
          Call SwitchK(1,iSame,KA,KB)
          Call SwitchRT(1,iSame,RA,TA)
          Call SwitchRT(1,iSame,RB,TB)
          Call SortAngP(iTgtP1,KA,RA,TA)
          Call SortAngD1(iTgtD1a,KB,RB,TB)
          Call GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
          Drv=(Coshuw*U*DSqrt(WSqr)*(15D0*TAx*TBx*TBy - 
     $ (3D0*RBy*TAx*TBx + 
     $ TBy*(3D0 + 3D0*RBx*TAx + 3D0*RAx*TBx - TAx*TBx*U**2D0))*WSqr + 
     $ (RBy + RBx*RBy*TAx + RAx*RBy*TBx + RAx*RBx*TBy)*WSqr**2D0) + 
     $ Sinhuw*(WSqr*(RBy*WSqr*(-1D0 - RAx*TBx + RAx*RBx*WSqr) + 
     $ TBy*(3D0 + 3D0*RAx*TBx + (-(RAx*RBx) + U**2D0 + 
     $ RAx*TBx*U**2D0)*WSqr)) + 
     $ TAx*(RBx*WSqr*(-(RBy*WSqr) + TBy*(3D0 + U**2D0*WSqr)) + 
     $ TBx*(RBy*WSqr*(3D0 + U**2D0*WSqr) - 3D0*TBy*(5D0 + 
     $ 2D0*U**2D0*WSqr)))))/
     $ (U*WSqr**3.5D0)     
        Endif
      Else If(NAngA.eq.1.and.NAngB.eq.2.and.MxB.eq.2) then 
        If(KA(1)+KB(1).eq.3.or.KA(2)+KB(2).eq.3
     $ .or.KA(3)+KB(3).eq.3) then
        !(*PvsD2D0-1D0 [100D0,200D0]*)
          Call SortAngP(iTgtP1,KA,RA,TA)
          Call SortAngD2(iTgtD2a,KB,RB,TB)
          Call GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
          Drv=(Coshuw*U*DSqrt(WSqr)*(WSqr*(-3D0*TBx*(2D0+RAx*TBx)+(RAx+ 
     $ 2D0*RBx + 2D0*RAx*RBx*TBx)*WSqr) + 
     $ TAx*(-6D0*RBx*TBx*WSqr+WSqr*(-3D0 + 
     $ RBx**2D0*WSqr) + TBx**2D0*(15D0 + U**2D0*WSqr))) + 
     $ Sinhuw*(TAx*(2D0*RBx*TBx*WSqr*(3D0 + U**2D0*WSqr) + WSqr*(3D0 - 
     $ RBx**2D0*WSqr + U**2D0*WSqr) - 3D0*TBx**2D0*(5D0 +
     $ 2D0*U**2D0*WSqr)) + 
     $ WSqr*(RAx*TBx**2D0*(3D0 + U**2D0*WSqr) + 
     $ 2D0*TBx*(3D0 - RAx*RBx*WSqr + 
     $ U**2D0*WSqr) + WSqr*(-2D0*RBx + RAx*(-1D0 + RBx**2D0*WSqr)))))/
     $ (U*WSqr**3.5D0)
        Else
        !(*PvsD2D0-2D0 [100D0,020D0]*)
          Call SortAngP(iTgtP1,KA,RA,TA)
          Call SortAngD2(iTgtD2b,KB,RB,TB)
          Call GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
          Drv=(Coshuw*U*DSqrt(WSqr)*(RAx*WSqr*(-3D0*TBy**2D0 + WSqr + 
     $ 2D0*RBy*TBy*WSqr) + TAx*(-6D0*RBy*TBy*WSqr +
     $ WSqr*(-3D0 + RBy**2D0*WSqr) 
     $ + TBy**2D0*(15D0 + U**2D0*WSqr))) + 
     $ Sinhuw*(RAx*WSqr*(-2D0*RBy*TBy*WSqr 
     $ + WSqr*(-1D0 + RBy**2D0*WSqr) + TBy**2D0*(3D0 + U**2D0*WSqr)) + 
     $ TAx*(2D0*RBy*TBy*WSqr*(3D0 + U**2D0*WSqr) + 
     $ WSqr*(3D0 - RBy**2D0*WSqr + 
     $ U**2D0*WSqr) - 3D0*TBy**2D0*(5D0 + 
     $ 2D0*U**2D0*WSqr))))/(U*WSqr**3.5D0)
        Endif
      Else If(NAngA.eq.2.and.MxA.eq.1.and.NAngB.eq.2.and.MxB.eq.1) then 
        If(KA(1).eq.KB(1).and.KA(2).eq.KB(2).and.KA(3).eq.KB(3)) then
        !(*D1D0vsD1D0-1D0 [110D0,110D0]*)
          Call SortAngD1(iTgtD1a,KA,RA,TA)
          Call SortAngD1(iTgtD1a,KB,RB,TB)
          Call GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
          Drv=(Coshuw*U*DSqrt(WSqr)*(WSqr*(WSqr*(-3D0*(1D0+ 
     $ RAx*TBx)*(1D0 + 
     $ RAy*TBy)+(RAy*RBy+RAx*(RBx+RAy*RBy*TBx + RAy*RBx*TBy))*WSqr)+ 
     $ TAy*(RBy*WSqr*(-3D0 - 3D0*RAx*TBx + RAx*RBx*WSqr) + TBy*(15D0 + 
     $ 15D0*RAx*TBx+(-3D0*RAx*RBx + U**2D0 + RAx*TBx*U**2D0)*WSqr))) + 
     $ TAx*(WSqr*(RBx*WSqr*(-3D0 - 3D0*RAy*TBy + RAy*RBy*WSqr) + 
     $ TBx*(15D0 + 
     $ 15D0*RAy*TBy +(-3D0*RAy*RBy + U**2D0 + RAy*TBy*U**2D0)*WSqr)) + 
     $ TAy*(RBx*WSqr*(-3D0*RBy*WSqr + TBy*(15D0 + U**2D0*WSqr)) + 
     $ TBx*(RBy*WSqr*(15D0 + U**2D0*WSqr) - 5D0*TBy*(21D0 +
     $ 2D0*U**2D0*WSqr))))) + 
     $ Sinhuw*(WSqr*(WSqr*(3D0 + 3D0*RAy*TBy + (-(RAy*RBy) + U**2D0 + 
     $ RAy*TBy*U**2D0)*WSqr+RAx*(RBx*WSqr*(-1D0-RAy*TBy+RAy*RBy*WSqr) +
     $ TBx*(3D0+3D0*RAy*TBy + (-(RAy*RBy)+
     $ U**2D0+RAy*TBy*U**2D0)*WSqr))) + 
     $ TAy*(RBy*WSqr*(3D0 + 3D0*RAx*TBx + (-(RAx*RBx) + U**2D0 + 
     $ RAx*TBx*U**2D0)*WSqr)+TBy*(-15D0-6D0*U**2D0*WSqr+
     $ RAx*(RBx*WSqr*(3D0 + 
     $ U**2D0*WSqr) - 3D0*TBx*(5D0 + 2D0*U**2D0*WSqr))))) + 
     $ TAx*(TAy*(RBx*WSqr*(RBy*WSqr*(3D0 + U**2D0*WSqr) - 
     $ 3D0*TBy*(5D0 + 2D0*U**2D0*WSqr)) + 
     $ TBx*(-3D0*RBy*WSqr*(5D0 + 2D0*U**2D0*WSqr) + 
     $ TBy*(105D0 + U**2D0*WSqr*(45D0 + U**2D0*WSqr)))) +
     $ WSqr*(RBx*WSqr*(3D0 + 
     $ 3D0*RAy*TBy + (-(RAy*RBy) + U**2D0 + RAy*TBy*U**2D0)*WSqr) + 
     $ TBx*(-15D0 - 6D0*U**2D0*WSqr + RAy*(RBy*WSqr*(3D0 +
     $ U**2D0*WSqr) - 
     $ 3D0*TBy*(5D0 + 2D0*U**2D0*WSqr)))))))/(U*WSqr**4.5D0)     
        Else
        !(*D1D0vsD1D0-2D0 [110D0,101D0]*)
          Do is=1,3
            If((KA(is).eq.1).and.(KA(is).eq.KB(is))) iSame=is
          Enddo
          Call SwitchK(1,iSame,KA,KB)
          Call SwitchRT(1,iSame,RA,TA)
          Call SwitchRT(1,iSame,RB,TB)
          Call SortAngD1(iTgtD1a,KA,RA,TA)
          Call SortAngD1(iTgtD1b,KB,RB,TB)
          Call GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
          Drv=(Coshuw*U*DSqrt(WSqr)*(WSqr*(RAy*WSqr*(-3D0*(TBz + 
     $ RAx*TBx*TBz) + (RBz + RAx*RBz*TBx + RAx*RBx*TBz)*WSqr) + 
     $ TAy*(RBz*WSqr*(-3D0 - 3D0*RAx*TBx + RAx*RBx*WSqr) + TBz*(15D0 + 
     $ 15D0*RAx*TBx + (-3D0*RAx*RBx + U**2D0 +
     $ RAx*TBx*U**2D0)*WSqr))) + 
     $ TAx*(RAy*WSqr*(RBx*WSqr*(-3D0*TBz + RBz*WSqr) +
     $ TBx*(-3D0*RBz*WSqr + 
     $ TBz*(15D0 + U**2D0*WSqr))) + TAy*(RBx*WSqr*(-3D0*RBz*WSqr +
     $ TBz*(15D0 + 
     $ U**2D0*WSqr)) + TBx*(RBz*WSqr*(15D0 + U**2D0*WSqr) -
     $ 5D0*TBz*(21D0 + 
     $ 2D0*U**2D0*WSqr)))))+
     $ Sinhuw*(TAx*(RAy*WSqr*(RBx*WSqr*(-(RBz*WSqr) + 
     $ TBz*(3D0 + U**2D0*WSqr))+TBx*(RBz*WSqr*(3D0 +
     $ U**2D0*WSqr) - 3D0*TBz*(5D0 + 
     $ 2D0*U**2D0*WSqr))) + TAy*(RBx*WSqr*(RBz*WSqr*(3D0 +
     $ U**2D0*WSqr) - 
     $ 3D0*TBz*(5D0 + 2D0*U**2D0*WSqr)) +
     $ TBx*(-3D0*RBz*WSqr*(5D0 + 2D0*U**2D0*WSqr) + 
     $ TBz*(105D0 + U**2D0*WSqr*(45D0 + U**2D0*WSqr))))) +
     $ WSqr*(RAy*WSqr*
     $ (RBz*WSqr*(-1D0 - RAx*TBx + RAx*RBx*WSqr) +
     $ TBz*(3D0 + 3D0*RAx*TBx + 
     $ (-(RAx*RBx) + U**2D0 + RAx*TBx*U**2D0)*WSqr)) +
     $ TAy*(RBz*WSqr*(3D0 + 
     $ 3D0*RAx*TBx + (-(RAx*RBx) + U**2D0 + RAx*TBx*U**2D0)*WSqr) + 
     $ TBz*(-15D0 -
     $ 6D0*U**2D0*WSqr + RAx*(RBx*WSqr*(3D0 + U**2D0*WSqr) - 
     $ 3D0*TBx*(5D0 + 2D0*U**2D0*WSqr)))))))/(U*WSqr**4.5D0)     
        Endif
      Else If(NAngA.eq.2.and.MxA.eq.1.and.NAngB.eq.2.and.MxB.eq.2) then 
        If(KA(1)+KB(1).eq.3.or.KA(2)+KB(2).eq.3
     $ .or.KA(3)+KB(3).eq.3) then
        !(*D1D0vsD2D0-1D0 [110D0,200D0]*)
          Do is=1,3
            If((KA(is).eq.1).and.(KB(is).eq.2)) iSame=is
          Enddo
          Call SwitchK(1,iSame,KA,KB)
          Call SwitchRT(1,iSame,RA,TA)
          Call SwitchRT(1,iSame,RB,TB)
          Call SortAngD1(iTgtD1a,KA,RA,TA)
          Call SortAngD2(iTgtD2a,KB,RB,TB)
          Call GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
          Drv=(Coshuw*U*DSqrt(WSqr)*(TAx*(RAy*WSqr*(-6D0*RBx*TBx*WSqr + 
     $ WSqr*(-3D0 + RBx**2D0*WSqr) + TBx**2D0*(15D0 + U**2D0*WSqr)) + 
     $ TAy*(2D0*RBx*TBx*WSqr*(15D0+U**2D0*WSqr)+WSqr*(15D0 -
     $ 3D0*RBx**2D0*WSqr + 
     $ U**2D0*WSqr) -
     $ 5D0*TBx**2D0*(21D0 + 2D0*U**2D0*WSqr))) + WSqr*(RAy*WSqr*
     $ (-3D0*TBx*(2D0 + RAx*TBx) + 
     $ (RAx + 2D0*RBx + 2D0*RAx*RBx*TBx)*WSqr) + 
     $ TAy*(RAx*TBx**2D0*(15D0 +
     $ U**2D0*WSqr) + 2D0*TBx*(15D0 - 3D0*RAx*RBx*WSqr + 
     $ U**2D0*WSqr) + WSqr*(-6D0*RBx + RAx*(-3D0 + RBx**2D0*WSqr)))))+ 
     $ Sinhuw*(WSqr*(RAy*WSqr*(RAx*TBx**2D0*(3D0 + U**2D0*WSqr) + 
     $ 2D0*TBx*(3D0 - RAx*RBx*WSqr + U**2D0*WSqr) + WSqr*(-2D0*RBx +
     $ RAx*(-1D0 + 
     $ RBx**2D0*WSqr))) +
     $ TAy*(-3D0*RAx*TBx**2D0*(5D0 + 2D0*U**2D0*WSqr) + 
     $ WSqr*(2D0*RBx*(3D0 +
     $ U**2D0*WSqr)+RAx*(3D0- RBx**2D0*WSqr + U**2D0*WSqr)) + 
     $ 2D0*TBx*(RAx*RBx*WSqr*(3D0 +
     $ U**2D0*WSqr) - 3D0*(5D0 + 2D0*U**2D0*WSqr)))) + 
     $ TAx*(RAy*WSqr*(2D0*RBx*TBx*WSqr*(3D0 +
     $ U**2D0*WSqr) + WSqr*(3D0 - 
     $ RBx**2D0*WSqr + U**2D0*WSqr)-3D0*TBx**2D0*(5D0 +
     $ 2D0*U**2D0*WSqr)) + 
     $ TAy*(-6D0*RBx*TBx*WSqr*(5D0 +
     $ 2D0*U**2D0*WSqr) + TBx**2D0*(105D0 + 
     $ U**2D0*WSqr*(45D0 +
     $ U**2D0*WSqr))+WSqr*(RBx**2D0*WSqr*(3D0+U**2D0*WSqr)- 
     $ 3D0*(5D0 + 2D0*U**2D0*WSqr))))))/(U*WSqr**4.5D0)     
        Else
        !(*D1D0vsD2D0-2D0 [101D0,020D0]*)
          Call SortAngD1(iTgtD1b,KA,RA,TA)
          Call SortAngD2(iTgtD2b,KB,RB,TB)
          Call GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
          Drv=(Coshuw*U*DSqrt(WSqr)*(RAx*WSqr*(RAz*WSqr*
     $ (-3D0*TBy**2D0 + 
     $ WSqr + 2D0*RBy*TBy*WSqr) + TAz*(-6D0*RBy*TBy*WSqr + WSqr*(-3D0 +
     $ RBy**2D0*WSqr) + TBy**2D0*(15D0 + U**2D0*WSqr))) + 
     $ TAx*(RAz*WSqr*(-6D0*RBy*TBy*WSqr + WSqr*(-3D0 + RBy**2D0*WSqr) + 
     $ TBy**2D0*(15D0 + U**2D0*WSqr)) + TAz*(2D0*RBy*TBy*WSqr*(15D0 + 
     $ U**2D0*WSqr) + WSqr*(15D0 - 3D0*RBy**2D0*WSqr + U**2D0*WSqr) - 
     $ 5D0*TBy**2D0*(21D0 + 2D0*U**2D0*WSqr)))) + 
     $ Sinhuw*(RAx*WSqr*(RAz*WSqr*(-2D0*RBy*TBy*WSqr + WSqr*(-1D0 + 
     $ RBy**2D0*WSqr) + TBy**2D0*(3D0 + U**2D0*WSqr)) + 
     $ TAz*(2D0*RBy*TBy*WSqr*(3D0 + U**2D0*WSqr) + WSqr*(3D0 -
     $ RBy**2D0*WSqr + 
     $ U**2D0*WSqr) - 3D0*TBy**2D0*(5D0 + 2D0*U**2D0*WSqr))) + 
     $ TAx*(RAz*WSqr*(2D0*RBy*TBy*WSqr*(3D0 +
     $ U**2D0*WSqr) + WSqr*(3D0 - 
     $ RBy**2D0*WSqr + U**2D0*WSqr) -
     $ 3D0*TBy**2D0*(5D0 + 2D0*U**2D0*WSqr)) + 
     $ TAz*(-6D0*RBy*TBy*WSqr*(5D0 +
     $ 2D0*U**2D0*WSqr) + TBy**2D0*(105D0 + 
     $ U**2D0*WSqr*(45D0 + U**2D0*WSqr)) + WSqr*(RBy**2D0*WSqr*(3D0 + 
     $ U**2D0*WSqr) -
     $ 3D0*(5D0 + 2D0*U**2D0*WSqr))))))/(U*WSqr**4.5D0)     
        Endif
      Else If(NAngA.eq.2.and.MxA.eq.2.and.NAngB.eq.2.and.MxB.eq.2) then 
        If(KA(1).eq.KB(1).and.KA(2).eq.KB(2).and.KA(3).eq.KB(3)) then
        !(*D2D0vsD2D0-1D0 [200D0,200D0]*)
          Call SortAngD2(iTgtD2a,KA,RA,TA)
          Call SortAngD2(iTgtD2a,KB,RB,TB)
          Call GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
          Drv=(Coshuw*U*DSqrt(WSqr)*(TAx**2D0*(2D0*RBx*TBx*WSqr*(15D0 + 
     $ U**2D0*WSqr) + WSqr*(15D0 - 3D0*RBx**2D0*WSqr + U**2D0*WSqr) - 
     $ 5D0*TBx**2D0*(21D0 + 2D0*U**2D0*WSqr)) + 
     $ WSqr*(WSqr*(-9D0 + (RAx**2D0 + 4D0*RAx*RBx + RBx**2D0)*WSqr) + 
     $ TBx**2D0*(15D0 - 3D0*RAx**2D0*WSqr + U**2D0*WSqr) + 
     $ 2D0*TBx*WSqr*(-3D0*RBx + RAx*(-6D0 + RAx*RBx*WSqr))) + 
     $ 2D0*TAx*WSqr*(RAx*TBx**2D0*(15D0 + U**2D0*WSqr)+2D0*TBx*(15D0 - 
     $ 3D0*RAx*RBx*WSqr + U**2D0*WSqr) + WSqr*(-6D0*RBx + RAx*(-3D0 + 
     $ RBx**2D0*WSqr)))) + Sinhuw*(WSqr*(WSqr*(9D0 - (RAx**2D0 + 
     $ 4D0*RAx*RBx + RBx**2D0 - 3D0*U**2D0)*WSqr +
     $ RAx**2D0*RBx**2D0*WSqr**2D0) + 
     $ 2D0*TBx*WSqr*(-(RAx**2D0*RBx*WSqr) +
     $ 2D0*RAx*(3D0 + U**2D0*WSqr) + 
     $ RBx*(3D0 + U**2D0*WSqr)) + TBx**2D0*(RAx**2D0*WSqr*(3D0 +
     $ U**2D0*WSqr) - 
     $ 3D0*(5D0 + 2D0*U**2D0*WSqr))) +
     $ 2D0*TAx*WSqr*(-3D0*RAx*TBx**2D0*(5D0 + 
     $ 2D0*U**2D0*WSqr)+WSqr*(2D0*RBx*(3D0+ U**2D0*WSqr) + RAx*(3D0 - 
     $ RBx**2D0*WSqr + U**2D0*WSqr)) + 
     $ 2D0*TBx*(RAx*RBx*WSqr*(3D0 + U**2D0*WSqr) -
     $ 3D0*(5D0 + 2D0*U**2D0*WSqr))) + 
     $ TAx**2D0*(-6D0*RBx*TBx*WSqr*(5D0 +
     $ 2D0*U**2D0*WSqr) + TBx**2D0*(105D0 + 
     $ U**2D0*WSqr*(45D0 + U**2D0*WSqr)) + WSqr*(RBx**2D0*WSqr*(3D0 + 
     $ U**2D0*WSqr) - 3D0*(5D0 + 2D0*U**2D0*WSqr)))))/(U*WSqr**4.5D0)
        Else
        !(*D2D0vsD2D0-2D0 [200D0,020D0]*)
          Call SortAngD2(iTgtD2a,KA,RA,TA)
          Call SortAngD2(iTgtD2b,KB,RB,TB)
          Call GetArgSph(EA,EB,RA,RB,TA,TB,U,TAx,TAy,TAz,
     $ TBx,TBy,TBz,RAx,RAy,RAz,RBx,RBy,RBz,WSqr,W,Sinhuw,Coshuw)
          Drv=(Coshuw*U*DSqrt(WSqr)*(2D0*RAx*TAx*WSqr*
     $ (-6D0*RBy*TBy*WSqr + 
     $ WSqr*(-3D0 + RBy**2D0*WSqr) + TBy**2D0*(15D0 + U**2D0*WSqr)) + 
     $ WSqr*(2D0*RBy*TBy*WSqr*(-3D0 +
     $ RAx**2D0*WSqr) + WSqr*(-3D0 + (RAx**2D0 + 
     $ RBy**2D0)*WSqr) +
     $ TBy**2D0*(15D0 - 3D0*RAx**2D0*WSqr + U**2D0*WSqr)) + 
     $ TAx**2D0*(2D0*RBy*TBy*WSqr*(15D0 + U**2D0*WSqr) + WSqr*(15D0 - 
     $ 3D0*RBy**2D0*WSqr + U**2D0*WSqr) -
     $ 5D0*TBy**2D0*(21D0 + 2D0*U**2D0*WSqr))) + 
     $ Sinhuw*(2D0*RAx*TAx*WSqr*(2D0*RBy*TBy*WSqr*(3D0 + U**2D0*WSqr)+ 
     $ WSqr*(3D0 - RBy**2D0*WSqr + U**2D0*WSqr)-3D0*TBy**2D0*(5D0+
     $ 2D0*U**2D0*WSqr)) + 
     $ TAx**2D0*(-6D0*RBy*TBy*WSqr*(5D0 +
     $ 2D0*U**2D0*WSqr) + TBy**2D0*(105D0 + 
     $ U**2D0*WSqr*(45D0 +
     $ U**2D0*WSqr))+WSqr*(RBy**2D0*WSqr*(3D0 + U**2D0*WSqr) - 
     $ 3D0*(5D0 + 2D0*U**2D0*WSqr))) +
     $ WSqr*(2D0*RBy*TBy*WSqr*(3D0 - RAx**2D0*WSqr +
     $ U**2D0*WSqr) + TBy**2D0*(RAx**2D0*WSqr*(3D0 + U**2D0*WSqr) - 
     $ 3D0*(5D0 + 2D0*U**2D0*WSqr)) + WSqr*(3D0 +
     $ WSqr*(-RBy**2D0 + U**2D0 + 
     $ RAx**2D0*(-1D0 + RBy**2D0*WSqr))))))/(U*WSqr**4.5D0)
        Endif
      Else 
        Write(*,*)"NAngA,NAngB:",NAngA,NAngB
        STOP 'IMPOSSIBLE'
      Endif
      Return
      END

*Deck azWghtUni
      Subroutine azWghtUni(SmallU,NU,Ub,Ux,UWei)
      Implicit Real*8(A-H,O-Z)
Cf2py intent(in) smallU,NU,UB
Cf2py intent(out) UWei 
Cf2py intent(out) Ux 
C     Trapezoid integration, grid points are uniform
      Dimension Ux(NU),UWei(NU)
      DH=(Ub-SmallU)/(NU-1)
      Do i=1,NU
        Ux(i)=SmallU+(i-1)*DH
        UWei(i)=DH
      Enddo
      UWei(1)=DH/2.D0
      UWei(NU)=DH/2.D0
      Return
      End
