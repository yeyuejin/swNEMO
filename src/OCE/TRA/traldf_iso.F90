# include "toggle.h90"

MODULE traldf_iso
   !!======================================================================
   !!                   ***  MODULE  traldf_iso  ***
   !! Ocean  tracers:  horizontal component of the lateral tracer mixing trend
   !!======================================================================
   !! History :  OPA  ! 1994-08  (G. Madec, M. Imbard)
   !!            8.0  ! 1997-05  (G. Madec)  split into traldf and trazdf
   !!            NEMO ! 2002-08  (G. Madec)  Free form, F90
   !!            1.0  ! 2005-11  (G. Madec)  merge traldf and trazdf :-)
   !!            3.3  ! 2010-09  (C. Ethe, G. Madec) Merge TRA-TRC
   !!            3.7  ! 2014-01  (G. Madec, S. Masson)  restructuration/simplification of aht/aeiv specification
   !!             -   ! 2014-02  (F. Lemarie, G. Madec)  triad operator (Griffies) + Method of Stabilizing Correction
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   tra_ldf_iso   : update the tracer trend with the horizontal component of a iso-neutral laplacian operator
   !!                   and with the vertical part of the isopycnal or geopotential s-coord. operator 
   !!----------------------------------------------------------------------
   USE oce            ! ocean dynamics and active tracers
   USE dom_oce        ! ocean space and time domain
   USE trc_oce        ! share passive tracers/Ocean variables
   USE zdf_oce        ! ocean vertical physics
   USE ldftra         ! lateral diffusion: tracer eddy coefficients
   USE ldfslp         ! iso-neutral slopes
   USE diaptr         ! poleward transport diagnostics
   USE diaar5         ! AR5 diagnostics
   !
   USE in_out_manager ! I/O manager
   USE iom            ! I/O library
   USE phycst         ! physical constants
   USE lbclnk         ! ocean lateral boundary conditions (or mpp link)
   use mpi

   IMPLICIT NONE
   PRIVATE

   PUBLIC   tra_ldf_iso   ! routine called by step.F90

   LOGICAL  ::   l_ptr   ! flag to compute poleward transport
   LOGICAL  ::   l_hst   ! flag to compute heat transport

   !! * Substitutions
#  include "vectopt_loop_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 4.0 , NEMO Consortium (2018)
   !! $Id: traldf_iso.F90 11993 2019-11-28 10:20:53Z cetlod $
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

  SUBROUTINE tra_ldf_iso( kt, kit000, cdtype, pahu, pahv, pgu , pgv ,   &
      &                                                   pgui, pgvi,   &
      &                                       ptb , ptbb, pta , kjpt, kpass )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE tra_ldf_iso  ***
      !!
      !! ** Purpose :   Compute the before horizontal tracer (t & s) diffusive 
      !!      trend for a laplacian tensor (ezxcept the dz[ dz[.] ] term) and 
      !!      add it to the general trend of tracer equation.
      !!
      !! ** Method  :   The horizontal component of the lateral diffusive trends 
      !!      is provided by a 2nd order operator rotated along neural or geopo-
      !!      tential surfaces to which an eddy induced advection can be added
      !!      It is computed using before fields (forward in time) and isopyc-
      !!      nal or geopotential slopes computed in routine ldfslp.
      !!
      !!      1st part :  masked horizontal derivative of T  ( di[ t ] )
      !!      ========    with partial cell update if ln_zps=T
      !!                  with top     cell update if ln_isfcav
      !!
      !!      2nd part :  horizontal fluxes of the lateral mixing operator
      !!      ========    
      !!         zftu =  pahu e2u*e3u/e1u di[ tb ]
      !!               - pahu e2u*uslp    dk[ mi(mk(tb)) ]
      !!         zftv =  pahv e1v*e3v/e2v dj[ tb ]
      !!               - pahv e2u*vslp    dk[ mj(mk(tb)) ]
      !!      take the horizontal divergence of the fluxes:
      !!         difft = 1/(e1e2t*e3t) {  di-1[ zftu ] +  dj-1[ zftv ]  }
      !!      Add this trend to the general trend (ta,sa):
      !!         ta = ta + difft
      !!
      !!      3rd part: vertical trends of the lateral mixing operator
      !!      ========  (excluding the vertical flux proportional to dk[t] )
      !!      vertical fluxes associated with the rotated lateral mixing:
      !!         zftw = - {  mi(mk(pahu)) * e2t*wslpi di[ mi(mk(tb)) ]
      !!                   + mj(mk(pahv)) * e1t*wslpj dj[ mj(mk(tb)) ]  }
      !!      take the horizontal divergence of the fluxes:
      !!         difft = 1/(e1e2t*e3t) dk[ zftw ]
      !!      Add this trend to the general trend (ta,sa):
      !!         pta = pta + difft
      !!
      !! ** Action :   Update pta arrays with the before rotated diffusion
      !!----------------------------------------------------------------------
      INTEGER                              , INTENT(in   ) ::   kt         ! ocean time-step index
      INTEGER                              , INTENT(in   ) ::   kit000     ! first time step index
      CHARACTER(len=3)                     , INTENT(in   ) ::   cdtype     ! =TRA or TRC (tracer indicator)
      INTEGER                              , INTENT(in   ) ::   kjpt       ! number of tracers
      INTEGER                              , INTENT(in   ) ::   kpass      ! =1/2 first or second passage
      REAL(wp), DIMENSION(jpi,jpj,jpk)     , INTENT(in   ) ::   pahu, pahv ! eddy diffusivity at u- and v-points  [m2/s]
      REAL(wp), DIMENSION(jpi,jpj    ,kjpt), INTENT(in   ) ::   pgu, pgv   ! tracer gradient at pstep levels
      REAL(wp), DIMENSION(jpi,jpj,    kjpt), INTENT(in   ) ::   pgui, pgvi ! tracer gradient at top   levels
      REAL(wp), DIMENSION(jpi,jpj,jpk,kjpt), INTENT(in   ) ::   ptb        ! tracer (kpass=1) or laplacian of tracer (kpass=2)
      REAL(wp), DIMENSION(jpi,jpj,jpk,kjpt), INTENT(in   ) ::   ptbb       ! tracer (only used in kpass=2)
      REAL(wp), DIMENSION(jpi,jpj,jpk,kjpt), INTENT(inout) ::   pta        ! tracer trend
      !
      INTEGER  ::  ji, jj, jk, jn   ! dummy loop indices
      INTEGER  ::  ikt
      INTEGER  ::  ierr             ! local integer
      REAL(wp) ::  zmsku, zahu_w, zabe1, zcof1, zcoef3   ! local scalars
      REAL(wp) ::  zmskv, zahv_w, zabe2, zcof2, zcoef4   !   -      -
      REAL(wp) ::  zcoef0, ze3w_2, zsign, z2dt, z1_2dt   !   -      -
      REAL(wp), DIMENSION(jpi,jpj)     ::   zdkt, zdk1t, z2d
      REAL(wp), DIMENSION(jpi,jpj,jpk) ::   zdit, zdjt, zftu, zftv, ztfw, zzz
# if defined iso_time
      real,dimension(4)::elapsed
      real :: st,ed
      integer :: print_flag
      common /flag/ print_flag
# endif
# if defined slave
      integer, external :: slave_iso1, slave_iso2, slave_iso3, slave_iso4
      type :: var3
         integer :: jpi, jpj, jpk, fs_two, fs_jpim1, jpjm1, jpkm1, jn, narea, kt
         real(wp) :: zsign
         integer*8 loc_wmask, loc_umask, loc_vmask, loc_pahu, loc_pahv, loc_wslpi, loc_wslpj, loc_ah_wslp2, loc_zdit, loc_zdjt, loc_ptb, loc_e2_e1u, loc_e3u_n, loc_e1_e2v, loc_e3v_n, loc_e2u, loc_uslp, loc_e1v, loc_vslp, loc_pta, loc_r1_e1e2t, loc_e3t_n, loc_e2t, loc_e1t, loc_e1e2t, loc_e3w_n, loc_akz, loc_ztfw, loc_zzz

      end type var3
      type(var3) :: v3
      v3%jpi = jpi
      v3%jpj = jpj
      v3%jpk = jpk
      v3%fs_two = fs_2
      v3%fs_jpim1 = fs_jpim1
      v3%jpjm1 = jpjm1
      v3%jpkm1 = jpkm1
      v3%narea = narea
      v3%kt = kt
      v3%loc_wmask = loc(wmask)
      v3%loc_umask = loc(umask)
      v3%loc_vmask = loc(vmask)
      v3%loc_pahu = loc(pahu)
      v3%loc_pahv = loc(pahv)
      v3%loc_wslpi = loc(wslpi)
      v3%loc_wslpj = loc(wslpj)
      v3%loc_ah_wslp2 = loc(ah_wslp2)
      v3%loc_zdit = loc(zdit)
      v3%loc_zdjt = loc(zdjt)
      v3%loc_ptb = loc(ptb)
      v3%loc_e2_e1u = loc(e2_e1u)
      v3%loc_e3u_n = loc(e3u_n)
      v3%loc_e1_e2v = loc(e1_e2v)
      v3%loc_e3v_n = loc(e3v_n)
      v3%loc_e2u = loc(e2u)
      v3%loc_uslp = loc(uslp)
      v3%loc_e1v = loc(e1v)
      v3%loc_vslp = loc(vslp)
      v3%loc_pta = loc(pta)
      v3%loc_r1_e1e2t = loc(r1_e1e2t)
      v3%loc_e3t_n = loc(e3t_n)
      v3%loc_e2t = loc(e2t)
      v3%loc_e1t = loc(e1t)
      v3%loc_e1e2t = loc(e1e2t)
      v3%loc_e3w_n = loc(e3w_n)
      v3%loc_akz = loc(akz)
      v3%loc_ztfw = loc(ztfw)
      v3%loc_zzz = loc(zzz)
# endif

      !!----------------------------------------------------------------------
      !
      IF( kpass == 1 .AND. kt == kit000 )  THEN
         IF(lwp) WRITE(numout,*)
         IF(lwp) WRITE(numout,*) 'tra_ldf_iso : rotated laplacian diffusion operator on ', cdtype
         IF(lwp) WRITE(numout,*) '~~~~~~~~~~~'
         !
         akz     (:,:,:) = 0._wp      
         ah_wslp2(:,:,:) = 0._wp
      ENDIF
      !   
      l_hst = .FALSE.
      l_ptr = .FALSE.
      
      IF( cdtype == 'TRA' .AND. ln_diaptr )                                                 l_ptr = .TRUE. 
      IF( cdtype == 'TRA' .AND. ( iom_use("uadv_heattr") .OR. iom_use("vadv_heattr") .OR. &
         &                        iom_use("uadv_salttr") .OR. iom_use("vadv_salttr")  ) )   l_hst = .TRUE.
      !
      !                                            ! set time step size (Euler/Leapfrog)
      IF( neuler == 0 .AND. kt == nit000 ) THEN   ;   z2dt =     rdt      ! at nit000   (Euler)
      ELSE                                        ;   z2dt = 2.* rdt      !             (Leapfrog)
      ENDIF
      z1_2dt = 1._wp / z2dt
      !
      IF( kpass == 1 ) THEN   ;   zsign =  1._wp      ! bilaplacian operator require a minus sign (eddy diffusivity >0)
      ELSE                    ;   zsign = -1._wp
      ENDIF
# if defined slave
      v3%zsign = zsign
# endif
      !!----------------------------------------------------------------------
      !!   0 - calculate  ah_wslp2 and akz
      !!----------------------------------------------------------------------
      !
# if defined iso_time
      if(lwp) st = MPI_WTime()
# endif
# if defined slave
      call athread_spawn(slave_iso1, v3)
      call athread_join()
# else
      DO jk = 2, jpkm1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1   ! vector opt.
               !
               zmsku = wmask(ji,jj,jk) / MAX(   umask(ji  ,jj,jk-1) + umask(ji-1,jj,jk)          &
                  &                           + umask(ji-1,jj,jk-1) + umask(ji  ,jj,jk) , 1._wp  )
               zmskv = wmask(ji,jj,jk) / MAX(   vmask(ji,jj  ,jk-1) + vmask(ji,jj-1,jk)          &
                  &                           + vmask(ji,jj-1,jk-1) + vmask(ji,jj  ,jk) , 1._wp  )
                  !
               zahu_w = (   pahu(ji  ,jj,jk-1) + pahu(ji-1,jj,jk)    &
                  &       + pahu(ji-1,jj,jk-1) + pahu(ji  ,jj,jk)  ) * zmsku
               zahv_w = (   pahv(ji,jj  ,jk-1) + pahv(ji,jj-1,jk)    &
                  &       + pahv(ji,jj-1,jk-1) + pahv(ji,jj  ,jk)  ) * zmskv
                  !
               ah_wslp2(ji,jj,jk) = zahu_w * wslpi(ji,jj,jk) * wslpi(ji,jj,jk)   &
                  &               + zahv_w * wslpj(ji,jj,jk) * wslpj(ji,jj,jk)
            END DO
         END DO
      END DO
# endif
# if defined iso_time
      if(lwp) then
          ed=MPI_WTIME()
          elapsed(1)=ed-st
      endif
# endif
      akz(fs_2:fs_jpim1,2:jpjm1,2:jpkm1) = ah_wslp2(fs_2:fs_jpim1,2:jpjm1,2:jpkm1)     

      DO jn = 1, kjpt                                            ! tracer loop
         !zdit (1,:,:) = 0._wp     ;     zdit (jpi,:,:) = 0._wp
         !zdjt (1,:,:) = 0._wp     ;     zdjt (jpi,:,:) = 0._wp
# if defined iso_time
         if(lwp) st = MPI_WTime()
# endif
# if defined slave
         v3%jn=jn
# endif
# if defined slave
         call athread_spawn(slave_iso2, v3)
         call athread_join()
# else
         DO jk = 1, jpkm1
            DO jj = 1, jpjm1
               DO ji = 1, fs_jpim1   ! vector opt.
                  zdit(ji,jj,jk) = ( ptb(ji+1,jj  ,jk,jn) - ptb(ji,jj,jk,jn) ) * umask(ji,jj,jk)
                  zdjt(ji,jj,jk) = ( ptb(ji  ,jj+1,jk,jn) - ptb(ji,jj,jk,jn) ) * vmask(ji,jj,jk)
               END DO
            END DO
         END DO
         DO jk = 1, jpkm1                                 ! Horizontal slab
            zdk1t(:,:) = ( ptb(:,:,jk,jn) - ptb(:,:,jk+1,jn) ) * wmask(:,:,jk+1)     ! level jk+1
            IF( jk == 1 ) THEN   ;   zdkt(:,:) = zdk1t(:,:)                          ! surface: zdkt(jk=1)=zdkt(jk=2)
            ELSE                 ;   zdkt(:,:) = ( ptb(:,:,jk-1,jn) - ptb(:,:,jk,jn) ) * wmask(:,:,jk)
            ENDIF
            DO jj = 1 , jpjm1            !==  Horizontal fluxes
               DO ji = 1, fs_jpim1   ! vector opt.
                  zabe1 = pahu(ji,jj,jk) * e2_e1u(ji,jj) * e3u_n(ji,jj,jk)
                  zabe2 = pahv(ji,jj,jk) * e1_e2v(ji,jj) * e3v_n(ji,jj,jk)
                  !
                  zmsku = 1. / MAX(  wmask(ji+1,jj,jk  ) + wmask(ji,jj,jk+1)   &
                     &             + wmask(ji+1,jj,jk+1) + wmask(ji,jj,jk  ), 1. )
                  !
                  zmskv = 1. / MAX(  wmask(ji,jj+1,jk  ) + wmask(ji,jj,jk+1)   &
                     &             + wmask(ji,jj+1,jk+1) + wmask(ji,jj,jk  ), 1. )
                  !
                  zcof1 = - pahu(ji,jj,jk) * e2u(ji,jj) * uslp(ji,jj,jk) * zmsku
                  zcof2 = - pahv(ji,jj,jk) * e1v(ji,jj) * vslp(ji,jj,jk) * zmskv
                  !
                  zftu(ji,jj,jk ) = (  zabe1 * zdit(ji,jj,jk)   &
                     &               + zcof1 * (  zdkt (ji+1,jj) + zdk1t(ji,jj)      &
                     &                          + zdk1t(ji+1,jj) + zdkt (ji,jj)  )  ) * umask(ji,jj,jk)
                  zftv(ji,jj,jk) = (  zabe2 * zdjt(ji,jj,jk)   &
                     &               + zcof2 * (  zdkt (ji,jj+1) + zdk1t(ji,jj)      &
                     &                          + zdk1t(ji,jj+1) + zdkt (ji,jj)  )  ) * vmask(ji,jj,jk)                  
               END DO
            END DO
            DO jj = 2 , jpjm1          !== horizontal divergence and add to pta
               DO ji = fs_2, fs_jpim1   ! vector opt.
                  pta(ji,jj,jk,jn) = pta(ji,jj,jk,jn) + zsign * (  zftu(ji,jj,jk) - zftu(ji-1,jj,jk)      &
                     &                                           + zftv(ji,jj,jk) - zftv(ji,jj-1,jk)  )   &
                     &                                        * r1_e1e2t(ji,jj) / e3t_n(ji,jj,jk)
               END DO
            END DO
         END DO                                        !   End of slab  
# endif
# if defined iso_time
         if(lwp) then
             ed=MPI_WTIME()
             elapsed(2)=ed-st
         endif
# endif
         ztfw(fs_2:1,:,:) = 0._wp     ;     ztfw(jpi:fs_jpim1,:,:) = 0._wp   ! avoid to potentially manipulate NaN values
         ztfw(:,:, 1 ) = 0._wp      ;      ztfw(:,:,jpk) = 0._wp
# if defined iso_time
         if(lwp) st = MPI_WTime()
# endif
# if defined slave
         call athread_spawn(slave_iso3, v3)
         call athread_join()
# else
         DO jk = 2, jpkm1           ! interior (2=<jk=<jpk-1)
            DO jj = 2, jpjm1
               DO ji = fs_2, fs_jpim1   ! vector opt.
                  !
                  zmsku = wmask(ji,jj,jk) / MAX(   umask(ji  ,jj,jk-1) + umask(ji-1,jj,jk)          &
                     &                           + umask(ji-1,jj,jk-1) + umask(ji  ,jj,jk) , 1._wp  )
                  zmskv = wmask(ji,jj,jk) / MAX(   vmask(ji,jj  ,jk-1) + vmask(ji,jj-1,jk)          &
                     &                           + vmask(ji,jj-1,jk-1) + vmask(ji,jj  ,jk) , 1._wp  )
                     !
                  zahu_w = (   pahu(ji  ,jj,jk-1) + pahu(ji-1,jj,jk)    &
                     &       + pahu(ji-1,jj,jk-1) + pahu(ji  ,jj,jk)  ) * zmsku
                  zahv_w = (   pahv(ji,jj  ,jk-1) + pahv(ji,jj-1,jk)    &
                     &       + pahv(ji,jj-1,jk-1) + pahv(ji,jj  ,jk)  ) * zmskv
                     !
                  zcoef3 = - zahu_w * e2t(ji,jj) * zmsku * wslpi (ji,jj,jk)   !wslpi & j are already w-masked
                  zcoef4 = - zahv_w * e1t(ji,jj) * zmskv * wslpj (ji,jj,jk)
                  !
                  ztfw(ji,jj,jk) = zcoef3 * (   zdit(ji  ,jj  ,jk-1) + zdit(ji-1,jj  ,jk)      &
                     &                        + zdit(ji-1,jj  ,jk-1) + zdit(ji  ,jj  ,jk)  )   &
                     &           + zcoef4 * (   zdjt(ji  ,jj  ,jk-1) + zdjt(ji  ,jj-1,jk)      &
                     &                        + zdjt(ji  ,jj-1,jk-1) + zdjt(ji  ,jj  ,jk)  )
                  ztfw(ji,jj,jk) = ztfw(ji,jj,jk) + e1e2t(ji,jj) / e3w_n(ji,jj,jk) * wmask(ji,jj,jk)   &
                     &                            * ( ah_wslp2(ji,jj,jk) - akz(ji,jj,jk) )             &
                     &                            * ( ptb(ji,jj,jk-1,jn) - ptb(ji,jj,jk,jn) )
               END DO
            END DO
         END DO
# endif
# if defined iso_time
         if(lwp) then
             ed=MPI_WTIME()
             elapsed(3)=ed-st
         endif
# endif
# if defined iso_time
         if(lwp) st = MPI_WTime()
# endif
# if defined slave
         call athread_spawn(slave_iso4, v3)
         call athread_join()
# else
         DO jk = 1, jpkm1                 !==  Divergence of vertical fluxes added to pta  ==!
            DO jj = 2, jpjm1
               DO ji = fs_2, fs_jpim1   ! vector opt.
                  pta(ji,jj,jk,jn) = pta(ji,jj,jk,jn) + zsign * (  ztfw (ji,jj,jk) - ztfw(ji,jj,jk+1)  )   &
                     &                                        * r1_e1e2t(ji,jj) / e3t_n(ji,jj,jk)
               END DO
            END DO
         END DO
# endif
# if defined iso_time
         if(lwp) then
             ed=MPI_WTIME()
             elapsed(4)=ed-st
         endif
# endif
# if defined iso_time
         if(lwp .and. print_flag.gt.0) then
             print_flag=print_flag-1
             print *,elapsed
         endif
# endif
      END DO                                                      ! end tracer loop
   END SUBROUTINE tra_ldf_iso

   !!==============================================================================
END MODULE traldf_iso
