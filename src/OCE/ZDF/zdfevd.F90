#include "toggle.h90"

MODULE zdfevd
   !!======================================================================
   !!                       ***  MODULE  zdfevd  ***
   !! Ocean physics: parameterization of convection through an enhancement
   !!                of vertical eddy mixing coefficient
   !!======================================================================
   !! History :  OPA  !  1997-06  (G. Madec, A. Lazar)  Original code
   !!   NEMO     1.0  !  2002-06  (G. Madec)  F90: Free form and module
   !!            3.2  !  2009-03  (M. Leclair, G. Madec, R. Benshila) test on both before & after
   !!            4.0  !  2017-04  (G. Madec)  evd applied on avm (at t-point) 
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   zdf_evd       : increase the momentum and tracer Kz at the location of
   !!                   statically unstable portion of the water column (ln_zdfevd=T)
   !!----------------------------------------------------------------------
   USE oce             ! ocean dynamics and tracers variables
   USE dom_oce         ! ocean space and time domain variables
   USE zdf_oce         ! ocean vertical physics variables
   USE trd_oce         ! trends: ocean variables
   USE trdtra          ! trends manager: tracers 
   !
   USE in_out_manager  ! I/O manager
   USE iom             ! for iom_put
   USE lbclnk          ! ocean lateral boundary conditions (or mpp link)
   USE timing          ! Timing
   use mpi

   IMPLICIT NONE
   PRIVATE

   PUBLIC   zdf_evd    ! called by step.F90

   !!----------------------------------------------------------------------
   !! NEMO/OCE 4.0 , NEMO Consortium (2018)
   !! $Id: zdfevd.F90 10068 2018-08-28 14:09:04Z nicolasmartin $
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE zdf_evd( kt, p_avm, p_avt )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE zdf_evd  ***
      !!                   
      !! ** Purpose :   Local increased the vertical eddy viscosity and diffu-
      !!      sivity coefficients when a static instability is encountered.
      !!
      !! ** Method  :   tracer (and momentum if nn_evdm=1) vertical mixing 
      !!              coefficients are set to rn_evd (namelist parameter) 
      !!              if the water column is statically unstable.
      !!                The test of static instability is performed using
      !!              Brunt-Vaisala frequency (rn2 < -1.e-12) of to successive
      !!              time-step (Leap-Frog environnement): before and
      !!              now time-step.
      !!
      !! ** Action  :   avt, avm   enhanced where static instability occurs
      !!----------------------------------------------------------------------
      INTEGER                    , INTENT(in   ) ::   kt             ! ocean time-step indexocean time step
      REAL(wp), DIMENSION(:,:,:) , INTENT(inout) ::   p_avm, p_avt   !  momentum and tracer Kz (w-points)
      !
      INTEGER ::   ji, jj, jk   ! dummy loop indices
      REAL(wp), DIMENSION(jpi,jpj,jpk) ::   zavt_evd, zavm_evd
      !!----------------------------------------------------------------------
      !
# if defined zdfevd_time
       external :: cyc_time
       integer*8 :: st, ed, elapsed 
       real(wp) :: qbw
       integer :: print_flag
       common /flag/ print_flag
# endif
# if defined slave
      integer, external :: slave_zdfevd
      type :: zdfevd_var
         integer   :: jpi, jpj, jpk, jpim1, jpjm1, jpkm1, narea
         real(wp)  :: rn_evd
         integer*8 :: loc_rn2, loc_rn2b, loc_wmask, loc_pavm, loc_pavt, loc_zavm, loc_zavt
      end type zdfevd_var
      type(zdfevd_var) :: var_z
      var_z%jpi = jpi
      var_z%jpj = jpj
      var_z%jpk = jpk
      var_z%jpim1 = jpim1
      var_z%jpjm1 = jpjm1
      var_z%jpkm1 = jpkm1
      var_z%narea = narea
      var_z%rn_evd = rn_evd
      var_z%loc_rn2 = loc(rn2)
      var_z%loc_rn2b = loc(rn2b)
      var_z%loc_wmask = loc(wmask)
      var_z%loc_pavm = loc(p_avm)
      var_z%loc_pavt = loc(p_avt)
      var_z%loc_zavm = loc(zavm_evd)
      var_z%loc_zavt = loc(zavt_evd)
# endif

      IF( kt == nit000 ) THEN
         IF(lwp) WRITE(numout,*)
         IF(lwp) WRITE(numout,*) 'zdf_evd : Enhanced Vertical Diffusion (evd)'
         IF(lwp) WRITE(numout,*) '~~~~~~~ '
         IF(lwp) WRITE(numout,*)
      ENDIF

# if defined zdfevd_time
         if(lwp) call cyc_time(st)
# endif
# if defined slave
         call athread_spawn(slave_zdfevd, var_z)
         zavm_evd(:,:,jpk) = 0._wp
         zavt_evd(:,:,jpk) = 0._wp
         call athread_join()
# else
      !
      zavt_evd(:,:,:) = p_avt(:,:,:)         ! set avt prior to evd application
      !
      SELECT CASE ( nn_evdm )
      !
      CASE ( 1 )           !==  enhance tracer & momentum Kz  ==!   (if rn2<-1.e-12)
         !
         zavm_evd(:,:,:) = p_avm(:,:,:)      ! set avm prior to evd application
         DO jk = 1, jpkm1 
            DO jj = 2, jpjm1
               DO ji = 2, jpim1
                  IF(  MIN( rn2(ji,jj,jk), rn2b(ji,jj,jk) ) <= -1.e-12 ) THEN
                     p_avt(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)
                     p_avm(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)
                  ENDIF
               END DO
            END DO
         END DO 
         zavm_evd(:,:,:) = p_avm(:,:,:) - zavm_evd(:,:,:)   ! change in avm due to evd
         CALL iom_put( "avm_evd", zavm_evd )                ! output this change
         !
      CASE DEFAULT         !==  enhance tracer Kz  ==!   (if rn2<-1.e-12) 
         !
         DO jk = 1, jpkm1
            DO jj = 2, jpjm1
               DO ji = 2, jpim1
                  IF(  MIN( rn2(ji,jj,jk), rn2b(ji,jj,jk) ) <= -1.e-12 )   &
                     p_avt(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)
               END DO
            END DO
         END DO
         !
      END SELECT 
      !
      zavt_evd(:,:,:) = p_avt(:,:,:) - zavt_evd(:,:,:)   ! change in avt due to evd
      CALL iom_put( "avt_evd", zavt_evd )              ! output this change
      !
# endif
# if defined zdfevd_time
         if(lwp) then
           call cyc_time(ed)
           elapsed = ed - st
           qbw = 8*2.25*(jpi*jpj*jpk*9)/((elapsed-60000)*45.0)
         end if
# endif
# if defined zdfevd_time
         if(lwp) then
           print *, 'zdfevd_time: ', elapsed, ' BW: ', qbw
         endif
# endif
# if defined slave
      CALL iom_put( "avm_evd", zavm_evd )                ! output this change
      CALL iom_put( "avt_evd", zavt_evd )              ! output this change
# endif

      ! 
      IF( l_trdtra ) CALL trd_tra( kt, 'TRA', jp_tem, jptra_evd, zavt_evd )
      !
   END SUBROUTINE zdf_evd

   !!======================================================================
END MODULE zdfevd
