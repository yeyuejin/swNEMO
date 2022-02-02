#include "toggle.h90"

MODULE dynzdf
   !!==============================================================================
   !!                 ***  MODULE  dynzdf  ***
   !! Ocean dynamics :  vertical component of the momentum mixing trend
   !!==============================================================================
   !! History :  1.0  !  2005-11  (G. Madec)  Original code
   !!            3.3  !  2010-10  (C. Ethe, G. Madec) reorganisation of initialisation phase
   !!            4.0  !  2017-06  (G. Madec) remove the explicit time-stepping option + avm at t-point
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   dyn_zdf       : compute the after velocity through implicit calculation of vertical mixing
   !!----------------------------------------------------------------------
   USE oce            ! ocean dynamics and tracers variables
   USE phycst         ! physical constants
   USE dom_oce        ! ocean space and time domain variables 
   USE sbc_oce        ! surface boundary condition: ocean
   USE zdf_oce        ! ocean vertical physics variables
   USE zdfdrg         ! vertical physics: top/bottom drag coef.
   USE dynadv    ,ONLY: ln_dynadv_vec    ! dynamics: advection form
   USE dynldf_iso,ONLY: akzu, akzv       ! dynamics: vertical component of rotated lateral mixing 
   USE ldfdyn         ! lateral diffusion: eddy viscosity coef. and type of operator
   USE trd_oce        ! trends: ocean variables
   USE trddyn         ! trend manager: dynamics
   !
   USE in_out_manager ! I/O manager
   USE lib_mpp        ! MPP library
   USE prtctl         ! Print control
   USE timing         ! Timing
   use mpi

   IMPLICIT NONE
   PRIVATE

   PUBLIC   dyn_zdf   !  routine called by step.F90

   REAL(wp) ::  r_vvl     ! non-linear free surface indicator: =0 if ln_linssh=T, =1 otherwise 

   !! * Substitutions
#  include "vectopt_loop_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 4.0 , NEMO Consortium (2018)
   !! $Id: dynzdf.F90 12292 2019-12-30 17:07:45Z mathiot $
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS
   
   SUBROUTINE dyn_zdf( kt )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE dyn_zdf  ***
      !!
      !! ** Purpose :   compute the trend due to the vert. momentum diffusion
      !!              together with the Leap-Frog time stepping using an 
      !!              implicit scheme.
      !!
      !! ** Method  :  - Leap-Frog time stepping on all trends but the vertical mixing
      !!         ua =         ub + 2*dt *       ua             vector form or linear free surf.
      !!         ua = ( e3u_b*ub + 2*dt * e3u_n*ua ) / e3u_a   otherwise
      !!               - update the after velocity with the implicit vertical mixing.
      !!      This requires to solver the following system: 
      !!         ua = ua + 1/e3u_a dk+1[ mi(avm) / e3uw_a dk[ua] ]
      !!      with the following surface/top/bottom boundary condition:
      !!      surface: wind stress input (averaged over kt-1/2 & kt+1/2)
      !!      top & bottom : top stress (iceshelf-ocean) & bottom stress (cf zdfdrg.F90)
      !!
      !! ** Action :   (ua,va)   after velocity 
      !!---------------------------------------------------------------------
      INTEGER, INTENT(in) ::   kt   ! ocean time-step index
      !
      INTEGER  ::   ji, jj, jk         ! dummy loop indices
      INTEGER  ::   iku, ikv           ! local integers
      REAL(wp) ::   zzwi, ze3ua, zdt   ! local scalars
      REAL(wp) ::   zzws, ze3va        !   -      -
      REAL(wp) ::   z1_e3ua, z1_e3va   !   -      -
      REAL(wp) ::   zWu , zWv          !   -      -
      REAL(wp) ::   zWui, zWvi         !   -      -
      REAL(wp) ::   zWus, zWvs         !   -      -
      REAL(wp), DIMENSION(jpi,jpj,jpk)        ::  zwi, zwd, zws,zzz   ! 3D workspace 
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   ztrdu, ztrdv   !  -      -
      !!---------------------------------------------------------------------
# if defined dyn_time
      real(8) :: st, ed
      real(8), dimension(9) :: elapsed
      integer :: print_flag
      common /flag/ print_flag
# endif
# if defined slave
      integer, external :: slave_dyn1, slave_dyn2, slave_dyn3, slave_dyn4, slave_dyn5, slave_dyn6, slave_dyn7, slave_dyn8
      type :: dyn_var
         integer :: jpi, jpj, jpk, jpjm1, jpkm1, fs_two, fs_jpim1, narea
         real(wp) :: r2dt, r_vvl, zdt, rau0
         integer*8 :: loc_ub, loc_ua, loc_umask, loc_vmask, loc_vb, loc_va, loc_ua_b, loc_va_b, loc_e3u_n, loc_e3u_a, loc_avm, loc_e3uw_n, loc_wumask, loc_zwi, loc_zws, loc_zwd, loc_utau_b, loc_utau, loc_e3v_n, loc_e3v_a, loc_e3vw_n, loc_wvmask, loc_vtau,loc_vtau_b
      end type dyn_var
      type(dyn_var) :: d_v
      d_v%jpi = jpi
      d_v%jpj = jpj
      d_v%jpk = jpk
      d_v%jpjm1 = jpjm1
      d_v%jpkm1 = jpkm1
      d_v%fs_two = fs_2
      d_v%fs_jpim1 = fs_jpim1
      d_v%narea = narea
      d_v%loc_ub = loc(ub)
      d_v%loc_ua = loc(ua)
      d_v%loc_umask = loc(umask)
      d_v%loc_vmask = loc(vmask)
      d_v%loc_vb = loc(vb)
      d_v%loc_va = loc(va)
      d_v%loc_ua_b = loc(ua_b)
      d_v%loc_va_b = loc(va_b)
      d_v%loc_e3u_n = loc(e3u_n)
      d_v%loc_e3u_a = loc(e3u_a)
      d_v%loc_avm = loc(avm)
      d_v%loc_e3uw_n = loc(e3uw_n)
      d_v%loc_wumask = loc(wumask)
      d_v%loc_zwi = loc(zwi)
      d_v%loc_zws = loc(zws)
      d_v%loc_zwd = loc(zwd)
      d_v%loc_utau = loc(utau)
      d_v%loc_utau_b = loc(utau_b)
      d_v%loc_e3v_n = loc(e3v_n)
      d_v%loc_e3v_a = loc(e3v_a)
      d_v%loc_e3vw_n = loc(e3vw_n)
      d_v%loc_wvmask = loc(wvmask)
      d_v%loc_vtau = loc(vtau)
      d_v%loc_vtau_b = loc(vtau_b)
      d_v%rau0 = rau0
# endif
      !
      IF( ln_timing )   CALL timing_start('dyn_zdf')
      !
      IF( kt == nit000 ) THEN       !* initialization
         IF(lwp) WRITE(numout,*)
         IF(lwp) WRITE(numout,*) 'dyn_zdf_imp : vertical momentum diffusion implicit operator'
         IF(lwp) WRITE(numout,*) '~~~~~~~~~~~ '
         !
         If( ln_linssh ) THEN   ;    r_vvl = 0._wp    ! non-linear free surface indicator
         ELSE                   ;    r_vvl = 1._wp
         ENDIF
      ENDIF
      !                             !* set time step
      IF( neuler == 0 .AND. kt == nit000     ) THEN   ;   r2dt =      rdt   ! = rdt (restart with Euler time stepping)
      ELSEIF(               kt <= nit000 + 1 ) THEN   ;   r2dt = 2. * rdt   ! = 2 rdt (leapfrog)
      ENDIF
      !
      !                             !* explicit top/bottom drag case
      !IF( .NOT.ln_drgimp )   CALL zdf_drg_exp( kt, ub, vb, ua, va )  ! add top/bottom friction trend to (ua,va)
      !
      !
      !IF( l_trddyn )   THEN         !* temporary save of ta and sa trends
      !   ALLOCATE( ztrdu(jpi,jpj,jpk), ztrdv(jpi,jpj,jpk) ) 
      !   ztrdu(:,:,:) = ua(:,:,:)
      !   ztrdv(:,:,:) = va(:,:,:)
      !ENDIF
      !
      !              !==  RHS: Leap-Frog time stepping on all trends but the vertical mixing  ==!   (put in ua,va)
      !
      !                    ! time stepping except vertical diffusion
# if defined dyn_time
      if(lwp) st = MPI_WTIME()
# endif
# if defined slave
      !DO jk = 1, jpkm1
      !   ua(:,:,jk) = ( ub(:,:,jk) + r2dt * ua(:,:,jk) ) * umask(:,:,jk)
      !   va(:,:,jk) = ( vb(:,:,jk) + r2dt * va(:,:,jk) ) * vmask(:,:,jk)
      !   ua(:,:,jk) = ( ua(:,:,jk) - ua_b(:,:) ) * umask(:,:,jk)
      !   va(:,:,jk) = ( va(:,:,jk) - va_b(:,:) ) * vmask(:,:,jk)
      !END DO
      d_v%r2dt = r2dt
      call athread_spawn(slave_dyn1, d_v)
      call athread_join()
# else
      !IF( ln_dynadv_vec .OR. ln_linssh ) THEN   ! applied on velocity
         DO jk = 1, jpkm1
            ua(:,:,jk) = ( ub(:,:,jk) + r2dt * ua(:,:,jk) ) * umask(:,:,jk)
            va(:,:,jk) = ( vb(:,:,jk) + r2dt * va(:,:,jk) ) * vmask(:,:,jk)
         END DO
      !ELSE                                      ! applied on thickness weighted velocity
      !   DO jk = 1, jpkm1
      !      ua(:,:,jk) = (         e3u_b(:,:,jk) * ub(:,:,jk)  &
      !         &          + r2dt * e3u_n(:,:,jk) * ua(:,:,jk)  ) / e3u_a(:,:,jk) * umask(:,:,jk)
      !      va(:,:,jk) = (         e3v_b(:,:,jk) * vb(:,:,jk)  &
      !         &          + r2dt * e3v_n(:,:,jk) * va(:,:,jk)  ) / e3v_a(:,:,jk) * vmask(:,:,jk)
      !   END DO
      !ENDIF
      !                    ! add top/bottom friction 
      !     With split-explicit free surface, barotropic stress is treated explicitly Update velocities at the bottom.
      !     J. Chanut: The bottom stress is computed considering after barotropic velocities, which does 
      !                not lead to the effective stress seen over the whole barotropic loop. 
      !     G. Madec : in linear free surface, e3u_a = e3u_n = e3u_0, so systematic use of e3u_a
      !IF( ln_drgimp .AND. ln_dynspg_ts ) THEN
         DO jk = 1, jpkm1        ! remove barotropic velocities
            ua(:,:,jk) = ( ua(:,:,jk) - ua_b(:,:) ) * umask(:,:,jk)
            va(:,:,jk) = ( va(:,:,jk) - va_b(:,:) ) * vmask(:,:,jk)
         END DO
# endif
# if defined dyn_time
      if(lwp) then
          ed = MPI_WTIME()
          elapsed(1) = ed - st
      end if
# endif
         DO jj = 2, jpjm1        ! Add bottom/top stress due to barotropic component only
            DO ji = fs_2, fs_jpim1   ! vector opt.
               iku = mbku(ji,jj)         ! ocean bottom level at u- and v-points 
               ikv = mbkv(ji,jj)         ! (deepest ocean u- and v-points)
               ze3ua =  ( 1._wp - r_vvl ) * e3u_n(ji,jj,iku) + r_vvl * e3u_a(ji,jj,iku)
               ze3va =  ( 1._wp - r_vvl ) * e3v_n(ji,jj,ikv) + r_vvl * e3v_a(ji,jj,ikv)
               ua(ji,jj,iku) = ua(ji,jj,iku) + r2dt * 0.5*( rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj) ) * ua_b(ji,jj) / ze3ua
               va(ji,jj,ikv) = va(ji,jj,ikv) + r2dt * 0.5*( rCdU_bot(ji,jj+1)+rCdU_bot(ji,jj) ) * va_b(ji,jj) / ze3va
            END DO
         END DO
      !   IF( ln_isfcav ) THEN    ! Ocean cavities (ISF)
      !      DO jj = 2, jpjm1        
      !         DO ji = fs_2, fs_jpim1   ! vector opt.
      !            iku = miku(ji,jj)         ! top ocean level at u- and v-points 
      !            ikv = mikv(ji,jj)         ! (first wet ocean u- and v-points)
      !            ze3ua =  ( 1._wp - r_vvl ) * e3u_n(ji,jj,iku) + r_vvl * e3u_a(ji,jj,iku)
      !            ze3va =  ( 1._wp - r_vvl ) * e3v_n(ji,jj,ikv) + r_vvl * e3v_a(ji,jj,ikv)
      !            ua(ji,jj,iku) = ua(ji,jj,iku) + r2dt * 0.5*( rCdU_top(ji+1,jj)+rCdU_top(ji,jj) ) * ua_b(ji,jj) / ze3ua
      !            va(ji,jj,ikv) = va(ji,jj,ikv) + r2dt * 0.5*( rCdU_top(ji,jj+1)+rCdU_top(ji,jj) ) * va_b(ji,jj) / ze3va
      !         END DO
      !      END DO
      !   END IF
      !ENDIF
      !
      !              !==  Vertical diffusion on u  ==!
      !
      !                    !* Matrix construction
      zdt = r2dt * 0.5
      !IF( ln_zad_Aimp ) THEN   !!
      !   SELECT CASE( nldf_dyn )
      !   CASE( np_lap_i )           ! rotated lateral mixing: add its vertical mixing (akzu)
      !      DO jk = 1, jpkm1
      !         DO jj = 2, jpjm1 
      !            DO ji = fs_2, fs_jpim1   ! vector opt.
      !               ze3ua =  ( 1._wp - r_vvl ) * e3u_n(ji,jj,jk) + r_vvl * e3u_a(ji,jj,jk)   ! after scale factor at U-point
      !               zzwi = - zdt * ( avm(ji+1,jj,jk  ) + avm(ji,jj,jk  ) + akzu(ji,jj,jk  ) )   &
      !                  &         / ( ze3ua * e3uw_n(ji,jj,jk  ) ) * wumask(ji,jj,jk  )
      !               zzws = - zdt * ( avm(ji+1,jj,jk+1) + avm(ji,jj,jk+1) + akzu(ji,jj,jk+1) )   &
      !                  &         / ( ze3ua * e3uw_n(ji,jj,jk+1) ) * wumask(ji,jj,jk+1)
      !               zWui = ( wi(ji,jj,jk  ) + wi(ji+1,jj,jk  ) ) / ze3ua
      !               zWus = ( wi(ji,jj,jk+1) + wi(ji+1,jj,jk+1) ) / ze3ua
      !               zwi(ji,jj,jk) = zzwi + zdt * MIN( zWui, 0._wp ) 
      !               zws(ji,jj,jk) = zzws - zdt * MAX( zWus, 0._wp )
      !               zwd(ji,jj,jk) = 1._wp - zzwi - zzws + zdt * ( MAX( zWui, 0._wp ) - MIN( zWus, 0._wp ) )
      !            END DO
      !         END DO
      !      END DO
      !   CASE DEFAULT               ! iso-level lateral mixing
      !      DO jk = 1, jpkm1
      !         DO jj = 2, jpjm1 
      !            DO ji = fs_2, fs_jpim1   ! vector opt.
      !               ze3ua =  ( 1._wp - r_vvl ) * e3u_n(ji,jj,jk) + r_vvl * e3u_a(ji,jj,jk)   ! after scale factor at U-point
      !               zzwi = - zdt * ( avm(ji+1,jj,jk  ) + avm(ji,jj,jk  ) ) / ( ze3ua * e3uw_n(ji,jj,jk  ) ) * wumask(ji,jj,jk  )
      !               zzws = - zdt * ( avm(ji+1,jj,jk+1) + avm(ji,jj,jk+1) ) / ( ze3ua * e3uw_n(ji,jj,jk+1) ) * wumask(ji,jj,jk+1)
      !               zWui = ( wi(ji,jj,jk  ) + wi(ji+1,jj,jk  ) ) / ze3ua
      !               zWus = ( wi(ji,jj,jk+1) + wi(ji+1,jj,jk+1) ) / ze3ua
      !               zwi(ji,jj,jk) = zzwi + zdt * MIN( zWui, 0._wp )
      !               zws(ji,jj,jk) = zzws - zdt * MAX( zWus, 0._wp )
      !               zwd(ji,jj,jk) = 1._wp - zzwi - zzws + zdt * ( MAX( zWui, 0._wp ) - MIN( zWus, 0._wp ) )
      !            END DO
      !         END DO
      !      END DO
      !   END SELECT
      !   DO jj = 2, jpjm1     !* Surface boundary conditions
      !      DO ji = fs_2, fs_jpim1   ! vector opt.
      !         zwi(ji,jj,1) = 0._wp
      !         ze3ua =  ( 1._wp - r_vvl ) * e3u_n(ji,jj,1) + r_vvl * e3u_a(ji,jj,1)
      !         zzws = - zdt * ( avm(ji+1,jj,2) + avm(ji  ,jj,2) ) / ( ze3ua * e3uw_n(ji,jj,2) ) * wumask(ji,jj,2)
      !         zWus = ( wi(ji  ,jj,2) +  wi(ji+1,jj,2) ) / ze3ua
      !         zws(ji,jj,1 ) = zzws - zdt * MAX( zWus, 0._wp )
      !         zwd(ji,jj,1 ) = 1._wp - zzws - zdt * ( MIN( zWus, 0._wp ) )
      !      END DO
      !   END DO
      !ELSE
      !   SELECT CASE( nldf_dyn )
      !   CASE( np_lap_i )           ! rotated lateral mixing: add its vertical mixing (akzu)
      !      DO jk = 1, jpkm1
      !         DO jj = 2, jpjm1 
      !            DO ji = fs_2, fs_jpim1   ! vector opt.
      !               ze3ua =  ( 1._wp - r_vvl ) * e3u_n(ji,jj,jk) + r_vvl * e3u_a(ji,jj,jk)   ! after scale factor at U-point
      !               zzwi = - zdt * ( avm(ji+1,jj,jk  ) + avm(ji,jj,jk  ) + akzu(ji,jj,jk  ) )   &
      !                  &         / ( ze3ua * e3uw_n(ji,jj,jk  ) ) * wumask(ji,jj,jk  )
      !               zzws = - zdt * ( avm(ji+1,jj,jk+1) + avm(ji,jj,jk+1) + akzu(ji,jj,jk+1) )   &
      !                  &         / ( ze3ua * e3uw_n(ji,jj,jk+1) ) * wumask(ji,jj,jk+1)
      !               zwi(ji,jj,jk) = zzwi
      !               zws(ji,jj,jk) = zzws
      !               zwd(ji,jj,jk) = 1._wp - zzwi - zzws
      !            END DO
      !         END DO
      !      END DO
      !   CASE DEFAULT               ! iso-level lateral mixing
# if defined dyn_time
            if(lwp) st = MPI_WTIME()
# endif
# if defined slave
            d_v%zdt = zdt
            d_v%r_vvl = r_vvl
            call athread_spawn(slave_dyn2, d_v)
            call athread_join()
# else
            DO jk = 1, jpkm1
               DO jj = 2, jpjm1 
                  DO ji = fs_2, fs_jpim1   ! vector opt.
                     ze3ua =  ( 1._wp - r_vvl ) * e3u_n(ji,jj,jk) + r_vvl * e3u_a(ji,jj,jk)   ! after scale factor at U-point
                     zzwi = - zdt * ( avm(ji+1,jj,jk  ) + avm(ji,jj,jk  ) ) / ( ze3ua * e3uw_n(ji,jj,jk  ) ) * wumask(ji,jj,jk  )
                     zzws = - zdt * ( avm(ji+1,jj,jk+1) + avm(ji,jj,jk+1) ) / ( ze3ua * e3uw_n(ji,jj,jk+1) ) * wumask(ji,jj,jk+1)
                     zwi(ji,jj,jk) = zzwi
                     zws(ji,jj,jk) = zzws
                     zwd(ji,jj,jk) = 1._wp - zzwi - zzws
                  END DO
               END DO
            END DO
# endif
# if defined dyn_time
            if(lwp) then
                ed = MPI_WTIME()
                elapsed(2) = ed - st
            end if
# endif
      !   END SELECT
         DO jj = 2, jpjm1     !* Surface boundary conditions
            DO ji = fs_2, fs_jpim1   ! vector opt.
               zwi(ji,jj,1) = 0._wp
               zwd(ji,jj,1) = 1._wp - zws(ji,jj,1)
            END DO
         END DO
      !ENDIF
      !
      !
      !              !==  Apply semi-implicit bottom friction  ==!
      !
      !     Only needed for semi-implicit bottom friction setup. The explicit
      !     bottom friction has been included in "u(v)a" which act as the R.H.S
      !     column vector of the tri-diagonal matrix equation
      !
      !IF ( ln_drgimp ) THEN      ! implicit bottom friction
         DO jj = 2, jpjm1
            DO ji = 2, jpim1
               iku = mbku(ji,jj)       ! ocean bottom level at u- and v-points
               ze3ua =  ( 1._wp - r_vvl ) * e3u_n(ji,jj,iku) + r_vvl * e3u_a(ji,jj,iku)   ! after scale factor at T-point
               zwd(ji,jj,iku) = zwd(ji,jj,iku) - r2dt * 0.5*( rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj) ) / ze3ua
            END DO
         END DO
      !   IF ( ln_isfcav ) THEN   ! top friction (always implicit)
      !      DO jj = 2, jpjm1
      !         DO ji = 2, jpim1
      !            !!gm   top Cd is masked (=0 outside cavities) no need of test on mik>=2  ==>> it has been suppressed
      !            iku = miku(ji,jj)       ! ocean top level at u- and v-points 
      !            ze3ua =  ( 1._wp - r_vvl ) * e3u_n(ji,jj,iku) + r_vvl * e3u_a(ji,jj,iku)   ! after scale factor at T-point
      !            zwd(ji,jj,iku) = zwd(ji,jj,iku) - r2dt * 0.5*( rCdU_top(ji+1,jj)+rCdU_top(ji,jj) ) / ze3ua
      !         END DO
      !      END DO
      !   END IF
      !ENDIF
      !
      ! Matrix inversion starting from the first level
      !-----------------------------------------------------------------------
      !   solve m.x = y  where m is a tri diagonal matrix ( jpk*jpk )
      !
      !        ( zwd1 zws1   0    0    0  )( zwx1 ) ( zwy1 )
      !        ( zwi2 zwd2 zws2   0    0  )( zwx2 ) ( zwy2 )
      !        (  0   zwi3 zwd3 zws3   0  )( zwx3 )=( zwy3 )
      !        (        ...               )( ...  ) ( ...  )
      !        (  0    0    0   zwik zwdk )( zwxk ) ( zwyk )
      !
      !   m is decomposed in the product of an upper and a lower triangular matrix
      !   The 3 diagonal terms are in 2d arrays: zwd, zws, zwi
      !   The solution (the after velocity) is in ua
      !-----------------------------------------------------------------------
      !
# if defined dyn_time
      if(lwp) st = MPI_WTIME()
# endif
# if defined slave
      call athread_spawn(slave_dyn3, d_v)
      call athread_join()
# else
      DO jk = 2, jpkm1        !==  First recurrence : Dk = Dk - Lk * Uk-1 / Dk-1   (increasing k)  ==
         DO jj = 2, jpjm1   
            DO ji = fs_2, fs_jpim1   ! vector opt.
               zwd(ji,jj,jk) = zwd(ji,jj,jk) - zwi(ji,jj,jk) * zws(ji,jj,jk-1) / zwd(ji,jj,jk-1)
            END DO
         END DO
      END DO
# endif
# if defined dyn_time
      if(lwp) then
          ed = MPI_WTIME()
          elapsed(3) = ed - st
      end if
# endif
      !
# if defined dyn_time
      if(lwp) st = MPI_WTIME()
# endif
# if defined slave
      call athread_spawn(slave_dyn4, d_v)
      call athread_join()
# else
      DO jj = 2, jpjm1        !==  second recurrence:    SOLk = RHSk - Lk / Dk-1  Lk-1  ==!
         DO ji = fs_2, fs_jpim1   ! vector opt.
            ze3ua =  ( 1._wp - r_vvl ) * e3u_n(ji,jj,1) + r_vvl * e3u_a(ji,jj,1) 
            ua(ji,jj,1) = ua(ji,jj,1) + r2dt * 0.5_wp * ( utau_b(ji,jj) + utau(ji,jj) )   &
               &                                      / ( ze3ua * rau0 ) * umask(ji,jj,1) 
         END DO
      END DO
      DO jk = 2, jpkm1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1
               ua(ji,jj,jk) = ua(ji,jj,jk) - zwi(ji,jj,jk) / zwd(ji,jj,jk-1) * ua(ji,jj,jk-1)
            END DO
         END DO
      END DO
# endif
# if defined dyn_time
      if(lwp) then
          ed = MPI_WTIME()
          elapsed(4) = ed - st
      end if
# endif
      !
# if defined dyn_time
      if(lwp) st = MPI_WTIME()
# endif
# if defined slave
      call athread_spawn(slave_dyn5, d_v)
      call athread_join()
# else
      DO jj = 2, jpjm1        !==  thrid recurrence : SOLk = ( Lk - Uk * Ek+1 ) / Dk  ==!
         DO ji = fs_2, fs_jpim1   ! vector opt.
            ua(ji,jj,jpkm1) = ua(ji,jj,jpkm1) / zwd(ji,jj,jpkm1)
         END DO
      END DO
      DO jk = jpk-2, 1, -1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1
               ua(ji,jj,jk) = ( ua(ji,jj,jk) - zws(ji,jj,jk) * ua(ji,jj,jk+1) ) / zwd(ji,jj,jk)
            END DO
         END DO
      END DO
# endif
# if defined dyn_time
      if(lwp) then
          ed = MPI_WTIME()
          elapsed(5) = ed - st
      end if
# endif
      !
      !              !==  Vertical diffusion on v  ==!
      !
      !                       !* Matrix construction
      zdt = r2dt * 0.5
      !IF( ln_zad_Aimp ) THEN   !!
      !   SELECT CASE( nldf_dyn )
      !   CASE( np_lap_i )           ! rotated lateral mixing: add its vertical mixing (akzv)
      !      DO jk = 1, jpkm1
      !         DO jj = 2, jpjm1 
      !            DO ji = fs_2, fs_jpim1   ! vector opt.
      !               ze3va =  ( 1._wp - r_vvl ) * e3v_n(ji,jj,jk) + r_vvl * e3v_a(ji,jj,jk)   ! after scale factor at V-point
      !               zzwi = - zdt * ( avm(ji,jj+1,jk  ) + avm(ji,jj,jk  ) + akzv(ji,jj,jk  ) )   &
      !                  &         / ( ze3va * e3vw_n(ji,jj,jk  ) ) * wvmask(ji,jj,jk  )
      !               zzws = - zdt * ( avm(ji,jj+1,jk+1) + avm(ji,jj,jk+1) + akzv(ji,jj,jk+1) )   &
      !                  &         / ( ze3va * e3vw_n(ji,jj,jk+1) ) * wvmask(ji,jj,jk+1)
      !               zWvi = ( wi(ji,jj,jk  ) + wi(ji,jj+1,jk  ) ) / ze3va
      !               zWvs = ( wi(ji,jj,jk+1) + wi(ji,jj+1,jk+1) ) / ze3va
      !               zwi(ji,jj,jk) = zzwi + zdt * MIN( zWvi, 0._wp )
      !               zws(ji,jj,jk) = zzws - zdt * MAX( zWvs, 0._wp )
      !               zwd(ji,jj,jk) = 1._wp - zzwi - zzws - zdt * ( - MAX( zWvi, 0._wp ) + MIN( zWvs, 0._wp ) )
      !            END DO
      !         END DO
      !      END DO
      !   CASE DEFAULT               ! iso-level lateral mixing
      !      DO jk = 1, jpkm1
      !         DO jj = 2, jpjm1 
      !            DO ji = fs_2, fs_jpim1   ! vector opt.
      !               ze3va =  ( 1._wp - r_vvl ) * e3v_n(ji,jj,jk) + r_vvl * e3v_a(ji,jj,jk)   ! after scale factor at V-point
      !               zzwi = - zdt * ( avm(ji,jj+1,jk  ) + avm(ji,jj,jk  ) ) / ( ze3va * e3vw_n(ji,jj,jk  ) ) * wvmask(ji,jj,jk  )
      !               zzws = - zdt * ( avm(ji,jj+1,jk+1) + avm(ji,jj,jk+1) ) / ( ze3va * e3vw_n(ji,jj,jk+1) ) * wvmask(ji,jj,jk+1)
      !               zWvi = ( wi(ji,jj,jk  ) + wi(ji,jj+1,jk  ) ) / ze3va
      !               zWvs = ( wi(ji,jj,jk+1) + wi(ji,jj+1,jk+1) ) / ze3va
      !               zwi(ji,jj,jk) = zzwi  + zdt * MIN( zWvi, 0._wp )
      !               zws(ji,jj,jk) = zzws  - zdt * MAX( zWvs, 0._wp )
      !               zwd(ji,jj,jk) = 1._wp - zzwi - zzws - zdt * ( - MAX( zWvi, 0._wp ) + MIN( zWvs, 0._wp ) )
      !            END DO
      !         END DO
      !      END DO
      !   END SELECT
      !   DO jj = 2, jpjm1     !* Surface boundary conditions
      !      DO ji = fs_2, fs_jpim1   ! vector opt.
      !         zwi(ji,jj,1) = 0._wp
      !         ze3va =  ( 1._wp - r_vvl ) * e3v_n(ji,jj,1) + r_vvl * e3v_a(ji,jj,1)
      !         zzws = - zdt * ( avm(ji,jj+1,2) + avm(ji,jj,2) ) / ( ze3va * e3vw_n(ji,jj,2) ) * wvmask(ji,jj,2)
      !         zWvs = ( wi(ji,jj  ,2) +  wi(ji,jj+1,2) ) / ze3va
      !         zws(ji,jj,1 ) = zzws - zdt * MAX( zWvs, 0._wp )
      !         zwd(ji,jj,1 ) = 1._wp - zzws - zdt * ( MIN( zWvs, 0._wp ) )
      !      END DO
      !   END DO
      !ELSE
      !   SELECT CASE( nldf_dyn )
      !   CASE( np_lap_i )           ! rotated lateral mixing: add its vertical mixing (akzu)
      !      DO jk = 1, jpkm1
      !         DO jj = 2, jpjm1   
      !            DO ji = fs_2, fs_jpim1   ! vector opt.
      !               ze3va =  ( 1._wp - r_vvl ) * e3v_n(ji,jj,jk) + r_vvl * e3v_a(ji,jj,jk)   ! after scale factor at V-point
      !               zzwi = - zdt * ( avm(ji,jj+1,jk  ) + avm(ji,jj,jk  ) + akzv(ji,jj,jk  ) )   &
      !                  &         / ( ze3va * e3vw_n(ji,jj,jk  ) ) * wvmask(ji,jj,jk  )
      !               zzws = - zdt * ( avm(ji,jj+1,jk+1) + avm(ji,jj,jk+1) + akzv(ji,jj,jk+1) )   &
      !                  &         / ( ze3va * e3vw_n(ji,jj,jk+1) ) * wvmask(ji,jj,jk+1)
      !               zwi(ji,jj,jk) = zzwi
      !               zws(ji,jj,jk) = zzws
      !               zwd(ji,jj,jk) = 1._wp - zzwi - zzws
      !            END DO
      !         END DO
      !      END DO
      !   CASE DEFAULT               ! iso-level lateral mixing
# if defined dyn_time
      if(lwp) st = MPI_WTIME()
# endif
# if defined slave
      d_v%zdt = zdt
      d_v%r_vvl = r_vvl
      call athread_spawn(slave_dyn6, d_v)
      call athread_join()
# else
            DO jk = 1, jpkm1
               DO jj = 2, jpjm1   
                  DO ji = fs_2, fs_jpim1   ! vector opt.
                     ze3va =  ( 1._wp - r_vvl ) * e3v_n(ji,jj,jk) + r_vvl * e3v_a(ji,jj,jk)   ! after scale factor at V-point
                     zzwi = - zdt * ( avm(ji,jj+1,jk  ) + avm(ji,jj,jk  ) ) / ( ze3va * e3vw_n(ji,jj,jk  ) ) * wvmask(ji,jj,jk  )
                     zzws = - zdt * ( avm(ji,jj+1,jk+1) + avm(ji,jj,jk+1) ) / ( ze3va * e3vw_n(ji,jj,jk+1) ) * wvmask(ji,jj,jk+1)
                     zwi(ji,jj,jk) = zzwi
                     zws(ji,jj,jk) = zzws
                     zwd(ji,jj,jk) = 1._wp - zzwi - zzws
                  END DO
               END DO
            END DO
# endif
!if(lwp) print *, zwd(2, 6, 1), zzz(2,6,1)
!do while(.true.)
!enddo
# if defined dyn_time
      if(lwp) then
          ed = MPI_WTIME()
          elapsed(6) = ed - st
      end if
# endif
      !   END SELECT
         DO jj = 2, jpjm1        !* Surface boundary conditions
            DO ji = fs_2, fs_jpim1   ! vector opt.
               zwi(ji,jj,1) = 0._wp
               zwd(ji,jj,1) = 1._wp - zws(ji,jj,1)
            END DO
         END DO
      !ENDIF
      !
      !              !==  Apply semi-implicit top/bottom friction  ==!
      !
      !     Only needed for semi-implicit bottom friction setup. The explicit
      !     bottom friction has been included in "u(v)a" which act as the R.H.S
      !     column vector of the tri-diagonal matrix equation
      !
      !IF( ln_drgimp ) THEN
         DO jj = 2, jpjm1
            DO ji = 2, jpim1
               ikv = mbkv(ji,jj)       ! (deepest ocean u- and v-points)
               ze3va =  ( 1._wp - r_vvl ) * e3v_n(ji,jj,ikv) + r_vvl * e3v_a(ji,jj,ikv)   ! after scale factor at T-point
               zwd(ji,jj,ikv) = zwd(ji,jj,ikv) - r2dt * 0.5*( rCdU_bot(ji,jj+1)+rCdU_bot(ji,jj) ) / ze3va           
            END DO
         END DO
      !   IF ( ln_isfcav ) THEN
      !      DO jj = 2, jpjm1
      !         DO ji = 2, jpim1
      !            ikv = mikv(ji,jj)       ! (first wet ocean u- and v-points)
      !            ze3va =  ( 1._wp - r_vvl ) * e3v_n(ji,jj,ikv) + r_vvl * e3v_a(ji,jj,ikv)   ! after scale factor at T-point
      !            zwd(ji,jj,ikv) = zwd(ji,jj,ikv) - r2dt * 0.5*( rCdU_top(ji,jj+1)+rCdU_top(ji,jj) ) / ze3va
      !         END DO
      !      END DO
      !   ENDIF
      !ENDIF

      ! Matrix inversion
      !-----------------------------------------------------------------------
      !   solve m.x = y  where m is a tri diagonal matrix ( jpk*jpk )
      !
      !        ( zwd1 zws1   0    0    0  )( zwx1 ) ( zwy1 )
      !        ( zwi2 zwd2 zws2   0    0  )( zwx2 ) ( zwy2 )
      !        (  0   zwi3 zwd3 zws3   0  )( zwx3 )=( zwy3 )
      !        (        ...               )( ...  ) ( ...  )
      !        (  0    0    0   zwik zwdk )( zwxk ) ( zwyk )
      !
      !   m is decomposed in the product of an upper and lower triangular matrix
      !   The 3 diagonal terms are in 2d arrays: zwd, zws, zwi
      !   The solution (after velocity) is in 2d array va
      !-----------------------------------------------------------------------
      !
# if defined dyn_time
      if(lwp) st = MPI_WTIME()
# endif
# if defined slave
      call athread_spawn(slave_dyn3, d_v)
      call athread_join()
# else
      DO jk = 2, jpkm1        !==  First recurrence : Dk = Dk - Lk * Uk-1 / Dk-1   (increasing k)  ==
         DO jj = 2, jpjm1   
            DO ji = fs_2, fs_jpim1   ! vector opt.
               zwd(ji,jj,jk) = zwd(ji,jj,jk) - zwi(ji,jj,jk) * zws(ji,jj,jk-1) / zwd(ji,jj,jk-1)
            END DO
         END DO
      END DO
# endif
# if defined dyn_time
      if(lwp) then
          ed = MPI_WTIME()
          elapsed(7) = ed - st
      end if
# endif
      !
# if defined dyn_time
      if(lwp) st = MPI_WTIME()
# endif
# if defined slave
      d_v%zdt = zdt
      d_v%r_vvl = r_vvl
      call athread_spawn(slave_dyn7, d_v)
      call athread_join()
# else
      DO jj = 2, jpjm1        !==  second recurrence:    SOLk = RHSk - Lk / Dk-1  Lk-1  ==!
         DO ji = fs_2, fs_jpim1   ! vector opt.          
            ze3va =  ( 1._wp - r_vvl ) * e3v_n(ji,jj,1) + r_vvl * e3v_a(ji,jj,1) 
            va(ji,jj,1) = va(ji,jj,1) + r2dt * 0.5_wp * ( vtau_b(ji,jj) + vtau(ji,jj) )   &
               &                                      / ( ze3va * rau0 ) * vmask(ji,jj,1) 
         END DO
      END DO
      DO jk = 2, jpkm1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1   ! vector opt.
               va(ji,jj,jk) = va(ji,jj,jk) - zwi(ji,jj,jk) / zwd(ji,jj,jk-1) * va(ji,jj,jk-1)
            END DO
         END DO
      END DO
# endif
# if defined dyn_time
      if(lwp) then
          ed = MPI_WTIME()
          elapsed(8) = ed - st
      end if
# endif
      !
# if defined dyn_time
      if(lwp) st = MPI_WTIME()
# endif
# if defined slave
      call athread_spawn(slave_dyn8, d_v)
      call athread_join()
# else
      DO jj = 2, jpjm1        !==  third recurrence : SOLk = ( Lk - Uk * SOLk+1 ) / Dk  ==!
         DO ji = fs_2, fs_jpim1   ! vector opt.
            va(ji,jj,jpkm1) = va(ji,jj,jpkm1) / zwd(ji,jj,jpkm1)
         END DO
      END DO
      DO jk = jpk-2, 1, -1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1
               va(ji,jj,jk) = ( va(ji,jj,jk) - zws(ji,jj,jk) * va(ji,jj,jk+1) ) / zwd(ji,jj,jk)
            END DO
         END DO
      END DO
# endif
# if defined dyn_time
      if(lwp) then
          ed = MPI_WTIME()
          elapsed(9) = ed - st
      end if
# endif
# if defined dyn_time
      if(lwp .and. print_flag>0) then
          print_flag = print_flag - 1
          print *, 'dyn_time:', elapsed
          print *, '------------------'
      endif
# endif
      !
      !IF( l_trddyn )   THEN                      ! save the vertical diffusive trends for further diagnostics
      !   ztrdu(:,:,:) = ( ua(:,:,:) - ub(:,:,:) ) / r2dt - ztrdu(:,:,:)
      !   ztrdv(:,:,:) = ( va(:,:,:) - vb(:,:,:) ) / r2dt - ztrdv(:,:,:)
      !   CALL trd_dyn( ztrdu, ztrdv, jpdyn_zdf, kt )
      !   DEALLOCATE( ztrdu, ztrdv ) 
      !ENDIF
      !                                          ! print mean trends (used for debugging)
      IF(ln_ctl)   CALL prt_ctl( tab3d_1=ua, clinfo1=' zdf  - Ua: ', mask1=umask,               &
         &                       tab3d_2=va, clinfo2=       ' Va: ', mask2=vmask, clinfo3='dyn' )
         !
      IF( ln_timing )   CALL timing_stop('dyn_zdf')
      !
   END SUBROUTINE dyn_zdf

   !!==============================================================================
END MODULE dynzdf
