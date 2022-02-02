#include "toggle.h90"

MODULE zdfmxl
   !!======================================================================
   !!                       ***  MODULE  zdfmxl  ***
   !! Ocean physics: mixed layer depth 
   !!======================================================================
   !! History :  1.0  ! 2003-08  (G. Madec)  original code
   !!            3.2  ! 2009-07  (S. Masson, G. Madec)  IOM + merge of DO-loop
   !!            3.7  ! 2012-03  (G. Madec)  make public the density criteria for trdmxl 
   !!             -   ! 2014-02  (F. Roquet)  mixed layer depth calculated using N2 instead of rhop 
   !!----------------------------------------------------------------------
   !!   zdf_mxl      : Compute the turbocline and mixed layer depths.
   !!----------------------------------------------------------------------
   USE oce            ! ocean dynamics and tracers variables
   USE dom_oce        ! ocean space and time domain variables
   USE trc_oce  , ONLY: l_offline         ! ocean space and time domain variables
   USE zdf_oce        ! ocean vertical physics
   !
   USE in_out_manager ! I/O manager
   USE prtctl         ! Print control
   USE phycst         ! physical constants
   USE iom            ! I/O library
   USE lib_mpp        ! MPP library
   use mpi

   IMPLICIT NONE
   PRIVATE

   PUBLIC   zdf_mxl   ! called by zdfphy.F90

   INTEGER , PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   nmln    !: number of level in the mixed layer (used by LDF, ZDF, TRD, TOP)
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   hmld    !: mixing layer depth (turbocline)      [m]   (used by TOP)
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   hmlp    !: mixed layer depth  (rho=rho0+zdcrit) [m]   (used by LDF)
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   hmlpt   !: depth of the last T-point inside the mixed layer [m] (used by LDF)

   REAL(wp), PUBLIC ::   rho_c = 0.01_wp    !: density criterion for mixed layer depth
   REAL(wp), PUBLIC ::   avt_c = 5.e-4_wp   ! Kz criterion for the turbocline depth

   !!----------------------------------------------------------------------
   !! NEMO/OCE 4.0 , NEMO Consortium (2018)
   !! $Id: zdfmxl.F90 10425 2018-12-19 21:54:16Z smasson $ 
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   INTEGER FUNCTION zdf_mxl_alloc()
      !!----------------------------------------------------------------------
      !!               ***  FUNCTION zdf_mxl_alloc  ***
      !!----------------------------------------------------------------------
      zdf_mxl_alloc = 0      ! set to zero if no array to be allocated
      IF( .NOT. ALLOCATED( nmln ) ) THEN
         ALLOCATE( nmln(jpi,jpj), hmld(jpi,jpj), hmlp(jpi,jpj), hmlpt(jpi,jpj), STAT= zdf_mxl_alloc )
         !
         CALL mpp_sum ( 'zdfmxl', zdf_mxl_alloc )
         IF( zdf_mxl_alloc /= 0 )   CALL ctl_stop( 'STOP', 'zdf_mxl_alloc: failed to allocate arrays.' )
         !
      ENDIF
   END FUNCTION zdf_mxl_alloc


   SUBROUTINE zdf_mxl( kt )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE zdfmxl  ***
      !!                   
      !! ** Purpose :   Compute the turbocline depth and the mixed layer depth
      !!              with density criteria.
      !!
      !! ** Method  :   The mixed layer depth is the shallowest W depth with 
      !!      the density of the corresponding T point (just bellow) bellow a
      !!      given value defined locally as rho(10m) + rho_c
      !!               The turbocline depth is the depth at which the vertical
      !!      eddy diffusivity coefficient (resulting from the vertical physics
      !!      alone, not the isopycnal part, see trazdf.F) fall below a given
      !!      value defined locally (avt_c here taken equal to 5 cm/s2 by default)
      !!
      !! ** Action  :   nmln, hmld, hmlp, hmlpt
      !!----------------------------------------------------------------------
      INTEGER, INTENT(in) ::   kt   ! ocean time-step index
      !
      INTEGER  ::   ji, jj, jk      ! dummy loop indices
      INTEGER  ::   iikn, iiki, ikt ! local integer
      REAL(wp) ::   zN2_c           ! local scalar
      INTEGER, DIMENSION(jpi,jpj) ::   imld   ! 2D workspace
      REAL(wp), DIMENSION(jpi,jpj) ::   gdepw_i 
      !!----------------------------------------------------------------------
      !
# if defined zdfmxl_time
       external :: cyc_time
       integer*8 :: st, ed, elapsed 
       real(wp) :: qbw
       integer :: print_flag
       common /flag/ print_flag
# endif
# if defined slave
      integer, external :: slave_zdfmxl
      type :: zdfmxl_var
         integer   :: jpi, jpj, jpk, jpim1, jpjm1, jpkm1, nlb10, narea, signal
         real(wp)  :: zN2_c, avt_c
         integer*8 :: loc_mbkt, loc_nmln
         integer*8 :: loc_rn2b, loc_e3wn, loc_avt, loc_wmask, loc_gdepwi, loc_gdepwn, loc_gdeptn, loc_ssmask, loc_hmld, loc_hmlp, loc_hmlpt
      end type zdfmxl_var
      type(zdfmxl_var) :: var_z
      var_z%jpi = jpi
      var_z%jpj = jpj
      var_z%jpk = jpk
      var_z%jpim1 = jpim1
      var_z%jpjm1 = jpjm1
      var_z%jpkm1 = jpkm1
      var_z%nlb10 = nlb10
      var_z%narea = narea
      var_z%signal = 0
      var_z%avt_c = avt_c
      var_z%loc_mbkt = loc(mbkt)
      var_z%loc_rn2b = loc(rn2b)
      var_z%loc_e3wn = loc(e3w_n)
      var_z%loc_avt = loc(avt)
      var_z%loc_wmask = loc(wmask)
      var_z%loc_gdepwi = loc(gdepw_i)
      var_z%loc_gdepwn = loc(gdepw_n)
      var_z%loc_gdeptn = loc(gdept_n)
      var_z%loc_ssmask = loc(ssmask)
# endif
      !
      IF( kt == nit000 ) THEN
         IF(lwp) WRITE(numout,*)
         IF(lwp) WRITE(numout,*) 'zdf_mxl : mixed layer depth'
         IF(lwp) WRITE(numout,*) '~~~~~~~ '
         !                             ! allocate zdfmxl arrays
         IF( zdf_mxl_alloc() /= 0 )   CALL ctl_stop( 'STOP', 'zdf_mxl : unable to allocate arrays' )
      ENDIF
      zN2_c = grav * rho_c * r1_rau0   ! convert density criteria into N^2 criteria
      !
# if defined slave
      var_z%zN2_c = zN2_c
      var_z%loc_nmln = loc(nmln)
      var_z%loc_hmld = loc(hmld)
      var_z%loc_hmlp = loc(hmlp)
      var_z%loc_hmlpt = loc(hmlpt)
# endif

# if defined zdfmxl_time
         if(lwp) call cyc_time(st)
# endif
# if defined slave
         call athread_spawn(slave_zdfmxl, var_z)
         DO jj = 1, jpjm1
            DO ji = 1, jpi
               gdepw_i(ji,jj) = gdepw_n(ji,jj,mbkt(ji,jj)+1)
            END DO
         END DO
         var_z%signal = 1
         DO ji = 1, jpi
            nmln(ji,jpj) = nlb10
            hmlp(ji,jpj) = 0._wp
            DO jk = nlb10, jpkm1
               ikt = mbkt(ji,jpj)
               hmlp(ji,jpj) = hmlp(ji,jpj) + MAX(rn2b(ji,jpj,jk), 0._wp)*e3w_n(ji,jpj,jk)
               IF(hmlp(ji,jpj) < zN2_c) nmln(ji,jpj) = MIN(jk, ikt) + 1
            END DO
            iikn = nmln(ji,jpj)
            hmlp (ji,jpj) = gdepw_n(ji,jpj,iikn  )*ssmask(ji,jpj)
            hmlpt(ji,jpj) = gdept_n(ji,jpj,iikn-1)*ssmask(ji,jpj)
            imld(ji,jpj) = mbkt(ji,jpj) + 1
            DO jk = jpkm1, nlb10, -1
               IF(avt(ji,jpj,jk) < avt_c*wmask(ji,jpj,jk)) imld(ji,jpj) = jk
            END DO
            iiki = imld(ji,jpj)
            hmld(ji,jpj) = gdepw_n(ji,jpj,iiki)*ssmask(ji,jpj) 
         END DO
         call athread_join()
# else
      ! w-level of the mixing and mixed layers
      nmln(:,:)  = nlb10               ! Initialization to the number of w ocean point
      hmlp(:,:)  = 0._wp               ! here hmlp used as a dummy variable, integrating vertically N^2
      DO jk = nlb10, jpkm1
         DO jj = 1, jpj                ! Mixed layer level: w-level 
            DO ji = 1, jpi
               ikt = mbkt(ji,jj)
               hmlp(ji,jj) = hmlp(ji,jj) + MAX( rn2b(ji,jj,jk) , 0._wp ) * e3w_n(ji,jj,jk)
               IF( hmlp(ji,jj) < zN2_c )   nmln(ji,jj) = MIN( jk , ikt ) + 1   ! Mixed layer level
            END DO
         END DO
      END DO
      ! w-level of the turbocline and mixing layer (iom_use)
      imld(:,:) = mbkt(:,:) + 1        ! Initialization to the number of w ocean point
      DO jk = jpkm1, nlb10, -1         ! from the bottom to nlb10 
         DO jj = 1, jpj
            DO ji = 1, jpi
               IF( avt (ji,jj,jk) < avt_c * wmask(ji,jj,jk) )   imld(ji,jj) = jk      ! Turbocline 
            END DO
         END DO
      END DO
      ! depth of the mixing and mixed layers
      DO jj = 1, jpj
         DO ji = 1, jpi
            iiki = imld(ji,jj)
            iikn = nmln(ji,jj)
            hmld (ji,jj) = gdepw_n(ji,jj,iiki  ) * ssmask(ji,jj)    ! Turbocline depth 
            hmlp (ji,jj) = gdepw_n(ji,jj,iikn  ) * ssmask(ji,jj)    ! Mixed layer depth
            hmlpt(ji,jj) = gdept_n(ji,jj,iikn-1) * ssmask(ji,jj)    ! depth of the last T-point inside the mixed layer
         END DO
      END DO
      !
# endif
# if defined zdfmxl_time
         if(lwp) then
           call cyc_time(ed)
           elapsed = ed - st
           qbw = 2.25*(8*jpi*jpj*jpk*6 + 8*jpi*jpj*5 + 4*jpi*jpj*2)/((elapsed-60000)*45.0)
         end if
# endif
# if defined zdfmxl_time
         if(lwp) then
           print *, 'zdfmxl_time: ', elapsed, ' BW: ', qbw
         endif
# endif

      !!! IF All FALSE
      IF( .NOT.l_offline ) THEN
         IF( iom_use("mldr10_1") ) THEN
            IF( ln_isfcav ) THEN  ;  CALL iom_put( "mldr10_1", hmlp - risfdep)   ! mixed layer thickness
            ELSE                  ;  CALL iom_put( "mldr10_1", hmlp )            ! mixed layer depth
            END IF
         END IF
         IF( iom_use("mldkz5") ) THEN
            IF( ln_isfcav ) THEN  ;  CALL iom_put( "mldkz5"  , hmld - risfdep )   ! turbocline thickness
            ELSE                  ;  CALL iom_put( "mldkz5"  , hmld )             ! turbocline depth
            END IF
         ENDIF
      ENDIF
      !
      IF(ln_ctl)   CALL prt_ctl( tab2d_1=REAL(nmln,wp), clinfo1=' nmln : ', tab2d_2=hmlp, clinfo2=' hmlp : ' )
      !
   END SUBROUTINE zdf_mxl

   !!======================================================================
END MODULE zdfmxl
