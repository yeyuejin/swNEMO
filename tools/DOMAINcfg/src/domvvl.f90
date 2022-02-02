MODULE domvvl
   !!======================================================================
   !!                       ***  MODULE domvvl   ***
   !! Ocean : 
   !!======================================================================
   !! History :  2.0  !  2006-06  (B. Levier, L. Marie)  original code
   !!            3.1  !  2009-02  (G. Madec, M. Leclair, R. Benshila)  pure z* coordinate
   !!            3.3  !  2011-10  (M. Leclair) totally rewrote domvvl:
   !!                                          vvl option includes z_star and z_tilde coordinates
   !!            3.6  !  2014-11  (P. Mathiot) add ice shelf capability
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   dom_vvl_init     : define initial vertical scale factors, depths and column thickness
   !!   dom_vvl_sf_nxt   : Compute next vertical scale factors
   !!   dom_vvl_sf_swp   : Swap vertical scale factors and update the vertical grid
   !!   dom_vvl_interpol : Interpolate vertical scale factors from one grid point to another
   !!   dom_vvl_rst      : read/write restart file
   !!   dom_vvl_ctl      : Check the vvl options
   !!----------------------------------------------------------------------
   USE oce             ! ocean dynamics and tracers
   USE phycst          ! physical constant
   USE dom_oce         ! ocean space and time domain
   !
   USE in_out_manager  ! I/O manager
   USE iom             ! I/O manager library
   USE lib_mpp         ! distributed memory computing library
   USE lbclnk          ! ocean lateral boundary conditions (or mpp link)
   USE wrk_nemo        ! Memory allocation
   USE timing          ! Timing

   IMPLICIT NONE
   PRIVATE

   PUBLIC  dom_vvl_init       ! called by domain.F90

   !                                                      !!* Namelist nam_vvl
   LOGICAL , PUBLIC :: ln_vvl_zstar           = .FALSE.    ! zstar  vertical coordinate
   LOGICAL , PUBLIC :: ln_vvl_ztilde          = .FALSE.    ! ztilde vertical coordinate
   LOGICAL , PUBLIC :: ln_vvl_layer           = .FALSE.    ! level  vertical coordinate
   LOGICAL , PUBLIC :: ln_vvl_ztilde_as_zstar = .FALSE.    ! ztilde vertical coordinate
   LOGICAL , PUBLIC :: ln_vvl_zstar_at_eqtor  = .FALSE.    ! ztilde vertical coordinate
   LOGICAL , PUBLIC :: ln_vvl_kepe            = .FALSE.    ! kinetic/potential energy transfer
   !                                                       ! conservation: not used yet
   REAL(wp)         :: rn_ahe3                             ! thickness diffusion coefficient
   REAL(wp)         :: rn_rst_e3t                          ! ztilde to zstar restoration timescale [days]
   REAL(wp)         :: rn_lf_cutoff                        ! cutoff frequency for low-pass filter  [days]
   REAL(wp)         :: rn_zdef_max                         ! maximum fractional e3t deformation
   LOGICAL , PUBLIC :: ln_vvl_dbg = .FALSE.                ! debug control prints

   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: un_td, vn_td                ! thickness diffusion transport
   REAL(wp)        , ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: hdiv_lf                     ! low frequency part of hz divergence
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: tilde_e3t_b, tilde_e3t_n    ! baroclinic scale factors
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: tilde_e3t_a, dtilde_e3t_a   ! baroclinic scale factors
   REAL(wp)        , ALLOCATABLE, SAVE, DIMENSION(:,:)   :: frq_rst_e3t                 ! retoring period for scale factors
   REAL(wp)        , ALLOCATABLE, SAVE, DIMENSION(:,:)   :: frq_rst_hdv                 ! retoring period for low freq. divergence

   !! * Substitutions
   !!----------------------------------------------------------------------
   !!                   ***  vectopt_loop_substitute  ***
   !!----------------------------------------------------------------------
   !! ** purpose :   substitute the inner loop start/end indices with CPP macro
   !!                allow unrolling of do-loop (useful with vector processors)
   !!----------------------------------------------------------------------
   !!----------------------------------------------------------------------
   !! NEMO/OPA 3.7 , NEMO Consortium (2014)
   !! $Id: vectopt_loop_substitute.h90 4990 2014-12-15 16:42:49Z timgraham $ 
   !! Software governed by the CeCILL licence (./LICENSE)
   !!----------------------------------------------------------------------
   !!----------------------------------------------------------------------
   !! NEMO/OPA 3.7 , NEMO-Consortium (2015) 
   !! $Id: domvvl.F90 6351 2016-02-24 18:50:11Z cetlod $
   !! Software governed by the CeCILL licence     (./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   INTEGER FUNCTION dom_vvl_alloc()
      !!----------------------------------------------------------------------
      !!                ***  FUNCTION dom_vvl_alloc  ***
      !!----------------------------------------------------------------------
      IF( ln_vvl_zstar )   dom_vvl_alloc = 0
      IF( ln_vvl_ztilde .OR. ln_vvl_layer ) THEN
         ALLOCATE( tilde_e3t_b(jpi,jpj,jpk)  , tilde_e3t_n(jpi,jpj,jpk) , tilde_e3t_a(jpi,jpj,jpk) ,   &
            &      dtilde_e3t_a(jpi,jpj,jpk) , un_td  (jpi,jpj,jpk)     , vn_td  (jpi,jpj,jpk)     ,   &
            &      STAT = dom_vvl_alloc        )
         IF( lk_mpp             )   CALL mpp_sum ( dom_vvl_alloc )
         IF( dom_vvl_alloc /= 0 )   CALL ctl_warn('dom_vvl_alloc: failed to allocate arrays')
         un_td = 0._wp
         vn_td = 0._wp
      ENDIF
      IF( ln_vvl_ztilde ) THEN
         ALLOCATE( frq_rst_e3t(jpi,jpj) , frq_rst_hdv(jpi,jpj) , hdiv_lf(jpi,jpj,jpk) , STAT= dom_vvl_alloc )
         IF( lk_mpp             )   CALL mpp_sum ( dom_vvl_alloc )
         IF( dom_vvl_alloc /= 0 )   CALL ctl_warn('dom_vvl_alloc: failed to allocate arrays')
      ENDIF
      !
   END FUNCTION dom_vvl_alloc


   SUBROUTINE dom_vvl_init
      !!----------------------------------------------------------------------
      !!                ***  ROUTINE dom_vvl_init  ***
      !!                   
      !! ** Purpose :  Initialization of all scale factors, depths
      !!               and water column heights
      !!
      !! ** Method  :  - use restart file and/or initialize
      !!               - interpolate scale factors
      !!
      !! ** Action  : - e3t_(n/b) and tilde_e3t_(n/b)
      !!              - Regrid: e3(u/v)_n
      !!                        e3(u/v)_b       
      !!                        e3w_n           
      !!                        e3(u/v)w_b      
      !!                        e3(u/v)w_n      
      !!                        gdept_n, gdepw_n and gde3w_n
      !!              - h(t/u/v)_0
      !!              - frq_rst_e3t and frq_rst_hdv
      !!
      !! Reference  : Leclair, M., and G. Madec, 2011, Ocean Modelling.
      !!----------------------------------------------------------------------
      INTEGER ::   ji, jj, jk
      INTEGER ::   ii0, ii1, ij0, ij1
      REAL(wp)::   zcoef
      !!----------------------------------------------------------------------
      !
      IF( nn_timing == 1 )   CALL timing_start('dom_vvl_init')
      !
      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*) 'dom_vvl_init : Variable volume activated'
      IF(lwp) WRITE(numout,*) '~~~~~~~~~~~~'
      !
      CALL dom_vvl_ctl     ! choose vertical coordinate (z_star, z_tilde or layer)
      !
      !                    ! Allocate module arrays
      IF( dom_vvl_alloc() /= 0 )   CALL ctl_stop( 'STOP', 'dom_vvl_init : unable to allocate arrays' )
      !
      !                    ! Read or initialize e3t_(b/n), tilde_e3t_(b/n) and hdiv_lf
      e3t_a(:,:,jpk) = e3t_0(:,:,jpk)  ! last level always inside the sea floor set one for all
      !
      !                    !== Set of all other vertical scale factors  ==!  (now and before)
      !                                ! Horizontal interpolation of e3t
      CALL dom_vvl_interpol( e3t_b(:,:,:), e3u_b(:,:,:), 'U' )    ! from T to U
      CALL dom_vvl_interpol( e3t_n(:,:,:), e3u_n(:,:,:), 'U' )
      CALL dom_vvl_interpol( e3t_b(:,:,:), e3v_b(:,:,:), 'V' )    ! from T to V 
      CALL dom_vvl_interpol( e3t_n(:,:,:), e3v_n(:,:,:), 'V' )
      CALL dom_vvl_interpol( e3u_n(:,:,:), e3f_n(:,:,:), 'F' )    ! from U to F
      !                                ! Vertical interpolation of e3t,u,v 
      CALL dom_vvl_interpol( e3t_n(:,:,:), e3w_n (:,:,:), 'W'  )  ! from T to W
      CALL dom_vvl_interpol( e3t_b(:,:,:), e3w_b (:,:,:), 'W'  )
      CALL dom_vvl_interpol( e3u_n(:,:,:), e3uw_n(:,:,:), 'UW' )  ! from U to UW
      CALL dom_vvl_interpol( e3u_b(:,:,:), e3uw_b(:,:,:), 'UW' )
      CALL dom_vvl_interpol( e3v_n(:,:,:), e3vw_n(:,:,:), 'VW' )  ! from V to UW
      CALL dom_vvl_interpol( e3v_b(:,:,:), e3vw_b(:,:,:), 'VW' )
      !
      !                    !==  depth of t and w-point  ==!   (set the isf depth as it is in the initial timestep)
      gdept_n(:,:,1) = 0.5_wp * e3w_n(:,:,1)       ! reference to the ocean surface (used for MLD and light penetration)
      gdepw_n(:,:,1) = 0.0_wp
      gde3w_n(:,:,1) = gdept_n(:,:,1) - sshn(:,:)  ! reference to a common level z=0 for hpg
      gdept_b(:,:,1) = 0.5_wp * e3w_b(:,:,1)
      gdepw_b(:,:,1) = 0.0_wp
      DO jk = 2, jpk                               ! vertical sum
         DO jj = 1,jpj
            DO ji = 1,jpi
               !    zcoef = tmask - wmask    ! 0 everywhere tmask = wmask, ie everywhere expect at jk = mikt
               !                             ! 1 everywhere from mbkt to mikt + 1 or 1 (if no isf)
               !                             ! 0.5 where jk = mikt     
!!gm ???????   BUG ?  gdept_n as well as gde3w_n  does not include the thickness of ISF ??
               zcoef = ( tmask(ji,jj,jk) - wmask(ji,jj,jk) )
               gdepw_n(ji,jj,jk) = gdepw_n(ji,jj,jk-1) + e3t_n(ji,jj,jk-1)
               gdept_n(ji,jj,jk) =      zcoef  * ( gdepw_n(ji,jj,jk  ) + 0.5 * e3w_n(ji,jj,jk))  &
                  &                + (1-zcoef) * ( gdept_n(ji,jj,jk-1) +       e3w_n(ji,jj,jk)) 
               gde3w_n(ji,jj,jk) = gdept_n(ji,jj,jk) - sshn(ji,jj)
               gdepw_b(ji,jj,jk) = gdepw_b(ji,jj,jk-1) + e3t_b(ji,jj,jk-1)
               gdept_b(ji,jj,jk) =      zcoef  * ( gdepw_b(ji,jj,jk  ) + 0.5 * e3w_b(ji,jj,jk))  &
                  &                + (1-zcoef) * ( gdept_b(ji,jj,jk-1) +       e3w_b(ji,jj,jk)) 
            END DO
         END DO
      END DO
      !
      !                    !==  thickness of the water column  !!   (ocean portion only)
      ht_n(:,:) = e3t_n(:,:,1) * tmask(:,:,1)   !!gm  BUG  :  this should be 1/2 * e3w(k=1) ....
      hu_b(:,:) = e3u_b(:,:,1) * umask(:,:,1)
      hu_n(:,:) = e3u_n(:,:,1) * umask(:,:,1)
      hv_b(:,:) = e3v_b(:,:,1) * vmask(:,:,1)
      hv_n(:,:) = e3v_n(:,:,1) * vmask(:,:,1)
      DO jk = 2, jpkm1
         ht_n(:,:) = ht_n(:,:) + e3t_n(:,:,jk) * tmask(:,:,jk)
         hu_b(:,:) = hu_b(:,:) + e3u_b(:,:,jk) * umask(:,:,jk)
         hu_n(:,:) = hu_n(:,:) + e3u_n(:,:,jk) * umask(:,:,jk)
         hv_b(:,:) = hv_b(:,:) + e3v_b(:,:,jk) * vmask(:,:,jk)
         hv_n(:,:) = hv_n(:,:) + e3v_n(:,:,jk) * vmask(:,:,jk)
      END DO
      !
      !                    !==  inverse of water column thickness   ==!   (u- and v- points)
      r1_hu_b(:,:) = ssumask(:,:) / ( hu_b(:,:) + 1._wp - ssumask(:,:) )    ! _i mask due to ISF
      r1_hu_n(:,:) = ssumask(:,:) / ( hu_n(:,:) + 1._wp - ssumask(:,:) )
      r1_hv_b(:,:) = ssvmask(:,:) / ( hv_b(:,:) + 1._wp - ssvmask(:,:) )
      r1_hv_n(:,:) = ssvmask(:,:) / ( hv_n(:,:) + 1._wp - ssvmask(:,:) )

      !                    !==   z_tilde coordinate case  ==!   (Restoring frequencies)
      IF( ln_vvl_ztilde ) THEN
!!gm : idea: add here a READ in a file of custumized restoring frequency
         !                                   ! Values in days provided via the namelist
         !                                   ! use rsmall to avoid possible division by zero errors with faulty settings
         frq_rst_e3t(:,:) = 2._wp * rpi / ( MAX( rn_rst_e3t  , rsmall ) * 86400.0_wp )
         frq_rst_hdv(:,:) = 2._wp * rpi / ( MAX( rn_lf_cutoff, rsmall ) * 86400.0_wp )
         !
         IF( ln_vvl_ztilde_as_zstar ) THEN   ! z-star emulation using z-tile
            frq_rst_e3t(:,:) = 0._wp               !Ignore namelist settings
            frq_rst_hdv(:,:) = 1._wp / rdt
         ENDIF
         IF ( ln_vvl_zstar_at_eqtor ) THEN   ! use z-star in vicinity of the Equator
            DO jj = 1, jpj
               DO ji = 1, jpi
!!gm  case |gphi| >= 6 degrees is useless   initialized just above by default
                  IF( ABS(gphit(ji,jj)) >= 6.) THEN
                     ! values outside the equatorial band and transition zone (ztilde)
                     frq_rst_e3t(ji,jj) =  2.0_wp * rpi / ( MAX( rn_rst_e3t  , rsmall ) * 86400.e0_wp )
                     frq_rst_hdv(ji,jj) =  2.0_wp * rpi / ( MAX( rn_lf_cutoff, rsmall ) * 86400.e0_wp )
                  ELSEIF( ABS(gphit(ji,jj)) <= 2.5) THEN    ! Equator strip ==> z-star
                     ! values inside the equatorial band (ztilde as zstar)
                     frq_rst_e3t(ji,jj) =  0.0_wp
                     frq_rst_hdv(ji,jj) =  1.0_wp / rdt
                  ELSE                                      ! transition band (2.5 to 6 degrees N/S)
                     !                                      ! (linearly transition from z-tilde to z-star)
                     frq_rst_e3t(ji,jj) = 0.0_wp + (frq_rst_e3t(ji,jj)-0.0_wp)*0.5_wp   &
                        &            * (  1.0_wp - COS( rad*(ABS(gphit(ji,jj))-2.5_wp)  &
                        &                                          * 180._wp / 3.5_wp ) )
                     frq_rst_hdv(ji,jj) = (1.0_wp / rdt)                                &
                        &            + (  frq_rst_hdv(ji,jj)-(1.e0_wp / rdt) )*0.5_wp   &
                        &            * (  1._wp  - COS( rad*(ABS(gphit(ji,jj))-2.5_wp)  &
                        &                                          * 180._wp / 3.5_wp ) )
                  ENDIF
               END DO
            END DO
            IF( cp_cfg == "orca" .AND. jp_cfg == 3 ) THEN   ! ORCA2: Suppress ztilde in the Foxe Basin for ORCA2
               ii0 = 103   ;   ii1 = 111       
               ij0 = 128   ;   ij1 = 135   ;   
               frq_rst_e3t( mi0(ii0):mi1(ii1) , mj0(ij0):mj1(ij1) ) =  0.0_wp
               frq_rst_hdv( mi0(ii0):mi1(ii1) , mj0(ij0):mj1(ij1) ) =  1.e0_wp / rdt
            ENDIF
         ENDIF
      ENDIF
      !
      IF( nn_timing == 1 )  CALL timing_stop('dom_vvl_init')
      !
   END SUBROUTINE dom_vvl_init


   SUBROUTINE dom_vvl_interpol( pe3_in, pe3_out, pout )
      !!---------------------------------------------------------------------
      !!                  ***  ROUTINE dom_vvl__interpol  ***
      !!
      !! ** Purpose :   interpolate scale factors from one grid point to another
      !!
      !! ** Method  :   e3_out = e3_0 + interpolation(e3_in - e3_0)
      !!                - horizontal interpolation: grid cell surface averaging
      !!                - vertical interpolation: simple averaging
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in   ) ::  pe3_in    ! input e3 to be interpolated
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(inout) ::  pe3_out   ! output interpolated e3
      CHARACTER(LEN=*)                , INTENT(in   ) ::  pout      ! grid point of out scale factors
      !                                                             !   =  'U', 'V', 'W, 'F', 'UW' or 'VW'
      !
      INTEGER ::   ji, jj, jk                                       ! dummy loop indices
      REAL(wp) ::  zlnwd                                            ! =1./0. when ln_wd = T/F
      !!----------------------------------------------------------------------
      !
      IF( nn_timing == 1 )   CALL timing_start('dom_vvl_interpol')
      !
      zlnwd = 0.0_wp
      !
      SELECT CASE ( pout )    !==  type of interpolation  ==!
         !
      CASE( 'U' )                   !* from T- to U-point : hor. surface weighted mean
         DO jk = 1, jpk
            DO jj = 1, jpjm1
               DO ji = 1, jpim1   ! vector opt.
                  pe3_out(ji,jj,jk) = 0.5_wp * (  umask(ji,jj,jk) * (1.0_wp - zlnwd) + zlnwd ) * r1_e1e2u(ji,jj)   &
                     &                       * (   e1e2t(ji  ,jj) * ( pe3_in(ji  ,jj,jk) - e3t_0(ji  ,jj,jk) )     &
                     &                           + e1e2t(ji+1,jj) * ( pe3_in(ji+1,jj,jk) - e3t_0(ji+1,jj,jk) ) )
               END DO
            END DO
         END DO
         CALL lbc_lnk( pe3_out(:,:,:), 'U', 1._wp )
         pe3_out(:,:,:) = pe3_out(:,:,:) + e3u_0(:,:,:)
         !
      CASE( 'V' )                   !* from T- to V-point : hor. surface weighted mean
         DO jk = 1, jpk
            DO jj = 1, jpjm1
               DO ji = 1, jpim1   ! vector opt.
                  pe3_out(ji,jj,jk) = 0.5_wp * ( vmask(ji,jj,jk)  * (1.0_wp - zlnwd) + zlnwd ) * r1_e1e2v(ji,jj)   &
                     &                       * (   e1e2t(ji,jj  ) * ( pe3_in(ji,jj  ,jk) - e3t_0(ji,jj  ,jk) )     &
                     &                           + e1e2t(ji,jj+1) * ( pe3_in(ji,jj+1,jk) - e3t_0(ji,jj+1,jk) ) )
               END DO
            END DO
         END DO
         CALL lbc_lnk( pe3_out(:,:,:), 'V', 1._wp )
         pe3_out(:,:,:) = pe3_out(:,:,:) + e3v_0(:,:,:)
         !
      CASE( 'F' )                   !* from U-point to F-point : hor. surface weighted mean
         DO jk = 1, jpk
            DO jj = 1, jpjm1
               DO ji = 1, jpim1   ! vector opt.
                  pe3_out(ji,jj,jk) = 0.5_wp * (  umask(ji,jj,jk) * umask(ji,jj+1,jk) * (1.0_wp - zlnwd) + zlnwd ) &
                     &                       *    r1_e1e2f(ji,jj)                                                  &
                     &                       * (   e1e2u(ji,jj  ) * ( pe3_in(ji,jj  ,jk) - e3u_0(ji,jj  ,jk) )     &
                     &                           + e1e2u(ji,jj+1) * ( pe3_in(ji,jj+1,jk) - e3u_0(ji,jj+1,jk) ) )
               END DO
            END DO
         END DO
         CALL lbc_lnk( pe3_out(:,:,:), 'F', 1._wp )
         pe3_out(:,:,:) = pe3_out(:,:,:) + e3f_0(:,:,:)
         !
      CASE( 'W' )                   !* from T- to W-point : vertical simple mean
         !
         pe3_out(:,:,1) = e3w_0(:,:,1) + pe3_in(:,:,1) - e3t_0(:,:,1)
         ! - ML - The use of mask in this formulea enables the special treatment of the last w-point without indirect adressing
!!gm BUG? use here wmask in case of ISF ?  to be checked
         DO jk = 2, jpk
            pe3_out(:,:,jk) = e3w_0(:,:,jk) + ( 1.0_wp - 0.5_wp * ( tmask(:,:,jk) * (1.0_wp - zlnwd) + zlnwd ) )   &
               &                            * ( pe3_in(:,:,jk-1) - e3t_0(:,:,jk-1) )                               &
               &                            +            0.5_wp * ( tmask(:,:,jk) * (1.0_wp - zlnwd) + zlnwd )     &
               &                            * ( pe3_in(:,:,jk  ) - e3t_0(:,:,jk  ) )
         END DO
         !
      CASE( 'UW' )                  !* from U- to UW-point : vertical simple mean
         !
         pe3_out(:,:,1) = e3uw_0(:,:,1) + pe3_in(:,:,1) - e3u_0(:,:,1)
         ! - ML - The use of mask in this formaula enables the special treatment of the last w- point without indirect adressing
!!gm BUG? use here wumask in case of ISF ?  to be checked
         DO jk = 2, jpk
            pe3_out(:,:,jk) = e3uw_0(:,:,jk) + ( 1.0_wp - 0.5_wp * ( umask(:,:,jk) * (1.0_wp - zlnwd) + zlnwd ) )  &
               &                             * ( pe3_in(:,:,jk-1) - e3u_0(:,:,jk-1) )                              &
               &                             +            0.5_wp * ( umask(:,:,jk) * (1.0_wp - zlnwd) + zlnwd )    &
               &                             * ( pe3_in(:,:,jk  ) - e3u_0(:,:,jk  ) )
         END DO
         !
      CASE( 'VW' )                  !* from V- to VW-point : vertical simple mean
         !
         pe3_out(:,:,1) = e3vw_0(:,:,1) + pe3_in(:,:,1) - e3v_0(:,:,1)
         ! - ML - The use of mask in this formaula enables the special treatment of the last w- point without indirect adressing
!!gm BUG? use here wvmask in case of ISF ?  to be checked
         DO jk = 2, jpk
            pe3_out(:,:,jk) = e3vw_0(:,:,jk) + ( 1.0_wp - 0.5_wp * ( vmask(:,:,jk) * (1.0_wp - zlnwd) + zlnwd ) )  &
               &                             * ( pe3_in(:,:,jk-1) - e3v_0(:,:,jk-1) )                              &
               &                             +            0.5_wp * ( vmask(:,:,jk) * (1.0_wp - zlnwd) + zlnwd )    &
               &                             * ( pe3_in(:,:,jk  ) - e3v_0(:,:,jk  ) )
         END DO
      END SELECT
      !
      IF( nn_timing == 1 )   CALL timing_stop('dom_vvl_interpol')
      !
   END SUBROUTINE dom_vvl_interpol


   SUBROUTINE dom_vvl_ctl
      !!---------------------------------------------------------------------
      !!                  ***  ROUTINE dom_vvl_ctl  ***
      !!                
      !! ** Purpose :   Control the consistency between namelist options
      !!                for vertical coordinate
      !!----------------------------------------------------------------------
      INTEGER ::   ioptio, ios
      !!
      NAMELIST/nam_vvl/ ln_vvl_zstar, ln_vvl_ztilde, ln_vvl_layer, ln_vvl_ztilde_as_zstar, &
         &              ln_vvl_zstar_at_eqtor      , rn_ahe3     , rn_rst_e3t            , &
         &              rn_lf_cutoff               , rn_zdef_max , ln_vvl_dbg                ! not yet implemented: ln_vvl_kepe
      !!---------------------------------------------------------------------- 
      !
      REWIND( numnam_ref )              ! Namelist nam_vvl in reference namelist : 
      READ  ( numnam_ref, nam_vvl, IOSTAT = ios, ERR = 901)
901   IF( ios /= 0 ) CALL ctl_nam ( ios , 'nam_vvl in reference namelist', lwp )
      !
      REWIND( numnam_cfg )              ! Namelist nam_vvl in configuration namelist : Parameters of the run
      READ  ( numnam_cfg, nam_vvl, IOSTAT = ios, ERR = 902 )
902   IF( ios /= 0 ) CALL ctl_nam ( ios , 'nam_vvl in configuration namelist', lwp )
      IF(lwm) WRITE ( numond, nam_vvl )
      !
      IF(lwp) THEN                    ! Namelist print
         WRITE(numout,*)
         WRITE(numout,*) 'dom_vvl_ctl : choice/control of the variable vertical coordinate'
         WRITE(numout,*) '~~~~~~~~~~~'
         WRITE(numout,*) '           Namelist nam_vvl : chose a vertical coordinate'
         WRITE(numout,*) '              zstar                      ln_vvl_zstar   = ', ln_vvl_zstar
         WRITE(numout,*) '              ztilde                     ln_vvl_ztilde  = ', ln_vvl_ztilde
         WRITE(numout,*) '              layer                      ln_vvl_layer   = ', ln_vvl_layer
         WRITE(numout,*) '              ztilde as zstar   ln_vvl_ztilde_as_zstar  = ', ln_vvl_ztilde_as_zstar
         WRITE(numout,*) '      ztilde near the equator    ln_vvl_zstar_at_eqtor  = ', ln_vvl_zstar_at_eqtor
         ! WRITE(numout,*) '           Namelist nam_vvl : chose kinetic-to-potential energy conservation'
         ! WRITE(numout,*) '                                         ln_vvl_kepe    = ', ln_vvl_kepe
         WRITE(numout,*) '           Namelist nam_vvl : thickness diffusion coefficient'
         WRITE(numout,*) '                                         rn_ahe3        = ', rn_ahe3
         WRITE(numout,*) '           Namelist nam_vvl : maximum e3t deformation fractional change'
         WRITE(numout,*) '                                         rn_zdef_max    = ', rn_zdef_max
         IF( ln_vvl_ztilde_as_zstar ) THEN
            WRITE(numout,*) '           ztilde running in zstar emulation mode; '
            WRITE(numout,*) '           ignoring namelist timescale parameters and using:'
            WRITE(numout,*) '                 hard-wired : z-tilde to zstar restoration timescale (days)'
            WRITE(numout,*) '                                         rn_rst_e3t     =    0.0'
            WRITE(numout,*) '                 hard-wired : z-tilde cutoff frequency of low-pass filter (days)'
            WRITE(numout,*) '                                         rn_lf_cutoff   =    1.0/rdt'
         ELSE
            WRITE(numout,*) '           Namelist nam_vvl : z-tilde to zstar restoration timescale (days)'
            WRITE(numout,*) '                                         rn_rst_e3t     = ', rn_rst_e3t
            WRITE(numout,*) '           Namelist nam_vvl : z-tilde cutoff frequency of low-pass filter (days)'
            WRITE(numout,*) '                                         rn_lf_cutoff   = ', rn_lf_cutoff
         ENDIF
         WRITE(numout,*) '           Namelist nam_vvl : debug prints'
         WRITE(numout,*) '                                         ln_vvl_dbg     = ', ln_vvl_dbg
      ENDIF
      !
      ioptio = 0                      ! Parameter control
      IF( ln_vvl_ztilde_as_zstar )   ln_vvl_ztilde = .true.
      IF( ln_vvl_zstar           )   ioptio = ioptio + 1
      IF( ln_vvl_ztilde          )   ioptio = ioptio + 1
      IF( ln_vvl_layer           )   ioptio = ioptio + 1
      !
      IF( ioptio /= 1 )   CALL ctl_stop( 'Choose ONE vertical coordinate in namelist nam_vvl' )
      !
      IF(lwp) THEN                   ! Print the choice
         WRITE(numout,*)
         IF( ln_vvl_zstar           ) WRITE(numout,*) '              zstar vertical coordinate is used'
         IF( ln_vvl_ztilde          ) WRITE(numout,*) '              ztilde vertical coordinate is used'
         IF( ln_vvl_layer           ) WRITE(numout,*) '              layer vertical coordinate is used'
         IF( ln_vvl_ztilde_as_zstar ) WRITE(numout,*) '              to emulate a zstar coordinate'
         ! - ML - Option not developed yet
         ! IF(       ln_vvl_kepe ) WRITE(numout,*) '              kinetic to potential energy transfer : option used'
         ! IF( .NOT. ln_vvl_kepe ) WRITE(numout,*) '              kinetic to potential energy transfer : option not used'
      ENDIF
      !
      !
   END SUBROUTINE dom_vvl_ctl

   !!======================================================================
END MODULE domvvl
