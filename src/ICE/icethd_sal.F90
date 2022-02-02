MODULE icethd_sal
   !!======================================================================
   !!                       ***  MODULE icethd_sal ***
   !!   sea-ice : computation of salinity variations in the ice
   !!======================================================================
   !! History :   -   !  2003-05  (M. Vancoppenolle) original code 1-D
   !!            3.0  !  2005-12  (M. Vancoppenolle) adapted to the 3-D version
   !!            4.0  !  2018     (many people)      SI3 [aka Sea Ice cube]
   !!---------------------------------------------------------------------
#if defined key_si3
   !!----------------------------------------------------------------------
   !!   'key_si3'                                       SI3 sea-ice model
   !!----------------------------------------------------------------------
   !!   ice_thd_sal      : salinity variations in the ice
   !!   ice_thd_sal_init : initialization
   !!----------------------------------------------------------------------
   USE dom_oce        ! ocean space and time domain
   USE phycst         ! physical constants
   USE ice            ! sea-ice: variables
   USE ice1D          ! sea-ice: thermodynamics variables
   USE icevar         ! sea-ice: operations
   !
   USE in_out_manager ! I/O manager
   USE lib_mpp        ! MPP library
   USE lib_fortran    ! fortran utilities (glob_sum + no signed zero)

   IMPLICIT NONE
   PRIVATE

   PUBLIC   ice_thd_sal        ! called by icethd
   PUBLIC   ice_thd_sal_init   ! called by ice_init
   
   ! ** namelist (namthd_sal) **
   REAL(wp) ::   rn_sal_gd     ! restoring salinity for gravity drainage [PSU]
   REAL(wp) ::   rn_time_gd    ! restoring time constant for gravity drainage (= 20 days) [s]
   REAL(wp) ::   rn_sal_fl     ! restoring salinity for flushing [PSU]
   REAL(wp) ::   rn_time_fl    ! restoring time constant for gravity drainage (= 10 days) [s]

   !!----------------------------------------------------------------------
   !! NEMO/ICE 4.0 , NEMO Consortium (2018)
   !! $Id: icethd_sal.F90 11536 2019-09-11 13:54:18Z smasson $
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE ice_thd_sal( ld_sal )
      !!-------------------------------------------------------------------
      !!                ***  ROUTINE ice_thd_sal  ***    
      !!   
      !! ** Purpose :   computes new salinities in the ice
      !!
      !! ** Method  :  3 possibilities
      !!               -> nn_icesal = 1 -> Sice = cst    [ice salinity constant in both time & space] 
      !!               -> nn_icesal = 2 -> Sice = S(z,t) [Vancoppenolle et al. 2005]
      !!               -> nn_icesal = 3 -> Sice = S(z)   [multiyear ice]
      !!---------------------------------------------------------------------
      LOGICAL, INTENT(in) ::   ld_sal            ! gravity drainage and flushing or not 
      !
      INTEGER  ::   ji, jk                       ! dummy loop indices 
      REAL(wp) ::   iflush, igravdr              ! local scalars
      REAL(wp) ::   zs_sni, zs_i_gd, zs_i_fl, zs_i_si, zs_i_bg   ! local scalars
      REAL(wp) ::   z1_time_gd, z1_time_fl
      !!---------------------------------------------------------------------

      SELECT CASE ( nn_icesal )
      !
      !               !---------------------------------------------!
      CASE( 2 )       !  time varying salinity with linear profile  !
         !            !---------------------------------------------!
         z1_time_gd = 1._wp / rn_time_gd * rdt_ice
         z1_time_fl = 1._wp / rn_time_fl * rdt_ice
         !
         DO ji = 1, npti
            !
            !---------------------------------------------------------
            !  Update ice salinity from snow-ice and bottom growth
            !---------------------------------------------------------
            IF( h_i_1d(ji) > 0._wp ) THEN
               zs_sni  = sss_1d(ji) * ( rhoi - rhos ) * r1_rhoi                     ! Salinity of snow ice
               zs_i_si = ( zs_sni      - s_i_1d(ji) ) * dh_snowice(ji) / h_i_1d(ji) ! snow-ice    
               zs_i_bg = ( s_i_new(ji) - s_i_1d(ji) ) * dh_i_bog  (ji) / h_i_1d(ji) ! bottom growth
               ! Update salinity (nb: salt flux already included in icethd_dh)
               s_i_1d(ji) = s_i_1d(ji) + zs_i_bg + zs_i_si
            ENDIF
            !
            IF( ld_sal ) THEN
               !---------------------------------------------------------
               !  Update ice salinity from brine drainage and flushing
               !---------------------------------------------------------
               iflush   = MAX( 0._wp , SIGN( 1._wp , t_su_1d(ji) - rt0         ) )  ! =1 if summer 
               igravdr  = MAX( 0._wp , SIGN( 1._wp , t_bo_1d(ji) - t_su_1d(ji) ) )  ! =1 if t_su < t_bo

               zs_i_gd = - igravdr * MAX( s_i_1d(ji) - rn_sal_gd , 0._wp ) * z1_time_gd  ! gravity drainage 
               zs_i_fl = - iflush  * MAX( s_i_1d(ji) - rn_sal_fl , 0._wp ) * z1_time_fl  ! flushing
               
               ! Update salinity   
               s_i_1d(ji) = s_i_1d(ji) + zs_i_fl + zs_i_gd
               
               ! Salt flux
               sfx_bri_1d(ji) = sfx_bri_1d(ji) - rhoi * a_i_1d(ji) * h_i_1d(ji) * ( zs_i_fl + zs_i_gd ) * r1_rdtice
            ENDIF
         END DO
         !
         ! Salinity profile
         CALL ice_var_salprof1d
         !
         !             !----------------------------------------!
      CASE( 3 )        ! constant salinity with a fixed profile ! (Schwarzacher (1959) multiyear salinity profile (mean = 2.30)
         !             !----------------------------------------!
         CALL ice_var_salprof1d
         !
      END SELECT
      !
   END SUBROUTINE ice_thd_sal


   SUBROUTINE ice_thd_sal_init
      !!-------------------------------------------------------------------
      !!                  ***  ROUTINE ice_thd_sal_init  ***
      !!
      !! ** Purpose :   initialization of ice salinity parameters
      !!
      !! ** Method  :   Read the namthd_sal namelist and check the parameter
      !!                values called at the first timestep (nit000)
      !!
      !! ** input   :   Namelist namthd_sal
      !!-------------------------------------------------------------------
      INTEGER  ::   ios   ! Local integer
      !!
      NAMELIST/namthd_sal/ nn_icesal, rn_icesal, rn_sal_gd, rn_time_gd,   &
         &                 rn_sal_fl, rn_time_fl, rn_simax , rn_simin 
      !!-------------------------------------------------------------------
      !
      REWIND( numnam_ice_ref )              ! Namelist namthd_sal in reference namelist : Ice salinity
      READ  ( numnam_ice_ref, namthd_sal, IOSTAT = ios, ERR = 901)
901   IF( ios /= 0 )   CALL ctl_nam ( ios , 'namthd_sal in reference namelist' )
      REWIND( numnam_ice_cfg )              ! Namelist namthd_sal in configuration namelist : Ice salinity
      READ  ( numnam_ice_cfg, namthd_sal, IOSTAT = ios, ERR = 902 )
902   IF( ios >  0 )   CALL ctl_nam ( ios , 'namthd_sal in configuration namelist' )
      IF(lwm) WRITE ( numoni, namthd_sal )
      !
      IF(lwp) THEN                           ! control print
         WRITE(numout,*)
         WRITE(numout,*) 'ice_thd_sal_init : Ice parameters for salinity '
         WRITE(numout,*) '~~~~~~~~~~~~~~~~'
         WRITE(numout,*) '   Namelist namthd_sal:'
         WRITE(numout,*) '      switch for salinity                                     nn_icesal  = ', nn_icesal
         WRITE(numout,*) '      bulk salinity value if nn_icesal = 1                    rn_icesal  = ', rn_icesal
         WRITE(numout,*) '      restoring salinity for gravity drainage                 rn_sal_gd  = ', rn_sal_gd
         WRITE(numout,*) '      restoring time for for gravity drainage                 rn_time_gd = ', rn_time_gd
         WRITE(numout,*) '      restoring salinity for flushing                         rn_sal_fl  = ', rn_sal_fl
         WRITE(numout,*) '      restoring time for flushing                             rn_time_fl = ', rn_time_fl
         WRITE(numout,*) '      Maximum tolerated ice salinity                          rn_simax   = ', rn_simax
         WRITE(numout,*) '      Minimum tolerated ice salinity                          rn_simin   = ', rn_simin
      ENDIF
      !
   END SUBROUTINE ice_thd_sal_init

#else
   !!----------------------------------------------------------------------
   !!   Default option         Dummy Module           No SI3 sea-ice model
   !!----------------------------------------------------------------------
#endif

   !!======================================================================
END MODULE icethd_sal
