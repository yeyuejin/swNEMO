!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief This module manage vertical grid.
!>
!> @details
!>    to set the depth of model levels and the resulting vertical scale
!> factors:<br/>
!> @code
!>    CALL vgrid_zgr_z(dd_gdepw(:), dd_gdept(:), dd_e3w(:), dd_e3t(:), 
!>                     dd_ppkth, dd_ppkth2, dd_ppacr, dd_ppacr2, 
!>                     dd_ppdzmin, dd_pphmax, dd_pp_to_be_computed, 
!>                     dd_ppa0, dd_ppa1, dd_ppa2, dd_ppsur)
!> @endcode
!>       - dd_gdepw is array of depth value on W point
!>       - dd_gdept is array of depth value on T point
!>       - dd_e3w   is array of vertical mesh size on W point
!>       - dd_e3t   is array of vertical mesh size on T point
!>       - dd_ppkth              see NEMO documentation
!>       - dd_ppkth2             see NEMO documentation
!>       - dd_ppacr              see NEMO documentation
!>       - dd_ppdzmin            see NEMO documentation
!>       - dd_pphmax             see NEMO documentation
!>       - dd_pp_to_be_computed  see NEMO documentation
!>       - dd_ppa1               see NEMO documentation
!>       - dd_ppa2               see NEMO documentation
!>       - dd_ppa0               see NEMO documentation
!>       - dd_ppsur              see NEMO documentation
!>    
!> 
!>    to set the depth and vertical scale factor in partial step z-coordinate
!>  case:<br/>
!> @code
!>    CALL vgrid_zgr_zps(id_mbathy(:,:), dd_bathy(:,:), id_jpkmax, dd_gdepw(:), 
!>                       dd_e3t(:), dd_e3zps_min, dd_e3zps_rat)
!> @endcode
!>       - id_mbathy is array of bathymetry level
!>       - dd_bathy  is array of bathymetry
!>       - id_jpkmax is the maximum number of level to be used
!>       - dd_gdepw  is array of vertical mesh size on W point
!>       - dd_e3t    is array of vertical mesh size on T point
!>       - dd_e3zps_min    see NEMO documentation
!>       - dd_e3zps_rat    see NEMO documentation
!>
!>    to check the bathymetry in levels:<br/>
!> @code
!>    CALL vgrid_zgr_bat_ctl(id_mbathy, id_jpkmax, id_jpk)
!> @endcode
!>       - id_mbathy is array of bathymetry level
!>       - id_jpkmax is the maximum number of level to be used
!>       - id_jpk    is the number of level
!>   
!>    to compute bathy level in T,U,V,F point from  Bathymetry file:<br/>
!> @code
!>    tl_level(:)=vgrid_get_level(td_bathy, [cd_namelist,] [td_dom,] [id_nlevel])
!> @endcode
!>       - td_bathy is Bathymetry file structure
!>       - cd_namelist is namelist [optional]
!>       - td_dom is domain structure [optional]
!>       - id_nlevel is number of lelvel to be used [optional]
!>    
!> @author
!> J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date Spetember, 2014
!> - add header
!> @date June, 2015 - update subroutine with NEMO 3.6
!>
!> @todo
!> - fusionner vgrid et grid_zgr
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE vgrid

   USE netcdf                          ! nf90 library
   USE kind                            ! F90 kind parameter
   USE fct                             ! basic usefull function
   USE global                          ! global parameter
   USE phycst                          ! physical constant
   USE logger                          ! log file manager
   USE file                            ! file manager
   USE var                             ! variable manager
   USE dim                             ! dimension manager
   USE dom                             ! domain manager
   USE grid                            ! grid manager
   USE iom                             ! I/O manager
   USE mpp                             ! MPP manager
   USE iom_mpp                         ! I/O MPP manager
   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! type and variable

   ! function and subroutine
   PUBLIC :: vgrid_zgr_z 
   PUBLIC :: vgrid_zgr_zps
   PUBLIC :: vgrid_zgr_bat_ctl
   PUBLIC :: vgrid_get_level

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE vgrid_zgr_z(dd_gdepw, dd_gdept, dd_e3w, dd_e3t,          &
         &                dd_e3w_1d, dd_e3t_1d,                        &
         &                dd_ppkth, dd_ppkth2, dd_ppacr, dd_ppacr2,    &
         &                dd_ppdzmin, dd_pphmax, dd_pp_to_be_computed, &
         &                dd_ppa0, dd_ppa1, dd_ppa2, dd_ppsur )
   !-------------------------------------------------------------------
   !> @brief This subroutine set the depth of model levels and the resulting 
   !>      vertical scale factors.
   !>
   !> @details
   !> ** Method  :   z-coordinate system (use in all type of coordinate)
   !>        The depth of model levels is defined from an analytical
   !>      function the derivative of which gives the scale factors.
   !>        both depth and scale factors only depend on k (1d arrays). <br/>
   !>              w-level: gdepw  = fsdep(k)                           <br/>
   !>                       e3w(k) = dk(fsdep)(k)     = fse3(k)         <br/>
   !>              t-level: gdept  = fsdep(k+0.5)                       <br/>
   !>                       e3t(k) = dk(fsdep)(k+0.5) = fse3(k+0.5)     <br/>
   !>
   !> ** Action  : - gdept, gdepw : depth of T- and W-point (m)          <br/>
   !>              -  e3t, e3w    : scale factors at T- and W-levels (m) <br/>
   !>
   !> @author G. Madec
   !> @date Marsh,2008 - F90: Free form and module
   !>
   !> @note Reference : Marti, Madec & Delecluse, 1992, JGR, 97, No8, 12,763-12,766.
   !>
   !> @param[inout] dd_gdepw
   !> @param[inout] dd_gedpt
   !> @param[inout] dd_e3w
   !> @param[inout] dd_e2t
   !> @param[in] dd_ppkth 
   !> @param[in] dd_ppkth2
   !> @param[in] dd_ppacr
   !> @param[in] dd_ppacr2
   !> @param[in] dd_ppdzmin
   !> @param[in] dd_pphmax 
   !> @param[in] dd_pp_to_be_computed
   !> @param[in] dd_ppa1
   !> @param[in] dd_ppa2 
   !> @param[in] dd_ppa0
   !> @param[in] dd_ppsur
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp), DIMENSION(:), INTENT(INOUT) :: dd_gdepw
      REAL(dp), DIMENSION(:), INTENT(INOUT) :: dd_gdept
      REAL(dp), DIMENSION(:), INTENT(INOUT) :: dd_e3w
      REAL(dp), DIMENSION(:), INTENT(INOUT) :: dd_e3t
      REAL(dp), DIMENSION(:), INTENT(INOUT) :: dd_e3w_1d
      REAL(dp), DIMENSION(:), INTENT(INOUT) :: dd_e3t_1d

      REAL(dp)              , INTENT(IN   ) :: dd_ppkth
      REAL(dp)              , INTENT(IN   ) :: dd_ppkth2
      REAL(dp)              , INTENT(IN   ) :: dd_ppacr
      REAL(dp)              , INTENT(IN   ) :: dd_ppacr2

      REAL(dp)              , INTENT(IN   ) :: dd_ppdzmin
      REAL(dp)              , INTENT(IN   ) :: dd_pphmax
      REAL(dp)              , INTENT(IN   ) :: dd_pp_to_be_computed

      REAL(dp)              , INTENT(IN   ) :: dd_ppa0
      REAL(dp)              , INTENT(IN   ) :: dd_ppa1
      REAL(dp)              , INTENT(IN   ) :: dd_ppa2
      REAL(dp)              , INTENT(IN   ) :: dd_ppsur

      ! local variable
      REAL(dp) :: dl_zkth
      REAL(dp) :: dl_zkth2
      REAL(dp) :: dl_zdzmin
      REAL(dp) :: dl_zhmax
      REAL(dp) :: dl_zacr
      REAL(dp) :: dl_zacr2

      REAL(dp) :: dl_ppacr
      REAL(dp) :: dl_ppacr2

      REAL(dp) :: dl_za0
      REAL(dp) :: dl_za1
      REAL(dp) :: dl_za2
      REAL(dp) :: dl_zsur
      REAL(dp) :: dl_zw
      REAL(dp) :: dl_zt

      INTEGER(i4) :: il_jpk

      ! loop indices
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

       dl_ppacr = dd_ppacr
       IF( dd_ppa1 == 0._dp ) dl_ppacr =1.0
       dl_ppacr2= dd_ppacr2
       IF( dd_ppa2 == 0._dp ) dl_ppacr2=1.0

      ! Set variables from parameters
      ! ------------------------------
       dl_zkth   = dd_ppkth       ;   dl_zacr  = dl_ppacr
       dl_zdzmin = dd_ppdzmin     ;   dl_zhmax = dd_pphmax
       dl_zkth2  = dd_ppkth2      ;   dl_zacr2 = dl_ppacr2

       il_jpk = SIZE(dd_gdepw(:))

      ! If ppa1 and ppa0 and ppsur are et to pp_to_be_computed
      !  za0, za1, zsur are computed from ppdzmin , pphmax, ppkth, ppacr
      !
       IF(  dd_ppa1  == dd_pp_to_be_computed  .AND.  &
         &  dd_ppa0  == dd_pp_to_be_computed  .AND.  &
         &  dd_ppsur == dd_pp_to_be_computed           ) THEN
         dl_za1 = ( dl_zdzmin - dl_zhmax / REAL((il_jpk-1),dp) ) &
             &     / ( TANH((1-dl_zkth)/dl_zacr) - dl_zacr/REAL((il_jpk-1),dp) &
             &     * (  LOG( COSH( (REAL(il_jpk,dp) - dl_zkth) / dl_zacr) )    &
             &        - LOG( COSH(( 1.0  - dl_zkth) / dl_zacr) ) )  )

         dl_za0  = dl_zdzmin - dl_za1 * TANH( (1.0-dl_zkth) / dl_zacr )
         dl_zsur = - dl_za0 - dl_za1 * dl_zacr * LOG( COSH( (1-dl_zkth) / dl_zacr )  )

       ELSE
         dl_za1 = dd_ppa1 ;       dl_za0 = dd_ppa0 ;  dl_zsur = dd_ppsur
         dl_za2 = dd_ppa2
       ENDIF

      ! Reference z-coordinate (depth - scale factor at T- and W-points)
      ! ======================
      IF(  dd_ppkth == 0. )THEN            !  uniform vertical grid       

         dl_za1 = dl_zhmax/REAL((il_jpk-1),dp) 
         DO jk = 1, il_jpk
            dl_zw = REAL(jk,dp)
            dl_zt = REAL(jk,dp) + 0.5_dp
            dd_gdepw(jk) = ( dl_zw - 1.0 ) * dl_za1
            dd_gdept(jk) = ( dl_zt - 1.0 ) * dl_za1
            dd_e3w  (jk) =  dl_za1
            dd_e3t  (jk) =  dl_za1
         END DO

      ELSE

         DO jk = 1, il_jpk
            dl_zw = REAL( jk,dp)
            dl_zt = REAL( jk,dp) + 0.5_dp
            dd_gdepw(jk) = ( dl_zsur + dl_za0 * dl_zw + &
            &                dl_za1 * dl_zacr * LOG( COSH( (dl_zw-dl_zkth)/dl_zacr ) ) + &
            &                dl_za2 * dl_zacr2* LOG( COSH( (dl_zw-dl_zkth2)/dl_zacr2 ) ) )
            dd_gdept(jk) = ( dl_zsur + dl_za0 * dl_zt + &
            &                dl_za1 * dl_zacr * LOG( COSH( (dl_zt-dl_zkth)/dl_zacr ) ) + &
            &                dl_za2 * dl_zacr2* LOG( COSH( (dl_zt-dl_zkth2)/dl_zacr2 ) ) )
            dd_e3w  (jk) =             dl_za0 + &
            &                          dl_za1 * TANH(      (dl_zw-dl_zkth)/dl_zacr   ) + &
            &                          dl_za2 * TANH(      (dl_zw-dl_zkth2)/dl_zacr2 )
            dd_e3t  (jk) =             dl_za0 + &
            &                          dl_za1 * TANH(      (dl_zt-dl_zkth)/dl_zacr   ) + &
            &                          dl_za2 * TANH(      (dl_zt-dl_zkth2)/dl_zacr2 )
         END DO
         dd_gdepw(1) = 0.e0   ! force first w-level to be exactly at zero

      ENDIF

   ! need to be like this to compute the pressure gradient with ISF.
   ! If not, level beneath the ISF are not aligned (sum(e3t) /= depth)
   ! define e3t_0 and e3w_0 as the differences between gdept and gdepw respectively
      DO jk = 1, il_jpk-1
         dd_e3t_1d(jk) = dd_gdepw(jk+1)-dd_gdepw(jk) 
      END DO
      dd_e3t_1d(il_jpk) = dd_e3t_1d(il_jpk-1) ! we don't care because this level is masked in NEMO

      DO jk = 2, il_jpk
         dd_e3w_1d(jk) = dd_gdept(jk) - dd_gdept(jk-1) 
      END DO
      dd_e3w_1d(1  ) = 2._dp * (dd_gdept(1) - dd_gdepw(1))

      ! Control and  print
      ! ==================

      DO jk = 1, il_jpk
         IF( dd_e3w(jk)  <= 0. .OR. dd_e3t(jk)  <= 0. )then
            CALL logger_debug("VGRID ZGR Z: e3w or e3t <= 0 ")
         ENDIF   

         IF( dd_e3w_1d(jk)  <= 0. .OR. dd_e3t_1d(jk)  <= 0. )then
            CALL logger_debug("VGRID ZGR Z: e3w_1d or e3t_1d <= 0 ")
         ENDIF

         IF( dd_gdepw(jk) < 0. .OR. dd_gdept(jk) < 0. )then
            CALL logger_debug("VGRID ZGR Z: gdepw or gdept < 0 ")
         ENDIF
      END DO

   END SUBROUTINE vgrid_zgr_z
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE vgrid_zgr_bat(dd_bathy, dd_gdepw, dd_hmin, dd_fill)
   !-------------------------------------------------------------------
   !> @brief This subroutine
   !>
   !> @todo add subroutine description
   !>
   !> @param[inout] dd_bathy
   !> @param[in] dd_gdepw
   !> @param[in] dd_hmin
   !> @param[in] dd_fill
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:,:), INTENT(INOUT) :: dd_bathy 
      REAL(dp), DIMENSION(:)  , INTENT(IN   ) :: dd_gdepw 
      REAL(dp)                , INTENT(IN   ) :: dd_hmin
      REAL(dp)                , INTENT(IN   ), OPTIONAL :: dd_fill

      ! local
      INTEGER(i4) :: il_jpk
      
      REAL(dp)    :: dl_hmin
      REAL(dp)    :: dl_fill

      ! loop indices
      INTEGER(i4) :: jk
      !----------------------------------------------------------------
      il_jpk = SIZE(dd_gdepw(:))

      dl_fill=0._dp
      IF( PRESENT(dd_fill) ) dl_fill=dd_fill

      IF( dd_hmin < 0._dp ) THEN
         jk = - INT( dd_hmin )     ! from a nb of level
      ELSE
         jk = MINLOC( dd_gdepw, mask = dd_gdepw > dd_hmin, dim = 1 )  ! from a depth
      ENDIF
      
      dl_hmin = dd_gdepw(jk+1) ! minimum depth = ik+1 w-levels 
      WHERE( dd_bathy(:,:) <= 0._wp .OR. dd_bathy(:,:) == dl_fill )
         dd_bathy(:,:) = dl_fill                         ! min=0     over the lands
      ELSE WHERE
         dd_bathy(:,:) = MAX(  dl_hmin , dd_bathy(:,:)  )   ! min=dl_hmin over the oceans
      END WHERE
      WRITE(*,*) 'Minimum ocean depth: ', dl_hmin, ' minimum number of ocean levels : ', jk      

   END SUBROUTINE vgrid_zgr_bat
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE vgrid_zgr_zps(id_mbathy, dd_bathy, id_jpkmax, &
         &                  dd_gdepw, dd_e3t,               &
         &                  dd_e3zps_min, dd_e3zps_rat,     &
         &                  dd_fill )
   !-------------------------------------------------------------------
   !> @brief This subroutine set the depth and vertical scale factor in partial step
   !>      z-coordinate case 
   !>
   !> @details
   !> ** Method  :   Partial steps : computes the 3D vertical scale factors
   !>      of T-, U-, V-, W-, UW-, VW and F-points that are associated with
   !>      a partial step representation of bottom topography.
   !>
   !>        The reference depth of model levels is defined from an analytical
   !>      function the derivative of which gives the reference vertical
   !>      scale factors.
   !>      From  depth and scale factors reference, we compute there new value
   !>      with partial steps  on 3d arrays ( i, j, k ).
   !>
   !>      w-level: 
   !>          - gdepw_ps(i,j,k)  = fsdep(k)
   !>          - e3w_ps(i,j,k) = dk(fsdep)(k)     = fse3(i,j,k)
   !>      t-level: 
   !>          - gdept_ps(i,j,k)  = fsdep(k+0.5)
   !>          - e3t_ps(i,j,k) = dk(fsdep)(k+0.5) = fse3(i,j,k+0.5)
   !>
   !>      With the help of the bathymetric file ( bathymetry_depth_ORCA_R2.nc),
   !>      we find the mbathy index of the depth at each grid point.
   !>      This leads us to three cases:
   !>          - bathy = 0 => mbathy = 0
   !>          - 1 < mbathy < jpkm1    
   !>          - bathy > gdepw(jpk) => mbathy = jpkm1  
   !>
   !>      Then, for each case, we find the new depth at t- and w- levels
   !>      and the new vertical scale factors at t-, u-, v-, w-, uw-, vw- 
   !>      and f-points.
   !> 
   !>        This routine is given as an example, it must be modified
   !>      following the user s desiderata. nevertheless, the output as
   !>      well as the way to compute the model levels and scale factors
   !>      must be respected in order to insure second order accuracy
   !>      schemes.
   !>
   !>  @warning 
   !>         - gdept, gdepw and e3 are positives
   !>         - gdept_ps, gdepw_ps and e3_ps are positives
   !>
   !> @author A. Bozec, G. Madec
   !> @date February, 2009 - F90: Free form and module
   !> @date February, 2009 
   !> - A. de Miranda : rigid-lid + islands
   !>
   !> @note Reference : Pacanowsky & Gnanadesikan 1997, Mon. Wea. Rev., 126, 3248-3270.
   !>
   !> @param[inout] id_mbathy
   !> @param[inout] dd_bathy
   !> @param[inout] id_jpkmax 
   !> @param[in] dd_gdepw
   !> @param[in] dd_e3t 
   !> @param[in] dd_e3zps_min
   !> @param[in] dd_e3zps_rat
   !> @param[in] dd_fill
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      INTEGER(i4), DIMENSION(:,:), INTENT(  OUT) :: id_mbathy
      REAL(dp)   , DIMENSION(:,:), INTENT(INOUT) :: dd_bathy
      INTEGER(i4)                , INTENT(INOUT) :: id_jpkmax
      REAL(dp)   , DIMENSION(:)  , INTENT(IN   ) :: dd_gdepw
      REAL(dp)   , DIMENSION(:)  , INTENT(IN   ) :: dd_e3t
      REAL(dp)                   , INTENT(IN   ) :: dd_e3zps_min
      REAL(dp)                   , INTENT(IN   ) :: dd_e3zps_rat
      REAL(dp)                   , INTENT(IN   ), OPTIONAL :: dd_fill

      ! local variable
      REAL(dp) :: dl_zmax     ! Maximum depth
      !REAL(dp) :: dl_zmin     ! Minimum depth
      REAL(dp) :: dl_zdepth   ! Ajusted ocean depth to avoid too small e3t 
      REAL(dp) :: dl_fill     

      INTEGER(i4) :: il_jpk
      INTEGER(i4) :: il_jpkm1
      INTEGER(i4) :: il_jpiglo
      INTEGER(i4) :: il_jpjglo

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      il_jpk=SIZE(dd_gdepw(:))
      il_jpiglo=SIZE(id_mbathy(:,:),DIM=1)
      il_jpjglo=SIZE(id_mbathy(:,:),DIM=2)

      dl_fill=0._dp
      IF( PRESENT(dd_fill) ) dl_fill=dd_fill

      ! Initialization of constant
      dl_zmax = dd_gdepw(il_jpk) + dd_e3t(il_jpk) ! maximum depth (i.e. the last ocean level thickness <= 2*e3t_1d(jpkm1) )

      ! bounded value of bathy (min already set at the end of zgr_bat)
      WHERE( dd_bathy(:,:) /= dl_fill )
         dd_bathy(:,:) = MIN( dl_zmax ,  dd_bathy(:,:) )
      END WHERE

      ! bathymetry in level (from bathy_meter)
      ! ===================
      il_jpkm1=il_jpk-1
      ! initialize mbathy to the maximum ocean level available
      id_mbathy(:,:) = il_jpkm1

      ! storage of land and island's number (zera and negative values) in mbathy
      DO jj = 1, il_jpjglo
         DO ji= 1, il_jpiglo
            IF( dd_bathy(ji,jj) <= 0._dp )THEN
               id_mbathy(ji,jj) = INT(dd_bathy(ji,jj),i4)
            ELSEIF( dd_bathy(ji,jj) == dl_fill )THEN
               id_mbathy(ji,jj) = 0_i4
            ENDIF
         END DO
      END DO

      ! Compute mbathy for ocean points (i.e. the number of ocean levels)
      ! find the number of ocean levels such that the last level thickness
      ! is larger than the minimum of e3zps_min and e3zps_rat * e3t (where
      ! e3t is the reference level thickness

      DO jk = il_jpkm1, 1, -1
         dl_zdepth = dd_gdepw(jk) + MIN( dd_e3zps_min, dd_e3t(jk)*dd_e3zps_rat )

         DO jj = 1, il_jpjglo
            DO ji = 1, il_jpiglo
               IF( dd_bathy(ji,jj) /= dl_fill )THEN
                  IF( 0. < dd_bathy(ji,jj) .AND. &
                  &       dd_bathy(ji,jj) <= dl_zdepth ) id_mbathy(ji,jj) = jk-1
               ENDIF
            END DO
         END DO
      END DO

      ! ================
      ! Bathymetry check
      ! ================

      CALL vgrid_zgr_bat_ctl( id_mbathy, id_jpkmax, il_jpk)

   END SUBROUTINE vgrid_zgr_zps
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE vgrid_zgr_bat_ctl(id_mbathy, id_jpkmax, id_jpk)
   !-------------------------------------------------------------------
   !> @brief This subroutine check the bathymetry in levels 
   !>
   !> @details
   !> ** Method  :   The array mbathy is checked to verified its consistency
   !>      with the model options. in particular:
   !>            mbathy must have at least 1 land grid-points (mbathy<=0)
   !>                  along closed boundary.
   !>            mbathy must be cyclic IF jperio=1.
   !>            mbathy must be lower or equal to jpk-1.
   !>            isolated ocean grid points are suppressed from mbathy
   !>                  since they are only connected to remaining
   !>                  ocean through vertical diffusion.
   !>      C A U T I O N : mbathy will be modified during the initializa-
   !>      tion phase to become the number of non-zero w-levels of a water
   !>      column, with a minimum value of 1.
   !>
   !> ** Action  : - update mbathy: level bathymetry (in level index)
   !>              - update bathy : meter bathymetry (in meters)
   !>
   !> @author G.Madec
   !> @date Marsh, 2008 - Original code
   !>
   !> @param[in] id_mbathy 
   !> @param[in] id_jpkmax
   !> @param[in] id_jpk
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      INTEGER(i4), DIMENSION(:,:), INTENT(INOUT) :: id_mbathy
      INTEGER(i4)                , INTENT(INOUT) :: id_jpkmax
      INTEGER(i4)                , INTENT(INOUT) :: id_jpk

      ! local variable
      INTEGER(i4) :: il_jpiglo
      INTEGER(i4) :: il_jpjglo

      INTEGER(i4) :: il_icompt
      INTEGER(i4) :: il_ibtest
      INTEGER(i4) :: il_ikmax
      INTEGER(i4) :: il_jpkm1

      INTEGER(i4) :: il_jim
      INTEGER(i4) :: il_jip
      INTEGER(i4) :: il_jjm
      INTEGER(i4) :: il_jjp

      ! loop indices
      INTEGER(i4) :: jl
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      il_jpiglo=SIZE(id_mbathy(:,:),DIM=1)
      il_jpjglo=SIZE(id_mbathy(:,:),DIM=2)

      ! ================
      ! Bathymetry check
      ! ================

      ! suppress isolated ocean grid points'
      il_icompt = 0

      DO jl = 1, 2
         DO jj = 1, il_jpjglo
            DO ji = 1, il_jpiglo
               il_jim=max(ji-1,1) ; il_jip=min(ji+1,il_jpiglo)
               il_jjm=max(jj-1,1) ; il_jjp=min(jj+1,il_jpjglo)

               if(il_jim==ji) il_jim=il_jip ; if(il_jip==ji) il_jip=il_jim
               if(il_jjm==jj) il_jjm=il_jjp ; if(il_jjp==jj) il_jjp=il_jjm

               il_ibtest = MAX( id_mbathy(il_jim,jj), id_mbathy(il_jip,jj),  &
                                id_mbathy(ji,il_jjm),id_mbathy(ji,il_jjp) )

               IF( il_ibtest < id_mbathy(ji,jj) ) THEN
                  id_mbathy(ji,jj) = il_ibtest
                  il_icompt = il_icompt + 1
               ENDIF
            END DO
         END DO

      END DO
      IF( il_icompt == 0 ) THEN
         CALL logger_info("VGRID ZGR BAT CTL:  no isolated ocean grid points")
      ELSE
         CALL logger_info("VGRID ZGR BAT CTL:"//TRIM(fct_str(il_icompt))//&
         &              " ocean grid points suppressed")
      ENDIF

      id_mbathy(:,:) = MAX( 0, id_mbathy(:,:))

      ! Number of ocean level inferior or equal to jpkm1

      il_ikmax = 0
      DO jj = 1, il_jpjglo
         DO ji = 1, il_jpiglo
            il_ikmax = MAX( il_ikmax, id_mbathy(ji,jj) )
         END DO
      END DO

      id_jpkmax=id_jpk

      il_jpkm1=id_jpk-1
      IF( il_ikmax > il_jpkm1 ) THEN
         CALL logger_error("VGRID ZGR BAT CTL: maximum number of ocean level = "//&
         &              TRIM(fct_str(il_ikmax))//" >  jpk-1."//&
         &              " Change jpk to "//TRIM(fct_str(il_ikmax+1))//&
         &              " to use the exact ead bathymetry" )
      ELSE IF( il_ikmax < il_jpkm1 ) THEN
         id_jpkmax=il_ikmax+1
      ENDIF

   END SUBROUTINE vgrid_zgr_bat_ctl
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION vgrid_get_level(td_bathy, cd_namelist, td_dom, id_nlevel) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function compute bathy level in T,U,V,F point, and return 
   !> them as array of variable structure
   !>
   !> @details
   !> Bathymetry is read on Bathymetry file, then bathy level is computed 
   !> on T point, and finally fit to U,V,F point.
   !>
   !> you could specify :<br/>
   !> - namelist where find parameter to set the depth of model levels
   !> (default use GLORYS 75 levels parameters)
   !> - domain structure to specify on e area to work on
   !> - number of level to be used
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_bathy     Bathymetry file structure 
   !> @param[in] cd_namelist  namelist 
   !> @param[in] td_dom       domain structure
   !> @param[in] id_nlevel    number of lelvel to be used 
   !> @return array of level on T,U,V,F point (variable structure)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP)      , INTENT(IN) :: td_bathy
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: cd_namelist
      TYPE(TDOM)      , INTENT(IN), OPTIONAL :: td_dom
      INTEGER(i4)     , INTENT(IN), OPTIONAL :: id_nlevel

      ! function
      TYPE(TVAR), DIMENSION(ip_npoint)       :: tf_var

      ! local variable
      REAL(dp)   , DIMENSION(:)      , ALLOCATABLE :: dl_gdepw 
      REAL(dp)   , DIMENSION(:)      , ALLOCATABLE :: dl_gdept 
      REAL(dp)   , DIMENSION(:)      , ALLOCATABLE :: dl_e3w 
      REAL(dp)   , DIMENSION(:)      , ALLOCATABLE :: dl_e3t
      REAL(dp)   , DIMENSION(:)      , ALLOCATABLE :: dl_e3w_1d 
      REAL(dp)   , DIMENSION(:)      , ALLOCATABLE :: dl_e3t_1d

      INTEGER(i4)                                  :: il_status
      INTEGER(i4)                                  :: il_fileid
      INTEGER(i4)                                  :: il_jpkmax
      INTEGER(i4), DIMENSION(2,2)                  :: il_xghost
      INTEGER(i4), DIMENSION(:,:)    , ALLOCATABLE :: il_mbathy
      INTEGER(i4), DIMENSION(:,:,:,:), ALLOCATABLE :: il_level
      
      LOGICAL                                      :: ll_exist

      TYPE(TDIM) , DIMENSION(ip_maxdim)            :: tl_dim

      TYPE(TDOM)                                   :: tl_dom

      TYPE(TVAR)                                   :: tl_var

      TYPE(TMPP)                                   :: tl_bathy

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj

      INTEGER(i4) :: jip
      INTEGER(i4) :: jjp

      !namelist (intialise with GLORYS 75 levels parameters)
      REAL(dp)                                :: dn_pp_to_be_computed = 0._dp
      REAL(dp)                                :: dn_ppsur     = -3958.951371276829_dp
      REAL(dp)                                :: dn_ppa0      =   103.9530096000000_dp
      REAL(dp)                                :: dn_ppa1      =     2.4159512690000_dp
      REAL(dp)                                :: dn_ppa2      =   100.7609285000000_dp
      REAL(dp)                                :: dn_ppkth     =    15.3510137000000_dp
      REAL(dp)                                :: dn_ppkth2    =    48.0298937200000_dp
      REAL(dp)                                :: dn_ppacr     =     7.0000000000000_dp
      REAL(dp)                                :: dn_ppacr2    =    13.000000000000_dp
      REAL(dp)                                :: dn_ppdzmin   = 6._dp
      REAL(dp)                                :: dn_pphmax    = 5750._dp
      INTEGER(i4)                             :: in_nlevel    = 75

      REAL(dp)                                :: dn_e3zps_min = 25._dp
      REAL(dp)                                :: dn_e3zps_rat = 0.2_dp
      !----------------------------------------------------------------
      NAMELIST /namzgr/ &
      &  dn_pp_to_be_computed, &
      &  dn_ppsur,     &
      &  dn_ppa0,      &
      &  dn_ppa1,      &
      &  dn_ppa2,      &
      &  dn_ppkth,     &
      &  dn_ppkth2,    &
      &  dn_ppacr,     &
      &  dn_ppacr2,    &
      &  dn_ppdzmin,   &
      &  dn_pphmax,    &
      &  in_nlevel

      NAMELIST /namzps/ &
      &  dn_e3zps_min, &
      &  dn_e3zps_rat
      !----------------------------------------------------------------

      IF( PRESENT(cd_namelist) )THEN
         !1- read namelist
         INQUIRE(FILE=TRIM(cd_namelist), EXIST=ll_exist)
         IF( ll_exist )THEN
 
            il_fileid=fct_getunit()

            OPEN( il_fileid, FILE=TRIM(cd_namelist), &
            &                FORM='FORMATTED',       &
            &                ACCESS='SEQUENTIAL',    &
            &                STATUS='OLD',           &
            &                ACTION='READ',          &
            &                IOSTAT=il_status)
            CALL fct_err(il_status)
            IF( il_status /= 0 )THEN
               CALL logger_fatal("VGRID GET LEVEL: ERROR opening "//&
               &  TRIM(cd_namelist))
            ENDIF

            READ( il_fileid, NML = namzgr )
            READ( il_fileid, NML = namzps )

            CLOSE( il_fileid, IOSTAT=il_status )
            CALL fct_err(il_status)
            IF( il_status /= 0 )THEN
               CALL logger_error("VGRID GET LEVELL: ERROR closing "//&
               &  TRIM(cd_namelist))
            ENDIF

         ELSE

            CALL logger_fatal("VGRID GET LEVEL: ERROR. can not find "//&
            &  TRIM(cd_namelist))

         ENDIF
      ENDIF

      ! copy structure
      tl_bathy=mpp_copy(td_bathy)

      ! get domain
      IF( PRESENT(td_dom) )THEN
         tl_dom=dom_copy(td_dom)
      ELSE
         CALL logger_debug("VGRID GET LEVEL: get dom from "//&
         &  TRIM(tl_bathy%c_name))
         tl_dom=dom_init(tl_bathy)
      ENDIF

      ! get ghost cell
      il_xghost(:,:)=grid_get_ghost(tl_bathy)

      ! open mpp files
      CALL iom_dom_open(tl_bathy, tl_dom)

      ! check namelist
      IF( PRESENT(id_nlevel) ) in_nlevel=id_nlevel
      IF( in_nlevel == 0 )THEN
         CALL logger_fatal("VGRID GET LEVEL: number of level to be used "//&
         &                 "is not specify. check namelist.")
      ENDIF

      ! read bathymetry
      tl_var=iom_dom_read_var(tl_bathy,'bathymetry',tl_dom)
      ! clean
      CALL dom_clean(tl_dom)

      ! remove ghost cell
      CALL grid_del_ghost(tl_var, il_xghost(:,:))

      ! force _FillValue (land) to be 0
      WHERE( tl_var%d_value(:,:,1,1) == tl_var%d_fill )
         tl_var%d_value(:,:,1,1)=0
      END WHERE

      ! clean
      CALL iom_dom_close(tl_bathy)
      CALL mpp_clean(tl_bathy)

      ! compute vertical grid
      ALLOCATE( dl_gdepw(in_nlevel), dl_gdept(in_nlevel) ) 
      ALLOCATE(   dl_e3w(in_nlevel),   dl_e3t(in_nlevel) ) 
      ALLOCATE(   dl_e3w_1d(in_nlevel),   dl_e3t_1d(in_nlevel) ) 
      CALL vgrid_zgr_z( dl_gdepw(:), dl_gdept(:), dl_e3w(:), dl_e3t(:), &
      &                 dl_e3w_1d, dl_e3t_1d, &
      &                 dn_ppkth, dn_ppkth2, dn_ppacr, dn_ppacr2,       &
      &                 dn_ppdzmin, dn_pphmax, dn_pp_to_be_computed,    &
      &                 dn_ppa0, dn_ppa1, dn_ppa2, dn_ppsur )

      ! compute bathy level on T point
      ALLOCATE( il_mbathy(tl_var%t_dim(1)%i_len, &
      &                   tl_var%t_dim(2)%i_len ) )
      CALL vgrid_zgr_zps( il_mbathy(:,:), tl_var%d_value(:,:,1,1), il_jpkmax, &
      &                   dl_gdepw(:), dl_e3t(:),               &
      &                   dn_e3zps_min, dn_e3zps_rat )

      DEALLOCATE( dl_gdepw, dl_gdept ) 
      DEALLOCATE(   dl_e3w,   dl_e3t )

      ! compute bathy level in T,U,V,F point
      ALLOCATE( il_level(tl_var%t_dim(1)%i_len, &
      &                  tl_var%t_dim(2)%i_len, &
      &                  ip_npoint,1) )

      DO jj=1,tl_var%t_dim(2)%i_len
         DO ji= 1,tl_var%t_dim(1)%i_len

         jip=MIN(ji+1,tl_var%t_dim(1)%i_len)
         jjp=MIN(jj+1,tl_var%t_dim(2)%i_len)

         ! T point
         il_level(ji,jj,jp_T,1)=il_mbathy(ji,jj)
         ! U point
         il_level(ji,jj,jp_U,1)=MIN( il_mbathy(ji, jj ), il_mbathy(jip, jj ))
         ! V point
         il_level(ji,jj,jp_V,1)=MIN( il_mbathy(ji, jj ), il_mbathy(ji , jjp))
         ! F point
         il_level(ji,jj,jp_F,1)=MIN( il_mbathy(ji, jj ), il_mbathy(jip, jj ), &
         &                           il_mbathy(ji, jjp), il_mbathy(jip, jjp))

         ENDDO
      ENDDO

      DEALLOCATE( il_mbathy )

      tl_dim(:)=dim_copy(tl_var%t_dim(:))
      ! clean
      CALL var_clean(tl_var)

      ! only 2 first dimension to be used
      tl_dim(3:4)%l_use=.FALSE.

      tf_var(jp_T)=var_init('tlevel', il_level(:,:,jp_T:jp_T,:), td_dim=tl_dim(:))
      tf_var(jp_U)=var_init('ulevel', il_level(:,:,jp_U:jp_U,:), td_dim=tl_dim(:))
      tf_var(jp_V)=var_init('vlevel', il_level(:,:,jp_V:jp_V,:), td_dim=tl_dim(:))
      tf_var(jp_F)=var_init('flevel', il_level(:,:,jp_F:jp_F,:), td_dim=tl_dim(:))

      DEALLOCATE( il_level )

      CALL grid_add_ghost( tf_var(jp_T), il_xghost(:,:) )
      CALL grid_add_ghost( tf_var(jp_U), il_xghost(:,:) )
      CALL grid_add_ghost( tf_var(jp_V), il_xghost(:,:) )
      CALL grid_add_ghost( tf_var(jp_F), il_xghost(:,:) )

      ! clean
      CALL dim_clean(tl_dim(:))

   END FUNCTION vgrid_get_level
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE vgrid

