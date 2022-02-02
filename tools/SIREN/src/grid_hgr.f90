!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief This module manage Horizontal grid.
!>
!> @details
!> ** Purpose :   Compute the geographical position (in degre) of the 
!>      model grid-points,  the horizontal scale factors (in meters) and 
!>      the Coriolis factor (in s-1).
!>
!> ** Method  :   The geographical position of the model grid-points is
!>    defined from analytical functions, fslam and fsphi, the derivatives of which gives the horizontal scale factors e1,e2.
!>    Defining two function fslam and fsphi and their derivatives in the two horizontal directions (fse1 and fse2), 
!>    the model grid-point position and scale factors are given by:
!>    - t-point:
!>       - glamt(i,j) = fslam(i    ,j    )   e1t(i,j) = fse1(i    ,j    )
!>       - gphit(i,j) = fsphi(i    ,j    )   e2t(i,j) = fse2(i    ,j    )
!>    - u-point:
!>       - glamu(i,j) = fslam(i+1/2,j    )   e1u(i,j) = fse1(i+1/2,j    )
!>       - gphiu(i,j) = fsphi(i+1/2,j    )   e2u(i,j) = fse2(i+1/2,j    )
!>    - v-point:
!>       - glamv(i,j) = fslam(i    ,j+1/2)   e1v(i,j) = fse1(i    ,j+1/2)
!>       - gphiv(i,j) = fsphi(i    ,j+1/2)   e2v(i,j) = fse2(i    ,j+1/2)
!>    - f-point:
!>       - glamf(i,j) = fslam(i+1/2,j+1/2)   e1f(i,j) = fse1(i+1/2,j+1/2)
!>       - gphif(i,j) = fsphi(i+1/2,j+1/2)   e2f(i,j) = fse2(i+1/2,j+1/2)
!>
!>    Where fse1 and fse2 are defined by:
!>       - fse1(i,j) = ra * rad * SQRT( (cos(phi) di(fslam))**2
!>                                     +          di(fsphi) **2 )(i,j)
!>       - fse2(i,j) = ra * rad * SQRT( (cos(phi) dj(fslam))**2
!>                                     +          dj(fsphi) **2 )(i,j)
!>
!>    The coriolis factor is given at z-point by:
!>        - ff = 2.*omega*sin(gphif)      (in s-1)<br/>
!>
!>        This routine is given as an example, it must be modified
!>      following the user s desiderata. nevertheless, the output as
!>      well as the way to compute the model grid-point position and
!>      horizontal scale factors must be respected in order to insure
!>      second order accuracy schemes.
!>
!> @note If the domain is periodic, verify that scale factors are also
!>      periodic, and the coriolis term again.
!>
!> ** Action  : 
!>    - define  glamt, glamu, glamv, glamf: longitude of t-, u-, v- and f-points (in degre)
!>    - define  gphit, gphiu, gphiv, gphit: latitude  of t-, u-, v-  and f-points (in degre)
!>    - define e1t, e2t, e1u, e2u, e1v, e2v, e1f, e2f: horizontal
!>    - scale factors (in meters) at t-, u-, v-, and f-points.
!>    - define ff: coriolis factor at f-point
!>
!> References :   Marti, Madec and Delecluse, 1992, JGR
!>                Madec, Imbard, 1996, Clim. Dyn.
!>
!> @author
!> G, Madec
!>
!> @date March, 1988 - Original code
!> @date January, 1996 
!> - terrain following coordinates
!> @date February, 1997 
!> - print mesh informations
!> @date November, 1999
!> - M. Imbard : NetCDF format with IO-IPSL
!> @date Augustr, 2000
!> - D. Ludicone : Reduced section at Bab el Mandeb
!> @date September, 2001
!> - M. Levy : eel config: grid in km, beta-plane
!> @date August, 2002
!> - G. Madec :  F90: Free form and module, namelist
!> @date January, 2004
!> - A.M. Treguier, J.M. Molines : Case 4 (Mercator mesh)
!> use of parameters in par_CONFIG-Rxx.h90, not in namelist
!> @date May, 2004
!> - A. Koch-Larrouy : Add Gyre configuration 
!> @date February, 2011
!> - G. Madec : add cell surface (e1e2t)
!> @date September, 2015 
!> - J, Paul : rewrite to SIREN format from $Id: domhgr.F90 5506 2015-06-29 15:19:38Z clevy $
!> @date October, 2016
!> - J, Paul : update from trunk (revision 6961): add wetting and drying, ice sheet coupling..
!> - J, Paul : compute coriolis factor at f-point and at t-point 
!> - J, Paul : do not use anymore special case for ORCA grid
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE grid_hgr

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
   USE lbc                             ! lateral boundary conditions

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! type and variable
   PUBLIC :: TNAMH

   PUBLIC :: tg_tmask
   PUBLIC :: tg_umask
   PUBLIC :: tg_vmask
   PUBLIC :: tg_fmask

!   PUBLIC :: tg_wmask
!   PUBLIC :: tg_wumask
!   PUBLIC :: tg_wvmask

   PUBLIC :: tg_ssmask
!   PUBLIC :: tg_ssumask
!   PUBLIC :: tg_ssvmask
!   PUBLIC :: tg_ssfmask

   PUBLIC :: tg_glamt
   PUBLIC :: tg_glamu
   PUBLIC :: tg_glamv
   PUBLIC :: tg_glamf

   PUBLIC :: tg_gphit
   PUBLIC :: tg_gphiu
   PUBLIC :: tg_gphiv
   PUBLIC :: tg_gphif
   
   PUBLIC :: tg_e1t
   PUBLIC :: tg_e1u
   PUBLIC :: tg_e1v
   PUBLIC :: tg_e1f
   
   PUBLIC :: tg_e2t
   PUBLIC :: tg_e2u
   PUBLIC :: tg_e2v
   PUBLIC :: tg_e2f
   
   PUBLIC :: tg_ff_t
   PUBLIC :: tg_ff_f

   PUBLIC :: tg_gcost
   PUBLIC :: tg_gcosu
   PUBLIC :: tg_gcosv
   PUBLIC :: tg_gcosf

   PUBLIC :: tg_gsint
   PUBLIC :: tg_gsinu
   PUBLIC :: tg_gsinv
   PUBLIC :: tg_gsinf

   ! function and subroutine
   PUBLIC :: grid_hgr_init 
   PUBLIC :: grid_hgr_fill
   PUBLIC :: grid_hgr_clean
   PUBLIC :: grid_hgr_nam 

   PRIVATE :: grid_hgr__fill_curv
   PRIVATE :: grid_hgr__fill_reg
   PRIVATE :: grid_hgr__fill_plan
   PRIVATE :: grid_hgr__fill_merc
   PRIVATE :: grid_hgr__fill_gyre
   PRIVATE :: grid_hgr__fill_coriolis
   PRIVATE :: grid_hgr__angle

   TYPE TNAMH

      CHARACTER(LEN=lc) :: c_coord   
      INTEGER(i4)       :: i_perio   
                
      INTEGER(i4)       :: i_mshhgr  
      REAL(dp)          :: d_ppglam0 
      REAL(dp)          :: d_ppgphi0 
                
      REAL(dp)          :: d_ppe1_deg
      REAL(dp)          :: d_ppe2_deg
!      REAL(dp)          :: d_ppe1_m  
!      REAL(dp)          :: d_ppe2_m     

!      INTEGER(i4)       :: i_cla     
                
!      CHARACTER(LEN=lc) :: c_cfg     
      INTEGER(i4)       :: i_cfg     
      LOGICAL           :: l_bench   
                
   END TYPE

   TYPE(TVAR), SAVE :: tg_tmask   
   TYPE(TVAR), SAVE :: tg_umask
   TYPE(TVAR), SAVE :: tg_vmask
   TYPE(TVAR), SAVE :: tg_fmask
!   TYPE(TVAR), SAVE :: tg_wmask
!   TYPE(TVAR), SAVE :: tg_wumask
!   TYPE(TVAR), SAVE :: tg_wvmask

   TYPE(TVAR), SAVE :: tg_ssmask 
!   TYPE(TVAR), SAVE :: tg_ssumask
!   TYPE(TVAR), SAVE :: tg_ssvmask
!   TYPE(TVAR), SAVE :: tg_ssfmask

   TYPE(TVAR), SAVE :: tg_glamt
   TYPE(TVAR), SAVE :: tg_glamu
   TYPE(TVAR), SAVE :: tg_glamv
   TYPE(TVAR), SAVE :: tg_glamf

   TYPE(TVAR), SAVE :: tg_gphit
   TYPE(TVAR), SAVE :: tg_gphiu
   TYPE(TVAR), SAVE :: tg_gphiv
   TYPE(TVAR), SAVE :: tg_gphif

   TYPE(TVAR), SAVE :: tg_e1t
   TYPE(TVAR), SAVE :: tg_e1u
   TYPE(TVAR), SAVE :: tg_e1v
   TYPE(TVAR), SAVE :: tg_e1f

   TYPE(TVAR), SAVE :: tg_e2t
   TYPE(TVAR), SAVE :: tg_e2u
   TYPE(TVAR), SAVE :: tg_e2v
   TYPE(TVAR), SAVE :: tg_e2f

   TYPE(TVAR), SAVE :: tg_ff_t
   TYPE(TVAR), SAVE :: tg_ff_f

   TYPE(TVAR), SAVE :: tg_gcost
   TYPE(TVAR), SAVE :: tg_gcosu
   TYPE(TVAR), SAVE :: tg_gcosv
   TYPE(TVAR), SAVE :: tg_gcosf

   TYPE(TVAR), SAVE :: tg_gsint
   TYPE(TVAR), SAVE :: tg_gsinu
   TYPE(TVAR), SAVE :: tg_gsinv
   TYPE(TVAR), SAVE :: tg_gsinf

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_hgr_init(jpi, jpj, jpk, ld_domcfg) 
   !-------------------------------------------------------------------
   !> @brief This subroutine initialise hgr structure
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial version
   !>
   !> @param[in] jpi
   !> @param[in] jpj
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      INTEGER(i4), INTENT(IN) :: jpi
      INTEGER(i4), INTENT(IN) :: jpj
      INTEGER(i4), INTENT(IN) :: jpk
      LOGICAL    , INTENT(IN) :: ld_domcfg

      REAL(dp), DIMENSION(jpi,jpj)     :: dl_tmp2D
      REAL(dp), DIMENSION(jpi,jpj,jpk) :: dl_tmp3D
      ! loop indices
      !----------------------------------------------------------------
      ! variable 2D
      dl_tmp2D(:,:)  =dp_fill_i1

      tg_ssmask   = var_init('ssmask' ,dl_tmp2D(:,:)  , dd_fill=dp_fill_i1, id_type=NF90_BYTE)
!      tg_ssumask  = var_init('ssumask',dl_tmp2D(:,:)  , dd_fill=dp_fill_i1, id_type=NF90_BYTE)
!      tg_ssvmask  = var_init('ssvmask',dl_tmp2D(:,:)  , dd_fill=dp_fill_i1, id_type=NF90_BYTE)
!      tg_ssfmask  = var_init('ssfmask',dl_tmp2D(:,:)  , dd_fill=dp_fill_i1, id_type=NF90_BYTE)

      dl_tmp2D(:,:)=dp_fill

      tg_glamt    = var_init('glamt',dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_glamu    = var_init('glamu',dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_glamv    = var_init('glamv',dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_glamf    = var_init('glamf',dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)

      tg_gphit    = var_init('gphit',dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_gphiu    = var_init('gphiu',dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_gphiv    = var_init('gphiv',dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_gphif    = var_init('gphif',dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)

      tg_e1t      = var_init('e1t'  ,dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_e1u      = var_init('e1u'  ,dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_e1v      = var_init('e1v'  ,dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_e1f      = var_init('e1f'  ,dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)

      tg_e2t      = var_init('e2t'  ,dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_e2u      = var_init('e2u'  ,dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_e2v      = var_init('e2v'  ,dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_e2f      = var_init('e2f'  ,dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)

      tg_ff_t     = var_init('ff_t' ,dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_ff_f     = var_init('ff_f' ,dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)

      IF( .NOT. ld_domcfg )THEN
         tg_gcost     =var_init('gcost',dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
         tg_gcosu     =var_init('gcosu',dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
         tg_gcosv     =var_init('gcosv',dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
         tg_gcosf     =var_init('gcosf',dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)

         tg_gsint     =var_init('gsint',dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
         tg_gsinu     =var_init('gsinu',dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
         tg_gsinv     =var_init('gsinv',dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
         tg_gsinf     =var_init('gsinf',dl_tmp2D(:,:)   , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      ENDIF

      ! variable 3D
      dl_tmp3D(:,:,:)=dp_fill_i1

      tg_tmask   = var_init('tmask'  ,dl_tmp3D(:,:,:), dd_fill=dp_fill_i1, id_type=NF90_BYTE)
      tg_umask   = var_init('umask'  ,dl_tmp3D(:,:,:), dd_fill=dp_fill_i1, id_type=NF90_BYTE)
      tg_vmask   = var_init('vmask'  ,dl_tmp3D(:,:,:), dd_fill=dp_fill_i1, id_type=NF90_BYTE)
      IF( .NOT. ld_domcfg )THEN
         tg_fmask   = var_init('fmask'  ,dl_tmp3D(:,:,:), dd_fill=dp_fill_i1, id_type=NF90_BYTE)
      ENDIF
      
!      tg_wmask   = var_init('wmask'  ,dl_tmp3D(:,:,:), dd_fill=dp_fill_i1, id_type=NF90_BYTE)
!      tg_wumask  = var_init('wumask' ,dl_tmp3D(:,:,:), dd_fill=dp_fill_i1, id_type=NF90_BYTE)
!      tg_wvmask  = var_init('wvmask' ,dl_tmp3D(:,:,:), dd_fill=dp_fill_i1, id_type=NF90_BYTE)

   END SUBROUTINE grid_hgr_init
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_hgr_clean(ld_domcfg) 
   !-------------------------------------------------------------------
   !> @brief This subroutine clean hgr structure
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial version
   !>
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      LOGICAL    , INTENT(IN) :: ld_domcfg

      ! local variable
      ! loop indices
      !----------------------------------------------------------------
      CALL var_clean(tg_ssmask  )

      CALL var_clean(tg_glamt)
      CALL var_clean(tg_glamu)
      CALL var_clean(tg_glamv)
      CALL var_clean(tg_glamf)

      CALL var_clean(tg_gphit)
      CALL var_clean(tg_gphiu)
      CALL var_clean(tg_gphiv)
      CALL var_clean(tg_gphif)

      CALL var_clean(tg_e1t  )
      CALL var_clean(tg_e1u  )
      CALL var_clean(tg_e1v  )
      CALL var_clean(tg_e1f  )

      CALL var_clean(tg_e2t  )
      CALL var_clean(tg_e2u  )
      CALL var_clean(tg_e2v  )
      CALL var_clean(tg_e2f  )

      CALL var_clean(tg_ff_t )
      CALL var_clean(tg_ff_f )

      IF( .NOT. ld_domcfg )THEN
         CALL var_clean(tg_gcost )
         CALL var_clean(tg_gcosu )
         CALL var_clean(tg_gcosv )
         CALL var_clean(tg_gcosf )

         CALL var_clean(tg_gsint )
         CALL var_clean(tg_gsinu )
         CALL var_clean(tg_gsinv )
         CALL var_clean(tg_gsinf )
      ENDIF

      CALL var_clean(tg_tmask   )
      CALL var_clean(tg_umask   )
      CALL var_clean(tg_vmask   )
      IF( .NOT. ld_domcfg )THEN
         CALL var_clean(tg_fmask   )
      ENDIF
   END SUBROUTINE grid_hgr_clean
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid_hgr_nam(cd_coord, id_perio, cd_namelist) &
         & RESULT (tf_namh)
   !-------------------------------------------------------------------
   !> @brief This function initialise hgr namelist structure
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial version
   !>
   !> @param[in] cd_coord   
   !> @param[in] id_perio  
   !> @param[in] cd_namelist
   !> @return hgr namelist structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=*), INTENT(IN) :: cd_coord
      INTEGER(i4)     , INTENT(IN) :: id_perio   
      CHARACTER(LEN=*), INTENT(IN) :: cd_namelist
      
      ! function
      TYPE(TNAMH)                  :: tf_namh

      ! local variable
      INTEGER(i4)        :: il_status
      INTEGER(i4)        :: il_fileid

      LOGICAL            :: ll_exist

      ! loop indices
      ! namelist

      ! namhgr
      INTEGER(i4)       :: in_mshhgr   = 0 
      REAL(dp)          :: dn_ppglam0  = NF90_FILL_DOUBLE
      REAL(dp)          :: dn_ppgphi0  = NF90_FILL_DOUBLE
      REAL(dp)          :: dn_ppe1_deg = NF90_FILL_DOUBLE
      REAL(dp)          :: dn_ppe2_deg = NF90_FILL_DOUBLE
!      REAL(dp)          :: dn_ppe1_m   = NF90_FILL_DOUBLE
!      REAL(dp)          :: dn_ppe2_m   = NF90_FILL_DOUBLE

!      ! namcla
!      INTEGER(i4)       :: in_cla      = 0

      ! namgrd
!      CHARACTER(LEN=lc) :: cn_cfg      = ''
      INTEGER(i4)       :: in_cfg      = 0
      LOGICAL           :: ln_bench    = .FALSE.

      !----------------------------------------------------------------
      NAMELIST /namhgr/ &  
      &  in_mshhgr,     &  !< type of horizontal mesh 
                           !< 0: curvilinear coordinate on the sphere read in coordinate.nc
                           !< 1: geographical mesh on the sphere with regular grid-spacing
                           !< 2: f-plane with regular grid-spacing
                           !< 3: beta-plane with regular grid-spacing
                           !< 4: Mercator grid with T/U point at the equator
                           !< 5: beta-plane with regular grid-spacing and rotated domain (GYRE configuration)
      &  dn_ppglam0,    &  !< longitude of first raw and column T-point (in_mshhgr = 1 or 4)
      &  dn_ppgphi0,    &  !< latitude  of first raw and column T-point (in_mshhgr = 1 or 4)
      &  dn_ppe1_deg,   &  !< zonal      grid-spacing (degrees)         (in_mshhgr = 1,2,3 or 4)
      &  dn_ppe2_deg       !< meridional grid-spacing (degrees)         (in_mshhgr = 1,2,3 or 4)
!      &  dn_ppe1_m,     &  !< zonal      grid-spacing (degrees)
!      &  dn_ppe2_m         !< meridional grid-spacing (degrees)

!      NAMELIST /namcla/ &
!      &  in_cla            !< =1 cross land advection for exchanges through some straits (ORCA2)

      NAMELIST/namgrd/  &  !< orca grid namelist
!      &  cn_cfg,        &  !< name of the configuration (orca)
      &  in_cfg,        &  !< resolution of the configuration (2,1,025..)
      &  ln_bench          !< benchmark parameter (in_mshhgr = 5 ). 

      !----------------------------------------------------------------
      ! read namelist
      INQUIRE(FILE=TRIM(cd_namelist), EXIST=ll_exist)
      IF( ll_exist )THEN
         
         il_fileid=fct_getunit()

         OPEN( il_fileid, FILE=TRIM(cd_namelist), &
            &             FORM='FORMATTED',       &
            &             ACCESS='SEQUENTIAL',    &
            &             STATUS='OLD',           &
            &             ACTION='READ',          &
            &             IOSTAT=il_status)
         CALL fct_err(il_status)
         IF( il_status /= 0 )THEN
            CALL logger_fatal("GRID HGR NAM: error opening "//&
               &  TRIM(cd_namelist))
         ENDIF

         READ( il_fileid, NML = namhgr  )
!         READ( il_fileid, NML = namcla  )
!         READ( il_fileid, NML = namgrd  )

         CLOSE( il_fileid, IOSTAT=il_status )
         CALL fct_err(il_status)
         IF( il_status /= 0 )THEN
            CALL logger_error("GRID HGR NAM: closing "//TRIM(cd_namelist))
         ENDIF
        
         tf_namh%c_coord   = TRIM(cd_coord)
         tf_namh%i_perio   = id_perio

         tf_namh%i_mshhgr  = in_mshhgr
         tf_namh%d_ppglam0 = dn_ppglam0
         tf_namh%d_ppgphi0 = dn_ppgphi0

         tf_namh%d_ppe1_deg= dn_ppe1_deg
         tf_namh%d_ppe2_deg= dn_ppe2_deg
!         tf_namh%d_ppe1_m  = dn_ppe1_m
!         tf_namh%d_ppe2_m  = dn_ppe2_m

!         tf_namh%i_cla     = in_cla

!         tf_namh%c_cfg     = TRIM(cn_cfg)
         tf_namh%i_cfg     = in_cfg
         tf_namh%l_bench   = ln_bench

      ELSE

         CALL logger_fatal(" GRID HGR NAM: can't find "//TRIM(cd_namelist))

      ENDIF

   END FUNCTION grid_hgr_nam
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_hgr_fill(td_nam, jpi, jpj, ld_domcfg) 
   !-------------------------------------------------------------------
   !> @brief This subroutine fill horizontal mesh (hgr structure)
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial version
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   !> @param[in] jpj
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMH), INTENT(IN) :: td_nam
      INTEGER(i4), INTENT(IN) :: jpi
      INTEGER(i4), INTENT(IN) :: jpj
      LOGICAL    , INTENT(IN) :: ld_domcfg

      ! local variable
      REAL(dp)    :: znorme
      ! loop indices
      !----------------------------------------------------------------
      CALL logger_info('GRID HGR FILL : define the horizontal mesh from the'//&
      &  ' type of horizontal mesh mshhgr = '//TRIM(fct_str(td_nam%i_mshhgr)))
      IF( td_nam%i_mshhgr == 1 )THEN
         CALL logger_info('   position of the first row and     ppglam0  = '//&
            &  TRIM(fct_str(td_nam%d_ppglam0  )) )
         CALL logger_info('   column grid-point (degrees)       ppgphi0  = '//&
            &  TRIM(fct_str(td_nam%d_ppgphi0  )) )
      ELSEIF( td_nam%i_mshhgr == 2 .OR. td_nam%i_mshhgr == 3  )THEN
         CALL logger_info('   zonal      grid-spacing (degrees) ppe1_deg = '//&
            &  TRIM(fct_str(td_nam%d_ppe1_deg )) )
         CALL logger_info('   meridional grid-spacing (degrees) ppe2_deg = '//&
            &  TRIM(fct_str(td_nam%d_ppe2_deg )) )
!         CALL logger_info('   zonal      grid-spacing (meters)  ppe1_m   = '//&
!            &  TRIM(fct_str(td_nam%d_ppe1_m   )) ) 
!         CALL logger_info('   meridional grid-spacing (meters)  ppe2_m   = '//&
!            &  TRIM(fct_str(td_nam%d_ppe2_m   )) ) 
      ENDIF

      SELECT CASE( td_nam%i_mshhgr ) ! type of horizontal mesh

         CASE(0)   !  curvilinear coordinate on the sphere read in coordinate.nc file

            CALL grid_hgr__fill_curv(td_nam)!,jpi,jpj)

         CASE(1)   ! geographical mesh on the sphere with regular grid-spacing

            CALL grid_hgr__fill_reg(td_nam,jpi,jpj)

         CASE(2:3) ! f- or beta-plane with regular grid-spacing

            CALL grid_hgr__fill_plan(td_nam,jpi,jpj)

         CASE(4)   ! geographical mesh on the sphere, isotropic MERCATOR type

            CALL grid_hgr__fill_merc(td_nam,jpi,jpj)

         CASE(5)   ! beta-plane with regular grid-spacing and rotated domain (GYRE configuration)

            CALL grid_hgr__fill_gyre(td_nam,jpi,jpj)

         CASE DEFAULT

            CALL logger_fatal('GRID HGR FILL : bad flag value for mshhgr = '//&
               &  TRIM(fct_str(td_nam%i_mshhgr)))

      END SELECT

      ! No Useful associated horizontal metrics
      ! ---------------------------------------

      ! create coriolis factor
      CALL grid_hgr__fill_coriolis(td_nam,jpi)!,jpj)
     
      ! Control of domain for symetrical condition
      ! ------------------------------------------
      ! The equator line must be the latitude coordinate axe

      IF( td_nam%i_perio == 2 ) THEN
         znorme = SQRT( SUM(tg_gphiu%d_value(:,2,1,1)*tg_gphiu%d_value(:,2,1,1)) ) / FLOAT( jpi )
         IF( znorme > 1.e-13 )THEN
            CALL logger_fatal( ' ===>>>> : symmetrical condition: rerun with good equator line' )
         ENDIF
      ENDIF

      ! compute angles between model grid lines and the North direction
      ! ---------------------------------------------------------------
      IF( .NOT. ld_domcfg )THEN
         CALL grid_hgr__angle(td_nam,jpi,jpj) 
      ENDIF

   END SUBROUTINE grid_hgr_fill
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_hgr__fill_curv(td_nam)!,jpi,jpj) 
   !-------------------------------------------------------------------
   !> @brief This subroutine fill horizontal mesh (hgr structure)
   !> for case of curvilinear coordinate on the sphere read in coordinate.nc file
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial version
   !> @date October, 2016
   !> - do not use anymore special case for ORCA grid
   !>
   !> @param[in] td_nam
   ! @param[in] jpi
   ! @param[in] jpj   
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMH), INTENT(IN) :: td_nam
!      INTEGER(i4), INTENT(IN) :: jpi
!      INTEGER(i4), INTENT(IN) :: jpj

      ! local variable
!      INTEGER(i4) :: ii0, ii1, ij0, ij1   ! temporary integers
!      INTEGER(i4) :: isrow                ! index for ORCA1 starting row

      TYPE(TMPP)  :: tl_coord

      ! loop indices
      !----------------------------------------------------------------

      ! read coordinates
      ! open file
      IF( td_nam%c_coord /= '' )THEN
         tl_coord=mpp_init( file_init(TRIM(td_nam%c_coord)), id_perio=td_nam%i_perio)
         CALL grid_get_info(tl_coord)
      ELSE
         CALL logger_fatal("GRID HGR FILL: no input coordinates file found. "//&
         &     "check namelist")      
      ENDIF

      CALL iom_mpp_open( tl_coord )

      ! read variable in coordinates
      tg_glamt=iom_mpp_read_var(tl_coord, 'glamt')
      tg_glamu=iom_mpp_read_var(tl_coord, 'glamu')
      tg_glamv=iom_mpp_read_var(tl_coord, 'glamv')
      tg_glamf=iom_mpp_read_var(tl_coord, 'glamf')

      tg_gphit=iom_mpp_read_var(tl_coord, 'gphit')
      tg_gphiu=iom_mpp_read_var(tl_coord, 'gphiu')
      tg_gphiv=iom_mpp_read_var(tl_coord, 'gphiv')
      tg_gphif=iom_mpp_read_var(tl_coord, 'gphif')

      ! force output type
      tg_glamt%i_type=NF90_DOUBLE
      tg_glamu%i_type=NF90_DOUBLE
      tg_glamv%i_type=NF90_DOUBLE
      tg_glamf%i_type=NF90_DOUBLE

      tg_gphit%i_type=NF90_DOUBLE
      tg_gphiu%i_type=NF90_DOUBLE
      tg_gphiv%i_type=NF90_DOUBLE
      tg_gphif%i_type=NF90_DOUBLE

      tg_e1t  =iom_mpp_read_var(tl_coord, 'e1t')
      tg_e1u  =iom_mpp_read_var(tl_coord, 'e1u')
      tg_e1v  =iom_mpp_read_var(tl_coord, 'e1v')
      tg_e1f  =iom_mpp_read_var(tl_coord, 'e1f')

      tg_e2t  =iom_mpp_read_var(tl_coord, 'e2t')
      tg_e2u  =iom_mpp_read_var(tl_coord, 'e2u')
      tg_e2v  =iom_mpp_read_var(tl_coord, 'e2v')
      tg_e2f  =iom_mpp_read_var(tl_coord, 'e2f')

      CALL iom_mpp_close( tl_coord )
      ! clean
      CALL mpp_clean(tl_coord)

      !! WARNING extended grid have to be correctly fill

!      !! special case for ORCA grid
!      ! ORCA R2 configuration
!      IF( TRIM(td_nam%c_cfg) == "orca" .AND. td_nam%i_cfg == 2 ) THEN
!            IF( td_nam%i_cla == 0 ) THEN
!               !
!               ! Gibraltar Strait (e2u = 20 km)
!               ii0 = 139   ;   ii1 = 140
!               ij0 = 102   ;   ij1 = 102   
!               ! e2u = 20 km
!               tg_e2u%d_value(ii0:ii1,ij0:ij1,1,1) =  20.e3
!               CALL logger_info('orca_r2: Gibraltar    : e2u reduced to 20 km')
!               !
!               ! Bab el Mandeb (e2u = 18 km)
!               ii0 = 160   ;   ii1 = 160 
!               ij0 =  88   ;   ij1 =  88   
!               ! e1v = 18 km
!               tg_e1v%d_value(ii0:ii1,ij0:ij1,1,1) =  18.e3
!               ! e2u = 30 km
!               tg_e2u%d_value(ii0:ii1,ij0:ij1,1,1) =  30.e3
!
!               CALL logger_info('orca_r2: Bab el Mandeb: e2u reduced to 30 km')
!               CALL logger_info('e1v reduced to 18 km')
!            ENDIF
!            ! Danish Straits
!            ii0 = 145   ;   ii1 = 146
!            ij0 = 116   ;   ij1 = 116   
!            ! e2u = 10 km
!            tg_e2u%d_value(ii0:ii1,ij0:ij1,1,1) =  10.e3
!            CALL logger_info('orca_r2: Danish Straits : e2u reduced to 10 km')
!      ENDIF
!
!      ! ORCA R1 configuration
!      IF( TRIM(td_nam%c_cfg) == "orca" .AND. td_nam%i_cfg == 1 ) THEN
!         ! This dirty section will be suppressed by simplification process: all this will come back in input files
!         ! Currently these hard-wired indices relate to configuration with
!         ! extend grid (jpjglo=332)
!         ! which had a grid-size of 362x292.
!
!         isrow = 332 - jpj
!
!         ! Gibraltar Strait (e2u = 20 km)
!         ii0 = 282           ;   ii1 = 283
!         ij0 = 201 + isrow   ;   ij1 = 241 - isrow 
!         ! e2u = 20 km
!         tg_e2u%d_value(ii0:ii1,ij0:ij1,1,1) =  20.e3
!         CALL logger_info('orca_r1: Gibraltar : e2u reduced to 20 km')
!
!         ! Bhosporus Strait (e2u = 10 km)
!         ii0 = 314           ;   ii1 = 315        ! Bhosporus Strait (e2u = 10 km)
!         ij0 = 208 + isrow   ;   ij1 = 248 - isrow
!         ! Bhosporus Strait (e2u = 10 km)
!         tg_e2u%d_value(ii0:ii1,ij0:ij1,1,1) =  10.e3
!         CALL logger_info('orca_r1: Bhosporus : e2u reduced to 10 km')
!
!         ! Lombok Strait (e1v = 13 km)
!         ii0 =  44           ;   ii1 =  44        ! Lombok Strait (e1v = 13 km)
!         ij0 = 124 + isrow   ;   ij1 = 165 - isrow
!         ! Lombok Strait (e1v = 13 km)
!         tg_e1v%d_value(ii0:ii1,ij0:ij1,1,1) =  13.e3
!         CALL logger_info('orca_r1: Lombok : e1v reduced to 10 km')
!
!         ! Sumba Strait (e1v = 8 km) [closed from bathy_11 on]
!         ii0 =  48           ;   ii1 =  48        ! Sumba Strait (e1v = 8 km) [closed from bathy_11 on]
!         ij0 = 124 + isrow   ;   ij1 = 165 - isrow
!         ! Sumba Strait (e1v = 8 km) [closed from bathy_11 on]
!         tg_e1v%d_value(ii0:ii1,ij0:ij1,1,1) =  8.e3
!         CALL logger_info('orca_r1: Sumba : e1v reduced to 8 km')
!
!         ! Ombai Strait (e1v = 13 km)
!         ii0 =  53           ;   ii1 =  53        ! Ombai Strait (e1v = 13 km)
!         ij0 = 124 + isrow   ;   ij1 = 165 - isrow
!         ! Ombai Strait (e1v = 13 km)
!         tg_e1v%d_value(ii0:ii1,ij0:ij1,1,1) = 13.e3
!         CALL logger_info('orca_r1: Ombai : e1v reduced to 13 km')
!
!         ! Timor Passage (e1v = 20 km)
!         ii0 =  56           ;   ii1 =  56        ! Timor Passage (e1v = 20 km)
!         ij0 = 124 + isrow   ;   ij1 = 145 - isrow
!         ! Timor Passage (e1v = 20 km)
!         tg_e1v%d_value(ii0:ii1,ij0:ij1,1,1) = 20.e3
!         CALL logger_info('orca_r1: Timor Passage : e1v reduced to 20 km')
!
!          ! West Halmahera Strait (e1v = 30 km)
!         ii0 =  55           ;   ii1 =  55        ! West Halmahera Strait (e1v = 30 km)
!         ij0 = 141 + isrow   ;   ij1 = 182 - isrow
!         ! West Halmahera Strait (e1v = 30 km)
!         tg_e1v%d_value(ii0:ii1,ij0:ij1,1,1) = 30.e3
!         CALL logger_info('orca_r1: W Halmahera : e1v reduced to 30 km')
!
!         ! East Halmahera Strait (e1v = 50 km)
!         ii0 =  58           ;   ii1 =  58        ! East Halmahera Strait (e1v = 50 km)
!         ij0 = 141 + isrow   ;   ij1 = 182 - isrow
!         ! East Halmahera Strait (e1v = 50 km)
!         tg_e1v%d_value(ii0:ii1,ij0:ij1,1,1) = 50.e3
!         CALL logger_info('orca_r1: E Halmahera : e1v reduced to 50 km')
!
!      ENDIF
!
!      ! ORCA R05 configuration
!      IF( TRIM(td_nam%c_cfg) == "orca" .AND. td_nam%i_cfg == 05 ) THEN
!
!         ! Gibraltar Strait (e2u = 20 km)
!         ii0 = 563   ;   ii1 = 564        ! Gibraltar Strait (e2u = 20 km)
!         ij0 = 327   ;   ij1 = 327
!         ! Gibraltar Strait (e2u = 20 km)
!         tg_e2u%d_value(ii0:ii1,ij0:ij1,1,1) =  20.e3
!         CALL logger_info('orca_r05: Reduced e2u at the Gibraltar Strait')
!         !
!         ! Bosphore Strait (e2u = 10 km)
!         ii0 = 627   ;   ii1 = 628        ! Bosphore Strait (e2u = 10 km)
!         ij0 = 343   ;   ij1 = 343
!         ! Bosphore Strait (e2u = 10 km)
!         tg_e2u%d_value(ii0:ii1,ij0:ij1,1,1) =  10.e3
!         CALL logger_info('orca_r05: Reduced e2u at the Bosphore Strait')
!         !
!         ! Sumba Strait (e2u = 40 km)
!         ii0 =  93   ;   ii1 =  94        ! Sumba Strait (e2u = 40 km)
!         ij0 = 232   ;   ij1 = 232
!         ! Sumba Strait (e2u = 40 km)
!         tg_e2u%d_value(ii0:ii1,ij0:ij1,1,1) =  40.e3
!         CALL logger_info('orca_r05: Reduced e2u at the Sumba Strait')
!         !
!         ! Ombai Strait (e2u = 15 km)
!         ii0 = 103   ;   ii1 = 103        ! Ombai Strait (e2u = 15 km)
!         ij0 = 232   ;   ij1 = 232
!         ! Ombai Strait (e2u = 15 km)
!         tg_e2u%d_value(ii0:ii1,ij0:ij1,1,1) =  15.e3
!         CALL logger_info('orca_r05: Reduced e2u at the Ombai Strait')
!         !
!         ! Palk Strait (e2u = 10 km)
!         ii0 =  15   ;   ii1 =  15        ! Palk Strait (e2u = 10 km)
!         ij0 = 270   ;   ij1 = 270
!         ! Palk Strait (e2u = 10 km)
!         tg_e2u%d_value(ii0:ii1,ij0:ij1,1,1) =  10.e3
!         CALL logger_info('orca_r05: Reduced e2u at the Palk Strait')
!         !
!         ! Lombok Strait (e1v = 10 km)
!         ii0 =  87   ;   ii1 =  87        ! Lombok Strait (e1v = 10 km)
!         ij0 = 232   ;   ij1 = 233
!         ! Lombok Strait (e1v = 10 km)
!         tg_e1v%d_value(ii0:ii1,ij0:ij1,1,1) =  10.e3
!         CALL logger_info('orca_r05: Reduced e1v at the Lombok Strait')
!         !
!         !
!         ! Bab el Mandeb (e1v = 25 km)
!         ii0 = 662   ;   ii1 = 662        ! Bab el Mandeb (e1v = 25 km)
!         ij0 = 276   ;   ij1 = 276
!         ! Bab el Mandeb (e1v = 25 km)
!         tg_e1v%d_value(ii0:ii1,ij0:ij1,1,1) =  25.e3
!         CALL logger_info('orca_r05: Reduced e1v at the Bab el Mandeb')
!
!      ENDIF

   END SUBROUTINE grid_hgr__fill_curv
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_hgr__fill_reg(td_nam, jpi, jpj) 
   !-------------------------------------------------------------------
   !> @brief This subroutine fill horizontal mesh (hgr structure)
   !> for case of geographical mesh on the sphere with regular grid-spacing
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial version
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   !> @param[in] jpj   
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMH), INTENT(IN) :: td_nam
      INTEGER(i4), INTENT(IN) :: jpi
      INTEGER(i4), INTENT(IN) :: jpj

      ! local variable
      REAL(dp)   ::   zti, zui, zvi, zfi   ! local scalars
      REAL(dp)   ::   ztj, zuj, zvj, zfj   !

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      CALL logger_info('GRID HGR FILL : geographical mesh on the sphere with'//&
         &  ' regular grid-spacing given by ppe1_deg and ppe2_deg')

      DO jj = 1, jpj
         DO ji = 1, jpi
            zti = FLOAT( ji - 1 )         ;   ztj = FLOAT( jj - 1 )
            zui = FLOAT( ji - 1 ) + 0.5   ;   zuj = FLOAT( jj - 1 )
            zvi = FLOAT( ji - 1 )         ;   zvj = FLOAT( jj - 1 ) + 0.5
            zfi = FLOAT( ji - 1 ) + 0.5   ;   zfj = FLOAT( jj - 1 ) + 0.5
      ! Longitude
            tg_glamt%d_value(ji,jj,1,1) = td_nam%d_ppglam0 + td_nam%d_ppe1_deg * zti
            tg_glamu%d_value(ji,jj,1,1) = td_nam%d_ppglam0 + td_nam%d_ppe1_deg * zui
            tg_glamv%d_value(ji,jj,1,1) = td_nam%d_ppglam0 + td_nam%d_ppe1_deg * zvi
            tg_glamf%d_value(ji,jj,1,1) = td_nam%d_ppglam0 + td_nam%d_ppe1_deg * zfi
      ! Latitude
            tg_gphit%d_value(ji,jj,1,1) = td_nam%d_ppgphi0 + td_nam%d_ppe2_deg * ztj
            tg_gphiu%d_value(ji,jj,1,1) = td_nam%d_ppgphi0 + td_nam%d_ppe2_deg * zuj
            tg_gphiv%d_value(ji,jj,1,1) = td_nam%d_ppgphi0 + td_nam%d_ppe2_deg * zvj
            tg_gphif%d_value(ji,jj,1,1) = td_nam%d_ppgphi0 + td_nam%d_ppe2_deg * zfj
      ! e1
            tg_e1t%d_value(ji,jj,1,1) = dp_rearth * dp_deg2rad * COS( dp_deg2rad * tg_gphit%d_value(ji,jj,1,1) ) * td_nam%d_ppe1_deg
            tg_e1u%d_value(ji,jj,1,1) = dp_rearth * dp_deg2rad * COS( dp_deg2rad * tg_gphiu%d_value(ji,jj,1,1) ) * td_nam%d_ppe1_deg
            tg_e1v%d_value(ji,jj,1,1) = dp_rearth * dp_deg2rad * COS( dp_deg2rad * tg_gphiv%d_value(ji,jj,1,1) ) * td_nam%d_ppe1_deg
            tg_e1f%d_value(ji,jj,1,1) = dp_rearth * dp_deg2rad * COS( dp_deg2rad * tg_gphif%d_value(ji,jj,1,1) ) * td_nam%d_ppe1_deg
      ! e2
            tg_e2t%d_value(ji,jj,1,1) = dp_rearth * dp_deg2rad * td_nam%d_ppe2_deg
            tg_e2u%d_value(ji,jj,1,1) = dp_rearth * dp_deg2rad * td_nam%d_ppe2_deg
            tg_e2v%d_value(ji,jj,1,1) = dp_rearth * dp_deg2rad * td_nam%d_ppe2_deg
            tg_e2f%d_value(ji,jj,1,1) = dp_rearth * dp_deg2rad * td_nam%d_ppe2_deg
         END DO
      END DO

   END SUBROUTINE grid_hgr__fill_reg
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_hgr__fill_plan(td_nam, jpi, jpj) 
   !-------------------------------------------------------------------
   !> @brief This subroutine fill horizontal mesh (hgr structure)
   !> for case of f- or beta-plane with regular grid-spacing
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial version
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   !> @param[in] jpj   
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMH), INTENT(IN) :: td_nam
      INTEGER(i4), INTENT(IN) :: jpi
      INTEGER(i4), INTENT(IN) :: jpj

      ! local variable
      REAL(dp)    :: dl_glam0
      REAL(dp)    :: dl_gphi0

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      CALL logger_info('GRID HGR FILL : f- or beta-plane with regular'//&
         &  ' grid-spacing given by ppe1_deg and ppe2_deg')
!         &  ' grid-spacing given by ppe1_m and ppe2_m')

      ! Position coordinates (in kilometers)
      !                          ==========
      dl_glam0 = 0.e0
      dl_gphi0 = - td_nam%d_ppe2_deg * 1.e-3
!      dl_gphi0 = - td_nam%d_ppe2_m * 1.e-3

         !
      DO jj = 1, jpj
         DO ji = 1, jpi
!            tg_glamt%d_value(ji,jj,1,1) = dl_glam0 + td_nam%d_ppe1_m * 1.e-3 * ( FLOAT( ji - 1 )       )
!            tg_glamu%d_value(ji,jj,1,1) = dl_glam0 + td_nam%d_ppe1_m * 1.e-3 * ( FLOAT( ji - 1 ) + 0.5 )
            tg_glamt%d_value(ji,jj,1,1) = dl_glam0 + td_nam%d_ppe1_deg * 1.e-3 * ( FLOAT( ji - 1 )       )
            tg_glamu%d_value(ji,jj,1,1) = dl_glam0 + td_nam%d_ppe1_deg * 1.e-3 * ( FLOAT( ji - 1 ) + 0.5 )
            tg_glamv%d_value(ji,jj,1,1) = tg_glamt%d_value(ji,jj,1,1)
            tg_glamf%d_value(ji,jj,1,1) = tg_glamu%d_value(ji,jj,1,1)
   
            !tg_gphit%d_value(ji,jj,1,1) = dl_gphi0 + td_nam%d_ppe2_m * 1.e-3 * ( FLOAT( jj - 1 )       )
            tg_gphit%d_value(ji,jj,1,1) = dl_gphi0 + td_nam%d_ppe2_deg * 1.e-3 * ( FLOAT( jj - 1 )       )
            tg_gphiu%d_value(ji,jj,1,1) = tg_gphit%d_value(ji,jj,1,1)
            !tg_gphiv%d_value(ji,jj,1,1) = dl_gphi0 + td_nam%d_ppe2_m * 1.e-3 * ( FLOAT( jj - 1 ) + 0.5 )
            tg_gphiv%d_value(ji,jj,1,1) = dl_gphi0 + td_nam%d_ppe2_deg * 1.e-3 * ( FLOAT( jj - 1 ) + 0.5 )
            tg_gphif%d_value(ji,jj,1,1) = tg_gphiv%d_value(ji,jj,1,1)
         END DO
      END DO

      ! Horizontal scale factors (in meters)
      !                              ======
!      tg_e1t%d_value(:,:,1,1) = td_nam%d_ppe1_m      
!      tg_e1u%d_value(:,:,1,1) = td_nam%d_ppe1_m      
!      tg_e1v%d_value(:,:,1,1) = td_nam%d_ppe1_m      
!      tg_e1f%d_value(:,:,1,1) = td_nam%d_ppe1_m      
      tg_e1t%d_value(:,:,1,1) = td_nam%d_ppe1_deg      
      tg_e1u%d_value(:,:,1,1) = td_nam%d_ppe1_deg      
      tg_e1v%d_value(:,:,1,1) = td_nam%d_ppe1_deg      
      tg_e1f%d_value(:,:,1,1) = td_nam%d_ppe1_deg      

!      tg_e2t%d_value(:,:,1,1) = td_nam%d_ppe2_m
!      tg_e2u%d_value(:,:,1,1) = td_nam%d_ppe2_m
!      tg_e2v%d_value(:,:,1,1) = td_nam%d_ppe2_m
!      tg_e2f%d_value(:,:,1,1) = td_nam%d_ppe2_m
      tg_e2t%d_value(:,:,1,1) = td_nam%d_ppe2_deg
      tg_e2u%d_value(:,:,1,1) = td_nam%d_ppe2_deg
      tg_e2v%d_value(:,:,1,1) = td_nam%d_ppe2_deg
      tg_e2f%d_value(:,:,1,1) = td_nam%d_ppe2_deg

   END SUBROUTINE grid_hgr__fill_plan
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_hgr__fill_merc(td_nam, jpi, jpj) 
   !-------------------------------------------------------------------
   !> @brief This subroutine fill horizontal mesh (hgr structure)
   !> for case of geographical mesh on the sphere, isotropic MERCATOR type
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial version
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   !> @param[in] jpj   
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMH), INTENT(IN) :: td_nam
      INTEGER(i4), INTENT(IN) :: jpi
      INTEGER(i4), INTENT(IN) :: jpj

      ! local variable
      INTEGER  ::   ijeq                 ! index of equator T point (used in case 4)

      REAL(dp) ::   zti, zui, zvi, zfi   ! local scalars
      REAL(dp) ::   ztj, zuj, zvj, zfj   !
      REAL(dp) ::   zarg

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      CALL logger_info('GRID HGR FILL : geographical mesh on the sphere, '//&
        &  'MERCATOR type longitudinal/latitudinal spacing given by ppe1_deg')

      IF( td_nam%d_ppgphi0 == -90 )THEN
         CALL logger_fatal(' Mercator grid cannot start at south pole !!!! ' )
      ENDIF

      !  Find index corresponding to the equator, given the grid spacing e1_deg
      !  and the (approximate) southern latitude ppgphi0.
      !  This way we ensure that the equator is at a "T / U" point, when in the domain.
      !  The formula should work even if the equator is outside the domain.
      zarg = dp_pi / 4. - dp_pi / 180. * td_nam%d_ppgphi0 / 2.
      ijeq = ABS( 180./dp_pi * LOG( COS( zarg ) / SIN( zarg ) ) / td_nam%d_ppe1_deg )
      IF( td_nam%d_ppgphi0 > 0 )  ijeq = -ijeq

      CALL logger_info('Index of the equator on the MERCATOR grid: '//TRIM(fct_str(ijeq)))

      DO jj = 1, jpj
         DO ji = 1, jpi
            zti = FLOAT( ji - 1 )         ;   ztj = FLOAT( jj - ijeq )
            zui = FLOAT( ji - 1 ) + 0.5   ;   zuj = FLOAT( jj - ijeq )
            zvi = FLOAT( ji - 1 )         ;   zvj = FLOAT( jj - ijeq ) + 0.5
            zfi = FLOAT( ji - 1 ) + 0.5   ;   zfj = FLOAT( jj - ijeq ) + 0.5
         ! Longitude
            tg_glamt%d_value(ji,jj,1,1) = td_nam%d_ppglam0 + td_nam%d_ppe1_deg * zti
            tg_glamu%d_value(ji,jj,1,1) = td_nam%d_ppglam0 + td_nam%d_ppe1_deg * zui
            tg_glamv%d_value(ji,jj,1,1) = td_nam%d_ppglam0 + td_nam%d_ppe1_deg * zvi
            tg_glamf%d_value(ji,jj,1,1) = td_nam%d_ppglam0 + td_nam%d_ppe1_deg * zfi
         ! Latitude
            tg_gphit%d_value(ji,jj,1,1) = 1./dp_deg2rad * ASIN ( TANH( td_nam%d_ppe1_deg *dp_deg2rad* ztj ) )
            tg_gphiu%d_value(ji,jj,1,1) = 1./dp_deg2rad * ASIN ( TANH( td_nam%d_ppe1_deg *dp_deg2rad* zuj ) )
            tg_gphiv%d_value(ji,jj,1,1) = 1./dp_deg2rad * ASIN ( TANH( td_nam%d_ppe1_deg *dp_deg2rad* zvj ) )
            tg_gphif%d_value(ji,jj,1,1) = 1./dp_deg2rad * ASIN ( TANH( td_nam%d_ppe1_deg *dp_deg2rad* zfj ) )
         ! e1
            tg_e1t%d_value(ji,jj,1,1) = dp_rearth * dp_deg2rad * COS( dp_deg2rad * tg_gphit%d_value(ji,jj,1,1) ) * td_nam%d_ppe1_deg
            tg_e1u%d_value(ji,jj,1,1) = dp_rearth * dp_deg2rad * COS( dp_deg2rad * tg_gphiu%d_value(ji,jj,1,1) ) * td_nam%d_ppe1_deg
            tg_e1v%d_value(ji,jj,1,1) = dp_rearth * dp_deg2rad * COS( dp_deg2rad * tg_gphiv%d_value(ji,jj,1,1) ) * td_nam%d_ppe1_deg
            tg_e1f%d_value(ji,jj,1,1) = dp_rearth * dp_deg2rad * COS( dp_deg2rad * tg_gphif%d_value(ji,jj,1,1) ) * td_nam%d_ppe1_deg
         ! e2
            tg_e2t%d_value(ji,jj,1,1) = dp_rearth * dp_deg2rad * COS( dp_deg2rad * tg_gphit%d_value(ji,jj,1,1) ) * td_nam%d_ppe1_deg
            tg_e2u%d_value(ji,jj,1,1) = dp_rearth * dp_deg2rad * COS( dp_deg2rad * tg_gphiu%d_value(ji,jj,1,1) ) * td_nam%d_ppe1_deg
            tg_e2v%d_value(ji,jj,1,1) = dp_rearth * dp_deg2rad * COS( dp_deg2rad * tg_gphiv%d_value(ji,jj,1,1) ) * td_nam%d_ppe1_deg
            tg_e2f%d_value(ji,jj,1,1) = dp_rearth * dp_deg2rad * COS( dp_deg2rad * tg_gphif%d_value(ji,jj,1,1) ) * td_nam%d_ppe1_deg
         END DO
      END DO

   END SUBROUTINE grid_hgr__fill_merc
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_hgr__fill_gyre(td_nam, jpi, jpj) 
   !-------------------------------------------------------------------
   !> @brief This subroutine fill horizontal mesh (hgr structure)
   !> for case of beta-plane with regular grid-spacing and rotated domain (GYRE configuration)
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial version
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   !> @param[in] jpj
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMH), INTENT(IN) :: td_nam
      INTEGER(i4), INTENT(IN) :: jpi
      INTEGER(i4), INTENT(IN) :: jpj

      ! local variable
      REAL(dp) ::   zlam1, zcos_alpha, zim1 , zjm1 , ze1, ze1deg
      REAL(dp) ::   zphi1, zsin_alpha, zim05, zjm05

      REAL(dp) ::   dl_glam0
      REAL(dp) ::   dl_gphi0

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      CALL logger_info('GRID HGR FILL : beta-plane with regular grid-spacing '//&
         &  'and rotated domain (GYRE configuration)')

      ! Position coordinates (in kilometers)
      !
      ! angle 45deg and ze1=106.e+3 / jp_cfg forced -> zlam1 = -85deg, zphi1 = 29degN
      zlam1 = -85
      zphi1 = 29
      ! resolution in meters
      ze1 = 106000. / FLOAT(td_nam%i_cfg)            
      ! benchmark: forced the resolution to be about 100 km
      IF( td_nam%l_bench )   ze1 = 106000.e0     
      zsin_alpha = - SQRT( 2. ) / 2.
      zcos_alpha =   SQRT( 2. ) / 2.
      ze1deg = ze1 / (dp_rearth * dp_deg2rad)
      dl_glam0 = zlam1 + zcos_alpha * ze1deg * FLOAT( jpj-2 )
      dl_gphi0 = zphi1 + zsin_alpha * ze1deg * FLOAT( jpj-2 )

      DO jj = 1, jpj
         DO ji = 1, jpi
            zim1 = FLOAT( ji - 1 )   ;   zim05 = FLOAT( ji ) - 1.5
            zjm1 = FLOAT( jj - 1 )   ;   zjm05 = FLOAT( jj ) - 1.5

            tg_glamf%d_value(ji,jj,1,1) = dl_glam0 &
                                        &  + zim1  * ze1deg * zcos_alpha &
                                        &  + zjm1  * ze1deg * zsin_alpha
            tg_gphif%d_value(ji,jj,1,1) = dl_gphi0 &
                                        & - zim1  * ze1deg * zsin_alpha &
                                        & + zjm1  * ze1deg * zcos_alpha

            tg_glamt%d_value(ji,jj,1,1) = dl_glam0 &
                                        & + zim05 * ze1deg * zcos_alpha &
                                        & + zjm05 * ze1deg * zsin_alpha
            tg_gphit%d_value(ji,jj,1,1) = dl_gphi0 &
                                        & - zim05 * ze1deg * zsin_alpha &
                                        & + zjm05 * ze1deg * zcos_alpha

            tg_glamu%d_value(ji,jj,1,1) = dl_glam0 &
                                        & + zim1  * ze1deg * zcos_alpha &
                                        & + zjm05 * ze1deg * zsin_alpha
            tg_gphiu%d_value(ji,jj,1,1) = dl_gphi0 & 
                                        & - zim1  * ze1deg * zsin_alpha &
                                        & + zjm05 * ze1deg * zcos_alpha

            tg_glamv%d_value(ji,jj,1,1) = dl_glam0 &
                                        & + zim05 * ze1deg * zcos_alpha &
                                        & + zjm1  * ze1deg * zsin_alpha
            tg_gphiv%d_value(ji,jj,1,1) = dl_gphi0 &
                                        & - zim05 * ze1deg * zsin_alpha &
                                        & + zjm1  * ze1deg * zcos_alpha

         END DO
      END DO

      ! Horizontal scale factors (in meters)
      !                              ======
      tg_e1t%d_value(:,:,1,1) = ze1      
      tg_e1u%d_value(:,:,1,1) = ze1      
      tg_e1v%d_value(:,:,1,1) = ze1      
      tg_e1f%d_value(:,:,1,1) = ze1      

      tg_e2t%d_value(:,:,1,1) = ze1
      tg_e2u%d_value(:,:,1,1) = ze1
      tg_e2v%d_value(:,:,1,1) = ze1
      tg_e2f%d_value(:,:,1,1) = ze1

   END SUBROUTINE grid_hgr__fill_gyre
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_hgr__fill_coriolis(td_nam, jpi)!,jpj) 
   !-------------------------------------------------------------------
   !> @brief This subroutine fill coriolis factor
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial version
   !> @date October, 2016
   !> - compute coriolis factor at f-point and at t-point
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   ! @param[in] jpj
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMH), INTENT(IN) :: td_nam
      INTEGER(i4), INTENT(IN) :: jpi
!      INTEGER(i4), INTENT(IN) :: jpj

      ! local variable
      REAL(dp) :: zbeta
      REAL(dp) :: zphi0  
      REAL(dp) :: zf0

      ! loop indices
      !----------------------------------------------------------------

      !  Coriolis factor
      SELECT CASE( td_nam%i_mshhgr )   ! type of horizontal mesh

         CASE ( 0, 1, 4 ) ! mesh on the sphere

            tg_ff_f%d_value(:,:,1,1) = 2. * dp_omega * SIN(dp_deg2rad * tg_gphif%d_value(:,:,1,1))
            tg_ff_t%d_value(:,:,1,1) = 2. * dp_omega * SIN(dp_deg2rad * tg_gphit%d_value(:,:,1,1)) ! at t-point

         CASE ( 2 )  ! f-plane at ppgphi0 

            tg_ff_f%d_value(:,:,1,1) = 2. * dp_omega * SIN( dp_deg2rad * td_nam%d_ppgphi0 )
            tg_ff_t%d_value(:,:,1,1) = 2. * dp_omega * SIN( dp_deg2rad * td_nam%d_ppgphi0 )
            CALL logger_info('f-plane: Coriolis parameter = constant = '//&
               &  TRIM(fct_str(tg_ff_f%d_value(1,1,1,1))) )

         CASE ( 3 )  ! beta-plane

            ! beta at latitude ppgphi0
            zbeta = 2. * dp_omega * COS( dp_deg2rad * td_nam%d_ppgphi0 ) / dp_rearth
            ! latitude of the first row F-points
!            zphi0 = td_nam%d_ppgphi0 - FLOAT( jpi/2 ) * td_nam%d_ppe2_m / ( dp_rearth * dp_deg2rad )
            zphi0 = td_nam%d_ppgphi0 - FLOAT( jpi/2 ) * td_nam%d_ppe2_deg / ( dp_rearth * dp_deg2rad )

            ! compute f0 1st point south
            zf0 = 2. * dp_omega * SIN( dp_deg2rad * zphi0 )
            ! f = f0 +beta* y ( y=0 at south)
            tg_ff_f%d_value(:,:,1,1) = zf0 + zbeta * tg_gphif%d_value(:,:,1,1) * 1.e3
            tg_ff_t%d_value(:,:,1,1) = zf0 + zbeta * tg_gphit%d_value(:,:,1,1) * 1.e3

         CASE ( 5 )  ! beta-plane and rotated domain (gyre configuration)

            ! beta at latitude ppgphi0
            zbeta = 2. * dp_omega * COS( dp_deg2rad * td_nam%d_ppgphi0 ) / dp_rearth
            ! latitude of the first row F-points
            zphi0 = 15.e0
            ! compute f0 1st point south
            zf0   = 2. * dp_omega * SIN( dp_deg2rad * zphi0 )

            ! f = f0 +beta* y ( y=0 at south)
            tg_ff_f%d_value(:,:,1,1) = ( zf0 + zbeta * ABS( tg_gphif%d_value(:,:,1,1) - zphi0 ) &
                                     & * dp_deg2rad * dp_rearth )
            tg_ff_t%d_value(:,:,1,1) = ( zf0 + zbeta * ABS( tg_gphit%d_value(:,:,1,1) - zphi0 ) &
                                     & * dp_deg2rad * dp_rearth )

      END SELECT

   END SUBROUTINE grid_hgr__fill_coriolis
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_hgr__angle(td_nam, jpi, jpj)
   !!----------------------------------------------------------------------
   !> @brief This subroutine compute angles between model grid lines and the North direction
   !>
   !> @details
   !> ** Method  :
   !>
   !> ** Action  :   Compute (gsint, gcost, gsinu, gcosu, gsinv, gcosv, gsinf, gcosf) arrays:
   !>      sinus and cosinus of the angle between the north-south axe and the 
   !>      j-direction at t, u, v and f-points
   !>
   !> History :
   !>   7.0  !  96-07  (O. Marti )  Original code
   !>   8.0  !  98-06  (G. Madec )
   !>   8.5  !  98-06  (G. Madec )  Free form, F90 + opt.
   !>   9.2  !  07-04  (S. Masson)  Add T, F points and bugfix in cos lateral boundary
   !>
   !> @author J.Paul
   !> @date September, 2015 - rewrite from geo2ocean
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   !> @param[in] jpj
   !!----------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TNAMH), INTENT(IN) :: td_nam
      INTEGER(i4), INTENT(IN) :: jpi
      INTEGER(i4), INTENT(IN) :: jpj

      ! local variable
      REAL(dp) :: zlam, zphi          
      REAL(dp) :: zlan, zphh          
      REAL(dp) :: zxnpt, zynpt, znnpt ! x,y components and norm of the vector: T point to North Pole
      REAL(dp) :: zxnpu, zynpu, znnpu ! x,y components and norm of the vector: U point to North Pole
      REAL(dp) :: zxnpv, zynpv, znnpv ! x,y components and norm of the vector: V point to North Pole
      REAL(dp) :: zxnpf, zynpf, znnpf ! x,y components and norm of the vector: F point to North Pole
      REAL(dp) :: zxvvt, zyvvt, znvvt ! x,y components and norm of the vector: between V points below and above a T point
      REAL(dp) :: zxffu, zyffu, znffu ! x,y components and norm of the vector: between F points below and above a U point
      REAL(dp) :: zxffv, zyffv, znffv ! x,y components and norm of the vector: between F points left  and right a V point
      REAL(dp) :: zxuuf, zyuuf, znuuf ! x,y components and norm of the vector: between U points below and above a F point

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !!----------------------------------------------------------------------

      ! ============================= !
      ! Compute the cosinus and sinus !
      ! ============================= !
      ! (computation done on the north stereographic polar plane)

      DO jj = 2, jpj-1
!CDIR NOVERRCHK
         DO ji = 2, jpi   ! vector opt.

            ! north pole direction & modulous (at t-point)
            zlam = tg_glamt%d_value(ji,jj,1,1)
            zphi = tg_gphit%d_value(ji,jj,1,1)
            zxnpt = 0._dp - 2._dp * COS( dp_deg2rad*zlam ) * TAN( dp_pi/4._dp - dp_deg2rad*zphi/2._dp )
            zynpt = 0._dp - 2._dp * SIN( dp_deg2rad*zlam ) * TAN( dp_pi/4._dp - dp_deg2rad*zphi/2._dp )
            znnpt = zxnpt*zxnpt + zynpt*zynpt

            ! north pole direction & modulous (at u-point)
            zlam = tg_glamu%d_value(ji,jj,1,1)
            zphi = tg_gphiu%d_value(ji,jj,1,1)
            zxnpu = 0._dp - 2._dp * COS( dp_deg2rad*zlam ) * TAN( dp_pi/4._dp - dp_deg2rad*zphi/2._dp )
            zynpu = 0._dp - 2._dp * SIN( dp_deg2rad*zlam ) * TAN( dp_pi/4._dp - dp_deg2rad*zphi/2._dp )
            znnpu = zxnpu*zxnpu + zynpu*zynpu

            ! north pole direction & modulous (at v-point)
            zlam = tg_glamv%d_value(ji,jj,1,1)
            zphi = tg_gphiv%d_value(ji,jj,1,1)
            zxnpv = 0._dp - 2._dp * COS( dp_deg2rad*zlam ) * TAN( dp_pi/4._dp - dp_deg2rad*zphi/2._dp )
            zynpv = 0._dp - 2._dp * SIN( dp_deg2rad*zlam ) * TAN( dp_pi/4._dp - dp_deg2rad*zphi/2._dp )
            znnpv = zxnpv*zxnpv + zynpv*zynpv

            ! north pole direction & modulous (at f-point)
            zlam = tg_glamf%d_value(ji,jj,1,1)
            zphi = tg_gphif%d_value(ji,jj,1,1)
            zxnpf = 0._dp - 2._dp * COS( dp_deg2rad*zlam ) * TAN( dp_pi/4._dp - dp_deg2rad*zphi/2._dp )
            zynpf = 0._dp - 2._dp * SIN( dp_deg2rad*zlam ) * TAN( dp_pi/4._dp - dp_deg2rad*zphi/2._dp )
            znnpf = zxnpf*zxnpf + zynpf*zynpf

            ! j-direction: v-point segment direction (around t-point)
            zlam = tg_glamv%d_value(ji,jj  ,1,1)
            zphi = tg_gphiv%d_value(ji,jj  ,1,1)
            zlan = tg_glamv%d_value(ji,jj-1,1,1)
            zphh = tg_gphiv%d_value(ji,jj-1,1,1)
            zxvvt =  2._dp * COS( dp_deg2rad*zlam ) * TAN( dp_pi/4._dp - dp_deg2rad*zphi/2._dp )   &
               &  -  2._dp * COS( dp_deg2rad*zlan ) * TAN( dp_pi/4._dp - dp_deg2rad*zphh/2._dp )
            zyvvt =  2._dp * SIN( dp_deg2rad*zlam ) * TAN( dp_pi/4._dp - dp_deg2rad*zphi/2._dp )   &
               &  -  2._dp * SIN( dp_deg2rad*zlan ) * TAN( dp_pi/4._dp - dp_deg2rad*zphh/2._dp )
            znvvt = SQRT( znnpt * ( zxvvt*zxvvt + zyvvt*zyvvt )  )
            znvvt = MAX( znvvt, dp_eps )

            ! j-direction: f-point segment direction (around u-point)
            zlam = tg_glamf%d_value(ji,jj  ,1,1)
            zphi = tg_gphif%d_value(ji,jj  ,1,1)
            zlan = tg_glamf%d_value(ji,jj-1,1,1)
            zphh = tg_gphif%d_value(ji,jj-1,1,1)
            zxffu =  2._dp * COS( dp_deg2rad*zlam ) * TAN( dp_pi/4._dp - dp_deg2rad*zphi/2._dp )   &
               &  -  2._dp * COS( dp_deg2rad*zlan ) * TAN( dp_pi/4._dp - dp_deg2rad*zphh/2._dp )
            zyffu =  2._dp * SIN( dp_deg2rad*zlam ) * TAN( dp_pi/4._dp - dp_deg2rad*zphi/2._dp )   &
               &  -  2._dp * SIN( dp_deg2rad*zlan ) * TAN( dp_pi/4._dp - dp_deg2rad*zphh/2._dp )
            znffu = SQRT( znnpu * ( zxffu*zxffu + zyffu*zyffu )  )
            znffu = MAX( znffu, dp_eps )

            ! i-direction: f-point segment direction (around v-point)
            zlam = tg_glamf%d_value(ji  ,jj,1,1)
            zphi = tg_gphif%d_value(ji  ,jj,1,1)
            zlan = tg_glamf%d_value(ji-1,jj,1,1)
            zphh = tg_gphif%d_value(ji-1,jj,1,1)
            zxffv =  2._dp * COS( dp_deg2rad*zlam ) * TAN( dp_pi/4._dp - dp_deg2rad*zphi/2._dp )   &
               &  -  2._dp * COS( dp_deg2rad*zlan ) * TAN( dp_pi/4._dp - dp_deg2rad*zphh/2._dp )
            zyffv =  2._dp * SIN( dp_deg2rad*zlam ) * TAN( dp_pi/4._dp - dp_deg2rad*zphi/2._dp )   &
               &  -  2._dp * SIN( dp_deg2rad*zlan ) * TAN( dp_pi/4._dp - dp_deg2rad*zphh/2._dp )
            znffv = SQRT( znnpv * ( zxffv*zxffv + zyffv*zyffv )  )
            znffv = MAX( znffv, dp_eps )

            ! j-direction: u-point segment direction (around f-point)
            zlam = tg_glamu%d_value(ji,jj+1,1,1)
            zphi = tg_gphiu%d_value(ji,jj+1,1,1)
            zlan = tg_glamu%d_value(ji,jj  ,1,1)
            zphh = tg_gphiu%d_value(ji,jj  ,1,1)
            zxuuf =  2._dp * COS( dp_deg2rad*zlam ) * TAN( dp_pi/4._dp - dp_deg2rad*zphi/2._dp )   &
               &  -  2._dp * COS( dp_deg2rad*zlan ) * TAN( dp_pi/4._dp - dp_deg2rad*zphh/2._dp )
            zyuuf =  2._dp * SIN( dp_deg2rad*zlam ) * TAN( dp_pi/4._dp - dp_deg2rad*zphi/2._dp )   &
               &  -  2._dp * SIN( dp_deg2rad*zlan ) * TAN( dp_pi/4._dp - dp_deg2rad*zphh/2._dp )
            znuuf = SQRT( znnpf * ( zxuuf*zxuuf + zyuuf*zyuuf )  )
            znuuf = MAX( znuuf, dp_eps )

            ! cosinus and sinus using scalar and vectorial products
            tg_gsint%d_value(ji,jj,1,1) = ( zxnpt*zyvvt - zynpt*zxvvt ) / znvvt
            tg_gcost%d_value(ji,jj,1,1) = ( zxnpt*zxvvt + zynpt*zyvvt ) / znvvt

            tg_gsinu%d_value(ji,jj,1,1) = ( zxnpu*zyffu - zynpu*zxffu ) / znffu
            tg_gcosu%d_value(ji,jj,1,1) = ( zxnpu*zxffu + zynpu*zyffu ) / znffu

            tg_gsinf%d_value(ji,jj,1,1) = ( zxnpf*zyuuf - zynpf*zxuuf ) / znuuf
            tg_gcosf%d_value(ji,jj,1,1) = ( zxnpf*zxuuf + zynpf*zyuuf ) / znuuf

            ! (caution, rotation of 90 degres)
            tg_gsinv%d_value(ji,jj,1,1) = ( zxnpv*zxffv + zynpv*zyffv ) / znffv
            tg_gcosv%d_value(ji,jj,1,1) =-( zxnpv*zyffv - zynpv*zxffv ) / znffv

         END DO
      END DO

      ! =============== !
      ! Geographic mesh !
      ! =============== !

      DO jj = 2, jpj-1
         DO ji = 2, jpi   ! vector opt.
            IF( MOD( ABS( tg_glamv%d_value(ji,jj,1,1) - tg_glamv%d_value(ji,jj-1,1,1) ), 360._dp ) < 1.e-8 ) THEN
               tg_gsint%d_value(ji,jj,1,1) = 0._dp
               tg_gcost%d_value(ji,jj,1,1) = 1._dp
            ENDIF
            IF( MOD( ABS( tg_glamf%d_value(ji,jj,1,1) - tg_glamf%d_value(ji,jj-1,1,1) ), 360._dp ) < 1.e-8 ) THEN
               tg_gsinu%d_value(ji,jj,1,1) = 0._dp
               tg_gcosu%d_value(ji,jj,1,1) = 1._dp
            ENDIF
            IF(      ABS( tg_gphif%d_value(ji,jj,1,1) - tg_gphif%d_value(ji-1,jj,1,1) )            < 1.e-8 ) THEN
               tg_gsinv%d_value(ji,jj,1,1) = 0._dp
               tg_gcosv%d_value(ji,jj,1,1) = 1._dp
            ENDIF
            IF( MOD( ABS( tg_glamu%d_value(ji,jj,1,1) - tg_glamu%d_value(ji,jj+1,1,1) ), 360._dp ) < 1.e-8 ) THEN
               tg_gsinf%d_value(ji,jj,1,1) = 0._dp
               tg_gcosf%d_value(ji,jj,1,1) = 1._dp
            ENDIF
         END DO
      END DO

      ! =========================== !
      ! Lateral boundary conditions !
      ! =========================== !

      ! lateral boundary cond.: T-, U-, V-, F-pts, sgn
      CALL lbc_lnk( tg_gcost%d_value(:,:,1,1), 'T', td_nam%i_perio, -1._dp )   
      CALL lbc_lnk( tg_gcosu%d_value(:,:,1,1), 'U', td_nam%i_perio, -1._dp )   
      CALL lbc_lnk( tg_gcosv%d_value(:,:,1,1), 'V', td_nam%i_perio, -1._dp )   
      CALL lbc_lnk( tg_gcosf%d_value(:,:,1,1), 'F', td_nam%i_perio, -1._dp )   

      CALL lbc_lnk( tg_gsint%d_value(:,:,1,1), 'T', td_nam%i_perio, -1._dp )
      CALL lbc_lnk( tg_gsinu%d_value(:,:,1,1), 'U', td_nam%i_perio, -1._dp )
      CALL lbc_lnk( tg_gsinv%d_value(:,:,1,1), 'V', td_nam%i_perio, -1._dp )
      CALL lbc_lnk( tg_gsinf%d_value(:,:,1,1), 'F', td_nam%i_perio, -1._dp )

   END SUBROUTINE grid_hgr__angle
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE grid_hgr
