!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief This module manage Vertical grid.
!>
!> @details
!> ** Purpose :   set the depth of model levels and the resulting 
!>              vertical scale factors.
!>
!> ** Method  : 
!>    - reference 1D vertical coordinate (gdep._1d, e3._1d)
!>    - read/set ocean depth and ocean levels (bathy, mbathy)
!>    - vertical coordinate (gdep., e3.) depending on the coordinate chosen :
!>       - ln_zco=T   z-coordinate   
!>       - ln_zps=T   z-coordinate with partial steps
!>       - ln_zco=T   s-coordinate 
!>
!> ** Action  :   define gdep., e3., mbathy and bathy
!>
!> @author
!> G, Madec
!>
!> @date December, 1995 - Original code : s vertical coordinate
!> @date July, 1997
!> - lbc_lnk call
!> @date September, 2002
!> - A. Bozec, G. Madec : F90: Free form and module
!> @date September, 2002
!> - A. de Miranda : rigid-lid + islands
!> @date August, 2003
!> - G. Madec : Free form and module
!> @date October, 2005
!> - A. Beckmann : modifications for hybrid s-ccordinates & new stretching function
!> @date April, 2006
!> - R. Benshila, G. Madec : add zgr_zco
!> @date June, 2008
!> - G. Madec : insertion of domzgr_zps.h90 & conding style
!> @date July, 2009
!> - R. Benshila : Suppression of rigid-lid option
!> @date November, 2011
!> - G. Madec : add mbk. arrays associated to the deepest ocean level
!> @date August, 2012
!> - J. Siddorn : added Siddorn and Furner stretching function
!> @date December, 2012
!> - R. Bourdalle-Badie and G. Reffray : modify C1D case
!> @date November, 2014
!> - P. Mathiot and C. Harris : add ice shelf capabilitye
!> @date November, 2015
!> - H. Liu : Modifications for Wetting/Drying
!> @date October, 2016
!> - J, Paul : update from trunk (revision 6961): add wetting and drying, ice sheet coupling..
!> - J, Paul : do not use anymore special case for ORCA grid.
!> @date November, 2016
!> - J, Paul : vertical scale factors e3. = dk[gdep] or old definition
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE grid_zgr

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
   USE grid_hgr                        ! Horizontal mesh

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible
   ! type and variable
   PUBLIC :: TNAMZ

   PUBLIC :: tg_gdepw_1d
   PUBLIC :: tg_gdept_1d
   PUBLIC :: tg_e3w_1d
   PUBLIC :: tg_e3t_1d
   PUBLIC :: tg_e3tp 
   PUBLIC :: tg_e3wp 

   PUBLIC :: tg_rx1 
   
   PUBLIC :: tg_mbathy
   PUBLIC :: tg_misfdep

   PUBLIC :: tg_gdept_0 
   PUBLIC :: tg_gdepw_0
!   PUBLIC :: tg_gdep3w_0  !useless to create meshmask
   PUBLIC :: tg_e3t_0
   PUBLIC :: tg_e3u_0
   PUBLIC :: tg_e3v_0
   PUBLIC :: tg_e3w_0
   PUBLIC :: tg_e3f_0     !useless to create meshmask
   PUBLIC :: tg_e3uw_0    !useless to create meshmask
   PUBLIC :: tg_e3vw_0    !useless to create meshmask
   
   PUBLIC :: tg_mbkt
!   PUBLIC :: tg_mbku      !useless to create meshmask
!   PUBLIC :: tg_mbkv      !useless to create meshmask
   PUBLIC :: tg_mikt
!   PUBLIC :: tg_miku      !useless to create meshmask
!   PUBLIC :: tg_mikv      !useless to create meshmask
!   PUBLIC :: tg_mikf      !useless to create meshmask

   PUBLIC :: tg_hbatt     !            sco
   PUBLIC :: tg_hbatu     !            sco
   PUBLIC :: tg_hbatv     !            sco
   PUBLIC :: tg_hbatf     !            sco

   PUBLIC :: tg_gsigt     !            sco(tanh)
   PUBLIC :: tg_gsigw     !            sco(tanh)
   PUBLIC :: tg_gsi3w     !            sco(tanh)
   PUBLIC :: tg_esigt     !            sco(tanh)
   PUBLIC :: tg_esigw     !            sco(tanh)

   ! function and subroutine
   PUBLIC :: grid_zgr_init 
   PUBLIC :: grid_zgr_nam
   PUBLIC :: grid_zgr_fill
   PUBLIC :: grid_zgr_clean

   PUBLIC :: grid_zgr_zps_init 
   PUBLIC :: grid_zgr_zps_clean
   PUBLIC :: grid_zgr_sco_init
   PUBLIC :: grid_zgr_sco_clean
   PUBLIC :: grid_zgr_sco_stiff 

   PRIVATE :: grid_zgr__z
   PRIVATE :: grid_zgr__bat 
   PRIVATE :: grid_zgr__zco 
!   PRIVATE :: grid_zgr__bat_zoom
   PRIVATE :: grid_zgr__bat_ctl
   PRIVATE :: grid_zgr__bot_level
   PRIVATE :: grid_zgr__top_level
   PRIVATE :: grid_zgr__zps_fill
   PRIVATE :: grid_zgr__isf_fill
!   PRIVATE :: grid_zgr__isf_fill_e3x
   PRIVATE :: grid_zgr__isf_fill_e3uw
!   PRIVATE :: grid_zgr__isf_fill_gdep3w_0
   PRIVATE :: grid_zgr__sco_fill
   PRIVATE :: grid_zgr__sco_s_sh94 
   PRIVATE :: grid_zgr__sco_s_sf12
   PRIVATE :: grid_zgr__sco_s_tanh
   PRIVATE :: grid_zgr__sco_fssig      !: tanh stretch function
   PRIVATE :: grid_zgr__sco_fssig1     !: Song and Haidvogel 1994 stretch function
   PRIVATE :: grid_zgr__sco_fgamma     !: Siddorn and Furner 2012 stretching function 

   TYPE TNAMZ

      CHARACTER(LEN=lc) :: c_coord   
      INTEGER(i4)       :: i_perio   
                
      LOGICAL           :: l_zco               
      LOGICAL           :: l_zps      
      LOGICAL           :: l_sco      
      LOGICAL           :: l_isfcav   
      LOGICAL           :: l_iscpl   
      LOGICAL           :: l_wd   
      INTEGER(i4)       :: i_nlevel   
                  
      REAL(dp)          :: d_ppsur    
      REAL(dp)          :: d_ppa0     
      REAL(dp)          :: d_ppa1     
      REAL(dp)          :: d_ppkth    
      REAL(dp)          :: d_ppacr    
      REAL(dp)          :: d_ppdzmin  
      REAL(dp)          :: d_pphmax   
      LOGICAL           :: l_dbletanh 
      REAL(dp)          :: d_ppa2     
      REAL(dp)          :: d_ppkth2   
      REAL(dp)          :: d_ppacr2   
                  
      REAL(dp)          :: d_hmin     
      REAL(dp)          :: d_isfhmin

      REAL(dp)          :: d_e3zps_min
      REAL(dp)          :: d_e3zps_rat
!      INTEGER(i4)       :: i_msh      
                  
      LOGICAL           :: l_s_sh94   
      LOGICAL           :: l_s_sf12   
      REAL(dp)          :: d_sbot_min 
      REAL(dp)          :: d_sbot_max 
      ! Song and Haidvogel 1994 stretching additional parameters
      REAL(dp)          :: d_rmax     
      REAL(dp)          :: d_hc       
      REAL(dp)          :: d_theta    
      REAL(dp)          :: d_thetb    
      REAL(dp)          :: d_bb       
      ! Siddorn and Furner stretching additional parameters 
      LOGICAL           :: l_sigcrit  
      REAL(dp)          :: d_alpha    
      REAL(dp)          :: d_efold    
      REAL(dp)          :: d_zs       
      REAL(dp)          :: d_zb_a     
      REAL(dp)          :: d_zb_b     
                  
      INTEGER(i4)       :: i_cla     

      REAL(dp)          :: d_wdmin1 
      REAL(dp)          :: d_wdmin2 
      REAL(dp)          :: d_wdld 

!      CHARACTER(LEN=lc) :: c_cfg     
!      INTEGER(i4)       :: i_cfg     
!      INTEGER(i4)       :: i_bench   
!      LOGICAL           :: l_zoom
      LOGICAL           :: l_c1d
      LOGICAL           :: l_e3_dep
                
!      CHARACTER(LEN=lc) :: c_cfz      
!      INTEGER(i4)       :: i_izoom    
!      INTEGER(i4)       :: i_jzoom    
!      LOGICAL           :: l_zoom_s   
!      LOGICAL           :: l_zoom_e   
!      LOGICAL           :: l_zoom_w   
!      LOGICAL           :: l_zoom_n

   END TYPE

   TYPE(TVAR), SAVE :: tg_gdepw_1d  !zco & zps & sco
   TYPE(TVAR), SAVE :: tg_gdept_1d  !zco & zps & sco
   TYPE(TVAR), SAVE :: tg_e3w_1d    !zco & zps
   TYPE(TVAR), SAVE :: tg_e3t_1d    !zco & zps
   TYPE(TVAR), SAVE :: tg_e3tp      !      zps
   TYPE(TVAR), SAVE :: tg_e3wp      !      zps
   
   TYPE(TVAR), SAVE :: tg_rx1       !            sco
   
   TYPE(TVAR), SAVE :: tg_mbathy    !zco & zps & sco
   TYPE(TVAR), SAVE :: tg_misfdep

   TYPE(TVAR), SAVE :: tg_gdept_0   !      zps & sco
   TYPE(TVAR), SAVE :: tg_gdepw_0   !      zps & sco
   !TYPE(TVAR), SAVE :: tg_gdep3w_0
   TYPE(TVAR), SAVE :: tg_e3t_0     !      zps & sco
   TYPE(TVAR), SAVE :: tg_e3u_0     !      zps & sco
   TYPE(TVAR), SAVE :: tg_e3v_0     !      zps & sco
   TYPE(TVAR), SAVE :: tg_e3w_0     !      zps & sco
   TYPE(TVAR), SAVE :: tg_e3f_0
   TYPE(TVAR), SAVE :: tg_e3uw_0
   TYPE(TVAR), SAVE :: tg_e3vw_0
   
   TYPE(TVAR), SAVE :: tg_mbkt      !zco & zps & sco
   !TYPE(TVAR), SAVE :: tg_mbku
   !TYPE(TVAR), SAVE :: tg_mbkv
   TYPE(TVAR), SAVE :: tg_mikt      !zco & zps & sco
   !TYPE(TVAR), SAVE :: tg_miku
   !TYPE(TVAR), SAVE :: tg_mikv
   !TYPE(TVAR), SAVE :: tg_mikf

   TYPE(TVAR), SAVE :: tg_hbatt     !            sco
   TYPE(TVAR), SAVE :: tg_hbatu     !            sco
   TYPE(TVAR), SAVE :: tg_hbatv     !            sco
   TYPE(TVAR), SAVE :: tg_hbatf     !            sco

   TYPE(TVAR), SAVE :: tg_gsigt     !            sco(tanh)
   TYPE(TVAR), SAVE :: tg_gsigw     !            sco(tanh)
   TYPE(TVAR), SAVE :: tg_gsi3w     !            sco(tanh)
   TYPE(TVAR), SAVE :: tg_esigt     !            sco(tanh)
   TYPE(TVAR), SAVE :: tg_esigw     !            sco(tanh)

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr_init(jpi, jpj, jpk, ld_sco) 
   !-------------------------------------------------------------------
   !> @brief This subroutine initialise global variable needed to compute vertical
   !>        mesh
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial version
   !>
   !> @param[in] jpi
   !> @param[in] jpj
   !> @param[in] jpk
   !> @param[in] ld_sco
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      INTEGER(i4), INTENT(IN) :: jpi
      INTEGER(i4), INTENT(IN) :: jpj
      INTEGER(i4), INTENT(IN) :: jpk
      LOGICAL    , INTENT(IN) :: ld_sco

      ! local variable
      REAL(dp), DIMENSION(jpk)         :: dl_tmp1D
      REAL(dp), DIMENSION(jpi,jpj)     :: dl_tmp2D
      REAL(dp), DIMENSION(jpi,jpj,jpk) :: dl_tmp3D
      ! loop indices
      !----------------------------------------------------------------

      ! variable 1D
      dl_tmp1D(:)    =dp_fill

      tg_gdepw_1d=var_init('gdepw_1d',dl_tmp1D(:)    , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_gdept_1d=var_init('gdept_1d',dl_tmp1D(:)    , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_e3w_1d  =var_init('e3w_1d  ',dl_tmp1D(:)    , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_e3t_1d  =var_init('e3t_1d  ',dl_tmp1D(:)    , dd_fill=dp_fill, id_type=NF90_DOUBLE)

      !only sco
      IF( ld_sco )THEN
         tg_gsigt   =var_init('gsigt   ',dl_tmp1D(:)    , dd_fill=dp_fill, id_type=NF90_DOUBLE)
         tg_gsigw   =var_init('gsigw   ',dl_tmp1D(:)    , dd_fill=dp_fill, id_type=NF90_DOUBLE)
         tg_gsi3w   =var_init('gsi3w   ',dl_tmp1D(:)    , dd_fill=dp_fill, id_type=NF90_DOUBLE)
         tg_esigt   =var_init('esigt   ',dl_tmp1D(:)    , dd_fill=dp_fill, id_type=NF90_DOUBLE)
         tg_esigw   =var_init('esigw   ',dl_tmp1D(:)    , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      ENDIF

      ! variable 2D
      dl_tmp2D(:,:)  =dp_fill_i2

      tg_mbkt    =var_init('mbkt    ',dl_tmp2D(:,:)  , dd_fill=dp_fill_i2, id_type=NF90_SHORT)
      !tg_mbku    =var_init('mbku    ',dl_tmp2D(:,:)  , dd_fill=dp_fill_i2, id_type=NF90_SHORT)
      !tg_mbkv    =var_init('mbkv    ',dl_tmp2D(:,:)  , dd_fill=dp_fill_i2, id_type=NF90_SHORT)
      tg_mikt    =var_init('mikt    ',dl_tmp2D(:,:)  , dd_fill=dp_fill_i2, id_type=NF90_SHORT)
      !tg_miku    =var_init('miku    ',dl_tmp2D(:,:)  , dd_fill=dp_fill_i2, id_type=NF90_SHORT)
      !tg_mikv    =var_init('mikv    ',dl_tmp2D(:,:)  , dd_fill=dp_fill_i2, id_type=NF90_SHORT)
      !tg_mikf    =var_init('mikf    ',dl_tmp2D(:,:)  , dd_fill=dp_fill_i2, id_type=NF90_SHORT)

      dl_tmp2D(:,:)  =dp_fill_i4

      tg_mbathy  =var_init('mbathy  ',dl_tmp2D(:,:)  , dd_fill=dp_fill_i4, id_type=NF90_INT)
      tg_misfdep =var_init('misfdep ',dl_tmp2D(:,:)  , dd_fill=dp_fill_i4, id_type=NF90_INT)

      dl_tmp2D(:,:)  =dp_fill

      ! only sco
      IF( ld_sco )THEN
         tg_hbatt   =var_init('hbatt   ',dl_tmp2D(:,:)  , dd_fill=dp_fill, id_type=NF90_DOUBLE)
         tg_hbatu   =var_init('hbatu   ',dl_tmp2D(:,:)  , dd_fill=dp_fill, id_type=NF90_DOUBLE)
         tg_hbatv   =var_init('hbatv   ',dl_tmp2D(:,:)  , dd_fill=dp_fill, id_type=NF90_DOUBLE)
         tg_hbatf   =var_init('hbatf   ',dl_tmp2D(:,:)  , dd_fill=dp_fill, id_type=NF90_DOUBLE)
      ENDIF

      ! variable 3D
      dl_tmp3D(:,:,:)=dp_fill

      tg_gdept_0 =var_init('gdept_0 ',dl_tmp3D(:,:,:), dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_gdepw_0 =var_init('gdepw_0 ',dl_tmp3D(:,:,:), dd_fill=dp_fill, id_type=NF90_DOUBLE)
      !tg_gdep3w_0=var_init('gdep3w_0',dl_tmp3D(:,:,:), dd_fill=dp_fill_sp, id_type=NF90_FLOAT)

      tg_e3t_0   =var_init('e3t_0   ',dl_tmp3D(:,:,:), dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_e3u_0   =var_init('e3u_0   ',dl_tmp3D(:,:,:), dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_e3v_0   =var_init('e3v_0   ',dl_tmp3D(:,:,:), dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_e3w_0   =var_init('e3w_0   ',dl_tmp3D(:,:,:), dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_e3f_0   =var_init('e3f_0   ',dl_tmp3D(:,:,:), dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_e3uw_0  =var_init('e3uw_0  ',dl_tmp3D(:,:,:), dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_e3vw_0  =var_init('e3vw_0  ',dl_tmp3D(:,:,:), dd_fill=dp_fill, id_type=NF90_DOUBLE)

   END SUBROUTINE grid_zgr_init
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr_clean(ld_sco) 
   !-------------------------------------------------------------------
   !> @brief This subroutine clean hgr structure
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial version
   !>
   !> @param[in] ld_sco
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      LOGICAL    , INTENT(IN) :: ld_sco

      ! local variable
      ! loop indices
      !----------------------------------------------------------------

      CALL var_clean(tg_gdepw_1d)
      CALL var_clean(tg_gdept_1d)
      CALL var_clean(tg_e3w_1d  )
      CALL var_clean(tg_e3t_1d  )

      IF( ld_sco )THEN
         CALL var_clean(tg_gsigt   )
         CALL var_clean(tg_gsigw   )
         CALL var_clean(tg_gsi3w   )
         CALL var_clean(tg_esigt   )
         CALL var_clean(tg_esigw   )
      ENDIF
      
      CALL var_clean(tg_mbathy  )
      CALL var_clean(tg_misfdep )

      CALL var_clean(tg_mbkt    )
      !CALL var_clean(tg_mbku    )
      !CALL var_clean(tg_mbkv    )
      CALL var_clean(tg_mikt    )
      !CALL var_clean(tg_miku    )
      !CALL var_clean(tg_mikv    )
      !CALL var_clean(tg_mikf    )

      IF( ld_sco )THEN
         CALL var_clean(tg_hbatt   )
         CALL var_clean(tg_hbatu   )
         CALL var_clean(tg_hbatv   )
         CALL var_clean(tg_hbatf   )
      ENDIF

      CALL var_clean(tg_gdept_0 )
      CALL var_clean(tg_gdepw_0 )
      !CALL var_clean(tg_gdep3w_0)

      CALL var_clean(tg_e3t_0   )
      CALL var_clean(tg_e3u_0   )
      CALL var_clean(tg_e3v_0   )
      CALL var_clean(tg_e3f_0   )
      CALL var_clean(tg_e3uw_0  )
      CALL var_clean(tg_e3vw_0  )
      
   END SUBROUTINE grid_zgr_clean
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid_zgr_nam(cd_coord, id_perio, cd_namelist) &
         & RESULT (tf_namz)
   !-------------------------------------------------------------------
   !> @brief This function initialise zgr namelist structure
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
      TYPE(TNAMZ)                  :: tf_namz

      ! local variable
      INTEGER(i4)        :: il_status
      INTEGER(i4)        :: il_fileid

      LOGICAL            :: ll_exist

      ! namelist
      ! namzgr
      LOGICAL           :: ln_zco      = .FALSE.
      LOGICAL           :: ln_zps      = .FALSE.
      LOGICAL           :: ln_sco      = .FALSE.
      LOGICAL           :: ln_isfcav   = .FALSE.
      LOGICAL           :: ln_iscpl    = .FALSE.
      LOGICAL           :: ln_wd       = .FALSE.
      INTEGER(i4)       :: in_nlevel   = 75

      ! namdmin
      REAL(dp)          :: dn_hmin     = NF90_FILL_DOUBLE
      REAL(dp)          :: dn_isfhmin  = NF90_FILL_DOUBLE

      ! namzco
      REAL(dp)          :: dn_ppsur    = -3958.951371276829 !NF90_FILL_DOUBLE
      REAL(dp)          :: dn_ppa0     = 103.953009600000   !NF90_FILL_DOUBLE
      REAL(dp)          :: dn_ppa1     = 2.415951269000     !NF90_FILL_DOUBLE
      REAL(dp)          :: dn_ppkth    = 15.351013700000    !NF90_FILL_DOUBLE
      REAL(dp)          :: dn_ppacr    = 7.000000000000     !NF90_FILL_DOUBLE
      REAL(dp)          :: dn_ppdzmin  = 6.                 !NF90_FILL_DOUBLE
      REAL(dp)          :: dn_pphmax   = 5750.              !NF90_FILL_DOUBLE
      LOGICAL           :: ln_dbletanh = .TRUE.
      REAL(dp)          :: dn_ppa2     = 100.760928500000   !NF90_FILL_DOUBLE
      REAL(dp)          :: dn_ppkth2   = 48.029893720000    !NF90_FILL_DOUBLE
      REAL(dp)          :: dn_ppacr2   = 13.000000000000    !NF90_FILL_DOUBLE

      ! namzps
      REAL(dp)          :: dn_e3zps_min= NF90_FILL_DOUBLE
      REAL(dp)          :: dn_e3zps_rat= NF90_FILL_DOUBLE
!      INTEGER(i4)       :: in_msh      = NF90_FILL_INT

      ! namsco
      LOGICAL           :: ln_s_sh94   = .FALSE.
      LOGICAL           :: ln_s_sf12   = .FALSE.
      REAL(dp)          :: dn_sbot_min = NF90_FILL_DOUBLE
      REAL(dp)          :: dn_sbot_max = NF90_FILL_DOUBLE
      REAL(dp)          :: dn_rmax     = NF90_FILL_DOUBLE
      REAL(dp)          :: dn_hc       = NF90_FILL_DOUBLE
      !
      REAL(dp)          :: dn_theta    = NF90_FILL_DOUBLE
      REAL(dp)          :: dn_thetb    = NF90_FILL_DOUBLE
      REAL(dp)          :: dn_bb       = NF90_FILL_DOUBLE
      !                               
      LOGICAL           :: ln_sigcrit  = .FALSE.
      REAL(dp)          :: dn_alpha    = NF90_FILL_DOUBLE
      REAL(dp)          :: dn_efold    = NF90_FILL_DOUBLE
      REAL(dp)          :: dn_zs       = NF90_FILL_DOUBLE
      REAL(dp)          :: dn_zb_a     = NF90_FILL_DOUBLE
      REAL(dp)          :: dn_zb_b     = NF90_FILL_DOUBLE

!      ! namcla
!      INTEGER(i4)       :: in_cla      = 0

      ! namwd
      REAL(dp)          :: dn_wdmin1   = NF90_FILL_DOUBLE
      REAL(dp)          :: dn_wdmin2   = NF90_FILL_DOUBLE
      REAL(dp)          :: dn_wdld     = NF90_FILL_DOUBLE

      ! namgrd
!      CHARACTER(LEN=lc) :: cn_cfg      = ''
!      INTEGER(i4)       :: in_cfg      = 0
!      INTEGER(i4)       :: in_bench    = 0
!      LOGICAL           :: ln_zoom     = .FALSE.
      LOGICAL           :: ln_c1d      = .FALSE.
      LOGICAL           :: ln_e3_dep   = .FALSE.

!      ! namzoom
!      CHARACTER(LEN=lc) :: cn_cfz      =''
!      INTEGER(i4)       :: in_izoom    = NF90_FILL_INT
!      INTEGER(i4)       :: in_jzoom    = NF90_FILL_INT
!      LOGICAL           :: ln_zoom_s   = .FALSE.
!      LOGICAL           :: ln_zoom_e   = .FALSE.
!      LOGICAL           :: ln_zoom_w   = .FALSE.
!      LOGICAL           :: ln_zoom_n   = .FALSE.
      !----------------------------------------------------------------
      NAMELIST /namzgr/ &
      &  ln_zco,        &  !< z-coordinate
      &  ln_zps,        &  !< z-coordinate with partial steps
      &  ln_sco,        &  !< s-coordinate
      &  ln_isfcav,     &  !< presence of ISF 
      &  ln_iscpl,      &  !< coupling with ice sheet
      &  ln_wd,         &  !< Wetting/drying activation      
      &  in_nlevel         !< number of vertical level

      NAMELIST /namdmin/ &
      &  dn_hmin,       &  !< minimum ocean depth (>0) or minimum number of ocean levels (<0)
      &  dn_isfhmin        !< threshold to discriminate grounded ice to floating ice

      NAMELIST /namzco/ &
      &  dn_ppsur,      &
      &  dn_ppa0,       &
      &  dn_ppa1,       &
      &  dn_ppkth,      &
      &  dn_ppacr,      &
      &  dn_ppdzmin,    &
      &  dn_pphmax,     &
      &  ln_dbletanh,   &
      &  dn_ppa2,       &
      &  dn_ppkth2,     &
      &  dn_ppacr2

      NAMELIST /namzps/ &
      &  dn_e3zps_min,  &
      &  dn_e3zps_rat!,  &
!      &  in_msh          

      NAMELIST /namsco/ &
      &   ln_s_sh94,    & !< use hybrid s-sig Song and Haidvogel 1994 stretching function fssig1
      &   ln_s_sf12,    & !< use hybrid s-z-sig Siddorn and Furner 2012 stretching function fgamma
      &   dn_sbot_min,  & !< minimum depth of s-bottom surface (>0) (m)
      &   dn_sbot_max,  & !< maximum depth of s-bottom surface (= ocean depth) (>0) (m)
      &   dn_hc,        & !< Critical depth for transition from sigma to stretched coordinates
      ! Song and Haidvogel 1994 stretching parameters
      &   dn_rmax,      & !< maximum cut-off r-value allowed (0<dn_rmax<1)
      &   dn_theta,     & !< surface control parameter (0<=dn_theta<=20)
      &   dn_thetb,     & !< bottom control parameter  (0<=dn_thetb<= 1)
      &   dn_bb,        & !< stretching parameter ( dn_bb=0; top only, dn_bb =1; top and bottom)
      ! Siddorn and Furner stretching parameters
      &   ln_sigcrit,   & !< use sigma coordinates below critical depth (T) or Z coordinates (F) for Siddorn & Furner stretch
      &   dn_alpha,     & !< control parameter ( > 1 stretch towards surface, < 1 towards seabed)
      &   dn_efold,     & !<  efold length scale for transition to stretched coord
      &   dn_zs,        & !<  depth of surface grid box
                          !<  bottom cell depth (Zb) is a linear function of water depth Zb = H*rn_zb_a + rn_zb_b'
      &   dn_zb_a,      & !<  bathymetry scaling factor for calculating Zb
      &   dn_zb_b         !<  offset for calculating Zb

!      NAMELIST /namcla/ &
!      &  in_cla            !< =1 cross land advection for exchanges through some straits (ORCA2)

      NAMELIST /namwd/ &  !< wetting and drying
      &  dn_wdmin1, &     !< minimum water depth on dried cells
      &  dn_wdmin2, &     !< tolerrance of minimum water depth on dried cells
      &  dn_wdld          !< land elevation below which wetting/drying

      NAMELIST/namgrd/  &  !< orca grid namelist
!      &  cn_cfg,        &  !< name of the configuration (orca)
!      &  in_cfg,        &  !< resolution of the configuration (2,1,025..)
!      &  in_bench,      &  !< benchmark parameter (in_mshhgr = 5 )
!      &  ln_zoom,       &  !< use zoom
      &  ln_c1d,         &  !< use configuration 1D
      &  ln_e3_dep          !< new vertical scale factors [T, F:old definition]

!      NAMELIST /namzoom/&
!      &  cn_cfz,        &  !< name of the zoom of configuration
!      &  in_izoom,      &  !< left bottom i-indices of the zoom in data domain indices
!      &  in_jzoom,      &  !< left bottom j-indices of the zoom in data domain indices
!      &  ln_zoom_s,     &  !< South zoom type flag
!      &  ln_zoom_e,     &  !< East  zoom type flag
!      &  ln_zoom_w,     &  !< West  zoom type flag
!      &  ln_zoom_n         !< North zoom type flag
      !----------------------------------------------------------------
   !1-2 read namelist
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
         CALL logger_fatal("GRID ZGR NAM: error opening "//&
            &  TRIM(cd_namelist))
      ENDIF

      READ( il_fileid, NML = namzgr  )
      READ( il_fileid, NML = namdmin )
      READ( il_fileid, NML = namzco  )

      IF( ln_zps    ) READ( il_fileid, NML = namzps  )
      IF( ln_sco    ) READ( il_fileid, NML = namsco  )
!      READ( il_fileid, NML = namcla  )
      READ( il_fileid, NML = namwd   )
      READ( il_fileid, NML = namgrd  )
!      IF( ln_zoom   ) READ( il_fileid, NML = namzoom )

      CLOSE( il_fileid, IOSTAT=il_status )
      CALL fct_err(il_status)
      IF( il_status /= 0 )THEN
         CALL logger_error("GRID ZGR NAM: closing "//TRIM(cd_namelist))
      ENDIF
     
      tf_namz%c_coord    = TRIM(cd_coord)
      tf_namz%i_perio    = id_perio

      tf_namz%l_zco      = ln_zco    
      tf_namz%l_zps      = ln_zps    
      tf_namz%l_sco      = ln_sco    
      tf_namz%l_isfcav   = ln_isfcav 
      tf_namz%l_iscpl    = ln_iscpl 
      tf_namz%l_wd       = ln_wd 
      tf_namz%i_nlevel   = in_nlevel

      tf_namz%d_hmin     = dn_hmin  
      tf_namz%d_isfhmin  = dn_isfhmin  

      tf_namz%d_ppsur    = dn_ppsur  
      tf_namz%d_ppa0     = dn_ppa0   
      tf_namz%d_ppa1     = dn_ppa1   
      tf_namz%d_ppkth    = dn_ppkth  
      tf_namz%d_ppacr    = dn_ppacr  
      tf_namz%d_ppdzmin  = dn_ppdzmin
      tf_namz%d_pphmax   = dn_pphmax 
                             
      tf_namz%l_dbletanh = ln_dbletanh
      tf_namz%d_ppa2     = dn_ppa2    
      tf_namz%d_ppkth2   = dn_ppkth2  
      tf_namz%d_ppacr2   = dn_ppacr2  

      tf_namz%d_e3zps_min= dn_e3zps_min
      tf_namz%d_e3zps_rat= dn_e3zps_rat
!      tf_namz%i_msh      = in_msh      

      tf_namz%l_s_sh94   = ln_s_sh94  
      tf_namz%l_s_sf12   = ln_s_sf12  
      tf_namz%d_sbot_min = dn_sbot_min
      tf_namz%d_sbot_max = dn_sbot_max
      tf_namz%d_rmax     = dn_rmax    
      tf_namz%d_hc       = dn_hc      
      !
      tf_namz%d_theta    = dn_theta   
      tf_namz%d_thetb    = dn_thetb   
      tf_namz%d_bb       = dn_bb      
      !
      tf_namz%l_sigcrit  = ln_sigcrit
      tf_namz%d_alpha    = dn_alpha  
      tf_namz%d_efold    = dn_efold  
      tf_namz%d_zs       = dn_zs     
      tf_namz%d_zb_a     = dn_zb_a
      tf_namz%d_zb_b     = dn_zb_b

!      tf_namz%i_cla      = in_cla

      tf_namz%d_wdmin1   = dn_wdmin1  
      tf_namz%d_wdmin2   = dn_wdmin2  
      tf_namz%d_wdld     = dn_wdld  

!      tf_namz%c_cfg      = TRIM(cn_cfg)
!      tf_namz%i_cfg      = in_cfg
!      tf_namz%i_bench    = in_bench
!      tf_namz%l_zoom     = ln_zoom
      tf_namz%l_c1d      = ln_c1d
      tf_namz%l_e3_dep   = ln_e3_dep

!      tf_namz%c_cfz      = cn_cfz   
!      tf_namz%i_izoom    = in_izoom 
!      tf_namz%i_jzoom    = in_jzoom 
!      tf_namz%l_zoom_s   = ln_zoom_s
!      tf_namz%l_zoom_e   = ln_zoom_e
!      tf_namz%l_zoom_w   = ln_zoom_w
!      tf_namz%l_zoom_n   = ln_zoom_n

   ELSE

      CALL logger_fatal(" GRID ZGR NAM: can't find "//TRIM(cd_namelist))

   ENDIF

   END FUNCTION grid_zgr_nam
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr_fill(td_nam, jpi, jpj, jpk, td_bathy, td_risfdep) 
   !-------------------------------------------------------------------
   !> @brief This subroutine fill vertical mesh
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial version
   !> @date October, 2016
   !> - ice shelf cavity
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   !> @param[in] jpj
   !> @param[in] jpk
   !> @param[in] td_bathy
   !> @param[in] td_risfdep
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMZ), INTENT(IN   ) :: td_nam
      INTEGER(i4), INTENT(IN   ) :: jpi
      INTEGER(i4), INTENT(IN   ) :: jpj
      INTEGER(i4), INTENT(IN   ) :: jpk
      TYPE(TVAR) , INTENT(INOUT) :: td_bathy
      TYPE(TVAR) , INTENT(INOUT) :: td_risfdep

      ! local variable
      INTEGER(i4) :: il_count

      REAL(dp)    :: dl_bat

      ! loop indices
      !----------------------------------------------------------------

      CALL logger_info('GRID ZGR : vertical coordinate')
      CALL logger_info('~~~~~~~')
      CALL logger_info(' Namelist namzgr : set vertical coordinate')
      CALL logger_info(' z-coordinate - full steps      ln_zco    = '//TRIM(fct_str(td_nam%l_zco)))
      CALL logger_info(' z-coordinate - partial steps   ln_zps    = '//TRIM(fct_str(td_nam%l_zps)))
      CALL logger_info(' s- or hybrid z-s-coordinate    ln_sco    = '//TRIM(fct_str(td_nam%l_sco)))
      CALL logger_info(' ice shelf cavities             ln_isfcav = '//TRIM(fct_str(td_nam%l_isfcav)))
      CALL logger_info(' vertical scale factors         ln_e3_dep = '//TRIM(fct_str(td_nam%l_e3_dep)))

      il_count=0
      IF( td_nam%l_zco      )   il_count = il_count + 1
      IF( td_nam%l_zps      )   il_count = il_count + 1
      IF( td_nam%l_sco      )   il_count = il_count + 1
      IF( il_count /= 1 )THEN
         CALL logger_fatal(' GRID ZGR : none or several vertical coordinate options used' )
      ENDIF
      !
      il_count=0
      IF ( td_nam%l_zco .AND. td_nam%l_isfcav ) il_count = il_count + 1
      IF ( td_nam%l_sco .AND. td_nam%l_isfcav ) il_count = il_count + 1
      IF( il_count > 0 )THEN
         CALL logger_fatal(' GRID ZGR : Cavity not tested/compatible with full step (zco) and sigma (ln_sco)' )
      ENDIF

      IF(.NOT. td_nam%l_e3_dep )THEN
         CALL logger_info("Obsolescent definition of e3 scale factors is used")
      ENDIF
      ! Build the vertical coordinate system
      ! ------------------------------------

      ! Reference z-coordinate system (always called)
      CALL grid_zgr__z( td_nam,jpk )

      ! Bathymetry fields (levels and meters)
      CALL grid_zgr__bat( td_nam,td_bathy,td_risfdep ) !jpi,jpj,td_bathy,td_risfdep )

      ! 1D config.: same bathy value over the 3x3 domain
      IF( td_nam%l_c1d ) CALL lbc_lnk( td_bathy%d_value(:,:,1,1),'T',td_nam%i_perio,1._dp )

      ! z-coordinate
      IF( td_nam%l_zco ) CALL grid_zgr__zco(jpk)

      ! Partial step z-coordinate
      IF( td_nam%l_zps ) CALL grid_zgr__zps_fill( td_nam,jpi,jpj,jpk,td_bathy,td_risfdep )

      ! s-coordinate or hybrid z-s coordinate
      IF( td_nam%l_sco ) CALL grid_zgr__sco_fill( td_nam,jpi,jpj,jpk,td_bathy )

      ! final adjustment of mbathy & check 
      ! ----------------------------------

!      ! correct mbathy in case of zoom subdomain
!      IF( td_nam%l_zoom )   CALL grid_zgr__bat_zoom( td_nam,jpi,jpj )

      ! check bathymetry (mbathy) and suppress isolated ocean points
      IF( .NOT. td_nam%l_c1d )   CALL grid_zgr__bat_ctl( td_nam,jpi,jpj,jpk )

      ! deepest ocean level for t-, u- and v-points
      CALL grid_zgr__bot_level( ) !td_nam,jpi,jpj )

      ! shallowest ocean level for T-, U-, V- points
      CALL grid_zgr__top_level( ) !td_nam,jpi,jpj )

      ! 1D config.: same mbathy value over the 3x3 domain
      IF( td_nam%l_c1d ) THEN
         dl_bat = tg_mbathy%d_value(2,2,1,1)
         tg_mbathy%d_value(:,:,1,1) = dl_bat
      END IF      

      CALL logger_info(' MIN val mbathy '//TRIM(fct_str(MINVAL( tg_mbathy%d_value(:,:,1,1) )))//&
         &   ' MAX '//TRIM(fct_str(MAXVAL( tg_mbathy%d_value(:,:,1,1) ))) )
      CALL logger_info(' MIN val depth  t '//TRIM(fct_str(MINVAL( tg_gdept_0%d_value(:,:,:,1) )))//&
         &   ' w '//TRIM(fct_str(MINVAL( tg_gdepw_0%d_value(:,:,:,1) )))//&
         !&   ' 3w '//TRIM(fct_str(MINVAL( tg_gdep3w_0%d_value(:,:,:,1) )))//&
         &   '  t '//TRIM(fct_str(MINVAL( tg_e3t_0%d_value(:,:,:,1) )))//&
         &   '  f '//TRIM(fct_str(MINVAL( tg_e3f_0%d_value(:,:,:,1) )))//&
         &   '  u '//TRIM(fct_str(MINVAL( tg_e3u_0%d_value(:,:,:,1) )))//&
         &   '  v '//TRIM(fct_str(MINVAL( tg_e3v_0%d_value(:,:,:,1) )))//&
         &   ' uw '//TRIM(fct_str(MINVAL( tg_e3uw_0%d_value(:,:,:,1) )))//&
         &   ' vw '//TRIM(fct_str(MINVAL( tg_e3vw_0%d_value(:,:,:,1) )))//&
         &   '  w '//TRIM(fct_str(MINVAL( tg_e3w_0%d_value(:,:,:,1) ))) )
      CALL logger_info(' MAX val depth t '//TRIM(fct_str(MAXVAL( tg_gdept_0%d_value(:,:,:,1) )))//&
         &   ' w '//TRIM(fct_str(MAXVAL( tg_gdepw_0%d_value(:,:,:,1) ))) )!//&
         !&   ' 3w '//TRIM(fct_str(MAXVAL( tg_gdep3w_0%d_value(:,:,:,1) ))) )
      CALL logger_info(' MAX val e3    t '//TRIM(fct_str(MAXVAL( tg_e3t_0%d_value(:,:,:,1) )))//&
         &   ' f '//TRIM(fct_str(MAXVAL( tg_e3f_0%d_value(:,:,:,1) )))//&
         &   ' u '//TRIM(fct_str(MAXVAL( tg_e3u_0%d_value(:,:,:,1) )))//&
         &   ' v '//TRIM(fct_str(MAXVAL( tg_e3v_0%d_value(:,:,:,1) )))//&
         &   ' uw '//TRIM(fct_str(MAXVAL( tg_e3uw_0%d_value(:,:,:,1) )))//&
         &   ' vw '//TRIM(fct_str(MAXVAL( tg_e3vw_0%d_value(:,:,:,1) )))//&
         &   ' w '//TRIM(fct_str(MAXVAL( tg_e3w_0%d_value(:,:,:,1) ))) )

   END SUBROUTINE grid_zgr_fill
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr__z(td_nam, jpk) 
   !-------------------------------------------------------------------
   !> @brief This subroutine set the depth of model levels and the resulting 
   !>        vertical scale factors.
   !>
   !> @details
   !>
   !> ** Method  :   z-coordinate system (use in all type of coordinate)
   !>        The depth of model levels is defined from an analytical
   !>      function the derivative of which gives the scale factors.
   !>        both depth and scale factors only depend on k (1d arrays).<br/>
   !>              w-level: 
   !>                       - gdepw_1d  = gdep(k)<br/>
   !>                       - e3w_1d(k) = dk(gdep)(k)     = e3(k)<br/>
   !>              t-level: 
   !>                       - gdept_1d  = gdep(k+0.5)<br/>
   !>                       - e3t_1d(k) = dk(gdep)(k+0.5) = e3(k+0.5)<br/>
   !>
   !>
   !> ** Action  : - gdept_1d, gdepw_1d : depth of T- and W-point (m)
   !>              - e3t_1d  , e3w_1d   : scale factors at T- and W-levels (m)
   !>
   !> !! Reference : Marti, Madec & Delecluse, 1992, JGR, 97, No8, 12,763-12,766.
   !>
   !> @author J.Paul
   !> @date September, 2015 - rewrite from zgr_z
   !>
   !> @param[in] td_nam
   !> @param[in] jpk
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMZ), INTENT(IN   ) :: td_nam
      INTEGER(i4), INTENT(IN   ) :: jpk

      ! local variable
      CHARACTER(LEN=lc)          :: cl_tmp

      REAL(dp)                   :: zsur, za0, za1, zkth   ! Values set from parameters in
      REAL(dp)                   :: zacr, zdzmin, zhmax    ! par_CONFIG_Rxx.h90
!      REAL(dp)                   :: zrefdep                ! depth of the reference level (~10m)
      REAL(dp)                   :: za2, zkth2, zacr2      ! Values for optional double tanh function set from parameters
      REAL(dp)                   :: zt, zw                 ! temporary scalars
      REAL(dp), PARAMETER        :: dp_pp_to_be_computed = NF90_FILL_DOUBLE

      ! loop indices
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      ! Set variables from parameters
      ! ------------------------------
      zkth   = td_nam%d_ppkth       
      zacr   = td_nam%d_ppacr
      zdzmin = td_nam%d_ppdzmin   
      zhmax  = td_nam%d_pphmax
      zkth2  = td_nam%d_ppkth2     
      zacr2  = td_nam%d_ppacr2

      ! If ppa1 and ppa0 and ppsur are set to pp_to_be_computed
      !  za0, za1, zsur are computed from ppdzmin , pphmax, ppkth, ppacr
      IF(   td_nam%d_ppa1  == dp_pp_to_be_computed  .AND.  &
         &  td_nam%d_ppa0  == dp_pp_to_be_computed  .AND.  &
         &  td_nam%d_ppsur == dp_pp_to_be_computed           ) THEN
         !
         za1  = (  zdzmin - zhmax / REAL(jpk-1,dp)  )                                               &
            & / ( TANH((1-zkth)/zacr) - zacr/REAL(jpk-1,dp) * ( LOG( COSH( (jpk - zkth) / zacr) )   &
            &                                               - LOG( COSH( ( 1  - zkth) / zacr) ) ) )
         za0  = zdzmin  - za1 *             TANH( (1-zkth) / zacr )
         zsur =   - za0 - za1 * zacr * LOG( COSH( (1-zkth) / zacr )  )
         ! za2 ???
      ELSE
         za1  = td_nam%d_ppa1 
         za0  = td_nam%d_ppa0
         zsur = td_nam%d_ppsur
         za2  = td_nam%d_ppa2                            ! optional (ldbletanh=T) double tanh parameter
      ENDIF      

      CALL logger_info(' GRID ZGR Z : Reference vertical z-coordinates')
      CALL logger_info('~~~~~~~~~~~')
      IF( zkth == 0._dp ) THEN 
         CALL logger_info('Uniform grid with '//TRIM(fct_str(jpk-1))//' layers')
         CALL logger_info('Total depth    :'//TRIM(fct_str(zhmax)))
         CALL logger_info('Layer thickness:'//TRIM(fct_str(zhmax/(jpk-1))))         
      ELSE
         IF( za1 == 0._dp .AND. za0 == 0._dp .AND. zsur == 0._dp ) THEN
            CALL logger_info('zsur, za0, za1 computed from ')
            CALL logger_info('        zdzmin = '//TRIM(fct_str(zdzmin)))
            CALL logger_info('        zhmax  = '//TRIM(fct_str(zhmax)))            
         ENDIF
            CALL logger_info('Value of coefficients for vertical mesh:')
            CALL logger_info('      zsur = '//TRIM(fct_str(zsur)))
            CALL logger_info('      za0  = '//TRIM(fct_str(za0)))
            CALL logger_info('      za1  = '//TRIM(fct_str(za1)))
            CALL logger_info('      zkth = '//TRIM(fct_str(zkth)))
            CALL logger_info('      zacr = '//TRIM(fct_str(zacr)))
         IF( td_nam%l_dbletanh ) THEN
            CALL logger_info(' (Double tanh    za2  = '//TRIM(fct_str(za2)))
            CALL logger_info('  parameters)    zkth2= '//TRIM(fct_str(zkth2)))
            CALL logger_info('                 zacr2= '//TRIM(fct_str(zacr2)))
         ENDIF
      ENDIF

      ! Reference z-coordinate (depth - scale factor at T- and W-points)
      ! ======================
      ! init
      IF( zkth == 0._dp ) THEN            !  uniform vertical grid       
         za1 = zhmax / REAL(jpk-1,dp) 
         DO jk = 1, jpk
            zw = REAL( jk, dp )
            zt = REAL( jk, dp ) + 0.5_dp
            tg_gdepw_1d%d_value(1,1,jk,1) = ( zw - 1 ) * za1
            tg_gdept_1d%d_value(1,1,jk,1) = ( zt - 1 ) * za1
            tg_e3w_1d%d_value  (1,1,jk,1) =  za1
            tg_e3t_1d%d_value  (1,1,jk,1) =  za1
         END DO
      ELSE                                ! Madec & Imbard 1996 function
         IF( .NOT. td_nam%l_dbletanh ) THEN
            DO jk = 1, jpk
               zw = REAL( jk , dp )
               zt = REAL( jk , dp ) + 0.5_dp
               tg_gdepw_1d%d_value(1,1,jk,1) = ( zsur + za0 * zw + za1 * zacr * LOG ( COSH( (zw-zkth) / zacr ) )  )
               tg_gdept_1d%d_value(1,1,jk,1) = ( zsur + za0 * zt + za1 * zacr * LOG ( COSH( (zt-zkth) / zacr ) )  )
               tg_e3w_1d%d_value  (1,1,jk,1) =          za0      + za1        * TANH(       (zw-zkth) / zacr   )
               tg_e3t_1d%d_value  (1,1,jk,1) =          za0      + za1        * TANH(       (zt-zkth) / zacr   )
            END DO
         ELSE
            DO jk = 1, jpk
               zw = REAL( jk, dp )
               zt = REAL( jk, dp ) + 0.5_dp
               ! Double tanh function
               tg_gdepw_1d%d_value(1,1,jk,1) = ( zsur + za0 * zw + za1 * zacr * LOG ( COSH( (zw-zkth ) / zacr  ) )    &
                           &                                     + za2 * zacr2* LOG ( COSH( (zw-zkth2) / zacr2 ) )  )
               tg_gdept_1d%d_value(1,1,jk,1) = ( zsur + za0 * zt + za1 * zacr * LOG ( COSH( (zt-zkth ) / zacr  ) )    &
                           &                                     + za2 * zacr2* LOG ( COSH( (zt-zkth2) / zacr2 ) )  )
               tg_e3w_1d  %d_value(1,1,jk,1) =          za0      + za1        * TANH(       (zw-zkth ) / zacr  )      &
                           &                                     + za2        * TANH(       (zw-zkth2) / zacr2 )
               tg_e3t_1d  %d_value(1,1,jk,1) =          za0      + za1        * TANH(       (zt-zkth ) / zacr  )      &
                  &                                              + za2        * TANH(       (zt-zkth2) / zacr2 )
            END DO
         ENDIF
         tg_gdepw_1d%d_value(1,1,1,1) = 0._dp                    ! force first w-level to be exactly at zero
      ENDIF      

      IF ( td_nam%l_isfcav .OR. td_nam%l_e3_dep ) THEN
         ! need to be like this to compute the pressure gradient with ISF. 
         ! If not, level beneath the ISF are not aligned (sum(e3t) /= depth)
         ! define e3t_0 and e3w_0 as the differences between gdept and gdepw respectively
         DO jk = 1, jpk-1
            tg_e3t_1d%d_value(1,1,jk,1) = tg_gdepw_1d%d_value(1,1,jk+1,1)-tg_gdepw_1d%d_value(1,1,jk,1) 
         END DO
         ! we don't care because this level is masked in NEMO
         tg_e3t_1d%d_value(1,1,jpk,1) = tg_e3t_1d%d_value(1,1,jpk-1,1)

         DO jk = 2, jpk
            tg_e3w_1d%d_value(1,1,jk,1) = tg_gdept_1d%d_value(1,1,jk,1) - tg_gdept_1d%d_value(1,1,jk-1,1) 
         END DO
         tg_e3w_1d%d_value(1,1,1,1) = 2._dp * (tg_gdept_1d%d_value(1,1,1,1) - tg_gdepw_1d%d_value(1,1,1,1)) 
      END IF 

! unused ?
!!!!gm BUG in s-coordinate this does not work!
!      ! deepest/shallowest W level Above/Below ~10m
!      
!      ! ref. depth with tolerance (10% of minimum layer thickness)
!      zrefdep = 10._dp - 0.1_dp * MINVAL( tg_e3w_1d%d_value(1,1,:,1) )
!      
!      ! shallowest W level Below ~10m
!      nlb10 = MINLOC( tg_gdepw_1d%d_value(1,1,:,1), mask = tg_gdepw_1d%d_value(1,1,:,1) > zrefdep, dim = 1 )
!      
!      ! deepest    W level Above ~10m
!      nla10 = nlb10 - 1
!!!!gm end bug

      ! control print
      CALL logger_info(' GRID ZGR Z : Reference z-coordinate depth and scale factors:')
      CALL logger_info('~~~~~~~~~~~')
      WRITE(cl_tmp, "(9x,' level  gdept_1d  gdepw_1d  e3t_1d   e3w_1d  ')" )
      CALL logger_info(cl_tmp)
      DO jk=1,jpk
         WRITE(cl_tmp, "(10x, i4, 4f9.2)" ) jk, tg_gdept_1d%d_value(1,1,jk,1), tg_gdepw_1d%d_value(1,1,jk,1), &
            &                                   tg_e3t_1d%d_value  (1,1,jk,1), tg_e3w_1d%d_value  (1,1,jk,1)
         CALL logger_info(cl_tmp)
      ENDDO

      ! control positivity
      DO jk = 1, jpk                      
         IF( tg_e3w_1d%d_value  (1,1,jk,1) <= 0._dp .OR. tg_e3t_1d%d_value  (1,1,jk,1) <= 0._dp )THEN
            CALL logger_fatal( 'GRID ZGR Z: e3w_1d or e3t_1d =< 0 '    )
         ENDIF
         IF( tg_gdepw_1d%d_value(1,1,jk,1) <  0._dp .OR. tg_gdept_1d%d_value(1,1,jk,1) <  0._dp )THEN
            CALL logger_fatal( 'GRID ZGR Z: gdepw_1d or gdept_1d < 0 ' )
         ENDIF
      END DO

   END SUBROUTINE grid_zgr__z
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr__bat(td_nam, td_bathy, td_risfdep) !jpi,jpj,td_bathy,td_risfdep ) 
   !-------------------------------------------------------------------
   !> @brief This subroutine set bathymetry both in levels and meters
   !>
   !> @details
   !>
   !> ** Method  :   read or define mbathy and bathy arrays
   !>       * level bathymetry:
   !>      The ocean basin geometry is given by a two-dimensional array,
   !>      mbathy, which is defined as follow :
   !>            mbathy(ji,jj) = 1, ..., jpk-1, the number of ocean level
   !>                              at t-point (ji,jj).
   !>                          = 0  over the continental t-point.
   !>      The array mbathy is checked to verified its consistency with
   !>      model option. in particular:
   !>            mbathy must have at least 1 land grid-points (mbathy<=0)
   !>                  along closed boundary.
   !>            mbathy must be cyclic IF jperio=1.
   !>            mbathy must be lower or equal to jpk-1.
   !>            isolated ocean grid points are suppressed from mbathy
   !>                  since they are only connected to remaining
   !>                  ocean through vertical diffusion.
   !>      ntopo=-1 :   rectangular channel or bassin with a bump 
   !>      ntopo= 0 :   flat rectangular channel or basin 
   !>      ntopo= 1 :   mbathy is read in 'bathy_level.nc' NetCDF file
   !>                   bathy  is read in 'bathy_meter.nc' NetCDF file
   !>
   !> ** Action  : - mbathy: level bathymetry (in level index)
   !>              - bathy : meter bathymetry (in meters)   
   !>
   !> @warning do not manage case ntopo=-1 or 0
   !>
   !> @author J.Paul
   !> @date September, 2015 - rewrite from zgr_bat
   !> @date October, 2016
   !> - do not use anymore special case for ORCA grid.
   !>
   !> @param[in] td_nam
   ! @param[in] jpi
   ! @param[in] jpj
   !> @param[in] td_bathy
   !> @param[in] td_risfdep
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMZ), INTENT(IN   ) :: td_nam
!      INTEGER(i4), INTENT(IN   ) :: jpi
!      INTEGER(i4), INTENT(IN   ) :: jpj
      TYPE(TVAR) , INTENT(INOUT) :: td_bathy
      TYPE(TVAR) , INTENT(INOUT) :: td_risfdep

      ! local variable
!      INTEGER(i4) :: ii0, ii1
!      INTEGER(i4) :: ij0, ij1

      REAL(dp)                                  :: zhmin

      ! loop indices
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      CALL logger_info(' GRID ZGR BAT : defines level and meter bathymetry')
      CALL logger_info(' ~~~~~~~~~~~~~')
      CALL logger_info(' GRID ZGR BAT : bathymetry read in file')

      IF( td_nam%l_zco )THEN
         ! zco : read level bathymetry

         ! read variable in bathymetry file
         tg_mbathy%d_value(:,:,1,1) = INT(td_bathy%d_value(:,:,1,1),i4)
         tg_misfdep%d_value(:,:,1,1)=1
         !                                                ! =====================
!         IF( TRIM(td_nam%c_cfg) == "orca" .AND. td_nam%i_cfg == 2 ) THEN    ! ORCA R2 configuration
!            !                                             ! =====================
!            IF( td_nam%i_cla == 0 ) THEN
!               ii0 = 140   ;   ii1 = 140                  ! Gibraltar Strait open 
!               ij0 = 102   ;   ij1 = 102                  ! (Thomson, Ocean Modelling, 1995)
!               tg_mbathy%d_value(ii0:ii1,ij0:ij1,1,1) = 15
!               CALL logger_info('orca_r2: Gibraltar strait open at i='//&
!                  &  TRIM(fct_str(ii0))//' j='//TRIM(fct_str(ij0)) )
!               !
!               ii0 = 160   ;   ii1 = 160                  ! Bab el mandeb Strait open
!               ij0 = 88    ;   ij1 = 88                   ! (Thomson, Ocean Modelling, 1995)
!               tg_mbathy%d_value(ii0:ii1,ij0:ij1,1,1) = 12
!               CALL logger_info('orca_r2: Bab el Mandeb strait open at i='//&
!                  &  TRIM(fct_str(ii0))//' j='//TRIM(fct_str(ij0)) )
!            ENDIF
!            !
!         ENDIF

      ENDIF

      IF( td_nam%l_zps .OR. td_nam%l_sco )THEN
         ! zps or sco : read meter bathymetry

         tg_misfdep%d_value(:,:,:,:)=1

         IF ( td_nam%l_isfcav ) THEN
            WHERE( td_bathy%d_value(:,:,1,1) <= 0._dp )
               td_risfdep%d_value(:,:,1,1) = 0._dp
            END WHERE

            ! set grounded point to 0
            ! (a treshold could be set here if needed, or set it offline based on the grounded fraction)
            WHERE ( td_bathy%d_value(:,:,1,1) <= td_risfdep%d_value(:,:,1,1) + td_nam%d_isfhmin )
               tg_misfdep%d_value(:,:,1,1) = 0
               td_risfdep%d_value(:,:,1,1) = 0._dp
               tg_mbathy%d_value (:,:,1,1) = 0
               td_bathy%d_value  (:,:,1,1) = 0._dp
            END WHERE
         END IF
         !       
!         IF( TRIM(td_nam%c_cfg) == "orca" .AND. td_nam%i_cfg == 2 ) THEN    ! ORCA R2 configuration
!            !
!           IF( td_nam%i_cla == 0 ) THEN
!              ii0 = 140   ;   ii1 = 140                   ! Gibraltar Strait open 
!              ij0 = 102   ;   ij1 = 102                   ! (Thomson, Ocean Modelling, 1995)
!              td_bathy%d_value(ii0:ii1,ij0:ij1,1,1) = 284._dp 
!              CALL logger_info('orca_r2: Gibraltar strait open at i='//&
!                 &   TRIM(fct_str(ii0))//' j='//TRIM(fct_str(ij0)) )
!              !
!              ii0 = 160   ;   ii1 = 160                   ! Bab el mandeb Strait open
!              ij0 = 88    ;   ij1 = 88                    ! (Thomson, Ocean Modelling, 1995)
!              td_bathy%d_value(ii0:ii1,ij0:ij1,1,1) = 137._dp
!              CALL logger_info('orca_r2: Bab el Mandeb strait open at i='//&
!                 &   TRIM(fct_str(ii0))//' j='//TRIM(fct_str(ij0)) )
!           ENDIF
!           !
!        ENDIF
         !
      ENDIF      

      !==  NO closed seas or lakes  ==!
      ! already done

      IF ( .NOT. td_nam%l_sco ) THEN
         !==  set a minimum depth  ==!
         
         IF( td_nam%d_hmin < 0._dp ) THEN
            ! from a nb of level
            jk = - INT(td_nam%d_hmin, i4)
         ELSE
            ! from a depth
            jk = MINLOC( tg_gdepw_1d%d_value(1,1,:,1), &
               &  MASK = tg_gdepw_1d%d_value(1,1,:,1) > td_nam%d_hmin, &
               &  DIM  = 1)
         ENDIF

         ! minimum depth = ik+1 w-levels
         zhmin = tg_gdepw_1d%d_value(1,1,jk+1,1)
         WHERE( td_bathy%d_value(:,:,1,1) <= 0._dp )
            ! min=0     over the lands
            td_bathy%d_value(:,:,1,1) = 0._dp
         ELSE WHERE
            ! min=zhmin over the oceans
            td_bathy%d_value(:,:,1,1) = MAX(zhmin, td_bathy%d_value(:,:,1,1))
         END WHERE
         CALL logger_info('GRID ZGR BAT: Minimum ocean depth: '//&
            &   TRIM(fct_str(zhmin))//' minimum number of ocean'//&
            &   ' levels : '//TRIM(fct_str(jk)))
      ENDIF

   END SUBROUTINE grid_zgr__bat
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr__zco(jpk) 
   !-------------------------------------------------------------------
   !> @brief This subroutine define the z-coordinate system
   !>
   !> @details
   !> set 3D coord. arrays to reference 1D array
   !>
   !> @author J.Paul
   !> @date September, 2015 - rewrite from zgr_zco
   !>
   !> @param[in] jpk
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      INTEGER(i4), INTENT(IN   ) :: jpk

      ! local variable
      ! loop indices
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      DO jk = 1, jpk
         tg_gdept_0%d_value (:,:,jk,1) = tg_gdept_1d%d_value(1,1,jk,1)
         tg_gdepw_0%d_value (:,:,jk,1) = tg_gdepw_1d%d_value(1,1,jk,1)
         !tg_gdep3w_0%d_value(:,:,jk,1) = tg_gdepw_1d%d_value(1,1,jk,1)
         tg_e3t_0%d_value   (:,:,jk,1) = tg_e3t_1d%d_value  (1,1,jk,1)
         tg_e3u_0%d_value   (:,:,jk,1) = tg_e3t_1d%d_value  (1,1,jk,1)
         tg_e3v_0%d_value   (:,:,jk,1) = tg_e3t_1d%d_value  (1,1,jk,1)
         tg_e3f_0%d_value   (:,:,jk,1) = tg_e3t_1d%d_value  (1,1,jk,1)
         tg_e3w_0%d_value   (:,:,jk,1) = tg_e3w_1d%d_value  (1,1,jk,1)
         tg_e3uw_0%d_value  (:,:,jk,1) = tg_e3w_1d%d_value  (1,1,jk,1)
         tg_e3vw_0%d_value  (:,:,jk,1) = tg_e3w_1d%d_value  (1,1,jk,1)
      END DO

   END SUBROUTINE grid_zgr__zco
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!   SUBROUTINE grid_zgr__bat_zoom(td_nam,jpi,jpj) 
!   !-------------------------------------------------------------------
!   !> @brief This subroutine : 
!   !> - close zoom domain boundary if necessary
!   !> - suppress Med Sea from ORCA R2 and R05 arctic zoom
!   !> 
!   !> @author J.Paul
!   !> @date September, 2015 - Initial version
!   !>
!   !> @param[in] td_nam
!   !> @param[in] jpi
!   !> @param[in] jpj
!   !-------------------------------------------------------------------
!
!      IMPLICIT NONE
!
!      ! Argument      
!      TYPE(TNAMZ), INTENT(IN   ) :: td_nam
!      INTEGER(i4), INTENT(IN   ) :: jpi
!      INTEGER(i4), INTENT(IN   ) :: jpj
!
!      ! local variable
!      INTEGER(i4) :: jpizoom
!      INTEGER(i4) :: jpjzoom
!
!      INTEGER(i4) :: ii0, ii1
!      INTEGER(i4) :: ij0, ij1
!      ! loop indices
!      !----------------------------------------------------------------
!
!      CALL logger_info('GRID ZGR BAT ZOOM : modify the level bathymetry for zoom domain')
!      CALL logger_info('~~~~~~~~~~~~')
!
!      jpizoom=td_nam%i_izoom
!      jpjzoom=td_nam%i_jzoom
!
!      ! Forced closed boundary if required
!      IF( td_nam%l_zoom_s ) tg_mbathy%d_value(    :         ,    jpjzoom  ,1,1) = 0
!      IF( td_nam%l_zoom_w ) tg_mbathy%d_value(     jpizoom  ,   :         ,1,1) = 0
!      IF( td_nam%l_zoom_e ) tg_mbathy%d_value( jpi+jpizoom-1,   :         ,1,1) = 0
!      IF( td_nam%l_zoom_n ) tg_mbathy%d_value(    :         ,jpj+jpjzoom-1,1,1) = 0
!
!      ! Configuration specific domain modifications
!      ! (here, ORCA arctic configuration: suppress Med Sea)
!      IF( TRIM(td_nam%c_cfg) == "orca" .AND. &
!        & TRIM(td_nam%c_cfz) == "arctic" ) THEN
!         SELECT CASE ( td_nam%i_cfg )
!         !                                        ! =======================
!         CASE ( 2 )                               !  ORCA_R2 configuration
!            !                                     ! =======================
!            CALL logger_info('ORCA R2 arctic zoom: suppress the Med Sea')
!            ii0 = 141   ;   ii1 = 162      ! Sea box i,j indices
!            ij0 =  98   ;   ij1 = 110
!            !                                     ! =======================
!         CASE ( 05 )                              !  ORCA_R05 configuration
!            !                                     ! =======================
!            CALL logger_info('ORCA R05 arctic zoom: suppress the Med Sea')
!            ii0 = 563   ;   ii1 = 642      ! zero over the Med Sea boxe
!            ij0 = 314   ;   ij1 = 370 
!         END SELECT
!         !
!         tg_mbathy%d_value( ii0:ii1, ij0:ij1, 1, 1) = 0   ! zero over the Med Sea boxe
!         !
!      ENDIF
!
!   END SUBROUTINE grid_zgr__bat_zoom
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr__bat_ctl(td_nam, jpi, jpj, jpk) 
   !-------------------------------------------------------------------
   !> @brief This subroutine check the bathymetry in levels
   !>
   !> @details
   !>
   !> 
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
   !> @author J.Paul
   !> @date September, 2015 - Initial version
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   !> @param[in] jpj
   !> @param[in] jpk
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMZ), INTENT(IN   ) :: td_nam
      INTEGER(i4), INTENT(IN   ) :: jpi
      INTEGER(i4), INTENT(IN   ) :: jpj
      INTEGER(i4), INTENT(IN   ) :: jpk

      ! local variable
      INTEGER(i4) :: icompt, ibtest, ikmax         ! temporary integers

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      CALL logger_info('GRID ZGR BAT CTL: check the bathymetry')
      CALL logger_info('~~~~~~~~~~~~~~~')
      CALL logger_info('                   suppress isolated ocean grid points')
      CALL logger_info('                   -----------------------------------')

      icompt = 0
      DO jl = 1, 2
         IF( td_nam%i_perio == 1 .OR. &
           & td_nam%i_perio == 4 .OR. &
           & td_nam%i_perio == 6 ) THEN
            tg_mbathy%d_value( 1 ,:,1,1) = tg_mbathy%d_value(jpi-1,:,1,1)           ! local domain is cyclic east-west
            tg_mbathy%d_value(jpi,:,1,1) = tg_mbathy%d_value(  2  ,:,1,1)
         ENDIF
         DO jj = 2, jpj-1
            DO ji = 2, jpi-1
               ibtest = MAX(  tg_mbathy%d_value(ji-1,jj  ,1,1), &
                  &           tg_mbathy%d_value(ji+1,jj  ,1,1), &
                  &           tg_mbathy%d_value(ji  ,jj-1,1,1), &
                  &           tg_mbathy%d_value(ji  ,jj+1,1,1)  )
               IF( ibtest < tg_mbathy%d_value(ji,jj,1,1) ) THEN
                  CALL logger_info(' the number of ocean level at '//&
                     &             'grid-point (i,j) =  ('//TRIM(fct_str(ji))//&
                     &             ','//TRIM(fct_str(jj))//') is changed from '//&
                     &             TRIM(fct_str(tg_mbathy%d_value(ji,jj,1,1)))//' to '//&
                     &             TRIM(fct_str(ibtest)) )
                  tg_mbathy%d_value(ji,jj,1,1) = ibtest
                  icompt = icompt + 1
               ENDIF
            END DO
         END DO
      END DO

      !! lk_mpp not used

      IF( icompt == 0 ) THEN
         CALL logger_info('     no isolated ocean grid points')
      ELSE
         CALL logger_info(TRIM(fct_str(icompt))//' ocean grid points suppressed')
      ENDIF

      !! lk_mpp not used

      !                                          ! East-west cyclic boundary conditions
      IF( td_nam%i_perio == 0 ) THEN
         CALL logger_info(' mbathy set to 0 along east and west boundary:'//&
            &  ' nperio = '//TRIM(fct_str(td_nam%i_perio)) )
         !! lk_mpp not used
         IF( td_nam%l_zco .OR. td_nam%l_zps ) THEN
            tg_mbathy%d_value( 1 ,:,1,1) = 0
            tg_mbathy%d_value(jpi,:,1,1) = 0
         ELSE
            tg_mbathy%d_value( 1 ,:,1,1) = jpk-1
            tg_mbathy%d_value(jpi,:,1,1) = jpk-1
         ENDIF
      ELSEIF( td_nam%i_perio == 1 .OR. &
            & td_nam%i_perio == 4 .OR. &
            & td_nam%i_perio == 6 ) THEN
         CALL logger_info(' east-west cyclic boundary conditions on mbathy:'//&
            &  ' nperio = '//TRIM(fct_str(td_nam%i_perio)) )
         tg_mbathy%d_value( 1 ,:,1,1) = tg_mbathy%d_value(jpi-1,:,1,1)
         tg_mbathy%d_value(jpi,:,1,1) = tg_mbathy%d_value(  2  ,:,1,1)
      ELSEIF( td_nam%i_perio == 2 ) THEN
         CALL logger_info('   equatorial boundary conditions on mbathy:'//&
            ' nperio = '//TRIM(fct_str(td_nam%i_perio)) )
      ELSE
         CALL logger_info('    e r r o r')
         CALL logger_info('    parameter , nperio = '//TRIM(fct_str(td_nam%i_perio)) )
         !         STOP 'dom_mba'
      ENDIF

      !  Boundary condition on mbathy
!!gm  !!bug ???  think about it !
      !   ... mono- or macro-tasking: T-point, >0, 2D array, no slab
      CALL lbc_lnk( tg_mbathy%d_value(:,:,1,1), 'T', td_nam%i_perio, 1._dp )


      ! Number of ocean level inferior or equal to jpkm1
      ikmax = 0
      DO jj = 1, jpj
         DO ji = 1, jpi
            ikmax = MAX( ikmax, INT(tg_mbathy%d_value(ji,jj,1,1),i4) )
         END DO
      END DO      
!!gm  !!! test to do:   ikmax = MAX( mbathy(:,:) )   ???
      IF( ikmax > jpk-1 ) THEN
         CALL logger_info(' maximum number of ocean level = '//TRIM(fct_str(ikmax))//' >  jpk-1')
         CALL logger_info(' change jpk to '//TRIM(fct_str(ikmax+1))//' to use the exact ead bathymetry')
      ELSE IF( ikmax < jpk-1 ) THEN
         CALL logger_info(' maximum number of ocean level = '//TRIM(fct_str(ikmax))//' < jpk-1')
         CALL logger_info(' you can decrease jpk to '//TRIM(fct_str(ikmax+1)))
      ENDIF      

   END SUBROUTINE grid_zgr__bat_ctl
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr__bot_level()!td_nam,jpi,jpj) 
   !-------------------------------------------------------------------
   !> @brief This subroutine defines the vertical index of ocean bottom (mbk. arrays)
   !>
   !> @details
   !>
   !> ** Method  :   computes from mbathy with a minimum value of 1 over land
   !>
   !> ** Action  :   mbkt, mbku, mbkv :   vertical indices of the deeptest 
   !>                                     ocean level at t-, u- & v-points
   !>                                     (min value = 1 over land)
   !> 
   !> 
   !> @author J.Paul
   !> @date September, 2015 - rewrite from zgr_bot_level
   !>
   ! @param[in] td_nam
   ! @param[in] jpi
   ! @param[in] jpj
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
!      TYPE(TNAMZ), INTENT(IN   ) :: td_nam
!      INTEGER(i4), INTENT(IN   ) :: jpi
!      INTEGER(i4), INTENT(IN   ) :: jpj

      ! local variable

      ! loop indices
!      INTEGER(i4) :: ji
!      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      CALL logger_info('GRID ZGR BOT LEVEL: ocean bottom k-index of T-, U-, V- and W-levels ')
      CALL logger_info('    ~~~~~~~~~~~~~')

      ! bottom k-index of T-level (=1 over land)
      tg_mbkt%d_value(:,:,1,1) = MAX( tg_mbathy%d_value(:,:,1,1) , 1._dp )

!      ! bottom k-index of W-level = mbkt+1
!      ! bottom k-index of u- (v-) level
!      DO jj = 1, jpj-1                      ! bottom k-index of u- (v-) level
!         DO ji = 1, jpi-1
!            tg_mbku%d_value(ji,jj,1,1) = MIN(  tg_mbkt%d_value(ji+1,jj  ,1,1) , &
!               &                               tg_mbkt%d_value(ji  ,jj  ,1,1)  )
!
!            tg_mbkv%d_value(ji,jj,1,1) = MIN(  tg_mbkt%d_value(ji  ,jj+1,1,1) , &
!               &                               tg_mbkt%d_value(ji  ,jj  ,1,1)  )
!         END DO
!      END DO
!
!      CALL lbc_lnk(tg_mbku%d_value(:,:,1,1),'U', td_nam%i_perio, 1._dp)
!      CALL lbc_lnk(tg_mbkv%d_value(:,:,1,1),'U', td_nam%i_perio, 1._dp)

   END SUBROUTINE grid_zgr__bot_level
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr__top_level()!td_nam,jpi,jpj) 
   !-------------------------------------------------------------------
   !> @brief This subroutine defines the vertical index of ocean top (mik. arrays)
   !>
   !> @details
   !>
   !> ** Method  :   computes from misfdep with a minimum value of 1
   !>
   !> ** Action  :   mikt, miku, mikv :   vertical indices of the shallowest 
   !>                                     ocean level at t-, u- & v-points
   !>                                     (min value = 1)
   !> 
   !> @author J.Paul
   !> @date September, 2015 - rewrite from zgr_top_level
   !>
   ! @param[in] td_nam
   ! @param[in] jpi
   ! @param[in] jpj
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
!      TYPE(TNAMZ), INTENT(IN   ) :: td_nam
!      INTEGER(i4), INTENT(IN   ) :: jpi
!      INTEGER(i4), INTENT(IN   ) :: jpj
      ! local variable

      ! loop indices
!      INTEGER(i4) :: ji
!      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      CALL logger_info('    GRID ZGR TOP LEVEL : ocean top k-index of T-, U-, V- and W-levels ')
      CALL logger_info('    ~~~~~~~~~~~~~')

      ! top k-index of T-level (=1)
      tg_mikt%d_value(:,:,1,1) = MAX( tg_misfdep%d_value(:,:,1,1) , 1._dp )

!      ! top k-index of W-level (=mikt)
!      ! top k-index of U- (U-) level
!      DO jj = 1, jpj-1                       ! top k-index of U- (U-) level
!         DO ji = 1, jpi-1
!            tg_miku%d_value(ji,jj,1,1) = MAX(  tg_mikt%d_value(ji+1,jj  ,1,1), &
!               &                               tg_mikt%d_value(ji  ,jj  ,1,1)  )
!
!            tg_mikv%d_value(ji,jj,1,1) = MAX(  tg_mikt%d_value(ji  ,jj+1,1,1), &
!               &                               tg_mikt%d_value(ji  ,jj  ,1,1)  )
!
!            tg_mikf%d_value(ji,jj,1,1) = MAX(  tg_mikt%d_value(ji  ,jj+1,1,1), &
!               &                               tg_mikt%d_value(ji  ,jj  ,1,1), &
!               &                               tg_mikt%d_value(ji+1,jj  ,1,1), &
!               &                               tg_mikt%d_value(ji+1,jj+1,1,1)  )
!         END DO
!      END DO
!
!      CALL lbc_lnk(tg_miku%d_value(:,:,1,1),'U',td_nam%i_perio,1._dp)
!      CALL lbc_lnk(tg_mikv%d_value(:,:,1,1),'V',td_nam%i_perio,1._dp)
!      CALL lbc_lnk(tg_mikf%d_value(:,:,1,1),'F',td_nam%i_perio,1._dp)

   END SUBROUTINE grid_zgr__top_level
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr_zps_init(jpi, jpj) 
   !-------------------------------------------------------------------
   !> @brief This subroutine initialise global variable needed to compute vertical
   !>        mesh
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

      ! local variable
      REAL(dp), DIMENSION(jpi,jpj) :: dl_tmp

      ! loop indices
      !----------------------------------------------------------------

      dl_tmp(:,:)=dp_fill

      tg_e3tp    =var_init('e3t_ps   ',dl_tmp(:,:), dd_fill=dp_fill, id_type=NF90_DOUBLE)
      tg_e3wp    =var_init('e3w_ps   ',dl_tmp(:,:), dd_fill=dp_fill, id_type=NF90_DOUBLE)

   END SUBROUTINE grid_zgr_zps_init
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr_zps_clean() 
   !-------------------------------------------------------------------
   !> @brief This subroutine clean hgr structure
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial version
   !>
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      ! local variable
      ! loop indices
      !----------------------------------------------------------------

      CALL var_clean(tg_e3tp     )
      CALL var_clean(tg_e3wp     )
      
   END SUBROUTINE grid_zgr_zps_clean
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr__zps_fill(td_nam, jpi, jpj, jpk, td_bathy, td_risfdep) 
   !-------------------------------------------------------------------
   !> @brief This subroutine define the depth and vertical scale factor in partial step
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
   !>        From  depth and scale factors reference, we compute there new value
   !>      with partial steps  on 3d arrays ( i, j, k ).
   !>
   !>              w-level: gdepw_0(i,j,k)  = gdep(k)
   !>                       e3w_0(i,j,k) = dk(gdep)(k)     = e3(i,j,k)
   !>              t-level: gdept_0(i,j,k)  = gdep(k+0.5)
   !>                       e3t_0(i,j,k) = dk(gdep)(k+0.5) = e3(i,j,k+0.5)
   !>
   !>        With the help of the bathymetric file ( bathymetry_depth_ORCA_R2.nc),
   !>      we find the mbathy index of the depth at each grid point.
   !>      This leads us to three cases:
   !>
   !>              - bathy = 0 => mbathy = 0
   !>              - 1 < mbathy < jpkm1    
   !>              - bathy > gdepw_0(jpk) => mbathy = jpkm1  
   !>
   !>        Then, for each case, we find the new depth at t- and w- levels
   !>      and the new vertical scale factors at t-, u-, v-, w-, uw-, vw- 
   !>      and f-points.
   !> 
   !>        This routine is given as an example, it must be modified
   !>      following the user s desiderata. nevertheless, the output as
   !>      well as the way to compute the model levels and scale factors
   !>      must be respected in order to insure second order accuracy
   !>      schemes.
   !>
   !>  @warrning gdept_1d, gdepw_1d and e3._1d are positives
   !>            gdept_0, gdepw_0 and e3. are positives
   !>      
   !>  Reference :   Pacanowsky & Gnanadesikan 1997, Mon. Wea. Rev., 126, 3248-3270.
   !> set 3D coord. arrays to reference 1D array
   !>
   !> @author J.Paul
   !> @date September, 2015 - rewrite from zgr_zps
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   !> @param[in] jpj
   !> @param[in] jpk
   !> @param[inout] td_bathy
   !> @param[inout] td_risfdep
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMZ), INTENT(IN   ) :: td_nam
      INTEGER(i4), INTENT(IN   ) :: jpi
      INTEGER(i4), INTENT(IN   ) :: jpj
      INTEGER(i4), INTENT(IN   ) :: jpk
      TYPE(TVAR) , INTENT(INOUT) :: td_bathy
      TYPE(TVAR) , INTENT(INOUT) :: td_risfdep

      ! local variable
      REAL(dp)    :: zmax             ! Maximum depth
      REAL(dp)    :: zdepwp, zdepth   ! Ajusted ocean depth to avoid too small e3t
      REAL(dp)    :: ze3tp , ze3wp    ! Last ocean level thickness at T- and W-points
      REAL(dp)    :: zdiff            ! temporary scalar

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk

      INTEGER(i4) :: ik
      INTEGER(i4) :: it
      !----------------------------------------------------------------

      CALL logger_info('GRID ZGR ZPS : z-coordinate with partial steps')
      CALL logger_info('mbathy is recomputed : bathy_level file is NOT used')

      ! bathymetry in level (from bathy_meter)

      ! maximum depth (i.e. the last ocean level thickness <= 2*e3t_1d(jpkm1) )
      zmax = tg_gdepw_1d%d_value(1,1,jpk,1) + tg_e3t_1d%d_value(1,1,jpk,1)

      ! bounded value of bathy (min already set at the end of zgr_bat)
      td_bathy%d_value(:,:,1,1) = MIN(zmax, td_bathy%d_value(:,:,1,1))

      WHERE( td_bathy%d_value(:,:,1,1) == 0._dp )
         ! land  : set mbathy to 0
         tg_mbathy%d_value(:,:,1,1) = 0
      ELSE WHERE
         ! ocean : initialize mbathy to the max ocean level
         tg_mbathy%d_value(:,:,1,1) = jpk-1
      END WHERE

      ! Compute mbathy for ocean points (i.e. the number of ocean levels)
      ! find the number of ocean levels such that the last level thickness
      ! is larger than the minimum of e3zps_min and e3zps_rat * e3t_1d (where
      ! e3t_1d is the reference level thickness
      DO jk = jpk-1, 1, -1
         zdepth = tg_gdepw_1d%d_value(1,1,jk,1) +  MIN( td_nam%d_e3zps_min, &
            &                                           td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,jk,1))

         WHERE( 0._dp < td_bathy%d_value(:,:,1,1) .AND. &
            &           td_bathy%d_value(:,:,1,1) <= zdepth )
            tg_mbathy%d_value(:,:,1,1) = jk-1
         END WHERE
      END DO

      ! Scale factors and depth at T- and W-points
      DO jk = 1, jpk                        
         ! intitialization to the reference z-coordinate
         tg_gdept_0%d_value(:,:,jk,1) = tg_gdept_1d%d_value(1,1,jk,1)
         tg_gdepw_0%d_value(:,:,jk,1) = tg_gdepw_1d%d_value(1,1,jk,1)
         tg_e3t_0%d_value  (:,:,jk,1) = tg_e3t_1d%d_value  (1,1,jk,1)
         tg_e3w_0%d_value  (:,:,jk,1) = tg_e3w_1d%d_value  (1,1,jk,1)
      END DO

      IF ( td_nam%l_isfcav )THEN
         ! Bathy, iceshelf draft, scale factor and depth at T- and W- points in case of isf
         CALL grid_zgr__isf_fill( td_nam, jpi,jpj,jpk, td_bathy, td_risfdep )

      ELSE ! .NOT. td_nam%l_isfcav
         !
         DO jj = 1, jpj
            DO ji = 1, jpi
               ik = tg_mbathy%d_value(ji,jj,1,1)
               IF( ik > 0 ) THEN               
                  ! ocean point only
                  IF( ik == jpk-1 ) THEN
                     ! max ocean level case
                     zdepwp = td_bathy%d_value(ji,jj,1,1)
                     ze3tp  = td_bathy%d_value(ji,jj,1,1) - tg_gdepw_1d%d_value(1,1,ik,1)
                     ze3wp  = 0.5_dp * tg_e3w_1d%d_value(1,1,ik,1) &
                        &            * ( 1._dp + ( ze3tp/tg_e3t_1d%d_value(1,1,ik,1) ) )
                     tg_e3t_0%d_value  (ji,jj,ik  ,1) = ze3tp
                     tg_e3t_0%d_value  (ji,jj,ik+1,1) = ze3tp
                     tg_e3w_0%d_value  (ji,jj,ik  ,1) = ze3wp
                     tg_e3w_0%d_value  (ji,jj,ik+1,1) = ze3tp

                     tg_gdepw_0%d_value(ji,jj,ik+1,1) = zdepwp
                     tg_gdept_0%d_value(ji,jj,ik  ,1) = tg_gdept_1d%d_value( 1, 1,ik-1,1) + ze3wp
                     tg_gdept_0%d_value(ji,jj,ik+1,1) = tg_gdept_0%d_value (ji,jj,ik  ,1) + ze3tp
                     !
                  ELSE
                     ! standard case
                     IF( td_bathy%d_value(ji,jj,1,1) <= tg_gdepw_1d%d_value(1,1,ik+1,1) ) THEN
                        tg_gdepw_0%d_value(ji,jj,ik+1,1) = td_bathy%d_value(ji,jj,1,1)
                     ELSE
                        tg_gdepw_0%d_value(ji,jj,ik+1,1) = tg_gdepw_1d%d_value(1,1,ik+1,1)
                     ENDIF
                     !gm Bug?  check the gdepw_1d
                     !       ... on ik
                     tg_gdept_0%d_value(ji,jj,ik,1) = tg_gdepw_1d%d_value( 1, 1,ik  ,1) + &
                     &        ( tg_gdepw_0%d_value(ji,jj,ik+1,1) - tg_gdepw_1d%d_value(1,1,ik,1) )  &
                     &    * ( (tg_gdept_1d%d_value( 1, 1,ik  ,1) - tg_gdepw_1d%d_value(1,1,ik,1) )  &
                     &      / (tg_gdepw_1d%d_value( 1, 1,ik+1,1) - tg_gdepw_1d%d_value(1,1,ik,1) ) )

                     tg_e3t_0%d_value  (ji,jj,ik,1) = tg_e3t_1d%d_value  ( 1, 1,ik  ,1)            &
                     &     * ( tg_gdepw_0%d_value (ji,jj,ik+1,1) - tg_gdepw_1d%d_value(1,1,ik,1) ) &
                     &     / ( tg_gdepw_1d%d_value( 1, 1,ik+1,1) - tg_gdepw_1d%d_value(1,1,ik,1) ) 

                     tg_e3w_0%d_value  (ji,jj,ik,1) = 0.5_dp   &
                     &                               * ( tg_gdepw_0%d_value(ji,jj,ik+1,1) &
                     &                                 + tg_gdepw_1d%d_value(1,1 ,ik+1,1) &
                     &                                 - tg_gdepw_1d%d_value(1,1 ,ik  ,1) * 2._dp ) &
                     &                               * ( tg_e3w_1d%d_value (1 ,1 ,ik  ,1) &
                     &                                 / (tg_gdepw_1d%d_value(1,1,ik+1,1) - tg_gdepw_1d%d_value(1,1,ik,1)) )

                     !       ... on ik+1
                     tg_e3w_0%d_value  (ji,jj,ik+1,1) = tg_e3t_0%d_value  (ji,jj,ik,1)
                     tg_e3t_0%d_value  (ji,jj,ik+1,1) = tg_e3t_0%d_value  (ji,jj,ik,1)
                     tg_gdept_0%d_value(ji,jj,ik+1,1) = tg_gdept_0%d_value(ji,jj,ik,1) + tg_e3t_0%d_value(ji,jj,ik,1)
                  ENDIF
               ENDIF
            END DO
         END DO
         !
         it = 0
         DO jj = 1, jpj
            DO ji = 1, jpi
               ik = tg_mbathy%d_value(ji,jj,1,1)
               IF( ik > 0 ) THEN               ! ocean point only
                  tg_e3tp%d_value (ji,jj,1,1) = tg_e3t_0%d_value(ji,jj,ik,1)
                  tg_e3wp%d_value (ji,jj,1,1) = tg_e3w_0%d_value(ji,jj,ik,1)
                  ! test
                  zdiff= tg_gdepw_0%d_value(ji,jj,ik+1,1) - tg_gdept_0%d_value(ji,jj,ik,1)
                  IF( zdiff <= 0._dp ) THEN 
                     it = it + 1
                     CALL logger_info(' it      = '//TRIM(fct_str(it))//&
                        &             ' ik      = '//TRIM(fct_str(ik))//&
                        &             ' (i,j)   = ('//TRIM(fct_str(ji))//','//TRIM(fct_str(jj))//')')
                     CALL logger_info(' bathy = '//TRIM(fct_str(td_bathy%d_value(ji,jj,1,1))))
                     CALL logger_info(' gdept_0 = '//TRIM(fct_str( tg_gdept_0%d_value(ji,jj,ik  ,1) ))//&
                        &             ' gdepw_0 = '//TRIM(fct_str( tg_gdepw_0%d_value(ji,jj,ik+1,1) ))//&
                        &             ' zdiff = '//TRIM(fct_str(zdiff)) )
                     CALL logger_info(' e3tp    = '//TRIM(fct_str(tg_e3t_0%d_value(ji,jj,ik,1)))//&
                        &             ' e3wp    = '//TRIM(fct_str(tg_e3w_0%d_value(ji,jj,ik,1)))  )
                  ENDIF
               ENDIF
            END DO
         END DO
      ENDIF
      !
!      IF ( td_nam%l_isfcav ) THEN
!      ! (ISF) Definition of e3t, u, v, w for ISF case
!         CALL grid_zgr__isf_fill_e3x( jpi,jpj, &
!            &                        td_risfdep )
!      END IF

      ! Scale factors and depth at U-, V-, UW and VW-points
      DO jk = 1, jpk
         ! initialisation to z-scale factors
         tg_e3u_0%d_value (:,:,jk,1) = tg_e3t_1d%d_value(1,1,jk,1)
         tg_e3v_0%d_value (:,:,jk,1) = tg_e3t_1d%d_value(1,1,jk,1)
         tg_e3uw_0%d_value(:,:,jk,1) = tg_e3w_1d%d_value(1,1,jk,1)
         tg_e3vw_0%d_value(:,:,jk,1) = tg_e3w_1d%d_value(1,1,jk,1)
      END DO

      ! Computed as the minimum of neighbooring scale factors
      DO jk = 1,jpk
         DO jj = 1, jpj - 1
            DO ji = 1, jpi - 1 
               tg_e3u_0%d_value (ji,jj,jk,1) = MIN( tg_e3t_0%d_value(ji,jj,jk,1), tg_e3t_0%d_value(ji+1,jj  ,jk,1) )
               tg_e3v_0%d_value (ji,jj,jk,1) = MIN( tg_e3t_0%d_value(ji,jj,jk,1), tg_e3t_0%d_value(ji  ,jj+1,jk,1) )
               tg_e3uw_0%d_value(ji,jj,jk,1) = MIN( tg_e3w_0%d_value(ji,jj,jk,1), tg_e3w_0%d_value(ji+1,jj  ,jk,1) )
               tg_e3vw_0%d_value(ji,jj,jk,1) = MIN( tg_e3w_0%d_value(ji,jj,jk,1), tg_e3w_0%d_value(ji  ,jj+1,jk,1) )
            END DO
         END DO
      END DO

      IF ( td_nam%l_isfcav ) THEN
         ! (ISF) define e3uw (adapted for 2 cells in the water column)
         CALL grid_zgr__isf_fill_e3uw(jpi,jpj)
      END IF

      ! lateral boundary conditions
      CALL lbc_lnk( tg_e3u_0%d_value (:,:,:,1), 'U', td_nam%i_perio, 1._dp )
      CALL lbc_lnk( tg_e3v_0%d_value (:,:,:,1), 'V', td_nam%i_perio, 1._dp )
      CALL lbc_lnk( tg_e3uw_0%d_value(:,:,:,1), 'U', td_nam%i_perio, 1._dp )
      CALL lbc_lnk( tg_e3vw_0%d_value(:,:,:,1), 'V', td_nam%i_perio, 1._dp )

      ! set to z-scale factor if zero (i.e. along closed boundaries)
      DO jk = 1, jpk
         WHERE( tg_e3u_0%d_value (:,:,jk,1) == 0._dp )   tg_e3u_0%d_value (:,:,jk,1) = tg_e3t_1d%d_value(1,1,jk,1)
         WHERE( tg_e3v_0%d_value (:,:,jk,1) == 0._dp )   tg_e3v_0%d_value (:,:,jk,1) = tg_e3t_1d%d_value(1,1,jk,1)
         WHERE( tg_e3uw_0%d_value(:,:,jk,1) == 0._dp )   tg_e3uw_0%d_value(:,:,jk,1) = tg_e3w_1d%d_value(1,1,jk,1)
         WHERE( tg_e3vw_0%d_value(:,:,jk,1) == 0._dp )   tg_e3vw_0%d_value(:,:,jk,1) = tg_e3w_1d%d_value(1,1,jk,1)
      END DO

      !! Scale factor at F-point
      !DO jk = 1, jpk                        
      !   ! initialisation to z-scale factors
      !   tg_e3f_0%d_value(:,:,jk,1) = tg_e3t_1d%d_value(1,1,jk,1)
      !END DO
      !
      !! Computed as the minimum of neighbooring V-scale factors
      !DO jk = 1, jpk
      !   DO jj = 1, jpj - 1
      !      DO ji = 1, jpi - 1
      !         tg_e3f_0%d_value(ji,jj,jk,1) = MIN( tg_e3v_0%d_value(ji,jj,jk,1), tg_e3v_0%d_value(ji+1,jj,jk,1) )
      !      END DO
      !   END DO
      !END DO
      !! Lateral boundary conditions
      !CALL lbc_lnk( tg_e3f_0%d_value(:,:,:,1), 'F', td_nam%i_perio, 1._dp )
      !
      !! set to z-scale factor if zero (i.e. along closed boundaries)
      !DO jk = 1, jpk
      !   WHERE( tg_e3f_0%d_value(:,:,jk,1) == 0._dp )   tg_e3f_0%d_value(:,:,jk,1) = tg_e3t_1d%d_value(1,1,jk,1)
      !END DO

!!gm  bug ? :  must be a do loop with mj0,mj1

      ! we duplicate factor scales for jj = 1 and jj = 2
      tg_e3t_0%d_value(:,1,:,1) = tg_e3t_0%d_value(:,2,:,1)
      tg_e3w_0%d_value(:,1,:,1) = tg_e3w_0%d_value(:,2,:,1) 
      tg_e3u_0%d_value(:,1,:,1) = tg_e3u_0%d_value(:,2,:,1) 
      tg_e3v_0%d_value(:,1,:,1) = tg_e3v_0%d_value(:,2,:,1) 
      !tg_e3f_0%d_value(:,1,:,1) = tg_e3f_0%d_value(:,2,:,1)

      ! Control of the sign
      IF( MINVAL( tg_e3t_0%d_value  (:,:,:,:) ) <= 0._dp )   CALL logger_fatal( ' GRID ZGR ZPS:   e r r o r   e3t_0 <= 0' )
      IF( MINVAL( tg_e3w_0%d_value  (:,:,:,:) ) <= 0._dp )   CALL logger_fatal( ' GRID ZGR ZPS:   e r r o r   e3w_0 <= 0' )
      IF( MINVAL( tg_gdept_0%d_value(:,:,:,:) ) <  0._dp )   CALL logger_fatal( ' GRID ZGR ZPS:   e r r o r   gdept_0 <  0' )
      IF( MINVAL( tg_gdepw_0%d_value(:,:,:,:) ) <  0._dp )   CALL logger_fatal( ' GRID ZGR ZPS:   e r r o r   gdepw_0 <  0' )

      !! Compute gdep3w_0 (vertical sum of e3w)
      !IF ( td_nam%l_isfcav ) THEN 
      !   ! if cavity
      !   CALL grid_zgr__isf_fill_gdep3w_0(jpi, jpj, jpk, td_risfdep)
      !ELSE 
      !   ! no cavity
      !   tg_gdep3w_0%d_value(:,:,1,1) = 0.5_dp * tg_e3w_0%d_value(:,:,1,1)
      !   DO jk = 2, jpk
      !      tg_gdep3w_0%d_value(:,:,jk,1) = tg_gdep3w_0%d_value(:,:,jk-1,1) + tg_e3w_0%d_value(:,:,jk,1)
      !   END DO
      !END IF

   END SUBROUTINE grid_zgr__zps_fill
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr__isf_fill(td_nam, jpi,jpj,jpk, td_bathy, td_risfdep) 
   !-------------------------------------------------------------------
   !> @brief This subroutine check the bathymetry in levels
   !>
   !> @details
   !> ** Method  :   THe water column have to contained at least 2 cells
   !>                Bathymetry and isfdraft are modified (dig/close) to respect
   !>                this criterion.
   !>                 
   !>   
   !> ** Action  : - test compatibility between isfdraft and bathy 
   !>              - bathy and isfdraft are modified   
   !> 
   !> @author J.Paul
   !> @date September, 2015 - rewrite from zgr_isf
   !> @date October, 2016
   !> - add ice sheet coupling case
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   !> @param[in] jpj
   !> @param[in] jpk
   !> @param[in] td_bathy
   !> @param[in] td_risfdep
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMZ), INTENT(IN   ) :: td_nam
      INTEGER(i4), INTENT(IN   ) :: jpi
      INTEGER(i4), INTENT(IN   ) :: jpj
      INTEGER(i4), INTENT(IN   ) :: jpk
      TYPE(TVAR) , INTENT(INOUT) :: td_bathy
      TYPE(TVAR) , INTENT(INOUT) :: td_risfdep

      ! local variable
      INTEGER(i4) :: ik, it
      INTEGER(i4) :: ibtest, ibtestim1, ibtestip1, ibtestjm1, ibtestjp1   ! (ISF)

      INTEGER(i4), ALLOCATABLE, DIMENSION(:,:) :: zmbathy, zmisfdep     ! 2D workspace (ISH)

      REAL(dp)    :: zdepth           ! Ajusted ocean depth to avoid too small e3t
      REAL(dp)    :: zmax             ! Maximum and minimum depth
      REAL(dp)    :: zbathydiff, zrisfdepdiff  ! isf temporary scalar
      REAL(dp)    :: zdepwp           ! Ajusted ocean depth to avoid too small e3t
      REAL(dp)    :: ze3tp , ze3wp    ! Last ocean level thickness at T- and W-points
      REAL(dp)    :: zdiff            ! temporary scalar

      REAL(dp), ALLOCATABLE, DIMENSION(:,:) :: zrisfdep, zmask   ! 2D workspace (ISH)

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      ! (ISF) compute misfdep
      WHERE( td_risfdep%d_value(:,:,1,1) == 0._dp .AND. &
           & td_bathy%d_value  (:,:,1,1) /= 0 )
        ! open water : set misfdep to 1
        tg_misfdep%d_value(:,:,1,1) = 1
      ELSEWHERE
         ! iceshelf : initialize misfdep to second level
         tg_misfdep%d_value(:,:,1,1) = 2
      END WHERE

      ALLOCATE(zmask   (jpi,jpj))
      ALLOCATE(zrisfdep(jpi,jpj))
      ALLOCATE(zmisfdep(jpi,jpj))

      ! Compute misfdep for ocean points (i.e. first wet level) 
      ! find the first ocean level such that the first level thickness 
      ! is larger than the bot_level of e3zps_min and e3zps_rat * e3t_0 (where 
      ! e3t_0 is the reference level thickness
      DO jk = 2, jpk-1 
         zdepth = tg_gdepw_1d%d_value(1,1,jk+1,1) - MIN( td_nam%d_e3zps_min, &
            &                                            td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,jk,1) ) 
         WHERE( 0._dp < td_risfdep%d_value(:,:,1,1) .AND. &
         &              td_risfdep%d_value(:,:,1,1) >= zdepth )
            tg_misfdep%d_value(:,:,1,1) = jk+1 
         END WHERE
      END DO 

      WHERE( 0._dp < td_risfdep%d_value(:,:,1,1) .AND. &
           &         td_risfdep%d_value(:,:,1,1) <= tg_e3t_1d%d_value(1,1,1,1) )
         td_risfdep%d_value(:,:,1,1) = 0.
         tg_misfdep%d_value(:,:,1,1) = 1
      END WHERE

      ! remove very shallow ice shelf (less than ~ 10m if 75L)
      WHERE( td_risfdep%d_value(:,:,1,1) <= 10._dp .AND. &
           & tg_misfdep%d_value(:,:,1,1) > 1 )
         tg_misfdep%d_value(:,:,1,1) = 0
         td_risfdep%d_value(:,:,1,1) = 0.0_dp
         tg_mbathy%d_value (:,:,1,1) = 0
         td_bathy%d_value  (:,:,1,1) = 0.0_dp
      END WHERE
      WHERE( td_bathy%d_value(:,:,1,1) <= 30.0_dp .AND. &
           & tg_gphit%d_value(:,:,1,1) < -60._dp )
         tg_misfdep%d_value(:,:,1,1) = 0
         td_risfdep%d_value(:,:,1,1) = 0.0_dp
         tg_mbathy%d_value (:,:,1,1) = 0
         td_bathy%d_value  (:,:,1,1) = 0.0_dp
      END WHERE

      ! basic check for the compatibility of bathy and risfdep. 
      ! I think it should be offline because it is not perfect and cannot solved all the situation
      ! run the bathy check 10 times to be sure all the modif in the bathy or iceshelf draft are compatible together
      DO jl = 1, 10

         WHERE( td_bathy%d_value(:,:,1,1) <= td_risfdep%d_value(:,:,1,1) + td_nam%d_isfhmin )
            tg_misfdep%d_value(:,:,1,1) = 0
            td_risfdep%d_value(:,:,1,1) = 0._dp
            tg_mbathy%d_value (:,:,1,1) = 0
            td_bathy%d_value  (:,:,1,1) = 0._dp
         END WHERE

         WHERE( tg_mbathy%d_value(:,:,1,1) <= 0 )
            tg_misfdep%d_value(:,:,1,1) = 0 
            td_risfdep%d_value(:,:,1,1) = 0._dp
            tg_mbathy%d_value (:,:,1,1) = 0
            td_bathy%d_value  (:,:,1,1) = 0._dp
         ENDWHERE

         !! lk_mpp not added

         IF( td_nam%i_perio == 1 .OR. &
           & td_nam%i_perio == 4 .OR. &
           & td_nam%i_perio == 6 )THEN 
            ! local domain is cyclic east-west
            tg_misfdep%d_value( 1 ,:,1,1) = tg_misfdep%d_value(jpi-1,:,1,1)
            tg_misfdep%d_value(jpi,:,1,1) = tg_misfdep%d_value(  2  ,:,1,1) 

            tg_mbathy%d_value( 1 ,:,1,1) = tg_mbathy%d_value(jpi-1,:,1,1)
            tg_mbathy%d_value(jpi,:,1,1) = tg_mbathy%d_value(  2  ,:,1,1)
         ENDIF

         ! split last cell if possible (only where water column is 2 cell or less)
         ! if coupled to ice sheet, we do not modify the bathymetry (can be discuss).
         IF( .NOT. td_nam%l_iscpl )THEN
            DO jk = jpk-1, 1, -1
               zmax = tg_gdepw_1d%d_value(1,1,jk,1) + &
                  &   MIN( td_nam%d_e3zps_min, &
                  &        td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,jk,1) )

               WHERE( td_bathy%d_value(:,:,1,1)       >  tg_gdepw_1d%d_value(1,1,jk,1).AND. &
                   &  td_bathy%d_value(:,:,1,1)       <= zmax                         .AND. &
                   &  tg_misfdep%d_value(:,:,1,1) + 1 >= tg_mbathy%d_value(:,:,1,1) )

                  tg_mbathy%d_value(:,:,1,1) = jk
                  td_bathy%d_value (:,:,1,1) = zmax

               END WHERE
            END DO
         ENDIF

         ! split top cell if possible (only where water column is 2 cell or less)
         DO jk = 2, jpk-1
            zmax = tg_gdepw_1d%d_value(1,1,jk+1,1) - &
               &   MIN( td_nam%d_e3zps_min, &
               &        td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,jk,1) )

            WHERE( td_risfdep%d_value(:,:,1,1)     <  tg_gdepw_1d%d_value(1,1,jk+1,1) .AND. &
                &  td_risfdep%d_value(:,:,1,1)     >= zmax                            .AND. &
                &  tg_misfdep%d_value(:,:,1,1) + 1 >= tg_mbathy%d_value(:,:,1,1) )

               tg_misfdep%d_value(:,:,1,1) = jk
               td_risfdep%d_value(:,:,1,1) = zmax

            END WHERE
         END DO

         ! Case where bathy and risfdep compatible but not the level 
         ! variable mbathy/misfdep because of partial cell condition
         DO jj = 1, jpj
            DO ji = 1, jpi
               ! find the minimum change option:
               ! test bathy
               IF( td_risfdep%d_value(ji,jj,1,1) > 1 )THEN
                  IF( .NOT. td_nam%l_iscpl )THEN

                     ik=tg_mbathy%d_value(ji,jj,1,1)
                     zbathydiff = ABS( td_bathy%d_value(ji,jj,1,1) - ( tg_gdepw_1d%d_value(1,1,ik+1,1) &
                        &              + MIN(td_nam%d_e3zps_min,                                       &
                        &                    td_nam%d_e3zps_rat * tg_e3t_1d%d_value(1,1,ik+1,1)) ))

                     ik=tg_misfdep%d_value(ji,jj,1,1)
                     zrisfdepdiff = ABS( td_risfdep%d_value(ji,jj,1,1) - ( tg_gdepw_1d%d_value(1,1,ik,1) &
                        &                - MIN( td_nam%d_e3zps_min,                                      &
                        &                       td_nam%d_e3zps_rat * tg_e3t_1d%d_value(1,1,ik-1,1)) ))
 
                     IF( td_bathy%d_value (ji,jj,1,1) > td_risfdep%d_value (ji,jj,1,1) .AND. &
                      &  tg_mbathy%d_value(ji,jj,1,1) < tg_misfdep%d_value(ji,jj,1,1) )THEN

                        IF( zbathydiff <= zrisfdepdiff )THEN

                           tg_mbathy%d_value(ji,jj,1,1) = tg_mbathy%d_value(ji,jj,1,1) + 1

                           ik=tg_mbathy%d_value(ji,jj,1,1)
                           td_bathy%d_value (ji,jj,1,1) = tg_gdepw_1d%d_value(1,1,ik,1) + &
                              &                           MIN( td_nam%d_e3zps_min, &
                              &                                td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik+1,1) )
                        ELSE

                           tg_misfdep%d_value(ji,jj,1,1) = tg_misfdep%d_value(ji,jj,1,1) - 1

                           ik=tg_misfdep%d_value(ji,jj,1,1)
                           td_risfdep%d_value(ji,jj,1,1) = tg_gdepw_1d%d_value(1,1,ik,1) - &
                              &                            MIN( td_nam%d_e3zps_min, &
                              &                                 td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik-1,1) )
                        ENDIF

                     ENDIF

                  ELSE

                     IF( td_bathy%d_value (ji,jj,1,1) > td_risfdep%d_value (ji,jj,1,1) .AND. &
                       & tg_mbathy%d_value(ji,jj,1,1) < tg_misfdep%d_value(ji,jj,1,1) )THEN

                        tg_misfdep%d_value(ji,jj,1,1) = tg_misfdep%d_value(ji,jj,1,1) - 1
                    
                        ik=tg_misfdep%d_value(ji,jj,1,1)
                        td_risfdep%d_value(ji,jj,1,1) = tg_gdepw_1d%d_value(1,1,ik,1) - &
                           &                            MIN( td_nam%d_e3zps_min, &
                           &                                 td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik-1,1) )
                     ENDIF

                  ENDIF

               ENDIF
            ENDDO
         ENDDO

          ! At least 2 levels for water thickness at T, U, and V point.
         DO jj = 1, jpj
            DO ji = 1, jpi
               ! find the minimum change option:
               ! test bathy
               IF( tg_misfdep%d_value(ji,jj,1,1) == tg_mbathy%d_value(ji,jj,1,1) .AND. &
                 & tg_mbathy%d_value (ji,jj,1,1) > 1) THEN

                  IF ( .NOT. td_nam%l_iscpl ) THEN

                     ik=tg_mbathy%d_value(ji,jj,1,1)
                     zbathydiff  = ABS( td_bathy%d_value(ji,jj,1,1)  - ( tg_gdepw_1d%d_value(1,1,ik+1,1) &
                        &               + MIN( td_nam%d_e3zps_min, &
                        &                      td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik+1,1)) ))

                     ik=tg_misfdep%d_value(ji,jj,1,1)
                     zrisfdepdiff = ABS( td_risfdep%d_value(ji,jj,1,1) - ( tg_gdepw_1d%d_value(1,1,ik,1) &
                        &                - MIN( td_nam%d_e3zps_min, &
                        &                       td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik-1,1)) ))

                     IF( zbathydiff <= zrisfdepdiff )THEN

                        tg_mbathy%d_value(ji,jj,1,1) = tg_mbathy%d_value(ji,jj,1,1) + 1

                        ik=tg_mbathy%d_value(ji,jj,1,1)
                        td_bathy%d_value(ji,jj,1,1)  = tg_gdepw_1d%d_value(1,1,ik,1) + &
                           &                           MIN( td_nam%d_e3zps_min, &
                           &                                td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik+1,1) )
                     ELSE

                        tg_misfdep%d_value(ji,jj,1,1) = tg_misfdep%d_value(ji,jj,1,1) - 1

                        ik=tg_misfdep%d_value(ji,jj,1,1)
                        td_risfdep%d_value(ji,jj,1,1)  = tg_gdepw_1d%d_value(1,1,ik+1,1) - &
                           &                            MIN( td_nam%d_e3zps_min, &
                           &                                 td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik,1) )
                     ENDIF

                  ELSE

                     tg_misfdep%d_value(ji,jj,1,1) = tg_misfdep%d_value(ji,jj,1,1) - 1

                     ik=tg_misfdep%d_value(ji,jj,1,1)
                     td_risfdep%d_value(ji,jj,1,1)  = tg_gdepw_1d%d_value(1,1,ik+1,1) - &
                        &                            MIN( td_nam%d_e3zps_min, &
                        &                                 td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik,1) )

                  ENDIF
               ENDIF

            ENDDO
         ENDDO

         ! point V mbathy(ji,jj  ) == misfdep(ji,jj+1) 
         DO jj = 1, jpj-1
            DO ji = 1, jpi-1

               IF( tg_mbathy%d_value(ji,jj  ,1,1) == tg_misfdep%d_value(ji,jj+1,1,1) .AND. &
                &  tg_mbathy%d_value(ji,jj  ,1,1) >  1 )THEN
                  IF ( .NOT. td_nam%l_iscpl ) THEN

                     ik=tg_mbathy%d_value(ji,jj  ,1,1)
                     zbathydiff = ABS( td_bathy%d_value(ji,jj  ,1,1) - ( tg_gdepw_1d%d_value(1,1,ik+1,1) &
                        &              + MIN( td_nam%d_e3zps_min,                                        &
                        &                     td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik+1,1)) ))

                     ik=tg_misfdep%d_value(ji,jj+1,1,1)
                     zrisfdepdiff = ABS( td_risfdep%d_value(ji,jj+1,1,1) - ( tg_gdepw_1d%d_value(1,1,ik,1) &
                        &                - MIN( td_nam%d_e3zps_min,                                        &
                        &                       td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik-1,1)) ))

                     IF( zbathydiff <= zrisfdepdiff )THEN

                        tg_mbathy%d_value(ji,jj  ,1,1) = tg_mbathy%d_value(ji,jj  ,1,1) + 1

                        ik=tg_mbathy%d_value(ji,jj  ,1,1)
                        td_bathy%d_value (ji,jj  ,1,1) = tg_gdepw_1d%d_value(1,1,ik,1) + &
                           &                           MIN( td_nam%d_e3zps_min, &
                           &                                td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik+1,1) )
                     ELSE

                        tg_misfdep%d_value(ji,jj+1,1,1) = tg_misfdep%d_value(ji,jj+1,1,1) - 1

                        ik=tg_misfdep%d_value(ji,jj+1,1,1)
                        td_risfdep%d_value(ji,jj+1,1,1) = tg_gdepw_1d%d_value(1,1,ik+1,1) - &
                           &                             MIN( td_nam%d_e3zps_min, &
                           &                                  td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik,1) )
                     ENDIF

                  ELSE
                     tg_misfdep%d_value(ji,jj+1,1,1) = tg_misfdep%d_value(ji,jj+1,1,1) - 1

                     ik=tg_misfdep%d_value(ji,jj+1,1,1)
                     td_risfdep%d_value(ji,jj+1,1,1) = tg_gdepw_1d%d_value(1,1,ik+1,1) - &
                        &                             MIN( td_nam%d_e3zps_min, &
                        &                                  td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik,1) )
                  ENDIF

               ENDIF

            ENDDO
         ENDDO

         !! lk_mpp not added

         ! point V mbathy(ji,jj+1) == misfdep(ji,jj  ) 
         DO jj = 1, jpj-1
            DO ji = 1, jpi-1

               IF( tg_mbathy%d_value(ji,jj+1,1,1) == tg_misfdep%d_value(ji,jj  ,1,1) .AND. &
                &  tg_mbathy%d_value(ji,jj  ,1,1) > 1) THEN
                  IF ( .NOT. td_nam%l_iscpl ) THEN

                     ik=tg_mbathy%d_value(ji,jj+1,1,1)
                     zbathydiff = ABS( td_bathy%d_value(ji,jj+1,1,1) - ( tg_gdepw_1d%d_value(1,1,ik+1,1) &
                        &              + MIN( td_nam%d_e3zps_min,                                        &
                        &                     td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik+1,1)) ))

                     ik=tg_misfdep%d_value(ji,jj  ,1,1)
                     zrisfdepdiff = ABS( td_risfdep%d_value(ji,jj,1,1) - ( tg_gdepw_1d%d_value(1,1,ik,1) &
                        &                - MIN( td_nam%d_e3zps_min,                                      &
                        &                       td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik-1,1)) ))

                     IF( zbathydiff <= zrisfdepdiff )THEN

                        tg_mbathy%d_value (ji,jj+1,1,1) = tg_mbathy%d_value(ji,jj+1,1,1) + 1

                        ik=tg_mbathy%d_value(ji,jj+1,1,1)
                        td_bathy%d_value  (ji,jj+1,1,1) = tg_gdepw_1d%d_value(1,1,ik,1) + &
                           &                            MIN( td_nam%d_e3zps_min,          &
                           &                                 td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik+1,1) )
                     ELSE

                        tg_misfdep%d_value(ji,jj  ,1,1) = tg_misfdep%d_value(ji,jj  ,1,1) - 1

                        ik=tg_misfdep%d_value(ji,jj  ,1,1)
                        td_risfdep%d_value(ji,jj  ,1,1) = tg_gdepw_1d%d_value(1,1,ik+1,1) - &
                           &                           MIN( td_nam%d_e3zps_min,              &
                           &                                td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik,1) )
                     END IF

                  ELSE

                     tg_misfdep%d_value(ji,jj  ,1,1) = tg_misfdep%d_value(ji,jj  ,1,1) - 1

                     ik=tg_misfdep%d_value(ji,jj  ,1,1)
                     td_risfdep%d_value(ji,jj  ,1,1) = tg_gdepw_1d%d_value(1,1,ik+1,1) - &
                        &                           MIN( td_nam%d_e3zps_min,           &
                        &                                td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik,1) )

                  ENDIF
               
               ENDIF

            ENDDO
         ENDDO         

         !! lk_mpp not added

         ! point U mbathy(ji  ,jj) EQ misfdep(ji+1,jj)
         DO jj = 1, jpj-1
            DO ji = 1, jpi-1

               IF( tg_mbathy%d_value(ji  ,jj,1,1) == tg_misfdep%d_value(ji+1,jj,1,1) .AND. &
                &  tg_mbathy%d_value(ji  ,jj,1,1) > 1 )THEN
                  IF ( .NOT. td_nam%l_iscpl ) THEN

                     ik=tg_mbathy%d_value(ji  ,jj,1,1)
                     zbathydiff = ABS(     td_bathy%d_value(ji  ,jj,1,1) - ( tg_gdepw_1d%d_value(1,1,ik+1,1) &
                        &              + MIN( td_nam%d_e3zps_min,                                     &
                        &                     td_nam%d_e3zps_rat* tg_e3t_1d%d_value(1,1,ik+1,1)) ))

                     ik=tg_misfdep%d_value(ji+1,jj,1,1)
                     zrisfdepdiff = ABS( td_risfdep%d_value(ji+1,jj,1,1) - ( tg_gdepw_1d%d_value(1,1,ik,1) &
                        &                - MIN( td_nam%d_e3zps_min,                                       &
                        &                       td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik-1,1)) ))

                     IF( zbathydiff <= zrisfdepdiff )THEN

                        tg_mbathy%d_value (ji  ,jj,1,1) = tg_mbathy%d_value(ji  ,jj,1,1) + 1

                        ik=tg_mbathy%d_value(ji  ,jj,1,1)
                        td_bathy%d_value  (ji  ,jj,1,1) = tg_gdepw_1d%d_value(1,1,ik,1) + &
                           &                           MIN( td_nam%d_e3zps_min, &
                           &                                td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik+1,1) )
                     ELSE
                        tg_misfdep%d_value(ji+1,jj,1,1) = tg_misfdep%d_value(ji+1,jj,1,1) - 1

                        ik=tg_misfdep%d_value(ji+1,jj,1,1)
                        td_risfdep%d_value(ji+1,jj,1,1) = tg_gdepw_1d%d_value(1,1,ik+1,1) - &
                           &                             MIN( td_nam%d_e3zps_min, &
                           &                                  td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik,1) )
                     END IF

                  ELSE
                     tg_misfdep%d_value(ji+1,jj,1,1)= tg_misfdep%d_value(ji+1,jj,1,1) - 1

                     ik=tg_misfdep%d_value(ji+1,jj,1,1)
                     td_risfdep%d_value(ji+1,jj,1,1) = tg_gdepw_1d%d_value(1,1,ik+1,1) - &
                        &                             MIN( td_nam%d_e3zps_min, &
                        &                                  td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik,1) )
                  ENDIF
               ENDIF

            ENDDO
         ENDDO

         !! lk_mpp not added

         ! point U mbathy(ji+1,jj) == misfdep(ji  ,jj) 
         DO jj = 1, jpj-1
            DO ji = 1, jpi-1

               IF( tg_mbathy%d_value(ji+1,jj,1,1) == tg_misfdep%d_value(ji  ,jj,1,1) .AND. &
                &  tg_mbathy%d_value(ji  ,jj,1,1) > 1 )THEN
                  IF ( .NOT. td_nam%l_iscpl ) THEN

                     ik=tg_mbathy%d_value(ji+1,jj,1,1)
                     zbathydiff = ABS(     td_bathy%d_value(ji+1,jj,1,1) - ( tg_gdepw_1d%d_value(1,1,ik+1,1) &
                        &              + MIN( td_nam%d_e3zps_min,                                        &
                        &                     td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik+1,1)) ))

                     ik=tg_misfdep%d_value(ji ,jj,1,1)
                     zrisfdepdiff = ABS( td_risfdep%d_value(ji  ,jj,1,1) - ( tg_gdepw_1d%d_value(1,1,ik,1) &
                        &                - MIN( td_nam%d_e3zps_min,                                      &
                        &                       td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik-1,1)) ))

                     IF( zbathydiff <= zrisfdepdiff )THEN

                        tg_mbathy%d_value (ji+1,jj,1,1) = tg_mbathy%d_value(ji+1,jj,1,1) + 1

                        ik=tg_mbathy%d_value(ji+1,jj,1,1)
                        td_bathy%d_value  (ji+1,jj,1,1) = tg_gdepw_1d%d_value(1,1,ik,1) + &
                           &                            MIN( td_nam%d_e3zps_min, &
                           &                                 td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik+1,1) )
                     ELSE
                        tg_misfdep%d_value(ji  ,jj,1,1) = tg_misfdep%d_value(ji  ,jj,1,1) - 1

                        ik=tg_misfdep%d_value(ji  ,jj,1,1)
                        td_risfdep%d_value(ji  ,jj,1,1) = tg_gdepw_1d%d_value(1,1,ik+1,1) - &
                           &                           MIN( td_nam%d_e3zps_min, &
                           &                                td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik,1) )
                     END IF

                  ELSE

                     tg_misfdep%d_value(ji ,jj,1,1) = tg_misfdep%d_value(ji ,jj,1,1) - 1

                     ik=tg_misfdep%d_value(ji  ,jj,1,1)
                     td_risfdep%d_value(ji  ,jj,1,1) = tg_gdepw_1d%d_value(1,1,ik+1,1) - &
                        &                           MIN( td_nam%d_e3zps_min, &
                        &                                td_nam%d_e3zps_rat*tg_e3t_1d%d_value(1,1,ik,1) )

                  ENDIF

               ENDIF

            ENDDO
         ENDDO

      END DO ! jl 
      ! end dig bathy/ice shelf to be compatible

      ! now fill single point in "coastline" of ice shelf, bathy, hole, and test again one cell tickness
      DO jl = 1,20

         ! remove single point "bay" on isf coast line in the ice shelf draft'
         DO jk = 2, jpk
            WHERE( tg_misfdep%d_value(:,:,1,1) == 0 )
               tg_misfdep%d_value(:,:,1,1)=jpk
            END WHERE

            zmask(:,:)=0
            WHERE( tg_misfdep%d_value(:,:,1,1) <= jk ) zmask(:,:)=1
            
            DO jj = 2, jpj-1
               DO ji = 2, jpi-1
                  IF( tg_misfdep%d_value(ji,jj,1,1) == jk )THEN

                     ibtest =   zmask(ji-1,jj  ) &
                        &     + zmask(ji+1,jj  ) &
                        &     + zmask(ji  ,jj-1) &
                        &     + zmask(ji  ,jj+1)
                     IF( ibtest <= 1 )THEN
                        td_risfdep%d_value(ji,jj,1,1) = tg_gdepw_1d%d_value(1,1,jk+1,1)
                        tg_misfdep%d_value(ji,jj,1,1) = jk+1
                        IF( tg_misfdep%d_value(ji,jj,1,1) > tg_mbathy%d_value(ji,jj,1,1) )THEN
                           tg_misfdep%d_value(ji,jj,1,1) = jpk
                        ENDIF
                     ENDIF

                  ENDIF
               ENDDO
            ENDDO
         ENDDO

         WHERE( tg_misfdep%d_value(:,:,1,1) == jpk )
             tg_misfdep%d_value(:,:,1,1)=0
             td_risfdep%d_value(:,:,1,1)=0.
             tg_mbathy%d_value (:,:,1,1)=0
             td_bathy%d_value  (:,:,1,1)=0.
         END WHERE

         !! lk_mpp not added

         ! remove single point "bay" on bathy coast line beneath an ice shelf'
         DO jk = jpk,1,-1

            zmask(:,:)=0
            WHERE( tg_mbathy%d_value(:,:,1,1) >= jk ) zmask(:,:)=1

            DO jj = 2, jpj-1
               DO ji = 2, jpi-1
                  IF( tg_mbathy%d_value (ji,jj,1,1) == jk .AND. &
                    & tg_misfdep%d_value(ji,jj,1,1) >= 2 )THEN

                     ibtest =   zmask(ji-1,jj  ) &
                        &     + zmask(ji+1,jj  ) &
                        &     + zmask(ji  ,jj-1) &
                        &     + zmask(ji  ,jj+1)
                     IF( ibtest <= 1 )THEN
                        td_bathy%d_value (ji,jj,1,1) = tg_gdepw_1d%d_value(1,1,jk,1)
                        tg_mbathy%d_value(ji,jj,1,1) = jk-1
                        IF( tg_misfdep%d_value(ji,jj,1,1) > tg_mbathy%d_value(ji,jj,1,1) )THEN
                           tg_mbathy%d_value(ji,jj,1,1) = 0
                        ENDIF
                     ENDIF

                  ENDIF
               ENDDO
            ENDDO
         ENDDO

         WHERE( tg_mbathy%d_value(:,:,1,1) == 0 )
             tg_misfdep%d_value(:,:,1,1)=0
             td_risfdep%d_value(:,:,1,1)=0.
             tg_mbathy%d_value (:,:,1,1)=0
             td_bathy%d_value  (:,:,1,1)=0.
         END WHERE

         !! lk_mpp not added

         ! fill hole in ice shelf
         zmisfdep(:,:) = tg_misfdep%d_value(:,:,1,1)
         zrisfdep(:,:) = td_risfdep%d_value(:,:,1,1)
         WHERE( zmisfdep(:,:) <= 1 ) zmisfdep(:,:)=jpk

         DO jj = 2, jpj-1
            DO ji = 2, jpi-1

               ibtestim1 = zmisfdep(ji-1,jj  )
               ibtestip1 = zmisfdep(ji+1,jj  )
               
               ibtestjm1 = zmisfdep(ji  ,jj-1)
               ibtestjp1 = zmisfdep(ji  ,jj+1)

               IF( zmisfdep(ji,jj) >= tg_mbathy%d_value(ji-1,jj  ,1,1) ) ibtestim1 = jpk
               IF( zmisfdep(ji,jj) >= tg_mbathy%d_value(ji+1,jj  ,1,1) ) ibtestip1 = jpk
               IF( zmisfdep(ji,jj) >= tg_mbathy%d_value(ji  ,jj-1,1,1) ) ibtestjm1 = jpk
               IF( zmisfdep(ji,jj) >= tg_mbathy%d_value(ji  ,jj+1,1,1) ) ibtestjp1 = jpk

               ibtest=MIN(ibtestim1, ibtestip1, ibtestjm1, ibtestjp1)
               IF( ibtest == jpk .AND. &
                 & tg_misfdep%d_value(ji,jj,1,1) >= 2 )THEN
                  tg_mbathy%d_value (ji,jj,1,1) = 0
                  td_bathy%d_value  (ji,jj,1,1) = 0.0_dp
                  tg_misfdep%d_value(ji,jj,1,1) = 0
                  td_risfdep%d_value(ji,jj,1,1) = 0.0_dp
               END IF

               IF( zmisfdep(ji,jj) < ibtest .AND. &
                 & tg_misfdep%d_value(ji,jj,1,1) >= 2 )THEN
                  tg_misfdep%d_value(ji,jj,1,1) = ibtest
                  td_risfdep%d_value(ji,jj,1,1) = tg_gdepw_1d%d_value(1,1,ibtest,1)
               ENDIF

            ENDDO
         ENDDO         

         !! lk_mpp not added

         !! fill hole in bathymetry
         zmbathy(:,:) = tg_mbathy%d_value(:,:,1,1)
         DO jj = 2, jpj-1
            DO ji = 2, jpi-1

               ibtestim1 = zmbathy(ji-1,jj  )
               ibtestip1 = zmbathy(ji+1,jj  )

               ibtestjm1 = zmbathy(ji  ,jj-1)
               ibtestjp1 = zmbathy(ji  ,jj+1)

               IF( zmbathy(ji,jj) < tg_misfdep%d_value(ji-1,jj  ,1,1) ) ibtestim1 = 0
               IF( zmbathy(ji,jj) < tg_misfdep%d_value(ji+1,jj  ,1,1) ) ibtestip1 = 0
               IF( zmbathy(ji,jj) < tg_misfdep%d_value(ji  ,jj-1,1,1) ) ibtestjm1 = 0
               IF( zmbathy(ji,jj) < tg_misfdep%d_value(ji  ,jj+1,1,1) ) ibtestjp1 = 0

               ibtest=MAX(ibtestim1, ibtestip1, ibtestjm1, ibtestjp1)
               IF( ibtest == 0 .AND. tg_misfdep%d_value(ji,jj,1,1) >= 2) THEN
                  tg_mbathy%d_value (ji,jj,1,1) = 0
                  td_bathy%d_value  (ji,jj,1,1) = 0.0_dp
                  tg_misfdep%d_value(ji,jj,1,1) = 0
                  td_risfdep%d_value(ji,jj,1,1) = 0.0_dp
               END IF

               IF( ibtest < zmbathy(ji,jj) .AND. &
               &   tg_misfdep%d_value(ji,jj,1,1) >= 2) THEN
                  tg_mbathy%d_value(ji,jj,1,1) = ibtest
                  td_bathy%d_value (ji,jj,1,1) = tg_gdepw_1d%d_value(1,1,ibtest+1,1) 
               ENDIF

            ENDDO
         ENDDO

         !! lk_mpp not added

         ! if not compatible after all check (ie U point water column less than 2 cells), mask U
         DO jj = 1, jpj-1
            DO ji = 1, jpi-1
               IF( tg_mbathy%d_value(ji  ,jj,1,1) == tg_misfdep%d_value(ji+1,jj,1,1) .AND. &
                 & tg_mbathy%d_value(ji  ,jj,1,1) >= 1   .AND. &
                 & tg_mbathy%d_value(ji+1,jj,1,1) >= 1   )THEN

                  tg_mbathy%d_value(ji,jj,1,1)  = tg_mbathy%d_value(ji,jj,1,1) - 1 

                  ik=tg_mbathy%d_value(ji,jj,1,1)
                  td_bathy%d_value (ji,jj,1,1)  = tg_gdepw_1d%d_value(1,1,ik+1,1)
               ENDIF
            ENDDO
         ENDDO

         !! lk_mpp not added

         ! if not compatible after all check (ie U point water column less than 2 cells), mask U
         DO jj = 1, jpj-1
            DO ji = 1, jpi-1
               IF( tg_misfdep%d_value(ji  ,jj,1,1) == tg_mbathy%d_value(ji+1,jj,1,1) .AND. &
                 & tg_mbathy%d_value (ji  ,jj,1,1) >= 1 .AND.&
                 & tg_mbathy%d_value (ji+1,jj,1,1) >= 1 )THEN

                  tg_mbathy%d_value(ji+1,jj,1,1)  = tg_mbathy%d_value(ji+1,jj,1,1) - 1

                  ik=tg_mbathy%d_value(ji+1,jj,1,1)
                  td_bathy%d_value(ji+1,jj,1,1)   = tg_gdepw_1d%d_value(1,1,ik+1,1)
               ENDIF
            ENDDO
         ENDDO         

         !! lk_mpp not added

         ! if not compatible after all check (ie V point water column less than 2 cells), mask V
         DO jj = 1, jpj-1
            DO ji = 1, jpi
               IF( tg_mbathy%d_value(ji,jj  ,1,1) == tg_misfdep%d_value(ji,jj+1,1,1) .AND. &
                 & tg_mbathy%d_value(ji,jj  ,1,1) >= 1 .AND. &
                 & tg_mbathy%d_value(ji,jj+1,1,1) >= 1 )THEN

                  tg_mbathy%d_value(ji,jj,1,1) = tg_mbathy%d_value(ji,jj,1,1) - 1

                  ik=tg_mbathy%d_value(ji,jj,1,1)
                  td_bathy%d_value (ji,jj,1,1) = tg_gdepw_1d%d_value(1,1,ik+1,1)

               ENDIF
            ENDDO
         ENDDO

         !! lk_mpp not added

         ! if not compatible after all check (ie V point water column less than 2 cells), mask V
         DO jj = 1, jpj-1
            DO ji = 1, jpi
               IF( tg_misfdep%d_value(ji,jj  ,1,1) == tg_mbathy%d_value(ji,jj+1,1,1) .AND.&
                 & tg_mbathy%d_value (ji,jj  ,1,1) >= 1 .AND.&
                 & tg_mbathy%d_value (ji,jj+1,1,1) >= 1 )THEN

                  tg_mbathy%d_value(ji,jj+1,1,1) = tg_mbathy%d_value(ji,jj+1,1,1) - 1

                  ik=tg_mbathy%d_value(ji,jj+1,1,1)
                  td_bathy%d_value (ji,jj+1,1,1) = tg_gdepw_1d%d_value(1,1,ik+1,1)
               ENDIF
            ENDDO
         ENDDO

         !! lk_mpp not added

         ! if not compatible after all check, mask T
         DO jj = 1, jpj
            DO ji = 1, jpi
               IF( tg_mbathy%d_value(ji,jj,1,1) <= tg_misfdep%d_value(ji,jj,1,1) )THEN
                  tg_misfdep%d_value(ji,jj,1,1) = 0
                  td_risfdep%d_value(ji,jj,1,1) = 0._dp
                  tg_mbathy%d_value (ji,jj,1,1) = 0
                  td_bathy%d_value  (ji,jj,1,1) = 0._dp
               ENDIF
            ENDDO
         ENDDO
 
         WHERE( tg_mbathy%d_value(:,:,1,1) == 1 )
            tg_mbathy%d_value (:,:,1,1) = 0
            td_bathy%d_value  (:,:,1,1) = 0.0_dp
            tg_misfdep%d_value(:,:,1,1) = 0
            td_risfdep%d_value(:,:,1,1) = 0.0_dp
         END WHERE
      ENDDO
      ! end check compatibility ice shelf/bathy

      ! remove very shallow ice shelf (less than ~ 10m if 75L)
      WHERE( td_risfdep%d_value(:,:,1,1) <= 10._dp )
         tg_misfdep%d_value(:,:,1,1) = 1
         td_risfdep%d_value(:,:,1,1) = 0.0_dp;
      END WHERE

      DEALLOCATE(zmask   )
      DEALLOCATE(zrisfdep)
      DEALLOCATE(zmisfdep)

      ! compute scale factor and depth at T- and W- points
      DO jj = 1, jpj
         DO ji = 1, jpi
            ik = tg_mbathy%d_value(ji,jj,1,1)
            IF( ik > 0 ) THEN               ! ocean point only      
               ! max ocean level case
               IF( ik == jpk-1 ) THEN

                  zdepwp = td_bathy%d_value(ji,jj,1,1)
                  ze3tp  = td_bathy%d_value(ji,jj,1,1) - tg_gdepw_1d%d_value(1,1,ik,1)
                  ze3wp  = 0.5_dp * tg_e3w_1d%d_value(1,1,ik,1) &
                     &            * ( 1._dp + ( ze3tp/tg_e3t_1d%d_value(1,1,ik,1) ) )
                  tg_e3t_0%d_value  (ji,jj,ik  ,1) = ze3tp
                  tg_e3t_0%d_value  (ji,jj,ik+1,1) = ze3tp
                  tg_e3w_0%d_value  (ji,jj,ik  ,1) = ze3wp
                  tg_e3w_0%d_value  (ji,jj,ik+1,1) = ze3wp

                  tg_gdepw_0%d_value(ji,jj,ik+1,1) = zdepwp
                  tg_gdept_0%d_value(ji,jj,ik  ,1) = tg_gdept_1d%d_value(1 ,1 ,ik-1,1) + ze3wp
                  tg_gdept_0%d_value(ji,jj,ik+1,1) = tg_gdept_0%d_value (ji,jj,ik  ,1) + ze3tp
                  !
               ELSE    ! standard case

                  IF( td_bathy%d_value(ji,jj,1,1) <= tg_gdepw_1d%d_value(1,1,ik+1,1) ) THEN
                     tg_gdepw_0%d_value(ji,jj,ik+1,1) = td_bathy%d_value(ji,jj,1,1)
                  ELSE
                     tg_gdepw_0%d_value(ji,jj,ik+1,1) = tg_gdepw_1d%d_value(1,1,ik+1,1)
                  ENDIF      
                  !
                  !gm Bug?  check the gdepw_1d
                  ! ... on ik
                  tg_gdept_0%d_value(ji,jj,ik,1) = tg_gdepw_1d%d_value(1,1,ik,1) + &
                     &     ( tg_gdepw_0%d_value(ji,jj,ik+1,1) - tg_gdepw_1d%d_value(1,1,ik,1) ) &
                     & * ( ( tg_gdept_1d%d_value(1,1, ik  ,1) - tg_gdepw_1d%d_value(1,1,ik,1) ) &
                     &   / ( tg_gdepw_1d%d_value(1,1, ik+1,1) - tg_gdepw_1d%d_value(1,1,ik,1) ) )

                  tg_e3t_0%d_value(ji,jj,ik  ,1) = tg_gdepw_0%d_value(ji,jj,ik+1,1) - tg_gdepw_1d%d_value(1,1,ik  ,1)
                  tg_e3w_0%d_value(ji,jj,ik  ,1) = tg_gdept_0%d_value(ji,jj,ik  ,1) - tg_gdept_1d%d_value(1,1,ik-1,1)
                  ! ... on ik+1
                  tg_e3w_0%d_value(ji,jj,ik+1,1) = tg_e3t_0%d_value(ji,jj,ik,1)
                  tg_e3t_0%d_value(ji,jj,ik+1,1) = tg_e3t_0%d_value(ji,jj,ik,1)
                  !
               ENDIF
            ENDIF
         END DO
      END DO
      !
      it = 0
      DO jj = 1, jpj
         DO ji = 1, jpi
            ik = tg_mbathy%d_value(ji,jj,1,1)
            IF( ik > 0 ) THEN               ! ocean point only
               tg_e3tp%d_value(ji,jj,1,1) = tg_e3t_0%d_value(ji,jj,ik,1)
               tg_e3wp%d_value(ji,jj,1,1) = tg_e3w_0%d_value(ji,jj,ik,1)
               ! test
               zdiff= tg_gdepw_0%d_value(ji,jj,ik+1,1) - tg_gdept_0%d_value(ji,jj,ik,1)
               IF( zdiff <= 0._dp ) THEN 
                  it = it + 1
                  CALL logger_info(' it    = '//TRIM(fct_str(it))//&
                     &             ' ik    = '//TRIM(fct_str(ik))//&
                     &             ' (i,j) = '//trim(fct_str(ji))//' '//TRIM(fct_str(jj)))
                  CALL logger_info(' bathy = '//TRIM(fct_str(td_bathy%d_value(ji,jj,1,1))))
                  CALL logger_info(' gdept_0 = '//TRIM(fct_str(tg_gdept_0%d_value(ji,jj,ik,1)))//&
                     &             ' gdepw_0 = '//TRIM(fct_str(tg_gdepw_0%d_value(ji,jj,ik+1,1)))//&
                     &             ' zdiff   = '//TRIM(fct_str(zdiff)))
                  CALL logger_info(' e3tp    = '//TRIM(fct_str(tg_e3t_0%d_value(ji,jj,ik,1)))//&
                     &             ' e3wp    = '//TRIM(fct_str(tg_e3w_0%d_value(ji,jj,ik,1))))
               ENDIF
            ENDIF
         END DO
      END DO
      !
      ! (ISF) Definition of e3t, u, v, w for ISF case
      DO jj = 1, jpj 
         DO ji = 1, jpi 
            ik = tg_misfdep%d_value(ji,jj,1,1) 
            IF( ik > 1 ) THEN               ! ice shelf point only 

               IF( td_risfdep%d_value(ji,jj,1,1) < tg_gdepw_1d%d_value(1,1,ik,1) )THEN
                   td_risfdep%d_value(ji,jj,1,1) = tg_gdepw_1d%d_value(1,1,ik,1) 
               ENDIF
               tg_gdepw_0%d_value(ji,jj,ik,1) = td_risfdep%d_value(ji,jj,1,1) 
!gm Bug?  check the gdepw_0 
            !       ... on ik 
               tg_gdept_0%d_value(ji,jj,ik,1) = tg_gdepw_1d%d_value(1,1,ik+1,1) &
                  &        - ( tg_gdepw_1d%d_value(1,1,ik+1,1) - tg_gdepw_0%d_value(ji,jj,ik,1) )   & 
                  &        * ( tg_gdepw_1d%d_value(1,1,ik+1,1) - tg_gdept_1d%d_value(1,1, ik,1) )   & 
                  &        / ( tg_gdepw_1d%d_value(1,1,ik+1,1) - tg_gdepw_1d%d_value(1,1, ik,1) ) 

               tg_e3t_0%d_value(ji,jj,ik  ,1) = tg_gdepw_1d%d_value(1,1,ik+1,1) - tg_gdepw_0%d_value(ji,jj,ik,1) 
               tg_e3w_0%d_value(ji,jj,ik+1,1) = tg_gdept_1d%d_value(1,1,ik+1,1) - tg_gdept_0%d_value(ji,jj,ik,1)

               IF( ik + 1 == tg_mbathy%d_value(ji,jj,1,1) )THEN    ! ice shelf point only (2 cell water column) 
                  tg_e3w_0%d_value(ji,jj,ik+1,1) = tg_gdept_0%d_value(ji,jj,ik+1,1) - tg_gdept_0%d_value(ji,jj,ik,1) 
               ENDIF 
            !       ... on ik / ik-1 
               tg_e3w_0%d_value  (ji,jj,ik  ,1) = tg_e3t_0%d_value  (ji,jj,ik,1) 
               tg_e3t_0%d_value  (ji,jj,ik-1,1) = tg_gdepw_0%d_value(ji,jj,ik,1) - tg_gdepw_1d%d_value(1,1,ik-1,1)
! The next line isn't required and doesn't affect results - included for consistency with bathymetry code 
               tg_gdept_0%d_value(ji,jj,ik-1,1) = tg_gdept_1d%d_value(1,1,ik-1,1)

            ENDIF 
         END DO 
      END DO

      it = 0 
      DO jj = 1, jpj 
         DO ji = 1, jpi 
            ik = tg_misfdep%d_value(ji,jj,1,1) 
            IF( ik > 1 ) THEN               ! ice shelf point only 
               tg_e3tp%d_value(ji,jj,1,1) = tg_e3t_0%d_value(ji,jj,ik  ,1) 
               tg_e3wp%d_value(ji,jj,1,1) = tg_e3w_0%d_value(ji,jj,ik+1,1) 
            ! test 
               zdiff= tg_gdept_0%d_value(ji,jj,ik,1) - tg_gdepw_0%d_value(ji,jj,ik,1) 
               IF( zdiff <= 0. ) THEN  
                  it = it + 1 
                  CALL logger_info(' it    = '//TRIM(fct_str(it))//&
                  &                ' ik    = '//TRIM(fct_str(ik))//&
                  &                ' (i,j) = '//trim(fct_str(ji))//' '//TRIM(fct_str(jj)))

                  CALL logger_info(' risfdep = '//TRIM(fct_str(td_risfdep%d_value(ji,jj,1,1)))) 
                  CALL logger_info(' gdept = '//TRIM(fct_str(tg_gdept_0%d_value(ji,jj,ik,1)))//&
                  &                ' gdepw = '//TRIM(fct_str(tg_gdepw_0%d_value(ji,jj,ik+1,1)))//&
                  &                ' zdiff = '//TRIM(fct_str(zdiff)))
                  CALL logger_info(' e3tp  = '//TRIM(fct_str(tg_e3t_0%d_value(ji,jj,ik  ,1)))//&
                  &                ' e3wp  = '//TRIM(fct_str(tg_e3w_0%d_value(ji,jj,ik+1,1))))
               ENDIF 
            ENDIF 
         END DO 
      END DO

   END SUBROUTINE grid_zgr__isf_fill
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!   SUBROUTINE grid_zgr__isf_fill_e3x( jpi,jpj, &
!         &                            td_risfdep) 
!   !-------------------------------------------------------------------
!   !> @brief This subroutine define e3t, u, v, w for ISF case 
!   !>
!   !> @details
!   !>
!   !> @author J.Paul
!   !> @date September, 2015 - rewrite from zgr_zps
!   !>
!   !> @param[in] jpi
!   !> @param[in] jpj
!   !> @param[in] td_risfdep
!   !-------------------------------------------------------------------
!
!      IMPLICIT NONE
!
!      ! Argument      
!      INTEGER(i4), INTENT(IN   ) :: jpi
!      INTEGER(i4), INTENT(IN   ) :: jpj
!      TYPE(TVAR) , INTENT(INOUT) :: td_risfdep
!
!      ! local variable
!      REAL(dp)    :: zdiff            ! temporary scalar
!
!      ! loop indices
!      INTEGER(i4) :: ji
!      INTEGER(i4) :: jj
!      INTEGER(i4) :: it
!      INTEGER(i4) :: ik
!      !----------------------------------------------------------------
!
!      ! (ISF) Definition of e3t, u, v, w for ISF case
!      DO jj = 1, jpj 
!         DO ji = 1, jpi 
!            ik = tg_misfdep%d_value(ji,jj,1,1) 
!
!            IF( ik > 1 ) THEN
!               ! ice shelf point only
!               IF( td_risfdep%d_value(ji,jj,1,1) < tg_gdepw_1d%d_value(1,1,ik,1) )THEN
!                   td_risfdep%d_value(ji,jj,1,1) = tg_gdepw_1d%d_value(1,1,ik,1) 
!               ENDIF
!               tg_gdepw_0%d_value(ji,jj,ik,1) = td_risfdep%d_value(ji,jj,1,1) 
!            !gm Bug?  check the gdepw_0 
!            !       ... on ik 
!               tg_gdept_0%d_value(ji,jj,ik  ,1) = tg_gdepw_1d%d_value(1,1,ik+1,1) - & 
!                  &                               (tg_gdepw_1d%d_value(1,1,ik+1,1) - tg_gdepw_0%d_value (ji,jj,ik,1)) * &
!                  &                               (tg_gdepw_1d%d_value(1,1,ik+1,1) - tg_gdept_1d%d_value( 1, 1,ik,1)) / & 
!                  &                               (tg_gdepw_1d%d_value(1,1,ik+1,1) - tg_gdepw_1d%d_value( 1, 1,ik,1)) 
!
!               tg_e3t_0%d_value  (ji,jj,ik  ,1) = tg_gdepw_1d%d_value( 1, 1,ik+1,1) - &
!                  &                               tg_gdepw_0%d_value (ji,jj,ik  ,1) 
!
!               tg_e3w_0%d_value  (ji,jj,ik+1,1) = tg_gdept_1d%d_value( 1, 1,ik+1,1) - &
!                  &                               tg_gdept_0%d_value (ji,jj,ik  ,1)
!
!               IF( ik + 1 == tg_mbathy%d_value(ji,jj,1,1) ) THEN
!
!                  ! ice shelf point only (2 cell water column)
!                  tg_e3w_0%d_value(ji,jj,ik+1,1) = tg_gdept_0%d_value(ji,jj,ik+1,1) - &
!                     &                             tg_gdept_0%d_value(ji,jj,ik  ,1) 
!
!               ENDIF 
!            !       ... on ik / ik-1 
!               tg_e3w_0%d_value  (ji,jj,ik  ,1) = 2._dp * (tg_gdept_0%d_value(ji,jj,ik,1) - &
!                  &                                        tg_gdepw_0%d_value(ji,jj,ik,1)) 
!
!               tg_e3t_0%d_value  (ji,jj,ik-1,1) = tg_gdepw_0%d_value (ji,jj,ik  ,1) - &
!                  &                               tg_gdepw_1d%d_value( 1, 1,ik-1,1)
!
!               ! The next line isn't required and doesn't affect results - included for consistency with bathymetry code 
!               tg_gdept_0%d_value(ji,jj,ik-1,1) = tg_gdept_1d%d_value(1,1,ik-1,1)
!            ENDIF 
!
!         END DO 
!      END DO 
!      
!      it = 0 
!      DO jj = 1, jpj 
!         DO ji = 1, jpi 
!            ik = tg_misfdep%d_value(ji,jj,1,1) 
!            IF( ik > 1 ) THEN               ! ice shelf point only 
!               tg_e3tp%d_value(ji,jj,1,1) = tg_e3t_0%d_value(ji,jj,ik  ,1) 
!               tg_e3wp%d_value(ji,jj,1,1) = tg_e3w_0%d_value(ji,jj,ik+1,1) 
!            ! test 
!               zdiff= tg_gdept_0%d_value(ji,jj,ik,1) - &
!                  &   tg_gdepw_0%d_value(ji,jj,ik,1) 
!
!               IF( zdiff <= 0. ) THEN  
!                  it = it + 1 
!                  CALL logger_info(' it      = '//TRIM(fct_str(it))//&
!                     &             ' ik      = '//TRIM(fct_str(ik))//&
!                     &             ' (i,j)   =('//TRIM(fct_str(ji))//','//TRIM(fct_str(jj))//')')  
!                  CALL logger_info(' risfdep = '//TRIM(fct_str(td_risfdep%d_value(ji,jj,1,1))) )
!                  CALL logger_info(' gdept = '//TRIM(fct_str(tg_gdept_0%d_value(ji,jj,ik  ,1)))//&
!                     &             ' gdepw = '//TRIM(fct_str(tg_gdepw_0%d_value(ji,jj,ik+1,1)))//&
!                     &             ' zdiff = '//TRIM(fct_str(zdiff)) ) 
!                  CALL logger_info(' e3tp  = '//TRIM(fct_str( tg_e3tp%d_value(ji,jj,1,1)))//&
!                     &             ' e3wp  = '//TRIM(fct_str( tg_e3wp%d_value(ji,jj,1,1))) ) 
!               ENDIF 
!            ENDIF 
!         END DO 
!      END DO 
!      ! END (ISF)
!
!   END SUBROUTINE grid_zgr__isf_fill_e3x
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr__isf_fill_e3uw(jpi, jpj) 
   !-------------------------------------------------------------------
   !> @brief This subroutine define e3uw 
   !>    (adapted for 2 cells in the water column) for ISF case 
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date September, 2015 - rewrite from zgr_zps
   !>
   !> @param[in] jpi
   !> @param[in] jpj
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      INTEGER(i4)   , INTENT(IN   ) :: jpi
      INTEGER(i4)   , INTENT(IN   ) :: jpj

      ! local variable
      INTEGER(i4) :: ikb, ikt

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      DO jj = 2, jpj - 1 
         DO ji = 2, jpi - 1 

            ikb = MAX(tg_mbathy%d_value (ji,jj,1,1), tg_mbathy%d_value (ji+1,jj,1,1))
            ikt = MAX(tg_misfdep%d_value(ji,jj,1,1), tg_misfdep%d_value(ji+1,jj,1,1))
            IF( ikb == ikt+1 )THEN
               tg_e3uw_0%d_value(ji,jj,ikb,1) = MIN( tg_gdept_0%d_value(ji,jj,ikb  ,1), tg_gdept_0%d_value(ji+1,jj  ,ikb  ,1) ) - &
               &                                MAX( tg_gdept_0%d_value(ji,jj,ikb-1,1), tg_gdept_0%d_value(ji+1,jj  ,ikb-1,1) )
            ENDIF
            
            ikb = MAX( tg_mbathy%d_value (ji,jj,1,1), tg_mbathy%d_value (ji,jj+1,1,1))
            ikt = MAX( tg_misfdep%d_value(ji,jj,1,1), tg_misfdep%d_value(ji,jj+1,1,1))
            IF( ikb == ikt+1 )THEN
               tg_e3vw_0%d_value(ji,jj,ikb,1) = MIN( tg_gdept_0%d_value(ji,jj,ikb  ,1), tg_gdept_0%d_value(ji  ,jj+1,ikb  ,1) ) - &
               &                                MAX( tg_gdept_0%d_value(ji,jj,ikb-1,1), tg_gdept_0%d_value(ji  ,jj+1,ikb-1,1) )
            ENDIF
         END DO
      END DO

   END SUBROUTINE grid_zgr__isf_fill_e3uw
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!   SUBROUTINE grid_zgr__isf_fill_gdep3w_0( jpi,jpj,jpk,td_risfdep )
!   !-------------------------------------------------------------------
!   !> @brief This subroutine compute gdep3w_0 (vertical sum of e3w) 
!   !>
!   !> @details
!   !>
!   !> @author J.Paul
!   !> @date September, 2015 - rewrite from zgr_zps
!   !>
!   !> @param[in] jpi
!   !> @param[in] jpj
!   !> @param[in] jpk
!   !> @param[in] td_risfdep
!   !-------------------------------------------------------------------
!
!      IMPLICIT NONE
!
!      ! Argument      
!      INTEGER(i4), INTENT(IN   ) :: jpi
!      INTEGER(i4), INTENT(IN   ) :: jpj
!      INTEGER(i4), INTENT(IN   ) :: jpk
!      TYPE(TVAR) , INTENT(INOUT) :: td_risfdep
!
!      ! local variable
!      INTEGER(i4) :: ik
!
!      ! loop indices
!      INTEGER(i4) :: ji
!      INTEGER(i4) :: jj
!      INTEGER(i4) :: jk
!      !----------------------------------------------------------------
!
!      WHERE( tg_misfdep%d_value(:,:,:,:) == 0 ) tg_misfdep%d_value(:,:,:,:) = 1
!      
!      DO jj = 1,jpj
!         DO ji = 1,jpi
!
!            tg_gdep3w_0%d_value(ji,jj,1,1) = 0.5_dp * tg_e3w_0%d_value(ji,jj,1,1)
!            DO jk = 2, INT(tg_misfdep%d_value(ji,jj,1,1),i4)
!               tg_gdep3w_0%d_value(ji,jj,jk,1) = tg_gdep3w_0%d_value(ji,jj,jk-1,1) + &
!                  &                              tg_e3w_0%d_value   (ji,jj,jk  ,1) 
!            END DO
!
!            ik=tg_misfdep%d_value(ji,jj,1,1)
!            IF( ik >= 2 )THEN
!               tg_gdep3w_0%d_value(ji,jj,ik,1) = td_risfdep%d_value(ji,jj, 1,1) + &
!                  &                       0.5_dp * tg_e3w_0%d_value(ji,jj,ik,1)
!            ENDIF
!
!            DO jk = ik + 1, jpk
!               tg_gdep3w_0%d_value(ji,jj,jk,1) = tg_gdep3w_0%d_value(ji,jj,jk-1,1) + &
!                  &                              tg_e3w_0%d_value   (ji,jj,jk  ,1) 
!            END DO
!
!         END DO
!      END DO
!
!   END SUBROUTINE grid_zgr__isf_fill_gdep3w_0
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr_sco_init(jpi, jpj) 
   !-------------------------------------------------------------------
   !> @brief This subroutine initialise global variable needed to compute vertical
   !>        mesh
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

      ! local variable
      REAL(dp), DIMENSION(jpi,jpj) :: dl_tmp

      ! loop indices
      !----------------------------------------------------------------

      dl_tmp(:,:)=dp_fill

      tg_rx1     =var_init('rx1      ',dl_tmp(:,:), dd_fill=dp_fill, id_type=NF90_DOUBLE)

   END SUBROUTINE grid_zgr_sco_init
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr_sco_clean() 
   !-------------------------------------------------------------------
   !> @brief This subroutine clean structure
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial version
   !>
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      ! local variable
      ! loop indices
      !----------------------------------------------------------------

      CALL var_clean(tg_rx1      )
      
   END SUBROUTINE grid_zgr_sco_clean
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr__sco_fill(td_nam, jpi, jpj, jpk, td_bathy) 
   !-------------------------------------------------------------------
   !> @brief This subroutine define the s-coordinate system
   !>
   !> @details
   !>
   !> ** Method  :   s-coordinate
   !>         The depth of model levels is defined as the product of an
   !>      analytical function by the local bathymetry, while the vertical
   !>      scale factors are defined as the product of the first derivative
   !>      of the analytical function by the bathymetry.
   !>      (this solution save memory as depth and scale factors are not
   !>      3d fields)
   !>          - Read bathymetry (in meters) at t-point and compute the
   !>         bathymetry at u-, v-, and f-points.
   !>            - hbatu = mi( hbatt )
   !>            - hbatv = mj( hbatt )
   !>            - hbatf = mi( mj( hbatt ) )
   !>          - Compute z_gsigt, z_gsigw, z_esigt, z_esigw from an analytical
   !>         function and its derivative given as function.
   !>            - z_gsigt(k) = fssig (k    )
   !>            - z_gsigw(k) = fssig (k-0.5)
   !>            - z_esigt(k) = fsdsig(k    )
   !>            - z_esigw(k) = fsdsig(k-0.5)
   !>      Three options for stretching are give, and they can be modified
   !>      following the users requirements. Nevertheless, the output as
   !>      well as the way to compute the model levels and scale factors
   !>      must be respected in order to insure second order accuracy
   !>      schemes.
   !>
   !>      The three methods for stretching available are:
   !> 
   !>           s_sh94 (Song and Haidvogel 1994)
   !>                a sinh/tanh function that allows sigma and stretched sigma
   !>
   !>           s_sf12 (Siddorn and Furner 2012?)
   !>                allows the maintenance of fixed surface and or
   !>                bottom cell resolutions (cf. geopotential coordinates) 
   !>                within an analytically derived stretched S-coordinate framework.
   !> 
   !>          s_tanh  (Madec et al 1996)
   !>                a cosh/tanh function that gives stretched coordinates
   !>
   !> @author J.Paul
   !> @date September, 2015 - rewrite from zgr_sco
   !> @date October, 2016
   !> - add wetting and drying boolean
   !>
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMZ), INTENT(IN   ) :: td_nam
      INTEGER(i4), INTENT(IN   ) :: jpi
      INTEGER(i4), INTENT(IN   ) :: jpj
      INTEGER(i4), INTENT(IN   ) :: jpk
      TYPE(TVAR) , INTENT(INOUT) :: td_bathy

      ! local variable
      INTEGER  ::   iip1, ijp1, iim1, ijm1   ! temporary integers

      REAL(dp) :: zrmax, zrfact
      REAL(dp) :: ztaper

      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: dl_hifv  !< interface depth between stretching at v--f
      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: dl_hiff
      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: dl_hift  !< and quasi-uniform spacing t--u points (m)
      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: dl_hifu
      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: dl_scosrf !< ocean surface topography
      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: dl_scobot !< ocean bottom topography

!      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: dl_hbatt
!      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: dl_hbatu
!      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: dl_hbatv
!      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: dl_hbatf

      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: zenv
      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: zri
      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: zrj
      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: zhbat
      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: ztmpi1
      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: ztmpi2
      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: ztmpj1
      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: ztmpj2

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      CALL logger_info('GRID ZGR SCO: s-coordinate or hybrid z-s-coordinate')
      CALL logger_info('~~~~~~~~~~~')
      CALL logger_info('   Namelist namzgr_sco')
      CALL logger_info('     stretching coeffs ')
      CALL logger_info('        maximum depth of s-bottom surface (>0)       dn_sbot_max   = '//TRIM(fct_str(td_nam%d_sbot_max)))
      CALL logger_info('        minimum depth of s-bottom surface (>0)       dn_sbot_min   = '//TRIM(fct_str(td_nam%d_sbot_min)))
      CALL logger_info('        Critical depth                               dn_hc         = '//TRIM(fct_str(td_nam%d_hc)))
      CALL logger_info('        maximum cut-off r-value allowed              dn_rmax       = '//TRIM(fct_str(td_nam%d_rmax)))
      CALL logger_info('     Song and Haidvogel 1994 stretching              ln_s_sh94     = '//TRIM(fct_str(td_nam%l_s_sh94)))
      CALL logger_info('        Song and Haidvogel 1994 stretching coefficients')
      CALL logger_info('        surface control parameter (0<=rn_theta<=20)  dn_theta      = '//TRIM(fct_str(td_nam%d_theta)))
      CALL logger_info('        bottom  control parameter (0<=rn_thetb<= 1)  dn_thetb      = '//TRIM(fct_str(td_nam%d_thetb)))
      CALL logger_info('        stretching parameter (song and haidvogel)    dn_bb         = '//TRIM(fct_str(td_nam%d_bb)))
      CALL logger_info('     Siddorn and Furner 2012 stretching              ln_s_sf12     = '//TRIM(fct_str(td_nam%l_s_sf12)))
      CALL logger_info('        switching to sigma (T) or Z (F) at H<Hc      ln_sigcrit    = '//TRIM(fct_str(td_nam%l_sigcrit)))
      CALL logger_info('        Siddorn and Furner 2012 stretching coefficients')
      CALL logger_info('        stretchin parameter ( >1 surface; <1 bottom) dn_alpha      = '//TRIM(fct_str(td_nam%d_alpha)))
      CALL logger_info('        e-fold length scale for transition region    dn_efold      = '//TRIM(fct_str(td_nam%d_efold)))
      CALL logger_info('        Surface cell depth (Zs) (m)                  dn_zs         = '//TRIM(fct_str(td_nam%d_zs)))
      CALL logger_info('        Bathymetry multiplier for Zb                 dn_zb_a       = '//TRIM(fct_str(td_nam%d_zb_a)))
      CALL logger_info('        Offset for Zb                                dn_zb_b       = '//TRIM(fct_str(td_nam%d_zb_b)))
      CALL logger_info('        Bottom cell (Zb) (m) = H*rn_zb_a + rn_zb_b')

      ALLOCATE(dl_hifv(jpi,jpj))
      ALLOCATE(dl_hiff(jpi,jpj))
      ALLOCATE(dl_hift(jpi,jpj))
      ALLOCATE(dl_hifu(jpi,jpj))

      ! set the minimum depth for the s-coordinate
      dl_hift(:,:) = td_nam%d_sbot_min
      dl_hifu(:,:) = td_nam%d_sbot_min
      dl_hifv(:,:) = td_nam%d_sbot_min
      dl_hiff(:,:) = td_nam%d_sbot_min

      ! set maximum ocean depth
      td_bathy%d_value(:,:,1,1) = MIN( td_nam%d_sbot_max, td_bathy%d_value(:,:,1,1) )
      IF( .NOT. td_nam%l_wd )THEN
         DO jj = 1, jpj
            DO ji = 1, jpi
              IF( td_bathy%d_value(ji,jj,1,1) > 0._dp )THEN
                 td_bathy%d_value(ji,jj,1,1) = MAX(td_nam%d_sbot_min, td_bathy%d_value(ji,jj,1,1))
              ENDIF
            END DO
         END DO
      ENDIF

      ! =============================
      ! Define the envelop bathymetry   (hbatt)
      ! =============================
      ! use r-value to create hybrid coordinates
      ALLOCATE(zenv(jpi,jpj))
      zenv(:,:) = td_bathy%d_value(:,:,1,1)

      IF( .NOT. td_nam%l_wd )THEN
      ! set first land point adjacent to a wet cell to sbot_min as this needs to be included in smoothing
         DO jj = 1, jpj
            DO ji = 1, jpi
              IF( td_bathy%d_value(ji,jj,1,1) == 0._dp )THEN
                iip1 = MIN( ji+1, jpi )
                ijp1 = MIN( jj+1, jpj )
                iim1 = MAX( ji-1,  1  )
                ijm1 = MAX( jj-1,  1  )
                IF( ( td_bathy%d_value(iim1,ijm1,1,1) + &
                   &  td_bathy%d_value(ji  ,ijp1,1,1) + &
                   &  td_bathy%d_value(iip1,ijp1,1,1) + &
                   &  td_bathy%d_value(iim1,jj  ,1,1) + &
                   &  td_bathy%d_value(iip1,jj  ,1,1) + &
                   &  td_bathy%d_value(iim1,ijm1,1,1) + &
                   &  td_bathy%d_value(ji  ,ijm1,1,1) + &
                   &  td_bathy%d_value(iip1,ijp1,1,1) ) > 0._dp )THEN
                  zenv(ji,jj) = td_nam%d_sbot_min
                ENDIF
              ENDIF
            END DO
         END DO
      ENDIF

      ! apply lateral boundary condition   CAUTION: keep the value when the lbc field is zero
      ! this is only in mpp case, so here just do nothing
      !! CALL lbc_lnk( zenv(:,:), 'T', td_nam%i_perio, 1._dp, 'no0' ) 

      ! smooth the bathymetry (if required)
      ALLOCATE(dl_scosrf(jpi,jpj))
      ALLOCATE(dl_scobot(jpi,jpj))
      dl_scosrf(:,:) = 0._dp                      ! ocean surface depth (here zero: no under ice-shelf sea)
      dl_scobot(:,:) = td_bathy%d_value(:,:,1,1)  ! ocean bottom  depth

      jl    = 0
      zrmax = 1._dp

      ! set scaling factor used in reducing vertical gradients
      zrfact = ( 1._dp - td_nam%d_rmax ) / ( 1._dp + td_nam%d_rmax )      

      ! initialise temporary envelope depth arrays
      ALLOCATE(ztmpi1(jpi,jpj))
      ALLOCATE(ztmpi2(jpi,jpj))
      ALLOCATE(ztmpj1(jpi,jpj))
      ALLOCATE(ztmpj2(jpi,jpj))

      ztmpi1(:,:) = zenv(:,:)
      ztmpi2(:,:) = zenv(:,:)
      ztmpj1(:,:) = zenv(:,:)
      ztmpj2(:,:) = zenv(:,:)

      ! initialise temporary r-value arrays
      ALLOCATE(zri(jpi,jpj))
      ALLOCATE(zrj(jpi,jpj))
      zri(:,:) = 1._dp
      zrj(:,:) = 1._dp

      !  Iterative loop  !
      ! ================ !
      DO WHILE( jl <= 10000 .AND. ( zrmax - td_nam%d_rmax ) > 1.e-8_dp )

         jl = jl + 1
         zrmax = 0._dp
         ! we set zrmax from previous r-values (zri and zrj) first
         ! if set after current r-value calculation (as previously)
         ! we could exit DO WHILE prematurely before checking r-value
         ! of current zenv
         DO jj = 1, jpj
            DO ji = 1, jpi
               zrmax = MAX( zrmax, ABS(zri(ji,jj)), ABS(zrj(ji,jj)) )
            END DO
         END DO
         zri(:,:) = 0._dp
         zrj(:,:) = 0._dp
         DO jj = 1, jpj
            DO ji = 1, jpi
               iip1 = MIN( ji+1, jpi )      ! force zri = 0 on last line (ji=ncli+1 to jpi)
               ijp1 = MIN( jj+1, jpj )      ! force zrj = 0 on last raw  (jj=nclj+1 to jpj)
               IF( (zenv(ji  ,jj) > 0._dp) .AND. &
                 & (zenv(iip1,jj) > 0._dp) )THEN
                  zri(ji,jj) = ( zenv(iip1,  jj) - zenv(ji,jj) ) / ( zenv(iip1,jj  ) + zenv(ji,jj) )
               END IF
               IF( (zenv(ji,jj  ) > 0._dp) .AND. &
                 & (zenv(ji,ijp1) > 0._dp) )THEN
                  zrj(ji,jj) = ( zenv(ji  ,ijp1) - zenv(ji,jj) ) / ( zenv(ji  ,ijp1) + zenv(ji,jj) )
               END IF
               IF( zri(ji,jj) >  td_nam%d_rmax ) ztmpi1(ji  ,jj  ) = zenv(iip1,jj  ) * zrfact
               IF( zri(ji,jj) < -td_nam%d_rmax ) ztmpi2(iip1,jj  ) = zenv(ji  ,jj  ) * zrfact
               IF( zrj(ji,jj) >  td_nam%d_rmax ) ztmpj1(ji  ,jj  ) = zenv(ji  ,ijp1) * zrfact
               IF( zrj(ji,jj) < -td_nam%d_rmax ) ztmpj2(ji  ,ijp1) = zenv(ji  ,jj  ) * zrfact
            END DO
         END DO
         !! 
         !
         CALL logger_info('zgr_sco :   iter= '//TRIM(fct_str(jl))//&
            &             ' rmax= '//TRIM(fct_str(zrmax)) )
         !
         DO jj = 1, jpj
            DO ji = 1, jpi
               zenv(ji,jj) = MAX( zenv  (ji,jj), &
                                & ztmpi1(ji,jj), &
                                & ztmpi2(ji,jj), &
                                & ztmpj1(ji,jj), &
                                & ztmpj2(ji,jj) )
            END DO
         END DO
         ! apply lateral boundary condition   CAUTION: keep the value when the lbc field is zero
         ! this is only in mpp case, so here just do nothing
         !!CALL lbc_lnk( zenv, 'T', td_nam%i_perio, 1._dp, 'no0' )
      ENDDO
      !     End loop     !
      ! ================ !

      DEALLOCATE(zri)
      DEALLOCATE(zrj)

      DEALLOCATE(ztmpi1)
      DEALLOCATE(ztmpi2)
      DEALLOCATE(ztmpj1)
      DEALLOCATE(ztmpj2)

      DO jj = 1, jpj
         DO ji = 1, jpi
            ! set all points to avoid undefined scale value warnings
            zenv(ji,jj) = MAX( zenv(ji,jj), td_nam%d_sbot_min )
         ENDDO
      ENDDO
      !
      ! Envelope bathymetry saved in hbatt
      !
!      ALLOCATE(dl_hbatt(jpi,jpj))
!      ALLOCATE(dl_hbatu(jpi,jpj))
!      ALLOCATE(dl_hbatv(jpi,jpj))
!      ALLOCATE(dl_hbatf(jpi,jpj))

      tg_hbatt%d_value(:,:,1,1) = zenv(:,:) 
      IF( MINVAL( tg_gphit%d_value(:,:,1,1) ) * &
        & MAXVAL( tg_gphit%d_value(:,:,1,1) ) <= 0._dp ) THEN
         CALL logger_warn( ' s-coordinates are tapered in vicinity of the Equator' )
         DO jj = 1, jpj
            DO ji = 1, jpi
               ztaper = EXP( -(tg_gphit%d_value(ji,jj,1,1)/8._dp)**2._dp )
               tg_hbatt%d_value(ji,jj,1,1) =   td_nam%d_sbot_max           * ztaper           &
               &                             + tg_hbatt%d_value(ji,jj,1,1) * (1._dp - ztaper)
            END DO
         END DO
      ENDIF

      DEALLOCATE(zenv)

      CALL logger_info(' bathy  MAX '//TRIM(fct_str(MAXVAL( td_bathy%d_value(:,:,1,1) )))//&
         &                    ' MIN '//TRIM(fct_str(MINVAL( td_bathy%d_value(:,:,1,1) ))) )
      CALL logger_info(' hbatt  MAX '//TRIM(fct_str(MAXVAL( tg_hbatt%d_value(:,:,1,1) )))//&
         &                    ' MIN '//TRIM(fct_str(MINVAL( tg_hbatt%d_value(:,:,1,1) ))) )
      !
      !   hbatu, hbatv, hbatf fields
      !
      tg_hbatu%d_value(:,:,1,1) = td_nam%d_sbot_min
      tg_hbatv%d_value(:,:,1,1) = td_nam%d_sbot_min
      tg_hbatf%d_value(:,:,1,1) = td_nam%d_sbot_min

      DO jj = 1, jpj-1
        DO ji = 1, jpi-1   ! NO vector opt.
           tg_hbatu%d_value(ji,jj,1,1) = 0.50_dp * ( tg_hbatt%d_value(ji  ,jj  ,1,1) + &
              &                                      tg_hbatt%d_value(ji+1,jj  ,1,1) )
           tg_hbatv%d_value(ji,jj,1,1) = 0.50_dp * ( tg_hbatt%d_value(ji  ,jj  ,1,1) + &
              &                                      tg_hbatt%d_value(ji  ,jj+1,1,1) )
           tg_hbatf%d_value(ji,jj,1,1) = 0.25_dp * ( tg_hbatt%d_value(ji  ,jj  ,1,1) + &
              &                                      tg_hbatt%d_value(ji  ,jj+1,1,1) + &
              &                                      tg_hbatt%d_value(ji+1,jj  ,1,1) + &
              &                                      tg_hbatt%d_value(ji+1,jj+1,1,1) )
        ENDDO
      ENDDO

      IF( td_nam%l_wd ) THEN               !avoid the zero depth on T- (U-,V-,F-) points
        DO jj = 1, jpj
          DO ji = 1, jpi
            IF( ABS(tg_hbatt%d_value(ji,jj,1,1)) < td_nam%d_wdmin1 )THEN
               tg_hbatt%d_value(ji,jj,1,1) = SIGN(1._dp, tg_hbatt%d_value(ji,jj,1,1)) * td_nam%d_wdmin1
            ENDIF
            IF( ABS(tg_hbatu%d_value(ji,jj,1,1)) < td_nam%d_wdmin1 )THEN
               tg_hbatu%d_value(ji,jj,1,1) = SIGN(1._dp, tg_hbatu%d_value(ji,jj,1,1)) * td_nam%d_wdmin1
            ENDIF
            IF( ABS(tg_hbatv%d_value(ji,jj,1,1)) < td_nam%d_wdmin1 )THEN
               tg_hbatv%d_value(ji,jj,1,1) = SIGN(1._dp, tg_hbatv%d_value(ji,jj,1,1)) * td_nam%d_wdmin1
            ENDIF
            IF( ABS(tg_hbatf%d_value(ji,jj,1,1)) < td_nam%d_wdmin1 )THEN
               tg_hbatf%d_value(ji,jj,1,1) = SIGN(1._dp, tg_hbatf%d_value(ji,jj,1,1)) * td_nam%d_wdmin1
            ENDIF
          END DO
        END DO
      ENDIF

      ! Apply lateral boundary condition
      ALLOCATE(zhbat(jpi,jpj))

!!gm  ! CAUTION: retain non zero value in the initial file this should be OK for orca cfg, not for EEL
      zhbat(:,:) = tg_hbatu%d_value(:,:,1,1)
      CALL lbc_lnk( tg_hbatu%d_value(:,:,1,1), 'U', td_nam%i_perio, 1._dp )
      DO jj = 1, jpj
         DO ji = 1, jpi
            IF( tg_hbatu%d_value(ji,jj,1,1) == 0._dp )THEN
               !No worries about the following line when l_wd == .true.
               IF( zhbat(ji,jj) == 0._dp ) tg_hbatu%d_value(ji,jj,1,1) = td_nam%d_sbot_min
               IF( zhbat(ji,jj) /= 0._dp ) tg_hbatu%d_value(ji,jj,1,1) = zhbat(ji,jj)
            ENDIF
         ENDDO
      ENDDO

      zhbat(:,:) = tg_hbatv%d_value(:,:,1,1)
      CALL lbc_lnk( tg_hbatv%d_value(:,:,1,1), 'V', td_nam%i_perio, 1._dp )
      DO jj = 1, jpj
         DO ji = 1, jpi
            IF( tg_hbatv%d_value(ji,jj,1,1) == 0._dp ) THEN
               IF( zhbat(ji,jj) == 0._dp ) tg_hbatv%d_value(ji,jj,1,1) = td_nam%d_sbot_min
               IF( zhbat(ji,jj) /= 0._dp ) tg_hbatv%d_value(ji,jj,1,1) = zhbat(ji,jj)
            ENDIF
         ENDDO
      ENDDO

      zhbat(:,:) = tg_hbatf%d_value(:,:,1,1)
      CALL lbc_lnk( tg_hbatf%d_value(:,:,1,1), 'F', td_nam%i_perio, 1._dp )
      DO jj = 1, jpj
         DO ji = 1, jpi
            IF( tg_hbatf%d_value(ji,jj,1,1) == 0._dp ) THEN
               IF( zhbat(ji,jj) == 0._dp ) tg_hbatf%d_value(ji,jj,1,1) = td_nam%d_sbot_min
               IF( zhbat(ji,jj) /= 0._dp ) tg_hbatf%d_value(ji,jj,1,1) = zhbat(ji,jj)
            ENDIF
         ENDDO
      ENDDO

      DEALLOCATE(zhbat)

!!bug:  key_helsinki a verifer
      IF( .NOT.td_nam%l_wd )THEN
         dl_hift(:,:) = MIN( dl_hift(:,:), tg_hbatt%d_value(:,:,1,1) )
         dl_hifu(:,:) = MIN( dl_hifu(:,:), tg_hbatu%d_value(:,:,1,1) )
         dl_hifv(:,:) = MIN( dl_hifv(:,:), tg_hbatv%d_value(:,:,1,1) )
         dl_hiff(:,:) = MIN( dl_hiff(:,:), tg_hbatf%d_value(:,:,1,1) )
      ENDIF

      CALL logger_info(' MAX val hif   t '//TRIM(fct_str(MAXVAL( dl_hift(:,:) )))//&
         &                           ' f '//TRIM(fct_str(MAXVAL( dl_hiff(:,:) )))//&
         &                           ' u '//TRIM(fct_str(MAXVAL( dl_hifu(:,:) )))//&
         &                           ' v '//TRIM(fct_str(MAXVAL( dl_hifv(:,:) ))) )
      CALL logger_info(' MIN val hif   t '//TRIM(fct_str(MINVAL( dl_hift(:,:) )))//&
         &                           ' f '//TRIM(fct_str(MINVAL( dl_hiff(:,:) )))//&
         &                           ' u '//TRIM(fct_str(MINVAL( dl_hifu(:,:) )))//&
         &                           ' v '//TRIM(fct_str(MINVAL( dl_hifv(:,:) ))) )
      CALL logger_info(' MAX val hbat  t '//TRIM(fct_str(MAXVAL( tg_hbatt%d_value(:,:,1,1) )))//&
         &                           ' f '//TRIM(fct_str(MAXVAL( tg_hbatf%d_value(:,:,1,1) )))//&
         &                           ' u '//TRIM(fct_str(MAXVAL( tg_hbatu%d_value(:,:,1,1) )))//&
         &                           ' v '//TRIM(fct_str(MAXVAL( tg_hbatv%d_value(:,:,1,1) ))) )
      CALL logger_info(' MIN val hbat  t '//TRIM(fct_str(MINVAL( tg_hbatt%d_value(:,:,1,1) )))//&
         &                           ' f '//TRIM(fct_str(MINVAL( tg_hbatf%d_value(:,:,1,1) )))//&
         &                           ' u '//TRIM(fct_str(MINVAL( tg_hbatu%d_value(:,:,1,1) )))//&
         &                           ' v '//TRIM(fct_str(MINVAL( tg_hbatv%d_value(:,:,1,1) ))) )
!! helsinki

      ! =======================
      !   s-ccordinate fields     (gdep., e3.)
      ! =======================
      !
      ! non-dimensional "sigma" for model level depth at w- and t-levels

!========================================================================
! Song and Haidvogel  1994 (ln_s_sh94=T)
! Siddorn and Furner  2012 (ln_sf12=T)
! or  tanh function        (both false)                    
!========================================================================
      IF( td_nam%l_s_sh94 ) THEN 
         CALL grid_zgr__sco_s_sh94( td_nam,jpi,jpj,jpk, &
         &                          dl_scosrf )
      ELSEIF( td_nam%l_s_sf12 ) THEN
         CALL grid_zgr__sco_s_sf12( td_nam,jpi,jpj,jpk, &
         &                          dl_scosrf )
      ELSE                 
         CALL grid_zgr__sco_s_tanh( td_nam,jpi,jpj,jpk, &
         &                          dl_scosrf, &
         &                          dl_hift, dl_hifu, dl_hifv, dl_hiff )
      ENDIF

      DEALLOCATE(dl_scosrf)

      DEALLOCATE(dl_hifv)
      DEALLOCATE(dl_hiff)
      DEALLOCATE(dl_hift)
      DEALLOCATE(dl_hifu)

      CALL lbc_lnk( tg_e3t_0%d_value(:,:,:,1) , 'T', td_nam%i_perio, 1._dp )
      CALL lbc_lnk( tg_e3u_0%d_value(:,:,:,1) , 'U', td_nam%i_perio, 1._dp )
      CALL lbc_lnk( tg_e3v_0%d_value(:,:,:,1) , 'V', td_nam%i_perio, 1._dp )
      CALL lbc_lnk( tg_e3f_0%d_value(:,:,:,1) , 'F', td_nam%i_perio, 1._dp )
      CALL lbc_lnk( tg_e3w_0%d_value(:,:,:,1) , 'W', td_nam%i_perio, 1._dp )
      CALL lbc_lnk( tg_e3uw_0%d_value(:,:,:,1), 'U', td_nam%i_perio, 1._dp )
      CALL lbc_lnk( tg_e3vw_0%d_value(:,:,:,1), 'V', td_nam%i_perio, 1._dp )

      IF( .NOT. td_nam%l_wd ) THEN
         WHERE( tg_e3t_0%d_value(:,:,:,1) == 0_dp )  tg_e3t_0%d_value(:,:,:,1) = 1._dp
         WHERE( tg_e3u_0%d_value(:,:,:,1) == 0_dp )  tg_e3u_0%d_value(:,:,:,1) = 1._dp
         WHERE( tg_e3v_0%d_value(:,:,:,1) == 0_dp )  tg_e3v_0%d_value(:,:,:,1) = 1._dp
         WHERE( tg_e3f_0%d_value(:,:,:,1) == 0_dp )  tg_e3f_0%d_value(:,:,:,1) = 1._dp
         WHERE( tg_e3w_0%d_value(:,:,:,1) == 0_dp )  tg_e3w_0%d_value(:,:,:,1) = 1._dp
         WHERE( tg_e3uw_0%d_value(:,:,:,1)== 0_dp )  tg_e3uw_0%d_value(:,:,:,1)= 1._dp
         WHERE( tg_e3vw_0%d_value(:,:,:,1)== 0_dp )  tg_e3vw_0%d_value(:,:,:,1)= 1._dp
      ENDIF

      ! HYBRID : 
      DO jj = 1, jpj
         DO ji = 1, jpi
            DO jk = 1, jpk-1
               IF( dl_scobot(ji,jj) >= tg_gdept_0%d_value(ji,jj,jk,1) )THEN
                  tg_mbathy%d_value(ji,jj,1,1) = REAL(MAX( 2, jk ),dp)
               ENDIF
            END DO

            IF( td_nam%l_wd ) THEN
               IF( dl_scobot(ji,jj) <= -(td_nam%d_wdld - td_nam%d_wdmin2) )THEN

                  tg_mbathy%d_value(ji,jj,1,1) = 0._dp

               ELSEIF( dl_scobot(ji,jj) <= td_nam%d_wdmin1 )THEN

                  tg_mbathy%d_value(ji,jj,1,1) = 2._dp

               ENDIF
            ELSE
               IF( dl_scobot(ji,jj) == 0._dp )THEN
                  tg_mbathy%d_value(ji,jj,1,1) = 0._dp
               ENDIF
            ENDIF

         ENDDO
      ENDDO

      DEALLOCATE(dl_scobot)

      CALL logger_info(' MIN val mbathy MIN '//TRIM(fct_str(MINVAL( tg_mbathy%d_value(:,:,1,1) )))//&
         &             ' MAX '//TRIM(fct_str(MAXVAL( tg_mbathy%d_value(:,:,1,1) ))) )
      CALL logger_info(' MIN val depth t '//TRIM(fct_str(MINVAL( tg_gdept_0%d_value(:,:,:,1) )))//&
         &                           ' w '//TRIM(fct_str(MINVAL( tg_gdepw_0%d_value(:,:,:,1) ))) )!//&
         !&                           '3w '//TRIM(fct_str(MINVAL( tg_gdep3w_0%d_value(:,:,:,1)))) )

      CALL logger_info(' MIN val e3    t '//TRIM(fct_str(MINVAL( tg_e3t_0%d_value(:,:,:,1) )))//&
         &                           ' f '//TRIM(fct_str(MINVAL( tg_e3f_0%d_value(:,:,:,1) )))//&
         &                           ' u '//TRIM(fct_str(MINVAL( tg_e3u_0%d_value(:,:,:,1) )))//&
         &                           ' v '//TRIM(fct_str(MINVAL( tg_e3v_0%d_value(:,:,:,1) )))//&
         &                          ' uw '//TRIM(fct_str(MINVAL( tg_e3uw_0%d_value(:,:,:,1) )))//&
         &                          ' vw '//TRIM(fct_str(MINVAL( tg_e3vw_0%d_value(:,:,:,1) )))//&
         &                           ' w '//TRIM(fct_str(MINVAL( tg_e3w_0%d_value(:,:,:,1) ))) )

      CALL logger_info(' MAX val depth t '//TRIM(fct_str(MAXVAL( tg_gdept_0%d_value(:,:,:,1) )))//&
         &                           ' w '//TRIM(fct_str(MAXVAL( tg_gdepw_0%d_value(:,:,:,1) ))) )!//&
         !&                           '3w '//TRIM(fct_str(MAXVAL( tg_gdep3w_0%d_value(:,:,:,1) ))) )

      CALL logger_info(' MAX val e3    t '//TRIM(fct_str(MAXVAL( tg_e3t_0%d_value(:,:,:,1) )))//&
         &                           ' f '//TRIM(fct_str(MAXVAL( tg_e3f_0%d_value(:,:,:,1) )))//&
         &                           ' u '//TRIM(fct_str(MAXVAL( tg_e3u_0%d_value(:,:,:,1) )))//&
         &                           ' v '//TRIM(fct_str(MAXVAL( tg_e3v_0%d_value(:,:,:,1) )))//&
         &                          ' uw '//TRIM(fct_str(MAXVAL( tg_e3uw_0%d_value(:,:,:,1) )))//&
         &                          ' vw '//TRIM(fct_str(MAXVAL( tg_e3vw_0%d_value(:,:,:,1) )))//&
         &                           ' w '//TRIM(fct_str(MAXVAL( tg_e3w_0%d_value(:,:,:,1) ))) )

!================================================================================
! check the coordinate makes sense
!================================================================================
      DO ji = 1, jpi
         DO jj = 1, jpj

            IF( tg_hbatt%d_value(ji,jj,1,1) > 0._dp )THEN
               DO jk = 1, INT(tg_mbathy%d_value(ji,jj,1,1),i4)
                  ! check coordinate is monotonically increasing
                  IF( tg_e3w_0%d_value(ji,jj,jk,1) <= 0._dp .OR. &
                    & tg_e3t_0%d_value(ji,jj,jk,1) <= 0._dp )THEN
                     CALL logger_fatal(' GRID ZGR SCO:   e3w   or e3t   =< 0  at point ('//&
                       &               TRIM(fct_str(ji))//","//&
                       &               TRIM(fct_str(jj))//","//&
                       &               TRIM(fct_str(jk))//")" )
                    !WRITE(numout,*) 'e3w',fse3w(ji,jj,:)
                    !WRITE(numout,*) 'e3t',fse3t(ji,jj,:)
                  ENDIF
                  ! and check it has never gone negative
                  IF( tg_gdepw_0%d_value(ji,jj,jk,1) < 0._dp .OR. &
                    & tg_gdept_0%d_value(ji,jj,jk,1) < 0._dp ) THEN
                     CALL logger_fatal('GRId ZGR SCO :   gdepw or gdept =< 0  at point ('//&
                       &               TRIM(fct_str(ji))//","//&
                       &               TRIM(fct_str(jj))//","//&
                       &               TRIM(fct_str(jk))//")" )
                    !WRITE(numout,*) 'gdepw',fsdepw(ji,jj,:)
                    !WRITE(numout,*) 'gdept',fsdept(ji,jj,:)
                  ENDIF
                  ! and check it never exceeds the total depth
                  IF( tg_gdepw_0%d_value(ji,jj,jk,1) > tg_hbatt%d_value(ji,jj,1,1) ) THEN
                     CALL logger_fatal('GRID ZGR SCO:   gdepw > hbatt  at point ('//&
                       &               TRIM(fct_str(ji))//","//&
                       &               TRIM(fct_str(jj))//","//&
                       &               TRIM(fct_str(jk))//")" )
                    !WRITE(numout,*) 'gdepw',fsdepw(ji,jj,:)
                  ENDIF
               ENDDO

               DO jk = 1, INT(tg_mbathy%d_value(ji,jj,1,1),i4)-1
                  ! and check it never exceeds the total depth
                  IF( tg_gdept_0%d_value(ji,jj,jk,1) > tg_hbatt%d_value(ji,jj,1,1) ) THEN
                    CALL logger_fatal('GRID ZGR SCO:   gdept > hbatt  at point ('//&
                       &               TRIM(fct_str(ji))//","//&
                       &               TRIM(fct_str(jj))//","//&
                       &               TRIM(fct_str(jk))//")" )
                    !WRITE(numout,*) 'gdept',fsdept(ji,jj,:)
                  ENDIF
               ENDDO
            ENDIF

         ENDDO
      ENDDO

   END SUBROUTINE grid_zgr__sco_fill
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr__sco_s_sh94(td_nam, jpi, jpj, jpk, dd_scosrf) 
   !-------------------------------------------------------------------
   !> @brief This subroutine stretch the s-coordinate system
   !>
   !> @details
   !> ** Method  :   s-coordinate stretch using the Song and Haidvogel 1994
   !>                mixed S/sigma coordinate
   !>
   !> Reference : Song and Haidvogel 1994.   
   !>
   !> @author J.Paul
   !> @date September, 2015 - rewrite from domzgr 
   !> @date October, 2016
   !> - add wetting and drying option
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   !> @param[in] jpj
   !> @param[in] jpk
   !> @param[in] dd_scosrf
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMZ),                 INTENT(IN   ) :: td_nam
      INTEGER(i4),                 INTENT(IN   ) :: jpi
      INTEGER(i4),                 INTENT(IN   ) :: jpj
      INTEGER(i4),                 INTENT(IN   ) :: jpk
      REAL(dp)   , DIMENSION(:,:), INTENT(IN   ) :: dd_scosrf

      ! local variable
      REAL(dp) :: zcoeft
      REAL(dp) :: zcoefw

      REAL(dp) ::   ztmpu
      REAL(dp) ::   ztmpv
      REAL(dp) ::   ztmpf
      REAL(dp) ::   ztmpu1
      REAL(dp) ::   ztmpv1
      REAL(dp) ::   ztmpf1

      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_gsigw3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_gsigt3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_gsi3w3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_esigt3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_esigw3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_esigtu3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_esigtv3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_esigtf3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_esigwu3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_esigwv3

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      ALLOCATE( z_gsigw3(jpi,jpj,jpk))
      ALLOCATE( z_gsigt3(jpi,jpj,jpk))
      ALLOCATE( z_gsi3w3(jpi,jpj,jpk))
      ALLOCATE( z_esigt3(jpi,jpj,jpk))
      ALLOCATE( z_esigw3(jpi,jpj,jpk))
      ALLOCATE( z_esigtu3(jpi,jpj,jpk))
      ALLOCATE( z_esigtv3(jpi,jpj,jpk))
      ALLOCATE( z_esigtf3(jpi,jpj,jpk))
      ALLOCATE( z_esigwu3(jpi,jpj,jpk))
      ALLOCATE( z_esigwv3(jpi,jpj,jpk))

      z_gsigw3(:,:,:) =0._dp
      z_gsigt3(:,:,:) =0._dp  
      z_gsi3w3(:,:,:) =0._dp  
      z_esigt3(:,:,:) =0._dp  
      z_esigw3(:,:,:) =0._dp  

      z_esigtu3(:,:,:)=0._dp 
      z_esigtv3(:,:,:)=0._dp
      z_esigtf3(:,:,:)=0._dp
      z_esigwu3(:,:,:)=0._dp
      z_esigwv3(:,:,:)=0._dp

      DO ji = 1, jpi
         DO jj = 1, jpj
   
            IF( tg_hbatt%d_value(ji,jj,1,1) > td_nam%d_hc ) THEN    !deep water, stretched sigma
               DO jk = 1, jpk
                  z_gsigw3(ji,jj,jk) = -grid_zgr__sco_fssig1( td_nam, jpk, REAL(jk,dp)-0.5_dp, td_nam%d_bb )
                  z_gsigt3(ji,jj,jk) = -grid_zgr__sco_fssig1( td_nam, jpk, REAL(jk,dp)       , td_nam%d_bb )
               END DO
            ELSE ! shallow water, uniform sigma
               DO jk = 1, jpk
                  z_gsigw3(ji,jj,jk) =   REAL(jk-1,dp)            / REAL(jpk-1,dp)
                  z_gsigt3(ji,jj,jk) = ( REAL(jk-1,dp) + 0.5_dp ) / REAL(jpk-1,dp)
               END DO
            ENDIF         

            DO jk = 1, jpk-1
               z_esigt3(ji,jj,jk  ) = z_gsigw3(ji,jj,jk+1) - z_gsigw3(ji,jj,jk)
               z_esigw3(ji,jj,jk+1) = z_gsigt3(ji,jj,jk+1) - z_gsigt3(ji,jj,jk)
            END DO            
            z_esigw3(ji,jj,1  ) = 2._dp * ( z_gsigt3(ji,jj,1  ) - z_gsigw3(ji,jj,1  ) )
            z_esigt3(ji,jj,jpk) = 2._dp * ( z_gsigt3(ji,jj,jpk) - z_gsigw3(ji,jj,jpk) )

            ! Coefficients for vertical depth as the sum of e3w scale factors
            z_gsi3w3(ji,jj,1) = 0.5_dp * z_esigw3(ji,jj,1)
            DO jk = 2, jpk
               z_gsi3w3(ji,jj,jk) = z_gsi3w3(ji,jj,jk-1) + z_esigw3(ji,jj,jk)
            END DO
            !
            DO jk = 1, jpk
               zcoeft = ( REAL(jk,dp) - 0.5_dp ) / REAL(jpk-1,dp)
               zcoefw = ( REAL(jk,dp) - 1.0_dp ) / REAL(jpk-1,dp)
               tg_gdept_0%d_value(ji,jj,jk,1)  = (  dd_scosrf(ji,jj) &
                  &                              + (tg_hbatt%d_value(ji,jj,1,1)-td_nam%d_hc)*z_gsigt3(ji,jj,jk) &
                  &                              +  td_nam%d_hc*zcoeft )
               tg_gdepw_0%d_value(ji,jj,jk,1)  = (  dd_scosrf(ji,jj) &
                  &                              + (tg_hbatt%d_value(ji,jj,1,1)-td_nam%d_hc)*z_gsigw3(ji,jj,jk) &
                  &                              +  td_nam%d_hc*zcoefw )
               !tg_gdep3w_0%d_value(ji,jj,jk,1) = (  dd_scosrf(ji,jj) &
               !   &                              + (tg_hbatt%d_value(ji,jj,1,1)-td_nam%d_hc)*z_gsi3w3(ji,jj,jk) &
               !   &                              +  td_nam%d_hc*zcoeft )
            END DO

         END DO   ! for all jj's
      END DO    ! for all ji's

      DO ji = 1, jpi-1
         DO jj = 1, jpj-1
            ! extended for Wetting/Drying case
            ztmpu  = tg_hbatt%d_value(ji  ,jj  ,1,1) + tg_hbatt%d_value(ji+1,jj  ,1,1)
            ztmpv  = tg_hbatt%d_value(ji  ,jj  ,1,1) + tg_hbatt%d_value(ji  ,jj+1,1,1)
            ztmpf  = tg_hbatt%d_value(ji  ,jj  ,1,1) + tg_hbatt%d_value(ji+1,jj  ,1,1) + &
                   & tg_hbatt%d_value(ji  ,jj+1,1,1) + tg_hbatt%d_value(ji+1,jj+1,1,1)

            ztmpu1 = tg_hbatt%d_value(ji  ,jj  ,1,1) * tg_hbatt%d_value(ji+1,jj  ,1,1)
            ztmpv1 = tg_hbatt%d_value(ji  ,jj  ,1,1) * tg_hbatt%d_value(ji  ,jj+1,1,1)
            ztmpf1 =    MIN(tg_hbatt%d_value(ji  ,jj  ,1,1), tg_hbatt%d_value(ji+1,jj  ,1,1), &
                   &        tg_hbatt%d_value(ji  ,jj+1,1,1), tg_hbatt%d_value(ji+1,jj+1,1,1)) &
                   & *  MAX(tg_hbatt%d_value(ji  ,jj  ,1,1), tg_hbatt%d_value(ji+1,jj  ,1,1), &
                   &        tg_hbatt%d_value(ji  ,jj+1,1,1), tg_hbatt%d_value(ji+1,jj+1,1,1))

            DO jk = 1, jpk

               IF( td_nam%l_wd .AND. &
                 & ( ztmpu1 < 0._dp .OR. ABS(ztmpu) < td_nam%d_wdmin1 ) )THEN
                  z_esigtu3(ji,jj,jk) = 0.5_dp * ( z_esigt3(ji,jj,jk) + z_esigt3(ji+1,jj,jk) )
                  z_esigwu3(ji,jj,jk) = 0.5_dp * ( z_esigw3(ji,jj,jk) + z_esigw3(ji+1,jj,jk) )
               ELSE
                  z_esigtu3(ji,jj,jk) = ( tg_hbatt%d_value(ji  ,jj,1,1)*z_esigt3(ji  ,jj,jk) &
                                      & + tg_hbatt%d_value(ji+1,jj,1,1)*z_esigt3(ji+1,jj,jk) ) / ztmpu
                  z_esigwu3(ji,jj,jk) = ( tg_hbatt%d_value(ji  ,jj,1,1)*z_esigw3(ji  ,jj,jk) &
                                      & + tg_hbatt%d_value(ji+1,jj,1,1)*z_esigw3(ji+1,jj,jk)) / ztmpu
               ENDIF

               IF( td_nam%l_wd .AND. &
                 & ( ztmpv1 < 0._dp .OR. ABS(ztmpv) < td_nam%d_wdmin1 ) )THEN
                  z_esigtv3(ji,jj,jk) = 0.5_dp * ( z_esigt3(ji,jj,jk) + z_esigt3(ji,jj+1,jk) )
                  z_esigwv3(ji,jj,jk) = 0.5_dp * ( z_esigw3(ji,jj,jk) + z_esigw3(ji,jj+1,jk) )
               ELSE
                  z_esigtv3(ji,jj,jk) = ( tg_hbatt%d_value(ji,jj  ,1,1)*z_esigt3(ji,jj  ,jk) &
                                      & + tg_hbatt%d_value(ji,jj+1,1,1)*z_esigt3(ji,jj+1,jk)) / ztmpv
                  z_esigwv3(ji,jj,jk) = ( tg_hbatt%d_value(ji,jj  ,1,1)*z_esigw3(ji,jj  ,jk) &
                                      & + tg_hbatt%d_value(ji,jj+1,1,1)*z_esigw3(ji,jj+1,jk)) / ztmpv
               ENDIF

               IF( td_nam%l_wd .AND. &
                 & ( ztmpf1 < 0._dp .OR. ABS(ztmpf) < td_nam%d_wdmin1 ) )THEN
                  z_esigtf3(ji,jj,jk) = 0.25_dp * ( z_esigt3(ji  ,jj  ,jk) &
                                      &           + z_esigt3(ji+1,jj  ,jk) &
                                      &           + z_esigt3(ji  ,jj+1,jk) &
                                      &           + z_esigt3(ji+1,jj+1,jk) )
               ELSE
                  z_esigtf3(ji,jj,jk) = ( tg_hbatt%d_value(ji  ,jj  ,1,1)*z_esigt3(ji  ,jj  ,jk) &
                                      & + tg_hbatt%d_value(ji+1,jj  ,1,1)*z_esigt3(ji+1,jj  ,jk) &
                                      & + tg_hbatt%d_value(ji  ,jj+1,1,1)*z_esigt3(ji  ,jj+1,jk) &
                                      & + tg_hbatt%d_value(ji+1,jj+1,1,1)*z_esigt3(ji+1,jj+1,jk) ) / ztmpf
               ENDIF
               !
               tg_e3t_0%d_value(ji,jj,jk,1) = ( ( tg_hbatt%d_value(ji,jj,1,1) - td_nam%d_hc )*z_esigt3 (ji,jj,jk) &
                  &                            + td_nam%d_hc / REAL(jpk-1,dp) )
               tg_e3u_0%d_value(ji,jj,jk,1) = ( ( tg_hbatu%d_value(ji,jj,1,1) - td_nam%d_hc )*z_esigtu3(ji,jj,jk) &
                  &                            + td_nam%d_hc / REAL(jpk-1,dp) )
               tg_e3v_0%d_value(ji,jj,jk,1) = ( ( tg_hbatv%d_value(ji,jj,1,1) - td_nam%d_hc )*z_esigtv3(ji,jj,jk) &
                  &                            + td_nam%d_hc / REAL(jpk-1,dp) )
               tg_e3f_0%d_value(ji,jj,jk,1) = ( ( tg_hbatf%d_value(ji,jj,1,1) - td_nam%d_hc ) *z_esigtf3(ji,jj,jk) &
                  &                            + td_nam%d_hc/REAL(jpk-1,dp) )
               
               tg_e3w_0%d_value (ji,jj,jk,1)= ( ( tg_hbatt%d_value(ji,jj,1,1) - td_nam%d_hc )*z_esigw3 (ji,jj,jk) &
                  &                            + td_nam%d_hc / REAL(jpk-1,dp) )
               tg_e3uw_0%d_value(ji,jj,jk,1)= ( ( tg_hbatu%d_value(ji,jj,1,1) - td_nam%d_hc)*z_esigwu3(ji,jj,jk) &
                 &                             + td_nam%d_hc/REAL(jpk-1,dp) )
               tg_e3vw_0%d_value(ji,jj,jk,1)= ( ( tg_hbatv%d_value(ji,jj,1,1) - td_nam%d_hc)*z_esigwv3(ji,jj,jk) &
                 &                             + td_nam%d_hc/REAL(jpk-1,dp) )
            END DO
        END DO
      END DO

      DEALLOCATE( z_gsigw3  )
      DEALLOCATE( z_gsigt3  )
      DEALLOCATE( z_gsi3w3  )
      DEALLOCATE( z_esigt3  )
      DEALLOCATE( z_esigw3  )
      DEALLOCATE( z_esigtu3 )
      DEALLOCATE( z_esigtv3 )
      DEALLOCATE( z_esigtf3 )
      DEALLOCATE( z_esigwu3 )
      DEALLOCATE( z_esigwv3 )

   END SUBROUTINE grid_zgr__sco_s_sh94
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr__sco_s_sf12(td_nam, jpi, jpj, jpk, dd_scosrf) 
   !-------------------------------------------------------------------
   !> @brief This subroutine stretch the s-coordinate system
   !>
   !> ** Method  :   s-coordinate stretch using the Siddorn and Furner 2012?
   !>                mixed S/sigma/Z coordinate
   !>
   !>                This method allows the maintenance of fixed surface and or
   !>                bottom cell resolutions (cf. geopotential coordinates) 
   !>                within an analytically derived stretched S-coordinate framework.
   !>
   !>
   !> Reference : Siddorn and Furner 2012 (submitted Ocean modelling).   
   !>
   !> @author J.Paul
   !> @date September, 2015 - rewrite from domzgr 
   !> @date October, 2016
   !> - add wetting and drying option
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   !> @param[in] jpj
   !> @param[in] jpk
   !> @param[in] dd_scosrf
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMZ),                 INTENT(IN   ) :: td_nam
      INTEGER(i4),                 INTENT(IN   ) :: jpi
      INTEGER(i4),                 INTENT(IN   ) :: jpj
      INTEGER(i4),                 INTENT(IN   ) :: jpk
      REAL(dp)   , DIMENSION(:,:), INTENT(IN   ) :: dd_scosrf

      ! local variable
      REAL(dp) ::   zsmth     ! smoothing around critical depth
      REAL(dp) ::   zzs, zzb  ! Surface and bottom cell thickness in sigma space

      REAL(dp) ::   ztmpu
      REAL(dp) ::   ztmpv
      REAL(dp) ::   ztmpf
      REAL(dp) ::   ztmpu1
      REAL(dp) ::   ztmpv1
      REAL(dp) ::   ztmpf1

      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_gsigw3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_gsigt3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_gsi3w3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_esigt3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_esigw3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_esigtu3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_esigtv3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_esigtf3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_esigwu3
      REAL(dp), DIMENSION(:,:,:), ALLOCATABLE :: z_esigwv3

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------
      ALLOCATE( z_gsigw3(jpi,jpj,jpk))
      ALLOCATE( z_gsigt3(jpi,jpj,jpk))
      ALLOCATE( z_gsi3w3(jpi,jpj,jpk))
      ALLOCATE( z_esigt3(jpi,jpj,jpk))
      ALLOCATE( z_esigw3(jpi,jpj,jpk))
      ALLOCATE( z_esigtu3(jpi,jpj,jpk))
      ALLOCATE( z_esigtv3(jpi,jpj,jpk))
      ALLOCATE( z_esigtf3(jpi,jpj,jpk))
      ALLOCATE( z_esigwu3(jpi,jpj,jpk))
      ALLOCATE( z_esigwv3(jpi,jpj,jpk))

      z_gsigw3(:,:,:) =0._dp
      z_gsigt3(:,:,:) =0._dp  
      z_gsi3w3(:,:,:) =0._dp  
      z_esigt3(:,:,:) =0._dp  
      z_esigw3(:,:,:) =0._dp  
      z_esigtu3(:,:,:)=0._dp 
      z_esigtv3(:,:,:)=0._dp
      z_esigtf3(:,:,:)=0._dp
      z_esigwu3(:,:,:)=0._dp
      z_esigwv3(:,:,:)=0._dp

      DO ji = 1, jpi
         DO jj = 1, jpj

          IF( tg_hbatt%d_value(ji,jj,1,1) > td_nam%d_hc )THEN !deep water, stretched sigma
              
             ! this forces a linear bottom cell depth relationship with H,.
             ! could be changed by users but care must be taken to do so carefully
              zzb = tg_hbatt%d_value(ji,jj,1,1)*td_nam%d_zb_a + td_nam%d_zb_b

              zzb = 1.0_dp-(zzb/tg_hbatt%d_value(ji,jj,1,1))
            
              zzs = td_nam%d_zs / tg_hbatt%d_value(ji,jj,1,1) 
              
              IF( td_nam%d_efold /= 0.0_dp )THEN
                zsmth = TANH( (tg_hbatt%d_value(ji,jj,1,1)-td_nam%d_hc ) / td_nam%d_efold )
              ELSE
                zsmth = 1.0_dp 
              ENDIF
               
              DO jk = 1, jpk
                z_gsigw3(ji,jj,jk) =  REAL(jk-1,dp)        /REAL(jpk-1,dp)
                z_gsigt3(ji,jj,jk) = (REAL(jk-1,dp)+0.5_dp)/REAL(jpk-1,dp)
              ENDDO
              z_gsigw3(ji,jj,:) = grid_zgr__sco_fgamma( td_nam, jpk, z_gsigw3(ji,jj,:), zzb, zzs, zsmth  )
              z_gsigt3(ji,jj,:) = grid_zgr__sco_fgamma( td_nam, jpk, z_gsigt3(ji,jj,:), zzb, zzs, zsmth  )
 
          ELSE IF( td_nam%l_sigcrit )THEN ! shallow water, uniform sigma

            DO jk = 1, jpk
              z_gsigw3(ji,jj,jk) =  REAL(jk-1,dp)     /REAL(jpk-1,dp)
              z_gsigt3(ji,jj,jk) = (REAL(jk-1,dp)+0.5)/REAL(jpk-1,dp)
            END DO

          ELSE  ! shallow water, z coordinates

            DO jk = 1, jpk
              z_gsigw3(ji,jj,jk) =  REAL(jk-1,dp)        /REAL(jpk-1,dp)*(td_nam%d_hc/tg_hbatt%d_value(ji,jj,1,1))
              z_gsigt3(ji,jj,jk) = (REAL(jk-1,dp)+0.5_dp)/REAL(jpk-1,dp)*(td_nam%d_hc/tg_hbatt%d_value(ji,jj,1,1))
            END DO

          ENDIF

          DO jk = 1, jpk-1
             z_esigt3(ji,jj,jk  ) = z_gsigw3(ji,jj,jk+1) - z_gsigw3(ji,jj,jk)
             z_esigw3(ji,jj,jk+1) = z_gsigt3(ji,jj,jk+1) - z_gsigt3(ji,jj,jk)
          END DO
          z_esigw3(ji,jj,1  ) = 2.0_dp * (z_gsigt3(ji,jj,1  ) - z_gsigw3(ji,jj,1  ))
          z_esigt3(ji,jj,jpk) = 2.0_dp * (z_gsigt3(ji,jj,jpk) - z_gsigw3(ji,jj,jpk))

          ! Coefficients for vertical depth as the sum of e3w scale factors
          z_gsi3w3(ji,jj,1) = 0.5_dp * z_esigw3(ji,jj,1)
          DO jk = 2, jpk
             z_gsi3w3(ji,jj,jk) = z_gsi3w3(ji,jj,jk-1) + z_esigw3(ji,jj,jk)
          END DO

          DO jk = 1, jpk
             tg_gdept_0%d_value(ji,jj,jk,1) = (dd_scosrf(ji,jj)+tg_hbatt%d_value(ji,jj,1,1))*z_gsigt3(ji,jj,jk)
             tg_gdepw_0%d_value(ji,jj,jk,1) = (dd_scosrf(ji,jj)+tg_hbatt%d_value(ji,jj,1,1))*z_gsigw3(ji,jj,jk)
             !tg_gdep3w_0%d_value(ji,jj,jk,1) = (dd_scosrf(ji,jj)+tg_hbatt%d_value(ji,jj,1,1))*z_gsi3w3(ji,jj,jk)
          END DO

        ENDDO   ! for all jj's
      ENDDO    ! for all ji's

      DO ji=1,jpi-1
        DO jj=1,jpj-1

           ! extend to suit for Wetting/Drying case
           ztmpu  = tg_hbatt%d_value(ji  ,jj  ,1,1) + tg_hbatt%d_value(ji+1,jj  ,1,1)
           ztmpv  = tg_hbatt%d_value(ji  ,jj  ,1,1) + tg_hbatt%d_value(ji  ,jj+1,1,1)
           ztmpf  = tg_hbatt%d_value(ji  ,jj  ,1,1) + tg_hbatt%d_value(ji+1,jj  ,1,1) + &
                  & tg_hbatt%d_value(ji  ,jj+1,1,1) + tg_hbatt%d_value(ji+1,jj+1,1,1)

           ztmpu1 = tg_hbatt%d_value(ji  ,jj  ,1,1) * tg_hbatt%d_value(ji+1,jj  ,1,1)
           ztmpv1 = tg_hbatt%d_value(ji  ,jj  ,1,1) * tg_hbatt%d_value(ji  ,jj+1,1,1)
           ztmpf1 =    MIN(tg_hbatt%d_value(ji  ,jj  ,1,1), tg_hbatt%d_value(ji+1,jj  ,1,1), &
                  &        tg_hbatt%d_value(ji  ,jj+1,1,1), tg_hbatt%d_value(ji+1,jj+1,1,1)) & 
                  &  * MAX(tg_hbatt%d_value(ji  ,jj  ,1,1), tg_hbatt%d_value(ji+1,jj  ,1,1), &
                  &        tg_hbatt%d_value(ji  ,jj+1,1,1), tg_hbatt%d_value(ji+1,jj+1,1,1))

          DO jk = 1, jpk

            IF( td_nam%l_wd .AND. &
              & ( ztmpu1 < 0._dp .OR. ABS(ztmpu) < td_nam%d_wdmin1 ) )THEN
               z_esigtu3(ji,jj,jk) = 0.5_dp * ( z_esigt3(ji,jj,jk) + z_esigt3(ji+1,jj,jk) )
               z_esigwu3(ji,jj,jk) = 0.5_dp * ( z_esigw3(ji,jj,jk) + z_esigw3(ji+1,jj,jk) )
            ELSE
               z_esigtu3(ji,jj,jk) = ( tg_hbatt%d_value(ji  ,jj,1,1)*z_esigt3(ji  ,jj,jk) &
                                   & + tg_hbatt%d_value(ji+1,jj,1,1)*z_esigt3(ji+1,jj,jk) ) / ztmpu
               z_esigwu3(ji,jj,jk) = ( tg_hbatt%d_value(ji  ,jj,1,1)*z_esigw3(ji  ,jj,jk) &
                                   & + tg_hbatt%d_value(ji+1,jj,1,1)*z_esigw3(ji+1,jj,jk) ) / ztmpu
            ENDIF

            IF( td_nam%l_wd .AND. &
              & ( ztmpv1 < 0._dp .OR. ABS(ztmpv) < td_nam%d_wdmin1 ) )THEN
               z_esigtv3(ji,jj,jk) = 0.5_dp * ( z_esigt3(ji,jj,jk) + z_esigt3(ji,jj+1,jk) )
               z_esigwv3(ji,jj,jk) = 0.5_dp * ( z_esigw3(ji,jj,jk) + z_esigw3(ji,jj+1,jk) )
            ELSE
               z_esigtv3(ji,jj,jk) = ( tg_hbatt%d_value(ji,jj  ,1,1)*z_esigt3(ji,jj  ,jk) &
                                   & + tg_hbatt%d_value(ji,jj+1,1,1)*z_esigt3(ji,jj+1,jk)) / ztmpv
               z_esigwv3(ji,jj,jk) = ( tg_hbatt%d_value(ji,jj  ,1,1)*z_esigw3(ji,jj  ,jk) &
                                   & + tg_hbatt%d_value(ji,jj+1,1,1)*z_esigw3(ji,jj+1,jk)) / ztmpv
            ENDIF

            IF( td_nam%l_wd .AND. &
              & ( ztmpf1 < 0._dp .OR. ABS(ztmpf) < td_nam%d_wdmin1 ) )THEN
               z_esigtf3(ji,jj,jk) = 0.25_dp * ( z_esigt3(ji  ,jj  ,jk) &
                                   &           + z_esigt3(ji+1,jj  ,jk) &
                                   &           + z_esigt3(ji  ,jj+1,jk) &
                                   &           + z_esigt3(ji+1,jj+1,jk) )
            ELSE
               z_esigtf3(ji,jj,jk) = ( tg_hbatt%d_value(ji  ,jj  ,1,1)*z_esigt3(ji  ,jj  ,jk) &
                                   & + tg_hbatt%d_value(ji+1,jj  ,1,1)*z_esigt3(ji+1,jj  ,jk) &
                                   & + tg_hbatt%d_value(ji  ,jj+1,1,1)*z_esigt3(ji  ,jj+1,jk) &
                                   & + tg_hbatt%d_value(ji+1,jj+1,1,1)*z_esigt3(ji+1,jj+1,jk) ) / ztmpf
            ENDIF

             tg_e3t_0%d_value(ji,jj,jk,1) = (dd_scosrf(ji,jj)+tg_hbatt%d_value(ji,jj,1,1))*z_esigt3 (ji,jj,jk)
             tg_e3u_0%d_value(ji,jj,jk,1) = (dd_scosrf(ji,jj)+tg_hbatu%d_value(ji,jj,1,1))*z_esigtu3(ji,jj,jk)
             tg_e3v_0%d_value(ji,jj,jk,1) = (dd_scosrf(ji,jj)+tg_hbatv%d_value(ji,jj,1,1))*z_esigtv3(ji,jj,jk)
             tg_e3f_0%d_value(ji,jj,jk,1) = (dd_scosrf(ji,jj)+tg_hbatf%d_value(ji,jj,1,1))*z_esigtf3(ji,jj,jk)
             !
             tg_e3w_0%d_value(ji,jj,jk,1)  = tg_hbatt%d_value(ji,jj,1,1)*z_esigw3 (ji,jj,jk)
             tg_e3uw_0%d_value(ji,jj,jk,1) = tg_hbatu%d_value(ji,jj,1,1)*z_esigwu3(ji,jj,jk)
             tg_e3vw_0%d_value(ji,jj,jk,1) = tg_hbatv%d_value(ji,jj,1,1)*z_esigwv3(ji,jj,jk)
          END DO

        ENDDO
      ENDDO

      CALL lbc_lnk(tg_e3t_0 %d_value(:,:,:,1),'T', td_nam%i_perio, 1._dp) 
      CALL lbc_lnk(tg_e3u_0 %d_value(:,:,:,1),'T', td_nam%i_perio, 1._dp)
      CALL lbc_lnk(tg_e3v_0 %d_value(:,:,:,1),'T', td_nam%i_perio, 1._dp) 
      CALL lbc_lnk(tg_e3f_0 %d_value(:,:,:,1),'T', td_nam%i_perio, 1._dp)
      CALL lbc_lnk(tg_e3w_0 %d_value(:,:,:,1),'T', td_nam%i_perio, 1._dp)
      CALL lbc_lnk(tg_e3uw_0%d_value(:,:,:,1),'T', td_nam%i_perio, 1._dp) 
      CALL lbc_lnk(tg_e3vw_0%d_value(:,:,:,1),'T', td_nam%i_perio, 1._dp)

      DEALLOCATE( z_gsigw3  )
      DEALLOCATE( z_gsigt3  )
      DEALLOCATE( z_gsi3w3  )
      DEALLOCATE( z_esigt3  )
      DEALLOCATE( z_esigw3  )
      DEALLOCATE( z_esigtu3 )
      DEALLOCATE( z_esigtv3 )
      DEALLOCATE( z_esigtf3 )
      DEALLOCATE( z_esigwu3 )
      DEALLOCATE( z_esigwv3 )

   END SUBROUTINE grid_zgr__sco_s_sf12
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr__sco_s_tanh(td_nam, jpi, jpj, jpk, &
         &                         dd_scosrf,             &
         &                         dd_hift, dd_hifu, dd_hifv, dd_hiff) 
   !-------------------------------------------------------------------
   !> @brief This subroutine stretch the s-coordinate system
   !>
   !>
   !> ** Method  :   s-coordinate stretch 
   !>
   !> Reference : Madec, Lott, Delecluse and Crepon, 1996. JPO, 26, 1393-1408.   
   !>
   !> @author J.Paul
   !> @date September, 2015 - rewrite from domzgr
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   !> @param[in] jpj
   !> @param[in] jpk
   !> @param[in] dd_scosrf
   !> @param[in] dd_hift 
   !> @param[in] dd_hifu 
   !> @param[in] dd_hifv 
   ! @param[in] dd_hiff 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMZ),                 INTENT(IN   ) :: td_nam
      INTEGER(i4),                 INTENT(IN   ) :: jpi
      INTEGER(i4),                 INTENT(IN   ) :: jpj
      INTEGER(i4),                 INTENT(IN   ) :: jpk
      REAL(dp)   , DIMENSION(:,:), INTENT(IN   ) :: dd_scosrf
      REAL(dp)   , DIMENSION(:,:), INTENT(IN   ) :: dd_hift
      REAL(dp)   , DIMENSION(:,:), INTENT(IN   ) :: dd_hifu
      REAL(dp)   , DIMENSION(:,:), INTENT(IN   ) :: dd_hifv
      REAL(dp)   , DIMENSION(:,:), INTENT(IN   ) :: dd_hiff

      ! local variable
      REAL(dp) :: zcoeft
      REAL(dp) :: zcoefw

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      tg_gsigt%d_value(1,1,:,1) =0._dp  
      tg_gsigw%d_value(1,1,:,1) =0._dp
      tg_gsi3w%d_value(1,1,:,1) =0._dp  
      tg_esigt%d_value(1,1,:,1) =0._dp  
      tg_esigw%d_value(1,1,:,1) =0._dp

      DO jk = 1, jpk
        tg_gsigw%d_value(1,1,jk,1) = -grid_zgr__sco_fssig( td_nam, jpk, REAL(jk,dp)-0.5_dp )
        tg_gsigt%d_value(1,1,jk,1) = -grid_zgr__sco_fssig( td_nam, jpk, REAL(jk,dp)        )
      END DO
      CALL logger_info('z_gsigw 1 jpk '//TRIM(fct_str(tg_gsigw%d_value(1,1,1,1)))//&
         &             TRIM(fct_str(tg_gsigw%d_value(1,1,jpk,1))) )

      ! Coefficients for vertical scale factors at w-, t- levels
!!gm bug :  define it from analytical function, not like juste bellow....
!!gm        or betteroffer the 2 possibilities....
      DO jk = 1, jpk-1
         tg_esigt%d_value(1,1,jk  ,1) = tg_gsigw%d_value(1,1,jk+1,1) - tg_gsigw%d_value(1,1,jk,1)
         tg_esigw%d_value(1,1,jk+1,1) = tg_gsigt%d_value(1,1,jk+1,1) - tg_gsigt%d_value(1,1,jk,1)
      END DO
      tg_esigw%d_value(1,1, 1 ,1) = 2._dp * ( tg_gsigt%d_value(1,1,1  ,1) - tg_gsigw%d_value(1,1,1  ,1) ) 
      tg_esigt%d_value(1,1,jpk,1) = 2._dp * ( tg_gsigt%d_value(1,1,jpk,1) - tg_gsigw%d_value(1,1,jpk,1) )

      ! Coefficients for vertical depth as the sum of e3w scale factors
      tg_gsi3w%d_value(1,1,1,1) = 0.5_dp * tg_esigw%d_value(1,1,1,1)
      DO jk = 2, jpk
         tg_gsi3w%d_value(1,1,jk,1) = tg_gsi3w%d_value(1,1,jk-1,1) + tg_esigw%d_value(1,1,jk,1)
      END DO
!!gm: depuw, depvw can be suppressed (modif in ldfslp) and depw=dep3w can be set (save 3 3D arrays)
      DO jk = 1, jpk
         zcoeft = ( REAL(jk,dp) - 0.5_dp ) / REAL(jpk-1,dp)
         zcoefw = ( REAL(jk,dp) - 1.0_dp ) / REAL(jpk-1,dp)
         tg_gdept_0%d_value (:,:,jk,1) = dd_scosrf(:,:) + ( tg_hbatt%d_value(:,:,1 ,1) - dd_hift(:,:) ) &
            &                                             * tg_gsigt%d_value(1,1,jk,1) + dd_hift(:,:)*zcoeft
         tg_gdepw_0%d_value (:,:,jk,1) = dd_scosrf(:,:) + ( tg_hbatt%d_value(:,:,1 ,1) - dd_hift(:,:) ) &
            &                                             * tg_gsigw%d_value(1,1,jk,1) + dd_hift(:,:)*zcoefw
         !tg_gdep3w_0%d_value(:,:,jk,1) = dd_scosrf(:,:) + ( tg_hbatt%d_value(:,:,1 ,1) - dd_hift(:,:)) &
         !                                                 * tg_gsi3w%d_value(1,1,jk,1) + dd_hift(:,:)*zcoeft
      END DO
!!gm: e3uw, e3vw can be suppressed  (modif in dynzdf, dynzdf_iso, zdfbfr) (save 2 3D arrays)
      DO jj = 1, jpj
         DO ji = 1, jpi
            DO jk = 1, jpk
              tg_e3t_0%d_value(ji,jj,jk,1) = ( (tg_hbatt%d_value(ji,jj,1 ,1) - dd_hift(ji,jj)) &
                 &                            * tg_esigt%d_value(1 ,1 ,jk,1) + dd_hift(ji,jj)/REAL(jpk-1,dp) )
              tg_e3u_0%d_value(ji,jj,jk,1) = ( (tg_hbatu%d_value(ji,jj,1 ,1) - dd_hifu(ji,jj)) &
                 &                            * tg_esigt%d_value(1 ,1 ,jk,1) + dd_hifu(ji,jj)/REAL(jpk-1,dp) )
              tg_e3v_0%d_value(ji,jj,jk,1) = ( (tg_hbatv%d_value(ji,jj,1 ,1) - dd_hifv(ji,jj)) &
                 &                            * tg_esigt%d_value(1 ,1 ,jk,1) + dd_hifv(ji,jj)/REAL(jpk-1,dp) )
              tg_e3f_0%d_value(ji,jj,jk,1) = ( (tg_hbatf%d_value(ji,jj,1 ,1) - dd_hiff(ji,jj)) &
                 &                            * tg_esigt%d_value(1 ,1, jk,1) + dd_hiff(ji,jj)/REAL(jpk-1,dp) )
               
              tg_e3w_0%d_value (ji,jj,jk,1)= ( (tg_hbatt%d_value(ji,jj,1 ,1) - dd_hift(ji,jj)) &
                 &                            * tg_esigw%d_value(1 ,1 ,jk,1) + dd_hift(ji,jj)/REAL(jpk-1,dp) )
              tg_e3uw_0%d_value(ji,jj,jk,1)= ( (tg_hbatu%d_value(ji,jj,1 ,1) - dd_hifu(ji,jj)) &
                 &                            * tg_esigw%d_value(1 ,1 ,jk,1) + dd_hifu(ji,jj)/REAL(jpk-1,dp) )
              tg_e3vw_0%d_value(ji,jj,jk,1)= ( (tg_hbatv%d_value(ji,jj,1 ,1) - dd_hifv(ji,jj)) &
                 &                            * tg_esigw%d_value(1 ,1 ,jk,1) + dd_hifv(ji,jj)/REAL(jpk-1,dp) )
            END DO
         END DO
      END DO

   END SUBROUTINE grid_zgr__sco_s_tanh
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid_zgr__sco_fssig(td_nam, jpk, pk) &
         &  RESULT( pf )
   !!----------------------------------------------------------------------
   !> @brief This function provide the analytical function in s-coordinate
   !>        
   !> @details
   !> ** Method  :   the function provide the non-dimensional position of
   !>                T and W (i.e. between 0 and 1)
   !>                T-points at integer values (between 1 and jpk)
   !>                W-points at integer values - 1/2 (between 0.5 and jpk-0.5)
   !>
   !> @author J.Paul
   !> @date September, 2015 - rewrite from domzgr
   !>
   !> @param[in] td_nam 
   !> @param[in] jpk
   !> @param[in] pk
   !!----------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TNAMZ), INTENT(IN   ) :: td_nam
      INTEGER(i4), INTENT(IN   ) :: jpk
      REAL(dp)   , INTENT(IN   ) :: pk   ! continuous "k" coordinate
      REAL(dp)                   :: pf   ! sigma value
      !!----------------------------------------------------------------------
      !
      pf =   (   TANH( td_nam%d_theta * ( -(pk-0.5_dp) / REAL(jpk-1,dp) + td_nam%d_thetb )  ) &
         &     - TANH( td_nam%d_thetb * td_nam%d_theta                                   )  ) &
         & * (   COSH( td_nam%d_theta                                      )                  &
         &     + COSH( td_nam%d_theta * ( 2._dp * td_nam%d_thetb - 1._dp ) )  )               &
         & / ( 2._dp * SINH( td_nam%d_theta ) )
      !
   END FUNCTION grid_zgr__sco_fssig
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid_zgr__sco_fssig1(td_nam, jpk, pk1, pbb) &
         & RESULT( pf1 )
   !!----------------------------------------------------------------------
   !> @brief This function provide the Song and Haidvogel version of the analytical function in s-coordinate
   !>
   !> @details
   !> ** Method  :   the function provides the non-dimensional position of
   !>                T and W (i.e. between 0 and 1)
   !>                T-points at integer values (between 1 and jpk)
   !>                W-points at integer values - 1/2 (between 0.5 and jpk-0.5)
   !>
   !> @author J.Paul
   !> @date September, 2015 - rewrite from domzgr
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   !> @param[in] jpj
   !> @param[in] pk1 
   !> @param[in] pbb
   !!----------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TNAMZ), INTENT(IN   ) :: td_nam
      INTEGER(i4), INTENT(IN   ) :: jpk
      REAL(dp)   , INTENT(IN   ) :: pk1   ! continuous "k" coordinate
      REAL(dp)   , INTENT(IN   ) :: pbb   ! Stretching coefficient
      REAL(dp)                   :: pf1   ! sigma value
      !!----------------------------------------------------------------------
      !
      IF ( td_nam%d_theta == 0 ) then      ! uniform sigma
         pf1 = - ( pk1 - 0.5_dp ) / REAL(jpk-1,dp)
      ELSE                        ! stretched sigma
         pf1 =   ( 1._dp - pbb ) * ( SINH( td_nam%d_theta*(-(pk1-0.5_dp)/REAL(jpk-1,dp)) ) ) / SINH( td_nam%d_theta )              &
            &  + pbb * (  (TANH( td_nam%d_theta*( (-(pk1-0.5_dp)/REAL(jpk-1,dp)) + 0.5_dp) ) - TANH( 0.5_dp * td_nam%d_theta )  )  &
            &        / ( 2._dp * TANH( 0.5_dp * td_nam%d_theta ) )  )
      ENDIF
      !
   END FUNCTION grid_zgr__sco_fssig1
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid_zgr__sco_fgamma(td_nam, jpk, pk1, pzb, pzs, psmth) &
         & RESULT( p_gamma )
   !!----------------------------------------------------------------------
   !> @brief This function provide analytical function for the s-coordinate
   !>
   !> @details
   !> ** Method  :   the function provides the non-dimensional position of
   !>                T and W (i.e. between 0 and 1)
   !>                T-points at integer values (between 1 and jpk)
   !>                W-points at integer values - 1/2 (between 0.5 and jpk-0.5)
   !>
   !>                This method allows the maintenance of fixed surface and or
   !>                bottom cell resolutions (cf. geopotential coordinates) 
   !>                within an analytically derived stretched S-coordinate framework.
   !>
   !> Reference  :   Siddorn and Furner, in prep
   !>
   !> @author J.Paul
   !> @date September, 2015 - rewrite from domzgr 
   !>
   !> @param[in] td_nam
   !> @param[in] jpk
   !> @param[in] pk1 
   !> @param[in] pzb
   !> @param[in] pzs
   !> @param[in] pzsmth
   !!----------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TNAMZ), INTENT(IN   ) :: td_nam
      INTEGER(i4), INTENT(IN   ) :: jpk
      REAL(dp)   , DIMENSION(:) , INTENT(IN   ) :: pk1       ! continuous "k" coordinate

      ! function
      REAL(dp)   , DIMENSION(jpk) :: p_gamma   ! stretched coordinate

      ! local variable
      REAL(dp)   , INTENT(IN   ) :: pzb           ! Bottom box depth
      REAL(dp)   , INTENT(IN   ) :: pzs           ! surface box depth
      REAL(dp)   , INTENT(IN   ) :: psmth       ! Smoothing parameter
      REAL(dp)                   :: za1,za2,za3    ! local variables
      REAL(dp)                   :: zn1,zn2        ! local variables
      REAL(dp)                   :: za,zb,zx       ! local variables

      ! loop indices
      INTEGER(i4)             ::   jk
      !!----------------------------------------------------------------------
      !
      zn1  =  1._dp / REAL(jpk-1,dp)
      zn2  =  1._dp -  zn1

      za1 = (td_nam%d_alpha+2.0_dp)*zn1**(td_nam%d_alpha+1.0_dp)-(td_nam%d_alpha+1.0_dp)*zn1**(td_nam%d_alpha+2.0_dp) 
      za2 = (td_nam%d_alpha+2.0_dp)*zn2**(td_nam%d_alpha+1.0_dp)-(td_nam%d_alpha+1.0_dp)*zn2**(td_nam%d_alpha+2.0_dp)
      za3 = (zn2**3.0_dp - za2)/( zn1**3.0_dp - za1)
     
      za = pzb - za3*(pzs-za1)-za2
      za = za/( zn2-0.5_dp*(za2+zn2**2.0_dp) - za3*(zn1-0.5_dp*(za1+zn1**2.0_dp) ) )
      zb = (pzs - za1 - za*( zn1-0.5_dp*(za1+zn1**2.0_dp ) ) ) / (zn1**3.0_dp - za1)
      zx = 1.0_dp-za/2.0_dp-zb
 
      DO jk = 1, jpk
        p_gamma(jk) = za*(pk1(jk)*(1.0_dp-pk1(jk)/2.0_dp))+zb*pk1(jk)**3.0_dp +  &
                    & zx*( (td_nam%d_alpha+2.0_dp)*pk1(jk)**(td_nam%d_alpha+1.0_dp)- &
                    &      (td_nam%d_alpha+1.0_dp)*pk1(jk)**(td_nam%d_alpha+2.0_dp) )
        p_gamma(jk) = p_gamma(jk)*psmth+pk1(jk)*(1.0_dp-psmth)
      ENDDO 

      !
   END FUNCTION grid_zgr__sco_fgamma
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_zgr_sco_stiff(td_nam, jpi, jpj, jpk)
   !-------------------------------------------------------------------
   !> @brief This subroutine stretch the s-coordinate system
   !>
   !>
   !> ** Method  :   s-coordinate stretch 
   !>
   !> Reference : Madec, Lott, Delecluse and Crepon, 1996. JPO, 26, 1393-1408.   
   !>
   !> @author J.Paul
   !> @date September, 2015 - rewrite from domain (dom_stiff) 
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   !> @param[in] jpj
   !> @param[in] jpk
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMZ), INTENT(IN   ) :: td_nam
      INTEGER(i4), INTENT(IN   ) :: jpi
      INTEGER(i4), INTENT(IN   ) :: jpj
      INTEGER(i4), INTENT(IN   ) :: jpk

      ! local variable
      REAL(dp) ::   zrxmax
      REAL(dp), DIMENSION(4) :: zr1

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------
      tg_rx1%d_value(:,:,1,1) = 0._dp

      zrxmax   = 0._dp
      zr1(:)   = 0._dp

      DO ji = 2, jpi-1
         DO jj = 2, jpj-1
            DO jk = 1, jpk-1
               zr1(1) = tg_umask%d_value(ji-1,jj  ,jk,1) &
                  &            * ABS( ( tg_gdepw_0%d_value(ji  ,jj  ,jk  ,1) &
                  &                   - tg_gdepw_0%d_value(ji-1,jj  ,jk  ,1) & 
                  &                   + tg_gdepw_0%d_value(ji  ,jj  ,jk+1,1) &
                  &                   - tg_gdepw_0%d_value(ji-1,jj  ,jk+1,1) ) &
                  &                 / ( tg_gdepw_0%d_value(ji  ,jj  ,jk  ,1) &
                  &                   + tg_gdepw_0%d_value(ji-1,jj  ,jk  ,1) &
                  &                   - tg_gdepw_0%d_value(ji  ,jj  ,jk+1,1) &
                  &                   - tg_gdepw_0%d_value(ji-1,jj  ,jk+1,1) &
                  &                   + dp_eps) )
               zr1(2) = tg_umask%d_value(ji  ,jj  ,jk,1) &
                  &            * ABS( ( tg_gdepw_0%d_value(ji+1,jj  ,jk  ,1) &
                  &                   - tg_gdepw_0%d_value(ji  ,jj  ,jk  ,1) &
                  &                   + tg_gdepw_0%d_value(ji+1,jj  ,jk+1,1) &
                  &                   - tg_gdepw_0%d_value(ji  ,jj  ,jk+1,1) ) &
                  &                 / ( tg_gdepw_0%d_value(ji+1,jj  ,jk  ,1)&
                  &                   + tg_gdepw_0%d_value(ji  ,jj  ,jk  ,1)&
                  &                   - tg_gdepw_0%d_value(ji+1,jj  ,jk+1,1)&
                  &                   - tg_gdepw_0%d_value(ji  ,jj  ,jk+1,1)&
                  &                   + dp_eps) )
               zr1(3) = tg_vmask%d_value(ji  ,jj  ,jk,1) &
                  &              * ABS( ( tg_gdepw_0%d_value(ji  ,jj+1,jk  ,1)&
                  &                     - tg_gdepw_0%d_value(ji  ,jj  ,jk  ,1)&
                  &                     + tg_gdepw_0%d_value(ji  ,jj+1,jk+1,1)&
                  &                     - tg_gdepw_0%d_value(ji  ,jj  ,jk+1,1) ) &
                  &                   / ( tg_gdepw_0%d_value(ji  ,jj+1,jk  ,1)&
                  &                     + tg_gdepw_0%d_value(ji  ,jj  ,jk  ,1)&
                  &                     - tg_gdepw_0%d_value(ji  ,jj+1,jk+1,1)&
                  &                     - tg_gdepw_0%d_value(ji  ,jj  ,jk+1,1)&
                  &                     + dp_eps) )
               zr1(4) = tg_vmask%d_value(ji  ,jj-1,jk,1) &
                  &              * ABS( ( tg_gdepw_0%d_value(ji  ,jj  ,jk  ,1)&
                  &                     - tg_gdepw_0%d_value(ji  ,jj-1,jk  ,1)&
                  &                     + tg_gdepw_0%d_value(ji  ,jj  ,jk+1,1)&
                  &                     - tg_gdepw_0%d_value(ji  ,jj-1,jk+1,1) ) &
                  &                   / ( tg_gdepw_0%d_value(ji  ,jj  ,jk  ,1)&
                  &                     + tg_gdepw_0%d_value(ji  ,jj-1,jk  ,1)&
                  &                     - tg_gdepw_0%d_value(ji  ,jj  ,jk+1,1)&
                  &                     - tg_gdepw_0%d_value(ji  ,jj-1,jk+1,1)&
                  &                     + dp_eps) )
               zrxmax = MAXVAL(zr1(1:4))
               tg_rx1%d_value(ji,jj,1,1) = MAX(tg_rx1%d_value(ji,jj,1,1), zrxmax)
            END DO
         END DO
      END DO

      CALL lbc_lnk( tg_rx1%d_value(:,:,1,1), 'T', td_nam%i_perio, 1._dp )

      zrxmax = MAXVAL(tg_rx1%d_value(:,:,1,1))
      CALL logger_info(' GRID ZGR SCO STIFF: maximum grid stiffness ratio: '//&
         &  TRIM(fct_str(zrxmax)) )

   END SUBROUTINE grid_zgr_sco_stiff
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE grid_zgr
