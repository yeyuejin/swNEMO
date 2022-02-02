!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @file
!> @brief
!>  This program creates the NetCDF file(s) which contain(s) all the
!>  ocean domain informations.
!>  it also permits to create the domain_cfg.nc file (needed to run NEMO v4.0
!>  and upper), or the mesh_mask file(s).
!>
!> @details
!> @section sec1 method
!>  bathymetry (and optionally ice shelf draft) is read on input file.<br/>
!>  horizontal grid-point position, scale factors, and the coriolis factor 
!>  are read in coordinates file or computed.<br/> 
!>  vertical coordinate is defined, and the bathymetry recomputed to fit the
!>  vertical grid.<br/>
!>  finally the masks from the bathymetry are computed.
!>
!>  all the variables read and or computed, are writen in one to three file(s) depending on
!>  output option.
!>  @note 
!>    the file contain depends on
!>    the vertical coordinate used (z-coord, partial steps, s-coord)
!>
!> @section sec2 how to
!> USAGE: create_meshmask create_meshmask.nam [-v] [-h]<br/>
!>    - positional arguments:<br/>
!>       - create_meshmask.nam<br/>
!>          namelist of create_meshmask
!>          @note
!>             a template of the namelist could be created running (in templates directory):
!>             @code{.sh}
!>                python create_templates.py create_meshmask
!>             @endcode
!>
!>    - optional arguments:<br/>
!>       - -h, --help<br/>
!>          show this help message (and exit)<br/>
!>       - -v, --version<br/>
!>          show Siren's version   (and exit)
!>
!> @section sec_meshmask create_meshmask.nam
!>    create_meshmask.nam contains 13 sub-namelists:<br/>
!>       - **namlog** to set logger parameters
!>       - **namcfg** to set configuration file parameters
!>       - **namsrc** to set source files parameters
!>       - **namhgr** to set horizontal grid parameters
!>       - **namzgr** to set vertical grid parameters
!>       - **namdmin** to set minimum depth parameters
!>       - **namzco** to set vertical coordinate parameters
!>       - **namzps** to set partial step parameters
!>       - **namsco** to set sigma or hybrid parameters
!>       - **namlbc** to set lateral boundary condition parameters
!>       - **namwd** to set wetting and drying parameters
!>       - **namgrd** to set grid parameters
!>       - **namout** to set output parameters
!>
!>    here after, each sub-namelist parameters is detailed.
!>    @note 
!>       default values are specified between brackets
!>
!> @subsection sublog namlog
!>    the logger sub-namelist parameters are :
!>
!>    - **cn_logfile** [@a create_meshmask.log]<br/>
!>       logger filename
!>
!>    - **cn_verbosity** [@a warning]<br/>
!>       verbosity level, choose between :
!>          - trace
!>          - debug
!>          - info
!>          - warning
!>          - error
!>          - fatal
!>          - none
!>
!>    - **in_maxerror** [@a 5]<br/> 
!>       maximum number of error allowed
!>
!> @subsection subcfg namcfg
!>    the configuration sub-namelist parameters are :
!>
!>    - **cn_varcfg** [@a ./cfg/variable.cfg]<br/>
!>       path to the variable configuration file.<br/>
!>       the variable configuration file defines standard name, 
!>       default interpolation method, axis,... 
!>       to be used for some known variables.<br/> 
!>
!>    - **cn_dimcfg** [@a ./cfg/dimension.cfg]<br/> 
!>       path to the dimension configuration file.<br/> 
!>       the dimension configuration file defines dimensions allowed.<br/> 
!>
!>    - **cn_dumcfg** [@a ./cfg/dummy.cfg]<br/> 
!>       path to the useless (dummy) configuration file.<br/>
!>       the dummy configuration file defines useless 
!>       dimension or variable. these dimension(s) or variable(s) will not be
!>       processed.<br/>
!>
!> @subsection subsrc namsrc 
!>    the source grid sub-namelist parameters are :
!>
!>    - **cn_bathy** [@a ]<br/>
!>       path to the bathymetry file
!>    - **cn_varbathy** [@a ]<br/>
!>       bathymetry variable name
!>    - **cn_coord** [@a ]<br/>
!>       path to the coordinate file (in_mshhgr=0)
!>    - **cn_isfdep** [@a ]<br/>
!>       iceshelf draft  (ln_isfcav=true, see namzgr)
!>    - **cn_varisfdep** [@a isfdraft]<br/>
!>       iceshelf draft variable name (ln_isfcav=true, see namzgr)
!>    - **in_perio** [@a ]<br/>
!>       NEMO periodicity
!>    - **ln_closea**  [@a .TRUE.]<br/>
!>       logical to fill closed sea or not
!>
!> @subsection subhgr namhgr
!>    the grid sub-namelist parameters are :
!>
!>    - **in_mshhgr** [@a 0]<br/>
!>       type of horizontal mesh
!>       - 0: curvilinear coordinate on the sphere read in coordinate.nc
!>       - 1: geographical mesh on the sphere with regular grid-spacing
!>       - 2: f-plane with regular grid-spacing
!>       - 3: beta-plane with regular grid-spacing
!>       - 4: Mercator grid with T/U point at the equator
!>       - 5: beta-plane with regular grid-spacing and rotated domain (GYRE configuration)
!>    - **dn_ppglam0** [@a ]<br/>
!>       longitude of first raw and column T-point (in_mshhgr = 1 or 4)
!>    - **dn_ppgphi0** [@a ]<br/>
!>       latitude  of first raw and column T-point (in_mshhgr = 1 or 4)
!>    - **dn_ppe1_deg** [@a ]<br/>
!>       zonal      grid-spacing (degrees)         (in_mshhgr = 1,2,3 or 4)
!>    - **dn_ppe2_deg** [@a ]<br/>
!>       meridional grid-spacing (degrees)         (in_mshhgr = 1,2,3 or 4)
!>
!>
!> @subsection subzgr namzgr
!>    the vertical grid sub-namelist parameters are :
!>
!>    - **ln_zco** [@a .FALSE.]<br/>
!>       z-coordinate - full steps
!>    - **ln_zps** [@a .FALSE.]<br/>
!>       z-coordinate - partial steps
!>    - **ln_sco** [@a .FALSE.]<br/>
!>       s- or hybrid z-s-coordinate
!>    - **ln_isfcav** [@a .FALSE.]<br/>
!>       ice shelf cavities
!>    - **ln_iscpl** [@a .FALSE.]<br/>
!>       coupling with ice sheet
!>    - **ln_wd** [@a .FALSE.]<br/>
!>       Wetting/drying activation
!>    - **in_nlevel** [@a 75]<br/>
!>       number of vertical level
!>
!> @subsection subdmin namdmin
!>    the minimum depth sub-namelist parameters are :
!>
!>    - **dn_hmin** [@a ]<br/>
!>       minimum ocean depth (>0) or minimum number of ocean levels (<0)
!>    - **dn_isfhmin** [@a ]<br/>
!>       threshold to discriminate grounded ice to floating ice
!>
!> @subsection subzco namzco
!>    the vertical coordinate sub-namelist parameters are :
!>
!>    - **dn_pp_to_be_computed** [@a 0]<br/>
!>
!>    - **dn_ppsur** [@a -3958.951371276829]<br/>
!>       coefficient to compute vertical grid
!>
!>    - **dn_ppa0** [@a 103.953009600000]<br/>
!>       coefficient to compute vertical grid
!>
!>    - **dn_ppa1** [@a 2.415951269000]<br/>
!>       coefficient to compute vertical grid
!>
!>    - **dn_ppkth** [@a 15.351013700000]<br/>
!>       coefficient to compute vertical grid
!>
!>    - **dn_ppacr** [@a 7.000000000000]<br/>
!>       coefficient to compute vertical grid
!>
!>    - **dn_ppdzmin** [@a 6.]<br/>
!>       minimum vertical spacing
!>
!>    - **dn_pphmax** [@a 5750.]<br/>
!>       maximum depth
!>
!>    - @b ln_dbletanh [@a .TRUE.]<br/>
!>       use double tanh to compute vartical grid
!>
!>    - **dn_ppa2** [@a 100.760928500000]<br/>
!>       double tanh function parameter
!>
!>    - **dn_ppkth2** [@a 48.029893720000]<br/>
!>       double tanh function parameter
!>
!>    - **dn_ppacr2** [@a 13.000000000000]<br/>
!>       double tanh function parameter
!>
!>     @note 
!>       If *dn_ppa1*, *dn_ppa0* and *dn_ppsur* are undefined,
!>       NEMO will compute them from *dn_ppdzmin, dn_pphmax, dn_ppkth, dn_ppacr*
!>
!>    @warning
!>       this namelist is also needed to define partial steps, sigma or hybrid coordinate.
!>
!> @subsection subzps namzps
!>    the partial step sub-namelist parameters are :
!>
!>    - **dn_e3zps_min** [@a 25.]<br/>
!>       minimum thickness of partial step level (meters)
!>    - **dn_e3zps_rat** [@a 0.2]<br/>
!>       minimum thickness ratio of partial step level
!>
!>
!> @subsection subsco namsco
!>    the sigma or hybrid sub-namelist parameters are :
!>
!>    - **ln_s_sh94** [@a .FALSE.]<br/>
!>       use hybrid s-sig Song and Haidvogel 1994 stretching function fssig1
!>    - **ln_s_sf12** [@a .FALSE.]<br/>
!>       use hybrid s-z-sig Siddorn and Furner 2012 stretching function fgamma
!>    - **dn_sbot_min** [@a ]<br/>
!>       minimum depth of s-bottom surface (>0) (m)
!>    - **dn_sbot_max** [@a ]<br/>
!>       maximum depth of s-bottom surface (= ocean depth) (>0) (m)
!>    - **dn_hc** [@a ]<br/>
!>       Critical depth for transition from sigma to stretched coordinates
!>    <br/> <br/>
!>    Song and Haidvogel 1994 stretching additional parameters
!>    - **dn_rmax** [@a ]<br/>
!>       maximum cut-off r-value allowed (0<dn_rmax<1)
!>    - **dn_theta** [@a ]<br/>
!>       surface control parameter (0<=dn_theta<=20)
!>    - **dn_thetb** [@a ]<br/>
!>       bottom control parameter  (0<=dn_thetb<= 1)
!>    - **dn_bb** [@a ]<br/>
!>       stretching parameter ( dn_bb=0; top only, dn_bb =1; top and bottom)
!>    <br/> <br/>
!>    Siddorn and Furner stretching additional parameters
!>    - **ln_sigcrit** [@a .FALSE.]<br/>
!>       switching to sigma (T) or Z (F) at H<Hc
!>    - **dn_alpha** [@a ]<br/>
!>       stretchin parameter ( >1 surface; <1 bottom)
!>    - **dn_efold** [@a ]<br/>
!>       e-fold length scale for transition region
!>    - **dn_zs** [@a ]<br/>
!>       Surface cell depth (Zs) (m)
!>       <br/>
!>       Bottom cell (Zb) (m) = H*rn_zb_a + rn_zb_b'
!>       - **dn_zb_a** [@a ]<br/>
!>          Bathymetry multiplier for Zb
!>       - **dn_zb_b** [@a ]<br/>
!>          Offset for Zb
!>
!> @subsection sublbc namlbc
!>    the lateral boundary condition sub-namelist parameters are :
!>
!>    - **rn_shlat** [@a 2.]<br/>
!>       lateral boundary conditions at the coast (modify fmask)
!>       -     shlat = 0 : free slip
!>       - 0 < shlat < 2 : partial slip 
!>       -     shlat = 2 : no slip
!>       -     shlat > 2 : strong slip
!>
!>    for more information see Boundary Condition at the Coast 
!>    in [NEMO documentation](https://forge.ipsl.jussieu.fr/nemo/chrome/site/doc/NEMO/manual/pdf/NEMO_manual.pdf)
!>
!> @subsection subwd namwd
!>    the wetting and drying sub-namelist parameters are :
!>
!>    - **dn_wdmin1** [@a ]<br/>
!>       minimum water depth on dried cells
!>    - **dn_wdmin2** [@a ]<br/>
!>       tolerrance of minimum water depth on dried cells
!>    - **dn_wdld** [@a ]<br/>
!>       land elevation below which wetting/drying
!>
!> @subsection subgrd namgrd
!>    the grid sub-namelist parameters are :
!>
!>    - **in_cfg** [@a 0]<br/>
!>       inverse resolution of the configuration (1/4° => 4)
!>    - **ln_bench** [@a .FALSE.]<br/>
!>       GYRE (in_mshhgr = 5 ) used as Benchmark.<br/>
!>       => forced the resolution to be about 100 km
!>    - **ln_c1d** [@a .FALSE.]<br/>
!>       use configuration 1D
!>    - **ln_e3_dep** [@a .FALSE.]<br/>
!>       vertical scale factors =T: e3.=dk[depth] =F: old definition
!>
!> @subsection subout namout
!>    the output sub-namelist parameters are :
!>
!>    - **cn_domcfg** [@a domain_cfg.nc]<br/>
!>       output file name
!>    - **in_msh** [@a 0]<br/>
!>       number of output file and contain (0-9)
!>    - **in_nproc** [@a 1]<br/>
!>       number of processor to be used
!>    - **in_niproc** [@a 1]<br/>
!>       i-direction number of processor
!>    - **in_njproc** [@a 1]<br/>
!>       j-direction numebr of processor
!>
!>       - if niproc, and njproc are provided : the program only look for land
!>         processor to be removed
!>       - if nproc is provided : the program compute each possible domain layout, 
!>         and save the one with the most land processor to be removed 
!>       - with no information about number of processors, the program
!>         assume to use only one processor
!>
!>    @note
!>        - if         in_msh = 0  : write '**domain_cfg.nc**' file.
!>        - if MOD(in_msh, 3) = 1  : write '<b>mesh_mask.nc</b>' file.
!>        - if MOD(in_msh, 3) = 2  : write '<b>mesh.nc</b>' and '<b>mask.nc</b>' files.
!>        - if MOD(in_msh, 3) = 0  : write '<b>mesh_hgr.nc</b>', '<b>mesh_zgr.nc</b>' and '<b>mask.nc</b>' files.<br/>
!>       For huge size domain, use option 2 or 3 depending on your vertical coordinate.<br/>
!>
!>    @note
!>        - if 0 < in_msh <= 3: write full 3D arrays for e3[tuvw] and gdep[tuvw]
!>        - if 3 < in_msh <= 6: write full 3D arrays for e3[tuvw] and 2D arrays 
!>                            corresponding to the depth of the bottom t- and w-points
!>        - if 6 < in_msh <= 9: write 2D arrays corresponding to the depth and the
!>                            thickness (e3[tw]_ps) of the bottom points
!>
!> <hr>
!> @author J.Paul
!>
!> @date September, 2015 - Initial Version (based on domhgr.F90, domzgr.F90, domwri.F90)
!> @date October, 2016
!> - update from trunk (revision 6961): add wetting and drying, ice sheet coupling..
!> @date October, 2016
!> - dimension to be used select from configuration file
!> - do not use anymore special case for ORCA grid
!> - allow to write domain_cfg file
!> @date November, 2016
!> - choose vertical scale factors (e3.=dk[depth] or old definition)
!> @date January, 2019
!> - add url path to global attributes of output file(s)
!> @date February, 2019
!> - rename sub namelist namin to namsrc
!> @date Ocober, 2019
!> - add help and version optional arguments
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
PROGRAM create_meshmask

   USE global                          ! global variable
   USE kind                            ! F90 kind parameter
   USE phycst                          ! physical constant
   USE logger                          ! log file manager
   USE date                            ! date manager
   USE fct                             ! basic useful function
   USE att                             ! attribute manager
   USE dim                             ! dimension manager
   USE var                             ! variable manager
   USE mpp                             ! MPP manager
   USE iom_mpp                         ! I/O MPP manager
   USE lbc                             ! lateral boundary conditions
   USE grid
   USE grid_hgr
   USE grid_zgr

   IMPLICIT NONE

   ! parameters
   CHARACTER(LEN=lc), PARAMETER  :: cp_myname = "create_meshmask"

   ! local variable
   CHARACTER(LEN=lc)                              :: cl_arg
   CHARACTER(LEN=lc)                              :: cl_namelist
   CHARACTER(LEN=lc)                              :: cl_date
   CHARACTER(LEN=lc)                              :: cl_url
   CHARACTER(LEN=lc)                              :: cl_errormsg

   INTEGER(i1), DIMENSION(:)        , ALLOCATABLE :: bl_tmp

   INTEGER(i4)                                    :: il_narg
   INTEGER(i4)                                    :: il_status
   INTEGER(i4)                                    :: il_fileid
   INTEGER(i4)                                    :: il_attid
   INTEGER(i4)                                    :: il_ew
   INTEGER(i4)                                    :: jpi
   INTEGER(i4)                                    :: jpj
   INTEGER(i4)                                    :: jpk
   INTEGER(i4), DIMENSION(:)        , ALLOCATABLE :: il_tmp
   INTEGER(i4), DIMENSION(:,:)      , ALLOCATABLE :: il_mask

   LOGICAL                                        :: ll_exist
   LOGICAL                                        :: ll_domcfg

   REAL(dp)   , DIMENSION(:,:)      , ALLOCATABLE :: dl_tmp2D
   REAL(dp)   , DIMENSION(:,:,:)    , ALLOCATABLE :: dl_tmp3D

   TYPE(TATT)                                     :: tl_att
   TYPE(TATT) , DIMENSION(:)        , ALLOCATABLE :: tl_gatt

   TYPE(TDIM)                                     :: tl_dim
   
   TYPE(TVAR)                                     :: tl_bathy
   TYPE(TVAR)                                     :: tl_risfdep
   TYPE(TVAR)                                     :: tl_misf
   TYPE(TVAR)                                     :: tl_gdepu
   TYPE(TVAR)                                     :: tl_gdepv
   TYPE(TVAR)                                     :: tl_hdept
   TYPE(TVAR)                                     :: tl_hdepw
   TYPE(TVAR)                                     :: tl_scalar

   TYPE(TNAMH)                                    :: tl_namh
   TYPE(TNAMZ)                                    :: tl_namz

   TYPE(TMPP)                                     :: tl_mpp
   TYPE(TMPP) , TARGET                            :: tl_mppout0
   TYPE(TMPP) , TARGET                            :: tl_mppout1
   TYPE(TMPP) , TARGET                            :: tl_mppout2
   TYPE(TMPP) , POINTER                           :: tl_mppmsk
   TYPE(TMPP) , POINTER                           :: tl_mpphgr
   TYPE(TMPP) , POINTER                           :: tl_mppzgr

   ! parameter
   INTEGER(i4) , PARAMETER :: ip_maxatt = 40

   ! loop indices
   INTEGER(i4) :: ji
   INTEGER(i4) :: jj
   INTEGER(i4) :: jk

   INTEGER(i4) :: ik

   ! namelist variable
   ! namlog
   CHARACTER(LEN=lc) :: cn_logfile  = 'create_meshmask.log' 
   CHARACTER(LEN=lc) :: cn_verbosity= 'warning' 
   INTEGER(i4)       :: in_maxerror = 5

   ! namcfg
   CHARACTER(LEN=lc) :: cn_varcfg   = './cfg/variable.cfg' 
   CHARACTER(LEN=lc) :: cn_dimcfg   = './cfg/dimension.cfg'
   CHARACTER(LEN=lc) :: cn_dumcfg   = './cfg/dummy.cfg'

   ! namsrc
   CHARACTER(LEN=lc) :: cn_bathy    = ''
   CHARACTER(LEN=lc) :: cn_varbathy = ''
   CHARACTER(LEN=lc) :: cn_coord    = ''
   CHARACTER(LEN=lc) :: cn_isfdep   = ''
   CHARACTER(LEN=lc) :: cn_varisfdep= 'isfdraft'
   INTEGER(i4)       :: in_perio    = -1
   LOGICAL           :: ln_closea   = .TRUE.

   ! namzgr
   LOGICAL           :: ln_zco      = .FALSE.
   LOGICAL           :: ln_zps      = .FALSE.
   LOGICAL           :: ln_sco      = .FALSE.
   LOGICAL           :: ln_isfcav   = .FALSE.
   LOGICAL           :: ln_iscpl    = .FALSE.
   LOGICAL           :: ln_wd       = .FALSE.
   INTEGER(i4)       :: in_nlevel   = 75

   ! namlbc
   REAL(sp)          :: rn_shlat    = 2.

   ! namout
   CHARACTER(LEN=lc) :: cn_domcfg   = 'domain_cfg.nc'
   INTEGER(i4)       :: in_msh      = 0 
   CHARACTER(LEN=lc) :: cn_type     = 'cdf'
   INTEGER(i4)       :: in_nproc    = 0
   INTEGER(i4)       :: in_niproc   = 0
   INTEGER(i4)       :: in_njproc   = 0
   !-------------------------------------------------------------------
   NAMELIST /namlog/ &  !< logger namelist
   &  cn_logfile,    &  !< log file
   &  cn_verbosity,  &  !< log verbosity
   &  in_maxerror       !< logger maximum error

   NAMELIST /namcfg/ &  !< configuration namelist
   &  cn_varcfg,     &  !< variable configuration file
   &  cn_dimcfg,     &  !< dimension configuration file
   &  cn_dumcfg         !< dummy configuration file

   NAMELIST /namsrc/ &  !< source namelist
   &  cn_bathy,      &  !< Bathymetry file
   &  cn_varbathy,   &  !< Bathymetry variable name
   &  cn_coord,      &  !< Coordinate file (in_mshhgr=0)
   &  cn_isfdep,     &  !< Iceshelf draft  (ln_isfcav=true)
   &  cn_varisfdep,  &  !< Iceshelf draft variable name (ln_isfcav=true)
   &  in_perio,      &  !< NEMO periodicity
   &  ln_closea

   NAMELIST /namzgr/ &
   &  ln_zco,        &  !< z-coordinate
   &  ln_zps,        &  !< z-coordinate with partial steps
   &  ln_sco,        &  !< s-coordinate
   &  ln_isfcav,     &  !< presence of ISF
   &  ln_iscpl,      &  !< coupling with ice sheet
   &  ln_wd,         &  !< Wetting/drying activation
   &  in_nlevel

   NAMELIST /namlbc/  &
   &  rn_shlat          !< lateral momentum boundary condition

   NAMELIST /namout/ &  !< output namlist
   &  cn_domcfg,     &  !< output file name
   &  in_msh,        &  !< number of output file (1,2,3)
   &  in_nproc,      &  !< number of processor to be used
   &  in_niproc,     &  !< i-direction number of processor
   &  in_njproc         !< j-direction numebr of processor
   !-------------------------------------------------------------------   

   !
   ! Initialisation
   ! --------------
   !
   il_narg=COMMAND_ARGUMENT_COUNT() !f03 intrinsec

   ! Traitement des arguments fournis
   ! --------------------------------
   IF( il_narg /= 1 )THEN
      WRITE(cl_errormsg,*) ' ERROR : one argument is needed '
      CALL fct_help(cp_myname,cl_errormsg) 
      CALL EXIT(1)
   ELSE

      CALL GET_COMMAND_ARGUMENT(1,cl_arg) !f03 intrinsec
      SELECT CASE (cl_arg)
         CASE ('-v', '--version')

            CALL fct_version(cp_myname)
            CALL EXIT(0)

         CASE ('-h', '--help')

            CALL fct_help(cp_myname)
            CALL EXIT(0)

         CASE DEFAULT

            cl_namelist=cl_arg

            ! read namelist
            INQUIRE(FILE=TRIM(cl_namelist), EXIST=ll_exist)
            IF( ll_exist )THEN

               il_fileid=fct_getunit()

               OPEN( il_fileid, FILE=TRIM(cl_namelist), &
               &                FORM='FORMATTED',       &
               &                ACCESS='SEQUENTIAL',    &
               &                STATUS='OLD',           &
               &                ACTION='READ',          &
               &                IOSTAT=il_status)
               CALL fct_err(il_status)
               IF( il_status /= 0 )THEN
                  WRITE(cl_errormsg,*) " ERROR : error opening "//TRIM(cl_namelist)
                  CALL fct_help(cp_myname,cl_errormsg) 
                  CALL EXIT(1)
               ENDIF

               READ( il_fileid, NML = namlog )
 
               ! define logger file
               CALL logger_open(TRIM(cn_logfile),TRIM(cn_verbosity),in_maxerror)
               CALL logger_header()

               READ( il_fileid, NML = namcfg )
               ! get variable extra information
               CALL var_def_extra(TRIM(cn_varcfg))

               ! get dimension allowed
               CALL dim_def_extra(TRIM(cn_dimcfg))

               ! get dummy variable
               CALL var_get_dummy(TRIM(cn_dumcfg))
               ! get dummy dimension
               CALL dim_get_dummy(TRIM(cn_dumcfg))
               ! get dummy attribute
               CALL att_get_dummy(TRIM(cn_dumcfg))

               READ( il_fileid, NML = namsrc )
               READ( il_fileid, NML = namzgr )
               READ( il_fileid, NML = namlbc )

               READ( il_fileid, NML = namout  )

               CLOSE( il_fileid, IOSTAT=il_status )
               CALL fct_err(il_status)
               IF( il_status /= 0 )THEN
                  CALL logger_error("CREATE MASK: closing "//TRIM(cl_namelist))
               ENDIF

            ELSE

               WRITE(cl_errormsg,*) " ERROR : can't find "//TRIM(cl_namelist)
               CALL fct_help(cp_myname,cl_errormsg) 
               CALL EXIT(1)

            ENDIF

      END SELECT
   ENDIF

   ll_domcfg=.FALSE.
   IF( in_msh == 0 )THEN
      ll_domcfg=.TRUE.
   ENDIF

   ! open file
   IF( cn_bathy /= '' )THEN
      tl_mpp=mpp_init( file_init(TRIM(cn_bathy)), id_perio=in_perio)
      CALL grid_get_info(tl_mpp)
   ELSE
      CALL logger_fatal("CREATE MESH MASK: no input bathymetry file found. "//&
      &     "check namelist")      
   ENDIF

   ! read bathymetry
   WRITE(*,*) 'FILE TO BE USED:',TRIM(cn_bathy)
   CALL iom_mpp_open(tl_mpp)

   ! get dimension
   jpi=tl_mpp%t_dim(jp_I)%i_len
   jpj=tl_mpp%t_dim(jp_J)%i_len
   jpk=in_nlevel

   WRITE(*,*) 'DIMENSION TO BE USED :',jpi,jpj,jpk

   ! read variable
   IF( TRIM(cn_varbathy) == '' )THEN
      IF( ln_zco )THEN
         cn_varbathy='Bathy_level'
      ELSEIF( ln_zps .OR. ln_sco )THEN
         IF( ln_isfcav )THEN
            cn_varbathy='Bathymetry_isf'
         ELSE
            cn_varbathy='Bathymetry'
         ENDIF
      ENDIF
   ENDIF
   WRITE(*,*) 'VARIABLE READ : '//TRIM(cn_varbathy)
   tl_bathy=iom_mpp_read_var(tl_mpp, TRIM(cn_varbathy))
   CALL iom_mpp_close(tl_mpp)

   WHERE( tl_bathy%d_value(:,:,1,1) == tl_bathy%d_fill .OR. &
        & tl_bathy%d_value(:,:,1,1) < 0._dp )
      tl_bathy%d_value(:,:,1,1) = 0._dp
   END WHERE
 
   IF ( ln_isfcav ) THEN
      WRITE(*,*) 'ICESHELF DRAFT FILE TO BE USED:',TRIM(cn_isfdep)
      WRITE(*,*) 'ICESHELF VARIABLE READ : '//TRIM(cn_varisfdep)
      ! open Iceshelf draft
      IF( cn_isfdep /= '' )THEN
         tl_mpp=mpp_init( file_init(TRIM(cn_isfdep)), id_perio=in_perio)
         CALL grid_get_info(tl_mpp)
      ELSE
         CALL logger_fatal("CREATE MESH MASK: no input Iceshelf draft '//&
            &  'file found. check namelist")      
      ENDIF

      ! read Iceshelf draft
      CALL iom_mpp_open(tl_mpp)
      IF( ln_zps .OR. ln_sco )   THEN
         tl_risfdep=iom_mpp_read_var(tl_mpp, cn_varisfdep)
      ENDIF
      CALL iom_mpp_close(tl_mpp)
   ELSE
      ALLOCATE(dl_tmp2D(jpi,jpj))
      dl_tmp2D(:,:)=0._dp

      tl_risfdep=var_init(cn_varisfdep, dl_tmp2D(:,:), id_type=NF90_DOUBLE)

      DEALLOCATE(dl_tmp2D)
   ENDIF

   ! fill closed sea
   IF( ln_closea )THEN
      WRITE(*,*) "CLOSED SEA"
      ALLOCATE( il_mask(tl_bathy%t_dim(1)%i_len, &
      &                 tl_bathy%t_dim(2)%i_len) )

      ! split domain in N sea subdomain
      il_mask(:,:)=grid_split_domain(tl_bathy)

      !  fill smallest domain
      CALL grid_fill_small_dom( tl_bathy, il_mask(:,:) )

      DEALLOCATE( il_mask )
   ENDIF

   in_perio = tl_mpp%i_perio
   il_ew    = tl_mpp%i_ew

   ! clean
   CALL mpp_clean(tl_mpp)

   ! Horizontal mesh (dom_hgr) -------------------------------------------------
   tl_namh=grid_hgr_nam( cn_coord, in_perio, cl_namelist )

   ! init Horizontal grid global variable
   CALL grid_hgr_init(jpi,jpj,jpk,ll_domcfg)

   ! compute horizontal mesh
   WRITE(*,*) "COMPUTE HORIZONTAL MESH"
   CALL grid_hgr_fill(tl_namh,jpi,jpj,ll_domcfg)      

   ! Vertyical  mesh (dom_zgr) -------------------------------------------------
   tl_namz=grid_zgr_nam( cn_coord, in_perio, cl_namelist )

   ! init Vertical grid global variable
   CALL grid_zgr_init(jpi,jpj,jpk,ln_sco)
   IF( ln_zps    ) CALL grid_zgr_zps_init(jpi,jpj)
   IF( ln_sco    ) CALL grid_zgr_sco_init(jpi,jpj) 

   ! compute vertical  mesh
   WRITE(*,*) "COMPUTE VERTICAL MESH"
   CALL grid_zgr_fill( tl_namz,jpi,jpj,jpk, tl_bathy, tl_risfdep )

   ! compute masks
   WRITE(*,*) "COMPUTE MASK"
   CALL create_meshmask__mask(tl_namh,jpi,jpj,jpk,ll_domcfg)

   ! Maximum stiffness ratio/hydrostatic consistency
   IF( ln_sco    ) CALL grid_zgr_sco_stiff(tl_namz,jpi,jpj,jpk)
 
   ! clean
   CALL var_clean(tl_bathy)

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   ! create ouptut structure 
   IF( in_niproc == 0 .AND. &
   &   in_njproc == 0 .AND. &
   &   in_nproc  == 0 )THEN
      in_niproc = 1
      in_njproc = 1
      in_nproc = 1
   ENDIF

   WRITE(*,*) "WRITE FILE(S)"
   IF( ll_domcfg )THEN
         !                                  ! ============================
         !                                  ! create 'domain_cfg.nc' file
         !                                  ! ============================
         tl_mppout0=mpp_init( cn_domcfg, tg_tmask, &
         &                    in_niproc, in_njproc, in_nproc, &
         &                    cd_type=cn_type )

         tl_mppmsk=>tl_mppout0
         tl_mpphgr=>tl_mppout0
         tl_mppzgr=>tl_mppout0

   ELSE
      SELECT CASE ( MOD(in_msh, 3) )
         !                                  ! ============================
      CASE ( 1 )                            !  create 'mesh_mask.nc' file
         !                                  ! ============================
         tl_mppout0=mpp_init( 'mesh_mask.nc', tg_tmask, &
         &                    in_niproc, in_njproc, in_nproc, &
         &                    cd_type=cn_type )

         tl_mppmsk=>tl_mppout0
         tl_mpphgr=>tl_mppout0
         tl_mppzgr=>tl_mppout0
         
         !                                  ! ============================
      CASE ( 2 )                            !  create 'mesh.nc' and 
         !                                  !         'mask.nc' files
         !                                  ! ============================
         tl_mppout0=mpp_init( 'mask.nc', tg_tmask, &
         &                    in_niproc, in_njproc, in_nproc, &
         &                    cd_type=cn_type )
         tl_mppout1=mpp_init( 'mesh.nc', tg_tmask, &
         &                    in_niproc, in_njproc, in_nproc, &
         &                    cd_type=cn_type )

         tl_mppmsk=>tl_mppout0
         tl_mpphgr=>tl_mppout1
         tl_mppzgr=>tl_mppout1

         !                                  ! ============================
      CASE ( 0 )                            !  create 'mesh_hgr.nc'
         !                                  !         'mesh_zgr.nc' and
         !                                  !         'mask.nc'     files
         !                                  ! ============================
         tl_mppout0=mpp_init( 'mask.nc', tg_tmask, &
         &                    in_niproc, in_njproc, in_nproc, &
         &                    cd_type=cn_type )
         tl_mppout1=mpp_init( 'mesh_hgr.nc', tg_tmask, &
         &                    in_niproc, in_njproc, in_nproc, &
         &                    cd_type=cn_type )
         tl_mppout2=mpp_init( 'mesh_zgr.nc', tg_tmask, &
         &                    in_niproc, in_njproc, in_nproc, &
         &                    cd_type=cn_type )
         !

         tl_mppmsk=>tl_mppout0
         tl_mpphgr=>tl_mppout1
         tl_mppzgr=>tl_mppout2

      END SELECT
   ENDIF

   ! add variables
   IF( ll_domcfg )THEN
      ALLOCATE(il_tmp(1))
      tl_dim%l_use=.FALSE.

      il_tmp(:)=jpi
      tl_scalar=var_init('jpiglo', il_tmp(:), id_type=NF90_INT, td_dim=tl_dim)
      CALL mpp_add_var(tl_mppmsk, tl_scalar)

      il_tmp(:)=jpj
      tl_scalar=var_init('jpjglo', il_tmp(:), id_type=NF90_INT, td_dim=tl_dim)
      CALL mpp_add_var(tl_mppmsk, tl_scalar)

      il_tmp(:)=jpk
      tl_scalar=var_init('jpkglo', il_tmp(:), id_type=NF90_INT, td_dim=tl_dim)
      CALL mpp_add_var(tl_mppmsk, tl_scalar)
      
      il_tmp(:)=tl_mppout0%i_perio
      tl_scalar=var_init('jperio', il_tmp(:), id_type=NF90_INT, td_dim=tl_dim)
      CALL mpp_add_var(tl_mppmsk, tl_scalar)

      DEALLOCATE(il_tmp)
      ALLOCATE(bl_tmp(1))

      bl_tmp(:)=0
      IF( ln_zco ) bl_tmp(:)=1
      tl_scalar=var_init('ln_zco',bl_tmp(:), id_type=NF90_BYTE, td_dim=tl_dim)
      CALL mpp_add_var(tl_mppmsk, tl_scalar)

      bl_tmp(:)=0
      IF( ln_zps ) bl_tmp(:)=1
      tl_scalar=var_init('ln_zps',bl_tmp(:), id_type=NF90_BYTE, td_dim=tl_dim)
      CALL mpp_add_var(tl_mppmsk, tl_scalar)

      bl_tmp(:)=0
      IF( ln_sco ) bl_tmp(:)=1
      tl_scalar=var_init('ln_sco',bl_tmp(:), id_type=NF90_BYTE, td_dim=tl_dim)
      CALL mpp_add_var(tl_mppmsk, tl_scalar)

      bl_tmp(:)=0
      IF( ln_isfcav ) bl_tmp(:)=1
      tl_scalar=var_init('ln_isfcav',bl_tmp(:), id_type=NF90_BYTE, td_dim=tl_dim)
      CALL mpp_add_var(tl_mppmsk, tl_scalar)

      DEALLOCATE(bl_tmp)
      CALL var_clean(tl_scalar)
   ENDIF

   !!! mask (msk)
   !!!----------------------
   IF( .NOT. ll_domcfg )THEN
      ! tmask
      CALL mpp_add_var(tl_mppmsk, tg_tmask)
      CALL var_clean(tg_tmask)
      ! umask
      CALL mpp_add_var(tl_mppmsk, tg_umask)
      CALL var_clean(tg_umask)
      ! vmask
      CALL mpp_add_var(tl_mppmsk, tg_vmask)
      CALL var_clean(tg_vmask)
      ! fmask
      CALL mpp_add_var(tl_mppmsk, tg_fmask)
      CALL var_clean(tg_fmask)
   ENDIF

   !!! horizontal mesh (hgr)
   !!!----------------------

   ! latitude
   ! glamt
   CALL mpp_add_var(tl_mpphgr, tg_glamt)
   CALL var_clean(tg_glamt)
   ! glamu
   CALL mpp_add_var(tl_mpphgr, tg_glamu)
   CALL var_clean(tg_glamu)
   ! glamV
   CALL mpp_add_var(tl_mpphgr, tg_glamv)
   CALL var_clean(tg_glamv)
   ! glamf
   CALL mpp_add_var(tl_mpphgr, tg_glamf)
   CALL var_clean(tg_glamf)

   ! longitude
   ! gphit
   CALL mpp_add_var(tl_mpphgr, tg_gphit)
   CALL var_clean(tg_gphit)
   ! gphiu
   CALL mpp_add_var(tl_mpphgr, tg_gphiu)
   CALL var_clean(tg_gphiu)
   ! gphiv
   CALL mpp_add_var(tl_mpphgr, tg_gphiv)
   CALL var_clean(tg_gphiv)
   ! gphif
   CALL mpp_add_var(tl_mpphgr, tg_gphif)
   CALL var_clean(tg_gphif)

   ! e1 scale factors
   ! e1t
   CALL mpp_add_var(tl_mpphgr, tg_e1t)
   CALL var_clean(tg_e1t)
   ! e1u
   CALL mpp_add_var(tl_mpphgr, tg_e1u)
   CALL var_clean(tg_e1u)
   ! e1v
   CALL mpp_add_var(tl_mpphgr, tg_e1v)
   CALL var_clean(tg_e1v)
   ! e1f
   CALL mpp_add_var(tl_mpphgr, tg_e1f)
   CALL var_clean(tg_e1f)

   ! e2 scale factors
   ! e2t
   CALL mpp_add_var(tl_mpphgr, tg_e2t)
   CALL var_clean(tg_e2t)
   ! e2u
   CALL mpp_add_var(tl_mpphgr, tg_e2u)
   CALL var_clean(tg_e2u)
   ! e2v
   CALL mpp_add_var(tl_mpphgr, tg_e2v)
   CALL var_clean(tg_e2v)
   ! e2f
   CALL mpp_add_var(tl_mpphgr, tg_e2f)
   CALL var_clean(tg_e2f)

   ! coriolis factor
   ! ff_t
   CALL mpp_add_var(tl_mpphgr, tg_ff_t)
   CALL var_clean(tg_ff_t)
   ! ff_f
   CALL mpp_add_var(tl_mpphgr, tg_ff_f)
   CALL var_clean(tg_ff_f)

   ! angles
   IF( .NOT. ll_domcfg )THEN
      ! cost
      CALL mpp_add_var(tl_mpphgr, tg_gcost)
      CALL var_clean(tg_gcost)
      ! cosu
      CALL mpp_add_var(tl_mpphgr, tg_gcosu)
      CALL var_clean(tg_gcosu)
      ! cosv
      CALL mpp_add_var(tl_mpphgr, tg_gcosv)
      CALL var_clean(tg_gcosv)
      ! cosf
      CALL mpp_add_var(tl_mpphgr, tg_gcosf)
      CALL var_clean(tg_gcosf)
      
      ! sint
      CALL mpp_add_var(tl_mpphgr, tg_gsint)
      CALL var_clean(tg_gsint)
      ! sinu
      CALL mpp_add_var(tl_mpphgr, tg_gsinu)
      CALL var_clean(tg_gsinu)
      ! sinv
      CALL mpp_add_var(tl_mpphgr, tg_gsinv)
      CALL var_clean(tg_gsinv)
      ! sinf
      CALL mpp_add_var(tl_mpphgr, tg_gsinf)
      CALL var_clean(tg_gsinf)
   ENDIF
   
   !!! vertical mesh (zgr)
   !!!----------------------
   ! note that mbkt is set to 1 over land ==> use surface tmask
   ! 
   ! mbathy
   tg_mbathy%d_value(:,:,:,:) = tg_ssmask%d_value(:,:,:,:) * &
   &                            tg_mbkt%d_value(:,:,:,:)
   !
   IF( ll_domcfg )THEN
      tg_mbathy%c_name='bottom_level'
   ENDIF
   CALL mpp_add_var(tl_mppzgr, tg_mbathy)
   CALL var_clean(tg_mbathy)

   ! misf
   ALLOCATE(dl_tmp2D(jpi,jpj))
   dl_tmp2D(:,:)=dp_fill

   tl_misf =var_init('misf     ',dl_tmp2D(:,:), id_type=NF90_INT)

   DEALLOCATE(dl_tmp2D)
   tl_misf%d_value(:,:,1,1) = tg_ssmask%d_value(:,:,1,1) * &
   &                          tg_mikt%d_value(:,:,1,1)
   !
   IF( ll_domcfg ) tl_misf%c_name='top_level'
   CALL mpp_add_var(tl_mppzgr, tl_misf)
   CALL var_clean(tl_misf)

   IF( .NOT. ll_domcfg )THEN
      ! isfdraft
      tl_risfdep%d_value(:,:,:,:) = tl_risfdep%d_value(:,:,:,:) * &
      &                             tg_mikt%d_value(:,:,:,:)
 
      CALL mpp_add_var(tl_mppzgr, tl_risfdep)
      CALL var_clean(tl_risfdep)
   ENDIF

   IF( ln_sco ) THEN ! s-coordinate

      IF( .NOT. ll_domcfg )THEN
         ! hbatt
         CALL mpp_add_var(tl_mppzgr, tg_hbatt)
         CALL var_clean(tg_hbatt)
         ! hbatu
         CALL mpp_add_var(tl_mppzgr, tg_hbatu)
         CALL var_clean(tg_hbatu)
         ! hbatv
         CALL mpp_add_var(tl_mppzgr, tg_hbatv)
         CALL var_clean(tg_hbatv)
         ! hbatf
         CALL mpp_add_var(tl_mppzgr, tg_hbatf)
         CALL var_clean(tg_hbatf)

         ! scaling coef.
         IF( .NOT. (tl_namz%l_s_sh94 .OR. tl_namz%l_s_sf12) )THEN
            ! gsigt
            CALL mpp_add_var(tl_mppzgr, tg_gsigt)
            CALL var_clean(tg_gsigt)
            ! gsigw
            CALL mpp_add_var(tl_mppzgr, tg_gsigw)
            CALL var_clean(tg_gsigw)
            ! gsi3w
            CALL mpp_add_var(tl_mppzgr, tg_gsi3w)
            CALL var_clean(tg_gsi3w)
            ! esigt
            CALL mpp_add_var(tl_mppzgr, tg_esigt)
            CALL var_clean(tg_esigt)
            ! esigw
            CALL mpp_add_var(tl_mppzgr, tg_esigw)
            CALL var_clean(tg_esigw)
         ENDIF
      ENDIF

      ! scale factors
      ! e3t_0
      CALL mpp_add_var(tl_mppzgr, tg_e3t_0)
      CALL var_clean(tg_e3t_0)
      ! e3u_0
      CALL mpp_add_var(tl_mppzgr, tg_e3u_0)
      CALL var_clean(tg_e3u_0)
      ! e3v_0
      CALL mpp_add_var(tl_mppzgr, tg_e3v_0)
      CALL var_clean(tg_e3v_0)
      ! e3f_0
      CALL mpp_add_var(tl_mppzgr, tg_e3f_0)
      CALL var_clean(tg_e3f_0)
      ! e3w_0
      CALL mpp_add_var(tl_mppzgr, tg_e3w_0)
      CALL var_clean(tg_e3w_0)
      ! e3uw_0
      CALL mpp_add_var(tl_mppzgr, tg_e3uw_0)
      CALL var_clean(tg_e3uw_0)
      ! e3vw_0
      CALL mpp_add_var(tl_mppzgr, tg_e3vw_0)
      CALL var_clean(tg_e3vw_0)

      ! Max. grid stiffness ratio
      ! rx1
      IF( ll_domcfg ) tg_rx1%c_name='stiffness'
      CALL mpp_add_var(tl_mppzgr, tg_rx1)
      CALL var_clean(tg_rx1)

      ! stretched system
      IF( .NOT. tl_namz%l_e3_dep )THEN
         ! gdept_1d
         CALL mpp_add_var(tl_mppzgr, tg_gdept_1d)
         CALL var_clean(tg_gdept_1d)
         ! gdepw_1d
         CALL mpp_add_var(tl_mppzgr, tg_gdepw_1d)
         CALL var_clean(tg_gdepw_1d)
         
         ! gdept_0
         CALL mpp_add_var(tl_mppzgr, tg_gdept_0)      
         CALL var_clean(tg_gdept_0)
         ! gdepw_0
         CALL mpp_add_var(tl_mppzgr, tg_gdepw_0)      
         CALL var_clean(tg_gdepw_0)
      ENDIF

   ENDIF

   IF( ln_zps ) THEN ! z-coordinate - partial steps

      IF( ll_domcfg .OR. in_msh <= 6 ) THEN ! 3D vertical scale factors

         ! e3t_0
         CALL mpp_add_var(tl_mppzgr, tg_e3t_0)
         CALL var_clean(tg_e3t_0)
         ! e3u_0
         CALL mpp_add_var(tl_mppzgr, tg_e3u_0)
         CALL var_clean(tg_e3u_0)
         ! e3v_0
         CALL mpp_add_var(tl_mppzgr, tg_e3v_0)
         CALL var_clean(tg_e3v_0)
         ! e3w_0
         CALL mpp_add_var(tl_mppzgr, tg_e3w_0)
         CALL var_clean(tg_e3w_0)

      ELSE

         DO jj = 1,jpj   
            DO ji = 1,jpi
               ik=tg_mbkt%d_value(ji,jj,1,1)
               tg_e3tp%d_value(ji,jj,1,1) = tg_e3t_0%d_value(ji,jj,ik,1) * &
                  &                         tg_ssmask%d_value(ji,jj,1,1)
               tg_e3wp%d_value(ji,jj,1,1) = tg_e3w_0%d_value(ji,jj,ik,1) * &
                  &                         tg_ssmask%d_value(ji,jj,1,1)
            END DO
         END DO         
         ! e3t_ps
         CALL mpp_add_var(tl_mppzgr, tg_e3tp)
         CALL var_clean(tg_e3tp)
         ! e3w_ps
         CALL mpp_add_var(tl_mppzgr, tg_e3wp)
         CALL var_clean(tg_e3wp)

      ENDIF ! 3D vertical scale factors

      IF( ll_domcfg .OR. in_msh <= 3 ) THEN ! 3D depth
         
         IF( .NOT. tl_namz%l_e3_dep )THEN
      
            ! gdepu, gdepv
            IF( .NOT. ll_domcfg )THEN
               ALLOCATE(dl_tmp3D(jpi,jpj,jpk))
               dl_tmp3D(:,:,:)=dp_fill

               tl_gdepu=var_init('gdepu',dl_tmp3D(:,:,:), id_type=NF90_FLOAT)
               tl_gdepv=var_init('gdepv',dl_tmp3D(:,:,:), id_type=NF90_FLOAT)

               DEALLOCATE(dl_tmp3D)
               DO jk = 1,jpk   
                  DO jj = 1, jpj-1   
                     DO ji = 1, jpi-1   ! vector opt.
                        tl_gdepu%d_value(ji,jj,jk,1) = MIN( tg_gdept_0%d_value(ji  ,jj  ,jk,1) , &
                           &                                tg_gdept_0%d_value(ji+1,jj  ,jk,1) )

                        tl_gdepv%d_value(ji,jj,jk,1) = MIN( tg_gdept_0%d_value(ji  ,jj  ,jk,1) , &
                           &                                tg_gdept_0%d_value(ji  ,jj+1,jk,1) )
                     END DO   
                  END DO   
               END DO         
               CALL lbc_lnk( tl_gdepu%d_value(:,:,:,1), 'U', in_perio, 1._dp )
               CALL lbc_lnk( tl_gdepv%d_value(:,:,:,1), 'V', in_perio, 1._dp )

               ! gdepu
               CALL mpp_add_var(tl_mppzgr, tl_gdepu)
               CALL var_clean(tl_gdepu)
               ! gdepv
               CALL mpp_add_var(tl_mppzgr, tl_gdepv)
               CALL var_clean(tl_gdepv)
            ENDIF

            ! gdept_0
            CALL mpp_add_var(tl_mppzgr, tg_gdept_0)
            CALL var_clean(tg_gdept_0)

            ! gdepw_0
            CALL mpp_add_var(tl_mppzgr, tg_gdepw_0)
            CALL var_clean(tg_gdepw_0)
         ENDIF

      ELSE ! 2D bottom depth
         ALLOCATE(dl_tmp2D(jpi,jpj))
         dl_tmp2D(:,:)=dp_fill

         tl_hdept=var_init('hdept',dl_tmp2D(:,:), id_type=NF90_INT)
         tl_hdepw=var_init('hdepw',dl_tmp2D(:,:), id_type=NF90_INT)

         DEALLOCATE(dl_tmp2D)
         DO jj = 1,jpj   
            DO ji = 1,jpi
               ik=tg_mbkt%d_value(ji,jj,1,1)
               tl_hdept%d_value(ji,jj,1,1) = tg_gdept_0%d_value(ji,jj,ik  ,1) * tg_ssmask%d_value(ji,jj,1,1)
               tl_hdepw%d_value(ji,jj,1,1) = tg_gdepw_0%d_value(ji,jj,ik+1,1) * tg_ssmask%d_value(ji,jj,1,1)
            END DO
         END DO
         ! hdept
         CALL mpp_add_var(tl_mppzgr, tl_hdept)
         CALL var_clean(tl_hdept)
         ! hdepw
         CALL mpp_add_var(tl_mppzgr, tl_hdepw)
         CALL var_clean(tl_hdepw)

         ! clean
         CALL var_clean(tg_gdept_0)

      ENDIF ! 3D depth

   ENDIF

   ! scale factors
   IF( ll_domcfg )THEN
      ! e3t_1d
      CALL mpp_add_var(tl_mppzgr, tg_e3t_1d)
      CALL var_clean(tg_e3t_1d)
      ! e3w_1d
      CALL mpp_add_var(tl_mppzgr, tg_e3w_1d)
      CALL var_clean(tg_e3w_1d)
   ENDIF

   IF( ln_zps .OR. ln_zco )THEN ! z-coordinate 
      IF( .NOT. tl_namz%l_e3_dep )THEN
         ! depth
         ! gdept_1d
         CALL mpp_add_var(tl_mppzgr, tg_gdept_1d)
         CALL var_clean(tg_gdept_1d)
         ! gdepw_1d
         CALL mpp_add_var(tl_mppzgr, tg_gdepw_1d)
         CALL var_clean(tg_gdepw_1d)
      ENDIF
   ENDIF

   ! define global attributes
   ALLOCATE(tl_gatt(ip_maxatt))

   tl_gatt(:) = create_meshmask__gloatt(cn_bathy,cn_coord,cn_isfdep,tl_namh,tl_namz)


   IF( in_msh == 0 ) in_msh=1
   SELECT CASE ( MOD(in_msh, 3) )
      CASE ( 1 )
         ! add some attribute
         tl_att=att_init("Created_by","SIREN create_meshmask")
         CALL mpp_add_att(tl_mppmsk, tl_att)
         
         !add source url
         cl_url=fct_split(fct_split(cp_url,2,'$'),2,'URL:')
         tl_att=att_init("SIREN_url",cl_url)
         CALL mpp_add_att(tl_mppmsk, tl_att)

         ! add date of creation
         cl_date=date_print(date_now())
         tl_att=att_init("Creation_date",TRIM(cl_date))
         CALL mpp_add_att(tl_mppmsk, tl_att)
         
         ! add attribute periodicity
         il_attid=0
         IF( ASSOCIATED(tl_mppmsk%t_proc(1)%t_att) )THEN
            il_attid=att_get_id(tl_mppmsk%t_proc(1)%t_att(:),'periodicity')
         ENDIF
         IF( in_perio >= 0 .AND. il_attid == 0 )THEN
            tl_att=att_init('periodicity',in_perio)
            CALL mpp_add_att(tl_mppmsk,tl_att)
         ENDIF
         
         il_attid=0
         IF( ASSOCIATED(tl_mppmsk%t_proc(1)%t_att) )THEN
            il_attid=att_get_id(tl_mppmsk%t_proc(1)%t_att(:),'ew_overlap')
         ENDIF
         IF( il_ew >= 0 .AND. il_attid == 0 )THEN
            tl_att=att_init('ew_overlap',il_ew)
            CALL mpp_add_att(tl_mppmsk,tl_att)
         ENDIF
         
         ji=1
         DO WHILE( tl_gatt(ji)%c_name /= '' )
            CALL mpp_add_att(tl_mppmsk,tl_gatt(ji))
            ji=ji+1
         ENDDO
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         ! create file
         CALL iom_mpp_create(tl_mppmsk)
         
         ! write file
         CALL iom_mpp_write_file(tl_mppmsk)
         ! close file
         CALL iom_mpp_close(tl_mppmsk)
         
         ! clean
         CALL mpp_clean(tl_mppmsk)

      CASE ( 2 )
         ! add some attribute
         tl_att=att_init("Created_by","SIREN create_meshmask")
         CALL mpp_add_att(tl_mppmsk, tl_att)
         CALL mpp_add_att(tl_mpphgr, tl_att)
         
         !add source url
         cl_url=fct_split(fct_split(cp_url,2,'$'),2,'URL:')
         tl_att=att_init("SIREN_url",cl_url)
         CALL mpp_add_att(tl_mppmsk, tl_att)
         CALL mpp_add_att(tl_mpphgr, tl_att)

         ! add date of creation
         cl_date=date_print(date_now())
         tl_att=att_init("Creation_date",TRIM(cl_date))
         CALL mpp_add_att(tl_mppmsk, tl_att)
         CALL mpp_add_att(tl_mpphgr, tl_att)
         
         ! add attribute periodicity
         il_attid=0
         IF( ASSOCIATED(tl_mppmsk%t_proc(1)%t_att) )THEN
            il_attid=att_get_id(tl_mppmsk%t_proc(1)%t_att(:),'periodicity')
         ENDIF
         IF( in_perio >= 0 .AND. il_attid == 0 )THEN
            tl_att=att_init('periodicity',in_perio)
            CALL mpp_add_att(tl_mppmsk,tl_att)
            CALL mpp_add_att(tl_mpphgr,tl_att)
         ENDIF
         
         il_attid=0
         IF( ASSOCIATED(tl_mppmsk%t_proc(1)%t_att) )THEN
            il_attid=att_get_id(tl_mppmsk%t_proc(1)%t_att(:),'ew_overlap')
         ENDIF
         IF( il_ew >= 0 .AND. il_attid == 0 )THEN
            tl_att=att_init('ew_overlap',il_ew)
            CALL mpp_add_att(tl_mppmsk,tl_att)
            CALL mpp_add_att(tl_mpphgr,tl_att)
         ENDIF

         ji=1
         DO WHILE( tl_gatt(ji)%c_name /= '' )
            CALL mpp_add_att(tl_mppmsk,tl_gatt(ji))
            CALL mpp_add_att(tl_mpphgr,tl_gatt(ji))
            ji=ji+1
         ENDDO
         
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         ! create mask file
         !-----------------
         CALL iom_mpp_create(tl_mppmsk)
         
         ! write file
         CALL iom_mpp_write_file(tl_mppmsk)
         ! close file
         CALL iom_mpp_close(tl_mppmsk)
         
         ! clean
         CALL mpp_clean(tl_mppmsk)

         ! create mesh file
         !-----------------
         CALL iom_mpp_create(tl_mpphgr)
         
         ! write file
         CALL iom_mpp_write_file(tl_mpphgr)
         ! close file
         CALL iom_mpp_close(tl_mpphgr)
         
         ! clean
         CALL mpp_clean(tl_mpphgr)

      CASE(0)
         ! add some attribute
         tl_att=att_init("Created_by","SIREN create_meshmask")
         CALL mpp_add_att(tl_mppmsk, tl_att)
         CALL mpp_add_att(tl_mpphgr, tl_att)
         CALL mpp_add_att(tl_mppzgr, tl_att)
         
         !add source url
         cl_url=fct_split(fct_split(cp_url,2,'$'),2,'URL:')
         tl_att=att_init("SIREN_url",cl_url)
         CALL mpp_add_att(tl_mppmsk, tl_att)
         CALL mpp_add_att(tl_mpphgr, tl_att)
         CALL mpp_add_att(tl_mppzgr, tl_att)

         ! add date of creation
         cl_date=date_print(date_now())
         tl_att=att_init("Creation_date",TRIM(cl_date))
         CALL mpp_add_att(tl_mppmsk, tl_att)
         CALL mpp_add_att(tl_mpphgr, tl_att)
         CALL mpp_add_att(tl_mppzgr, tl_att)
         
         ! add attribute periodicity
         il_attid=0
         IF( ASSOCIATED(tl_mppmsk%t_proc(1)%t_att) )THEN
            il_attid=att_get_id(tl_mppmsk%t_proc(1)%t_att(:),'periodicity')
         ENDIF
         IF( in_perio >= 0 .AND. il_attid == 0 )THEN
            tl_att=att_init('periodicity',in_perio)
            CALL mpp_add_att(tl_mppmsk,tl_att)
            CALL mpp_add_att(tl_mpphgr,tl_att)
            CALL mpp_add_att(tl_mppzgr,tl_att)
         ENDIF
 
         il_attid=0
         IF( ASSOCIATED(tl_mppmsk%t_proc(1)%t_att) )THEN
            il_attid=att_get_id(tl_mppmsk%t_proc(1)%t_att(:),'ew_overlap')
         ENDIF
         IF( il_ew >= 0 .AND. il_attid == 0 )THEN
            tl_att=att_init('ew_overlap',il_ew)
            CALL mpp_add_att(tl_mppmsk,tl_att)
            CALL mpp_add_att(tl_mpphgr,tl_att)
            CALL mpp_add_att(tl_mppzgr,tl_att)
         ENDIF

         ji=1
         DO WHILE( tl_gatt(ji)%c_name /= '' )
            CALL mpp_add_att(tl_mppmsk,tl_gatt(ji))
            CALL mpp_add_att(tl_mpphgr,tl_gatt(ji))
            CALL mpp_add_att(tl_mppzgr,tl_gatt(ji))
            ji=ji+1
         ENDDO

         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         ! create mask file
         !-----------------
         CALL iom_mpp_create(tl_mppmsk)
 
         ! write file
         CALL iom_mpp_write_file(tl_mppmsk)
         ! close file
         CALL iom_mpp_close(tl_mppmsk)
 
         ! clean
         WRITE(*,*) "CLEAN MSK"
         CALL mpp_clean(tl_mppmsk)

         ! create mesh_hgr file
         !-----------------
         CALL iom_mpp_create(tl_mpphgr)
 
         ! write file
         CALL iom_mpp_write_file(tl_mpphgr)
         ! close file
         CALL iom_mpp_close(tl_mpphgr)
 
         ! clean
         WRITE(*,*) "CLEAN HGR"
         CALL mpp_clean(tl_mpphgr)

         ! create mesh_zgr file
         !-----------------
         WRITE(*,*) "CREATE ZGR"
         CALL iom_mpp_create(tl_mppzgr)
 
         ! write file
         WRITE(*,*) "WRITE ZGR"
         CALL iom_mpp_write_file(tl_mppzgr)
         ! close file
         WRITE(*,*) "CLOSE ZGR"
         CALL iom_mpp_close(tl_mppzgr)
 
         ! clean
         WRITE(*,*) "CLEAN ZGR"
         CALL mpp_clean(tl_mppzgr)

      CASE DEFAULT
         CALL logger_fatal("CREATE MESHMASK : something wrong with in_msh("//&
            &  TRIM(fct_str(in_msh))//"), check your namelist.")
    END SELECT

   ! clean
   WRITE(*,*) "CLEAN"
   CALL att_clean(tl_att)
   CALL att_clean(tl_gatt)

   DEALLOCATE(tl_gatt)

   CALL grid_hgr_clean(ll_domcfg)
   CALL grid_zgr_clean(ln_sco)
   IF( ln_zps    ) CALL grid_zgr_zps_clean()
   IF( ln_sco    ) CALL grid_zgr_sco_clean()
   CALL var_clean_extra()

   ! close log file
   CALL logger_footer()
   CALL logger_close()
CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE create_meshmask__mask(td_nam,jpi,jpj,jpk,ld_domcfg) 
   !-------------------------------------------------------------------
   !> @brief This subroutine compute land/ocean mask arrays at tracer points, 
   !>      horizontal velocity points (u & v), vorticity points (f) and 
   !>      barotropic stream function  points (b). 
   !>
   !> @details
   !>
   !> ** Method  :   The ocean/land mask is computed from the basin bathymetry in level (mbathy) 
   !>      which is defined or read in dommba.
   !>      mbathy equals 0 over continental T-point and the number of ocean level over the ocean.
   !>
   !>      At a given position (ji,jj,jk) the ocean/land mask is given by:
   !>      - t-point : 
   !>             - 0. IF mbathy( ji ,jj) =< 0
   !>             - 1. IF mbathy( ji ,jj) >= jk
   !>      - u-point : 
   !>             - 0. IF mbathy( ji ,jj)  or mbathy(ji+1, jj ) =< 0
   !>             - 1. IF mbathy( ji ,jj) and mbathy(ji+1, jj ) >= jk.
   !>      - v-point : 
   !>             - 0. IF mbathy( ji ,jj)  or mbathy( ji ,jj+1) =< 0
   !>             - 1. IF mbathy( ji ,jj) and mbathy( ji ,jj+1) >= jk.
   !>      - f-point : 
   !>             - 0. IF mbathy( ji ,jj)  or mbathy( ji ,jj+1) or mbathy(ji+1,jj)  or mbathy(ji+1,jj+1) =< 0
   !>             - 1. IF mbathy( ji ,jj) and mbathy( ji ,jj+1) and mbathy(ji+1,jj) and mbathy(ji+1,jj+1) >= jk.
   !>      - b-point : the same definition as for f-point of the first ocean
   !>                level (surface level) but with 0 along coastlines.
   !>      - tmask_i : interior ocean mask at t-point, i.e. excluding duplicated
   !>                rows/lines due to cyclic or North Fold boundaries as well
   !>                as MPP halos.
   !>
   !> @warning do not set the lateral friction through the value of fmask along
   !>      the coast and topography. 
   !>
   !> @note If nperio not equal to 0, the land/ocean mask arrays
   !>      are defined with the proper value at lateral domain boundaries,
   !>      but bmask. indeed, bmask defined the domain over which the
   !>      barotropic stream function is computed. this domain cannot
   !>      contain identical columns because the matrix associated with
   !>      the barotropic stream function equation is then no more inverti-
   !>      ble. therefore bmask is set to 0 along lateral domain boundaries
   !>      even IF nperio is not zero.
   !>
   !>      In case of open boundaries (lk_bdy=T):
   !>        - tmask is set to 1 on the points to be computed bay the open
   !>          boundaries routines.
   !>        - bmask is  set to 0 on the open boundaries.
   !>
   !> ** Action :   
   !>       - tmask    : land/ocean mask at t-point (=0. or 1.)
   !>       - umask    : land/ocean mask at u-point (=0. or 1.)
   !>       - vmask    : land/ocean mask at v-point (=0. or 1.)
   !>       - fmask    : land/ocean mask at f-point (=0. or 1.)
   !>       - bmask    : land/ocean mask at barotropic stream
   !>                  function point (=0. or 1.) and set to 0 along lateral boundaries
   !>       - tmask_i  : interior ocean mask
   !>
   !> @author J.Paul
   !> @date September, 2015 - rewrite from dom_msk
   !> @date October, 2016
   !> - do not use anymore special case for ORCA grid
   !>
   !> @param[in] td_nam
   !> @param[in] jpi
   !> @param[in] jpj
   !> @param[in] jpk
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TNAMH), INTENT(IN) :: td_nam
      INTEGER(i4), INTENT(IN) :: jpi
      INTEGER(i4), INTENT(IN) :: jpj
      INTEGER(i4), INTENT(IN) :: jpk
      LOGICAL    , INTENT(IN) :: ld_domcfg

      ! local variable
!      INTEGER(i4)  ::   ii0, ii1  ! local integers
!      INTEGER(i4)  ::   ij0, ij1
!      INTEGER(i4)  ::   isrow

!      INTEGER(i4), DIMENSION(:,:), ALLOCATABLE ::  imsk
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE ::  zwf

!      REAL(dp)   , DIMENSION(:)  , ALLOCATABLE :: dl_tpol
!      REAL(dp)   , DIMENSION(:)  , ALLOCATABLE :: dl_fpol

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

!      ALLOCATE( imsk(jpi,jpj) )

!      ALLOCATE( dl_tpol(jpi) )
!      ALLOCATE( dl_fpol(jpi) )

      CALL logger_info('dommsk : ocean mask ')
      CALL logger_info('~~~~~~')
      IF    (      rn_shlat == 0.               )THEN ; CALL logger_info('   ocean lateral  free-slip ')
      ELSEIF(      rn_shlat == 2.               )THEN ; CALL logger_info('   ocean lateral  no-slip ')
      ELSEIF( 0. < rn_shlat .AND. rn_shlat < 2. )THEN ; CALL logger_info('   ocean lateral  partial-slip ')
      ELSEIF( 2. < rn_shlat                     )THEN ; CALL logger_info('   ocean lateral  strong-slip ')
      ELSE ; CALL logger_info(' rn_shlat is negative = '//TRIM(fct_str(rn_shlat)))
      ENDIF

      ! 1. Ocean/land mask at t-point (computed from mbathy)
      ! -----------------------------
      ! N.B. tmask has already the right boundary conditions since mbathy is ok
      !
      tg_tmask%d_value(:,:,:,1) = 0._dp
      DO jk = 1, jpk
         DO jj = 1, jpj
            DO ji = 1, jpi
               IF( tg_mbathy%d_value(ji,jj,1,1) - REAL(jk,dp) + 0.1_dp >= 0._dp )THEN
                  tg_tmask%d_value(ji,jj,jk,1) = 1._dp
               ENDIF
            ENDDO  
         ENDDO  
      ENDDO  
      
      ! (ISF) define barotropic mask and mask the ice shelf point
      tg_ssmask%d_value(:,:,1,1)=tg_tmask%d_value(:,:,1,1) ! at this stage ice shelf is not masked
      
      DO jk = 1, jpk
         DO jj = 1, jpj
            DO ji = 1, jpi
               IF( tg_misfdep%d_value(ji,jj,1,1) - REAL(jk,dp) - 0.1_dp >= 0._dp )   THEN
                  tg_tmask%d_value(ji,jj,jk,1) = 0._dp
               END IF
            ENDDO  
         ENDDO  
      ENDDO

!      ! Interior domain mask (used for global sum)
!      ! --------------------
!      tg_tmask_i%d_value(:,:,1,1) = tg_ssmask%d_value(:,:,1,1)  ! (ISH) tmask_i = 1 even on the ice shelf
!
!      tg_tmask_h%d_value(:,:,1,1) = 1._dp          ! 0 on the halo and 1 elsewhere
!
!      tg_tmask_h%d_value( 1 , : ,1,1) = 0._dp      ! first columns
!      tg_tmask_h%d_value(jpi, : ,1,1) = 0._dp      ! last  columns
!      tg_tmask_h%d_value( : , 1 ,1,1) = 0._dp      ! first rows
!      tg_tmask_h%d_value( : ,jpj,1,1) = 0._dp      ! last  rows
!
!      ! north fold mask
!      ! ---------------
!      dl_tpol(1:jpi) = 1._dp 
!      dl_fpol(1:jpi) = 1._dp
!      IF( td_nam%i_perio == 3 .OR. td_nam%i_perio == 4 )THEN      ! T-point pivot
!         dl_tpol(jpi/2+1:jpi) = 0._dp
!         dl_fpol(   1   :jpi) = 0._dp
!         IF( mjg(nlej) == jpj ) THEN                  ! only half of the nlcj-1 row
!            DO ji = iif+1, iil-1
!               tg_tmask_h%d_value(ji,nlej-1,1,1) = tg_tmask_h%d_value(ji,nlej-1,1,1) * dl_tpol(mig(ji))
!            END DO
!         ENDIF
!      ENDIF
!
!      tg_tmask_i%d_value(:,:,1,1) = tg_tmask_i%d_value(:,:,1,1) * tg_tmask_h%d_value(:,:,1,1)
!
!      IF( td_nam%i_perio == 5 .OR. td_nam%i_perio == 6 )THEN      ! F-point pivot
!         dl_tpol(   1   :jpi) = 0._dp
!         dl_fpol(jpi/2+1:jpi) = 0._dp
!      ENDIF

      ! 2. Ocean/land mask at u-,  v-, and z-points (computed from tmask)
      ! -------------------------------------------
      DO jk = 1, jpk
         DO jj = 1, jpj-1
            DO ji = 1, jpi-1  ! vector loop
               tg_umask%d_value(ji,jj,jk,1) = tg_tmask%d_value(ji  ,jj  ,jk,1) * &
                                            & tg_tmask%d_value(ji+1,jj  ,jk,1)

               tg_vmask%d_value(ji,jj,jk,1) = tg_tmask%d_value(ji  ,jj  ,jk,1) * &
                                            & tg_tmask%d_value(ji  ,jj+1,jk,1)
            !END DO
            !DO ji = 1, jpi-1  ! NO vector opt.
               IF( .NOT. ld_domcfg )THEN
                  tg_fmask%d_value(ji,jj,jk,1) = tg_tmask%d_value(ji  ,jj  ,jk,1) * &
                                               & tg_tmask%d_value(ji+1,jj  ,jk,1) * &
                                               & tg_tmask%d_value(ji  ,jj+1,jk,1) * &
                                               & tg_tmask%d_value(ji+1,jj+1,jk,1)
               ENDIF
            ENDDO
         ENDDO
      ENDDO      

!      ! (ISF) MIN(1,SUM(umask)) is here to check if you have effectively at least 1 wet cell at u point
!      DO jj = 1, jpjm1
!         DO ji = 1, fs_jpim1   ! vector loop
!            tg_ssumask%d_value(ji,jj,1,1)  = tg_ssmask%d_value(ji  ,jj  ,1,1) * &
!                                           & tg_ssmask%d_value(ji+1,jj  ,1,1) * &
!                                           & MIN( 1._wp, SUM(tg_umask%d_value(ji,jj,:,1)) )
!            tg_ssvmask%d_value(ji,jj,1,1)  = tg_ssmask%d_value(ji  ,jj  ,1,1) * &
!                                           & tg_ssmask%d_value(ji  ,jj+1,1,1) * &
!                                           & MIN( 1._wp, SUM(tg_vmask%d_value(ji,jj,:,1)) )
!         END DO
!         DO ji = 1, jpim1      ! NO vector opt.
!            tg_ssfmask%d_value(ji,jj,1,1) =  tg_ssmask%d_value(ji  ,jj  ,1,1) * &
!                                          &  tg_ssmask%d_value(ji+1,jj  ,1,1) * &
!                                          &  tg_ssmask%d_value(ji  ,jj+1,1,1) * &
!                                          &  tg_ssmask%d_value(ji+1,jj+1,1,1) * &
!                                          &  MIN( 1._wp, SUM(tg_fmask%d_value(ji,jj,:,1)) )
!         END DO
!      END DO
      CALL lbc_lnk( tg_umask%d_value  (:,:,:,1), 'U', td_nam%i_perio, 1._dp )      ! Lateral boundary conditions
      CALL lbc_lnk( tg_vmask%d_value  (:,:,:,1), 'V', td_nam%i_perio, 1._dp )
      IF( .NOT. ld_domcfg )THEN
         CALL lbc_lnk( tg_fmask%d_value  (:,:,:,1), 'F', td_nam%i_perio, 1._dp )
      ENDIF
!      CALL lbc_lnk( tg_ssumask%d_value(:,:,:,1), 'U', td_nam%i_perio, 1._dp )      ! Lateral boundary conditions
!      CALL lbc_lnk( tg_ssvmask%d_value(:,:,:,1), 'V', td_nam%i_perio, 1._dp )
!      CALL lbc_lnk( tg_ssfmask%d_value(:,:,:,1), 'F', td_nam%i_perio, 1._dp )

      ! 3. Ocean/land mask at wu-, wv- and w points 
      !----------------------------------------------
!      tg_wmask%d_value (:,:,1,1) = tg_tmask%d_value(:,:,1,1) ! surface
!      tg_wumask%d_value(:,:,1,1) = tg_umask%d_value(:,:,1,1)
!      tg_wvmask%d_value(:,:,1,1) = tg_vmask%d_value(:,:,1,1)
!
!      DO jk=2,jpk                                            ! interior values
!         tg_wmask%d_value (:,:,jk,1) = tg_tmask%d_value(:,:,jk  ,1) * &
!                                     & tg_tmask%d_value(:,:,jk-1,1)
!         tg_wumask%d_value(:,:,jk,1) = tg_umask%d_value(:,:,jk  ,1) * &
!                                     & tg_umask%d_value(:,:,jk-1,1)   
!         tg_wvmask%d_value(:,:,jk,1) = tg_vmask%d_value(:,:,jk  ,1) * &
!                                     & tg_vmask%d_value(:,:,jk-1,1)
!      ENDDO

      ! Lateral boundary conditions on velocity (modify fmask)
      ! ---------------------------------------     
      IF( .NOT. ld_domcfg )THEN
         ALLOCATE( zwf(jpi,jpj) )
         DO jk = 1, jpk
            zwf(:,:) = tg_fmask%d_value(:,:,jk,1)         
            DO jj = 2, jpj-1
               DO ji = 2, jpi-1   ! vector opt.
                  IF( tg_fmask%d_value(ji,jj,jk,1) == 0._dp )THEN
                     tg_fmask%d_value(ji,jj,jk,1) = rn_shlat * &
                                                  & MIN(1._dp , MAX(zwf(ji+1,jj), zwf(ji,jj+1), &
                                                  &                 zwf(ji-1,jj), zwf(ji,jj-1)) )
                  ENDIF
               END DO
            END DO
            DO jj = 2, jpj-1
               IF( tg_fmask%d_value(1,jj,jk,1) == 0._dp )THEN
                  tg_fmask%d_value(1  ,jj,jk,1) = rn_shlat * &
                                                & MIN(1._dp, MAX(zwf(2,jj), zwf(1,jj+1), zwf(1,jj-1)))
               ENDIF
               IF( tg_fmask%d_value(jpi,jj,jk,1) == 0._dp )THEN
                  tg_fmask%d_value(jpi,jj,jk,1) = rn_shlat * &
                                                & MIN(1._wp, MAX(zwf(jpi,jj+1), zwf(jpi-1,jj), zwf(jpi,jj-1)))
               ENDIF
            END DO         
            DO ji = 2, jpi-1
               IF( tg_fmask%d_value(ji,1,jk,1) == 0._dp )THEN
                  tg_fmask%d_value(ji, 1 ,jk,1) = rn_shlat * &
                                                & MIN(1._dp, MAX(zwf(ji+1,1), zwf(ji,2), zwf(ji-1,1)))
               ENDIF
               IF( tg_fmask%d_value(ji,jpj,jk,1) == 0._dp )THEN
                  tg_fmask%d_value(ji,jpj,jk,1) = rn_shlat * &
                                                & MIN(1._dp, MAX(zwf(ji+1,jpj), zwf(ji-1,jpj), zwf(ji,jpj-1)))
               ENDIF
            END DO
         END DO
         DEALLOCATE( zwf )

!      IF( td_nam%c_cfg == "orca" .AND. td_nam%i_cfg == 2 )THEN   ! ORCA_R2 configuration
!         !                                                 ! Increased lateral friction near of some straits
!         IF( td_nam%i_cla == 0 ) THEN
!            ! Gibraltar strait  : partial slip (fmask=0.5)
!            ij0 = 101   ;   ij1 = 101
!            ii0 = 139   ;   ii1 = 140   
!            tg_fmask%d_value(ii0:ii1,ij0:ij1,1:jpk,1) =  0.5_dp
!
!            ij0 = 102   ;   ij1 = 102
!            ii0 = 139   ;   ii1 = 140
!            tg_fmask%d_value(ii0:ii1,ij0:ij1,1:jpk,1) =  0.5_dp
!            !
!            !Bab el Mandeb : partial slip (fmask=1)
!            ij0 =  87   ;   ij1 =  88
!            ii0 = 160   ;   ii1 = 160
!            tg_fmask%d_value(ii0:ii1,ij0:ij1,1:jpk,1) =  1._dp
!
!            ij0 =  88   ;   ij1 =  88
!            ii0 = 159   ;   ii1 = 159
!            tg_fmask%d_value(ii0:ii1,ij0:ij1,1:jpk,1) =  1._dp
!            !
!         ENDIF
!      ENDIF
!      !
!      IF( td_nam%c_cfg == "orca" .AND. td_nam%i_cfg == 1 )THEN   ! ORCA R1 configuration
!         ! Increased lateral friction near of some straits
!         ! This dirty section will be suppressed by simplification process:
!         ! all this will come back in input files
!         ! Currently these hard-wired indices relate to configuration with
!         ! extend grid (jpjglo=332)
!         !
!         isrow = 332 - jpj
!         ! Gibraltar Strait
!         ii0 = 282           ;   ii1 = 283 
!         ij0 = 201 + isrow   ;   ij1 = 241 - isrow
!         tg_fmask%d_value(ii0:ii1,ij0:ij1,1:jpk,1) = 2._dp  
!
!         ! Bhosporus Strait
!         ii0 = 314           ;   ii1 = 315
!         ij0 = 208 + isrow   ;   ij1 = 248 - isrow 
!         tg_fmask%d_value(ii0:ii1,ij0:ij1,1:jpk,1 ) = 2._dp  
!
!         ! Makassar Strait (Top) 
!         ii0 =  48           ;   ii1 =  48
!         ij0 = 149 + isrow   ;   ij1 = 190 - isrow
!         tg_fmask%d_value(ii0:ii1,ij0:ij1,1:jpk,1 ) = 3._dp  
!
!         ! Lombok Strait
!         ii0 =  44           ;   ii1 =  44
!         ij0 = 124 + isrow   ;   ij1 = 165 - isrow
!         tg_fmask%d_value(ii0:ii1,ij0:ij1,1:jpk,1 ) = 2._dp  
!
!         ! Ombai Strait
!         ii0 =  53           ;   ii1 =  53
!         ij0 = 124 + isrow   ;   ij1 = 165 - isrow
!         tg_fmask%d_value(ii0:ii1,ij0:ij1,1:jpk,1 ) = 2._dp  
!
!         ! Timor Passage
!         ii0 =  56           ;   ii1 =  56
!         ij0 = 124 + isrow   ;   ij1 = 165 - isrow
!         tg_fmask%d_value(ii0:ii1,ij0:ij1,1:jpk,1 ) = 2._dp  
!
!         ! West Halmahera Strait
!         ii0 =  58           ;   ii1 =  58
!         ij0 = 141 + isrow   ;   ij1 = 182 - isrow
!         tg_fmask%d_value(ii0:ii1,ij0:ij1,1:jpk,1 ) = 3._dp  
!
!         ! East Halmahera Strait
!         ii0 =  55           ;   ii1 =  55
!         ij0 = 141 + isrow   ;   ij1 = 182 - isrow
!         tg_fmask%d_value(ii0:ii1,ij0:ij1,1:jpk,1 ) = 3._dp  
!         !
!      ENDIF
      !
         CALL lbc_lnk( tg_fmask%d_value(:,:,:,1), 'F', td_nam%i_perio, 1._dp )      ! Lateral boundary conditions on fmask
      ENDIF

!      DEALLOCATE( imsk )

!      DEALLOCATE( dl_tpol )
!      DEALLOCATE( dl_fpol )
      
   END SUBROUTINE create_meshmask__mask
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION create_meshmask__gloatt(cd_bathy,cd_coord,cd_isfdep,td_namh,td_namz) &
         & RESULT(tf_att)
   !-------------------------------------------------------------------
   !> @brief 
   !> this function create array of global attributes.
   !>
   !> @author J.Paul
   !> @date October, 2016 - initial release
   !>
   !> @param[in] cd_bathy
   !> @param[in] cd_coord
   !> @param[in] cd_isfdep
   !> @param[in] td_namh
   !> @param[in] td_namz
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=*), INTENT(IN   )  :: cd_bathy 
      CHARACTER(LEN=*), INTENT(IN   )  :: cd_coord 
      CHARACTER(LEN=*), INTENT(IN   )  :: cd_isfdep 
      TYPE(TNAMH)     , INTENT(IN   )  :: td_namh 
      TYPE(TNAMZ)     , INTENT(IN   )  :: td_namz 

      ! function
      TYPE(TATT), DIMENSION(ip_maxatt) :: tf_att

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ji=1    ; tf_att(ji)=att_init("src_bathy",TRIM(cd_bathy))
      ! horizontal grid
      ji=ji+1 ; tf_att(ji)=att_init("in_mshhgr",td_namh%i_mshhgr)
      SELECT CASE(td_namh%i_mshhgr)
         CASE(0)
            ji=ji+1 ; tf_att(ji)=att_init("src_coord",TRIM(cd_coord))
         CASE(1,4)
            ji=ji+1 ; tf_att(ji)=att_init("ppglam0",td_namh%d_ppglam0)
            ji=ji+1 ; tf_att(ji)=att_init("ppgphi0",td_namh%d_ppgphi0)
         CASE(2,3)
            ji=ji+1 ; tf_att(ji)=att_init("ppglam0",td_namh%d_ppglam0)
            ji=ji+1 ; tf_att(ji)=att_init("ppgphi0",td_namh%d_ppgphi0)
            ji=ji+1 ; tf_att(ji)=att_init("ppe1_deg",td_namh%d_ppe1_deg)
            ji=ji+1 ; tf_att(ji)=att_init("ppe2_deg",td_namh%d_ppe2_deg)
      END SELECT

      IF( td_namz%l_isfcav )THEN
         ji=ji+1 ; tf_att(ji)=att_init("ice_shelf_cavities","activated")
         ji=ji+1 ; tf_att(ji)=att_init("src_isfdep",TRIM(cd_isfdep))
      ENDIF
      IF( td_namz%l_iscpl )THEN
         ji=ji+1 ; tf_att(ji)=att_init("ice_sheet_coupling","activated")
      ENDIF

      ! vertical grid
      IF( td_namz%l_zco )THEN
         ji=ji+1 ; tf_att(ji)=att_init("z_coord","full steps")
      ENDIF
      IF( td_namz%l_zps )THEN
         ji=ji+1 ; tf_att(ji)=att_init("z_coord","partial steps")
      ENDIF
      IF( td_namz%l_sco )THEN
         IF( td_namz%l_s_sh94 )THEN
            ji=ji+1 ; tf_att(ji)=att_init("z_coord","hybrid Song and Haidvogel 1994")
         ELSEIF( td_namz%l_s_sf12 )THEN
            ji=ji+1 ; tf_att(ji)=att_init("z_coord","hybrid Siddorn and Furner 2012")
         ELSE
            ji=ji+1 ; tf_att(ji)=att_init("z_coord","sigma")
         ENDIF
      ENDIF
      ji=ji+1 ; tf_att(ji)=att_init("hmin",td_namz%d_hmin)
      IF( td_namz%l_isfcav ) ji=ji+1 ; tf_att(ji)=att_init("isfhmin",td_namz%d_isfhmin)

      ! zco
      IF( td_namz%d_ppsur /= NF90_FILL_DOUBLE )THEN
         ji=ji+1 ; tf_att(ji)=att_init("ppsur",td_namz%d_ppsur)
      ELSE
         ji=ji+1 ; tf_att(ji)=att_init("ppsur","to_be_computed")
      ENDIF
      IF( td_namz%d_ppa0 /= NF90_FILL_DOUBLE )THEN
         ji=ji+1 ; tf_att(ji)=att_init("ppa0",td_namz%d_ppa0)
      ELSE
         ji=ji+1 ; tf_att(ji)=att_init("ppa0","to_be_computed")
      ENDIF
      IF( td_namz%d_ppa1 /= NF90_FILL_DOUBLE )THEN
         ji=ji+1 ; tf_att(ji)=att_init("ppa1",td_namz%d_ppa1)
      ELSE
         ji=ji+1 ; tf_att(ji)=att_init("ppa1","to_be_computed")
      ENDIF

      ji=ji+1 ; tf_att(ji)=att_init("ppkth",td_namz%d_ppkth)
      ji=ji+1 ; tf_att(ji)=att_init("ppacr",td_namz%d_ppacr)
      ji=ji+1 ; tf_att(ji)=att_init("ppdzmin",td_namz%d_ppdzmin)
      ji=ji+1 ; tf_att(ji)=att_init("pphmax",td_namz%d_pphmax)

      IF( td_namz%l_dbletanh )THEN
         ji=ji+1 ; tf_att(ji)=att_init("ppa2",td_namz%d_ppa2)
         ji=ji+1 ; tf_att(ji)=att_init("ppkth2",td_namz%d_ppkth2)
         ji=ji+1 ; tf_att(ji)=att_init("ppacr2",td_namz%d_ppacr2)
      ENDIF

      IF( td_namz%l_zps )THEN
         ji=ji+1 ; tf_att(ji)=att_init("e3zps_min",td_namz%d_e3zps_min)
         ji=ji+1 ; tf_att(ji)=att_init("e3zps_rat",td_namz%d_e3zps_rat)
      ENDIF

      IF( td_namz%l_sco )THEN
         ji=ji+1 ; tf_att(ji)=att_init("sbot_min",td_namz%d_sbot_min)
         ji=ji+1 ; tf_att(ji)=att_init("sbot_max",td_namz%d_sbot_max)
         ji=ji+1 ; tf_att(ji)=att_init("hc",td_namz%d_hc)
         IF( td_namz%l_s_sh94 )THEN
            ji=ji+1 ; tf_att(ji)=att_init("rmax",td_namz%d_rmax)
            ji=ji+1 ; tf_att(ji)=att_init("theta",td_namz%d_theta)
            ji=ji+1 ; tf_att(ji)=att_init("thetb",td_namz%d_thetb)
            ji=ji+1 ; tf_att(ji)=att_init("bb",td_namz%d_bb)
         ELSEIF( td_namz%l_s_sf12 )THEN
            IF( td_namz%l_sigcrit ) ji=ji+1 ; tf_att(ji)=att_init("sigma_below_critical_depth","activated")
            ji=ji+1 ; tf_att(ji)=att_init("alpha",td_namz%d_alpha)
            ji=ji+1 ; tf_att(ji)=att_init("efold",td_namz%d_efold)
            ji=ji+1 ; tf_att(ji)=att_init("zs",td_namz%d_zs)
            ji=ji+1 ; tf_att(ji)=att_init("zb_a",td_namz%d_zb_a)
            ji=ji+1 ; tf_att(ji)=att_init("zb_b",td_namz%d_zb_b)
         ENDIF
      ENDIF

      IF( td_namz%l_wd )THEN
         ji=ji+1 ; tf_att(ji)=att_init("wetting_drying","activated")
      ENDIF

      IF( td_namz%l_wd )THEN
         ji=ji+1 ; tf_att(ji)=att_init("wdmin1",td_namz%d_wdmin1)
         ji=ji+1 ; tf_att(ji)=att_init("wdmin2",td_namz%d_wdmin2)
         ji=ji+1 ; tf_att(ji)=att_init("wdld",td_namz%d_wdld)
      ENDIF
 
   END FUNCTION create_meshmask__gloatt
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END PROGRAM
