!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @file
!> @brief 
!> this program creates restart file or initial state.
!>
!> @details
!> @section sec1 method
!> variables could be 
!>  - extracted from fine grid file
!>  - interpolated from coarse grid file
!>  - interpolated from restart file
!>  - handwritten
!>
!> then they are split over new layout. 
!> @note
!>    interpolation and/or extrapolation method could be different for each variable.
!>
!> @section sec2 how to
!> USAGE: create_restart create_restart.nam [-v] [-h]<br/>
!>    - positional arguments:<br/>
!>       - create_restart.nam<br/>
!>          namelist of create_restart
!>          @note
!>             a template of the namelist could be created running (in templates directory):
!>             @code{.sh}
!>                python create_templates.py create_restart
!>             @endcode
!>
!>    - optional arguments:<br/>
!>       - -h, --help<br/>
!>          show this help message (and exit)<br/>
!>       - -v, --version<br/>
!>          show Siren's version   (and exit)
!>
!> @section sec_restart create_restart.nam
!>    create_restart.nam contains 9 namelists:<br/>
!>       - **namlog** to set logger parameters
!>       - **namcfg** to set configuration file parameters
!>       - **namsrc** to set source/coarse grid parameters
!>       - **namtgt** to set target/fine grid parameters
!>       - **namzgr** to set vertical grid parameters
!>       - **namzps** to set partial step parameters
!>       - **namvar** to set variable parameters
!>       - **namnst** to set sub domain and nesting paramters
!>       - **namout** to set output parameters
!>
!>    here after, each sub-namelist parameters is detailed.
!>    @note 
!>       default values are specified between brackets
!>
!> @subsection sublog namlog
!>    the logger sub-namelist parameters are :
!>
!>    - **cn_logfile** [@a create_restart.log]<br/>
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
!>    the coarse grid sub-namelist parameters are :
!>
!>    - **cn_coord0** [@a ]<br/> 
!>       path to the coordinate file
!>
!>    - **in_perio0** [@a ]<br/> 
!>       NEMO periodicity index<br/> 
!>       the NEMO periodicity could be choose between 0 to 6:
!>       <dl>
!>          <dt>in_perio=0</dt>
!>          <dd>standard regional model</dd>
!>          <dt>in_perio=1</dt>
!>          <dd>east-west cyclic model</dd>
!>          <dt>in_perio=2</dt>
!>          <dd>model with symmetric boundary condition across the equator</dd>
!>          <dt>in_perio=3</dt>
!>          <dd>regional model with North fold boundary and T-point pivot</dd>
!>          <dt>in_perio=4</dt>
!>          <dd>global model with a T-point pivot.<br/>
!>          example: ORCA2, ORCA025, ORCA12</dd>
!>          <dt>in_perio=5</dt>
!>          <dd>regional model with North fold boundary and F-point pivot</dd>
!>          <dt>in_perio=6</dt>
!>          <dd>global model with a F-point pivot<br/>
!>          example: ORCA05</dd>
!>          </dd>
!>       </dl>
!>       @sa For more information see @ref md_src_docsrc_6_perio
!>       and Model Boundary Condition paragraph in the 
!>       [NEMO documentation](https://forge.ipsl.jussieu.fr/nemo/chrome/site/doc/NEMO/manual/pdf/NEMO_manual.pdf)
!>
!> @subsection subtgt namtgt 
!>    the fine grid sub-namelist parameters are :
!>
!>    - **cn_coord1** [@a ]<br/> 
!>       path to coordinate file
!>
!>    - **cn_bathy1** [@a ]<br/> 
!>       path to bathymetry file
!>
!>    - **in_perio1** [@a ]<br/>
!>       NEMO periodicity index (see above)
!>    @note if the fine/target coordinates file (cn_coord1) was created by SIREN, you do
!>    not need to fill this parameter. SIREN will read it on the global attributes of
!>    the coordinates file.
!>
!> @subsection subzgr namzgr
!>    the vertical grid sub-namelist parameters are :
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
!>    - **dn_ppa2** [@a 100.760928500000]<br/>
!>       double tanh function parameter
!>
!>    - **dn_ppkth** [@a 15.351013700000]<br/>
!>       coefficient to compute vertical grid
!>
!>    - **dn_ppkth2** [@a 48.029893720000]<br/>
!>       double tanh function parameter
!>
!>    - **dn_ppacr** [@a 7.000000000000]<br/>
!>       coefficient to compute vertical grid
!>
!>    - **dn_ppacr2** [@a 13.000000000000]<br/>
!>       double tanh function parameter
!>
!>    - **dn_ppdzmin** [@a 6.]<br/>
!>       minimum vertical spacing
!>
!>    - **dn_pphmax** [@a 5750.]<br/>
!>       maximum depth
!>
!>    - **in_nlevel** [@a 75]<br/>
!>       number of vertical level
!>
!>     @note 
!>       If *dn_ppa1*, *dn_ppa0* and *dn_ppsur* are undefined,
!>       NEMO will compute them from *dn_ppdzmin, dn_pphmax, dn_ppkth, dn_ppacr*
!>
!> @subsection subzps namzps
!>    the partial step sub-namelist parameters are :
!>
!>    - **dn_e3zps_min** [@a 25.]<br/>
!>       minimum thickness of partial step level (meters)
!>    - **dn_e3zps_rat** [@a 0.2]<br/>
!>       minimum thickness ratio of partial step level
!>
!> @subsection subvar namvar 
!>    the variable sub-namelist parameters are :
!>
!>    - **cn_varfile** [@a ]<br/> 
!>       list of variable, and associated file 
!>
!>       *cn_varfile* is the path and filename of the file where find
!>       variable.
!>       @note 
!>          *cn_varfile* could be a matrix of value, if you want to handwrite
!>          variable value.<br/>
!>          the variable array of value is split into equal subdomain.<br/>
!>          each subdomain is filled with the corresponding value 
!>          of the matrix.<br/>          
!>          separators used to defined matrix are:
!>             - ',' for line
!>             - '/' for row
!>             Example:<br/>
!>                3,2,3/1,4,5  =>  @f$ \left( \begin{array}{ccc}
!>                                      3 & 2 & 3 \\
!>                                      1 & 4 & 5 \end{array} \right) @f$
!>
!>       Examples: 
!>          - 'votemper:gridT.nc', 'vozocrtx:gridU.nc'
!>          - 'votemper:10\25', 'vozocrtx:gridU.nc'<br/>
!>
!>       @note
!>          to get all variables from one file:
!>
!>       Example:
!>          - 'all:restart.dimg'
!>
!>       @note 
!>          Optionnaly, NEMO periodicity could be added following the filename.
!>          the periodicity must be separated by ';'
!>
!>       Example:
!>          - 'votemper:gridT.nc ; perio=4'
!>
!>    - **cn_varinfo** [@a ]<br/> 
!>       list of variable and extra information about request(s) to be used<br/>
!>
!>       each elements of *cn_varinfo* is a string character (separated by ',').<br/>
!>       it is composed of the variable name follow by ':', 
!>       then request(s) to be used on this variable.<br/> 
!>       request could be:
!>          - int = interpolation method
!>          - ext = extrapolation method
!>          - flt = filter method
!>          - min = minimum value
!>          - max = maximum value
!>          - unt = new units
!>          - unf = unit scale factor (linked to new units)
!>
!>             requests must be separated by ';'.<br/>
!>             order of requests does not matter.<br/>
!>
!>       informations about available method could be find in @ref interp,
!>       @ref extrap and @ref filter modules.<br/>
!>       Example: 
!>          - 'votemper: int=linear; flt=hann; ext=dist_weight',
!>            'vosaline: int=cubic'
!>
!>       @note 
!>          If you do not specify a method which is required, 
!>          default one is apply.
!>
!> @subsection subnst namnst 
!>    the nesting sub-namelist parameters are :
!>
!>    - **in_rhoi**  [@a 1]<br/> 
!>       refinement factor in i-direction
!>
!>    - **in_rhoj**  [@a 1]<br/> 
!>       refinement factor in j-direction
!>
!>    @note 
!>       coarse grid indices will be deduced from fine grid
!>       coordinate file.
!>
!> @subsection subout namout 
!>    the output sub-namelist parameter is :
!>
!>    - **cn_fileout** [@a restart.nc]<br/>
!>       output bathymetry filename
!>
!>    - **ln_extrap** [@a .FALSE.]<br/>
!>       logical to extrapolate land point or not
!>
!>    - **in_niproc** [@a 1]<br/>
!>       number of processor in i-direction
!>
!>    - **in_njproc** [@a 1]<br/>
!>       number of processor in j-direction
!>
!>    - **in_nproc** [@a 1]<br/>
!>       total number of processor to be used
!>
!>    - **cn_type** [@a ]<br/>
!>       output format ('dimg', 'cdf')
!>
!>    @note
!>       - if *in_niproc*, and *in_njproc* are provided : the program only look for land
!>         processor to be removed
!>       - if *in_nproc* is provided : the program compute each possible domain layout, 
!>         and save the one with the most land processor to be removed 
!>       - with no information about number of processors, the program
!>         assume to use only one processor
!>
!> <hr>
!> @author J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date September, 2014
!> - add header for user
!> - offset computed considering grid point
!> - add attributes in output variable
!> @date June, 2015
!> - extrapolate all land points, and add ln_extrap in namelist.
!> - allow to change unit.
!> @date September, 2015
!> - manage useless (dummy) variable, attributes, and dimension
!> @date October, 2016
!> - dimension to be used select from configuration file
!> @date January, 2019
!> - add url path to global attributes of output file(s)
!> - check name and standard name for longitude and latitude
!> @date February, 2019
!> - rename sub namelist namcrs to namsrc
!> - rename sub namelist namfin to namtgt
!> @date May, 2019
!> - create and clean file structure to avoid memory leaks
!> @date August, 2019
!> - use periodicity read from namelist, and store in multi structure
!> @date Ocober, 2019
!> - add help and version optional arguments
!>
!> @todo
!> - rewrite using meshmask instead of bathymetry and coordinates files
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
PROGRAM create_restart

   USE global                          ! global variable
   USE kind                            ! F90 kind parameter
   USE logger                          ! log file manager
   USE fct                             ! basic useful function
   USE date                            ! date manager
   USE att                             ! attribute manager
   USE dim                             ! dimension manager
   USE var                             ! variable manager
   USE file                            ! file manager
   USE multi                           ! multi file manager
   USE iom                             ! I/O manager
   USE grid                            ! grid manager
   USE vgrid                           ! vertical grid manager
   USE extrap                          ! extrapolation manager
   USE interp                          ! interpolation manager
   USE filter                          ! filter manager
   USE mpp                             ! MPP manager
   USE dom                             ! domain manager
   USE iom_mpp                         ! MPP I/O manager
   USE iom_dom                         ! DOM I/O manager

   IMPLICIT NONE

   ! parameters
   CHARACTER(LEN=lc), PARAMETER  :: cp_myname = "create_restart"

   ! local variable
   CHARACTER(LEN=lc)                                  :: cl_arg
   CHARACTER(LEN=lc)                                  :: cl_namelist
   CHARACTER(LEN=lc)                                  :: cl_date
   CHARACTER(LEN=lc)                                  :: cl_name
   CHARACTER(LEN=lc)                                  :: cl_data
   CHARACTER(LEN=lc)                                  :: cl_fileout 
   CHARACTER(LEN=lc)                                  :: cl_url
   CHARACTER(LEN=lc)                                  :: cl_errormsg

   INTEGER(i4)                                        :: il_narg
   INTEGER(i4)                                        :: il_status
   INTEGER(i4)                                        :: il_fileid
   INTEGER(i4)                                        :: il_varid
   INTEGER(i4)                                        :: il_attid
   INTEGER(i4)                                        :: il_nvar
   INTEGER(i4)                                        :: il_imin1
   INTEGER(i4)                                        :: il_imax1
   INTEGER(i4)                                        :: il_jmin1
   INTEGER(i4)                                        :: il_jmax1
   INTEGER(i4)                                        :: il_imin0
   INTEGER(i4)                                        :: il_imax0
   INTEGER(i4)                                        :: il_jmin0
   INTEGER(i4)                                        :: il_jmax0
   INTEGER(i4)                                        :: il_index
   INTEGER(i4)      , DIMENSION(ip_maxdim)            :: il_rho
   INTEGER(i4)      , DIMENSION(2,2)                  :: il_xghost
   INTEGER(i4)      , DIMENSION(2,2)                  :: il_offset
   INTEGER(i4)      , DIMENSION(2,2)                  :: il_ind

   LOGICAL                                            :: ll_exist
   LOGICAL                                            :: ll_sameGrid

   TYPE(TDOM)                                         :: tl_dom1
   TYPE(TDOM)                                         :: tl_dom0

   TYPE(TATT)                                         :: tl_att
   
   TYPE(TVAR)                                         :: tl_depth
   TYPE(TVAR)                                         :: tl_time
   TYPE(TVAR)                                         :: tl_lon
   TYPE(TVAR)                                         :: tl_lat
   TYPE(TVAR)       , DIMENSION(:)      , ALLOCATABLE :: tl_var
   TYPE(TVAR)       , DIMENSION(:)      , ALLOCATABLE :: tl_level
   
   TYPE(TDIM)       , DIMENSION(ip_maxdim)            :: tl_dim

   TYPE(TFILE)                                        :: tl_file

   TYPE(TMPP)                                         :: tl_coord0
   TYPE(TMPP)                                         :: tl_coord1
   TYPE(TMPP)                                         :: tl_bathy1
   TYPE(TMPP)                                         :: tl_mpp
   TYPE(TMPP)                                         :: tl_mppout

   TYPE(TMULTI)                                       :: tl_multi

   ! loop indices
   INTEGER(i4) :: ji
   INTEGER(i4) :: jj
   INTEGER(i4) :: jvar

   ! namelist variable
   ! namlog
   CHARACTER(LEN=lc)                       :: cn_logfile = 'create_restart.log' 
   CHARACTER(LEN=lc)                       :: cn_verbosity = 'warning' 
   INTEGER(i4)                             :: in_maxerror = 5

   ! namcfg
   CHARACTER(LEN=lc)                       :: cn_varcfg = './cfg/variable.cfg' 
   CHARACTER(LEN=lc)                       :: cn_dimcfg = './cfg/dimension.cfg'
   CHARACTER(LEN=lc)                       :: cn_dumcfg = './cfg/dummy.cfg'

   ! namsrc
   CHARACTER(LEN=lc)                       :: cn_coord0 = '' 
   INTEGER(i4)                             :: in_perio0 = -1

   ! namtgt
   CHARACTER(LEN=lc)                       :: cn_coord1 = ''
   CHARACTER(LEN=lc)                       :: cn_bathy1 = ''
   INTEGER(i4)                             :: in_perio1 = -1

   !namzgr
   REAL(dp)                                :: dn_pp_to_be_computed = 0._dp
   REAL(dp)                                :: dn_ppsur   = -3958.951371276829_dp
   REAL(dp)                                :: dn_ppa0    =   103.953009600000_dp
   REAL(dp)                                :: dn_ppa1    =     2.415951269000_dp
   REAL(dp)                                :: dn_ppa2    =   100.760928500000_dp
   REAL(dp)                                :: dn_ppkth   =    15.351013700000_dp
   REAL(dp)                                :: dn_ppkth2  =    48.029893720000_dp
   REAL(dp)                                :: dn_ppacr   =     7.000000000000_dp
   REAL(dp)                                :: dn_ppacr2  =    13.000000000000_dp
   REAL(dp)                                :: dn_ppdzmin = 6._dp
   REAL(dp)                                :: dn_pphmax  = 5750._dp
   INTEGER(i4)                             :: in_nlevel  = 75

   !namzps
   REAL(dp)                                :: dn_e3zps_min = 25._dp
   REAL(dp)                                :: dn_e3zps_rat = 0.2_dp

   ! namvar
   CHARACTER(LEN=lc), DIMENSION(ip_maxvar) :: cn_varfile = ''
   CHARACTER(LEN=lc), DIMENSION(ip_maxvar) :: cn_varinfo = ''

   ! namnst
   INTEGER(i4)                             :: in_rhoi = 1
   INTEGER(i4)                             :: in_rhoj = 1

   ! namout
   CHARACTER(LEN=lc)                       :: cn_fileout = 'restart.nc' 
   LOGICAL                                 :: ln_extrap  = .FALSE.
   INTEGER(i4)                             :: in_nproc   = 0
   INTEGER(i4)                             :: in_niproc  = 0
   INTEGER(i4)                             :: in_njproc  = 0
   CHARACTER(LEN=lc)                       :: cn_type    = ''

   !-------------------------------------------------------------------

   NAMELIST /namlog/ &  !< logger namelist
   &  cn_logfile,    &  !< log file
   &  cn_verbosity,  &  !< log verbosity
   &  in_maxerror       !< logger maximum error

   NAMELIST /namcfg/ &  !< configuration namelist
   &  cn_varcfg,     &  !< variable configuration file
   &  cn_dimcfg,     &  !< dimension configuration file
   &  cn_dumcfg         !< dummy configuration file

   NAMELIST /namsrc/ &  !< source/coarse grid namelist
   &  cn_coord0,     &  !< coordinate file
   &  in_perio0         !< periodicity index

   NAMELIST /namtgt/ &  !< target/fine grid namelist
   &  cn_coord1,     &  !< coordinate file
   &  cn_bathy1,     &  !< bathymetry file
   &  in_perio1         !< periodicity index
 
   NAMELIST /namzgr/ &
   &  dn_pp_to_be_computed, &
   &  dn_ppsur,      &
   &  dn_ppa0,       &
   &  dn_ppa1,       &
   &  dn_ppa2,       &
   &  dn_ppkth,      &
   &  dn_ppkth2,     &
   &  dn_ppacr,      &
   &  dn_ppacr2,     &
   &  dn_ppdzmin,    &
   &  dn_pphmax,     &
   &  in_nlevel         !< number of vertical level

   NAMELIST /namzps/ &
   &  dn_e3zps_min,  &
   &  dn_e3zps_rat

   NAMELIST /namvar/ &  !< variable namelist
   &  cn_varfile,    &  !< list of variable file
   &  cn_varinfo        !< list of variable and interpolation method to be used.

   NAMELIST /namnst/ &  !< nesting namelist
   &  in_rhoi,       &  !< refinement factor in i-direction
   &  in_rhoj           !< refinement factor in j-direction

   NAMELIST /namout/ &  !< output namelist
   &  cn_fileout,    &  !< fine grid bathymetry file
   &  ln_extrap,     &  !< extrapolate or not
   &  in_niproc,     &  !< i-direction number of processor
   &  in_njproc,     &  !< j-direction numebr of processor
   &  in_nproc,      &  !< number of processor to be used
   &  cn_type           !< output type format (dimg, cdf)
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

               OPEN( il_fileid, FILE=TRIM(cl_namelist),  &
               &                FORM='FORMATTED',        &
               &                ACCESS='SEQUENTIAL',     &
               &                STATUS='OLD',            &
               &                ACTION='READ',           &
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
               READ( il_fileid, NML = namtgt )
               READ( il_fileid, NML = namzgr )
               READ( il_fileid, NML = namvar )
               ! add user change in extra information
               CALL var_chg_extra(cn_varinfo)
               ! match variable with file
               tl_multi=multi_init(cn_varfile)

               READ( il_fileid, NML = namnst )
               READ( il_fileid, NML = namout )

               CLOSE( il_fileid, IOSTAT=il_status )
               CALL fct_err(il_status)
               IF( il_status /= 0 )THEN
                  CALL logger_error("CREATE RESTART: closing "//TRIM(cl_namelist))
               ENDIF

            ELSE

               WRITE(cl_errormsg,*) " ERROR : can't find "//TRIM(cl_namelist)
               CALL fct_help(cp_myname,cl_errormsg) 
               CALL EXIT(1)

            ENDIF

      END SELECT
   ENDIF

   CALL multi_print(tl_multi)

   IF( tl_multi%i_nvar <= 0 )THEN
      CALL logger_fatal("CREATE RESTART: no variable to be used."//&
      &  " check namelist.")
   ENDIF

   ! open files
   IF( cn_coord0 /= '' )THEN
      tl_file=file_init(TRIM(cn_coord0))
      tl_coord0=mpp_init( tl_file, id_perio=in_perio0)
      ! clean
      CALL file_clean(tl_file)
      CALL grid_get_info(tl_coord0)
   ELSE
      CALL logger_fatal("CREATE RESTART: no coarse grid coordinate found. "//&
      &     "check namelist")      
   ENDIF

   IF( TRIM(cn_coord1) /= '' )THEN
      tl_file=file_init(TRIM(cn_coord1))
      tl_coord1=mpp_init( tl_file, id_perio=in_perio1)
      ! clean
      CALL file_clean(tl_file)
      CALL grid_get_info(tl_coord1)
   ELSE
      CALL logger_fatal("CREATE RESTART: no fine grid coordinate found. "//&
      &     "check namelist")
   ENDIF

   IF( TRIM(cn_bathy1) /= '' )THEN
      tl_file=file_init(TRIM(cn_bathy1))
      tl_bathy1=mpp_init( tl_file, id_perio=in_perio1)
      ! clean
      CALL file_clean(tl_file)
      CALL grid_get_info(tl_bathy1)
   ELSE
      CALL logger_fatal("CREATE RESTART: no fine grid bathymetry found. "//&
      &     "check namelist")
   ENDIF

   ! check
   ! check output file do not already exist
   IF( in_nproc > 0 )THEN
      cl_fileout=file_rename(cn_fileout,1)
   ELSE
      cl_fileout=file_rename(cn_fileout)
   ENDIF
   INQUIRE(FILE=TRIM(cl_fileout), EXIST=ll_exist)
   IF( ll_exist )THEN
      CALL logger_fatal("CREATE RESTART: output file "//TRIM(cl_fileout)//&
      &  " already exist.")
   ENDIF

   ! check refinement factor
   il_rho(:)=1
   IF( in_rhoi < 1 .OR. in_rhoj < 1 )THEN
      CALL logger_error("CREATE RESTART: invalid refinement factor."//&
      &  " check namelist "//TRIM(cl_namelist))
   ELSE
      il_rho(jp_I)=in_rhoi
      il_rho(jp_J)=in_rhoj
   ENDIF

   ! check domain indices
   ! compute coarse grid indices around fine grid
   il_ind(:,:)=grid_get_coarse_index(tl_coord0, tl_coord1, &
   &                                 id_rho=il_rho(:))

   il_imin0=il_ind(1,1) ; il_imax0=il_ind(1,2)
   il_jmin0=il_ind(2,1) ; il_jmax0=il_ind(2,2)

   ! check domain validity
   CALL grid_check_dom(tl_coord0, il_imin0, il_imax0, il_jmin0, il_jmax0)

   !3-2-4 check coincidence between coarse and fine grid
   CALL grid_check_coincidence( tl_coord0, tl_coord1, &
   &                            il_imin0, il_imax0, &
   &                            il_jmin0, il_jmax0, &
   &                            il_rho(:) )

   ! fine grid ghost cell
   il_xghost(:,:)=grid_get_ghost(tl_bathy1)

   ! work on variables
   IF( .NOT. ASSOCIATED(tl_multi%t_mpp) )THEN
      CALL logger_error("CREATE RESTART: no file to work on. "//&
      &                 "check cn_varfile in namelist.")
   ELSE
      ALLOCATE( tl_var( tl_multi%i_nvar ) )

      jvar=0
      ! for each file
      DO ji=1,tl_multi%i_nmpp

         WRITE(cl_data,'(a,i2.2)') 'data-',jvar+1
         IF( .NOT. ASSOCIATED(tl_multi%t_mpp(ji)%t_proc(1)%t_var) )THEN

            CALL logger_error("CREATE RESTART: no variable to work on for "//&
            &                 "mpp "//TRIM(tl_multi%t_mpp(ji)%c_name)//&
            &                 ". check cn_varfile in namelist.")

         ELSEIF( TRIM(tl_multi%t_mpp(ji)%c_name) == TRIM(cl_data) )THEN
         !- use input matrix to fill variable

            WRITE(*,'(a)') "work on data"
            ! for each variable initialise from matrix
            DO jj=1,tl_multi%t_mpp(ji)%t_proc(1)%i_nvar

               jvar=jvar+1

               WRITE(*,'(2x,a,a)') "work on variable "//&
               &  TRIM(tl_multi%t_mpp(ji)%t_proc(1)%t_var(jj)%c_name)

               ! fill value with matrix data
               tl_var(jvar) = create_restart_matrix( &
               &  tl_multi%t_mpp(ji)%t_proc(1)%t_var(jj), tl_coord1, &
               &  in_nlevel, il_xghost(:,:) )

               ! add ghost cell
               CALL grid_add_ghost(tl_var(jvar), il_xghost(:,:))

            ENDDO
         !- end of use input matrix to fill variable
         ELSE
         !- use mpp file to fill variable

            WRITE(*,'(a)') "work on file "//TRIM(tl_multi%t_mpp(ji)%c_name)
            !
            tl_file=file_init(TRIM(tl_multi%t_mpp(ji)%t_proc(1)%c_name), &
               &              id_perio=tl_multi%t_mpp(ji)%i_perio)
            tl_mpp=mpp_init( tl_file )
            ! clean
            CALL file_clean(tl_file)
            CALL grid_get_info(tl_mpp)

            ! check vertical dimension
            IF( tl_mpp%t_dim(jp_K)%l_use .AND. &
            &   tl_mpp%t_dim(jp_K)%i_len /= in_nlevel  )THEN
               CALL logger_error("CREATE RESTART: dimension in file "//&
               &  TRIM(tl_mpp%c_name)//" not agree with namelist in_nlevel ")
            ENDIF

            ! open mpp file
            CALL iom_mpp_open(tl_mpp)

            ! get or check depth value
            CALL create_restart_check_depth( tl_mpp, tl_depth )

            ! get or check time value
            CALL create_restart_check_time( tl_mpp, tl_time )

            ! close mpp file
            CALL iom_mpp_close(tl_mpp)

            IF( ANY(tl_mpp%t_dim(1:2)%i_len /= tl_coord0%t_dim(1:2)%i_len) .OR.&
            &   ALL(il_rho(:)==1) )THEN
            !!! extract value from fine grid 

               IF( ANY( tl_mpp%t_dim(1:2)%i_len < &
               &        tl_coord1%t_dim(1:2)%i_len) )THEN
                  CALL logger_fatal("CREATE RESTART: dimensions in file "//&
                  &  TRIM(tl_mpp%c_name)//" smaller than those in fine"//&
                  &  " grid coordinates.")
               ENDIF

               ! use coord0 instead of mpp for restart file case 
               !  (without lon,lat)
               ll_sameGrid=.FALSE.
               IF( ALL(tl_mpp%t_dim(1:2)%i_len /= tl_coord0%t_dim(1:2)%i_len) &
               &   )THEN
                  ll_sameGrid=.TRUE. 
               ENDIF

               ! compute domain on fine grid
               IF( ll_sameGrid )THEN
                  il_ind(:,:)=grid_get_coarse_index(tl_mpp, tl_coord1 )
               ELSE
                  il_ind(:,:)=grid_get_coarse_index(tl_coord0, tl_coord1 )
               ENDIF

               il_imin1=il_ind(1,1) ; il_imax1=il_ind(1,2)
               il_jmin1=il_ind(2,1) ; il_jmax1=il_ind(2,2)

               !- check grid coincidence
               IF( ll_sameGrid )THEN
                  il_rho(:)=1
                  CALL grid_check_coincidence( tl_mpp, tl_coord1, &
                  &                            il_imin1, il_imax1, &
                  &                            il_jmin1, il_jmax1, &
                  &                            il_rho(:) )
               ELSE
                  CALL grid_check_coincidence( tl_coord0, tl_coord1, &
                  &                            il_imin1, il_imax1, &
                  &                            il_jmin1, il_jmax1, &
                  &                            il_rho(:) )
               ENDIF

               ! compute domain
               tl_dom1=dom_init(tl_mpp,         &
               &                il_imin1, il_imax1, &
               &                il_jmin1, il_jmax1)
               
               ! open mpp files
               CALL iom_dom_open(tl_mpp, tl_dom1)

               ! for each variable of this file
               DO jj=1,tl_multi%t_mpp(ji)%t_proc(1)%i_nvar

                  WRITE(*,'(2x,a,a)') "work on (extract) variable "//&
                  &  TRIM(tl_multi%t_mpp(ji)%t_proc(1)%t_var(jj)%c_name)

                  jvar=jvar+1
                  cl_name=tl_multi%t_mpp(ji)%t_proc(1)%t_var(jj)%c_name
                  ! read variable over domain
                  tl_var(jvar)=iom_dom_read_var(tl_mpp, TRIM(cl_name), tl_dom1)

                  ! add attribute to variable
                  tl_att=att_init('src_file',TRIM(fct_basename(tl_mpp%c_name)))
                  CALL var_move_att(tl_var(jvar), tl_att)

                  tl_att=att_init('src_i_indices',(/il_imin0, il_imax0/))
                  CALL var_move_att(tl_var(jvar), tl_att)

                  tl_att=att_init('src_j_indices',(/il_jmin0, il_jmax0/))
                  CALL var_move_att(tl_var(jvar), tl_att)

                  ! clean structure
                  CALL att_clean(tl_att)

                  ! add ghost cell
                  CALL grid_add_ghost(tl_var(jvar), tl_dom1%i_ghost(:,:))

               ENDDO

               ! close mpp file
               CALL iom_dom_close(tl_mpp)

               ! clean structure
               CALL mpp_clean(tl_mpp)
               CALL dom_clean(tl_dom1)

            ELSE
            !!! get value from coarse grid 

               ! compute domain on coarse grid
               tl_dom0=dom_init(tl_mpp,             &
               &                il_imin0, il_imax0, &
               &                il_jmin0, il_jmax0 )

               ! add extra band (if possible) to compute interpolation
               CALL dom_add_extra(tl_dom0)

               ! open mpp files
               CALL iom_dom_open(tl_mpp, tl_dom0)
               ! for each variable of this file
               DO jj=1,tl_multi%t_mpp(ji)%t_proc(1)%i_nvar

                  WRITE(*,'(2x,a,a)') "work on (interp) variable "//&
                  &  TRIM(tl_multi%t_mpp(ji)%t_proc(1)%t_var(jj)%c_name)

                  jvar=jvar+1
                  cl_name=tl_multi%t_mpp(ji)%t_proc(1)%t_var(jj)%c_name

                  ! read variable over domain
                  tl_var(jvar)=iom_dom_read_var(tl_mpp, TRIM(cl_name), tl_dom0)

                  il_offset(:,:)=grid_get_fine_offset(tl_coord0, &
                     &                                il_imin0, il_jmin0, &
                     &                                il_imax0, il_jmax0, &
                     &                                tl_coord1, &
                     &                                id_rho=il_rho(:), &
                     &                                cd_point=TRIM(tl_var(jvar)%c_point))

                  ! interpolate variable
                  CALL create_restart_interp(tl_var(jvar), & 
                     &                       il_rho(:), &
                     &                       id_offset=il_offset(:,:))

                  ! remove extraband added to domain
                  CALL dom_del_extra( tl_var(jvar), tl_dom0, il_rho(:) )

                  ! add attribute to variable
                  tl_att=att_init('src_file',TRIM(fct_basename(tl_mpp%c_name)))
                  CALL var_move_att(tl_var(jvar), tl_att)

                  tl_att=att_init('src_i-indices',(/il_imin0, il_imax0/))
                  CALL var_move_att(tl_var(jvar), tl_att)

                  tl_att=att_init('src_j-indices',(/il_jmin0, il_jmax0/))
                  CALL var_move_att(tl_var(jvar), tl_att)

                  IF( ANY(il_rho(:)/=1) )THEN
                     tl_att=att_init("refinment_factor", &
                     &               (/il_rho(jp_I),il_rho(jp_J)/))
                     CALL var_move_att(tl_var(jvar), tl_att)
                  ENDIF

                  ! clean structure
                  CALL att_clean(tl_att)

                  ! add ghost cell
                  CALL grid_add_ghost(tl_var(jvar), il_xghost(:,:))
               ENDDO

               ! close mpp file
               CALL iom_dom_close(tl_mpp)

               ! clean structure
               CALL mpp_clean(tl_mpp)
               CALL dom_clean(tl_dom0)

            ENDIF

            ! clean structure
            CALL mpp_clean(tl_mpp)
         ENDIF
      ENDDO
   ENDIF

   il_nvar=tl_multi%i_nvar

   ! clean
   CALL multi_clean(tl_multi)
   CALL mpp_clean(tl_coord0)

   IF( .NOT. ln_extrap )THEN
      ! compute level
      ALLOCATE(tl_level(ip_npoint))
      tl_level(:)=vgrid_get_level(tl_bathy1, cl_namelist )
   ENDIF

   ! clean
   CALL mpp_clean(tl_bathy1)

   ! use additional request
   DO jvar=1,il_nvar

      ! change unit and apply factor
      CALL var_chg_unit(tl_var(jvar))

      ! forced min and max value
      CALL var_limit_value(tl_var(jvar))

      ! filter
      CALL filter_fill_value(tl_var(jvar))

      IF( .NOT. ln_extrap )THEN
         ! use mask
         CALL create_restart_mask(tl_var(jvar), tl_level(:))
      ENDIF

   ENDDO

   ! create file
   IF( in_niproc == 0 .AND. &
   &   in_njproc == 0 .AND. &
   &   in_nproc == 0 )THEN
      in_niproc = 1
      in_njproc = 1
      in_nproc = 1
   ENDIF

   ! add dimension
   tl_dim(:)=var_max_dim(tl_var(:))

   DO ji=1,il_nvar

      IF( ALL(tl_var(ji)%t_dim(:)%i_len == tl_dim(:)%i_len) )THEN
         tl_mppout=mpp_init( TRIM(cn_fileout), tl_var(ji), &
         &                   in_niproc, in_njproc, in_nproc, &
         &                   cd_type=cn_type)
         EXIT
      ENDIF

   ENDDO

   DO ji=1,ip_maxdim

      IF( tl_dim(ji)%l_use )THEN
         CALL mpp_move_dim(tl_mppout, tl_dim(ji))
         SELECT CASE(TRIM(tl_dim(ji)%c_sname))
         CASE('z','t')
            DO jj=1,tl_mppout%i_nproc
               CALL file_add_dim(tl_mppout%t_proc(jj), tl_dim(ji))
            ENDDO
         END SELECT 
      ENDIF

   ENDDO

   ! add variables
   IF( ALL( tl_dim(1:2)%l_use ) )THEN

      ! open mpp files
      CALL iom_mpp_open(tl_coord1)

      ! add longitude
      il_varid=var_get_id(tl_coord1%t_proc(1)%t_var(:),'longitude')
      IF( il_varid == 0 )THEN
         il_varid=var_get_id(tl_coord1%t_proc(1)%t_var(:),'longitude_T')
      ENDIF
      tl_lon=iom_mpp_read_var(tl_coord1, il_varid)
      CALL mpp_add_var(tl_mppout, tl_lon)
      CALL var_clean(tl_lon)

      ! add latitude
      il_varid=var_get_id(tl_coord1%t_proc(1)%t_var(:),'latitude')
      IF( il_varid == 0 )THEN
         il_varid=var_get_id(tl_coord1%t_proc(1)%t_var(:),'latitude_T')
      ENDIF
      tl_lat=iom_mpp_read_var(tl_coord1, il_varid)
      CALL mpp_add_var(tl_mppout, tl_lat)
      CALL var_clean(tl_lat)

      ! close mpp files
      CALL iom_mpp_close(tl_coord1)

   ENDIF

   IF( tl_dim(3)%l_use )THEN
      IF( ASSOCIATED(tl_depth%d_value) )THEN
         ! add depth
         CALL mpp_add_var(tl_mppout, tl_depth)
      ELSE
         CALL logger_warn("CREATE RESTART: no value for depth variable.")
      ENDIF
   ENDIF
   IF( ASSOCIATED(tl_depth%d_value) ) CALL var_clean(tl_depth)

   IF( tl_dim(4)%l_use )THEN
      IF( ASSOCIATED(tl_time%d_value) )THEN
         ! add time
         CALL mpp_add_var(tl_mppout, tl_time)
      ELSE
         CALL logger_warn("CREATE RESTART: no value for time variable.")
      ENDIF
   ENDIF
   IF( ASSOCIATED(tl_time%d_value) ) CALL var_clean(tl_time)

   ! add other variable
   DO jvar=il_nvar,1,-1
      ! check if variable already add
      il_index=var_get_index(tl_mppout%t_proc(1)%t_var(:), tl_var(jvar)%c_name)
      IF( il_index == 0 )THEN
         CALL mpp_add_var(tl_mppout, tl_var(jvar))
         CALL var_clean(tl_var(jvar))
      ENDIF
   ENDDO

   ! add some attribute
   tl_att=att_init("Created_by","SIREN create_restart")
   CALL mpp_add_att(tl_mppout, tl_att)

   !add source url
   cl_url=fct_split(fct_split(cp_url,2,'$'),2,'URL:')
   tl_att=att_init("SIREN_url",cl_url)
   CALL mpp_add_att(tl_mppout, tl_att)

   ! add date of creation
   cl_date=date_print(date_now())
   tl_att=att_init("Creation_date",TRIM(cl_date))
   CALL mpp_add_att(tl_mppout, tl_att)

   ! add attribute periodicity
   il_attid=0
   IF( ASSOCIATED(tl_mppout%t_proc(1)%t_att) )THEN
      il_attid=att_get_id(tl_mppout%t_proc(1)%t_att(:),'periodicity')
   ENDIF
   IF( tl_coord1%i_perio >= 0 .AND. il_attid == 0 )THEN
      tl_att=att_init('periodicity',tl_coord1%i_perio)
      CALL mpp_add_att(tl_mppout,tl_att)
   ENDIF

   il_attid=0
   IF( ASSOCIATED(tl_mppout%t_proc(1)%t_att) )THEN
      il_attid=att_get_id(tl_mppout%t_proc(1)%t_att(:),'ew_overlap')
   ENDIF
   IF( tl_coord1%i_ew >= 0 .AND. il_attid == 0 )THEN
      tl_att=att_init('ew_overlap',tl_coord1%i_ew)
      CALL mpp_add_att(tl_mppout,tl_att)
   ENDIF

   ! print
   CALL mpp_print(tl_mppout)

   ! create file
   CALL iom_mpp_create(tl_mppout)

   ! write file
   CALL iom_mpp_write_file(tl_mppout)
   ! close file
   CALL iom_mpp_close(tl_mppout)

   ! clean
   CALL att_clean(tl_att)
   CALL var_clean(tl_var(:))
   DEALLOCATE(tl_var)
   IF( .NOT. ln_extrap )THEN
      CALL var_clean(tl_level(:))
      DEALLOCATE(tl_level)
   ENDIF

   CALL mpp_clean(tl_mppout)
   CALL mpp_clean(tl_coord1)
   CALL var_clean_extra()

   ! close log file
   CALL logger_footer()
   CALL logger_close()

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION create_restart_matrix(td_var, td_coord, id_nlevel, id_xghost) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief
   !> This function create variable, filled with matrix value
   !> 
   !> @details 
   !> A variable is create with the same name that the input variable, 
   !> and with dimension of the coordinate file.<br/> 
   !> Then the variable array of value is split into equal subdomain.
   !> Each subdomain is filled with the associated value of the matrix.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - do not use level anymore 
   !>
   !> @param[in] td_var    variable structure 
   !> @param[in] td_coord  coordinate file structure 
   !> @param[in] id_nlevel number of vertical level  
   !> @param[in] id_xghost ghost cell array
   !> @return variable structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR)                 , INTENT(IN) :: td_var
      TYPE(TMPP)                 , INTENT(IN) :: td_coord
      INTEGER(i4)                , INTENT(IN) :: id_nlevel
      INTEGER(i4), DIMENSION(:,:), INTENT(IN) :: id_xghost

      ! function
      TYPE(TVAR)                              :: tf_var

      ! local variable
      INTEGER(i4)      , DIMENSION(3)                    :: il_dim
      INTEGER(i4)      , DIMENSION(3)                    :: il_size
      INTEGER(i4)      , DIMENSION(3)                    :: il_rest

      INTEGER(i4)      , DIMENSION(:)      , ALLOCATABLE :: il_ishape
      INTEGER(i4)      , DIMENSION(:)      , ALLOCATABLE :: il_jshape
      INTEGER(i4)      , DIMENSION(:)      , ALLOCATABLE :: il_kshape

      REAL(dp)         , DIMENSION(:,:,:,:), ALLOCATABLE :: dl_value

      TYPE(TDIM)       , DIMENSION(ip_maxdim)            :: tl_dim

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      ! write value on grid
      ! get matrix dimension
      il_dim(:)=td_var%t_dim(1:3)%i_len

      ! output dimension
      tl_dim(jp_I:jp_J)=dim_copy(td_coord%t_dim(jp_I:jp_J))
      IF( id_nlevel >= 1 )THEN
         tl_dim(jp_K)=dim_init('Z',id_nlevel)
      ENDIF

      ! remove ghost cell
      tl_dim(jp_I)%i_len=tl_dim(jp_I)%i_len - SUM(id_xghost(jp_I,:))*ip_ghost
      tl_dim(jp_J)%i_len=tl_dim(jp_J)%i_len - SUM(id_xghost(jp_J,:))*ip_ghost

      ! split output domain in N subdomain depending of matrix dimension 
      il_size(:) = tl_dim(1:3)%i_len / il_dim(:)
      il_rest(:) = MOD(tl_dim(1:3)%i_len, il_dim(:))

      ALLOCATE( il_ishape(il_dim(1)+1) )
      il_ishape(:)=0
      DO ji=2,il_dim(1)+1
         il_ishape(ji)=il_ishape(ji-1)+il_size(1)
      ENDDO
      ! add rest to last cell
      il_ishape(il_dim(1)+1)=il_ishape(il_dim(1)+1)+il_rest(1)

      ALLOCATE( il_jshape(il_dim(2)+1) )
      il_jshape(:)=0
      DO jj=2,il_dim(2)+1
         il_jshape(jj)=il_jshape(jj-1)+il_size(2)
      ENDDO
      ! add rest to last cell
      il_jshape(il_dim(2)+1)=il_jshape(il_dim(2)+1)+il_rest(2)

      ALLOCATE( il_kshape(il_dim(3)+1) )
      il_kshape(:)=0
      DO jk=2,il_dim(3)+1
         il_kshape(jk)=il_kshape(jk-1)+il_size(3)
      ENDDO
      ! add rest to last cell
      il_kshape(il_dim(3)+1)=il_kshape(il_dim(3)+1)+il_rest(3)

      ! write ouput array of value 
      ALLOCATE(dl_value( tl_dim(1)%i_len, &
      &                  tl_dim(2)%i_len, &
      &                  tl_dim(3)%i_len, &
      &                  tl_dim(4)%i_len) )

      dl_value(:,:,:,:)=0

      DO jk=2,il_dim(3)+1
         DO jj=2,il_dim(2)+1
            DO ji=2,il_dim(1)+1
               
               dl_value( 1+il_ishape(ji-1):il_ishape(ji), &
               &         1+il_jshape(jj-1):il_jshape(jj), &
               &         1+il_kshape(jk-1):il_kshape(jk), &
               &         1 ) = td_var%d_value(ji-1,jj-1,jk-1,1)

            ENDDO
         ENDDO
      ENDDO

      ! keep attribute and type
      tf_var=var_copy(td_var)
      DEALLOCATE( tf_var%d_value )
      ! save new dimension
      tf_var%t_dim(:)=dim_copy(tl_dim(:))
      ! add variable value
      CALL var_add_value( tf_var, dl_value(:,:,:,:), &
      &                   id_type=td_var%i_type)

      DEALLOCATE(dl_value)

      ! clean 
      DEALLOCATE(il_ishape)
      DEALLOCATE(il_jshape)
      DEALLOCATE(il_kshape)

   END FUNCTION create_restart_matrix
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE create_restart_mask(td_var, td_mask)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine use mask to filled land point with _FillValue
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] td_mask   mask variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR)              , INTENT(INOUT) :: td_var
      TYPE(TVAR), DIMENSION(:), INTENT(IN   ) :: td_mask

      ! local variable
      INTEGER(i4), DIMENSION(:,:), ALLOCATABLE :: il_mask

      ! loop indices
      INTEGER(i4) :: jl
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      IF( ALL(td_var%t_dim(1:2)%l_use) )THEN
         IF( ANY(td_var%t_dim(1:2)%i_len /= td_mask(1)%t_dim(1:2)%i_len) )THEN
            CALL logger_error("CREATE RESTART MASK: dimension differ between"//&
            &                 " variable "//TRIM(td_var%c_name)//" ("//&
            &                 TRIM(fct_str(td_var%t_dim(1)%i_len))//","//&
            &                 TRIM(fct_str(td_var%t_dim(2)%i_len))//&
            &                 ") and level ("//&
            &                 TRIM(fct_str(td_mask(1)%t_dim(1)%i_len))//","//&
            &                 TRIM(fct_str(td_mask(1)%t_dim(2)%i_len))//")")
         ELSE
            ALLOCATE( il_mask(td_var%t_dim(1)%i_len, &
            &                 td_var%t_dim(2)%i_len) )

            SELECT CASE(TRIM(td_var%c_point))
            CASE('T')
               il_mask(:,:)=INT(td_mask(jp_T)%d_value(:,:,1,1))
            CASE('U')
               il_mask(:,:)=INT(td_mask(jp_U)%d_value(:,:,1,1))
            CASE('V')
               il_mask(:,:)=INT(td_mask(jp_V)%d_value(:,:,1,1))
            CASE('F')
               il_mask(:,:)=INT(td_mask(jp_F)%d_value(:,:,1,1))
            END SELECT

            DO jl=1,td_var%t_dim(4)%i_len
               DO jk=1,td_var%t_dim(3)%i_len
                  WHERE( il_mask(:,:) < jk )
                     td_var%d_value(:,:,jk,jl)=td_var%d_fill
                  END WHERE
               ENDDO
            ENDDO

            DEALLOCATE( il_mask )
         ENDIF
      ENDIF
   END SUBROUTINE create_restart_mask
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE create_restart_interp(td_var, id_rho, id_offset, &
         &                          id_iext, id_jext)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine interpolate variable
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - do not use level anymore (for extrapolation)
   !>
   !> @param[inout] td_var    variable structure 
   !> @param[in] id_rho       array of refinment factor
   !> @param[in] id_offset    array of offset between fine and coarse grid
   !> @param[in] id_iext      i-direction size of extra bands (default=im_minext)
   !> @param[in] id_jext      j-direction size of extra bands (default=im_minext)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) ,                 INTENT(INOUT) :: td_var
      INTEGER(i4), DIMENSION(:)  , INTENT(IN   ) :: id_rho
      INTEGER(i4), DIMENSION(:,:), INTENT(IN   ) :: id_offset
      INTEGER(i4),                 INTENT(IN   ), OPTIONAL :: id_iext
      INTEGER(i4),                 INTENT(IN   ), OPTIONAL :: id_jext

      ! local variable
      INTEGER(i4) :: il_iext
      INTEGER(i4) :: il_jext

      ! loop indices
      !----------------------------------------------------------------

      il_iext=3
      IF( PRESENT(id_iext) ) il_iext=id_iext

      il_jext=3
      IF( PRESENT(id_jext) ) il_jext=id_jext

      IF( il_iext < 2 .AND. td_var%c_interp(1) == 'cubic' )THEN
         CALL logger_warn("CREATE RESTART INTERP: at least extrapolation "//&
         &  "on two points are required with cubic interpolation ")
         il_iext=2
      ENDIF

      IF( il_jext < 2 .AND. td_var%c_interp(1) == 'cubic' )THEN
         CALL logger_warn("CREATE RESTART INTERP: at least extrapolation "//&
         &  "on two points are required with cubic interpolation ")
         il_jext=2
      ENDIF
      ! work on variable
      ! add extraband
      CALL extrap_add_extrabands(td_var, il_iext, il_jext)

      ! extrapolate variable
      CALL extrap_fill_value( td_var )

      ! interpolate variable
      CALL interp_fill_value( td_var, id_rho(:), &
      &                       id_offset=id_offset(:,:) )

      ! remove extraband
      CALL extrap_del_extrabands(td_var, il_iext*id_rho(jp_I), il_jext*id_rho(jp_J))

   END SUBROUTINE create_restart_interp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE create_restart_check_depth(td_mpp, td_depth)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine get depth variable value in an open mpp structure
   !> and check if agree with already input depth variable.
   !> 
   !> @details 
   !>
   !> @author J.Paul
   !> @date November, 2014 - Initial Version
   !>
   !> @param[in] td_mpp       mpp structure
   !> @param[inout] td_depth  depth variable structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP), INTENT(IN   ) :: td_mpp
      TYPE(TVAR), INTENT(INOUT) :: td_depth

      ! local variable
      INTEGER(i4) :: il_varid
      TYPE(TVAR)  :: tl_depth
      ! loop indices
      !----------------------------------------------------------------

      ! get or check depth value
      IF( td_mpp%t_proc(1)%i_depthid /= 0 )THEN

         il_varid=td_mpp%t_proc(1)%i_depthid
         IF( ASSOCIATED(td_depth%d_value) )THEN

            tl_depth=iom_mpp_read_var(td_mpp, il_varid)
            IF( ANY( td_depth%d_value(:,:,:,:) /= &
            &        tl_depth%d_value(:,:,:,:) ) )THEN

               CALL logger_warn("CREATE RESTART: depth value from "//&
               &  TRIM(td_mpp%c_name)//" not conform "//&
               &  " to those from former file(s).")

            ENDIF
            CALL var_clean(tl_depth)

         ELSE
            td_depth=iom_mpp_read_var(td_mpp,il_varid)
         ENDIF

      ENDIF
      
   END SUBROUTINE create_restart_check_depth
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE create_restart_check_time(td_mpp, td_time)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine get date and time in an open mpp structure
   !> and check if agree with date and time already read.
   !> 
   !> @details 
   !>
   !> @author J.Paul
   !> @date November, 2014 - Initial Version
   !>
   !> @param[in] td_mpp      mpp structure
   !> @param[inout] td_time  time variable structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP), INTENT(IN   ) :: td_mpp
      TYPE(TVAR), INTENT(INOUT) :: td_time

      ! local variable
      INTEGER(i4) :: il_varid
      TYPE(TVAR)  :: tl_time

      TYPE(TDATE) :: tl_date1
      TYPE(TDATE) :: tl_date2
      ! loop indices
      !----------------------------------------------------------------

      ! get or check depth value

      IF( td_mpp%t_proc(1)%i_timeid /= 0 )THEN

         il_varid=td_mpp%t_proc(1)%i_timeid
         IF( ASSOCIATED(td_time%d_value) )THEN

            tl_time=iom_mpp_read_var(td_mpp, il_varid)

            tl_date1=var_to_date(td_time)
            tl_date2=var_to_date(tl_time)
            IF( tl_date1 - tl_date2 /= 0 )THEN

               CALL logger_warn("CREATE BOUNDARY: date from "//&
               &  TRIM(td_mpp%c_name)//" not conform "//&
               &  " to those from former file(s).")

            ENDIF
            CALL var_clean(tl_time)

         ELSE
            td_time=iom_mpp_read_var(td_mpp,il_varid)
         ENDIF

      ENDIF
      
   END SUBROUTINE create_restart_check_time
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END PROGRAM create_restart
