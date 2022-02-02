!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @file
!> @brief 
!> this program creates fine grid coordinate file.
!>
!> @details
!> @section sec1 method
!> variables from the input coordinates coarse/source grid file, are extracted
!> and interpolated to create a fine/taget grid coordinates file.<br/>
!> @note 
!>    interpolation method could be different for each variable.
!>
!> \image html  header_coord_40.png 
!> <center> \image latex header_coord_40.png 
!> </center>
!>
!> @section sec2 how to
!> USAGE: create_coord create_coord.nam [-v] [-h]<br/>
!>    - positional arguments:<br/>
!>       - create_coord.nam<br/>
!>          namelist of create_coord
!>          @note
!>             a template of the namelist could be created running (in templates directory):
!>             @code{.sh}
!>                python create_templates.py create_coord
!>             @endcode
!>
!>    - optional arguments:<br/>
!>       - -h, --help<br/>
!>          show this help message (and exit)<br/>
!>       - -v, --version<br/>
!>          show Siren's version   (and exit)
!>
!> @section sec_coord create_coord.nam
!>    create_coord.nam contains 6 sub-namelists:<br/>
!>       - **namlog** to set logger parameters
!>       - **namcfg** to set configuration file parameters
!>       - **namsrc** to set source/coarse grid parameters
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
!>    - **cn_logfile** [@a create_coord.log]<br/>
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
!>    the source/coarse grid sub-namelist parameters are :
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
!> @subsection subvar namvar 
!>    the variable sub-namelist parameters are :
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
!>
!>             requests must be separated by ';'.<br/>
!>             order of requests does not matter.<br/>
!>
!>       informations about available method could be find in @ref interp,
!>       @ref extrap and @ref filter modules.<br/>
!>       Example: 
!>          - 'glamt: int=linear; ext=dist_weight', 'e1t: int=cubic/rhoi'
!>
!>       @note 
!>          If you do not specify a method which is required, 
!>          default one is apply.
!>
!> @subsection subnst namnst 
!>    the nesting sub-namelist parameters are :
!>
!>    - **in_imin0** [@a ]<br/>
!>       i-direction lower left  point indice of coarse grid subdomain to be used
!>    - **in_imax0** [@a ]<br/>
!>       i-direction upper right point indice of coarse grid subdomain to be used
!>    - **in_jmin0** [@a ]<br/>
!>       j-direction lower left  point indice of coarse grid subdomain to be used
!>    - **in_jmax0** [@a ]<br/>
!>       j-direction upper right point indice of coarse grid subdomain to be used
!> <br/>or<br/>
!>    - **rn_lonmin0** [@a ]<br/>
!>       lower left  longitude of coarse grid subdomain to be used
!>    - **rn_lonmax0** [@a ]<br/>
!>       upper right longitude of coarse grid subdomain to be used
!>    - **rn_latmin0** [@a ]<br/>
!>       lower left  latitude  of coarse grid subdomain to be used
!>    - **rn_latmax0** [@a ]<br/>
!>       upper right latitude  of coarse grid subdomain to be used
!>    @note you could define sub domain with 
!>       - coarse/source grid indices
!>       <br/>or<br/>
!>       - coordinates.<br/>
!>    if coordinates are defined (-180 < lon < 360 and -90 < lat < 90),
!>    SIREN does not take into account indices.
!>
!>    - **in_rhoi**  [@a 1]<br/> 
!>       refinement factor in i-direction
!>
!>    - **in_rhoj**  [@a 1]<br/> 
!>       refinement factor in j-direction
!>
!>       \image html  grid_zoom_60.png 
!>       <center> \image latex grid_zoom_40.png 
!>       </center>
!>
!> @subsection subout namout 
!>    the output sub-namelist parameter is :
!>
!>    - **cn_fileout** [@a coord_fine.nc]<br/>
!>       output bathymetry filename
!>
!> <hr>
!> @author J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date September, 2014
!> - add header for user
!> - compute offset considering grid point
!> - add global attributes in output file
!> @date September, 2015
!> - manage useless (dummy) variable, attributes, and dimension
!> @date September, 2016
!> - allow to use coordinate to define subdomain
!> @date October, 2016
!> - dimension to be used select from configuration file
!> @date January, 2019
!> - add url path to global attributes of output file(s)
!> @date February, 2019
!> - rename sub namelist namcrs to namsrc
!> - create and clean file structure to avoid memory leaks
!> @date Ocober, 2019
!> - add help and version optional arguments
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
PROGRAM create_coord

   USE global                          ! global variable
   USE kind                            ! F90 kind parameter
   USE logger                          ! log file manager
   USE fct                             ! basic useful function
   USE date                            ! date manager
   USE att                             ! attribute manager
   USE dim                             ! dimension manager
   USE var                             ! variable manager
   USE file                            ! file manager
   USE iom                             ! I/O manager
   USE grid                            ! grid manager
   USE extrap                          ! extrapolation manager
   USE interp                          ! interpolation manager
   USE filter                          ! filter manager
   USE mpp                             ! MPP manager
   USE dom                             ! domain manager
   USE iom_mpp                         ! MPP I/O manager
   USE iom_dom                         ! DOM I/O manager

   IMPLICIT NONE

   ! parameters
   CHARACTER(LEN=lc), PARAMETER  :: cp_myname = "create_coord"

   ! local variable
   CHARACTER(LEN=lc)                                    :: cl_arg
   CHARACTER(LEN=lc)                                    :: cl_namelist
   CHARACTER(LEN=lc)                                    :: cl_date
   CHARACTER(LEN=lc)                                    :: cl_url
   CHARACTER(LEN=lc)                                    :: cl_errormsg

   INTEGER(i4)                                          :: il_narg
   INTEGER(i4)                                          :: il_status
   INTEGER(i4)                                          :: il_fileid
   INTEGER(i4)                                          :: il_attid
   INTEGER(i4)                                          :: il_ind
   INTEGER(i4)                                          :: il_nvar
   INTEGER(i4)                                          :: il_ew
   INTEGER(i4)                                          :: il_imin0
   INTEGER(i4)                                          :: il_imax0
   INTEGER(i4)                                          :: il_jmin0
   INTEGER(i4)                                          :: il_jmax0

   INTEGER(i4)      , DIMENSION(ip_maxdim)              :: il_rho
   INTEGER(i4)      , DIMENSION(2)                      :: il_index
   INTEGER(i4)      , DIMENSION(2,2,ip_npoint)          :: il_offset

   LOGICAL                                              :: ll_exist

   TYPE(TATT)                                           :: tl_att

   TYPE(TDOM)                                           :: tl_dom

   TYPE(TVAR)       , DIMENSION(:)        , ALLOCATABLE :: tl_var

   TYPE(TDIM)       , DIMENSION(ip_maxdim)              :: tl_dim

   TYPE(TFILE)                                          :: tl_file

   TYPE(TMPP)                                           :: tl_coord0
   TYPE(TFILE)                                          :: tl_fileout

   ! loop indices
   INTEGER(i4) :: ji
   INTEGER(i4) :: jj

   ! namelist variable
   ! namlog
   CHARACTER(LEN=lc) :: cn_logfile  = 'create_coord.log' 
   CHARACTER(LEN=lc) :: cn_verbosity= 'warning' 
   INTEGER(i4)       :: in_maxerror = 5

   ! namcfg
   CHARACTER(LEN=lc) :: cn_varcfg   = './cfg/variable.cfg' 
   CHARACTER(LEN=lc) :: cn_dimcfg   = './cfg/dimension.cfg' 
   CHARACTER(LEN=lc) :: cn_dumcfg   = './cfg/dummy.cfg'

   ! namsrc
   CHARACTER(LEN=lc) :: cn_coord0   = '' 
   INTEGER(i4)       :: in_perio0   = -1

   ! namvar
   CHARACTER(LEN=lc), DIMENSION(ip_maxvar) :: cn_varinfo = ''

   !namnst
   REAL(sp)          :: rn_lonmin0  = -360.
   REAL(sp)          :: rn_lonmax0  = -360.
   REAL(sp)          :: rn_latmin0  = -360.
   REAL(sp)          :: rn_latmax0  = -360.
   INTEGER(i4)       :: in_imin0 = 0
   INTEGER(i4)       :: in_imax0 = 0
   INTEGER(i4)       :: in_jmin0 = 0
   INTEGER(i4)       :: in_jmax0 = 0
   INTEGER(i4)       :: in_rhoi  = 1
   INTEGER(i4)       :: in_rhoj  = 1

   !namout
   CHARACTER(LEN=lc) :: cn_fileout  = 'coord_fine.nc'
   !-------------------------------------------------------------------

   NAMELIST /namlog/ &  !  logger namelist
   &  cn_logfile,    &  !< logger file name
   &  cn_verbosity,  &  !< logger verbosity
   &  in_maxerror       !< logger maximum error

   NAMELIST /namcfg/ &  !< configuration namelist
   &  cn_varcfg,     &  !< variable configuration file
   &  cn_dimcfg,     &  !< dimension configuration file
   &  cn_dumcfg         !< dummy configuration file

   NAMELIST /namsrc/ &  !< source/coarse grid namelist
   &  cn_coord0 ,    &  !< coordinate file
   &  in_perio0         !< periodicity index

   NAMELIST /namvar/ &  !< variable namelist
   &  cn_varinfo        !< list of variable and extra information about 
                        !< interpolation, extrapolation or filter method to be used. 
                        !< (ex: 'votemper:linear,hann,dist_weight','vosaline:cubic' ) 
 
   NAMELIST /namnst/ &  !< nesting namelist
   &  rn_lonmin0,    &  !< lower left  coarse grid longitude
   &  rn_lonmax0,    &  !< upper right coarse grid longitude
   &  rn_latmin0,    &  !< lower left  coarse grid latitude
   &  rn_latmax0,    &  !< upper right coarse grid latitude
   &  in_imin0,      &  !< i-direction lower left  point indice 
   &  in_imax0,      &  !< i-direction upper right point indice
   &  in_jmin0,      &  !< j-direction lower left  point indice
   &  in_jmax0,      &  !< j-direction upper right point indice
   &  in_rhoi,       &  !< refinement factor in i-direction
   &  in_rhoj           !< refinement factor in j-direction

   NAMELIST /namout/ &  !< output namelist
   &  cn_fileout        !< fine grid coordinate file   
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
               ! get variable extra information on configuration file
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
               READ( il_fileid, NML = namvar )
               ! add user change in extra information
               CALL var_chg_extra( cn_varinfo )

               READ( il_fileid, NML = namnst )
               READ( il_fileid, NML = namout )

               CLOSE( il_fileid, IOSTAT=il_status )
               CALL fct_err(il_status)
               IF( il_status /= 0 )THEN
                  CALL logger_error("CREATE COORD: closing "//TRIM(cl_namelist))
               ENDIF

            ELSE

               WRITE(cl_errormsg,*) " ERROR : can't find "//TRIM(cl_namelist)
               CALL fct_help(cp_myname,cl_errormsg) 
               CALL EXIT(1)

            ENDIF

      END SELECT
   ENDIF

   ! open files
   IF( cn_coord0 /= '' )THEN
      tl_file=file_init(TRIM(cn_coord0))
      tl_coord0=mpp_init( tl_file, id_perio=in_perio0)
      ! clean
      CALL file_clean(tl_file)
      CALL grid_get_info(tl_coord0)
   ELSE
      CALL logger_fatal("CREATE COORD: no coarse grid coordinate found. "//&
      &     "check namelist")      
   ENDIF

   ! check
   ! check output file do not already exist
   INQUIRE(FILE=TRIM(cn_fileout), EXIST=ll_exist)
   IF( ll_exist )THEN
      CALL logger_fatal("CREATE COORD: output file "//TRIM(cn_fileout)//&
      &  " already exist.")
   ENDIF

   ! check nesting parameters
   il_index(:)=0
   IF( rn_lonmin0 >= -180. .AND. rn_lonmin0 <= 360 .AND. &
     & rn_latmin0 >= -90.  .AND. rn_latmin0 <= 90. )THEN

      il_index(:)=grid_get_closest(tl_coord0, &
         &                         REAL(rn_lonmin0,dp), REAL(rn_latmin0,dp), &
         &                         cd_pos='ll') 
      il_imin0=il_index(1)
      il_jmin0=il_index(2)
   ELSE
      il_imin0=in_imin0
      il_jmin0=in_jmin0
   ENDIF

   il_index(:)=0
   IF( rn_lonmax0 >= -180. .AND. rn_lonmax0 <= 360 .AND. &
     & rn_latmax0 >= -90.  .AND. rn_latmax0 <= 90. )THEN

      il_index(:)=grid_get_closest(tl_coord0, &
         &                         REAL(rn_lonmax0,dp), REAL(rn_latmax0,dp), &
         &                         cd_pos='ur') 
      il_imax0=il_index(1)
      il_jmax0=il_index(2)
   ELSE
      il_imax0=in_imax0
      il_jmax0=in_jmax0
   ENDIF

   ! forced indices for east west cyclic domain
   IF( rn_lonmin0 == rn_lonmax0 .AND. &
     & rn_lonmin0 /= -360. )THEN
      il_imin0=0
      il_imax0=0
   ENDIF

   IF( il_imin0 < 0 .OR. il_imax0 < 0 .OR. il_jmin0 < 0 .OR. il_jmax0 < 0)THEN
      CALL logger_fatal("CREATE COORD: invalid points indices."//&
      &  " check namelist "//TRIM(cl_namelist))
   ENDIF

   il_rho(:)=1
   IF( in_rhoi < 1 .OR. in_rhoj < 1 )THEN
      CALL logger_error("CREATE COORD: invalid refinement factor."//&
      &  " check namelist "//TRIM(cl_namelist))
   ELSE
      il_rho(jp_I)=in_rhoi
      il_rho(jp_J)=in_rhoj      

      il_offset(:,:,:)=create_coord_get_offset(il_rho(:))
   ENDIF

   ! check domain validity
   CALL grid_check_dom(tl_coord0, il_imin0, il_imax0, il_jmin0, il_jmax0 )

   ! compute domain
   tl_dom=dom_init( tl_coord0,         &
   &                il_imin0, il_imax0,&
   &                il_jmin0, il_jmax0 )

   ! add extra band (if need be) to compute interpolation
   CALL dom_add_extra(tl_dom)

   ! open mpp files
   CALL iom_dom_open(tl_coord0, tl_dom)

   il_nvar=tl_coord0%t_proc(1)%i_nvar
   ALLOCATE( tl_var(il_nvar) )
   DO ji=1,il_nvar

      tl_var(ji)=iom_dom_read_var(tl_coord0, &
      &                          TRIM(tl_coord0%t_proc(1)%t_var(ji)%c_name),&
      &                          tl_dom)

      SELECT CASE(TRIM(tl_var(ji)%c_point))
         CASE('T')
            jj=jp_T
         CASE('U')
            jj=jp_U
         CASE('V')
            jj=jp_V
         CASE('F')
            jj=jp_F
      END SELECT

      ! interpolate variables
      CALL create_coord_interp( tl_var(ji), il_rho(:), &
      &                         il_offset(:,:,jj) )

      ! remove extraband added to domain
      CALL dom_del_extra( tl_var(ji), tl_dom, il_rho(:), .true. )

      ! filter
      CALL filter_fill_value(tl_var(ji))      

   ENDDO

   ! clean
   CALL dom_clean_extra( tl_dom )

   ! close mpp files
   CALL iom_dom_close(tl_coord0)

   ! clean
   CALL mpp_clean(tl_coord0)

   ! create file
   tl_fileout=file_init(TRIM(cn_fileout))

   ! add dimension
   ! save biggest dimension
   tl_dim(:)=var_max_dim(tl_var(:))

   DO ji=1,ip_maxdim
      IF( tl_dim(ji)%l_use ) CALL file_add_dim(tl_fileout, tl_dim(ji))
   ENDDO

   ! add variables
   DO ji=il_nvar,1,-1
      CALL file_add_var(tl_fileout, tl_var(ji))
      CALL var_clean(tl_var(ji))
   ENDDO

   ! add some attribute
   tl_att=att_init("Created_by","SIREN create_coord")
   CALL file_add_att(tl_fileout, tl_att)

   !add source url
   cl_url=fct_split(fct_split(cp_url,2,'$'),2,'URL:')
   tl_att=att_init("SIREN_url",cl_url)
   CALL file_add_att(tl_fileout, tl_att)

   ! add date of creation
   cl_date=date_print(date_now())
   tl_att=att_init("Creation_date",cl_date)
   CALL file_add_att(tl_fileout, tl_att)

   tl_att=att_init("src_file",TRIM(fct_basename(cn_coord0)))
   CALL file_add_att(tl_fileout, tl_att)   

   tl_att=att_init("src_i_indices",(/tl_dom%i_imin,tl_dom%i_imax/))
   CALL file_add_att(tl_fileout, tl_att)   
   tl_att=att_init("src_j_indices",(/tl_dom%i_jmin,tl_dom%i_jmax/))
   CALL file_add_att(tl_fileout, tl_att)
   IF( .NOT. ALL(il_rho(:)==1) )THEN
      tl_att=att_init("refinment_factor",(/il_rho(jp_I),il_rho(jp_J)/))
      CALL file_add_att(tl_fileout, tl_att)
   ENDIF

   ! add attribute periodicity
   il_attid=0
   IF( ASSOCIATED(tl_fileout%t_att) )THEN
      il_attid=att_get_index(tl_fileout%t_att(:),'periodicity')
   ENDIF
   IF( tl_dom%i_perio >= 0 .AND. il_attid == 0 )THEN
      tl_att=att_init('periodicity',tl_dom%i_perio)
      CALL file_add_att(tl_fileout,tl_att)
   ENDIF

   ! add attribute east west overlap
   il_attid=0
   IF( ASSOCIATED(tl_fileout%t_att) )THEN
      il_attid=att_get_index(tl_fileout%t_att(:),'ew_overlap')
   ENDIF
   IF( il_attid == 0 )THEN
      il_ind=var_get_index(tl_fileout%t_var(:),'longitude')
      IF( il_ind == 0 )THEN
         il_ind=var_get_index(tl_fileout%t_var(:),'longitude_T')
      ENDIF
      il_ew=grid_get_ew_overlap(tl_fileout%t_var(il_ind))
      IF( il_ew >= 0 )THEN
         tl_att=att_init('ew_overlap',il_ew)
         CALL file_add_att(tl_fileout,tl_att)
      ENDIF
   ENDIF

   ! create file
   CALL iom_create(tl_fileout)

   ! write file
   CALL iom_write_file(tl_fileout)

   ! close file
   CALL iom_close(tl_fileout)

   ! clean
   CALL att_clean(tl_att)
   CALL var_clean(tl_var(:))
   DEALLOCATE( tl_var) 

   CALL file_clean(tl_fileout)
   CALL var_clean_extra()

   ! close log file
   CALL logger_footer()
   CALL logger_close() 

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION create_coord_get_offset(id_rho) &
         & RESULT (if_offset)
   !-------------------------------------------------------------------
   !> @brief
   !> This function compute offset over Arakawa grid points, 
   !> given refinement factor.
   !> 
   !> @author J.Paul
   !> @date August, 2014 - Initial Version
   !>
   !> @param[in] id_rho array of refinement factor
   !> @return array of offset
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      INTEGER(i4), DIMENSION(:), INTENT(IN) :: id_rho

      ! function
      INTEGER(i4), DIMENSION(2,2,ip_npoint) :: if_offset

      ! local variable
      ! loop indices
      !----------------------------------------------------------------

      ! case 'T'
      if_offset(jp_I,:,jp_T)=FLOOR(REAL(id_rho(jp_I)-1,dp)*0.5)
      if_offset(jp_J,:,jp_T)=FLOOR(REAL(id_rho(jp_J)-1,dp)*0.5)
      ! case 'U'
      if_offset(jp_I,1,jp_U)=0
      if_offset(jp_I,2,jp_U)=id_rho(jp_I)-1
      if_offset(jp_J,:,jp_U)=FLOOR(REAL(id_rho(jp_J)-1,dp)*0.5)
      ! case 'V'
      if_offset(jp_I,:,jp_V)=FLOOR(REAL(id_rho(jp_I)-1,dp)*0.5)
      if_offset(jp_J,1,jp_V)=0
      if_offset(jp_J,2,jp_V)=id_rho(jp_J)-1
      ! case 'F'
      if_offset(jp_I,1,jp_F)=0
      if_offset(jp_I,2,jp_F)=id_rho(jp_I)-1
      if_offset(jp_J,1,jp_F)=0
      if_offset(jp_J,2,jp_F)=id_rho(jp_J)-1

   END FUNCTION create_coord_get_offset
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE create_coord_interp(td_var, id_rho, id_offset, &
         &                        id_iext, id_jext)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine interpolate variable, given refinment factor.
   !> 
   !> @details 
   !>  Optionaly, you could specify number of points 
   !>    to be extrapolated in i- and j-direction.<br/>
   !>  variable mask is first computed (using _FillValue) and interpolated.<br/>
   !>  variable is then extrapolated, and interpolated.<br/> 
   !>  Finally interpolated mask is applied on refined variable.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable strcuture 
   !> @param[in] id_rho    array of refinement factor
   !> @param[in] id_offset offset between fine grid and coarse grid
   !> @param[in] id_iext   number of points to be extrapolated in i-direction
   !> @param[in] id_jext   number of points to be extrapolated in j-direction
   !>
   !> @todo check if mask is really needed
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) ,                 INTENT(INOUT) :: td_var
      INTEGER(i4), DIMENSION(:)  , INTENT(IN   ) :: id_rho
      INTEGER(i4), DIMENSION(:,:), INTENT(IN)    :: id_offset
      INTEGER(i4),                 INTENT(IN   ), OPTIONAL :: id_iext
      INTEGER(i4),                 INTENT(IN   ), OPTIONAL :: id_jext

      ! local variable
      TYPE(TVAR)  :: tl_mask

      INTEGER(i1), DIMENSION(:,:,:,:), ALLOCATABLE :: bl_mask

      INTEGER(i4) :: il_iext
      INTEGER(i4) :: il_jext

      ! loop indices
      !----------------------------------------------------------------

      IF( ANY(SHAPE(id_offset(:,:)) /= 2) )THEN
         CALL logger_error("CREATE COORD INTERP: invalid dimension of "//&
         &                 "offset array")
      ENDIF

      !WARNING: two extrabands are required for cubic interpolation
      il_iext=2
      IF( PRESENT(id_iext) ) il_iext=id_iext

      il_jext=2
      IF( PRESENT(id_jext) ) il_jext=id_jext
      
      IF( il_iext < 2 .AND. td_var%c_interp(1) == 'cubic' )THEN
         CALL logger_warn("CREATE COORD INTERP: at least extrapolation "//&
         &  "on two points are required with cubic interpolation ")
         il_iext=2
      ENDIF

      IF( il_jext < 2 .AND. td_var%c_interp(1) == 'cubic' )THEN
         CALL logger_warn("CREATE COORD INTERP: at least extrapolation "//&
         &  "on two points are required with cubic interpolation ")
         il_jext=2
      ENDIF

      IF( ANY(id_rho(:)>1) )THEN
         ! work on mask
         ! create mask
         ALLOCATE(bl_mask(td_var%t_dim(1)%i_len, &
         &                td_var%t_dim(2)%i_len, &
         &                td_var%t_dim(3)%i_len, &
         &                td_var%t_dim(4)%i_len) )

         bl_mask(:,:,:,:)=1
         WHERE(td_var%d_value(:,:,:,:)==td_var%d_fill) bl_mask(:,:,:,:)=0      

         SELECT CASE(TRIM(td_var%c_point))
         CASE DEFAULT ! 'T'
            tl_mask=var_init('tmask',bl_mask(:,:,:,:),td_dim=td_var%t_dim(:),&
            &                id_ew=td_var%i_ew )
         CASE('U')
            tl_mask=var_init('umask',bl_mask(:,:,:,:),td_dim=td_var%t_dim(:),&
            &                id_ew=td_var%i_ew )
         CASE('V')
            tl_mask=var_init('vmask',bl_mask(:,:,:,:),td_dim=td_var%t_dim(:),&
            &                id_ew=td_var%i_ew )
         CASE('F')
            tl_mask=var_init('fmask',bl_mask(:,:,:,:),td_dim=td_var%t_dim(:),&
            &                id_ew=td_var%i_ew )
         END SELECT         

         DEALLOCATE(bl_mask)

         ! interpolate mask
         CALL interp_fill_value( tl_mask, id_rho(:), &
         &                       id_offset=id_offset(:,:) )

         ! work on variable
         ! add extraband
         CALL extrap_add_extrabands(td_var, il_iext, il_jext)

         ! extrapolate variable
         CALL extrap_fill_value( td_var )

         ! interpolate variable
         CALL interp_fill_value( td_var, id_rho(:), &
         &                       id_offset=id_offset(:,:))

         ! remove extraband
         CALL extrap_del_extrabands(td_var, il_iext*id_rho(jp_I), il_jext*id_rho(jp_J))

         ! keep original mask 
         WHERE( tl_mask%d_value(:,:,:,:) == 0 )
            td_var%d_value(:,:,:,:)=td_var%d_fill
         END WHERE
      ENDIF

      ! clean variable structure
      CALL var_clean(tl_mask)

   END SUBROUTINE create_coord_interp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END PROGRAM create_coord
