!----------------------------------------------------------------------
! MERCATOR OCEAN, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @file
!> @brief 
!> This program creates/computes the domain layout for you configuration.
!>
!> @details
!> @section sec1 method
!>
!> Domain layout is computed, with domain dimension, overlap between subdomain,
!> and the number of processors available or following i and j-direction.
!> Then the number of sea/land processors is compute with mask.
!>
!> The optimized domain layout is assumed to be the domain layout, with the the most land 
!> processors removed. If no land processor could be removed, it assumed to be the domain layout
!> with the most sea processors.
!>
!> @section sec2 how to
!> USAGE: create_layout create_layout.nam [-v] [-h]<br/>
!>    - positional arguments:<br/>
!>       - create_layout.nam<br/>
!>          namelist of create_layout
!>          @note
!>             a template of the namelist could be created running (in templates directory):
!>             @code{.sh}
!>                python create_templates.py create_layout
!>             @endcode
!>
!>    - optional arguments:<br/>
!>       - -h, --help<br/>
!>          show this help message (and exit)<br/>
!>       - -v, --version<br/>
!>          show Siren's version   (and exit)
!>
!> @section sec_layout create_layout.nam
!>    create_layout.nam contains 4 namelists:<br/>
!>       - **namlog** to set logger parameters
!>       - **namcfg** to set configuration file parameters
!>       - **namvar** to set variable parameters
!>       - **namout** to set output parameters
!>
!>    here after, each sub-namelist parameters is detailed.
!>    @note 
!>       default values are specified between brackets
!>
!> @subsection sublog namlog
!>    the logger sub-namelist parameters are :
!>
!>    - **cn_logfile** [@a create_layout.log]<br/>
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
!> @subsection subvar namvar 
!>    the variable sub-namelist parameters are :
!>
!>    - **cn_varfile** [@a ]<br/> 
!>       list of variable, and associated file 
!>       @warning 
!>          variable name must be __Bathymetry__ here.
!>
!>    - **cn_varfile** [@a ]<br/>:
!>       list of variable, and associated file.<br/> 
!>       *cn_varfile* is the path and filename of the file where find
!>       variable to be used as mask grid.<br/>
!>
!>       Examples: 
!>          - 'Bathymetry:bathy_meter.nc'
!>
!> @subsection subout namout 
!>    the output sub-namelist parameters are :
!>
!>    - **in_niproc** [@a 1]<br/>:
!>       number of processor in i-direction
!>    - **in_njproc** [@a 1]<br/>:
!>       number of processor in j-direction
!>    - **in_nproc** [@a 1]<br/>:
!>       total number of processor to be used
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
!> @author
!> J.Paul
!>
!> @date January, 2019 - Initial Version
!> @date Ocober, 2019
!> - add help and version optional arguments
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
PROGRAM create_layout

   USE global                          ! global variable
   USE kind                            ! F90 kind parameter
   USE logger                          ! log file manager
   USE fct                             ! basic useful function
   USE date                            ! date manager
   USE math                            ! 
   USE att                             ! attribute manager
   USE dim                             ! dimension manager
   USE var                             ! variable manager
   USE file                            ! file manager
   USE multi                           ! multi file manager
   USE iom                             ! I/O manager
   USE dom                             ! domain manager
   USE grid                            ! grid manager
   USE mpp                             ! MPP manager
   USE iom_mpp                         ! MPP I/O manager

   IMPLICIT NONE

   ! parameters
   CHARACTER(LEN=lc), PARAMETER  :: cp_myname = "create_layout"

   ! local variable
   CHARACTER(LEN=lc)                       :: cl_arg
   CHARACTER(LEN=lc)                       :: cl_namelist
   CHARACTER(LEN=lc)                       :: cl_var
   CHARACTER(LEN=lc)                       :: cl_errormsg


   INTEGER(i4)                             :: il_narg
   INTEGER(i4)                             :: il_status
   INTEGER(i4)                             :: il_fileid

   LOGICAL                                 :: ll_exist

   TYPE(TVAR)                              :: tl_var

   TYPE(TFILE)                             :: tl_file

   TYPE(TMPP)                              :: tl_mpp
   TYPE(TMPP)                              :: tl_mppout

   TYPE(TMULTI)                            :: tl_multi
   
   ! namelist variable
   ! namlog
   CHARACTER(LEN=lc)                       :: cn_logfile = 'create_layout.log' 
   CHARACTER(LEN=lc)                       :: cn_verbosity = 'warning' 
   INTEGER(i4)                             :: in_maxerror = 5

   ! namcfg
   CHARACTER(LEN=lc)                       :: cn_varcfg = './cfg/variable.cfg' 
   CHARACTER(LEN=lc)                       :: cn_dimcfg = './cfg/dimension.cfg'
   CHARACTER(LEN=lc)                       :: cn_dumcfg = './cfg/dummy.cfg'

   ! namvar
   CHARACTER(LEN=lc), DIMENSION(ip_maxvar) :: cn_varfile = ''

   ! namout
   INTEGER(i4)                             :: in_niproc = 0
   INTEGER(i4)                             :: in_njproc = 0 
   INTEGER(i4)                             :: in_nproc  = 0
   !-------------------------------------------------------------------

   NAMELIST /namlog/ &  !< logger namelist
   &  cn_logfile,    &  !< log file
   &  cn_verbosity,  &  !< log verbosity
   &  in_maxerror       !< logger maximum error

   NAMELIST /namcfg/ &  !< configuration namelist
   &  cn_varcfg,     &  !< variable configuration file
   &  cn_dimcfg,     &  !< dimension configuration file
   &  cn_dumcfg         !< dummy configuration file

   NAMELIST /namvar/ &  !< source grid namelist
   &  cn_varfile        !< input file and mask variable   

   NAMELIST /namout/ &  !< output namelist
   &  in_niproc,     &
   &  in_njproc,     &
   &  in_nproc
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
               ! define log file
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

               READ( il_fileid, NML = namvar  )

               ! match variable with file
               tl_multi=multi_init(cn_varfile)

               READ( il_fileid, NML = namout  )

               CLOSE( il_fileid, IOSTAT=il_status )
               CALL fct_err(il_status)
               IF( il_status /= 0 )THEN
                  CALL logger_error("CREATE LAYOUT: closing "//TRIM(cl_namelist))
               ENDIF

            ELSE

               WRITE(cl_errormsg,*) " ERROR : can't find "//TRIM(cl_namelist)
               CALL fct_help(cp_myname,cl_errormsg) 
               CALL EXIT(1)

            ENDIF

      END SELECT
   ENDIF

   IF( .NOT. ASSOCIATED(tl_multi%t_mpp) .AND. tl_multi%i_nmpp /= 1 )THEN
      CALL logger_error("CREATE LAYOUT: no (or too much) mpp file to work on. "//&
      &                 "check cn_varfile in namelist.")
      CALL logger_fatal("CREATE LAYOUT: no input grid found. "//&
      &     "check namelist")
   ELSE

      CALL multi_print(tl_multi)
      
      ! open file
      tl_file=file_init(TRIM(tl_multi%t_mpp(1)%c_name))
      tl_mpp=mpp_init( tl_file )
      ! clean
      CALL file_clean(tl_file)
      !
      CALL grid_get_info(tl_mpp)

      CALL iom_mpp_open(tl_mpp)

      cl_var=TRIM((tl_multi%t_mpp(1)%t_proc(1)%t_var(1)%c_name))
      tl_var=iom_mpp_read_var(tl_mpp,cl_var)

      CALL iom_mpp_close(tl_mpp)
      ! clean structure
      CALL mpp_clean(tl_mpp)

      tl_mppout=mpp_init('layout.nc',tl_var,in_niproc, in_njproc,in_nproc)

      CALL mpp_print(tl_mppout)

      ! clean structure
      CALL var_clean(tl_var)
      CALL mpp_clean(tl_mppout)

   ENDIF

   ! clean
   CALL multi_clean(tl_multi)

   ! close log file
   CALL logger_footer()
   CALL logger_close()

END PROGRAM
