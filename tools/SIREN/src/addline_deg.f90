!----------------------------------------------------------------------
! MERCATOR OCEAN, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!> @file
!> @brief 
!> This program add line to all variables of the input file.
!>
!> @details
!> @section sec2 how to
!>    to add line to file:<br/>
!> @code{.sh}
!>    ./SIREN/bin/addline addline.nam
!> @endcode
!> the namelist file (**addline.nam**) sets up program parameters.
!>
!> to set up program parameters, you just have to fill the namelist file (**add_line.nam**).
!> @note
!>    you could find a template of the namelist in templates directory.
!>
!>    create_bathy.nam comprise 4 namelists:<br/>
!>       - **namlog** to set logger parameters
!>       - **namcfg** to set configuration file parameters
!>       - **namsrc** to set source grid parameters
!>       - **namout** to set output parameters
!>
!>    here after, each sub-namelist parameters is detailed.
!>    @note 
!>       default values are specified between brackets
!>
!> @subsection sublog namlog
!>    the logger sub-namelist parameters are :
!>
!>    - **cn_logfile** [@a addline.log]<br/>
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
!>          - 'Bathymetry:gridT.nc'
!>
!>       @note 
!>          Optionnaly, NEMO periodicity could be added following the filename.
!>          the periodicity must be separated by ';'
!>
!>       Example:
!>          - 'Bathymetry:gridT.nc ; perio=4'<br/>
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
!>          - 'Bathymetry: flt=2*hamming(2,3); min=0'
!>
!>       @note 
!>          If you do not specify a method which is required, 
!>          default one is apply.
!>
!> @subsection subout namout 
!>    the output sub-namelist parameter is :
!>
!>    - **cn_fileout** [@a addline_deg.nc]<br/>
!>       output filename
!>    - @b ln_extrap [@a .FALSE.]<br/>
!>       extrapolate extra line 
!>    - @b ln_copy [@a .FALSE.]<br/>
!>       copy extra line from above
!>    - **in_nproc** [@a 1]<br/>
!>       number of processor to be used
!>    - **in_niproc** [@a 1]<br/>
!>       i-direction number of processor
!>    - **in_njproc** [@a 1]<br/>
!>       j-direction numebr of processor
!>
!> <hr>
!> @author J.Paul
!> @date October, 2015 - Initial Version
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
PROGRAM addline_deg

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
   USE extrap                          ! extrapolation manager
   USE interp                          ! interpolation manager
   USE filter                          ! filter manager
   USE mpp                             ! MPP manager
   USE iom_mpp                         ! MPP I/O manager

   IMPLICIT NONE

   ! local variable
   CHARACTER(LEN=lc)                                  :: cl_namelist
   CHARACTER(LEN=lc)                                  :: cl_date

   INTEGER(i4)                                        :: il_narg
   INTEGER(i4)                                        :: il_status
   INTEGER(i4)                                        :: il_fileid
   INTEGER(i4)                                        :: il_varid
   INTEGER(i4)                                        :: il_attid
   INTEGER(i4)                                        :: il_index
   INTEGER(i4)                                        :: il_nvar

   LOGICAL                                            :: ll_exist

   TYPE(TMPP)                                         :: tl_coord0
   TYPE(TMPP)                                         :: tl_mpp
   TYPE(TMPP)                                         :: tl_mppout

   TYPE(TATT)                                         :: tl_att
   
   TYPE(TVAR)                                         :: tl_lon
   TYPE(TVAR)                                         :: tl_lat
   TYPE(TVAR)                                         :: tl_depth
   TYPE(TVAR)                                         :: tl_time

   TYPE(TVAR)                                         :: tl_tmp
   TYPE(TVAR)       , DIMENSION(:), ALLOCATABLE       :: tl_var
   
   TYPE(TDIM)       , DIMENSION(ip_maxdim)            :: tl_dim

   TYPE(TMULTI)                                       :: tl_multi

   ! loop indices
   INTEGER(i4) :: ji
   INTEGER(i4) :: jj
   INTEGER(i4) :: jk
   INTEGER(i4) :: jvar

   ! namelist variable
   ! namlog
   CHARACTER(LEN=lc)                       :: cn_logfile = 'addline.log' 
   CHARACTER(LEN=lc)                       :: cn_verbosity = 'warning' 
   INTEGER(i4)                             :: in_maxerror = 5

   ! namcfg
   CHARACTER(LEN=lc)                       :: cn_varcfg = 'variable.cfg' 
   CHARACTER(LEN=lc)                       :: cn_dimcfg = 'dimension.cfg' 
   CHARACTER(LEN=lc)                       :: cn_dumcfg = 'dummy.cfg' 

   ! namsrc
   CHARACTER(LEN=lc)                       :: cn_coord0 = '' 
   INTEGER(i4)                             :: in_perio0 = -1

   ! namvar
   CHARACTER(LEN=lc), DIMENSION(ip_maxvar) :: cn_varinfo = ''
   CHARACTER(LEN=lc), DIMENSION(ip_maxvar) :: cn_varfile = ''

   ! namout
   CHARACTER(LEN=lc)                       :: cn_fileout = 'addline_deg.nc' 
   LOGICAL                                 :: ln_extrap  = .FALSE.
   LOGICAL                                 :: ln_copy    = .FALSE.
   INTEGER(i4)                             :: in_nproc   = 0
   INTEGER(i4)                             :: in_niproc  = 0
   INTEGER(i4)                             :: in_njproc  = 0
   CHARACTER(LEN=lc)                       :: cn_type    = 'cdf'
   !-------------------------------------------------------------------

   NAMELIST /namlog/ &   !< logger namelist
   &  cn_logfile,    &   !< log file
   &  cn_verbosity,  &   !< log verbosity
   &  in_maxerror        !< logger maximum error

   NAMELIST /namcfg/ &   !< configuration namelist
   &  cn_varcfg, &       !< variable configuration file
   &  cn_dimcfg, &       !< dimension configuration file
   &  cn_dumcfg          !< dummy configuration file

   NAMELIST /namsrc/ &   !< source/coarse grid namelist
   &  cn_coord0,  &      !< coordinate file
   &  in_perio0          !< periodicity index
   
   NAMELIST /namvar/ &   !< variable namelist
   &  cn_varinfo, &      !< list of variable and interpolation method to be used. (ex: 'votemper:linear','vosaline:cubic' )
   &  cn_varfile         !< list of variable file

   NAMELIST /namout/ &   !< output namlist
   &  cn_fileout, &      !< fine grid bathymetry file
   &  ln_extrap, &
   &  ln_copy,   &
   &  in_niproc,  &     !< i-direction number of processor
   &  in_njproc,  &     !< j-direction numebr of processor
   &  in_nproc,   &     !< number of processor to be used
   &  cn_type           !< output type format (dimg, cdf)
   !-------------------------------------------------------------------

   ! namelist
   ! get namelist
   il_narg=COMMAND_ARGUMENT_COUNT() !f03 intrinsec
   IF( il_narg/=1 )THEN
      PRINT *,"ERROR in addline: need a namelist"
      STOP
   ELSE
      CALL GET_COMMAND_ARGUMENT(1,cl_namelist) !f03 intrinsec
   ENDIF
 
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
         PRINT *,"ERROR in addline: error opening "//TRIM(cl_namelist)
         STOP
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

      READ( il_fileid, NML = namsrc )
      READ( il_fileid, NML = namvar )
      ! add user change in extra information
      CALL var_chg_extra( cn_varinfo )
      ! match variable with file
      tl_multi=multi_init(cn_varfile)
      
      READ( il_fileid, NML = namout )

      CLOSE( il_fileid, IOSTAT=il_status )
      CALL fct_err(il_status)
      IF( il_status /= 0 )THEN
         CALL logger_error("ADD LINE: closing "//TRIM(cl_namelist))
      ENDIF

   ELSE

      PRINT *,"ERROR in addline: can't find "//TRIM(cl_namelist)
      STOP

   ENDIF

   CALL multi_print(tl_multi)

   ! open files
   IF( cn_coord0 /= '' )THEN
      tl_coord0=mpp_init( file_init(TRIM(cn_coord0)), id_perio=in_perio0)
      CALL grid_get_info(tl_coord0)
   ELSE
      CALL logger_fatal("ADD LINE: no coarse grid coordinate found. "//&
      &     "check namelist")      
   ENDIF

   ! check
   ! check output file do not already exist
   print *,'cn_fileout ',TRIM(cn_fileout)
   INQUIRE(FILE=TRIM(cn_fileout), EXIST=ll_exist)
   IF( ll_exist )THEN
      CALL logger_fatal("ADD LINE: output file "//TRIM(cn_fileout)//&
      &  " already exist.")
   ENDIF

   IF( .NOT. ASSOCIATED(tl_multi%t_mpp) )THEN
      CALL logger_error("ADD LINE: no mpp file to work on. "//&
      &                 "check cn_varfile in namelist.")
   ELSE

      ALLOCATE( tl_var( tl_multi%i_nvar ) )
      jk=0
      DO ji=1,tl_multi%i_nmpp
      
         IF( .NOT. ASSOCIATED(tl_multi%t_mpp(ji)%t_proc(1)%t_var) )THEN

            CALL logger_fatal("ADD LINE: no variable to work on for "//&
            &                 "mpp file"//TRIM(tl_multi%t_mpp(ji)%c_name)//&
            &                 ". check cn_varfile in namelist.")
         ELSE

            WRITE(*,'(a)') "work on file "//TRIM(tl_multi%t_mpp(ji)%c_name)
            tl_mpp=mpp_init( file_init(TRIM(tl_multi%t_mpp(ji)%t_proc(1)%c_name)) )
            CALL grid_get_info(tl_mpp)

            ! open mpp file
            CALL iom_mpp_open(tl_mpp)

            ! get or check depth value
            IF( tl_multi%t_mpp(ji)%t_proc(1)%i_depthid /= 0 )THEN
               il_varid=tl_multi%t_mpp(ji)%t_proc(1)%i_depthid
               IF( ASSOCIATED(tl_depth%d_value) )THEN
                  tl_tmp=iom_mpp_read_var(tl_mpp,il_varid)
                  IF( ANY( tl_depth%d_value(:,:,:,:) /= &
                  &        tl_tmp%d_value(:,:,:,:) ) )THEN
                     CALL logger_fatal("ADD LINE: depth value from "//&
                     &  TRIM(tl_multi%t_mpp(ji)%c_name)//" not conform "//&
                     &  " to those from former file(s).")
                  ENDIF
                  CALL var_clean(tl_tmp)
               ELSE
                  tl_depth=iom_mpp_read_var(tl_mpp,il_varid)
               ENDIF
            ENDIF

            ! get or check time value
            IF( tl_multi%t_mpp(ji)%t_proc(1)%i_timeid /= 0 )THEN
               il_varid=tl_multi%t_mpp(ji)%t_proc(1)%i_timeid
               IF( ASSOCIATED(tl_time%d_value) )THEN
                  tl_tmp=iom_mpp_read_var(tl_mpp,il_varid)
                  IF( ANY( tl_time%d_value(:,:,:,:) /= &
                  &        tl_tmp%d_value(:,:,:,:) ) )THEN
                     CALL logger_fatal("ADD LINE: time value from "//&
                     &  TRIM(tl_multi%t_mpp(ji)%c_name)//" not conform "//&
                     &  " to those from former file(s).")
                  ENDIF
                  CALL var_clean(tl_tmp)
               ELSE
                  tl_time=iom_mpp_read_var(tl_mpp,il_varid)
               ENDIF
            ENDIF

            ! close mpp file
            CALL iom_mpp_close(tl_mpp)

            !- add line to input file variable 
            DO jj=1,tl_multi%t_mpp(ji)%t_proc(1)%i_nvar
               jk=jk+1
               tl_tmp=var_copy(tl_multi%t_mpp(ji)%t_proc(1)%t_var(jj))
               WRITE(*,'(2x,a)') "work on variable "//TRIM(tl_tmp%c_name)
            
               tl_var(jk)=add_line( tl_tmp, tl_mpp, &
               &                    tl_coord0 )

               IF( ln_copy )THEN
                  tl_var(jk)%d_value(:,2,:,:)=tl_var(jk)%d_value(:,3,:,:)
               ELSEIF( ln_extrap )THEN
                  ! extrapolate variable
                  CALL extrap_fill_value( tl_var(jk) )
               ENDIF
               ! clean
               CALL var_clean(tl_tmp)

            ENDDO

         ENDIF

      ENDDO

   ENDIF

   il_nvar=tl_multi%i_nvar
   ! clean
   CALL multi_clean(tl_multi)

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
      CALL iom_mpp_open(tl_coord0)

      ! add longitude
      tl_lon=iom_mpp_read_var(tl_coord0,'longitude')
      CALL mpp_add_var(tl_mppout, tl_lon)
      CALL var_clean(tl_lon)

      ! add latitude
      tl_lat=iom_mpp_read_var(tl_coord0,'latitude')
      CALL mpp_add_var(tl_mppout, tl_lat)
      CALL var_clean(tl_lat)

      ! close mpp files
      CALL iom_mpp_close(tl_coord0)

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

   ! add other variables
   DO jvar=il_nvar,1,-1
      ! check if variable already add
      il_index=var_get_index(tl_mppout%t_proc(1)%t_var(:), tl_var(jvar)%c_name)
      IF( il_index == 0 )THEN
         CALL mpp_add_var(tl_mppout, tl_var(jvar))
         CALL var_clean(tl_var(jvar))
      ENDIF
   ENDDO

   ! add some attribute
   tl_att=att_init("Created_by","SIREN addline_deg")
   CALL mpp_add_att(tl_mppout, tl_att)

   cl_date=date_print(date_now())
   tl_att=att_init("Creation_date",cl_date)
   CALL mpp_add_att(tl_mppout, tl_att)

   ! add attribute periodicity
   il_attid=0
   IF( ASSOCIATED(tl_mppout%t_proc(1)%t_att) )THEN
      il_attid=att_get_id(tl_mppout%t_proc(1)%t_att(:),'periodicity')
   ENDIF
   IF( tl_coord0%i_perio >= 0 .AND. il_attid == 0 )THEN
      tl_att=att_init('periodicity',tl_coord0%i_perio)
      CALL mpp_add_att(tl_mppout,tl_att)
   ENDIF

   il_attid=0
   IF( ASSOCIATED(tl_mppout%t_proc(1)%t_att) )THEN
      il_attid=att_get_id(tl_mppout%t_proc(1)%t_att(:),'ew_overlap')
   ENDIF
   IF( tl_coord0%i_ew >= 0 .AND. il_attid == 0 )THEN
      tl_att=att_init('ew_overlap',tl_coord0%i_ew)
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

   CALL mpp_clean(tl_mppout)
   CALL mpp_clean(tl_coord0)

   ! close log file
   CALL logger_footer()
   CALL logger_close()

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION add_line(td_var, td_mpp, td_coord) &
         &  RESULT(tf_var)
   !-------------------------------------------------------------------
   !> @brief
   !> This function add line to variable and return variable structure
   !> 
   !> @author J.Paul
   !> @date October, 2015 - Initial Version
   !>
   !> @param[in] td_var    variable structure 
   !> @param[in] td_mpp    mpp file structure
   !> @param[in] td_coord  coordinate file structure
   !> @return variable structure
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), INTENT(IN) :: td_var  
      TYPE(TMPP), INTENT(IN) :: td_mpp
      TYPE(TMPP), INTENT(IN) :: td_coord

      ! function
      TYPE(TVAR)             :: tf_var

      ! local variable
      INTEGER(i4), DIMENSION(2,2) :: il_ghost

      TYPE(TMPP)  :: tl_mpp

      TYPE(TATT)  :: tl_att

      TYPE(TDIM), DIMENSION(ip_maxdim)  :: tl_dim
      ! loop indices
      !----------------------------------------------------------------

      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN
         CALL logger_error("ADD LINE: no processor associated "//&
         &  "to mpp "//TRIM(td_mpp%c_name))
      ELSE

         !init
         tl_mpp=mpp_copy(td_mpp)
         il_ghost(:,:)=0

         tl_dim(:)=dim_copy(td_coord%t_dim(:))

         ! ghost cell to be added
         il_ghost(jp_I,:)=(/0,0/)
         il_ghost(jp_J,:)=(/1,0/)

         ! open mpp files
         CALL iom_mpp_open(tl_mpp)

         ! read variable
         tf_var=iom_mpp_read_var(tl_mpp,TRIM(td_var%c_name))

         ! close mpp file
         CALL iom_mpp_close(tl_mpp)

         ! add ghost cell
         CALL grid_add_ghost(tf_var,il_ghost(:,:))

         ! add attribute to variable
         tl_att=att_init('src_file',TRIM(fct_basename(tl_mpp%c_name)))
         CALL var_move_att(tf_var, tl_att)         

         tl_att=att_init('add_i_line',(/il_ghost(jp_I,1), il_ghost(jp_I,2)/))
         CALL var_move_att(tf_var, tl_att)

         tl_att=att_init('add_j_line',(/il_ghost(jp_J,1), il_ghost(jp_J,2)/))
         CALL var_move_att(tf_var, tl_att)

         ! clean structure
         CALL att_clean(tl_att)
         CALL mpp_clean(tl_mpp)
      ENDIF

   END FUNCTION add_line
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END PROGRAM 
