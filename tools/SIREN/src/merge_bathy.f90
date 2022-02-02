!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @file
!> @brief 
!> this program merges bathymetry file at boundaries.
!>
!> @details
!> @section sec1 method
!> coarse grid bathymetry is interpolated on fine grid 
!> (nearest interpolation method is used).<br/>  
!> then fine bathymetry and refined coarse bathymetry are merged at boundaries.<br/>
!>    @f[BathyFine= Weight * BathyCoarse + (1-Weight)*BathyFine@f]
!> the weight function used is :<br/>
!>       @f[Weight = 0.5 + 0.5*COS( \frac{\pi*dist}{width} )@f]<br/>
!> with
!> - dist : number of point to border 
!> - width : boundary size
!>
!> @section sec2 how to
!> USAGE: merge_bathy merge_bathy.nam [-v] [-h]<br/>
!>    - positional arguments:<br/>
!>       - merge_bathy.nam<br/>
!>          namelist of merge_bathy
!>          @note
!>             a template of the namelist could be created running (in templates directory):
!>             @code{.sh}
!>                python create_templates.py merge_bathy
!>             @endcode
!>
!>    - optional arguments:<br/>
!>       - -h, --help<br/>
!>          show this help message (and exit)<br/>
!>       - -v, --version<br/>
!>          show Siren's version   (and exit)
!>
!> @section sec_merge merge_bathy.nam
!>    merge_bathy.nam contains 7 namelists:
!>       - **namlog** to set logger parameters
!>       - **namcfg** to set configuration file parameters
!>       - **namsrc** to set source/coarse grid parameters
!>       - **namtgt** to set target/fine grid parameters
!>       - **namnst** to set sub domain and nesting paramters
!>       - **nambdy** to set boundary parameters
!>       - **namout** to set output parameters
!>
!>    here after, each sub-namelist parameters is detailed.
!>    @note 
!>       default values are specified between brackets
!>
!> @subsection sublog namlog
!>    the logger sub-namelist parameters are :
!>
!>    - **cn_logfile** [@a merge_bathy.log]<br/>
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
!>    - **cn_bathy0** [@a ]<br/> 
!>       path to the bathymetry file
!>       @warning 
!>          variable name must be __Bathymetry__ here.
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
!>    the target/fine grid sub-namelist parameters are :
!>
!>    - **cn_bathy1** [@a ]<br/> 
!>       path to bathymetry file
!>       @warning 
!>          variable name must be __Bathymetry__ here.
!>
!>    - **in_perio1** [@a ]<br/>
!>       NEMO periodicity index (see above)
!>    @note if the fine/target coordinates file (cn_coord1) was created by SIREN, you do
!>    not need to fill this parameter. SIREN will read it on the global attributes of
!>    the coordinates file.
!>
!> @subsection subnst namnst 
!>    the nesting sub-namelist parameters are (default value are specified between brackets):
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
!> @subsection subbdy nambdy
!>    the boundary sub-namelist parameters are :
!>
!>    - **ln_north** [@a .TRUE.]<br/> 
!>       logical to use north boundary or not
!>    - **ln_south** [@a .TRUE.]<br/>
!>       logical to use south boundary or not
!>    - **ln_east**  [@a .TRUE.]<br/>
!>       logical to use east boundary or not
!>    - **ln_west**  [@a .TRUE.]<br/>
!>       logical to use west  boundary or not
!>    <br/> <br/>
!>    - **cn_north** [@a ]<br/>
!>       north boundary indices on fine grid<br/>
!>    - **cn_south** [@a ]<br/>
!>       south boundary indices on fine grid<br/>
!>    - **cn_east**  [@a ]<br/>
!>       east  boundary indices on fine grid<br/>
!>    - **cn_west**  [@a ]<br/>
!>       west  boundary indices on fine grid<br/>
!>
!>       *cn_north* is a string character defining boundary
!>       segmentation.<br/>
!>       segments are separated by '|'.<br/>
!>       each segments of the boundary is composed of:
!>          - indice of velocity (orthogonal to boundary .ie. 
!>             for north boundary, J-indice). 
!>          - indice of segment start (I-indice for north boundary) 
!>          - indice of segment end   (I-indice for north boundary)<br/>
!>             indices must be separated by ':' .<br/>
!>          - optionally, boundary size could be added between '(' and ')' 
!>          in the first segment defined.
!>             @note 
!>                boundary size is the same for all segments of one boundary.
!>
!>       Examples:
!>          - cn_north='index1,first1:last1(width)'
!>          - cn_north='index1(width),first1:last1|index2,first2:last2'
!>
!>       @image html  boundary_50.png 
!>       <center>@image latex boundary_50.png
!>       </center>
!>
!>    - **in_ncrs**  [@a 2]<br/> 
!>       number of point(s) with coarse value save at boundaries
!>
!>    - **ln_oneseg** [@a .TRUE.]<br/>
!>       logical to use only one segment for each boundary or not
!>
!> @subsection subout namout 
!>    the output sub-namelist parameter is :
!>
!>    - **cn_fileout** [@a bathy_merged.nc]<br/>
!>       output bathymetry filename
!>
!> <hr>
!> @author J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date Sepember, 2014 
!> - add header for user
!> @date July, 2015 
!> - extrapolate all land points
!> - add attributes with boundary string character (as in namelist)
!> @date September, 2015
!> - manage useless (dummy) variable, attributes, and dimension
!> @date October, 2016
!> - allow to choose the number of boundary point with coarse grid value.
!> - dimension to be used select from configuration file
!> @date January, 2019
!> - add url path to global attributes of output file(s)
!> @date February, 2019
!> - rename sub namelist namcrs to namsrc
!> - rename sub namelist namfin to namtgt
!> @date May, 2019
!> - create and clean file structure to avoid memory leaks
!> @date Ocober, 2019
!> - add help and version optional arguments
!>
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
PROGRAM merge_bathy

   USE netcdf                          ! nf90 library
   USE global                          ! global variable
   USE phycst                          ! physical constant
   USE kind                            ! F90 kind parameter
   USE logger                          ! log file manager
   USE fct                             ! basic useful function
   USE date                            ! date manager
   USE att                             ! attribute manager
   USE dim                             ! dimension manager
   USE var                             ! variable manager
   USE file                            ! file manager
   USE boundary                        ! boundary manager
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
   CHARACTER(LEN=lc), PARAMETER  :: cp_myname = "merge_bathy"

   ! local variable
   CHARACTER(LEN=lc)                                  :: cl_arg
   CHARACTER(LEN=lc)                                  :: cl_namelist
   CHARACTER(LEN=lc)                                  :: cl_date
   CHARACTER(LEN=lc)                                  :: cl_tmp
   CHARACTER(LEN=lc)                                  :: cl_url
   CHARACTER(LEN=lc)                                  :: cl_errormsg

   INTEGER(i4)                                        :: il_narg
   INTEGER(i4)                                        :: il_status
   INTEGER(i4)                                        :: il_fileid
   INTEGER(i4)                                        :: il_attind
   INTEGER(i4)                                        :: il_imin0
   INTEGER(i4)                                        :: il_imax0
   INTEGER(i4)                                        :: il_jmin0
   INTEGER(i4)                                        :: il_jmax0
   INTEGER(i4)                                        :: il_shift
   INTEGER(i4)      , DIMENSION(ip_maxdim)            :: il_rho
   INTEGER(i4)      , DIMENSION(2,2)                  :: il_ind

   LOGICAL                                            :: ll_exist

   REAL(dp)                                           :: dl_fill
   REAL(dp)         , DIMENSION(:,:,:,:), ALLOCATABLE :: dl_refined
   REAL(dp)         , DIMENSION(:,:,:,:), ALLOCATABLE :: dl_weight

   TYPE(TMPP)                                         :: tl_bathy0
   TYPE(TMPP)                                         :: tl_bathy1

   TYPE(TFILE)                                        :: tl_file
   TYPE(TFILE)                                        :: tl_fileout
   
   TYPE(TATT)                                         :: tl_att
   
   TYPE(TVAR)                                         :: tl_var
   TYPE(TVAR)                                         :: tl_lon
   TYPE(TVAR)                                         :: tl_lat
   
   TYPE(TDIM)       , DIMENSION(ip_maxdim)            :: tl_dim
   
   TYPE(TBDY)       , DIMENSION(ip_ncard)             :: tl_bdy

   ! loop indices
   INTEGER(i4) :: ji
   INTEGER(i4) :: jk
   INTEGER(i4) :: jl

   ! namelist variable
   ! namlog
   CHARACTER(LEN=lc)                       :: cn_logfile = 'merge_bathy.log' 
   CHARACTER(LEN=lc)                       :: cn_verbosity = 'warning' 
   INTEGER(i4)                             :: in_maxerror = 5

   ! namcfg
   CHARACTER(LEN=lc)                       :: cn_varcfg = './cfg/variable.cfg' 
   CHARACTER(LEN=lc)                       :: cn_dimcfg = './cfg/dimension.cfg'
   CHARACTER(LEN=lc)                       :: cn_dumcfg = './cfg/dummy.cfg'

   ! namsrc
   CHARACTER(LEN=lc)                       :: cn_bathy0 = '' 
   INTEGER(i4)                             :: in_perio0 = -1

   ! namtgt
   CHARACTER(LEN=lc)                       :: cn_bathy1 = '' 
   INTEGER(i4)                             :: in_perio1 = -1

!   ! namvar
!   CHARACTER(LEN=lc), DIMENSION(ip_maxvar) :: cn_varinfo = ''

   ! namnst
   INTEGER(i4)                             :: in_rhoi  = 1
   INTEGER(i4)                             :: in_rhoj  = 1

   ! nambdy
   LOGICAL                                 :: ln_north = .TRUE.
   LOGICAL                                 :: ln_south = .TRUE.
   LOGICAL                                 :: ln_east  = .TRUE.
   LOGICAL                                 :: ln_west  = .TRUE.
   CHARACTER(LEN=lc)                       :: cn_north = ''
   CHARACTER(LEN=lc)                       :: cn_south = ''
   CHARACTER(LEN=lc)                       :: cn_east  = ''
   CHARACTER(LEN=lc)                       :: cn_west  = ''
   INTEGER(i4)                             :: in_ncrs  = 2
   LOGICAL                                 :: ln_oneseg= .TRUE.

   ! namout
   CHARACTER(LEN=lc)                       :: cn_fileout = 'bathy_merged.nc' 
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
   &  cn_bathy0,  &      !< bathymetry file
   &  in_perio0          !< periodicity index
   
   NAMELIST /namtgt/ &   !< target/fine grid namelist
   &  cn_bathy1,     &   !< bathymetry file
   &  in_perio1          !< periodicity index
 
!   NAMELIST /namvar/ &  !< variable namelist
!   &  cn_varinfo        !< list of variable and interpolation 
!                        !< method to be used. 
!                        !< (ex: 'votemper|linear','vosaline|cubic' ) 
   
   NAMELIST /namnst/ &  !< nesting namelist
   &  in_rhoi,    &     !< refinement factor in i-direction
   &  in_rhoj           !< refinement factor in j-direction

   NAMELIST /nambdy/ &  !< boundary namelist
   &  ln_north,   &     !< use north boundary
   &  ln_south,   &     !< use south boundary
   &  ln_east ,   &     !< use east  boundary
   &  ln_west ,   &     !< use west  boundary
   &  cn_north,   &     !< north boundary indices on fine grid
   &  cn_south,   &     !< south boundary indices on fine grid
   &  cn_east ,   &     !< east  boundary indices on fine grid
   &  cn_west ,   &     !< west  boundary indices on fine grid
   &  in_ncrs,    &     !< number of point with coarse value save at boundaries 
   &  ln_oneseg         !< use only one segment for each boundary or not

   NAMELIST /namout/ &  !< output namelist
   &  cn_fileout        !< fine grid merged bathymetry file   
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

               READ( il_fileid, NML = namsrc )
               READ( il_fileid, NML = namtgt )
!               READ( il_fileid, NML = namvar )
!               ! add user change in extra information
!               CALL var_chg_extra(cn_varinfo)

               READ( il_fileid, NML = namnst )
               READ( il_fileid, NML = nambdy )

               READ( il_fileid, NML = namout )

               CLOSE( il_fileid, IOSTAT=il_status )
               CALL fct_err(il_status)
               IF( il_status /= 0 )THEN
                  CALL logger_error("MERGE BATHY: ERROR closing "//TRIM(cl_namelist))
               ENDIF

            ELSE

               WRITE(cl_errormsg,*) " ERROR : can't find "//TRIM(cl_namelist)
               CALL fct_help(cp_myname,cl_errormsg) 
               CALL EXIT(1)

            ENDIF

      END SELECT
   ENDIF

   ! open files
   IF( TRIM(cn_bathy0) /= '' )THEN
      tl_file=file_init(TRIM(cn_bathy0))
      tl_bathy0=mpp_init( tl_file, id_perio=in_perio0)
      ! clean
      CALL file_clean(tl_file)
      CALL grid_get_info(tl_bathy0)
   ELSE
      CALL logger_fatal("MERGE BATHY: can not find coarse grid bathymetry "//&
      &  "file. check namelist")
   ENDIF

   IF( TRIM(cn_bathy1) /= '' )THEN
      tl_file=file_init(TRIM(cn_bathy1))
      tl_bathy1=mpp_init( tl_file, id_perio=in_perio1)
      ! clean
      CALL file_clean(tl_file)
      CALL grid_get_info(tl_bathy1)
   ELSE
      CALL logger_fatal("MERGE BATHY: can not find fine grid bathymetry "//&
      &  "file. check namelist")
   ENDIF

   ! check
   ! check output file do not already exist
   INQUIRE(FILE=TRIM(cn_fileout), EXIST=ll_exist)
   IF( ll_exist )THEN
      CALL logger_fatal("CREATE BATHY: output file "//TRIM(cn_fileout)//&
      &  " already exist.")
   ENDIF

   ! check namelist
   ! check refinament factor
   il_rho(:)=1
   IF( in_rhoi < 1 .OR. in_rhoj < 1 )THEN
      CALL logger_error("MERGE BATHY: invalid refinement factor."//&
      &  " check namelist "//TRIM(cl_namelist))
   ELSE
      il_rho(jp_I)=in_rhoi
      il_rho(jp_J)=in_rhoj
   ENDIF

   ! check domain indices
   ! compute coarse grid indices around fine grid
   il_ind(:,:)=grid_get_coarse_index(tl_bathy0, tl_bathy1, &
   &                                 id_rho=il_rho(:) )

   il_imin0=il_ind(1,1) ; il_imax0=il_ind(1,2)
   il_jmin0=il_ind(2,1) ; il_jmax0=il_ind(2,2)

   ! check domain validity
   CALL grid_check_dom(tl_bathy0, il_imin0, il_imax0, il_jmin0, il_jmax0)

   ! check coincidence between coarse and fine grid
   CALL grid_check_coincidence( tl_bathy0, tl_bathy1, &
   &                            il_imin0, il_imax0, &
   &                            il_jmin0, il_jmax0, &
   &                            il_rho(:) )

   ! open mpp files
   CALL iom_mpp_open(tl_bathy1)

   ! read or compute boundary
   tl_var=iom_mpp_read_var(tl_bathy1,'Bathymetry')

   ! close mpp files
   CALL iom_mpp_close(tl_bathy1)

   tl_bdy(:)=boundary_init(tl_var, ln_north, ln_south, ln_east, ln_west, &
   &                               cn_north, cn_south, cn_east, cn_west, &
   &                               ln_oneseg ) 

   ! get boundary on coarse grid
   ! define refined bathymetry array (for coarse grid)
   dl_fill=tl_var%d_fill
   ALLOCATE( dl_refined(tl_var%t_dim(1)%i_len, &
   &                    tl_var%t_dim(2)%i_len, &
   &                    tl_var%t_dim(3)%i_len, &
   &                    tl_var%t_dim(4)%i_len) )

   dl_refined(:,:,:,:)=dl_fill

   ! define weight array
   ALLOCATE( dl_weight(tl_var%t_dim(1)%i_len, &
   &                   tl_var%t_dim(2)%i_len, &
   &                   1,1) )

   dl_weight(:,:,:,:)=dl_fill 

   ! compute coarse grid refined bathymetry on boundary.
   DO jk=1,ip_ncard
      CALL merge_bathy_get_boundary(tl_bathy0, tl_bathy1, tl_bdy(jk), &
      &                             il_rho(:), in_ncrs,                      &
      &                             dl_refined(:,:,:,:), dl_weight(:,:,:,:), &
      &                             dl_fill)

   ENDDO

   ! merge bathy on boundary
   DO jl=1,tl_var%t_dim(4)%i_len
      DO jk=1,tl_var%t_dim(3)%i_len
         WHERE(    dl_refined(:,:,jk,jl) /= dl_fill .AND. &
         &      tl_var%d_value(:,:,jk,jl)/= tl_var%d_fill )
            tl_var%d_value(:,:,jk,jl)=  &
            &           dl_weight(:,:,1,1) *    dl_refined(:,:,jk,jl) + &
            &        (1-dl_weight(:,:,1,1))*tl_var%d_value(:,:,jk,jl)
         ELSE WHERE( dl_refined(:,:,jk,jl)== dl_fill .AND. &
         &           dl_weight(:,:,1,1)  /= dl_fill )
         ! to keep coarse grid mask on boundary
            tl_var%d_value(:,:,jk,jl)=dl_fill
         END WHERE
      ENDDO
   ENDDO

   DEALLOCATE(dl_refined)

   ! create file
   tl_fileout=file_init(TRIM(cn_fileout),id_perio=in_perio1)

   ! add dimension
   tl_dim(:)=dim_copy(tl_var%t_dim(:))

   DO ji=1,ip_maxdim
      IF( tl_dim(ji)%l_use ) CALL file_add_dim(tl_fileout, tl_dim(ji))
   ENDDO

   ! add variables
   IF( ALL( tl_dim(1:2)%l_use ) )THEN
      ! open mpp files
      CALL iom_mpp_open(tl_bathy1)

      ! add longitude
      tl_lon=iom_mpp_read_var(tl_bathy1,'longitude')
      CALL file_add_var(tl_fileout, tl_lon)
      CALL var_clean(tl_lon)

      ! add latitude
      tl_lat=iom_mpp_read_var(tl_bathy1,'latitude')
      CALL file_add_var(tl_fileout, tl_lat)
      CALL var_clean(tl_lat)

      ! close mpp files
      CALL iom_mpp_close(tl_bathy1)      
   ENDIF

   CALL file_add_var(tl_fileout, tl_var)
   CALL var_clean(tl_var)

   ! only 2 first dimension to be used
   tl_dim(:)=dim_copy(tl_fileout%t_dim(:))
   tl_dim(3:4)%l_use=.FALSE.
   tl_var=var_init('weight',dl_weight(:,:,:,:),td_dim=tl_dim(:),dd_fill=dl_fill)
   CALL file_add_var(tl_fileout, tl_var)
   CALL var_clean(tl_var)

   ! add some attribute
   tl_att=att_init("Created_by","SIREN merge_bathy")
   CALL file_add_att(tl_fileout, tl_att)

   !add source url
   cl_url=fct_split(fct_split(cp_url,2,'$'),2,'URL:')
   tl_att=att_init("SIREN_url",cl_url)
   CALL file_add_att(tl_fileout, tl_att)

   ! add date of creation
   cl_date=date_print(date_now())
   tl_att=att_init("Creation_date",cl_date)
   CALL file_add_att(tl_fileout, tl_att)

   tl_att=att_init("coarse_grid_source_file",TRIM(fct_basename(tl_bathy0%c_name)))
   CALL file_add_att(tl_fileout, tl_att)

   tl_att=att_init("fine_grid_source_file",TRIM(fct_basename(tl_bathy1%c_name)))
   CALL file_add_att(tl_fileout, tl_att)

   ! add attribute periodicity
   il_attind=0
   IF( ASSOCIATED(tl_fileout%t_att) )THEN
      il_attind=att_get_index(tl_fileout%t_att(:),'periodicity')
   ENDIF
   IF( tl_bathy1%i_perio >= 0 .AND. il_attind == 0 )THEN
      tl_att=att_init('periodicity',tl_bathy1%i_perio)
      CALL file_add_att(tl_fileout,tl_att)
   ENDIF

   il_attind=0
   IF( ASSOCIATED(tl_fileout%t_att) )THEN
      il_attind=att_get_index(tl_fileout%t_att(:),'ew_overlap')
   ENDIF
   IF( tl_bathy1%i_ew >= 0 .AND. il_attind == 0 )THEN
      tl_att=att_init('ew_overlap',tl_bathy1%i_ew)
      CALL file_add_att(tl_fileout,tl_att)
   ENDIF


   IF( tl_bdy(jp_north)%l_use )THEN
      ! add shift on north boundary
      ! boundary compute on T point but express on U or V point
      il_shift=1

      cl_tmp=TRIM(fct_str(tl_bdy(jp_north)%t_seg(1)%i_index-il_shift))//','//&
         &   TRIM(fct_str(tl_bdy(jp_north)%t_seg(1)%i_first))//':'//&
         &   TRIM(fct_str(tl_bdy(jp_north)%t_seg(1)%i_last))//&
         &   '('//TRIM(fct_str(tl_bdy(jp_north)%t_seg(1)%i_width))//')'
      DO ji=2,tl_bdy(jp_north)%i_nseg
         cl_tmp=TRIM(cl_tmp)//'|'//&
            &   TRIM(fct_str(tl_bdy(jp_north)%t_seg(ji)%i_index-il_shift))//','//&
            &   TRIM(fct_str(tl_bdy(jp_north)%t_seg(ji)%i_first))//':'//&
            &   TRIM(fct_str(tl_bdy(jp_north)%t_seg(ji)%i_last))
      ENDDO
      tl_att=att_init("bdy_north",TRIM(cl_tmp))
      CALL file_add_att(tl_fileout, tl_att)
   ENDIF

   IF( tl_bdy(jp_south)%l_use )THEN
      
      cl_tmp=TRIM(fct_str(tl_bdy(jp_south)%t_seg(1)%i_index))//','//&
         &   TRIM(fct_str(tl_bdy(jp_south)%t_seg(1)%i_first))//':'//&
         &   TRIM(fct_str(tl_bdy(jp_south)%t_seg(1)%i_last))//&
         &   '('//TRIM(fct_str(tl_bdy(jp_south)%t_seg(1)%i_width))//')'
      DO ji=2,tl_bdy(jp_south)%i_nseg
         cl_tmp=TRIM(cl_tmp)//'|'//&
            &   TRIM(fct_str(tl_bdy(jp_south)%t_seg(ji)%i_index))//','//&
            &   TRIM(fct_str(tl_bdy(jp_south)%t_seg(ji)%i_first))//':'//&
            &   TRIM(fct_str(tl_bdy(jp_south)%t_seg(ji)%i_last))
      ENDDO

      tl_att=att_init("bdy_south",TRIM(cl_tmp))
      CALL file_add_att(tl_fileout, tl_att)
   ENDIF

   IF( tl_bdy(jp_east)%l_use )THEN
      ! add shift on east boundary
      ! boundary compute on T point but express on U or V point
      il_shift=1

      cl_tmp=TRIM(fct_str(tl_bdy(jp_east)%t_seg(1)%i_index-il_shift))//','//&
         &   TRIM(fct_str(tl_bdy(jp_east)%t_seg(1)%i_first))//':'//&
         &   TRIM(fct_str(tl_bdy(jp_east)%t_seg(1)%i_last))//&
         &   '('//TRIM(fct_str(tl_bdy(jp_east)%t_seg(1)%i_width))//')'
      DO ji=2,tl_bdy(jp_east)%i_nseg
         cl_tmp=TRIM(cl_tmp)//'|'//&
            &   TRIM(fct_str(tl_bdy(jp_east)%t_seg(ji)%i_index-il_shift))//','//&
            &   TRIM(fct_str(tl_bdy(jp_east)%t_seg(ji)%i_first))//':'//&
            &   TRIM(fct_str(tl_bdy(jp_east)%t_seg(ji)%i_last))
      ENDDO

      tl_att=att_init("bdy_east",TRIM(cl_tmp))
      CALL file_add_att(tl_fileout, tl_att)
   ENDIF

   IF( tl_bdy(jp_west)%l_use )THEN

      cl_tmp=TRIM(fct_str(tl_bdy(jp_west)%t_seg(1)%i_index))//','//&
         &   TRIM(fct_str(tl_bdy(jp_west)%t_seg(1)%i_first))//':'//&
         &   TRIM(fct_str(tl_bdy(jp_west)%t_seg(1)%i_last))//&
         &   '('//TRIM(fct_str(tl_bdy(jp_west)%t_seg(1)%i_width))//')'
      DO ji=2,tl_bdy(jp_west)%i_nseg
         cl_tmp=TRIM(cl_tmp)//'|'//&
            &   TRIM(fct_str(tl_bdy(jp_west)%t_seg(ji)%i_index))//','//&
            &   TRIM(fct_str(tl_bdy(jp_west)%t_seg(ji)%i_first))//':'//&
            &   TRIM(fct_str(tl_bdy(jp_west)%t_seg(ji)%i_last))
      ENDDO

      tl_att=att_init("bdy_west",TRIM(cl_tmp))
      CALL file_add_att(tl_fileout, tl_att)
   ENDIF

   ! create file
   CALL iom_create(tl_fileout)

   ! write file
   CALL iom_write_file(tl_fileout)

   ! close file
   CALL iom_close(tl_fileout)

   ! clean
   CALL att_clean(tl_att)
   CALL file_clean(tl_fileout)
   CALL mpp_clean(tl_bathy1)
   CALL mpp_clean(tl_bathy0)
   DEALLOCATE(dl_weight)
   CALL boundary_clean(tl_bdy(:))
   CALL var_clean_extra()

   ! close log file
   CALL logger_footer()
   CALL logger_close()

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE merge_bathy_get_boundary(td_bathy0, td_bathy1, td_bdy, &
         &                             id_rho, id_ncrs,              &
         &                             dd_refined, dd_weight, dd_fill)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute refined bathymetry on boundary from coarse grid.
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_bathy0       coarse grid bathymetry file structure 
   !> @param[in] td_bathy1       fine grid bathymetry file structure
   !> @param[in] td_bdy          boundary structure
   !> @param[in] id_rho          array of refinement factor
   !> @param[in] id_ncrs         number of point with coarse value save at boundaries
   !> @param[inout] dd_refined   array of refined bathymetry 
   !> @param[inout] dd_weight    array of weight
   !> @param[in] dd_fill         fillValue
   !>
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP)                     , INTENT(IN   ) :: td_bathy0
      TYPE(TMPP)                     , INTENT(IN   ) :: td_bathy1
      TYPE(TBDY)                     , INTENT(IN   ) :: td_bdy
      INTEGER(i4), DIMENSION(:)      , INTENT(IN   ) :: id_rho
      INTEGER(i4)                    , INTENT(IN   ) :: id_ncrs
      REAL(dp)   , DIMENSION(:,:,:,:), INTENT(INOUT) :: dd_refined
      REAL(dp)   , DIMENSION(:,:,:,:), INTENT(INOUT) :: dd_weight
      REAL(dp)                       , INTENT(IN   ) :: dd_fill  

      ! local variable
      INTEGER(i4) :: il_imin1
      INTEGER(i4) :: il_imax1
      INTEGER(i4) :: il_jmin1
      INTEGER(i4) :: il_jmax1
      
      INTEGER(i4) :: il_imin0
      INTEGER(i4) :: il_imax0
      INTEGER(i4) :: il_jmin0
      INTEGER(i4) :: il_jmax0

      INTEGER(i4) :: il_width

      INTEGER(i4), DIMENSION(2,2)         :: il_offset
      INTEGER(i4), DIMENSION(2,2)         :: il_ind

      REAL(dp)   , DIMENSION(:)      , ALLOCATABLE :: dl_tmp1d
      REAL(dp)   , DIMENSION(:,:)    , ALLOCATABLE :: dl_tmp2d
      REAL(dp)   , DIMENSION(:,:)    , ALLOCATABLE :: dl_wseg

      TYPE(TVAR) :: tl_var0
      TYPE(TVAR) :: tl_lon1
      TYPE(TVAR) :: tl_lat1

      TYPE(TMPP)  :: tl_bathy1
      TYPE(TMPP)  :: tl_bathy0

      TYPE(TDOM)  :: tl_dom1
      TYPE(TDOM)  :: tl_dom0

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      IF( td_bdy%l_use )THEN
         DO jl=1,td_bdy%i_nseg
            ! get boundary definition
            SELECT CASE(TRIM(td_bdy%c_card))
            CASE('north')

               il_imin1=td_bdy%t_seg(jl)%i_first
               il_imax1=td_bdy%t_seg(jl)%i_last 
               il_jmin1=td_bdy%t_seg(jl)%i_index-(td_bdy%t_seg(jl)%i_width-1)
               il_jmax1=td_bdy%t_seg(jl)%i_index

               ! do not used grid point to compute 
               ! boundaries indices (cf create_boundary)
               ! as Bathymetry always on T point

            CASE('south')

               il_imin1=td_bdy%t_seg(jl)%i_first
               il_imax1=td_bdy%t_seg(jl)%i_last 
               il_jmin1=td_bdy%t_seg(jl)%i_index
               il_jmax1=td_bdy%t_seg(jl)%i_index+(td_bdy%t_seg(jl)%i_width-1)

            CASE('east')

               il_imin1=td_bdy%t_seg(jl)%i_index-(td_bdy%t_seg(jl)%i_width-1)
               il_imax1=td_bdy%t_seg(jl)%i_index
               il_jmin1=td_bdy%t_seg(jl)%i_first
               il_jmax1=td_bdy%t_seg(jl)%i_last 

               ! do not used grid point to compute 
               ! boundaries indices (cf create_boundary)
               ! as Bathymetry always on T point

            CASE('west')

               il_imin1=td_bdy%t_seg(jl)%i_index
               il_imax1=td_bdy%t_seg(jl)%i_index+(td_bdy%t_seg(jl)%i_width-1)
               il_jmin1=td_bdy%t_seg(jl)%i_first
               il_jmax1=td_bdy%t_seg(jl)%i_last 

            END SELECT

            ! -read fine grid domain
            tl_bathy1=mpp_copy(td_bathy1)

            ! compute domain
            tl_dom1=dom_init( tl_bathy1,         &
            &                 il_imin1, il_imax1,&
            &                 il_jmin1, il_jmax1,&
            &                 TRIM(td_bdy%c_card))

            ! add extra band to fine grid domain (if possible)
            ! to avoid dimension of one and so be able to compute offset
            CALL dom_add_extra(tl_dom1, id_rho(jp_I), id_rho(jp_J))

            ! open mpp files over domain
            CALL iom_dom_open(tl_bathy1, tl_dom1)

            ! read variable value on domain
            tl_lon1=iom_dom_read_var(tl_bathy1,'longitude',tl_dom1)
            tl_lat1=iom_dom_read_var(tl_bathy1,'latitude' ,tl_dom1)

            ! close mpp files
            CALL iom_dom_close(tl_bathy1)

            ! clean structure
            CALL mpp_clean(tl_bathy1)

            ! get coarse grid indices
            il_ind(:,:)=grid_get_coarse_index(td_bathy0, tl_lon1, tl_lat1, &
            &                                 id_rho=id_rho(:))

            il_imin0=il_ind(1,1)
            il_imax0=il_ind(1,2)

            il_jmin0=il_ind(2,1)
            il_jmax0=il_ind(2,2)

            ! read coarse grid bathymetry on domain
            tl_bathy0=mpp_copy(td_bathy0)

            ! compute domain
            tl_dom0=dom_init( tl_bathy0,         &
            &                 il_imin0, il_imax0,&
            &                 il_jmin0, il_jmax0 )

            il_offset(:,:)= grid_get_fine_offset(tl_bathy0,         &
            &                                    il_imin0, il_jmin0,&
            &                                    il_imax0, il_jmax0,&
            &                                    tl_lon1%d_value(:,:,1,1), &
            &                                    tl_lat1%d_value(:,:,1,1), &
            &                                    id_rho=id_rho(:))

            ! clean
            CALL var_clean(tl_lon1)
            CALL var_clean(tl_lat1)

            ! add extra band (if possible) to compute interpolation
            CALL dom_add_extra(tl_dom0)

            ! open mpp files over domain
            CALL iom_dom_open(tl_bathy0, tl_dom0)

            ! read variable value on domain
            tl_var0=iom_dom_read_var(tl_bathy0,'Bathymetry',tl_dom0)

            ! force to use nearest interpolation
            tl_var0%c_interp(1)='nearest'

            ! close mpp files
            CALL iom_dom_close(tl_bathy0)

            ! clean structure
            CALL mpp_clean(tl_bathy0)

            ! interpolate variable
            CALL merge_bathy_interp( tl_var0,         &
            &                        id_rho(:),       &
            &                        il_offset(:,:) )

            ! remove extraband added to domain
            CALL dom_del_extra( tl_var0, tl_dom0, id_rho(:) )

            ! remove extraband added to domain
            CALL dom_clean_extra( tl_dom0 )

            ! remove extraband added to fine grid domain
            CALL dom_del_extra( tl_var0, tl_dom1 )

            ! remove extraband added to fine grid domain
            CALL dom_clean_extra( tl_dom1 )

            ! fill refined array
            dd_refined( il_imin1:il_imax1, &
            &           il_jmin1:il_jmax1, &
            &           :,: )= tl_var0%d_value(:,:,:,:)

            ! clean
            CALL var_clean(tl_var0)

            ! compute weight function
            ALLOCATE( dl_tmp1d(td_bdy%t_seg(jl)%i_width) )

            SELECT CASE(TRIM(td_bdy%c_card))
            CASE('north')


               ! save n coarse point
               il_width=td_bdy%t_seg(jl)%i_width-id_ncrs
               ! compute "distance"
               dl_tmp1d(:)=(/(ji,ji=il_width,1,-1),(0,ji=1,id_ncrs)/)

               ! compute weight on segment
               dl_tmp1d(:)= 0.5 + 0.5*COS( (dp_pi*dl_tmp1d(:)) / &
               &                           (il_width) )

               ALLOCATE( dl_wseg(tl_dom1%t_dim(1)%i_len, &
               &                 tl_dom1%t_dim(2)%i_len) )
               dl_wseg(:,:)=dd_fill
               dl_wseg(:,:)=SPREAD( dl_tmp1d(:), &
               &                    DIM=1,       &
               &                    NCOPIES=tl_dom1%t_dim(1)%i_len )

            CASE('south')

               ! save n coarse point
               il_width=td_bdy%t_seg(jl)%i_width-id_ncrs
               ! compute "distance"
               dl_tmp1d(:)=(/(0,ji=1,id_ncrs),(ji,ji=1,il_width)/)

               ! compute weight on segment
               dl_tmp1d(:)= 0.5 + 0.5*COS( (dp_pi*dl_tmp1d(:)) / &
               &                           (il_width) )

               ALLOCATE( dl_wseg(tl_dom1%t_dim(1)%i_len, &
               &                 tl_dom1%t_dim(2)%i_len) )
               dl_wseg(:,:)=dd_fill
               dl_wseg(:,:)=SPREAD( dl_tmp1d(:), &
               &                    DIM=1,       &
               &                    NCOPIES=tl_dom1%t_dim(1)%i_len )

            CASE('east')

               ! save n coarse point
               il_width=td_bdy%t_seg(jl)%i_width-id_ncrs
               ! compute "distance"
               dl_tmp1d(:)=(/(ji,ji=il_width,1,-1),(0,ji=1,id_ncrs)/)

               ! compute weight on segment
               dl_tmp1d(:)= 0.5 + 0.5*COS( (dp_pi*dl_tmp1d(:)) / &
               &                           (il_width) )

               ALLOCATE( dl_wseg(tl_dom1%t_dim(1)%i_len, &
               &                 tl_dom1%t_dim(2)%i_len) )
               dl_wseg(:,:)=dd_fill
               dl_wseg(:,:)=SPREAD( dl_tmp1d(:), &
               &                    DIM=2,       &
               &                    NCOPIES=tl_dom1%t_dim(2)%i_len )

            CASE('west')

               ! save n coarse point
               il_width=td_bdy%t_seg(jl)%i_width-id_ncrs
               ! compute "distance"
               dl_tmp1d(:)=(/(0,ji=1,id_ncrs),(ji,ji=1,il_width)/)

               ! compute weight on segment
               dl_tmp1d(:)= 0.5 + 0.5*COS( (dp_pi*dl_tmp1d(:)) / &
               &                           (il_width) )

               ALLOCATE( dl_wseg(tl_dom1%t_dim(1)%i_len, &
               &                 tl_dom1%t_dim(2)%i_len) )
               dl_wseg(:,:)=dd_fill
               dl_wseg(:,:)=SPREAD( dl_tmp1d(:), &
               &                    DIM=2,       &
               &                    NCOPIES=tl_dom1%t_dim(2)%i_len )

            END SELECT

            DEALLOCATE( dl_tmp1d )

            ! fill weight array
            ALLOCATE( dl_tmp2d( tl_dom1%t_dim(1)%i_len, &
            &                   tl_dom1%t_dim(2)%i_len) )

            dl_tmp2d(:,:)=dd_weight( il_imin1:il_imax1, &
            &                        il_jmin1:il_jmax1, &
            &                        1,1 )

            WHERE( dl_tmp2d(:,:) == dd_fill )
               dl_tmp2d(:,:)= dl_wseg(:,:)
            ELSE WHERE
               dl_tmp2d(:,:)= MAX( dl_tmp2d(:,:), dl_wseg(:,:) )
            END WHERE

            dd_weight( il_imin1:il_imax1, &
            &          il_jmin1:il_jmax1, &
            &          1,1 ) = dl_tmp2d(:,:)

            DEALLOCATE( dl_wseg )
            DEALLOCATE( dl_tmp2d )

         ENDDO
      ENDIF
   END SUBROUTINE merge_bathy_get_boundary
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE merge_bathy_interp(td_var, id_rho, id_offset, id_iext, id_jext)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine interpolate variable.
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] id_rho    array of refinment factor
   !> @param[in] id_offset array of offset between fine and coarse grid
   !> @param[in] id_iext   i-direction size of extra bands (default=im_minext) 
   !> @param[in] id_jext   j-direction size of extra bands (default=im_minext)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) ,                 INTENT(INOUT) :: td_var
      INTEGER(i4), DIMENSION(:)  , INTENT(IN   ) :: id_rho
      INTEGER(i4), DIMENSION(:,:), INTENT(IN   ) :: id_offset
      INTEGER(i4),                 INTENT(IN   ), OPTIONAL :: id_iext
      INTEGER(i4),                 INTENT(IN   ), OPTIONAL :: id_jext

      ! local variable
      TYPE(TVAR)  :: tl_mask

      INTEGER(i1), DIMENSION(:,:,:,:), ALLOCATABLE :: bl_mask

      INTEGER(i4) :: il_iext
      INTEGER(i4) :: il_jext

      ! loop indices
      !----------------------------------------------------------------

      !WARNING: two extrabands are required for cubic interpolation
      il_iext=3
      IF( PRESENT(id_iext) ) il_iext=id_iext

      il_jext=3
      IF( PRESENT(id_jext) ) il_jext=id_jext

      IF( il_iext < 2 .AND. td_var%c_interp(1) == 'cubic' )THEN
         CALL logger_warn("MERGE BATHY INTERP: at least extrapolation "//&
         &  "on two points are required with cubic interpolation ")
         il_iext=2
      ENDIF

      IF( il_jext < 2 .AND. td_var%c_interp(1) == 'cubic' )THEN
         CALL logger_warn("MERGE BATHY INTERP: at least extrapolation "//&
         &  "on two points are required with cubic interpolation ")
         il_jext=2
      ENDIF

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
      CASE('U','V','F')
         CALL logger_fatal("MERGE BATHY INTERP: can not computed "//&
         &                 "interpolation on "//TRIM(td_var%c_point)//&
         &                 " grid point (variable "//TRIM(td_var%c_name)//&
         &                 "). check namelist.")
      END SELECT

      DEALLOCATE(bl_mask)

      ! interpolate mask
      CALL interp_fill_value( tl_mask, id_rho(:),  &
      &                       id_offset=id_offset(:,:) )

      ! work on variable
      ! add extraband
      CALL extrap_add_extrabands(td_var, il_iext, il_iext)

      ! extrapolate variable
      CALL extrap_fill_value( td_var )

      ! interpolate Bathymetry
      CALL interp_fill_value( td_var, id_rho(:), &
      &                       id_offset=id_offset(:,:) )

      ! remove extraband
      CALL extrap_del_extrabands(td_var, il_iext*id_rho(jp_I), il_jext*id_rho(jp_J))

      ! keep original mask 
      WHERE( tl_mask%d_value(:,:,:,:) == 0 )
         td_var%d_value(:,:,:,:)=td_var%d_fill
      END WHERE

   END SUBROUTINE merge_bathy_interp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END PROGRAM merge_bathy
