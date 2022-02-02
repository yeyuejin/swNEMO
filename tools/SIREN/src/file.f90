!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
!> @brief 
!> This module manage file structure.
!>
!> @details
!>    define type TFILE:<br/>
!> @code
!>    TYPE(TFILE) :: tl_file
!> @endcode
!>
!>    to initialize a file structure:<br/>
!> @code
!>    tl_file=file_init(cd_file [,cd_type] [,ld_wrt] [,cd_grid])
!%    tl_file=file_init(cd_file [,cd_type] [,ld_wrt] [,id_ew] [,id_perio] [,id_pivot] [,cd_grid])
!> @endcode
!>       - cd_file is the file name
!>       - cd_type is the type of the file ('cdf', 'dimg') [optional]
!>       - ld_wrt  file in write mode or not [optional]
!%       - id_ew is the number of point for east-west overlap [optional]
!%       - id_perio is the NEMO periodicity index [optional]
!%       - id_pivot is the NEMO pivot point index F(0),T(1) [optional]
!>       - cd_grid is the grid type (default 'ARAKAWA-C')
!>
!>    to get file name:<br/>
!>    - tl_file\%c_name
!>
!>    to get file id (units):<br/>
!>    - tl_file\%i_id
!>    
!>    to get the type of the file (cdf, cdf4, dimg):<br/>
!>    - tl_file\%c_type
!> 
!>    to know if file was open in write mode:<br/>
!>    - tl_file\%l_wrt
!>
!>    to get the record length of the file:<br/>
!>    - tl_file\%i_recl
!>
!>    Files variables<br/>
!>    to get the number of variable in the file:<br/>
!>    - tl_file\%i_nvar
!>
!>    to get the array of variable structure associated to the file:<br/>
!>    - tl_file\%t_var(:)
!>
!>    Files attributes<br/>
!>    to get the nmber of global attributes of the file:<br/>
!>    - tl_file\%i_natt
!>
!>    to get the array of attributes structure associated to the file:<br/>
!>    - tl_file\%t_att(:)
!>
!>    Files dimensions<br/>
!>    to get the number of dimension used in the file:<br/>
!>    - tl_file\%i_ndim
!>
!>    to get the array of dimension structure (4 elts) associated to the
!>    file:<br/>
!>    - tl_file\%t_dim(:)
!>
!>    to print information about file structure:<br/>
!> @code
!>    CALL file_print(td_file)
!> @endcode
!>
!>    to clean file structure:<br/>
!> @code
!>    CALL file_clean(td_file)
!> @endcode
!>
!>    to add a global attribute structure in file structure:<br/>
!> @code
!>    CALL file_add_att(td_file, td_att)
!> @endcode
!>       - td_att is an attribute structure
!>
!>    to add a dimension structure in file structure:<br/>
!> @code
!>    CALL file_add_dim(td_file, td_dim)
!> @endcode
!>       - td_dim is a dimension structure
!>
!>    to add a variable structure in file structure:<br/>
!> @code
!>    CALL file_add_var(td_file, td_var)
!> @endcode
!>       - td_var is a variable structure
!>
!>    to delete a global attribute structure in file structure:<br/>
!> @code
!>    CALL file_del_att(td_file, td_att)
!> @endcode
!>       - td_att is an attribute structure
!>
!>    to delete a dimension structure in file structure:<br/>
!> @code
!>    CALL file_del_dim(td_file, td_dim)
!> @endcode
!>       - td_dim is a dimension structure
!>
!>    to delete a variable structure in file structure:<br/>
!> @code
!>    CALL file_del_var(td_file, td_var)
!> @endcode
!>       - td_var is a variable structure
!>
!>    to overwrite one attribute structure in file structure:<br/>
!> @code
!>    CALL file_move_att(td_file, td_att)
!> @endcode
!>       - td_att is an attribute structure
!>
!>    to  overwrite one dimension strucutre in file structure:<br/>
!> @code
!>    CALL file_move_dim(td_file, td_dim)
!> @endcode
!>       - td_dim is a dimension structure
!>
!>    to overwrite one variable  structure in file structure:<br/>
!> @code
!>    CALL file_move_var(td_file, td_var)
!> @endcode
!>       - td_var is a variable structure
!> 
!>    to check if file and variable structure share same dimension:<br/>
!> @code
!>    ll_check_dim = file_check_var_dim(td_file, td_var)
!> @endcode
!>       - td_var is a variable structure
!>
!> @author
!> J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date November, 2014 
!> - Fix memory leaks bug
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE file

   USE kind                            ! F90 kind parameter
   USE global                          ! global variable
   USE fct                             ! basic useful function
   USE logger                          ! log file manager
   USE dim                             ! dimension manager
   USE att                             ! attribute manager
   USE var                             ! variable manager

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! type and variable
   PUBLIC :: TFILE   !< file structure

   ! function and subroutine
   PUBLIC :: file_copy           !< copy file structure
   PUBLIC :: file_print          !< print information about file structure
   PUBLIC :: file_clean          !< clean file structure
   PUBLIC :: file_init           !< initialize file structure
   PUBLIC :: file_add_att        !< add one attribute structure in file structure
   PUBLIC :: file_add_var        !< add one variable  structure in file structure
   PUBLIC :: file_add_dim        !< add one dimension strucutre in file structure
   PUBLIC :: file_del_att        !< delete one attribute structure of file structure
   PUBLIC :: file_del_var        !< delete one variable  structure of file structure
   PUBLIC :: file_del_dim        !< delete one dimension strucutre of file structure
   PUBLIC :: file_move_att       !< overwrite one attribute structure in file structure
   PUBLIC :: file_move_var       !< overwrite one variable  structure in file structure
   PUBLIC :: file_move_dim       !< overwrite one dimension strucutre in file structure
   PUBLIC :: file_check_var_dim  !< check if file and variable structure use same dimension.
   PUBLIC :: file_get_type       !< get type of file
   PUBLIC :: file_get_id         !< get file id
   PUBLIC :: file_rename         !< rename file name
   PUBLIC :: file_add_suffix     !< add suffix to file name
 
   PRIVATE :: file__clean_unit    ! clean file structure
   PRIVATE :: file__clean_arr     ! clean array of file structure
   PRIVATE :: file__del_var_name ! delete a variable structure in file structure, given variable name or standard name
   PRIVATE :: file__del_var_str  ! delete a variable structure in file structure, given variable structure
   PRIVATE :: file__del_att_name ! delete a attribute structure in file structure, given attribute name
   PRIVATE :: file__del_att_str  ! delete a attribute structure in file structure, given attribute structure
   PRIVATE :: file__get_number   ! get number in file name without suffix
   PRIVATE :: file__get_suffix   ! get suffix of file name
   PRIVATE :: file__copy_unit    ! copy file structure
   PRIVATE :: file__copy_arr     ! copy array of file structure
   PRIVATE :: file__rename_char  ! rename file name, given processor number.
   PRIVATE :: file__rename_str   ! rename file name, given file structure.

   TYPE TFILE !< file structure

      ! general 
      CHARACTER(LEN=lc)                 :: c_name = ""       !< file name
      CHARACTER(LEN=lc)                 :: c_type = ""       !< type of the file (cdf, cdf4, dimg)
      INTEGER(i4)                       :: i_id   = 0        !< file id
      LOGICAL                           :: l_wrt  = .FALSE.  !< read or write mode
      INTEGER(i4)                       :: i_nvar = 0        !< number of variable
      TYPE(TVAR), DIMENSION(:), POINTER :: t_var  => NULL()  !< file variables

      CHARACTER(LEN=lc)                 :: c_grid = 'ARAKAWA-C' !< grid type

      INTEGER(i4)                       :: i_ew    =-1       !< east-west overlap
      INTEGER(i4)                       :: i_perio =-1       !< NEMO periodicity index
      INTEGER(i4)                       :: i_pivot =-1       !< NEMO pivot point index F(0),T(1)

      INTEGER(i4)                       :: i_depthid = 0     !< variable id of depth
      INTEGER(i4)                       :: i_timeid  = 0     !< variable id of time

      ! netcdf file
      INTEGER(i4)                       :: i_ndim  = 0       !< number of dimensions used in the file
      INTEGER(i4)                       :: i_natt  = 0       !< number of global attributes in the file
      INTEGER(i4)                       :: i_uldid = 0       !< id of the unlimited dimension in the file
      LOGICAL                           :: l_def   = .FALSE. !< define mode or not
      TYPE(TATT), DIMENSION(:), POINTER :: t_att   => NULL() !< global attributes
      TYPE(TDIM), DIMENSION(ip_maxdim)  :: t_dim             !< dimension structure
      
      ! dimg file
      INTEGER(i4)                       :: i_recl = 0        !< record length (binary file)
      INTEGER(i4)                       :: i_n0d  = 0        !< number of scalar variable
      INTEGER(i4)                       :: i_n1d  = 0        !< number of 1D variable
      INTEGER(i4)                       :: i_n2d  = 0        !< number of 2D variable
      INTEGER(i4)                       :: i_n3d  = 0        !< number of 3D variable
      INTEGER(i4)                       :: i_rhd  = 0        !< record of the header infos (last record)

      ! mpp
      ! only use for massively parallel processing
      INTEGER(i4)                       :: i_pid  = -1       !< processor id (start to 1)
      INTEGER(i4)                       :: i_impp = 0        !< i-indexes for mpp-subdomain left bottom
      INTEGER(i4)                       :: i_jmpp = 0        !< j-indexes for mpp-subdomain left bottom
      INTEGER(i4)                       :: i_lci  = 0        !< i-dimensions of subdomain
      INTEGER(i4)                       :: i_lcj  = 0        !< j-dimensions of subdomain
      INTEGER(i4)                       :: i_ldi  = 0        !< first indoor i-indices
      INTEGER(i4)                       :: i_ldj  = 0        !< first indoor j-indices
      INTEGER(i4)                       :: i_lei  = 0        !< last  indoor i-indices
      INTEGER(i4)                       :: i_lej  = 0        !< last  indoor j-indices

      LOGICAL                           :: l_ctr  = .FALSE.  !< domain is on border
      LOGICAL                           :: l_use  = .FALSE.  !< domain is used

      ! only use to draw domain decomposition when initialize with mpp_init
      INTEGER(i4)                       :: i_iind = 0        !< i-direction indices
      INTEGER(i4)                       :: i_jind = 0        !< j-direction indices

   END TYPE TFILE

   INTERFACE file_clean
      MODULE PROCEDURE file__clean_unit
      MODULE PROCEDURE file__clean_arr
   END INTERFACE file_clean

   INTERFACE file_del_var
      MODULE PROCEDURE file__del_var_name
      MODULE PROCEDURE file__del_var_str
   END INTERFACE file_del_var

   INTERFACE file_del_att
      MODULE PROCEDURE file__del_att_name
      MODULE PROCEDURE file__del_att_str
   END INTERFACE file_del_att
   
   INTERFACE file_rename
      MODULE PROCEDURE file__rename_char
      MODULE PROCEDURE file__rename_str
   END INTERFACE file_rename

    INTERFACE file_copy
      MODULE PROCEDURE file__copy_unit   
      MODULE PROCEDURE file__copy_arr    
   END INTERFACE

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION file__copy_unit(td_file) &
         & RESULT (tf_file)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine copy file structure in another one
   !> @details 
   !> file variable and attribute value are copied in a temporary array, 
   !> so input and output file structure value do not point on the same 
   !> "memory cell", and so on are independant. 
   !>
   !> @note new file is assume to be closed.
   !>
   !> @warning do not use on the output of a function who create or read an
   !> structure (ex: tl_file=file_copy(file_init()) is forbidden).
   !> This will create memory leaks.
   !> @warning to avoid infinite loop, do not use any function inside 
   !> this subroutine
   !>   
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date November, 2014
   !> - use function instead of overload assignment operator 
   !> (to avoid memory leak)
   !> @date January, 2019
   !> - clean variable structure
   !> 
   !> @param[in] td_file  file structure
   !> @return copy of input file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TFILE), INTENT(IN) :: td_file

      ! function
      TYPE(TFILE)             :: tf_file

      ! local variable
      TYPE(TVAR) :: tl_var
      TYPE(TATT) :: tl_att

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      CALL logger_trace("FILE COPY: file "//TRIM(td_file%c_name) )

      ! copy file variable
      tf_file%c_name = TRIM(td_file%c_name)
      tf_file%c_type = TRIM(td_file%c_type)
      ! file1 should be closed even if file2 is opened right now
      tf_file%i_id   = 0
      tf_file%l_wrt  = td_file%l_wrt
      tf_file%i_nvar = td_file%i_nvar

      tf_file%c_grid = td_file%c_grid

      tf_file%i_ew   = td_file%i_ew
      tf_file%i_perio= td_file%i_perio
      tf_file%i_pivot= td_file%i_pivot

      tf_file%i_depthid = td_file%i_depthid
      tf_file%i_timeid  = td_file%i_timeid

      ! copy variable structure
      IF( ASSOCIATED(tf_file%t_var) )THEN
         CALL var_clean(tf_file%t_var(:))
         DEALLOCATE(tf_file%t_var)
      ENDIF
      IF( ASSOCIATED(td_file%t_var) .AND. tf_file%i_nvar > 0 )THEN
         ALLOCATE( tf_file%t_var(tf_file%i_nvar) )
         DO ji=1,tf_file%i_nvar
            tl_var = var_copy(td_file%t_var(ji))
            tf_file%t_var(ji) = var_copy(tl_var)
            ! clean
            CALL var_clean(tl_var)
         ENDDO
      ENDIF
      
      ! copy netcdf variable
      tf_file%i_ndim   = td_file%i_ndim
      tf_file%i_natt   = td_file%i_natt
      tf_file%i_uldid  = td_file%i_uldid
      tf_file%l_def    = td_file%l_def

      ! copy dimension
      tf_file%t_dim(:) = dim_copy(td_file%t_dim(:))
      
      ! copy attribute structure
      IF( ASSOCIATED(tf_file%t_att) )THEN
         CALL att_clean(tf_file%t_att(:))
         DEALLOCATE(tf_file%t_att)
      ENDIF
      IF( ASSOCIATED(td_file%t_att) .AND. tf_file%i_natt > 0 )THEN
         ALLOCATE( tf_file%t_att(tf_file%i_natt) )
         DO ji=1,tf_file%i_natt
            tl_att = att_copy(td_file%t_att(ji))
            tf_file%t_att(ji) = att_copy(tl_att)
         ENDDO
      ENDIF

      ! clean
      CALL att_clean(tl_att)

      ! copy dimg variable
      tf_file%i_recl = td_file%i_recl
      tf_file%i_n0d  = td_file%i_n0d
      tf_file%i_n1d  = td_file%i_n1d
      tf_file%i_n2d  = td_file%i_n2d
      tf_file%i_n3d  = td_file%i_n3d 
      tf_file%i_rhd  = td_file%i_rhd
      
      ! copy mpp variable
      tf_file%i_pid  = td_file%i_pid
      tf_file%i_impp = td_file%i_impp
      tf_file%i_jmpp = td_file%i_jmpp
      tf_file%i_lci  = td_file%i_lci
      tf_file%i_lcj  = td_file%i_lcj
      tf_file%i_ldi  = td_file%i_ldi
      tf_file%i_ldj  = td_file%i_ldj
      tf_file%i_lei  = td_file%i_lei
      tf_file%i_lej  = td_file%i_lej
      tf_file%l_ctr  = td_file%l_ctr
      tf_file%l_use  = td_file%l_use
      tf_file%i_iind = td_file%i_iind
      tf_file%i_jind = td_file%i_jind

   END FUNCTION file__copy_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION file__copy_arr(td_file) &
         & RESULT (tf_file)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine copy a array of file structure in another one
   !> @details 
   !> file variable and attribute value are copied in a temporary array, 
   !> so input and output file structure value do not point on the same 
   !> "memory cell", and so on are independant. 
   !>
   !> @note new file is assume to be closed.
   !>
   !> @warning do not use on the output of a function who create or read an
   !> structure (ex: tl_file=file_copy(file_init()) is forbidden).
   !> This will create memory leaks.
   !> @warning to avoid infinite loop, do not use any function inside 
   !> this subroutine
   !>   
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date November, 2014
   !> - use function instead of overload assignment operator 
   !> (to avoid memory leak)
   !>
   !> @param[in] td_file  file structure
   !> @return copy of input array of file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TFILE), DIMENSION(:), INTENT(IN   ) :: td_file

      ! function
      TYPE(TFILE), DIMENSION(SIZE(td_file(:))) :: tf_file

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      DO ji=1,SIZE(td_file(:))
         tf_file(ji)=file_copy(td_file(ji))
      ENDDO

   END FUNCTION file__copy_arr
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION file_init(cd_file, cd_type, ld_wrt, &
         &                        id_ew, id_perio, id_pivot,&
         &                        cd_grid) &
         & RESULT (tf_file)
   !-------------------------------------------------------------------
   !> @brief This function initialize file structure.<br/> 
   !> @details
   !> If cd_type is not specify, check if file name include '.nc' or
   !> '.dimg'<br/>
   !> Optionally, you could specify:<br/>
   !> - write mode (default .FALSE., ld_wrt)
   !> - East-West overlap (id_ew)
   !> - NEMO periodicity index (id_perio)
   !> - NEMO pivot point index F(0),T(1) (id_pivot)
   !> - grid type (default: 'ARAKAWA-C')
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_file   file name
   !> @param[in] cd_type   file type ('cdf', 'dimg')
   !> @param[in] ld_wrt    write mode (default .FALSE.)
   !> @param[in] id_ew     east-west overlap
   !> @param[in] id_perio  NEMO periodicity index
   !> @param[in] id_pivot  NEMO pivot point index F(0),T(1)
   !> @param[in] cd_grid   grid type (default 'ARAKAWA-C')
   !> @return file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=*), INTENT(IN) :: cd_file
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: cd_type
      LOGICAL         , INTENT(IN), OPTIONAL :: ld_wrt
      INTEGER(i4)     , INTENT(IN), OPTIONAL :: id_ew
      INTEGER(i4)     , INTENT(IN), OPTIONAL :: id_perio
      INTEGER(i4)     , INTENT(IN), OPTIONAL :: id_pivot
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: cd_grid

      ! function
      TYPE(TFILE)                  :: tf_file

      ! local variable
      TYPE(TATT)  :: tl_att
      !----------------------------------------------------------------

      ! clean file
      CALL file_clean(tf_file)

      tf_file%c_name=TRIM(ADJUSTL(cd_file))
      CALL logger_trace("FILE INIT: initialize file "//TRIM(tf_file%c_name))

      ! check type
      IF( PRESENT(cd_type) )THEN
         SELECT CASE(TRIM(cd_type))
            CASE('cdf')
               tf_file%c_type='cdf'
            CASE('dimg')
               tf_file%c_type='dimg'
            CASE DEFAULT
               CALL logger_error( " FILE INIT: can't initialize file "//&
               &               TRIM(tf_file%c_name)//" : type unknown " )
         END SELECT
      ELSE
         CALL logger_debug("FILE INIT: look for file type "//TRIM(tf_file%c_name))
         tf_file%c_type=TRIM(file_get_type(cd_file))
      ENDIF

      ! create some global attribute
      IF( TRIM(tf_file%c_type) == 'cdf' )THEN
         tl_att=att_init("Conventions","CF-1.5")
         CALL file_add_att(tf_file,tl_att)
      ENDIF
     
      tl_att=att_init("Grid",TRIM(tf_file%c_grid))
      CALL file_add_att(tf_file,tl_att)

      IF( PRESENT(ld_wrt) )THEN
         tf_file%l_wrt=ld_wrt
      ENDIF

      IF( PRESENT(id_ew) )THEN
         tf_file%i_ew=id_ew
         IF( id_ew >= 0 )THEN
            tl_att=att_init('ew_overlap',id_ew)
            CALL file_move_att(tf_file, tl_att)
         ENDIF
      ENDIF

      IF( PRESENT(id_perio) )THEN
         tf_file%i_perio=id_perio
         IF( id_perio >= 0 )THEN
            tl_att=att_init('periodicity',id_perio)
            CALL file_move_att(tf_file, tl_att)
         ENDIF
      ENDIF

      IF( PRESENT(id_pivot) )THEN
         tf_file%i_pivot=id_pivot
         IF( id_pivot > 0 )THEN
            tl_att=att_init('pivot_point',id_pivot)
            CALL file_move_att(tf_file, tl_att)
         ENDIF
      ENDIF

      IF( PRESENT(cd_grid) )THEN
         tf_file%c_grid=cd_grid
      ENDIF

      ! clean
      CALL att_clean(tl_att)

   END FUNCTION file_init
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION file_get_type(cd_file) &
         & RESULT (cf_type)
   !-------------------------------------------------------------------
   !> @brief 
   !> This function get type of file, given file name.
   !> @details
   !> Actually it get suffix of the file name, and compare it to 'nc', 'cdf' or
   !> 'dimg'<br/>
   !> If no suffix or suffix not identify, we assume file is dimg
   !
   !> @details
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - netcdf4 files identify as netcdf file 
   !>
   !> @param[in] cd_file   file name
   !> @return type of file
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=*), INTENT(IN) :: cd_file

      ! function
      CHARACTER(LEN=lc)            :: cf_type

      !local variable
      CHARACTER(LEN=lc) :: cl_suffix
      !----------------------------------------------------------------

      cl_suffix=file__get_suffix(cd_file)
      SELECT CASE( TRIM(fct_lower(cl_suffix)) )
         CASE('.nc','.cdf','.nc4')
            CALL logger_debug(" FILE GET TYPE: file "//TRIM(cd_file)//" is cdf")
            ! Warning : type could be change to cdf4 when opening file.
            cf_type='cdf'
         CASE('.dimg')
            CALL logger_debug(" FILE GET TYPE: file "//TRIM(cd_file)//" is dimg" )
            cf_type='dimg'
         CASE DEFAULT
            CALL logger_warn(" FILE GET TYPE: type unknown, we assume file: "//&
            &              TRIM(cd_file)//" is dimg ")
            cf_type='dimg'
      END SELECT

   END FUNCTION file_get_type
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION file_check_var_dim(td_file, td_var, ld_chklen) &
         & RESULT (lf_dim)
   !-------------------------------------------------------------------
   !> @brief This function check that variable dimension to be used 
   !> of both variable and file structure are convenient (axis, length).
   !
   !> @details
   !> optionaly you could choose to not check length
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2017
   !> - add option to not check dimension length
   !
   !> @param[in] td_file   file structure
   !> @param[in] td_var    variable structure
   !> @param[in] ld_chklen check length
   !> @return true if dimension of variable and file structure agree
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(IN) :: td_file
      TYPE(TVAR),  INTENT(IN) :: td_var
      LOGICAL,     INTENT(IN), OPTIONAL :: ld_chklen

      ! function
      LOGICAL                 :: lf_dim

      ! local variable
      CHARACTER(LEN=lc) :: cl_dim
      LOGICAL           :: ll_error
      LOGICAL           :: ll_warn 
      LOGICAL           :: ll_chklen
      LOGICAL           :: ll_use
      LOGICAL           :: ll_len

      INTEGER(i4)       :: il_ind

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      lf_dim=.TRUE.

      CALL logger_debug( " FILE CHECK VAR DIM: check: "//TRIM(td_var%c_name) )
      ! check dimension length
      ll_chklen=.TRUE.
      IF( PRESENT(ld_chklen) ) ll_chklen=ld_chklen

      ! check used dimension 
      ll_error=.FALSE.
      ll_warn=.FALSE.
      DO ji=1,ip_maxdim
         il_ind=dim_get_index( td_file%t_dim(:), &
         &                     TRIM(td_var%t_dim(ji)%c_name), &
         &                     TRIM(td_var%t_dim(ji)%c_sname))

         IF( il_ind /= 0 )THEN
            ll_use=(td_var%t_dim(ji)%l_use .AND. td_file%t_dim(il_ind)%l_use)

            ll_len=.TRUE.
            IF( ll_chklen )THEN
               ! check dimension length
               ll_len=(td_var%t_dim(ji)%i_len == td_file%t_dim(il_ind)%i_len)
            ENDIF
            IF( ll_use .AND. .NOT. ll_len )THEN
               IF( INDEX( TRIM(td_var%c_axis), &
               &          TRIM(fct_upper(td_var%t_dim(ji)%c_name))) == 0 )THEN
                  ll_warn=.TRUE.
               ELSE
                  ll_error=.TRUE.
               ENDIF
            ENDIF
         ENDIF
      ENDDO

      IF( ll_error )THEN

         cl_dim='(/'
         DO ji = 1, td_file%i_ndim
            IF( td_file%t_dim(ji)%l_use )THEN
               cl_dim=TRIM(cl_dim)//&
               &  TRIM(fct_upper(td_file%t_dim(ji)%c_sname))//':'//&
               &  TRIM(fct_str(td_file%t_dim(ji)%i_len))//','
            ENDIF
         ENDDO
         cl_dim=TRIM(cl_dim)//'/)'
         CALL logger_debug( " file dimension: "//TRIM(cl_dim) )

         cl_dim='(/'
         DO ji = 1, td_var%i_ndim
            IF( td_var%t_dim(ji)%l_use )THEN
               cl_dim=TRIM(cl_dim)//&
               &  TRIM(fct_upper(td_var%t_dim(ji)%c_sname))//':'//&
               &  TRIM(fct_str(td_var%t_dim(ji)%i_len))//','
            ENDIF
         ENDDO
         cl_dim=TRIM(cl_dim)//'/)'
         CALL logger_debug( " variable dimension: "//TRIM(cl_dim) )

         lf_dim=.FALSE.

         CALL logger_error( &
         &  " FILE CHECK VAR DIM: variable and file dimension differ"//&
         &  " for variable "//TRIM(td_var%c_name)//&
         &  " and file "//TRIM(td_file%c_name))

      ELSEIF( ll_warn )THEN
         CALL logger_warn( &
         &  " FILE CHECK VAR DIM: variable and file dimension differ"//&
         &  " for variable "//TRIM(td_var%c_name)//&
         &  " and file "//TRIM(td_file%c_name)//". you should use"//&
         &  " var_check_dim to remove useless dimension.")
      ELSE

         IF( td_var%i_ndim >  td_file%i_ndim )THEN
            CALL logger_info("FILE CHECK VAR DIM: variable "//&
            &  TRIM(td_var%c_name)//" use more dimension than file "//&
            &  TRIM(td_file%c_name)//" do until now.")
         ENDIF

      ENDIF

   END FUNCTION file_check_var_dim
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE file_add_var(td_file, td_var)
   !-------------------------------------------------------------------
   !> @brief This subroutine add a variable structure in a file structure.<br/>
   !> Do not overwrite, if variable already in file structure.
   !
   !> @note variable value is suppose to be ordered ('x','y','z','t')
   !
   !> @details
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014
   !> - add dimension in file if need be
   !> - do not reorder dimension from variable, before put in file
   !> @date September, 2015
   !> - check variable dimension expected
   !> @date January, 2019
   !> - clean variable structure
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] td_var       variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file
      TYPE(TVAR) , INTENT(INOUT) :: td_var

      ! local variable
      INTEGER(i4) :: il_status
      !INTEGER(i4) :: il_rec
      INTEGER(i4) :: il_ind

      TYPE(TVAR), DIMENSION(:), ALLOCATABLE :: tl_var

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! check if file opened
      IF( TRIM(td_file%c_name) == '' )THEN

         CALL logger_debug( " FILE ADD VAR: you should have used file_init before "//&
         & "running file_add_var" )
         CALL logger_error( " FILE ADD VAR: structure file unknown" )

      ELSE
         ! check if variable exist
         IF( TRIM(td_var%c_name) == '' .AND. &
         &   TRIM(td_var%c_stdname) == '' )THEN
            CALL logger_error(" FILE ADD VAR: variable without name ")
         ELSE
            ! check if variable already in file structure
            il_ind=0
            IF( ASSOCIATED(td_file%t_var) )THEN
               il_ind=var_get_index( td_file%t_var(:), td_var%c_name,   &
               &                                       td_var%c_stdname )
            ENDIF
            CALL logger_debug( &
            &  " FILE ADD VAR: ind "//TRIM(fct_str(il_ind)) )
            IF( il_ind /= 0 )THEN

               CALL logger_error( &
               &  " FILE ADD VAR: variable "//TRIM(td_var%c_name)//&
               &  ", standard name "//TRIM(td_var%c_stdname)//&
               &  ", already in file "//TRIM(td_file%c_name) )

               DO ji=1,td_file%i_nvar
                  CALL logger_debug( " ADD VAR: in file : &
                  &  variable "//TRIM(td_file%t_var(ji)%c_name)//&
                  &  ", standard name "//TRIM(td_file%t_var(ji)%c_stdname) )
               ENDDO

            ELSE

               CALL logger_debug( &
               &  " FILE ADD VAR: add variable "//TRIM(td_var%c_name)//&
               &  ", standard name "//TRIM(td_var%c_stdname)//&
               &  ", in file "//TRIM(td_file%c_name) )

               ! check used dimension 
               IF( file_check_var_dim(td_file, td_var) )THEN

                  ! check variable dimension expected
                  CALL var_check_dim(td_var)

                  ! update dimension if need be
                  DO ji=1,ip_maxdim
                     IF( td_var%t_dim(ji)%l_use .AND. &
                     &   .NOT. td_file%t_dim(ji)%l_use )THEN
                        CALL file_add_dim(td_file,td_var%t_dim(ji))
                     ENDIF
                  ENDDO

                  ! get index of new variable
                  SELECT CASE(td_var%i_ndim)
                     CASE(0)
                        il_ind=td_file%i_n0d+1
                        !il_rec=0
                     CASE(1)
                        il_ind=td_file%i_n0d+td_file%i_n1d+1
                        !il_rec=1
                     CASE(2)
                        il_ind=td_file%i_n0d+td_file%i_n1d+td_file%i_n2d+1
                        !il_rec=1
                     CASE(3,4)
                        il_ind=td_file%i_n0d+td_file%i_n1d+td_file%i_n2d+td_file%i_n3d+1
                        !il_rec=td_file%t_dim(3)%i_len
                  END SELECT

                  IF( td_file%i_nvar > 0 )THEN
                  ! already other variable in file structure
                     ALLOCATE( tl_var(td_file%i_nvar), stat=il_status )
                     IF(il_status /= 0 )THEN

                        CALL logger_error( &
                        &  " FILE ADD VAR: not enough space to put variables "//&
                        &  "from "//TRIM(td_file%c_name)//&
                        &  " in variable structure")

                     ELSE

                        ! save temporary variable of file structure
                        tl_var(:)=var_copy(td_file%t_var(:))

                        CALL var_clean( td_file%t_var(:) )
                        DEALLOCATE(td_file%t_var)
                        ALLOCATE( td_file%t_var(td_file%i_nvar+1), &
                        &         stat=il_status)
                        IF(il_status /= 0 )THEN

                           CALL logger_error( &
                           &  " FILE ADD VAR: not enough space to put variable "//&
                           &  "in file structure "//TRIM(td_file%c_name) )

                        ENDIF

                        ! copy variable in file before
                        ! variable with less than or equal dimension that new variable
                        IF( il_ind > 1 )THEN
                           td_file%t_var( 1:il_ind-1 ) = var_copy(tl_var(1:il_ind-1))
                        ENDIF

                        IF( il_ind < td_file%i_nvar+1 )THEN
                           ! variable with more dimension than new variable
                           td_file%t_var( il_ind+1 : td_file%i_nvar+1 ) = &
                           &        var_copy( tl_var(il_ind : td_file%i_nvar) )
                        ENDIF

                        ! clean
                        CALL var_clean(tl_var(:))
                     ENDIF
                     DEALLOCATE(tl_var)

                  ELSE
                  ! no variable in file structure
                     IF( ASSOCIATED(td_file%t_var) )THEN
                        CALL var_clean(td_file%t_var(:))
                        DEALLOCATE(td_file%t_var)
                     ENDIF
                     ALLOCATE( td_file%t_var(td_file%i_nvar+1), stat=il_status )
                     IF(il_status /= 0 )THEN

                        CALL logger_error( &
                        &  " FILE ADD VAR: not enough space to put variable "//&
                        &  "in file structure "//TRIM(td_file%c_name) )

                     ENDIF

                  ENDIF

                  ! add new variable in array of variable
                  ALLOCATE( tl_var(1), stat=il_status )
                  IF(il_status /= 0 )THEN

                     CALL logger_error( &
                     &  " FILE ADD VAR: not enough space to put variables from "//&
                     &  TRIM(td_var%c_name)//" in variable structure")

                  ELSE
                     tl_var(1)=var_copy(td_var)
                     ! remove old id
                     tl_var(1)%i_id=0

                     ! update dimension name in new variable
                     tl_var(1)%t_dim(:)%c_name = td_file%t_dim(:)%c_name
                  
                     ! add new variable
                     td_file%t_var(il_ind)=var_copy(tl_var(1))

                     ! update number of variable
                     td_file%i_nvar=td_file%i_nvar+1
                     SELECT CASE(tl_var(1)%i_ndim)
                        CASE(0)
                           td_file%i_n0d=td_file%i_n0d+1
                        CASE(1)
                           td_file%i_n1d=td_file%i_n1d+1
                        CASE(2)
                           td_file%i_n2d=td_file%i_n2d+1
                        CASE(3,4)
                           td_file%i_n3d=td_file%i_n3d+1
                     END SELECT

                     ! update variable id
                     td_file%t_var(il_ind)%i_id=var_get_unit(td_file%t_var(:))

                     ! update dimension used
                     td_file%t_dim(:)%l_use=.FALSE.
                     DO ji=1,ip_maxdim
                        IF( ANY(td_file%t_var(:)%t_dim(ji)%l_use) )THEN
                           td_file%t_dim(ji)%l_use=.TRUE.
                        ENDIF
                     ENDDO

                     ! update number of dimension
                     td_file%i_ndim=COUNT(td_file%t_dim(:)%l_use)

                     ! clean
                     CALL var_clean( tl_var(:) )
                  ENDIF
                  DEALLOCATE(tl_var)

               ENDIF
            ENDIF
         ENDIF
      ENDIF

   END SUBROUTINE file_add_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE file__del_var_name(td_file, cd_name)
   !-------------------------------------------------------------------
   !> @brief This subroutine delete a variable structure 
   !> in file structure, given variable name or standard name.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date February, 2015 
   !> - define local variable structure to avoid mistake with pointer
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] cd_name      variable name or standard name
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE)     , INTENT(INOUT) :: td_file
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_name

      ! local variable
      INTEGER(i4)       :: il_ind
      TYPE(TVAR)        :: tl_var
      !----------------------------------------------------------------

      ! check if file opened
      IF( TRIM(td_file%c_name) == '' )THEN

         CALL logger_error( " FILE DEL VAR NAME: file structure unknown ")
         CALL logger_debug( " FILE DEL VAR NAME: you should have used file_init before "//&
         & "running file_del_var" )

      ELSE

         IF( td_file%i_nvar /= 0 )THEN

            ! get the variable index, in file variable structure
            il_ind=0
            IF( ASSOCIATED(td_file%t_var) )THEN
               il_ind=var_get_index(td_file%t_var(:), cd_name )
            ENDIF

            IF( il_ind /= 0 )THEN
   
               tl_var=var_copy(td_file%t_var(il_ind))
               CALL file_del_var(td_file, tl_var)
               ! clean
               CALL var_clean(tl_var)
            ELSE

               CALL logger_debug( &
               &  " FILE DEL VAR NAME: there is no variable with name or "//&
               &  "standard name "//TRIM(cd_name)//" in file "//&
               &  TRIM(td_file%c_name))

            ENDIF

         ELSE
            CALL logger_debug( " FILE DEL VAR NAME: "//&
            &        "no variable associated to file "//&
            &        TRIM(td_file%c_name) )
         ENDIF

      ENDIF

   END SUBROUTINE file__del_var_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE file__del_var_str(td_file, td_var)
   !-------------------------------------------------------------------
   !> @brief This subroutine delete a variable structure 
   !> in file structure, given variable structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - clean variable structure
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] td_var       variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file
      TYPE(TVAR),  INTENT(IN)    :: td_var

      ! local variable
      INTEGER(i4) :: il_status
      INTEGER(i4) :: il_ind
      INTEGER(i4) :: il_rec
      TYPE(TVAR), DIMENSION(:), ALLOCATABLE :: tl_var

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! check if file opened
      IF( TRIM(td_file%c_name) == '' )THEN

         CALL logger_error( " FILE DEL VAR: file structure unknown ")
         CALL logger_debug( " FILE DEL VAR: you should have used "//&
         &  "file_init before running file_del_var" )      

      ELSE

         ! check if variable is member of a file
         IF( td_var%l_file )THEN
            CALL logger_warn( &
            &  " FILE DEL VAR: variable "//TRIM(td_var%c_name)//&
            &  ", belong to file "//TRIM(td_file%c_name)//&
            &  " and can not be removed.")
         ELSE
            ! check if variable already in file structure
            il_ind=0
            IF( ASSOCIATED(td_file%t_var) )THEN
               il_ind=var_get_index( td_file%t_var(:), td_var%c_name, &
               &                                       td_var%c_stdname )
            ENDIF

            IF( il_ind == 0 )THEN

               CALL logger_warn( "FILE DEL VAR: no variable "//&
               &     TRIM(td_var%c_name)//", in file "//TRIM(td_file%c_name) )

               DO ji=1,td_file%i_nvar
                  CALL logger_debug( "FILE DEL VAR: in file "//&
                  &  TRIM(td_file%t_var(ji)%c_name)//", standard name "//&
                  &  TRIM(td_file%t_var(ji)%c_stdname) )
               ENDDO

            ELSE
               
               CALL logger_trace( "FILE DEL VAR: delete variable "//&
               &  TRIM(td_var%c_name)//", from file "//TRIM(td_file%c_name) )

               ALLOCATE( tl_var(td_file%i_nvar-1), stat=il_status )
               IF(il_status /= 0 )THEN

                  CALL logger_error( &
                  &  " FILE DEL VAR: not enough space to put variables from "//&
                  &  TRIM(td_file%c_name)//" in temporary variable structure")

               ELSE

                  ! save temporary variable's file structure
                  IF( il_ind > 1 )THEN
                     tl_var(1:il_ind-1)=var_copy(td_file%t_var(1:il_ind-1))
                  ENDIF

                  IF( il_ind < td_file%i_nvar )THEN
                     tl_var(il_ind:)=var_copy(td_file%t_var(il_ind+1:))
                  ENDIF

                  ! new number of variable in file
                  td_file%i_nvar=td_file%i_nvar-1
                  SELECT CASE(td_var%i_ndim)
                     CASE(0)
                        td_file%i_n0d=td_file%i_n0d-1
                        il_rec=0
                     CASE(1)
                        td_file%i_n1d=td_file%i_n1d-1
                        il_rec=1
                     CASE(2)
                        td_file%i_n2d=td_file%i_n2d-1
                        il_rec=1
                     CASE(3,4)
                        td_file%i_n3d=td_file%i_n3d-1
                        il_rec=td_file%t_dim(3)%i_len
                  END SELECT

                  CALL var_clean( td_file%t_var(:) )
                  DEALLOCATE(td_file%t_var)

                  IF( td_file%i_nvar > 0 )THEN
                     ALLOCATE( td_file%t_var(td_file%i_nvar), stat=il_status )
                     IF(il_status /= 0 )THEN

                        CALL logger_error( " FILE DEL VAR: not enough space"//&
                        &  "to put variables in file structure "//&
                        &  TRIM(td_file%c_name) )

                     ENDIF

                     ! copy attribute in file before
                     td_file%t_var(:)=var_copy(tl_var(:))

                     ! update dimension used
                     td_file%t_dim(:)%l_use=.FALSE.
                     DO ji=1,ip_maxdim
                        IF( ANY(td_file%t_var(:)%t_dim(ji)%l_use) )THEN
                           td_file%t_dim(ji)%l_use=.TRUE.
                        ENDIF
                     ENDDO

                     ! update number of dimension
                     td_file%i_ndim=COUNT(td_file%t_dim(:)%l_use)

                  ENDIF

                  ! clean
                  CALL var_clean(tl_var(:))
               ENDIF 
               DEALLOCATE(tl_var)

            ENDIF
         ENDIF
      ENDIF

   END SUBROUTINE file__del_var_str
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE file_move_var(td_file, td_var)
   !-------------------------------------------------------------------
   !> @brief This subroutine overwrite variable structure 
   !> in file structure.
   !
   !> @warning change variable id in file structure.
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[inout] td_file   file structure
   !> @param[in] td_var       variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file
      TYPE(TVAR),  INTENT(IN)    :: td_var

      ! local variable
      TYPE(TVAR) :: tl_var
      !----------------------------------------------------------------

      ! copy variable
      tl_var=var_copy(td_var)

      ! remove variable with same name or standard name
      CALL file_del_var(td_file, tl_var)

      ! add new variable
      CALL file_add_var(td_file, tl_var)

      ! clean
      CALL var_clean(tl_var)

   END SUBROUTINE file_move_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE file_add_att(td_file, td_att)
   !-------------------------------------------------------------------
   !> @brief This subroutine add a global attribute 
   !> in a file structure.<br/>
   !> Do not overwrite, if attribute already in file structure.
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - clean attribute structure
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] td_att       attribute structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file
      TYPE(TATT),  INTENT(IN)    :: td_att

      ! local variable
      INTEGER(i4) :: il_status
      INTEGER(i4) :: il_ind
      TYPE(TATT), DIMENSION(:), ALLOCATABLE :: tl_att

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! check if file opened
      IF( TRIM(td_file%c_name) == '' )THEN

         CALL logger_error( " FILE ADD ATT: file structure unknown ")
         CALL logger_debug( " FILE ADD ATT: you should have used file_init before "//&
         & "running file_add_att" )      

      ELSE

         ! check if attribute already in file structure
         il_ind=0
         IF( ASSOCIATED(td_file%t_att) )THEN
            il_ind=att_get_index( td_file%t_att(:), td_att%c_name )
         ENDIF

         IF( il_ind /= 0 )THEN
         
            CALL logger_error( &
            &  " FILE ADD ATT: attribute "//TRIM(td_att%c_name)//&
            &  ", already in file "//TRIM(td_file%c_name) )

            DO ji=1,td_file%i_natt
               CALL logger_debug( &
               &  " FILE ADD ATT: in file "//TRIM(td_file%t_att(ji)%c_name) )
            ENDDO

         ELSE
            
            CALL logger_trace( &
            &  " FILE ADD ATT: add attribute "//TRIM(td_att%c_name)//&
            &  ", in file "//TRIM(td_file%c_name) )

            IF( td_file%i_natt > 0 )THEN
            ! already other attribute in file structure
               ALLOCATE( tl_att(td_file%i_natt), stat=il_status )
               IF(il_status /= 0 )THEN

                  CALL logger_error( &
                  &  " FILE ADD ATT: not enough space to put attributes from "//&
                  &  TRIM(td_file%c_name)//" in temporary attribute structure")

               ELSE

                  ! save temporary global attribute's file structure
                  tl_att(:)=att_copy(td_file%t_att(:))

                  CALL att_clean( td_file%t_att(:) )
                  DEALLOCATE(td_file%t_att)
                  ALLOCATE( td_file%t_att(td_file%i_natt+1), stat=il_status )
                  IF(il_status /= 0 )THEN

                     CALL logger_error( &
                     &  " FILE ADD ATT: not enough space to put attributes "//&
                     &  "in file structure "//TRIM(td_file%c_name) )

                  ENDIF

                  ! copy attribute in file before
                  td_file%t_att(1:td_file%i_natt)=att_copy(tl_att(:))

                   ! clean
                  CALL att_clean(tl_att(:))
               ENDIF
               DEALLOCATE(tl_att)

            ELSE
            ! no attribute in file structure
               IF( ASSOCIATED(td_file%t_att) )THEN
                  CALL att_clean(td_file%t_att(:))
                  DEALLOCATE(td_file%t_att)
               ENDIF

               ALLOCATE( td_file%t_att(td_file%i_natt+1), stat=il_status )
               IF(il_status /= 0 )THEN

                  CALL logger_error( &
                  &  " FILE ADD ATT: not enough space to put attributes "//&
                  &  "in file structure "//TRIM(td_file%c_name) )

               ENDIF
            ENDIF
            ! add new attribute
            td_file%t_att(td_file%i_natt+1)=att_copy(td_att)

            ! update number of attribute
            td_file%i_natt=td_file%i_natt+1
         ENDIF
      ENDIF

   END SUBROUTINE file_add_att
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE file__del_att_name(td_file, cd_name)
   !-------------------------------------------------------------------
   !> @brief This subroutine delete a global attribute structure 
   !> in file structure, given attribute name.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date February, 2015 
   !> - define local attribute structure to avoid mistake
   !> with pointer
   !> @date January, 2019
   !> - clean attribute structure
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] cd_name      attribute name
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE)     , INTENT(INOUT) :: td_file
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_name

      ! local variable
      INTEGER(i4)       :: il_ind
      TYPE(TATT)        :: tl_att
      !----------------------------------------------------------------

      ! check if file opened
      IF( TRIM(td_file%c_name) == '' )THEN

         CALL logger_error( " FILE DEL ATT NAME: file structure unknown ")
         CALL logger_debug( " FILE DEL ATT NAME: you should have "//&
         &  "used file_init before running file_del_att" )

      ELSE

         IF( td_file%i_natt /= 0 )THEN

            ! get the variable id, in file variable structure
            il_ind=0
            IF( ASSOCIATED(td_file%t_att) )THEN
               il_ind=att_get_index(td_file%t_att(:), cd_name )
            ENDIF

            IF( il_ind /= 0 )THEN
   
               tl_att=att_copy(td_file%t_att(il_ind))
               CALL file_del_att(td_file, tl_att)
               ! clean
               CALL att_clean(tl_att)
            ELSE

               CALL logger_debug( &
               &  " FILE DEL ATT NAME: there is no attribute with name "//&
               &  TRIM(cd_name)//" in file "//TRIM(td_file%c_name))

            ENDIF

         ELSE
            CALL logger_debug( " FILE DEL ATT NAME: no attribute "//&
            &  "associated to file "//TRIM(td_file%c_name) )
         ENDIF

      ENDIF

   END SUBROUTINE file__del_att_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE file__del_att_str(td_file, td_att)
   !-------------------------------------------------------------------
   !> @brief This subroutine delete a global attribute structure 
   !> from file structure, given attribute structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - clean attribute structure
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] td_att       attribute structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file
      TYPE(TATT),  INTENT(IN)    :: td_att

      ! local variable
      INTEGER(i4) :: il_status
      INTEGER(i4) :: il_ind
      TYPE(TATT), DIMENSION(:), ALLOCATABLE :: tl_att

      ! loop indices
      !----------------------------------------------------------------

      ! check if file opened
      IF( TRIM(td_file%c_name) == '' )THEN

         CALL logger_error( " FILE DEL ATT: file structure unknown ")
         CALL logger_debug( " FILE DEL ATT: you should have used "//&
         &  "file_init before running file_del_att" )      

      ELSE

         ! check if attribute already in file structure
         il_ind=0
         IF( ASSOCIATED(td_file%t_att) )THEN
            il_ind=att_get_index( td_file%t_att(:), td_att%c_name )
         ENDIF

         IF( il_ind == 0 )THEN

            CALL logger_error( &
            &  " FILE DEL ATT: no attribute "//TRIM(td_att%c_name)//&
            &  ", in file "//TRIM(td_file%c_name) )

         ELSE
            
            CALL logger_trace( &
            &  " FILE DEL ATT: del attribute "//TRIM(td_att%c_name)//&
            &  ", in file "//TRIM(td_file%c_name) )

            ALLOCATE( tl_att(td_file%i_natt-1), stat=il_status )
            IF(il_status /= 0 )THEN

               CALL logger_error( &
               &  " FILE ADD ATT: not enough space to put attributes from "//&
               &  TRIM(td_file%c_name)//" in temporary attribute structure")

            ELSE

               ! save temporary global attribute's file structure
               IF( il_ind > 1 )THEN
                  tl_att(1:il_ind-1)=att_copy(td_file%t_att(1:il_ind-1))
               ENDIF

               IF( il_ind < td_file%i_natt )THEN
                  tl_att(il_ind:)=att_copy(td_file%t_att(il_ind+1:))
               ENDIF

               CALL att_clean( td_file%t_att(:) )
               DEALLOCATE(td_file%t_att)

               ! new number of attribute in file
               td_file%i_natt=td_file%i_natt-1

               ALLOCATE( td_file%t_att(td_file%i_natt), stat=il_status )
               IF(il_status /= 0 )THEN

                  CALL logger_error( &
                  &  " FILE ADD ATT: not enough space to put attributes "//&
                  &  "in file structure "//TRIM(td_file%c_name) )

               ENDIF

               ! copy attribute in file before
               td_file%t_att(1:td_file%i_natt)=att_copy(tl_att(:))

               ! clean 
               CALL att_clean(tl_att(:))
            ENDIF 
            DEALLOCATE(tl_att)

         ENDIF
      ENDIF

   END SUBROUTINE file__del_att_str
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE file_move_att(td_file, td_att)
   !-------------------------------------------------------------------
   !> @brief This subroutine move a global attribute structure 
   !> from file structure.
   !> @warning change attribute id in file structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] td_att       attribute structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file
      TYPE(TATT),  INTENT(IN)    :: td_att

      ! local variable
      TYPE(TATT)  :: tl_att
      INTEGER(i4) :: il_ind
      !----------------------------------------------------------------

      ! copy attribute
      tl_att=att_copy(td_att)

      IF( ASSOCIATED(td_file%t_att) )THEN
         il_ind=att_get_index(td_file%t_att(:),TRIM(tl_att%c_name))
         IF( il_ind /= 0 )THEN
            ! remove attribute with same name
            CALL file_del_att(td_file, tl_att)
         ENDIF
      ENDIF

      ! add new attribute
      CALL file_add_att(td_file, tl_att)

       ! clean
       CALL att_clean(tl_att)

   END SUBROUTINE file_move_att
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE file_add_dim(td_file, td_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine add a dimension structure in file 
   !> structure.
   !> Do not overwrite, if dimension already in file structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014
   !> - do not reorder dimension, before put in file
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] td_dim       dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE)     , INTENT(INOUT) :: td_file
      TYPE(TDIM)      , INTENT(IN   ) :: td_dim

      ! local variable
      INTEGER(i4) :: il_ind

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! check if file opened
      IF( TRIM(td_file%c_name) == '' )THEN

         CALL logger_error( " FILE ADD DIM: file structure unknown ")
         CALL logger_debug( " FILE ADD DIM: you should have used "//&
         &  "file_init before running file_add_dim" )      

      ELSE

         IF( td_file%i_ndim <= ip_maxdim )THEN

            ! check if dimension already in file structure
            il_ind=dim_get_index(td_file%t_dim(:), td_dim%c_sname)
            IF( il_ind /= 0 )THEN
               IF( td_file%t_dim(il_ind)%l_use )THEN
                  CALL logger_error( &
                  &  "FILE ADD DIM: dimension "//TRIM(td_dim%c_name)//&
                  &  ", short name "//TRIM(td_dim%c_sname)//&
                  &  ", already used in file "//TRIM(td_file%c_name) )
               ELSE
                  ! replace dimension
                  td_file%t_dim(il_ind)=dim_copy(td_dim)
                  td_file%t_dim(il_ind)%i_id=il_ind
                  td_file%t_dim(il_ind)%l_use=.TRUE.
               ENDIF
            ELSE
               IF( td_file%i_ndim == ip_maxdim )THEN
                  CALL logger_error( &
                  &  "FILE ADD DIM: can not add dimension "//TRIM(td_dim%c_name)//&
                  &  ", short name "//TRIM(td_dim%c_sname)//&
                  &  ", in file "//TRIM(td_file%c_name)//". Already "//&
                  &  TRIM(fct_str(ip_maxdim))//" dimensions." )
               ELSE
                  ! search empty dimension
                  DO ji=1,ip_maxdim
                     IF( td_file%t_dim(ji)%i_id == 0 )THEN
                        il_ind=ji 
                        EXIT
                     ENDIF
                  ENDDO
 
                  ! add new dimension    
                  td_file%t_dim(il_ind)=dim_copy(td_dim)
                  ! update number of attribute
                  td_file%i_ndim=COUNT(td_file%t_dim(:)%l_use)

                  td_file%t_dim(il_ind)%i_id=td_file%i_ndim
                  td_file%t_dim(il_ind)%l_use=.TRUE.
               ENDIF               
            ENDIF

         ELSE
            CALL logger_error( &
            &  " FILE ADD DIM: too much dimension in file "//&
            &  TRIM(td_file%c_name)//" ("//TRIM(fct_str(td_file%i_ndim))//")")
         ENDIF

      ENDIF

   END SUBROUTINE file_add_dim
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE file_del_dim(td_file, td_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine delete a dimension structure in file 
   !> structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - clean dimension structure
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] td_dim       dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE)     , INTENT(INOUT) :: td_file
      TYPE(TDIM)      , INTENT(IN   ) :: td_dim

      ! local variable
      INTEGER(i4) :: il_status
      INTEGER(i4) :: il_ind

      TYPE(TDIM), DIMENSION(:), ALLOCATABLE  :: tl_dim

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! check if file opened
      IF( TRIM(td_file%c_name) == '' )THEN

         CALL logger_error( " FILE DEL DIM: file structure unknown ")
         CALL logger_debug( " FILE DEL DIM: you should have used "//&
         &  "file_init before running file_del_dim" )      

      ELSE

         ! check if dimension already in file structure
         il_ind=dim_get_index(td_file%t_dim(:), td_dim%c_sname)
         IF( il_ind == 0 )THEN

            CALL logger_error( &
            &  "FILE DEL DIM: no dimension "//TRIM(td_dim%c_name)//&
            &  ", short name "//TRIM(td_dim%c_sname)//&
            &  ", in file "//TRIM(td_file%c_name) )

         ELSE
            ALLOCATE( tl_dim(td_file%i_ndim-1), stat=il_status )
            IF(il_status /= 0 )THEN

               CALL logger_error( &
               &  "FILE DEL DIM: not enough space to put dimensions from "//&
               &  TRIM(td_file%c_name)//" in temporary dimension structure")

            ELSE            
               ! save temporary dimension's mpp structure
               tl_dim( 1 : il_ind-1 ) = dim_copy(td_file%t_dim(1 : il_ind-1))
               tl_dim( il_ind : td_file%i_ndim-1 ) = &
               &      dim_copy(td_file%t_dim(il_ind+1 : td_file%i_ndim))

               ! remove dimension from file
               CALL dim_clean(td_file%t_dim(:))
               ! copy dimension in file, except one
               td_file%t_dim(1:td_file%i_ndim)=dim_copy(tl_dim(:))

               ! update number of dimension
               td_file%i_ndim=td_file%i_ndim-1

               ! update dimension id
               DO ji=1,td_file%i_ndim
                  td_file%t_dim(ji)%i_id=ji
               ENDDO

               ! clean
               CALL dim_clean(tl_dim(:))
            ENDIF
            DEALLOCATE(tl_dim)

         ENDIF
      ENDIF

   END SUBROUTINE file_del_dim
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE file_move_dim(td_file, td_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine move a dimension structure 
   !> in file structure.
   !> @warning change dimension order in file structure. 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] td_dim       dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE)     , INTENT(INOUT) :: td_file
      TYPE(TDIM)      , INTENT(IN   ) :: td_dim

      ! local variable
      INTEGER(i4) :: il_ind
      INTEGER(i4) :: il_dimid
      !----------------------------------------------------------------
      IF( td_file%i_ndim <= ip_maxdim )THEN

         ! check if dimension already in mpp structure
         il_ind=dim_get_index(td_file%t_dim(:), td_dim%c_name, td_dim%c_sname)
         IF( il_ind /= 0 )THEN

            il_dimid=td_file%t_dim(il_ind)%i_id
            ! replace dimension
            td_file%t_dim(il_ind)=dim_copy(td_dim)
            td_file%t_dim(il_ind)%i_id=il_dimid
            td_file%t_dim(il_ind)%l_use=.TRUE.

         ELSE
            CALL file_add_dim(td_file, td_dim)
         ENDIF

      ELSE
         CALL logger_error( &
         &  "FILE MOVE DIM: too much dimension in mpp "//&
         &  TRIM(td_file%c_name)//" ("//TRIM(fct_str(td_file%i_ndim))//")")
      ENDIF

   END SUBROUTINE file_move_dim
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE file_print(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine print some information about file strucutre.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(IN) :: td_file

      ! local variable
      CHARACTER(LEN=lc) :: cl_mode

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      cl_mode='READ'
      IF( td_file%l_wrt ) cl_mode='WRITE'

      WRITE(*,'((a,a),2(/3x,a,a),4(/3x,a,i0))')&
      &  "File : ",TRIM(td_file%c_name), &
      &  " type : ",TRIM(td_file%c_type), &
      &  " mode : ",TRIM(cl_mode), &
      &  " id   : ",td_file%i_id, &
      &  " ndim : ",td_file%i_ndim, &
      &  " natt : ",td_file%i_natt, &
      &  " nvar : ",td_file%i_nvar

      SELECT CASE(TRIM(td_file%c_type))
         CASE('cdf')
            WRITE(*,'((/3x,a,a),(/3x,a,i3))')&
            &  "define mode : ",TRIM(fct_str(td_file%l_def)),&
            &  "unlimited id : ",td_file%i_uldid
         CASE('dimg')
            WRITE(*,'(5(/3x,a,i0))')&
            &  " record length : ",td_file%i_recl, &
            &  " n0d : ",td_file%i_n0d, &
            &  " n1d : ",td_file%i_n1d, &
            &  " n2d : ",td_file%i_n2d, &
            &  " n3d : ",td_file%i_n3d
      END SELECT

      ! print dimension
      IF(  td_file%i_ndim /= 0 )THEN
         WRITE(*,'(/a)') " File dimension"
         DO ji=1,ip_maxdim
            IF( td_file%t_dim(ji)%l_use )THEN
               CALL dim_print(td_file%t_dim(ji))
            ENDIF
         ENDDO
      ENDIF

      ! print global attribute
      IF( td_file%i_natt /= 0 )THEN
         WRITE(*,'(/a)') " File attribute"
         DO ji=1,td_file%i_natt
            CALL att_print(td_file%t_att(ji))
         ENDDO
      ENDIF

      ! print variable
      IF( td_file%i_nvar /= 0 )THEN
         WRITE(*,'(/a)') " File variable"
         DO ji=1,td_file%i_nvar
            CALL var_print(td_file%t_var(ji),.FALSE.)
         ENDDO
      ENDIF

   END SUBROUTINE file_print
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION file__get_suffix(cd_file) &
         & RESULT (cf_suffix)
   !-------------------------------------------------------------------
   !> @brief This function get suffix of file name.
   !> @details
   !> we assume suffix is define as alphanumeric character following the
   !> last '.' in file name.<br/>
   !> If no suffix is found, return empty character.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_file   file structure
   !> @return suffix
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=*), INTENT(IN) :: cd_file

      ! function
      CHARACTER(LEN=lc)            :: cf_suffix
   
      ! local variable
      INTEGER(i4) :: il_ind
      !----------------------------------------------------------------

      CALL logger_trace( "FILE GET SUFFIX: look for suffix in file name "//&
      &               TRIM(cd_file) )

      il_ind=INDEX(TRIM(cd_file),'.',BACK=.TRUE.)
      IF( il_ind /= 0 )THEN
         ! read number in basename
         READ( cd_file(il_ind:),'(a)' ) cf_suffix

         IF( fct_is_num(cf_suffix(2:)) )THEN
            cf_suffix=''
         ENDIF

      ELSE
         cf_suffix=''
      ENDIF

   END FUNCTION file__get_suffix
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION file__get_number(cd_file) &
         & RESULT (cf_number)
   !-------------------------------------------------------------------
   !> @brief This function get number in file name without suffix.
   !> @details
   !> Actually it get the number following the last separator.
   !> separator could be '.' or '_'.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date February, 2015 
   !> - add case to not return date (yyyymmdd) at the end of filename
   !> @date February, 2015 
   !> - add case to not return release number
   !> we assume release number only on one digit (ex : file_v3.5.nc)
   !>
   !> @param[in] cd_file   file name (without suffix)
   !> @return character file number.
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=lc), INTENT(IN) :: cd_file

      ! function
      CHARACTER(LEN=lc)             :: cf_number

      ! local variable
      INTEGER(i4) :: il_indmax
      INTEGER(i4) :: il_ind

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! get number position in file name
      il_indmax=0
      DO ji=1,ip_nsep
         il_ind=INDEX(TRIM(cd_file),TRIM(cp_sep(ji)),BACK=.TRUE.)
         IF( il_ind > il_indmax )THEN
            il_indmax=il_ind
         ENDIF
      ENDDO

      IF( il_indmax /= 0 )THEN
         ! read number in basename
         READ( cd_file(il_indmax:),'(a)' ) cf_number

         IF( .NOT. fct_is_num(cf_number(2:)) )THEN
            cf_number=''
         ELSEIF( LEN(TRIM(cf_number))-1 == 8 )THEN
            ! date case yyyymmdd
            cf_number=''
         ELSEIF( LEN(TRIM(cf_number))-1 == 1 )THEN
            ! release number case 
            cf_number=''
         ENDIF
      ELSE
         cf_number=''
      ENDIF

   END FUNCTION file__get_number
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION file__rename_char(cd_file, id_num) &
         & RESULT (cf_file)
   !-------------------------------------------------------------------
   !> @brief This function rename file name, given processor number.
   !> @details
   !> If no processor number is given, return file name without number
   !> If processor number is given, return file name with new number
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_file   file structure
   !> @param[in] id_num    processor number (start to 1)
   !> @return file name
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=*), INTENT(IN) :: cd_file
      INTEGER(i4),      INTENT(IN), OPTIONAL :: id_num

      ! function
      CHARACTER(LEN=lc)            :: cf_file

      ! local variable
      CHARACTER(LEN=lc) :: cl_suffix
      CHARACTER(LEN=lc) :: cl_file
      CHARACTER(LEN=lc) :: cl_number
      CHARACTER(LEN=lc) :: cl_base
      CHARACTER(LEN=lc) :: cl_sep
      CHARACTER(LEN=lc) :: cl_format
      INTEGER(i4)       :: il_ind
      INTEGER(i4)       :: il_numlen
      !----------------------------------------------------------------

      ! get suffix
      cl_suffix=file__get_suffix(cd_file)
      IF( TRIM(cl_suffix) /= '' )THEN
         il_ind=INDEX(TRIM(cd_file),TRIM(cl_suffix(1:1)),BACK=.TRUE.)
         cl_file=TRIM(cd_file(:il_ind-1))
      ELSE
         cl_file=TRIM(cd_file)
      ENDIF

      cl_number=file__get_number(cl_file)
      IF( TRIM(cl_number) /= '' )THEN
         il_ind=INDEX(TRIM(cl_file),TRIM(cl_number(1:1)),BACK=.TRUE.)
         cl_base=TRIM(cl_file(:il_ind-1))

         cl_sep=TRIM(cl_number(1:1))
         il_numlen=LEN(TRIM(cl_number))-1
      ELSE
         cl_base=TRIM(cl_file)
         il_numlen=4
         cl_sep='_'
      ENDIF

      IF( PRESENT(id_num) )THEN
         ! format
         WRITE(cl_format,'(a,i1.1,a,i1.1,a)') '(a,a,i',il_numlen,'.',il_numlen,',a)'
         WRITE(cf_file,cl_format) TRIM(cl_base),TRIM(cl_sep),id_num,TRIM(cl_suffix)
      ELSE
         WRITE(cf_file,'(a,a)') TRIM(cl_base),TRIM(cl_suffix)
      ENDIF
      CALL logger_trace(" FILE RENAME : "//TRIM(cf_file))

   END FUNCTION file__rename_char
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION file__rename_str(td_file, id_num) &
         & RESULT (tf_file)
   !-------------------------------------------------------------------
   !> @brief This function rename file name, given file structure.
   !> @details
   !> If no processor number is given, return file name without number
   !> I processor number is given, return file name with new number
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_file   file structure
   !> @param[in] id_num    processor number (start to 1)
   !> @return file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(IN) :: td_file
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_num

      ! function
      TYPE(TFILE)             :: tf_file

      ! local variable
      CHARACTER(LEN=lc) :: cl_name
      !----------------------------------------------------------------

      ! change name
      cl_name=TRIM( file_rename(td_file%c_name, id_num) )

      tf_file=file_init(TRIM(cl_name), TRIM(td_file%c_type))

   END FUNCTION file__rename_str
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION file_add_suffix(cd_file, cd_type) &
         & RESULT (cf_file)
   !-------------------------------------------------------------------
   !> @brief This function add suffix to file name.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_file   file structure
   !> @return file name
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=*), INTENT(IN) :: cd_file
      CHARACTER(LEN=*), INTENT(IN) :: cd_type

      ! function
      CHARACTER(LEN=lc)            :: cf_file

      ! local variable
      INTEGER(i4)       :: il_ind
      CHARACTER(LEN=lc) :: cl_file
      CHARACTER(LEN=lc) :: cl_suffix
      !----------------------------------------------------------------

      ! get suffix 
      cl_suffix=file__get_suffix(cd_file)
      IF( TRIM(cl_suffix) /= '' )THEN
         il_ind=INDEX(TRIM(cd_file),TRIM(cl_suffix(1:1)),BACK=.TRUE.)
         cl_file=TRIM(cd_file(:il_ind-1))
      ELSE
         cl_file=TRIM(cd_file)
      ENDIF

      SELECT CASE(TRIM(cd_type))
         CASE('cdf')
            cf_file=TRIM(cl_file)//TRIM(cl_suffix)
         CASE('dimg')
            IF( TRIM(cl_suffix) /= '' )THEN
               cf_file=TRIM(cl_file)//'.dimg'
            ELSE
               cf_file=TRIM(cl_file)
            ENDIF
         CASE DEFAULT
            CALL logger_error(" FILE ADD SUFFIX: type unknown "//TRIM(cd_type))
      END SELECT

   END FUNCTION file_add_suffix
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE file__clean_unit(td_file)
   !-------------------------------------------------------------------
   !> @brief 
   !>  This subroutine clean file strcuture.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Inital version
   !> @date January, 2019
   !> - nullify attribute structure inside file structure
   !> - nullify variable structure inside file structure
   !>
   !> @param[inout] td_file   file strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TFILE),  INTENT(INOUT) :: td_file

      ! local variable
      TYPE(TFILE) :: tl_file ! empty file structure

      ! loop indices
      !----------------------------------------------------------------

      CALL logger_trace( &
      &  " FILE CLEAN: reset file "//TRIM(td_file%c_name) )

      ! del attribute
      IF( ASSOCIATED( td_file%t_att ) )THEN
         CALL att_clean( td_file%t_att(:) )
         DEALLOCATE(td_file%t_att)
         NULLIFY(td_file%t_att)
      ENDIF

      ! del dimension
      IF( td_file%i_ndim /= 0 )THEN
         CALL dim_clean( td_file%t_dim(:) )
      ENDIF

      ! del variable
      IF( ASSOCIATED( td_file%t_var ) )THEN
         CALL var_clean( td_file%t_var(:) )
         DEALLOCATE(td_file%t_var)
         NULLIFY(td_file%t_var)
      ENDIF

      ! replace by empty structure
      td_file=file_copy(tl_file)

   END SUBROUTINE file__clean_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE file__clean_arr(td_file)
   !-------------------------------------------------------------------
   !> @brief 
   !>  This subroutine clean file array of file strcuture.
   !>
   !> @author J.Paul
   !> @date Marsh, 2014 - Inital version
   !>
   !> @param[inout] td_file   array file strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TFILE), DIMENSION(:), INTENT(INOUT) :: td_file

      ! local variable
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      DO ji=SIZE(td_file(:)),1,-1
         CALL file_clean(td_file(ji))
      ENDDO

   END SUBROUTINE file__clean_arr
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION file_get_id(td_file, cd_name) &
         & RESULT (if_id)
   !-------------------------------------------------------------------
   !> @brief This function return the file id, in a array of file
   !> structure,  given file name. 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_file   array of file structure
   !> @param[in] cd_name   file name
   !> @return file id in array of file structure (0 if not found)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE)     , DIMENSION(:), INTENT(IN) :: td_file
      CHARACTER(LEN=*),               INTENT(IN) :: cd_name

      ! function
      INTEGER(i4)                                :: if_id

      ! local variable
      INTEGER(i4) :: il_size

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      if_id=0
      il_size=SIZE(td_file(:))

      ! check if file is in array of file structure
      DO ji=1,il_size
         ! look for file name
         IF( fct_lower(td_file(ji)%c_name) == fct_lower(cd_name) )THEN
         
            if_id=td_file(ji)%i_id
            EXIT

         ENDIF
      ENDDO

   END FUNCTION file_get_id
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION file_get_unit(td_file) &
         & RESULT (if_unit)
   !-------------------------------------------------------------------
   !> @brief
   !> This function get the next unused unit in array of file structure.
   !> 
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[in] td_file   array of file 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TFILE), DIMENSION(:), INTENT(IN   ) :: td_file

      ! function
      INTEGER(i4)                              :: if_unit

      ! local variable
      ! loop indices
      !----------------------------------------------------------------

      if_unit=MAXVAL(td_file(:)%i_id)+1

   END FUNCTION file_get_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE file

