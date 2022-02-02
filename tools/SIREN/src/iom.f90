!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief Input/Output manager :  Library to read input files
!>
!> @details
!>    to open file:<br/>
!> @code
!>    CALL iom_open(td_file)
!> @endcode
!>       - td_file is file structure
!>
!>    to create file:<br/>
!> @code
!>    CALL iom_create(td_file)
!> @endcode
!>       - td_file is file structure
!> 
!>    to write in file:<br/>
!> @code
!>    CALL  iom_write_file(td_file)
!> @endcode
!>
!>    to close file:<br/>
!> @code
!>    CALL iom_close(tl_file)
!> @endcode
!>
!>    to read one dimension in file:<br/>
!> @code
!>    tl_dim = iom_read_dim(tl_file, id_dimid)
!> @endcode
!>    or<br/>
!> @code
!>    tl_dim = iom_read_dim(tl_file, cd_name)
!> @endcode
!>       - id_dimid is dimension id
!>       - cd_name is dimension name
!>
!>    to read variable or global attribute in file:<br/>
!> @code
!>    tl_att = iom_read_att(tl_file, id_varid, id_attid)
!> @endcode
!>    or
!> @code
!>    tl_att = iom_read_att(tl_file, id_varid, cd_attname)
!> @endcode
!>    or
!> @code
!>    tl_att = iom_read_att(tl_file, cd_varname, id_attid)
!> @endcode
!>    or
!> @code
!>    tl_att = iom_read_att(tl_file, cd_varname, cd_attname)
!> @endcode
!>       - id_varid is variable id
!>       - id_attid is attribute id
!>       - cd_attname is attribute name
!>       - cd_varname is variable name or standard name
!>    
!>    to read one variable in file:<br/>
!> @code
!>    tl_var = iom_read_var(td_file, id_varid, [id_start, id_count])
!> @endcode
!>    or
!> @code
!>    tl_var = iom_read_var(td_file, cd_name, [id_start, [id_count,]])
!> @endcode
!>       - id_varid is variabale id
!>       - cd_name is variabale name or standard name.
!>       - id_start is a integer(4) 1D array of index from which the data 
!>          values will be read [optional]
!>       - id_count is a integer(4) 1D array of the number of indices selected
!>          along each dimension [optional]
!>
!> @author
!> J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date August, 2017
!> - permit to write header and variable independantly
!>
!> @todo
!> - see lbc_lnk
!> - see goup netcdf4
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE iom

   USE netcdf                          ! nf90 library
   USE global                          ! global parameter
   USE kind                            ! F90 kind parameter
   USE fct                             ! basic useful function
   USE logger                          ! log file manager
   USE dim                             ! dimension manager
   USE att                             ! attribute manager
   USE var                             ! variable manager
   USE file                            ! file manager
   USE iom_cdf                         ! netcdf I/O manager
   USE iom_rstdimg                     ! restart dimg I/O manager

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! function and subroutine
   PUBLIC :: iom_open        !< open or create file, fill file structure
   PUBLIC :: iom_create      !< create file, fill file structure
   PUBLIC :: iom_close       !< close file
   PUBLIC :: iom_read_dim    !< read one dimension in an opened file
   PUBLIC :: iom_read_att    !< read one attribute in an opened file
   PUBLIC :: iom_read_var    !< read one variable  in an opened file   
   PUBLIC :: iom_write_file  !< write file structure contents in an opened file
   PUBLIC :: iom_write_header!< write header in an opened file
   PUBLIC :: iom_write_var   !< write variable an opened file

                                          ! read variable or global attribute in an opened file
   PRIVATE :: iom__read_att_varname_id   ! given variable name or standard name and attribute id.
   PRIVATE :: iom__read_att_varid_id     ! given variable id and attribute id.
   PRIVATE :: iom__read_att_varname_name ! given variable name or standard name, and attribute name.
   PRIVATE :: iom__read_att_varid_name   ! given variable id and attribute name.

   PRIVATE :: iom__read_dim_id            ! read one dimension in an opened file, given dimension id.
   PRIVATE :: iom__read_dim_name          ! read one dimension in an opened netcdf file, given dimension name.
   PRIVATE :: iom__read_var_id            ! read variable value in an opened file, given variable id.
   PRIVATE :: iom__read_var_name          ! read variable value in an opened file, given variable name or standard name.

   INTERFACE iom_read_var
      MODULE PROCEDURE iom__read_var_id
      MODULE PROCEDURE iom__read_var_name
   END INTERFACE iom_read_var

   INTERFACE iom_read_dim
      MODULE PROCEDURE iom__read_dim_id
      MODULE PROCEDURE iom__read_dim_name
   END INTERFACE iom_read_dim

   INTERFACE iom_read_att    !< read variable or global attribute in an opened file
      MODULE PROCEDURE iom__read_att_varname_id   !< given variable name or standard name and attribute id.
      MODULE PROCEDURE iom__read_att_varid_id     !< given variable id and attribute id.
      MODULE PROCEDURE iom__read_att_varname_name !< given variable name or standard name, and attribute name.
      MODULE PROCEDURE iom__read_att_varid_name   !< given variable id and attribute name.
   END INTERFACE iom_read_att

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_open(td_file)
   !-------------------------------------------------------------------
   !> @brief This function open a file in read or write mode
   !> @details
   !> If try to open a file in write mode that did not exist, create it.<br/>
   !> 
   !> If file exist, get information about:
   !> - the number of variables
   !> - the number of dimensions
   !> - the number of global attributes
   !> - the ID of the unlimited dimension
   !> - the file format
   !> and finally read dimensions.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_file file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT)  :: td_file
      !----------------------------------------------------------------

      ! add suffix to file name
      td_file%c_name = file_add_suffix( TRIM(td_file%c_name), &
      &                                 TRIM(td_file%c_type)  )
      ! check type
      SELECT CASE(TRIM(ADJUSTL(fct_lower(td_file%c_type))))

         CASE('cdf')
            CALL iom_cdf_open(td_file)
         !CASE('cdf4')
         CASE('dimg')
            CALL iom_rstdimg_open(td_file)
         CASE DEFAULT
            CALL logger_error("IOM OPEN: unknow type : "//TRIM(td_file%c_name))

      END SELECT

   END SUBROUTINE iom_open
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_create(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine create a file.
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_file file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT)  :: td_file

      ! local variable
      LOGICAL     :: ll_exist
      !----------------------------------------------------------------

      INQUIRE(FILE=TRIM(td_file%c_name), EXIST=ll_exist )
      IF( ll_exist )THEN
         CALL logger_fatal("IOM CREATE: can not create file "//&
         &  TRIM(td_file%c_name)//". file exist already.")
      ENDIF

      ! forced to open in write mode
      td_file%l_wrt=.TRUE.
      ! check type
      SELECT CASE(TRIM(td_file%c_type))
         CASE('cdf')
            CALL iom_cdf_open(td_file)
         CASE('dimg')
            CALL iom_rstdimg_open(td_file)
         CASE DEFAULT
            CALL logger_error( "IOM CREATE: can't create file "//&
            &               TRIM(td_file%c_name)//": type unknown " )
      END SELECT

   END SUBROUTINE iom_create
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_close(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine close file
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_file file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file
      !----------------------------------------------------------------

      ! open file
      SELECT CASE(TRIM(td_file%c_type))
         CASE('cdf')
            CALL iom_cdf_close(td_file)
         CASE('dimg')
            CALL iom_rstdimg_close(td_file)
         CASE DEFAULT
            CALL logger_debug( "IOM CLOSE: type "//TRIM(td_file%c_type))
            CALL logger_error( "IOM CLOSE: can't close file "//&
            &               TRIM(td_file%c_name)//": type unknown " )
      END SELECT

   END SUBROUTINE iom_close
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom__read_att_varname_id(td_file, cd_varname, id_attid) &
         & RESULT (tf_att)
   !-------------------------------------------------------------------
   !> @brief This function read attribute (of variable or global) in an opened 
   !> file, given variable name or standard name and attribute id.
   !> @details
   !>  - to get global attribute use 'GLOBAL' as variable name. 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_file      file structure
   !> @param[in] cd_varname   variable name. use 'GLOBAL' to read global 
   !> attribute in a file
   !> @param[in] id_attid     attribute id
   !> @return  attribute structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TFILE),       INTENT(IN) :: td_file
      CHARACTER(LEN=lc), INTENT(IN) :: cd_varname
      INTEGER(i4),       INTENT(IN) :: id_attid

      ! function
      TYPE(TATT)                    :: tf_att

      ! local variable
      INTEGER(i4) :: il_varid
      !----------------------------------------------------------------

      ! get variable id
      IF( TRIM(fct_upper(cd_varname)) == 'GLOBAL' )THEN
         il_varid=NF90_GLOBAL
      ELSE
         il_varid=var_get_id(td_file%t_var(:), cd_varname)
      ENDIF

      IF( il_varid /= 0 .OR. TRIM(fct_upper(cd_varname)) == 'GLOBAL' )THEN
         ! open file
         SELECT CASE(TRIM(td_file%c_type))
            CASE('cdf')
               tf_att=iom_read_att( td_file, il_varid, id_attid)
            CASE('dimg')
               CALL logger_warn( " IOM READ ATT: can't read attribute "//&
               &              "in dimg file : "//TRIM(td_file%c_name) )
            CASE DEFAULT
               CALL logger_error( " IOM READ ATT: can't read attribute "//&
               &    " in file "//TRIM(td_file%c_name)//" : type unknown " )
         END SELECT
      ENDIF

   END FUNCTION iom__read_att_varname_id
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom__read_att_varid_id(td_file, id_varid, id_attid) &
         & RESULT (tf_att)
   !-------------------------------------------------------------------
   !> @brief This function read attribute (of variable or global) in an opened 
   !> file, given variable id and attribute id.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_file   file structure
   !> @param[in] id_varid  variable id. use NF90_GLOBAL to read global 
   !> attribute in a file
   !> @param[in] id_attid  attribute id
   !> @return  attribute structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TFILE), INTENT(IN) :: td_file
      INTEGER(i4), INTENT(IN) :: id_varid
      INTEGER(i4), INTENT(IN) :: id_attid

      ! function
      TYPE(TATT)              :: tf_att
      !----------------------------------------------------------------

      ! open file
      SELECT CASE(TRIM(td_file%c_type))
         CASE('cdf')
            tf_att=iom_cdf_read_att(td_file, id_varid, id_attid)
         CASE('dimg')
            CALL logger_warn( " IOM READ ATT: can't read attribute in dimg file "//&
            &              TRIM(td_file%c_name) )
         CASE DEFAULT
            CALL logger_error( " IOM READ ATT: can't read attribute in file "//&
            &               TRIM(td_file%c_name)//" : type unknown " )
      END SELECT

   END FUNCTION iom__read_att_varid_id
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom__read_att_varname_name(td_file, cd_varname, cd_attname) &
         & RESULT (tf_att)
   !-------------------------------------------------------------------
   !> @brief This function read attribute (of variable or global) in an opened 
   !> file, given variable name or standard name, and attribute name.
   !> @details
   !> - to get global attribute use 'GLOBAL' as variable name.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_file      file structure
   !> @param[in] cd_varname   variable name or standard name. use 'GLOBAL' to read global
   !> attribute in a file
   !> @param[in] cd_attname   attribute name
   !> @return  attribute structure    
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TFILE),      INTENT(IN) :: td_file
      CHARACTER(LEN=*), INTENT(IN) :: cd_varname
      CHARACTER(LEN=*), INTENT(IN) :: cd_attname

      ! function
      TYPE(TATT)                   :: tf_att

      ! local variable
      INTEGER(i4) :: il_varid
      !----------------------------------------------------------------

      ! get variable id
      IF( TRIM(fct_upper(cd_varname)) == 'GLOBAL' )THEN
         il_varid=NF90_GLOBAL
      ELSE
         il_varid=var_get_id(td_file%t_var(:), cd_varname)
      ENDIF

      IF( il_varid /= 0 .OR. TRIM(fct_upper(cd_varname)) == 'GLOBAL' )THEN
         ! open file
         SELECT CASE(TRIM(td_file%c_type))
            CASE('cdf')
               tf_att=iom_cdf_read_att(td_file, il_varid, cd_attname)
            CASE('dimg')
               CALL logger_warn( " IOM READ ATT: can't read attribute "//&
               &              "in dimg file :"//TRIM(td_file%c_name) )
            CASE DEFAULT
               CALL logger_error( " IOM READ ATT: can't read attribute in file "//&
               &               TRIM(td_file%c_name)//" : type unknown " )
         END SELECT
      ENDIF

   END FUNCTION iom__read_att_varname_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom__read_att_varid_name(td_file, id_varid, cd_attname) &
         & RESULT (tf_att)
   !-------------------------------------------------------------------
   !> @brief This function read attribute (of variable or global) in an opened 
   !> file, given variable id and attribute name.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_file      file structure
   !> @param[in] id_varid     variable id. use NF90_GLOBAL to read global
   !> attribute in a file
   !> @param[in] cd_attname   attribute name
   !> @return  attribute structure    
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TFILE), INTENT(IN) :: td_file
      INTEGER(i4), INTENT(IN) :: id_varid
      CHARACTER(LEN=*), INTENT(IN) :: cd_attname

      ! function
      TYPE(TATT)              :: tf_att
      !----------------------------------------------------------------

      ! open file
      SELECT CASE(TRIM(td_file%c_type))
         CASE('cdf')
            tf_att=iom_cdf_read_att(td_file, id_varid, cd_attname)
         CASE('dimg')
            CALL logger_warn( " IOM READ ATT: can't read attribute in dimg file :"&
            &              //TRIM(td_file%c_name) )
         CASE DEFAULT
            CALL logger_error( " IOM READ ATT: can't read attribute in file "//&
            &               TRIM(td_file%c_name)//" : type unknown " )
      END SELECT

   END FUNCTION iom__read_att_varid_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom__read_dim_id(td_file, id_dimid) &
         & RESULT (tf_dim)
   !-------------------------------------------------------------------
   !> @brief This function read one dimension in an opened file, 
   !> given dimension id.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_file   file structure
   !> @param[in] id_dimid  dimension id
   !> @return  dimension structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(IN) :: td_file
      INTEGER(i4), INTENT(IN) :: id_dimid

      ! function
      TYPE(TDIM)              :: tf_dim
      !----------------------------------------------------------------

      ! open file
      SELECT CASE(TRIM(td_file%c_type))
         CASE('cdf')
            tf_dim=iom_cdf_read_dim(td_file, id_dimid)
         CASE('dimg')
            tf_dim=iom_rstdimg_read_dim(td_file, id_dimid)
         CASE DEFAULT
            CALL logger_error( " IOM READ DIM: can't read dimension in file "//&
            &               TRIM(td_file%c_name)//" : type unknown " )
      END SELECT      

   END FUNCTION iom__read_dim_id
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom__read_dim_name(td_file, cd_name) &
         & RESULT(tf_dim)
   !-------------------------------------------------------------------
   !> @brief This function read one dimension in an opened netcdf file, 
   !> given dimension name.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_file   file structure
   !> @param[in] cd_name   dimension name
   !> @return  dimension structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE),      INTENT(IN) :: td_file
      CHARACTER(LEN=*), INTENT(IN) :: cd_name

      ! function
      TYPE(TDIM)                   :: tf_dim
      !----------------------------------------------------------------

      ! open file
      SELECT CASE(TRIM(td_file%c_type))
         CASE('cdf')
            tf_dim=iom_cdf_read_dim(td_file, cd_name)
         CASE('dimg')
            tf_dim=iom_rstdimg_read_dim(td_file, cd_name)
         CASE DEFAULT
            CALL logger_error( " IOM READ DIM: can't read dimension in file "//&
            &               TRIM(td_file%c_name)//" : type unknown " )
      END SELECT      

   END FUNCTION iom__read_dim_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom__read_var_id(td_file, id_varid, id_start, id_count) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function read variable value in an opened 
   !> file, given variable id.
   !> @details
   !> start indices and number of indices selected along each dimension 
   !> could be specify in a 4 dimension array (/'x','y','z','t'/)
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_file   file structure
   !> @param[in] id_varid  variable id
   !> @param[in] id_start  index in the variable from which the data values 
   !> will be read
   !> @param[in] id_count  number of indices selected along each dimension
   !> @return  variable structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE),                       INTENT(IN) :: td_file
      INTEGER(i4),                       INTENT(IN) :: id_varid
      INTEGER(i4), DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4), DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_count

      ! function
      TYPE(TVAR)                                    :: tf_var
      !----------------------------------------------------------------

      ! open file
      SELECT CASE(TRIM(td_file%c_type))
         CASE('cdf')
            tf_var=iom_cdf_read_var(td_file, id_varid, id_start, id_count)
         CASE('dimg')
            tf_var=iom_rstdimg_read_var(td_file, id_varid, id_start, id_count)
         CASE DEFAULT
            CALL logger_error( " IOM READ VAR: can't read variable in file "//&
            &               TRIM(td_file%c_name)//" : type unknown " )
      END SELECT

   END FUNCTION iom__read_var_id
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom__read_var_name(td_file, cd_name, id_start, id_count) &
      & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function read variable value in an opened 
   !> file, given variable name or standard name.
   !> @details
   !> start indices and number of indices selected along each dimension 
   !> could be specify in a 4 dimension array (/'x','y','z','t'/)
   !>
   !> look first for variable name. If it doesn't
   !> exist in file, look for variable standard name.<br/>
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_file   file structure
   !> @param[in] cd_name   variable name or standard name
   !> @param[in] id_start  index in the variable from which the data values
   !> will be read
   !> @param[in] id_count  number of indices selected along each dimension
   !> @return  variable structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE)                   , INTENT(IN) :: td_file
      CHARACTER(LEN=*)              , INTENT(IN) :: cd_name
      INTEGER(i4)     , DIMENSION(:), INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:), INTENT(IN), OPTIONAL :: id_count

      ! function
      TYPE(TVAR)                                 :: tf_var
      !----------------------------------------------------------------
      
      ! open file
      SELECT CASE(TRIM(td_file%c_type))
         CASE('cdf')
            tf_var=iom_cdf_read_var(td_file, cd_name, id_start, id_count )
         CASE('dimg')
            tf_var=iom_rstdimg_read_var(td_file, cd_name, id_start, id_count )
         CASE DEFAULT
            CALL logger_error( " IOM READ VAR: can't read variable in file "//&
            &               TRIM(td_file%c_name)//" : type unknown " )
      END SELECT

   END FUNCTION iom__read_var_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_write_file(td_file, cd_dimorder)
   !-------------------------------------------------------------------
   !> @brief This subroutine write file structure in an opened file.
   !>
   !> @details
   !> optionally, you could specify dimension order (default 'xyzt')
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2015 - add dimension order option
   !> @date August, 2017
   !> - split in write_header and write_var
   !>
   !> @param[in] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE)     , INTENT(INOUT) :: td_file
      CHARACTER(LEN=*), INTENT(IN   ), OPTIONAL :: cd_dimorder
      !----------------------------------------------------------------

      CALL iom_write_header(td_file, cd_dimorder)

      CALL iom_write_var(td_file)

   END SUBROUTINE iom_write_file
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_write_header(td_file, cd_dimorder, td_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine write header from file structure
   !> of an opened file.
   !>
   !> @details
   !> optionally, you could specify dimension order (default 'xyzt'), 
   !> and dimension structure for netcdf case.
   !>
   !> @author J.Paul
   !> @date August, 2017 - Initial Version
   !>
   !> @param[inout] td_file      file structure
   !> @param[in] cd_dimorder  dimension order
   !> @param[in] td_dim       array of dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE)                           , INTENT(INOUT) :: td_file
      CHARACTER(LEN=*)                      , INTENT(IN   ), OPTIONAL :: cd_dimorder
      TYPE(TDIM)      , DIMENSION(ip_maxdim), INTENT(IN   ), OPTIONAL :: td_dim
      !----------------------------------------------------------------

      ! open file
      SELECT CASE(TRIM(td_file%c_type))
         CASE('cdf')
            CALL iom_cdf_write_header(td_file, cd_dimorder, td_dim)
         CASE('dimg')
            ! note: can not change dimension order in restart dimg file
            CALL iom_rstdimg_write_header(td_file)
         CASE DEFAULT
            CALL logger_error( " IOM WRITE HEADER: can't write header&
            &                  , file "//TRIM(td_file%c_name)//" : &
            &                  type unknown " )
      END SELECT

   END SUBROUTINE iom_write_header
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_write_var(td_file, cd_dimorder, id_start, id_count)
   !-------------------------------------------------------------------
   !> @brief This subroutine write variables from file structure 
   !> in an opened file.
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date August, 2017 - Initial Version
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] cd_dimorder  dimension order
   !> @param[in] id_start  index in the variable from which the data values
   !> will be read
   !> @param[in] id_count  number of indices selected along each dimension
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE)                   , INTENT(INOUT) :: td_file
      CHARACTER(LEN=*)              , INTENT(IN   ), OPTIONAL :: cd_dimorder
      INTEGER(i4)     , DIMENSION(:), INTENT(IN   ), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:), INTENT(IN   ), OPTIONAL :: id_count
      !----------------------------------------------------------------

      ! open file
      SELECT CASE(TRIM(td_file%c_type))
         CASE('cdf')
            CALL iom_cdf_write_var(td_file, cd_dimorder, &
                 &                 id_start, id_count)
         CASE('dimg')
            ! note: can not change dimension order in restart dimg file
            CALL iom_rstdimg_write_var(td_file)
         CASE DEFAULT
            CALL logger_error( " IOM WRITE VAR: can't write variable, file "//&
            &               TRIM(td_file%c_name)//" : type unknown " )
      END SELECT

   END SUBROUTINE iom_write_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE iom

