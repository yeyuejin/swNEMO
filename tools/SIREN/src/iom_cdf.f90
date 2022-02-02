!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief NETCDF Input/Output manager :  Library to read Netcdf input files
!>
!> @details
!>    to open netcdf file:<br/>
!> @code
!>    CALL iom_cdf_open(td_file)
!> @endcode
!>       - td_file is file structure (see @ref file)
!>
!>    to write in netcdf file:<br/>
!> @code
!>    CALL  iom_cdf_write_file(td_file)
!> @endcode
!>
!>    to close netcdf file:<br/>
!> @code
!>    CALL iom_cdf_close(tl_file)
!> @endcode
!>
!>    to read one dimension in netcdf file:<br/>
!> @code
!>    tl_dim = iom_cdf_read_dim(tl_file, id_dimid)
!> @endcode
!>    or
!> @code
!>    tl_dim = iom_cdf_read_dim(tl_file, cd_name)
!> @endcode
!>       - id_dimid is dimension id<br/>
!>       - cd_name is dimension name
!>
!>    to read one attribute in netcdf file:<br/>
!> @code
!>    tl_att = iom_cdf_read_att(tl_file, id_varid, id_attid)
!> @endcode
!>    or
!> @code
!>    tl_att = iom_cdf_read_att(tl_file, id_varid, cd_name)
!> @endcode
!>       - id_varid is variable id
!>       - id_attid is attribute id<br/>
!>       - cd_name is attribute name
!>    
!>    to read one variable in netcdf file:<br/>
!> @code
!>    tl_var = iom_cdf_read_var(td_file, id_varid, [id_start, id_count])
!> @endcode
!>    or
!> @code
!>    tl_var = iom_cdf_read_var(td_file, cd_name, [id_start, [id_count,]])
!> @endcode
!>       - id_varid is variabale id
!>       - cd_name is variabale name
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
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE iom_cdf

   USE netcdf                          ! nf90 library
   USE global                          ! global parameter
   USE kind                            ! F90 kind parameter
   USE fct                             ! basic useful function
   USE logger                          ! log file manager
   USE att                             ! attribute manage
   USE dim                             ! dimension manager
   USE var                             ! variable manager
   USE file                            ! file manager

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! function and subroutine
   PUBLIC :: iom_cdf_open           !< open or create netcdf file, return file structure
   PUBLIC :: iom_cdf_close          !< close netcdf file
   PUBLIC :: iom_cdf_read_dim       !< read one dimension in an opened netcdf file, return dimension structure
   PUBLIC :: iom_cdf_read_att       !< read one attribute in an opened netcdf file, return attribute structure
   PUBLIC :: iom_cdf_read_var       !< read one variable  in an opened netcdf file, return variable  structure
   PUBLIC :: iom_cdf_fill_var       !< fill variable value in an opened netcdf file
   PUBLIC :: iom_cdf_write_header   !< write header in an opened netcdf file
   PUBLIC :: iom_cdf_write_var      !< write variables in an opened netcdf file

   PRIVATE :: iom_cdf__check           ! provides a simple interface to netcdf error message
   PRIVATE :: iom_cdf__get_info        ! get global information in an opened netcdf file
   PRIVATE :: iom_cdf__get_file_dim    ! read dimension on an opened netcdf file, and reorder it
   PRIVATE :: iom_cdf__get_file_att    ! read global attribute on an opened netcdf file
   PRIVATE :: iom_cdf__get_file_var    ! read information about variable on an opened netcdf file
   PRIVATE :: iom_cdf__read_dim_id     ! read one dimension in an opened netcdf file, given dimension id.
   PRIVATE :: iom_cdf__read_dim_name   ! read one dimension in an opened netcdf file, given dimension name.
   PRIVATE :: iom_cdf__read_att_name   ! read variable or global attribute in an opened netcdf file, given attribute name. 
   PRIVATE :: iom_cdf__read_att_id     ! read variable or global attribute in an opened netcdf file, given attribute id.
   PRIVATE :: iom_cdf__read_var_id     ! read variable value in an opened netcdf file, given variable id.
   PRIVATE :: iom_cdf__read_var_name   ! read variable value in an opened netcdf file, given variable name or standard name.
   PRIVATE :: iom_cdf__read_var_meta   ! read metadata of a variable in an opened netcdf file.
   PRIVATE :: iom_cdf__read_var_dim    ! read variable dimension in an opened netcdf file.
   PRIVATE :: iom_cdf__read_var_att    ! read variable attributes in an opened netcdf file.
   PRIVATE :: iom_cdf__read_var_value  ! read variable value in an opened netcdf file.
   PRIVATE :: iom_cdf__write_dim_def   ! write dimension definition in an opened netcdf file.
   PRIVATE :: iom_cdf__write_att_def   ! write attribute definition in an opened netcdf file.
   PRIVATE :: iom_cdf__write_var_def   ! write variable definition in an opened netcdf file.
   PRIVATE :: iom_cdf__write_var       ! write a variable in an opened netcdf file.
   PRIVATE :: iom_cdf__write_var_value ! put variable value in an opened netcdf file.
   PRIVATE :: iom_cdf__fill_var_id     ! fill variable value in an opened netcdf file, given variable id
   PRIVATE :: iom_cdf__fill_var_name   ! fill variable value in an opened netcdf file, given variable name
   PRIVATE :: iom_cdf__fill_var_all    ! fill all variable value in an opened netcdf file
   PRIVATE :: iom_cdf__del_coord_var   ! remove coordinate variable from an opened netcdf file

   INTERFACE iom_cdf_read_var
      MODULE PROCEDURE iom_cdf__read_var_id
      MODULE PROCEDURE iom_cdf__read_var_name
   END INTERFACE iom_cdf_read_var

   INTERFACE iom_cdf_fill_var
      MODULE PROCEDURE iom_cdf__fill_var_id
      MODULE PROCEDURE iom_cdf__fill_var_name
      MODULE PROCEDURE iom_cdf__fill_var_all
   END INTERFACE iom_cdf_fill_var

   INTERFACE iom_cdf_read_dim
      MODULE PROCEDURE iom_cdf__read_dim_id
      MODULE PROCEDURE iom_cdf__read_dim_name
   END INTERFACE iom_cdf_read_dim

   INTERFACE iom_cdf_read_att
      MODULE PROCEDURE iom_cdf__read_att_id
      MODULE PROCEDURE iom_cdf__read_att_name
   END INTERFACE iom_cdf_read_att

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf__check(id_status, cd_msg)
   !-------------------------------------------------------------------
   !> @brief This subroutine provides a simple interface to 
   !> netcdf error message 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date May, 2015 
   !> - add optional message to netcdf error message
   !>
   !> @param[in] id_status error status
   !> @param[in] cd_msg    message
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      INTEGER(i4)     , INTENT(IN)           :: id_status
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: cd_msg
      ! local variable
      CHARACTER(LEN=lc) :: cl_msg
      !----------------------------------------------------------------

      cl_msg=""
      IF( PRESENT(cd_msg) ) cl_msg=cd_msg

      IF( id_status /= NF90_NOERR )THEN
         CALL logger_error(TRIM(cl_msg)//TRIM(NF90_STRERROR(id_status)))
      ENDIF

   END SUBROUTINE iom_cdf__check
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf_open(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine open a netcdf file in read or write mode.
   !> @details
   !> if try to open a file in write mode that did not exist, create it.<br/>
   !> if file already exist, get information about0:<br/>
   !> - the number of variables
   !> - the number of dimensions
   !> - the number of global attributes
   !> - the ID of the unlimited dimension
   !> - the file format
   !> Finally it read dimensions, and 'longitude' variable to compute East-West
   !> overlap.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date November, 2018
   !> - write netcdf file as netcdf4
   !>
   !> @param[inout] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT)  :: td_file

      ! local variable
      LOGICAL     :: ll_exist
      LOGICAL     :: ll_open

      INTEGER(i4) :: il_status
      INTEGER(i4) :: il_oldmode
      !----------------------------------------------------------------

      ! check file existence
      INQUIRE(FILE=TRIM(td_file%c_name), EXIST=ll_exist, OPENED=ll_open)
      ! ll_open do not work for netcdf file, always return FALSE
      IF( .NOT. ll_exist .OR. TRIM(td_file%c_type) /= 'cdf' )THEN

         IF( .NOT. td_file%l_wrt )THEN

            CALL logger_fatal( " IOM CDF OPEN: can not open file "//&
            &               TRIM(td_file%c_name) )
 
         ELSE

            CALL logger_info( " IOM CDF CREATE: file "//TRIM(td_file%c_name) )

            il_status = NF90_CREATE(TRIM(td_file%c_name),&
               &                    cmode=NF90_NETCDF4,  &
               &                    ncid=td_file%i_id)
            CALL iom_cdf__check(il_status," IOM CDF CREATE: ")
            il_status = NF90_SET_FILL(td_file%i_id,   &
               &                      NF90_NOFILL,    &
               &                      il_oldmode)
            CALL iom_cdf__check(il_status," IOM CDF SET FILL: ")

            td_file%l_def=.TRUE.
            CALL logger_debug( " IOM CDF CREATE: td_file%l_def"//fct_str(td_file%l_def))

         ENDIF

      ELSE

         IF( td_file%i_id /= 0 )THEN

            CALL logger_error( " IOM CDF OPEN: file "//&
            &               TRIM(td_file%c_name)//" already opened")

         ELSE
 
            IF( .NOT. td_file%l_wrt )THEN

               CALL logger_info( " IOM CDF OPEN: file "//&
               &              TRIM(td_file%c_name)//" in read only mode" )

               il_status = NF90_OPEN( TRIM(td_file%c_name), &
               &                      NF90_NOWRITE,         &
               &                      td_file%i_id)
               CALL iom_cdf__check(il_status," IOM CDF OPEN: ")

            ELSE

               CALL logger_info( "IOM CDF OPEN: file "//&
               &              TRIM(td_file%c_name)//" in write mode" )

               il_status = NF90_OPEN( TRIM(td_file%c_name), &
               &                      NF90_WRITE,           &
               &                      td_file%i_id)
               CALL iom_cdf__check(il_status,"IOM CDF OPEN: ")

            ENDIF

            ! get general information about file
            CALL iom_cdf__get_info(td_file)

            ! read dimension in file
            CALL iom_cdf__get_file_dim(td_file) 

            ! read global attribute in file
            CALL iom_cdf__get_file_att(td_file)

            ! get information about variables in file
            CALL iom_cdf__get_file_var(td_file)

            ! remove dimension variable from list of variable
            CALL iom_cdf__del_coord_var(td_file)

         ENDIF

      ENDIF

   END SUBROUTINE iom_cdf_open
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf_close(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine close netcdf file.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file

      ! local variable
      INTEGER(i4) :: il_status
      !----------------------------------------------------------------

      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  " IOM CDF CLOSE: no id associated to file "//TRIM(td_file%c_name))

      ELSE
         CALL logger_info( &
         &  " IOM CDF CLOSE: file "//TRIM(td_file%c_name))

         il_status = NF90_CLOSE(td_file%i_id)
         CALL iom_cdf__check(il_status,"IOM CDF CLOSE: ")

         td_file%i_id = 0

      ENDIF

   END SUBROUTINE iom_cdf_close
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf__get_info(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine get global information in an opened netcdf 
   !> file.
   !> @details
   !> It gets the number of variables, the number of dimensions, 
   !> the number of global attributes, the ID of the unlimited dimension
   !> and finally the format version and filled file strucuture with it.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date October, 2016
   !> - define cdf4 as cdf.
   !>
   !> @param[inout] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file

      ! local variable
      INTEGER(i4) :: il_fmt   ! format version
      INTEGER(i4) :: il_status
      !----------------------------------------------------------------

      CALL logger_trace( &
      &  " IOM CDF GET INFO: about netcdf file "//TRIM(td_file%c_name))

      il_status=NF90_INQUIRE(td_file%i_id, td_file%i_ndim, &
      &     td_file%i_nvar, td_file%i_natt, td_file%i_uldid, il_fmt)
      CALL iom_cdf__check(il_status,"IOM CDF GET INFO: ")

      SELECT CASE(il_fmt)
         CASE(nf90_format_classic, nf90_format_64bit)
            td_file%c_type='cdf'
         CASE(nf90_format_netcdf4,nf90_format_netcdf4_classic)
            td_file%c_type='cdf'
      END SELECT
      CALL logger_debug("IOM CDF GET INFO: type "//TRIM(td_file%c_type)) 

      ! record header infos
      td_file%i_rhd=1

   END SUBROUTINE iom_cdf__get_info
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf__get_file_dim(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine read dimension on an opened netcdf file, and
   !> reorder dimension to ('x', 'y', 'z', 't').
   !> The dimension structure inside file structure is then completed.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date October, 2016
   !> - check unknown dimension
   !> @date January, 2019
   !> - clean dimension structure
   !>
   !> @param[inout] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file

      ! local variable
      TYPE(TDIM) :: tl_dim

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: ii
      !----------------------------------------------------------------

      ! clean dimension
      DO ji=1,ip_maxdim
         CALL dim_clean(td_file%t_dim(ji))
      ENDDO

      IF( td_file%i_ndim > 0 )THEN

         ii=1
         DO ji = 1, td_file%i_ndim
            ! read dimension information
            tl_dim=iom_cdf_read_dim( td_file, ji)
            ! sname == 'u' if dimension is unknown (not to be used)
            IF( TRIM(tl_dim%c_sname) /= 'u' )THEN
               IF( ii > ip_maxdim )THEN
                  CALL logger_fatal("IOM CDF OPEN: too much dimension "//&
                  & "to be read. you could choose dimension to be used. see "//&
                  & " configuration file")
               ENDIF
               td_file%t_dim(ii)=dim_copy(tl_dim)
               ii=ii+1
            ENDIF
            ! clean
            CALL dim_clean(tl_dim)
         ENDDO

         ! inform unlimited dimension
         IF( td_file%i_uldid == -1 )THEN
            CALL logger_warn( &
            &  " IOM CDF GET FILE DIM: there is no unlimited dimension in file "//&
            &  TRIM(td_file%c_name))
         !ELSE
         !   td_file%t_dim( td_file%i_uldid )%l_uld=.TRUE.
         ENDIF

      ELSE

         CALL logger_warn( &
         &  " IOM CDF GET FILE DIM: there is no dimension in file "//&
         &  TRIM(td_file%c_name))

      ENDIF

      ! reorder dimension to ('x','y','z','t')
      CALL dim_reorder(td_file%t_dim(:))

   END SUBROUTINE iom_cdf__get_file_dim
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf__get_file_att(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine read global attribute on an opened netcdf 
   !> file.
   !> The attribute structure inside file structure is then completed.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014
   !> - use attribute periodicity read from the file if present.
   !> @date January, 2019
   !> - clean attribute structure
   !>
   !> @param[inout] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file

      ! local variable
      TYPE(TATT) :: tl_att

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: ii
      !----------------------------------------------------------------
      CALL logger_trace("IOM CDF GET FILE ATT : get attr(s) in &
         &     file "//TRIM(td_file%c_name))

      IF( td_file%i_natt > 0 )THEN
         IF(ASSOCIATED(td_file%t_att))THEN
            CALL att_clean(td_file%t_att(:))
            DEALLOCATE(td_file%t_att)
         ENDIF
         ALLOCATE(td_file%t_att(td_file%i_natt))

         ii=1
         DO ji = 1, td_file%i_natt
            ! read global attribute
            tl_att=iom_cdf_read_att( td_file, NF90_GLOBAL, ji)
            IF( .NOT. att_is_dummy(tl_att) )THEN
               td_file%t_att(ii)=att_copy(tl_att)
               ii=ii+1
            ENDIF
            ! clean
            CALL att_clean(tl_att)
         ENDDO

      ELSE
         CALL logger_debug( &
         &  " IOM CDF GET FILE ATT: there is no global attribute in file "//&
         &  TRIM(td_file%c_name))
      ENDIF

   END SUBROUTINE iom_cdf__get_file_att
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf__get_file_var(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine read information about variable of an 
   !> opened netcdf file.
   !> The variable structure inside file structure is then completed.
   !> @note variable value are not read !
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2015
   !> - manage useless (dummy) variable
   !> @date January, 2016
   !> - increment n3d for 4D variable
   !> @date October, 2016
   !> - check variable to be used (variable's dimension allowed and variable
   !> not "dummy")
   !>
   !> @param[inout] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file

      ! local variable
      INTEGER(i4) :: il_attid
      INTEGER(i4) :: il_nvar

      TYPE(TVAR), DIMENSION(:), ALLOCATABLE  :: tl_var

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: ii
      !----------------------------------------------------------------

      IF( td_file%i_nvar > 0 )THEN

         IF(ASSOCIATED(td_file%t_var))THEN
            CALL var_clean(td_file%t_var(:))
            DEALLOCATE(td_file%t_var)
         ENDIF

         il_nvar=td_file%i_nvar
         ALLOCATE(tl_var(il_nvar))
         DO ji = 1, il_nvar
           ! read variable information
           tl_var(ji)=iom_cdf__read_var_meta( td_file, ji) 
         ENDDO

         ! update number of variable used
         td_file%i_nvar=COUNT(tl_var(:)%l_use)

         ALLOCATE(td_file%t_var(td_file%i_nvar))

         ii=0
         DO ji = 1, il_nvar
            IF( tl_var(ji)%l_use )THEN
               ii=ii+1
               td_file%t_var(ii)=var_copy(tl_var(ji))
               SELECT CASE(td_file%t_var(ii)%i_ndim)
                  CASE(0)
                     td_file%i_n0d=td_file%i_n0d+1
                  CASE(1)
                     td_file%i_n1d=td_file%i_n1d+1
                     td_file%i_rhd=td_file%i_rhd+1
                  CASE(2)
                     td_file%i_n2d=td_file%i_n2d+1
                     td_file%i_rhd=td_file%i_rhd+1
                  CASE(3,4)
                     td_file%i_n3d=td_file%i_n3d+1
                     td_file%i_rhd=td_file%i_rhd+td_file%t_dim(3)%i_len
               END SELECT

               ! look for depth id
               IF( INDEX(TRIM(fct_lower(td_file%t_var(ii)%c_name)),'depth')/=0 )THEN
                  IF( td_file%i_depthid == 0 )THEN
                     td_file%i_depthid=ji
                  ELSE
                     IF( td_file%i_depthid /= ji )THEN
                        CALL logger_error("IOM CDF GET FILE VAR: find more"//&
                           &  " than one depth variable in file "//&
                           &  TRIM(td_file%c_name) )
                     ENDIF
                  ENDIF
               ENDIF

               ! look for time id
               IF( INDEX(TRIM(fct_lower(td_file%t_var(ii)%c_name)),'time')/=0 )THEN
                  IF( td_file%i_timeid == 0 )THEN
                     td_file%i_timeid=ji
                  ELSE
                     IF( td_file%i_timeid /= ji )THEN
                        CALL logger_warn("IOM CDF GET FILE VAR: find more "//&
                        &                 "than one time variable in file "//&
                        &                 TRIM(td_file%c_name)//". see "//&
                        &                 "dummy.cfg configuration file to"//&
                        &                 " not used dummy variables.")
                     ENDIF
                     il_attid=0
                     IF( ASSOCIATED(td_file%t_var(ii)%t_att) )THEN
                        il_attid=att_get_id(td_file%t_var(ii)%t_att(:),'calendar')
                     ENDIF
                     IF( il_attid /= 0 )THEN
                        td_file%i_timeid=ji
                     ENDIF
                  ENDIF
               ENDIF

            ENDIF
         ENDDO

         CALL var_clean(tl_var(:))
         DEALLOCATE(tl_var)

      ELSE
         CALL logger_debug( &
         &  " IOM CDF GET FILE VAR: there is no variable in file "//&
         &  TRIM(td_file%c_name))
      ENDIF

   END SUBROUTINE iom_cdf__get_file_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf__del_coord_var(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine delete coordinate variable from an 
   !> opened netcdf file if present.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file

      ! local variable
      CHARACTER(LEN=lc) :: cl_name
      CHARACTER(LEN=lc) :: cl_sname

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      IF( td_file%i_nvar > 0 )THEN
         DO ji=td_file%i_nvar,1,-1
            cl_name=TRIM(td_file%t_var(ji)%c_name)
            DO jj=1,ip_maxdim
               IF( td_file%t_dim(jj)%l_use )THEN
                  cl_sname=fct_upper(td_file%t_dim(jj)%c_sname)
                  IF( TRIM(cl_name) == TRIM(cl_sname) )THEN
                     CALL file_del_var(td_file,TRIM(cl_name))
                     EXIT
                  ENDIF
               ENDIF
            ENDDO
         ENDDO
      ELSE
         CALL logger_debug( &
         &  " IOM CDF DEL VAR DIM: there is no variable in file "//&
         &  TRIM(td_file%c_name))
      ENDIF

   END SUBROUTINE iom_cdf__del_coord_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_cdf__read_dim_id(td_file, id_dimid) &
         & RESULT (tf_dim)
   !-------------------------------------------------------------------
   !> @brief This function read one dimension in an opened netcdf file, 
   !> given dimension id.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date February, 2015 - create unused dimension, when reading dimension
   !> of length less or equal to zero
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

      ! local variable
      INTEGER(i4)       :: il_status
      INTEGER(i4)       :: il_len
      CHARACTER(LEN=lc) :: cl_name
      LOGICAL           :: ll_use
      !----------------------------------------------------------------

      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  " IOM CDF READ DIM: no id associated to file "//TRIM(td_file%c_name))

      ELSE

         CALL logger_trace( &
         &  " IOM CDF READ DIM: dimension "//TRIM(fct_str(id_dimid))//&
         &  " in file "//TRIM(td_file%c_name))

         il_status=NF90_INQUIRE_DIMENSION(td_file%i_id, id_dimid, &
         &                                cl_name, il_len )
         CALL iom_cdf__check(il_status,"IOM CDF READ DIM: ")

         ll_use=.TRUE.
         IF( il_len <= 0 )THEN
            CALL logger_warn( &
         &  " IOM CDF READ DIM: dimension "//TRIM(fct_str(id_dimid))//&
         &  " in file "//TRIM(td_file%c_name)//" is less or equel to zero")
            il_len=1
            ll_use=.FALSE.
         ENDIF
         tf_dim=dim_init(cl_name, il_len, ld_use=ll_use)

      ENDIF

      tf_dim%i_id=id_dimid

   END FUNCTION iom_cdf__read_dim_id
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_cdf__read_dim_name(td_file, cd_name) &
         & RESULT (tf_dim)
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

      ! local variable
      INTEGER(i4) :: il_status
      INTEGER(i4) :: il_dimid
      !----------------------------------------------------------------

      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  " IOM CDF READ DIM: no id associated to file "//&
         &  TRIM(td_file%c_name))

      ELSE      

         il_status=NF90_INQ_DIMID( td_file%i_id, TRIM(ADJUSTL(cd_name)), &
         &                         il_dimid)
         CALL iom_cdf__check(il_status,"IOM CDF READ DIM: ")

         tf_dim=iom_cdf_read_dim(td_file, il_dimid)

      ENDIF

   END FUNCTION iom_cdf__read_dim_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_cdf__read_att_name(td_file, id_varid, cd_name) &
         & RESULT (tf_att)
   !-------------------------------------------------------------------
   !> @brief This function read variable or global attribute in an opened 
   !> netcdf file, given attribute name.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date November 2017
   !> - check if cl_value is not bug 
   !>
   !> @param[in] td_file   file structure
   !> @param[in] id_varid  variable id. use NF90_GLOBAL to read global
   !> attribute in a file
   !> @param[in] cd_name   attribute name
   !> @return  attribute structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE),      INTENT(IN) :: td_file
      INTEGER(i4),      INTENT(IN) :: id_varid
      CHARACTER(LEN=*), INTENT(IN) :: cd_name
   
      ! function
      TYPE(TATT)                   :: tf_att

      ! local variable
      CHARACTER(LEN=lc) :: cl_name

      INTEGER(i4) :: il_status
      INTEGER(i4) :: il_attid
      INTEGER(i4) :: il_type
      INTEGER(i4) :: il_len

      CHARACTER(LEN=lc) :: cl_value
      
      INTEGER(i1), DIMENSION(:), ALLOCATABLE :: bl_value
      INTEGER(i2), DIMENSION(:), ALLOCATABLE :: sl_value
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_value
      REAL(sp)   , DIMENSION(:), ALLOCATABLE :: rl_value
      REAL(dp)   , DIMENSION(:), ALLOCATABLE :: dl_value
      !----------------------------------------------------------------
      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
            &  " IOM CDF READ ATT: no id associated to file "//TRIM(td_file%c_name))

      ELSE      

         cl_name=TRIM(ADJUSTL(cd_name))

         ! inquire attribute
         IF( id_varid == NF90_GLOBAL )THEN

            CALL logger_trace( &
               &  " IOM CDF READ ATT: inquire global attribute "//&
               &  " in file "//TRIM(td_file%c_name))

         ELSE

            CALL logger_trace( &
               &  " IOM CDF READ ATT: inquire attribute "//&
               &  " of variable "//TRIM(fct_str(id_varid))//&
               &  " in file "//TRIM(td_file%c_name))

         ENDIF

         il_status=NF90_INQUIRE_ATTRIBUTE(td_file%i_id, id_varid,  &
            &                             cl_name,&
            &                             il_type,&
            &                             il_len, &
            &                             il_attid )
         CALL iom_cdf__check(il_status,"IOM CDF READ ATT: ")

         !! get attribute value
         CALL logger_debug( " IOM CDF READ ATT: get attribute "//&
            &            TRIM(cl_name)//" in file "//TRIM(td_file%c_name))

         SELECT CASE( il_type )

            CASE(NF90_CHAR)
               CALL logger_debug( " IOM CDF READ ATT: get NF90_CHAR ")

               ! check string lengths
               IF( LEN(cl_value) < il_len )THEN

                  CALL logger_warn( &
                     &  " IOM CDF READ ATT: not enough space to put "//&
                     &  "attribute "//TRIM(cl_name) )

               ELSE

                  ! Read the attributes
                  il_status=NF90_GET_ATT(td_file%i_id, id_varid, &
                     &                   cl_name, &
                     &                   cl_value )
                  CALL iom_cdf__check(il_status,"IOM CDF READ ATT: ")

                  ! check cl_value
                  CALL logger_trace("IOM CDF READ ATT: value = "//TRIM(cl_value))
                  IF( LLT(cl_value,'') ) cl_value = ''
                  tf_att=att_init(cl_name, cl_value)

               ENDIF
         
            CASE(NF90_BYTE)
               CALL logger_debug( " IOM CDF READ ATT: get NF90_BYTE ")

               ALLOCATE( bl_value(il_len), &
                  &      stat=il_status)
               IF(il_status /= 0 )THEN

                  CALL logger_error( "IOM CDF READ ATT: "//&
                     &  "not enough space to put attribute "//TRIM(cl_name) )

               ELSE

                  ! Read the attributes
                  il_status=NF90_GET_ATT(td_file%i_id, id_varid, &
                     &                   cl_name, &
                     &                   bl_value(:))
                  CALL iom_cdf__check(il_status,"IOM CDF READ ATT: ")   

                  tf_att=att_init(cl_name, bl_value(:))

               ENDIF

               DEALLOCATE(bl_value)

            CASE(NF90_SHORT)
               CALL logger_debug( " IOM CDF READ ATT: get NF90_SHORT ")

               ALLOCATE( sl_value(il_len), &
                  &      stat=il_status)
               IF(il_status /= 0 )THEN

                  CALL logger_error( &
                     &  " IOM CDF READ ATT: not enough space to put "//&
                     &  "attribute "//TRIM(cl_name) )

               ELSE

                  ! Read the attributes
                  il_status=NF90_GET_ATT(td_file%i_id, id_varid, &
                     &                   cl_name, &
                     &                   sl_value(:))
                  CALL iom_cdf__check(il_status,"IOM CDF READ ATT: ")   

                  tf_att=att_init(cl_name, sl_value(:))

               ENDIF

               DEALLOCATE(sl_value)

            CASE(NF90_INT)
               CALL logger_debug( " IOM CDF READ ATT: get NF90_INT ")

               ALLOCATE( il_value(il_len), &
                  &      stat=il_status)
               IF(il_status /= 0 )THEN

                  CALL logger_error( &
                     &  " IOM CDF READ ATT: not enough space to put "//&
                     &  "attribute "//TRIM(cl_name) )

               ELSE

                  ! Read the attributes
                  il_status=NF90_GET_ATT(td_file%i_id, id_varid, &
                     &                   cl_name, &
                     &                   il_value(:))
                  CALL iom_cdf__check(il_status,"IOM CDF READ ATT: ")   

                  tf_att=att_init(cl_name, il_value(:))
               ENDIF

               DEALLOCATE(il_value)

            CASE(NF90_FLOAT)
               CALL logger_debug( " IOM CDF READ ATT: get NF90_FLOAT ")

               ALLOCATE( rl_value(il_len), &
                  &      stat=il_status)
               IF(il_status /= 0 )THEN

                  CALL logger_error( &
                  &  " IOM CDF READ ATT: not enough space to put "//&
                  &  "attribute "//TRIM(cl_name) )

               ELSE

                  ! Read the attributes
                  il_status=NF90_GET_ATT(td_file%i_id, id_varid, &
                     &                   cl_name, &
                     &                   rl_value(:))
                  CALL iom_cdf__check(il_status,"IOM CDF READ ATT: ")   

                  tf_att=att_init(cl_name, rl_value(:))

               ENDIF

               DEALLOCATE(rl_value)

            CASE(NF90_DOUBLE)
               CALL logger_debug( " IOM CDF READ ATT: get NF90_DOUBLE ")

               ALLOCATE( dl_value(il_len), &
                  &      stat=il_status)
               IF(il_status /= 0 )THEN

                  CALL logger_error( &
                     &  " IOM CDF READ ATT: not enough space to put "//&
                     &  "attribute "//TRIM(cl_name) )

               ELSE

                  ! Read the attributes
                  il_status=NF90_GET_ATT(td_file%i_id, id_varid, &
                     &                   cl_name, &
                     &                   dl_value(:))
                  CALL iom_cdf__check(il_status,"IOM CDF READ ATT: ")   

                  tf_att=att_init(cl_name, dl_value(:))

               ENDIF

               DEALLOCATE(dl_value)

         END SELECT

         tf_att%i_id=il_attid

      ENDIF

   END FUNCTION iom_cdf__read_att_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_cdf__read_att_id(td_file, id_varid, id_attid) &
         & RESULT (tf_att)
   !-------------------------------------------------------------------
   !> @brief This function read variable or global attribute in an opened 
   !> netcdf file, given attribute id.
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

      ! local variable
      INTEGER(i4)       :: il_status
      CHARACTER(LEN=lc) :: cl_name
      !----------------------------------------------------------------
      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
            &  "IOM CDF READ ATT: no id associated to file "//TRIM(td_file%c_name))

      ELSE

         CALL logger_trace( " IOM CDF READ ATT ID: get attribute "//&
            &               TRIM(fct_str(id_attid))//" of var "//&
            &               TRIM(fct_str(id_varid))//" in file "//&
            &               TRIM(td_file%c_name) &
         )

         ! get attribute name
         il_status=NF90_INQ_ATTNAME(td_file%i_id, id_varid, id_attid, cl_name)
         CALL iom_cdf__check(il_status,"IOM CDF READ ATT: ")

         ! read attribute
         tf_att=iom_cdf__read_att_name(td_file, id_varid, cl_name)

      ENDIF

   END FUNCTION iom_cdf__read_att_id
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_cdf__read_var_id(td_file, id_varid, id_start, id_count) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function read variable value in an opened 
   !> netcdf file, given variable id.
   !> @details
   !> Optionaly, start indices and number of indices selected along each dimension 
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
      TYPE(TFILE),               INTENT(IN) :: td_file
      INTEGER(i4),               INTENT(IN) :: id_varid
      INTEGER(i4), DIMENSION(:), INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4), DIMENSION(:), INTENT(IN), OPTIONAL :: id_count

      ! function
      TYPE(TVAR)                            :: tf_var

      ! local variable
      INTEGER(i4), DIMENSION(1) :: il_ind
      !----------------------------------------------------------------
      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  " IOM CDF READ VAR: no id associated to file "//TRIM(td_file%c_name))

      ELSE

         ! look for variable index
         il_ind(:)=MINLOC(td_file%t_var(:)%i_id,mask=(td_file%t_var(:)%i_id==id_varid))
         IF( il_ind(1) /= 0 )THEN

            tf_var=var_copy(td_file%t_var(il_ind(1)))

            !!! read variable value
            CALL iom_cdf__read_var_value(td_file, tf_var, id_start, id_count)

         ELSE
            CALL logger_error( &
            &  " IOM CDF READ VAR: there is no variable with id "//&
            &  TRIM(fct_str(id_varid))//" in file "//TRIM(td_file%c_name))
         ENDIF

      ENDIF
   END FUNCTION iom_cdf__read_var_id
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_cdf__read_var_name(td_file, cd_name, id_start, id_count) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function read variable value in an opened 
   !> netcdf file, given variable name or standard name.
   !> @details
   !> Optionaly, start indices and number of indices selected along each dimension 
   !> could be specify in a 4 dimension array (/'x','y','z','t'/)
   !>
   !> look first for variable name. If it doesn't
   !> exist in file, look for variable standard name.<br/>
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_file   file structure
   !> @param[in] cd_name   variable name or standard name.
   !> @param[in] id_start  index in the variable from which the data values will be read
   !> @param[in] id_count  number of indices selected along each dimension
   !> @return  variable structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE)     ,                INTENT(IN) :: td_file
      CHARACTER(LEN=*),                INTENT(IN), OPTIONAL :: cd_name
      INTEGER(i4)     , DIMENSION(:),  INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:),  INTENT(IN), OPTIONAL :: id_count

      ! function
      TYPE(TVAR)                                  :: tf_var

      ! local variable
      INTEGER(i4)       :: il_varid
      !----------------------------------------------------------------
      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  " IOM CDF READ VAR: no id associated to file "//TRIM(td_file%c_name))

      ELSE

         IF( .NOT. PRESENT(cd_name) )THEN

            CALL logger_error( &
            &  " IOM CDF READ VAR: you must specify a variable to read "//&
            &  " in file "//TRIM(td_file%c_name))

         ELSE

            il_varid=var_get_index(td_file%t_var(:), cd_name)
            IF( il_varid /= 0 )THEN

               tf_var=var_copy(td_file%t_var(il_varid))

               !!! read variable value
               CALL iom_cdf__read_var_value( td_file, tf_var, id_start, id_count)

            ELSE

               CALL logger_error( &
               &  " IOM CDF READ VAR: there is no variable with "//&
               &  " name or standard name "//TRIM(cd_name)//&
               &  " in file "//TRIM(td_file%c_name) )
            ENDIF

         ENDIF

      ENDIF
 
   END FUNCTION iom_cdf__read_var_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf__fill_var_all(td_file, id_start, id_count)
   !-------------------------------------------------------------------
   !> @brief This subroutine fill all variable value from an opened 
   !> netcdf file.
   !> @details
   !> Optionaly, start indices and number of indices selected along each dimension 
   !> could be specify in a 4 dimension array (/'x','y','z','t'/)
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] id_start     index in the variable from which the data values 
   !> will be read
   !> @param[in] id_count     number of indices selected along each dimension
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE),               INTENT(INOUT) :: td_file
      INTEGER(i4), DIMENSION(:), INTENT(IN   ),  OPTIONAL :: id_start
      INTEGER(i4), DIMENSION(:), INTENT(IN   ),  OPTIONAL :: id_count

      ! local variable

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  " IOM CDF FILL VAR: no id associated to file "//TRIM(td_file%c_name))

      ELSE

         DO ji=1,td_file%i_nvar
            CALL iom_cdf_fill_var(td_file, td_file%t_var(ji)%i_id, &
            &                     id_start, id_count)
         ENDDO

      ENDIF

   END SUBROUTINE iom_cdf__fill_var_all
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf__fill_var_id(td_file, id_varid, id_start, id_count)
   !-------------------------------------------------------------------
   !> @brief This subroutine fill variable value in an opened 
   !> netcdf file, given variable id.
   !> @details
   !> Optionaly, start indices and number of indices selected along each dimension 
   !> could be specify in a 4 dimension array (/'x','y','z','t'/)
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[inout] td_file   file structure
   !> @param[in] id_varid     variable id
   !> @param[in] id_start     index in the variable from which the data values 
   !> will be read
   !> @param[in] id_count     number of indices selected along each dimension
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE),               INTENT(INOUT) :: td_file
      INTEGER(i4),               INTENT(IN)    :: id_varid
      INTEGER(i4), DIMENSION(:), INTENT(IN),  OPTIONAL :: id_start
      INTEGER(i4), DIMENSION(:), INTENT(IN),  OPTIONAL :: id_count

      ! local variable
      INTEGER(i4), DIMENSION(1) :: il_varid

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  "IOM CDF FILL VAR: no id associated to file "//TRIM(td_file%c_name))

      ELSE

         ! look for variable id
         il_varid(:)=MINLOC( td_file%t_var(:)%i_id, &
         &                 mask=(td_file%t_var(:)%i_id==id_varid))
         IF( il_varid(1) /= 0 )THEN

            !!! read variable value
            CALL iom_cdf__read_var_value(td_file, td_file%t_var(il_varid(1)), &
            &                            id_start, id_count)

            DO ji=1,td_file%i_nvar
               CALL logger_debug(" IOM CDF FILL VAR: var id "//&
               &     TRIM(td_file%t_var(ji)%c_name)//" "//&
               &     TRIM(fct_str(td_file%t_var(ji)%i_id)) )
            ENDDO
         ELSE
            CALL logger_error( &
            &  " IOM CDF FILL VAR: there is no variable with id "//&
            &  TRIM(fct_str(id_varid))//" in file "//TRIM(td_file%c_name))
         ENDIF

      ENDIF

   END SUBROUTINE iom_cdf__fill_var_id
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf__fill_var_name(td_file, cd_name, id_start, id_count)
   !-------------------------------------------------------------------
   !> @brief This subroutine fill variable value in an opened 
   !> netcdf file, given variable name or standard name.
   !> @details
   !> Optionaly, start indices and number of indices selected along each dimension 
   !> could be specify in a 4 dimension array (/'x','y','z','t'/)
   !>
   !> look first for variable name. If it doesn't
   !> exist in file, look for variable standard name.<br/>
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] cd_name      variable name or standard name
   !> @param[in] id_start     index in the variable from which the data values will be read
   !> @param[in] id_count     number of indices selected along each dimension
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE),                   INTENT(INOUT) :: td_file
      CHARACTER(LEN=*),              INTENT(IN)    :: cd_name
      INTEGER(i4),     DIMENSION(:), INTENT(IN),  OPTIONAL :: id_start
      INTEGER(i4),     DIMENSION(:), INTENT(IN),  OPTIONAL :: id_count

      ! local variable
      INTEGER(i4)       :: il_varid
      !----------------------------------------------------------------
      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  "IOM CDF FILL VAR: no id associated to file "//TRIM(td_file%c_name))

      ELSE

            il_varid=var_get_index(td_file%t_var(:), cd_name)
            IF( il_varid /= 0 )THEN

               !!! read variable value
               CALL iom_cdf__read_var_value(td_file, td_file%t_var(il_varid), &
               &                            id_start, id_count)

            ELSE

               CALL logger_error( &
               &  "IOM CDF FILL VAR: there is no variable with "//&
               &  "name or standard name"//TRIM(cd_name)//&
               &  " in file "//TRIM(td_file%c_name))
            ENDIF

      ENDIF
      
   END SUBROUTINE iom_cdf__fill_var_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_cdf__read_var_meta(td_file, id_varid) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function read metadata of a variable in an opened 
   !> netcdf file.
   !>
   !> @note variable value are not read
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014
   !> - force to use FillValue=1.e20 if no FillValue for coordinate variable.
   !> @date September, 2015
   !> - manage useless (dummy) attribute
   !>
   !> @param[in] td_file   file structure
   !> @param[in] id_varid  variable id
   !> @return  variable structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(IN) :: td_file
      INTEGER(i4), INTENT(IN) :: id_varid

      ! function
      TYPE(TVAR)              :: tf_var

      ! local variable
      CHARACTER(LEN=lc)                                       :: cl_name

      INTEGER(i4)                                             :: il_status
      INTEGER(i4)                                             :: il_type
      INTEGER(i4)                                             :: il_ndim
      INTEGER(i4)                                             :: il_natt
      INTEGER(i4)                                             :: il_attid
      INTEGER(i4), DIMENSION(NF90_MAX_VAR_DIMS)               :: il_dimid

      TYPE(TDIM) , DIMENSION(ip_maxdim)                       :: tl_dim
      TYPE(TATT)                                              :: tl_fill
      TYPE(TATT) , DIMENSION(:)                 , ALLOCATABLE :: tl_att
      TYPE(TATT) , DIMENSION(:)                 , ALLOCATABLE :: tl_tmp

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
            &  " IOM CDF READ VAR META: no id associated to file "//&
            &   TRIM(td_file%c_name))

      ELSE

         ! inquire variable
         CALL logger_debug( &
            &  " IOM CDF READ VAR META: inquire variable "//&
            &  TRIM(fct_str(id_varid))//&
            &  " in file "//TRIM(td_file%c_name))
         
         il_dimid(:)=0

         il_status=NF90_INQUIRE_VARIABLE( td_file%i_id, id_varid,        &
            &                             cl_name,    &
            &                             il_type,    &
            &                             il_ndim,    &
            &                             il_dimid(:),&
            &                             il_natt )
         CALL iom_cdf__check(il_status,"IOM CDF READ VAR META: ")

         !!! fill variable dimension structure
         tl_dim(:)=iom_cdf__read_var_dim( td_file, il_ndim, cl_name, il_dimid(:) )

         IF( il_natt /= 0 )THEN
            ALLOCATE( tl_att(il_natt) )
            !!! fill variable attribute structure
            tl_att(:)=iom_cdf__read_var_att(td_file, id_varid, il_natt)

            !! look for _FillValue. if none add one
            il_attid=att_get_id(tl_att(:),'_FillValue')
            IF( il_attid == 0 )THEN
               CALL logger_info("IOM CDF READ VAR META: no _FillValue for variable "//&
                  &  TRIM(cl_name)//" in file "//TRIM(td_file%c_name) )

               il_attid=att_get_id(tl_att(:),'missing_value')
               IF( il_attid /= 0 )THEN
                  ! create attribute _FillValue
                  CALL logger_info("IOM CDF READ VAR META: assume _FillValue is equal to "//&
                     &             "missing_value for variable "//TRIM(cl_name) )
                  tl_fill=att_init('_FillValue',tl_att(il_attid)%d_value(:), &
                     &             id_type=tl_att(il_attid)%i_type)
               ELSE
                  ! create attribute _FillValue
                  SELECT CASE(TRIM(fct_lower(cl_name)))
                     CASE DEFAULT
                        CALL logger_info("IOM CDF READ VAR META: assume _FillValue is equal to "//&
                           &             "zero for variable "//TRIM(cl_name) )
                        tl_fill=att_init('_FillValue',0.)
                     CASE('nav_lon','nav_lat', 'nav_lev', &
                        &  'glamt','glamu','glamv','glamf', &
                        &  'gphit','gphiu','gphiv','gphif')
                        CALL logger_info("IOM CDF READ VAR META: assume _FillValue is equal to "//&
                           &             "dummy fillValue (1.e20) for variable "//TRIM(cl_name) )
                        tl_fill=att_init('_FillValue',1.e20)
                  END SELECT
               ENDIF

               ALLOCATE( tl_tmp(il_natt) )
               ! save read attribut
               tl_tmp(:)=att_copy(tl_att(:))
               ! change number of attribute in array
               CALL att_clean(tl_att(:))
               DEALLOCATE( tl_att )
               ALLOCATE( tl_att(il_natt+1) )
               ! copy read attribut
               tl_att(1:il_natt)=att_copy(tl_tmp(:))
               ! clean
               CALL att_clean(tl_tmp(:))
               DEALLOCATE( tl_tmp )

               ! create attribute _FillValue
               tl_att(il_natt+1)=att_copy(tl_fill)

            ENDIF

         ELSE
            ALLOCATE(tl_att(il_natt+1) )
            ! create attribute _FillValue
            SELECT CASE(TRIM(fct_lower(cl_name)))
               CASE DEFAULT
                  CALL logger_info("IOM CDF READ VAR META: assume _FillValue is equal to "//&
                     &             "zero for variable "//TRIM(cl_name) )
                  tl_fill=att_init('_FillValue',0.)
               CASE('nav_lon','nav_lat', &
                  &  'glamt','glamu','glamv','glamf', &
                  &  'gphit','gphiu','gphiv','gphif')
                  CALL logger_info("IOM CDF READ VAR META: assume _FillValue is equal to "//&
                     &             "dummy fillValue (1.e20) for variable "//TRIM(cl_name) )
                  tl_fill=att_init('_FillValue',1.e20)
            END SELECT            
            ! create attribute _FillValue
            tl_att(il_natt+1)=att_copy(tl_fill)
         ENDIF

         !! initialize variable
         tf_var=var_init( cl_name, il_type, tl_dim(:), tl_att(:), id_id=id_varid )

         !! look for dummy attribute
         DO ji=il_natt,1,-1
            IF( att_is_dummy(tl_att(ji)) )THEN
               CALL var_del_att(tf_var, tl_att(ji))
            ENDIF
         ENDDO

         !! check if variable is dummy
         IF( var_is_dummy(tf_var) )THEN
            tf_var%l_use=.FALSE.
         ENDIF

         !! check if all dimensions are allowed
         DO ji=1,il_ndim
            IF( ALL(td_file%t_dim(:)%i_id /= il_dimid(ji)) )THEN
               tf_var%l_use=.FALSE.
            ENDIF
         ENDDO

         ! clean
         CALL dim_clean(tl_dim(:))
         CALL att_clean(tl_fill)
         CALL att_clean(tl_att(:))
         DEALLOCATE( tl_att )

      ENDIF

   END FUNCTION iom_cdf__read_var_meta
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_cdf__read_var_dim(td_file, id_ndim, cd_name, id_dimid) &
         & RESULT (tf_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine read variable dimension
   !> in an opened netcdf file.
   !>
   !> @details
   !> the number of dimension can't exceed 4, 
   !> and should be 'x', 'y', 'z', 't' (whatever their order).<br/>
   !> If the number of dimension read is less than 4, the array of dimension
   !> strucure is filled with unused dimension.<br/>
   !> So the array of dimension structure of a variable is always compose of 4
   !> dimension (use or not). 
   !>
   !> @warn dummy dimension are not used. 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2015 
   !> - Bug fix: use order to disorder table (see dim_init)
   !> @date September, 2015
   !> - check dummy dimension
   !> @date April, 2018
   !> - Bug fix : use dimid to look for the index of the dimension, and not as
   !> dimension index
   !>
   !> @param[in] td_file   file structure
   !> @param[in] id_ndim   number of dimension
   !> @param[in] cd_name   variable name
   !> @param[in] id_dimid  array of dimension id
   !> @return array dimension structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE),               INTENT(IN) :: td_file
      INTEGER(i4),               INTENT(IN) :: id_ndim
      CHARACTER(LEN=*)         , INTENT(IN) :: cd_name
      INTEGER(i4), DIMENSION(:), INTENT(IN) :: id_dimid

      ! function
      TYPE(TDIM), DIMENSION(ip_maxdim)      :: tf_dim

      ! local variable
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_xyzt2
      INTEGER(i4)                       :: il_idx

      TYPE(TDIM) , DIMENSION(ip_maxdim) :: tl_dim

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: ii
      !----------------------------------------------------------------

      IF( id_ndim == 0 )THEN

         tl_dim(:)%l_use=.FALSE.

         ! reorder dimension to ('x','y','z','t')
         CALL dim_reorder(tl_dim(:))

         tf_dim(:)=dim_copy(tl_dim(:))

         ! clean
         CALL dim_clean(tl_dim(:))

      ELSE IF( id_ndim > 0 )THEN

         ii=1
         DO ji = 1, id_ndim

            ! check if dimension to be used, is allowed
            IF( ANY(td_file%t_dim(:)%i_id == id_dimid(ji)) )THEN
               IF( ii > ip_maxdim )THEN
                  CALL logger_error(" IOM CDF READ VAR DIM: "//&
                     &  "too much dimensions for variable "//&
                     &  TRIM(cd_name)//". check dummy configuration file.")
               ENDIF

               CALL logger_debug( " IOM CDF READ VAR DIM: get variable "//&
                  &  "dimension "//TRIM(fct_str(ji)) )

               ! look for dimension index
               DO jj=1,ip_maxdim
                  IF( td_file%t_dim(jj)%i_id == id_dimid(ji) )THEN
                     il_idx=jj
                     EXIT
                  ENDIF
               ENDDO
               !il_xyzt2(ii)=td_file%t_dim(id_dimid(ji))%i_xyzt2
               il_xyzt2(ii)=td_file%t_dim(il_idx)%i_xyzt2

               ! read dimension information
               tl_dim(ii) = dim_init( td_file%t_dim(il_xyzt2(ii))%c_name, &
                  &                   td_file%t_dim(il_xyzt2(ii))%i_len )

               ii=ii+1
            ELSE
               CALL logger_debug(" IOM CDF READ VAR DIM: dummy variable "//&
                  &              "dimension "//TRIM(fct_str(ji))//" not used.")
            ENDIF
         ENDDO

         ! reorder dimension to ('x','y','z','t')
         CALL dim_reorder(tl_dim(:))
 
         tf_dim(:)=dim_copy(tl_dim(:))

         ! clean
         CALL dim_clean(tl_dim(:))

      ENDIF

   END FUNCTION iom_cdf__read_var_dim
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_cdf__read_var_att(td_file, id_varid, id_natt) &
         & RESULT (tf_att)
   !-------------------------------------------------------------------
   !> @brief This subroutine read variable attributes
   !> in an opened netcdf file.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_file   file structure
   !> @param[in] id_varid  variable id
   !> @param[in] id_natt   number of attributes
   !> @return array of attribute structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(IN) :: td_file
      INTEGER(i4), INTENT(IN) :: id_varid
      INTEGER(i4), INTENT(IN) :: id_natt      

      ! function
      TYPE(TATT), DIMENSION(id_natt) :: tf_att

      ! local variable

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( id_natt > 0 )THEN
      
         ! read attributes
         DO ji = 1, id_natt
            CALL logger_trace( " IOM CDF READ VAR ATT: get attribute "//&
               &               TRIM(fct_str(ji))//" in file "//&
               &               TRIM(td_file%c_name) &
            )

            tf_att(ji)=iom_cdf_read_att(td_file, id_varid, ji)

         ENDDO

      ELSE

         CALL logger_debug( " IOM CDF READ VAR ATT: no attribute for variable " )

      ENDIF

   END FUNCTION iom_cdf__read_var_att
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf__read_var_value(td_file, td_var, id_start, id_count)
   !-------------------------------------------------------------------
   !> @brief This subroutine read variable value
   !> in an opened netcdf file.
   !> @details
   !> Optionaly, start indices and number of indices selected along each dimension 
   !> could be specify in a 4 dimension array (/'x','y','z','t'/)
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015 
   !> - use scale factor and offset, as soon as read variable value
   !> @date January, 2019
   !> - read array in netcdf file, level by level, and time step by time step
   !> - apply scale factor and offset, level by level, and time step by time step
   !>
   !> @param[in] td_file   file structure
   !> @param[inout] td_var variable structure
   !> @param[in] id_start  index in the variable from which the data values will be read
   !> @param[in] id_count  number of indices selected along each dimension
   !> @return variable structure completed 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE),               INTENT(IN)    :: td_file
      TYPE(TVAR) ,               INTENT(INOUT) :: td_var
      INTEGER(i4), DIMENSION(:), INTENT(IN),   OPTIONAL :: id_start
      INTEGER(i4), DIMENSION(:), INTENT(IN),   OPTIONAL :: id_count

      ! local variable
      INTEGER(i4)                                    :: il_status
      INTEGER(i4)                                    :: il_tmp1
      INTEGER(i4)                                    :: il_tmp2
      INTEGER(i4)                                    :: il_varid
      INTEGER(i4), DIMENSION(ip_maxdim)              :: il_start
      INTEGER(i4), DIMENSION(ip_maxdim)              :: il_count
      INTEGER(i4), DIMENSION(ip_maxdim)              :: il_start_ord
      INTEGER(i4), DIMENSION(ip_maxdim)              :: il_count_ord
      INTEGER(i4), DIMENSION(ip_maxdim)              :: il_start_tmp
      INTEGER(i4), DIMENSION(ip_maxdim)              :: il_count_tmp

      REAL(dp)   , DIMENSION(:,:,:,:)  , ALLOCATABLE :: dl_value
      REAL(dp)   , DIMENSION(:,:,:,:)  , ALLOCATABLE :: dl_tmp

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      ! check if variable in file structure
      il_varid=var_get_id(td_file%t_var(:),TRIM(td_var%c_name))
      IF( il_varid /= 0 )THEN

         ! check id_count and id_start optionals parameters...
         IF( (       PRESENT(id_start)  .AND. (.NOT. PRESENT(id_count))) .OR. &
             ((.NOT. PRESENT(id_start)) .AND.        PRESENT(id_count) ) )THEN
            CALL logger_warn( "IOM CDF READ VAR VALUE: id_start and id_count"//&
               & " should be both specify")
         ENDIF
         IF( PRESENT(id_start).AND.PRESENT(id_count) )THEN

            IF( SIZE(id_start(:)) /= ip_maxdim .OR. &
            &   SIZE(id_count(:)) /= ip_maxdim )THEN
               CALL logger_error("IOM CDF READ VAR: dimension of array start"//&
                  &  " or count are invalid to read variable "//&
                  &  TRIM(td_var%c_name)//" in file "//TRIM(td_file%c_name) )
            ENDIF

            ! change dimension order from ('x','y','z','t')
            il_start(:)=dim_reorder_xyzt2(td_var%t_dim, id_start(:))
            il_count(:)=dim_reorder_xyzt2(td_var%t_dim, id_count(:))

            ! keep ordered array ('x','y','z','t')
            il_start_ord(:)=id_start(:)
            il_count_ord(:)=id_count(:)

         ELSE

            ! change dimension order from ('x','y','z','t')
            il_start(:)=(/1,1,1,1/)
            il_count(:)=dim_reorder_xyzt2(td_var%t_dim(:),td_var%t_dim(:)%i_len)

            ! keep ordered array ('x','y','z','t')
            il_start_ord(:)=(/1,1,1,1/)
            il_count_ord(:)=td_var%t_dim(:)%i_len

         ENDIF

         ! check dimension
         IF( .NOT. ALL(il_start_ord(:)>=(/1,1,1,1/)) )THEN

            CALL logger_error( "IOM CDF READ VAR VALUE: start indices should"//&
            &  " be greater than or equal to 1")

         ENDIF

         IF(.NOT.ALL(il_start_ord(:)+il_count_ord(:)-1 <= &
            &  (/td_var%t_dim( 1 )%i_len,&
            &    td_var%t_dim( 2 )%i_len,&
            &    td_var%t_dim( 3 )%i_len,&
            &    td_var%t_dim( 4 )%i_len &
            &                                            /)) )THEN

            DO ji = 1, ip_maxdim
               il_tmp1=il_start_ord(ji)+il_count_ord(ji)-1
               il_tmp2=td_var%t_dim(ji)%i_len
               CALL logger_debug( "IOM CDF READ VAR VALUE: start + count -1:"//&
               &  TRIM(fct_str(il_tmp1))//" variable dimension"//&
               &  TRIM(fct_str(il_tmp2)))
            ENDDO
            CALL logger_error( "IOM CDF READ VAR VALUE: start + count exceed "//&
            &  "variable dimension for "//TRIM(td_var%c_name) )

         ELSE

            ! Allocate space to hold variable value (disorder)
            ALLOCATE(dl_value( il_count(1), &
               &               il_count(2), &
               &               il_count(3), &
               &               il_count(4)),&
               &               stat=il_status)
            IF( il_status /= 0 )THEN

              CALL logger_fatal( &
               &  "IOM CDF READ VAR VALUE: not enough space to put variable "//&
               &  TRIM(td_var%c_name))

            ENDIF

            ! read values
            CALL logger_debug( &
            &  "IOM CDF READ VAR VALUE: get variable "//TRIM(td_var%c_name)//&
            &  " in file "//TRIM(td_file%c_name))

            il_start_tmp(:)=il_start(:)
            il_count_tmp(:)=il_count(:)
            DO jl=il_start(jp_L),il_start(jp_L)+il_count(jp_L)-1
               il_start_tmp(jp_L)=jl
               il_count_tmp(jp_L) = 1
               DO jk=il_start(jp_K),il_start(jp_K)+il_count(jp_K)-1
                  il_start_tmp(jp_K)=jk
                  il_count_tmp(jp_K)=1 
                  il_status = NF90_GET_VAR( td_file%i_id, il_varid,           &
                  &                                       dl_value(:,:,jk,jl),&
                  &                                       start = il_start_tmp(:),&
                  &                                       count = il_count_tmp(:) )
            !   il_status = NF90_GET_VAR( td_file%i_id, il_varid,           &
            !   &                                       dl_value(:,:,:,:),&
               CALL iom_cdf__check(il_status,"IOM CDF READ VAR VALUE: ")
               ENDDO
            ENDDO

            ! Allocate space to hold variable value in structure
            IF( ASSOCIATED(td_var%d_value) )THEN
               DEALLOCATE(td_var%d_value)   
            ENDIF
 
            ! new dimension length
            td_var%t_dim(:)%i_len=il_count_ord(:)

!>   dummy patch for pgf95
            ALLOCATE( dl_tmp( td_var%t_dim(1)%i_len, &
            &                 td_var%t_dim(2)%i_len, &
            &                 td_var%t_dim(3)%i_len, &
            &                 td_var%t_dim(4)%i_len),&
            &        stat=il_status)
            IF(il_status /= 0 )THEN

               CALL logger_fatal( &
               &  "IOM CDF READ VAR VALUE: not enough space to put variable "//&
               &  TRIM(td_var%c_name)//&
               &  " in variable structure")
            ENDIF
            dl_tmp(:,:,:,:)=td_var%d_fill

            ! reshape values to be ordered as ('x','y','z','t')
            dl_tmp(:,:,:,:)=dim_reshape_2xyzt(td_var%t_dim(:), &
            &                                 dl_value(:,:,:,:))

            DEALLOCATE(dl_value)

            ALLOCATE(td_var%d_value( td_var%t_dim(1)%i_len, &
            &                        td_var%t_dim(2)%i_len, &
            &                        td_var%t_dim(3)%i_len, &
            &                        td_var%t_dim(4)%i_len),&
            &        stat=il_status)
            IF(il_status /= 0 )THEN

               CALL logger_fatal( &
               &  "IOM CDF READ VAR VALUE: not enough space to put variable "//&
               &  TRIM(td_var%c_name)//&
               &  " in variable structure")

            ENDIF
!            ! FillValue by default
!            td_var%d_value(:,:,:,:)=td_var%d_fill
!
!            ! reshape values to be ordered as ('x','y','z','t')
!            td_var%d_value(:,:,:,:)=dim_reshape_2xyzt(td_var%t_dim(:), &
!            &                                         dl_value(:,:,:,:))
!
!            DEALLOCATE(dl_value)

            DO jl=1,td_var%t_dim(jp_L)%i_len
               DO jk=1,td_var%t_dim(jp_K)%i_len
                  DO jj=1,td_var%t_dim(jp_J)%i_len
                     DO ji=1,td_var%t_dim(jp_I)%i_len
                        td_var%d_value(ji,jj,jk,jl)=dl_tmp(ji,jj,jk,jl)
                     ENDDO
                  ENDDO
               ENDDO
            ENDDO
            DEALLOCATE(dl_tmp)
!<   dummy patch for pgf95

            ! force to change _FillValue to avoid mistake 
            ! with dummy zero _FillValue
            IF( td_var%d_fill == 0._dp )THEN
               CALL var_chg_FillValue(td_var)
            ENDIF

            ! use scale factor and offset
            DO jl=1,td_var%t_dim(jp_L)%i_len
               DO jk=1,td_var%t_dim(jp_K)%i_len
                  WHERE( td_var%d_value(:,:,jk,jl) /= td_var%d_fill )
                     td_var%d_value(:,:,jk,jl) = &
                     &  td_var%d_value(:,:,jk,jl)*td_var%d_scf + td_var%d_ofs
                  END WHERE
               ENDDO
            ENDDO

         ENDIF
      ELSE
         CALL logger_error( &
         &  "IOM CDF READ VAR VALUE: no variable "//TRIM(td_var%c_name)//&
         &  " in file structure "//TRIM(td_file%c_name))
      ENDIF

   END SUBROUTINE iom_cdf__read_var_value
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf_write_header(td_file, cd_dimorder, td_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine write file header in an opened netcdf file.
   !>
   !> @details
   !> optionally, you could specify dimension order (default 'xyzt'),
   !> and/or dimension structure to be used.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2015 
   !> - add dimension order option 
   !> @date August, 2017
   !> - split write_file into write_header and write_var
   !> - add dimension structure as optional argument
   !> @date September, 2017
   !> - do not check variable dimension if dimension forced
   !>
   !> @param[inout] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE)                           , INTENT(INOUT) :: td_file
      CHARACTER(LEN=*)                      , INTENT(IN   ), OPTIONAL :: cd_dimorder
      TYPE(TDIM )     , DIMENSION(ip_maxdim), INTENT(IN   ), OPTIONAL :: td_dim

      ! local variable
      INTEGER(i4)                            :: il_status
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_value

      CHARACTER(LEN=lc)                      :: cl_dimorder

      LOGICAL                                :: ll_chkdim

      TYPE(TVAR), DIMENSION(ip_maxdim)       :: tl_var

      TYPE(TDIM), DIMENSION(ip_maxdim)       :: tl_dim

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jvar
      !----------------------------------------------------------------

      cl_dimorder='xyzt'
      IF( PRESENT(cd_dimorder) ) cl_dimorder=TRIM(cd_dimorder)

      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  " IOM CDF WRITE HEADER: no id associated to file "//&
         &  TRIM(td_file%c_name)//". Check if file is opened.")

      ELSE
         IF( td_file%l_wrt )THEN

            ! remove dummy variable
            CALL file_del_var(td_file,'no0d')
            CALL file_del_var(td_file,'no1d')
            CALL file_del_var(td_file,'no2d')
            CALL file_del_var(td_file,'no3d')

            DO ji = 1, td_file%i_nvar
               CALL var_check_dim( td_file%t_var(ji) )
            ENDDO

            IF( PRESENT(td_dim) )THEN
               ! special case to rebuild mpp layout
               DO ji=1,ip_maxdim
                  IF( td_dim(ji)%l_use ) CALL file_move_dim(td_file, td_dim(ji))
               ENDDO
            ELSE
               ! save useful dimension
               IF( ASSOCIATED(td_file%t_var) )THEN
                  tl_dim(:) = var_max_dim( td_file%t_var(:) )

                  DO ji=1,ip_maxdim
                     IF( tl_dim(ji)%l_use ) CALL file_move_dim(td_file, tl_dim(ji))
                  ENDDO
                  ! clean
                  CALL dim_clean(tl_dim(:))
               ENDIF
            ENDIF

            ! Enter in define mode
            IF( .NOT. td_file%l_def )THEN
               CALL logger_debug( &
               &  " IOM CDF WRITE HEADER: Enter define mode, file "//&
               &  TRIM(td_file%c_name))

               ! Enter define mode
               il_status=NF90_REDEF(td_file%i_id)
               CALL iom_cdf__check(il_status,"IOM CDF WRITE HEADER: ")

               td_file%l_def=.TRUE.
            ENDIF            

            ! write dimension definition in header of file
            IF( TRIM(cl_dimorder) /= 'xyzt' )THEN
               CALL dim_reorder(td_file%t_dim(:),TRIM(cl_dimorder))
               DO jvar=1,td_file%i_nvar
                  CALL logger_debug("VAR REORDER: "//TRIM(td_file%t_var(jvar)%c_name))
                  CALL var_reorder(td_file%t_var(jvar),TRIM(cl_dimorder))
               ENDDO
            ENDIF

            ! write dimension in file
            DO ji = 1, ip_maxdim
               IF( td_file%t_dim(ji)%l_use )THEN
                  CALL iom_cdf__write_dim_def(td_file, td_file%t_dim(ji))

                  ! write dimension variable
                  ALLOCATE(il_value(td_file%t_dim(ji)%i_len))
                  il_value(:)=(/(jj,jj=1,td_file%t_dim(ji)%i_len)/)

                  tl_var(ji)=var_init( fct_upper(td_file%t_dim(ji)%c_sname), &
                  &                    il_value(:),                          &
                  &                    td_dim=td_file%t_dim(ji) )

                  DEALLOCATE(il_value)

                  ! do not use FillValue for dimension variable
                  CALL var_del_att(tl_var(ji), "_FillValue")
                   
                  ! write dimension variable definition in header of file
                  CALL iom_cdf__write_var_def(td_file,tl_var(ji))
                  
               ENDIF
            ENDDO

            ! write global attibute in file
            DO ji = 1, td_file%i_natt
               CALL iom_cdf__write_att_def(td_file, NF90_GLOBAL, td_file%t_att(ji))
            ENDDO
 
            ! write variable definition in header of file
            ll_chkdim=.TRUE.
            IF( PRESENT(td_dim) )THEN
               ! special case to rebuild mpp layout
               ! do not check dimension length of variable
               ll_chkdim=.FALSE.
            ENDIF
            DO ji=1,td_file%i_nvar
               CALL iom_cdf__write_var_def(td_file, td_file%t_var(ji),ll_chkdim)
            ENDDO

            ! Leave define mode
            IF( td_file%l_def )THEN
               CALL logger_debug( &
               &  " IOM CDF WRITE HEADER: Leave define mode, file "//&
               &  TRIM(td_file%c_name))

               ! Leave define mode
               il_status=NF90_ENDDEF(td_file%i_id)
               CALL iom_cdf__check(il_status,"IOM CDF WRITE HEADER: ")

               td_file%l_def=.FALSE.
            ENDIF

            ! write dimension variable in file
            DO ji = 1, ip_maxdim
               IF( td_file%t_dim(ji)%l_use )THEN
                  ! do not use FillValue for dimension variable
                  CALL var_del_att(tl_var(ji), "_FillValue")
                  CALL iom_cdf__write_var(td_file,tl_var(ji))
               ENDIF
            ENDDO
            ! clean
            CALL var_clean(tl_var(:))

         ELSE

            CALL logger_error( &
            &  "IOM CDF WRITE HEADER: try to write in file "//TRIM(td_file%c_name)//&
            &  ", not opened in write mode")

         ENDIF
      ENDIF

   END SUBROUTINE iom_cdf_write_header
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf_write_var(td_file, cd_dimorder, id_start, id_count)!, ld_first)
   !-------------------------------------------------------------------
   !> @brief This subroutine write variable(s) in an opened netcdf file.
   !>
   !> @details
   !> optionally, you could specify dimension order (default 'xyzt')
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2015 
   !> - add dimension order option 
   !> @date August, 2017
   !> - add start and count array as optional argument
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] td_var       array of variable structure
   !> @param[in] cd_dimorder  dimension order
   !> @param[in] id_start     index in the variable from which the data values 
   !> will be read
   !> @param[in] id_count     number of indices selected along each dimension
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE)                   , INTENT(INOUT) :: td_file
      CHARACTER(LEN=*)              , INTENT(IN   ), OPTIONAL :: cd_dimorder
      INTEGER(i4)     , DIMENSION(:), INTENT(IN   ), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:), INTENT(IN   ), OPTIONAL :: id_count
      ! local variable
      CHARACTER(LEN=lc)                      :: cl_dimorder

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      cl_dimorder='xyzt'
      IF( PRESENT(cd_dimorder) ) cl_dimorder=TRIM(cd_dimorder)

      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  " IOM CDF WRITE VAR: no id associated to file "//TRIM(td_file%c_name))

      ELSE
         IF( td_file%l_wrt )THEN

            ! write variable in file
            DO ji = 1, td_file%i_nvar

               ! change dimension order
               IF( TRIM(cl_dimorder) /= 'xyzt' )THEN
                  CALL logger_debug("VAR REORDER: "//TRIM(td_file%t_var(ji)%c_name))
                  CALL var_reorder(td_file%t_var(ji),TRIM(cl_dimorder))
               ENDIF

               IF( ASSOCIATED(td_file%t_var(ji)%d_value) )THEN
                  ! write variable value in file
                  CALL iom_cdf__write_var_value( td_file, td_file%t_var(ji), &
                     &                           id_start, id_count)
               ENDIF
            ENDDO

         ELSE

            CALL logger_error( &
            &  "IOM CDF WRITE VAR: try to write in file "//TRIM(td_file%c_name)//&
            &  ", not opened in write mode")

         ENDIF
      ENDIF

   END SUBROUTINE iom_cdf_write_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf__write_dim_def(td_file, td_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine define a dimension in the header of a netcdf 
   !> file.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date August, 2017
   !> - rename in write_dim_def
   !> - do not check define mode here anymore
   !>
   !> @param[inout] td_file   file structure
   !> @param[inout] td_dim    dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file
      TYPE(TDIM),  INTENT(INOUT) :: td_dim

      ! local variable
      INTEGER(i4) :: il_status
      !----------------------------------------------------------------

      IF( td_dim%l_use )THEN
         IF( td_dim%l_uld )THEN
            ! write unlimited dimension
            CALL logger_trace( &
            &  "IOM CDF WRITE FILE DIM: write unlimited dimension "//&
            &  TRIM(td_dim%c_name)//" in file "//TRIM(td_file%c_name))

            il_status=NF90_DEF_DIM(td_file%i_id, fct_upper(td_dim%c_sname), &
            &                      NF90_UNLIMITED, td_dim%i_id)
            CALL iom_cdf__check(il_status,"IOM CDF WRITE FILE DIM: ")

         ELSE
            ! write not unlimited dimension
            CALL logger_debug( &
            &  "IOM CDF WRITE FILE DIM: write dimension "//TRIM(td_dim%c_name)//&
            &  " in file "//TRIM(td_file%c_name))
            
            CALL logger_debug("IOM CDF WRITE FILE DIM: id "//TRIM(fct_str(td_file%i_id))//&
               & " sname "//TRIM(td_dim%c_sname))
            il_status=NF90_DEF_DIM(td_file%i_id, fct_upper(td_dim%c_sname), &
            &                      td_dim%i_len, td_dim%i_id)
            CALL iom_cdf__check(il_status,"IOM CDF WRITE FILE DIM: ")

         ENDIF
      ENDIF

   END SUBROUTINE iom_cdf__write_dim_def
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf__write_att_def(td_file, id_varid, td_att)
   !-------------------------------------------------------------------
   !> @brief This subroutine write a variable attribute in
   !> an opened netcdf file.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date August, 2017
   !> - rename in write_att_def
   !> - do not check define mode here anymore
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] id_varid     variable id. use NF90_GLOBAL to write 
   !> global attribute in a file
   !> @param[in] td_att       attribute structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file
      INTEGER(i4), INTENT(IN)    :: id_varid
      TYPE(TATT),  INTENT(IN)    :: td_att

      ! local variable
      INTEGER(i4) :: il_status
      !----------------------------------------------------------------

      !! put attribute value
      CALL logger_trace( &
      &  "IOM CDF WRITE FILE ATT: write attribute "//TRIM(td_att%c_name)//&
      &  " of variable "//TRIM(fct_str(id_varid))//&
      &  " in file "//TRIM(td_file%c_name))
      SELECT CASE( td_att%i_type )

         CASE(NF90_CHAR)
            ! put the attribute
            il_status = NF90_PUT_ATT(td_file%i_id, id_varid, &
            &  td_att%c_name, td_att%c_value )
            CALL iom_cdf__check(il_status,"IOM CDF WRITE FILE ATT: ")

         CASE(NF90_BYTE, NF90_SHORT, NF90_INT, NF90_FLOAT, NF90_DOUBLE)
            ! put the attribute
            il_status = NF90_PUT_ATT(td_file%i_id, id_varid, &
            &  td_att%c_name, td_att%d_value )
            CALL iom_cdf__check(il_status,"IOM CDF WRITE FILE ATT: ")

      END SELECT

   END SUBROUTINE iom_cdf__write_att_def
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf__write_var(td_file, td_var, id_start, id_count)
   !-------------------------------------------------------------------
   !> @brief This subroutine write a variable in an opened netcdf file.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2015
   !> - do not force to use zero as FillValue for any meshmask variable
   !> @date August, 2017
   !> - add start and count array as optional argument
   !> - variable definition now done in write_var_def 
   !>
   !> @param[inout] td_file   file structure
   !> @param[inout] td_var    variable structure
   !> @param[in] id_start     index in the variable from which the data 
   !> values will be read
   !> @param[in] id_count     number of indices selected along each dimension
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE)              , INTENT(INOUT) :: td_file
      TYPE(TVAR)               , INTENT(INOUT) :: td_var
      INTEGER(i4), DIMENSION(:), INTENT(IN   ), OPTIONAL :: id_start
      INTEGER(i4), DIMENSION(:), INTENT(IN   ), OPTIONAL :: id_count

      ! local variable
      !----------------------------------------------------------------

      IF( ASSOCIATED(td_var%d_value) )THEN
         
         ! write variable value in file
         CALL iom_cdf__write_var_value(td_file, td_var, id_start, id_count)
      ENDIF

   END SUBROUTINE iom_cdf__write_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf__write_var_def(td_file, td_var,ld_chkdim)
   !-------------------------------------------------------------------
   !> @brief This subroutine define variable in the header of a netcdf
   !> file.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2017
   !> - add option to not check dimension length
   !> @date August, 2017
   !> - extract from write_var
   !> @date November, 2019
   !> - compress 2D,3D, and 4D variables in netcdf4 files
   !> - set no_fill mode on every variables
   !>
   !> @param[in] td_file   file structure
   !> @param[in] td_var    variable structure
   !> @param[in] ld_chkdim check dimension length
   !> @return  variable id
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(IN   ) :: td_file
      TYPE(TVAR),  INTENT(INOUT) :: td_var
      LOGICAL    , INTENT(IN   ), OPTIONAL :: ld_chkdim

      ! local variable
      INTEGER(i4)                       :: il_status
      INTEGER(i4)                       :: il_ind
      INTEGER(i4)                       :: il_deflate
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_dimid

      LOGICAL                           :: ll_chg
      LOGICAL                           :: ll_defdim

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      ll_defdim=file_check_var_dim(td_file, td_var, ld_chkdim )
      IF( ll_defdim )THEN
         ll_chg=.TRUE.
         DO ji=1,ip_maxdim
            IF( TRIM(fct_lower(cp_dimorder(ji:ji))) == &
            &   TRIM(fct_lower(td_var%c_name)) )THEN
               ll_chg=.FALSE.
               CALL logger_trace(TRIM(fct_lower(td_var%c_name))//' is var dimension')
               EXIT
            ENDIF
         ENDDO
         ! ugly patch until NEMO do not force to use 0. as FillValue 
         IF( ll_chg )THEN
            ! not a dimension variable
            ! change FillValue
            SELECT CASE( TRIM(fct_lower(td_var%c_name)) )
               CASE DEFAULT
                  CALL var_chg_FillValue(td_var,0._dp)
               CASE('nav_lon','nav_lat', &
                  & 'glamt','glamu','glamv','glamf', &
                  & 'gphit','gphiu','gphiv','gphif', &
                  & 'e1t','e1u','e1v','e1f',         &
                  & 'e2t','e2u','e2v','e2f','ff',    &
                  & 'gcost','gcosu','gcosv','gcosf', &
                  & 'gsint','gsinu','gsinv','gsinf', &
                  & 'mbathy','misf','isf_draft','isfdraft',     &
                  & 'hbatt','hbatu','hbatv','hbatf', &
                  & 'gsigt','gsigu','gsigv','gsigf', &
                  & 'e3t_0','e3u_0','e3v_0','e3w_0', &
                  & 'e3f_0','gdepw_1d','gdept_1d',   &
                  & 'e3tp','e3wp','gdepw_0','rx1',   &
                  & 'gdept_0','gdepu','gdepv',       &
                  & 'hdept','hdepw','e3w_1d','e3t_1d',&
                  & 'tmask','umask','vmask','fmask'  )
                  ! do not change for coordinates and meshmask variables
               !CASE('no3','o2','po4','si', &
               !   & 'solub','ndepo','dust','fr_par', &
               !   & 'alk','dic','doc','fer' )
               !   ! do not change for BGC variables
            END SELECT
         ENDIF         

         ! forced to use float type
         IF( td_var%d_unf /= 1. .AND. td_var%i_type==NF90_SHORT )THEN
            td_var%i_type=NF90_FLOAT
         ENDIF

         IF( ALL( .NOT. td_var%t_dim(:)%l_use ) )THEN
            CALL logger_debug( &
            &  "IOM CDF WRITE VAR DEF scalar: define variable "//&
            &  TRIM(td_var%c_name)//" in file "//TRIM(td_file%c_name))
            ! scalar value
            il_status = NF90_DEF_VAR(td_file%i_id,          &
               &                     TRIM(td_var%c_name),   &
               &                     td_var%i_type,         &
               &                     varid=td_var%i_id) 
            CALL iom_cdf__check(il_status,"IOM CDF WRITE VAR DEF: ")
         ELSE

            ! check which dimension use
            jj=0
            il_dimid(:)=0
            ! reorder dimension, so unused dimension won't be written
            DO ji = 1,  ip_maxdim
               IF( td_var%t_dim(ji)%l_use )THEN
                  jj=jj+1
                  il_dimid(jj)=dim_get_id(td_file%t_dim(:),td_var%t_dim(ji)%c_name)
               ENDIF
            ENDDO

            CALL logger_debug( &
            &  "IOM CDF WRITE VAR DEF: define dimension to be used for variable "//&
            &  TRIM(td_var%c_name)//" in file "//TRIM(td_file%c_name))

            DO ji=1,jj
               CALL logger_debug("IOM CDF WRITE VAR DEF: dimname : "//TRIM(td_var%t_dim(ji)%c_name))
               CALL logger_debug("IOM CDF WRITE VAR DEF: dimid "//TRIM(fct_str(il_dimid(ji))) )
            ENDDO

            il_deflate=0
            ! compress 2D,3D, and 4D variables
            if( jj > 1 ) il_deflate=1
            CALL logger_debug("IOM CDF WRITE VAR DEF: deflate = "//TRIM(fct_str(il_deflate)))
            il_status = NF90_DEF_VAR(td_file%i_id,       &
               &                     TRIM(td_var%c_name),&
               &                     td_var%i_type,      &
               &                     il_dimid(1:jj),     &
               &                     varid=td_var%i_id,  &
               &                     deflate_level=il_deflate )
            CALL iom_cdf__check(il_status,"IOM CDF WRITE VAR DEF: ")
         ENDIF
         CALL logger_debug("IOM CDF WRITE VAR DEF: type = "//TRIM(fct_str(td_var%i_type)))

         ! remove useless attribute
         il_ind=att_get_index( td_var%t_att(:), "ew_overlap" )
         IF( il_ind /= 0 )THEN
            IF( td_var%t_att(il_ind)%d_value(1) == -1 )THEN
               CALL var_del_att(td_var, td_var%t_att(il_ind))
            ENDIF
         ENDIF

         DO ji = 1, td_var%i_natt
            CALL logger_debug( &
            &  " IOM CDF WRITE VAR DEF: put attribute "//TRIM(td_var%t_att(ji)%c_name)//&
            &  " for variable "//TRIM(td_var%c_name)//&
            &  " in file "//TRIM(td_file%c_name) )

            ! forced FillValue to have same type than variable
            IF( TRIM(td_var%t_att(ji)%c_name) == '_FillValue' )THEN

               td_var%t_att(ji)%i_type=td_var%i_type

               SELECT CASE(td_var%t_att(ji)%i_type)
                  CASE(NF90_BYTE)
                     il_status = NF90_DEF_VAR_FILL(td_file%i_id,  &
                        &                          td_var%i_id,   &
                        &                          1_i4,          &
                        &                          INT(td_var%d_fill,i1))
                  CASE(NF90_SHORT)
                     il_status = NF90_DEF_VAR_FILL(td_file%i_id,  &
                        &                          td_var%i_id,   &
                        &                          1_i4,          &
                        &                          INT(td_var%d_fill,i2))
                  CASE(NF90_INT)
                     il_status = NF90_DEF_VAR_FILL(td_file%i_id,  &
                        &                          td_var%i_id,   &
                        &                          1_i4,          &
                        &                          INT(td_var%d_fill,i4))
                  CASE(NF90_FLOAT)
                     il_status = NF90_DEF_VAR_FILL(td_file%i_id,  &
                        &                          td_var%i_id,   &
                        &                          1_i4,          &
                        &                          REAL(td_var%d_fill,sp))
                  CASE(NF90_DOUBLE)
                     il_status = NF90_DEF_VAR_FILL(td_file%i_id,  &
                        &                          td_var%i_id,   &
                        &                          1_i4,          &
                        &                          REAL(td_var%d_fill,dp))
               END SELECT
               CALL iom_cdf__check(il_status,"IOM CDF WRITE VAR DEF FILL: ")

            ELSE

               SELECT CASE(td_var%t_att(ji)%i_type)
                  CASE(NF90_CHAR)
                     IF( TRIM(td_var%t_att(ji)%c_value) /= '' )THEN
                        il_status = NF90_PUT_ATT(td_file%i_id, td_var%i_id,      &
                        &                        TRIM(td_var%t_att(ji)%c_name),  &
                        &                        TRIM(td_var%t_att(ji)%c_value)  )
                     ENDIF
                  CASE(NF90_BYTE)
                     il_status = NF90_PUT_ATT(td_file%i_id,                   &
                     &                        td_var%i_id,                    &
                     &                        TRIM(td_var%t_att(ji)%c_name),  &
                     &                        INT(td_var%t_att(ji)%d_value(:),i1))
                  CASE(NF90_SHORT)
                     il_status = NF90_PUT_ATT(td_file%i_id,                   &
                     &                        td_var%i_id,                    &
                     &                        TRIM(td_var%t_att(ji)%c_name),  &
                     &                        INT(td_var%t_att(ji)%d_value(:),i2))
                  CASE(NF90_INT)
                     il_status = NF90_PUT_ATT(td_file%i_id,                   &
                     &                        td_var%i_id,                    &
                     &                        TRIM(td_var%t_att(ji)%c_name),  &
                     &                        INT(td_var%t_att(ji)%d_value(:),i4))
                  CASE(NF90_FLOAT)
                     il_status = NF90_PUT_ATT(td_file%i_id,                   &
                     &                        td_var%i_id,                    &
                     &                        TRIM(td_var%t_att(ji)%c_name),  &
                     &                        REAL(td_var%t_att(ji)%d_value(:),sp))
                  CASE(NF90_DOUBLE)
                     il_status = NF90_PUT_ATT(td_file%i_id,                   &
                     &                        td_var%i_id,                    &
                     &                        TRIM(td_var%t_att(ji)%c_name),  &
                     &                        REAL(td_var%t_att(ji)%d_value(:),dp))
               END SELECT
               CALL iom_cdf__check(il_status,"IOM CDF WRITE VAR DEF: ")

            ENDIF

         ENDDO

      ENDIF
      CALL logger_debug("IOM CDF WRITE VAR DEF: type = "//TRIM(fct_str(td_var%i_type)))

   END SUBROUTINE iom_cdf__write_var_def
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_cdf__write_var_value(td_file, td_var, id_start, id_count)
   !-------------------------------------------------------------------
   !> @brief This subroutine put variable value in an opened netcdf file.
   !>
   !> @details
   !> The variable is written in the type define in variable structure.
   !> Only dimension used are printed, and fillValue in array are
   !> replaced by default fill values defined in module netcdf for each type. 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - reuse scale factor and offset, before writing variable
   !> @August, 2017
   !> - use start and count array to write variable value
   !>
   !> @param[in] td_file   file structure
   !> @param[in] td_var    variable structure
   !> @param[in] id_start  index in the variable from which the data 
   !> values will be read
   !> @param[in] id_count  number of indices selected along each dimension
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE)                   , INTENT(IN   ) :: td_file
      TYPE(TVAR)                    , INTENT(IN   ) :: td_var
      INTEGER(i4)     , DIMENSION(:), INTENT(IN   ), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:), INTENT(IN   ), OPTIONAL :: id_count

      ! local variable
      INTEGER(i4)                       :: il_status
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_order
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_shape
      REAL(dp),    DIMENSION(:,:,:,:), ALLOCATABLE :: dl_value

      INTEGER(i4), DIMENSION(ip_maxdim) :: il_start
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_count      

      INTEGER(i4), DIMENSION(ip_maxdim) :: il_start_ord
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_count_ord      
      ! loop indices
      INTEGER(i4) :: ji, jj
      !----------------------------------------------------------------

      ! check which dimension use
      CALL logger_debug( &
      &  "IOM CDF WRITE VAR VALUE: get dimension to be used for variable "//&
      &  TRIM(td_var%c_name)//" in file "//TRIM(td_file%c_name))
 
      il_start(:)=1
      IF( PRESENT(id_start) ) il_start(:)=id_start(:)

      il_count(:)=td_var%t_dim(:)%i_len
      IF( PRESENT(id_count) ) il_count(:)=id_count(:)
 
      ! use scale factor and offset
      WHERE( td_var%d_value(:,:,:,:) /= td_var%d_fill )
         td_var%d_value(:,:,:,:) = &
         &  (td_var%d_value(:,:,:,:)-td_var%d_ofs)/td_var%d_scf
      END WHERE

      jj=0
      DO ji = 1, ip_maxdim
         IF( td_var%t_dim(ji)%l_use )THEN
            jj=jj+1
            il_order(jj)=ji
            il_shape(jj)=td_var%t_dim(ji)%i_len
         ENDIF
      ENDDO
      ! dimension not use
      DO ji = 1, ip_maxdim
         IF( .NOT. td_var%t_dim(ji)%l_use )THEN
            jj=jj+1
            il_order(jj)=ji
            il_shape(jj)=td_var%t_dim(ji)%i_len
         ENDIF
      ENDDO

      ALLOCATE( dl_value( il_shape(1),il_shape(2),il_shape(3),il_shape(4) ) )

      ! reshape array, so useless dimension won't be written
      dl_value(:,:,:,:)=RESHAPE(source=td_var%d_value(:,:,:,:),&
      &                         SHAPE = il_shape(:), &
      &                         ORDER = il_order(:))

      DO ji = 1, ip_maxdim
         il_start_ord(il_order(ji))=il_start(ji)
         il_count_ord(il_order(ji))=il_count(ji)
      ENDDO

      ! put value
      CALL logger_debug( &
      &  "IOM CDF WRITE VAR VALUE: put "//TRIM(td_var%c_name)//" value "//&
      &  "in file "//TRIM(td_file%c_name))
      il_status = NF90_PUT_VAR( td_file%i_id, td_var%i_id, dl_value(:,:,:,:), &
      &                      start=il_start_ord(:), &
      &                      count=il_count_ord(:) )
      CALL iom_cdf__check(il_status,"IOM CDF WRITE VAR VALUE ("//&
      &  TRIM(td_var%c_name)//") :" )

      DEALLOCATE( dl_value )

   END SUBROUTINE iom_cdf__write_var_value
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE iom_cdf
