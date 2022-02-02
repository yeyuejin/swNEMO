!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief
!> This module is a library to read/write dimg file.
!>
!> @details
!>    to open dimg file (create file structure):<br/>
!> @code
!>    CALL iom_rstdimg_open(td_file)
!> @endcode
!>       - td_file is file structure (see file.f90)
!>
!>    to write in dimg file:<br/>
!> @code
!>    CALL  iom_rstdimg_write_file(td_file)
!> @endcode
!>
!>    to close dimg file:<br/>
!> @code
!>    CALL iom_rstdimg_close(tl_file)
!> @endcode
!>
!>    to read one dimension in dimg file:<br/>
!> @code
!>    tl_dim = iom_rstdimg_read_dim(tl_file, id_dimid)
!> @endcode
!>    or
!> @code
!>    tl_dim = iom_rstdimg_read_dim(tl_file, cd_name)
!> @endcode
!>       - id_dimid is dimension id<br/>
!>       - cd_name is dimension name
!>
!>    to read one variable in dimg file:<br/>
!> @code
!>    tl_var = iom_rstdimg_read_var(td_file, id_varid, [id_start, id_count])
!> @endcode
!>    or
!> @code
!>    tl_var = iom_rstdimg_read_var(td_file, cd_name, [id_start, [id_count]])
!> @endcode
!>       - id_varid is variabale id
!>       - cd_name is variabale name or standard name
!>       - id_start is a integer(4) 1D array of index from which the data 
!>          values will be read [optional]
!>       - id_count is a integer(4) 1D array of the number of indices selected
!>          along each dimension [optional]
!>
!>    to get sub domain decomppistion in a dimg file:<br/>
!> @code
!>    CALL iom_rstdimg_get_mpp(td_file)
!> @endcode
!>
!> @author
!> J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date August, 2017 
!> - handle use of domain decomposition for monoproc file
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE iom_rstdimg

   USE netcdf                          ! nf90 library
   USE global                          ! global parameter
   USE kind                            ! F90 kind parameter
   USE fct                             ! basic useful function
   USE logger                          ! log file manager
   USE att                             ! attribute manager
   USE dim                             ! dimension manager
   USE var                             ! variable manager
   USE file                            ! file manager

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! type and variable
   PRIVATE ::  im_vnl !< variable name length 

   ! function and subroutine
   PUBLIC :: iom_rstdimg_open        !< open or create dimg file, return file structure
   PUBLIC :: iom_rstdimg_close       !< close dimg file
   PUBLIC :: iom_rstdimg_read_dim    !< read one dimension in an opened dimg file, return variable structure
   PUBLIC :: iom_rstdimg_read_var    !< read one variable  in an opened dimg file, return dimension structure
   PUBLIC :: iom_rstdimg_write_header!< write header in an opened dimg file
   PUBLIC :: iom_rstdimg_write_var   !< write variable in an opened dimg file
   PUBLIC :: iom_rstdimg_get_mpp     !< get sub domain decomppistion in a dimg file

   PRIVATE :: iom_rstdimg__get_info        ! get global information in an opened dimg file
   PRIVATE :: iom_rstdimg__get_file_var    ! read information about variable on an opened dimg file.
   PRIVATE :: iom_rstdimg__get_file_var_0d ! put information about scalar variable in file structure
   PRIVATE :: iom_rstdimg__get_file_var_1d ! put information about variable 1D in file structure
   PRIVATE :: iom_rstdimg__get_file_var_2d ! put information about variable 2D in file structure
   PRIVATE :: iom_rstdimg__get_file_var_3d ! put information about variable 3D in file structure
   PRIVATE :: iom_rstdimg__read_dim_id     ! read dimension structure in an opened dimg file, given variable id.
   PRIVATE :: iom_rstdimg__read_dim_name   ! read dimension structure in an opened dimg file, given variable name or standard name.
   PRIVATE :: iom_rstdimg__read_var_id     ! read variable value in an opened dimg file, given variable id.
   PRIVATE :: iom_rstdimg__read_var_name   ! read variable value in an opened dimg file, given variable name or standard name.
   PRIVATE :: iom_rstdimg__read_var_value  ! read variable value in an opened dimg file, for variable 1,2,3d
   PRIVATE :: iom_rstdimg__get_rec         ! compute record number before writing file
   PRIVATE :: iom_rstdimg__write_header    ! write header in an opened dimg file
   PRIVATE :: iom_rstdimg__write_var       ! write variables in an opened dimg file 

   ! module variable
   INTEGER(i4), PARAMETER :: im_vnl = 32 ! variable name length

   INTERFACE iom_rstdimg_read_dim
      MODULE PROCEDURE iom_rstdimg__read_dim_id
      MODULE PROCEDURE iom_rstdimg__read_dim_name
   END INTERFACE iom_rstdimg_read_dim

   INTERFACE iom_rstdimg_read_var
      MODULE PROCEDURE iom_rstdimg__read_var_id
      MODULE PROCEDURE iom_rstdimg__read_var_name
   END INTERFACE iom_rstdimg_read_var

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_rstdimg_open(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine open a dimg file in read or write mode.
   !> @details
   !> if try to open a file in write mode that did not exist, create it.<br/>
   !> if file already exist, get information about:
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
   !>
   !> @param[inout] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT)  :: td_file

      ! local variable
      LOGICAL           :: ll_exist
      LOGICAL           :: ll_open

      INTEGER(i4)       :: il_status
      !----------------------------------------------------------------

      ! check file existence
      ! WARNING may be some issue with dimg file !!!
      INQUIRE(FILE=TRIM(td_file%c_name), EXIST=ll_exist, OPENED=ll_open)
      IF( .NOT. ll_exist .OR. TRIM(td_file%c_type) /= 'dimg' )THEN

         IF( .NOT. td_file%l_wrt )THEN

            CALL logger_fatal( " OPEN: can not open dimg file "//&
            &               TRIM(td_file%c_name) )
         
         ELSE

            CALL logger_info( " CREATE: dimg file "//TRIM(td_file%c_name) )

            ! get free unit
            td_file%i_id=fct_getunit()

            OPEN( td_file%i_id, FILE=TRIM(td_file%c_name),&
            &                         FORM='UNFORMATTED',              &
            &                         ACCESS='DIRECT',                 &
            &                         STATUS='NEW',                    &
            &                         ACTION='WRITE',                  &
            &                         RECL=8,                          &
            &                         IOSTAT=il_status)
            CALL fct_err(il_status)
            IF( il_status /= 0 )THEN
               CALL logger_error("CREATE: dimg file "//&
               &              TRIM(td_file%c_name))
            ENDIF

         ENDIF

      ELSE

         IF( ll_open )THEN

            CALL logger_error( " OPEN: dimg file "//&
            &                TRIM(td_file%c_name)//" already opened")

         ELSE

            ! get free unit
            td_file%i_id=fct_getunit()

            ! open temporary in read only mode
            OPEN( td_file%i_id, FILE=TRIM(td_file%c_name),&
            &                         FORM='UNFORMATTED',              &
            &                         ACCESS='DIRECT',                 &
            &                         STATUS='OLD',                    &
            &                         ACTION='READ',                   &
            &                         RECL=8,                          &
            &                         IOSTAT=il_status)
            CALL fct_err(il_status)
            IF( il_status /= 0 )THEN
               CALL logger_error("OPEN: file "//TRIM(td_file%c_name))
            ENDIF

            ! get record length
            READ( td_file%i_id, IOSTAT=il_status, &
            &                         REC=1) td_file%i_recl
            CALL fct_err(il_status)
            IF( il_status /= 0 )THEN
               CALL logger_error("OPEN: read record length : "//&
               &    TRIM(fct_str(td_file%i_recl))//" in file "//&
               &    TRIM(td_file%c_name) )
            ENDIF

            CLOSE( td_file%i_id, IOSTAT=il_status )
            CALL fct_err(il_status)
            IF( il_status /= 0 )THEN
               CALL logger_error("OPEN: close file "//TRIM(td_file%c_name))
            ENDIF

            IF( .NOT. td_file%l_wrt )THEN

               CALL logger_info( " OPEN: dimg file "//&
               &              TRIM(td_file%c_name)//" in read only mode" )

               ! open file in read mode
               OPEN( td_file%i_id, FILE=TRIM(td_file%c_name),&
               &                         FORM='UNFORMATTED',              &
               &                         ACCESS='DIRECT',                 &
               &                         STATUS='OLD',                    &
               &                         ACTION='READ',                   &
               &                         RECL=td_file%i_recl,    &
               &                         IOSTAT=il_status)
               CALL fct_err(il_status)
               IF( il_status /= 0 )THEN
                  CALL logger_debug("IOM RSTDIMG OPEN: open staus "//&
                  &  TRIM(fct_str(il_status)))
                  CALL logger_fatal("IOM RSTDIMG OPEN: file "//&
                  &  TRIM(td_file%c_name)&
                  &  //" with record length "//TRIM(fct_str(td_file%i_recl)))
               ENDIF               

            ELSE

               CALL logger_info( " OPEN: dimg file "//&
               &              TRIM(td_file%c_name)//&
               &              " in read and write mode")

               ! open file in read mode
               OPEN( td_file%i_id, FILE=TRIM(td_file%c_name),&
               &                         FORM='UNFORMATTED',              &
               &                         ACCESS='DIRECT',                 &
               &                         STATUS='OLD',                    &
               &                         ACTION='READWRITE',              &
               &                         RECL=td_file%i_recl,    &
               &                         IOSTAT=il_status)
               CALL fct_err(il_status)
               IF( il_status /= 0 )THEN
                  CALL logger_debug("IOM RSTDIMG OPEN: open staus "//&
                  &  TRIM(fct_str(il_status)))
                  CALL logger_error("IOM RSTDIMG  OPEN: file "//&
                  & TRIM(td_file%c_name))
               ENDIF        

            ENDIF

            ! get general information about file
            CALL iom_rstdimg__get_info(td_file)

            ! get domain decomposition in file
            CALL iom_rstdimg_get_mpp(td_file)

            ! get information about variables in file
            CALL iom_rstdimg__get_file_var(td_file)

         ENDIF

      ENDIF

   END SUBROUTINE iom_rstdimg_open
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_rstdimg_close(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine close dimg file.
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
         &  " CLOSE: no id associated to file "//TRIM(td_file%c_name))

      ELSE      
         CALL logger_info( &
         &  " CLOSE: file "//TRIM(td_file%c_name))

         CLOSE( td_file%i_id, IOSTAT=il_status )
         CALL fct_err(il_status)
         IF( il_status /= 0 )THEN
            CALL logger_error("CLOSE "//TRIM(td_file%c_name))
         ENDIF

         td_file%i_id = 0 

      ENDIF

   END SUBROUTINE iom_rstdimg_close
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_rstdimg__get_info(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine get global information in an opened dimg 
   !> file.
   !> @details
   !> It gets the number of variables, the  domain decompistion, 
   !> the record of the header.<br/>
   !> It read dimensions, and add it to dimension structure inside 
   !> file structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January,2019
   !> - clean dimension structure
   !>
   !> @param[inout] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file

      ! local variable
      INTEGER(i4) :: il_status
      INTEGER(i4) :: il_recl                          ! record length 
      INTEGER(i4) :: il_nx, il_ny, il_nz              ! x,y,z dimension 
      INTEGER(i4) :: il_n0d, il_n1d, il_n2d, il_n3d   ! number of 0/1/2/3D variables 
      INTEGER(i4) :: il_rhd                           ! record of the header infos

      TYPE(TDIM)  :: tl_dim ! dimension structure 
      !----------------------------------------------------------------

      CALL logger_debug( &
      &  " IOM RSTDIMG GET INFO: about dimg file "//TRIM(td_file%c_name))

      ! read first record 
      READ( td_file%i_id, IOSTAT=il_status, REC=1 )& 
      &     il_recl,                         &
      &     il_nx, il_ny, il_nz,             &
      &     il_n0d, il_n1d, il_n2d, il_n3d,  &
      &     il_rhd
      CALL fct_err(il_status)
      IF( il_status /= 0 )THEN
         CALL logger_debug(" READ status: "//TRIM(fct_str(il_status)))
         CALL logger_fatal("IOM RSTDIMG GET INFO: read first line of "//&
         &  TRIM(td_file%c_name))
      ENDIF

      td_file%c_type='dimg'

      ! add dimension to file structure
      tl_dim=dim_init('X', il_nx)
      CALL file_move_dim(td_file, tl_dim)
      tl_dim=dim_init('Y', il_ny)
      CALL file_move_dim(td_file, tl_dim)
      tl_dim=dim_init('Z', il_nz)
      CALL file_move_dim(td_file, tl_dim)

      ! reorder dimension to ('x','y','z','t')
      ! actually fill unused dimension
      CALL dim_reorder(td_file%t_dim)

      ! save total number of variable 
      td_file%i_n0d=il_n0d
      td_file%i_n1d=il_n1d
      td_file%i_n2d=il_n2d
      td_file%i_n3d=il_n3d
      td_file%i_nvar=il_n0d+il_n1d+il_n2d+il_n3d

      ! record header infos
      td_file%i_rhd=il_rhd

      ! clean
      CALL dim_clean(tl_dim)

   END SUBROUTINE iom_rstdimg__get_info
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_rstdimg_get_mpp(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine get sub domain decomposition in a dimg file.
   !> @details
   !> domain decomposition informations are saved in attributes.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2016
   !> - mismatch with "halo" indices
   !>
   !> @param[inout] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file

      ! local variable
      TYPE(TATT)  :: tl_att
      INTEGER(i4) :: il_status
      INTEGER(i4) :: il_recl                          ! record length 
      INTEGER(i4) :: il_nx, il_ny, il_nz              ! x,y,z dimension 
      INTEGER(i4) :: il_n0d, il_n1d, il_n2d, il_n3d   ! number of 0/1/2/3D variables 
      INTEGER(i4) :: il_iglo, il_jglo                 ! domain global size
      INTEGER(i4) :: il_rhd                           ! record of the header infos
      INTEGER(i4) :: il_niproc, il_njproc, il_nproc   ! domain decomposition
      INTEGER(i4) :: il_area                          ! domain index

      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_impp
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_jmpp
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_lci
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_lcj
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_ldi
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_ldj
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_lei
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_lej
      !----------------------------------------------------------------

      CALL logger_debug( " IOM RSTDIMG GET MPP: dimg file "//&
      &  TRIM(td_file%c_name))

      ! read first record 
      READ( td_file%i_id, IOSTAT=il_status, REC=1 )& 
      &     il_recl,                         &
      &     il_nx, il_ny, il_nz,             &
      &     il_n0d, il_n1d, il_n2d, il_n3d,  &
      &     il_rhd,                          &
      &     il_niproc, il_njproc, il_nproc,  &
      &     il_area,                         &
      &     il_iglo, il_jglo
      CALL fct_err(il_status)
      IF( il_status /= 0 )THEN
         CALL logger_debug(" IOM RSTDIMG GET MPP: read status: "//&
         &  TRIM(fct_str(il_status)))
         CALL logger_error(" IOM RSTDIMG GET MPP: read first line of "//&
         &  TRIM(td_file%c_name))
      ENDIF

      ! create attributes to save mpp value
      tl_att=att_init( "DOMAIN_number_total", il_nproc)
      CALL file_move_att(td_file, tl_att)

      tl_att=att_init( "DOMAIN_I_number_total", il_niproc)
      CALL file_move_att(td_file, tl_att)

      tl_att=att_init( "DOMAIN_J_number_total", il_njproc)
      CALL file_move_att(td_file, tl_att)

      tl_att=att_init( "DOMAIN_number", il_area)
      CALL file_move_att(td_file, tl_att)

      tl_att=att_init( "DOMAIN_size_global", (/il_iglo, il_jglo/))
      CALL file_move_att(td_file, tl_att)

      ! allocate local variable
      ALLOCATE( il_impp(il_nproc), il_jmpp(il_nproc),&
      &         il_lci(il_nproc),  il_lcj(il_nproc), &
      &         il_ldi(il_nproc),  il_ldj(il_nproc), &
      &         il_lei(il_nproc),  il_lej(il_nproc), &
      &         stat=il_status)
      IF(il_status /= 0 )THEN

         CALL logger_error( " IOM RSTDIMG GET MPP: not enough space to put "//&
         &  "domain decomposition in file "//TRIM(td_file%c_name) )

      ENDIF

      ! read first record 
      READ( td_file%i_id, IOSTAT=il_status, REC=1 )& 
      &     il_recl,                         &
      &     il_nx, il_ny, il_nz,             &
      &     il_n0d, il_n1d, il_n2d, il_n3d,  &
      &     il_rhd,                          &
      &     il_niproc, il_njproc, il_nproc,  &
      &     il_area,                         &
      &     il_iglo, il_jglo,                &
      &     il_lci(:), il_lcj(:),            &
      &     il_ldi(:), il_ldj(:),            &
      &     il_lei(:), il_lej(:),            &
      &     il_impp(:),il_jmpp(:)
      CALL fct_err(il_status)
      IF( il_status /= 0 )THEN
         CALL logger_debug(" IOM RSTDIMG GET MPP: read status: "//&
         &  TRIM(fct_str(il_status)))
         CALL logger_fatal("IOM RSTDIMG GET MPP: read domain decomposition "//&
         &           "on first line of "//TRIM(td_file%c_name))
      ENDIF

      tl_att=att_init( "SUBDOMAIN_I_left_bottom_indices", il_impp(:) )
      CALL file_move_att(td_file, tl_att)
      tl_att=att_init( "SUBDOMAIN_J_left_bottom_indices", il_jmpp(:) )
      CALL file_move_att(td_file, tl_att)

      tl_att=att_init( "SUBDOMAIN_I_dimensions", il_lci(:))
      CALL file_move_att(td_file, tl_att)
      tl_att=att_init( "SUBDOMAIN_J_dimensions", il_lcj(:))
      CALL file_move_att(td_file, tl_att)

      tl_att=att_init( "SUBDOMAIN_I_first_indoor_indices", il_ldi(:))
      CALL file_move_att(td_file, tl_att)
      tl_att=att_init( "SUBDOMAIN_J_first_indoor_indices", il_ldj(:))
      CALL file_move_att(td_file, tl_att)

      tl_att=att_init( "SUBDOMAIN_I_last_indoor_indices", il_lei(:))
      CALL file_move_att(td_file, tl_att)
      tl_att=att_init( "SUBDOMAIN_J_last_indoor_indices", il_lej(:))
      CALL file_move_att(td_file, tl_att)

      ! clean
      CALL att_clean(tl_att)

      DEALLOCATE( il_impp, il_jmpp,&
      &           il_lci,  il_lcj, &
      &           il_ldi,  il_ldj, &
      &           il_lei,  il_lej  )

   END SUBROUTINE iom_rstdimg_get_mpp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_rstdimg__get_file_var(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine read information about variable on an 
   !> opened dimg file.
   !> @details
   !> The variables structures inside file structure are then completed.
   !> Variables no0d, no1d, no2d, no3d are deleted from file strucutre.
   !> @note variable value are read only for scalar variable (0d).
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
      CHARACTER(LEN=im_vnl), DIMENSION(:), ALLOCATABLE :: cl_name

      REAL(dp)             , DIMENSION(:), ALLOCATABLE :: dl_value

      INTEGER(i4)                                      :: il_status
      INTEGER(i4)          , DIMENSION(:), ALLOCATABLE :: il_start
      INTEGER(i4)          , DIMENSION(:), ALLOCATABLE :: il_count
      !----------------------------------------------------------------

      IF( td_file%i_nvar > 0 )THEN

         ALLOCATE( il_start(4), il_count(4) )

         il_start(1) = 1
         il_count(1) = td_file%i_n0d
         
         il_start(2) = 1 + il_count(1)
         il_count(2) = il_start(2) - 1 + td_file%i_n1d
         
         il_start(3) = 1 + il_count(2)
         il_count(3) = il_start(3) - 1 + td_file%i_n2d
         
         il_start(4) = 1 + il_count(3)
         il_count(4) = il_start(4) - 1 + td_file%i_n3d
         
         ALLOCATE( cl_name(td_file%i_nvar), dl_value(td_file%i_nvar) )

         ! read first record 
         READ( td_file%i_id, IOSTAT=il_status, REC=td_file%i_rhd )& 
         & cl_name(il_start(1):il_count(1)), dl_value(il_start(1):il_count(1)),&
         & cl_name(il_start(2):il_count(2)), dl_value(il_start(2):il_count(2)),&
         & cl_name(il_start(3):il_count(3)), dl_value(il_start(3):il_count(3)),&
         & cl_name(il_start(4):il_count(4)), dl_value(il_start(4):il_count(4))
         CALL fct_err(il_status)
         IF( il_status /= 0 )THEN
            CALL logger_error("GET FILE: reading headers in file "//&
            &   TRIM(td_file%c_name))
         ENDIF

         DEALLOCATE( il_start, il_count )

         IF(ASSOCIATED(td_file%t_var))THEN
            CALL var_clean(td_file%t_var(:))
            DEALLOCATE(td_file%t_var)
         ENDIF
         ALLOCATE(td_file%t_var(td_file%i_nvar))

         ! put information about variable 0D inside file structure
         CALL iom_rstdimg__get_file_var_0d(td_file, cl_name(:), dl_value(:))

         ! put information about variable 1D inside file structure
         CALL iom_rstdimg__get_file_var_1d(td_file, cl_name(:), dl_value(:))

         ! put information about variable 2D inside file structure
         CALL iom_rstdimg__get_file_var_2d(td_file, cl_name(:), dl_value(:))

         ! put information about variable 3D inside file structure
         CALL iom_rstdimg__get_file_var_3d(td_file, cl_name(:), dl_value(:))

         DEALLOCATE( cl_name, dl_value )

         ! delete dummy variable
         CALL file_del_var( td_file, 'no0d' )
         CALL file_del_var( td_file, 'no1d' )
         CALL file_del_var( td_file, 'no2d' )
         CALL file_del_var( td_file, 'no3d' )

      ELSE

         CALL logger_debug( &
         &  " GET FILE VAR: there is no variable in file "//&
         &  TRIM(td_file%c_name))

      ENDIF

   END SUBROUTINE iom_rstdimg__get_file_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_rstdimg__get_file_var_0d(td_file, cd_name, dd_value)
   !-------------------------------------------------------------------
   !> @brief This subroutine put informations about scalar variable 
   !> inside file structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] cd_name      array of variable name
   !> @param[in] dd_value     array of variable value
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE),                         INTENT(INOUT) :: td_file
      CHARACTER(LEN=im_vnl), DIMENSION(:), INTENT(IN)    :: cd_name
      REAL(dp),              DIMENSION(:), INTENT(IN)    :: dd_value

      ! local variable
      TYPE(TDIM), DIMENSION(ip_maxdim) :: tl_dim

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! define same dimension as in file
      tl_dim(:)=dim_copy(td_file%t_dim(:))
      ! do not use any dimension
      tl_dim(:)%l_use=.FALSE.
      tl_dim(:)%i_len=1
      
      ! case scalar variable
      DO ji = 1, td_file%i_n0d
   
         td_file%t_var(ji)=var_init( TRIM(cd_name(ji)), NF90_DOUBLE, &
         &                           tl_dim(:), dd_fill=0._dp,       &
         &                           id_id=ji, id_rec=1 )

         ! get value of scalar
         IF( ASSOCIATED(td_file%t_var(ji)%d_value) )THEN
            DEALLOCATE(td_file%t_var(ji)%d_value)
         ENDIF
         ALLOCATE(td_file%t_var(ji)%d_value(1,1,1,1))

         td_file%t_var(ji)%d_value(1,1,1,1)=dd_value(ji)

      ENDDO

      ! clean
      CALL dim_clean(tl_dim(:))

   END SUBROUTINE iom_rstdimg__get_file_var_0d
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_rstdimg__get_file_var_1d(td_file, cd_name, dd_value)
   !-------------------------------------------------------------------
   !> @brief This subroutine put informations about variable 1D 
   !> inside file structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2016
   !>    - change right dimension struct 
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] cd_name      array of variable name
   !> @param[in] dd_value     array of variable record
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE),                         INTENT(INOUT) :: td_file
      CHARACTER(LEN=im_vnl), DIMENSION(:), INTENT(IN)    :: cd_name
      REAL(dp),              DIMENSION(:), INTENT(IN)    :: dd_value

      ! local variable
      TYPE(TDIM), DIMENSION(ip_maxdim) :: tl_dim

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! case variable 1D
      DO ji = td_file%i_n0d + 1, &
      &       td_file%i_n0d + td_file%i_n1d

         ! define same dimension as in file
         tl_dim(:)=dim_copy(td_file%t_dim(:))
         ! do not use X and Y dimension
         tl_dim(1:2)%l_use=.FALSE.
         tl_dim(1:2)%i_len=1
      
         td_file%t_var(ji)=var_init( TRIM(cd_name(ji)), NF90_DOUBLE, &
         &                           tl_dim(:), dd_fill=0._dp,       &
         &                           id_id=ji, id_rec=INT(dd_value(ji),i4) )

         ! clean
         CALL dim_clean(tl_dim(:))

      ENDDO

   END SUBROUTINE iom_rstdimg__get_file_var_1d
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_rstdimg__get_file_var_2d(td_file, cd_name, dd_value)
   !-------------------------------------------------------------------
   !> @brief This subroutine put informations about variable 2D 
   !> inside file structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] cd_name      array of variable name
   !> @param[in] dd_value     array of variable record
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE),                         INTENT(INOUT) :: td_file
      CHARACTER(LEN=im_vnl), DIMENSION(:), INTENT(IN)    :: cd_name
      REAL(dp),              DIMENSION(:), INTENT(IN)    :: dd_value

      ! local variable
      TYPE(TDIM), DIMENSION(ip_maxdim) :: tl_dim

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! case variable 2D (X,Y)
      DO ji = td_file%i_n0d + td_file%i_n1d + 1 , &
      &       td_file%i_n0d + td_file%i_n1d + td_file%i_n2d

         ! define same dimension as in file
         tl_dim(:)=dim_copy(td_file%t_dim(:))
         ! do not use Z dimension
         tl_dim(3)%l_use=.FALSE.
         tl_dim(3)%i_len=1       

         td_file%t_var(ji)=var_init( TRIM(cd_name(ji)), NF90_DOUBLE, &
         &                           tl_dim(:), dd_fill=0._dp,       &
         &                           id_id=ji, id_rec=INT(dd_value(ji),i4) )

         ! clean
         CALL dim_clean(tl_dim(:))

      ENDDO

   END SUBROUTINE iom_rstdimg__get_file_var_2d
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_rstdimg__get_file_var_3d(td_file, cd_name, dd_value)
   !-------------------------------------------------------------------
   !> @brief This subroutine put informations about variable 3D 
   !> inside file structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_file   file structure
   !> @param[in] cd_name      array of variable name
   !> @param[in] dd_value     array of variable record
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE),                         INTENT(INOUT) :: td_file
      CHARACTER(LEN=im_vnl), DIMENSION(:), INTENT(IN)    :: cd_name
      REAL(dp),              DIMENSION(:), INTENT(IN)    :: dd_value

      ! local variable
      TYPE(TDIM), DIMENSION(ip_maxdim) :: tl_dim

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! case variable 3D (X,Y,Z)
      DO ji = td_file%i_n0d + td_file%i_n1d + td_file%i_n2d +1 , &
      &       td_file%i_n0d + td_file%i_n1d + td_file%i_n2d + td_file%i_n3d

         ! define same dimension as in file
         tl_dim(:)=dim_copy(td_file%t_dim(:)) 

         td_file%t_var(ji)=var_init( TRIM(cd_name(ji)), NF90_DOUBLE, &
         &                           tl_dim(:), dd_fill=0._dp,       &
         &                           id_id=ji, id_rec=INT(dd_value(ji),i4) )

         ! clean
         CALL dim_clean(tl_dim(:))

      ENDDO

   END SUBROUTINE iom_rstdimg__get_file_var_3d
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_rstdimg__read_dim_id(td_file, id_dimid) &
         & RESULT (tf_dim)
   !-------------------------------------------------------------------
   !> @brief This function read one dimension in an opened netcdf file, 
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

      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  " READ DIM: no id associated to dimg file "//TRIM(td_file%c_name))

      ELSE      

         tf_dim%i_id=id_dimid

         CALL logger_debug( &
         &  " READ DIM: dimension "//TRIM(fct_str(id_dimid))//&
         &  " in file "//TRIM(td_file%c_name))

         IF( id_dimid <= 4 )THEN
            tf_dim=td_file%t_dim(id_dimid)
         ELSE
            CALL logger_error( &
            &  " READ DIM: no dimension with id "//TRIM(fct_str(id_dimid))//&
            &  " in file "//TRIM(td_file%c_name))
         ENDIF

      ENDIF

   END FUNCTION iom_rstdimg__read_dim_id
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_rstdimg__read_dim_name(td_file, cd_name) &
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
      INTEGER(i4) :: il_dimid
      !----------------------------------------------------------------

      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  " READ DIM: no id associated to dimg file "//TRIM(td_file%c_name))

      ELSE      

         il_dimid=dim_get_id(td_file%t_dim(:), TRIM(cd_name))
         IF( il_dimid /= 0 )THEN
            tf_dim=iom_rstdimg_read_dim(td_file, il_dimid)
         ELSE
            CALL logger_error( &
            &  " READ DIM: no dimension "//TRIM(cd_name)//&
            &  " in file "//TRIM(td_file%c_name))
         ENDIF

      ENDIF

   END FUNCTION iom_rstdimg__read_dim_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_rstdimg__read_var_id(td_file, id_varid, id_start, id_count) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function read variable value in an opened 
   !> dimg file, given variable id.
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
      INTEGER(i4), DIMENSION(1) :: il_varid
      !----------------------------------------------------------------
      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  " READ VAR: no id associated to file "//TRIM(td_file%c_name))

      ELSE

         ! look for variable id
         il_varid(:)=MINLOC(td_file%t_var(:)%i_id,mask=(td_file%t_var(:)%i_id==id_varid))
         IF( il_varid(1) /= 0 )THEN

            tf_var=var_copy(td_file%t_var(il_varid(1)))

            IF( tf_var%i_ndim /= 0 )THEN
               !!! read variable value
               CALL iom_rstdimg__read_var_value( td_file, tf_var, id_start, id_count)
            ELSE
               CALL logger_debug( " READ VAR: variable 0d "//&
               &               TRIM(td_file%t_var(il_varid(1))%c_name)//&
               &               " should be already read ")
            ENDIF            

         ELSE
            CALL logger_error( &
            &  " READ VAR: there is no variable with id "//&
            &  TRIM(fct_str(id_varid))//" in file "//TRIM(td_file%c_name))
         ENDIF

      ENDIF
   END FUNCTION iom_rstdimg__read_var_id
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_rstdimg__read_var_name(td_file, cd_name, id_start, id_count) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function read variable value in an opened 
   !> dimg file, given variable name or standard name.
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
   !> @param[in] cd_name   variable name or standard name 
   !> @param[in] id_start  index in the variable from which the data values 
   !> will be read
   !> @param[in] id_count  number of indices selected along each dimension
   !> @return  variable structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument 
      TYPE(TFILE),                     INTENT(IN) :: td_file
      CHARACTER(LEN=*),                INTENT(IN) :: cd_name
      INTEGER(i4),      DIMENSION(:),  INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4),      DIMENSION(:),  INTENT(IN), OPTIONAL :: id_count

      ! function
      TYPE(TVAR)                                  :: tf_var

      ! local variable
      INTEGER(i4)       :: il_varid
      !----------------------------------------------------------------
      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  " READ VAR: no id associated to file "//TRIM(td_file%c_name))

      ELSE

         il_varid=var_get_index(td_file%t_var(:), cd_name)
         IF( il_varid /= 0 )THEN

            tf_var=var_copy(td_file%t_var(il_varid))

            IF( td_file%t_var(il_varid)%i_ndim /= 0 )THEN
               !!! read variable value
               CALL iom_rstdimg__read_var_value( td_file, tf_var, id_start, id_count)
            ELSE
               CALL logger_debug( " READ VAR: variable 0d "//&
               &               TRIM(td_file%t_var(il_varid)%c_name)//&
               &               " should have been already read ")
            ENDIF

         ELSE

            CALL logger_error( &
            &  " READ VAR NAME: there is no variable with "//&
            &  " name or standard name "//TRIM(cd_name)//&
            &  " in file "//TRIM(td_file%c_name) )

         ENDIF

      ENDIF
      
   END FUNCTION iom_rstdimg__read_var_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_rstdimg__read_var_value(td_file, td_var, id_start, id_count)
   !-------------------------------------------------------------------
   !> @brief This subroutine read variable value in an opened dimg file, for
   !> variable 1,2,3d.
   !> @details
   !> Optionaly,start indices and number of indices selected along each dimension 
   !> could be specify in a 4 dimension array (/'x','y','z','t'/)
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date February, 2016
   !> - use temporary array to read value from file
   !>
   !> @param[in] td_file   file structure
   !> @param[inout] td_var variable structure
   !> @param[in] id_start  index in the variable from which the data values will be read
   !> @param[in] id_count  number of indices selected along each dimension
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE),               INTENT(IN)    :: td_file
      TYPE(TVAR) ,               INTENT(INOUT) :: td_var
      INTEGER(i4), DIMENSION(:), INTENT(IN),  OPTIONAL :: id_start
      INTEGER(i4), DIMENSION(:), INTENT(IN),  OPTIONAL :: id_count

      ! local variable
      INTEGER(i4)                                  :: il_status
      INTEGER(i4)                                  :: il_tmp1, il_tmp2
      INTEGER(i4), DIMENSION(ip_maxdim)            :: il_start
      INTEGER(i4), DIMENSION(ip_maxdim)            :: il_count

      REAL(dp),    DIMENSION(:,:,:)  , ALLOCATABLE :: dl_tmp
      REAL(dp),    DIMENSION(:,:,:,:), ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! check id_count and id_start optionals parameters...
      IF( (       PRESENT(id_start)  .AND. (.NOT. PRESENT(id_count))) .OR. &
          ((.NOT. PRESENT(id_start)) .AND.        PRESENT(id_count) ) )THEN
         CALL logger_warn( &
         &  " READ VAR VALUE: id_start and id_count should be both specify")
      ENDIF

      IF( PRESENT(id_start).AND.PRESENT(id_count) )THEN

         IF( SIZE(id_start(:)) /= ip_maxdim .OR. &
         &   SIZE(id_count(:)) /= ip_maxdim )THEN
            CALL logger_error("READ VAR: dimension of array start or count "//&
            &      " are invalid to read variable "//TRIM(td_var%c_name)//&
            &      " in file "//TRIM(td_file%c_name) )
         ENDIF

         ! dimension order assume to be ('x','y','z','t')
         il_start(:)=id_start(:)
         il_count(:)=id_count(:)

      ELSE

         ! dimension order assume to be ('x','y','z','t')
         il_start(:)=(/1,1,1,1/)
         il_count(:)=td_var%t_dim(:)%i_len

      ENDIF

      ! check dimension
      IF( .NOT. ALL(il_start(:)>=(/1,1,1,1/)) )THEN

         CALL logger_error( " READ VAR VALUE: "//&
         &               " start indices should be greater than or equal to 1")

      ENDIF

      IF(.NOT.ALL(il_start(:)+il_count(:)-1<=(/td_var%t_dim(1)%i_len,&
         &                                     td_var%t_dim(2)%i_len,&
         &                                     td_var%t_dim(3)%i_len,&
         &                                     td_var%t_dim(4)%i_len &
         &                                    /)) )THEN

         CALL logger_error( " READ VAR VALUE: "//&
         &               "start + count exceed variable dimension" )

         DO ji = 1, ip_maxdim
            il_tmp1=il_start(ji)+il_count(ji)-1
            il_tmp2=td_var%t_dim(ji)%i_len
            CALL logger_debug( &
            &  " READ VAR VALUE: start + count - 1 "//TRIM(fct_str(il_tmp1))//&
            &  " variable dimension"//TRIM(fct_str(il_tmp2)))
         ENDDO

      ELSE

         ! Allocate space to hold variable value
         ALLOCATE(dl_value( td_var%t_dim(1)%i_len, &
         &                  td_var%t_dim(2)%i_len, &
         &                  td_var%t_dim(3)%i_len, &
         &                  td_var%t_dim(4)%i_len),&
         &        stat=il_status)
         IF(il_status /= 0 )THEN

           CALL logger_error( &
            &  " READ VAR VALUE: not enough space to put variable "//&
            &  TRIM(td_var%c_name)//&
            &  " in temporary array")

         ENDIF

         ! read values
         CALL logger_trace( &
         &  " READ VAR VALUE: get variable "//TRIM(td_var%c_name)//&
         &  " in file "//TRIM(td_file%c_name))     

         IF( ALL(td_var%t_dim(1:3)%l_use) )THEN
            ! 3D variable (X,Y,Z)
            ALLOCATE(dl_tmp( td_var%t_dim(1)%i_len, &
            &                td_var%t_dim(2)%i_len, &
            &                td_var%t_dim(4)%i_len) )           
            DO ji=1,td_var%t_dim(3)%i_len
               READ(td_file%i_id, IOSTAT=il_status, REC=td_var%i_rec +ji-1) &
               &  dl_tmp(:,:,:)
               CALL fct_err(il_status)
               IF( il_status /= 0 )THEN
                  CALL logger_error("READ VAR VALUE: reading 3D variable "//&
                  &              TRIM(td_var%c_name))
               ENDIF
               dl_value(:,:,ji,:)=dl_tmp(:,:,:)
            ENDDO
            DEALLOCATE(dl_tmp)
         ELSEIF( ALL(td_var%t_dim(1:2)%l_use) )THEN
            ! 2D variable (X,Y)
            READ(td_file%i_id, IOSTAT=il_status, REC=td_var%i_rec ) &
            &  dl_value(:,:,:,:)
            CALL fct_err(il_status)
            IF( il_status /= 0 )THEN
               CALL logger_error("READ VAR VALUE: reading 2D variable "//&
                  &            TRIM(td_var%c_name))
            ENDIF
         ELSEIF( td_var%t_dim(3)%l_use )THEN
            ! 1D variable (Z)
            READ(td_file%i_id, IOSTAT=il_status, REC=td_var%i_rec ) &
            &  dl_value(:,:,:,:)
            CALL fct_err(il_status)
            IF( il_status /= 0 )THEN
               CALL logger_error("READ VAR VALUE: reading 1D variable "//&
                  &            TRIM(td_var%c_name))
            ENDIF
         ENDIF
 
         ! Allocate space to hold variable value in structure
         IF( ASSOCIATED(td_var%d_value) )THEN
            DEALLOCATE(td_var%d_value)   
         ENDIF

         ALLOCATE(td_var%d_value( il_count(1), &
         &                        il_count(2), &
         &                        il_count(3), &
         &                        il_count(4)),&
         &        stat=il_status)
         IF(il_status /= 0 )THEN

           CALL logger_error( &
            &  " READ VAR VALUE: not enough space to put variable "//&
            &  TRIM(td_var%c_name)//&
            &  " in variable structure")

         ENDIF
         ! FillValue by default
         td_var%d_value(:,:,:,:)=td_var%d_fill

         ! new dimension length
         td_var%t_dim(:)%i_len=il_count(:)

         ! extract value 
         td_var%d_value(:,:,:,:) = dl_value(il_start(1):il_start(1)+il_count(1)-1,&
         &                                  il_start(2):il_start(2)+il_count(2)-1,&
         &                                  il_start(3):il_start(3)+il_count(3)-1,&
         &                                  il_start(4):il_start(4)+il_count(4)-1)

         DEALLOCATE(dl_value)

      ENDIF

      ! force to change _FillValue to avoid mistake 
      ! with dummy zero _FillValue
      IF( td_var%d_fill == 0._dp )THEN
         CALL var_chg_FillValue(td_var)
      ENDIF

      ! use scale factor and offset
      WHERE( td_var%d_value(:,:,:,:) /= td_var%d_fill )
         td_var%d_value(:,:,:,:) = &
         &  td_var%d_value(:,:,:,:)*td_var%d_scf + td_var%d_ofs
      END WHERE

   END SUBROUTINE iom_rstdimg__read_var_value
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_rstdimg_write_header(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine write header of dimg file from file structure.
   !>
   !> @details
   !> dimg file have to be already opened in write mode.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014
   !> - use iom_rstdimg__get_rec
   !> @date August, 2017
   !> - split in write_header and write_var 
   !>
   !> @param[inout] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file

      ! local variable
      INTEGER(i4)           :: il_status
      INTEGER(i4)           :: il_ind
      !----------------------------------------------------------------
      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  " WRITE FILE: no id associated to file "//TRIM(td_file%c_name))

      ELSE
         IF( td_file%l_wrt )THEN

            ! check dimension
            IF( td_file%t_dim(jp_L)%l_use .AND. &
            &   td_file%t_dim(jp_L)%i_len /= 1 )THEN
               CALL logger_fatal("WRITE FILE: can not write dimg file with "//&
               &  " several time step.")
            ENDIF

            ! close and open file with right record length
            CALL iom_rstdimg_close(td_file)

            ! compute record number to be used 
            ! and add variable no0d, no1d,.. if need be
            CALL iom_rstdimg__get_rec(td_file)

            ! compute record length
            il_ind=att_get_index(td_file%t_att(:),"DOMAIN_number_total")
            IF( il_ind /= 0 )THEN
               td_file%i_recl = MAX( &
               &     td_file%t_dim(1)%i_len * td_file%t_dim(2)%i_len * 8, &
               &     ( 8 * INT(td_file%t_att(il_ind)%d_value(1)) + 15 ) * 4 )
            ELSE
               td_file%i_recl = td_file%t_dim(1)%i_len * &
               &                td_file%t_dim(2)%i_len * 8
            ENDIF
            ! check record length
            IF( td_file%i_nvar*(im_vnl+dp) > td_file%i_recl )THEN
               CALL logger_fatal("WRITE FILE: record length is too small. "//&
               &  " Try to reduce the output number of processor.")
            ENDIF

            ! get free unit
            td_file%i_id=fct_getunit()

            OPEN( td_file%i_id, FILE=TRIM(td_file%c_name),&
            &                         FORM='UNFORMATTED',              &
            &                         ACCESS='DIRECT',                 &
            &                         STATUS='REPLACE',                &
            &                         ACTION='WRITE',                  &
            &                         RECL=td_file%i_recl,             &
            &                         IOSTAT=il_status)
            CALL fct_err(il_status)
            IF( il_status /= 0 )THEN
               CALL logger_error("WRITE FILE: REPLACE file "//TRIM(td_file%c_name)//&
               &  " with record length "//TRIM(fct_str(td_file%i_recl)))
            ELSE
               CALL logger_debug("WRITE FILE: REPLACE file "//TRIM(td_file%c_name)//&
               &  " with record length "//TRIM(fct_str(td_file%i_recl)))
            ENDIF

            ! write header
            CALL iom_rstdimg__write_header(td_file)

         ELSE

            CALL logger_error( &
            &  " WRITE FILE: try to write in file "//TRIM(td_file%c_name)//&
            &  ", not opened in write mode")

         ENDIF
      ENDIF

   END SUBROUTINE iom_rstdimg_write_header
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_rstdimg_write_var(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine write variable in dimg file from file structure.
   !>
   !> @details
   !> dimg file have to be already opened in write mode.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014
   !> - use iom_rstdimg__get_rec
   !> @date August, 2017
   !> - split in write_header and write_var 
   !>
   !> @param[inout] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file

      ! local variable
      !----------------------------------------------------------------
      ! check if file opened
      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  " WRITE FILE: no id associated to file "//TRIM(td_file%c_name))

      ELSE
         IF( td_file%l_wrt )THEN

            ! write variable in file
            CALL iom_rstdimg__write_var(td_file) 

         ELSE

            CALL logger_error( &
            &  " WRITE FILE: try to write in file "//TRIM(td_file%c_name)//&
            &  ", not opened in write mode")

         ENDIF
      ENDIF

   END SUBROUTINE iom_rstdimg_write_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_rstdimg__get_rec(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine compute record number to be used.
   !>
   !> @details
   !> Moreover it adds variable no0d, no1d, no2d and no3d if need be.
   !>
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[inout] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file

      ! local variable
      INTEGER(i4) :: il_rec
      TYPE(TVAR)  :: tl_var

      INTEGER(i4), DIMENSION(:)    , ALLOCATABLE :: il_tmp1d
      INTEGER(i4), DIMENSION(:,:)  , ALLOCATABLE :: il_tmp2d
      INTEGER(i4), DIMENSION(:,:,:), ALLOCATABLE :: il_tmp3d

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! add dummy variable if necessary
      IF( td_file%i_n0d == 0 )THEN
         ! create var
         tl_var=var_init('no0d')

         CALL file_add_var( td_file, tl_var )
      ENDIF

      IF( td_file%i_n1d == 0 )THEN
         ! create var
         ALLOCATE( il_tmp1d( td_file%t_dim(3)%i_len ) )
         il_tmp1d(:)=-1

         tl_var=var_init( 'no1d', il_tmp1d(:)) 

         DEALLOCATE( il_tmp1d )

         CALL file_add_var( td_file, tl_var ) 
      ENDIF
 
      IF( td_file%i_n2d == 0 )THEN
         ! create var
         ALLOCATE( il_tmp2d( td_file%t_dim(1)%i_len, &
         &                   td_file%t_dim(2)%i_len ) )
         il_tmp2d(:,:)=-1

         tl_var=var_init('no2d', il_tmp2d(:,:) )

         DEALLOCATE( il_tmp2d )

         CALL file_add_var( td_file, tl_var ) 

      ENDIF
 
      IF( td_file%i_n3d == 0 )THEN
         ! create var
         ALLOCATE( il_tmp3d( td_file%t_dim(1)%i_len, &
         &                   td_file%t_dim(2)%i_len, &
         &                   td_file%t_dim(3)%i_len ) )
         il_tmp3d(:,:,:)=-1

         tl_var=var_init('no3d', il_tmp3d(:,:,:) )

         DEALLOCATE( il_tmp3d )

         CALL file_add_var( td_file, tl_var ) 
      ENDIF

      ! clean
      CALL var_clean(tl_var)

      il_rec=2
      DO ji=1,td_file%i_nvar
         SELECT CASE(td_file%t_var(ji)%i_ndim)
            CASE(0)
               IF( INDEX(td_file%t_var(ji)%c_name, 'no0d' ) == 0 )THEN
                  td_file%t_var(ji)%i_rec=il_rec
                  il_rec = il_rec  + 0
               ENDIF
            CASE(1)
               IF( INDEX(td_file%t_var(ji)%c_name, 'no1d' ) == 0 )THEN
                  td_file%t_var(ji)%i_rec=il_rec
                  il_rec = il_rec  + 1
               ENDIF
            CASE(2)
               IF( INDEX(td_file%t_var(ji)%c_name, 'no2d' ) == 0 )THEN
                  td_file%t_var(ji)%i_rec=il_rec
                  il_rec = il_rec  + 1
               ENDIF
            CASE(3)
               IF( INDEX(td_file%t_var(ji)%c_name, 'no3d' ) == 0 )THEN
                  td_file%t_var(ji)%i_rec=il_rec
                  il_rec = il_rec  + td_file%t_dim(3)%i_len
               ENDIF
         END SELECT
      ENDDO
      td_file%i_rhd  = il_rec

   END SUBROUTINE iom_rstdimg__get_rec
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_rstdimg__write_header(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine write header in an opened dimg 
   !> file in write mode.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2016
   !> - mismatch with "halo" indices
   !>
   !> @param[inout] td_file   file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file

      ! local variable
      INTEGER(i4) :: il_status
      INTEGER(i4) :: il_ind
      INTEGER(i4) :: il_nproc
      INTEGER(i4) :: il_niproc
      INTEGER(i4) :: il_njproc
      INTEGER(i4) :: il_area
      INTEGER(i4) :: il_iglo
      INTEGER(i4) :: il_jglo

      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_impp
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_jmpp
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_lci
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_lcj
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_ldi
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_ldj
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_lei
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_lej
      
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! check record length
      IF( td_file%i_recl <= 8 )THEN
         CALL logger_warn(" WRITE FILE: record length seems to be tiny!! &
         & ("//TRIM(fct_str(td_file%i_recl))//")")
      ENDIF

      ! check dimension
      IF( ANY(td_file%t_dim(1:3)%i_len <= 0 ) )THEN
         CALL logger_error(" WRITE FILE: at least one dimension size is less &
         &                than or equal to zero !! ")
         DO ji=1,3
            CALL logger_debug(" WRITE FILE: dimension "//&
            &               TRIM(td_file%t_dim(ji)%c_name)//" : "//&
            &               TRIM(fct_str(td_file%t_dim(ji)%i_len)) )
         ENDDO
      ENDIF

      ! get domain decomposition
      il_ind=att_get_index( td_file%t_att, "DOMAIN_number_total" )
      il_nproc = 1
      IF( il_ind /= 0 )THEN
         il_nproc = INT(td_file%t_att(il_ind)%d_value(1))
      ENDIF

      il_ind=att_get_index( td_file%t_att, "DOMAIN_I_number_total" )
      il_niproc = 0
      IF( il_ind /= 0 )THEN
         il_niproc = INT(td_file%t_att(il_ind)%d_value(1))
      ENDIF

      il_ind=att_get_index( td_file%t_att, "DOMAIN_J_number_total" )
      il_njproc = 0
      IF( il_ind /= 0 )THEN
         il_njproc = INT(td_file%t_att(il_ind)%d_value(1))
      ENDIF

      ! check domain decomposition
      IF( il_niproc  <= 0 .OR. &
      &   il_njproc  <= 0 .OR. &
      &   il_nproc <= 0 .OR. &
      &   il_nproc > il_niproc * il_njproc )THEN
      
         CALL logger_error(" WRITE FILE: invalid domain splitting ")

         CALL logger_debug(" WRITE FILE: niproc "//TRIM(fct_str(il_niproc)) )
         CALL logger_debug(" WRITE FILE: njproc "//TRIM(fct_str(il_njproc)) )
         CALL logger_debug(" WRITE FILE: nproc "//TRIM(fct_str(il_nproc)) )

      ENDIF
      
      ! get domain number
      il_ind=att_get_index( td_file%t_att, "DOMAIN_number" )
      il_area = 0
      IF( il_ind /= 0 )THEN
         il_area = INT(td_file%t_att(il_ind)%d_value(1))
      ENDIF

      ! get domain global size
      il_ind=att_get_index( td_file%t_att, "DOMAIN_size_global" )
      il_iglo = 0
      il_jglo = 0
      IF( il_ind /= 0 )THEN
         il_iglo = INT(td_file%t_att(il_ind)%d_value(1))
         il_jglo = INT(td_file%t_att(il_ind)%d_value(2))
      ENDIF

      ! check domain global size
      IF( il_iglo < td_file%t_dim(1)%i_len .OR. &
      &   il_jglo < td_file%t_dim(2)%i_len )THEN
         CALL logger_error(" WRITE FILE: invalid global domain ")

         CALL logger_debug(" WRITE FILE: global domain : "//&
         &              TRIM(fct_str(il_iglo))//" x "//&
         &              TRIM(fct_str(il_jglo)) )
         CALL logger_debug(" WRITE FILE: local domain : "//&
         &              TRIM(fct_str(td_file%t_dim(1)%i_len))//" x "//&
         &              TRIM(fct_str(td_file%t_dim(2)%i_len)) )
      ENDIF

      ! allocate local variable
      ALLOCATE( il_impp(il_nproc), il_jmpp(il_nproc),&
      &         il_lci(il_nproc),  il_lcj(il_nproc), &
      &         il_ldi(il_nproc),  il_ldj(il_nproc), &
      &         il_lei(il_nproc),  il_lej(il_nproc) )

      ! get left bottom indices
      il_ind=att_get_index( td_file%t_att, "SUBDOMAIN_I_left_bottom_indices" )
      il_impp(:) = 0
      IF( il_ind /= 0 )THEN
         il_impp(:) = INT(td_file%t_att(il_ind)%d_value(:))
      ENDIF

      il_ind=att_get_index( td_file%t_att, "SUBDOMAIN_J_left_bottom_indices" )
      il_jmpp(:) = 0
      IF( il_ind /= 0 )THEN
         il_jmpp(:) = INT(td_file%t_att(il_ind)%d_value(:))
      ENDIF
      
      ! check left bottom indices
      IF( ANY(il_impp(:)==0) .OR. ANY(il_jmpp(:)==0) )THEN
         CALL logger_warn("WRITE FILE: no data for subdomain left bottom indices")
      ENDIF

      ! get subdomain dimensions
      il_ind=att_get_index( td_file%t_att, "SUBDOMAIN_I_dimensions" )
      il_lci(:) = 0
      IF( il_ind /= 0 )THEN
         il_lci(:) = INT(td_file%t_att(il_ind)%d_value(:))
      ENDIF

      il_ind=att_get_index( td_file%t_att, "SUBDOMAIN_J_dimensions" )
      il_lcj(:) = 0
      IF( il_ind /= 0 )THEN
         il_lcj(:) = INT(td_file%t_att(il_ind)%d_value(:))
      ENDIF

      ! check subdomain dimension
      IF( ANY(il_lci(:)==0) .OR. ANY(il_lcj(:)==0) )THEN
         CALL logger_warn("WRITE FILE: no data for subdomain dimensions")
      ENDIF

      ! get first indoor indices
      il_ind=att_get_index( td_file%t_att, "SUBDOMAIN_I_first_indoor_indices" )
      il_ldi(:) = 0
      IF( il_ind /= 0 )THEN
         il_ldi(:) = INT(td_file%t_att(il_ind)%d_value(:))
      ENDIF

      il_ind=att_get_index( td_file%t_att, "SUBDOMAIN_J_first_indoor_indices" )
      il_ldj(:) = 0
      IF( il_ind /= 0 )THEN
         il_ldj(:) = INT(td_file%t_att(il_ind)%d_value(:))
      ENDIF
      
      ! check first indoor indices
      IF( ANY(il_ldi(:)==0) .OR. ANY(il_ldj(:)==0) )THEN
         CALL logger_warn("WRITE FILE: no data for subdomain first indoor indices")
      ENDIF

      ! get last indoor indices
      il_ind=att_get_index( td_file%t_att, "SUBDOMAIN_I_last_indoor_indices" )
      il_lei(:) = 0
      IF( il_ind /= 0 )THEN
         il_lei(:) = INT(td_file%t_att(il_ind)%d_value(:))
      ENDIF

      il_ind=att_get_index( td_file%t_att, "SUBDOMAIN_J_last_indoor_indices" )
      il_lej(:) = 0
      IF( il_ind /= 0 )THEN
         il_lej(:) = INT(td_file%t_att(il_ind)%d_value(:))
      ENDIF

      ! check last indoor indices
      IF( ANY(il_lei(:)==0) .OR. ANY(il_lej(:)==0) )THEN
         CALL logger_warn("WRITE FILE: no data for subdomain last indoor indices")
      ENDIF      

      ! write file header
      WRITE(td_file%i_id, IOSTAT=il_status, REC=1 )&
      &  td_file%i_recl, &
      &  td_file%t_dim(1)%i_len, &
      &  td_file%t_dim(2)%i_len, &
      &  td_file%t_dim(3)%i_len, &
      &  td_file%i_n0d, &
      &  td_file%i_n1d, &
      &  td_file%i_n2d, &
      &  td_file%i_n3d, &
      &  td_file%i_rhd, &
      &  il_niproc, il_njproc, il_nproc, &
      &  il_area,                &
      &  il_iglo, il_jglo,       &
      &  il_lci(:), il_lcj(:),   &
      &  il_ldi(:), il_ldj(:),   &
      &  il_lei(:), il_lej(:),   &
      &  il_impp(:), il_jmpp(:)

      DEALLOCATE( il_impp, il_jmpp,&
      &           il_lci,  il_lcj, &
      &           il_ldi,  il_ldj, &
      &           il_lei,  il_lej  )

   END SUBROUTINE iom_rstdimg__write_header 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_rstdimg__write_var(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine write variables in an opened dimg file.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2015
   !> - bug fix: do not use scale factor an offset for case no0d, no1d...
   !>
   !> @param[in] td_file file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file

      ! local variable
      INTEGER(i4) :: il_status
      INTEGER(i4) :: il_rec

      INTEGER(i4),            DIMENSION(:), ALLOCATABLE :: il_start
      INTEGER(i4),            DIMENSION(:), ALLOCATABLE :: il_count
      CHARACTER(LEN=im_vnl),  DIMENSION(:), ALLOCATABLE :: cl_name
      REAL(dp),               DIMENSION(:), ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      ! reform name and record
      ALLOCATE( cl_name(td_file%i_nvar), dl_value(td_file%i_nvar) )

      DO ji=1,td_file%i_nvar

         ! change FillValue to 0.
         CALL var_chg_FillValue(td_file%t_var(ji),0._dp)

         cl_name(ji)  = TRIM(td_file%t_var(ji)%c_name)
         dl_value(ji) = REAL(td_file%t_var(ji)%i_rec,dp)
         
         SELECT CASE (TRIM(td_file%t_var(ji)%c_name))
            CASE('no0d','no1d','no2d','no3d')
            CASE DEFAULT
               
               ! use scale factor and offset
               WHERE( td_file%t_var(ji)%d_value(:,:,:,:) /= &
               &      td_file%t_var(ji)%d_fill )
                  td_file%t_var(ji)%d_value(:,:,:,:) = &
                  &   ( td_file%t_var(ji)%d_value(:,:,:,:) - &
                  &     td_file%t_var(ji)%d_ofs ) / &
                  &   td_file%t_var(ji)%d_scf
               END WHERE

               DO jk=1,td_file%t_var(ji)%t_dim(3)%i_len
                  SELECT CASE (td_file%t_var(ji)%i_ndim)
                     CASE(0)
                        ! special case for 0d, value save in rec
                        dl_value(ji)=td_file%t_var(ji)%d_value(1,1,1,1)
                        il_rec = td_file%t_var(ji)%i_rec
                     CASE(1,2)
                        il_rec = td_file%t_var(ji)%i_rec
                     CASE(3)
                        il_rec = td_file%t_var(ji)%i_rec + jk -1
                  END SELECT 
                  WRITE( td_file%i_id, IOSTAT=il_status, REC=il_rec ) &
                  &  td_file%t_var(ji)%d_value(:,:,jk,1)
                  CALL fct_err(il_status)
                  IF( il_status /= 0 )THEN
                     CALL logger_error("IOM RSTDIMG WRITE VAR: can not "//&
                     &  "write variable "//TRIM(td_file%t_var(ji)%c_name)//&
                     &  " in record "//TRIM(fct_str(il_rec)))
                  ENDIF
               ENDDO
            END SELECT

      ENDDO

      ALLOCATE( il_start(4), il_count(4) )

      il_start(1) = 1
      il_count(1) = td_file%i_n0d
      
      il_start(2) = 1 + il_count(1)
      il_count(2) = il_start(2) - 1 + td_file%i_n1d
      
      il_start(3) = 1 + il_count(2)
      il_count(3) = il_start(3) - 1 + td_file%i_n2d
      
      il_start(4) = 1 + il_count(3)
      il_count(4) = il_start(4) - 1 + td_file%i_n3d

      WRITE(td_file%i_id, IOSTAT=il_status, REC=td_file%i_rhd )&
      &  cl_name(il_start(1):il_count(1)), dl_value(il_start(1):il_count(1)),&
      &  cl_name(il_start(2):il_count(2)), dl_value(il_start(2):il_count(2)),&
      &  cl_name(il_start(3):il_count(3)), dl_value(il_start(3):il_count(3)),&
      &  cl_name(il_start(4):il_count(4)), dl_value(il_start(4):il_count(4))
      CALL fct_err(il_status)
      IF( il_status /= 0 )THEN
         CALL logger_error("IOM RSTDIMG WRITE VAR: can not "//&
         &  "write restart header in record "//TRIM(fct_str(td_file%i_rhd)))
      ENDIF

      ! clean
      DEALLOCATE( cl_name, dl_value )
      DEALLOCATE( il_start, il_count )

   END SUBROUTINE iom_rstdimg__write_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE iom_rstdimg
