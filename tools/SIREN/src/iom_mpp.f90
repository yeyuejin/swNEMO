!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief This module manage massively parallel processing Input/Output manager.
!> Library to read/write mpp files.
!>
!> @details
!>    to open mpp files (only file to be used (see mpp_get_use) 
!>    will be open):<br/>
!> @code
!>    CALL iom_mpp_open(td_mpp)
!> @endcode
!>       - td_mpp is a mpp structure
!>
!>    to creates mpp files:<br/>
!> @code
!>    CALL iom_mpp_create(td_mpp)
!> @endcode
!>       - td_mpp is a mpp structure
!>
!>    to write in mpp files :<br/>
!> @code
!>    CALL  iom_mpp_write_file(td_mpp)
!> @endcode
!>       - td_mpp is a mpp structure
!>
!>    to close mpp files:<br/>
!> @code
!>    CALL iom_mpp_close(td_mpp)
!> @endcode
!>
!>    to read one variable in an mpp files:<br/>
!> @code
!>    tl_var=iom_mpp_read_var( td_mpp, id_varid, [id_start, id_count] [,id_ew] ) 
!> @endcode
!>    or
!> @code
!>    tl_var=iom_mpp_read_var( td_mpp, cd_name, [id_start, id_count] [,id_ew] ) 
!> @endcode
!>       - td_mpp is a mpp structure
!>       - id_varid is a variable id
!>       - cd_name is variable name or standard name
!>       - id_start is a integer(4) 1D array of index from which the data 
!>          values will be read [optional]
!>       - id_count is a integer(4) 1D array of the number of indices selected
!>          along each dimension [optional]
!>       - id_ew East West overlap [optional]
!>
!>    to fill variable value in mpp structure:<br/>
!> @code
!>    CALL iom_mpp_fill_var(td_mpp, id_varid, [id_start, id_count] [,id_ew] )
!> @endcode
!>    or<br/>
!> @code
!>    CALL iom_mpp_fill_var(td_mpp, cd_name, [id_start, id_count] [,id_ew] )
!> @endcode
!>       - td_mpp is mpp structure
!>       - id_varid is variable id
!>       - cd_name is variable name or standard name
!>       - id_start is a integer(4) 1D array of index from which the data 
!>          values will be read [optional]
!>       - id_count is a integer(4) 1D array of the number of indices selected
!>          along each dimension [optional]
!>       - id_ew East West overlap [optional]
!>
!>    to fill all variable in mpp structure:<br/>
!> @code
!>    CALL iom_mpp_fill_var(td_mpp, [id_start, id_count] [,id_ew] )
!> @endcode
!>       - td_mpp is mpp structure
!>       - id_start is a integer(4) 1D array of index from which the data 
!>          values will be read [optional]
!>       - id_count is a integer(4) 1D array of the number of indices selected
!>          along each dimension [optional]
!>       - id_ew East West overlap
!>
!>    to write files composong mpp strucutre:<br/>
!> @code
!>    CALL iom_mpp_write_file(td_mpp) 
!> @endcode
!>
!> @author
!> J.Paul
!>
!> @date November, 2013 - Initial Version
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE iom_mpp

   USE netcdf                          ! nf90 library
   USE global                          ! global parameter
   USE kind                            ! F90 kind parameter
   USE fct                             ! basic useful function
   USE logger                          ! log file manager
   USE dim                             ! dimension manager
   USE att                             ! attribute manager
   USE var                             ! variable manager
   USE file                            ! file manager
   USE iom                             ! I/O manager
   USE mpp                             ! mpp manager

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! function and subroutine
   PUBLIC :: iom_mpp_open                    !< open all files composing mpp structure
   PUBLIC :: iom_mpp_create                  !< creates files composing mpp structure
   PUBLIC :: iom_mpp_close                   !< close file composing mpp structure
   PUBLIC :: iom_mpp_read_var                !< read one variable in an mpp structure
   PUBLIC :: iom_mpp_write_file              !< write mpp structure in files

   PRIVATE :: iom_mpp__read_var_id           ! read one variable in an mpp structure, given variable id
   PRIVATE :: iom_mpp__read_var_name         ! read one variable in an mpp structure, given variable name
   PRIVATE :: iom_mpp__read_var_value        ! read variable value in an mpp structure

   INTERFACE iom_mpp_read_var                   ! read one variable in an mpp structure
      MODULE PROCEDURE iom_mpp__read_var_id     ! given variable id
      MODULE PROCEDURE iom_mpp__read_var_name   ! given variable name 
   END INTERFACE iom_mpp_read_var

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_mpp_open(td_mpp, id_perio, id_ew)
   !-------------------------------------------------------------------
   !> @brief This subroutine open files composing mpp structure to be used.
   !> @details
   !> If try to open a file in write mode that did not exist, create it.<br/>
   !> 
   !> If file already exist, get information about:
   !> - the number of variables
   !> - the number of dimensions
   !> - the number of global attributes
   !> - the ID of the unlimited dimension
   !> - the file format
   !> and finally read dimensions.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date August, 2017 
   !> - handle use of domain decomposition for monoproc file
   !>
   !> @param[inout] td_mpp mpp structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP) , INTENT(INOUT)  :: td_mpp
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_perio
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_ew

      ! local variable
      CHARACTER(LEN=lc) :: cl_name
      INTEGER(i4) :: il_pid  
      INTEGER(i4) :: il_impp 
      INTEGER(i4) :: il_jmpp 
      INTEGER(i4) :: il_lci  
      INTEGER(i4) :: il_lcj  
      INTEGER(i4) :: il_ldi  
      INTEGER(i4) :: il_ldj  
      INTEGER(i4) :: il_lei  
      INTEGER(i4) :: il_lej  
      LOGICAL     :: ll_ctr  
      LOGICAL     :: ll_use  
      LOGICAL     :: ll_create  
      INTEGER(i4) :: il_iind 
      INTEGER(i4) :: il_jind 

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! check if mpp exist
      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN

         CALL logger_error( " IOM MPP OPEN: domain decomposition not define "//&
         &               " in mpp strcuture "//TRIM(td_mpp%c_name))

      ELSE
         ! 
         td_mpp%i_id=1

         ! if no processor file selected
         ! force to open all files 
         IF( .NOT. ANY( td_mpp%t_proc(:)%l_use ) )THEN
            td_mpp%t_proc(:)%l_use=.TRUE.
         ENDIF

         ! add suffix to mpp name
         td_mpp%c_name=file_add_suffix( TRIM(td_mpp%c_name), &
                                      & TRIM(td_mpp%c_type) )

         td_mpp%t_proc(:)%c_type=TRIM(td_mpp%c_type) 
         IF( td_mpp%i_nproc > 1 .AND. td_mpp%l_usempp )THEN
            DO ji=1,td_mpp%i_nproc
               IF( td_mpp%t_proc(ji)%l_use )THEN

                  SELECT CASE(TRIM(td_mpp%c_type))
                  CASE('cdf')
                     cl_name=TRIM( file_rename(td_mpp%c_name, ji-1) )
                  CASE('dimg')
                     cl_name=TRIM( file_rename(td_mpp%c_name, ji) )
                  CASE DEFAULT
                     CALL logger_fatal("IOM MPP OPEN: can not open file "//&
                     &  "of type "//TRIM(td_mpp%c_type))
                  END SELECT

                  td_mpp%t_proc(ji)%c_name=TRIM(cl_name)

                  CALL iom_open(td_mpp%t_proc(ji))

               ENDIF
            ENDDO
         ELSE ! td_mpp%i_nproc == 1 
               cl_name=TRIM( file_rename(td_mpp%c_name) )
               td_mpp%t_proc(1)%c_name=TRIM(cl_name)

               CALL iom_open(td_mpp%t_proc(1))

               IF( .NOT. td_mpp%l_usempp )THEN
                  ! copy file structure of first proc, except layout decomposition
                  ! do not do it when creating output file.
                  ll_create=( ALL(td_mpp%t_proc(:)%l_wrt) .AND. &
                  &           ALL(td_mpp%t_proc(:)%l_use) )
                  IF( .NOT. ll_create )THEN
                     DO ji=2,td_mpp%i_nproc
                        IF( td_mpp%t_proc(ji)%l_use )THEN
                           il_pid  = td_mpp%t_proc(ji)%i_pid  
                           il_impp = td_mpp%t_proc(ji)%i_impp 
                           il_jmpp = td_mpp%t_proc(ji)%i_jmpp 
                           il_lci  = td_mpp%t_proc(ji)%i_lci  
                           il_lcj  = td_mpp%t_proc(ji)%i_lcj  
                           il_ldi  = td_mpp%t_proc(ji)%i_ldi  
                           il_ldj  = td_mpp%t_proc(ji)%i_ldj  
                           il_lei  = td_mpp%t_proc(ji)%i_lei  
                           il_lej  = td_mpp%t_proc(ji)%i_lej  
                           ll_ctr  = td_mpp%t_proc(ji)%l_ctr  
                           ll_use  = td_mpp%t_proc(ji)%l_use  
                           il_iind = td_mpp%t_proc(ji)%i_iind 
                           il_jind = td_mpp%t_proc(ji)%i_jind 

                           td_mpp%t_proc(ji)=file_copy(td_mpp%t_proc(1))
                           td_mpp%t_proc(ji)%i_id=td_mpp%t_proc(1)%i_id
                           td_mpp%t_proc(ji)%l_def=.FALSE.

                           td_mpp%t_proc(ji)%i_pid  = il_pid  
                           td_mpp%t_proc(ji)%i_impp = il_impp 
                           td_mpp%t_proc(ji)%i_jmpp = il_jmpp 
                           td_mpp%t_proc(ji)%i_lci  = il_lci  
                           td_mpp%t_proc(ji)%i_lcj  = il_lcj  
                           td_mpp%t_proc(ji)%i_ldi  = il_ldi  
                           td_mpp%t_proc(ji)%i_ldj  = il_ldj  
                           td_mpp%t_proc(ji)%i_lei  = il_lei  
                           td_mpp%t_proc(ji)%i_lej  = il_lej  
                           td_mpp%t_proc(ji)%l_ctr  = ll_ctr  
                           td_mpp%t_proc(ji)%l_use  = ll_use  
                           td_mpp%t_proc(ji)%i_iind = il_iind 
                           td_mpp%t_proc(ji)%i_jind = il_jind 
                        ENDIF
                     ENDDO
                  ELSE
                     ! keep file id
                     DO ji=2,td_mpp%i_nproc
                        IF( td_mpp%t_proc(ji)%l_use )THEN 
                           td_mpp%t_proc(ji)%i_id=td_mpp%t_proc(1)%i_id
                           td_mpp%t_proc(ji)%l_def=.FALSE.
                        ENDIF
                     ENDDO
                  ENDIF
               ENDIF

         ENDIF

         IF( PRESENT(id_ew) )THEN
            td_mpp%i_ew=id_ew
            ! add east west overlap to each variable
            DO ji=1,td_mpp%i_nproc
               WHERE(td_mpp%t_proc(ji)%t_var(:)%t_dim(1)%l_use)
                  td_mpp%t_proc(ji)%t_var(:)%i_ew=td_mpp%i_ew
               ENDWHERE
            ENDDO
         ENDIF

         IF( PRESENT(id_perio) )THEN
            td_mpp%i_perio=id_perio
         ENDIF

      ENDIF

   END SUBROUTINE iom_mpp_open
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_mpp_create(td_mpp)
   !-------------------------------------------------------------------
   !> @brief This subroutine create files, composing mpp structure to be used,
   !> in write mode.
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_mpp mpp structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP), INTENT(INOUT)  :: td_mpp
      !----------------------------------------------------------------
      ! check if mpp exist
      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN

         CALL logger_error( " IOM MPP CREATE: domain decomposition not define "//&
         &               " in mpp strcuture "//TRIM(td_mpp%c_name))

      ELSE
         ! forced to open in write mode
         td_mpp%t_proc(:)%l_wrt=.TRUE.
         td_mpp%t_proc(:)%l_use=.TRUE.
         CALL iom_mpp_open(td_mpp)
      ENDIF

   END SUBROUTINE iom_mpp_create
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_mpp_close(td_mpp)
   !-------------------------------------------------------------------
   !> @brief This subroutine close files composing mpp structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_mpp mpp structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP), INTENT(INOUT) :: td_mpp

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! check if mpp exist
      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN

         CALL logger_error( " IOM MPP CLOSE: domain decomposition not define "//&
         &               " in mpp strcuture "//TRIM(td_mpp%c_name))

      ELSE
         ! 
         td_mpp%i_id=0         

         IF( td_mpp%l_usempp )THEN
            DO ji=1,td_mpp%i_nproc
               IF( td_mpp%t_proc(ji)%i_id /= 0 )THEN
                  CALL iom_close(td_mpp%t_proc(ji))
               ENDIF
            ENDDO
         ELSE
            IF( td_mpp%t_proc(1)%i_id /= 0 )THEN
               CALL iom_close(td_mpp%t_proc(1))
               td_mpp%t_proc(:)%i_id=0
            ENDIF
         ENDIF
         td_mpp%t_proc(:)%l_use=.FALSE.

      ENDIF

   END SUBROUTINE iom_mpp_close
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_mpp__read_var_id(td_mpp, id_varid, id_start, id_count) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function read variable value in opened mpp files,
   !> given variable id.
   !>
   !> @details
   !> Optionally start indices and number of point to be read could be specify.
   !> as well as East West ovelap of the global domain.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date October, 2014
   !> - use start and count array instead of domain structure.
   !>
   !> @param[in] td_mpp    mpp structure
   !> @param[in] id_varid  variable id
   !> @param[in] id_start  index in the variable from which the data values 
   !> will be read
   !> @param[in] id_count  number of indices selected along each dimension
   !> @return  variable structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP),                INTENT(IN) :: td_mpp
      INTEGER(i4),               INTENT(IN) :: id_varid
      INTEGER(i4), DIMENSION(:), INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4), DIMENSION(:), INTENT(IN), OPTIONAL :: id_count      

      ! function
      TYPE(TVAR)                            :: tf_var

      ! local variable
      INTEGER(i4), DIMENSION(1) :: il_ind
      !----------------------------------------------------------------
      ! check if mpp exist
      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN

         CALL logger_error( " IOM MPP READ VAR: domain decomposition not define "//&
         &               " in mpp strcuture "//TRIM(td_mpp%c_name))

      ELSEIF( td_mpp%i_id == 0 )THEN

         CALL logger_error( " IOM MPP READ VAR: mpp structure not opened. "//&
         &               " can not read variable in "//TRIM(td_mpp%c_name))   
      
      ELSE


         IF( ANY(td_mpp%t_proc(:)%i_id /= 0) )THEN
            ! look for variable id
            il_ind(:)=MINLOC( td_mpp%t_proc(1)%t_var(:)%i_id, &
            &           mask=(td_mpp%t_proc(1)%t_var(:)%i_id==id_varid))
            IF( il_ind(1) /= 0 )THEN

               tf_var=var_copy(td_mpp%t_proc(1)%t_var(il_ind(1)))

               !!! read variable value
               CALL iom_mpp__read_var_value(td_mpp, tf_var, id_start, id_count)

            ELSE
               CALL logger_error( &
               &  " IOM MPP READ VAR: there is no variable with id "//&
               &  TRIM(fct_str(id_varid))//" in processor/file "//&
               &  TRIM(td_mpp%t_proc(1)%c_name))
            ENDIF
         ELSE
            CALL logger_error(" IOM MPP READ VAR: can't read variable, mpp "//&
            &  TRIM(td_mpp%c_name)//" not opened")
         ENDIF

      ENDIF

   END FUNCTION iom_mpp__read_var_id
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_mpp__read_var_name(td_mpp, cd_name, id_start, id_count) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function read variable value in opened mpp files, 
   !> given variable name or standard name.
   !>
   !> @details
   !> Optionally start indices and number of point to be read could be specify.
   !> as well as East West ovelap of the global domain.
   !>
   !> look first for variable name. If it doesn't
   !> exist in file, look for variable standard name.<br/>
   !> If variable name is not present, check variable standard name.<br/>
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date October, 2014
   !> - use start and count array instead of domain structure.
   !>
   !> @param[in] td_mpp    mpp structure
   !> @param[in] cd_name   variable name
   !> @param[in] id_start  index in the variable from which the data values 
   !> will be read
   !> @param[in] id_count  number of indices selected along each dimension
   !> @return  variable structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP),                INTENT(IN) :: td_mpp
      CHARACTER(LEN=*),          INTENT(IN) :: cd_name
      INTEGER(i4), DIMENSION(:), INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4), DIMENSION(:), INTENT(IN), OPTIONAL :: id_count

      ! function
      TYPE(TVAR)                            :: tf_var

      ! local variable
      INTEGER(i4)       :: il_ind
      !----------------------------------------------------------------
      ! check if mpp exist
      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN

         CALL logger_error( " IOM MPP READ VAR: domain decomposition not define "//&
         &               " in mpp strcuture "//TRIM(td_mpp%c_name))

      ELSEIF( td_mpp%i_id == 0 )THEN

         CALL logger_error( " IOM MPP READ VAR: mpp structure not opened. "//&
         &               " can not read variable in "//TRIM(td_mpp%c_name))   
 
      ELSE

            il_ind=var_get_index( td_mpp%t_proc(1)%t_var(:), cd_name)
            IF( il_ind /= 0 )THEN

               tf_var=var_copy(td_mpp%t_proc(1)%t_var(il_ind))

               !!! read variable value
               CALL iom_mpp__read_var_value( td_mpp, tf_var, id_start, id_count)

            ELSE

               CALL logger_fatal( &
               &  " IOM MPP READ VAR: there is no variable with "//&
               &  "name or standard name "//TRIM(cd_name)//&
               &  " in processor/file "//TRIM(td_mpp%t_proc(1)%c_name))
            ENDIF

      ENDIF
      
   END FUNCTION iom_mpp__read_var_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_mpp__read_var_value(td_mpp, td_var, id_start, id_count)
   !-------------------------------------------------------------------
   !> @brief This subroutine read variable value
   !> in an mpp structure.
   !>
   !> @details
   !> Optionally start indices and number of point to be read could be specify.
   !> as well as East West ovelap of the global domain.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date October, 2014
   !> - use start and count array instead of domain structure.
   !>
   !> @param[in] td_mpp    mpp structure
   !> @param[inout] td_var variable structure
   !> @param[in] id_start  index in the variable from which the data values 
   !> will be read
   !> @param[in] id_count  number of indices selected along each dimension
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP)               , INTENT(IN   ) :: td_mpp
      TYPE(TVAR)               , INTENT(INOUT) :: td_var
      INTEGER(i4), DIMENSION(:), INTENT(IN   ), OPTIONAL :: id_start
      INTEGER(i4), DIMENSION(:), INTENT(IN   ), OPTIONAL :: id_count      

      ! local variable
      INTEGER(i4)                       :: il_status
      INTEGER(i4), DIMENSION(4)         :: il_ind
      INTEGER(i4)                       :: il_i1p
      INTEGER(i4)                       :: il_i2p
      INTEGER(i4)                       :: il_j1p
      INTEGER(i4)                       :: il_j2p
      INTEGER(i4)                       :: il_i1
      INTEGER(i4)                       :: il_i2
      INTEGER(i4)                       :: il_j1
      INTEGER(i4)                       :: il_j2

      INTEGER(i4), DIMENSION(ip_maxdim) :: il_start
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_end
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_count      

      INTEGER(i4), DIMENSION(ip_maxdim) :: il_strt
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_cnt      

      TYPE(TATT)                        :: tl_att
      TYPE(TVAR)                        :: tl_var

      ! loop indices
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      il_start(:)=1
      IF( PRESENT(id_start) ) il_start(:)=id_start(:)

      il_count(:)=td_mpp%t_dim(:)%i_len
      IF( PRESENT(id_count) ) il_count(:)=id_count(:)

      CALL logger_debug("IOM MPP READ VAR VALUE: start "//&
               &  TRIM(fct_str(il_start(jp_I)))//","//&
               &  TRIM(fct_str(il_start(jp_J)))//","//&
               &  TRIM(fct_str(il_start(jp_K)))//","//&
               &  TRIM(fct_str(il_start(jp_L))) )
      CALL logger_debug("IOM MPP READ VAR VALUE: count "//&
               &  TRIM(fct_str(il_count(jp_I)))//","//&
               &  TRIM(fct_str(il_count(jp_J)))//","//&
               &  TRIM(fct_str(il_count(jp_K)))//","//&
               &  TRIM(fct_str(il_count(jp_L))) )

      !IF( td_mpp%l_usempp .AND. (PRESENT(id_start) .OR. PRESENT(id_count)))THEN
      !   CALL logger_fatal("IOM MPP READ VAR VALUE: should not use"//&
      !      &  " start or count arguments when usempp is False.")
      !ENDIF

      DO jk=1,ip_maxdim
         IF( .NOT. td_var%t_dim(jk)%l_use )THEN
            il_start(jk) = 1
            il_count(jk) = 1
         ENDIF

         il_end(jk)=il_start(jk)+il_count(jk)-1
      ENDDO

      IF( ANY(il_end(:) > td_mpp%t_dim(:)%i_len) )THEN
            CALL logger_debug("IOM MPP READ VAR VALUE: start + count "//&
               &  TRIM(fct_str(il_end(jp_I)))//","//&
               &  TRIM(fct_str(il_end(jp_J)))//","//&
               &  TRIM(fct_str(il_end(jp_K)))//","//&
               &  TRIM(fct_str(il_end(jp_L))) )
            CALL logger_debug("IOM MPP READ VAR VALUE: dimension "//&
               &  TRIM(fct_str(td_mpp%t_dim(jp_I)%i_len))//","//&
               &  TRIM(fct_str(td_mpp%t_dim(jp_J)%i_len))//","//&
               &  TRIM(fct_str(td_mpp%t_dim(jp_K)%i_len))//","//&
               &  TRIM(fct_str(td_mpp%t_dim(jp_L)%i_len)) )
            CALL logger_fatal("IOM MPP READ VAR VALUE: start + count "//&
            &                 "exceed dimension bound.")
      ENDIF

      ! use domain dimension 
      td_var%t_dim(:)%i_len=il_count(:)

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
         &  " IOM MPP READ VAR VALUE: not enough space to put variable "//&
         &  TRIM(td_var%c_name)//&
         &  " in variable structure")

      ENDIF

      CALL logger_debug("IOM MPP READ VAR VALUE: shape ("//&
      &  TRIM(fct_str(SIZE(td_var%d_value(:,:,:,:),DIM=1)))//","//&
      &  TRIM(fct_str(SIZE(td_var%d_value(:,:,:,:),DIM=2)))//","//&
      &  TRIM(fct_str(SIZE(td_var%d_value(:,:,:,:),DIM=3)))//","//&
      &  TRIM(fct_str(SIZE(td_var%d_value(:,:,:,:),DIM=4)))//")" )
      ! FillValue by default
      td_var%d_value(:,:,:,:)=td_var%d_fill

      ! read processor 
      DO jk=1,td_mpp%i_nproc
         IF( td_mpp%t_proc(jk)%l_use )THEN
             
            ! get processor indices
            il_ind(:)=mpp_get_proc_index( td_mpp, jk )
            il_i1p = il_ind(1)
            il_i2p = il_ind(2)
            il_j1p = il_ind(3)
            il_j2p = il_ind(4)
 
            IF( .NOT. td_var%t_dim(1)%l_use )THEN
               il_i1p=il_start(1) ; il_i2p=il_end(1)
            ENDIF
            IF( .NOT. td_var%t_dim(2)%l_use )THEN
               il_j1p=il_start(2) ; il_j2p=il_end(2)
            ENDIF            
            
            il_i1=MAX(il_i1p, il_start(1))
            il_i2=MIN(il_i2p, il_end(1))

            il_j1=MAX(il_j1p, il_start(2))
            il_j2=MIN(il_j2p, il_end(2))

            IF( (il_i1<=il_i2).AND.(il_j1<=il_j2) )THEN
               IF( td_mpp%l_usempp )THEN
                  il_strt(:)=(/ il_i1-il_i1p+1, &
                  &             il_j1-il_j1p+1, &
                  &             1,1 /)
               ELSE
                  il_strt(:)=(/ il_i1, &
                  &             il_j1, &
                  &             1,1 /)
               ENDIF

               il_cnt(:)=(/ il_i2-il_i1+1,         &
               &            il_j2-il_j1+1,         &
               &            td_var%t_dim(3)%i_len, &
               &            td_var%t_dim(4)%i_len /)

               tl_var=iom_read_var( td_mpp%t_proc(jk), td_var%c_name,&
               &                    il_strt(:), il_cnt(:) )
               ! replace value in output variable structure
               td_var%d_value( il_i1 - il_start(1) + 1 : &
               &               il_i2 - il_start(1) + 1,  &
               &               il_j1 - il_start(2) + 1 : &
               &               il_j2 - il_start(2) + 1,  &
               &               :,:) = tl_var%d_value(:,:,:,:)

               ! clean
               CALL var_clean(tl_var)
            ENDIF

         ENDIF
      ENDDO

      IF( td_var%t_dim(1)%l_use .AND. &
      &   td_var%t_dim(1)%i_len == td_mpp%t_dim(1)%i_len )THEN
         IF( td_mpp%i_ew >= 0 )THEN
            tl_att=att_init("ew_overlap",td_mpp%i_ew)
            CALL var_move_att(td_var,tl_att)
            ! clean 
            CALL att_clean(tl_att)
         ENDIF
      ENDIF

      ! force to change _FillValue to avoid mistake 
      ! with dummy zero _FillValue
      IF( td_var%d_fill == 0._dp )THEN
         CALL var_chg_FillValue(td_var)
      ENDIF      

   END SUBROUTINE iom_mpp__read_var_value
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_mpp_write_file(td_mpp, cd_dimorder)
   !-------------------------------------------------------------------
   !> @brief This subroutine write files composing mpp structure.
   !>
   !> @details
   !> optionally, you could specify the dimension order (default 'xyzt')
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2015 
   !> - add dimension order option 
   !> @date August, 2017 
   !> - handle use of domain decomposition for monoproc file
   !>
   !> @param[inout] td_mpp mpp structure
   !> @param[in] cd_dimorder dimension order
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP)      , INTENT(INOUT) :: td_mpp
      CHARACTER(LEN=*), INTENT(IN   ), OPTIONAL :: cd_dimorder

      ! local variable
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! check if mpp exist
      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN

         CALL logger_error( " MPP WRITE: domain decomposition not define "//&
         &               " in mpp strcuture "//TRIM(td_mpp%c_name))

      ELSE
         IF( td_mpp%l_usempp )THEN
            DO ji=1, td_mpp%i_nproc
               IF( td_mpp%t_proc(ji)%i_id /= 0 )THEN
                  CALL logger_debug("MPP WRITE: proc "//TRIM(fct_str(ji)))
                  CALL iom_write_file(td_mpp%t_proc(ji), cd_dimorder)
               ELSE
                  CALL logger_debug( " MPP WRITE: no id associated to file "//&
                  &              TRIM(td_mpp%t_proc(ji)%c_name) )
               ENDIF
            ENDDO
         ELSE
            CALL iom_write_header(td_mpp%t_proc(1), cd_dimorder, td_mpp%t_dim(:))

            CALL iom_mpp__write_var(td_mpp, cd_dimorder)
         ENDIF
      ENDIF

   END SUBROUTINE iom_mpp_write_file
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_mpp__write_var(td_mpp, cd_dimorder)
   !-------------------------------------------------------------------
   !> @brief This subroutine write variables from mpp structure in one output
   !> file.
   !>
   !> @details
   !> optionally, you could specify the dimension order (default 'xyzt')
   !>
   !> @author J.Paul
   !> @date August, 2017 - Initial Version
   !>
   !> @param[inout] td_mpp mpp structure
   !> @param[in] cd_dimorder dimension order
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP)      , INTENT(INOUT) :: td_mpp
      CHARACTER(LEN=*), INTENT(IN   ), OPTIONAL :: cd_dimorder

      ! local variable
      INTEGER(i4), DIMENSION(4)         :: il_ind
      INTEGER(i4)                       :: il_i1p
      INTEGER(i4)                       :: il_i2p
      INTEGER(i4)                       :: il_j1p
      INTEGER(i4)                       :: il_j2p
      INTEGER(i4)                       :: il_i1
      INTEGER(i4)                       :: il_i2
      INTEGER(i4)                       :: il_j1
      INTEGER(i4)                       :: il_j2

      INTEGER(i4), DIMENSION(ip_maxdim) :: il_start
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_count     

      INTEGER(i4), DIMENSION(ip_maxdim) :: il_strt
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_cnt      

      REAL(dp)                          :: dl_fill

      TYPE(TFILE)                       :: tl_file

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      ! write variable in file
      DO jj = 1, td_mpp%i_nproc
         
         ! link
         tl_file=td_mpp%t_proc(jj)
         CALL logger_debug("IOM MPP WRITE: proc "//fct_str(jj))

         ! get processor indices
         il_ind(:)=mpp_get_proc_index( td_mpp, jj )
         il_i1p = il_ind(1)
         il_i2p = il_ind(2)
         il_j1p = il_ind(3)
         il_j2p = il_ind(4)
      
         IF( jj > 1 )THEN
            ! force to use id from variable write on first proc
            tl_file%t_var(:)%i_id=td_mpp%t_proc(1)%t_var(:)%i_id
         ENDIF

         DO ji = 1, tl_file%i_nvar

            IF( jj > 1 )THEN
               ! check _FillValue
               dl_fill=td_mpp%t_proc(1)%t_var(ji)%d_fill
               IF( tl_file%t_var(ji)%d_fill /= dl_fill )THEN
                  CALL var_chg_FillValue( tl_file%t_var(ji), dl_fill )
               ENDIF
            ENDIF

            il_start(:)=1
            il_count(:)=td_mpp%t_dim(:)%i_len

            IF( .NOT. tl_file%t_var(ji)%t_dim(1)%l_use )THEN
               il_i1p=1 ; il_i2p=1
               il_count(1) = 1
            ENDIF
            IF( .NOT. tl_file%t_var(ji)%t_dim(2)%l_use )THEN
               il_j1p=1 ; il_j2p=1
               il_count(2) = 1
            ENDIF            
            
            il_i1=MAX(il_i1p, il_start(1))
            il_i2=MIN(il_i2p, il_count(1))

            il_j1=MAX(il_j1p, il_start(2))
            il_j2=MIN(il_j2p, il_count(2))
         
            IF( (il_i1<=il_i2).AND.(il_j1<=il_j2) )THEN
               il_strt(:)=(/ il_i1, &
               &             il_j1, &
               &             1,1 /)

               il_cnt(:)=(/ il_i2-il_i1+1,         &
               &            il_j2-il_j1+1,         &
               &            tl_file%t_var(ji)%t_dim(3)%i_len, &
               &            tl_file%t_var(ji)%t_dim(4)%i_len /)

               CALL iom_write_var(tl_file, cd_dimorder, &
               &                  id_start=il_strt(:), &
               &                  id_count=il_cnt(:))
            ENDIF

         ENDDO
      ENDDO

   END SUBROUTINE iom_mpp__write_var   
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE iom_mpp
