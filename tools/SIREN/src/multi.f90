!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> This module manage multi file structure.
!>
!> @details
!>    define type TMULTI:<br/>
!> @code
!>    TYPE(TMULTI) :: tl_multi
!> @endcode
!>
!>    to initialize a multi-file structure:<br/>
!> @code
!>    tl_multi=multi_init(cd_varfile(:))
!> @endcode
!>       - cd_varfile : array of variable with file path 
!>       ('var1:file1','var2:file2')<br/>
!>          file path could be replaced by a matrix of value.<br/>
!>          separators used to defined matrix are:
!>             - ',' for line
!>             - '/' for row
!>             - '\' for level<br/>
!>             Example:<br/>
!>                - 'var1:3,2,3/1,4,5'
!>                - 3,2,3/1,4,5  =>  
!>                      @f$ \left( \begin{array}{ccc}
!>                           3 & 2 & 3 \\
!>                           1 & 4 & 5 \end{array} \right) @f$<br/>
!> 
!>    to get the number of mpp file in mutli file structure:<br/>
!>    - tl_multi\%i_nmpp
!>
!>    to get the total number of variable in mutli file structure:<br/>
!>    - tl_multi\%i_nvar
!>
!>    @note number of variable and number of file could differ cause several variable
!>    could be in the same file.
!>
!>    to get array of mpp structure in mutli file structure:<br/>
!>    - tl_multi\%t_mpp(:)
!>
!>    to print information about multi structure:<br/>
!> @code
!>    CALL multi_print(td_multi)
!> @endcode
!>
!>    to clean multi file strucutre:<br/>
!> @code
!>    CALL multi_clean(td_multi)
!> @endcode
!>       - td_multi is multi file structure
!>
!> @author
!>  J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date October, 2014
!> - use mpp file structure instead of file
!> @date November, 2014 
!> - Fix memory leaks bug
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE multi

   USE kind                            ! F90 kind parameter
   USE logger                          ! log file manager
   USE fct                             ! basic useful function
   USE dim                             ! dimension manager
   USE var                             ! variable manager
   USE file                            ! file manager
   USE iom                             ! I/O manager
   USE mpp                             ! MPP manager
   USE iom_mpp                         ! MPP I/O manager

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! type and variable
   PUBLIC :: TMULTI       !< multi file structure

   ! function and subroutine
   PUBLIC :: multi_copy        !< copy multi structure
   PUBLIC :: multi_init        !< initialise multi structure
   PUBLIC :: multi_clean       !< clean multi strcuture
   PUBLIC :: multi_print       !< print information about milti structure

   PRIVATE :: multi__add_mpp   !< add file strucutre to multi file structure
   PRIVATE :: multi__copy_unit !< copy multi file structure
   PRIVATE :: multi__get_perio !< read periodicity from namelist

   TYPE TMULTI !< multi file structure
      ! general 
      INTEGER(i4)                         :: i_nmpp  = 0         !< number of mpp files 
      INTEGER(i4)                         :: i_nvar  = 0         !< total number of variables
      TYPE(TMPP) , DIMENSION(:), POINTER  :: t_mpp => NULL()     !< mpp files composing multi
   END TYPE

   INTERFACE multi_copy
      MODULE PROCEDURE multi__copy_unit   ! copy multi file structure
   END INTERFACE   

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION multi__copy_unit(td_multi) &
         & RESULT (tf_multi)
   !-------------------------------------------------------------------
   !> @brief
   !> This function copy multi mpp structure in another one
   !> @details 
   !> file variable value are copied in a temporary array, 
   !> so input and output file structure value do not point on the same 
   !> "memory cell", and so on are independant. 
   !>
   !> @warning do not use on the output of a function who create or read an
   !> attribute (ex: tl_att=att_copy(att_init()) is forbidden).
   !> This will create memory leaks.
   !> @warning to avoid infinite loop, do not use any function inside 
   !> this subroutine
   !>   
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date November, 2014
   !>    - use function instead of overload assignment operator (to avoid memory leak)
   !>
   !> @param[in] td_multi    mpp structure
   !> @return copy of input multi structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMULTI), INTENT(IN)  :: td_multi

      ! function
      TYPE(TMULTI)              :: tf_multi

      ! local variable
      TYPE(TMPP) :: tl_mpp

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      tf_multi%i_nmpp = td_multi%i_nmpp
      tf_multi%i_nvar = td_multi%i_nvar

      ! copy variable structure
      IF( ASSOCIATED(tf_multi%t_mpp) )THEN
         CALL mpp_clean(tf_multi%t_mpp(:))
         DEALLOCATE(tf_multi%t_mpp)
      ENDIF
      IF( ASSOCIATED(td_multi%t_mpp) .AND. tf_multi%i_nmpp > 0 )THEN
         ALLOCATE( tf_multi%t_mpp(tf_multi%i_nmpp) )
         DO ji=1,tf_multi%i_nmpp
            tl_mpp = mpp_copy(td_multi%t_mpp(ji))
            tf_multi%t_mpp(ji) = mpp_copy(tl_mpp)
         ENDDO
         ! clean
         CALL mpp_clean(tl_mpp)
      ENDIF

   END FUNCTION multi__copy_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION multi_init(cd_varfile) &
         & RESULT (tf_multi)
   !-------------------------------------------------------------------
   !> @brief This subroutine initialize multi file structure.
   !>
   !> @details
   !> if variable name is 'all', add all the variable of the file in mutli file
   !> structure.
   !> Optionnaly, periodicity could be read behind filename.
   !>
   !> @note if first character of filename is numeric, assume matrix is given as
   !> input.<br/>
   !> create pseudo file named 'data-*', with matrix read as variable value.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2015 
   !> - check if variable to be read is in file
   !> @date January, 2016
   !> - read variable dimensions
   !> @date July, 2016
   !> - get variable to be read and associated file first
   !> @date August, 2017
   !> - get perio from namelist 
   !> @date January, 2019
   !> - create and clean file structure to avoid memory leaks
   !> - fill value read from array of variable structure
   !> @date May, 2019
   !> - compare each elt of cl_tabfile to cl_file
   !> @date August, 2019
   !> - use periodicity read from namelist, and store in multi structure
   !>
   !> @param[in] cd_varfile   variable location information (from namelist) 
   !> @return multi file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), DIMENSION(:), INTENT(IN) :: cd_varfile

      ! function
      TYPE(TMULTI)                               :: tf_multi

      ! parameters
      INTEGER(i4)   , PARAMETER        :: ip_nmaxfiles = 50
      INTEGER(i4)   , PARAMETER        :: ip_nmaxvars = 100

      ! local variable
      INTEGER(i4)                                             :: il_nvar
      INTEGER(i4)                                             :: il_nvarin
      INTEGER(i4)                                             :: il_nfiles
      INTEGER(i4)                                             :: il_varid
      INTEGER(i4)                                             :: il_perio

      REAL(dp)                                                :: dl_fill
      CHARACTER(LEN=lc)                                       :: cl_name
      CHARACTER(LEN=lc)                                       :: cl_varname
      CHARACTER(LEN=lc)                                       :: cl_lower
      CHARACTER(LEN=lc)                                       :: cl_file
      CHARACTER(LEN=lc)                                       :: cl_matrix

      CHARACTER(LEN=lc), DIMENSION(ip_nmaxfiles)              :: cl_tabfile
      CHARACTER(LEN=lc), DIMENSION(ip_nmaxfiles, ip_nmaxvars) :: cl_tabvar

      LOGICAL                                                 :: ll_dim

      TYPE(TDIM), DIMENSION(ip_maxdim)                        :: tl_dim

      TYPE(TVAR)                                              :: tl_var
      TYPE(TVAR) , DIMENSION(:), ALLOCATABLE                  :: tl_varin

      TYPE(TMPP)                                              :: tl_mpp

      TYPE(TFILE)                                             :: tl_file

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      INTEGER(i4) :: jf
      INTEGER(i4) , DIMENSION(ip_nmaxvars) :: jv
      !----------------------------------------------------------------

      ji=1
      jf=0
      jv(:)=0
      cl_tabfile(:)=''
      DO WHILE( TRIM(cd_varfile(ji)) /= '' )

         cl_name=fct_split(cd_varfile(ji),1,':')
         IF( TRIM(cl_name) == '' )THEN
            CALL logger_error("MULTI INIT: variable name "//&
            &                 "is empty. check namelist.")
         ENDIF

         cl_file=fct_split(cd_varfile(ji),2,':')
         IF( TRIM(cl_file) == '' )THEN
            CALL logger_error("MULTI INIT: file name matching variable "//&
            &                 TRIM(cl_name)//" is empty. check namelist.")
         ENDIF
         IF( LEN(TRIM(cl_file)) >= lc )THEN
            CALL logger_fatal("MULTI INIT: file name too long (>"//&
            &          TRIM(fct_str(lc))//"). check namelist.")
         ENDIF
         
         IF( TRIM(cl_file) /= '' )THEN
            jk=0
            DO jj=1,jf
               IF( TRIM(cl_file) == TRIM(cl_tabfile(jj)) )THEN            
                  jk=jj
                  EXIT
               ENDIF
            ENDDO
            IF ( jk /= 0 )then
               jv(jk)=jv(jk)+1
               cl_tabvar(jk,jv(jk))=TRIM(cl_name)
            ELSE ! jk == 0
               jf=jf+1
               IF( jf > ip_nmaxfiles )THEN
                  CALL logger_fatal("MULTI INIT: too much files in "//&
                  &  "varfile (>"//TRIM(fct_str(ip_nmaxfiles))//&
                  &  "). check namelist.")
               ENDIF
               cl_tabfile(jf)=TRIM(cl_file)
               jv(jf)=jv(jf)+1
               cl_tabvar(jf,jv(jf))=TRIM(cl_name)
            ENDIF
         ENDIF

         ji=ji+1
      ENDDO

!print *,'============'
!print *,jf,' files ','============'
!DO ji=1,jf
!   print *,'file ',trim(cl_tabfile(ji))
!   print *,jv(ji),' vars '
!   DO jj=1,jv(ji)
!      print *,'var ',trim(cl_tabvar(ji,jj))
!   ENDDO
!ENDDO
!print *,'============'


      il_nfiles=jf
      il_nvar=0
      DO ji=1,il_nfiles
         cl_file=TRIM(cl_tabfile(ji))

         cl_matrix=''
         IF( fct_is_num(cl_file(1:1)) )THEN
            cl_matrix=TRIM(cl_file)
            WRITE(cl_file,'(a,i2.2)')'data-',ji

            DO jj=1,jv(ji)
               cl_name=TRIM(cl_tabvar(ji,jv(ji)))
               cl_lower=TRIM(fct_lower(cl_name))

               tl_var=var_init(TRIM(cl_name))
               CALL var_read_matrix(tl_var, cl_matrix)

               IF( jj == 1 )THEN
                  ! create mpp structure
                  tl_mpp=mpp_init(TRIM(cl_file), tl_var)
               ENDIF

               ! add variable
               CALL mpp_add_var(tl_mpp,tl_var)
               ! number of variable
               il_nvar=il_nvar+1

            ENDDO

         ELSE
            CALL multi__get_perio(cl_file, il_perio)

            tl_file=file_init(TRIM(cl_file), id_perio=il_perio)
            tl_mpp=mpp_init( tl_file, id_perio=il_perio )
            ! clean
            CALL file_clean(tl_file)

            il_nvarin=tl_mpp%t_proc(1)%i_nvar
            ALLOCATE(tl_varin(il_nvarin))
            DO jj=1,il_nvarin
               tl_varin(jj)=var_copy(tl_mpp%t_proc(1)%t_var(jj))
               DO jl=1,ip_maxdim
                  IF( tl_varin(jj)%t_dim(jl)%l_use )THEN
                     tl_varin(jj)%t_dim(jl)=dim_copy(tl_mpp%t_dim(jl))
                  ENDIF
               ENDDO
            ENDDO

            ! clean all varible
            CALL mpp_del_var(tl_mpp)

            DO jj=1,jv(ji)
               cl_name=TRIM(cl_tabvar(ji,jj))
               cl_lower=TRIM(fct_lower(cl_name))
               ! define variable
               IF( TRIM(fct_lower(cl_lower)) /= 'all' )THEN

                  ! check if variable is in file
                  il_varid=var_get_index(tl_varin(:),cl_lower)
                  IF( il_varid == 0 )THEN
                     CALL logger_fatal("MULTI INIT: variable "//&
                        & TRIM(cl_name)//" not in file "//&
                        & TRIM(cl_file) )
                  ENDIF

                  ! get (global) variable dimension
                  tl_dim(jp_I)=dim_copy(tl_varin(il_varid)%t_dim(jp_I))
                  tl_dim(jp_J)=dim_copy(tl_varin(il_varid)%t_dim(jp_J))
                  tl_dim(jp_K)=dim_copy(tl_varin(il_varid)%t_dim(jp_K))
                  tl_dim(jp_L)=dim_copy(tl_varin(il_varid)%t_dim(jp_L))

                  cl_varname=tl_varin(il_varid)%c_name
                  dl_fill=tl_varin(il_varid)%d_fill

                  tl_var=var_init(TRIM(cl_varname), td_dim=tl_dim(:), &
                     &            dd_fill=dl_fill)

                  ! add variable
                  CALL mpp_add_var(tl_mpp,tl_var)

                  ! number of variable
                  il_nvar=il_nvar+1

                  ! clean structure
                  CALL var_clean(tl_var)

               ELSE ! cl_lower == 'all'

                  DO jk=il_nvarin,1,-1

                     ! check if variable is dimension
                     ll_dim=.FALSE.
                     DO jl=1,ip_maxdim
                        IF( TRIM(tl_mpp%t_proc(1)%t_dim(jl)%c_name) == &
                        &   TRIM(tl_varin(jk)%c_name) )THEN
                           ll_dim=.TRUE.
                           CALL logger_trace("MULTI INIT: "//&
                           &  TRIM(tl_varin(jk)%c_name)//&
                           &  ' is var dimension')
                           EXIT
                        ENDIF
                     ENDDO
                     ! do not use variable dimension
                     IF( ll_dim )THEN
                        tl_var=var_init( TRIM(tl_varin(jk)%c_name) )
                        ! delete variable
                        CALL mpp_del_var(tl_mpp,tl_var)
                        ! clean structure
                        CALL var_clean(tl_var)
                     ELSE
                        ! add variable
                        CALL mpp_add_var(tl_mpp, tl_varin(jk))
                        ! number of variable
                        il_nvar=il_nvar+1
                     ENDIF

                  ENDDO

               ENDIF
            ENDDO
            ! clean structure
            CALL var_clean(tl_varin)
            DEALLOCATE(tl_varin)

         ENDIF

         CALL multi__add_mpp(tf_multi, tl_mpp)

         ! update total number of variable
         tf_multi%i_nvar=tf_multi%i_nvar+tl_mpp%t_proc(1)%i_nvar

         ! clean
         CALL mpp_clean(tl_mpp)

      ENDDO

   END FUNCTION multi_init
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE multi_clean(td_multi)
   !-------------------------------------------------------------------
   !> @brief This subroutine clean multi file strucutre.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - nullify mpp structure in multi file structure
   !>
   !> @param[in] td_multi  multi file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMULTI), INTENT(INOUT) :: td_multi

      ! local variable
      TYPE(TMULTI) :: tl_multi ! empty multi file structure

      ! loop indices
      !----------------------------------------------------------------

      CALL logger_info( " CLEAN: reset multi file " )

      IF( ASSOCIATED( td_multi%t_mpp ) )THEN
         CALL mpp_clean(td_multi%t_mpp(:))
         DEALLOCATE(td_multi%t_mpp)
         NULLIFY(td_multi%t_mpp)
      ENDIF

      ! replace by empty structure
      td_multi=multi_copy(tl_multi)

   END SUBROUTINE multi_clean
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE multi_print(td_multi)
   !-------------------------------------------------------------------
   !> @brief This subroutine print some information about mpp strucutre.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - print periodicity
   !> @date May, 2019
   !> - specify format output
   !>
   !> @param[in] td_multi multi file structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMULTI), INTENT(IN) :: td_multi

      ! local variable

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      ! print file
      IF( td_multi%i_nmpp /= 0 .AND. ASSOCIATED(td_multi%t_mpp) )THEN
         WRITE(*,'(/a,i3)') 'MULTI: total number of file(s): ',&
         &  td_multi%i_nmpp
         WRITE(*,'(6x,a,i3)') ' total number of variable(s): ',&
         &  td_multi%i_nvar
         DO ji=1,td_multi%i_nmpp
            WRITE(*,'(3x,3a)') 'FILE ',TRIM(td_multi%t_mpp(ji)%c_name),&
            & ' CONTAINS'
            DO jj=1,td_multi%t_mpp(ji)%t_proc(1)%i_nvar
               IF( ASSOCIATED(td_multi%t_mpp(ji)%t_proc(1)%t_var) )THEN
                  WRITE(*,'(6x,a)') &
                  &  TRIM(td_multi%t_mpp(ji)%t_proc(1)%t_var(jj)%c_name)
                  !WRITE(*,'(6x,a,i0)') 'perio ',td_multi%t_mpp(ji)%t_proc(1)%i_perio
               ENDIF
            ENDDO
         ENDDO
      ENDIF

   END SUBROUTINE multi_print
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE multi__add_mpp(td_multi, td_mpp)
   !-------------------------------------------------------------------
   !> @brief
   !>    This subroutine add file to multi file structure.
   !>
   !> @detail
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date October, 2014
   !> - use mpp file structure instead of file
   !> @date January, 2019
   !> - deallocate mpp structure whatever happens
   !>
   !> @param[inout] td_multi  multi mpp file strcuture
   !> @param[in]    td_mpp    mpp file strcuture
   !> @return mpp file id in multi mpp file structure
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      TYPE(TMULTI), INTENT(INOUT) :: td_multi
      TYPE(TMPP)  , INTENT(IN)    :: td_mpp

      ! local variable
      INTEGER(i4) :: il_status
      INTEGER(i4) :: il_mppid
      
      TYPE(TMPP), DIMENSION(:), ALLOCATABLE :: tl_mpp

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      il_mppid=0
      IF( ASSOCIATED(td_multi%t_mpp) )THEN
         il_mppid=mpp_get_index(td_multi%t_mpp(:),TRIM(td_mpp%c_name))
      ENDIF

      IF( il_mppid /= 0 )THEN

            CALL logger_debug( " MULTI ADD FILE: mpp file "//TRIM(td_mpp%c_name)//&
            &               " already in multi mpp file structure")

            ! add new variable
            DO ji=1,td_mpp%t_proc(1)%i_nvar
               CALL mpp_add_var(td_multi%t_mpp(il_mppid), td_mpp%t_proc(1)%t_var(ji))
            ENDDO

      ELSE
 
         CALL logger_trace("MULTI ADD MPP: add mpp "//&
         &               TRIM(td_mpp%c_name)//" in multi mpp file structure")

         IF( td_multi%i_nmpp > 0 )THEN
            ! 
            ! already other mpp file in multi file structure
            ALLOCATE( tl_mpp(td_multi%i_nmpp), stat=il_status )
            IF(il_status /= 0 )THEN

               CALL logger_error( " MULTI ADD MPP FILE: not enough space to put &
               &               mpp file in multi mpp file structure")

            ELSE
               ! save temporary multi file structure
               tl_mpp(:)=mpp_copy(td_multi%t_mpp(:))

               CALL mpp_clean(td_multi%t_mpp(:))
               DEALLOCATE( td_multi%t_mpp )
               ALLOCATE( td_multi%t_mpp(td_multi%i_nmpp+1), stat=il_status)
               IF(il_status /= 0 )THEN

                  CALL logger_error( " MULTI ADD MPP FILE: not enough space "//&
                  &  "to put mpp file in multi mpp file structure ")

               ENDIF

               ! copy mpp file in multi mpp file before
               td_multi%t_mpp(1:td_multi%i_nmpp) = mpp_copy(tl_mpp(:))

               ! clean
               CALL mpp_clean(tl_mpp(:))
            ENDIF
            DEALLOCATE(tl_mpp)

         ELSE
            ! no file in multi file structure
            IF( ASSOCIATED(td_multi%t_mpp) )THEN
               CALL mpp_clean(td_multi%t_mpp(:))
               DEALLOCATE(td_multi%t_mpp)
            ENDIF
            ALLOCATE( td_multi%t_mpp(td_multi%i_nmpp+1), stat=il_status )
            IF(il_status /= 0 )THEN

               CALL logger_error( " MULTI ADD MPP FILE: not enough space "//&
               &  "to put mpp file in multi mpp file structure " )

            ENDIF
         ENDIF

         ! update number of mpp
         td_multi%i_nmpp=td_multi%i_nmpp+1

         ! add new mpp
         td_multi%t_mpp(td_multi%i_nmpp)=mpp_copy(td_mpp)

      ENDIF

   END SUBROUTINE multi__add_mpp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE multi__get_perio(cd_file, id_perio)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine check if variable file, read in namelist, contains 
   !> periodicity value and return it if true. 
   !> 
   !> @details
   !> periodicity value is assume to follow string "perio ="
   !>
   !> @author J.Paul
   !> @date January, 2019 - Initial Version
   !> @date August, 209
   !> - rewrite function to subroutine
   !> - output filename string contains only filename (no more periodicity if
   !> given)
   !>
   !> @param[inout] cd_file    file name
   !> @param[  out] id_perio   NEMO periodicity 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(INOUT) :: cd_file
      INTEGER(i4)     , INTENT(  OUT) :: id_perio

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      CHARACTER(LEN=lc) :: cl_perio
 
      INTEGER(i4)       :: il_ind

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      ! init
      cl_perio=''
      id_perio=-1

      ji=1
      cl_tmp=fct_split(cd_file,ji,';')
      DO WHILE( TRIM(cl_tmp) /= '' )
         il_ind=INDEX(TRIM(cl_tmp),'perio')
         IF( il_ind /= 0 )THEN
            ! check character just after
            jj=il_ind+LEN('perio')
            IF(  TRIM(cl_tmp(jj:jj)) == ' ' .OR. &
            &    TRIM(cl_tmp(jj:jj)) == '=' )THEN
               cl_perio=fct_split(cl_tmp,2,'=')
               EXIT
            ENDIF
         ENDIF
         ji=ji+1
         cl_tmp=fct_split(cd_file,ji,';')         
      ENDDO
      cd_file=fct_split(cd_file,1,';')

      IF( TRIM(cl_perio) /= '' )THEN
         IF( fct_is_num(cl_perio) )THEN
            READ(cl_perio,*) id_perio
            CALL logger_debug("MULTI GET PERIO: will use periodicity value of "//&
            &  TRIM(fct_str(id_perio))//" for file "//TRIM(cd_file) )
         ELSE
            CALL logger_error("MULTI GET PERIO: invalid periodicity value ("//&
               & TRIM(cl_perio)//") for file "//TRIM(cd_file)//&
               & ". check namelist." )
         ENDIF
      ENDIF

   END SUBROUTINE multi__get_perio
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE multi

