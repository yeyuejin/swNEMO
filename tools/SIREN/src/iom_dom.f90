!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief This module allow to read domain (defined as domain structure) in a mpp files.
!>
!> @details
!>    to read one variable in an mpp files over domain defined as domain structure:<br/>
!> @code
!>    tl_var=iom_dom_read_var( td_mpp, id_varid, td_dom ) 
!> @endcode
!>    or
!> @code
!>    tl_var=iom_dom_read_var( td_mpp, cd_name, td_dom ) 
!> @endcode
!>       - td_mpp is a mpp structure
!>       - id_varid is a variable id
!>       - cd_name is variable name or standard name
!>       - td_dom is a domain structure
!>
!> @author
!> J.Paul
!>
!> @date October, 2014 - Initial Version
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE iom_dom

   USE netcdf                          ! nf90 library
   USE global                          ! global parameter
   USE kind                            ! F90 kind parameter
   USE fct                             ! basic useful function
   USE logger                          ! log file manager
   USE dim                             ! dimension manager
   USE att                             ! attribute manager
   USE var                             ! variable manager
   USE iom                             ! I/O manager
   USE mpp                             ! mpp manager
   USe dom                             ! domain manager
   USE iom_mpp                         ! I/O mpp manager

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! function and subroutine
   PUBLIC :: iom_dom_open                    !< open files composing mpp structure over domain to be used
   PUBLIC :: iom_dom_read_var                !< read one variable in an mpp structure over domain to be used
   PUBLIC :: iom_dom_close                   !< close file composing mpp structure over domain

   PRIVATE :: iom_dom__read_var_id           ! read one variable in an mpp structure, given variable id
   PRIVATE :: iom_dom__read_var_name         ! read one variable in an mpp structure, given variable name
   PRIVATE :: iom_dom__read_var_value        ! read variable value in an mpp structure
   PRIVATE :: iom_dom__no_pole_no_overlap    ! do not overlap north fold boundary or east-west boundary
   PRIVATE :: iom_dom__no_pole_cyclic        ! do not overlap north fold boundary. However uses cyclic east-west boundary
   PRIVATE :: iom_dom__no_pole_overlap       ! do not overlap north fold boundary. However overlaps east-west boundary
!   PRIVATE :: iom_dom__pole_no_overlap       ! overlaps north fold boundary. However do not overlap east-west boundary
!   PRIVATE :: iom_dom__pole_cyclic           ! overlaps north fold boundary and uses cyclic east-west boundary
!   PRIVATE :: iom_dom__pole_overlap          ! overlaps north fold boundary and east-west boundary

   INTERFACE iom_dom_read_var                   ! read one variable in an mpp structure
      MODULE PROCEDURE iom_dom__read_var_id     ! given variable id
      MODULE PROCEDURE iom_dom__read_var_name   ! given variable name 
   END INTERFACE iom_dom_read_var

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_dom_open(td_mpp, td_dom, id_perio, id_ew)
   !-------------------------------------------------------------------
   !> @brief This subroutine open files composing mpp structure 
   !> over domain to be used.
   !>
   !> @author J.Paul
   !> @date October, 2014 - Initial Version
   !
   !> @param[inout] td_mpp mpp structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP) , INTENT(INOUT) :: td_mpp
      TYPE(TDOM) , INTENT(IN)    :: td_dom
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_perio
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_ew

      ! local variable
      ! loop indices
      !----------------------------------------------------------------
      ! check if mpp exist
      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN

         CALL logger_error( " IOM DOM OPEN: domain decomposition not define "//&
         &               " in mpp strcuture "//TRIM(td_mpp%c_name))

      ELSE
         ! get processor to be used
         CALL mpp_get_use( td_mpp, td_dom%i_imin, td_dom%i_imax, &
         &                         td_dom%i_jmin, td_dom%i_jmax )

         CALL iom_mpp_open(td_mpp, id_perio, id_ew)

      ENDIF

   END SUBROUTINE iom_dom_open
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_dom_close(td_mpp)
   !-------------------------------------------------------------------
   !> @brief This subroutine close files composing mpp structure.
   !>
   !> @author J.Paul
   !> @date October, 2014 - Initial Version
   !
   !> @param[in] td_mpp mpp structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP), INTENT(INOUT) :: td_mpp

      ! loop indices
      !----------------------------------------------------------------

      CALL iom_mpp_close(td_mpp)

   END SUBROUTINE iom_dom_close
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_dom__read_var_id(td_mpp, id_varid, td_dom) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function read variable value in opened mpp files,
   !> given variable id and domain strcuture.
   !>
   !> @details
   !> Optionally start indices and number of point to be read could be specify.
   !> as well as East West ovelap of the global domain.
   !>
   !> @author J.Paul
   !> @date October, 2014 - Initial Version
   !>
   !> @param[in] td_mpp    mpp structure
   !> @param[in] id_varid  variable id
   !> @param[in] td_dom    domain structure
   !> @return  variable structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP) , INTENT(IN) :: td_mpp
      INTEGER(i4), INTENT(IN) :: id_varid
      TYPE(TDOM) , INTENT(IN) :: td_dom

      ! function
      TYPE(TVAR)              :: tf_var

      ! local variable
      INTEGER(i4), DIMENSION(1) :: il_ind
      !----------------------------------------------------------------
      ! check if mpp exist
      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN

         CALL logger_error(" IOM DOM READ VAR: domain decomposition "//&
            &              "not define in mpp strcuture "//TRIM(td_mpp%c_name))

      ELSE

         IF( ANY(td_mpp%t_proc(:)%i_id /= 0) )THEN
            ! look for variable id
            il_ind(:)=MINLOC( td_mpp%t_proc(1)%t_var(:)%i_id, &
            &           mask=(td_mpp%t_proc(1)%t_var(:)%i_id==id_varid))
            IF( il_ind(1) /= 0 )THEN

               tf_var=var_copy(td_mpp%t_proc(1)%t_var(il_ind(1)))

               !!! read variable value
               CALL iom_dom__read_var_value(td_mpp, tf_var, td_dom)

            ELSE
               CALL logger_error( &
                  &  " IOM DOM READ VAR: there is no variable with id "//&
                  &  TRIM(fct_str(id_varid))//" in processor/file "//&
                  &  TRIM(td_mpp%t_proc(1)%c_name))
            ENDIF
         ELSE
            CALL logger_error(" IOM DOM READ VAR: can't read variable, mpp "//&
               &              TRIM(td_mpp%c_name)//" not opened")
         ENDIF

      ENDIF

   END FUNCTION iom_dom__read_var_id
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION iom_dom__read_var_name(td_mpp, cd_name, td_dom) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function read variable value in opened mpp files, 
   !> given variable name or standard name, and domain structure.
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
   !> @date October, 2014 - Initial Version
   !> @date May, 2019
   !> - copy variable struct without array of value, then read array of value.
   !>
   !> @param[in] td_mpp    mpp structure
   !> @param[in] cd_name   variable name
   !> @param[in] td_dom    domain structure
   !> @return  variable structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP),       INTENT(IN) :: td_mpp
      CHARACTER(LEN=*), INTENT(IN) :: cd_name
      TYPE(TDOM)      , INTENT(IN) :: td_dom

      ! function
      TYPE(TVAR)                   :: tf_var

      ! local variable
      INTEGER(i4)       :: il_ind

      !----------------------------------------------------------------
      ! check if mpp exist
      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN

         CALL logger_error( " IOM DOM READ VAR: domain decomposition not define "//&
            &               " in mpp strcuture "//TRIM(td_mpp%c_name))

      ELSE

         il_ind=var_get_index( td_mpp%t_proc(1)%t_var(:), cd_name)
         IF( il_ind /= 0 )THEN

            tf_var=var_copy(td_mpp%t_proc(1)%t_var(il_ind), ld_value=.FALSE.)

            !!! read variable value
            CALL iom_dom__read_var_value( td_mpp, tf_var, td_dom )

         ELSE

            CALL logger_error( &
               &  " IOM DOM READ VAR: there is no variable with "//&
               &  "name or standard name "//TRIM(cd_name)//&
               &  " in processor/file "//TRIM(td_mpp%t_proc(1)%c_name))
         ENDIF

      ENDIF
 
   END FUNCTION iom_dom__read_var_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_dom__read_var_value(td_mpp, td_var, td_dom)
   !-------------------------------------------------------------------
   !> @brief This subroutine read variable value
   !> in an mpp structure, given domain structure.
   !>
   !> @author J.Paul
   !> @date October, 2014 - Initial Version
   !>
   !> @todo
   !> - handle north fold
   !>
   !> @param[in] td_mpp    mpp structure
   !> @param[inout] td_var variable structure
   !> @param[in] td_dom    domain structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP),   INTENT(IN)    :: td_mpp
      TYPE(TVAR),   INTENT(INOUT) :: td_var
      TYPE(TDOM),   INTENT(IN)    :: td_dom

      ! local variable
      INTEGER(i4)                 :: il_status

      TYPE(TATT)                  :: tl_att
      TYPE(TMPP)                  :: tl_mpp
      TYPE(TDOM)                  :: tl_dom

      ! loop indices
      INTEGER(i4)                 :: jk
      !----------------------------------------------------------------

      CALL logger_debug(" IOM DOM READ VAR VALUE: name "//&
      &  TRIM(td_var%c_name)//" "//TRIM(td_var%c_point) )
      CALL logger_debug(" IOM DOM READ VAR VALUE: ndim "//&
      &  TRIM(fct_str(td_var%i_ndim)) )

      ! copy mpp structure
      tl_mpp=mpp_copy(td_mpp)
      ! forced to keep same id
      tl_mpp%t_proc(:)%i_id=td_mpp%t_proc(:)%i_id         

      ! Allocate space to hold variable value in structure
      IF( ASSOCIATED(td_var%d_value) )THEN
         DEALLOCATE(td_var%d_value)   
      ENDIF
      
      ! copy domain structure
      tl_dom=dom_copy(td_dom)
      DO jk=1,ip_maxdim
         IF( .NOT. td_var%t_dim(jk)%l_use ) tl_dom%t_dim(jk)%i_len = 1
      ENDDO

      ! use domain dimension 
      td_var%t_dim(1:2)%i_len=tl_dom%t_dim(1:2)%i_len

      ALLOCATE(td_var%d_value( tl_dom%t_dim(1)%i_len, &
      &                        tl_dom%t_dim(2)%i_len, &
      &                        td_var%t_dim(3)%i_len, &
      &                        td_var%t_dim(4)%i_len),&
      &        stat=il_status)
      IF(il_status /= 0 )THEN

        CALL logger_error( &
         &  " IOM DOM READ VAR VALUE: not enough space to put variable "//&
         &  TRIM(td_var%c_name)//&
         &  " in variable structure")

      ENDIF
      CALL logger_debug("IOM DOM READ VAR VALUE: shape ("//&
      &  TRIM(fct_str(SIZE(td_var%d_value(:,:,:,:),DIM=1)))//","//&
      &  TRIM(fct_str(SIZE(td_var%d_value(:,:,:,:),DIM=2)))//","//&
      &  TRIM(fct_str(SIZE(td_var%d_value(:,:,:,:),DIM=3)))//","//&
      &  TRIM(fct_str(SIZE(td_var%d_value(:,:,:,:),DIM=4)))//")" )         
      ! FillValue by default
      td_var%d_value(:,:,:,:)=td_var%d_fill

      IF( tl_dom%i_perio0 < 3 .OR. &
      &   tl_dom%i_jmax /= tl_dom%t_dim0(2)%i_len )THEN
      ! no north pole

         IF( (tl_dom%i_perio0 == 1 .OR. &
         &    tl_dom%i_perio0 == 4 .OR. &
         &    tl_dom%i_perio0 == 6) .AND. &
         &   tl_dom%i_imin == 1 .AND. &
         &   tl_dom%i_imax == tl_dom%t_dim0(1)%i_len )THEN
         ! east west cyclic

            CALL iom_dom__no_pole_cyclic(tl_mpp, td_var, tl_dom)

         ELSEIF( tl_dom%i_imin <= tl_dom%i_imax )THEN
         ! no east west overlap
            
            CALL iom_dom__no_pole_no_overlap(tl_mpp, td_var, tl_dom)

            ! no more EW overlap in variable
            td_var%i_ew=-1

         ELSEIF( (tl_dom%i_perio0 == 1 .OR. &
         &        tl_dom%i_perio0 == 4 .OR. &
         &        tl_dom%i_perio0 == 6) .AND. &
         &       tl_dom%i_imin > tl_dom%i_imax )THEN
         ! east west overlap

            CALL iom_dom__no_pole_overlap(tl_mpp, td_var, tl_dom)

            ! no more EW overlap in variable
            td_var%i_ew=-1

         ELSE

            CALL logger_fatal(" IOM DOM READ VAR VALUE: invalid domain definition.")

         ENDIF

      ELSE ! tl_dom%i_jmax == tl_dom%t_dim0(2)%i_len
         ! north pole

         CALL logger_error("IOM DOM READ VAR VALUE: "//&
         &                 TRIM(fct_str(tl_dom%i_jmin))//" "//&
         &                 TRIM(fct_str(tl_dom%i_jmax)) )
         CALL logger_fatal("IOM DOM READ VAR VALUE: siren is not able to "//&
         &                 "use north pole now, maybe in the next release")
      !   IF( tl_dom%i_imin < tl_dom%i_imax )THEN
      !   ! no east west overlap

      !      CALL iom_dom__pole_no_overlap(tl_mpp, td_var, tl_dom)

      !   ELSEIF(tl_dom%i_imin == tl_dom%i_imax)THEN
      !   ! east west cyclic

      !      CALL iom_dom__pole_cyclic(tl_mpp, td_var, tl_dom)

      !    ELSE ! tl_dom%i_imin > tl_dom%i_imax
      !    ! east west overlap

      !      CALL iom_dom__pole_overlap(tl_mpp, td_var, tl_dom)

      !   ENDIF
      ENDIF

      ! clean 
      CALL mpp_clean(tl_mpp)
      CALL dom_clean(tl_dom)

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

   END SUBROUTINE iom_dom__read_var_value
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_dom__no_pole_no_overlap(td_mpp, td_var, td_dom)
   !-------------------------------------------------------------------
   !> @brief This subroutine read variable value
   !> in an mpp structure. 
   !> @details
   !> The output domain do not overlap
   !> north fold boundary or east-west boundary.
   !>
   !> @author J.Paul
   !> @date October, 2014 - Initial Version
   !>
   !> @param[in] td_mpp    mpp structure
   !> @param[inout] td_var variable structure
   !> @param[in] td_dom    domain structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP),  INTENT(IN)    :: td_mpp
      TYPE(TVAR),  INTENT(INOUT) :: td_var
      TYPE(TDOM),  INTENT(IN)    :: td_dom

      ! local variable
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_start
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_count

      TYPE(TDOM)                :: tl_dom

      ! loop indices
      !----------------------------------------------------------------
      
      ! copy domain structure
      tl_dom=dom_copy(td_dom)

      ! change dimension length if not use
      IF( .NOT. td_var%t_dim(1)%l_use )THEN 
         tl_dom%i_imin=1 ; tl_dom%i_imax=1
      ENDIF
      IF( .NOT. td_var%t_dim(2)%l_use )THEN 
         tl_dom%i_jmin=1 ; tl_dom%i_jmax=1
      ENDIF

      il_start(:)=(/tl_dom%i_imin,tl_dom%i_jmin,1,1/)

      il_count(:)=(/tl_dom%i_imax-tl_dom%i_imin+1, &
      &             tl_dom%i_jmax-tl_dom%i_jmin+1, &
      &             td_var%t_dim(3)%i_len, &
      &             td_var%t_dim(4)%i_len/)

      td_var=iom_mpp_read_var(td_mpp, TRIM(td_var%c_name), &
      &                       il_start(:), il_count(:) )

      CALL dom_clean(tl_dom)

   END SUBROUTINE iom_dom__no_pole_no_overlap
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_dom__no_pole_cyclic(td_mpp, td_var, td_dom)
   !-------------------------------------------------------------------
   !> @brief This subroutine read cyclic variable value
   !> in an mpp structure. 
   !> @details
   !> The output domain do not overlap north fold boundary.
   !> However it uses cyclic east-west boundary.
   !>
   !> @author J.Paul
   !> @date October, 2014 - Initial Version
   !>
   !> @param[in] td_mpp    mpp structure
   !> @param[inout] td_var variable structure
   !> @param[in] td_dom    domain structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP),   INTENT(IN   ) :: td_mpp
      TYPE(TVAR),   INTENT(INOUT) :: td_var
      TYPE(TDOM),   INTENT(IN   ) :: td_dom

      ! local variable
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_start
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_count

      TYPE(TDOM)                :: tl_dom

      ! loop indices
      !----------------------------------------------------------------

      ! copy domain structure
      tl_dom=dom_copy(td_dom)

      ! cyclic domain
      tl_dom%i_imin=1
      tl_dom%i_imax=tl_dom%t_dim(1)%i_len

      ! change dimension length if not use
      IF( .NOT. td_var%t_dim(1)%l_use )THEN 
         tl_dom%i_imin=1 ; tl_dom%i_imax=1
      ENDIF
      IF( .NOT. td_var%t_dim(2)%l_use )THEN 
         tl_dom%i_jmin=1 ; tl_dom%i_jmax=1
      ENDIF

      il_start(:)=(/tl_dom%i_imin,tl_dom%i_jmin,1,1/)

      il_count(:)=(/tl_dom%i_imax-tl_dom%i_imin+1, &
      &             tl_dom%i_jmax-tl_dom%i_jmin+1, &
      &             td_var%t_dim(3)%i_len, &
      &             td_var%t_dim(4)%i_len /)

      td_var=iom_mpp_read_var(td_mpp, TRIM(td_var%c_name), &
      &                       il_start(:), il_count(:) )

      ! clean
      CALL dom_clean(tl_dom)

   END SUBROUTINE iom_dom__no_pole_cyclic
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE iom_dom__no_pole_overlap(td_mpp, td_var, td_dom)
   !-------------------------------------------------------------------
   !> @brief This subroutine read East West overlap variable value
   !> in an mpp structure.
   !> @details
   !> The output domain do not overlap north fold boundary. 
   !> However it overlaps east-west boundary. 
   !>
   !> @author J.Paul
   !> @date October, 2014 - Initial Version
   !>
   !> @param[in] td_mpp    mpp structure
   !> @param[inout] td_var variable structure
   !> @param[in] td_dom    domain structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP),   INTENT(IN)    :: td_mpp
      TYPE(TVAR),   INTENT(INOUT) :: td_var
      TYPE(TDOM),   INTENT(IN),   OPTIONAL :: td_dom

      ! local variable
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_start
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_count

      INTEGER(i4)               :: il_dim1
      INTEGER(i4)               :: il_dim2

      TYPE(TVAR)                :: tl_var1
      TYPE(TVAR)                :: tl_var2

      TYPE(TDOM)                :: tl_dom

      ! loop indices
      !----------------------------------------------------------------

      ! copy domain structure
      tl_dom=dom_copy(td_dom)

      ! change dimension length if not use
      IF( .NOT. td_var%t_dim(1)%l_use )THEN 
         tl_dom%i_imin=1 ; tl_dom%i_imax=1
      ENDIF
      IF( .NOT. td_var%t_dim(2)%l_use )THEN 
         tl_dom%i_jmin=1 ; tl_dom%i_jmax=1
      ENDIF

      ! get first part of domain
      tl_var1=var_copy(td_var)
      DEALLOCATE(tl_var1%d_value)

      il_start(:)=(/tl_dom%i_imin,tl_dom%i_jmin,1,1/)

      il_dim1 = td_mpp%t_dim(1)%i_len - td_mpp%i_ew - tl_dom%i_imin + 1

      il_count(:)=(/il_dim1, &
      &             tl_dom%i_jmax-tl_dom%i_jmin+1, &
      &             td_var%t_dim(3)%i_len, &
      &             td_var%t_dim(4)%i_len /)

      ! dimension part 1
      tl_var1%t_dim(:)%i_len=il_count(:)

      ALLOCATE(tl_var1%d_value(tl_var1%t_dim(1)%i_len, &
      &                        tl_var1%t_dim(2)%i_len, &
      &                        tl_var1%t_dim(3)%i_len, &
      &                        tl_var1%t_dim(4)%i_len) )

      tl_var1=iom_mpp_read_var(td_mpp, TRIM(td_var%c_name), &
      &                        il_start(:), il_count(:) )      

      IF( td_var%t_dim(jp_I)%l_use )THEN
         ! get second part of domain
         tl_var2=var_copy(td_var)
         DEALLOCATE(tl_var2%d_value)

         il_start(:)=(/1,tl_dom%i_jmin,1,1/)

         il_dim2 = tl_dom%i_imax

         il_count(:)=(/il_dim2, &
         &             tl_dom%i_jmax-tl_dom%i_jmin+1, &
         &             td_var%t_dim(3)%i_len, &
         &             td_var%t_dim(4)%i_len /)

         ! dimension part 2
         tl_var2%t_dim(:)%i_len=il_count(:)

         ALLOCATE(tl_var2%d_value(tl_var2%t_dim(1)%i_len, &
         &                        tl_var2%t_dim(2)%i_len, &
         &                        tl_var2%t_dim(3)%i_len, &
         &                        tl_var2%t_dim(4)%i_len) )

         tl_var2=iom_mpp_read_var(td_mpp, TRIM(td_var%c_name), &
         &                        il_start(:), il_count(:) ) 

         ! concatenate both part
         td_var=var_concat(tl_var1, tl_var2, jp_I)
      
         ! clean
         CALL var_clean(tl_var1)
         CALL var_clean(tl_var2)
      ELSE
         td_var=var_copy(tl_var1)
         ! clean
         CALL var_clean(tl_var1)
      ENDIF

      ! clean
      CALL dom_clean(tl_dom)

   END SUBROUTINE iom_dom__no_pole_overlap
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!   SUBROUTINE iom_dom__pole_no_overlap(td_mpp, td_var, td_dom)
   !-------------------------------------------------------------------
   !> @brief This subroutine read north fold variable value
   !> in an mpp structure. 
   !> @details
   !> The output domain overlaps
   !> north fold boundary. However it do not overlap east-west boundary.
   !>
   !> @author J.Paul
   !> @date October, 2014 - Initial Version
   !>
   !> @param[in] td_mpp    mpp structure
   !> @param[inout] td_var variable structure
   !> @param[in] td_dom    domain structure
   !-------------------------------------------------------------------
!
!      IMPLICIT NONE
!
!      ! Argument      
!      TYPE(TMPP),   INTENT(IN)    :: td_mpp
!      TYPE(TVAR),   INTENT(INOUT) :: td_var
!      TYPE(TDOM),   INTENT(IN),   OPTIONAL :: td_dom
!
!      ! local variable
!
!      ! loop indices
!      !----------------------------------------------------------------
!
!   END SUBROUTINE iom_dom__pole_no_overlap
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!   SUBROUTINE iom_dom__pole_cyclic(td_mpp, td_var, td_dom)
   !-------------------------------------------------------------------
   !> @brief This subroutine read semi global variable value
   !> in an mpp structure. 
   !> @details
   !> The output domain overlaps north fold boundary.
   !> and uses cyclic east-west boundary.
   !>
   !> @author J.Paul
   !> @date October, 2014 - Initial Version
   !>
   !> @param[in] td_mpp    mpp structure
   !> @param[inout] td_var variable structure
   !> @param[in] td_dom    domain structure
   !> @return variable structure completed 
   !-------------------------------------------------------------------
!
!      IMPLICIT NONE
!
!      ! Argument      
!      TYPE(TMPP),   INTENT(IN)    :: td_mpp
!      TYPE(TVAR),   INTENT(INOUT) :: td_var
!      TYPE(TDOM),   INTENT(IN),   OPTIONAL :: td_dom
!
!      ! local variable
!
!      ! loop indices
!      !----------------------------------------------------------------
!
!   END SUBROUTINE iom_dom__pole_cyclic
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!   SUBROUTINE iom_dom__pole_overlap(td_mpp, td_var, td_dom)
   !-------------------------------------------------------------------
   !> @brief This subroutine read north fold East West overlap variable value
   !> in an mpp structure. 
   !> @details
   !> The output domain overlaps north fold boundary. 
   !> and east-west boundary. 
   !>
   !> @author J.Paul
   !> @date October, 2014 - Initial Version
   !>
   !> @param[in] td_mpp    mpp structure
   !> @param[inout] td_var variable structure
   !> @param[in] td_dom    domain structure
   !> @return variable structure completed 
   !-------------------------------------------------------------------
!
!      IMPLICIT NONE
!
!      ! Argument      
!      TYPE(TMPP),   INTENT(IN)    :: td_mpp
!      TYPE(TVAR),   INTENT(INOUT) :: td_var
!      TYPE(TDOM),   INTENT(IN),   OPTIONAL :: td_dom
!
!      ! local variable
!
!      ! loop indices
!      !----------------------------------------------------------------
!
!   END SUBROUTINE iom_dom__pole_overlap
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE iom_dom
