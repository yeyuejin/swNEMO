!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief 
!> This module manage interpolation on regular grid.
!>
!> @details Interpolation method to be used is specify inside variable 
!>    strcuture, as array of string character.<br/> 
!>    - td_var\%c_interp(1) string character is the interpolation name choose between:
!>       - 'nearest'
!>       - 'cubic  '
!>       - 'linear '
!>    - td_var\%c_interp(2) string character is an operation to be used 
!>    on interpolated value.<br/>
!>          operation have to be mulitplication '*' or division '/'.<br/>
!>          coefficient have to be refinement factor following i-direction 'rhoi', 
!>          j-direction 'rhoj', or k-direction 'rhok'.<br/>
!>
!>          Examples: '*rhoi', '/rhoj'.
!> 
!>    @note Those informations are read from namelist or variable configuration file (default).<br/>
!>    Interplation method could be specify for each variable in namelist _namvar_,
!>    defining string character _cn\_varinfo_.<br/>
!>    Example:
!>       - cn_varinfo='varname1:int=cubic/rhoi', 'varname2:int=linear' 
!>
!>    to create mixed grid (with coarse grid point needed to compute
!> interpolation):<br/>
!> @code
!>    CALL interp_create_mixed_grid( td_var, td_mix [,id_rho] )
!> @endcode
!>       - td_var is coarse grid variable (should be extrapolated)
!>       - td_mix is mixed grid variable structure [output]
!>       - id_rho is array of refinment factor [optional]
!>
!>    to detected point to be interpolated:<br/>
!> @code
!>    il_detect(:,:,:)=interp_detect( td_mix [,id_rho] )
!> @endcode
!>       - il_detect(:,:,:) is 3D array of detected point to be interpolated
!>       - td_mix is mixed grid variable
!>       - id_rho is array of refinement factor [optional]
!>
!>    to interpolate variable value:<br/>
!> @code
!>    CALL  interp_fill_value( td_var [,id_rho] [,id_offset] )
!> @endcode
!>       - td_var is variable structure
!>       - id_rho is array of refinement factor [optional]
!>       - id_offset is array of offset between fine and coarse grid [optional]
!>
!>    to clean mixed grid (remove points added on mixed grid to compute interpolation):<br/>
!> @code
!>    CALL interp_clean_mixed_grid( td_mix, td_var, id_rho )
!> @endcode
!>       - td_mix is mixed grid variable structure
!>       - td_var is variable structure [output]
!>       - id_rho is array of refinement factor [optional]
!>       - id_offset is array of offset between fine and coarse grid [optional]
!>
!> @note It use to work on ORCA grid, as we work only with grid indices.
!>
!> @warning due to the use of second derivative when using cubic interpolation
!> you should add at least 2 extrabands.
!>
!> @author
!> J.Paul
!> 
!> @date November, 2013 - Initial Version
!> @date September, 2014
!> - add header
!> - use interpolation method modules
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE interp

   USE netcdf                          ! nf90 library
   USE global                          ! global variable
   USE kind                            ! F90 kind parameter
   USE logger                          ! log file manager
   USE fct                             ! basic useful function
   USE date                            ! date manager
   USE att                             ! attribute manager
   USE dim                             ! dimension manager
   USE var                             ! variable manager
   USE grid                            ! grid manager
   USE extrap                          ! extrapolation manager
   USE interp_cubic                    !   cubic interpolation manager
   USE interp_linear                   !  linear interpolation manager
   USE interp_nearest                  ! nearest interpolation manager

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! type and variable

   ! function and subroutine
   PUBLIC :: interp_detect             !< detected point to be interpolated
   PUBLIC :: interp_fill_value         !< interpolate value
   PUBLIC :: interp_create_mixed_grid  !< create mixed grid
   PUBLIC :: interp_clean_mixed_grid   !< clean mixed grid

   PRIVATE :: interp__detect              ! detected point to be interpolated
   PRIVATE :: interp__detect_wrapper      ! detected point to be interpolated
   PRIVATE :: interp__fill_value_wrapper  ! interpolate value over detectected point
   PRIVATE :: interp__fill_value          ! interpolate value over detectected point
   PRIVATE :: interp__clean_even_grid     ! clean even mixed grid
   PRIVATE :: interp__check_method        ! check if interpolation method available
   
   TYPE TINTERP
      CHARACTER(LEN=lc) :: c_name   = '' !< interpolation method name
      CHARACTER(LEN=lc) :: c_factor = '' !< interpolation factor
      CHARACTER(LEN=lc) :: c_divisor= '' !< interpolation divisor
   END TYPE TINTERP

   INTERFACE interp_detect
      MODULE PROCEDURE interp__detect_wrapper 
   END INTERFACE interp_detect

   INTERFACE interp_fill_value
      MODULE PROCEDURE interp__fill_value_wrapper 
   END INTERFACE interp_fill_value

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION interp__check_method(cd_method) &
         & RESULT (lf_avail)
   !-------------------------------------------------------------------
   !> @brief
   !> This function check if interpolation method is available.
   !> 
   !> @details 
   !> check if name of interpolation method is present in global list of string
   !> character cp_interp_list (see global.f90).
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[in] cd_method interpolation method
   !> @return true if interpolation method is available
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=lc) :: cd_method

      ! function
      LOGICAL           :: lf_avail

      ! local variable
      CHARACTER(LEN=lc) :: cl_interp
      CHARACTER(LEN=lc) :: cl_method

      ! loop indices
      INTEGER(I4) :: ji
      !----------------------------------------------------------------

      cl_method=fct_lower(cd_method)

      lf_avail=.FALSE.
      DO ji=1,ip_ninterp
         cl_interp=fct_lower(cp_interp_list(ji))
         IF( TRIM(cl_interp) == TRIM(cl_method) )THEN
            lf_avail=.TRUE.
            EXIT
         ENDIF
      ENDDO

   END FUNCTION interp__check_method
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION interp__detect_wrapper(td_mix, id_rho) &
         & RESULT (if_detect)
   !-------------------------------------------------------------------
   !> @brief
   !> This function detected point to be interpolated.
   !> 
   !> @details 
   !> Actually it checks, the number of dimension used for this variable
   !> and launch interp__detect which detected point to be interpolated.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_mix mixed grid variable (to interpolate)
   !> @param[in] id_rho array of refinement factor
   !> @return 3D array of detected point to be interpolated 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) , INTENT(IN) :: td_mix
      INTEGER(I4), DIMENSION(:), INTENT(IN), OPTIONAL :: id_rho

      ! function
      INTEGER(i4), DIMENSION(td_mix%t_dim(1)%i_len,&
         &                   td_mix%t_dim(2)%i_len,&
         &                   td_mix%t_dim(3)%i_len )  :: if_detect

      ! local variable
      ! loop indices
      !----------------------------------------------------------------

      IF( .NOT. ANY(td_mix%t_dim(1:3)%l_use) )THEN
         ! no dimension I-J-K used
         CALL logger_debug(" INTERP DETECT: nothing done for variable"//&
            &              TRIM(td_mix%c_name) )
         
         if_detect(:,:,:)=0

      ELSE IF( ALL(td_mix%t_dim(1:3)%l_use) )THEN
         
         ! detect point to be interpolated on I-J-K
         CALL logger_debug(" INTERP DETECT: detect point "//&
            &              TRIM(td_mix%c_point)//" for variable "//&
            &              TRIM(td_mix%c_name) )
         
         if_detect(:,:,:)=interp__detect( td_mix, id_rho(:) )

      ELSE IF( ALL(td_mix%t_dim(1:2)%l_use) )THEN

         ! detect point to be interpolated on I-J
         CALL logger_debug(" INTERP DETECT: detect point "//&
            &              TRIM(td_mix%c_point)//" for variable "//&
            &              TRIM(td_mix%c_name) )
         
         if_detect(:,:,1:1)=interp__detect( td_mix, id_rho(:))

      ELSE IF( td_mix%t_dim(3)%l_use )THEN
         
         ! detect point to be interpolated on K
         CALL logger_debug(" INTERP DETECT: detect vertical point "//&
            &              " for variable "//TRIM(td_mix%c_name) )
         
         if_detect(1:1,1:1,:)=interp__detect( td_mix, id_rho(:) )

      ENDIF              

   END FUNCTION interp__detect_wrapper
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION interp__detect(td_mix, id_rho) &
         & RESULT (if_detect)
   !-------------------------------------------------------------------
   !> @brief
   !> This function detected point to be interpolated.
   !> 
   !> @details 
   !> A special case is done for even refinement on ARAKAWA-C grid.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[in] td_mix mixed grid variable (to interpolate)
   !> @param[in] id_rho array of refinement factor
   !> @return 3D array of detected point to be interpolated 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) , INTENT(IN) :: td_mix
      INTEGER(I4), DIMENSION(:), INTENT(IN), OPTIONAL :: id_rho

      ! function
      INTEGER(i4), DIMENSION(td_mix%t_dim(1)%i_len,&
         &                   td_mix%t_dim(2)%i_len,&
         &                   td_mix%t_dim(3)%i_len )  :: if_detect

      ! local variable
      INTEGER(I4), DIMENSION(:), ALLOCATABLE :: il_rho

      INTEGER(I4) :: il_xextra
      INTEGER(I4) :: il_yextra
      INTEGER(I4) :: il_zextra

      INTEGER(i4), DIMENSION(3) :: il_dim

      LOGICAL    , DIMENSION(3) :: ll_even

      ! loop indices
      INTEGER(I4) :: ji
      INTEGER(I4) :: jj
      INTEGER(I4) :: jk
      !----------------------------------------------------------------
      ALLOCATE( il_rho(ip_maxdim) )
      il_rho(:)=1
      IF( PRESENT(id_rho) ) il_rho(1:SIZE(id_rho(:)))=id_rho(:)

      ! special case for even refinement on ARAKAWA-C grid
      ll_even(:)=.FALSE.
      IF( MOD(il_rho(jp_I),2) == 0 ) ll_even(1)=.TRUE.
      IF( MOD(il_rho(jp_J),2) == 0 ) ll_even(2)=.TRUE.
      IF( MOD(il_rho(jp_K),2) == 0 ) ll_even(3)=.TRUE.         

      SELECT CASE(TRIM(td_mix%c_point))
         CASE('U')
            ll_even(1)=.FALSE.
         CASE('V')
            ll_even(2)=.FALSE.
         CASE('F')
            ll_even(:)=.FALSE.
      END SELECT

      IF( ll_even(1) ) il_rho(jp_I)=il_rho(jp_I)+1
      IF( ll_even(2) ) il_rho(jp_J)=il_rho(jp_J)+1
      IF( ll_even(3) ) il_rho(jp_K)=il_rho(jp_K)+1

      ! special case for cubic interpolation
      il_xextra=0
      il_yextra=0
      il_zextra=0
      SELECT CASE(TRIM(td_mix%c_interp(1)))
      CASE('cubic')
         ! those points can not be compute cause cubic interpolation
         ! need second derivative.
         IF( il_rho(jp_I) /= 1 ) il_xextra=3*il_rho(jp_I)
         IF( il_rho(jp_J) /= 1 ) il_yextra=3*il_rho(jp_J)
         IF( il_rho(jp_K) /= 1 ) il_zextra=3*il_rho(jp_K)
      END SELECT

      il_dim(:)=td_mix%t_dim(1:3)%i_len

      ! init
      if_detect(:,:,:)=1

      ! do not compute coarse grid point
      if_detect(1:td_mix%t_dim(1)%i_len:il_rho(jp_I), &
         &      1:td_mix%t_dim(2)%i_len:il_rho(jp_J), &
         &      1:td_mix%t_dim(3)%i_len:il_rho(jp_K)  ) = 0

      ! do not compute point near fill value
      DO jk=1,il_dim(3),il_rho(jp_K)
         DO jj=1,il_dim(2),il_rho(jp_J)
            DO ji=1,il_dim(1),il_rho(jp_I)

               IF( td_mix%d_value(ji,jj,jk,1) == td_mix%d_fill )THEN

                  ! i-direction
                  if_detect(MAX(1,ji-il_xextra):MIN(ji+il_xextra,il_dim(1)),&
                     &      MAX(1,jj-(il_rho(jp_J)-1)):MIN(jj+(il_rho(jp_J)-1),il_dim(2)),&
                     &      MAX(1,jk-(il_rho(jp_K)-1)):MIN(jk+(il_rho(jp_K)-1),il_dim(3)) )=0 
                  ! j-direction
                  if_detect(MAX(1,ji-(il_rho(jp_I)-1)):MIN(ji+(il_rho(jp_I)-1),il_dim(1)),&
                     &      MAX(1,jj-il_yextra):MIN(jj+il_yextra,il_dim(2)),&
                     &      MAX(1,jk-(il_rho(jp_K)-1)):MIN(jk+(il_rho(jp_K)-1),il_dim(3)) )=0
                  ! k-direction
                  if_detect(MAX(1,ji-(il_rho(jp_I)-1)):MIN(ji+(il_rho(jp_I)-1),il_dim(1)),&
                     &      MAX(1,jj-(il_rho(jp_J)-1)):MIN(jj+(il_rho(jp_J)-1),il_dim(2)),&
                     &      MAX(1,jk-il_zextra):MIN(jk+il_zextra,il_dim(3)) )=0         

               ENDIF

            ENDDO
         ENDDO
      ENDDO

      DEALLOCATE( il_rho )

   END FUNCTION interp__detect
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp_create_mixed_grid(td_var, td_mix, id_rho)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine create mixed grid.
   !> 
   !> @details 
   !> Created grid is fine resolution grid.
   !> First and last point are coasre grid point. 
   !>
   !> A special case is done for even refinement on ARAKAWA-C grid.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_var    coarse grid variable (should be extrapolated)
   !> @param[out] td_mix   mixed grid variable
   !> @param[in] id_rho    array of refinment factor (default 1) 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) ,               INTENT(IN   ) :: td_var 
      TYPE(TVAR) ,               INTENT(  OUT) :: td_mix 
      INTEGER(I4), DIMENSION(:), INTENT(IN   ), OPTIONAL :: id_rho

      ! local variable
      INTEGER(I4), DIMENSION(:), ALLOCATABLE :: il_rho

      INTEGER(i4) :: il_xextra
      INTEGER(i4) :: il_yextra
      INTEGER(i4) :: il_zextra

      LOGICAL, DIMENSION(3) :: ll_even

      ! loop indices
      !----------------------------------------------------------------
      ALLOCATE(il_rho(ip_maxdim))
      il_rho(:)=1
      IF( PRESENT(id_rho) ) il_rho(1:SIZE(id_rho(:)))=id_rho(:)

      ! special case for even refinement on ARAKAWA-C grid
      ll_even(:)=.FALSE.
      IF( MOD(il_rho(jp_I),2) == 0 ) ll_even(1)=.TRUE.
      IF( MOD(il_rho(jp_J),2) == 0 ) ll_even(2)=.TRUE.
      IF( MOD(il_rho(jp_K),2) == 0 ) ll_even(3)=.TRUE.         

      SELECT CASE(TRIM(td_var%c_point))
         CASE('U')
            ll_even(1)=.FALSE.
         CASE('V')
            ll_even(2)=.FALSE.
         CASE('F')
            ll_even(:)=.FALSE.
      END SELECT

      IF( ll_even(1) ) il_rho(jp_I)=il_rho(jp_I)+1
      IF( ll_even(2) ) il_rho(jp_J)=il_rho(jp_J)+1
      IF( ll_even(3) ) il_rho(jp_K)=il_rho(jp_K)+1

      ! copy variable
      td_mix=var_copy(td_var)

      ! compute new dimension length
      il_xextra=il_rho(jp_I)-1
      td_mix%t_dim(1)%i_len=td_mix%t_dim(1)%i_len*il_rho(jp_I)-il_xextra

      il_yextra=il_rho(jp_J)-1
      td_mix%t_dim(2)%i_len=td_mix%t_dim(2)%i_len*il_rho(jp_J)-il_yextra

      il_zextra=il_rho(jp_K)-1
      td_mix%t_dim(3)%i_len=td_mix%t_dim(3)%i_len*il_rho(jp_K)-il_zextra

      IF( ASSOCIATED(td_mix%d_value) ) DEALLOCATE( td_mix%d_value )
      ALLOCATE( td_mix%d_value( td_mix%t_dim(1)%i_len, &
      &                         td_mix%t_dim(2)%i_len, &
      &                         td_mix%t_dim(3)%i_len, &
      &                         td_mix%t_dim(4)%i_len) )

      ! initialise to FillValue
      td_mix%d_value(:,:,:,:)=td_mix%d_fill

      ! quid qd coord ou bathy fourni par user ?? (offset ??)
      td_mix%d_value(1:td_mix%t_dim(1)%i_len:il_rho(jp_I), &
      &              1:td_mix%t_dim(2)%i_len:il_rho(jp_J), &
      &              1:td_mix%t_dim(3)%i_len:il_rho(jp_K), :) = &
      &     td_var%d_value(:,:,:,:)

      DEALLOCATE(il_rho)

   END SUBROUTINE interp_create_mixed_grid
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp__clean_even_grid(td_mix, id_rho)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine remove points added to mixed grid to compute
   !> interpolation in the special case of even refinement on ARAKAWA-C grid.
   !> 
   !> @details 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_mix mixed grid variable
   !> @param[in] id_rho    array of refinment factor 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) ,               INTENT(INOUT) :: td_mix 
      INTEGER(I4), DIMENSION(:), INTENT(IN   ), OPTIONAL :: id_rho

      ! local variable
      INTEGER(I4), DIMENSION(:), ALLOCATABLE :: il_rho

      INTEGER(i4) :: il_xextra
      INTEGER(i4) :: il_yextra
      INTEGER(i4) :: il_zextra

      LOGICAL, DIMENSION(3) :: ll_even

      LOGICAL, DIMENSION(:,:,:,:), ALLOCATABLE :: ll_mask

      REAL(dp), DIMENSION(:)     , ALLOCATABLE :: dl_vect

      TYPE(TVAR) :: tl_mix

      ! loop indices
      !----------------------------------------------------------------
      ALLOCATE(il_rho(ip_maxdim))
      il_rho(:)=1
      IF( PRESENT(id_rho) ) il_rho(:)=id_rho(:)

      ! special case for even refinement on ARAKAWA-C grid
      ll_even(:)=.FALSE.
      IF( MOD(il_rho(jp_I),2) == 0 ) ll_even(1)=.TRUE.
      IF( MOD(il_rho(jp_J),2) == 0 ) ll_even(2)=.TRUE.
      IF( MOD(il_rho(jp_K),2) == 0 ) ll_even(3)=.TRUE.         

      SELECT CASE(TRIM(td_mix%c_point))
         CASE('U')
            ll_even(1)=.FALSE.
         CASE('V')
            ll_even(2)=.FALSE.
         CASE('F')
            ll_even(:)=.FALSE.
      END SELECT

      ! remove some point only if refinement in some direction is even
      IF( ANY(ll_even(:)) )THEN

         ! copy variable
         tl_mix=var_copy(td_mix)

         ALLOCATE( ll_mask( tl_mix%t_dim(1)%i_len, &
         &                  tl_mix%t_dim(2)%i_len, &
         &                  tl_mix%t_dim(3)%i_len, &
         &                  tl_mix%t_dim(4)%i_len) )

         ll_mask(:,:,:,:)=.TRUE.

         IF( tl_mix%t_dim(1)%l_use .AND. ll_even(1) )THEN

            il_rho(jp_I)=il_rho(jp_I)+1

            ! locate wrong point on mixed grid
            ll_mask(1:td_mix%t_dim(1)%i_len:il_rho(jp_I),:,:,:)=.FALSE.

            ! compute coasre grid dimension length
            il_xextra=il_rho(jp_I)-1
            td_mix%t_dim(1)%i_len=(tl_mix%t_dim(1)%i_len+il_xextra)/il_rho(jp_I)

            il_rho(jp_I)=il_rho(jp_I)-1
            ! compute right fine grid dimension length
            td_mix%t_dim(1)%i_len=td_mix%t_dim(1)%i_len*il_rho(jp_I)-il_xextra

         ENDIF

         IF( tl_mix%t_dim(2)%l_use .AND. ll_even(2) )THEN

            il_rho(jp_J)=il_rho(jp_J)+1

            ! locate wrong point on mixed grid
            ll_mask(:,1:tl_mix%t_dim(2)%i_len:il_rho(jp_J),:,:)=.FALSE.

            ! compute coasre grid dimension length
            il_yextra=il_rho(jp_J)-1
            td_mix%t_dim(2)%i_len=(tl_mix%t_dim(2)%i_len+il_yextra)/il_rho(jp_J)

            il_rho(jp_J)=il_rho(jp_J)-1
            ! compute right fine grid dimension length
            td_mix%t_dim(2)%i_len=td_mix%t_dim(2)%i_len*il_rho(jp_J)-il_yextra
            
         ENDIF

         IF( tl_mix%t_dim(3)%l_use .AND. ll_even(3) )THEN

            il_rho(jp_K)=il_rho(jp_K)+1

            ! locate wrong point on mixed grid
            ll_mask(:,:,1:tl_mix%t_dim(3)%i_len:il_rho(jp_K),:)=.FALSE.

            ! compute coasre grid dimension length
            il_zextra=il_rho(jp_K)-1
            td_mix%t_dim(3)%i_len=(tl_mix%t_dim(3)%i_len+il_zextra)/il_rho(jp_K)

            il_rho(jp_K)=il_rho(jp_K)-1
            ! compute right fine grid dimension length
            td_mix%t_dim(3)%i_len=td_mix%t_dim(3)%i_len*il_rho(jp_K)-il_zextra

         ENDIF      

         IF( ASSOCIATED(td_mix%d_value) ) DEALLOCATE( td_mix%d_value )
         ALLOCATE( td_mix%d_value( td_mix%t_dim(1)%i_len, &
         &                         td_mix%t_dim(2)%i_len, &
         &                         td_mix%t_dim(3)%i_len, &
         &                         td_mix%t_dim(4)%i_len) )

         ! initialise to FillValue
         td_mix%d_value(:,:,:,:)=td_mix%d_fill

         IF( COUNT(ll_mask(:,:,:,:)) /= SIZE(td_mix%d_value(:,:,:,:)) )THEN

            CALL logger_error("INTERP CLEAN EVEN GRID: output value size "//&
            &  " and mask count differ ")

         ELSE

            ALLOCATE( dl_vect(COUNT(ll_mask(:,:,:,:))) )

            dl_vect(:)= PACK( tl_mix%d_value(:,:,:,:), &
            &                 MASK=ll_mask(:,:,:,:)     )

            td_mix%d_value(:,:,:,:)=RESHAPE( dl_vect(:), &
            &                                SHAPE=(/td_mix%t_dim(1)%i_len, &
            &                                        td_mix%t_dim(2)%i_len, &
            &                                        td_mix%t_dim(3)%i_len, &
            &                                        td_mix%t_dim(4)%i_len/) )

            DEALLOCATE( dl_vect )

         ENDIF

         DEALLOCATE( ll_mask )

         ! clean
         CALL var_clean(tl_mix)

      ENDIF

      ! clean
      DEALLOCATE(il_rho)

   END SUBROUTINE interp__clean_even_grid
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp_clean_mixed_grid(td_mix, td_var, &
         &                            id_rho, id_offset)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine remove points added on mixed grid 
   !> to compute interpolation. And save interpolated value over domain. 
   !> 
   !> @details 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014
   !> - use offset to save useful domain
   !>
   !> @param[in] td_mix    mixed grid variable structure
   !> @param[out] td_var   variable structure
   !> @param[in] id_rho    array of refinement factor (default 1)
   !> @param[in] id_offset 2D array of offset between fine and coarse grid
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR)                 , INTENT(IN   ) :: td_mix
      TYPE(TVAR)                 , INTENT(  OUT) :: td_var
      INTEGER(I4), DIMENSION(:)  , INTENT(IN   ) :: id_rho
      INTEGER(I4), DIMENSION(2,2), INTENT(IN   ) :: id_offset

      ! local variable
      INTEGER(i4) :: il_imin0
      INTEGER(i4) :: il_imax0
      INTEGER(i4) :: il_jmin0
      INTEGER(i4) :: il_jmax0

      INTEGER(i4) :: il_imin1
      INTEGER(i4) :: il_jmin1
      INTEGER(i4) :: il_imax1
      INTEGER(i4) :: il_jmax1

      REAL(dp), DIMENSION(:,:,:,:) , ALLOCATABLE :: dl_value

      TYPE(TVAR) :: tl_mix

      ! loop indices
      !----------------------------------------------------------------
      ! copy mixed variable in temporary structure
      tl_mix=var_copy(td_mix)

      ! remove useless points over mixed grid for even refinement
      CALL interp__clean_even_grid(tl_mix, id_rho(:))

      ! copy cleaned mixed variable
      td_var=var_copy(tl_mix)

      ! delete array of value
      CALL var_del_value(td_var)

      ! compute domain indices in i-direction
      il_imin0=1  ; il_imax0=td_var%t_dim(1)%i_len
 
      IF( td_var%t_dim(1)%l_use )THEN
         il_imin1=il_imin0+id_offset(Jp_I,1)
         il_imax1=il_imax0-id_offset(Jp_I,2)
      ELSE

         il_imin1=il_imin0
         il_imax1=il_imax0

      ENDIF

      ! compute domain indices in j-direction
      il_jmin0=1  ; il_jmax0=td_var%t_dim(2)%i_len
 
      IF( td_var%t_dim(2)%l_use )THEN
         il_jmin1=il_jmin0+id_offset(Jp_J,1)
         il_jmax1=il_jmax0-id_offset(Jp_J,2)
      ELSE

         il_jmin1=il_jmin0
         il_jmax1=il_jmax0
 
      ENDIF

      ! compute new dimension
      td_var%t_dim(1)%i_len=il_imax1-il_imin1+1
      td_var%t_dim(2)%i_len=il_jmax1-il_jmin1+1
 
      ALLOCATE(dl_value(td_var%t_dim(1)%i_len, &
      &                 td_var%t_dim(2)%i_len, &
      &                 td_var%t_dim(3)%i_len, &
      &                 td_var%t_dim(4)%i_len) )

      dl_value( 1:td_var%t_dim(1)%i_len, &
      &         1:td_var%t_dim(2)%i_len, &
      &         :,:) = tl_mix%d_value( il_imin1:il_imax1, &
      &                                il_jmin1:il_jmax1, &
      &                                     :, : )

      ! add variable value
      CALL var_add_value(td_var,dl_value(:,:,:,:))

      DEALLOCATE(dl_value)
 
      ! clean
      CALL var_clean(tl_mix)

   END SUBROUTINE interp_clean_mixed_grid
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp__fill_value_wrapper(td_var, & 
         &                               id_rho, &
         &                               id_offset)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine interpolate variable value.
   !> 
   !> @details 
   !> Actually it checks, the number of dimension used for this variable
   !> and launch interp__fill_value.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] id_rho    array of refinement factor
   !> @param[in] id_offset 2D array of offset between fine and coarse grid
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) ,                 INTENT(INOUT) :: td_var
      INTEGER(I4), DIMENSION(:)  , INTENT(IN   ), OPTIONAL :: id_rho
      INTEGER(I4), DIMENSION(:,:), INTENT(IN   ), OPTIONAL :: id_offset

      ! local variable
      CHARACTER(LEN=lc)                            :: cl_method
      INTEGER(i4)      , DIMENSION(:), ALLOCATABLE :: il_rho
      INTEGER(i4)      , DIMENSION(2,2)            :: il_offset

      ! loop indices
      !----------------------------------------------------------------

      ALLOCATE( il_rho(ip_maxdim) )
      il_rho(:)=1
      IF( PRESENT(id_rho) ) il_rho(1:SIZE(id_rho(:)))=id_rho(:)
      IF( ANY(il_rho(:) < 0) )THEN
         CALL logger_error("INTERP FILL VALUE: invalid "//&
         &  " refinement factor ")
      ENDIF

      il_offset(:,:)=0
      IF( PRESENT(id_offset) )THEN
         IF( ANY(SHAPE(id_offset(:,:)) /= (/2,2/)) )THEN
            CALL logger_error("INTERP FILL VALUE: invalid array of offset")
         ELSE
            il_offset(:,:)=id_offset(:,:)
         ENDIF
      ENDIF

      IF( (il_rho(jp_I) /= 1 .AND. td_var%t_dim(1)%l_use) .OR. &
      &   (il_rho(jp_J) /= 1 .AND. td_var%t_dim(2)%l_use) .OR. &
      &   (il_rho(jp_K) /= 1 .AND. td_var%t_dim(3)%l_use) )THEN

         SELECT CASE(TRIM(td_var%c_interp(1)))
            CASE('cubic','linear','nearest')
               cl_method=TRIM(td_var%c_interp(1))
            CASE DEFAULT
               CALL logger_warn("INTERP FILL VALUE: interpolation method unknown."//&
               &  " use linear interpolation")
               cl_method='linear'
               ! update variable structure value
               td_var%c_interp(1)='linear'
         END SELECT

         CALL logger_info("INTERP FILL: interpolate "//TRIM(td_var%c_name)//&
         &             " using "//TRIM(cl_method)//" method." )
         CALL logger_info("INTERP FILL: refinement factor "//&
         &                        TRIM(fct_str(il_rho(jp_I)))//&
         &                   " "//TRIM(fct_str(il_rho(jp_J)))//&
         &                   " "//TRIM(fct_str(il_rho(jp_K))) )
 
         CALL interp__fill_value( td_var, cl_method, & 
         &                        il_rho(:), il_offset(:,:) )

         SELECT CASE(TRIM(td_var%c_interp(2)))
         CASE('/rhoi')
            WHERE( td_var%d_value(:,:,:,:) /= td_var%d_fill )
               td_var%d_value(:,:,:,:) = &
               &  td_var%d_value(:,:,:,:) / REAL(il_rho(jp_I),dp)
            END WHERE
         CASE('/rhoj')
            WHERE( td_var%d_value(:,:,:,:) /= td_var%d_fill )
               td_var%d_value(:,:,:,:) = &
               &  td_var%d_value(:,:,:,:) / REAL(il_rho(jp_J),dp)
            END WHERE
         CASE('/rhok')
            WHERE( td_var%d_value(:,:,:,:) /= td_var%d_fill )
               td_var%d_value(:,:,:,:) = &
               &  td_var%d_value(:,:,:,:) / REAL(il_rho(jp_K),dp)
            END WHERE
         CASE('*rhoi')
            WHERE( td_var%d_value(:,:,:,:) /= td_var%d_fill )
               td_var%d_value(:,:,:,:) = &
               &  td_var%d_value(:,:,:,:) * REAL(il_rho(jp_I),dp)
            END WHERE
         CASE('*rhoj')
            WHERE( td_var%d_value(:,:,:,:) /= td_var%d_fill ) 
               td_var%d_value(:,:,:,:) = &
               &  td_var%d_value(:,:,:,:) * REAL(il_rho(jp_J),dp)
            END WHERE
         CASE('*rhok')
            WHERE( td_var%d_value(:,:,:,:) /= td_var%d_fill ) 
               td_var%d_value(:,:,:,:) = &
               &  td_var%d_value(:,:,:,:) * REAL(il_rho(jp_K),dp)
            END WHERE
         CASE DEFAULT
            td_var%c_interp(2)=''
         END SELECT

      ELSE
         td_var%c_interp(:)=''
      ENDIF

      DEALLOCATE(il_rho)

   END SUBROUTINE interp__fill_value_wrapper
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp__fill_value(td_var, cd_method, &
         &                       id_rho, id_offset)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine interpolate value over mixed grid.
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014
   !> - use interpolation method modules
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] cd_method interpolation method 
   !> @param[in] id_rho    array of refinment factor 
   !> @param[in] id_offset 2D array of offset between fine and coarse grid
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR)                      , INTENT(INOUT) :: td_var 
      CHARACTER(LEN=*)                , INTENT(IN   ) :: cd_method
      INTEGER(I4)     , DIMENSION(:)  , INTENT(IN   ) :: id_rho
      INTEGER(I4)     , DIMENSION(2,2), INTENT(IN   ) :: id_offset

      ! local variable
      CHARACTER(LEN=lc)                                :: cl_interp

      INTEGER(I4)    , DIMENSION(:)      , ALLOCATABLE :: il_rho
      INTEGER(i4)    , DIMENSION(:,:,:)  , ALLOCATABLE :: il_detect
 
      REAL(dp)                                         :: dl_min
      REAL(dp)                                         :: dl_max

      LOGICAL        , DIMENSION(3)                    :: ll_even
      LOGICAL                                          :: ll_discont
 
      TYPE(TVAR)                                       :: tl_mix

      TYPE(TATT)                                       :: tl_att
 
      ! loop indices
      !----------------------------------------------------------------

      !1- create mixed grid
      CALL interp_create_mixed_grid( td_var, tl_mix, id_rho(:) )

      ! clean variable structure
      CALL var_clean(td_var)

      !2- detect point to be interpolated
      ALLOCATE( il_detect( tl_mix%t_dim(1)%i_len, &
      &                    tl_mix%t_dim(2)%i_len, &
      &                    tl_mix%t_dim(3)%i_len) )

      il_detect(:,:,:)=0

      il_detect(:,:,:)=interp_detect(tl_mix, id_rho(:) )

      ! add attribute to variable
      cl_interp=fct_concat(tl_mix%c_interp(:))
      tl_att=att_init('interpolation',cl_interp)
      CALL var_move_att(tl_mix, tl_att)

      ! clean
      CALL att_clean(tl_att)

      ! special case for even refinement on ARAKAWA-C grid
      ll_even(:)=.FALSE.
      IF( MOD(id_rho(jp_I),2) == 0 ) ll_even(1)=.TRUE.
      IF( MOD(id_rho(jp_J),2) == 0 ) ll_even(2)=.TRUE.
      IF( MOD(id_rho(jp_K),2) == 0 ) ll_even(3)=.TRUE.         

      SELECT CASE(TRIM(tl_mix%c_point))
         CASE('U')
            ll_even(1)=.FALSE.
         CASE('V')
            ll_even(2)=.FALSE.
         CASE('F')
            ll_even(:)=.FALSE.
      END SELECT

      ALLOCATE(il_rho(ip_maxdim))
      il_rho(:)=id_rho(:)

      IF( ll_even(1) ) il_rho(jp_I)=id_rho(jp_I)+1
      IF( ll_even(2) ) il_rho(jp_J)=id_rho(jp_J)+1
      IF( ll_even(3) ) il_rho(jp_K)=id_rho(jp_K)+1

      ! special case for longitude
      ll_discont=.FALSE.
      IF( TRIM(tl_mix%c_units) == 'degrees_east' )THEN
         dl_min=MINVAL( tl_mix%d_value(:,:,:,:), &
         &              tl_mix%d_value(:,:,:,:)/=tl_mix%d_fill)
         dl_max=MAXVAL( tl_mix%d_value(:,:,:,:), &
         &              tl_mix%d_value(:,:,:,:)/=tl_mix%d_fill)         
         IF( dl_min < -170_dp .AND. dl_max > 170_dp .OR. &
         &   dl_min <   10_dp .AND. dl_max > 350_dp )THEN
            ll_discont=.TRUE.
         ENDIF
      ENDIF

      !3- interpolate
      CALL logger_debug("INTERP 2D: interpolation method "//TRIM(cd_method)//&
      &  " discont "//TRIM(fct_str(ll_discont)) )
      SELECT CASE(TRIM(cd_method))
      CASE('cubic')
         CALL interp_cubic_fill(tl_mix%d_value(:,:,:,:), tl_mix%d_fill, &
           &                    il_detect(:,:,:),                        &
           &                    il_rho(:), ll_even(:), ll_discont )
      CASE('nearest')
         CALL interp_nearest_fill(tl_mix%d_value(:,:,:,:), &
              &                   il_detect(:,:,:),         &
              &                   il_rho(:) )
      CASE DEFAULT ! linear
         CALL interp_linear_fill(tl_mix%d_value(:,:,:,:), tl_mix%d_fill, &
              &                  il_detect(:,:,:),                        &
              &                  il_rho(:), ll_even(:), ll_discont )
      END SELECT         

      IF( ANY(il_detect(:,:,:)==1) )THEN
         CALL logger_warn("INTERP FILL: some points can not be interpolated "//&
         &             "for variable "//TRIM(tl_mix%c_name) )
      ENDIF

      DEALLOCATE(il_detect)

      !4- save useful domain (remove offset)
      CALL interp_clean_mixed_grid( tl_mix, td_var, &
      &                             id_rho(:), id_offset(:,:)  )  

      ! clean variable structure
      DEALLOCATE(il_rho)
      CALL var_clean(tl_mix)

   END SUBROUTINE interp__fill_value
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE interp
