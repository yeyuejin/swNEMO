!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief 
!> This module manage extrapolation.
!>
!> @details
!>    Extrapolation method to be used is specify inside variable 
!>    strcuture, as array of string character.<br/> 
!>    - td_var\%c_extrap(1) string character is the interpolation name choose between:
!>       - 'dist_weight'
!>       - 'min_error'
!>
!>    @note Extrapolation method could be specify for each variable in namelist _namvar_,
!>    defining string character _cn\_varinfo_. By default _dist_weight_.<br/>
!>    Example:
!>       - cn_varinfo='varname1:ext=dist_weight', 'varname2:ext=min_error'
!>
!>    to detect point to be extrapolated:<br/>
!> @code
!>    il_detect(:,:,:)=extrap_detect(td_var)
!> @endcode
!>       - il_detect(:,:,:) is 3D array of point to be extrapolated
!>       - td_var  is coarse grid variable to be extrapolated
!>
!>    to extrapolate variable:<br/>
!> @code
!>    CALL extrap_fill_value( td_var, [id_radius])
!> @endcode
!>       - td_var  is coarse grid variable to be extrapolated
!>       - id_radius is radius of the halo used to compute extrapolation [optional]
!>
!>    to add extraband to the variable (to be extrapolated):<br/>
!> @code
!>    CALL extrap_add_extrabands(td_var, [id_isize,] [id_jsize] )
!> @endcode
!>       - td_var is variable structure
!>       - id_isize : i-direction size of extra bands [optional]
!>       - id_jsize : j-direction size of extra bands [optional]
!>
!>    to delete extraband of a variable:<br/>
!> @code
!>    CALL extrap_del_extrabands(td_var, [id_isize,] [id_jsize] )
!> @endcode
!>       - td_var is variable structure
!>       - id_isize : i-direction size of extra bands [optional]
!>       - id_jsize : j-direction size of extra bands [optional]
!>
!> @warning _FillValue must not be zero (use var_chg_FillValue())
!>
!> @author
!> J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date September, 2014
!> - add header
!> @date June, 2015
!> - extrapolate all land points (_FillValue)
!> - move deriv function to math module
!> @date July, 2015
!> - compute extrapolation from north west to south east, 
!> and from south east to north west
!>
!> @todo
!> - create module for each extrapolation method
!> - smooth extrapolated points
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE extrap

   USE netcdf                          ! nf90 library
   USE kind                            ! F90 kind parameter
   USE phycst                          ! physical constant
   USE global                          ! global variable
   USE fct                             ! basic useful function
   USE date                            ! date manager
   USE logger                          ! log file manager
   USE math                            ! mathematical function
   USE att                             ! attribute manager
   USE dim                             ! dimension manager
   USE var                             ! variable manager

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! type and variable
   PRIVATE :: im_minext    !< default minumum number of point to extrapolate
   PRIVATE :: im_mincubic  !< default minumum number of point to extrapolate for cubic interpolation

   ! function and subroutine
   PUBLIC :: extrap_detect         !< detected point to be extrapolated
   PUBLIC :: extrap_fill_value     !< extrapolate value over detected point 
   PUBLIC :: extrap_add_extrabands !< add extraband to the variable (to be extrapolated) 
   PUBLIC :: extrap_del_extrabands !< delete extraband of the variable 

   PRIVATE :: extrap__detect_wrapper      ! detected point to be extrapolated wrapper
   PRIVATE :: extrap__detect              ! detected point to be extrapolated
   PRIVATE :: extrap__fill_value_wrapper  ! extrapolate value over detected point wrapper 
   PRIVATE :: extrap__fill_value          ! extrapolate value over detected point 
   PRIVATE :: extrap__3D                  !
   PRIVATE :: extrap__3D_min_error_coef   !
   PRIVATE :: extrap__3D_min_error_fill   !
   PRIVATE :: extrap__3D_dist_weight_coef !
   PRIVATE :: extrap__3D_dist_weight_fill ! 

   INTEGER(i4), PARAMETER :: im_minext  = 2  !< default minumum number of point to extrapolate
   INTEGER(i4), PARAMETER :: im_mincubic= 4  !< default minumum number of point to extrapolate for cubic interpolation

   INTERFACE extrap_detect
      MODULE PROCEDURE extrap__detect_wrapper     !< detected point to be extrapolated
   END INTERFACE extrap_detect

   INTERFACE extrap_fill_value
      MODULE PROCEDURE extrap__fill_value_wrapper !< detected point to be interpolated
   END INTERFACE extrap_fill_value
   
CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION extrap__detect(td_var0) &
         &  RESULT (if_detect)
   !-------------------------------------------------------------------
   !> @brief
   !> This function detected point to be extrapolated, given variable structure.
   !> 
   !> @details 
   !> optionaly, you could sepcify fine grid level, refinment factor (default 1), 
   !> offset between fine and coarse grid (default compute from refinment factor
   !> as offset=(rho-1)/2), number of point to be extrapolated in each direction
   !> (default im_minext).<br/>
   !>
   !> First coarsening fine grid level, if need be, then select point near 
   !> grid point already inform.
   !>
   !> @note point to be extrapolated are selected using FillValue, 
   !> so to avoid mistake FillValue should not be zero (use var_chg_FillValue)
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - do not use level to select points to be extrapolated
   !>
   !> @param[in] td_var0   coarse grid variable to extrapolate
   !> @return array of point to be extrapolated
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) ,                      INTENT(IN   ) :: td_var0

      ! function
      INTEGER(i4), DIMENSION(td_var0%t_dim(1)%i_len,&
      &                      td_var0%t_dim(2)%i_len,&
      &                      td_var0%t_dim(3)%i_len ) :: if_detect

      ! local variable
      ! loop indices
      INTEGER(i4) :: ji0
      INTEGER(i4) :: jj0
      INTEGER(i4) :: jk0
      !----------------------------------------------------------------

      ! force to extrapolated all points
      if_detect(:,:,:)=1

      ! do not compute grid point already inform
      DO jk0=1,td_var0%t_dim(3)%i_len
         DO jj0=1,td_var0%t_dim(2)%i_len
            DO ji0=1,td_var0%t_dim(1)%i_len
               IF( td_var0%d_value(ji0,jj0,jk0,1) /= td_var0%d_fill )THEN
                  if_detect(ji0,jj0,jk0)=0
               ENDIF
            ENDDO
         ENDDO
      ENDDO

   END FUNCTION extrap__detect
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION extrap__detect_wrapper(td_var) &
         & RESULT (if_detect)
   !-------------------------------------------------------------------
   !> @brief
   !> This function sort variable to be extrapolated, depending on number of
   !> dimentsion, then detected point to be extrapolated.
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - select all land points for extrapolation
   !>
   !> @param[in] td_var    coarse grid variable to extrapolate
   !> @return 3D array of point to be extrapolated
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) ,                 INTENT(IN   ) :: td_var

      ! function
      INTEGER(i4), DIMENSION(td_var%t_dim(1)%i_len,&
      &                      td_var%t_dim(2)%i_len,&
      &                      td_var%t_dim(3)%i_len ) :: if_detect

      ! local variable
      ! loop indices
      !----------------------------------------------------------------
      ! init
      if_detect(:,:,:)=0

      IF( .NOT. ANY(td_var%t_dim(1:3)%l_use) )THEN
         ! no dimension I-J-K used
         CALL logger_debug(" EXTRAP DETECT: nothing done for variable"//&
            &              TRIM(td_var%c_name) )
      ELSE IF( ALL(td_var%t_dim(1:3)%l_use) )THEN
         
         ! detect point to be extrapolated on I-J-K
         CALL logger_debug(" EXTRAP DETECT: detect point "//&
            &              " for variable "//TRIM(td_var%c_name) )
         
      if_detect(:,:,:)=extrap__detect( td_var )

      ELSE IF( ALL(td_var%t_dim(1:2)%l_use) )THEN

         ! detect point to be extrapolated on I-J
         CALL logger_debug(" EXTRAP DETECT: detect horizontal point "//&
            &              " for variable "//TRIM(td_var%c_name) )
         
         if_detect(:,:,1:1)=extrap__detect( td_var )

      ELSE IF( td_var%t_dim(3)%l_use )THEN
         
         ! detect point to be extrapolated on K
         CALL logger_debug(" EXTRAP DETECT: detect vertical point "//&
            &              " for variable "//TRIM(td_var%c_name) )
         
         if_detect(1:1,1:1,:)=extrap__detect( td_var )

      ENDIF              

      CALL logger_debug(" EXTRAP DETECT: "//&
         &  TRIM(fct_str(SUM(if_detect(:,:,:))))//&
         &  " points to be extrapolated" )
      
   END FUNCTION extrap__detect_wrapper
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE extrap__fill_value_wrapper(td_var, id_radius)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine select method to be used for extrapolation.
   !> If need be, increase number of points to be extrapolated.
   !> Finally launch extrap__fill_value.
   !> 
   !> @details
   !> optionaly, you could specify :<br/>
   !>  - refinment factor (default 1)
   !>  - offset between fine and coarse grid (default compute from refinment factor
   !> as offset=(rho-1)/2) 
   !>  - number of point to be extrapolated in each direction (default im_minext)
   !>  - radius of the halo used to compute extrapolation
   !>  - maximum number of iteration
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - select all land points for extrapolation
   !>
   !> @param[inout] td_var    variable structure
   !> @param[in] id_radius    radius of the halo used to compute extrapolation 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) ,                  INTENT(INOUT) :: td_var
      INTEGER(i4),                  INTENT(IN   ), OPTIONAL :: id_radius

      ! local variable
      INTEGER(i4) :: il_radius

      CHARACTER(LEN=lc) :: cl_method

      ! loop indices
      !----------------------------------------------------------------
      IF( .NOT. ASSOCIATED(td_var%d_value) )THEN
         CALL logger_error("EXTRAP FILL VALUE: no value "//&
         &  "associted to variable "//TRIM(td_var%c_name) )
      ELSE

         SELECT CASE(TRIM(td_var%c_extrap(1)))
            CASE('min_error')
               cl_method='min_error'
            CASE DEFAULT
               cl_method='dist_weight'

               !update variable structure
               td_var%c_extrap(1)='dist_weight'
         END SELECT

         ! number of point use to compute box
         il_radius=1
         IF( PRESENT(id_radius) ) il_radius=id_radius
         IF( il_radius < 0 )THEN
            CALL logger_error("EXTRAP FILL VALUE: invalid "//&
            &  " radius of the box used to compute extrapolation "//&
            &  "("//TRIM(fct_str(il_radius))//")")
         ENDIF

         CALL logger_info("EXTRAP FILL: extrapolate "//TRIM(td_var%c_name)//&
         &  " using "//TRIM(cl_method)//" method." )

         CALL extrap__fill_value( td_var, cl_method, &
         &                        il_radius )
 
      ENDIF

   END SUBROUTINE extrap__fill_value_wrapper
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE extrap__fill_value(td_var, cd_method, id_radius)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute point to be extrapolated, then extrapolate point.
   !> 
   !> @details 
   !> optionaly, you could specify :<br/>
   !>  - refinment factor (default 1)
   !>  - offset between fine and coarse grid (default compute from refinment factor
   !> as offset=(rho-1)/2) 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - select all land points for extrapolation
   !>
   !> @param[inout] td_var    variable structure
   !> @param[in] cd_method    extrapolation method
   !> @param[in] id_radius    radius of the halo used to compute extrapolation
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR)      ,                 INTENT(INOUT) :: td_var
      CHARACTER(LEN=*),                 INTENT(IN   ) :: cd_method
      INTEGER(i4)     ,                 INTENT(IN   ) :: id_radius

      ! local variable
      CHARACTER(LEN=lc)                            :: cl_extrap

      INTEGER(i4), DIMENSION(:,:,:)  , ALLOCATABLE :: il_detect

      TYPE(TATT)                                   :: tl_att

      ! loop indices
      !----------------------------------------------------------------

      !1- detect point to be extrapolated
      ALLOCATE( il_detect( td_var%t_dim(1)%i_len, &
      &                    td_var%t_dim(2)%i_len, &
      &                    td_var%t_dim(3)%i_len) )

      il_detect(:,:,:) = extrap_detect( td_var )

      !2- add attribute to variable
      cl_extrap=fct_concat(td_var%c_extrap(:))
      tl_att=att_init('extrapolation',cl_extrap)
      CALL var_move_att(td_var, tl_att)

      CALL att_clean(tl_att)

      IF( ALL(il_detect(:,:,:)==1) )THEN
         CALL logger_warn(" EXTRAP FILL: "//&
            &  " can not extrapolate "//TRIM(td_var%c_name)//&
            &  ". no value inform." )
      ELSE
         CALL logger_info(" EXTRAP FILL: "//&
            &              TRIM(fct_str(SUM(il_detect(:,:,:))))//&
            &              " point(s) to extrapolate " )

         CALL logger_info(" EXTRAP FILL: method "//&
            &  TRIM(cd_method) )

         !3- extrapolate
         CALL extrap__3D(td_var%d_value(:,:,:,:), td_var%d_fill, &
         &               il_detect(:,:,:),                       &
         &               cd_method, id_radius )
      ENDIF

      DEALLOCATE(il_detect)

   END SUBROUTINE extrap__fill_value
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE extrap__3D(dd_value, dd_fill, id_detect,&
         &               cd_method, id_radius)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute point to be extrapolated in 3D array.
   !> 
   !> @details 
   !> in case of 'min_error' method:<br/>
   !>    - compute derivative in i-, j- and k- direction
   !>    - compute minimum error coefficient (distance to center of halo)
   !>    - compute extrapolatd values by calculated minimum error using taylor expansion 
   !> in case of 'dist_weight' method:<br/>
   !>    - compute distance weight coefficient (inverse of distance to center of halo) 
   !>    - compute extrapolatd values using Inverse Distance Weighting
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2015 
   !> - compute coef indices to be used
   !> - bug fix: force coef indice to 1, for dimension lenth equal to 1
   !>
   !> @param[inout] dd_value  3D array of variable to be extrapolated
   !> @param[in] dd_fill      FillValue of variable
   !> @param[inout] id_detect array of point to extrapolate
   !> @param[in] cd_method    extrapolation method
   !> @param[in] id_radius    radius of the halo used to compute extrapolation
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)   , DIMENSION(:,:,:,:), INTENT(INOUT) :: dd_value
      REAL(dp)   ,                     INTENT(IN   ) :: dd_fill
      INTEGER(i4), DIMENSION(:,:,:)  , INTENT(INOUT) :: id_detect
      CHARACTER(LEN=*),                INTENT(IN   ) :: cd_method
      INTEGER(i4),                     INTENT(IN   ) :: id_radius

      ! local variable
      INTEGER(i4)                                :: il_imin
      INTEGER(i4)                                :: il_imax
      INTEGER(i4)                                :: il_jmin
      INTEGER(i4)                                :: il_jmax
      INTEGER(i4)                                :: il_kmin
      INTEGER(i4)                                :: il_kmax
      INTEGER(i4)                                :: il_iter
      INTEGER(i4)                                :: il_radius
      INTEGER(i4)                                :: il_i1
      INTEGER(i4)                                :: il_i2
      INTEGER(i4)                                :: il_j1
      INTEGER(i4)                                :: il_j2
      INTEGER(i4)                                :: il_k1
      INTEGER(i4)                                :: il_k2

      INTEGER(i4), DIMENSION(4)                  :: il_shape
      INTEGER(i4), DIMENSION(3)                  :: il_dim

      INTEGER(i4), DIMENSION(:,:,:), ALLOCATABLE :: il_detect

      REAL(dp)   , DIMENSION(:,:,:), ALLOCATABLE :: dl_dfdx
      REAL(dp)   , DIMENSION(:,:,:), ALLOCATABLE :: dl_dfdy
      REAL(dp)   , DIMENSION(:,:,:), ALLOCATABLE :: dl_dfdz
      REAL(dp)   , DIMENSION(:,:,:), ALLOCATABLE :: dl_coef

      LOGICAL                                    :: ll_iter

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      il_shape(:)=SHAPE(dd_value)

      ALLOCATE( il_detect( il_shape(1), il_shape(2), il_shape(3)) )

      SELECT CASE(TRIM(cd_method))
      CASE('min_error')
         DO jl=1,il_shape(4)

            ! initialise table of point to be extrapolated
            il_detect(:,:,:)=id_detect(:,:,:)

            il_iter=1
            DO WHILE( ANY(il_detect(:,:,:)==1) )
               ! change extend value to minimize number of iteration
               il_radius=id_radius+(il_iter-1)
               ll_iter=.TRUE.

               ALLOCATE( dl_dfdx(il_shape(1), il_shape(2), il_shape(3)) ) 
               ALLOCATE( dl_dfdy(il_shape(1), il_shape(2), il_shape(3)) ) 
               ALLOCATE( dl_dfdz(il_shape(1), il_shape(2), il_shape(3)) ) 

               ! compute derivative in i-direction
               dl_dfdx(:,:,:)=dd_fill
               IF( il_shape(1) > 1 )THEN
                  dl_dfdx(:,:,:)=math_deriv_3D( dd_value(:,:,:,jl), &
                     &                          dd_fill, 'I' )
               ENDIF

               ! compute derivative in j-direction
               dl_dfdy(:,:,:)=dd_fill
               IF( il_shape(2) > 1 )THEN
                  dl_dfdy(:,:,:)=math_deriv_3D( dd_value(:,:,:,jl), &
                     &                          dd_fill, 'J' )
               ENDIF
 
               ! compute derivative in k-direction
               dl_dfdz(:,:,:)=dd_fill
               IF( il_shape(3) > 1 )THEN
                  dl_dfdz(:,:,:)=math_deriv_3D( dd_value(:,:,:,jl), &
                     &                          dd_fill, 'K' )
               ENDIF
 
               il_dim(1)=2*il_radius+1
               IF( il_shape(1) < 2*il_radius+1 ) il_dim(1)=1
               il_dim(2)=2*il_radius+1
               IF( il_shape(2) < 2*il_radius+1 ) il_dim(2)=1
               il_dim(3)=2*il_radius+1
               IF( il_shape(3) < 2*il_radius+1 ) il_dim(3)=1

               ALLOCATE( dl_coef(il_dim(1), il_dim(2), il_dim(3)) ) 

               dl_coef(:,:,:)=extrap__3D_min_error_coef(dd_value( 1:il_dim(1), &
               &                                                  1:il_dim(2), &
               &                                                  1:il_dim(3), &
               &                                                  jl ))

               DO jk=1,il_shape(3)
                  ! from North West(1,1) to South East(il_shape(1),il_shape(2))
                  IF( ALL(il_detect(:,:,jk) == 0) ) CYCLE
                  DO jj=1,il_shape(2)
                     IF( ALL(il_detect(:,jj,jk) == 0) ) CYCLE
                     DO ji=1,il_shape(1)

                        IF( il_detect(ji,jj,jk) == 1 )THEN
                          
                           il_imin=MAX(ji-il_radius,1)
                           il_imax=MIN(ji+il_radius,il_shape(1))
                           ! coef indices to be used
                           il_i1 = il_radius-(ji-il_imin)+1
                           il_i2 = il_radius+(il_imax-ji)+1
                           IF( il_dim(1) == 1 )THEN
                              il_imin=ji
                              il_imax=ji
                              ! coef indices to be used
                              il_i1 = 1
                              il_i2 = 1
                           ENDIF


                           il_jmin=MAX(jj-il_radius,1)
                           il_jmax=MIN(jj+il_radius,il_shape(2))
                           ! coef indices to be used
                           il_j1 = il_radius-(jj-il_jmin)+1
                           il_j2 = il_radius+(il_jmax-jj)+1
                           IF( il_dim(2) == 1 )THEN
                              il_jmin=jj
                              il_jmax=jj
                              ! coef indices to be used
                              il_j1 = 1
                              il_j2 = 1
                           ENDIF

                           il_kmin=MAX(jk-il_radius,1)
                           il_kmax=MIN(jk+il_radius,il_shape(3))
                           ! coef indices to be used
                           il_k1 = il_radius-(jk-il_kmin)+1
                           il_k2 = il_radius+(il_kmax-jk)+1
                           IF( il_dim(3) == 1 )THEN
                              il_kmin=jk
                              il_kmax=jk
                              ! coef indices to be used
                              il_k1 = 1
                              il_k2 = 1
                           ENDIF

                           dd_value(ji,jj,jk,jl)=extrap__3D_min_error_fill( &
                           &  dd_value( il_imin:il_imax, &
                           &            il_jmin:il_jmax, &
                           &            il_kmin:il_kmax,jl ), dd_fill, il_radius, &
                           &  dl_dfdx(  il_imin:il_imax, &
                           &            il_jmin:il_jmax, &
                           &            il_kmin:il_kmax ), &
                           &  dl_dfdy(  il_imin:il_imax, &
                           &            il_jmin:il_jmax, &
                           &            il_kmin:il_kmax ), &
                           &  dl_dfdz(  il_imin:il_imax, &
                           &            il_jmin:il_jmax, &
                           &            il_kmin:il_kmax ), &
                           &  dl_coef(il_i1:il_i2, &
                           &          il_j1:il_j2, &
                           &          il_k1:il_k2) )

                           IF( dd_value(ji,jj,jk,jl) /= dd_fill )THEN
                              il_detect(ji,jj,jk)= 0
                              ll_iter=.FALSE.
                           ENDIF

                        ENDIF

                     ENDDO
                  ENDDO
                  ! from South East(il_shape(1),il_shape(2)) to North West(1,1)
                  IF( ALL(il_detect(:,:,jk) == 0) ) CYCLE
                  DO jj=il_shape(2),1,-1
                     IF( ALL(il_detect(:,jj,jk) == 0) ) CYCLE
                     DO ji=il_shape(1),1,-1

                        IF( il_detect(ji,jj,jk) == 1 )THEN
                          
                           il_imin=MAX(ji-il_radius,1)
                           il_imax=MIN(ji+il_radius,il_shape(1))
                           ! coef indices to be used
                           il_i1 = il_radius-(ji-il_imin)+1
                           il_i2 = il_radius+(il_imax-ji)+1
                           IF( il_dim(1) == 1 )THEN
                              il_imin=ji
                              il_imax=ji
                              ! coef indices to be used
                              il_i1 = 1
                              il_i2 = 1
                           ENDIF


                           il_jmin=MAX(jj-il_radius,1)
                           il_jmax=MIN(jj+il_radius,il_shape(2))
                           ! coef indices to be used
                           il_j1 = il_radius-(jj-il_jmin)+1
                           il_j2 = il_radius+(il_jmax-jj)+1
                           IF( il_dim(2) == 1 )THEN
                              il_jmin=jj
                              il_jmax=jj
                              ! coef indices to be used
                              il_j1 = 1
                              il_j2 = 1
                           ENDIF

                           il_kmin=MAX(jk-il_radius,1)
                           il_kmax=MIN(jk+il_radius,il_shape(3))
                           ! coef indices to be used
                           il_k1 = il_radius-(jk-il_kmin)+1
                           il_k2 = il_radius+(il_kmax-jk)+1
                           IF( il_dim(3) == 1 )THEN
                              il_kmin=jk
                              il_kmax=jk
                              ! coef indices to be used
                              il_k1 = 1
                              il_k2 = 1
                           ENDIF

                           dd_value(ji,jj,jk,jl)=extrap__3D_min_error_fill( &
                           &  dd_value( il_imin:il_imax, &
                           &            il_jmin:il_jmax, &
                           &            il_kmin:il_kmax,jl ), dd_fill, il_radius, &
                           &  dl_dfdx(  il_imin:il_imax, &
                           &            il_jmin:il_jmax, &
                           &            il_kmin:il_kmax ), &
                           &  dl_dfdy(  il_imin:il_imax, &
                           &            il_jmin:il_jmax, &
                           &            il_kmin:il_kmax ), &
                           &  dl_dfdz(  il_imin:il_imax, &
                           &            il_jmin:il_jmax, &
                           &            il_kmin:il_kmax ), &
                           &  dl_coef(il_i1:il_i2, &
                           &          il_j1:il_j2, &
                           &          il_k1:il_k2) )

                           IF( dd_value(ji,jj,jk,jl) /= dd_fill )THEN
                              il_detect(ji,jj,jk)= 0
                              ll_iter=.FALSE.
                           ENDIF

                        ENDIF

                     ENDDO
                  ENDDO
               ENDDO

               DEALLOCATE( dl_dfdx )
               DEALLOCATE( dl_dfdy )
               DEALLOCATE( dl_dfdz )
               DEALLOCATE( dl_coef )

               IF( ll_iter ) il_iter=il_iter+1
            ENDDO
         ENDDO

      CASE DEFAULT ! 'dist_weight'
         DO jl=1,il_shape(4)

            ! intitialise table of poitn to be extrapolated
            il_detect(:,:,:)=id_detect(:,:,:)

            il_iter=1
            DO WHILE( ANY(il_detect(:,:,:)==1) )
               ! change extend value to minimize number of iteration
               il_radius=id_radius+(il_iter-1)
               ll_iter=.TRUE.

               il_dim(1)=2*il_radius+1
               IF( il_shape(1) < 2*il_radius+1 ) il_dim(1)=1
               il_dim(2)=2*il_radius+1
               IF( il_shape(2) < 2*il_radius+1 ) il_dim(2)=1
               il_dim(3)=2*il_radius+1
               IF( il_shape(3) < 2*il_radius+1 ) il_dim(3)=1
               
               ALLOCATE( dl_coef(il_dim(1), il_dim(2), il_dim(3)) )

               dl_coef(:,:,:)=extrap__3D_dist_weight_coef(dd_value(1:il_dim(1),&
               &                                                   1:il_dim(2),&
               &                                                   1:il_dim(3),&
               &                                                   jl ) )
               
               DO jk=1,il_shape(3)
                  ! from North West(1,1) to South East(il_shape(1),il_shape(2))
                  IF( ALL(il_detect(:,:,jk) == 0) ) CYCLE
                  DO jj=1,il_shape(2)
                     IF( ALL(il_detect(:,jj,jk) == 0) ) CYCLE
                     DO ji=1,il_shape(1)

                        IF( il_detect(ji,jj,jk) == 1 )THEN
                           
                           il_imin=MAX(ji-il_radius,1)
                           il_imax=MIN(ji+il_radius,il_shape(1))
                           ! coef indices to be used
                           il_i1 = il_radius-(ji-il_imin)+1
                           il_i2 = il_radius+(il_imax-ji)+1
                           IF( il_dim(1) == 1 )THEN
                              il_imin=ji
                              il_imax=ji
                              ! coef indices to be used
                              il_i1 = 1
                              il_i2 = 1
                           ENDIF

                           il_jmin=MAX(jj-il_radius,1)
                           il_jmax=MIN(jj+il_radius,il_shape(2))
                           ! coef indices to be used
                           il_j1 = il_radius-(jj-il_jmin)+1
                           il_j2 = il_radius+(il_jmax-jj)+1
                           IF( il_dim(2) == 1 )THEN
                              il_jmin=jj
                              il_jmax=jj
                              ! coef indices to be used
                              il_j1 = 1
                              il_j2 = 1
                           ENDIF

                           il_kmin=MAX(jk-il_radius,1)
                           il_kmax=MIN(jk+il_radius,il_shape(3))
                           ! coef indices to be used
                           il_k1 = il_radius-(jk-il_kmin)+1
                           il_k2 = il_radius+(il_kmax-jk)+1
                           IF( il_dim(3) == 1 )THEN
                              il_kmin=jk
                              il_kmax=jk
                              ! coef indices to be used
                              il_k1 = 1
                              il_k2 = 1
                           ENDIF

                           dd_value(ji,jj,jk,jl)=extrap__3D_dist_weight_fill( &
                           &  dd_value( il_imin:il_imax, &
                           &            il_jmin:il_jmax, &
                           &            il_kmin:il_kmax, &
                           &            jl), dd_fill, il_radius, &
                           &  dl_coef(il_i1:il_i2, &
                           &          il_j1:il_j2, &
                           &          il_k1:il_k2) )

                           IF( dd_value(ji,jj,jk,jl) /= dd_fill )THEN
                              il_detect(ji,jj,jk)= 0
                              ll_iter=.FALSE.
                           ENDIF

                        ENDIF

                     ENDDO
                  ENDDO
                  ! from South East(il_shape(1),il_shape(2)) to North West(1,1)
                  IF( ALL(il_detect(:,:,jk) == 0) ) CYCLE
                  DO jj=il_shape(2),1,-1
                     IF( ALL(il_detect(:,jj,jk) == 0) ) CYCLE
                     DO ji=il_shape(1),1,-1

                        IF( il_detect(ji,jj,jk) == 1 )THEN
                           
                           il_imin=MAX(ji-il_radius,1)
                           il_imax=MIN(ji+il_radius,il_shape(1))
                           ! coef indices to be used
                           il_i1 = il_radius-(ji-il_imin)+1
                           il_i2 = il_radius+(il_imax-ji)+1
                           IF( il_dim(1) == 1 )THEN
                              il_imin=ji
                              il_imax=ji
                              ! coef indices to be used
                              il_i1 = 1
                              il_i2 = 1
                           ENDIF

                           il_jmin=MAX(jj-il_radius,1)
                           il_jmax=MIN(jj+il_radius,il_shape(2))
                           ! coef indices to be used
                           il_j1 = il_radius-(jj-il_jmin)+1
                           il_j2 = il_radius+(il_jmax-jj)+1
                           IF( il_dim(2) == 1 )THEN
                              il_jmin=jj
                              il_jmax=jj
                              ! coef indices to be used
                              il_j1 = 1
                              il_j2 = 1
                           ENDIF

                           il_kmin=MAX(jk-il_radius,1)
                           il_kmax=MIN(jk+il_radius,il_shape(3))
                           ! coef indices to be used
                           il_k1 = il_radius-(jk-il_kmin)+1
                           il_k2 = il_radius+(il_kmax-jk)+1
                           IF( il_dim(3) == 1 )THEN
                              il_kmin=jk
                              il_kmax=jk
                              ! coef indices to be used
                              il_k1 = 1
                              il_k2 = 1
                           ENDIF

                           dd_value(ji,jj,jk,jl)=extrap__3D_dist_weight_fill( &
                           &  dd_value( il_imin:il_imax, &
                           &            il_jmin:il_jmax, &
                           &            il_kmin:il_kmax, &
                           &            jl), dd_fill, il_radius, &
                           &  dl_coef(il_i1:il_i2, &
                           &          il_j1:il_j2, &
                           &          il_k1:il_k2) )

                           IF( dd_value(ji,jj,jk,jl) /= dd_fill )THEN
                              il_detect(ji,jj,jk)= 0
                              ll_iter=.FALSE.
                           ENDIF

                        ENDIF

                     ENDDO
                  ENDDO
               ENDDO
               CALL logger_info(" EXTRAP 3D: "//&
               &              TRIM(fct_str(SUM(il_detect(:,:,:))))//&
               &              " point(s) to extrapolate " )
            
               DEALLOCATE( dl_coef )
               IF( ll_iter ) il_iter=il_iter+1
            ENDDO
         ENDDO            
      END SELECT

      DEALLOCATE( il_detect )

   END SUBROUTINE extrap__3D
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION extrap__3D_min_error_coef(dd_value) &
         & RESULT (df_value)
   !-------------------------------------------------------------------
   !> @brief
   !> This function compute coefficient for min_error extrapolation.
   !> 
   !> @details 
   !> coefficients are  "grid distance" to the center of the box 
   !> choosed to compute extrapolation.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2015 
   !> - decrease weight of third dimension
   !>
   !> @param[in] dd_value  3D array of variable to be extrapolated
   !> @return 3D array of coefficient for minimum error extrapolation
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)   , DIMENSION(:,:,:), INTENT(IN) :: dd_value

      ! function
      REAL(dp), DIMENSION(SIZE(dd_value(:,:,:),DIM=1), &
      &                   SIZE(dd_value(:,:,:),DIM=2), &
      &                   SIZE(dd_value(:,:,:),DIM=3) ) :: df_value

      ! local variable
      INTEGER(i4), DIMENSION(3) :: il_shape

      INTEGER(i4) :: il_imid
      INTEGER(i4) :: il_jmid
      INTEGER(i4) :: il_kmid

      REAL(dp)   , DIMENSION(:,:,:), ALLOCATABLE :: dl_dist

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      ! init
      df_value(:,:,:)=0

      il_shape(:)=SHAPE(dd_value(:,:,:))

      il_imid=INT(REAL(il_shape(1),sp)*0.5+1)
      il_jmid=INT(REAL(il_shape(2),sp)*0.5+1)
      il_kmid=INT(REAL(il_shape(3),sp)*0.5+1)

      ALLOCATE( dl_dist(il_shape(1),il_shape(2),il_shape(3)) )

      DO jk=1,il_shape(3)
         DO jj=1,il_shape(2)
            DO ji=1,il_shape(1)

               ! compute distance
               ! "vertical weight" is lower than horizontal 
               dl_dist(ji,jj,jk) = (ji-il_imid)**2 + &
               &                   (jj-il_jmid)**2 + &
               &                 3*(jk-il_kmid)**2

               IF( dl_dist(ji,jj,jk) /= 0 )THEN
                  dl_dist(ji,jj,jk)=SQRT( dl_dist(ji,jj,jk) )
               ENDIF

            ENDDO
         ENDDO
      ENDDO

      WHERE( dl_dist(:,:,:) /= 0 )
         df_value(:,:,:)=dl_dist(:,:,:)
      END WHERE

      DEALLOCATE( dl_dist )

   END FUNCTION extrap__3D_min_error_coef
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION extrap__3D_min_error_fill(dd_value, dd_fill, id_radius,&
         &                                 dd_dfdx, dd_dfdy, dd_dfdz,   &
         &                                 dd_coef) &
         & RESULT (df_value)
   !-------------------------------------------------------------------
   !> @brief
   !> This function compute extrapolatd value by calculated minimum error using
   !> taylor expansion
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_value  3D array of variable to be extrapolated
   !> @param[in] dd_fill   FillValue of variable
   !> @param[in] id_radius radius of the halo used to compute extrapolation
   !> @param[in] dd_dfdx   derivative of function in i-direction
   !> @param[in] dd_dfdy   derivative of function in j-direction
   !> @param[in] dd_dfdz   derivative of function in k-direction
   !> @param[in] dd_coef   array of coefficient for min_error extrapolation 
   !> @return extrapolatd value
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)   , DIMENSION(:,:,:), INTENT(IN) :: dd_value
      REAL(dp)   ,                   INTENT(IN) :: dd_fill
      INTEGER(i4),                   INTENT(IN) :: id_radius
      REAL(dp)   , DIMENSION(:,:,:), INTENT(IN) :: dd_dfdx
      REAL(dp)   , DIMENSION(:,:,:), INTENT(IN) :: dd_dfdy
      REAL(dp)   , DIMENSION(:,:,:), INTENT(IN) :: dd_dfdz
      REAL(dp)   , DIMENSION(:,:,:), INTENT(IN) :: dd_coef

      ! function
      REAL(dp)                                  :: df_value

      ! local variable
      INTEGER(i4), DIMENSION(3) :: il_shape
      INTEGER(i4), DIMENSION(3) :: il_ind

      INTEGER(i4), DIMENSION(:,:,:), ALLOCATABLE :: il_mask
      REAL(dp)   , DIMENSION(:,:,:), ALLOCATABLE :: dl_error

      INTEGER(i4) :: il_min
      ! loop indices

      !----------------------------------------------------------------

      ! init
      df_value=dd_fill

      il_min=MAX(1,(SIZE(dd_value(:,:,:)))/(1+id_radius*2))

      IF( COUNT(dd_value(:,:,:) /= dd_fill) >= il_min )THEN

         il_shape(:)=SHAPE(dd_value(:,:,:))
         ALLOCATE( il_mask( il_shape(1),il_shape(2),il_shape(3)) )
         ALLOCATE( dl_error(il_shape(1),il_shape(2),il_shape(3)) )

         ! compute error
         dl_error(:,:,:)=0.
         il_mask(:,:,:)=0
         WHERE( dd_dfdx(:,:,:) /= dd_fill )
            dl_error(:,:,:)=dd_coef(:,:,:)*dd_dfdx(:,:,:)
            il_mask(:,:,:)=1
         END WHERE
         WHERE( dd_dfdy(:,:,:) /= dd_fill  )
            dl_error(:,:,:)=(dl_error(:,:,:)+dd_coef(:,:,:)*dd_dfdy(:,:,:))
            il_mask(:,:,:)=1
         END WHERE
         WHERE( dd_dfdz(:,:,:) /= dd_fill  )
            dl_error(:,:,:)=(dl_error(:,:,:)+dd_coef(:,:,:)*dd_dfdz(:,:,:))
            il_mask(:,:,:)=1
         END WHERE

         ! get minimum error indices
         il_ind(:)=MINLOC(dl_error(:,:,:),il_mask(:,:,:)==1)

         ! return value
         IF( ALL(il_ind(:)/=0) )THEN
            df_value=dd_value(il_ind(1),il_ind(2),il_ind(3))
         ENDIF

         DEALLOCATE( il_mask )
         DEALLOCATE( dl_error )

      ENDIF

   END FUNCTION extrap__3D_min_error_fill
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION extrap__3D_dist_weight_coef(dd_value) &
         & RESULT (df_value)
   !-------------------------------------------------------------------
   !> @brief
   !> This function compute coefficient for inverse distance weighted method 
   !> 
   !> @details 
   !> coefficients are inverse "grid distance" to the center of the box choosed to compute
   !> extrapolation.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2015 
   !> - decrease weight of third dimension
   !>
   !> @param[in] dd_value  3D array of variable to be extrapolated
   !> @return 3D array of coefficient for inverse distance weighted extrapolation
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:,:,:), INTENT(IN) :: dd_value

      ! function
      REAL(dp), DIMENSION(SIZE(dd_value(:,:,:),DIM=1), &
      &                   SIZE(dd_value(:,:,:),DIM=2), &
      &                   SIZE(dd_value(:,:,:),DIM=3) ) :: df_value

      ! local variable
      INTEGER(i4), DIMENSION(3) :: il_shape

      INTEGER(i4) :: il_imid
      INTEGER(i4) :: il_jmid
      INTEGER(i4) :: il_kmid

      REAL(dp)   , DIMENSION(:,:,:), ALLOCATABLE :: dl_dist

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      ! init
      df_value(:,:,:)=0

      il_shape(:)=SHAPE(dd_value(:,:,:))

      il_imid=INT(REAL(il_shape(1),sp)*0.5+1,i4)
      il_jmid=INT(REAL(il_shape(2),sp)*0.5+1,i4)
      il_kmid=INT(REAL(il_shape(3),sp)*0.5+1,i4)

      ALLOCATE( dl_dist(il_shape(1),il_shape(2),il_shape(3)) )

      DO jk=1,il_shape(3)
         DO jj=1,il_shape(2)
            DO ji=1,il_shape(1)

               ! compute distance
               ! "vertical weight" is lower than horizontal 
               dl_dist(ji,jj,jk) = (ji-il_imid)**2 + &
               &                   (jj-il_jmid)**2 + &
               &                 3*(jk-il_kmid)**2

               IF( dl_dist(ji,jj,jk) /= 0 )THEN
                  dl_dist(ji,jj,jk)=SQRT( dl_dist(ji,jj,jk) )
               ENDIF

            ENDDO
         ENDDO
      ENDDO

      WHERE( dl_dist(:,:,:) /= 0 ) 
         df_value(:,:,:)=1./dl_dist(:,:,:)
      END WHERE

      DEALLOCATE( dl_dist )

   END FUNCTION extrap__3D_dist_weight_coef
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION extrap__3D_dist_weight_fill(dd_value, dd_fill, id_radius, &
         &                              dd_coef) &
         &  RESULT (df_value)
   !-------------------------------------------------------------------
   !> @brief
   !> This function compute extrapolatd value using inverse distance weighted
   !> method
   !> 
   !> @details 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_value  3D array of variable to be extrapolated
   !> @param[in] dd_fill   FillValue of variable
   !> @param[in] id_radius radius of the halo used to compute extrapolation
   !> @param[in] dd_coef   3D array of coefficient for inverse distance weighted extrapolation 
   !> @return extrapolatd value
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)   , DIMENSION(:,:,:), INTENT(IN) :: dd_value
      REAL(dp)   ,                   INTENT(IN) :: dd_fill
      INTEGER(i4),                   INTENT(IN) :: id_radius
      REAL(dp)   , DIMENSION(:,:,:), INTENT(IN) :: dd_coef

      ! function
      REAL(dp)                                  :: df_value

      ! local variable
      INTEGER(i4), DIMENSION(3) :: il_shape

      REAL(dp)   , DIMENSION(:,:,:), ALLOCATABLE :: dl_value
      REAL(dp)   , DIMENSION(:,:,:), ALLOCATABLE :: dl_coef

      INTEGER(i4) :: il_min
      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      ! init
      df_value=dd_fill

      il_min=MAX(1,(SIZE(dd_value(:,:,:)))/(1+id_radius*2))

      IF( COUNT(dd_value(:,:,:)/= dd_fill) >= il_min )THEN

         il_shape(:)=SHAPE(dd_value(:,:,:))
         ALLOCATE( dl_value( il_shape(1),il_shape(2),il_shape(3)) )
         ALLOCATE( dl_coef( il_shape(1),il_shape(2),il_shape(3)) )

         dl_value(:,:,:)=0
         dl_coef(:,:,:)=0

         DO jk=1,il_shape(3)
            DO jj=1,il_shape(2)
               DO ji=1,il_shape(1)

                  IF( dd_value(ji,jj,jk) /= dd_fill )THEN
                     ! compute factor
                     dl_value(ji,jj,jk)=dd_coef(ji,jj,jk)*dd_value(ji,jj,jk)
                     dl_coef(ji,jj,jk)=dd_coef(ji,jj,jk)
                  ENDIF

               ENDDO
            ENDDO
         ENDDO


         ! return value
         IF( SUM( dl_coef(:,:,:) ) /= 0 )THEN
            df_value = SUM( dl_value(:,:,:) )/SUM( dl_coef(:,:,:) )
         ENDIF

         DEALLOCATE( dl_value )
         DEALLOCATE( dl_coef )

      ENDIF

   END FUNCTION extrap__3D_dist_weight_fill
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE extrap_add_extrabands(td_var, id_isize, id_jsize)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine add to the variable (to be extrapolated) an 
   !> extraband of N points at north,south,east and west boundaries.
   !> 
   !> @details
   !> optionaly you could specify size of extra bands in i- and j-direction
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_var variable 
   !> @param[in] id_isize  i-direction size of extra bands (default=im_minext)
   !> @param[in] id_jsize  j-direction size of extra bands (default=im_minext)
   !> @todo
   !> - invalid special case for grid with north fold
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) , INTENT(INOUT)  :: td_var
      INTEGER(i4), INTENT(IN   ), OPTIONAL :: id_isize
      INTEGER(i4), INTENT(IN   ), OPTIONAL :: id_jsize

      ! local variable
      REAL(dp), DIMENSION(:,:,:,:) , ALLOCATABLE :: dl_value

      INTEGER(i4) :: il_isize
      INTEGER(i4) :: il_jsize
      INTEGER(i4) :: il_tmp

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: ij
      !----------------------------------------------------------------
      il_isize=im_minext
      IF(PRESENT(id_isize)) il_isize=id_isize
      IF( il_isize < im_minext .AND. &
      &   TRIM(td_var%c_interp(1)) == 'cubic' )THEN
         CALL logger_warn("EXTRAP ADD EXTRABANDS: size of extrabands "//&
         &  "should be at least "//TRIM(fct_str(im_minext))//" for "//&
         &  " cubic interpolation ")
      ENDIF

      il_jsize=im_minext
      IF(PRESENT(id_jsize)) il_jsize=id_jsize
      IF( il_jsize < im_minext .AND. &
      &   TRIM(td_var%c_interp(1)) == 'cubic' )THEN
         CALL logger_warn("EXTRAP ADD EXTRABANDS: size of extrabands "//&
         &  "should be at least "//TRIM(fct_str(im_minext))//" for "//&
         &  " cubic interpolation ")
      ENDIF

      IF( .NOT. td_var%t_dim(1)%l_use ) il_isize=0
      IF( .NOT. td_var%t_dim(2)%l_use ) il_jsize=0

      CALL logger_trace( "EXTRAP ADD EXTRABANDS: dimension change "//&
      &              "in variable "//TRIM(td_var%c_name) )

      ! add extrabands in variable
      ALLOCATE(dl_value( td_var%t_dim(1)%i_len, &
      &                  td_var%t_dim(2)%i_len, &
      &                  td_var%t_dim(3)%i_len, &
      &                  td_var%t_dim(4)%i_len ))

      dl_value(:,:,:,:)=td_var%d_value(:,:,:,:)


      td_var%t_dim(1)%i_len = td_var%t_dim(1)%i_len + 2*il_isize
      td_var%t_dim(2)%i_len = td_var%t_dim(2)%i_len + 2*il_jsize

      DEALLOCATE(td_var%d_value)
      ALLOCATE( td_var%d_value(td_var%t_dim(1)%i_len, &
      &                        td_var%t_dim(2)%i_len, &
      &                        td_var%t_dim(3)%i_len, &
      &                        td_var%t_dim(4)%i_len ) )

      ! intialise
      td_var%d_value(:,:,:,:)=td_var%d_fill

      ! fill center
      td_var%d_value( 1+il_isize:td_var%t_dim(1)%i_len-il_isize, &
      &               1+il_jsize:td_var%t_dim(2)%i_len-il_jsize, &
      &                :,:) = dl_value(:,:,:,:)

      ! special case for overlap
      IF( td_var%i_ew >= 0 .AND. il_isize /= 0 )THEN
         DO ji=1,il_isize
            ! from east to west
            il_tmp=td_var%t_dim(1)%i_len-td_var%i_ew+ji-2*il_isize
            td_var%d_value(ji,:,:,:) = td_var%d_value(il_tmp,:,:,:)

            ! from west to east
            ij=td_var%t_dim(1)%i_len-ji+1
            il_tmp=td_var%i_ew-ji+2*il_isize+1
            td_var%d_value(ij,:,:,:) = td_var%d_value(il_tmp,:,:,:)
         ENDDO
      ENDIF

      DEALLOCATE( dl_value )

   END SUBROUTINE extrap_add_extrabands
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE extrap_del_extrabands(td_var, id_isize, id_jsize)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine remove of the variable an extraband 
   !> of N points at north,south,east and west boundaries.
   !> 
   !> @details
   !> optionaly you could specify size of extra bands in i- and j-direction
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_var variable 
   !> @param[in] id_isize  i-direction size of extra bands (default=im_minext)
   !> @param[in] id_jsize  j-direction size of extra bands (default=im_minext)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) , INTENT(INOUT) :: td_var
      INTEGER(i4), INTENT(IN   ), OPTIONAL :: id_isize
      INTEGER(i4), INTENT(IN   ), OPTIONAL :: id_jsize

      ! local variable
      REAL(dp), DIMENSION(:,:,:,:) , ALLOCATABLE :: dl_value

      INTEGER(i4) :: il_isize
      INTEGER(i4) :: il_jsize
 
      INTEGER(i4) :: il_imin
      INTEGER(i4) :: il_imax
      INTEGER(i4) :: il_jmin
      INTEGER(i4) :: il_jmax

      ! loop indices
      !----------------------------------------------------------------
      il_isize=im_minext
      IF(PRESENT(id_isize)) il_isize=id_isize

      il_jsize=im_minext
      IF(PRESENT(id_jsize)) il_jsize=id_jsize

      IF( .NOT. td_var%t_dim(1)%l_use ) il_isize=0
      IF( .NOT. td_var%t_dim(2)%l_use ) il_jsize=0

      CALL logger_trace( "EXTRAP DEL EXTRABANDS: dimension change "//&
      &              "in variable "//TRIM(td_var%c_name) )

      ! add extrabands in variable
      ALLOCATE(dl_value( td_var%t_dim(1)%i_len, &
      &                  td_var%t_dim(2)%i_len, &
      &                  td_var%t_dim(3)%i_len, &
      &                  td_var%t_dim(4)%i_len ))

      dl_value(:,:,:,:)=td_var%d_value(:,:,:,:)

      ! fill center
      il_imin=1+il_isize
      il_imax=td_var%t_dim(1)%i_len-il_isize

      il_jmin=1+il_jsize
      il_jmax=td_var%t_dim(2)%i_len-il_jsize
      
      td_var%t_dim(1)%i_len = td_var%t_dim(1)%i_len - 2*il_isize
      td_var%t_dim(2)%i_len = td_var%t_dim(2)%i_len - 2*il_jsize

      DEALLOCATE(td_var%d_value)
      ALLOCATE( td_var%d_value(td_var%t_dim(1)%i_len, &
      &                        td_var%t_dim(2)%i_len, &
      &                        td_var%t_dim(3)%i_len, &
      &                        td_var%t_dim(4)%i_len ) )

      ! intialise
      td_var%d_value(:,:,:,:)=td_var%d_fill

      td_var%d_value(:,:,:,:)=dl_value(il_imin:il_imax,&
      &                                il_jmin:il_jmax,&
      &                                :,:)

      DEALLOCATE( dl_value )

   END SUBROUTINE extrap_del_extrabands
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE extrap
