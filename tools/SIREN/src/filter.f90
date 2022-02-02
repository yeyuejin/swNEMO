!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief This module is filter manager.
!>
!> @details Filtering method to be used is specify inside variable strcuture,
!>    as array of string character.<br/>
!>    td_var\%c_filter(1) string character is the filter name choose between:<br/>
!>       - 'hann'
!>          - rad < cutoff : @f$ filter=0.5+0.5*COS(\pi*\frac{rad}{cutoff}) @f$
!>          - rad > cutoff : @f$ filter=0 @f$
!>       - 'hamming'
!>          - rad < cutoff : @f$ filter=0.54+0.46*COS(\pi*\frac{rad}{cutoff}) @f$
!>          - rad > cutoff : @f$ filter=0 @f$               
!>       - 'blackman'
!>          - rad < cutoff : @f$ filter=0.42 + 0.5*COS(\pi*\frac{rad}{cutoff}) + 
!>                                      0.08*COS(2\pi*\frac{rad}{cutoff}) @f$
!>          - rad > cutoff : @f$ filter=0 @f$
!>       - 'gauss'
!>          - @f$filter=exp(-(\alpha * rad^2) / (2*cutoff^2))@f$
!>       - 'butterworth'
!>          - @f$ filer=1 / (1+(rad^2 / cutoff^2)^{\alpha}) @f$
!>             .
!>
!>       with @f$ rad= \sqrt{(dist-radius)^2} @f$
!>
!>    td_var\%c_filter(2) string character is the number of turn to be done<br/>
!>    td_var\%c_filter(3) string character is the cut-off frequency 
! >                       (count in number of mesh grid)<br/>
!>    td_var\%c_filter(4) string character is the halo radius 
!>                        (count in number of mesh grid)<br/>
!>    td_var\%c_filter(5) string character is the alpha parameter 
!>                        (for gauss and butterworth method)<br/>
!>    
!>    @note Filter method could be specify for each variable in namelist _namvar_,
!>    defining string character _cn\_varinfo_. None by default.<br/>
!>    Filter method parameters are informed inside bracket.
!>       - @f$\alpha@f$ parameter is added for _gauss_ and _butterworth_ methods
!> 
!>    The number of turn is specify using '*' separator.<br/>
!>    Example:
!>       - cn_varinfo='varname1:flt=2*hamming(@f$cutoff@f$,@f$radius@f$)', 
!>                    'varname2:flt=gauss(@f$cutoff@f$,@f$radius@f$,@f$\alpha@f$)'
!>
!>    to filter variable value:<br/>
!> @code
!>    CALL filter_fill_value( td_var )
!> @endcode
!>       - td_var is variable structure
!>
!> @author
!> J.Paul
!>
!> @date November, 2013 - Initial Version
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE filter

   USE kind                            ! F90 kind parameter
   USE phycst                          ! physical constant
   USE logger                          ! log file manager
   USE fct                             ! basic usefull function
   use att                             ! attribute manager
   USE var                             ! variable manager
   USE extrap                          ! extrapolation manager

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! type and variable


   ! function and subroutine
   PUBLIC :: filter_fill_value   !< filter variable value

   PRIVATE :: filter__fill_value_wrapper !
   PRIVATE :: filter__fill_value         !
   PRIVATE :: filter__3D_fill_value      ! 
   PRIVATE :: filter__2D_fill_value      !
   PRIVATE :: filter__2D                 !
   PRIVATE :: filter__2D_coef            !
   PRIVATE :: filter__2D_hann            !
   PRIVATE :: filter__2D_hamming         !
   PRIVATE :: filter__2D_blackman        !
   PRIVATE :: filter__2D_gauss           !
   PRIVATE :: filter__2D_butterworth     !
   PRIVATE :: filter__1D_fill_value      !
   PRIVATE :: filter__1D                 !
   PRIVATE :: filter__1D_coef            !
   PRIVATE :: filter__1D_hann            !
   PRIVATE :: filter__1D_hamming         !
   PRIVATE :: filter__1D_blackman        !
   PRIVATE :: filter__1D_gauss           !
   PRIVATE :: filter__1D_butterworth     !

   INTERFACE filter_fill_value
      MODULE PROCEDURE filter__fill_value_wrapper
   END INTERFACE filter_fill_value

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE filter__fill_value_wrapper(td_var)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine filter variable value.
   !>
   !> @details
   !> it checks if filtering method is available,
   !>  gets parameter value, and launch filter__fill_value 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), INTENT(INOUT) :: td_var

      ! local variable
      CHARACTER(LEN=lc) :: cl_filter
      CHARACTER(LEN=lc) :: cl_method
      INTEGER(I4)       :: il_radius
      INTEGER(I4)       :: il_nturn
      REAL(dp)          :: dl_cutoff 
      REAL(dp)          :: dl_alpha

      TYPE(TATT)        :: tl_att

      ! loop indices
      INTEGER(I4) :: jl
      !----------------------------------------------------------------

      IF( .NOT. ASSOCIATED(td_var%d_value) )THEN
         CALL logger_error("FILTER FILL VALUE: no array of value "//&
         &  "associted to variable "//TRIM(td_var%c_name) )
      ELSE

         SELECT CASE(TRIM(td_var%c_filter(1)))

         CASE DEFAULT
         
            CALL logger_trace("FILTER FILL VALUE: no filter selected "//&
            &  "for variable "//TRIM(td_var%c_name))

         CASE('hann','hamming','blackman','gauss','butterworth')

            cl_method=TRIM(td_var%c_filter(1))

            ! look for number of turn to be done
            READ(td_var%c_filter(2),*) il_nturn
            IF( il_nturn < 0 )THEN
               CALL logger_error("FILTER FILL VALUE: invalid "//&
               &  "number of turn ("//TRIM(td_var%c_filter(2))//")")
            ENDIF

            ! look for cut-off frequency
            dl_cutoff=2
            IF( TRIM(td_var%c_filter(3)) /= '' )THEN
               READ(td_var%c_filter(3),*) dl_cutoff
            ENDIF
            IF( dl_cutoff < 0 )THEN
               CALL logger_error("FILTER FILL VALUE: invalid cut-off "//&
               &  "frequency ("//TRIM(td_var%c_filter(3))//")")
            ENDIF

            ! look for halo size
            il_radius=1
            IF( TRIM(td_var%c_filter(4)) /= '' )THEN
               READ(td_var%c_filter(4),*) il_radius
            ENDIF
            IF( il_radius < 0 )THEN
               CALL logger_error("FILTER FILL VALUE: invalid halo radius "//&
               &  " ("//TRIM(td_var%c_filter(4))//")")
            ENDIF

            IF( REAL(2*il_radius+1,dp) < dl_cutoff )THEN
               CALL logger_error("FILTER FILL VALUE: radius of halo and "//&
               &  "spatial cut-off frequency are not suitable.")
            ENDIF

            ! look for alpha parameter
            dl_alpha=2
            IF( TRIM(td_var%c_filter(5)) /= '' )THEN
               READ(td_var%c_filter(5),*) dl_alpha
            ENDIF

            SELECT CASE(TRIM(cl_method))
            CASE('gauss','butterworth')
               CALL logger_info("FILTER FILL VALUE: filtering "//&
               &   " variable "//TRIM(td_var%c_name)//&
               &   " using "//TRIM(fct_str(il_nturn))//" turn"//&
               &   " of "//TRIM(cl_method)//" method,"//&
               &   " with cut-off frequency of "//&
               &        TRIM(fct_str(REAL(dl_cutoff,sp)))//&
               &   ", halo's radius of "//&
               &        TRIM(fct_str(il_radius))//&
               &   ", and alpha parameter of "//&
               &        TRIM(fct_str(REAL(dl_alpha,sp))) )
            CASE DEFAULT
               CALL logger_info("FILTER FILL VALUE: filtering "//&
               &   " variable "//TRIM(td_var%c_name)//&
               &   " using "//TRIM(fct_str(il_nturn))//" turn"//&
               &   " of "//TRIM(cl_method)//" method,"//&
               &   " with cut-off frequency of "//&
               &        TRIM(fct_str(REAL(dl_cutoff,sp)))//&
               &   " and halo's radius of "//&
               &        TRIM(fct_str(il_radius)) )
            END SELECT
      
            IF( .NOT. ANY(td_var%t_dim(1:3)%l_use) )THEN
               ! no dimension I-J-K used
               CALL logger_debug("FILTER FILL VALUE: no filtering can "//&
               &  "be done for variable "//TRIM(td_var%c_name))
            ELSE 

               ! add attribute to variable
               SELECT CASE(TRIM(cl_method))
               CASE('gauss','butterworth')
                  cl_filter=TRIM(fct_str(il_nturn))//'*'//TRIM(cl_method)//&
                  &                    '('//TRIM(fct_str(REAL(dl_cutoff,sp)))//","//&
                  &                         TRIM(fct_str(il_radius))//","//&
                  &                         TRIM(fct_str(REAL(dl_alpha,sp)))//')'
               CASE DEFAULT
                  cl_filter=TRIM(fct_str(il_nturn))//'*'//TRIM(cl_method)//&
                  &                    '('//TRIM(fct_str(REAL(dl_cutoff,sp)))//","//&
                  &                         TRIM(fct_str(il_radius))//')'
               END SELECT
               tl_att=att_init('filter',cl_filter)
               CALL var_move_att(td_var,tl_att)
               ! clean
               CALL att_clean(tl_att)

               DO jl=1,il_nturn
                  CALL filter__fill_value( td_var, TRIM(cl_method),  & 
                  &                        dl_cutoff, il_radius, dl_alpha )
               ENDDO
            ENDIF               

         END SELECT

      ENDIF
   END SUBROUTINE filter__fill_value_wrapper
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE filter__fill_value(td_var, cd_name, &
         &                       dd_cutoff, id_radius, dd_alpha)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine filtering variable value, given cut-off frequency
   !> halo radius and alpha parameter.
   !> 
   !> @details 
   !>    First extrabands are added to array of variable value.
   !>    Then values are extrapolated, before apply filter.
   !>    Finally extrabands are removed.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable 
   !> @param[in] cd_name   filter name
   !> @param[in] dd_cutoff cut-off frequency
   !> @param[in] id_radius filter halo radius
   !> @param[in] dd_alpha  filter parameter
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR)      , INTENT(INOUT) :: td_var
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_name
      REAL(dp)        , INTENT(IN   ) :: dd_cutoff 
      INTEGER(I4)     , INTENT(IN   ) :: id_radius
      REAL(dp)        , INTENT(IN   ) :: dd_alpha

      ! local variable
      TYPE(TVAR)                                         :: tl_mask

      INTEGER(i1)      , DIMENSION(:,:,:,:), ALLOCATABLE :: bl_mask

      ! loop indices
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      CALL logger_debug("FILTER: "//TRIM(fct_str(td_var%d_fill)) )

      !1-add extraband
      CALL extrap_add_extrabands(td_var, id_radius, id_radius)

      !2-compute mask
      ALLOCATE(bl_mask(td_var%t_dim(1)%i_len, &
      &                td_var%t_dim(2)%i_len, &
      &                td_var%t_dim(3)%i_len, &
      &                td_var%t_dim(4)%i_len) )

      bl_mask(:,:,:,:)=1
      WHERE(td_var%d_value(:,:,:,:)==td_var%d_fill) bl_mask(:,:,:,:)=0      

      tl_mask=var_init('tmask', bl_mask(:,:,:,:))

      DEALLOCATE(bl_mask)

      !3-extrapolate
      CALL extrap_fill_value( td_var ) !, id_iext=id_radius, id_jext=id_radius )

      !4-filtering
      DO jl=1,td_var%t_dim(4)%i_len
         IF( ALL(td_var%t_dim(1:3)%l_use) )THEN
            ! dimension I-J-K used
            CALL filter__3D_fill_value( td_var%d_value(:,:,:,jl),       &
            &                           td_var%d_fill, TRIM(cd_name), &
            &                           dd_cutoff, id_radius, dd_alpha)
         ELSE IF( ALL(td_var%t_dim(1:2)%l_use) )THEN 
            ! dimension I-J used
            CALL filter__2D_fill_value( td_var%d_value(:,:,1,jl),       &
            &                           td_var%d_fill, TRIM(cd_name), &
            &                           dd_cutoff, id_radius, dd_alpha)         
         ELSE IF( td_var%t_dim(3)%l_use )THEN 
            ! dimension K used
            CALL filter__1D_fill_value( td_var%d_value(1,1,:,jl),       &
            &                           td_var%d_fill, TRIM(cd_name), &
            &                           dd_cutoff, id_radius, dd_alpha)         
         ENDIF
      ENDDO

      !5-keep original mask
      WHERE( tl_mask%d_value(:,:,:,:) == 0 )
         td_var%d_value(:,:,:,:)=td_var%d_fill
      END WHERE

      ! clean
      CALL var_clean(tl_mask)

      !6-remove extraband
      CALL extrap_del_extrabands(td_var, id_radius, id_radius)

   END SUBROUTINE filter__fill_value
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE filter__3D_fill_value(dd_value, dd_fill, cd_name, &
         &                          dd_cutoff, id_radius, dd_alpha)
   !-------------------------------------------------------------------
   !> @brief This subroutine compute filtered value of 3D array. 
   !>
   !> @details
   !>    First compute filter coefficient.
   !>    Then apply it on each level of variable value.
   !>
   !> @warning array of value should have been already extrapolated before
   !> running this subroutine.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] dd_value  array of value to be filtered 
   !> @param[in] dd_fill      fill value 
   !> @param[in] cd_name      filter name
   !> @param[in] dd_cutoff    cut-off frequency
   !> @param[in] id_radius    filter halo radius
   !> @param[in] dd_alpha     filter parameter
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp)        , DIMENSION(:,:,:), INTENT(INOUT) :: dd_value
      REAL(dp)        ,                   INTENT(IN   ) :: dd_fill
      CHARACTER(LEN=*),                   INTENT(IN   ) :: cd_name
      REAL(dp)        ,                   INTENT(IN   ) :: dd_cutoff
      INTEGER(i4)     ,                   INTENT(IN   ) :: id_radius
      REAL(dp)        ,                   INTENT(IN   ) :: dd_alpha      

      ! local variable
      INTEGER(i4), DIMENSION(3)                :: il_shape
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_coef

      ! loop indices
      INTEGER(i4) :: jk
      !----------------------------------------------------------------
      
      il_shape(:)=SHAPE(dd_value(:,:,:))

      ALLOCATE( dl_coef(2*id_radius+1,2*id_radius+1) )

      dl_coef(:,:)=filter__2D_coef(cd_name, dd_cutoff, id_radius, dd_alpha)

      DO jk=1,il_shape(3)
         CALL filter__2D(dd_value(:,:,jk), dd_fill,dl_coef(:,:),id_radius)
      ENDDO

      DEALLOCATE( dl_coef )

   END SUBROUTINE filter__3D_fill_value
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE filter__2D_fill_value(dd_value, dd_fill, cd_name, &
         &                          dd_cutoff, id_radius, dd_alpha)
   !-------------------------------------------------------------------
   !> @brief This subroutine compute filtered value of 2D array.
   !>
   !> @details
   !>    First compute filter coefficient.
   !>    Then apply it on variable value.
   !>
   !> @warning array of value should have been already extrapolated before
   !> running this subroutine.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] dd_value  array of value to be filtered 
   !> @param[in] dd_fill      fill value 
   !> @param[in] cd_name      filter name
   !> @param[in] dd_cutoff    cut-off frequency
   !> @param[in] id_radius    filter halo radius
   !> @param[in] dd_alpha     filter parameter
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)        , DIMENSION(:,:), INTENT(INOUT) :: dd_value
      REAL(dp)        ,                 INTENT(IN   ) :: dd_fill
      CHARACTER(LEN=*),                 INTENT(IN   ) :: cd_name
      REAL(dp)        ,                 INTENT(IN   ) :: dd_cutoff
      INTEGER(i4)     ,                 INTENT(IN   ) :: id_radius
      REAL(dp)        ,                 INTENT(IN   ) :: dd_alpha      

      ! local variable

      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: dl_coef
      ! loop indices
      !----------------------------------------------------------------

      ALLOCATE( dl_coef(2*id_radius+1,2*id_radius+1) )

      dl_coef(:,:)=filter__2D_coef(cd_name, dd_cutoff, id_radius, dd_alpha)

      CALL filter__2D(dd_value(:,:), dd_fill, dl_coef(:,:), id_radius)

      DEALLOCATE( dl_coef )

   END SUBROUTINE filter__2D_fill_value
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE filter__1D_fill_value(dd_value, dd_fill, cd_name, &
         &                          dd_cutoff, id_radius, dd_alpha)
   !-------------------------------------------------------------------
   !> @brief This subroutine compute filtered value of 1D array.
   !>
   !> @details
   !>    First compute filter coefficient.
   !>    Then apply it on variable value.
   !>
   !> @warning array of value should have been already extrapolated before
   !> running this subroutine.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] dd_value  array of value to be filtered 
   !> @param[in] dd_fill      fill value 
   !> @param[in] cd_name      filter name
   !> @param[in] dd_cutoff    cut-off frequency
   !> @param[in] id_radius    filter halo radius
   !> @param[in] dd_alpha     filter parameter
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp)        , DIMENSION(:), INTENT(INOUT) :: dd_value
      REAL(dp)        ,               INTENT(IN   ) :: dd_fill
      CHARACTER(LEN=*),               INTENT(IN   ) :: cd_name
      REAL(dp)        ,               INTENT(IN   ) :: dd_cutoff
      INTEGER(i4)     ,               INTENT(IN   ) :: id_radius
      REAL(dp)        ,               INTENT(IN   ) :: dd_alpha      

      ! local variable

      REAL(dp), DIMENSION(:), ALLOCATABLE :: dl_coef
      ! loop indices
      !----------------------------------------------------------------

      ALLOCATE( dl_coef(2*id_radius+1) )

      dl_coef(:)=filter__1D_coef(cd_name, dd_cutoff, id_radius, dd_alpha)

      CALL filter__1D(dd_value(:), dd_fill, dl_coef(:),id_radius)

      DEALLOCATE( dl_coef )

   END SUBROUTINE filter__1D_fill_value
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE filter__2D(dd_value, dd_fill, dd_coef, id_radius)
   !-------------------------------------------------------------------
   !> @brief This subroutine filtered 2D array of value 
   !>
   !> @details
   !>    loop on first and second dimension, 
   !>    and apply coefficient 2D array on each point
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] dd_value  array of value to be filtered 
   !> @param[in] dd_fill      fill value 
   !> @param[in] dd_coef      filter coefficent array
   !> @param[in] id_radius    filter halo radius
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp)        , DIMENSION(:,:), INTENT(INOUT) :: dd_value
      REAL(dp)        ,                 INTENT(IN   ) :: dd_fill 
      REAL(dp)        , DIMENSION(:,:), INTENT(IN   ) :: dd_coef 
      INTEGER(i4)     ,                 INTENT(IN   ) :: id_radius

      ! local variable
      INTEGER(i4), DIMENSION(2)                :: il_shape
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_value
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_halo

      ! loop indices
      INTEGER(i4) :: jj
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      il_shape(:)=SHAPE(dd_value(:,:))

      ALLOCATE(dl_value(il_shape(1),il_shape(2)))
      dl_value(:,:)=dd_value(:,:)

      ALLOCATE(dl_halo(2*id_radius+1,2*id_radius+1))

      DO jj=1+id_radius,il_shape(2)-id_radius
         DO ji=1+id_radius,il_shape(1)-id_radius

            dl_halo(:,:)=dd_fill
            dl_halo(:,:)=dl_value(ji-id_radius:ji+id_radius, &
            &                     jj-id_radius:jj+id_radius)

            dd_value(ji,jj)=SUM(dl_halo(:,:)*dd_coef(:,:))

         ENDDO
      ENDDO

      DEALLOCATE(dl_halo)
      DEALLOCATE(dl_value)

   END SUBROUTINE filter__2D
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE filter__1D(dd_value, dd_fill, dd_coef, id_radius)
   !-------------------------------------------------------------------
   !> @brief This subroutine filtered 1D array of value  
   !>
   !> @details
   !>    loop on first dimension, 
   !>    and apply coefficient 1D array on each point
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] dd_value  array of value to be filtered 
   !> @param[in] dd_fill      fill value 
   !> @param[in] dd_coef      filter coefficent array
   !> @param[in] id_radius    filter halo radius
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp)        , DIMENSION(:), INTENT(INOUT) :: dd_value
      REAL(dp)        ,               INTENT(IN   ) :: dd_fill 
      REAL(dp)        , DIMENSION(:), INTENT(IN   ) :: dd_coef 
      INTEGER(i4)     ,               INTENT(IN   ) :: id_radius

      ! local variable
      INTEGER(i4), DIMENSION(1)              :: il_shape
      REAL(dp)   , DIMENSION(:), ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      il_shape(:)=SHAPE(dd_value(:))

      ALLOCATE(dl_value(2*id_radius+1))

      DO ji=1+id_radius,il_shape(1)-id_radius

         dl_value(:)=dd_fill
         dl_value(:)=dd_value(ji-id_radius:ji+id_radius)

         dd_value(ji)=SUM(dl_value(:)*dd_coef(:))

      ENDDO

      DEALLOCATE(dl_value)

   END SUBROUTINE filter__1D
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION filter__2D_coef(cd_name, dd_cutoff, id_radius, dd_alpha) &
         & RESULT (df_coef)
   !-------------------------------------------------------------------
   !> @brief This function compute filter coefficient. 
   !>
   !> @details
   !> 
   !> filter could be choose between :
   !> - hann
   !> - hamming
   !> - blackman
   !> - gauss
   !> - butterworth
   !> Cut-off frequency could be specify.
   !> As well as a filter parameter for gauss and butterworth filter
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_name   filter name
   !> @param[in] dd_cutoff cut-off frequency
   !> @param[in] id_radius filter halo radius
   !> @param[in] dd_alpha  filter parameter 
   !> @return array of filter coefficient
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=*), INTENT(IN) :: cd_name
      REAL(dp)        , INTENT(IN) :: dd_cutoff
      INTEGER(i4)     , INTENT(IN) :: id_radius
      REAL(dp)        , INTENT(IN) :: dd_alpha

      ! function
      REAL(dp), DIMENSION(2*id_radius+1,2*id_radius+1) :: df_coef

      ! local variable

      ! loop indices
      !----------------------------------------------------------------
      IF( REAL(id_radius,dp) < dd_cutoff )THEN
         CALL logger_warn("FILTER COEF: radius of the filter halo "//&
         &  "is lower than cut-off frequency")
      ENDIF

      SELECT CASE(TRIM(fct_lower(cd_name)))
      CASE('hann')
         df_coef(:,:)=filter__2D_hann(dd_cutoff, id_radius)
      CASE('hamming')
         df_coef(:,:)=filter__2D_hamming(dd_cutoff, id_radius)
      CASE('blackman')
         df_coef(:,:)=filter__2D_blackman(dd_cutoff, id_radius)
      CASE('gauss')
         df_coef(:,:)=filter__2D_gauss(dd_cutoff, id_radius, dd_alpha)
      CASE('butterworth')
         df_coef(:,:)=filter__2D_butterworth(dd_cutoff, id_radius, dd_alpha)
      CASE DEFAULT
         CALL logger_error("FILTER COEF: invalid filter name :"//TRIM(cd_name))
      END SELECT

   END FUNCTION filter__2D_coef
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION filter__1D_coef(cd_name, dd_cutoff, id_radius, dd_alpha) &
         & RESULT (df_coef)
   !-------------------------------------------------------------------
   !> @brief This function compute filter coefficient. 
   !>
   !> @details
   !> 
   !> filter could be choose between :
   !> - hann
   !> - hamming
   !> - blackman
   !> - gauss
   !> - butterworth
   !> Cut-off frequency could be specify.
   !> As well as a filter parameter for gauss an butterworth filter
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_name   filter name
   !> @param[in] dd_cutoff cut-off frequency
   !> @param[in] id_radius filter halo radius
   !> @param[in] dd_alpha  filter parameter 
   !> @return array of filter coefficient
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=*), INTENT(IN) :: cd_name
      REAL(dp)        , INTENT(IN) :: dd_cutoff
      INTEGER(i4)     , INTENT(IN) :: id_radius
      REAL(dp)        , INTENT(IN) :: dd_alpha

      ! function
      REAL(dp), DIMENSION(2*id_radius+1) :: df_coef

      ! local variable

      ! loop indices
      !----------------------------------------------------------------

      SELECT CASE(TRIM(fct_lower(cd_name)))
      CASE('hann')
         df_coef(:)=filter__1D_hann(dd_cutoff, id_radius)
      CASE('hamming')
         df_coef(:)=filter__1D_hamming(dd_cutoff, id_radius)
      CASE('blackman')
         df_coef(:)=filter__1D_blackman(dd_cutoff, id_radius)
      CASE('gauss')
         df_coef(:)=filter__1D_gauss(dd_cutoff, id_radius, dd_alpha)
      CASE('butterworth')
         df_coef(:)=filter__1D_butterworth(dd_cutoff, id_radius, dd_alpha)
      CASE DEFAULT
         CALL logger_error("FILTER COEF: invalid filter name :"//TRIM(cd_name))
      END SELECT

   END FUNCTION filter__1D_coef
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION filter__1D_hann(dd_cutoff, id_radius) &
         & RESULT (df_coef)
   !-------------------------------------------------------------------
   !> @brief This function compute coefficient for HANN filter.
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_cutoff cut-off frequency
   !> @param[in] id_radius filter halo radius
   !> @return array of hann filter coefficient 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp)        , INTENT(IN) :: dd_cutoff 
      INTEGER(i4)     , INTENT(IN) :: id_radius

      ! function
      REAL(dp), DIMENSION(2*id_radius+1) :: df_coef

      ! local variable
      REAL(dp) :: dl_rad
      REAL(dp) :: dl_sum

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( dd_cutoff < 1 )THEN
         CALL logger_error("FILTER COEF: cut-off frequency "//&
         &  "should be greater than or equal to 1. No filter will be apply ")
         df_coef(:)=0.
         df_coef(id_radius+1)=1.
      ELSE
         DO ji=1,2*id_radius+1

            dl_rad=SQRT(REAL(ji-id_radius+1,dp)**2 )
            
            IF( dl_rad < dd_cutoff )THEN
               df_coef(ji)=0.5 + 0.5*COS(dp_pi*dl_rad/dd_cutoff)
            ELSE
               df_coef(ji)=0
            ENDIF

         ENDDO

         ! normalize
         dl_sum=SUM(df_coef(:))

         df_coef(:)=df_coef(:)/dl_sum
      ENDIF

   END FUNCTION filter__1D_hann
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION filter__2D_hann(dd_cutoff, id_radius) &
         & RESULT (df_coef)
   !-------------------------------------------------------------------
   !> @brief This function compute coefficient for HANN filter.
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_cutoff cut-off frequency
   !> @param[in] id_radius filter halo radius
   !> @return array of hann filter coefficient 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp)   , INTENT(IN) :: dd_cutoff 
      INTEGER(i4), INTENT(IN) :: id_radius

      ! function
      REAL(dp), DIMENSION(2*id_radius+1,2*id_radius+1) :: df_coef

      ! local variable
      REAL(dp) :: dl_rad
      REAL(dp) :: dl_sum

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      IF( dd_cutoff < 1.0_dp )THEN
         CALL logger_error("FILTER COEF: cut-off frequency "//&
         &  "should be greater than or equal to 1. No filter will be apply ")
         df_coef(:,:)=0.
         df_coef(id_radius+1,id_radius+1)=1.
      ELSE
         DO jj=1,2*id_radius+1
            DO ji=1,2*id_radius+1

               ! radius
               dl_rad= SQRT( REAL(ji-(id_radius+1),dp)**2 + &
               &             REAL(jj-(id_radius+1),dp)**2 )
               
               IF( dl_rad < dd_cutoff )THEN
                  df_coef(ji,jj)=0.5 + 0.5*COS(dp_pi*dl_rad/dd_cutoff)
               ELSE
                  df_coef(ji,jj)=0
               ENDIF

            ENDDO
         ENDDO

         ! normalize
         dl_sum=SUM(df_coef(:,:))

         df_coef(:,:)=df_coef(:,:)/dl_sum
      ENDIF

   END FUNCTION filter__2D_hann
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION filter__1D_hamming(dd_cutoff, id_radius) &
         & RESULT (df_coef)
   !-------------------------------------------------------------------
   !> @brief This function compute coefficient for HAMMING filter.
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_cutoff cut-off frequency
   !> @param[in] id_radius filter halo radius
   !> @return array of hamming filter coefficient 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp)        , INTENT(IN) :: dd_cutoff 
      INTEGER(i4)     , INTENT(IN) :: id_radius

      ! function
      REAL(dp), DIMENSION(2*id_radius+1) :: df_coef

      ! local variable
      REAL(dp) :: dl_rad
      REAL(dp) :: dl_sum

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( dd_cutoff < 1 )THEN
         CALL logger_error("FILTER COEF: cut-off frequency "//&
         &  "should be greater than or equal to 1. No filter will be apply ")
         df_coef(:)=0.
         df_coef(id_radius+11)=1.
      ELSE
         DO ji=1,2*id_radius+1

            dl_rad= SQRT( REAL(ji-(id_radius+1),dp)**2 )
         
            IF( dl_rad < dd_cutoff )THEN
               df_coef(ji)= 0.54 + 0.46*COS(dp_pi*dl_rad/dd_cutoff)
            ELSE
               df_coef(ji)=0
            ENDIF

         ENDDO

         ! normalize
         dl_sum=SUM(df_coef(:))

         df_coef(:)=df_coef(:)/dl_sum
      ENDIF

   END FUNCTION filter__1D_hamming
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION filter__2D_hamming(dd_cutoff, id_radius) &
         & RESULT (df_coef)
   !-------------------------------------------------------------------
   !> @brief This function compute coefficient for HAMMING filter.
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_cutoff cut-off frequency
   !> @param[in] id_radius filter halo radius
   !> @return array of hamming filter coefficient 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp)        , INTENT(IN) :: dd_cutoff 
      INTEGER(i4)     , INTENT(IN) :: id_radius

      ! function
      REAL(dp), DIMENSION(2*id_radius+1,2*id_radius+1) :: df_coef

      ! local variable
      REAL(dp) :: dl_rad
      REAL(dp) :: dl_sum

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      IF( dd_cutoff < 1 )THEN
         CALL logger_error("FILTER COEF: cut-off frequency "//&
         &  "should be greater than or equal to 1. No filter will be apply ")
         df_coef(:,:)=0.
         df_coef(id_radius+1,id_radius+1)=1.
      ELSE
         DO jj=1,2*id_radius+1
            DO ji=1,2*id_radius+1

               dl_rad= SQRT( REAL(ji-(id_radius+1),dp)**2 + &
                  &          REAL(jj-(id_radius+1),dp)**2 )
            
               IF( dl_rad < dd_cutoff )THEN
                  df_coef(ji,jj)= 0.54 + 0.46*COS(dp_pi*dl_rad/dd_cutoff)
               ELSE
                  df_coef(ji,jj)=0
               ENDIF

            ENDDO
         ENDDO

         ! normalize
         dl_sum=SUM(df_coef(:,:))

         df_coef(:,:)=df_coef(:,:)/dl_sum
      ENDIF

   END FUNCTION filter__2D_hamming
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION filter__1D_blackman(dd_cutoff, id_radius) &
         & RESULT (df_coef)
   !-------------------------------------------------------------------
   !> @brief This function compute coefficient for BLACKMAN filter.
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_cutoff cut-off frequency
   !> @param[in] id_radius filter halo radius
   !> @return array of blackman filter coefficient 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp)        , INTENT(IN) :: dd_cutoff
      INTEGER(i4)     , INTENT(IN) :: id_radius

      ! function
      REAL(dp), DIMENSION(2*id_radius+1) :: df_coef

      ! local variable
      REAL(dp) :: dl_rad
      REAL(dp) :: dl_sum

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( dd_cutoff < 1 )THEN
         CALL logger_error("FILTER COEF: cut-off frequency "//&
         &  "should be greater than or equal to 1. No filter will be apply ")
         df_coef(:)=0.
         df_coef(id_radius+1)=1.
      ELSE      
         DO ji=1,2*id_radius+1

            dl_rad= SQRT( REAL(ji-(id_radius+1),dp)**2 )
            
            IF( dl_rad < dd_cutoff )THEN
               df_coef(ji)= 0.42 + 0.5 *COS(  dp_pi*dl_rad/dd_cutoff) &
                  &              + 0.08*COS(2*dp_pi*dl_rad/dd_cutoff)
            ELSE
               df_coef(ji)=0
            ENDIF                                

         ENDDO

         ! normalize
         dl_sum=SUM(df_coef(:))

         df_coef(:)=df_coef(:)/dl_sum
      ENDIF

   END FUNCTION filter__1D_blackman
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION filter__2D_blackman(dd_cutoff, id_radius) &
         & RESULT (df_coef)
   !-------------------------------------------------------------------
   !> @brief This function compute coefficient for BLACKMAN filter.
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_cutoff cut-off frequency
   !> @param[in] id_radius filter halo radius
   !> @return array of blackman filter coefficient 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp)        , INTENT(IN) :: dd_cutoff 
      INTEGER(i4)     , INTENT(IN) :: id_radius

      ! function
      REAL(dp), DIMENSION(2*id_radius+1,2*id_radius+1) :: df_coef

      ! local variable
      REAL(dp) :: dl_rad
      REAL(dp) :: dl_sum

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      IF( dd_cutoff < 1 )THEN
         CALL logger_error("FILTER COEF: cut-off frequency "//&
         &  "should be greater than or equal to 1. No filter will be apply ")
         df_coef(:,:)=0.
         df_coef(id_radius+1,id_radius+1)=1.
      ELSE      
         DO jj=1,2*id_radius+1
            DO ji=1,2*id_radius+1

               dl_rad= SQRT( REAL(ji-(id_radius+1),dp)**2 + &
               &             REAL(jj-(id_radius+1),dp)**2 )
               
               IF( dl_rad < dd_cutoff )THEN
                  df_coef(ji,jj)= 0.42 + 0.5 *COS(  dp_pi*dl_rad/dd_cutoff) &
                     &                 + 0.08*COS(2*dp_pi*dl_rad/dd_cutoff)
               ELSE
                  df_coef(ji,jj)=0
               ENDIF                                

            ENDDO
         ENDDO

         ! normalize
         dl_sum=SUM(df_coef(:,:))

         df_coef(:,:)=df_coef(:,:)/dl_sum
      ENDIF

   END FUNCTION filter__2D_blackman
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION filter__1D_gauss(dd_cutoff, id_radius, dd_alpha) &
         & RESULT (df_coef)
   !-------------------------------------------------------------------
   !> @brief This function compute coefficient for GAUSS filter.
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_cutoff cut-off frequency
   !> @param[in] id_radius filter halo radius
   !> @param[in] dd_alpha  filter parameter
   !> @return array of gauss filter coefficient 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp)        , INTENT(IN) :: dd_cutoff 
      INTEGER(i4)     , INTENT(IN) :: id_radius
      REAL(dp)        , INTENT(IN) :: dd_alpha 

      ! function
      REAL(dp), DIMENSION(2*id_radius+1) :: df_coef

      ! local variable
      REAL(dp) :: dl_rad
      REAL(dp) :: dl_sum

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( dd_cutoff < 1 )THEN
         CALL logger_error("FILTER COEF: cut-off frequency "//&
         &  "should be greater than or equal to 1. No filter will be apply ")
         df_coef(:)=0.
         df_coef(id_radius+1)=1.
      ELSE
         DO ji=1,2*id_radius+1

            dl_rad= SQRT( REAL(ji-(id_radius+1),dp)**2 )
            
            df_coef(ji)=EXP(-(dd_alpha*dl_rad**2)/(2*dd_cutoff**2))

         ENDDO

         ! normalize
         dl_sum=SUM(df_coef(:))

         df_coef(:)=df_coef(:)/dl_sum
      ENDIF

   END FUNCTION filter__1D_gauss
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION filter__2D_gauss(dd_cutoff, id_radius, dd_alpha) &
         & RESULT (df_coef)
   !-------------------------------------------------------------------
   !> @brief This function compute coefficient for GAUSS filter.
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_cutoff cut-off frequency
   !> @param[in] id_radius filter halo radius
   !> @param[in] dd_alpha  filter parameter
   !> @return array of gauss filter coefficient 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp)        , INTENT(IN) :: dd_cutoff 
      INTEGER(i4)     , INTENT(IN) :: id_radius
      REAL(dp)        , INTENT(IN) :: dd_alpha 

      ! function
      REAL(dp), DIMENSION(2*id_radius+1,2*id_radius+1) :: df_coef

      ! local variable
      REAL(dp) :: dl_rad
      REAL(dp) :: dl_sum

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      IF( dd_cutoff < 1 )THEN
         CALL logger_error("FILTER COEF: cut-off frequency "//&
         &  "should be greater than or equal to 1. No filter will be apply ")
         df_coef(:,:)=0.
         df_coef(id_radius+1,id_radius+1)=1.
      ELSE
         DO jj=1,2*id_radius+1
            DO ji=1,2*id_radius+1

               dl_rad= SQRT( REAL(ji-(id_radius+1),dp)**2 + &
                  &          REAL(jj-(id_radius+1),dp)**2 )
               
               df_coef(ji,jj)=EXP(-(dd_alpha*dl_rad**2)/(2*dd_cutoff**2))

            ENDDO
         ENDDO

         ! normalize
         dl_sum=SUM(df_coef(:,:))

         df_coef(:,:)=df_coef(:,:)/dl_sum
      ENDIF

   END FUNCTION filter__2D_gauss
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION filter__1D_butterworth(dd_cutoff, id_radius, dd_alpha) &
         & RESULT (df_coef)
   !-------------------------------------------------------------------
   !> @brief This function compute coefficient for BUTTERWORTH filter.
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_cutoff cut-off frequency
   !> @param[in] id_radius filter halo radius
   !> @param[in] dd_alpha  filter parameter
   !> @return array of butterworth filter coefficient 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp)        , INTENT(IN) :: dd_cutoff 
      INTEGER(i4)     , INTENT(IN) :: id_radius
      REAL(dp)        , INTENT(IN) :: dd_alpha 

      ! function
      REAL(dp), DIMENSION(2*id_radius+1) :: df_coef

      ! local variable
      REAL(dp) :: dl_rad
      REAL(dp) :: dl_sum

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( dd_cutoff <= 1 )THEN
         CALL logger_error("FILTER COEF: cut-off frequency "//&
         &  "should be greater than 1. No filter will be apply ")
         df_coef(:)=0.
         df_coef(id_radius+1)=1.
      ELSE
         DO ji=1,2*id_radius+1

            dl_rad= SQRT( REAL(ji-(id_radius+1),dp)**2 )
            
            df_coef(ji)= 1 / (1+(dl_rad**2/dd_cutoff**2)**dd_alpha)

         ENDDO

         ! normalize
         dl_sum=SUM(df_coef(:))

         df_coef(:)=df_coef(:)/dl_sum
      ENDIF

   END FUNCTION filter__1D_butterworth
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION filter__2D_butterworth(dd_cutoff,  id_radius, dd_alpha) &
         & RESULT (df_coef)
   !-------------------------------------------------------------------
   !> @brief This function compute coefficient for BUTTERWORTH filter.
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_cutoff cut-off frequency
   !> @param[in] id_radius filter halo radius
   !> @param[in] dd_alpha  filter parameter
   !> @return array of butterworth filter coefficient 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp)        , INTENT(IN) :: dd_cutoff 
      INTEGER(i4)     , INTENT(IN) :: id_radius
      REAL(dp)        , INTENT(IN) :: dd_alpha 

      ! function
      REAL(dp), DIMENSION(2*id_radius+1,2*id_radius+1) :: df_coef

      ! local variable
      REAL(dp) :: dl_rad
      REAL(dp) :: dl_sum

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      IF( dd_cutoff <= 1 )THEN
         CALL logger_error("FILTER COEF: cut-off frequency "//&
         &  "should be greater than 1. No filter will be apply ")
         df_coef(:,:)=0.
         df_coef(id_radius+1,id_radius+1)=1.
      ELSE
         DO jj=1,2*id_radius+1
            DO ji=1,2*id_radius+1

               dl_rad= SQRT( REAL(ji-(id_radius+1),dp)**2 + &
                  &          REAL(jj-(id_radius+1),dp)**2 )
               
               df_coef(ji,jj)= 1 / (1+(dl_rad**2/dd_cutoff**2)**dd_alpha)

            ENDDO
         ENDDO

         ! normalize
         dl_sum=SUM(df_coef(:,:))

         df_coef(:,:)=df_coef(:,:)/dl_sum
      ENDIF

   END FUNCTION filter__2D_butterworth
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE filter

