!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief 
!> This module manage linear interpolation on regular grid.
!>
!> @details
!> to compute linear interpolation:<br/>
!> @code
!> CALL interp_linear_fill(dd_value, dd_fill, id_detect, id_rho, ld_even [,ld_discont] )
!> @endcode
!>    - dd_value is 2D array of variable value 
!>    - dd_fill is the FillValue of variable
!>    - id_detect is 2D array of point to be interpolated (see interp module) 
!>    - id_rho  is array of refinment factor
!>    - ld_even indicates even refinment or not 
!>    - ld_discont indicates longitudinal discontinuity (-180°/180°, 0°/360°) or not
!>
!> @author
!> J.Paul
!>
!> @date September, 2014 - Initial version
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE interp_linear

   USE netcdf                          ! nf90 library
   USE global                          ! global variable
   USE kind                            ! F90 kind parameter
   USE logger                          ! log file manager
   USE fct                             ! basic useful function
   USE extrap                          ! extrapolation manager

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! type and variable

   ! function and subroutine
   PUBLIC :: interp_linear_fill  !< compute interpolation using linear method

   PRIVATE :: interp_linear__2D           !< compute bilinear interpolation on 2D gid
   PRIVATE :: interp_linear__1D           !< compute   linear interpolation on 1D gid
   PRIVATE :: interp_linear__2D_coef      !< compute coefficient for bilinear interpolation
   PRIVATE :: interp_linear__2D_fill      !< fill value using bilinear interpolation
   PRIVATE :: interp_linear__1D_coef      !< compute coefficient for   linear interpolation
   PRIVATE :: interp_linear__1D_fill      !< fill value using   linear interpolation
   PRIVATE :: interp_linear__get_weight2D !< compute interpoaltion weight for 2D array.
   PRIVATE :: interp_linear__get_weight1D !< compute interpoaltion weight for 1D array.

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp_linear_fill(dd_value, dd_fill, id_detect, &
         &                       id_rho, ld_even, ld_discont)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute horizontal linear interpolation on 4D array of value. 
   !> 
   !> @details 
   !>
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !> @date July, 2015 - reinitialise detect array for each level
   !>
   !> @param[inout] dd_value  2D array of variable value 
   !> @param[in] dd_fill      FillValue of variable
   !> @param[inout] id_detect 2D array of point to be interpolated 
   !> @param[in] id_rho       array of refinment factor
   !> @param[in] ld_even      even refinment or not 
   !> @param[in] ld_discont   longitudinal discontinuity (-180°/180°, 0°/360°) or not
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)        , DIMENSION(:,:,:,:), INTENT(INOUT) :: dd_value 
      REAL(dp)                            , INTENT(IN   ) :: dd_fill 
      INTEGER(I4)     , DIMENSION(:,:,:)  , INTENT(INOUT) :: id_detect
      INTEGER(I4)     , DIMENSION(:)      , INTENT(IN   ) :: id_rho
      LOGICAL         , DIMENSION(:)      , INTENT(IN   ) :: ld_even
      LOGICAL                             , INTENT(IN   ), OPTIONAL :: ld_discont

      ! local variable
      INTEGER(i4), DIMENSION(4)                  :: il_shape

      INTEGER(I4), DIMENSION(:,:,:), ALLOCATABLE :: il_detect

      LOGICAL                                    :: ll_discont
 
      REAL(dp)   , DIMENSION(:,:)  , ALLOCATABLE :: dl_weight_IJ
      REAL(dp)   , DIMENSION(:,:)  , ALLOCATABLE :: dl_weight_I
      REAL(dp)   , DIMENSION(:,:)  , ALLOCATABLE :: dl_weight_J
      
      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------
      ll_discont=.FALSE.
      IF( PRESENT(ld_discont) ) ll_discont=ld_discont

      il_shape(:)=SHAPE(dd_value)

      ! compute vect2D
      ALLOCATE(dl_weight_IJ(4,((id_rho(jp_I)+1)*(id_rho(jp_J)+1))) )
      CALL interp_linear__get_weight2D(dl_weight_IJ(:,:), &
      &                               id_rho(:), ld_even(:))

      ALLOCATE( dl_weight_I( 2,((id_rho(jp_I)+1)                 )) )
      ALLOCATE( dl_weight_J( 2,(                 (id_rho(jp_J)+1))) )
      CALL interp_linear__get_weight1D(dl_weight_I(:,:), &
      &                               id_rho(jp_I), ld_even(jp_I))
      CALL interp_linear__get_weight1D(dl_weight_J(:,:), &
      &                               id_rho(jp_J), ld_even(jp_J))

      ALLOCATE(il_detect(il_shape(1),il_shape(2),il_shape(3)))

      DO jl=1,il_shape(4)
         il_detect(:,:,:)=id_detect(:,:,:)
         ! loop on vertical level
         DO jk=1,il_shape(3)

            ! I-J plan
            CALL interp_linear__2D(dd_value(:,:,jk,jl), dd_fill,&
            &                     il_detect(:,:,jk),            &
            &                     dl_weight_IJ(:,:),            &
            &                     id_rho(jp_I), id_rho(jp_J),   &
            &                     ll_discont)            
            IF( ANY(il_detect(:,:,jk)==1) )THEN
               ! I direction
               DO jj=1,il_shape(2)
                  CALL interp_linear__1D( dd_value(:,jj,jk,jl), dd_fill,&
                  &                       il_detect(:,jj,jk),           &
                  &                       dl_weight_I(:,:),             &
                  &                       id_rho(jp_I), ll_discont )
               ENDDO
               IF( ALL(il_detect(:,:,jk)==0) )THEN
                  CYCLE
               ELSE
                  ! J direction
                  DO ji=1,il_shape(1)
                     CALL interp_linear__1D( dd_value(ji,:,jk,jl), dd_fill,&
                     &                       il_detect(ji,:,jk),           &
                     &                       dl_weight_J(:,:),             &
                     &                       id_rho(jp_J), ll_discont )
                  ENDDO
               ENDIF
            ENDIF

         ENDDO
      ENDDO

      id_detect(:,:,:)=il_detect(:,:,:)
      DEALLOCATE(il_detect)

      DEALLOCATE(dl_weight_IJ)
      DEALLOCATE(dl_weight_I)
      DEALLOCATE(dl_weight_J)
 
   END SUBROUTINE interp_linear_fill
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp_linear__2D(dd_value,  dd_fill,   &
         &                      id_detect,            &
         &                      dd_weight,            &
         &                      id_rhoi, id_rhoj,     &
         &                      ld_discont)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute linear interpolation on 2D array of value. 
   !> 
   !> @details 
   !>
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[inout] dd_value  2D array of variable value 
   !> @param[in] dd_fill      FillValue of variable
   !> @param[inout] id_detect 2D array of point to be interpolated 
   !> @param[in] id_rhoi      refinment factor in i-direction
   !> @param[in] id_rhoj      refinment factor in j-direction
   !> @param[in] id_rhok      refinment factor in k-direction
   !> @param[in] ld_even      even refinment or not 
   !> @param[in] ld_discont   longitudinal discontinuity (-180°/180°, 0°/360°) or not
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)        , DIMENSION(:,:), INTENT(INOUT) :: dd_value 
      REAL(dp)                        , INTENT(IN   ) :: dd_fill 
      INTEGER(I4)     , DIMENSION(:,:), INTENT(INOUT) :: id_detect
      REAL(dp)        , DIMENSION(:,:), INTENT(IN   ) :: dd_weight
      INTEGER(I4)                     , INTENT(IN   ) :: id_rhoi
      INTEGER(I4)                     , INTENT(IN   ) :: id_rhoj
      LOGICAL                         , INTENT(IN   ) :: ld_discont

      ! local variable
      INTEGER(I4)                              :: il_xextra
      INTEGER(I4)                              :: il_yextra
      INTEGER(i4), DIMENSION(2)                :: il_shape
      INTEGER(i4), DIMENSION(2)                :: il_dim

      REAL(dp)                                 :: dl_min 
      REAL(dp)                                 :: dl_max 
      REAL(dp)   , DIMENSION(:)  , ALLOCATABLE :: dl_coef
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_coarse
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_tmp

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: ii
      INTEGER(i4) :: ij

      !----------------------------------------------------------------

      IF( ANY(id_detect(:,:)==1) )THEN
         il_shape(:)=SHAPE(dd_value)

         ! compute coarse grid dimension
         il_xextra=id_rhoi-1
         il_dim(1)=(il_shape(1)+il_xextra)/id_rhoi

         il_yextra=id_rhoj-1
         il_dim(2)=(il_shape(2)+il_yextra)/id_rhoj

         ALLOCATE( dl_coarse(il_dim(1),il_dim(2)) )

         ! value on coarse grid
         dl_coarse(:,:)=dd_value( 1:il_shape(1):id_rhoi, &
         &                        1:il_shape(2):id_rhoj )

         ALLOCATE( dl_tmp(2,2) )
         ALLOCATE( dl_coef(4) )

         DO jj=1,il_shape(2)-1,id_rhoj
            ij=((jj-1)/id_rhoj)+1
            DO ji=1,il_shape(1)-1,id_rhoi
               ii=((ji-1)/id_rhoi)+1
         
               ! check if point to be interpolated
               IF( ALL(id_detect(ji:ji+id_rhoi,   &
               &                 jj:jj+id_rhoj)==0) ) CYCLE
               ! check data needed to interpolate
               IF( ANY(dl_coarse(ii:ii+1,ij:ij+1)==dd_fill) ) CYCLE
               ! check longitude discontinuity
               dl_tmp(:,:)=dl_coarse(ii:ii+1,ij:ij+1)
               IF( ld_discont )THEN

                  dl_min=MINVAL( dl_tmp(:,:), dl_tmp(:,:)/=dd_fill )
                  dl_max=MAXVAL( dl_tmp(:,:), dl_tmp(:,:)/=dd_fill )
                  IF( dl_min < -170_dp .AND. dl_max > 170_dp )THEN
                     WHERE( dl_tmp(:,:) < 0_dp ) 
                        dl_tmp(:,:) = dl_tmp(:,:)+360._dp
                     END WHERE
                  ELSEIF( dl_min < 10_dp .AND. dl_max > 350_dp )THEN
                     WHERE( dl_tmp(:,:) > 180_dp ) 
                        dl_tmp(:,:) = dl_tmp(:,:)-180._dp
                     END WHERE
                  ENDIF

               ENDIF

               ! compute bilinear coefficient
               dl_coef(:)=interp_linear__2D_coef(dl_tmp(:,:),&
               &                                 dd_fill )

               ! compute value on detetected point
               CALL interp_linear__2D_fill(dd_value( ji:ji+id_rhoi,   &
               &                                     jj:jj+id_rhoj ), &
               &                           id_detect(ji:ji+id_rhoi,   &
               &                                     jj:jj+id_rhoj ), &
               &                           dd_weight(:,:), dl_coef(:),&
               &                           dd_fill, id_rhoi, id_rhoj )

               IF( ld_discont )THEN
                  WHERE( dd_value( ji:ji+id_rhoi, &
                     &             jj:jj+id_rhoj ) >= 180._dp .AND. &
                     &   dd_value( ji:ji+id_rhoi, &
                     &             jj:jj+id_rhoj ) /= dd_fill )
                     dd_value( ji:ji+id_rhoi, &
                     &         jj:jj+id_rhoj ) = &
                     &           dd_value( ji:ji+id_rhoi, &
                     &                     jj:jj+id_rhoj ) - 360._dp
                  END WHERE
               ENDIF

            ENDDO
         ENDDO

         DEALLOCATE(dl_coef)
         DEALLOCATE(dl_tmp )

         DEALLOCATE( dl_coarse )
      ENDIF

   END SUBROUTINE interp_linear__2D
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp_linear__1D(dd_value,  dd_fill,   &
         &                      id_detect,            &
         &                      dd_weight,            &
         &                      id_rhoi,              &
         &                      ld_discont)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute linear interpolation on 1D array of value. 
   !> 
   !> @details 
   !>
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[inout] dd_value  1D array of variable value 
   !> @param[in] dd_fill      FillValue of variable
   !> @param[inout] id_detect 1D array of point to be interpolated 
   !> @param[in] id_rhoi      refinment factor
   !> @param[in] ld_even      even refinment or not 
   !> @param[in] ld_discont   longitudinal discontinuity (-180°/180°, 0°/360°) or not
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)        , DIMENSION(:)  , INTENT(INOUT) :: dd_value 
      REAL(dp)                        , INTENT(IN   ) :: dd_fill 
      INTEGER(I4)     , DIMENSION(:)  , INTENT(INOUT) :: id_detect
      REAL(dp)        , DIMENSION(:,:), INTENT(IN   ) :: dd_weight
      INTEGER(I4)                     , INTENT(IN   ) :: id_rhoi
      LOGICAL                         , INTENT(IN   ) :: ld_discont

      ! local variable
      INTEGER(I4)                            :: il_xextra
      INTEGER(i4), DIMENSION(1)              :: il_shape
      INTEGER(i4), DIMENSION(1)              :: il_dim

      REAL(dp)                               :: dl_min 
      REAL(dp)                               :: dl_max 
      REAL(dp)   , DIMENSION(:), ALLOCATABLE :: dl_coef
      REAL(dp)   , DIMENSION(:), ALLOCATABLE :: dl_coarse
      REAL(dp)   , DIMENSION(:), ALLOCATABLE :: dl_tmp

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: ii

      !----------------------------------------------------------------

      IF( ANY(id_detect(:)==1) )THEN
         il_shape(:)=SHAPE(dd_value)

         ! compute coarse grid dimension
         il_xextra=id_rhoi-1
         il_dim(1)=(il_shape(1)+il_xextra)/id_rhoi

         ALLOCATE( dl_coarse(il_dim(1)) )

         ! value on coarse grid
         dl_coarse(:)=dd_value( 1:il_shape(1):id_rhoi )

         ALLOCATE( dl_tmp(2) )
         ALLOCATE( dl_coef(4) )

         DO ji=1,il_shape(1)-1,id_rhoi
            ii=((ji-1)/id_rhoi)+1
         
            ! check if point to be interpolated
            IF( ALL(id_detect(ji:ji+id_rhoi)==0) ) CYCLE
            ! check data needed to interpolate
            IF( ANY(dl_coarse(ii:ii+1)==dd_fill) ) CYCLE
            ! check longitude discontinuity
            dl_tmp(:)=dl_coarse(ii:ii+1)
            IF( ld_discont )THEN

               dl_min=MINVAL( dl_tmp(:), dl_tmp(:)/=dd_fill )
               dl_max=MAXVAL( dl_tmp(:), dl_tmp(:)/=dd_fill )
               IF( dl_min < -170_dp .AND. dl_max > 170_dp )THEN
                  WHERE( dl_tmp(:) < 0_dp ) 
                     dl_tmp(:) = dl_tmp(:)+360._dp
                  END WHERE
               ELSEIF( dl_min < 10_dp .AND. dl_max > 350_dp )THEN
                  WHERE( dl_tmp(:) > 180_dp ) 
                     dl_tmp(:) = dl_tmp(:)-180._dp
                  END WHERE
               ENDIF

            ENDIF

            ! compute bilinear coefficient
            dl_coef(:)=interp_linear__1D_coef(dl_tmp(:),       &
            &                                 dd_fill )

            ! compute value on detetected point
            CALL interp_linear__1D_fill( dd_value( ji:ji+id_rhoi ), &
            &                            id_detect(ji:ji+id_rhoi ), &
            &                            dd_weight(:,:), dl_coef(:),&
            &                            dd_fill, id_rhoi )

            IF( ld_discont )THEN
               WHERE( dd_value( ji:ji+id_rhoi ) >= 180._dp .AND. &
                  &   dd_value( ji:ji+id_rhoi ) /= dd_fill )
                  dd_value(ji:ji+id_rhoi) = dd_value(ji:ji+id_rhoi) - 360._dp
               END WHERE
            ENDIF

         ENDDO

         DEALLOCATE(dl_coef)
         DEALLOCATE(dl_tmp )

         DEALLOCATE( dl_coarse )
      ENDIF

   END SUBROUTINE interp_linear__1D
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION interp_linear__2D_coef(dd_value, dd_fill) &
         & RESULT (df_coef)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute 2D array of coefficient for linear interpolation. 
   !> 
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[in] dd_value  2D array of value
   !> @param[in] dd_fill   FillValue of variable
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:,:)  , INTENT(IN) :: dd_value 
      REAL(dp)                  , INTENT(IN) :: dd_fill 

      ! function
      REAL(dp), DIMENSION(4)                 :: df_coef

      ! local variable
      REAL(dp), DIMENSION(4,4), PARAMETER :: dl_matrix = RESHAPE( &
      & (/  1 ,-1 ,-1 , 1 ,& 
            0 , 1 , 0 ,-1 ,& 
            0 , 0 , 1 ,-1 ,& 
            0 , 0 , 0 , 1 /), &
      & (/ 4, 4 /) )

      REAL(dp), DIMENSION(4) :: dl_vect

      !----------------------------------------------------------------
      ! init
      df_coef(:)=dd_fill

      dl_vect( 1: 4)=PACK(dd_value(:,:),.TRUE. )
      df_coef(:)=MATMUL(dl_matrix(:,:),dl_vect(:))

   END FUNCTION interp_linear__2D_coef
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp_linear__2D_fill(dd_value, id_detect, &
         &                           dd_weight, dd_coef,  &
         &                           dd_fill, id_rhoi, id_rhoj)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute linear interpolation of a 2D array of value. 
   !> 
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[inout] dd_value  2D array of mixed grid value
   !> @param[inout] id_detect 2D array of point to be interpolated
   !> @param[in] dd_coef      2D array of coefficient
   !> @param[in] dd_fill      FillValue of variable
   !> @param[in] ld_even      even refinment or not
   !> @param[in] id_rhoi      refinement factor in i-direction 
   !> @param[in] id_rhoj      refinement factor in j-direction
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)        , DIMENSION(:,:), INTENT(INOUT) :: dd_value 
      INTEGER(i4)     , DIMENSION(:,:), INTENT(INOUT) :: id_detect
      REAL(dp)        , DIMENSION(:,:), INTENT(IN   ) :: dd_weight
      REAL(dp)        , DIMENSION(:)  , INTENT(IN   ) :: dd_coef 
      REAL(dp)                        , INTENT(IN   ) :: dd_fill 
      INTEGER(I4)     ,                 INTENT(IN   ) :: id_rhoi
      INTEGER(I4)     ,                 INTENT(IN   ) :: id_rhoj

      ! local variable

      ! loop indices
      INTEGER(i4) :: ii

      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      IF( ANY( dd_coef(:)==dd_fill ) )THEN
         CALL logger_error("INTERP LINEAR FILL: fill value detected in coef. "//&
         &              "can not compute interpolation.")
      ELSE

         ii=0
         DO jj=1,id_rhoj+1
            DO ji=1,id_rhoi+1

               ii=ii+1
               IF(id_detect(ji,jj)==1)THEN

                  dd_value(ji,jj)=DOT_PRODUCT(dd_coef(:),dd_weight(:,ii))
                  id_detect(ji,jj)=0

               ENDIF

            ENDDO
         ENDDO

      ENDIF

   END SUBROUTINE interp_linear__2D_fill
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION interp_linear__1D_coef(dd_value, dd_fill) &
         & RESULT (df_coef)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute 1D array of coefficient for linear interpolation. 
   !> 
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[in] dd_value  1D array of value
   !> @param[in] dd_fill   FillValue of variable
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:)  , INTENT(IN) :: dd_value 
      REAL(dp)                , INTENT(IN) :: dd_fill  

      ! function
      REAL(dp), DIMENSION(2)               :: df_coef

      ! local variable
      REAL(dp), DIMENSION(2,2), PARAMETER :: dl_matrix = RESHAPE( &
      & (/  1 ,-1 ,& 
            0 , 1  /), &
      &  (/ 2, 2 /) )

      REAL(dp), DIMENSION(2) :: dl_vect

      !----------------------------------------------------------------
      ! init
      df_coef(:)=dd_fill

      dl_vect( 1: 2)=PACK(dd_value(:),.TRUE. )
      df_coef(:)=MATMUL(dl_matrix(:,:),dl_vect(:))

   END FUNCTION interp_linear__1D_coef
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp_linear__1D_fill(dd_value, id_detect, &
         &                           dd_weight, dd_coef,  &
         &                           dd_fill, id_rho)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute linear interpolation of a 1D array of value. 
   !> 
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !> 
   !> @param[inout] dd_value  1D array of mixed grid value
   !> @param[inout] id_detect 1D array of point to be interpolated
   !> @param[in] dd_coef      1D array of coefficient
   !> @param[in] dd_fill      FillValue of variable
   !> @param[in] ld_even      even refinment or not
   !> @param[in] id_rho       refinement factor
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)        , DIMENSION(:)  , INTENT(INOUT) :: dd_value 
      INTEGER(i4)     , DIMENSION(:)  , INTENT(INOUT) :: id_detect
      REAL(dp)        , DIMENSION(:,:), INTENT(IN   ) :: dd_weight
      REAL(dp)        , DIMENSION(:)  , INTENT(IN   ) :: dd_coef 
      REAL(dp)                        , INTENT(IN   ) :: dd_fill 
      INTEGER(I4)                     , INTENT(IN   ) :: id_rho      

      ! local variable

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( ANY( dd_coef(:)==dd_fill ) )THEN
         CALL logger_error("INTERP LINEAR FILL: fill value detected. "//&
         &              "can not compute interpolation")
      ELSE

         DO ji=1,id_rho+1
            
            IF(id_detect(ji)==1)THEN

               dd_value(ji)=DOT_PRODUCT(dd_coef(:),dd_weight(:,ji))
               id_detect(ji)=0

            ENDIF

         ENDDO

      ENDIF

   END SUBROUTINE interp_linear__1D_fill
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp_linear__get_weight2D(dd_weight, id_rho, ld_even)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute interpoaltion weight for 2D array. 
   !> 
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[in] dd_weight interpolation weight of 2D array
   !> @param[in] ld_even   even refinment or not
   !> @param[in] id_rho    refinement factor 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument

      REAL(dp)   , DIMENSION(:,:), INTENT(INOUT) :: dd_weight
      INTEGER(I4), DIMENSION(:)  , INTENT(IN   ) :: id_rho
      LOGICAL    , DIMENSION(:)  , INTENT(IN   ) :: ld_even

      ! local variable
      REAL(dp)                  :: dl_dx
      REAL(dp)                  :: dl_x      
      REAL(dp)                  :: dl_dy
      REAL(dp)                  :: dl_y      
      
      ! loop indices
      INTEGER(i4) :: ii 

      INTEGER(i4) :: ji 
      INTEGER(i4) :: jj
      !----------------------------------------------------------------
      
      IF( ld_even(jp_I) )THEN
         dl_dx=1._dp/REAL(id_rho(jp_I)-1,dp)
      ELSE ! odd refinement
         dl_dx=1._dp/REAL(id_rho(jp_I),dp)
      ENDIF

      IF( ld_even(jp_J) )THEN
         dl_dy=1._dp/REAL(id_rho(jp_J)-1,dp)
      ELSE ! odd refinement
         dl_dy=1._dp/REAL(id_rho(jp_J),dp)
      ENDIF

      ii=0
      DO jj=1,id_rho(jp_J)+1

         IF( ld_even(jp_J) )THEN
            dl_y=REAL(jj-1,dp)*dl_dy - dl_dy*0.5_dp
         ELSE ! odd refinement
            dl_y=REAL(jj-1,dp)*dl_dy 
         ENDIF

         DO ji=1,id_rho(jp_I)+1

            ! iter
            ii=ii+1

            IF( ld_even(jp_I) )THEN
               dl_x=REAL(ji-1,dp)*dl_dx - dl_dx*0.5_dp 
            ELSE ! odd refinement
               dl_x=REAL(ji-1,dp)*dl_dx 
            ENDIF

            dd_weight(:,ii)=(/1._dp, dl_x, dl_y, dl_x*dl_y /)

         ENDDO
      ENDDO

   END SUBROUTINE interp_linear__get_weight2D
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp_linear__get_weight1D(dd_weight, id_rho, ld_even)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute interpoaltion weight for 1D array. 
   !> 
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[in] dd_weight interpolation weight of 1D array
   !> @param[in] ld_even   even refinment or not
   !> @param[in] id_rho    refinement factor 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)   , DIMENSION(:,:), INTENT(INOUT) :: dd_weight
      INTEGER(I4)                , INTENT(IN   ) :: id_rho
      LOGICAL                    , INTENT(IN   ) :: ld_even

      ! local variable
      REAL(dp)                  :: dl_dx
      REAL(dp)                  :: dl_x      

      ! loop indices
      INTEGER(i4) :: ji   
      !----------------------------------------------------------------
      
      IF( ld_even )THEN
         dl_dx=1._dp/REAL(id_rho-1,dp)
      ELSE ! odd refinement
         dl_dx=1._dp/REAL(id_rho,dp)
      ENDIF

      DO ji=1,id_rho+1
         IF( ld_even )THEN
            dl_x=REAL(ji-1,dp)*dl_dx - dl_dx*0.5_dp 
         ELSE ! odd refinement
            dl_x=REAL(ji-1,dp)*dl_dx 
         ENDIF

         dd_weight(:,ji)=(/1._dp, dl_x /)
      ENDDO

   END SUBROUTINE interp_linear__get_weight1D
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE interp_linear
