!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief 
!> This module manage nearest interpolation on regular grid.
!>
!> @details
!> to compute nearest interpolation:<br/>
!> @code
!> CALL interp_nearest_fill(dd_value, dd_fill, id_detect, id_rho, ld_even [,ld_discont] )
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
MODULE interp_nearest

   USE netcdf                          ! nf90 library
   USE global                          ! global variable
   USE kind                            ! F90 kind parameter
   USE logger                          ! log file manager
   USE fct                             ! basic useful function

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! type and variable

   ! function and subroutine
   PUBLIC :: interp_nearest_fill  !< compute interpolation using nearest method

   PRIVATE :: interp_nearest__2D       !< compute binearest interpolation on 2D gid
   PRIVATE :: interp_nearest__1D       !< compute   nearest interpolation on 1D gid
   PRIVATE :: interp_nearest__2D_fill  !< fill value using binearest interpolation
   PRIVATE :: interp_nearest__1D_fill  !< fill value using   nearest interpolation

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp_nearest_fill(dd_value, id_detect, id_rho)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute horizontal nearest interpolation on 4D array of value. 
   !> 
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[inout] dd_value  2D array of variable value 
   !> @param[inout] id_detect 2D array of point to be interpolated 
   !> @param[in]    id_rho    array of refinment factor
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)        , DIMENSION(:,:,:,:), INTENT(INOUT) :: dd_value 
      INTEGER(I4)     , DIMENSION(:,:,:)  , INTENT(INOUT) :: id_detect
      INTEGER(I4)     , DIMENSION(:)      , INTENT(IN   ) :: id_rho

      ! local variable
      INTEGER(i4), DIMENSION(4)                  :: il_shape

      INTEGER(I4), DIMENSION(:,:,:), ALLOCATABLE :: il_detect

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      il_shape(:)=SHAPE(dd_value)

      ALLOCATE(il_detect(il_shape(1),il_shape(2),il_shape(3)))
      DO jl=1,il_shape(4)
         il_detect(:,:,:)=id_detect(:,:,:)
         ! loop on vertical level
         DO jk=1,il_shape(3)

            ! I-J plan
            CALL interp_nearest__2D(dd_value(:,:,jk,jl),&
            &                       il_detect(:,:,jk),  &
            &                       id_rho(jp_I), id_rho(jp_J) )            
            IF( ANY(il_detect(:,:,jk)==1) )THEN
               ! I direction
               DO jj=1,il_shape(2)
                  CALL interp_nearest__1D( dd_value(:,jj,jk,jl),&
                  &                        il_detect(:,jj,jk),  &
                  &                        id_rho(jp_I) )
               ENDDO
               IF( ALL(il_detect(:,:,jk)==0) )THEN
                  CYCLE
               ELSE
                  ! J direction
                  DO ji=1,il_shape(1)
                     CALL interp_nearest__1D( dd_value(ji,:,jk,jl),&
                     &                        il_detect(ji,:,jk),  &
                     &                        id_rho(jp_J) )
                  ENDDO
               ENDIF
            ENDIF

         ENDDO
      ENDDO

      id_detect(:,:,:)=il_detect(:,:,:)
      DEALLOCATE(il_detect)

   END SUBROUTINE interp_nearest_fill
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp_nearest__2D(dd_value, id_detect, id_rhoi, id_rhoj)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute nearest interpolation on 2D array of value. 
   !> 
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[inout] dd_value  2D array of variable value 
   !> @param[inout] id_detect 2D array of point to be interpolated 
   !> @param[in] id_rhoi      refinment factor in i-direction
   !> @param[in] id_rhoj      refinment factor in j-direction
   !> @param[in] id_rhok      refinment factor in k-direction
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)        , DIMENSION(:,:), INTENT(INOUT) :: dd_value 
      INTEGER(I4)     , DIMENSION(:,:), INTENT(INOUT) :: id_detect
      INTEGER(I4)                     , INTENT(IN   ) :: id_rhoi
      INTEGER(I4)                     , INTENT(IN   ) :: id_rhoj

      ! local variable
      INTEGER(i4), DIMENSION(2)                :: il_shape

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj

      !----------------------------------------------------------------

      IF( ANY(id_detect(:,:)==1) )THEN

         il_shape(:)=SHAPE(dd_value)

         DO jj=1,il_shape(2)-1,id_rhoj
            DO ji=1,il_shape(1)-1,id_rhoi
         
               ! check if point to be interpolated
               IF( ALL(id_detect(ji:ji+id_rhoi,   &
               &                 jj:jj+id_rhoj)==0) ) CYCLE

               ! compute value on detetected point
               CALL interp_nearest__2D_fill(dd_value( ji:ji+id_rhoi,   &
               &                                      jj:jj+id_rhoj ), &
               &                            id_detect(ji:ji+id_rhoi,   &
               &                                      jj:jj+id_rhoj ) )

            ENDDO
         ENDDO

      ENDIF

   END SUBROUTINE interp_nearest__2D
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp_nearest__1D(dd_value, id_detect, id_rhoi)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute nearest interpolation on 1D array of value. 
   !> 
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[inout] dd_value  1D array of variable value 
   !> @param[inout] id_detect 1D array of point to be interpolated 
   !> @param[in]    id_rhoi   refinment factor
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)        , DIMENSION(:), INTENT(INOUT) :: dd_value 
      INTEGER(I4)     , DIMENSION(:), INTENT(INOUT) :: id_detect
      INTEGER(I4)                   , INTENT(IN   ) :: id_rhoi

      ! local variable
      INTEGER(i4), DIMENSION(1)              :: il_shape

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( ANY(id_detect(:)==1) )THEN
         il_shape(:)=SHAPE(dd_value)

         DO ji=1,il_shape(1)-1,id_rhoi
         
            ! check if point to be interpolated
            IF( ALL(id_detect(ji:ji+id_rhoi)==0) ) CYCLE

            ! compute value on detetected point
            CALL interp_nearest__1D_fill( dd_value( ji:ji+id_rhoi ), &
            &                             id_detect(ji:ji+id_rhoi ) )

         ENDDO

      ENDIF

   END SUBROUTINE interp_nearest__1D
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp_nearest__2D_fill(dd_value, id_detect)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute nearest interpolation of a 2D array of value. 
   !> 
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[inout] dd_value  2D array of mixed grid value
   !> @param[inout] id_detect 2D array of point to be interpolated
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)   , DIMENSION(:,:)  , INTENT(INOUT) :: dd_value 
      INTEGER(i4), DIMENSION(:,:)  , INTENT(INOUT) :: id_detect

      ! local variable
      INTEGER(i4), DIMENSION(2) :: il_shape

      INTEGER(i4) :: il_i1
      INTEGER(i4) :: il_i2
      INTEGER(i4) :: il_j1
      INTEGER(i4) :: il_j2

      INTEGER(i4) :: il_half1
      INTEGER(i4) :: il_half2

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      il_shape(:)=SHAPE(dd_value(:,:))

      il_i1=1
      il_i2=il_shape(1)

      il_j1=1
      il_j2=il_shape(2)

      il_half1=CEILING(il_shape(1)*0.5)
      il_half2=CEILING(il_shape(2)*0.5)

      DO jj=1,il_half2

         DO ji=1,il_half1
            
            ! lower left point
            IF(id_detect(ji,jj)==1)THEN

               dd_value( ji,jj)=dd_value(il_i1,il_j1)
               id_detect(ji,jj)=0

            ENDIF

            ! lower right point
            IF(id_detect(il_shape(1)-ji+1,jj)==1)THEN

               dd_value( il_shape(1)-ji+1,jj)=dd_value(il_i2,il_j1)
               id_detect(il_shape(1)-ji+1,jj)=0

            ENDIF

            ! upper left point
            IF(id_detect(ji,il_shape(2)-jj+1)==1)THEN

               dd_value( ji,il_shape(2)-jj+1)=dd_value(il_i1,il_j2)
               id_detect(ji,il_shape(2)-jj+1)=0

            ENDIF            

            ! upper right point
            IF(id_detect(il_shape(1)-ji+1,il_shape(2)-jj+1)==1)THEN

               dd_value( il_shape(1)-ji+1,il_shape(2)-jj+1)=dd_value(il_i2,il_j2)
               id_detect(il_shape(1)-ji+1,il_shape(2)-jj+1)=0

            ENDIF            

         ENDDO

      ENDDO

   END SUBROUTINE interp_nearest__2D_fill
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE interp_nearest__1D_fill(dd_value, id_detect)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute nearest interpolation of a 1D array of value. 
   !> 
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[inout] dd_value  1D array of mixed grid value
   !> @param[inout] id_detect 1D array of point to be interpolated
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)   , DIMENSION(:), INTENT(INOUT) :: dd_value 
      INTEGER(i4), DIMENSION(:), INTENT(INOUT) :: id_detect

      ! local variable
      INTEGER(i4), DIMENSION(1) :: il_shape

      INTEGER(i4) :: il_i1
      INTEGER(i4) :: il_i2

      INTEGER(i4) :: il_half1
      
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      il_shape(:)=SHAPE(dd_value)

      il_i1=1
      il_i2=il_shape(1)

      il_half1=CEILING(il_shape(1)*0.5)
      
      DO ji=1,il_half1
         
         ! lower left point
         IF(id_detect(ji)==1)THEN

            dd_value( ji)=dd_value(il_i1)
            id_detect(ji)=0

         ENDIF

         ! lower right point
         IF(id_detect(il_shape(1)-ji+1)==1)THEN

            dd_value( il_shape(1)-ji+1)=dd_value(il_i2)
            id_detect(il_shape(1)-ji+1)=0

         ENDIF

      ENDDO

   END SUBROUTINE interp_nearest__1D_fill
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE interp_nearest
