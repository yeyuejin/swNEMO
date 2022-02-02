!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief
!> This module groups some useful mathematical function.
!>
!> @details
!>
!>    to compute the mean of an array:<br/>
!> @code
!>    dl_value=math_mean( dl_value, dd_fill )
!> @endcode
!>       - dl_value is 1D or 2D array
!>       - dd_fill is FillValue
!>
!>    to compute the median of an array:<br/>
!> @code
!>    dl_value=math_median( dl_value, dd_fill )
!> @endcode
!>       - dl_value is 1D or 2D array
!>       - dd_fill is FillValue
!>
!>    to compute the mean without extremum of an array:<br/>
!> @code
!>    dl_value=math_mwe( dl_value, id_next, dd_fill )
!> @endcode
!>       - dl_value is 1D or 2D array
!>       - id_next is the number of extremum to be removed
!>       - dd_fill is FillValue
!>
!>    to sort an 1D array:<br/>
!> @code
!>    CALL math_QsortC(dl_value)
!> @endcode
!>       - dl_value is 1D array
!>
!>    to correct phase angles to produce smoother phase:<br/>
!> @code
!>    CALL math_unwrap(dl_value, [dl_discont])
!> @endcode
!>       - dl_value is 1D array
!>       - dl_discont maximum discontinuity between values, default pi
!>
!>    to compute simple operation
!> @code
!>    dl_res=math_compute(cl_var)
!> @endcode
!>       - cl_var operation to compute (string of character)
!>       - dl_res result of the operation, real(dp)
!>
!>    to compute first derivative of 1D array:<br/>
!> @code
!>    dl_value(:)=math_deriv_1D( dd_value(:), dd_fill, [ld_discont] )
!> @endcode
!>       - dd_value is 1D array of variable
!>       - dd_fill is FillValue of variable
!>       - ld_discont is logical to take into account longitudinal 
!>         East-West discontinuity [optional]
!>
!>    to compute first derivative of 2D array:<br/>
!> @code
!>    dl_value(:,:)=math_deriv_2D( dd_value(:,:), dd_fill, cd_dim, 
!>                  [ld_discont] )
!> @endcode
!>       - dd_value is 2D array of variable
!>       - dd_fill is FillValue of variable
!>       - cd_dim is character to compute derivative on first (I) or
!>         second (J) dimension
!>       - ld_discont is logical to take into account longitudinal 
!>         East-West discontinuity [optional]
!>
!>    to compute first derivative of 3D array:<br/>
!> @code
!>    dl_value(:,:,:)=math_deriv_3D( dd_value(:,:,:), dd_fill, cd_dim,
!>                    [ld_discont] )
!> @endcode
!>       - dd_value is 3D array of variable
!>       - dd_fill is FillValue of variable
!>       - cd_dim is character to compute derivative on first (I), second (J),
!>         or third (K) dimension
!>       - ld_discont is logical to take into account longitudinal East-West 
!>         discontinuity [optional]
!>
!>
!> @author
!> J.Paul
!>
!> @date January, 2015 - Initial version
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE math

   USE kind                            ! F90 kind parameter
   USE global                          ! global variable
   USE phycst                          ! physical constant
   USE fct                             ! basic useful function
   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! function and subroutine
   PUBLIC :: math_mean      !< return mean of an array
   PUBLIC :: math_median    !< return median of an array
   PUBLIC :: math_mwe       !< return mean without extremum of an array
   PUBLIC :: math_QsortC    !< sort an 1D array
   PUBLIC :: math_unwrap    !< correct phase angles to produce smoother phase
   PUBLIC :: math_compute   !< compute simple operation
   PUBLIC :: math_deriv_1D  !< compute first derivative of 1D array 
   PUBLIC :: math_deriv_2D  !< compute first derivative of 2D array 
   PUBLIC :: math_deriv_3D  !< compute first derivative of 3D array
   PUBLIC :: math_ortho     !< compute orthodome distance
   PUBLIC :: math_euclid    !< compute euclidian distance

   PRIVATE :: math__Partition
   PRIVATE :: math__mean_1d
   PRIVATE :: math__mean_2d
   PRIVATE :: math__median_1d
   PRIVATE :: math__median_2d
   PRIVATE :: math__mwe_1d
   PRIVATE :: math__mwe_2d
   PRIVATE :: math__parentheses


   INTERFACE math_mean
      MODULE PROCEDURE math__mean_1d ! return mean of an array 1D
      MODULE PROCEDURE math__mean_2d ! return mean of an array 2D
   END INTERFACE math_mean

   INTERFACE math_median
      MODULE PROCEDURE math__median_1d ! return median of an array 1D
      MODULE PROCEDURE math__median_2d ! return median of an array 2D
   END INTERFACE math_median

   INTERFACE math_mwe
      MODULE PROCEDURE math__mwe_1d ! return mean without extremum of an array 1D
      MODULE PROCEDURE math__mwe_2d ! return mean without extremum of an array 2D
   END INTERFACE math_mwe

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION math__mean_1d(dd_array, dd_fill) &
         & RESULT (df_mean)
   !------------------------------------------------------------------- 
   !> @brief This function compute the mean of a 1D array. 
   !>
   !> @author J.Paul 
   !> @date January, 2015 - Initial Version 
   !>
   !> @param[in] dd_array  1D array 
   !> @param[in] dd_fill   fillValue
   !> @return mean value, real(dp)
   !------------------------------------------------------------------- 

      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:), INTENT(IN) :: dd_array
      REAL(dp),               INTENT(IN), OPTIONAL :: dd_fill

      ! function
      REAL(dp)                           :: df_mean

      ! local variable
      INTEGER(i4)  :: il_count
      REAL(dp)     :: dl_sum
      REAL(dp)     :: dl_count
      !----------------------------------------------------------------
      
      IF( PRESENT(dd_fill) )THEN
         il_count=COUNT(dd_array(:)/=dd_fill)
         IF( il_count > 0 )THEN
            dl_sum  =SUM ( dd_array(:), dd_array(:)/= dd_fill )
            dl_count=REAL( il_count, dp)

            df_mean=dl_sum/dl_count
         ELSE
            df_mean=dd_fill
         ENDIF
      ELSE
         il_count=SIZE(dd_array(:))
         IF( il_count > 0 )THEN
            dl_sum  =SUM ( dd_array(:) )
            dl_count=REAL( il_count, dp)

            df_mean=dl_sum/dl_count
         ELSE
            df_mean=0
         ENDIF
      ENDIF

   END FUNCTION math__mean_1d
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION math__mean_2d(dd_array, dd_fill) &
         & RESULT (df_mean)
   !------------------------------------------------------------------- 
   !> @brief This function compute the mean of a 2D array. 
   !>
   !> @author J.Paul 
   !> @date January, 2015 - Initial Version 
   !>
   !> @param[in] dd_array  2D array 
   !> @param[in] dd_fill   fillValue
   !> @return mean value, real(dp)
   !------------------------------------------------------------------- 
      
      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:,:), INTENT(IN) :: dd_array
      REAL(dp),                 INTENT(IN), OPTIONAL :: dd_fill

      ! function
      REAL(dp)                             :: df_mean

      ! local variable
      INTEGER(i4)                         :: il_count

      REAL(dp), DIMENSION(:), ALLOCATABLE :: dl_list
      !----------------------------------------------------------------

      IF( PRESENT(dd_fill)  )THEN
         il_count=COUNT(dd_array(:,:)/=dd_fill)
         IF( il_count > 0 )THEN
            ALLOCATE( dl_list(il_count) )
            dl_list(:)=PACK(dd_array(:,:),dd_array(:,:)/=dd_fill)
         ELSE
            df_mean=dd_fill
         ENDIF
      ELSE
         il_count=SIZE(dd_array)
         IF( il_count > 0 )THEN
            ALLOCATE( dl_list(il_count) )
            dl_list(:)=PACK(dd_array(:,:), MASK=.TRUE.)
         ELSE
            df_mean=0
         ENDIF
      ENDIF

      IF( ALLOCATED(dl_list) )THEN
         df_mean=math_mean(dl_list(:))
         DEALLOCATE( dl_list )
      ENDIF

   END FUNCTION math__mean_2d
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION math__median_1d(dd_array, dd_fill) &
         & RESULT (df_median)
   !------------------------------------------------------------------- 
   !> @brief This function compute the median of a 1D array. 
   !>
   !> @author J.Paul 
   !> @date January, 2015 - Initial Version 
   !>
   !> @param[in] dd_array  1D array 
   !> @param[in] dd_fill   fillValue
   !> @return median value, real(dp)
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:), INTENT(IN) :: dd_array
      REAL(dp),               INTENT(IN), OPTIONAL :: dd_fill

      ! function
      REAL(dp)                           :: df_median

      ! local variable
      INTEGER(i4)                         :: il_count

      REAL(dp), DIMENSION(:), ALLOCATABLE :: dl_list
      !----------------------------------------------------------------
      
      IF( PRESENT(dd_fill)  )THEN
         il_count=COUNT(dd_array(:)/=dd_fill)
         IF( il_count > 0 )THEN
            ALLOCATE( dl_list(il_count) )
            dl_list(:)=PACK(dd_array(:),dd_array(:)/=dd_fill)
         ELSE
            df_median=dd_fill
         ENDIF
      ELSE
         il_count=SIZE(dd_array(:))
         IF( il_count > 0 )THEN
            ALLOCATE( dl_list(il_count) )
            dl_list(:)=dd_array(:)
         ELSE
            df_median=0
         ENDIF
      ENDIF

      IF( ALLOCATED(dl_list) )THEN
         CALL math_QsortC(dl_list(:))

         IF( MOD(il_count,2) == 0 )THEN
            df_median=(dl_list(il_count/2+1)+dl_list(il_count/2))/2_dp
         ELSE
            df_median=dl_list(il_count/2+1)
         ENDIF

         DEALLOCATE(dl_list)
      ENDIF

   END FUNCTION math__median_1d
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION math__median_2d(dd_array, dd_fill) &
         & RESULT (df_median)
   !------------------------------------------------------------------- 
   !> @brief This function compute the median of a 2D array. 
   !> 
   !> @author J.Paul 
   !> @date January, 2015 - Initial Version 
   !>
   !> @param[in] dd_array  2D array 
   !> @param[in] dd_fill   fillValue
   !> @return median value, real(dp)
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:,:), INTENT(IN) :: dd_array
      REAL(dp),                 INTENT(IN), OPTIONAL :: dd_fill

      ! funtion
      REAL(dp)                             :: df_median
      
      ! local variable
      INTEGER(i4)                         :: il_count

      REAL(dp), DIMENSION(:), ALLOCATABLE :: dl_list
      !----------------------------------------------------------------
      
      IF( PRESENT(dd_fill)  )THEN
         il_count=COUNT(dd_array(:,:)/=dd_fill)
         IF( il_count > 0 )THEN
            ALLOCATE( dl_list(il_count) )
            dl_list(:)=PACK(dd_array(:,:),dd_array(:,:)/=dd_fill)
         ELSE
            df_median=dd_fill
         ENDIF
      ELSE
         il_count=SIZE(dd_array(:,:))
         IF( il_count > 0 )THEN
            ALLOCATE( dl_list(il_count) )
            dl_list(:)=PACK(dd_array(:,:), MASK=.TRUE.)
         ELSE
            df_median=0
         ENDIF
      ENDIF

      IF( ALLOCATED(dl_list) )THEN
         df_median=math_median(dl_list(:))
         DEALLOCATE(dl_list)
      ENDIF

   END FUNCTION math__median_2d
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION math__mwe_1d(dd_array, id_next, dd_fill) &
         & RESULT (df_mwe)
   !------------------------------------------------------------------- 
   !> @brief This function compute the mean without extremum of a 1D array. 
   !> 
   !> @author J.Paul 
   !> @date January, 2015 - Initial Version 
   !>
   !> @param[in] dd_array  1D array 
   !> @param[in] id_next   number of extremum to be removed
   !> @param[in] dd_fill   fillValue
   !> @return median value, real(dp)
   !-------------------------------------------------------------------
      
      IMPLICIT NONE
      
      ! Argument
      REAL(dp), DIMENSION(:), INTENT(IN) :: dd_array
      INTEGER(i4)           , INTENT(IN), OPTIONAL :: id_next
      REAL(dp),               INTENT(IN), OPTIONAL :: dd_fill

      ! function
      REAL(dp)                           :: df_mwe

      ! local variable
      INTEGER(i4)                         :: il_next
      INTEGER(i4)                         :: il_count
      INTEGER(i4)                         :: il_size

      REAL(dp), DIMENSION(:), ALLOCATABLE :: dl_list
      !----------------------------------------------------------------

      il_next=2
      IF( PRESENT(id_next) ) il_next=id_next
      
      il_size=SIZE(dd_array(:))
      IF( PRESENT(dd_fill)  )THEN
         il_count=COUNT(dd_array(:)/=dd_fill)
         IF( il_count > 0 )THEN
            ALLOCATE( dl_list(il_count) )
            dl_list(:)=PACK(dd_array(:),dd_array(:)/=dd_fill)
         ELSE
            df_mwe=dd_fill
         ENDIF
      ELSE
         il_count=SIZE(dd_array(:))
         IF( il_count > 0 )THEN
            ALLOCATE( dl_list(il_count) )
            dl_list(:)=dd_array(:)
         ELSE
            df_mwe=0
         ENDIF
      ENDIF

      IF( ALLOCATED(dl_list) )THEN
         CALL math_QsortC(dl_list(:))

         IF( il_count == il_size )THEN
            ! no fillValue
            df_mwe=math_mean(dl_list(il_next+1:il_size-il_next))
         ELSEIF( il_count > il_size-2*il_next )THEN
            ! remove one extremum each side
            df_mwe=math_mean(dl_list(2:il_size-1))
         ELSE ! il_count <= il_size-2*il_next
            ! more than 2*il_next fillValue
            ! compute mean only
            df_mwe=math_mean(dl_list(:))
         ENDIF

         DEALLOCATE(dl_list)
      ENDIF

   END FUNCTION math__mwe_1d
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION math__mwe_2d(dd_array, id_next, dd_fill) &
         & RESULT (df_mwe)
   !------------------------------------------------------------------- 
   !> @brief This function compute the mean without extremum of a 2D array. 
   !>
   !> @author J.Paul 
   !> @date January, 2015 - Initial Version 
   !>
   !> @param[in] dd_array  2D array 
   !> @param[in] id_next   number of extremum to be removed
   !> @param[in] dd_fill   fillValue
   !> @return median value, real(dp)
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:,:), INTENT(IN) :: dd_array
      INTEGER(i4)             , INTENT(IN), OPTIONAL :: id_next
      REAL(dp),                 INTENT(IN), OPTIONAL :: dd_fill

      ! function
      REAL(dp)                             :: df_mwe

      ! local variable
      INTEGER(i4)                         :: il_count

      REAL(dp), DIMENSION(:), ALLOCATABLE :: dl_list
      !----------------------------------------------------------------
      
      IF( PRESENT(dd_fill)  )THEN
         il_count=COUNT(dd_array(:,:)/=dd_fill)
         IF( il_count > 0 )THEN
            ALLOCATE( dl_list(il_count) )
            dl_list(:)=PACK(dd_array(:,:),dd_array(:,:)/=dd_fill)

            df_mwe=math_mwe(dl_list(:), id_next)
         ELSE
            df_mwe=dd_fill
         ENDIF
      ELSE
         il_count=SIZE(dd_array(:,:))
         IF( il_count > 0 )THEN
            ALLOCATE( dl_list(il_count) )
            dl_list(:)=PACK(dd_array(:,:), MASK=.TRUE.)

            df_mwe=math_mwe(dl_list(:), id_next)
         ELSE
            df_mwe=0
         ENDIF
      ENDIF

      IF( ALLOCATED(dl_list) ) DEALLOCATE(dl_list)

   END FUNCTION math__mwe_2d
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE RECURSIVE SUBROUTINE math_QsortC(dd_array)
   !------------------------------------------------------------------- 
   !> @brief This subroutine sort a 1D array. 
   !>
   !> @details
   !> Recursive Fortran 95 quicksort routine
   !> sorts real numbers into ascending numerical order
   !> Author: Juli Rew, SCD Consulting (juliana@ucar.edu), 9/03
   !> Based on algorithm from Cormen et al., Introduction to Algorithms,
   !> 1997 printing
   !> 
   !> @author J.Paul 
   !> @date January, 2015 - Rewrite with SIREN coding rules 
   !>
   !> @param[inout] dd_array  1D array 
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:), INTENT(INOUT) :: dd_array

      ! local variable
      INTEGER(i4) :: il_iq
      !----------------------------------------------------------------
     
      IF( SIZE(dd_array(:)) > 1 )THEN
         CALL math__Partition(dd_array, il_iq)
         CALL math_QsortC(dd_array(:il_iq-1))
         CALL math_QsortC(dd_array(il_iq:))
      ENDIF

   END SUBROUTINE math_QsortC
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE SUBROUTINE math__Partition(dd_array, id_marker)
   !------------------------------------------------------------------- 
   !> @brief This subroutine partition a 1D array. 
   !>
   !> @details
   !> Author: Juli Rew, SCD Consulting (juliana@ucar.edu), 9/03
   !> Based on algorithm from Cormen et al., Introduction to Algorithms,
   !> 1997 printing
   !> 
   !> @author J.Paul 
   !> @date January, 2015 - Rewrite with SIREN coding rules 
   !> @date November, 2017
   !> - use the correct loop index to look for element bigger than pivot point.
   !>
   !> @param[inout] dd_array  1D array 
   !> @param[in] id_marker 
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      REAL(dp)   , DIMENSION(:), INTENT(INOUT) :: dd_array
      INTEGER(i4),               INTENT(  OUT) :: id_marker

      ! local variable
      REAL(dp)    :: dl_temp
      REAL(dp)    :: dl_x     ! pivot point

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      dl_x = dd_array(1)
      ji= 0
      jj= SIZE(dd_array(:)) + 1

      DO 
         jj=jj-1
         DO
            IF( dd_array(jj) <= dl_x ) EXIT
            jj=jj-1
         ENDDO
         ji=ji+1
         DO
            IF( dd_array(ji) >= dl_x ) EXIT
            ji=ji+1
         ENDDO
         IF( ji < jj )THEN
            ! exchange dd_array(ji) and dd_array(jj)
            dl_temp= dd_array(ji)
            dd_array(ji) = dd_array(jj)
            dd_array(jj) = dl_temp
         ELSEIF( ji==jj )THEN
            id_marker=ji+1
            RETURN
         ELSE
            id_marker=ji
            RETURN
         ENDIF
      ENDDO

   END SUBROUTINE math__Partition
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE SUBROUTINE math_unwrap(dd_array, dd_discont)
   !------------------------------------------------------------------- 
   !> @brief This subroutine correct phase angles to produce smoother 
   !> phase plots. 
   !>
   !> @details
   !> This code is based on numpy unwrap function
   !>
   !> Unwrap by changing deltas between values to 2*pi complement.
   !>
   !> Unwrap radian phase `dd_array` by changing absolute jumps greater than
   !> `dd_discont` to their 2*pi complement.
   !>
   !> @note If the discontinuity in `dd_array` is smaller than ``pi``, 
   !> but larger than `dd_discont`, no unwrapping is done because taking
   !> the 2*pi complement would only make the discontinuity larger.
   !>
   !> @author J.Paul 
   !> @date Marsh, 2015 - Rewrite in fortran, with SIREN coding rules 
   !>
   !> @param[inout] dd_array  1D array 
   !> @param[in] dd_discont maximum discontinuity between values, default pi
   !-------------------------------------------------------------------
      
      IMPLICIT NONE
      
      ! Argument
      REAL(dp)   , DIMENSION(:), INTENT(INOUT) :: dd_array
      REAL(dp)   ,               INTENT(IN   ), OPTIONAL :: dd_discont

      ! local variable
      INTEGER(i4)                         :: il_size

      REAL(dp)                            :: dl_discont

      REAL(dp), DIMENSION(:), ALLOCATABLE :: dl_diff
      REAL(dp), DIMENSION(:), ALLOCATABLE :: dl_mod
      REAL(dp), DIMENSION(:), ALLOCATABLE :: dl_correct
      REAL(dp), DIMENSION(:), ALLOCATABLE :: dl_tmp

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      dl_discont=dp_pi
      IF( PRESENT(dd_discont) ) dl_discont=dd_discont

      il_size=SIZE(dd_array)
      ALLOCATE(dl_diff(il_size-1))
      DO ji=1,il_size-1
         dl_diff(ji)=dd_array(ji+1)-dd_array(ji)
      ENDDO

      ALLOCATE(dl_mod(il_size-1))
      DO ji=1,il_size-1
         dl_mod(ji) = MOD(dl_diff(ji) + dp_pi, 2*dp_pi) - dp_pi
      ENDDO

      WHERE( (dl_mod(:) == -dp_pi) .AND. (dl_diff(:) > 0._dp ) )
         dl_mod(:)=dp_pi
      END WHERE

      ALLOCATE(dl_correct(il_size-1))
      dl_correct(:)=dl_mod(:)-dl_diff(:)

      DEALLOCATE(dl_mod)

      WHERE( ABS(dl_diff(:)) < dl_discont )
         dl_correct(:)=0._dp
      END WHERE

      DEALLOCATE(dl_diff)
      
      ALLOCATE(dl_tmp(il_size))
      dl_tmp(:)=dd_array(:)

      DO ji=1,il_size-1
         dd_array(ji+1)=dl_tmp(ji+1)+SUM(dl_correct(1:ji))
      ENDDO

      DEALLOCATE(dl_correct)
      DEALLOCATE(dl_tmp)

   END SUBROUTINE math_unwrap
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   RECURSIVE FUNCTION math_compute(cd_var) &
         &  RESULT(df_res)
   !------------------------------------------------------------------- 
   !> @brief This function compute simple operation 
   !>
   !> @details
   !> - operation should be write as a string of character.
   !> - operators allowed are : +,-,*,/ 
   !> - to ordered operation you should use parentheses
   !>
   !> exemples: '1e6/(16/122)', '(3/2)*(2+1)' 
   !>
   !> @author J.Paul 
   !> @date June, 2015 - initial version
   !>
   !> @param[in] cd_var operation to compute (string of character)  
   !> @return result of the operation, real(dp)
   !-------------------------------------------------------------------
      
      IMPLICIT NONE
      
      ! Argument      
      CHARACTER(LEN=*), INTENT(IN) :: cd_var
   
      ! fucntion
      REAL(dp)                     :: df_res

      ! local variables
      CHARACTER(LEN=lc) :: cl_var
      CHARACTER(LEN=lc) :: cl_str1
      CHARACTER(LEN=lc) :: cl_str2
   
      INTEGER(i4)       :: il_ind
      ! loop indices
      !----------------------------------------------------------------
   
   
      IF(fct_is_real(cd_var))THEN
         READ(cd_var,*) df_res
      ELSE
      
   
         CALL math__parentheses(cd_var, cl_var)
         
         IF(fct_is_real(cl_var))THEN
            READ(cl_var,*) df_res
         ELSE
            il_ind=SCAN(TRIM(cl_var),'*')
            IF( il_ind /= 0 )THEN
               cl_str1=cl_var(1:il_ind-1)
               cl_str2=cl_var(il_ind+1:)
               df_res=math_compute(cl_str1)*math_compute(cl_str2)
            ELSE
               il_ind=SCAN(TRIM(cl_var),'/')
               IF( il_ind /= 0 )THEN
                  cl_str1=cl_var(1:il_ind-1)
                  cl_str2=cl_var(il_ind+1:)
                  df_res=math_compute(cl_str1)/math_compute(cl_str2)
               ELSE
                  il_ind=SCAN(TRIM(cl_var),'+')
                  IF( il_ind /= 0 )THEN
                     cl_str1=cl_var(1:il_ind-1)
                     cl_str2=cl_var(il_ind+1:)
                     df_res=math_compute(cl_str1)+math_compute(cl_str2)
                  ELSE
                     il_ind=SCAN(TRIM(cl_var),'-')
                     IF( il_ind /= 0 )THEN
                        cl_str1=cl_var(1:il_ind-1)
                        cl_str2=cl_var(il_ind+1:)
                        df_res=math_compute(cl_str1)-math_compute(cl_str2)
                     ELSE
                        df_res=dp_fill
                     ENDIF
                  ENDIF
               ENDIF
            ENDIF
         ENDIF
      ENDIF
   
   END FUNCTION math_compute
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE math__parentheses(cd_varin, cd_varout)
   !------------------------------------------------------------------- 
   !> @brief This subroutine replace sub string inside parentheses
   !> by the value of the operation inside.
   !>
   !> @details
   !> exemple : 
   !> - '2.6+(3/2)' => '2.6+1.5000'
   !>
   !> @author J.Paul 
   !> @date June, 2015 - initial version
   !> 
   !> @param[in] cd_varin  string of character with operation inside
   !> parentheses 
   !> @param[out] cd_varout string of character with result of 
   !> operation inside parentheses
   !-------------------------------------------------------------------   
      
      IMPLICIT NONE
      
      ! Argument      
      CHARACTER(LEN=*) , INTENT(IN)  :: cd_varin
      CHARACTER(LEN=lc), INTENT(OUT) :: cd_varout
      
      ! local variables
      CHARACTER(LEN=lc)     :: cl_cpt
      INTEGER(i4)           :: il_ind
      INTEGER(i4)           :: il_count
      
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      il_ind=INDEX(cd_varin,'(')
      IF( il_ind /= 0 )THEN
         il_count=0
         DO ji=il_ind+1,LEN(cd_varin)
            IF( cd_varin(ji:ji) == '(' )THEN
               il_count=il_count+1
            ELSEIF( cd_varin(ji:ji) == ')' )THEN
               IF( il_count == 0 )THEN
                  WRITE(cl_cpt,*) math_compute(cd_varin(il_ind+1:ji-1))
                  cd_varout=TRIM(cd_varin(1:il_ind-1))//TRIM(ADJUSTL(cl_cpt))//&
                     &        TRIM(cd_varin(ji+1:))
                  EXIT
               ELSE
                  il_count=il_count-1
               ENDIF
            ENDIF
         ENDDO
      ELSE
         cd_varout=TRIM(cd_varin)
      ENDIF
   
   END SUBROUTINE math__parentheses
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION math_deriv_1D(dd_value, dd_fill, ld_discont) &
         & RESULT (df_deriv)
   !-------------------------------------------------------------------
   !> @brief
   !> This function compute derivative of 1D array.
   !> 
   !> @details 
   !> optionaly you could specify to take into account east west discontinuity
   !> (-180° 180° or 0° 360° for longitude variable)
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_value     1D array of variable to be extrapolated
   !> @param[in] dd_fill      FillValue of variable
   !> @param[in] ld_discont   logical to take into account east west discontinuity 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)   , DIMENSION(:), INTENT(IN) :: dd_value
      REAL(dp)                 , INTENT(IN) :: dd_fill
      LOGICAL                  , INTENT(IN), OPTIONAL :: ld_discont

      ! function
      REAL(dp), DIMENSION(SIZE(dd_value,DIM=1) ) :: df_deriv

      ! local variable
      INTEGER(i4)                            :: il_imin
      INTEGER(i4)                            :: il_imax
      INTEGER(i4), DIMENSION(1)              :: il_shape

      REAL(dp)                               :: dl_min
      REAL(dp)                               :: dl_max
      REAL(dp)   , DIMENSION(:), ALLOCATABLE :: dl_value

      LOGICAL                                :: ll_discont

      ! loop indices
      INTEGER(i4) :: ji

      INTEGER(i4) :: i1
      INTEGER(i4) :: i2
      !----------------------------------------------------------------
      ! init
      df_deriv(:)=dd_fill

      ll_discont=.FALSE.
      IF( PRESENT(ld_discont) ) ll_discont=ld_discont

      il_shape(:)=SHAPE(dd_value(:))

      ALLOCATE( dl_value(3))

      ! compute derivative in i-direction
      DO ji=1,il_shape(1)
         
            il_imin=MAX(ji-1,1)
            il_imax=MIN(ji+1,il_shape(1))

            IF( il_imin==ji-1 .AND. il_imax==ji+1 )THEN
               i1=1  ; i2=3
            ELSEIF( il_imin==ji .AND. il_imax==ji+1 )THEN
               i1=1  ; i2=2
            ELSEIF( il_imin==ji-1 .AND. il_imax==ji )THEN
               i1=2  ; i2=3
            ENDIF

            dl_value(i1:i2)=dd_value(il_imin:il_imax)
            IF( il_imin == 1 )THEN
               dl_value(:)=EOSHIFT( dl_value(:), &
               &                    DIM=1,         &
               &                    SHIFT=-1,      &
               &                    BOUNDARY=dl_value(1) )
            ENDIF
            IF( il_imax == il_shape(1) )THEN
               dl_value(:)=EOSHIFT( dl_value(:), &
               &                    DIM=1,         &
               &                    SHIFT=1,       &
               &                    BOUNDARY=dl_value(3))
            ENDIF

            IF( ll_discont )THEN
               dl_min=MINVAL( dl_value(:), dl_value(:)/=dd_fill )
               dl_max=MAXVAL( dl_value(:), dl_value(:)/=dd_fill )
               IF( dl_min < -170_dp .AND. dl_max > 170_dp )THEN
                  WHERE( dl_value(:) < 0._dp ) 
                     dl_value(:) = dl_value(:)+360._dp
                  END WHERE
               ELSEIF( dl_min < 10_dp .AND. dl_max > 350_dp )THEN
                  WHERE( dl_value(:) > 180._dp ) 
                     dl_value(:) = dl_value(:)-180._dp
                  END WHERE
               ENDIF
            ENDIF

         IF( dl_value( 2) /= dd_fill .AND. & ! ji
         &   dl_value( 3) /= dd_fill .AND. & ! ji+1
         &   dl_value( 1) /= dd_fill )THEN   ! ji-1

            df_deriv(ji)= (dl_value(3) - dl_value(1)) / REAL(il_imax-il_imin,dp)

         ENDIF

      ENDDO

      DEALLOCATE( dl_value )

   END FUNCTION math_deriv_1D
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION math_deriv_2D(dd_value, dd_fill, cd_dim, ld_discont) &
         & RESULT (df_deriv)
   !-------------------------------------------------------------------
   !> @brief
   !> This function compute derivative of 2D array.
   !> you have to specify in which direction derivative have to be computed:
   !> first (I) or second (J) dimension. 
   !>
   !> @details 
   !> optionaly you could specify to take into account east west discontinuity
   !> (-180° 180° or 0° 360° for longitude variable)
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_value     2D array of variable to be extrapolated
   !> @param[in] dd_fill      FillValue of variable
   !> @param[in] cd_dim       compute derivative on first (I) or second (J) dimension 
   !> @param[in] ld_discont   logical to take into account east west discontinuity 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)   , DIMENSION(:,:), INTENT(IN) :: dd_value
      REAL(dp)                   , INTENT(IN) :: dd_fill
      CHARACTER(LEN=*)           , INTENT(IN) :: cd_dim
      LOGICAL                    , INTENT(IN), OPTIONAL :: ld_discont

      ! function
      REAL(dp), DIMENSION(SIZE(dd_value,DIM=1), &
         &                SIZE(dd_value,DIM=2) ) :: df_deriv

      ! local variable
      INTEGER(i4)                              :: il_imin
      INTEGER(i4)                              :: il_imax
      INTEGER(i4)                              :: il_jmin
      INTEGER(i4)                              :: il_jmax
      INTEGER(i4), DIMENSION(2)                :: il_shape

      REAL(dp)                                 :: dl_min
      REAL(dp)                                 :: dl_max
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_value

      LOGICAL                                  :: ll_discont

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj

      INTEGER(i4) :: i1
      INTEGER(i4) :: i2

      INTEGER(i4) :: j1
      INTEGER(i4) :: j2
      !----------------------------------------------------------------
      ! init
      df_deriv(:,:)=dd_fill

      ll_discont=.FALSE.
      IF( PRESENT(ld_discont) ) ll_discont=ld_discont

      il_shape(:)=SHAPE(dd_value(:,:))

      SELECT CASE(TRIM(fct_upper(cd_dim)))

      CASE('I')

         ALLOCATE( dl_value(3,il_shape(2)) )
         ! compute derivative in i-direction
         DO ji=1,il_shape(1)

            ! init
            dl_value(:,:)=dd_fill
            
            il_imin=MAX(ji-1,1)
            il_imax=MIN(ji+1,il_shape(1))

            IF( il_imin==ji-1 .AND. il_imax==ji+1 )THEN
               i1=1  ; i2=3
            ELSEIF( il_imin==ji .AND. il_imax==ji+1 )THEN
               i1=1  ; i2=2
            ELSEIF( il_imin==ji-1 .AND. il_imax==ji )THEN
               i1=2  ; i2=3
            ENDIF

            dl_value(i1:i2,:)=dd_value(il_imin:il_imax,:)
            IF( il_imin == 1 )THEN
               dl_value(:,:)=EOSHIFT( dl_value(:,:), &
                  &                   DIM=1,         &
                  &                   SHIFT=-1,      &
                  &                   BOUNDARY=dl_value(1,:) )
            ENDIF
            IF( il_imax == il_shape(1) )THEN
               dl_value(:,:)=EOSHIFT( dl_value(:,:), &
                  &                   DIM=1,         &
                  &                   SHIFT=1,       &
                  &                   BOUNDARY=dl_value(3,:))
            ENDIF

            IF( ll_discont )THEN
               dl_min=MINVAL( dl_value(:,:), dl_value(:,:)/=dd_fill )
               dl_max=MAXVAL( dl_value(:,:), dl_value(:,:)/=dd_fill )
               IF( dl_min < -170_dp .AND. dl_max > 170_dp )THEN
                  WHERE( dl_value(:,:) < 0_dp ) 
                     dl_value(:,:) = dl_value(:,:)+360._dp
                  END WHERE
               ELSEIF( dl_min < 10_dp .AND. dl_max > 350_dp )THEN
                  WHERE( dl_value(:,:) > 180 ) 
                     dl_value(:,:) = dl_value(:,:)-180._dp
                  END WHERE
               ENDIF
            ENDIF
            
            WHERE( dl_value(2,:) /= dd_fill .AND. &  ! ji
               &   dl_value(3,:) /= dd_fill .AND. &  ! ji+1
               &   dl_value(1,:) /= dd_fill )        ! ji-1

               df_deriv(ji,:)= (dl_value(3,:) - dl_value(1,:)) / REAL(il_imax-il_imin,dp)

            END WHERE

         ENDDO

      CASE('J')

         ALLOCATE( dl_value(il_shape(1),3) )
         ! compute derivative in j-direction
         DO jj=1,il_shape(2)
         
            il_jmin=MAX(jj-1,1)
            il_jmax=MIN(jj+1,il_shape(2))

            IF( il_jmin==jj-1 .AND. il_jmax==jj+1 )THEN
               j1=1  ; j2=3
            ELSEIF( il_jmin==jj .AND. il_jmax==jj+1 )THEN
               j1=1  ; j2=2
            ELSEIF( il_jmin==jj-1 .AND. il_jmax==jj )THEN
               j1=2  ; j2=3
            ENDIF

            dl_value(:,j1:j2)=dd_value(:,il_jmin:il_jmax)
            IF( il_jmin == 1 )THEN
               dl_value(:,:)=EOSHIFT( dl_value(:,:), &
                  &                   DIM=2,         &
                  &                   SHIFT=-1,      &
                  &                   BOUNDARY=dl_value(:,1))
            ENDIF
            IF( il_jmax == il_shape(2) )THEN
               dl_value(:,:)=EOSHIFT( dl_value(:,:), &
                  &                   DIM=2,         &
                  &                   SHIFT=1,       &
                  &                   BOUNDARY=dl_value(:,3))
            ENDIF

            IF( ll_discont )THEN
               dl_min=MINVAL( dl_value(:,:), dl_value(:,:)/=dd_fill )
               dl_max=MAXVAL( dl_value(:,:), dl_value(:,:)/=dd_fill )
               IF( dl_min < -170_dp .AND. dl_max > 170_dp )THEN
                  WHERE( dl_value(:,:) < 0_dp ) 
                     dl_value(:,:) = dl_value(:,:)+360._dp
                  END WHERE
               ELSEIF( dl_min < 10_dp .AND. dl_max > 350_dp )THEN
                  WHERE( dl_value(:,:) > 180 ) 
                     dl_value(:,:) = dl_value(:,:)-180._dp
                  END WHERE
               ENDIF
            ENDIF

            WHERE( dl_value(:, 2) /= dd_fill .AND. & ! jj
               &   dl_value(:, 3) /= dd_fill .AND. & ! jj+1
               &   dl_value(:, 1) /= dd_fill )       ! jj-1

               df_deriv(:,jj)= (dl_value(:,3) - dl_value(:,1)) / REAL(il_jmax-il_jmin,dp)         

            END WHERE

         ENDDO
         
      END SELECT

      DEALLOCATE( dl_value )

   END FUNCTION math_deriv_2D
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION math_deriv_3D(dd_value, dd_fill, cd_dim, ld_discont) &
         & RESULT (df_deriv)
   !-------------------------------------------------------------------
   !> @brief
   !> This function compute derivative of 3D array.
   !> you have to specify in which direction derivative have to be computed:
   !> first (I), second (J) or third (K) dimension.
   !> 
   !> @details 
   !> optionaly you could specify to take into account east west discontinuity
   !> (-180° 180° or 0° 360° for longitude variable)
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] dd_value  3D array of variable to be extrapolated
   !> @param[in] dd_fill      FillValue of variable
   !> @param[in] cd_dim       compute derivative on first (I) second (J) or third (K) dimension   
   !> @param[in] ld_discont   logical to take into account east west discontinuity
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)        , DIMENSION(:,:,:), INTENT(IN) :: dd_value
      REAL(dp)                          , INTENT(IN) :: dd_fill
      CHARACTER(LEN=*)                  , INTENT(IN) :: cd_dim
      LOGICAL                           , INTENT(IN), OPTIONAL :: ld_discont

      ! function
      REAL(dp), DIMENSION(SIZE(dd_value,DIM=1), &
      &                   SIZE(dd_value,DIM=2), &
      &                   SIZE(dd_value,DIM=3)) :: df_deriv

      ! local variable
      INTEGER(i4)                                :: il_imin
      INTEGER(i4)                                :: il_imax
      INTEGER(i4)                                :: il_jmin
      INTEGER(i4)                                :: il_jmax
      INTEGER(i4)                                :: il_kmin
      INTEGER(i4)                                :: il_kmax
      INTEGER(i4), DIMENSION(3)                  :: il_shape

      REAL(dp)                                   :: dl_min
      REAL(dp)                                   :: dl_max
      REAL(dp)   , DIMENSION(:,:,:), ALLOCATABLE :: dl_value

      LOGICAL                                    :: ll_discont

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk

      INTEGER(i4) :: i1
      INTEGER(i4) :: i2

      INTEGER(i4) :: j1
      INTEGER(i4) :: j2
      
      INTEGER(i4) :: k1
      INTEGER(i4) :: k2      
      !----------------------------------------------------------------
      ! init
      df_deriv(:,:,:)=dd_fill

      ll_discont=.FALSE.
      IF( PRESENT(ld_discont) ) ll_discont=ld_discont

      il_shape(:)=SHAPE(dd_value(:,:,:))


      SELECT CASE(TRIM(fct_upper(cd_dim)))

      CASE('I')

         ALLOCATE( dl_value(3,il_shape(2),il_shape(3)) )
         ! compute derivative in i-direction
         DO ji=1,il_shape(1)
            
            il_imin=MAX(ji-1,1)
            il_imax=MIN(ji+1,il_shape(1))

            IF( il_imin==ji-1 .AND. il_imax==ji+1 )THEN
               i1=1  ; i2=3
            ELSEIF( il_imin==ji .AND. il_imax==ji+1 )THEN
               i1=1  ; i2=2
            ELSEIF( il_imin==ji-1 .AND. il_imax==ji )THEN
               i1=2  ; i2=3
            ENDIF

            dl_value(i1:i2,:,:)=dd_value(il_imin:il_imax,:,:)
            IF( il_imin == 1 )THEN
               dl_value(:,:,:)=EOSHIFT( dl_value(:,:,:), &
                  &                     DIM=1,         &
                  &                     SHIFT=-1,      &
                  &                     BOUNDARY=dl_value(1,:,:) )
            ENDIF
            IF( il_imax == il_shape(1) )THEN
               dl_value(:,:,:)=EOSHIFT( dl_value(:,:,:), &
                  &                     DIM=1,         &
                  &                     SHIFT=1,       &
                  &                     BOUNDARY=dl_value(3,:,:))
            ENDIF

            IF( ll_discont )THEN
               dl_min=MINVAL( dl_value(:,:,:), dl_value(:,:,:)/=dd_fill )
               dl_max=MAXVAL( dl_value(:,:,:), dl_value(:,:,:)/=dd_fill )
               IF( dl_min < -170_dp .AND. dl_max > 170_dp )THEN
                  WHERE( dl_value(:,:,:) < 0_dp ) 
                     dl_value(:,:,:) = dl_value(:,:,:)+360._dp
                  END WHERE
               ELSEIF( dl_min < 10_dp .AND. dl_max > 350_dp )THEN
                  WHERE( dl_value(:,:,:) > 180 ) 
                     dl_value(:,:,:) = dl_value(:,:,:)-180._dp
                  END WHERE
               ENDIF
            ENDIF

            WHERE( dl_value(2,:,:) /= dd_fill .AND. & ! ji
               &   dl_value(3,:,:) /= dd_fill .AND. & !ji+1 
               &   dl_value(1,:,:) /= dd_fill )       !ji-1

               df_deriv(ji,:,:)= (dl_value(3,:,:) - dl_value(1,:,:)) / REAL(il_imax-il_imin,dp)

            END WHERE

         ENDDO

      CASE('J')

         ALLOCATE( dl_value(il_shape(1),3,il_shape(3)) )
         ! compute derivative in j-direction
         DO jj=1,il_shape(2)
         
            il_jmin=MAX(jj-1,1)
            il_jmax=MIN(jj+1,il_shape(2))

            IF( il_jmin==jj-1 .AND. il_jmax==jj+1 )THEN
               j1=1  ; j2=3
            ELSEIF( il_jmin==jj .AND. il_jmax==jj+1 )THEN
               j1=1  ; j2=2
            ELSEIF( il_jmin==jj-1 .AND. il_jmax==jj )THEN
               j1=2  ; j2=3
            ENDIF

            dl_value(:,j1:j2,:)=dd_value(:,il_jmin:il_jmax,:)
            IF( il_jmin == 1 )THEN
               dl_value(:,:,:)=EOSHIFT( dl_value(:,:,:), &
                  &                     DIM=2,         &
                  &                     SHIFT=-1,      &
                  &                     BOUNDARY=dl_value(:,1,:) )
            ENDIF
            IF( il_jmax == il_shape(2) )THEN
               dl_value(:,:,:)=EOSHIFT( dl_value(:,:,:), &
                  &                     DIM=2,         &
                  &                     SHIFT=1,       &
                  &                     BOUNDARY=dl_value(:,3,:))
            ENDIF

            IF( ll_discont )THEN
               dl_min=MINVAL( dl_value(:,:,:), dl_value(:,:,:)/=dd_fill )
               dl_max=MAXVAL( dl_value(:,:,:), dl_value(:,:,:)/=dd_fill )
               IF( dl_min < -170_dp .AND. dl_max > 170_dp )THEN
                  WHERE( dl_value(:,:,:) < 0_dp ) 
                     dl_value(:,:,:) = dl_value(:,:,:)+360._dp
                  END WHERE
               ELSEIF( dl_min < 10_dp .AND. dl_max > 350_dp )THEN
                  WHERE( dl_value(:,:,:) > 180 ) 
                     dl_value(:,:,:) = dl_value(:,:,:)-180._dp
                  END WHERE
               ENDIF
            ENDIF

            WHERE( dl_value(:, 2,:) /= dd_fill .AND. & ! jj
               &   dl_value(:, 3,:) /= dd_fill .AND. & ! jj+1
               &   dl_value(:, 1,:) /= dd_fill )       ! jj-1

               df_deriv(:,jj,:)= (dl_value(:,3,:) - dl_value(:,1,:)) / REAL(il_jmax - il_jmin,dp)         

            END WHERE

         ENDDO
         
      CASE('K')

         ALLOCATE( dl_value(il_shape(1),il_shape(2),3) )
         ! compute derivative in k-direction
         DO jk=1,il_shape(3)

            il_kmin=MAX(jk-1,1)
            il_kmax=MIN(jk+1,il_shape(3))

            IF( il_kmin==jk-1 .AND. il_kmax==jk+1 )THEN
               k1=1  ; k2=3
            ELSEIF( il_kmin==jk .AND. il_kmax==jk+1 )THEN
               k1=1  ; k2=2
            ELSEIF( il_kmin==jk-1 .AND. il_kmax==jk )THEN
               k1=2  ; k2=3
            ENDIF

            dl_value(:,:,k1:k2)=dd_value(:,:,il_kmin:il_kmax)
            IF( il_kmin == 1 )THEN
               dl_value(:,:,:)=EOSHIFT( dl_value(:,:,:), &
                  &                     DIM=3,         &
                  &                     SHIFT=-1,      &
                  &                     BOUNDARY=dl_value(:,:,1) )
            ENDIF
            IF( il_kmax == il_shape(3) )THEN
               dl_value(:,:,:)=EOSHIFT( dl_value(:,:,:), &
                  &                     DIM=3,         &
                  &                     SHIFT=1,       &
                  &                     BOUNDARY=dl_value(:,:,3))
            ENDIF

            IF( ll_discont )THEN
               dl_min=MINVAL( dl_value(:,:,:), dl_value(:,:,:)/=dd_fill )
               dl_max=MAXVAL( dl_value(:,:,:), dl_value(:,:,:)/=dd_fill )
               IF( dl_min < -170_dp .AND. dl_max > 170_dp )THEN
                  WHERE( dl_value(:,:,:) < 0_dp ) 
                     dl_value(:,:,:) = dl_value(:,:,:)+360._dp
                  END WHERE
               ELSEIF( dl_min < 10_dp .AND. dl_max > 350_dp )THEN
                  WHERE( dl_value(:,:,:) > 180 ) 
                     dl_value(:,:,:) = dl_value(:,:,:)-180._dp
                  END WHERE
               ENDIF
            ENDIF         

            WHERE( dl_value(:,:,2) /= dd_fill .AND. & ! jk
               &   dl_value(:,:,3) /= dd_fill .AND. & ! jk+1
               &   dl_value(:,:,1) /= dd_fill )       ! jk-1

               df_deriv(:,:,jk)= (dl_value(:,:,3) - dl_value(:,:,1)) / REAL(il_kmax-il_kmin,dp)         

            END WHERE

         ENDDO

      END SELECT

      DEALLOCATE( dl_value )

   END FUNCTION math_deriv_3D
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION math_ortho(dd_latm) &
         &  RESULT(df_ortho)
   !-------------------------------------------------------------------
   !> @brief
   !> This function compute orthodome distance between opposite point of a cell
   !> of one degree.
   !> 
   !> @details 
   !>
   !> @author J.Paul
   !> @date April, 2017 - Initial Version
   !>
   !> @param[in] dd_latm   mean latitude of the cell
   !> @return orthodome distance
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      REAL(dp), TARGET :: dd_latm
      
      ! function
      REAL(dp)         :: df_ortho
      
      ! local
      REAL(dp) :: dl_dlat
      REAL(dp) :: dl_dlon
      REAL(dp) :: dl_lat1
      REAL(dp) :: dl_lat2
      REAL(dp) :: dl_tmp
      !----------------------------------------------------------------

      ! one degree cell
      dl_dlat= 1._dp * dp_deg2rad
      dl_dlon= 1._dp * dp_deg2rad

      ! 
      dl_lat1 = (dd_latm - 0.5_dp) * dp_deg2rad
      dl_lat2 = (dd_latm + 0.5_dp) * dp_deg2rad

      dl_tmp = SQRT( SIN(dl_dlat*0.5)**2 + &
         &           COS(dl_lat1)*COS(dl_lat2)*SIN(dl_dlon*0.5)**2 )

      df_ortho= 2* dp_rearth * ASIN( dl_tmp )

   END FUNCTION math_ortho
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION math_euclid(dd_lonm,dd_latm) &
         & RESULT(df_euclid)
   !-------------------------------------------------------------------
   !> @brief
   !> This function compute euclidian distance between opposite point of a cell
   !> of one degree, center on (lonm,latm).
   !> 
   !> @details 
   !>
   !> @author J.Paul
   !> @date April, 2017 - Initial Version
   !>
   !> @param[in] dd_lonm   mean longitude of the cell
   !> @param[in] dd_latm   mean latitude of the cell
   !> @return orthodome distance
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      REAL(dp), TARGET :: dd_lonm
      REAL(dp), TARGET :: dd_latm
      
      ! function
      REAL(dp)         :: df_euclid
      
      ! local
      REAL(dp) :: dl_lata
      REAL(dp) :: dl_lona
      REAL(dp) :: dl_latb
      REAL(dp) :: dl_lonb
      REAL(dp) :: xa,ya,za
      REAL(dp) :: xb,yb,zb
      !----------------------------------------------------------------

      dl_lata=(dd_latm-0.5)*dp_deg2rad
      dl_lona=(dd_lonm-0.5)*dp_deg2rad

      xa = dp_rearth * COS(dl_lata) * COS(dl_lona)
      ya = dp_rearth * COS(dl_lata) * SIN(dl_lona)
      za = dp_rearth * SIN(dl_lata)

      dl_latb=(dd_latm+0.5)*dp_deg2rad
      dl_lonb=(dd_lonm+0.5)*dp_deg2rad

      xb = dp_rearth * COS(dl_latb) * COS(dl_lonb)
      yb = dp_rearth * COS(dl_latb) * SIN(dl_lonb)
      zb = dp_rearth * SIN(dl_latb)

      df_euclid = ((xb-xa)**2 + (yb-ya)**2 + (zb-za)**2)

   END FUNCTION math_euclid
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE math
