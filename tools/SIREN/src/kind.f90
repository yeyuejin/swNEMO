!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> This module defines the F90 kind parameter for common data types.
!> 
!>
!> @author
!> G. Madec
! REVISION HISTORY:
!> @date June, 2006 - Initial Version
!> @date December, 2012 - G. Madec
!>  - add a standard length of character strings
!>
!> @todo
!> - check i8 max value
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE kind

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   !                                                                !!** Floating point **
   ! SELECTED_REAL_KIND(P,R) returns the kind value of a real data type 
   ! with decimal precision of at least P digits, exponent range of at least R
   INTEGER, PUBLIC, PARAMETER ::   sp = SELECTED_REAL_KIND( 6, 37)   !< single precision (real 4)
   INTEGER, PUBLIC, PARAMETER ::   dp = SELECTED_REAL_KIND(12,307)   !< double precision (real 8)
   INTEGER, PUBLIC, PARAMETER ::   wp = dp                           !< working precision

   !                                                                !!** Integer **
   ! SELECTED_INT_KIND(R) return the kind value of the smallest integer type 
   ! that can represent all values ranging ] -10^R , 10^R [
   INTEGER, PUBLIC, PARAMETER ::   i1 = SELECTED_INT_KIND( 1)        !< single precision (integer 1) 
   INTEGER, PUBLIC, PARAMETER ::   i2 = SELECTED_INT_KIND( 4)        !< single precision (integer 2) 
   INTEGER, PUBLIC, PARAMETER ::   i4 = SELECTED_INT_KIND( 9)        !< single precision (integer 4) 
   INTEGER, PUBLIC, PARAMETER ::   i8 = SELECTED_INT_KIND(14)        !< double precision (integer 8)
   
   !                                                                !!** Integer **
   INTEGER, PUBLIC, PARAMETER ::   lc = 256                          !< Length of Character strings

END MODULE kind

