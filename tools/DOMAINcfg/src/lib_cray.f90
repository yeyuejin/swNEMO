!  Cray subroutines or functions used by OPA model and possibly 
!  not found on other platforms.
!
!  check their existence
!  
!  wheneq
!!----------------------------------------------------------------------
!!  OPA 9.0 , LOCEAN-IPSL (2005) 
!! $Id: lib_cray.f90 3680 2012-11-27 14:42:24Z rblod $ 
!! This software is governed by the CeCILL licence see modipsl/doc/NEMO_CeCILL.txt 
!!----------------------------------------------------------------------
SUBROUTINE lib_cray
      WRITE(*,*) 'lib_cray: You should not have seen this print! error?'
END SUBROUTINE lib_cray

SUBROUTINE wheneq ( i, x, j, t, ind, nn )
        IMPLICIT NONE

        INTEGER , INTENT (  in ) :: i, j
        INTEGER , INTENT ( out ) :: nn
        REAL    , INTENT (  in ), DIMENSION (1+(i-1)*j) :: x
        REAL    , INTENT (  in ) :: t
        INTEGER , INTENT ( out ), DIMENSION (1+(i-1)*j) :: ind
        INTEGER :: n, k
        nn = 0
        DO n = 1, i
          k = 1 + (n-1) * j
          IF ( x ( k) == t ) THEN 
              nn = nn + 1
              ind (nn) = k
          ENDIF
        END DO 

END SUBROUTINE wheneq
