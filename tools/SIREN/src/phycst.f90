!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief This module defines physical constant.
!>
!> @author
!> J.paul
! REVISION HISTORY:
!> @date November, 2013 - Initial Version
!> @date September, 2015
!> - add physical constant to compute meshmask
!> @date January, 2019
!> - half reduce epsilon value
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE phycst

   USE kind                         ! F90 kind parameter

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   PUBLIC :: dp_pi      !< pi
   PUBLIC :: dp_eps     !< epsilon value
   PUBLIC :: dp_rearth  !< earth radius [m]
   PUBLIC :: dp_deg2rad !< degree to radian ratio 
   PUBLIC :: dp_rad2deg !< radian to degree ratio 
   PUBLIC :: dp_delta   !<  
   PUBLIC :: dp_omega   !< earth rotation parameter [s-1] 
   PUBLIC :: dp_day     !< day                                [s]
   PUBLIC :: dp_siyea   !< sideral year                       [s]
   PUBLIC :: dp_siday   !< sideral day                        [s]

   REAL(dp), PUBLIC ::   rday = 24.*60.*60.     !: day                                [s]
   REAL(dp), PUBLIC ::   rsiyea                 !: sideral year                       [s]
   REAL(dp), PUBLIC ::   rsiday                 !: sideral day                        [s]

   REAL(dp), PARAMETER :: dp_pi = 3.14159274101257_dp
   REAL(dp), PARAMETER :: dp_eps = 0.5 * EPSILON(1._dp)
   REAL(dp), PARAMETER :: dp_rearth = 6371229._dp !m
   REAL(dp), PARAMETER :: dp_deg2rad = dp_pi/180.0
   REAL(dp), PARAMETER :: dp_rad2deg = 180.0/dp_pi

   REAL(dp), PARAMETER :: dp_day = 24.*60.*60.     
   REAL(dp), PARAMETER :: dp_siyea = 365.25_dp * dp_day * &
      &  2._dp * dp_pi / 6.283076_dp
   REAL(dp), PARAMETER :: dp_siday = dp_day / ( 1._dp + dp_day / dp_siyea )

   REAL(dp), PARAMETER :: dp_delta=1.e-6
   REAL(dp), PARAMETER :: dp_omega= 2._dp * dp_pi / dp_siday

END MODULE phycst

