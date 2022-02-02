MODULE custom

   USE utils

   IMPLICIT NONE
   PUBLIC

   CONTAINS

   SUBROUTINE custom_resto( presto )
      !!---------------------------------
      !!     **ROUTINE: custom_resto
      !!
      !!  ** Purpose: Module to be edited by users to create custom restoration
      !!              coefficient files (e.g. regional damping).
      !!
      !!-------------------------------------
      REAL(wp), DIMENSION(jpi,jpk), INTENT(inout) :: presto

   END SUBROUTINE custom_resto

END MODULE custom
