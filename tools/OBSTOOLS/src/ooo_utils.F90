MODULE ooo_utils
   !! =================================================================
   !!                    *** MODULE ooo_utils ***
   !! =================================================================
   USE par_oce

   IMPLICIT NONE

   ! Define double precision obfillflt
   REAL(kind=dp), PARAMETER :: obfilldbl=99999.

   !! $Id: ooo_utils.F90 5215 2015-04-15 16:11:56Z nicolasmartin $
   CONTAINS


   SUBROUTINE date_format(date_str)
      !---------------------------------------
      ! Routine to create creation date string
      !---------------------------------------

      ! Routine arguments
      CHARACTER(len=*), INTENT(OUT) :: date_str
      ! Local variables
      CHARACTER(8)  :: date
      CHARACTER(10) :: time

      CHARACTER(10)  :: date_part
      CHARACTER(8)  :: time_part

      CALL date_and_time(DATE=date)
      CALL date_and_time(TIME=time)
      date_part  = date(1:4)//'/'//date(5:6)//'/'//date(7:8)
      time_part  = time(1:2)//':'//time(3:4)//':'//time(5:6)

      date_str   = date_part//' '//time_part

   END SUBROUTINE date_format

END MODULE ooo_utils
