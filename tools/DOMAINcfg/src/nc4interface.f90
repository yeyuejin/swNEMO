MODULE nc4interface
!-
!-
! This software is governed by the CeCILL license
! See IOIPSL/IOIPSL_License_CeCILL.txt
!---------------------------------------------------------------------

      !!--------------------------------------------------------------------
      !! NOT 'key_netcdf4' Defines dummy routines for netcdf4
      !!                   calls when compiling without netcdf4 libraries
      !!--------------------------------------------------------------------
  !- netcdf4 chunking control structure
  !- (optional on histbeg and histend calls)
!$AGRIF_DO_NOT_TREAT
  TYPE, PUBLIC :: snc4_ctl
     SEQUENCE
     INTEGER :: ni
     INTEGER :: nj
     INTEGER :: nk
     LOGICAL :: luse
  END TYPE snc4_ctl
!$AGRIF_END_DO_NOT_TREAT

CONTAINS
!===
   SUBROUTINE GET_NF90_SYMBOL(sym_name, ivalue)
      CHARACTER(len=*),      INTENT(in)  :: sym_name
      INTEGER,               INTENT(out) :: ivalue
      ivalue = -999
   END SUBROUTINE GET_NF90_SYMBOL
   INTEGER FUNCTION SET_NF90_DEF_VAR_CHUNKING(idum1, idum2, idum3, iarr1)
      !!--------------------------------------------------------------------
      !!                   ***  SUBROUTINE NF90_DEF_VAR_CHUNKING  ***
      !!
      !! ** Purpose :   Dummy NetCDF4 routine to enable compiling with NetCDF3 libraries
      !!--------------------------------------------------------------------
      INTEGER,               INTENT(in) :: idum1, idum2, idum3
      INTEGER, DIMENSION(4), INTENT(in) :: iarr1
      WRITE(*,*) 'Warning: Attempt to chunk output variable without NetCDF4 support'
      SET_NF90_DEF_VAR_CHUNKING = -1
   END FUNCTION SET_NF90_DEF_VAR_CHUNKING

   INTEGER FUNCTION SET_NF90_DEF_VAR_DEFLATE(idum1, idum2, idum3, idum4, idum5)
      !!--------------------------------------------------------------------
      !!                   ***  SUBROUTINE NF90_DEF_VAR_DEFLATE  ***
      !!
      !! ** Purpose :   Dummy NetCDF4 routine to enable compiling with NetCDF3 libraries
      !!--------------------------------------------------------------------
      INTEGER,               INTENT(in) :: idum1, idum2, idum3, idum4, idum5
      WRITE(*,*) 'Warning: Attempt to compress output variable without NetCDF4 support'
      SET_NF90_DEF_VAR_DEFLATE = -1
   END FUNCTION SET_NF90_DEF_VAR_DEFLATE

!------------------
END MODULE nc4interface
