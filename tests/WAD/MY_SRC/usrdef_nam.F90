MODULE usrdef_nam
   !!======================================================================
   !!                       ***  MODULE usrdef_nam  ***
   !!
   !!                  ===  WAD_TEST_CASES configuration  ===
   !!
   !! User defined : set the domain characteristics of a user configuration
   !!======================================================================
   !! History :  4.0  ! 2016-03  (S. Flavoni, G. Madec)  Original code
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   usr_def_nam   : read user defined namelist and set global domain size
   !!   usr_def_hgr   : initialize the horizontal mesh 
   !!----------------------------------------------------------------------
   USE dom_oce  , ONLY: nimpp , njmpp            ! i- & j-indices of the local domain
   USE par_oce        ! ocean space and time domain
   USE phycst         ! physical constants
   !
   USE in_out_manager ! I/O manager
   USE lib_mpp        ! MPP library
   USE timing         ! Timing
   
   IMPLICIT NONE
   PRIVATE

   PUBLIC   usr_def_nam   ! called by nemogcm.F90

   !                              !!* namusr_def namelist *!!
   REAL(wp), PUBLIC ::   rn_dx     ! resolution in meters defining the horizontal domain size
   REAL(wp), PUBLIC ::   rn_dz     ! resolution in meters defining the vertical   domain size
   INTEGER , PUBLIC :: nn_wad_test ! resolution in meters defining the vertical   domain size

   !!----------------------------------------------------------------------
   !! NEMO/OCE 4.0 , NEMO Consortium (2018)
   !! $Id: usrdef_nam.F90 11536 2019-09-11 13:54:18Z smasson $ 
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE usr_def_nam( cd_cfg, kk_cfg, kpi, kpj, kpk, kperio )
      !!----------------------------------------------------------------------
      !!                     ***  ROUTINE dom_nam  ***
      !!                    
      !! ** Purpose :   read user defined namelist and define the domain size
      !!
      !! ** Method  :   read in namusr_def containing all the user specific namelist parameter
      !!
      !!                Here WAD_TEST_CASES configuration
      !!
      !! ** input   : - namusr_def namelist found in namelist_cfg
      !!----------------------------------------------------------------------
      CHARACTER(len=*)              , INTENT(out) ::   cd_cfg          ! configuration name
      INTEGER                       , INTENT(out) ::   kk_cfg          ! configuration resolution
      INTEGER                       , INTENT(out) ::   kpi, kpj, kpk   ! global domain sizes 
      INTEGER                       , INTENT(out) ::   kperio          ! lateral global domain b.c. 
      !
      INTEGER ::   ios   ! Local integer
      !!
      NAMELIST/namusr_def/ rn_dx, rn_dz, nn_wad_test
      !!----------------------------------------------------------------------
      !
      REWIND( numnam_cfg )          ! Namelist namusr_def (exist in namelist_cfg only)
      READ  ( numnam_cfg, namusr_def, IOSTAT = ios, ERR = 902 )
902   IF( ios /= 0 )   CALL ctl_nam ( ios , 'namusr_def in configuration namelist' )
      !
      IF(lwm)   WRITE( numond, namusr_def )
      !
      !
      cd_cfg = 'wad'      ! name & resolution (not used)
      nn_cfg = nn_wad_test
      kk_cfg = nn_wad_test
      !
      ! Global Domain size:  WAD_TEST_CASES domain is 52 km x 34 km x 10 m
      kpi = INT(  50.e3 / rn_dx ) + 2
      kpj = INT(  32.e3 / rn_dx ) + 2
      kpk = INT(  10.  / rn_dz ) + 1
      !                             ! Set the lateral boundary condition of the global domain
      kperio = 0                    ! WAD_TEST_CASES configuration : closed domain
      IF( nn_wad_test == 8 ) kperio = 7 ! North-South cyclic test
      !
      !                             ! control print
      IF(lwp) THEN
         WRITE(numout,*) '   '
         WRITE(numout,*) 'usr_def_nam  : read the user defined namelist (namusr_def) in namelist_cfg'
         WRITE(numout,*) '~~~~~~~~~~~ '
         WRITE(numout,*) '   Namelist namusr_def : WAD_TEST_CASES test case'
         WRITE(numout,*) '      horizontal resolution                    rn_dx  = ', rn_dx, ' meters'
         WRITE(numout,*) '      vertical   resolution                    rn_dz  = ', rn_dz, ' meters'
         WRITE(numout,*) '      WAD_TEST_CASES domain = 52 km  x  34 km x 10 m'
         WRITE(numout,*) '         resulting global domain size :        jpiglo = ', kpi
         WRITE(numout,*) '                                               jpjglo = ', kpj
         WRITE(numout,*) '                                               jpkglo = ', kpk
         WRITE(numout,*) '   '
         WRITE(numout,*) '   Lateral boundary condition of the global domain'
         WRITE(numout,*) '      closed                                   jperio = ', kperio
      ENDIF
      !
   END SUBROUTINE usr_def_nam

   !!======================================================================
END MODULE usrdef_nam
