MODULE usrdef_nam
   !!======================================================================
   !!                       ***  MODULE  usrdef_nam  ***
   !!
   !!                      ===  VORTEX configuration  ===
   !!
   !! User defined : set the domain characteristics of a user configuration
   !!======================================================================
   !! History :  NEMO ! 2017-10  (J. Chanut)  Original code
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
   REAL(wp), PUBLIC ::   rn_dx      ! resolution in meters defining the horizontal domain size
   REAL(wp), PUBLIC ::   rn_dy      ! resolution in meters defining the horizontal domain size
   REAL(wp), PUBLIC ::   rn_dz      ! vertical resolution 
   REAL(wp), PUBLIC ::   rn_ppgphi0 ! reference latitude for beta-plane 

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
      !!                Here VORTEX configuration
      !!
      !! ** input   : - namusr_def namelist found in namelist_cfg
      !!----------------------------------------------------------------------
      CHARACTER(len=*)              , INTENT(out) ::   cd_cfg          ! configuration name
      INTEGER                       , INTENT(out) ::   kk_cfg          ! configuration resolution
      INTEGER                       , INTENT(out) ::   kpi, kpj, kpk   ! global domain sizes 
      INTEGER                       , INTENT(out) ::   kperio          ! lateral global domain b.c. 
      !
      INTEGER ::   ios          ! Local integer
      REAL(wp)::   zlx, zly, zh ! Local scalars
      !!
      NAMELIST/namusr_def/  rn_dx, rn_dy, rn_dz, rn_ppgphi0
      !!----------------------------------------------------------------------
      !
      REWIND( numnam_cfg )          ! Namelist namusr_def (exist in namelist_cfg only)
      READ  ( numnam_cfg, namusr_def, IOSTAT = ios, ERR = 902 )
902   IF( ios /= 0 )   CALL ctl_nam ( ios , 'namusr_def in configuration namelist' )
      !
#if defined key_agrif 
      ! Domain parameters are taken from parent:
      IF( .NOT. Agrif_Root() ) THEN
         rn_dx = Agrif_Parent(rn_dx)/Agrif_Rhox()
         rn_dy = Agrif_Parent(rn_dy)/Agrif_Rhoy()
         rn_dz = Agrif_Parent(rn_dz)
         rn_ppgphi0 = Agrif_Parent(rn_ppgphi0)
      ENDIF
#endif
      !
      IF(lwm)   WRITE( numond, namusr_def )
      !
      cd_cfg = 'VORTEX'             ! name & resolution (not used)
      kk_cfg = nINT( rn_dx )
      !
      IF( Agrif_Root() ) THEN       ! Global Domain size:  VORTEX global domain is  1800 km x 1800 Km x 5000 m
         kpi = NINT( 1800.e3  / rn_dx ) + 3  
         kpj = NINT( 1800.e3  / rn_dy ) + 3 
      ELSE
         kpi  = nbcellsx + 2 + 2*nbghostcells
         kpj  = nbcellsy + 2 + 2*nbghostcells
      ENDIF
      kpk = NINT( 5000._wp / rn_dz ) + 1
      !
      zlx = (kpi-2)*rn_dx*1.e-3
      zly = (kpj-2)*rn_dy*1.e-3
      zh  = (kpk-1)*rn_dz
      !                             ! Set the lateral boundary condition of the global domain
      kperio = 0                    ! VORTEX configuration : closed basin
      !                             ! control print
      IF(lwp) THEN
         WRITE(numout,*) '   '
         WRITE(numout,*) 'usr_def_nam  : read the user defined namelist (namusr_def) in namelist_cfg'
         WRITE(numout,*) '~~~~~~~~~~~ '
         WRITE(numout,*) '   Namelist namusr_def : VORTEX test case'
         WRITE(numout,*) '      horizontal resolution             rn_dx  = ', rn_dx, ' m'
         WRITE(numout,*) '      horizontal resolution             rn_dy  = ', rn_dy, ' m'
         WRITE(numout,*) '      vertical resolution               rn_dz  = ', rn_dz, ' m'
         WRITE(numout,*) '      VORTEX domain: '
         WRITE(numout,*) '         LX [km]: ', zlx
         WRITE(numout,*) '         LY [km]: ', zly
         WRITE(numout,*) '          H [m] : ', zh
         WRITE(numout,*) '      Reference latitude            rn_ppgphi0 = ', rn_ppgphi0
         WRITE(numout,*) '   '
         WRITE(numout,*) '   Lateral boundary condition of the global domain'
         WRITE(numout,*) '      VORTEX : closed basin            jperio = ', kperio
      ENDIF
      !
   END SUBROUTINE usr_def_nam

   !!======================================================================
END MODULE usrdef_nam
