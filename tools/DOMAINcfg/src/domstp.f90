MODULE domstp
   !!==============================================================================
   !!                       ***  MODULE domstp   ***
   !! Ocean initialization : time domain
   !!==============================================================================

   !!----------------------------------------------------------------------
   !!   dom_stp        : ocean time domain initialization
   !!----------------------------------------------------------------------
   !! History :  OPA  ! 1990-10  (O. Marti)  Original code
   !!                 ! 1996-01  (G. Madec)  terrain following coordinates
   !!   NEMO     1.0  ! 2002-08  (G. Madec)  F90: Free form and module
   !!----------------------------------------------------------------------
   USE oce            ! ocean dynamics and tracers
   USE dom_oce        ! ocean space and time domain
   USE in_out_manager ! I/O manager
   USE lib_mpp        ! MPP library

   IMPLICIT NONE
   PRIVATE

   PUBLIC   dom_stp   ! routine called by inidom.F90

   !!----------------------------------------------------------------------
   !! NEMO/OCE 4.0 , NEMO Consortium (2018)
   !! $Id: domstp.F90 6140 2015-12-21 11:35:23Z timgraham $ 
   !! Software governed by the CeCILL licence     (./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE dom_stp
      !!----------------------------------------------------------------------
      !!                    ***  ROUTINE dom_stp  ***
      !!          
      !! ** Purpose :   Intialize ocean time step for the run
      !!
      !! ** Method  : - Initialization of a coef. use in the Asselin time
      !!      filter:  atfp1 = 1 - 2 * atfp  where atfp is the Asselin time
      !!      filter parameter read in namelist
      !!              - Model time step:
      !!                synchronous time intergration.
      !!      There is one time step only, defined by: rdt for dynamics and
      !!      tracer,wind stress, surface heat and salt fluxes
      !!
      !! ** Action  : [REMOVED - rdttra: vertical profile of tracer time step]
      !!              - atfp1    : = 1 - 2*atfp
      !!
      !! References :   Bryan, K., 1984, J. Phys. Oceanogr., 14, 666-673.
      !!----------------------------------------------------------------------
      INTEGER ::   jk              ! dummy loop indice
      !!----------------------------------------------------------------------

      IF(lwp) THEN
         WRITE(numout,*)
         WRITE(numout,*) 'dom_stp : time stepping setting'
         WRITE(numout,*) '~~~~~~~'
      ENDIF

      ! 0. Asselin Time filter
      ! ----------------------
      
      atfp1 = 1. - 2. * atfp

      IF(lwp) WRITE(numout,*)'               synchronous time stepping'
      IF(lwp) WRITE(numout,*)'               dynamics and tracer time step = ', rdt/3600., ' hours'


   END SUBROUTINE dom_stp

   !!======================================================================
END MODULE domstp
