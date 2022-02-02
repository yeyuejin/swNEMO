MODULE step_oce
   !!======================================================================
   !!                       ***  MODULE step_oce  ***
   !! Ocean time-stepping : module used in both initialisation phase and time stepping
   !!======================================================================
   !! History :   3.3  !  2010-08  (C. Ethe)  Original code - reorganisation of the initial phase
   !!             3.7  !  2014-01  (G. Madec) LDF simplication 
   !!----------------------------------------------------------------------
   USE oce              ! ocean dynamics and tracers variables
   USE dom_oce          ! ocean space and time domain variables

   USE daymod           ! calendar                         (day     routine)

   USE eosbn2           ! equation of state                (eos_bn2 routine)

   USE prtctl           ! Print control                    (prt_ctl routine)

   USE in_out_manager   ! I/O manager
   USE iom              !
   USE lbclnk
   USE timing           ! Timing



   !!----------------------------------------------------------------------
   !! NEMO/OPA 3.7 , NEMO Consortium (2014)
   !! $Id: step_oce.F90 6140 2015-12-21 11:35:23Z timgraham $
   !! Software governed by the CeCILL licence     (./LICENSE)
   !!======================================================================
END MODULE step_oce
