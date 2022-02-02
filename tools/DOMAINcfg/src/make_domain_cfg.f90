PROGRAM make_domain_cfg
   !!======================================================================
   !!                     ***  PROGRAM  make_domain_cfg  ***
   !!
   !! ** Purpose :  tool to create domain_cfg.nc file via nemogcm
   !!======================================================================
   !! History :   OPA  ! 2001-02  (M. Imbard, A. Weaver)  Original code
   !!   NEMO      1.0  ! 2003-10  (G. Madec) F90
   !!----------------------------------------------------------------------
   USE nemogcm   ! NEMO system   (nemo_gcm routine)
   !!----------------------------------------------------------------------
   !! NEMO/OCE 4.0 , NEMO Consortium (2018)
   !! $Id: nemo.f90 2528 2010-12-27 17:33:53Z rblod $ 
   !! Software governed by the CeCILL licence     (./LICENSE)
   !!----------------------------------------------------------------------
   !
   CALL nemo_gcm           ! NEMO direct code
   ! 
   !!======================================================================
END PROGRAM make_domain_cfg 
