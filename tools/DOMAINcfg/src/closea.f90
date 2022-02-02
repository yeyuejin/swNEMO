MODULE closea
   !!======================================================================
   !!                       ***  MODULE  closea  ***
   !! Closed Seas  : specific treatments associated with closed seas
   !!======================================================================
   !! History :   8.2  !  00-05  (O. Marti)  Original code
   !!             8.5  !  02-06  (E. Durand, G. Madec)  F90
   !!             9.0  !  06-07  (G. Madec)  add clo_rnf, clo_ups, clo_bat
   !!        NEMO 3.4  !  03-12  (P.G. Fogli) sbc_clo bug fix & mpp reproducibility
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   dom_clo    : modification of the ocean domain for closed seas cases
   !!   sbc_clo    : Special handling of closed seas
   !!   clo_rnf    : set close sea outflows as river mouths (see sbcrnf)
   !!   clo_ups    : set mixed centered/upstream scheme in closed sea (see traadv_cen2)
   !!   clo_bat    : set to zero a field over closed sea (see domzrg)
   !!----------------------------------------------------------------------
   USE oce             ! dynamics and tracers
   USE dom_oce         ! ocean space and time domain
   USE phycst          ! physical constants
   USE in_out_manager  ! I/O manager
   USE lib_fortran,    ONLY: glob_sum, DDPDD
   USE lbclnk          ! lateral boundary condition - MPP exchanges
   USE lib_mpp         ! MPP library
   USE timing

   IMPLICIT NONE
   PRIVATE

   PUBLIC dom_clo      ! routine called by domain module
   PUBLIC clo_bat      ! routine called in domzgr module

   INTEGER, PUBLIC, PARAMETER          ::   jpncs   = 4      !: number of closed sea
   INTEGER, PUBLIC, DIMENSION(jpncs)   ::   ncstt            !: Type of closed sea
   INTEGER, PUBLIC, DIMENSION(jpncs)   ::   ncsi1, ncsj1     !: south-west closed sea limits (i,j)
   INTEGER, PUBLIC, DIMENSION(jpncs)   ::   ncsi2, ncsj2     !: north-east closed sea limits (i,j)
   INTEGER, PUBLIC, DIMENSION(jpncs)   ::   ncsnr            !: number of point where run-off pours
   INTEGER, PUBLIC, DIMENSION(jpncs,4) ::   ncsir, ncsjr     !: Location of runoff

   REAL(wp), DIMENSION (jpncs+1)       ::   surf             ! closed sea surface

   !! * Substitutions
   !!----------------------------------------------------------------------
   !!                   ***  vectopt_loop_substitute  ***
   !!----------------------------------------------------------------------
   !! ** purpose :   substitute the inner loop start/end indices with CPP macro
   !!                allow unrolling of do-loop (useful with vector processors)
   !!----------------------------------------------------------------------
   !!----------------------------------------------------------------------
   !! NEMO/OPA 3.7 , NEMO Consortium (2014)
   !! $Id: vectopt_loop_substitute.h90 4990 2014-12-15 16:42:49Z timgraham $ 
   !! Software governed by the CeCILL licence (./LICENSE)
   !!----------------------------------------------------------------------
   !!----------------------------------------------------------------------
   !! NEMO/OCE 4.0 , NEMO Consortium (2018)
   !! $Id: closea.F90 5836 2015-10-26 14:49:40Z cetlod $
   !! Software governed by the CeCILL licence     (./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE dom_clo
      !!---------------------------------------------------------------------
      !!                  ***  ROUTINE dom_clo  ***
      !!        
      !! ** Purpose :   Closed sea domain initialization
      !!
      !! ** Method  :   if a closed sea is located only in a model grid point
      !!                just the thermodynamic processes are applied.
      !!
      !! ** Action  :   ncsi1(), ncsj1() : south-west closed sea limits (i,j)
      !!                ncsi2(), ncsj2() : north-east Closed sea limits (i,j)
      !!                ncsir(), ncsjr() : Location of runoff
      !!                ncsnr            : number of point where run-off pours
      !!                ncstt            : Type of closed sea
      !!                                   =0 spread over the world ocean
      !!                                   =2 put at location runoff
      !!----------------------------------------------------------------------
      INTEGER ::   jc      ! dummy loop indices
      INTEGER ::   isrow   ! local index
      !!----------------------------------------------------------------------
      !
      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*)'dom_clo : closed seas '
      IF(lwp) WRITE(numout,*)'~~~~~~~'
      !
      ! initial values
      ncsnr(:) = 1  ;  ncsi1(:) = 1  ;  ncsi2(:) = 1  ;  ncsir(:,:) = 1
      ncstt(:) = 0  ;  ncsj1(:) = 1  ;  ncsj2(:) = 1  ;  ncsjr(:,:) = 1
      !
      ! set the closed seas (in data domain indices)
      ! -------------------
      !
      IF( cp_cfg == "orca" ) THEN
         !
         SELECT CASE ( jp_cfg )
         !                                           ! =======================
         CASE ( 1 )                                  ! ORCA_R1 configuration
            !                                        ! =======================
            ! This dirty section will be suppressed by simplification process:
            ! all this will come back in input files
            ! Currently these hard-wired indices relate to configuration with
            ! extend grid (jpjglo=332)
            isrow = 332 - jpjglo
            !
            ncsnr(1)   = 1    ; ncstt(1)   = 0           ! Caspian Sea
            ncsi1(1)   = 332  ; ncsj1(1)   = 243 - isrow
            ncsi2(1)   = 344  ; ncsj2(1)   = 275 - isrow
            ncsir(1,1) = 1    ; ncsjr(1,1) = 1
            !                                        
            !                                        ! =======================
         CASE ( 2 )                                  !  ORCA_R2 configuration
            !                                        ! =======================
            !                                            ! Caspian Sea
            ncsnr(1)   =   1  ;  ncstt(1)   =   0           ! spread over the globe
            ncsi1(1)   =  11  ;  ncsj1(1)   = 103
            ncsi2(1)   =  17  ;  ncsj2(1)   = 112
            ncsir(1,1) =   1  ;  ncsjr(1,1) =   1 
            !                                            ! Great North American Lakes
            ncsnr(2)   =   1  ;  ncstt(2)   =   2           ! put at St Laurent mouth
            ncsi1(2)   =  97  ;  ncsj1(2)   = 107
            ncsi2(2)   = 103  ;  ncsj2(2)   = 111
            ncsir(2,1) = 110  ;  ncsjr(2,1) = 111           
            !                                            ! Black Sea (crossed by the cyclic boundary condition)
            ncsnr(3:4) =   4  ;  ncstt(3:4) =   2           ! put in Med Sea (north of Aegean Sea)
            ncsir(3:4,1) = 171;  ncsjr(3:4,1) = 106         !
            ncsir(3:4,2) = 170;  ncsjr(3:4,2) = 106 
            ncsir(3:4,3) = 171;  ncsjr(3:4,3) = 105 
            ncsir(3:4,4) = 170;  ncsjr(3:4,4) = 105 
            ncsi1(3)   = 174  ;  ncsj1(3)   = 107           ! 1 : west part of the Black Sea      
            ncsi2(3)   = 181  ;  ncsj2(3)   = 112           !            (ie west of the cyclic b.c.)
            ncsi1(4)   =   2  ;  ncsj1(4)   = 107           ! 2 : east part of the Black Sea 
            ncsi2(4)   =   6  ;  ncsj2(4)   = 112           !           (ie east of the cyclic b.c.)
             
          

            !                                        ! =======================
         CASE ( 4 )                                  !  ORCA_R4 configuration
            !                                        ! =======================
            !                                            ! Caspian Sea
            ncsnr(1)   =  1  ;  ncstt(1)   =  0  
            ncsi1(1)   =  4  ;  ncsj1(1)   = 53 
            ncsi2(1)   =  4  ;  ncsj2(1)   = 56
            ncsir(1,1) =  1  ;  ncsjr(1,1) =  1
            !                                            ! Great North American Lakes
            ncsnr(2)   =  1  ;  ncstt(2)   =  2 
            ncsi1(2)   = 49  ;  ncsj1(2)   = 55
            ncsi2(2)   = 51  ;  ncsj2(2)   = 56
            ncsir(2,1) = 57  ;  ncsjr(2,1) = 55
            !                                            ! Black Sea
            ncsnr(3)   =  4  ;  ncstt(3)   =  2  
            ncsi1(3)   = 88  ;  ncsj1(3)   = 55 
            ncsi2(3)   = 91  ;  ncsj2(3)   = 56
            ncsir(3,1) = 86  ;  ncsjr(3,1) = 53
            ncsir(3,2) = 87  ;  ncsjr(3,2) = 53 
            ncsir(3,3) = 86  ;  ncsjr(3,3) = 52 
            ncsir(3,4) = 87  ;  ncsjr(3,4) = 52
            !                                            ! Baltic Sea
            ncsnr(4)   =  1  ;  ncstt(4)   =  2
            ncsi1(4)   = 75  ;  ncsj1(4)   = 59
            ncsi2(4)   = 76  ;  ncsj2(4)   = 61
            ncsir(4,1) = 84  ;  ncsjr(4,1) = 59 
            !                                        ! =======================
         CASE ( 025 )                                ! ORCA_R025 configuration
            !                                        ! =======================
            ncsnr(1)   = 1    ; ncstt(1)   = 0               ! Caspian + Aral sea
            ncsi1(1)   = 1330 ; ncsj1(1)   = 645
            ncsi2(1)   = 1400 ; ncsj2(1)   = 795
            ncsir(1,1) = 1    ; ncsjr(1,1) = 1
            !                                        
            ncsnr(2)   = 1    ; ncstt(2)   = 0               ! Azov Sea 
            ncsi1(2)   = 1284 ; ncsj1(2)   = 722
            ncsi2(2)   = 1304 ; ncsj2(2)   = 747
            ncsir(2,1) = 1    ; ncsjr(2,1) = 1
            !
         END SELECT
         !
      ENDIF

      ! convert the position in local domain indices
      ! --------------------------------------------
      DO jc = 1, jpncs
         ncsi1(jc)   = mi0( ncsi1(jc) )
         ncsj1(jc)   = mj0( ncsj1(jc) )

         ncsi2(jc)   = mi1( ncsi2(jc) )   
         ncsj2(jc)   = mj1( ncsj2(jc) )  
      END DO
      !
   END SUBROUTINE dom_clo


   SUBROUTINE clo_bat( pbat, kbat )
      !!---------------------------------------------------------------------
      !!                  ***  ROUTINE clo_bat  ***
      !!                    
      !! ** Purpose :   suppress closed sea from the domain
      !!
      !! ** Method  :   set to 0 the meter and level bathymetry (given in 
      !!                arguments) over the closed seas.
      !!
      !! ** Action  :   set pbat=0 and kbat=0 over closed seas
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj), INTENT(inout) ::   pbat   ! bathymetry in meters (bathy array)
      INTEGER , DIMENSION(jpi,jpj), INTENT(inout) ::   kbat   ! bathymetry in levels (mbathy array)
      !
      INTEGER  ::   jc, ji, jj      ! dummy loop indices
      !!----------------------------------------------------------------------
      !
      DO jc = 1, jpncs
         DO jj = ncsj1(jc), ncsj2(jc)
            DO ji = ncsi1(jc), ncsi2(jc)
               pbat(ji,jj) = 0._wp   
               kbat(ji,jj) = 0   
            END DO 
         END DO 
       END DO 
       !
   END SUBROUTINE clo_bat

   !!======================================================================
END MODULE closea

