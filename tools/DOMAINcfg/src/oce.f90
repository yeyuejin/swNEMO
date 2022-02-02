MODULE oce
   !!======================================================================
   !!                      ***  MODULE  oce  ***
   !! Ocean        :  dynamics and active tracers defined in memory 
   !!======================================================================
   !! History :  1.0  !  2002-11  (G. Madec)  F90: Free form and module
   !!            3.1  !  2009-02  (G. Madec, M. Leclair)  pure z* coordinate
   !!            3.3  !  2010-09  (C. Ethe) TRA-TRC merge: add ts, gtsu, gtsv 4D arrays
   !!            3.7  !  2014-01  (G. Madec) suppression of curl and before hdiv from in-core memory
   !!----------------------------------------------------------------------
   USE par_oce        ! ocean parameters
   USE lib_mpp        ! MPP library

   IMPLICIT NONE
   PRIVATE

   PUBLIC oce_alloc ! routine called by nemo_init in nemogcm.F90

   !! dynamics and tracer fields                            ! before ! now    ! after  ! the after trends becomes the fields
   !! --------------------------                            ! fields ! fields ! trends ! only after tra_zdf and dyn_spg
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   ub   ,  un    , ua     !: i-horizontal velocity        [m/s]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   vb   ,  vn    , va     !: j-horizontal velocity        [m/s]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::           wn             !: vertical velocity            [m/s]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::           hdivn          !: horizontal divergence        [s-1]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:,:) ::   tsb  ,  tsn   , tsa    !: 4D T-S fields                  [Celcius,psu] 
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:,:) ::   rab_b,  rab_n          !: thermal/haline expansion coef. [Celcius-1,psu-1]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   rn2b ,  rn2            !: brunt-vaisala frequency**2     [s-2]
   !
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:) ::   rhd    !: in situ density anomalie rhd=(rho-rau0)/rau0  [no units]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:) ::   rhop   !: potential volumic mass                           [kg/m3]

   !! free surface                                      !  before  ! now    ! after  !
   !! ------------                                      !  fields  ! fields ! fields !
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   ub_b   ,  un_b  ,  ua_b  !: Barotropic velocities at u-point [m/s]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   vb_b   ,  vn_b  ,  va_b  !: Barotropic velocities at v-point [m/s]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   sshb   ,  sshn  ,  ssha  !: sea surface height at t-point [m]

   !! Arrays at barotropic time step:                   ! befbefore! before !  now   ! after  !
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   ubb_e  ,  ub_e  ,  un_e  , ua_e   !: u-external velocity
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   vbb_e  ,  vb_e  ,  vn_e  , va_e   !: v-external velocity
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   sshbb_e,  sshb_e,  sshn_e, ssha_e !: external ssh
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::                              hu_e   !: external u-depth
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::                              hv_e   !: external v-depth
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::                              hur_e  !: inverse of u-depth
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::                              hvr_e  !: inverse of v-depth
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   ub2_b  , vb2_b           !: Half step fluxes (ln_bt_fw=T)



   !
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   spgu, spgv               !: horizontal surface pressure gradient

   !! interpolated gradient (only used in zps case)
   !! ---------------------
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:) ::   gtsu, gtsv   !: horizontal gradient of T, S bottom u-point 
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)   ::   gru , grv    !: horizontal gradient of rd at bottom u-point

   !! (ISF) interpolated gradient (only used for ice shelf case) 
   !! --------------------- 
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:) ::   gtui, gtvi   !: horizontal gradient of T, S and rd at top u-point 
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)   ::   grui, grvi   !: horizontal gradient of T, S and rd at top v-point  
   !! (ISF) ice load
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)   ::   riceload

   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:) ::   rke          !: kinetic energy

   !! arrays relating to embedding ice in the ocean. These arrays need to be declared 
   !! even if no ice model is required. In the no ice model or traditional levitating 
   !! ice cases they contain only zeros
   !! ---------------------
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)   ::   snwice_mass        !: mass of snow and ice at current  ice time step   [Kg/m2]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)   ::   snwice_mass_b      !: mass of snow and ice at previous ice time step   [Kg/m2]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)   ::   snwice_fmass       !: time evolution of mass of snow+ice               [Kg/m2/s]

   !! Energy budget of the leads (open water embedded in sea ice)
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)   ::   fraqsr_1lev        !: fraction of solar net radiation absorbed in the first ocean level [-]

   !!----------------------------------------------------------------------
   !! NEMO/OPA 3.7 , NEMO Consortium (2014)
   !! $Id: oce.F90 6140 2015-12-21 11:35:23Z timgraham $ 
   !! Software governed by the CeCILL licence     (./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   INTEGER FUNCTION oce_alloc()
      !!----------------------------------------------------------------------
      !!                   ***  FUNCTION oce_alloc  ***
      !!----------------------------------------------------------------------
      INTEGER :: ierr(7)
      !!----------------------------------------------------------------------
      !
      ierr(:) = 0 
      ALLOCATE( ub   (jpi,jpj,jpk)      , un   (jpi,jpj,jpk)      , ua(jpi,jpj,jpk)       ,     &
         &      vb   (jpi,jpj,jpk)      , vn   (jpi,jpj,jpk)      , va(jpi,jpj,jpk)       ,     &          
         &      wn   (jpi,jpj,jpk)      , hdivn(jpi,jpj,jpk)      ,                             &
         &      tsb  (jpi,jpj,jpk,jpts) , tsn  (jpi,jpj,jpk,jpts) , tsa(jpi,jpj,jpk,jpts) ,     &
         &      rab_b(jpi,jpj,jpk,jpts) , rab_n(jpi,jpj,jpk,jpts) ,                             &
         &      rn2b (jpi,jpj,jpk)      , rn2  (jpi,jpj,jpk)      ,                             &
         &      rhd  (jpi,jpj,jpk)      , rhop (jpi,jpj,jpk)                              , STAT=ierr(1) )
         !
      ALLOCATE(rke(jpi,jpj,jpk)  ,                                         &
         &     sshb(jpi,jpj)     , sshn(jpi,jpj)   , ssha(jpi,jpj)   ,     &
         &     ub_b(jpi,jpj)     , un_b(jpi,jpj)   , ua_b(jpi,jpj)   ,     &
         &     vb_b(jpi,jpj)     , vn_b(jpi,jpj)   , va_b(jpi,jpj)   ,     &
         &     spgu  (jpi,jpj)   , spgv(jpi,jpj)   ,                       &
         &     gtsu(jpi,jpj,jpts), gtsv(jpi,jpj,jpts),                     &
         &     gru(jpi,jpj)      , grv(jpi,jpj)      ,                     &
         &     gtui(jpi,jpj,jpts), gtvi(jpi,jpj,jpts),                     &
         &     grui(jpi,jpj)     , grvi(jpi,jpj)     ,                     &
         &     riceload(jpi,jpj),                             STAT=ierr(2) )
         !
      ALLOCATE( snwice_mass(jpi,jpj) , snwice_mass_b(jpi,jpj), snwice_fmass(jpi,jpj) , STAT=ierr(3) )
         !
      ALLOCATE( fraqsr_1lev(jpi,jpj) , STAT=ierr(4) )
         !
      ALLOCATE( ssha_e(jpi,jpj),  sshn_e(jpi,jpj), sshb_e(jpi,jpj), sshbb_e(jpi,jpj), &
         &        ua_e(jpi,jpj),    un_e(jpi,jpj),   ub_e(jpi,jpj),   ubb_e(jpi,jpj), &
         &        va_e(jpi,jpj),    vn_e(jpi,jpj),   vb_e(jpi,jpj),   vbb_e(jpi,jpj), &
         &        hu_e(jpi,jpj),   hur_e(jpi,jpj),   hv_e(jpi,jpj),   hvr_e(jpi,jpj), STAT=ierr(5) )
         !
      ALLOCATE( ub2_b(jpi,jpj), vb2_b(jpi,jpj)                                      , STAT=ierr(6) )



         !
      oce_alloc = MAXVAL( ierr )
      IF( oce_alloc /= 0 )   CALL ctl_warn('oce_alloc: failed to allocate arrays')
      !
   END FUNCTION oce_alloc

   !!======================================================================
END MODULE oce
