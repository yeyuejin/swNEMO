MODULE domhgr
   !!==============================================================================
   !!                       ***  MODULE domhgr   ***
   !! Ocean initialization : domain initialization
   !!==============================================================================
   !! History :  OPA  ! 1988-03  (G. Madec) Original code
   !!            7.0  ! 1996-01  (G. Madec)  terrain following coordinates
   !!            8.0  ! 1997-02  (G. Madec)  print mesh informations
   !!            8.1  ! 1999-11  (M. Imbard) NetCDF format with IO-IPSL
   !!            8.2  ! 2000-08  (D. Ludicone) Reduced section at Bab el Mandeb
   !!             -   ! 2001-09  (M. Levy)  eel config: grid in km, beta-plane
   !!  NEMO      1.0  ! 2002-08  (G. Madec)  F90: Free form and module, namelist
   !!             -   ! 2004-01  (A.M. Treguier, J.M. Molines) Case 4 (Mercator mesh)
   !!                            use of parameters in par_CONFIG-Rxx.h90, not in namelist
   !!             -   ! 2004-05  (A. Koch-Larrouy) Add Gyre configuration 
   !!            3.7  ! 2015-09  (G. Madec, S. Flavoni) add cell surface and their inverse
   !!                                       add optional read of e1e2u & e1e2v
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   dom_hgr       : initialize the horizontal mesh 
   !!   hgr_read      : read "coordinate" NetCDF file 
   !!----------------------------------------------------------------------
   USE dom_oce        ! ocean space and time domain
   USE phycst         ! physical constants
   USE domwri         ! write 'meshmask.nc' & 'coordinate_e1e2u_v.nc' files
   !
   USE in_out_manager ! I/O manager
   USE lib_mpp        ! MPP library
   USE timing         ! Timing

   IMPLICIT NONE
   PRIVATE

   REAL(wp) ::   glam0, gphi0   ! variables corresponding to parameters ppglam0 ppgphi0 set in par_oce

   PUBLIC   dom_hgr   ! called by domain.F90

   !!----------------------------------------------------------------------
   !! NEMO/OPA 3.7 , NEMO Consortium (2014)
   !! $Id: domhgr.F90 6140 2015-12-21 11:35:23Z timgraham $ 
   !! Software governed by the CeCILL licence     (./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE dom_hgr
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE dom_hgr  ***
      !!
      !! ** Purpose :   Compute the geographical position (in degre) of the 
      !!      model grid-points,  the horizontal scale factors (in meters) and 
      !!      the Coriolis factor (in s-1).
      !!
      !! ** Method  :   The geographical position of the model grid-points is
      !!      defined from analytical functions, fslam and fsphi, the deriva-
      !!      tives of which gives the horizontal scale factors e1,e2.
      !!      Defining two function fslam and fsphi and their derivatives in 
      !!      the two horizontal directions (fse1 and fse2), the model grid-
      !!      point position and scale factors are given by:
      !!         t-point:
      !!      glamt(i,j) = fslam(i    ,j    )   e1t(i,j) = fse1(i    ,j    )
      !!      gphit(i,j) = fsphi(i    ,j    )   e2t(i,j) = fse2(i    ,j    )
      !!         u-point:
      !!      glamu(i,j) = fslam(i+1/2,j    )   e1u(i,j) = fse1(i+1/2,j    )
      !!      gphiu(i,j) = fsphi(i+1/2,j    )   e2u(i,j) = fse2(i+1/2,j    )
      !!         v-point:
      !!      glamv(i,j) = fslam(i    ,j+1/2)   e1v(i,j) = fse1(i    ,j+1/2)
      !!      gphiv(i,j) = fsphi(i    ,j+1/2)   e2v(i,j) = fse2(i    ,j+1/2)
      !!            f-point:
      !!      glamf(i,j) = fslam(i+1/2,j+1/2)   e1f(i,j) = fse1(i+1/2,j+1/2)
      !!      gphif(i,j) = fsphi(i+1/2,j+1/2)   e2f(i,j) = fse2(i+1/2,j+1/2)
      !!      Where fse1 and fse2 are defined by:
      !!         fse1(i,j) = ra * rad * SQRT( (cos(phi) di(fslam))**2
      !!                                     +          di(fsphi) **2 )(i,j)
      !!         fse2(i,j) = ra * rad * SQRT( (cos(phi) dj(fslam))**2
      !!                                     +          dj(fsphi) **2 )(i,j)
      !!
      !!        The coriolis factor is given at z-point by:
      !!                     ff = 2.*omega*sin(gphif)      (in s-1)
      !!
      !!        This routine is given as an example, it must be modified
      !!      following the user s desiderata. nevertheless, the output as
      !!      well as the way to compute the model grid-point position and
      !!      horizontal scale factors must be respected in order to insure
      !!      second order accuracy schemes.
      !!
      !! N.B. If the domain is periodic, verify that scale factors are also
      !!      periodic, and the coriolis term again.
      !!
      !! ** Action  : - define  glamt, glamu, glamv, glamf: longitude of t-, 
      !!                u-, v- and f-points (in degre)
      !!              - define  gphit, gphiu, gphiv, gphit: latitude  of t-,
      !!               u-, v-  and f-points (in degre)
      !!        define e1t, e2t, e1u, e2u, e1v, e2v, e1f, e2f: horizontal
      !!      scale factors (in meters) at t-, u-, v-, and f-points.
      !!        define ff: coriolis factor at f-point
      !!
      !! References :   Marti, Madec and Delecluse, 1992, JGR
      !!                Madec, Imbard, 1996, Clim. Dyn.
      !!----------------------------------------------------------------------
      INTEGER  ::   ji, jj               ! dummy loop indices
      INTEGER  ::   ii0, ii1, ij0, ij1   ! temporary integers
      INTEGER  ::   ijeq                 ! index of equator T point (used in case 4)
      REAL(wp) ::   zti, zui, zvi, zfi   ! local scalars
      REAL(wp) ::   ztj, zuj, zvj, zfj   !   -      -
      REAL(wp) ::   zphi0, zbeta, znorme !
      REAL(wp) ::   zarg, zf0, zminff, zmaxff
      REAL(wp) ::   zlam1, zcos_alpha, zim1 , zjm1 , ze1, ze1deg
      REAL(wp) ::   zphi1, zsin_alpha, zim05, zjm05
      INTEGER  ::   isrow                ! index for ORCA1 starting row
      INTEGER  ::   ie1e2u_v             ! fag for u- & v-surface read in coordinate file or not
      !!----------------------------------------------------------------------
      !
      IF( nn_timing == 1 )  CALL timing_start('dom_hgr')
      !
      IF(lwp) THEN
         WRITE(numout,*)
         WRITE(numout,*) 'dom_hgr : define the horizontal mesh from ithe following par_oce parameters '
         WRITE(numout,*) '~~~~~~~      type of horizontal mesh           jphgr_msh = ', jphgr_msh
         WRITE(numout,*) '             position of the first row and     ppglam0  = ', ppglam0
         WRITE(numout,*) '             column grid-point (degrees)       ppgphi0  = ', ppgphi0
         WRITE(numout,*) '             zonal      grid-spacing (degrees) ppe1_deg = ', ppe1_deg
         WRITE(numout,*) '             meridional grid-spacing (degrees) ppe2_deg = ', ppe2_deg
         WRITE(numout,*) '             zonal      grid-spacing (meters)  ppe1_m   = ', ppe1_m  
         WRITE(numout,*) '             meridional grid-spacing (meters)  ppe2_m   = ', ppe2_m  
      ENDIF
      !
      !
      SELECT CASE( jphgr_msh )   !  type of horizontal mesh  
      !
      CASE ( 0 )                     !==  read in coordinate.nc file  ==!
         !
         IF(lwp) WRITE(numout,*)
         IF(lwp) WRITE(numout,*) '          curvilinear coordinate on the sphere read in "coordinate" file'
         !
         ie1e2u_v = 0                  ! set to unread e1e2u and e1e2v
         !
         CALL hgr_read( ie1e2u_v )     ! read the coordinate.nc file
         !
         IF( ie1e2u_v == 0 ) THEN      ! e1e2u and e1e2v have not been read: compute them
            !                          ! e2u and e1v does not include a reduction in some strait: apply reduction
            e1e2u (:,:) = e1u(:,:) * e2u(:,:)   
            e1e2v (:,:) = e1v(:,:) * e2v(:,:) 
         ENDIF
         !
      CASE ( 1 )                     !==  geographical mesh on the sphere with regular (in degree) grid-spacing  ==!
         !
         IF(lwp) WRITE(numout,*)
         IF(lwp) WRITE(numout,*) '          geographical mesh on the sphere with regular grid-spacing'
         IF(lwp) WRITE(numout,*) '          given by ppe1_deg and ppe2_deg' 
         !
         DO jj = 1, jpj
            DO ji = 1, jpi
               zti = REAL( ji - 1 + nimpp - 1 )         ;   ztj = REAL( jj - 1 + njmpp - 1 )
               zui = REAL( ji - 1 + nimpp - 1 ) + 0.5   ;   zuj = REAL( jj - 1 + njmpp - 1 )
               zvi = REAL( ji - 1 + nimpp - 1 )         ;   zvj = REAL( jj - 1 + njmpp - 1 ) + 0.5
               zfi = REAL( ji - 1 + nimpp - 1 ) + 0.5   ;   zfj = REAL( jj - 1 + njmpp - 1 ) + 0.5
         ! Longitude
               glamt(ji,jj) = ppglam0 + ppe1_deg * zti
               glamu(ji,jj) = ppglam0 + ppe1_deg * zui
               glamv(ji,jj) = ppglam0 + ppe1_deg * zvi
               glamf(ji,jj) = ppglam0 + ppe1_deg * zfi
         ! Latitude
               gphit(ji,jj) = ppgphi0 + ppe2_deg * ztj
               gphiu(ji,jj) = ppgphi0 + ppe2_deg * zuj
               gphiv(ji,jj) = ppgphi0 + ppe2_deg * zvj
               gphif(ji,jj) = ppgphi0 + ppe2_deg * zfj
         ! e1
               e1t(ji,jj) = ra * rad * COS( rad * gphit(ji,jj) ) * ppe1_deg
               e1u(ji,jj) = ra * rad * COS( rad * gphiu(ji,jj) ) * ppe1_deg
               e1v(ji,jj) = ra * rad * COS( rad * gphiv(ji,jj) ) * ppe1_deg
               e1f(ji,jj) = ra * rad * COS( rad * gphif(ji,jj) ) * ppe1_deg
         ! e2
               e2t(ji,jj) = ra * rad * ppe2_deg
               e2u(ji,jj) = ra * rad * ppe2_deg
               e2v(ji,jj) = ra * rad * ppe2_deg
               e2f(ji,jj) = ra * rad * ppe2_deg
            END DO
         END DO
         !
      CASE ( 2:3 )                   !==  f- or beta-plane with regular grid-spacing  ==!
         !
         IF(lwp) WRITE(numout,*)
         IF(lwp) WRITE(numout,*) '          f- or beta-plane with regular grid-spacing'
         IF(lwp) WRITE(numout,*) '          given by ppe1_m and ppe2_m' 
         !
         ! Position coordinates (in kilometers)
         !                          ==========
         glam0 = 0._wp
         gphi0 = - ppe2_m * 1.e-3
         !
         DO jj = 1, jpj
            DO ji = 1, jpi
               glamt(ji,jj) = glam0 + ppe1_m * 1.e-3 * ( REAL( ji - 1 + nimpp - 1 )       )
               glamu(ji,jj) = glam0 + ppe1_m * 1.e-3 * ( REAL( ji - 1 + nimpp - 1 ) + 0.5 )
               glamv(ji,jj) = glamt(ji,jj)
               glamf(ji,jj) = glamu(ji,jj)
               !
               gphit(ji,jj) = gphi0 + ppe2_m * 1.e-3 * ( REAL( jj - 1 + njmpp - 1 )       )
               gphiu(ji,jj) = gphit(ji,jj)
               gphiv(ji,jj) = gphi0 + ppe2_m * 1.e-3 * ( REAL( jj - 1 + njmpp - 1 ) + 0.5 )
               gphif(ji,jj) = gphiv(ji,jj)
            END DO
         END DO
         !
         ! Horizontal scale factors (in meters)
         !                              ======
         e1t(:,:) = ppe1_m      ;      e2t(:,:) = ppe2_m
         e1u(:,:) = ppe1_m      ;      e2u(:,:) = ppe2_m
         e1v(:,:) = ppe1_m      ;      e2v(:,:) = ppe2_m
         e1f(:,:) = ppe1_m      ;      e2f(:,:) = ppe2_m
         !
      CASE ( 4 )                     !==  geographical mesh on the sphere, isotropic MERCATOR type  ==!
         !
         IF(lwp) WRITE(numout,*)
         IF(lwp) WRITE(numout,*) '          geographical mesh on the sphere, MERCATOR type'
         IF(lwp) WRITE(numout,*) '          longitudinal/latitudinal spacing given by ppe1_deg'
         IF ( ppgphi0 == -90 ) CALL ctl_stop( ' Mercator grid cannot start at south pole !!!! ' )
         !
         !  Find index corresponding to the equator, given the grid spacing e1_deg
         !  and the (approximate) southern latitude ppgphi0.
         !  This way we ensure that the equator is at a "T / U" point, when in the domain.
         !  The formula should work even if the equator is outside the domain.
         zarg = rpi / 4. - rpi / 180. * ppgphi0 / 2.
         ijeq = ABS( 180./rpi * LOG( COS( zarg ) / SIN( zarg ) ) / ppe1_deg )
         IF(  ppgphi0 > 0 )  ijeq = -ijeq
         !
         IF(lwp) WRITE(numout,*) '          Index of the equator on the MERCATOR grid:', ijeq
         !
         DO jj = 1, jpj
            DO ji = 1, jpi
               zti = REAL( ji - 1 + nimpp - 1 )         ;   ztj = REAL( jj - ijeq + njmpp - 1 )
               zui = REAL( ji - 1 + nimpp - 1 ) + 0.5   ;   zuj = REAL( jj - ijeq + njmpp - 1 )
               zvi = REAL( ji - 1 + nimpp - 1 )         ;   zvj = REAL( jj - ijeq + njmpp - 1 ) + 0.5
               zfi = REAL( ji - 1 + nimpp - 1 ) + 0.5   ;   zfj = REAL( jj - ijeq + njmpp - 1 ) + 0.5
         ! Longitude
               glamt(ji,jj) = ppglam0 + ppe1_deg * zti
               glamu(ji,jj) = ppglam0 + ppe1_deg * zui
               glamv(ji,jj) = ppglam0 + ppe1_deg * zvi
               glamf(ji,jj) = ppglam0 + ppe1_deg * zfi
         ! Latitude
               gphit(ji,jj) = 1./rad * ASIN ( TANH( ppe1_deg *rad* ztj ) )
               gphiu(ji,jj) = 1./rad * ASIN ( TANH( ppe1_deg *rad* zuj ) )
               gphiv(ji,jj) = 1./rad * ASIN ( TANH( ppe1_deg *rad* zvj ) )
               gphif(ji,jj) = 1./rad * ASIN ( TANH( ppe1_deg *rad* zfj ) )
         ! e1
               e1t(ji,jj) = ra * rad * COS( rad * gphit(ji,jj) ) * ppe1_deg
               e1u(ji,jj) = ra * rad * COS( rad * gphiu(ji,jj) ) * ppe1_deg
               e1v(ji,jj) = ra * rad * COS( rad * gphiv(ji,jj) ) * ppe1_deg
               e1f(ji,jj) = ra * rad * COS( rad * gphif(ji,jj) ) * ppe1_deg
         ! e2
               e2t(ji,jj) = ra * rad * COS( rad * gphit(ji,jj) ) * ppe1_deg
               e2u(ji,jj) = ra * rad * COS( rad * gphiu(ji,jj) ) * ppe1_deg
               e2v(ji,jj) = ra * rad * COS( rad * gphiv(ji,jj) ) * ppe1_deg
               e2f(ji,jj) = ra * rad * COS( rad * gphif(ji,jj) ) * ppe1_deg
            END DO
         END DO
         !
      CASE ( 5 )                   !==  beta-plane with regular grid-spacing and rotated domain ==! (GYRE configuration)
         !
         IF(lwp) WRITE(numout,*)
         IF(lwp) WRITE(numout,*) '          beta-plane with regular grid-spacing and rotated domain (GYRE configuration)'
         IF(lwp) WRITE(numout,*) '          given by ppe1_m and ppe2_m'
         !
         ! Position coordinates (in kilometers)
         !                          ==========
         !
         ! angle 45deg and ze1=106.e+3 / jp_cfg forced -> zlam1 = -85deg, zphi1 = 29degN
         zlam1 = -85._wp
         zphi1 =  29._wp
         ! resolution in meters
         ze1 = 106000. / REAL( jp_cfg , wp )            
         ! benchmark: forced the resolution to be about 100 km
         IF( nbench /= 0 )   ze1 = 106000._wp     
         zsin_alpha = - SQRT( 2._wp ) * 0.5_wp
         zcos_alpha =   SQRT( 2._wp ) * 0.5_wp
         ze1deg = ze1 / (ra * rad)
         IF( nbench /= 0 )   ze1deg = ze1deg / REAL( jp_cfg , wp )   ! benchmark: keep the lat/+lon
         !                                                           ! at the right jp_cfg resolution
         glam0 = zlam1 + zcos_alpha * ze1deg * REAL( jpjglo-2 , wp )
         gphi0 = zphi1 + zsin_alpha * ze1deg * REAL( jpjglo-2 , wp )
         !
         IF( nprint==1 .AND. lwp )   THEN
            WRITE(numout,*) '          ze1', ze1, 'cosalpha', zcos_alpha, 'sinalpha', zsin_alpha
            WRITE(numout,*) '          ze1deg', ze1deg, 'glam0', glam0, 'gphi0', gphi0
         ENDIF
         !
         DO jj = 1, jpj
            DO ji = 1, jpi
               zim1 = REAL( ji + nimpp - 1 ) - 1.   ;   zim05 = REAL( ji + nimpp - 1 ) - 1.5
               zjm1 = REAL( jj + njmpp - 1 ) - 1.   ;   zjm05 = REAL( jj + njmpp - 1 ) - 1.5
               !
               glamf(ji,jj) = glam0 + zim1  * ze1deg * zcos_alpha + zjm1  * ze1deg * zsin_alpha
               gphif(ji,jj) = gphi0 - zim1  * ze1deg * zsin_alpha + zjm1  * ze1deg * zcos_alpha
               !
               glamt(ji,jj) = glam0 + zim05 * ze1deg * zcos_alpha + zjm05 * ze1deg * zsin_alpha
               gphit(ji,jj) = gphi0 - zim05 * ze1deg * zsin_alpha + zjm05 * ze1deg * zcos_alpha
               !
               glamu(ji,jj) = glam0 + zim1  * ze1deg * zcos_alpha + zjm05 * ze1deg * zsin_alpha
               gphiu(ji,jj) = gphi0 - zim1  * ze1deg * zsin_alpha + zjm05 * ze1deg * zcos_alpha
               !
               glamv(ji,jj) = glam0 + zim05 * ze1deg * zcos_alpha + zjm1  * ze1deg * zsin_alpha
               gphiv(ji,jj) = gphi0 - zim05 * ze1deg * zsin_alpha + zjm1  * ze1deg * zcos_alpha
            END DO
         END DO
         !
         ! Horizontal scale factors (in meters)
         !                              ======
         e1t(:,:) =  ze1     ;      e2t(:,:) = ze1
         e1u(:,:) =  ze1     ;      e2u(:,:) = ze1
         e1v(:,:) =  ze1     ;      e2v(:,:) = ze1
         e1f(:,:) =  ze1     ;      e2f(:,:) = ze1
         !
      CASE DEFAULT
         WRITE(ctmp1,*) '          bad flag value for jphgr_msh = ', jphgr_msh
         CALL ctl_stop( ctmp1 )
         !
      END SELECT
      
      ! associated horizontal metrics
      ! -----------------------------
      !
      r1_e1t(:,:) = 1._wp / e1t(:,:)   ;   r1_e2t (:,:) = 1._wp / e2t(:,:)
      r1_e1u(:,:) = 1._wp / e1u(:,:)   ;   r1_e2u (:,:) = 1._wp / e2u(:,:)
      r1_e1v(:,:) = 1._wp / e1v(:,:)   ;   r1_e2v (:,:) = 1._wp / e2v(:,:)
      r1_e1f(:,:) = 1._wp / e1f(:,:)   ;   r1_e2f (:,:) = 1._wp / e2f(:,:)
      !
      e1e2t (:,:) = e1t(:,:) * e2t(:,:)   ;   r1_e1e2t(:,:) = 1._wp / e1e2t(:,:)
      e1e2f (:,:) = e1f(:,:) * e2f(:,:)   ;   r1_e1e2f(:,:) = 1._wp / e1e2f(:,:)
      IF( jphgr_msh /= 0 ) THEN               ! e1e2u and e1e2v have not been set: compute them
         e1e2u (:,:) = e1u(:,:) * e2u(:,:)   
         e1e2v (:,:) = e1v(:,:) * e2v(:,:) 
      ENDIF
      r1_e1e2u(:,:) = 1._wp / e1e2u(:,:)     ! compute their invert in both cases
      r1_e1e2v(:,:) = 1._wp / e1e2v(:,:)
      !   
      e2_e1u(:,:) = e2u(:,:) / e1u(:,:)
      e1_e2v(:,:) = e1v(:,:) / e2v(:,:)

      IF( lwp .AND. nn_print >=1 .AND. .NOT.ln_rstart ) THEN      ! Control print : Grid informations (if not restart)
         WRITE(numout,*)
         WRITE(numout,*) '          longitude and e1 scale factors'
         WRITE(numout,*) '          ------------------------------'
         WRITE(numout,9300) ( ji, glamt(ji,1), glamu(ji,1),   &
            glamv(ji,1), glamf(ji,1),   &
            e1t(ji,1), e1u(ji,1),   &
            e1v(ji,1), e1f(ji,1), ji = 1, jpi,10)
9300     FORMAT( 1x, i4, f8.2,1x, f8.2,1x, f8.2,1x, f8.2, 1x,    &
            f19.10, 1x, f19.10, 1x, f19.10, 1x, f19.10 )
            !
         WRITE(numout,*)
         WRITE(numout,*) '          latitude and e2 scale factors'
         WRITE(numout,*) '          -----------------------------'
         WRITE(numout,9300) ( jj, gphit(1,jj), gphiu(1,jj),   &
            &                     gphiv(1,jj), gphif(1,jj),   &
            &                     e2t  (1,jj), e2u  (1,jj),   &
            &                     e2v  (1,jj), e2f  (1,jj), jj = 1, jpj, 10 )
      ENDIF


      ! ================= !
      !  Coriolis factor  !
      ! ================= !

      SELECT CASE( jphgr_msh )   ! type of horizontal mesh
      !
      CASE ( 0, 1, 4 )               ! mesh on the sphere
         !
         ff_f(:,:) = 2. * omega * SIN( rad * gphif(:,:) ) 
         ff_t(:,:) = 2. * omega * SIN( rad * gphit(:,:) )     !    -        -       - at t-point
         !
      CASE ( 2 )                     ! f-plane at ppgphi0 
         !
         ff_f(:,:) = 2. * omega * SIN( rad * ppgphi0 )
         ff_t(:,:) = 2. * omega * SIN( rad * ppgphi0 )
         !
         IF(lwp) WRITE(numout,*) '          f-plane: Coriolis parameter = constant = ', ff_f(1,1)
         !
      CASE ( 3 )                     ! beta-plane
         !
         zbeta   = 2. * omega * COS( rad * ppgphi0 ) / ra                       ! beta at latitude ppgphi0
         zphi0   = ppgphi0 - REAL( jpjglo/2) * ppe2_m / ( ra * rad )           ! latitude of the first row F-points
         !
         zf0     = 2. * omega * SIN( rad * zphi0 )                              ! compute f0 1st point south
         !
         ff_f(:,:) = ( zf0  + zbeta * gphif(:,:) * 1.e+3 )                        ! f = f0 +beta* y ( y=0 at south)
         ff_t(:,:) = ( zf0  + zbeta * gphit(:,:) * 1.e+3 )                        ! f = f0 +beta* y ( y=0 at south)
         !
         IF(lwp) THEN
            WRITE(numout,*) 
            WRITE(numout,*) '          Beta-plane: Beta parameter = constant = ', ff_f(nldi,nldj)
            WRITE(numout,*) '          Coriolis parameter varies from ', ff_f(nldi,nldj),' to ', ff_f(nldi,nlej)
         ENDIF
         IF( lk_mpp ) THEN 
            zminff=ff_f(nldi,nldj)
            zmaxff=ff_f(nldi,nlej)
            CALL mpp_min( zminff )   ! min over the global domain
            CALL mpp_max( zmaxff )   ! max over the global domain
            IF(lwp) WRITE(numout,*) '          Coriolis parameter varies globally from ', zminff,' to ', zmaxff
         END IF
         !
      CASE ( 5 )                     ! beta-plane and rotated domain (gyre configuration)
         !
         zbeta = 2. * omega * COS( rad * ppgphi0 ) / ra                     ! beta at latitude ppgphi0
         zphi0 = 15._wp                                                     ! latitude of the first row F-points
         zf0   = 2. * omega * SIN( rad * zphi0 )                            ! compute f0 1st point south
         !
         ff_f(:,:) = ( zf0 + zbeta * ABS( gphif(:,:) - zphi0 ) * rad * ra )   ! f = f0 +beta* y ( y=0 at south)
         ff_t(:,:) = ( zf0 + zbeta * ABS( gphit(:,:) - zphi0 ) * rad * ra )   ! f = f0 +beta* y ( y=0 at south)
         !
         IF(lwp) THEN
            WRITE(numout,*) 
            WRITE(numout,*) '          Beta-plane and rotated domain : '
            WRITE(numout,*) '          Coriolis parameter varies in this processor from ', ff_f(nldi,nldj),' to ', ff_f(nldi,nlej)
         ENDIF
         !
         IF( lk_mpp ) THEN 
            zminff=ff_f(nldi,nldj)
            zmaxff=ff_f(nldi,nlej)
            CALL mpp_min( zminff )   ! min over the global domain
            CALL mpp_max( zmaxff )   ! max over the global domain
            IF(lwp) WRITE(numout,*) '          Coriolis parameter varies globally from ', zminff,' to ', zmaxff
         END IF
         !
      END SELECT


      ! Control of domain for symetrical condition
      ! ------------------------------------------
      ! The equator line must be the latitude coordinate axe

      IF( nperio == 2 ) THEN
         znorme = SQRT( SUM( gphiu(:,2) * gphiu(:,2) ) ) / REAL( jpi )
         IF( znorme > 1.e-13 ) CALL ctl_stop( ' ===>>>> : symmetrical condition: rerun with good equator line' )
      ENDIF
      !
      IF( nn_timing == 1 )  CALL timing_stop('dom_hgr')
      !
   END SUBROUTINE dom_hgr


   SUBROUTINE hgr_read( ke1e2u_v )
      !!---------------------------------------------------------------------
      !!              ***  ROUTINE hgr_read  ***
      !!
      !! ** Purpose :   Read a coordinate file in NetCDF format using IOM
      !!
      !!----------------------------------------------------------------------
      USE iom
      !!
      INTEGER, INTENT( inout ) ::   ke1e2u_v   ! fag: e1e2u & e1e2v read in coordinate file (=1) or not (=0)
      !
      INTEGER ::   inum   ! temporary logical unit
      !!----------------------------------------------------------------------
      !
      IF(lwp) THEN
         WRITE(numout,*)
         WRITE(numout,*) 'hgr_read : read the horizontal coordinates'
         WRITE(numout,*) '~~~~~~~~      jpiglo = ', jpiglo, ' jpjglo = ', jpjglo, ' jpk = ', jpk
      ENDIF
      !
      CALL iom_open( 'coordinates', inum )
      !
      CALL iom_get( inum, jpdom_data, 'glamt', glamt, lrowattr=ln_use_jattr )
      CALL iom_get( inum, jpdom_data, 'glamu', glamu, lrowattr=ln_use_jattr )
      CALL iom_get( inum, jpdom_data, 'glamv', glamv, lrowattr=ln_use_jattr )
      CALL iom_get( inum, jpdom_data, 'glamf', glamf, lrowattr=ln_use_jattr )
      !
      CALL iom_get( inum, jpdom_data, 'gphit', gphit, lrowattr=ln_use_jattr )
      CALL iom_get( inum, jpdom_data, 'gphiu', gphiu, lrowattr=ln_use_jattr )
      CALL iom_get( inum, jpdom_data, 'gphiv', gphiv, lrowattr=ln_use_jattr )
      CALL iom_get( inum, jpdom_data, 'gphif', gphif, lrowattr=ln_use_jattr )
      !
      CALL iom_get( inum, jpdom_data, 'e1t'  , e1t  , lrowattr=ln_use_jattr )
      CALL iom_get( inum, jpdom_data, 'e1u'  , e1u  , lrowattr=ln_use_jattr )
      CALL iom_get( inum, jpdom_data, 'e1v'  , e1v  , lrowattr=ln_use_jattr )
      CALL iom_get( inum, jpdom_data, 'e1f'  , e1f  , lrowattr=ln_use_jattr )
      !
      CALL iom_get( inum, jpdom_data, 'e2t'  , e2t  , lrowattr=ln_use_jattr )
      CALL iom_get( inum, jpdom_data, 'e2u'  , e2u  , lrowattr=ln_use_jattr )
      CALL iom_get( inum, jpdom_data, 'e2v'  , e2v  , lrowattr=ln_use_jattr )
      CALL iom_get( inum, jpdom_data, 'e2f'  , e2f  , lrowattr=ln_use_jattr )
      !
      IF( iom_varid( inum, 'e1e2u', ldstop = .FALSE. ) > 0 ) THEN
         IF(lwp) WRITE(numout,*) 'hgr_read : e1e2u & e1e2v read in coordinates file'
         CALL iom_get( inum, jpdom_data, 'e1e2u'  , e1e2u  , lrowattr=ln_use_jattr )
         CALL iom_get( inum, jpdom_data, 'e1e2v'  , e1e2v  , lrowattr=ln_use_jattr )
         ke1e2u_v = 1
      ELSE
         ke1e2u_v = 0
      ENDIF
      !
      CALL iom_close( inum )
      
    END SUBROUTINE hgr_read
    
   !!======================================================================
END MODULE domhgr
