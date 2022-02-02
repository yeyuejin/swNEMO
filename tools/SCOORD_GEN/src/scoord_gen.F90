PROGRAM SCOORD_GEN
      !!----------------------------------------------------------------------
      !!                  ***  PROGRAM scoord_gen  ***
      !!                     
      !! ** Purpose :   define the s-coordinate system
      !!
      !! ** Method  :   s-coordinate
      !!         The depth of model levels is defined as the product of an
      !!      analytical function by the local bathymetry, while the vertical
      !!      scale factors are defined as the product of the first derivative
      !!      of the analytical function by the bathymetry.
      !!      (this solution save memory as depth and scale factors are not
      !!      3d fields)
      !!          - Read bathymetry (in meters) at t-point and compute the
      !!         bathymetry at u-, v-, and f-points.
      !!            hbatu = mi( hbatt )
      !!            hbatv = mj( hbatt )
      !!            hbatf = mi( mj( hbatt ) )
      !!          - Compute z_gsigt, z_gsigw, z_esigt, z_esigw from an analytical
      !!         function and its derivative given as function.
      !!            z_gsigt(k) = fssig (k    )
      !!            z_gsigw(k) = fssig (k-0.5)
      !!            z_esigt(k) = fsdsig(k    )
      !!            z_esigw(k) = fsdsig(k-0.5)
      !!      Three options for stretching are give, and they can be modified
      !!      following the users requirements. Nevertheless, the output as
      !!      well as the way to compute the model levels and scale factors
      !!      must be respected in order to insure second order accuracy
      !!      schemes.
      !!
      !!      The three methods for stretching available are:
      !! 
      !!           s_sh94 (Song and Haidvogel 1994)
      !!                a sinh/tanh function that allows sigma and stretched sigma
      !!
      !!           s_sf12 (Siddorn and Furner 2012?)
      !!                allows the maintenance of fixed surface and or
      !!                bottom cell resolutions (cf. geopotential coordinates) 
      !!                within an analytically derived stretched S-coordinate framework.
      !! 
      !!          s_tanh  (Madec et al 1996)
      !!                a cosh/tanh function that gives stretched coordinates        
      !!
      !! ** History: 2015: Tim Graham - Code created based on online zdf_sco routine
      !!
      !!
      !!----------------------------------------------------------------------
      !
      USE utils
      IMPLICIT NONE


     !!----------------------------------------------------------------------
      !
      !
      OPEN( UNIT=numnam, FILE='namelist', FORM='FORMATTED', STATUS='OLD' )
      READ( numnam, namzgr_sco )
      CLOSE( numnam )
      jpk = INT(rn_jpk)

      WRITE(*,*)
      WRITE(*,*) 'scoord_gen : s-coordinate or hybrid z-s-coordinate'
      WRITE(*,*) '~~~~~~~~~~~'
      WRITE(*,*) '   Namelist namzgr_sco'
      WRITE(*,*) '     stretching coeffs '
      WRITE(*,*) '        Number of levels                             rn_jpk        = ',rn_jpk
      WRITE(*,*) '        maximum depth of s-bottom surface (>0)       rn_sbot_max   = ',rn_sbot_max
      WRITE(*,*) '        minimum depth of s-bottom surface (>0)       rn_sbot_min   = ',rn_sbot_min
      WRITE(*,*) '        Critical depth                               rn_hc         = ',rn_hc
      WRITE(*,*) '        maximum cut-off r-value allowed              rn_rmax       = ',rn_rmax
      WRITE(*,*) '        Tapering in vicinity of equator              ln_eq_taper   = ',ln_eq_taper
      WRITE(*,*) '        Horizontal Coordinate File                   cn_coord_hgr  = ',cn_coord_hgr
      WRITE(*,*) '     Song and Haidvogel 1994 stretching              ln_s_sh94     = ',ln_s_sh94
      WRITE(*,*) '        Song and Haidvogel 1994 stretching coefficients'
      WRITE(*,*) '        surface control parameter (0<=rn_theta<=20)  rn_theta      = ',rn_theta
      WRITE(*,*) '        bottom  control parameter (0<=rn_thetb<= 1)  rn_thetb      = ',rn_thetb
      WRITE(*,*) '        stretching parameter (song and haidvogel)    rn_bb         = ',rn_bb
      WRITE(*,*) '     Siddorn and Furner 2012 stretching              ln_s_sf12     = ',ln_s_sf12
      WRITE(*,*) '        switching to sigma (T) or Z (F) at H<Hc      ln_sigcrit    = ',ln_sigcrit
      WRITE(*,*) '        Siddorn and Furner 2012 stretching coefficients'
      WRITE(*,*) '        stretchin parameter ( >1 surface; <1 bottom) rn_alpha      = ',rn_alpha
      WRITE(*,*) '        e-fold length scale for transition region    rn_efold      = ',rn_efold
      WRITE(*,*) '        Surface cell depth (Zs) (m)                  rn_zs         = ',rn_zs
      WRITE(*,*) '        Bathymetry multiplier for Zb                 rn_zb_a       = ',rn_zb_a
      WRITE(*,*) '        Offset for Zb                                rn_zb_b       = ',rn_zb_b
      WRITE(*,*) '        Bottom cell (Zb) (m) = H*rn_zb_a + rn_zb_b'

      
      ! Read in bathy, jpi and jpj from bathy.nc
      CALL read_bathy()

      !Allocate all other allocatable variables
      ios = dom_oce_alloc()
      IF( ios .NE. 0) THEN
         WRITE(0,*) 'Unable to allocate all arrays'
         STOP 1
      ENDIF

      hift(:,:) = rn_sbot_min                     ! set the minimum depth for the s-coordinate
      hifu(:,:) = rn_sbot_min
      hifv(:,:) = rn_sbot_min
      hiff(:,:) = rn_sbot_min

      !                                        ! set maximum ocean depth
      bathy(:,:) = MIN( rn_sbot_max, bathy(:,:) )

      DO jj = 1, jpj
         DO ji = 1, jpi
           IF( bathy(ji,jj) > 0. )   bathy(ji,jj) = MAX( rn_sbot_min, bathy(ji,jj) )
         END DO
      END DO
      !                                        ! =============================
      !                                        ! Define the envelop bathymetry   (hbatt)
      !                                        ! =============================
      ! use r-value to create hybrid coordinates
      zenv(:,:) = bathy(:,:)
      !
      ! set first land point adjacent to a wet cell to sbot_min as this needs to be included in smoothing
      DO jj = 1, jpj
         DO ji = 1, jpi
           IF( bathy(ji,jj) == 0. ) THEN
             iip1 = MIN( ji+1, jpi )
             ijp1 = MIN( jj+1, jpj )
             iim1 = MAX( ji-1, 1 )
             ijm1 = MAX( jj-1, 1 )
             IF( (bathy(iip1,jj) + bathy(iim1,jj) + bathy(ji,ijp1) + bathy(ji,ijm1) +              &
        &         bathy(iip1,ijp1) + bathy(iim1,ijm1) + bathy(iip1,ijp1) + bathy(iim1,ijm1)) > 0. ) THEN
               zenv(ji,jj) = rn_sbot_min
             ENDIF
           ENDIF
         END DO
      END DO
      ! 
      ! smooth the bathymetry (if required)
      scosrf(:,:) = 0.             ! ocean surface depth (here zero: no under ice-shelf sea)
      scobot(:,:) = bathy(:,:)        ! ocean bottom  depth
      !
      jl = 0
      zrmax = 1.
      !   
      !     
      ! set scaling factor used in reducing vertical gradients
      zrfact = ( 1. - rn_rmax ) / ( 1. + rn_rmax )
      !
      ! initialise temporary evelope depth arrays
      ztmpi1(:,:) = zenv(:,:)
      ztmpi2(:,:) = zenv(:,:)
      ztmpj1(:,:) = zenv(:,:)
      ztmpj2(:,:) = zenv(:,:)
      !
      ! initialise temporary r-value arrays
      zri(:,:) = 1.
      zrj(:,:) = 1.

      !                                                            ! ================ !
      DO WHILE( jl <= 10000 .AND. ( zrmax - rn_rmax ) > 1.e-8 ) !  Iterative loop  !
         !                                                         ! ================ !
         jl = jl + 1
         zrmax = 0.
         ! we set zrmax from previous r-values (zri and zrj) first
         ! if set after current r-value calculation (as previously)
         ! we could exit DO WHILE prematurely before checking r-value
         ! of current zenv
         DO jj = 1, jpj 
            DO ji = 1, jpi
               zrmax = MAX( zrmax, ABS(zri(ji,jj)), ABS(zrj(ji,jj)) )
            END DO
         END DO
         zri(:,:) = 0.
         zrj(:,:) = 0.
         DO jj = 1, jpj
            DO ji = 1, jpi
               iip1 = MIN(ji+1,jpi) 
               ijp1 = MIN(jj+1,jpj)     
               IF( (zenv(ji,jj) > 0.) .AND. (zenv(iip1,jj) > 0.)) THEN
                  zri(ji,jj) = ( zenv(iip1,jj  ) - zenv(ji,jj) ) / ( zenv(iip1,jj  ) + zenv(ji,jj) )
               END IF
               IF( (zenv(ji,jj) > 0.) .AND. (zenv(ji,ijp1) > 0.)) THEN
                  zrj(ji,jj) = ( zenv(ji  ,ijp1) - zenv(ji,jj) ) / ( zenv(ji  ,ijp1) + zenv(ji,jj) )
               END IF
               IF( zri(ji,jj) >  rn_rmax )   ztmpi1(ji  ,jj  ) = zenv(iip1,jj  ) * zrfact
               IF( zri(ji,jj) < -rn_rmax )   ztmpi2(iip1,jj  ) = zenv(ji  ,jj  ) * zrfact
               IF( zrj(ji,jj) >  rn_rmax )   ztmpj1(ji  ,jj  ) = zenv(ji  ,ijp1) * zrfact
               IF( zrj(ji,jj) < -rn_rmax )   ztmpj2(ji  ,ijp1) = zenv(ji  ,jj  ) * zrfact
            END DO
         END DO
         !
         WRITE(*,*) 'zgr_sco :   iter= ',jl, ' rmax= ', zrmax
         !
         DO jj = 1, jpj
            DO ji = 1, jpi
               zenv(ji,jj) = MAX(zenv(ji,jj), ztmpi1(ji,jj), ztmpi2(ji,jj), ztmpj1(ji,jj), ztmpj2(ji,jj) )
            END DO
         END DO
         !                                                  ! ================ !
      END DO                                                !     End loop     !
      !                                                     ! ================ !
      DO jj = 1, jpj
         DO ji = 1, jpi
            zenv(ji,jj) = MAX( zenv(ji,jj), rn_sbot_min ) ! set all points to avoid undefined scale value warnings
         END DO
      END DO
      !
      ! Envelope bathymetry saved in hbatt
      hbatt(:,:) = zenv(:,:) 
      IF( ln_eq_taper) THEN
        CALL READ_GPHIT()
        IF( MINVAL( gphit(:,:) ) * MAXVAL( gphit(:,:) ) <= 0. ) THEN
          WRITE(*,*) 's-coordinates are tapered in vicinity of the Equator' 
          DO jj = 1, jpj
             DO ji = 1, jpi
                ztaper = EXP( -(gphit(ji,jj)/8.)**2. )
                hbatt(ji,jj) = rn_sbot_max * ztaper + hbatt(ji,jj) * ( 1. - ztaper )
             END DO
          END DO
        ENDIF
      ENDIF
      !
      !                                        ! ==============================
      !                                        !   hbatu, hbatv, hbatf fields
      !                                        ! ==============================
      WRITE(*,*)
      WRITE(*,*) ' zgr_sco: minimum depth of the envelop topography set to : ', rn_sbot_min
      hbatu(:,:) = rn_sbot_min
      hbatv(:,:) = rn_sbot_min
      hbatf(:,:) = rn_sbot_min
      DO jj = 1, jpj-1
        DO ji = 1, jpi-1   ! NO vector opt.
           hbatu(ji,jj) = 0.50 * ( hbatt(ji  ,jj) + hbatt(ji+1,jj  ) )
           hbatv(ji,jj) = 0.50 * ( hbatt(ji  ,jj) + hbatt(ji  ,jj+1) )
           hbatf(ji,jj) = 0.25 * ( hbatt(ji  ,jj) + hbatt(ji  ,jj+1)   &
              &                     + hbatt(ji+1,jj) + hbatt(ji+1,jj+1) )
        END DO
      END DO
      ! 
      zhbat(:,:) = hbatu(:,:)   
      DO jj = 1, jpj
         DO ji = 1, jpi
            IF( hbatu(ji,jj) == 0. ) THEN
               IF( zhbat(ji,jj) == 0. )   hbatu(ji,jj) = rn_sbot_min
               IF( zhbat(ji,jj) /= 0. )   hbatu(ji,jj) = zhbat(ji,jj)
            ENDIF
         END DO
      END DO
      zhbat(:,:) = hbatv(:,:)  
      DO jj = 1, jpj
         DO ji = 1, jpi
            IF( hbatv(ji,jj) == 0. ) THEN
               IF( zhbat(ji,jj) == 0. )   hbatv(ji,jj) = rn_sbot_min
               IF( zhbat(ji,jj) /= 0. )   hbatv(ji,jj) = zhbat(ji,jj)
            ENDIF
         END DO
      END DO
      zhbat(:,:) = hbatf(:,:)   
      DO jj = 1, jpj
         DO ji = 1, jpi
            IF( hbatf(ji,jj) == 0. ) THEN
               IF( zhbat(ji,jj) == 0. )   hbatf(ji,jj) = rn_sbot_min
               IF( zhbat(ji,jj) /= 0. )   hbatf(ji,jj) = zhbat(ji,jj)
            ENDIF
         END DO
      END DO

!!bug:  key_helsinki a verifer
      hift(:,:) = MIN( hift(:,:), hbatt(:,:) )
      hifu(:,:) = MIN( hifu(:,:), hbatu(:,:) )
      hifv(:,:) = MIN( hifv(:,:), hbatv(:,:) )
      hiff(:,:) = MIN( hiff(:,:), hbatf(:,:) )

         WRITE(*,*) ' MAX val hif   t ', MAXVAL( hift (:,:) ), ' f ', MAXVAL( hiff (:,:) ),  &
            &                        ' u ',   MAXVAL( hifu (:,:) ), ' v ', MAXVAL( hifv (:,:) )
         WRITE(*,*) ' MIN val hif   t ', MINVAL( hift (:,:) ), ' f ', MINVAL( hiff (:,:) ),  &
            &                        ' u ',   MINVAL( hifu (:,:) ), ' v ', MINVAL( hifv (:,:) )
         WRITE(*,*) ' MAX val hbat  t ', MAXVAL( hbatt(:,:) ), ' f ', MAXVAL( hbatf(:,:) ),  &
            &                        ' u ',   MAXVAL( hbatu(:,:) ), ' v ', MAXVAL( hbatv(:,:) )
         WRITE(*,*) ' MIN val hbat  t ', MINVAL( hbatt(:,:) ), ' f ', MINVAL( hbatf(:,:) ),  &
            &                        ' u ',   MINVAL( hbatu(:,:) ), ' v ', MINVAL( hbatv(:,:) )
      
      ! Create the output file
      CALL make_coord_file()

      !                                            ! =======================
      !                                            !   s-coordinate fields     (gdep., e3.)
      !                                            ! =======================
      !
      ! non-dimensional "sigma" for model level depth at w- and t-levels


!========================================================================
! Song and Haidvogel  1994 (ln_s_sh94=T)
! Siddorn and Furner 2012 (ln_sf12=T)
! or  tanh function       (both false)                    
! To reduce memory loop over jpk and write each level to file
!========================================================================
      IF      ( ln_s_sh94 ) THEN 
                           CALL s_sh94()
      ELSE IF ( ln_s_sf12 ) THEN
                           CALL s_sf12()
      ELSE                 
                           CALL s_tanh()
      ENDIF

!Write all 2D variables to output file
      CALL write_netcdf_2d_vars()
      CALL check_nf90( nf90_close(ncout) )

 
      !
END PROGRAM SCOORD_GEN

!!======================================================================
   SUBROUTINE s_sh94()

      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE s_sh94  ***
      !!                     
      !! ** Purpose :   stretch the s-coordinate system
      !!
      !! ** Method  :   s-coordinate stretch using the Song and Haidvogel 1994
      !!                mixed S/sigma coordinate
      !!
      !! Reference : Song and Haidvogel 1994. 
      !!----------------------------------------------------------------------
      !
      USE utils
      REAL(wp) ::   zcoeft, zcoefw   ! temporary scalars
      !
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: z_gsigw3, z_gsigt3, z_gsi3w3
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: z_gsigw3p1, z_gsigt3m1, z_gsi3w3m1
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: z_esigt3, z_esigw3, z_esigtu3 
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: z_esigtv3, z_esigtf3, z_esigwu3, z_esigwv3           

      ALLOCATE( z_gsigw3(jpi,jpj), z_gsigt3(jpi,jpj), z_gsi3w3(jpi,jpj) )
      ALLOCATE( z_esigt3(jpi,jpj), z_esigw3(jpi,jpj), z_esigtu3(jpi,jpj)) 
      ALLOCATE( z_esigtv3(jpi,jpj), z_esigtf3(jpi,jpj), z_esigwu3(jpi,jpj)) 
      ALLOCATE( z_esigwv3(jpi,jpj), z_gsigw3p1(jpi,jpj), z_gsigt3m1(jpi,jpj) )
      ALLOCATE( z_gsi3w3m1(jpi,jpj) )
 
      z_gsigw3  = 0.   ;   z_gsigt3  = 0.   ;   z_gsi3w3  = 0.
      z_esigt3  = 0.   ;   z_esigw3  = 0. 
      z_esigtu3 = 0.   ;   z_esigtv3 = 0.   ;   z_esigtf3 = 0.
      z_esigwu3 = 0.   ;   z_esigwv3 = 0.

      DO jk = 1,jpk
         DO ji = 1, jpi
            DO jj = 1, jpj

               IF( hbatt(ji,jj) > rn_hc ) THEN    !deep water, stretched sigma
                     z_gsigw3(ji,jj) = -fssig1( REAL(jk,wp)-0.5, rn_bb )
                     z_gsigw3p1(ji,jj) = -fssig1( REAL(jk+1,wp)-0.5, rn_bb )
                     z_gsigt3(ji,jj) = -fssig1( REAL(jk,wp)    , rn_bb )
               ELSE ! shallow water, uniform sigma
                     z_gsigw3(ji,jj) =   REAL(jk-1,wp)            / REAL(jpk-1,wp)
                     z_gsigw3p1(ji,jj) =   REAL(jk,wp)            / REAL(jpk-1,wp)
                     z_gsigt3(ji,jj) = ( REAL(jk-1,wp) + 0.5 ) / REAL(jpk-1,wp)
               ENDIF
               !
             !gsi3w3m1 & gsigt3m1 only used if jk /= 1 and is set at the end of the loop over jk
               IF( jk .EQ. 1) THEN
                  z_esigw3(ji,jj ) = 2. * ( z_gsigt3(ji,jj ) - z_gsigw3(ji,jj ) )
                  z_gsi3w3(ji,jj) = 0.5 * z_esigw3(ji,jj)
               ELSE
                  z_esigw3(ji,jj) = z_gsigt3(ji,jj) - z_gsigt3m1(ji,jj)
                  z_gsi3w3(ji,jj) = z_gsi3w3m1(ji,jj) + z_esigw3(ji,jj)
               ENDIF
               IF( jk .EQ. jpk) THEN
                  z_esigt3(ji,jj) = 2. * ( z_gsigt3(ji,jj) - z_gsigw3(ji,jj) )
               ELSE
                  z_esigt3(ji,jj ) = z_gsigw3p1(ji,jj) - z_gsigw3(ji,jj)
               ENDIF
               !

               zcoeft = ( REAL(jk,wp) - 0.5 ) / REAL(jpk-1,wp)
               zcoefw = ( REAL(jk,wp) - 1.0 ) / REAL(jpk-1,wp)
               gdept_0(ji,jj) = ( scosrf(ji,jj) + (hbatt(ji,jj)-rn_hc)*z_gsigt3(ji,jj)+rn_hc*zcoeft )
               gdepw_0(ji,jj) = ( scosrf(ji,jj) + (hbatt(ji,jj)-rn_hc)*z_gsigw3(ji,jj)+rn_hc*zcoefw )
               gdep3w_0(ji,jj) = ( scosrf(ji,jj) + (hbatt(ji,jj)-rn_hc)*z_gsi3w3(ji,jj)+rn_hc*zcoeft )
              !
            END DO   ! for all jj's
         END DO    ! for all ji's

         DO ji = 1, jpi-1
            DO jj = 1, jpj-1
               z_esigtu3(ji,jj) = ( hbatt(ji,jj)*z_esigt3(ji,jj)+hbatt(ji+1,jj)*z_esigt3(ji+1,jj) )   &
                  &              / ( hbatt(ji,jj)+hbatt(ji+1,jj) )
               z_esigtv3(ji,jj) = ( hbatt(ji,jj)*z_esigt3(ji,jj)+hbatt(ji,jj+1)*z_esigt3(ji,jj+1) )   &
                  &              / ( hbatt(ji,jj)+hbatt(ji,jj+1) )
               z_esigtf3(ji,jj) = ( hbatt(ji,jj)*z_esigt3(ji,jj)+hbatt(ji+1,jj)*z_esigt3(ji+1,jj)     &
                  &                + hbatt(ji,jj+1)*z_esigt3(ji,jj+1)+hbatt(ji+1,jj+1)*z_esigt3(ji+1,jj+1) )   &
                  &              / ( hbatt(ji,jj)+hbatt(ji+1,jj)+hbatt(ji,jj+1)+hbatt(ji+1,jj+1) )
               z_esigwu3(ji,jj) = ( hbatt(ji,jj)*z_esigw3(ji,jj)+hbatt(ji+1,jj)*z_esigw3(ji+1,jj) )   &
                  &              / ( hbatt(ji,jj)+hbatt(ji+1,jj) )
               z_esigwv3(ji,jj) = ( hbatt(ji,jj)*z_esigw3(ji,jj)+hbatt(ji,jj+1)*z_esigw3(ji,jj+1) )   &
                  &              / ( hbatt(ji,jj)+hbatt(ji,jj+1) )
               !
               e3t_0(ji,jj) = ( (hbatt(ji,jj)-rn_hc)*z_esigt3 (ji,jj) + rn_hc/REAL(jpk-1,wp) )
               e3u_0(ji,jj) = ( (hbatu(ji,jj)-rn_hc)*z_esigtu3(ji,jj) + rn_hc/REAL(jpk-1,wp) )
               e3v_0(ji,jj) = ( (hbatv(ji,jj)-rn_hc)*z_esigtv3(ji,jj) + rn_hc/REAL(jpk-1,wp) )
               e3f_0(ji,jj) = ( (hbatf(ji,jj)-rn_hc)*z_esigtf3(ji,jj) + rn_hc/REAL(jpk-1,wp) )
               !
               e3w_0 (ji,jj) = ( (hbatt(ji,jj)-rn_hc)*z_esigw3 (ji,jj) + rn_hc/REAL(jpk-1,wp) )
               e3uw_0(ji,jj) = ( (hbatu(ji,jj)-rn_hc)*z_esigwu3(ji,jj) + rn_hc/REAL(jpk-1,wp) )
               e3vw_0(ji,jj) = ( (hbatv(ji,jj)-rn_hc)*z_esigwv3(ji,jj) + rn_hc/REAL(jpk-1,wp) )
            END DO
         END DO

         z_gsigt3m1 = z_gsigt3
         z_gsi3w3m1 = z_gsi3w3
 
         where (e3t_0   (:,:).eq.0.0)  e3t_0(:,:) = 1.0
         where (e3u_0   (:,:).eq.0.0)  e3u_0(:,:) = 1.0
         where (e3v_0   (:,:).eq.0.0)  e3v_0(:,:) = 1.0
         where (e3f_0   (:,:).eq.0.0)  e3f_0(:,:) = 1.0
         where (e3w_0   (:,:).eq.0.0)  e3w_0(:,:) = 1.0
         where (e3uw_0  (:,:).eq.0.0)  e3uw_0(:,:) = 1.0
         where (e3vw_0  (:,:).eq.0.0)  e3vw_0(:,:) = 1.0
        
         CALL write_netcdf_3d_vars(jk)
         DO jj = 1, jpj
            DO ji = 1, jpi
                  IF( scobot(ji,jj) >= gdept_0(ji,jj) )   mbathy(ji,jj) = MAX( 2, jk )
                  IF( scobot(ji,jj) == 0.             )   mbathy(ji,jj) = 0
            END DO
         END DO
      END DO !End of loop over jk

      DEALLOCATE( z_gsigw3, z_gsigt3, z_gsi3w3, z_gsigw3p1, z_gsigt3m1, z_gsi3w3m1)
      DEALLOCATE( z_esigt3, z_esigw3, z_esigtu3, z_esigtv3, z_esigtf3, z_esigwu3, z_esigwv3 )

   END SUBROUTINE s_sh94

   SUBROUTINE s_sf12

      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE s_sf12 *** 
      !!                     
      !! ** Purpose :   stretch the s-coordinate system
      !!
      !! ** Method  :   s-coordinate stretch using the Siddorn and Furner 2012?
      !!                mixed S/sigma/Z coordinate
      !!
      !!                This method allows the maintenance of fixed surface and or
      !!                bottom cell resolutions (cf. geopotential coordinates) 
      !!                within an analytically derived stretched S-coordinate framework.
      !!
      !!
      !! Reference : Siddorn and Furner 2012 (submitted Ocean modelling).
      !!----------------------------------------------------------------------
      !
      USE utils
      REAL(wp) ::   zsmth               ! smoothing around critical depth
      REAL(wp) ::   zzs, zzb           ! Surface and bottom cell thickness in sigma space
      !
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: z_gsigw3, z_gsigt3, z_gsi3w3
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: z_gsigw3p1, z_gsigt3m1, z_gsi3w3m1
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: z_esigt3, z_esigw3, z_esigtu3 
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: z_esigtv3, z_esigtf3, z_esigwu3, z_esigwv3           

      ALLOCATE( z_gsigw3(jpi,jpj), z_gsigt3(jpi,jpj), z_gsi3w3(jpi,jpj) )
      ALLOCATE( z_esigt3(jpi,jpj), z_esigw3(jpi,jpj), z_esigtu3(jpi,jpj)) 
      ALLOCATE( z_esigtv3(jpi,jpj), z_esigtf3(jpi,jpj), z_esigwu3(jpi,jpj)) 
      ALLOCATE( z_esigwv3(jpi,jpj), z_gsigw3p1(jpi,jpj), z_gsigt3m1(jpi,jpj) )
      ALLOCATE( z_gsi3w3m1(jpi,jpj) )

      z_gsigw3  = 0.   ;   z_gsigt3  = 0.   ;   z_gsi3w3  = 0.
      z_esigt3  = 0.   ;   z_esigw3  = 0. 
      z_esigtu3 = 0.   ;   z_esigtv3 = 0.   ;   z_esigtf3 = 0.
      z_esigwu3 = 0.   ;   z_esigwv3 = 0.
      

      DO jk = 1,jpk
         DO ji = 1, jpi
            DO jj = 1, jpj
             IF (hbatt(ji,jj)>rn_hc) THEN !deep water, stretched sigma
              
                zzb = hbatt(ji,jj)*rn_zb_a + rn_zb_b   ! this forces a linear bottom cell depth relationship with H,.
                                                     ! could be changed by users but care must be taken to do so carefully
                zzb = 1.0-(zzb/hbatt(ji,jj))
            
                zzs = rn_zs / hbatt(ji,jj) 
              
                IF (rn_efold /= 0.0) THEN
                   zsmth   = tanh( (hbatt(ji,jj)- rn_hc ) / rn_efold )
                ELSE
                   zsmth = 1.0 
                ENDIF
                  
                z_gsigw3(ji,jj) =  REAL(jk-1,wp)        /REAL(jpk-1,wp)
                z_gsigw3p1(ji,jj) =  REAL(jk,wp)        /REAL(jpk-1,wp)
                z_gsigt3(ji,jj) = (REAL(jk-1,wp)+0.5)/REAL(jpk-1,wp)
                z_gsigw3(ji,jj) = fgamma( z_gsigw3(ji,jj), zzb, zzs, zsmth  )
                z_gsigw3p1(ji,jj) = fgamma( z_gsigw3p1(ji,jj), zzb, zzs, zsmth  )
                z_gsigt3(ji,jj) = fgamma( z_gsigt3(ji,jj), zzb, zzs, zsmth  )
 
             ELSE IF(ln_sigcrit) THEN ! shallow water, uniform sigma

                z_gsigw3(ji,jj) =  REAL(jk-1,wp)     /REAL(jpk-1,wp)
                z_gsigw3p1(ji,jj) =  REAL(jk,wp)     /REAL(jpk-1,wp)
                z_gsigt3(ji,jj) = (REAL(jk-1,wp)+0.5)/REAL(jpk-1,wp)

             ELSE  ! shallow water, z coordinates

               z_gsigw3(ji,jj) =  REAL(jk-1,wp)        /REAL(jpk-1,wp)*(rn_hc/hbatt(ji,jj))
               z_gsigw3p1(ji,jj) =  REAL(jk,wp)        /REAL(jpk-1,wp)*(rn_hc/hbatt(ji,jj))
               z_gsigt3(ji,jj) = (REAL(jk-1,wp)+0.5)/REAL(jpk-1,wp)*(rn_hc/hbatt(ji,jj))

             ENDIF

             !gsi3w3m1 & z_gsigt3m1 only used if jk /= 1 and is set at the end of the loop over jk
             IF( jk .EQ. 1) THEN
                z_esigw3(ji,jj  ) = 2.0 * (z_gsigt3(ji,jj ) - z_gsigw3(ji,jj ))
                z_gsi3w3(ji,jj) = 0.5 * z_esigw3(ji,jj) 
             ELSE
                z_esigw3(ji,jj) = z_gsigt3(ji,jj) - z_gsigt3m1(ji,jj)
                z_gsi3w3(ji,jj) = z_gsi3w3m1(ji,jj) + z_esigw3(ji,jj)
             ENDIF
             IF( jk .EQ. jpk) THEN
                z_esigt3(ji,jj) = 2.0 * (z_gsigt3(ji,jj) - z_gsigw3(ji,jj))
             ELSE
                z_esigt3(ji,jj) = z_gsigw3p1(ji,jj) - z_gsigw3(ji,jj)
             ENDIF

             gdept_0(ji,jj) = (scosrf(ji,jj)+hbatt(ji,jj))*z_gsigt3(ji,jj)
             gdepw_0(ji,jj) = (scosrf(ji,jj)+hbatt(ji,jj))*z_gsigw3(ji,jj)
             gdep3w_0(ji,jj) = (scosrf(ji,jj)+hbatt(ji,jj))*z_gsi3w3(ji,jj)

            ENDDO   ! for all jj's
         ENDDO    ! for all ji's

         DO ji=1,jpi-1
            DO jj=1,jpj-1

               z_esigtu3(ji,jj) = ( hbatt(ji,jj)*z_esigt3(ji,jj)+hbatt(ji+1,jj)*z_esigt3(ji+1,jj) ) / &
                                     ( hbatt(ji,jj)+hbatt(ji+1,jj) )
               z_esigtv3(ji,jj) = ( hbatt(ji,jj)*z_esigt3(ji,jj)+hbatt(ji,jj+1)*z_esigt3(ji,jj+1) ) / &
                                     ( hbatt(ji,jj)+hbatt(ji,jj+1) )
               z_esigtf3(ji,jj) = ( hbatt(ji,jj)*z_esigt3(ji,jj)+hbatt(ji+1,jj)*z_esigt3(ji+1,jj) +  &
                                       hbatt(ji,jj+1)*z_esigt3(ji,jj+1)+hbatt(ji+1,jj+1)*z_esigt3(ji+1,jj+1) ) / &
                                     ( hbatt(ji,jj)+hbatt(ji+1,jj)+hbatt(ji,jj+1)+hbatt(ji+1,jj+1) )
               z_esigwu3(ji,jj) = ( hbatt(ji,jj)*z_esigw3(ji,jj)+hbatt(ji+1,jj)*z_esigw3(ji+1,jj) ) / &
                                     ( hbatt(ji,jj)+hbatt(ji+1,jj) )
               z_esigwv3(ji,jj) = ( hbatt(ji,jj)*z_esigw3(ji,jj)+hbatt(ji,jj+1)*z_esigw3(ji,jj+1) ) / &
                                     ( hbatt(ji,jj)+hbatt(ji,jj+1) )

               e3t_0(ji,jj)=(scosrf(ji,jj)+hbatt(ji,jj))*z_esigt3(ji,jj)
               e3u_0(ji,jj)=(scosrf(ji,jj)+hbatu(ji,jj))*z_esigtu3(ji,jj)
               e3v_0(ji,jj)=(scosrf(ji,jj)+hbatv(ji,jj))*z_esigtv3(ji,jj)
               e3f_0(ji,jj)=(scosrf(ji,jj)+hbatf(ji,jj))*z_esigtf3(ji,jj)
               !
               e3w_0(ji,jj)=hbatt(ji,jj)*z_esigw3(ji,jj)
               e3uw_0(ji,jj)=hbatu(ji,jj)*z_esigwu3(ji,jj)
               e3vw_0(ji,jj)=hbatv(ji,jj)*z_esigwv3(ji,jj)
            ENDDO
         ENDDO
         ! Keep some arrays for next level
         z_gsigt3m1 = z_gsigt3
         z_gsi3w3m1 = z_gsi3w3
 
         where (e3t_0   (:,:).eq.0.0)  e3t_0(:,:) = 1.0
         where (e3u_0   (:,:).eq.0.0)  e3u_0(:,:) = 1.0
         where (e3v_0   (:,:).eq.0.0)  e3v_0(:,:) = 1.0
         where (e3f_0   (:,:).eq.0.0)  e3f_0(:,:) = 1.0
         where (e3w_0   (:,:).eq.0.0)  e3w_0(:,:) = 1.0
         where (e3uw_0  (:,:).eq.0.0)  e3uw_0(:,:) = 1.0
         where (e3vw_0  (:,:).eq.0.0)  e3vw_0(:,:) = 1.0
         
         CALL write_netcdf_3d_vars(jk)

         DO jj = 1, jpj
            DO ji = 1, jpi
                  IF( scobot(ji,jj) >= gdept_0(ji,jj) )   mbathy(ji,jj) = MAX( 2, jk )
                  IF( scobot(ji,jj) == 0.             )   mbathy(ji,jj) = 0
            END DO
         END DO

      ENDDO ! End of loop over jk

      DEALLOCATE( z_gsigw3, z_gsigt3, z_gsi3w3, z_gsigw3p1, z_gsigt3m1, z_gsi3w3m1)
      DEALLOCATE( z_esigt3, z_esigw3, z_esigtu3, z_esigtv3, z_esigtf3, z_esigwu3, z_esigwv3 )

   END SUBROUTINE s_sf12

   SUBROUTINE s_tanh()

      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE s_tanh*** 
      !!                     
      !! ** Purpose :   stretch the s-coordinate system
      !!
      !! ** Method  :   s-coordinate stretch 
      !!
      !! Reference : Madec, Lott, Delecluse and Crepon, 1996. JPO, 26, 1393-1408.
      !!----------------------------------------------------------------------

      USE utils
      REAL(wp) ::   zcoeft, zcoefw   ! temporary scalars

      REAL(wp), ALLOCATABLE, DIMENSION(:) :: z_gsigw, z_gsigt, z_gsi3w
      REAL(wp), ALLOCATABLE, DIMENSION(:) :: z_esigt, z_esigw

      ALLOCATE( z_gsigw(jpk), z_gsigt(jpk), z_gsi3w(jpk) )
      ALLOCATE( z_esigt(jpk), z_esigw(jpk) )

      z_gsigw  = 0.   ;   z_gsigt  = 0.   ;   z_gsi3w  = 0.
      z_esigt  = 0.   ;   z_esigw  = 0. 

      DO jk = 1, jpk
        z_gsigw(jk) = -fssig( REAL(jk,wp)-0.5 )
        z_gsigt(jk) = -fssig( REAL(jk,wp)        )
      END DO
      WRITE(*,*) 'z_gsigw 1 jpk    ', z_gsigw(1), z_gsigw(jpk)
      !
      ! Coefficients for vertical scale factors at w-, t- levels
!!gm bug :  define it from analytical function, not like juste bellow....
!!gm        or betteroffer the 2 possibilities....
      DO jk = 1, jpk-1
         z_esigt(jk  ) = z_gsigw(jk+1) - z_gsigw(jk)
         z_esigw(jk+1) = z_gsigt(jk+1) - z_gsigt(jk)
      END DO
      z_esigw( 1 ) = 2. * ( z_gsigt(1  ) - z_gsigw(1  ) ) 
      z_esigt(jpk) = 2. * ( z_gsigt(jpk) - z_gsigw(jpk) )
      !
      ! Coefficients for vertical depth as the sum of e3w scale factors
      z_gsi3w(1) = 0.5 * z_esigw(1)
      DO jk = 2, jpk
         z_gsi3w(jk) = z_gsi3w(jk-1) + z_esigw(jk)
      END DO
!!gm: e3uw, e3vw can be suppressed  (modif in dynzdf, dynzdf_iso, zdfbfr) (save 2 3D arrays)
      DO jk = 1, jpk
         DO jj = 1, jpj
            DO ji = 1, jpi
              zcoeft = ( REAL(jk,wp) - 0.5 ) / REAL(jpk-1,wp)
              zcoefw = ( REAL(jk,wp) - 1.0 ) / REAL(jpk-1,wp)
              gdept_0(ji,jj) = ( scosrf(ji,jj) + (hbatt(ji,jj)-hift(ji,jj))*z_gsigt(jk) + hift(ji,jj)*zcoeft )
              gdepw_0(:,:) = ( scosrf(ji,jj) + (hbatt(ji,jj)-hift(ji,jj))*z_gsigw(jk) + hift(ji,jj)*zcoefw )
              gdep3w_0(:,:) = ( scosrf(ji,jj) + (hbatt(ji,jj)-hift(ji,jj))*z_gsi3w(jk) + hift(ji,jj)*zcoeft )
              e3t_0(ji,jj) = ( (hbatt(ji,jj)-hift(ji,jj))*z_esigt(jk) + hift(ji,jj)/REAL(jpk-1,wp) )
              e3u_0(ji,jj) = ( (hbatu(ji,jj)-hifu(ji,jj))*z_esigt(jk) + hifu(ji,jj)/REAL(jpk-1,wp) )
              e3v_0(ji,jj) = ( (hbatv(ji,jj)-hifv(ji,jj))*z_esigt(jk) + hifv(ji,jj)/REAL(jpk-1,wp) )
              e3f_0(ji,jj) = ( (hbatf(ji,jj)-hiff(ji,jj))*z_esigt(jk) + hiff(ji,jj)/REAL(jpk-1,wp) )
              !
              e3w_0(ji,jj) = ( (hbatt(ji,jj)-hift(ji,jj))*z_esigw(jk) + hift(ji,jj)/REAL(jpk-1,wp) )
              e3uw_0(ji,jj) = ( (hbatu(ji,jj)-hifu(ji,jj))*z_esigw(jk) + hifu(ji,jj)/REAL(jpk-1,wp) )
              e3vw_0(ji,jj) = ( (hbatv(ji,jj)-hifv(ji,jj))*z_esigw(jk) + hifv(ji,jj)/REAL(jpk-1,wp) )
              IF( scobot(ji,jj) >= gdept_0(ji,jj) )   mbathy(ji,jj) = MAX( 2, jk )
              IF( scobot(ji,jj) == 0.             )   mbathy(ji,jj) = 0
            END DO
         END DO
         where (e3t_0   (:,:).eq.0.0)  e3t_0(:,:) = 1.0
         where (e3u_0   (:,:).eq.0.0)  e3u_0(:,:) = 1.0
         where (e3v_0   (:,:).eq.0.0)  e3v_0(:,:) = 1.0
         where (e3f_0   (:,:).eq.0.0)  e3f_0(:,:) = 1.0
         where (e3w_0   (:,:).eq.0.0)  e3w_0(:,:) = 1.0
         where (e3uw_0  (:,:).eq.0.0)  e3uw_0(:,:) = 1.0
         where (e3vw_0  (:,:).eq.0.0)  e3vw_0(:,:) = 1.0
 
         CALL write_netcdf_3d_vars(jk)
      ENDDO ! End of loop over jk


      DEALLOCATE( z_gsigw, z_gsigt, z_gsi3w )
      DEALLOCATE( z_esigt, z_esigw )

   END SUBROUTINE s_tanh
