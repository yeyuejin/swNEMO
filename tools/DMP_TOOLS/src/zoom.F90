MODULE zoom
   
   USE utils

   CONTAINS

   SUBROUTINE dtacof_zoom( presto, mask)
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE dtacof_zoom  ***
      !!
      !! ** Purpose :   Compute the damping coefficient for zoom domain
      !!
      !! ** Method  : - set along closed boundary due to zoom a damping over
      !!                6 points with a max time scale of 5 days.
      !!              - ORCA arctic/antarctic zoom: set the damping along
      !!                south/north boundary over a latitude strip.
      !!
      !! ** Action  : - resto, the damping coeff. for T and S
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj), INTENT(inout)  ::   presto   ! restoring coeff. (s-1)
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in)  ::   mask   ! restoring coeff. (s-1)
      !
      INTEGER  ::   ji, jj, jn   ! dummy loop indices
      REAL(wp) ::   zlat, zlat0, zlat1, zlat2, z1_5d   ! local scalar
      REAL(wp), DIMENSION(6)  ::   zfact               ! 1Dworkspace

      !Namelist variables
      LOGICAL :: lzoom_w, lzoom_e, lzoom_n, lzoom_s 
      NAMELIST/nam_zoom_dmp/lzoom_n,lzoom_e,lzoom_w,lzoom_s
      !!----------------------------------------------------------------------

      ! Read namelist
      OPEN( UNIT=numnam, FILE='namelist', FORM='FORMATTED', STATUS='OLD' )
      READ( numnam, nam_dmp_create )
      CLOSE( numnam )

      zfact(1) =  1._wp
      zfact(2) =  1._wp
      zfact(3) = 11._wp / 12._wp
      zfact(4) =  8._wp / 12._wp
      zfact(5) =  4._wp / 12._wp
      zfact(6) =  1._wp / 12._wp
      zfact(:) = zfact(:) / ( 5._wp * rday )    ! 5 days max restoring time scale

      presto(:,:) = 0._wp

      ! damping along the forced closed boundary over 6 grid-points
      DO jn = 1, 6
         IF( lzoom_w )   presto( jn, : )                    = zfact(jn)   ! west  closed
         IF( lzoom_s )   presto( : , jn )                    = zfact(jn)   ! south closed 
         IF( lzoom_e )   presto( jpi+1-jn , : ) = zfact(jn)   ! east  closed 
         IF( lzoom_n )   presto( : , jpj+1-jn ) = zfact(jn)   ! north closed
      END DO

      !                                           ! ====================================================
      IF( cp_cfz == "arctic" .OR. cp_cfz == "antarctic" ) THEN   !  ORCA configuration : arctic or antarctic zoom
         !                                        ! ====================================================
         WRITE(numout,*)
         IF(cp_cfz == "arctic" ) WRITE(numout,*) '              dtacof_zoom : ORCA    Arctic zoom'
         IF(cp_cfz == "antarctic" ) WRITE(numout,*) '           dtacof_zoom : ORCA Antarctic zoom'
         WRITE(numout,*)
         !
         !                          ! Initialization : 
         presto(:,:) = 0._wp
         zlat0 = 10._wp                     ! zlat0 : latitude strip where resto decreases
         zlat1 = 30._wp                     ! zlat1 : resto = 1 before zlat1
         zlat2 = zlat1 + zlat0              ! zlat2 : resto decreases from 1 to 0 between zlat1 and zlat2
         z1_5d = 1._wp / ( 5._wp * rday )   ! z1_5d : 1 / 5days

         DO jj = 1, jpj
            DO ji = 1, jpi
               zlat = ABS( gphit(ji,jj) )
               IF( zlat1 <= zlat .AND. zlat <= zlat2 ) THEN
                  presto(ji,jj) = 0.5_wp * z1_5d * (  1._wp - COS( rpi*(zlat2-zlat)/zlat0 )  ) 
               ELSEIF( zlat < zlat1 ) THEN
                  presto(ji,jj) = z1_5d
               ENDIF
            END DO
         END DO
         !
      ENDIF
      !                             ! Mask resto array
      presto(:,:) = presto(:,:) * mask(:,:)

   END SUBROUTINE dtacof_zoom

END MODULE zoom
