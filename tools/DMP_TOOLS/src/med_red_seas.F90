MODULE med_red_seas

   USE utils
   
   IMPLICIT NONE
   PUBLIC

   CONTAINS  

   SUBROUTINE med_red_dmp(presto, jk, ln_31_lev)
      !!------------------------------------
      !!    **ROUTINE: med_red_dmp
      !! 
      !! **Purpose: Apply specific modifications to damping coefficients on ORCA
      !!            grids in Med and Red Seas
      !!
      !!-----------------------------------
      INTEGER :: ij0,ij1,ii0,ii1,ji,jj     
      INTEGER, INTENT(in) :: jk
      REAL(wp), DIMENSION(:,:), ALLOCATABLE :: zmrs
      REAL(wp) :: zhfac, zsdmp, zbdmp
      REAL(wp), DIMENSION(jpi,jpj), INTENT(inout) :: presto
      LOGICAL, INTENT(in), OPTIONAL :: ln_31_lev
      LOGICAL :: l_31_lev

      WRITE(numout,*) 'ORCA Med and Red Seas Damping'
      
      IF ( PRESENT(ln_31_lev)) THEN
         l_31_lev = ln_31_lev
      ELSE
         l_31_lev = .false.
      ENDIF
      
      ALLOCATE( zmrs(jpi, jpj) )
         !
         zmrs(:,:) = 0._wp
         !
         SELECT CASE ( jp_cfg )
         !                                           ! =======================
         CASE ( 4 )                                  !  ORCA_R4 configuration 
            !                                        ! =======================
            ij0 =  50   ;   ij1 =  56                    ! Mediterranean Sea

            ii0 =  81   ;   ii1 =  91   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 1.
            ij0 =  50   ;   ij1 =  55
            ii0 =  75   ;   ii1 =  80   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 1.
            ij0 =  52   ;   ij1 =  53
            ii0 =  70   ;   ii1 =  74   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 1.
           !
           !                                        ! =======================
         CASE ( 2 )                                  !  ORCA_R2 configuration 
            !                                        ! =======================
            ij0 =  96   ;   ij1 = 110                    ! Mediterranean Sea
            ii0 = 157   ;   ii1 = 181   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 1._wp
            ij0 = 100   ;   ij1 = 110
            ii0 = 144   ;   ii1 = 156   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 1._wp
            ij0 = 100   ;   ij1 = 103
            ii0 = 139   ;   ii1 = 143   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 1._wp
            !
            ij0 = 101   ;   ij1 = 102                    ! Decrease before Gibraltar Strait
            ii0 = 139   ;   ii1 = 141   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 0._wp
            ii0 = 142   ;   ii1 = 142   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 1._wp / 90._wp
            ii0 = 143   ;   ii1 = 143   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 0.40_wp
            ii0 = 144   ;   ii1 = 144   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 0.75_wp
            !
            ij0 =  87   ;   ij1 =  96                    ! Red Sea
            ii0 = 147   ;   ii1 = 163   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 1._wp
            !
            ij0 =  91   ;   ij1 =  91                    ! Decrease before Bab el Mandeb Strait
            ii0 = 153   ;   ii1 = 160   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 0.80_wp
            ij0 =  90   ;   ij1 =  90
            ii0 = 153   ;   ii1 = 160   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 0.40_wp
            ij0 =  89   ;   ij1 =  89
            ii0 = 158   ;   ii1 = 160   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 1._wp / 90._wp
            ij0 =  88   ;   ij1 =  88
            ii0 = 160   ;   ii1 = 163   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 0._wp
            !
            !                                        ! =======================
         CASE ( 05 )                                 !  ORCA_R05 configuration
            !                                        ! =======================
            ii0 = 568   ;   ii1 = 574                    ! Mediterranean Sea
            ij0 = 324   ;   ij1 = 333   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 1._wp
            ii0 = 575   ;   ii1 = 658
            ij0 = 314   ;   ij1 = 366   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 1._wp
            !
            ii0 = 641   ;   ii1 = 651                    ! Black Sea (remaining part
            ij0 = 367   ;   ij1 = 372   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 1._wp
            !
            ij0 = 324   ;   ij1 = 333                    ! Decrease before Gibraltar Strait
            ii0 = 565   ;   ii1 = 565   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 1._wp / 90._wp
            ii0 = 566   ;   ii1 = 566   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 0.40_wp
            ii0 = 567   ;   ii1 = 567   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 0.75_wp
            !
            ii0 = 641   ;   ii1 = 665                    ! Red Sea
            ij0 = 270   ;   ij1 = 310   ;   zmrs( ii0:ii1 , ij0:ij1 ) = 1._wp
            !
            ii0 = 666   ;   ii1 = 675                    ! Decrease before Bab el Mandeb Strait
            ij0 = 270   ;   ij1 = 290   
            DO ji = ii0, ii1
               zmrs( ji , ij0:ij1 ) = 0.1_wp * ABS( FLOAT(ji - ii1) )
            END DO 
            !                                       ! ========================
         CASE ( 025 )                               !  ORCA_R025 configuration 
            !                                       ! ========================
            WRITE(numerr,*) ' Mediterranean and Red Sea damping option not implemented for ORCA_R025'
            WRITE(numerr,*) ' Set ln_med_red = .false.'
            STOP
            !
         END SELECT

         zsdmp = 1._wp / ( pn_surf * rday )
         zbdmp = 1._wp / ( pn_bot  * rday )

         ! The l_31_lev option is used to reproduce the old behaviour of
         ! defining the restoration coefficient based on the level number.
         ! This is included to allow damping coefficients for reference
         ! configurations to be kept the same.
         IF (l_31_lev) THEN
            IF (jk <= 17) THEN
               zhfac = 0.5_wp * (  1. - COS( rpi * REAL(jk-1,wp) / 16._wp )  ) / rday
            ELSE
               zhfac = 1._wp / rday
            ENDIF
         ELSE
            zhfac = (  zbdmp + (zsdmp-zbdmp) * EXP( -gdept(1,1)/pn_dep )  )
         ENDIF

         presto(:,:) = zmrs(:,:) * zhfac + ( 1._wp - zmrs(:,:) ) * presto(:,:)

         DEALLOCATE( zmrs )         

   END SUBROUTINE med_red_dmp


END MODULE med_red_seas
