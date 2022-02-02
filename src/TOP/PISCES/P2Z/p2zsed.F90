# include "toggle.h90"

MODULE p2zsed
   !!======================================================================
   !!                         ***  MODULE p2zsed  ***
   !! TOP :   PISCES Compute loss of organic matter in the sediments
   !!======================================================================
   !! History :    -   !  1995-06 (M. Levy)  original code
   !!              -   !  2000-12 (E. Kestenare)  clean up
   !!             2.0  !  2007-12  (C. Deltel, G. Madec)  F90 + simplifications
   !!----------------------------------------------------------------------
   !!   p2z_sed        :  Compute loss of organic matter in the sediments
   !!----------------------------------------------------------------------
   USE oce_trc         !
   USE trd_oce         !
   USE trdtrc          !
   USE trc             !
   USE sms_pisces      !
   !
   USE lbclnk          !
   USE iom             !
   USE prtctl_trc      ! Print control for debbuging

   IMPLICIT NONE
   PRIVATE

   PUBLIC   p2z_sed         ! called in ???
   PUBLIC   p2z_sed_init    ! called in ???

   REAL(wp), PUBLIC ::   sedlam      !: time coefficient of POC remineralization in sediments
   REAL(wp), PUBLIC ::   sedlostpoc  !: mass of POC lost in sediments 
   REAL(wp), PUBLIC ::   vsed        !: detritus sedimentation speed [m/s] 
   REAL(wp), PUBLIC ::   xhr         !: coeff for martin''s remineralisation profile

   !!----------------------------------------------------------------------
   !! NEMO/TOP 4.0 , NEMO Consortium (2018)
   !! $Id: p2zsed.F90 11536 2019-09-11 13:54:18Z smasson $ 
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE p2z_sed( kt )
      !!---------------------------------------------------------------------
      !!                     ***  ROUTINE p2z_sed  ***
      !!
      !! ** Purpose :   compute the now trend due to the vertical sedimentation of
      !!              detritus and add it to the general trend of detritus equations
      !!
      !! ** Method  :   this ROUTINE compute not exactly the advection but the
      !!              transport term, i.e.  dz(wt) and dz(ws)., dz(wtr)
      !!              using an upstream scheme
      !!              the now vertical advection of tracers is given by:
      !!                      dz(trn wn) = 1/bt dk+1( e1t e2t vsed (trn) )
      !!              add this trend now to the general trend of tracer (ta,sa,tra):
      !!                             tra = tra + dz(trn wn)
      !!        
      !!              IF 'key_diabio' is defined, the now vertical advection
      !!              trend of passive tracers is saved for futher diagnostics.
      !!---------------------------------------------------------------------
      INTEGER, INTENT( in ) ::   kt      ! ocean time-step index      
      !
      INTEGER  ::   ji, jj, jk, jl, ierr
      CHARACTER (len=25) :: charout
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: zw2d
      REAL(wp), DIMENSION(jpi,jpj,jpk) :: zwork, ztra
      !!---------------------------------------------------------------------
      !
! yyj debug-->      
      real(wp) ,dimension(jpi,jpj,jpk) :: check_tmp1, check_tmp2
      integer*8 :: ct1, ct2, ct3, ct4
      external :: cyc_time
      integer :: tag 
#if defined slave
      external :: slave_p2zsed
      type p2zsed_var
        integer :: jpi,jpj,jpk,jpkm1, jpdet, narea
        real*8 :: vsed
        integer*8 :: loc_trn, loc_tra, loc_ztra, loc_e3t_n
      endtype

      type(p2zsed_var) :: v
      v%jpi = jpi
      v%jpj = jpj
      v%jpk = jpk
      v%jpkm1 = jpkm1
      v%jpdet = jpdet
      v%narea = narea
      v%vsed = vsed
      
      v%loc_trn = loc(trn)
      v%loc_tra = loc(tra)
      v%loc_ztra = loc(ztra)
      v%loc_e3t_n = loc(e3t_n)

#endif      

      IF( ln_timing )   CALL timing_start('p2z_sed')
      !
      IF( kt == nittrc000 ) THEN
         IF(lwp) WRITE(numout,*)
         IF(lwp) WRITE(numout,*) ' p2z_sed: LOBSTER sedimentation'
         IF(lwp) WRITE(numout,*) ' ~~~~~~~'
      ENDIF

      ! sedimentation of detritus  : upstream scheme
      ! --------------------------------------------
#if defined slave

      call athread_spawn(slave_p2zsed, v)
      call athread_join()

     ! do jk=1, jpkm1
     !   if(jk ==1) then
     !     do jj=1, jpj
     !     do ji=1,jpi
     !       ztra(ji,jj,jk) = -vsed * trn(ji,jj,jk) / e3t_n(ji,jj,jk)
     !       check_tmp1(ji,jj,jk) = check_tmp1(ji,jj,jk) + ztra(ji,jj,jk)
     !     enddo
     !     enddo
     !   else if(jk == jpkm1) then
     !     do jj=1,jpj
     !     do ji=1,jpi
     !       ztra(ji,jj,jk) = vsed * trn(ji,jj,jk-1) / e3t_n(ji,jj,jk)
     !       check_tmp1(ji,jj,jk) = check_tmp1(ji,jj,jk) + ztra(ji,jj,jk)
     !     enddo
     !     enddo
     !   else
     !     do jj =1,jpi
     !     do ji=1,jpi
     !       ztra(ji,jj,jk) = -vsed*(trn(ji,jj,jk) - trn(ji,jj,jk-1)) / e3t_n(ji,jj,jk)
     !       check_tmp1(ji,jj,jk) = check_tmp1(ji,jj,jk) + ztra(ji,jj,jk)
     !     enddo
     !     enddo
     !   endif
     ! enddo  

#else     
!      call cyc_time(ct3)
      ! for detritus sedimentation only - jpdet
      zwork(:,:,1  ) = 0.e0      ! surface value set to zero
      zwork(:,:,jpk) = 0.e0      ! bottom value  set to zero

      ! tracer flux at w-point: we use -vsed (downward flux)  with simplification : no e1*e2
      DO jk = 2, jpkm1
         zwork(:,:,jk) = -vsed * trn(:,:,jk-1,jpdet)
      END DO

      ! tracer flux divergence at t-point added to the general trend
      DO jk = 1, jpkm1
         DO jj = 1, jpj
            DO ji = 1, jpi
               ztra(ji,jj,jk)  = - ( zwork(ji,jj,jk) - zwork(ji,jj,jk+1) ) / e3t_n(ji,jj,jk)
               tra(ji,jj,jk,jpdet) = tra(ji,jj,jk,jpdet) + ztra(ji,jj,jk)
            END DO
         END DO
      END DO

      IF( lk_iomput )  THEN
         IF( iom_use( "TDETSED" ) ) THEN
            ALLOCATE( zw2d(jpi,jpj) )
            zw2d(:,:) =  ztra(:,:,1) * e3t_n(:,:,1) * 86400._wp
            DO jk = 2, jpkm1
               zw2d(:,:) = zw2d(:,:) + ztra(:,:,jk) * e3t_n(:,:,jk) * 86400._wp
            END DO
            CALL iom_put( "TDETSED", zw2d )
            DEALLOCATE( zw2d )
         ENDIF
      ENDIF
!      call cyc_time(ct4)
      !
     ! if(lwp) print*, "speedup ", (ct4-ct3)/(ct2-ct1)
  
#endif      

      IF(ln_ctl)   THEN  ! print mean trends (used for debugging)
         WRITE(charout, FMT="('sed')")
         CALL prt_ctl_trc_info(charout)
         CALL prt_ctl_trc(tab4d=tra, mask=tmask, clinfo=ctrcnm)
      ENDIF
      !
      IF( ln_timing )   CALL timing_stop('p2z_sed')
      !
   END SUBROUTINE p2z_sed


   SUBROUTINE p2z_sed_init
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE p2z_sed_init  ***
      !!
      !! ** Purpose :   Parameters from aphotic layers to sediment
      !!
      !! ** Method  :   Read the namlobsed namelist and check the parameters
      !!
      !!----------------------------------------------------------------------
      INTEGER ::   ios   ! Local integer
      !!
      NAMELIST/namlobsed/ sedlam, sedlostpoc, vsed, xhr
      !!----------------------------------------------------------------------
      !
      REWIND( numnatp_ref )              ! Namelist namlobsed in reference namelist : Lobster sediments
      READ  ( numnatp_ref, namlobsed, IOSTAT = ios, ERR = 901)
901   IF( ios /= 0 )   CALL ctl_nam ( ios , 'namlosed in reference namelist' )
      REWIND( numnatp_cfg )              ! Namelist namlobsed in configuration namelist : Lobster sediments
      READ  ( numnatp_cfg, namlobsed, IOSTAT = ios, ERR = 902 )
902   IF( ios >  0 )   CALL ctl_nam ( ios , 'namlobsed in configuration namelist' )
      IF(lwm) WRITE ( numonp, namlobsed )
      !
      IF(lwp) THEN
          WRITE(numout,*) '   Namelist namlobsed'
          WRITE(numout,*) '      time coeff of POC in sediments                sedlam    =', sedlam
          WRITE(numout,*) '      Sediment geol loss for POC                    sedlostpoc=', sedlostpoc
          WRITE(numout,*) '      detritus sedimentation speed                  vsed      =', 86400 * vsed  , ' d'
          WRITE(numout,*) '      coeff for martin''s remineralistion           xhr       =', xhr
          WRITE(numout,*) ' '
      ENDIF
      !
   END SUBROUTINE p2z_sed_init

   !!======================================================================
END MODULE p2zsed
