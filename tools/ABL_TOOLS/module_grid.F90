MODULE module_grid
   !!======================================================================
   !!                   ***  MODULE  module_grid  ***
   !! ABL utilities to define and store vertical grids 
   !!=====================================================================
   !! History : 2016-10  (F. Lemarié)  Original code
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   FUNCTIONS   : None    
   !!   SUBROUTINES : get_atm_grid, init_atm_mask, get_pot_temp
   !!                 Write_Grid_File, Init_output_File, init_target_grid
   !!                 flip_vert_dim, smooth_field
   !!----------------------------------------------------------------------
   IMPLICIT NONE

CONTAINS

   SUBROUTINE get_atm_grid( jpi, jpj, jpka, slp, temp, humi, Aw, Bw, &   ! in
      &                                                      e3t, ghw )   ! out
      !!--------------------------------------------------------------------------
      !!                    ***  ROUTINE get_atm_grid  ***
      !!                   
      !! ** Purpose : compute layer thickness and altitude of interfaces   
      !!              of the ECMWF atmospheric grid
      !!
      !! ** Method  : 
      !!        (1) recompute the pressure levels thanks to the sea-level pressure
      !!        (2) use the hydrostatic relation to convert pressure into altitudes
      !!
      !!---------------------------------------------------------------------------      
      INTEGER, INTENT(in   ) ::   jpi, jpj
      INTEGER, INTENT(in   ) ::   jpka
      REAL(8), INTENT(in   ) ::   slp   ( 1:jpi, 1:jpj         )
      REAL(8), INTENT(in   ) ::   temp  ( 1:jpi, 1:jpj, 1:jpka )      
      REAL(8), INTENT(in   ) ::   humi  ( 1:jpi, 1:jpj, 1:jpka ) 
      REAL(8), INTENT(in   ) ::   Aw    (               0:jpka )
      REAL(8), INTENT(in   ) ::   Bw    (               0:jpka )                 
      REAL(8), INTENT(  out) ::   e3t   ( 1:jpi, 1:jpj, 1:jpka )  
      REAL(8)                ::   ghw   ( 1:jpi, 1:jpj, 0:jpka )                           
      REAL(8), PARAMETER     ::   g    =   9.80665
      REAL(8), PARAMETER     ::   Rd   = 287.058   
      REAL(8), PARAMETER     ::   zvir =   0.609133 
      REAL(8), PARAMETER     ::   ig   =   1./g
      !!
      INTEGER                :: ji,jj,jk
      REAL(8)                :: tv,ph(0:jpka)
      !
      DO jj = 1, jpj
         DO ji = 1, jpi

            DO jk=0,jpka
               ph(jk) = Aw( jk ) + Bw( jk ) * slp( ji, jj )    !<-- Pa
            END DO
            
            DO jk=jpka,1,-1
              tv      = temp( ji, jj, jk ) * ( 1. + zvir*humi( ji, jj, jk ) )  !<-- Virtual temperature
              e3t ( ji, jj, jk   ) =  ig*( Rd * tv * log( ph( jk ) / ph( jk-1 ) ) )
              ghw ( ji, jj, jk-1 ) =  e3t( ji, jj, jk ) + ghw( ji, jj, jk )
            END DO  

         END DO
      END DO
      !
   END SUBROUTINE get_atm_grid




   SUBROUTINE init_atm_mask( jpi, jpj, mask_file, mask_name, ln_lsm_land, tmask )
      USE module_io
      !!---------------------------------------------------------------------
      !!                    ***  ROUTINE INIT_atm_MASK  ***
      !!                   
      !! ** Purpose : extract the land/sea mask and remove isolated sea points 
      !!
      !! ** Method  : mask is 1 over the ocean and 0 over land 
      !!
      !!----------------------------------------------------------------------         
      INTEGER, INTENT(in   ) ::   jpi, jpj 
      LOGICAL, INTENT(in   ) ::   ln_lsm_land
      REAL(8)                ::   tmask  ( 1:jpi, 1:jpj, 1 ) 
      INTEGER                ::   jj, ji        
      INTEGER                ::   status,ncid,varid  
      CHARACTER(len = *  )   ::   mask_file, mask_name
      REAL(8)                ::   cff
        
      ! Read land-sea mask variable
      status    = nf90_open(trim(mask_file),NF90_NOWRITE,ncid)      
      status    = nf90_inq_varid(ncid,mask_name,varid)        
      status    = nf90_get_var(ncid,varid,tmask,start=(/1,1,1/))                                             
      status    = nf90_close(ncid)  
      ! invert the mask (1 over the ocean, 0 over land if ln_lsm_land)
      IF(ln_lsm_land) THEN
         DO jj=1,jpj
            DO ji=1,jpi
               IF( tmask(ji,jj,1) <= 0. ) THEN
                  tmask(ji,jj,1) = 1.     !! Ocean points
               ELSE
                  tmask(ji,jj,1) = 0.     !! Land points     
               END IF
            END DO
         END DO  
      ENDIF
      ! remove some closed seas
      DO jj=2,jpj-1
         DO ji=2,jpi-1
            cff = MAX( tmask(ji+1,jj  ,1),tmask(ji-1,jj  ,1), &
               &       tmask(ji  ,jj+1,1),tmask(ji  ,jj-1,1), &
               &       tmask(ji+1,jj+1,1),tmask(ji-1,jj-1,1), &
               &       tmask(ji+1,jj-1,1),tmask(ji-1,jj+1,1) )     
            IF( tmask( ji, jj, 1 ) .gt. 0.5 .and. cff .lt. 0.5 ) THEN
               tmask( ji, jj, 1 ) = 0.
            END IF
         END DO
      END DO
      WRITE(*,*)' init_atm_mask: ',mask_name,' in ',mask_file, ' OK'
      !!---------------------------------------------------------------------- 
      !
   END SUBROUTINE init_atm_mask


       
   SUBROUTINE get_pot_temp( jpi, jpj, jpka, slp, temp, Aw, Bw, tmask, method, hum1, z1 )         
      !!---------------------------------------------------------------------
      !!                    ***  ROUTINE get_pot_temp  ***
      !!                   
      !! ** Purpose :  compute the potential temperature based on the 
      !!               absolute temperature and the sea level pressure
      !!
      !! ** Method  : five different ways are implemented depending on the 
      !!              value of 'method'
      !!       (0) potential temperature = absolute temperature (not recommended)
      !!       (1) potential temperature is computed using a local reference 
      !!                                pressure equal to the sea-level-pressure
      !!       (2) potential temperature is computed only on a perturbation of 
      !!                                  the absolute temperature around t0
      !!       (3) a local reference pressure is used consistently with AEROBULK gamma_moist
      !!       (4) a constant global reference pressure is used
      !!----------------------------------------------------------------------  
      INTEGER, INTENT(in   ) ::   jpi, jpj
      INTEGER, INTENT(in   ) ::   jpka, method
      REAL(8), INTENT(in   ) ::   slp   ( 1:jpi, 1:jpj         )
      REAL(8), INTENT(in   ) ::   tmask ( 1:jpi, 1:jpj         )   
      REAL(8), INTENT(in   ) ::   z1    ( 1:jpi, 1:jpj         )            
      REAL(8), INTENT(inout) ::   temp  ( 1:jpi, 1:jpj, 1:jpka ) 
      REAL(8), INTENT(in   ) ::   hum1  ( 1:jpi, 1:jpj         )       
      REAL(8), INTENT(in   ) ::   Aw    (               0:jpka )
      REAL(8), INTENT(in   ) ::   Bw    (               0:jpka )                         
      REAL(8), PARAMETER     ::   grav   =   9.80665
      REAL(8), PARAMETER     ::   R_dry  =   287.058 
      REAL(8), PARAMETER     ::   R_vap  =   461.495
      REAL(8), PARAMETER     ::   Cp_dry =   1005.             
      REAL(8), PARAMETER     ::   cevap  =   2.5E+06       
      REAL(8), PARAMETER     ::   reps0  =   R_dry / R_vap        
      REAL(8), PARAMETER     ::   gamma  =   2./7.
      REAL(8)                ::   pres0, gamma_moist, zrv, zirt     
      REAL(8), PARAMETER     ::   t0     =      288.     !<-- K
      !!
      INTEGER                :: ji,jj,jk
      REAL(8)                :: pres,ph(0:jpka), cff         
      !!
      SELECT CASE ( method ) 
      CASE(0) 
         RETURN
      CASE(1)         
         DO jj = 1, jpj
            DO ji = 1, jpi

               IF( tmask(ji,jj) .gt. 0.5 ) THEN
                  DO jk=0,jpka
                     ph(jk) = Aw( jk ) + Bw( jk ) * slp( ji, jj )
                  END DO            

                  pres0 = 0.5*(ph(jpka)+ph(jpka-1))
            
                  DO jk=1,jpka
                     pres = 0.5*(ph(jk)+ph(jk-1))
                     cff  = ( pres0/pres )**gamma
                     temp( ji, jj, jk ) = cff * temp( ji, jj, jk )
                  END DO
               ELSE
                  temp( ji, jj, 1:jpka ) = 273.15
               END IF
            END DO
         END DO        
      CASE(2)
         pres0 =   100900.      
         DO jj = 1, jpj
            DO ji = 1, jpi
               IF( tmask(ji,jj) .gt. 0.5 ) THEN
                  DO jk=0,jpka
                     ph(jk) = Aw( jk ) + Bw( jk ) * slp( ji, jj )
                  END DO            
            
                  DO jk=1,jpka
                     pres = 0.5*(ph(jk)+ph(jk-1))
                     cff  = ( pres0/pres )**gamma
                     temp( ji, jj, jk ) = cff * ( temp( ji, jj, jk ) - t0 ) + t0
                  END DO
               ELSE
                  temp( ji, jj, 1:jpka ) = 273.15
               END IF
            END DO
         END DO  
      CASE(3)
         DO jj = 1, jpj
            DO ji = 1, jpi      
               IF( tmask(ji,jj) .gt. 0.5 ) THEN
                  DO jk=0,jpka
                     ph(jk) = Aw( jk ) + Bw( jk ) * slp( ji, jj )
                  END DO 
                  !! compute gamma_moist consistently with AEROBULK
                  zrv         = hum1( ji, jj ) / ( 1. - hum1( ji, jj ) )
                  zirt        = 1. / ( R_dry * temp( ji, jj, jpka ) )
                  gamma_moist = grav * ( 1. + cevap*zrv*ziRT )   & 
                     &     / ( Cp_dry + cevap*cevap*zrv*reps0*ziRT/temp( ji, jj, jpka ) )       
                  !! pressure at z = z1
                  pres        = 0.5*(ph(jpka)+ph(jpka-1))                  
                  pres0       = pres * ( 1. + gamma_moist * z1(ji,jj) / temp( ji, jj, jpka ) )**(1./gamma)
                  DO jk=1,jpka
                     pres = 0.5*(ph(jk)+ph(jk-1))
                     cff  = ( pres0/pres )**gamma
                     temp( ji, jj, jk ) = cff * temp( ji, jj, jk )
                  END DO
               ELSE
                  temp( ji, jj, 1:jpka ) = 273.15
               END IF
            END DO
         END DO            
      CASE(4)
         pres0 =   100900.      
         DO jj = 1, jpj
            DO ji = 1, jpi
               IF( tmask(ji,jj) .gt. 0.5 ) THEN
                  DO jk=0,jpka
                     ph(jk) = Aw( jk ) + Bw( jk ) * slp( ji, jj )
                  END DO            
            
                  DO jk=1,jpka
                     pres = 0.5*(ph(jk)+ph(jk-1))
                     cff  = ( pres0/pres )**gamma
                     temp( ji, jj, jk ) = cff * temp( ji, jj, jk )
                  END DO
               ELSE
                  temp( ji, jj, 1:jpka ) = 273.15
               END IF
            END DO
         END DO       
      END SELECT
      !
   END SUBROUTINE get_pot_temp


   SUBROUTINE Write_Grid_File( jpka, ght, ghw, e3t, e3w, grd_file  )  
      !!
      USE module_io
      !!---------------------------------------------------------------------
      !!                    ***  ROUTINE write_Grid_File  ***
      !!                   
      !! ** Purpose : write the ABL grid file  
      !!
      !! ** Method  : store the layer thickness and altitude of grid points 
      !!
      !!----------------------------------------------------------------------
      INTEGER,      INTENT(in   )  ::   jpka
      REAL(8),      INTENT(in   )  ::   ght ( 1:jpka+1 )
      REAL(8),      INTENT(in   )  ::   ghw ( 1:jpka+1 )      
      REAL(8),      INTENT(in   )  ::   e3t ( 1:jpka+1 )
      REAL(8),      INTENT(in   )  ::   e3w ( 1:jpka+1 )            
      CHARACTER(*), INTENT(in   )  ::   grd_file
      !!
      INTEGER                      :: status, ncid
      !!
      status   = nf90_create( grd_file, NF90_WRITE, ncid )
      status   = nf90_close ( ncid )    
      !!  
      Call Write_Ncdf_dim ( 'jpka' , grd_file, jpka+1 )       
      Call Write_Ncdf_var ( 'ghw', 'jpka', grd_file, ghw, 'double' ) 
      Call Write_Ncdf_var ( 'ght', 'jpka', grd_file, ght, 'double' )
      Call Write_Ncdf_var ( 'e3t', 'jpka', grd_file, e3t, 'double' )
      Call Write_Ncdf_var ( 'e3w', 'jpka', grd_file, e3w, 'double' )
      !
   END SUBROUTINE Write_Grid_File

       



   SUBROUTINE Init_output_File ( jpi, jpj, jpka, atm_file, abl_file, tmask  )  
      !!
      USE module_io
      !!---------------------------------------------------------------------
      !!                    ***  ROUTINE Init_output_File  ***
      !!                   
      !! ** Purpose :  write longitude, latitude and mask in the output file
      !!
      !! ** Method  : define dimensions in the netcdf file
      !!----------------------------------------------------------------------
      INTEGER,                    INTENT(in   )  ::   jpi,jpj,jpka
      CHARACTER(*),               INTENT(in   )  ::   atm_file      
      CHARACTER(*),               INTENT(in   )  ::   abl_file
      REAL(8)                                    ::   tmask(1:jpi,1:jpj)
      !!
      REAL(8), ALLOCATABLE, DIMENSION(:      )   ::   tmp1d
      INTEGER                                    ::   status, ncid
      CHARACTER(len= 20),DIMENSION(4)            :: dimnames
      !!
      status   = nf90_create( abl_file, NF90_WRITE, ncid )
      status   = nf90_close ( ncid )    
      !!     
      CALL Write_Ncdf_dim ( 'lon'    , abl_file, jpi     )
      CALL Write_Ncdf_dim ( 'lat'    , abl_file, jpj     )
      CALL Write_Ncdf_dim ( 'jpka'   , abl_file, jpka+1  )         
      CALL Write_Ncdf_dim ( 'time'   , abl_file, 0       )     
      !
      ALLOCATE( tmp1d( 1:jpi ) )
      !
      CALL Read_Ncdf_var  ( 'lon', atm_file, tmp1d )
      CALL Write_Ncdf_var ( 'lon', 'lon', abl_file, tmp1d, 'double' ) 
      !
      DEALLOCATE( tmp1d )
      !
      ALLOCATE( tmp1d( 1:jpj ) )           
      !
      CALL Read_Ncdf_var ( 'lat', atm_file, tmp1d )   
      CALL Write_Ncdf_var( 'lat', 'lat', abl_file, tmp1d, 'double' )     
      !
      DEALLOCATE( tmp1d )   
      !
      dimnames(1) = 'lon'
      dimnames(2) = 'lat'
      CALL Write_Ncdf_var( 'lsm', dimnames , abl_file, tmask, 'float' )
      !
   END SUBROUTINE Init_output_File
       
       
   SUBROUTINE Init_output_File_c1d ( jpi, jpj, jpka, atm_file, abl_file, tmask, iloc, jloc )
      !!
      USE module_io
      !!---------------------------------------------------------------------
      !!                    ***  ROUTINE Init_output_File  ***
      !!
      !! ** Purpose :  write longitude, latitude and mask in the output file
      !!
      !! ** Method  : define dimensions in the netcdf file
      !!----------------------------------------------------------------------
      INTEGER,                    INTENT(in   )  ::   iloc,jloc,jpi,jpj,jpka
      CHARACTER(*),               INTENT(in   )  ::   atm_file
      CHARACTER(*),               INTENT(in   )  ::   abl_file
      REAL(8)                                    ::   tmask(1:jpi,1:jpj)
      !!
      REAL(8), ALLOCATABLE, DIMENSION(:      )   ::   tmp1d
      REAL(8)                                    ::   tmp1d_loc(3),tmp2d_loc(3,3)
      INTEGER                                    ::   status, ncid
      CHARACTER(len= 20),DIMENSION(4)            :: dimnames
      !!
      status   = nf90_create( abl_file, NF90_WRITE, ncid )
      status   = nf90_close ( ncid )
      !!
      CALL Write_Ncdf_dim ( 'lon'    , abl_file, 3       )
      CALL Write_Ncdf_dim ( 'lat'    , abl_file, 3       )
      CALL Write_Ncdf_dim ( 'jpka'   , abl_file, jpka+1  )
      CALL Write_Ncdf_dim ( 'time'   , abl_file, 0       )
      !
      ALLOCATE( tmp1d    ( 1:jpi ) )
      !
      CALL Read_Ncdf_var  ( 'lon', atm_file, tmp1d )
      tmp1d_loc( 1 ) = tmp1d(iloc)
      tmp1d_loc( 2 ) = tmp1d(iloc)
      tmp1d_loc( 3 ) = tmp1d(iloc)
      CALL Write_Ncdf_var ( 'lon', 'lon', abl_file, tmp1d_loc, 'double' )
      print*,'1D column located at : '
      print*,'longitude = ',tmp1d(iloc),' degree_east'
      !
      DEALLOCATE( tmp1d )
      !
      ALLOCATE( tmp1d( 1:jpj ) )
      !
      CALL Read_Ncdf_var ( 'lat', atm_file, tmp1d )
      tmp1d_loc( 1 ) = tmp1d(jloc)
      tmp1d_loc( 2 ) = tmp1d(jloc)
      tmp1d_loc( 3 ) = tmp1d(jloc)
      CALL Write_Ncdf_var( 'lat', 'lat', abl_file, tmp1d_loc, 'double' )
      print*,'latitude = ',tmp1d(jloc),' degree_north'
      !
      DEALLOCATE( tmp1d )
      !
      tmp2d_loc(:,:) = tmask(iloc,jloc)       
      dimnames(1) = 'lon'
      dimnames(2) = 'lat'
      CALL Write_Ncdf_var( 'lsm', dimnames , abl_file, tmp2d_loc, 'float' )
      !
   END SUBROUTINE Init_output_File_c1d       
       

   SUBROUTINE init_target_grid( jpka, ght, ghw, e3t, e3w, hmax, hc, theta_s,  &
      &                                                        force_z1, z1 )
      !!---------------------------------------------------------------------
      !!                    ***  ROUTINE init_target_grid  ***
      !!                   
      !! ** Purpose : compute the layer thickness and altitude of grid points 
      !!              for the ABL model based on the namelist parameter values  
      !!
      !! ** Method  : depending on the logical 'force_z1' two methods are used
      !!              (1) true -> the user chooses the value of the first vertical
      !!                  grid point. A few Newton iterations are used to correct  
      !!                  the value of the parameter theta_s to satisfy this constraint
      !!              (2) false -> use the parameter values in the namelist to 
      !!                  compute the vertical grid
      !!
      !!----------------------------------------------------------------------
      INTEGER, intent(in   )    :: jpka
      REAL(8), intent(in   )    :: hmax,hc
      REAL(8), intent(inout)    :: theta_s      
      LOGICAL, intent(in   )    :: force_z1
      REAL(8), intent(in   )    :: z1
      !!
      REAL(8), intent(  out)    :: ghw( 1:jpka+1 )
      REAL(8), intent(  out)    :: ght( 1:jpka+1 )      
      REAL(8), intent(  out)    :: e3w( 1:jpka+1 )
      REAL(8), intent(  out)    :: e3t( 1:jpka+1 ) 
      !! 
      REAL(8)                   ::  ds,cff,sc_w,sc_r,alpha,x
      REAL(8)                   ::  fx,fxp
      INTEGER                   ::  jk,maxiter,jiter
      REAL(8), PARAMETER        :: tol = 1.E-12
      !!
      IF(force_z1) THEN
        IF(z1.LT.10.) THEN
          WRITE(*,*) " ERROR: z1 < 1st ECMWF level height (~10m)"
          STOP
        ELSE
          !! Newton iterations to find the appropriate value of theta_s
          maxiter = 1000
          x       = theta_s
          sc_r    = (float(1)-0.5)/float(jpka)
          alpha   = (z1 - hc*sc_r) / (hmax - hc)
          !
          DO jiter=1,maxiter
             fx   = (sinh(sc_r*x)/sinh(x))-alpha
             fxp  = (sc_r*cosh(sc_r*x)-sinh(sc_r*x)*cosh(x)/sinh(x))/sinh(x)
             IF( abs(fx) .lt. tol ) THEN
                   exit
             ENDIF
             cff  = fx / fxp
             x    = x - cff      
          ENDDO
          !
          theta_s = x
        END IF
        !   
      ENDIF
      !
      ds =1./float(jpka)
      cff=(hmax-hc)/sinh(theta_s)
      !
      DO jk = jpka,1,-1
        sc_w      = ds*float(jk)
        ghw(jk+1) = hc*sc_w + cff*sinh(theta_s*sc_w)
        sc_r      = ds*(float(jk)-0.5)
        ght(jk+1) = hc*sc_r + cff*sinh(theta_s*sc_r)
      END DO
      !
      ghw(1) = 0.
      e3t(1) = 0.
      ght(1) = 0.
      !
      DO jk = 2,jpka+1
        e3t(jk) = ghw(jk)-ghw(jk-1)
      END DO
      !
      DO jk=1,jpka
        e3w(jk)  = ght(jk+1)-ght(jk)
      END DO
      !
      e3w(jpka+1) = ghw(jpka+1) - ght(jpka+1)          
      !
      IF(force_z1) THEN   !++ print the new parameter values
         print*,'*** Updated grid parameters'
         print*,'theta_s = ',theta_s
         print*,'hc      = ',hc      
         print*,'hmax    = ',hmax         
         print*,'ght(2)  = ',ght(2)
      ENDIF      
      !
   END SUBROUTINE init_target_grid
   
   
   
   
   

   SUBROUTINE flip_vert_dim( kstr, kend, jpi, jpj, tabin )
      !!---------------------------------------------------------------------
      !!                    ***  ROUTINE flip_vert_dim  ***
      !!                   
      !! ** Purpose : flip the vertical axis of the array tabin so that    
      !!              the vertical grid goes from k=kstr at the bottom 
      !!              of the ABL to k=kend at the top
      !!----------------------------------------------------------------------
      INTEGER, intent(in   )   :: kstr,kend
      INTEGER, intent(in   )   :: jpi, jpj 
      REAL(8), intent(inout)   :: tabin( 1:jpi, 1:jpj, kstr:kend )
      !!
      INTEGER                  :: ji,jj,jk,ks
      REAL(8)                  :: tabfl(kstr:kend)
      !
      DO jj = 1,jpj
         DO ji = 1,jpi
            DO jk=kstr,kend
               ks=(kend-jk)+kstr
               tabfl(ks) = tabin( ji, jj, jk ) 
            END DO
            tabin( ji, jj, kstr:kend ) = tabfl( kstr:kend )
         END DO 
      END DO 
      !
   END SUBROUTINE flip_vert_dim
   
  





   SUBROUTINE smooth_field( jpi, jpj, varin, tmask, niter )         
      !!---------------------------------------------------------------------
      !!                    ***  ROUTINE smooth_field  ***
      !!                   
      !! ** Purpose : smooth the sea level pressure over the ocean
      !!              to attenuate Gibbs oscillation 
      !! ** Method  : a 9-point isotropic laplacian filter is applied
      !!              iteratively on ocean grid points only
      !!
      !! Proper treatment of the periodicity is still not yet implemented
      !!
      !!----------------------------------------------------------------------     
      INTEGER, INTENT(in   ) ::   jpi, jpj, niter  
      REAL(8), INTENT(in   ) ::   tmask  ( 1:jpi, 1:jpj )        
      REAL(8), INTENT(inout) ::   varin  ( 1:jpi, 1:jpj )
      INTEGER                :: ji,jj,nit
      REAL(8)                :: smth_a,smth_b,umask,vmask 
      REAL(8)                :: FX ( 0:jpi  , 0:jpj+1 )
      REAL(8)                :: FE1( 0:jpi+1, 0:jpj   )
      REAL(8)                :: FE ( 1:jpi  , 0:jpj   )       
      !!  
      !!=========================================================
      !!    
      ! Hanning filter
      !smth_a = 1./8.
      !smth_b = 1./4.     
      ! 9-point isotropic laplacian filter
      smth_a = 1./12.
      smth_b = 3./16.      

      FX ( 0:jpi  , 0:jpj+1 ) = 0.
      FE1( 0:jpi+1, 0:jpj   ) = 0.
      FE ( 1:jpi  , 0:jpj   ) = 0.

      !!+++++++++
      DO nit = 1,niter
      !!+++++++++
         DO jj=1,jpj              
            DO ji=1,jpi-1          
            umask = tmask(ji,jj)*tmask(ji+1,jj)  
            FX ( ji, jj ) = ( varin( ji+1,jj ) - varin( ji  ,jj ) ) * umask  
            END DO                    
         END DO
         FX(   0  , 1:jpj ) = FX( jpi-1, 1:jpj  )
         FX  ( jpi, 1:jpj ) = FX(     1, 1:jpj  )      
         FX( 0:jpi, 0     ) = FX( 0:jpi,     1  )
         FX( 0:jpi, jpj+1 ) = FX( 0:jpi,     jpj)           
         !!
         DO jj=1,jpj-1                 
            DO ji=1,jpi
               vmask = tmask(ji,jj)*tmask(ji,jj+1)  
               FE1( ji, jj ) = ( varin( ji, jj+1 ) - varin( ji  ,jj ) ) * vmask 
           END DO
         END DO
         !! 
         FE1( 0      , 1:jpj-1 ) = FE1( jpi , 1:jpj-1 )
         FE1( jpi+1  , 1:jpj-1 ) = FE1(   1 , 1:jpj-1 )      
         FE1( 0:jpi+1,    0    ) = 0.
         FE1( 0:jpi+1,    jpj  ) = 0. 
         !!
         DO jj=0,jpj                 
            DO ji=1,jpi
               FE ( ji, jj ) =   FE1( ji, jj   )                   &
               & +  smth_a*(  FX ( ji, jj+1 ) + FX( ji-1, jj   )   &
               &           -  FX ( ji, jj   ) - FX( ji-1, jj+1 ) )
            END DO
         END DO
         !!
         DO jj = 1, jpj
            DO ji = 0,jpi
               FX( ji, jj ) = FX( ji, jj )         &
              &    + smth_a*( FE1( ji+1, jj ) + FE1( ji  , jj-1 )    &
              &              -FE1( ji  , jj ) - FE1( ji+1, jj-1 ) )
            END DO
            DO ji = 1,jpi
               varin( ji  ,jj ) = varin( ji  ,jj ) &
     &         + tmask(ji,jj) * smth_b * (         &
     &                  FX( ji, jj ) - FX( ji-1, jj )          &
     &                 +FE( ji, jj ) - FE( ji, jj-1 )  )
            END DO
         END DO      
         !!
      !!+++++++++      
      END DO
      !!+++++++++   
            
   END SUBROUTINE smooth_field
















   

  




!   SUBROUTINE DTV_Filter( jpi, jpj, varin, tmask, niter, time )         
!      USE module_io
!      !!---------------------------------------------------------------------
!      !!                    ***  ROUTINE DTV_Filter  ***
!      !!                   
!      !! ** Purpose :   
!      !!
!      !! ** Method  : 
!      !!
!      !! ** Action  : 
!      !!----------------------------------------------------------------------   
!      INTEGER, INTENT(in   ) ::   jpi, jpj, niter, time 
!      REAL(8), INTENT(in   ) ::   tmask  ( 1:jpi, 1:jpj )        
!      REAL(8), INTENT(inout) ::   varin  ( 1:jpi, 1:jpj )
!      INTEGER                ::   jn, jj, ji, nit, nt_n, nt_a, nl2      
!      REAL(8)                ::   var0,mean,sigma2,cff,lambda
!      REAL(8)                ::   FX(0:jpi+1,0:jpj+1)
!      REAL(8)                ::   FE(0:jpi+1,0:jpj+1)
!      REAL(8)                ::   FL(0:jpi+1,0:jpj+1)
!      REAL(8)                ::   FR(0:jpi+1,0:jpj+1)  
!      REAL(8)                ::   wrk ( 1:jpi, 1:jpj, 2 ) 
!      REAL(8)                ::   div ( 0:jpi+1, 0:jpj+1 ), diag               
!      REAL(8), PARAMETER     ::   rsmall = 1.E-08
!      REAL(8), PARAMETER     ::   rbig   = 1.E+14
!      REAL(8)                ::   umask, vmask, fmask, wght(8), dt
!      REAL(8)                ::   L2norm_n, L2norm_a
!      REAL(8), ALLOCATABLE, DIMENSION(:      )   ::   tmp1d      
!      INTEGER                :: status,ncid,varid1,varid2,dimid1,dimid2  
!      !!=========================================================
!      CHARACTER(len=  3  )   :: nn
!      CHARACTER(len = 500)   :: erai_file, smth_file
!      LOGICAL                :: ln_diag_smoothing
!      ln_diag_smoothing = .true.
!      IF( ln_diag_smoothing ) THEN
!         erai_file = '/Users/florianlemarie/Documents/INRIA/SIMBAD/ERA_INTERIM_DECEMBRE_2015/phi_ml1_6h_erai_201512.nc'
!         WRITE(nn,'(I3)') time
!         DO ji=1,3
!            IF(nn(ji:ji)==' ') nn(ji:ji)='0'
!         END DO
!      
!         smth_file = 'smoothing_results_DTV_'//nn//'.nc'  
!         status    = nf90_create( trim(smth_file) , NF90_WRITE, ncid )
!         status    = nf90_close ( ncid )    
!         !   
!         CALL Write_Ncdf_dim ( 'lon'    , trim(smth_file) , jpi     )
!         CALL Write_Ncdf_dim ( 'lat'    , trim(smth_file) , jpj     )
!         !
!         ALLOCATE( tmp1d( 1:jpi ) )
!         CALL Read_Ncdf_var  ( 'lon', erai_file, tmp1d )
!         CALL Write_Ncdf_var ( 'lon', 'lon', trim(smth_file), tmp1d, 'double' ) 
!         DEALLOCATE( tmp1d )
!         !
!         ALLOCATE( tmp1d( 1:jpj ) )           
!         CALL Read_Ncdf_var ( 'lat', erai_file, tmp1d )   
!         CALL Write_Ncdf_var( 'lat', 'lat', trim(smth_file), tmp1d, 'double' )     
!         DEALLOCATE( tmp1d ) 
!         !         
!         status = nf90_open(trim(smth_file),NF90_WRITE,ncid)     
!         status = nf90_inq_dimid(ncid, 'lon', dimid1)
!         status = nf90_inq_dimid(ncid, 'lat', dimid2)
!         status = nf90_redef(ncid)      
!         status = nf90_def_var(ncid,'varinp',nf90_double,(/dimid1,dimid2/),varid1)        
!         status = nf90_def_var(ncid,'varout',nf90_double,(/dimid1,dimid2/),varid2)   
!         status = nf90_enddef(ncid)
!         ! 
!         !
!         status = nf90_put_var(ncid,varid1,varin)
!      END IF
!      !!=========================================================
!      nit    = 0
!      nl2    = 0
!      L2norm_a = 0.
!      L2norm_n = rbig      
!      nt_n   = 1 + MOD( nit  , 2 )
!      nt_a   = 1 + MOD( nit+1, 2 )      
!      var0   = varin(  nint(0.5*jpi) , nint(0.5*jpj) )  
!      varin( 1:jpi, 1:jpj       ) = varin( 1:jpi, 1:jpj ) - var0
!      wrk  ( 1:jpi, 1:jpj, nt_n ) = varin( 1:jpi, 1:jpj )
!      !! Compute the mean first
!      mean   = Get_Mean ( jpi , jpj,  varin, tmask       )
!      sigma2 = Get_Vari ( jpi , jpj,  varin, tmask, mean )
!      !!
!      print*,'mean value over the ocean = ',mean + var0
!      print*,'variance of the input field = ',sigma2      
!      !!
!      lambda = (1. / sigma2      )
!      dt     =  1. / (16.+lambda)
!      diag   =  0.
!      
!      !!>>>>>>>>>>>>>>>>
!      DO nit = 1,niter
!      !!>>>>>>>>>>>>>>>>
!      FX(0:jpi+1,0:jpj+1) = 0.
!      FE(0:jpi+1,0:jpj+1) = 0.
!      FL(0:jpi+1,0:jpj+1) = 0.
!      FR(0:jpi+1,0:jpj+1) = 0.
!      !!
!      DO jj = 1,jpj
!         DO ji = 1,jpi-1
!            umask        = tmask(ji,jj)*tmask(ji+1,jj)  
!            FX( ji, jj ) = umask * ( wrk( ji+1, jj, nt_n ) - wrk( ji, jj, nt_n ) )
!         END DO
!      END DO  
!      !!
!      DO jj = 1,jpj-1
!         DO ji = 1,jpi
!            vmask        = tmask(ji,jj)*tmask(ji,jj+1)
!            FE( ji, jj ) = vmask * ( wrk( ji, jj+1, nt_n ) - wrk( ji, jj, nt_n ) )
!         END DO
!      END DO       
!      !!
!      DO jj = 1,jpj-1
!         DO ji = 1,jpi-1
!            fmask        = tmask(ji,jj)*tmask(ji+1,jj+1)*diag
!            FL( ji, jj ) = fmask * ( wrk( ji+1, jj+1, nt_n ) - wrk( ji  , jj, nt_n   ) )
!            fmask        = tmask(ji+1,jj)*tmask(ji,jj+1)*diag       
!            FR( ji, jj ) = fmask * ( wrk( ji  , jj+1, nt_n ) - wrk( ji+1, jj, nt_n ) )
!         END DO
!      END DO 
!      !!
!      div( 0:jpi+1, 0:jpj+1 ) = rsmall
!      !!
!      DO jj = 1,jpj+1
!         DO ji = 1,jpi+1           
!            div(ji,jj) = MAX( sqrt(  FX(ji,jj)**2 + FX(ji-1,jj  )**2 + FE(ji,jj  )**2 + FX(ji,jj-1)**2          &
!            &                      + FL(ji,jj)**2 + FL(ji-1,jj-1)**2 + FR(ji,jj-1)**2 + FR(ji-1,jj)**2 ), rsmall ) 
!         END DO
!      END DO
!      
!      
!       
!      
!      DO jj=1,jpj
!         DO ji=1,jpi
!            IF( div(ji,jj) .eq. rsmall .or. tmask(ji,jj).lt.0.5 ) THEN
!               wrk(ji,jj,nt_a) = wrk(ji,jj,nt_n)
!            ELSE
!               wght(1)     = 1. +  div(ji,jj) / div(ji+1,jj  ) 
!               wght(2)     = 1. +  div(ji,jj) / div(ji-1,jj  ) 
!               wght(3)     = 1. +  div(ji,jj) / div(ji  ,jj+1)                        
!               wght(4)     = 1. +  div(ji,jj) / div(ji  ,jj-1)           
!               wght(5)     = 1. +  div(ji,jj) / div(ji+1,jj+1)   
!               wght(6)     = 1. +  div(ji,jj) / div(ji-1,jj+1)
!               wght(7)     = 1. +  div(ji,jj) / div(ji-1,jj-1)                       
!               wght(8)     = 1. +  div(ji,jj) / div(ji+1,jj-1)             
!              
!            wrk(ji,jj,nt_a) = wrk(ji,jj,nt_n) +  dt*(                   &
!               &            + FX(ji  ,jj  ) * wght(1)                   &
!               &            - FX(ji-1,jj  ) * wght(2)                   &
!               &            + FE(ji  ,jj  ) * wght(3)                   &
!               &            - FE(ji  ,jj-1) * wght(4)                   &
!               &            + FL(ji  ,jj  ) * wght(5)                   &
!               &            + FR(ji-1,jj  ) * wght(6)                   &
!               &            - FL(ji-1,jj-1) * wght(7)                   &
!               &            - FR(ji  ,jj-1) * wght(8)                   &
!               &            - lambda * div(ji,jj) * ( wrk(ji,jj,nt_n) - varin(ji,jj) ) )  
!               
!              IF( isnan(wrk(ji,jj,nt_a)) ) THEN 
!                  print*,'Nan in smoothing at iteration ',nit
!                  print*,'at grid point ',ji,jj
!              END IF    
!              IF( abs(wrk(ji,jj,nt_a)) .gt. rbig ) THEN
!                  print*,'Inf in smoothing at iteration ',nit
!                  print*,'at grid point ',ji,jj                                    
!              END IF
!                 L2norm_a = L2norm_a + ( wrk(ji,jj,nt_n)-wrk(ji,jj,nt_a) )**2
!              END IF                 
!         END DO
!      END DO
!
!
!      mean   = Get_Mean ( jpi , jpj,  wrk(:,:,nt_a), tmask       )
!      sigma2 = Get_Vari ( jpi , jpj,  wrk(:,:,nt_a), tmask, mean )
!      !!
!      print*,'mean value over the ocean = ',mean + var0
!      print*,'variance of the input field = ',sigma2   
!
!      
!      IF( L2norm_a .gt. L2norm_n ) THEN
!         print*,'convergence after ',nit,' iterations'
!!         EXIT
!      END IF     
!      L2norm_n = L2norm_a 
!      L2norm_a = 0.     
!      nt_n = 1 + MOD( nit  , 2 )
!      nt_a = 1 + MOD( nit+1, 2 ) 
!           
!      !!>>>>>>>>>>>>>>>>
!      END DO
!      !!>>>>>>>>>>>>>>>>
!
!       
!      DO jj=1,jpj
!         DO ji=1,jpi
!            varin(ji,jj) = ( wrk(ji,jj,nt_n) + var0 )*tmask(ji,jj)
!         END DO
!      END DO
!
!      IF( ln_diag_smoothing ) THEN
!         status = nf90_put_var(ncid,varid2,varin)
!         status = nf90_close(ncid)
!      END IF   
!      !
!   END SUBROUTINE dtv_filter
!
!
!
!   REAL(8) FUNCTION Get_Mean ( jpi , jpj, tabvar, tmask )
!      !!---------------------------------------------------------------------
!      !!                    ***  FUNCTION Get_Mean  ***
!      !!                   
!      !! ** Purpose : get the mean of the input field 
!      !!
!      !!----------------------------------------------------------------------
!      INTEGER,   INTENT(in)      :: jpi,jpj
!      REAL(8),   INTENT(in)      :: tabvar(jpi,jpj)
!      REAL(8),   INTENT(in)      :: tmask (jpi,jpj)
!      INTEGER                    :: ji,jj,jn    
!      
!      !! Compute the mean first
!      jn       = 0
!      Get_Mean = 0.
!      DO jj = 1, jpj
!         DO ji = 1, jpi
!            IF(tmask(ji,jj) .gt. 0.5) THEN
!               Get_Mean = Get_Mean + tabvar( ji,jj )  
!               jn=jn+1
!            END IF  
!         END DO
!      END DO      
!      Get_Mean   = (1./jn)   * Get_Mean  
!      !
!   END FUNCTION Get_Mean
!      
!
!
!
!
!   REAL(8) FUNCTION Get_Vari ( jpi , jpj, tabvar, tmask, mean )
!      !!---------------------------------------------------------------------
!      !!                    ***  FUNCTION Get_Mean  ***
!      !!                   
!      !! ** Purpose : get the mean of the input field 
!      !!
!      !!----------------------------------------------------------------------
!      INTEGER,   INTENT(in)      :: jpi,jpj
!      REAL(8),   INTENT(in)      :: tabvar(jpi,jpj)
!      REAL(8),   INTENT(in)      :: tmask (jpi,jpj), mean
!      INTEGER                    :: ji,jj,jn    
!      
!      jn       = 0
!      Get_Vari = 0
!      !!
!      DO jj = 1, jpj
!         DO ji = 1, jpi
!            IF(tmask(ji,jj) .gt. 0.5) THEN
!              Get_Vari = Get_Vari + ( tabvar(ji,jj) - mean )**2 
!              jn = jn+1 
!            ENDIF  
!         END DO
!      END DO 
!      Get_Vari   = (1./jn)   * Get_Vari 
!      !
!   END FUNCTION Get_Vari


   
END MODULE module_grid
