MODULE domwri
   !!======================================================================
   !!                       ***  MODULE domwri  ***
   !! Ocean initialization : write the ocean domain mesh file(s)
   !!======================================================================
   !! History :  OPA  ! 1997-02  (G. Madec)  Original code
   !!            8.1  ! 1999-11  (M. Imbard)  NetCDF FORMAT with IOIPSL
   !!   NEMO     1.0  ! 2002-08  (G. Madec)  F90 and several file
   !!            3.0  ! 2008-01  (S. Masson)  add dom_uniq 
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   dom_wri        : create and write mesh and mask file(s)
   !!   dom_uniq       : identify unique point of a grid (TUVF)
   !!   dom_stiff      : diagnose maximum grid stiffness/hydrostatic consistency (s-coordinate)
   !!----------------------------------------------------------------------
   USE dom_oce         ! ocean space and time domain
   USE in_out_manager  ! I/O manager
   USE iom             ! I/O library
   USE lbclnk          ! lateral boundary conditions - mpp exchanges
   USE lib_mpp         ! MPP library
   USE wrk_nemo        ! Memory allocation
   USE timing          ! Timing
   USE phycst

   IMPLICIT NONE
   PRIVATE

   PUBLIC   dom_wri              ! routine called by inidom.F90
   PUBLIC   dom_wri_coordinate   ! routine called by domhgr.F90
   PUBLIC   dom_stiff            ! routine called by inidom.F90

   !!----------------------------------------------------------------------
   !! NEMO/OPA 3.7 , NEMO Consortium (2014)
   !! $Id: vectopt_loop_substitute.h90 4990 2014-12-15 16:42:49Z timgraham $ 
   !! Software governed by the CeCILL licence (./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE dom_wri_coordinate
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE dom_wri_coordinate  ***
      !!                   
      !! ** Purpose :   Create the NetCDF file which contains all the
      !!              standard coordinate information plus the surface,
      !!              e1e2u and e1e2v. By doing so, those surface will
      !!              not be changed by the reduction of e1u or e2v scale 
      !!              factors in some straits. 
      !!                 NB: call just after the read of standard coordinate
      !!              and the reduction of scale factors in some straits
      !!
      !! ** output file :   coordinate_e1e2u_v.nc
      !!----------------------------------------------------------------------
      INTEGER           ::   inum0    ! temprary units for 'coordinate_e1e2u_v.nc' file
      CHARACTER(len=21) ::   clnam0   ! filename (mesh and mask informations)
      !                                   !  workspaces
      REAL(wp), POINTER, DIMENSION(:,:  ) :: zprt, zprw 
      REAL(wp), POINTER, DIMENSION(:,:,:) :: zdepu, zdepv
      !!----------------------------------------------------------------------
      !
      IF( nn_timing == 1 )  CALL timing_start('dom_wri_coordinate')
      !
      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*) 'dom_wri_coordinate : create NetCDF coordinate file'
      IF(lwp) WRITE(numout,*) '~~~~~~~~~~~~~~~~~~'
      
      clnam0 = 'coordinate_e1e2u_v'  ! filename (mesh and mask informations)
      
      !  create 'coordinate_e1e2u_v.nc' file
      ! ============================
      !
      CALL iom_open( TRIM(clnam0), inum0, ldwrt = .TRUE., kiolib = jprstlib )
      !
      !                                                         ! horizontal mesh (inum3)
      CALL iom_rstput( 0, 0, inum0, 'glamt', glamt, ktype = jp_r8 )     !    ! latitude
      CALL iom_rstput( 0, 0, inum0, 'glamu', glamu, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum0, 'glamv', glamv, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum0, 'glamf', glamf, ktype = jp_r8 )
      
      CALL iom_rstput( 0, 0, inum0, 'gphit', gphit, ktype = jp_r8 )     !    ! longitude
      CALL iom_rstput( 0, 0, inum0, 'gphiu', gphiu, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum0, 'gphiv', gphiv, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum0, 'gphif', gphif, ktype = jp_r8 )
      
      CALL iom_rstput( 0, 0, inum0, 'e1t', e1t, ktype = jp_r8 )         !    ! e1 scale factors
      CALL iom_rstput( 0, 0, inum0, 'e1u', e1u, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum0, 'e1v', e1v, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum0, 'e1f', e1f, ktype = jp_r8 )
      
      CALL iom_rstput( 0, 0, inum0, 'e2t', e2t, ktype = jp_r8 )         !    ! e2 scale factors
      CALL iom_rstput( 0, 0, inum0, 'e2u', e2u, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum0, 'e2v', e2v, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum0, 'e2f', e2f, ktype = jp_r8 )
      
      CALL iom_rstput( 0, 0, inum0, 'e1e2u', e1e2u, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum0, 'e1e2v', e1e2v, ktype = jp_r8 )

      CALL iom_close( inum0 )
      !
      IF( nn_timing == 1 )  CALL timing_stop('dom_wri_coordinate')
      !
   END SUBROUTINE dom_wri_coordinate


   SUBROUTINE dom_wri
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE dom_wri  ***
      !!                   
      !! ** Purpose :   Create the NetCDF file(s) which contain(s) all the
      !!      ocean domain informations (mesh and mask arrays). This (these)
      !!      file(s) is (are) used for visualisation (SAXO software) and
      !!      diagnostic computation.
      !!
      !! ** Method  :   Write in a file all the arrays generated in routines
      !!      domhgr, domzgr, and dommsk. Note: the file contain depends on
      !!      the vertical coord. used (z-coord, partial steps, s-coord)
      !!            MOD(nmsh, 3) = 1  :   'mesh_mask.nc' file
      !!                         = 2  :   'mesh.nc' and mask.nc' files
      !!                         = 0  :   'mesh_hgr.nc', 'mesh_zgr.nc' and
      !!                                  'mask.nc' files
      !!      For huge size domain, use option 2 or 3 depending on your 
      !!      vertical coordinate.
      !!
      !!      if     nmsh <= 3: write full 3D arrays for e3[tuvw] and gdep[tuvw]
      !!      if 3 < nmsh <= 6: write full 3D arrays for e3[tuvw] and 2D arrays 
      !!                        corresponding to the depth of the bottom t- and w-points
      !!      if 6 < nmsh <= 9: write 2D arrays corresponding to the depth and the
      !!                        thickness (e3[tw]_ps) of the bottom points 
      !!
      !! ** output file :   meshmask.nc  : domain size, horizontal grid-point position,
      !!                                   masks, depth and vertical scale factors
      !!----------------------------------------------------------------------
      !!
      INTEGER           ::   inum0    ! temprary units for 'mesh_mask.nc' file
      INTEGER           ::   inum1    ! temprary units for 'mesh.nc'      file
      INTEGER           ::   inum2    ! temprary units for 'mask.nc'      file
      INTEGER           ::   inum3    ! temprary units for 'mesh_hgr.nc'  file
      INTEGER           ::   inum4    ! temprary units for 'mesh_zgr.nc'  file
      CHARACTER(len=21) ::   clnam0   ! filename (mesh and mask informations)
      CHARACTER(len=21) ::   clnam1   ! filename (mesh informations)
      CHARACTER(len=21) ::   clnam2   ! filename (mask informations)
      CHARACTER(len=21) ::   clnam3   ! filename (horizontal mesh informations)
      CHARACTER(len=21) ::   clnam4   ! filename (vertical   mesh informations)
      INTEGER           ::   ji, jj, jk   ! dummy loop indices
      !                                   !  workspaces
      REAL(wp), POINTER, DIMENSION(:,:  ) :: zprt, zprw 
      REAL(wp), POINTER, DIMENSION(:,:,:) :: zdepu, zdepv
      !!----------------------------------------------------------------------
      !
      IF( nn_timing == 1 )  CALL timing_start('dom_wri')
      !
      CALL wrk_alloc( jpi, jpj, zprt, zprw )
      CALL wrk_alloc( jpi, jpj, jpk, zdepu, zdepv )
      !
      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*) 'dom_wri : create NetCDF mesh and mask information file(s)'
      IF(lwp) WRITE(numout,*) '~~~~~~~'
      
      clnam0 = 'mesh_mask'  ! filename (mesh and mask informations)
      clnam1 = 'mesh'       ! filename (mesh informations)
      clnam2 = 'mask'       ! filename (mask informations)
      clnam3 = 'mesh_hgr'   ! filename (horizontal mesh informations)
      clnam4 = 'mesh_zgr'   ! filename (vertical   mesh informations)
      
      SELECT CASE ( MOD(nmsh, 3) )
         !                                  ! ============================
      CASE ( 1 )                            !  create 'mesh_mask.nc' file
         !                                  ! ============================
         CALL iom_open( TRIM(clnam0), inum0, ldwrt = .TRUE., kiolib = jprstlib )
         inum2 = inum0                                            ! put all the informations
         inum3 = inum0                                            ! in unit inum0
         inum4 = inum0
         
         !                                  ! ============================
      CASE ( 2 )                            !  create 'mesh.nc' and 
         !                                  !         'mask.nc' files
         !                                  ! ============================
         CALL iom_open( TRIM(clnam1), inum1, ldwrt = .TRUE., kiolib = jprstlib )
         CALL iom_open( TRIM(clnam2), inum2, ldwrt = .TRUE., kiolib = jprstlib )
         inum3 = inum1                                            ! put mesh informations 
         inum4 = inum1                                            ! in unit inum1 
         !                                  ! ============================
      CASE ( 0 )                            !  create 'mesh_hgr.nc'
         !                                  !         'mesh_zgr.nc' and
         !                                  !         'mask.nc'     files
         !                                  ! ============================
         CALL iom_open( TRIM(clnam2), inum2, ldwrt = .TRUE., kiolib = jprstlib )
         CALL iom_open( TRIM(clnam3), inum3, ldwrt = .TRUE., kiolib = jprstlib )
         CALL iom_open( TRIM(clnam4), inum4, ldwrt = .TRUE., kiolib = jprstlib )
         !
      END SELECT
      
      !                                                         ! masks (inum2) 
      CALL iom_rstput( 0, 0, inum2, 'tmask', tmask, ktype = jp_i1 )     !    ! land-sea mask
      CALL iom_rstput( 0, 0, inum2, 'umask', umask, ktype = jp_i1 )
      CALL iom_rstput( 0, 0, inum2, 'vmask', vmask, ktype = jp_i1 )
      CALL iom_rstput( 0, 0, inum2, 'fmask', fmask, ktype = jp_i1 )
      
      CALL dom_uniq( zprw, 'T' )
      DO jj = 1, jpj
         DO ji = 1, jpi
            jk=mikt(ji,jj) 
            zprt(ji,jj) = tmask(ji,jj,jk) * zprw(ji,jj)                        !    ! unique point mask
         END DO
      END DO                             !    ! unique point mask
      CALL iom_rstput( 0, 0, inum2, 'tmaskutil', zprt, ktype = jp_i1 )  
      CALL dom_uniq( zprw, 'U' )
      DO jj = 1, jpj
         DO ji = 1, jpi
            jk=miku(ji,jj) 
            zprt(ji,jj) = umask(ji,jj,jk) * zprw(ji,jj)                        !    ! unique point mask
         END DO
      END DO
      CALL iom_rstput( 0, 0, inum2, 'umaskutil', zprt, ktype = jp_i1 )  
      CALL dom_uniq( zprw, 'V' )
      DO jj = 1, jpj
         DO ji = 1, jpi
            jk=mikv(ji,jj) 
            zprt(ji,jj) = vmask(ji,jj,jk) * zprw(ji,jj)                        !    ! unique point mask
         END DO
      END DO
      CALL iom_rstput( 0, 0, inum2, 'vmaskutil', zprt, ktype = jp_i1 )  
      CALL dom_uniq( zprw, 'F' )
      DO jj = 1, jpj
         DO ji = 1, jpi
            jk=mikf(ji,jj) 
            zprt(ji,jj) = fmask(ji,jj,jk) * zprw(ji,jj)                        !    ! unique point mask
         END DO
      END DO
      CALL iom_rstput( 0, 0, inum2, 'fmaskutil', zprt, ktype = jp_i1 )  

      !                                                         ! horizontal mesh (inum3)
      CALL iom_rstput( 0, 0, inum3, 'glamt', glamt, ktype = jp_r8 )     !    ! latitude
      CALL iom_rstput( 0, 0, inum3, 'glamu', glamu, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum3, 'glamv', glamv, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum3, 'glamf', glamf, ktype = jp_r8 )
      
      CALL iom_rstput( 0, 0, inum3, 'gphit', gphit, ktype = jp_r8 )     !    ! longitude
      CALL iom_rstput( 0, 0, inum3, 'gphiu', gphiu, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum3, 'gphiv', gphiv, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum3, 'gphif', gphif, ktype = jp_r8 )
      
      CALL iom_rstput( 0, 0, inum3, 'e1t', e1t, ktype = jp_r8 )         !    ! e1 scale factors
      CALL iom_rstput( 0, 0, inum3, 'e1u', e1u, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum3, 'e1v', e1v, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum3, 'e1f', e1f, ktype = jp_r8 )
      
      CALL iom_rstput( 0, 0, inum3, 'e2t', e2t, ktype = jp_r8 )         !    ! e2 scale factors
      CALL iom_rstput( 0, 0, inum3, 'e2u', e2u, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum3, 'e2v', e2v, ktype = jp_r8 )
      CALL iom_rstput( 0, 0, inum3, 'e2f', e2f, ktype = jp_r8 )
      
      CALL iom_rstput( 0, 0, inum3, 'ff_f', ff_f, ktype = jp_r8 )           !    ! coriolis factor
      CALL iom_rstput( 0, 0, inum3, 'ff_t', ff_t, ktype = jp_r8 )           !    ! coriolis factor
      
      ! note that mbkt is set to 1 over land ==> use surface tmask
      zprt(:,:) = ssmask(:,:) * REAL( mbkt(:,:) , wp )
      CALL iom_rstput( 0, 0, inum4, 'mbathy', zprt, ktype = jp_i2 )     !    ! nb of ocean T-points
      zprt(:,:) = ssmask(:,:) * REAL( mikt(:,:) , wp )
      CALL iom_rstput( 0, 0, inum4, 'misf', zprt, ktype = jp_i2 )       !    ! nb of ocean T-points
      zprt(:,:) = ssmask(:,:) * REAL( risfdep(:,:) , wp )
      CALL iom_rstput( 0, 0, inum4, 'isfdraft', zprt, ktype = jp_r8 )       !    ! nb of ocean T-points
            
      IF( ln_sco ) THEN                                         ! s-coordinate
         CALL iom_rstput( 0, 0, inum4, 'hbatt', hbatt )
         CALL iom_rstput( 0, 0, inum4, 'hbatu', hbatu )
         CALL iom_rstput( 0, 0, inum4, 'hbatv', hbatv )
         CALL iom_rstput( 0, 0, inum4, 'hbatf', hbatf )
         !
         CALL iom_rstput( 0, 0, inum4, 'gsigt', gsigt )         !    ! scaling coef.
         CALL iom_rstput( 0, 0, inum4, 'gsigw', gsigw )  
         CALL iom_rstput( 0, 0, inum4, 'gsi3w', gsi3w )
         CALL iom_rstput( 0, 0, inum4, 'esigt', esigt )
         CALL iom_rstput( 0, 0, inum4, 'esigw', esigw )
         !
         CALL iom_rstput( 0, 0, inum4, 'e3t_0', e3t_0 )         !    ! scale factors
         CALL iom_rstput( 0, 0, inum4, 'e3u_0', e3u_0 )
         CALL iom_rstput( 0, 0, inum4, 'e3v_0', e3v_0 )
         CALL iom_rstput( 0, 0, inum4, 'e3w_0', e3w_0 )
         !
         CALL iom_rstput( 0, 0, inum4, 'gdept_1d' , gdept_1d )  !    ! stretched system
         CALL iom_rstput( 0, 0, inum4, 'gdepw_1d' , gdepw_1d )
         CALL iom_rstput( 0, 0, inum4, 'gdept_0', gdept_0, ktype = jp_r8 )     
         CALL iom_rstput( 0, 0, inum4, 'gdepw_0', gdepw_0, ktype = jp_r8 )
         CALL dom_stiff( zprt )
         CALL iom_rstput( 0, 0, inum4, 'stiffness', zprt )       !    ! Max. grid stiffness ratio
      ENDIF
      
      IF( ln_zps ) THEN                                         ! z-coordinate - partial steps
         !
         IF( nmsh <= 6 ) THEN                                   !    ! 3D vertical scale factors
            CALL iom_rstput( 0, 0, inum4, 'e3t_0', e3t_0 )         
            CALL iom_rstput( 0, 0, inum4, 'e3u_0', e3u_0 )
            CALL iom_rstput( 0, 0, inum4, 'e3v_0', e3v_0 )
            CALL iom_rstput( 0, 0, inum4, 'e3w_0', e3w_0 )
         ELSE                                                   !    ! 2D masked bottom ocean scale factors
            DO jj = 1,jpj   
               DO ji = 1,jpi
                  e3tp(ji,jj) = e3t_0(ji,jj,mbkt(ji,jj)) * ssmask(ji,jj)
                  e3wp(ji,jj) = e3w_0(ji,jj,mbkt(ji,jj)) * ssmask(ji,jj)
               END DO
            END DO
            CALL iom_rstput( 0, 0, inum4, 'e3t_ps', e3tp )      
            CALL iom_rstput( 0, 0, inum4, 'e3w_ps', e3wp )
         END IF
         !
         IF( nmsh <= 3 ) THEN                                   !    ! 3D depth
            CALL iom_rstput( 0, 0, inum4, 'gdept_0', gdept_0, ktype = jp_r8 )     
            DO jk = 1,jpk   
               DO jj = 1, jpjm1   
                  DO ji = 1, jpim1   ! vector opt.
                     zdepu(ji,jj,jk) = MIN( gdept_0(ji,jj,jk) , gdept_0(ji+1,jj  ,jk) )
                     zdepv(ji,jj,jk) = MIN( gdept_0(ji,jj,jk) , gdept_0(ji  ,jj+1,jk) )
                  END DO   
               END DO   
            END DO
            CALL lbc_lnk( zdepu, 'U', 1. )   ;   CALL lbc_lnk( zdepv, 'V', 1. ) 
            CALL iom_rstput( 0, 0, inum4, 'gdepu', zdepu, ktype = jp_r8 )
            CALL iom_rstput( 0, 0, inum4, 'gdepv', zdepv, ktype = jp_r8 )
            CALL iom_rstput( 0, 0, inum4, 'gdepw_0', gdepw_0, ktype = jp_r8 )
         ELSE                                                   !    ! 2D bottom depth
            DO jj = 1,jpj   
               DO ji = 1,jpi
                  zprt(ji,jj) = gdept_0(ji,jj,mbkt(ji,jj)  ) * ssmask(ji,jj)
                  zprw(ji,jj) = gdepw_0(ji,jj,mbkt(ji,jj)+1) * ssmask(ji,jj)
               END DO
            END DO
            CALL iom_rstput( 0, 0, inum4, 'hdept', zprt, ktype = jp_r8 )     
            CALL iom_rstput( 0, 0, inum4, 'hdepw', zprw, ktype = jp_r8 ) 
         ENDIF
         !
         CALL iom_rstput( 0, 0, inum4, 'gdept_1d', gdept_1d )   !    ! reference z-coord.
         CALL iom_rstput( 0, 0, inum4, 'gdepw_1d', gdepw_1d )
         CALL iom_rstput( 0, 0, inum4, 'e3t_1d'  , e3t_1d   )
         CALL iom_rstput( 0, 0, inum4, 'e3w_1d'  , e3w_1d   )
      ENDIF
      
      IF( ln_zco ) THEN
         !                                                      ! z-coordinate - full steps
         CALL iom_rstput( 0, 0, inum4, 'gdept_1d', gdept_1d )   !    ! depth
         CALL iom_rstput( 0, 0, inum4, 'gdepw_1d', gdepw_1d )
         CALL iom_rstput( 0, 0, inum4, 'e3t_1d'  , e3t_1d   )   !    ! scale factors
         CALL iom_rstput( 0, 0, inum4, 'e3w_1d'  , e3w_1d   )
      ENDIF
      !                                     ! ============================
      !                                     !        close the files 
      !                                     ! ============================
      SELECT CASE ( MOD(nmsh, 3) )
      CASE ( 1 )                
         CALL iom_close( inum0 )
      CASE ( 2 )
         CALL iom_close( inum1 )
         CALL iom_close( inum2 )
      CASE ( 0 )
         CALL iom_close( inum2 )
         CALL iom_close( inum3 )
         CALL iom_close( inum4 )
      END SELECT
      !
      CALL wrk_dealloc( jpi, jpj, zprt, zprw )
      CALL wrk_dealloc( jpi, jpj, jpk, zdepu, zdepv )
      !
      IF( nn_timing == 1 )  CALL timing_stop('dom_wri')
      !
   END SUBROUTINE dom_wri


   SUBROUTINE dom_uniq( puniq, cdgrd )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE dom_uniq  ***
      !!                   
      !! ** Purpose :   identify unique point of a grid (TUVF)
      !!
      !! ** Method  :   1) aplly lbc_lnk on an array with different values for each element
      !!                2) check which elements have been changed
      !!----------------------------------------------------------------------
      !
      CHARACTER(len=1)        , INTENT(in   ) ::   cdgrd   ! 
      REAL(wp), DIMENSION(:,:), INTENT(inout) ::   puniq   ! 
      !
      REAL(wp) ::  zshift   ! shift value link to the process number
      INTEGER  ::  ji       ! dummy loop indices
      LOGICAL, DIMENSION(SIZE(puniq,1),SIZE(puniq,2),1) ::  lldbl  ! store whether each point is unique or not
      REAL(wp), POINTER, DIMENSION(:,:) :: ztstref
      !!----------------------------------------------------------------------
      !
      IF( nn_timing == 1 )  CALL timing_start('dom_uniq')
      !
      CALL wrk_alloc( jpi, jpj, ztstref )
      !
      ! build an array with different values for each element 
      ! in mpp: make sure that these values are different even between process
      ! -> apply a shift value according to the process number
      zshift = jpi * jpj * ( narea - 1 )
      ztstref(:,:) = RESHAPE( (/ (zshift + REAL(ji,wp), ji = 1, jpi*jpj) /), (/ jpi, jpj /) )
      !
      puniq(:,:) = ztstref(:,:)                   ! default definition
      CALL lbc_lnk( puniq, cdgrd, 1. )            ! apply boundary conditions
      lldbl(:,:,1) = puniq(:,:) == ztstref(:,:)   ! check which values have been changed 
      !
      puniq(:,:) = 1.                             ! default definition
      ! fill only the inner part of the cpu with llbl converted into real 
      puniq(nldi:nlei,nldj:nlej) = REAL( COUNT( lldbl(nldi:nlei,nldj:nlej,:), dim = 3 ) , wp )
      !
      CALL wrk_dealloc( jpi, jpj, ztstref )
      !
      IF( nn_timing == 1 )  CALL timing_stop('dom_uniq')
      !
   END SUBROUTINE dom_uniq


   SUBROUTINE dom_stiff( px1 )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE dom_stiff  ***
      !!                     
      !! ** Purpose :   Diagnose maximum grid stiffness/hydrostatic consistency
      !!
      !! ** Method  :   Compute Haney (1991) hydrostatic condition ratio
      !!                Save the maximum in the vertical direction
      !!                (this number is only relevant in s-coordinates)
      !!
      !!                Haney, 1991, J. Phys. Oceanogr., 21, 610-619.
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(:,:), INTENT(out), OPTIONAL ::   px1   ! stiffness
      !
      INTEGER  ::   ji, jj, jk 
      REAL(wp) ::   zrxmax
      REAL(wp), DIMENSION(4) ::   zr1
      REAL(wp), DIMENSION(jpi,jpj) ::   zx1
      !!----------------------------------------------------------------------
      zx1(:,:) = 0._wp
      zrxmax   = 0._wp
      zr1(:)   = 0._wp
      !
      DO ji = 2, jpim1
         DO jj = 2, jpjm1
            DO jk = 1, jpkm1
!!gm   remark: dk(gdepw) = e3t   ===>>>  possible simplification of the following calculation....
!!             especially since it is gde3w which is used to compute the pressure gradient
!!             furthermore, I think gdept_0 should be used below instead of w point in the numerator
!!             so that the ratio is computed at the same point (i.e. uw and vw) ....
               zr1(1) = ABS(  ( gdepw_0(ji  ,jj,jk  )-gdepw_0(ji-1,jj,jk  )               & 
                    &          +gdepw_0(ji  ,jj,jk+1)-gdepw_0(ji-1,jj,jk+1) )             &
                    &       / ( gdepw_0(ji  ,jj,jk  )+gdepw_0(ji-1,jj,jk  )               &
                    &          -gdepw_0(ji  ,jj,jk+1)-gdepw_0(ji-1,jj,jk+1) + rsmall )  ) * umask(ji-1,jj,jk)
               zr1(2) = ABS(  ( gdepw_0(ji+1,jj,jk  )-gdepw_0(ji  ,jj,jk  )               &
                    &          +gdepw_0(ji+1,jj,jk+1)-gdepw_0(ji  ,jj,jk+1) )             &
                    &       / ( gdepw_0(ji+1,jj,jk  )+gdepw_0(ji  ,jj,jk  )               &
                    &          -gdepw_0(ji+1,jj,jk+1)-gdepw_0(ji  ,jj,jk+1) + rsmall )  ) * umask(ji  ,jj,jk)
               zr1(3) = ABS(  ( gdepw_0(ji,jj+1,jk  )-gdepw_0(ji,jj  ,jk  )               &
                    &          +gdepw_0(ji,jj+1,jk+1)-gdepw_0(ji,jj  ,jk+1) )             &
                    &       / ( gdepw_0(ji,jj+1,jk  )+gdepw_0(ji,jj  ,jk  )               &
                    &          -gdepw_0(ji,jj+1,jk+1)-gdepw_0(ji,jj  ,jk+1) + rsmall )  ) * vmask(ji,jj  ,jk)
               zr1(4) = ABS(  ( gdepw_0(ji,jj  ,jk  )-gdepw_0(ji,jj-1,jk  )               &
                    &          +gdepw_0(ji,jj  ,jk+1)-gdepw_0(ji,jj-1,jk+1) )             &
                    &       / ( gdepw_0(ji,jj  ,jk  )+gdepw_0(ji,jj-1,jk  )               &
                    &          -gdepw_0(ji,jj  ,jk+1)-gdepw_0(ji,jj-1,jk+1) + rsmall )  ) * vmask(ji,jj-1,jk)
               zrxmax = MAXVAL( zr1(1:4) )
               zx1(ji,jj) = MAX( zx1(ji,jj) , zrxmax )
            END DO
         END DO
      END DO
      CALL lbc_lnk( zx1, 'T', 1. )
      !
      IF( PRESENT( px1 ) )    px1 = zx1
      !
      zrxmax = MAXVAL( zx1 )
      !
      IF( lk_mpp )   CALL mpp_max( zrxmax ) ! max over the global domain
      !
      IF(lwp) THEN
         WRITE(numout,*)
         WRITE(numout,*) 'dom_stiff : maximum grid stiffness ratio: ', zrxmax
         WRITE(numout,*) '~~~~~~~~~'
      ENDIF
      !
   END SUBROUTINE dom_stiff

   !!======================================================================
END MODULE domwri
