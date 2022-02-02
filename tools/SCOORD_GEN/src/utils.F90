MODULE utils

   USE netcdf

   IMPLICIT NONE
   PUBLIC             ! allows the acces to par_oce when dom_oce is used
   !                  ! exception to coding rules... to be suppressed ???

!   PUBLIC dom_oce_alloc

   INTEGER, PARAMETER   :: dp=8 , sp=4, wp=dp

   !! All coordinates
   !! ---------------
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   gdep3w_0           !: depth of t-points (sum of e3w) (m)
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   gdept_0, gdepw_0   !: analytical (time invariant) depth at t-w  points (m)
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   gphit              !: latitude at t points
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   e3v_0  , e3f_0     !: analytical (time invariant) vertical scale factors at  v-f
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   e3t_0  , e3u_0     !:                                      t-u  points (m)
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   e3vw_0             !: analytical (time invariant) vertical scale factors at  vw
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   e3w_0  , e3uw_0    !:                                      w-uw points (m)

   !! s-coordinate and hybrid z-s-coordinate
   !! =----------------======---------------
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   hbatv , hbatf      !: ocean depth at the vertical of  v--f
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   hbatt , hbatu      !:                                 t--u points (m)
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   scosrf, scobot     !: ocean surface and bottom topographies 
   !                                                                           !  (if deviating from coordinate surfaces in HYBRID)
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   hifv  , hiff       !: interface depth between stretching at v--f
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   hift  , hifu       !: and quasi-uniform spacing             t--u points (m)
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   rx1                !: Maximum grid stiffness ratio

   !!----------------------------------------------------------------------
   !! masks, bathymetry
   !! ---------------------------------------------------------------------
   INTEGER , PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   mbathy             !: number of ocean level (=0, 1, ... , jpk-1)
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   bathy              !: ocean depth (meters - read from file)

   !! Other variables needed by scoord_gen
   INTEGER  ::   jpi, jpj, jpk            ! Size of the domain - read from bathy or namelist?
   INTEGER  ::   ji, jj, jk, jl           ! dummy loop argument
   INTEGER  ::   iip1, ijp1, iim1, ijm1   ! temporary integers
   INTEGER  ::   ios                      ! Local integer output status for namelist read and allocation
   INTEGER,PARAMETER  ::   numnam=8       ! File handle for namelist 
   REAL(wp) ::   zrmax, ztaper   ! temporary scalars
   REAL(wp) ::   zrfact
   !
   REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: ztmpi1, ztmpi2, ztmpj1, ztmpj2
   REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: zenv, ztmp, zmsk, zri, zrj, zhbat

   !Namelist variables
   REAL(wp) :: rn_jpk, rn_sbot_min, rn_sbot_max, rn_hc, rn_rmax, rn_theta
   REAL(wp) :: rn_thetb, rn_bb, rn_alpha, rn_efold, rn_zs, rn_zb_a, rn_zb_b
   LOGICAL :: ln_s_sh94, ln_s_sf12, ln_sigcrit, ln_eq_taper
   CHARACTER(len=50) :: cn_coord_hgr

   NAMELIST/namzgr_sco/rn_jpk, ln_s_sh94, ln_s_sf12, ln_sigcrit, ln_eq_taper, & 
                  &      cn_coord_hgr, rn_sbot_min, rn_sbot_max, rn_hc, rn_rmax,rn_theta, &
                  &      rn_thetb, rn_bb, rn_alpha, rn_efold, rn_zs, rn_zb_a, rn_zb_b

  ! IDs for output netcdf file
  INTEGER :: id_x, id_y, id_z
  INTEGER :: ncout
  INTEGER, DIMENSION(20) :: var_ids  !Array to contain all variable IDs

   CONTAINS

   INTEGER FUNCTION dom_oce_alloc()
      !!----------------------------------------------------------------------
      INTEGER, DIMENSION(4) :: ierr
      !!----------------------------------------------------------------------
      ierr(:) = 0
      !
      ALLOCATE( zenv(jpi,jpj), ztmp(jpi,jpj), zmsk(jpi,jpj), zri(jpi,jpj), zrj(jpi,jpj), &
         &      zhbat(jpi,jpj) , ztmpi1(jpi,jpj), ztmpi2(jpi,jpj), ztmpj1(jpi,jpj), ztmpj2(jpi,jpj), STAT=ierr(1) )
         !
      ALLOCATE( gdep3w_0(jpi,jpj) , e3v_0(jpi,jpj) , e3f_0(jpi,jpj) ,                         &
         &      gdept_0(jpi,jpj) , e3t_0(jpi,jpj) , e3u_0 (jpi,jpj) ,                         &
         &      gdepw_0(jpi,jpj) , e3w_0(jpi,jpj) , e3vw_0(jpi,jpj) ,                         &
         &      gphit(jpi,jpj)   , e3uw_0(jpi,jpj) , STAT=ierr(2) )
         !
         !
      ALLOCATE( hbatv (jpi,jpj) , hbatf (jpi,jpj) ,     &
         &      hbatt (jpi,jpj) , hbatu (jpi,jpj) ,     &
         &      scosrf(jpi,jpj) , scobot(jpi,jpj) ,     &
         &      hifv  (jpi,jpj) , hiff  (jpi,jpj) ,     &
         &      hift  (jpi,jpj) , hifu  (jpi,jpj) , rx1 (jpi,jpj) , STAT=ierr(3) )

      ALLOCATE( mbathy(jpi,jpj) , STAT=ierr(4) )
     !
      dom_oce_alloc = MAXVAL(ierr)
      !
   END FUNCTION dom_oce_alloc
 

   SUBROUTINE read_bathy()
     !! Read bathymetry from input netcdf file
     INTEGER :: var_id, ncin

     CALL check_nf90( nf90_open('bathy_meter.nc', NF90_NOWRITE, ncin), 'Error opening bathy_meter.nc file' )

     ! Find the size of the input bathymetry
     CALL dimlen(ncin, 'lon', jpi)    
     CALL dimlen(ncin, 'lat', jpj)    
     
     ALLOCATE( bathy(jpi, jpj) )
     
     ! Read the bathymetry variable from file
     CALL check_nf90( nf90_inq_varid( ncin, 'Bathymetry', var_id ), 'Cannot get variable ID for bathymetry')
     CALL check_nf90( nf90_get_var( ncin, var_id, bathy, (/ 1,1 /), (/ jpi, jpj /) ) )

     CALL check_nf90( nf90_close(ncin), 'Error closing bathy.nc file' )

   END SUBROUTINE read_bathy

   SUBROUTINE read_gphit()
   !! Read gphit from horizontal coordinate file if required
     INTEGER :: var_id, ncin

     CALL check_nf90( nf90_open(cn_coord_hgr, NF90_NOWRITE, ncin), 'Error opening horizontal coordinate file' )

     ! Read gphit variable from file
     CALL check_nf90( nf90_inq_varid( ncin, 'gphit', var_id ), 'Cannot get variable ID for bathymetry')
     CALL check_nf90( nf90_get_var( ncin, var_id, gphit, (/ 1,1 /), (/ jpi, jpj /) ) )

     CALL check_nf90( nf90_close(ncin), 'Error closing horizontal coordinate file' )

   END SUBROUTINE read_gphit

   SUBROUTINE dimlen( ncid, dimname, len )
     ! Determine the length of dimension dimname
     INTEGER, INTENT(in)          :: ncid
     CHARACTER(LEN=*), INTENT(in) :: dimname
     INTEGER, INTENT(out)         :: len
     ! Local variables
     INTEGER :: id_var

     id_var = 1
     CALL check_nf90( nf90_inq_dimid(ncid, dimname, id_var), 'Dimension not found in file')
     CALL check_nf90( nf90_inquire_dimension(ncid,id_var,len=len))

   END SUBROUTINE dimlen
  
 
   SUBROUTINE make_coord_file()
     ! Create new coordinates file and define dimensions and variables ready for
     ! writing
     

     !Create the file
     CALL check_nf90( nf90_create('coord_zgr.nc', NF90_NETCDF4, ncout), 'Could not create output file')
     !
     !Define dimensions
     CALL check_nf90( nf90_def_dim(ncout, 'x', jpi, id_x) )
     CALL check_nf90( nf90_def_dim(ncout, 'y', jpj, id_y) )
     CALL check_nf90( nf90_def_dim(ncout, 'z', jpk, id_z) )
     !
     !Define variables - include all varibles that would be put into the mesh
     !mask file
     CALL check_nf90( nf90_def_var(ncout, 'gdept_0', nf90_double, (/id_x, id_y,id_z/), var_ids(1)) )
     CALL check_nf90( nf90_def_var(ncout, 'gdepw_0', nf90_double, (/id_x, id_y,id_z/), var_ids(2)) )
     CALL check_nf90( nf90_def_var(ncout, 'gdep3w_0', nf90_double, (/id_x, id_y,id_z/), var_ids(3)) )
     CALL check_nf90( nf90_def_var(ncout, 'e3f_0', nf90_double, (/id_x, id_y,id_z/), var_ids(4)) )
     CALL check_nf90( nf90_def_var(ncout, 'e3t_0', nf90_double, (/id_x, id_y,id_z/), var_ids(5)) )
     CALL check_nf90( nf90_def_var(ncout, 'e3u_0', nf90_double, (/id_x, id_y,id_z/), var_ids(6)) )
     CALL check_nf90( nf90_def_var(ncout, 'e3v_0', nf90_double, (/id_x, id_y,id_z/), var_ids(7)) )
     CALL check_nf90( nf90_def_var(ncout, 'e3w_0', nf90_double, (/id_x, id_y,id_z/), var_ids(8)) )
     CALL check_nf90( nf90_def_var(ncout, 'e3uw_0', nf90_double, (/id_x, id_y,id_z/), var_ids(9)) )
     CALL check_nf90( nf90_def_var(ncout, 'e3vw_0', nf90_double, (/id_x, id_y,id_z/), var_ids(10)) )
     ! 2D fields
     CALL check_nf90( nf90_def_var(ncout, 'mbathy', nf90_double, (/id_x, id_y/), var_ids(11)) )
     CALL check_nf90( nf90_def_var(ncout, 'hbatt', nf90_double, (/id_x, id_y/), var_ids(12)) )
     CALL check_nf90( nf90_def_var(ncout, 'hbatu', nf90_double, (/id_x, id_y/), var_ids(13)) )
     CALL check_nf90( nf90_def_var(ncout, 'hbatv', nf90_double, (/id_x, id_y/), var_ids(14)) )
     CALL check_nf90( nf90_def_var(ncout, 'hbatf', nf90_double, (/id_x, id_y/), var_ids(15)) )
     CALL check_nf90( nf90_def_var(ncout, 'rx1', nf90_double, (/id_x, id_y/), var_ids(16)) )

     
     ! End define mode
     CALL check_nf90( nf90_enddef(ncout) )
     
     WRITE(*,*) 'Opened coord_zgr.nc file and defined variables'

   END SUBROUTINE make_coord_file

   SUBROUTINE write_netcdf_2d_vars()

     CALL check_nf90( nf90_put_var(ncout, var_ids(11), mbathy, (/ 1,1 /), (/ jpi, jpj /) ) )
     CALL check_nf90( nf90_put_var(ncout, var_ids(12), hbatt, (/ 1,1 /), (/ jpi, jpj /) ) )
     CALL check_nf90( nf90_put_var(ncout, var_ids(13), hbatu, (/ 1,1 /), (/ jpi, jpj /) ) )
     CALL check_nf90( nf90_put_var(ncout, var_ids(14), hbatv, (/ 1,1 /), (/ jpi, jpj /) ) )
     CALL check_nf90( nf90_put_var(ncout, var_ids(15), hbatf, (/ 1,1 /), (/ jpi, jpj /) ) )
     CALL check_nf90( nf90_put_var(ncout, var_ids(16), rx1, (/ 1,1 /), (/ jpi, jpj /) ) )

   END SUBROUTINE write_netcdf_2d_vars

   SUBROUTINE write_netcdf_3d_vars(kk)
   ! Write  variables to the netcdf file at level kk
     INTEGER, INTENT(in) :: kk

     CALL check_nf90( nf90_put_var(ncout, var_ids(1), gdept_0, (/ 1,1,kk /), (/ jpi, jpj,1 /) ) )
     CALL check_nf90( nf90_put_var(ncout, var_ids(2), gdepw_0, (/ 1,1,kk /), (/ jpi, jpj,1 /) ) )
     CALL check_nf90( nf90_put_var(ncout, var_ids(3), gdep3w_0, (/ 1,1,kk /), (/ jpi, jpj,1 /) ) )
     CALL check_nf90( nf90_put_var(ncout, var_ids(4), e3f_0, (/ 1,1,kk /), (/ jpi, jpj,1 /) ) )
     CALL check_nf90( nf90_put_var(ncout, var_ids(5), e3t_0, (/ 1,1,kk /), (/ jpi, jpj,1 /) ) )
     CALL check_nf90( nf90_put_var(ncout, var_ids(6), e3u_0, (/ 1,1,kk /), (/ jpi, jpj,1 /) ) )
     CALL check_nf90( nf90_put_var(ncout, var_ids(7), e3v_0, (/ 1,1,kk /), (/ jpi, jpj,1 /) ) )
     CALL check_nf90( nf90_put_var(ncout, var_ids(8), e3w_0, (/ 1,1,kk /), (/ jpi, jpj,1 /) ) )
     CALL check_nf90( nf90_put_var(ncout, var_ids(9), e3uw_0, (/ 1,1,kk /), (/ jpi, jpj,1 /) ) )
     CALL check_nf90( nf90_put_var(ncout, var_ids(10), e3vw_0, (/ 1,1,kk /), (/ jpi, jpj,1 /) ) )

   END SUBROUTINE write_netcdf_3d_vars

   SUBROUTINE check_nf90( istat, message )
      !Check for netcdf errors
      INTEGER, INTENT(in) :: istat
      CHARACTER(LEN=*), INTENT(in), OPTIONAL :: message

      IF (istat /= nf90_noerr) THEN
         WRITE(*,*) 'ERROR! : '//TRIM(nf90_strerror(istat))
         IF ( PRESENT(message) ) THEN ; WRITE(*,*) message ; ENDIF
         STOP
      ENDIF

   END SUBROUTINE check_nf90
   FUNCTION fssig( pk ) RESULT( pf )
      !!----------------------------------------------------------------------
      !!                 ***  ROUTINE fssig ***
      !!       
      !! ** Purpose :   provide the analytical function in s-coordinate
      !!          
      !! ** Method  :   the function provide the non-dimensional position of
      !!                T and W (i.e. between 0 and 1)
      !!                T-points at integer values (between 1 and jpk)
      !!                W-points at integer values - 1/2 (between 0.5 and jpk-0.5)
      !!----------------------------------------------------------------------
!     USE utils, ONLY : wp,rn_theta,rn_thetb,jpk
      IMPLICIT NONE
      REAL(wp), INTENT(in) ::   pk   ! continuous "k" coordinate
      REAL(wp)             ::   pf   ! sigma value
      !!----------------------------------------------------------------------
      !
      pf =   (   TANH( rn_theta * ( -(pk-0.5) / REAL(jpk-1) + rn_thetb )  )   &
         &     - TANH( rn_thetb * rn_theta                                )  )   &
         & * (   COSH( rn_theta                           )                      &
         &     + COSH( rn_theta * ( 2. * rn_thetb - 1. ) )  )              &
         & / ( 2. * SINH( rn_theta ) )
      !
   END FUNCTION fssig

   FUNCTION fssig1( pk1, pbb ) RESULT( pf1 )
      !!----------------------------------------------------------------------
      !!                 ***  ROUTINE fssig1 ***
      !!
      !! ** Purpose :   provide the Song and Haidvogel version of the analytical function in s-coordinate
      !!
      !! ** Method  :   the function provides the non-dimensional position of
      !!                T and W (i.e. between 0 and 1)
      !!                T-points at integer values (between 1 and jpk)
      !!                W-points at integer values - 1/2 (between 0.5 and jpk-0.5)
      !!----------------------------------------------------------------------
!     USE utils, ONLY : wp, jpk, rn_theta
      IMPLICIT NONE
      REAL(wp), INTENT(in) ::   pk1   ! continuous "k" coordinate
      REAL(wp), INTENT(in) ::   pbb   ! Stretching coefficient
      REAL(wp)             ::   pf1   ! sigma value
      !!----------------------------------------------------------------------
      !
      IF ( rn_theta == 0 ) then      ! uniform sigma
         pf1 = - ( pk1 - 0.5 ) / REAL( jpk-1 )
      ELSE                        ! stretched sigma
         pf1 =   ( 1. - pbb ) * ( SINH( rn_theta*(-(pk1-0.5)/REAL(jpk-1)) ) ) / SINH( rn_theta )              &
            &  + pbb * (  (TANH( rn_theta*( (-(pk1-0.5)/REAL(jpk-1)) + 0.5) ) - TANH( 0.5 * rn_theta )  )  &
            &        / ( 2. * TANH( 0.5 * rn_theta ) )  )
      ENDIF
      !
   END FUNCTION fssig1

   FUNCTION fgamma( pk1, pzb, pzs, psmth) RESULT( p_gamma )
      !!----------------------------------------------------------------------
      !!                 ***  ROUTINE fgamma  ***
      !!
      !! ** Purpose :   provide analytical function for the s-coordinate
      !!
      !! ** Method  :   the function provides the non-dimensional position of
      !!                T and W (i.e. between 0 and 1)
      !!                T-points at integer values (between 1 and jpk)
      !!                W-points at integer values - 1/2 (between 0.5 and jpk-0.5)
      !!
      !!                This method allows the maintenance of fixed surface and or
      !!                bottom cell resolutions (cf. geopotential coordinates) 
      !!                within an analytically derived stretched S-coordinate framework.
      !!
      !! Reference  :   Siddorn and Furner, in prep
      !!----------------------------------------------------------------------
!     USE utils, ONLY : jpk,wp,rn_alpha
      IMPLICIT NONE
      REAL(wp), INTENT(in   ) ::   pk1           ! continuous "k" coordinate
      REAL(wp)                ::   p_gamma       ! stretched coordinate
      REAL(wp), INTENT(in   ) ::   pzb           ! Bottom box depth
      REAL(wp), INTENT(in   ) ::   pzs           ! surface box depth
      REAL(wp), INTENT(in   ) ::   psmth         ! Smoothing parameter
      REAL(wp)                ::   za1,za2,za3   ! local variables
      REAL(wp)                ::   zn1,zn2       ! local variables
      REAL(wp)                ::   za,zb,zx      ! local variables
      !!----------------------------------------------------------------------
      !

      zn1  =  1./(jpk-1.)
      zn2  =  1. -  zn1

      za1 = (rn_alpha+2.0)*zn1**(rn_alpha+1.0)-(rn_alpha+1.0)*zn1**(rn_alpha+2.0) 
      za2 = (rn_alpha+2.0)*zn2**(rn_alpha+1.0)-(rn_alpha+1.0)*zn2**(rn_alpha+2.0)
      za3 = (zn2**3.0 - za2)/( zn1**3.0 - za1)
     
      za = pzb - za3*(pzs-za1)-za2
      za = za/( zn2-0.5*(za2+zn2**2.0) - za3*(zn1-0.5*(za1+zn1**2.0) ) )
      zb = (pzs - za1 - za*( zn1-0.5*(za1+zn1**2.0 ) ) ) / (zn1**3.0 - za1)
      zx = 1.0-za/2.0-zb

      p_gamma = za*(pk1*(1.0-pk1/2.0))+zb*pk1**3.0 +  &
                  & zx*( (rn_alpha+2.0)*pk1**(rn_alpha+1.0)- &
                  &      (rn_alpha+1.0)*pk1**(rn_alpha+2.0) )
      p_gamma = p_gamma*psmth+pk1*(1.0-psmth)

      !
   END FUNCTION fgamma


END MODULE utils
