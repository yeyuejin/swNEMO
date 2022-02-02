!************************************************************************
! Fortran 95 OPA Nesting tools						*
!									*
!     Copyright (C) 2005 Florian Lemarié (Florian.Lemarie@imag.fr)	*
!									*
!************************************************************************
!
MODULE agrif_readwrite
  !
  USE agrif_types
  !
  IMPLICIT NONE
  !
CONTAINS
  !
  !************************************************************************
  ! 									*
  ! MODULE  AGRIF_READWRITE						*
  !									*
  ! module containing subroutine used for : 				*
  !   - Coordinates files reading/writing				*
  !   - Bathymetry files reading/writing (meter and levels)		*
  !   - Naming of child grid files					*
  !									*
  !************************************************************************
  !       
  !*****************************************************
  !   function Read_Coordinates(name,Grid)
  !*****************************************************

  INTEGER FUNCTION Read_Coordinates(name,Grid,Pacifique)
    !
    USE io_netcdf
    !    
    !  file name to open
    ! 
    CHARACTER(*) name
    LOGICAL,OPTIONAL :: Pacifique
    ! 
    TYPE(Coordinates) :: Grid
    !      
    CALL read_ncdf_var('glamt',name,Grid%glamt)
    CALL read_ncdf_var('glamu',name,Grid%glamu)
    CALL read_ncdf_var('glamv',name,Grid%glamv)
    CALL read_ncdf_var('glamf',name,Grid%glamf)
    CALL read_ncdf_var('gphit',name,Grid%gphit)
    CALL read_ncdf_var('gphiu',name,Grid%gphiu)
    CALL read_ncdf_var('gphiv',name,Grid%gphiv)
    CALL read_ncdf_var('gphif',name,Grid%gphif)
    CALL read_ncdf_var('e1t',name,Grid%e1t)
    CALL read_ncdf_var('e1u',name,Grid%e1u)
    CALL read_ncdf_var('e1v',name,Grid%e1v)
    CALL read_ncdf_var('e1f',name,Grid%e1f)
    CALL read_ncdf_var('e2t',name,Grid%e2t)
    CALL read_ncdf_var('e2u',name,Grid%e2u)
    CALL read_ncdf_var('e2v',name,Grid%e2v)
    CALL read_ncdf_var('e2f',name,Grid%e2f)
    CALL read_ncdf_var('nav_lon',name,Grid%nav_lon)
    CALL read_ncdf_var('nav_lat',name,Grid%nav_lat)       
    ! 
    IF( PRESENT(Pacifique) )THEN
       IF ( Grid%glamt(1,1) > Grid%glamt(nxfin,nyfin) ) THEN           
       Pacifique = .TRUE.
       WHERE ( Grid%glamt < 0 )
          Grid%glamt = Grid%glamt + 360.
       END WHERE
       WHERE ( Grid%glamf < 0 )
          Grid%glamf = Grid%glamf + 360.
       END WHERE
       WHERE ( Grid%glamu < 0 )
          Grid%glamu = Grid%glamu + 360.
       END WHERE
       WHERE ( Grid%glamv < 0 )
          Grid%glamv = Grid%glamv + 360.
       END WHERE
       WHERE ( Grid%nav_lon < 0 )
          Grid%nav_lon = Grid%nav_lon + 360.
       END WHERE
       ENDIF
    ENDIF
    !            
    WRITE(*,*) ' '
    WRITE(*,*) 'Reading coordinates file: ',name
    WRITE(*,*) ' '
    !      
    Read_Coordinates = 1
    !      
  END FUNCTION Read_Coordinates

  !*****************************************************
  !   function Read_Coordinates(name,Grid)
  !*****************************************************

  INTEGER FUNCTION Read_Local_Coordinates(name,Grid,strt,cnt)
    !
    USE io_netcdf
    !    
    !  file name to open
    ! 
    CHARACTER(*) name
    INTEGER, DIMENSION(2) :: strt,cnt
    ! 
    TYPE(Coordinates) :: Grid
    !      
    CALL read_ncdf_var('glamt',name,Grid%glamt,strt,cnt)
    CALL read_ncdf_var('glamu',name,Grid%glamu,strt,cnt)
    CALL read_ncdf_var('glamv',name,Grid%glamv,strt,cnt)
    CALL read_ncdf_var('glamf',name,Grid%glamf,strt,cnt)
    CALL read_ncdf_var('gphit',name,Grid%gphit,strt,cnt)
    CALL read_ncdf_var('gphiu',name,Grid%gphiu,strt,cnt)
    CALL read_ncdf_var('gphiv',name,Grid%gphiv,strt,cnt)
    CALL read_ncdf_var('gphif',name,Grid%gphif,strt,cnt)
    CALL read_ncdf_var('e1t',name,Grid%e1t,strt,cnt)
    CALL read_ncdf_var('e1u',name,Grid%e1u,strt,cnt)
    CALL read_ncdf_var('e1v',name,Grid%e1v,strt,cnt)
    CALL read_ncdf_var('e1f',name,Grid%e1f,strt,cnt)
    CALL read_ncdf_var('e2t',name,Grid%e2t,strt,cnt)
    CALL read_ncdf_var('e2u',name,Grid%e2u,strt,cnt)
    CALL read_ncdf_var('e2v',name,Grid%e2v,strt,cnt)
    CALL read_ncdf_var('e2f',name,Grid%e2f,strt,cnt)
    CALL read_ncdf_var('nav_lon',name,Grid%nav_lon,strt,cnt)
    CALL read_ncdf_var('nav_lat',name,Grid%nav_lat,strt,cnt)       
    !
    WRITE(*,*) ' '
    WRITE(*,*) 'Reading coordinates file: ',name
    WRITE(*,*) ' '
    !      
    Read_Local_Coordinates = 1
    !      
  END FUNCTION Read_Local_Coordinates

  !*****************************************************
  !   function Write_Coordinates(name,Grid)
  !*****************************************************

  INTEGER FUNCTION Write_Coordinates(name,Grid)
    !
    USE io_netcdf
    CHARACTER(*) name
    TYPE(Coordinates) :: Grid
    INTEGER :: status,ncid
    CHARACTER(len=1),DIMENSION(2) :: dimnames
    !
    status = nf90_create(name,NF90_WRITE,ncid)
    status = nf90_close(ncid)
    !           
    dimnames = (/ 'x','y' /)
    CALL write_ncdf_dim(dimnames(1),name,nxfin)
    CALL write_ncdf_dim(dimnames(2),name,nyfin)
    !      
    CALL write_ncdf_var('nav_lon',dimnames,name,Grid%nav_lon,'float')      
    CALL write_ncdf_var('nav_lat',dimnames,name,Grid%nav_lat,'float')
    !
    CALL write_ncdf_var('glamt',dimnames,name,Grid%glamt,'double')
    CALL write_ncdf_var('glamu',dimnames,name,Grid%glamu,'double')
    CALL write_ncdf_var('glamv',dimnames,name,Grid%glamv,'double')
    CALL write_ncdf_var('glamf',dimnames,name,Grid%glamf,'double')
    CALL write_ncdf_var('gphit',dimnames,name,Grid%gphit,'double')
    CALL write_ncdf_var('gphiu',dimnames,name,Grid%gphiu,'double')
    CALL write_ncdf_var('gphiv',dimnames,name,Grid%gphiv,'double')
    CALL write_ncdf_var('gphif',dimnames,name,Grid%gphif,'double')      
    CALL write_ncdf_var('e1t',dimnames,name,Grid%e1t,'double')      
    CALL write_ncdf_var('e1u',dimnames,name,Grid%e1u,'double')     
    CALL write_ncdf_var('e1v',dimnames,name,Grid%e1v,'double')      
    CALL write_ncdf_var('e1f',dimnames,name,Grid%e1f,'double')
    CALL write_ncdf_var('e2t',dimnames,name,Grid%e2t,'double')
    CALL write_ncdf_var('e2u',dimnames,name,Grid%e2u,'double')
    CALL write_ncdf_var('e2v',dimnames,name,Grid%e2v,'double')
    CALL write_ncdf_var('e2f',dimnames,name,Grid%e2f,'double')
    !      
    CALL copy_ncdf_att('nav_lon',TRIM(parent_coordinate_file),name,MINVAL(Grid%nav_lon),MAXVAL(Grid%nav_lon))
    CALL copy_ncdf_att('nav_lat',TRIM(parent_coordinate_file),name,MINVAL(Grid%nav_lat),MAXVAL(Grid%nav_lat))
    CALL copy_ncdf_att('glamt',TRIM(parent_coordinate_file),name)
    CALL copy_ncdf_att('glamu',TRIM(parent_coordinate_file),name)
    CALL copy_ncdf_att('glamv',TRIM(parent_coordinate_file),name)
    CALL copy_ncdf_att('glamf',TRIM(parent_coordinate_file),name)
    CALL copy_ncdf_att('gphit',TRIM(parent_coordinate_file),name)
    CALL copy_ncdf_att('gphiu',TRIM(parent_coordinate_file),name)
    CALL copy_ncdf_att('gphiv',TRIM(parent_coordinate_file),name)
    CALL copy_ncdf_att('gphif',TRIM(parent_coordinate_file),name)
    CALL copy_ncdf_att('e1t',TRIM(parent_coordinate_file),name)
    CALL copy_ncdf_att('e1u',TRIM(parent_coordinate_file),name)
    CALL copy_ncdf_att('e1v',TRIM(parent_coordinate_file),name)
    CALL copy_ncdf_att('e1f',TRIM(parent_coordinate_file),name)
    CALL copy_ncdf_att('e2t',TRIM(parent_coordinate_file),name)
    CALL copy_ncdf_att('e2u',TRIM(parent_coordinate_file),name)
    CALL copy_ncdf_att('e2v',TRIM(parent_coordinate_file),name)
    CALL copy_ncdf_att('e2f',TRIM(parent_coordinate_file),name)            
    !
    WRITE(*,*) ' '
    WRITE(*,*) 'Writing coordinates file: ',name
    WRITE(*,*) ' '
    !
    Write_Coordinates = 1
    !      
  END FUNCTION Write_Coordinates
  !
  !
  !
  !*****************************************************
  !   function Read_Bathy_level(name,Grid)
  !*****************************************************
  !
  INTEGER FUNCTION Read_Bathy_level(name,Grid)
    !
    USE io_netcdf
    !      
    CHARACTER(*) name
    TYPE(Coordinates) :: Grid
    !
    CALL read_ncdf_var(parent_level_name,name,Grid%Bathy_level)    
    !
    WRITE(*,*) ' '
    WRITE(*,*) 'Reading bathymetry file: ',name
    WRITE(*,*) ' '      
    !
    Read_Bathy_level = 1
    !      
  END FUNCTION Read_Bathy_level
  !
  !*****************************************************
  !   function Write_Bathy_level(name,Grid)
  !*****************************************************
  !
  INTEGER FUNCTION Write_Bathy_level(name,Grid)
    !
    USE io_netcdf
    !      
    CHARACTER(*) name
    TYPE(Coordinates) :: Grid
    INTEGER :: status,ncid
    CHARACTER(len=1),DIMENSION(2) :: dimnames
    !
    status = nf90_create(name,NF90_WRITE,ncid)
    status = nf90_close(ncid)          
    !
    dimnames = (/ 'x','y' /)
    CALL write_ncdf_dim(dimnames(1),name,nxfin)
    CALL write_ncdf_dim(dimnames(2),name,nyfin)
    !      
    CALL write_ncdf_var('nav_lon'        ,dimnames,name,Grid%nav_lon    ,'float')
    CALL write_ncdf_var('nav_lat'        ,dimnames,name,Grid%nav_lat    ,'float')
    CALL write_ncdf_var(parent_level_name,dimnames,name,NINT(Grid%bathy_level),'integer')
    !
    CALL copy_ncdf_att('nav_lon'        ,TRIM(parent_bathy_level),name,MINVAL(Grid%nav_lon),MAXVAL(Grid%nav_lon))
    CALL copy_ncdf_att('nav_lat'        ,TRIM(parent_bathy_level),name,MINVAL(Grid%nav_lat),MAXVAL(Grid%nav_lat))
    CALL copy_ncdf_att(parent_level_name,TRIM(parent_bathy_level),name)       
    !
    WRITE(*,*) ' '
    WRITE(*,*) 'Writing bathymetry (levels) in: ',name
    WRITE(*,*) ' '
    !
    Write_Bathy_level = 1
    !
  END FUNCTION Write_Bathy_level
  !
  !*****************************************************
  !   function read_bathy_coord(name,CoarseGrid,ChildGrid)
  !*****************************************************
  !
  INTEGER FUNCTION read_bathy_coord(name,CoarseGrid,ChildGrid,Pacifique)
    !
    USE io_netcdf
    CHARACTER(*) name
    INTEGER :: i,j,tabdim1,tabdim2
    INTEGER, DIMENSION(1) :: i_min,i_max,j_min,j_max
    REAL*8,POINTER,DIMENSION(:) :: topo_lon,topo_lat
    INTEGER :: status,ncid,varid 
    LOGICAL,OPTIONAL :: Pacifique
    TYPE(Coordinates) :: CoarseGrid,ChildGrid
    REAL*8 :: zdel
    zdel = 0.5 ! Offset in degrees to extend extraction of bathymetry data
    !
    IF( Dims_Existence('lon',name) .AND. Dims_Existence('lat',name) ) THEN
       WRITE(*,*) '****'
       WRITE(*,*) ' etopo format for external high resolution database  '
       WRITE(*,*) '****'
       CALL read_ncdf_var('lon',name,topo_lon)
       CALL read_ncdf_var('lat',name,topo_lat)
    ELSE IF( Dims_Existence('x',name) .AND. Dims_Existence('y',name) ) THEN
       WRITE(*,*) '****'
       WRITE(*,*) ' OPA format for external high resolution database  '
       WRITE(*,*) '****'
       CALL read_ncdf_var('nav_lon',name,CoarseGrid%nav_lon)
       CALL read_ncdf_var('nav_lat',name,CoarseGrid%nav_lat)
       CALL read_ncdf_var(parent_meter_name,name,CoarseGrid%Bathy_meter)
       !            
       IF ( PRESENT(Pacifique) ) THEN
          IF(Pacifique) THEN
             WHERE(CoarseGrid%nav_lon < 0.001) 
                CoarseGrid%nav_lon = CoarseGrid%nav_lon + 360.
             END WHERE
          ENDIF
       ENDIF
       !      
       read_bathy_coord = 1
       RETURN      
    ELSE
       WRITE(*,*) '****'
       WRITE(*,*) '*** ERROR Bad format for external high resolution database'
       WRITE(*,*) '****'
       STOP  
    ENDIF
    !
    IF( MAXVAL(ChildGrid%glamt) > 180. ) THEN                 
       !          
       WHERE( topo_lon < 0. )   topo_lon = topo_lon + 360.
       !          
       i_min = MAXLOC(topo_lon,mask = topo_lon < MINVAL(ChildGrid%nav_lon)-zdel)
       i_max = MINLOC(topo_lon,mask = topo_lon > MAXVAL(ChildGrid%nav_lon)+zdel)                    
       j_min = MAXLOC(topo_lat,mask = topo_lat < MINVAL(ChildGrid%nav_lat)-zdel)
       j_max = MINLOC(topo_lat,mask = topo_lat > MAXVAL(ChildGrid%nav_lat)+zdel)
       !          
       tabdim1 = ( SIZE(topo_lon) - i_min(1) + 1 ) + i_max(1)                    
       !          
       IF( ln_agrif_domain ) THEN
          IF(j_min(1)-2 >= 1 .AND. j_max(1)+3 <= SIZE(topo_lat,1) ) THEN
             j_min(1) = j_min(1)-2
             j_max(1) = j_max(1)+3
          ENDIF
       ENDIF
       tabdim2 = j_max(1) - j_min(1) + 1
       !
       ALLOCATE(CoarseGrid%nav_lon(tabdim1,tabdim2))
       ALLOCATE(CoarseGrid%nav_lat(tabdim1,tabdim2))
       ALLOCATE(CoarseGrid%Bathy_meter(tabdim1,tabdim2))          
       !
       DO i = 1,tabdim1
          CoarseGrid%nav_lat(i,:) = topo_lat(j_min(1):j_max(1))
       END DO
       !
       DO j = 1, tabdim2
          !                     
          CoarseGrid%nav_lon(1:SIZE(topo_lon)-i_min(1)+1      ,j) = topo_lon(i_min(1):SIZE(topo_lon))
          CoarseGrid%nav_lon(2+SIZE(topo_lon)-i_min(1):tabdim1,j) = topo_lon(1:i_max(1))
          !    
       END DO
       status = nf90_open(name,NF90_NOWRITE,ncid)
       status = nf90_inq_varid(ncid,elevation_name,varid)
       !
       IF (status/=nf90_noerr) THEN
          WRITE(*,*)"Can't find variable: ", elevation_name
          STOP
       ENDIF
       !          
       status=nf90_get_var(ncid,varid,CoarseGrid%Bathy_meter(1:SIZE(topo_lon)-i_min(1)+1,:), &
            start=(/i_min(1),j_min(1)/),count=(/SIZE(topo_lon)-i_min(1),tabdim2/))

       status=nf90_get_var(ncid,varid,CoarseGrid%Bathy_meter(2+SIZE(topo_lon)-i_min(1):tabdim1,:), &
            start=(/1,j_min(1)/),count=(/i_max(1),tabdim2/))                
       !
    ELSE
       !
       WHERE( topo_lon > 180. )   topo_lon = topo_lon - 360.
       !
       i_min = MAXLOC(topo_lon,mask = topo_lon < MINVAL(ChildGrid%nav_lon)-zdel)
       i_max = MINLOC(topo_lon,mask = topo_lon > MAXVAL(ChildGrid%nav_lon)+zdel)
       j_min = MAXLOC(topo_lat,mask = topo_lat < MINVAL(ChildGrid%nav_lat)-zdel)
       j_max = MINLOC(topo_lat,mask = topo_lat > MAXVAL(ChildGrid%nav_lat)+zdel)
       !      
       IF( ln_agrif_domain ) THEN
          IF(i_min(1)-2 >= 1 .AND. i_max(1)+3 <= SIZE(topo_lon,1) ) THEN
             i_min(1) = i_min(1)-2
             i_max(1) = i_max(1)+3
          ENDIF
       ENDIF
       tabdim1 = i_max(1) - i_min(1) + 1
       !
       IF( ln_agrif_domain ) THEN
          IF(j_min(1)-2 >= 1 .AND. j_max(1)+3 <= SIZE(topo_lat,1) ) THEN
             j_min(1) = j_min(1)-2
             j_max(1) = j_max(1)+3
          ENDIF
       ENDIF
       tabdim2 = j_max(1) - j_min(1) + 1
       !
       WRITE(*,*) ' '
       WRITE(*,*) 'Reading bathymetry file: ',name
       WRITE(*,*) ' '
       !
       ALLOCATE(CoarseGrid%nav_lon(tabdim1,tabdim2))
       ALLOCATE(CoarseGrid%nav_lat(tabdim1,tabdim2))
       ALLOCATE(CoarseGrid%Bathy_meter(tabdim1,tabdim2))
       !
       DO j = 1,tabdim2
          CoarseGrid%nav_lon(:,j) = topo_lon(i_min(1):i_max(1))
       END DO
       !      
       DO i = 1,tabdim1
          CoarseGrid%nav_lat(i,:) = topo_lat(j_min(1):j_max(1)) 
       END DO
       !
       status = nf90_open(name,NF90_NOWRITE,ncid)
       status = nf90_inq_varid(ncid,elevation_name,varid)
       IF (status/=nf90_noerr) THEN
          WRITE(*,*)"Can't find variable: ", elevation_name
          STOP
       ENDIF

       status = nf90_get_var(ncid,varid,CoarseGrid%Bathy_meter, &
          &                  start=(/i_min(1),j_min(1)/),count=(/tabdim1,tabdim2/))
       !
    ENDIF
    !
    status = nf90_close(ncid)     
    !
    WHERE(CoarseGrid%Bathy_meter.GE.0)
       CoarseGrid%Bathy_meter = 0.0
    END WHERE
    !      
    CoarseGrid%Bathy_meter(:,:) = -1.0 * CoarseGrid%Bathy_meter(:,:)
    !      
    read_bathy_coord = 1
    RETURN
    !      
  END FUNCTION read_bathy_coord
  ! 
  !
  !*****************************************************
  !   function read_bathy_meter(name,CoarseGrid,ChildGrid)
  !*****************************************************
  !
  INTEGER FUNCTION read_bathy_meter(name,Grid)
    !
    !
    USE io_netcdf
    !      
    CHARACTER(*) name
    TYPE(Coordinates) :: Grid
    !
    CALL read_ncdf_var(parent_meter_name,name,Grid%Bathy_meter)    
    !
    WRITE(*,*) ' '
    WRITE(*,*) 'Reading bathymetry file: ',name
    WRITE(*,*) ' '      
    !
    read_bathy_meter = 1
    !      
  END FUNCTION read_bathy_meter
  !     
  !*****************************************************
  !   function Write_Bathy_meter(name,Grid)
  !*****************************************************
  !
  INTEGER FUNCTION Write_Bathy_meter(name,Grid)
    !
    USE io_netcdf
    !
    CHARACTER(*) name
    TYPE(Coordinates) :: Grid
    INTEGER :: status,ncid
    CHARACTER(len=1),DIMENSION(2) :: dimnames
    INTEGER :: nx,ny
    !
    status = nf90_create(name,NF90_WRITE,ncid)
    status = nf90_close(ncid)     
    !
    nx = SIZE(Grid%bathy_meter,1)
    ny = SIZE(Grid%bathy_meter,2)
    dimnames = (/ 'x','y' /)

    CALL write_ncdf_dim(dimnames(1),name,nx)
    CALL write_ncdf_dim(dimnames(2),name,ny)
    !     
    CALL write_ncdf_var('nav_lon'        ,dimnames,name,Grid%nav_lon    ,'float')
    CALL write_ncdf_var('nav_lat'        ,dimnames,name,Grid%nav_lat    ,'float')
    CALL write_ncdf_var(parent_meter_name,dimnames,name,Grid%bathy_meter,'float')
    CALL write_ncdf_var('weight'         ,dimnames,name,Grid%wgt        ,'float')
    !
    CALL copy_ncdf_att('nav_lon'        ,TRIM(parent_bathy_meter),name,MINVAL(Grid%nav_lon),MAXVAL(Grid%nav_lon))
    CALL copy_ncdf_att('nav_lat'        ,TRIM(parent_bathy_meter),name,MINVAL(Grid%nav_lat),MAXVAL(Grid%nav_lat))  
    CALL copy_ncdf_att(parent_meter_name,TRIM(parent_bathy_meter),name)
    !
    WRITE(*,*) ' '
    WRITE(*,*) 'Writing bathymetry (meters) in: ',name
    WRITE(*,*) ' '
    !
    Write_Bathy_meter = 1
    !      
  END FUNCTION Write_Bathy_meter
  !
  !*****************************************************
  !   function write_domcfg(name,Grid)
  !*****************************************************

  INTEGER FUNCTION write_domcfg(name,Grid)
     !-----------------------------------------
     ! It creates a domain_cfg.nc used in NEMO4
     !-----------------------------------------    
     !
     USE io_netcdf
     !
     CHARACTER(*) name
     TYPE(Coordinates) :: Grid
     !
     INTEGER :: status, ncid
     INTEGER :: nx, ny, jk
     INTEGER :: ln_sco, ln_isfcav, ln_zco, ln_zps, jperio
     REAL*8  :: rpi, rad, rday, rsiyea, rsiday, omega
     !
     CHARACTER(len=1),     DIMENSION(3)     ::   dimnames
     REAL*8 ,              DIMENSION(N)     ::   e3t_1d, e3w_1d, gdept_1d, gdepw_1d
     REAL*8 , ALLOCATABLE, DIMENSION(:,:)   ::   ff_t, ff_f, zbathy
     INTEGER, ALLOCATABLE, DIMENSION(:,:)   ::   bottom_level, top_level, mbathy
     REAL*8 , ALLOCATABLE, DIMENSION(:,:,:) ::   e3t_0, e3u_0, e3v_0, e3f_0, e3w_0, e3uw_0, e3vw_0
     !
     ! size of the Grid
     nx = SIZE(Grid%bathy_meter,1)
     ny = SIZE(Grid%bathy_meter,2)

     ! allocate needed arrays for domain_cfg
     ALLOCATE( ff_t(nx,ny), ff_f(nx,ny), zbathy(nx,ny) )
     ALLOCATE( bottom_level(nx,ny), top_level(nx,ny), mbathy(nx,ny) )
     ALLOCATE( e3t_0(nx,ny,N), e3u_0 (nx,ny,N), e3v_0 (nx,ny,N), e3f_0(nx,ny,N),  &
        &      e3w_0(nx,ny,N), e3uw_0(nx,ny,N), e3vw_0(nx,ny,N) )

     ! some physical parameters
     rpi    = 3.141592653589793
     rad    = 3.141592653589793 / 180.
     rday   = 24.*60.*60.
     rsiyea = 365.25 * rday * 2. * rpi / 6.283076
     rsiday = rday / ( 1. + rday / rsiyea )
     omega  = 2. * rpi / rsiday

     ! Coriolis
     ff_f(:,:) = 2. * omega * SIN( rad * Grid%gphif(:,:) )     ! compute it on the sphere at f-point
     ff_t(:,:) = 2. * omega * SIN( rad * Grid%gphit(:,:) )     !    -        -       -    at t-point

     ! bathy in meters
     zbathy(:,:) = Grid%bathy_meter

     ! vertical scale factors
     CALL zgr_z( e3t_1d, e3w_1d, gdept_1d, gdepw_1d )    
!     DO jk = 1, N
!        e3t_0  (:,:,jk) = e3t_1d  (jk)
!        e3u_0  (:,:,jk) = e3t_1d  (jk)
!        e3v_0  (:,:,jk) = e3t_1d  (jk)
!        e3f_0  (:,:,jk) = e3t_1d  (jk)
!        e3w_0  (:,:,jk) = e3w_1d  (jk)
!        e3uw_0 (:,:,jk) = e3w_1d  (jk)
!        e3vw_0 (:,:,jk) = e3w_1d  (jk)
!     END DO

     ! logicals and others
     ln_sco = 0
     ln_isfcav = 0
!!$     IF( partial_steps ) THEN
        ln_zps = 1
        ln_zco = 0
!!$     ELSE
!!$        ln_zps = 0
!!$        ln_zco = 1
!!$     ENDIF

     CALL zgr_zps( nx, ny, gdept_1d, gdepw_1d, e3t_1d, e3w_1d, zbathy, &        ! input
        &          mbathy, e3t_0, e3u_0, e3v_0, e3f_0, e3w_0, e3uw_0, e3vw_0 )  ! output

     ! top/bottom levels
     bottom_level(:,:) = mbathy(:,:)
     top_level(:,:)    = MIN( 1, mbathy(:,:) )

     ! closed domain for child grid and namelist defined domain for parent grid
     IF( TRIM(name) == TRIM(parent_domcfg_updated) .OR. TRIM(name) == TRIM(parent_domcfg_out) ) THEN
        jperio = parent_jperio
     ELSE
        jperio = 0
     ENDIF

     !IF( .NOT.ln_agrif_domain ) THEN
     !   bottom_level(1:nx,1   ) = 0
     !   bottom_level(1:nx,  ny) = 0
     !   bottom_level(1   ,1:ny) = 0
     !   bottom_level(  nx,1:ny) = 0

     !   top_level(1:nx,1   ) = 0
     !   top_level(1:nx,  ny) = 0
     !   top_level(1   ,1:ny) = 0
     !   top_level(  nx,1:ny) = 0

     !   zbathy(1:nx,1   ) = 0.
     !   zbathy(1:nx,  ny) = 0.
     !   zbathy(1   ,1:ny) = 0.
     !   zbathy(  nx,1:ny) = 0.     
     !ENDIF
     
     ! -------------------
     ! write domain_cfg.nc
     ! -------------------
     status = nf90_create(name,IOR(NF90_64BIT_OFFSET,NF90_WRITE),ncid)
     status = nf90_close(ncid)
     !
     ! dimensions
     dimnames = (/'x','y','z'/)
     CALL write_ncdf_dim(dimnames(1),name,nx)
     CALL write_ncdf_dim(dimnames(2),name,ny)
     CALL write_ncdf_dim(dimnames(3),name,N)
     !      
     ! variables
     CALL write_ncdf_var('nav_lon',dimnames(1:2),name,Grid%nav_lon,'float')      
     CALL write_ncdf_var('nav_lat',dimnames(1:2),name,Grid%nav_lat,'float')
     CALL write_ncdf_var('nav_lev',dimnames(3)  ,name,gdept_1d    ,'float')
     !
     CALL write_ncdf_var('jpiglo',name,nx    ,'integer')
     CALL write_ncdf_var('jpjglo',name,ny    ,'integer')
     CALL write_ncdf_var('jpkglo',name,N     ,'integer')
     CALL write_ncdf_var('jperio',name,jperio,'integer')
     !
     CALL write_ncdf_var('ln_zco'   ,name,ln_zco   ,'integer')
     CALL write_ncdf_var('ln_zps'   ,name,ln_zps   ,'integer')
     CALL write_ncdf_var('ln_sco'   ,name,ln_sco   ,'integer')
     CALL write_ncdf_var('ln_isfcav',name,ln_isfcav,'integer')

     CALL write_ncdf_var('glamt',dimnames(1:2),name,Grid%glamt,'double')
     CALL write_ncdf_var('glamu',dimnames(1:2),name,Grid%glamu,'double')
     CALL write_ncdf_var('glamv',dimnames(1:2),name,Grid%glamv,'double')
     CALL write_ncdf_var('glamf',dimnames(1:2),name,Grid%glamf,'double')
     CALL write_ncdf_var('gphit',dimnames(1:2),name,Grid%gphit,'double')
     CALL write_ncdf_var('gphiu',dimnames(1:2),name,Grid%gphiu,'double')
     CALL write_ncdf_var('gphiv',dimnames(1:2),name,Grid%gphiv,'double')
     CALL write_ncdf_var('gphif',dimnames(1:2),name,Grid%gphif,'double')      

     CALL write_ncdf_var('e1t',dimnames(1:2),name,Grid%e1t,'double')      
     CALL write_ncdf_var('e1u',dimnames(1:2),name,Grid%e1u,'double')     
     CALL write_ncdf_var('e1v',dimnames(1:2),name,Grid%e1v,'double')      
     CALL write_ncdf_var('e1f',dimnames(1:2),name,Grid%e1f,'double')
     CALL write_ncdf_var('e2t',dimnames(1:2),name,Grid%e2t,'double')
     CALL write_ncdf_var('e2u',dimnames(1:2),name,Grid%e2u,'double')
     CALL write_ncdf_var('e2v',dimnames(1:2),name,Grid%e2v,'double')
     CALL write_ncdf_var('e2f',dimnames(1:2),name,Grid%e2f,'double')

     CALL write_ncdf_var('ff_f',dimnames(1:2),name,ff_f,'double')
     CALL write_ncdf_var('ff_t',dimnames(1:2),name,ff_t,'double')

     CALL write_ncdf_var('e3t_1d',dimnames(3),name,e3t_1d,'double')
     CALL write_ncdf_var('e3w_1d',dimnames(3),name,e3w_1d,'double')

     CALL write_ncdf_var('e3t_0' ,dimnames(:),name,e3t_0 ,'double')
     CALL write_ncdf_var('e3w_0' ,dimnames(:),name,e3w_0 ,'double')
     CALL write_ncdf_var('e3u_0' ,dimnames(:),name,e3u_0 ,'double')
     CALL write_ncdf_var('e3v_0' ,dimnames(:),name,e3v_0 ,'double')
     CALL write_ncdf_var('e3f_0' ,dimnames(:),name,e3f_0 ,'double')
     CALL write_ncdf_var('e3w_0' ,dimnames(:),name,e3w_0 ,'double')
     CALL write_ncdf_var('e3uw_0',dimnames(:),name,e3uw_0,'double')
     CALL write_ncdf_var('e3vw_0',dimnames(:),name,e3vw_0,'double')

     CALL write_ncdf_var('bottom_level',dimnames(1:2),name,bottom_level,'integer')
     CALL write_ncdf_var('top_level'   ,dimnames(1:2),name,top_level   ,'integer')
     CALL write_ncdf_var('bathy_meter' ,dimnames(1:2),name,zbathy      ,'float')

     ! some attributes      
     CALL copy_ncdf_att('nav_lon',TRIM(parent_coordinate_file),name,MINVAL(Grid%nav_lon),MAXVAL(Grid%nav_lon))
     CALL copy_ncdf_att('nav_lat',TRIM(parent_coordinate_file),name,MINVAL(Grid%nav_lat),MAXVAL(Grid%nav_lat))

     CALL copy_ncdf_att('glamt',TRIM(parent_coordinate_file),name)
     CALL copy_ncdf_att('glamu',TRIM(parent_coordinate_file),name)
     CALL copy_ncdf_att('glamv',TRIM(parent_coordinate_file),name)
     CALL copy_ncdf_att('glamf',TRIM(parent_coordinate_file),name)
     CALL copy_ncdf_att('gphit',TRIM(parent_coordinate_file),name)
     CALL copy_ncdf_att('gphiu',TRIM(parent_coordinate_file),name)
     CALL copy_ncdf_att('gphiv',TRIM(parent_coordinate_file),name)
     CALL copy_ncdf_att('gphif',TRIM(parent_coordinate_file),name)

     CALL copy_ncdf_att('e1t',TRIM(parent_coordinate_file),name)
     CALL copy_ncdf_att('e1u',TRIM(parent_coordinate_file),name)
     CALL copy_ncdf_att('e1v',TRIM(parent_coordinate_file),name)
     CALL copy_ncdf_att('e1f',TRIM(parent_coordinate_file),name)
     CALL copy_ncdf_att('e2t',TRIM(parent_coordinate_file),name)
     CALL copy_ncdf_att('e2u',TRIM(parent_coordinate_file),name)
     CALL copy_ncdf_att('e2v',TRIM(parent_coordinate_file),name)
     CALL copy_ncdf_att('e2f',TRIM(parent_coordinate_file),name)            
     !
     ! control print
     WRITE(*,*) ' '
     WRITE(*,*) 'Writing domcfg file: ',name
     WRITE(*,*) ' '
     !
     DEALLOCATE( ff_t, ff_f, zbathy )
     DEALLOCATE( bottom_level, top_level, mbathy )
     DEALLOCATE( e3t_0, e3u_0, e3v_0, e3f_0, e3w_0, e3uw_0, e3vw_0 )
     !      
     write_domcfg = 1

  END FUNCTION write_domcfg
  !
  SUBROUTINE zgr_z(e3t_1d, e3w_1d, gdept_1d, gdepw_1d )
     !!----------------------------------------------------------------------
     !!                   ***  ROUTINE zgr_z (from NEMO4) ***
     !!                   
     !! ** Purpose :   set the depth of model levels and the resulting 
     !!      vertical scale factors.
     !!
     !! ** Method  :   z-coordinate system (use in all type of coordinate)
     !!        The depth of model levels is defined from an analytical
     !!      function the derivative of which gives the scale factors.
     !!        both depth and scale factors only depend on k (1d arrays).
     !!              w-level: gdepw_1d  = gdep(k)
     !!                       e3w_1d(k) = dk(gdep)(k)     = e3(k)
     !!              t-level: gdept_1d  = gdep(k+0.5)
     !!                       e3t_1d(k) = dk(gdep)(k+0.5) = e3(k+0.5)
     !!
     !! ** Action  : - gdept_1d, gdepw_1d : depth of T- and W-point (m)
     !!              - e3t_1d  , e3w_1d   : scale factors at T- and W-levels (m)
     !!
     !! Reference : Marti, Madec & Delecluse, 1992, JGR, 97, No8, 12,763-12,766.
     !!----------------------------------------------------------------------
     INTEGER  ::   jk                   ! dummy loop indices
     REAL*8 ::   zt, zw                 ! temporary scalars
     REAL*8 ::   zsur, za0, za1, zkth   ! Values set from parameters in
     REAL*8 ::   zacr, zdzmin, zhmax    ! par_CONFIG_Rxx.h90
     REAL*8 ::   za2, zkth2, zacr2      ! Values for optional double tanh function set from parameters 

     REAL*8, DIMENSION(:), INTENT(out) ::  e3t_1d, e3w_1d, gdept_1d, gdepw_1d
     !!----------------------------------------------------------------------
     !
     !
     ! Set variables from parameters
     ! ------------------------------
     zkth = ppkth       ;   zacr = ppacr
     zdzmin = ppdzmin   ;   zhmax = pphmax
     zkth2 = ppkth2     ;   zacr2 = ppacr2   ! optional (ldbletanh=T) double tanh parameters

     ! If pa1 and pa0 and psur are et to pp_to_be_computed
     !  za0, za1, zsur are computed from ppdzmin , pphmax, ppkth, ppacr
    IF ( ( pa0 == 0 .OR. pa1 == 0 .OR. psur == 0 ) &
         .AND. ppdzmin.NE.0 .AND. pphmax.NE.0 ) THEN
        !
        za1  = (  ppdzmin - pphmax / FLOAT(N-1)  )                                                      &
           & / ( TANH((1-ppkth)/ppacr) - ppacr/FLOAT(N-1) * (  LOG( COSH( (N - ppkth) / ppacr) )      &
           &                                                   - LOG( COSH( ( 1  - ppkth) / ppacr) )  )  )
        za0  = ppdzmin - za1 *              TANH( (1-ppkth) / ppacr )
        zsur =   - za0 - za1 * ppacr * LOG( COSH( (1-ppkth) / ppacr )  )
     ELSE
        za1 = pa1 ;       za0 = pa0 ;          zsur = psur
        za2 = pa2                            ! optional (ldbletanh=T) double tanh parameter
     ENDIF

     ! Reference z-coordinate (depth - scale factor at T- and W-points)
     ! ======================
     IF( ppkth == 0. ) THEN            !  uniform vertical grid 

        za1 = zhmax / FLOAT(N-1)

        DO jk = 1, N
           zw = FLOAT( jk )
           zt = FLOAT( jk ) + 0.5
           gdepw_1d(jk) = ( zw - 1 ) * za1
           gdept_1d(jk) = ( zt - 1 ) * za1
           e3w_1d  (jk) =  za1
           e3t_1d  (jk) =  za1
        END DO
     ELSE                                ! Madec & Imbard 1996 function
        IF( .NOT. ldbletanh ) THEN
           DO jk = 1, N
              zw = REAL( jk )
              zt = REAL( jk ) + 0.5
              gdepw_1d(jk) = ( zsur + za0 * zw + za1 * zacr * LOG ( COSH( (zw-zkth) / zacr ) )  )
              gdept_1d(jk) = ( zsur + za0 * zt + za1 * zacr * LOG ( COSH( (zt-zkth) / zacr ) )  )
              e3w_1d  (jk) =          za0      + za1        * TANH(       (zw-zkth) / zacr   )
              e3t_1d  (jk) =          za0      + za1        * TANH(       (zt-zkth) / zacr   )
           END DO
        ELSE
           DO jk = 1, N
              zw = FLOAT( jk )
              zt = FLOAT( jk ) + 0.5
              ! Double tanh function
              gdepw_1d(jk) = ( zsur + za0 * zw + za1 * zacr * LOG ( COSH( (zw-zkth ) / zacr  ) )    &
                 &                             + za2 * zacr2* LOG ( COSH( (zw-zkth2) / zacr2 ) )  )
              gdept_1d(jk) = ( zsur + za0 * zt + za1 * zacr * LOG ( COSH( (zt-zkth ) / zacr  ) )    &
                 &                             + za2 * zacr2* LOG ( COSH( (zt-zkth2) / zacr2 ) )  )
              e3w_1d  (jk) =          za0      + za1        * TANH(       (zw-zkth ) / zacr  )      &
                 &                             + za2        * TANH(       (zw-zkth2) / zacr2 )
              e3t_1d  (jk) =          za0      + za1        * TANH(       (zt-zkth ) / zacr  )      &
                 &                             + za2        * TANH(       (zt-zkth2) / zacr2 )
           END DO
        ENDIF
        gdepw_1d(1) = 0.                    ! force first w-level to be exactly at zero
     ENDIF

     IF ( ln_e3_dep ) THEN      ! e3. = dk[gdep]   
        !
        !==>>>   need to be like this to compute the pressure gradient with ISF. 
        !        If not, level beneath the ISF are not aligned (sum(e3t) /= depth)
        !        define e3t_0 and e3w_0 as the differences between gdept and gdepw respectively
        !
        DO jk = 1, N-1
           e3t_1d(jk) = gdepw_1d(jk+1)-gdepw_1d(jk)
        END DO
        e3t_1d(N) = e3t_1d(N-1)   ! we don't care because this level is masked in NEMO

        DO jk = 2, N
           e3w_1d(jk) = gdept_1d(jk) - gdept_1d(jk-1)
        END DO
        e3w_1d(1  ) = 2. * (gdept_1d(1) - gdepw_1d(1))
     END IF

     !
  END SUBROUTINE zgr_z

  SUBROUTINE zgr_zps( nx, ny, gdept_1d, gdepw_1d, e3t_1d, e3w_1d, zbathy, &
     &                mbathy, e3t_0, e3u_0, e3v_0, e3f_0, e3w_0, e3uw_0, e3vw_0 )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE zgr_zps (from NEMO4)  ***
      !!                     
      !! ** Purpose :   the depth and vertical scale factor in partial step
      !!              reference z-coordinate case
      !!
      !! ** Method  :   Partial steps : computes the 3D vertical scale factors
      !!      of T-, U-, V-, W-, UW-, VW and F-points that are associated with
      !!      a partial step representation of bottom topography.
      !!
      !!        The reference depth of model levels is defined from an analytical
      !!      function the derivative of which gives the reference vertical
      !!      scale factors.
      !!        From  depth and scale factors reference, we compute there new value
      !!      with partial steps  on 3d arrays ( i, j, k ).
      !!
      !!              w-level: gdepw_0(i,j,k)  = gdep(k)
      !!                       e3w_0(i,j,k) = dk(gdep)(k)     = e3(i,j,k)
      !!              t-level: gdept_0(i,j,k)  = gdep(k+0.5)
      !!                       e3t_0(i,j,k) = dk(gdep)(k+0.5) = e3(i,j,k+0.5)
      !!
      !!        With the help of the bathymetric file ( bathymetry_depth_ORCA_R2.nc),
      !!      we find the mbathy index of the depth at each grid point.
      !!      This leads us to three cases:
      !!
      !!              - bathy = 0 => mbathy = 0
      !!              - 1 < mbathy < jpkm1    
      !!              - bathy > gdepw_0(jpk) => mbathy = jpkm1  
      !!
      !!        Then, for each case, we find the new depth at t- and w- levels
      !!      and the new vertical scale factors at t-, u-, v-, w-, uw-, vw- 
      !!      and f-points.
      !! 
      !!        This routine is given as an example, it must be modified
      !!      following the user s desiderata. nevertheless, the output as
      !!      well as the way to compute the model levels and scale factors
      !!      must be respected in order to insure second order accuracy
      !!      schemes.
      !!
      !!         c a u t i o n : gdept_1d, gdepw_1d and e3._1d are positives
      !!         - - - - - - -   gdept_0, gdepw_0 and e3. are positives
      !!      
      !!  Reference :   Pacanowsky & Gnanadesikan 1997, Mon. Wea. Rev., 126, 3248-3270.
      !!----------------------------------------------------------------------
      INTEGER  ::   ji, jj, jk       ! dummy loop indices
      INTEGER  ::   ik
      INTEGER  ::   ikb, ikt       ! temporary integers
      REAL*8 ::   ze3tp , ze3wp    ! Last ocean level thickness at T- and W-points
      REAL*8 ::   zdepwp, zdepth   ! Ajusted ocean depth to avoid too small e3t
      REAL*8 ::   zmax             ! temporary scalar
      !
      REAL*8, ALLOCATABLE, DIMENSION(:,:,:) ::  gdept_0, gdepw_0, zprt
      !
      INTEGER,                   INTENT(in   ) ::  nx, ny
      REAL*8 , DIMENSION(:)    , INTENT(in   ) ::  gdept_1d, gdepw_1d, e3t_1d, e3w_1d
      REAL*8 , DIMENSION(:,:)  , INTENT(inout) ::  zbathy
      INTEGER, DIMENSION(:,:)  , INTENT(  out) ::  mbathy
      REAL*8 , DIMENSION(:,:,:), INTENT(  out) ::  e3t_0, e3u_0, e3v_0, e3f_0, e3w_0, e3uw_0, e3vw_0
      !
      !!---------------------------------------------------------------------
      
      ALLOCATE( zprt(nx,ny,N), gdept_0(nx,ny,N), gdepw_0(nx,ny,N) )
      !
      ! bathymetry in level (from bathy_meter)
      ! ===================
      zmax = gdepw_1d(N) + e3t_1d(N)        ! maximum depth (i.e. the last ocean level thickness <= 2*e3t_1d(N-1) )
      zbathy(:,:) = MIN( zmax ,  zbathy(:,:) )    ! bounded value of bathy (min already set at the end of zgr_bat)
      WHERE( zbathy(:,:) == 0. )   ;   mbathy(:,:) = 0     ! land  : set mbathy to 0
      ELSEWHERE                    ;   mbathy(:,:) = N-1   ! ocean : initialize mbathy to the max ocean level
      END WHERE

!!$      IF( partial_steps ) THEN
         ! Compute mbathy for ocean points (i.e. the number of ocean levels)
         ! find the number of ocean levels such that the last level thickness
         ! is larger than the minimum of e3zps_min and e3zps_rat * e3t_1d (where
         ! e3t_1d is the reference level thickness
         DO jk = N-1, 1, -1
            zdepth = gdepw_1d(jk) + MIN( e3zps_min, e3t_1d(jk)*e3zps_rat )
            WHERE( 0. < zbathy(:,:) .AND. zbathy(:,:) <= zdepth )   mbathy(:,:) = jk-1
         END DO
!!$      ELSE
!!$         DO jk = 1, N
!!$            WHERE( 0. < zbathy(:,:) .AND. zbathy(:,:) >= gdepw_1d(jk) )   mbathy(:,:) = jk-1
!!$         END DO        
!!$      ENDIF
      
      ! Scale factors and depth at T- and W-points
      DO jk = 1, N                        ! intitialization to the reference z-coordinate
         gdept_0(:,:,jk) = gdept_1d(jk)
         gdepw_0(:,:,jk) = gdepw_1d(jk)
         e3t_0  (:,:,jk) = e3t_1d  (jk)
         e3w_0  (:,:,jk) = e3w_1d  (jk)
      END DO
      
      ! Bathy, iceshelf draft, scale factor and depth at T- and W- points in case of isf
      !!clem: not implemented yet
      !!      IF ( ln_isfcav == 1 ) CALL zgr_isf
      !
      ! Scale factors and depth at T- and W-points
      ! IF ( ln_isfcav == 0 ) THEN
      DO jj = 1, ny
         DO ji = 1, nx
            ik = mbathy(ji,jj)
            IF( ik > 0 ) THEN               ! ocean point only
               ! max ocean level case
               IF( ik == N-1 ) THEN
                  zdepwp = zbathy(ji,jj)
                  ze3tp  = zbathy(ji,jj) - gdepw_1d(ik)
                  ze3wp = 0.5 * e3w_1d(ik) * ( 1. + ( ze3tp/e3t_1d(ik) ) )
                  e3t_0(ji,jj,ik  ) = ze3tp
                  e3t_0(ji,jj,ik+1) = ze3tp
                  e3w_0(ji,jj,ik  ) = ze3wp
                  e3w_0(ji,jj,ik+1) = ze3tp
                  gdepw_0(ji,jj,ik+1) = zdepwp
                  gdept_0(ji,jj,ik  ) = gdept_1d(ik-1) + ze3wp
                  gdept_0(ji,jj,ik+1) = gdept_0(ji,jj,ik) + ze3tp
                  !
               ELSE                         ! standard case
                  IF( zbathy(ji,jj) <= gdepw_1d(ik+1) ) THEN  ;   gdepw_0(ji,jj,ik+1) = zbathy(ji,jj)
                  ELSE                                        ;   gdepw_0(ji,jj,ik+1) = gdepw_1d(ik+1)
                  ENDIF
                  !gm Bug?  check the gdepw_1d
                  !       ... on ik
                  gdept_0(ji,jj,ik) = gdepw_1d(ik) + ( gdepw_0(ji,jj,ik+1) - gdepw_1d(ik) )   &
                     &                             * ((gdept_1d(     ik  ) - gdepw_1d(ik) )   &
                     &                             / ( gdepw_1d(     ik+1) - gdepw_1d(ik) ))
                  e3t_0  (ji,jj,ik) = e3t_1d  (ik) * ( gdepw_0 (ji,jj,ik+1) - gdepw_1d(ik) )   & 
                     &                             / ( gdepw_1d(      ik+1) - gdepw_1d(ik) ) 
                  e3w_0(ji,jj,ik) = 0.5 * ( gdepw_0(ji,jj,ik+1) + gdepw_1d(ik+1) - 2. * gdepw_1d(ik) )   &
                     &                     * ( e3w_1d(ik) / ( gdepw_1d(ik+1) - gdepw_1d(ik) ) )
                  !       ... on ik+1
                  e3w_0  (ji,jj,ik+1) = e3t_0  (ji,jj,ik)
                  e3t_0  (ji,jj,ik+1) = e3t_0  (ji,jj,ik)
                  gdept_0(ji,jj,ik+1) = gdept_0(ji,jj,ik) + e3t_0(ji,jj,ik)
               ENDIF
            ENDIF
         END DO
      END DO
      !
      ! DO jj = 1, ny
      !    DO ji = 1, nx
      !       ik = mbathy(ji,jj)
      !       IF( ik > 0 ) THEN               ! ocean point only
      !          e3tp (ji,jj) = e3t_0(ji,jj,ik)
      !          e3wp (ji,jj) = e3w_0(ji,jj,ik)
      !       ENDIF
      !    END DO
      ! END DO
      ! END IF
      !
      ! Scale factors and depth at U-, V-, UW and VW-points
      DO jk = 1, N                        ! initialisation to z-scale factors
         e3u_0 (:,:,jk) = e3t_1d(jk)
         e3v_0 (:,:,jk) = e3t_1d(jk)
         e3uw_0(:,:,jk) = e3w_1d(jk)
         e3vw_0(:,:,jk) = e3w_1d(jk)
      END DO

      DO jk = 1,N                         ! Computed as the minimum of neighbooring scale factors
         DO jj = 1, ny-1
            DO ji = 1, nx-1   ! vector opt.
               e3u_0 (ji,jj,jk) = MIN( e3t_0(ji,jj,jk), e3t_0(ji+1,jj,jk) )
               e3v_0 (ji,jj,jk) = MIN( e3t_0(ji,jj,jk), e3t_0(ji,jj+1,jk) )
               e3uw_0(ji,jj,jk) = MIN( e3w_0(ji,jj,jk), e3w_0(ji+1,jj,jk) )
               e3vw_0(ji,jj,jk) = MIN( e3w_0(ji,jj,jk), e3w_0(ji,jj+1,jk) )
            END DO
         END DO
      END DO
      
!      IF ( ln_isfcav == 1 ) THEN
!      ! (ISF) define e3uw (adapted for 2 cells in the water column)
!         DO jj = 2, ny-1 
!            DO ji = 2, nx-1   ! vector opt. 
!               ikb = MAX(mbathy (ji,jj),mbathy (ji+1,jj))
!               ikt = MAX(misfdep(ji,jj),misfdep(ji+1,jj))
!               IF (ikb == ikt+1) e3uw_0(ji,jj,ikb) =  MIN( gdept_0(ji,jj,ikb  ), gdept_0(ji+1,jj  ,ikb  ) ) &
!                                       &            - MAX( gdept_0(ji,jj,ikb-1), gdept_0(ji+1,jj  ,ikb-1) )
!               ikb = MAX(mbathy (ji,jj),mbathy (ji,jj+1))
!               ikt = MAX(misfdep(ji,jj),misfdep(ji,jj+1))
!               IF (ikb == ikt+1) e3vw_0(ji,jj,ikb) =  MIN( gdept_0(ji,jj,ikb  ), gdept_0(ji  ,jj+1,ikb  ) ) &
!                                       &            - MAX( gdept_0(ji,jj,ikb-1), gdept_0(ji  ,jj+1,ikb-1) )
!            END DO
!         END DO
!      END IF
      !

      DO jk = 1, N                        ! set to z-scale factor if zero (i.e. along closed boundaries)
         WHERE( e3u_0 (:,:,jk) == 0. )   e3u_0 (:,:,jk) = e3t_1d(jk)
         WHERE( e3v_0 (:,:,jk) == 0. )   e3v_0 (:,:,jk) = e3t_1d(jk)
         WHERE( e3uw_0(:,:,jk) == 0. )   e3uw_0(:,:,jk) = e3w_1d(jk)
         WHERE( e3vw_0(:,:,jk) == 0. )   e3vw_0(:,:,jk) = e3w_1d(jk)
      END DO
      
      ! Scale factor at F-point
      DO jk = 1, N                        ! initialisation to z-scale factors
         e3f_0(:,:,jk) = e3t_1d(jk)
      END DO
      DO jk = 1, N                        ! Computed as the minimum of neighbooring V-scale factors
         DO jj = 1, ny-1
            DO ji = 1, nx-1   ! vector opt.
               e3f_0(ji,jj,jk) = MIN( e3v_0(ji,jj,jk), e3v_0(ji+1,jj,jk) )
            END DO
         END DO
      END DO
      !
      DO jk = 1, N                        ! set to z-scale factor if zero (i.e. along closed boundaries)
         WHERE( e3f_0(:,:,jk) == 0. )   e3f_0(:,:,jk) = e3t_1d(jk)
      END DO

      ! Control of the sign
      IF( MINVAL( e3t_0  (:,:,:) ) <= 0. )   STOP '    zgr_zps :   e r r o r   e3t_0 <= 0' 
      IF( MINVAL( e3w_0  (:,:,:) ) <= 0. )   STOP '    zgr_zps :   e r r o r   e3w_0 <= 0' 
      IF( MINVAL( gdept_0(:,:,:) ) <  0. )   STOP '    zgr_zps :   e r r o r   gdept_0 <  0' 
      IF( MINVAL( gdepw_0(:,:,:) ) <  0. )   STOP '    zgr_zps :   e r r o r   gdepw_0 <  0' 
     
      ! Compute gde3w_0 (vertical sum of e3w)
!      IF ( ln_isfcav ==1 ) THEN ! if cavity
!         WHERE( misfdep == 0 )   misfdep = 1
!         DO jj = 1,ny
!            DO ji = 1,nx
!               gde3w_0(ji,jj,1) = 0.5 * e3w_0(ji,jj,1)
!               DO jk = 2, misfdep(ji,jj)
!                  gde3w_0(ji,jj,jk) = gde3w_0(ji,jj,jk-1) + e3w_0(ji,jj,jk) 
!               END DO
!               IF( misfdep(ji,jj) >= 2 )   gde3w_0(ji,jj,misfdep(ji,jj)) = risfdep(ji,jj) + 0.5 * e3w_0(ji,jj,misfdep(ji,jj))
!               DO jk = misfdep(ji,jj) + 1, N
!                  gde3w_0(ji,jj,jk) = gde3w_0(ji,jj,jk-1) + e3w_0(ji,jj,jk) 
!               END DO
!            END DO
!         END DO
!      ELSE ! no cavity
!         gde3w_0(:,:,1) = 0.5 * e3w_0(:,:,1)
!         DO jk = 2, N
!            gde3w_0(:,:,jk) = gde3w_0(:,:,jk-1) + e3w_0(:,:,jk)
!         END DO
!      END IF
      !
      DEALLOCATE( zprt, gdept_0, gdepw_0 )
      !
   END SUBROUTINE zgr_zps

  !*****************************************************
  !   function set_child_name(Parentname,Childname)
  !*****************************************************
  !
  SUBROUTINE set_child_name(Parentname,Childname)
    !
    CHARACTER(*),INTENT(in) :: Parentname
    CHARACTER(*),INTENT(out) :: Childname
    CHARACTER(2) :: prefix
    INTEGER :: pos
    !   
    pos  = INDEX(TRIM(Parentname),'/',back=.TRUE.)
    !
    prefix=Parentname(pos+1:pos+2)
    IF (prefix == '1_') THEN
       Childname = '2_'//Parentname(pos+3:LEN(Parentname)) 
    ELSEIF (prefix == '2_') THEN
       Childname = '3_'//Parentname(pos+3:LEN(Parentname)) 
    ELSEIF (prefix == '3_') THEN
       Childname = '4_'//Parentname(pos+3:LEN(Parentname)) 
    ELSEIF (prefix == '4_') THEN
       Childname = '5_'//Parentname(pos+3:LEN(Parentname)) 
    ELSE
       Childname = '1_'//Parentname(pos+1:LEN(Parentname)) 
    ENDIF
    !   
  END SUBROUTINE set_child_name
  !
  !*****************************************************
  !   subroutine get_interptype(varname,interp_type,conservation)
  !*****************************************************
  !
  SUBROUTINE get_interptype( varname,interp_type,conservation )
    !
    LOGICAL,OPTIONAL :: conservation
    CHARACTER(*) :: interp_type,varname
    INTEGER :: pos,pos1,pos2,pos3,i,k 
    LOGICAL :: find        
    i=1
    DO WHILE ( TRIM(VAR_INTERP(i)) /= '' )     
       pos = INDEX( TRIM(VAR_INTERP(i)) , TRIM(varname) )
       IF ( pos .NE. 0 ) THEN      
          pos1 = INDEX( TRIM(VAR_INTERP(i)) , 'bicubic' )
          pos2 = INDEX( TRIM(VAR_INTERP(i)) , 'bilinear' )
          pos3 = INDEX( TRIM(VAR_INTERP(i)) , 'conservative' )
          ! initialize interp_type
          IF( pos1 .NE. 0 ) interp_type = 'bicubic'
          IF( pos2 .NE. 0 ) interp_type = 'bilinear'
          IF( pos1 .EQ. 0 .AND. pos2 .EQ. 0) interp_type = 'bicubic'
          ! initialize conservation                                           
          IF( pos3 .NE. 0 .AND. PRESENT(conservation) ) THEN
             conservation = .TRUE.
             RETURN
          ELSE 
             conservation = .FALSE.
          ENDIF
          find = .FALSE.
          IF( PRESENT(conservation) ) THEN
             k=0
             conservation = .FALSE.         
             DO WHILE( k < SIZE(flxtab) .AND. .NOT.find ) 
                k = k+1
                IF( TRIM(varname) .EQ. TRIM(flxtab(k)) ) THEN       
                   conservation = .TRUE.
                   find = .TRUE.
                ENDIF
             END DO
          ENDIF
          RETURN
       ENDIF
       i = i+1
    END DO
    !default values interp_type = bicubic // conservation = false      
    interp_type = 'bicubic'      
    IF( PRESENT(conservation) ) conservation = .FALSE.

    RETURN
    !   
  END SUBROUTINE get_interptype
  !      
  !*****************************************************
  !   subroutine Init_mask(name,Grid)
  !*****************************************************
  !
  SUBROUTINE Init_mask(name,Grid,jpiglo,jpjglo)
    !
    USE io_netcdf
    !      
    CHARACTER(*) name
    INTEGER :: nx,ny,k,i,j,jpiglo,jpjglo
    TYPE(Coordinates) :: Grid
    REAL*8, POINTER, DIMENSION(:,:) ::zwf => NULL()
    !
    IF(jpiglo == 1 .AND. jpjglo == 1) THEN
       CALL read_ncdf_var('Bathy_level',name,Grid%Bathy_level)
    ELSE
       CALL read_ncdf_var('Bathy_level',name,Grid%Bathy_level,(/jpizoom,jpjzoom/),(/jpiglo,jpjglo/) )
    ENDIF


    !
    WRITE(*,*) 'Init masks in T,U,V,F points'    
    !
    nx = SIZE(Grid%Bathy_level,1)
    ny = SIZE(Grid%Bathy_level,2)
    !
    !      
    ALLOCATE(Grid%tmask(nx,ny,N), &
         Grid%umask(nx,ny,N), &
         Grid%vmask(nx,ny,N), &
         Grid%fmask(nx,ny,N))
    !
    DO k = 1,N
       !      
       WHERE(Grid%Bathy_level(:,:) <= k-1 )    
          Grid%tmask(:,:,k) = 0
       ELSEWHERE
          Grid%tmask(:,:,k) = 1
       END WHERE
       !
    END DO
    !
    Grid%umask(1:nx-1,:,:) = Grid%tmask(1:nx-1,:,:)*Grid%tmask(2:nx,:,:)
    Grid%vmask(:,1:ny-1,:) = Grid%tmask(:,1:ny-1,:)*Grid%tmask(:,2:ny,:)
    !
    Grid%umask(nx,:,:) = Grid%tmask(nx,:,:)
    Grid%vmask(:,ny,:) = Grid%tmask(:,ny,:)
    !      
    Grid%fmask(1:nx-1,1:ny-1,:) = Grid%tmask(1:nx-1,1:ny-1,:)*Grid%tmask(2:nx,1:ny-1,:)* &
         Grid%tmask(1:nx-1,2:ny,:)*Grid%tmask(2:nx,2:ny,:) 
    !
    Grid%fmask(nx,:,:) = Grid%tmask(nx,:,:)
    Grid%fmask(:,ny,:) = Grid%tmask(:,ny,:)
    !
    ALLOCATE(zwf(nx,ny))
    !     
    DO k = 1,N
       !
       zwf(:,:) = Grid%fmask(:,:,k)
       !         
       DO j = 2, ny-1
          DO i = 2,nx-1            
             IF( Grid%fmask(i,j,k) == 0. ) THEN               
                Grid%fmask(i,j,k) = shlat * MIN(1.,MAX( zwf(i+1,j),zwf(i,j+1),zwf(i-1,j),zwf(i,j-1)))
             END IF
          END DO
       END DO
       !
       DO j = 2, ny-1
          IF( Grid%fmask(1,j,k) == 0. ) THEN
             Grid%fmask(1,j,k) = shlat * MIN(1.,MAX(zwf(2,j),zwf(1,j+1),zwf(1,j-1)))
          ENDIF

          IF( Grid%fmask(nx,j,k) == 0. ) THEN
             Grid%fmask(nx,j,k) = shlat * MIN(1.,MAX(zwf(nx,j+1),zwf(nx-1,j),zwf(nx,j-1)))
          ENDIF
       END DO
       !         
       DO i = 2, nx-1         
          IF( Grid%fmask(i,1,k) == 0. ) THEN
             Grid%fmask(i, 1 ,k) = shlat*MIN( 1.,MAX(zwf(i+1,1),zwf(i,2),zwf(i-1,1)))
          ENDIF
          !            
          IF( Grid%fmask(i,ny,k) == 0. ) THEN
             Grid%fmask(i,ny,k) = shlat * MIN(1.,MAX(zwf(i+1,ny),zwf(i-1,ny),zwf(i,ny-1)))
          ENDIF
       END DO
       !!
    END DO
    !!      
  END SUBROUTINE Init_mask
  !
  !*****************************************************
  !   subroutine Init_Tmask(name,Grid)
  !*****************************************************
  !
  SUBROUTINE Init_Tmask(name,Grid,jpiglo,jpjglo)
    !
    USE io_netcdf
    !      
    CHARACTER(*) name
    INTEGER :: nx,ny,k,i,j,jpiglo,jpjglo
    TYPE(Coordinates) :: Grid
    REAL*8, POINTER, DIMENSION(:,:) ::zwf => NULL()
    !
    IF(jpiglo == 1 .AND. jpjglo == 1) THEN
       CALL read_ncdf_var('Bathy_level',name,Grid%Bathy_level)
    ELSE
       CALL read_ncdf_var('Bathy_level',name,Grid%Bathy_level,(/jpizoom,jpjzoom/),(/jpiglo,jpjglo/) )
    ENDIF
    !
    nx = SIZE(Grid%Bathy_level,1)
    ny = SIZE(Grid%Bathy_level,2) 
    !
    WRITE(*,*) 'Init masks in T points'    
    !     
    ALLOCATE( Grid%tmask(nx,ny,N) )
    !
    DO k = 1,N
       !      
       WHERE(Grid%Bathy_level(:,:) <= k-1 )    
          Grid%tmask(:,:,k) = 0.
       ELSEWHERE
          Grid%tmask(:,:,k) = 1.
       END WHERE
       !
    END DO
    !      
  END SUBROUTINE Init_Tmask
  !
  !*****************************************************
  !   subroutine Init_ssmask(name,Grid)
  !*****************************************************
  !
!  SUBROUTINE Init_ssmask(varname,filename,Grid,jpiglo,jpjglo)
!    !
!    USE io_netcdf
!    !      
!    CHARACTER(*) varname,filename
!    INTEGER :: nx,ny,k,i,j,jpiglo,jpjglo
!    TYPE(Coordinates) :: Grid
!    REAL*8, POINTER, DIMENSION(:,:) ::zwf => NULL()
!    !
!    IF(jpiglo == 1 .AND. jpjglo == 1) THEN
!       CALL read_ncdf_var(varname,filename,Grid%bathy_level)
!    ELSE
!       CALL read_ncdf_var(varname,filename,Grid%bathy_level,(/jpizoom,jpjzoom/),(/jpiglo,jpjglo/) )
!    ENDIF
!    !
!    nx = SIZE(Grid%bathy_level,1)
!    ny = SIZE(Grid%bathy_level,2) 
!    !
!    WRITE(*,*) 'Init surface masks in T points'    
!    !     
!    ALLOCATE( Grid%ssmask(nx,ny), Grid%ssumask(nx,ny), Grid%ssvmask(nx,ny) )
!    !
!    WHERE(Grid%bathy_level(:,:) <= 0. )    
!       Grid%ssmask(:,:) = 0.
!    ELSEWHERE
!       Grid%ssmask(:,:) = 1.
!    END WHERE
!    !
!    Grid%ssumask(1:nx-1,:) = Grid%ssmask(1:nx-1,:)*Grid%ssmask(2:nx,:)
!    Grid%ssvmask(:,1:ny-1) = Grid%ssmask(:,1:ny-1)*Grid%ssmask(:,2:ny)
!    !
!    Grid%ssumask(nx,:) = Grid%ssmask(nx,:)
!    Grid%ssvmask(:,ny) = Grid%ssmask(:,ny)
!    !      
!  END SUBROUTINE Init_ssmask
  !
  !*****************************************************
  !   subroutine get_mask(name,Grid)
  !*****************************************************
  !
  SUBROUTINE get_mask(level,posvar,mask,filename)
    !
    USE io_netcdf
    !      
    CHARACTER(*) filename
    CHARACTER(*) posvar
    INTEGER :: level, nx, ny
    LOGICAL,DIMENSION(:,:),POINTER :: mask
    INTEGER,DIMENSION(:,:),POINTER :: maskT,maskU,maskV
    !      
    TYPE(Coordinates) :: Grid
    !
    CALL read_ncdf_var('Bathy_level',filename,Grid%Bathy_level)
    ! 
    nx = SIZE(Grid%Bathy_level,1)
    ny = SIZE(Grid%Bathy_level,2)
    ALLOCATE(maskT(nx,ny),mask(nx,ny))
    mask = .TRUE.
    !
    WHERE(Grid%Bathy_level(:,:) <= level-1 )    
       maskT(:,:) = 0
    ELSEWHERE
       maskT(:,:) = 1
    END WHERE
    !
    SELECT CASE(posvar)
       !
    CASE('T')
       !
       WHERE(maskT > 0)
          mask = .TRUE.
       ELSEWHERE
          mask = .FALSE.
       END WHERE
       DEALLOCATE(maskT)
       !
    CASE('U')
       !
       ALLOCATE(maskU(nx,ny))
       maskU(1:nx-1,:) = maskT(1:nx-1,:)*maskT(2:nx,:)
       maskU(nx,:) = maskT(nx,:)
       WHERE(maskU > 0)
          mask = .TRUE.
       ELSEWHERE
          mask = .FALSE.
       END WHERE
       DEALLOCATE(maskU,maskT)
       !
    CASE('V')
       !
       ALLOCATE(maskV(nx,ny))
       maskV(:,1:ny-1) = maskT(:,1:ny-1)*maskT(:,2:ny)
       maskV(:,ny) = maskT(:,ny)
       WHERE(maskT > 0)
          mask = .TRUE.
       ELSEWHERE
          mask = .FALSE.
       END WHERE
       DEALLOCATE(maskV,maskT)
       !
    END SELECT
    !      
  END SUBROUTINE get_mask
  !
  !
  !*****************************************************
  !   subroutine read_dimg_var(unit,irec,field)
  !*****************************************************
  !
  SUBROUTINE read_dimg_var(unit,irec,field,jpk)
    !      
    INTEGER :: unit,irec,jpk
    REAL*8,DIMENSION(:,:,:,:),POINTER :: field
    INTEGER :: k
    !      
    DO k = 1,jpk
       READ(unit,REC=irec) field(:,:,k,1)
       irec = irec + 1
    ENDDO
    !
  END SUBROUTINE read_dimg_var
  ! 
  !
  !*****************************************************
  !   subroutine read_dimg_var(unit,irec,field)
  !*****************************************************
  !
  SUBROUTINE write_dimg_var(unit,irec,field,jpk)
    !      
    INTEGER :: unit,irec,jpk
    REAL*8,DIMENSION(:,:,:,:),POINTER :: field
    INTEGER :: k
    !      
    DO k = 1,jpk
       WRITE(unit,REC=irec) field(:,:,k,1)
       irec = irec + 1
    ENDDO
    !
  END SUBROUTINE write_dimg_var

END MODULE agrif_readwrite
