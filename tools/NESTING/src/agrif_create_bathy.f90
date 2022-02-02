!************************************************************************
! Fortran 95 OPA Nesting tools						*
!									*
!     Copyright (C) 2005 Florian Lemarié (Florian.Lemarie@imag.fr)	*
!                        Laurent Debreu (Laurent.Debreu@imag.fr)	*
!************************************************************************
!
PROGRAM create_bathy
  !
  USE NETCDF
  USE bilinear_interp
  USE agrif_readwrite
  USE agrif_partial_steps
  USE agrif_connect_topo
  USE agrif_interpolation
  !
  IMPLICIT NONE
  !
  !************************************************************************
  ! 									*
  ! PROGRAM  CREATE_BATHY						*
  !									*
  ! program to implement bathymetry interpolation to generate 		*
  ! child grid bathymetry file						*
  !									*
  ! various options :							*
  !									*
  ! 1- Interpolation directly from parent bathymetry file (z-coord)	*
  ! 2- Use new topo file in meters (for example etopo)  		*
  !									*
  ! vertical coordinates permitted : z-coord and partial steps		*
  ! sigma coordinates is not yet implemented				*
  !									*
  !Interpolation is carried out using bilinear interpolation		*
  !routine from SCRIP package or median average				*		
  !									*
  !http://climate.lanl.gov/Software/SCRIP/				*
  !************************************************************************
  !
  ! variables declaration
  !      
  CHARACTER(len=80) ::   namelistname
  CHARACTER*100     ::   child_coordinates, child_level, child_meter, child_domcfg    
  LOGICAL ::   identical_grids     
  INTEGER ::   nbadd,status,narg,iargc     
  INTEGER ::   jpj,jpi
  LOGICAL,DIMENSION(:,:),POINTER     ::   masksrc => NULL()  
  INTEGER,DIMENSION(:,:),ALLOCATABLE ::   mask_oce,trouble_points
  INTEGER,DIMENSION(:)  ,POINTER     ::   src_add,dst_add => NULL() 
  REAL*8, DIMENSION(:,:),POINTER     ::   matrix,interpdata => NULL()     
  REAL*8, DIMENSION(:,:),POINTER     ::   bathy_fin_constant => NULL()  
  REAL*8, DIMENSION(:,:),ALLOCATABLE ::   bathy_test,vardep
  REAL*8, DIMENSION(:)  ,ALLOCATABLE ::   vardep1d
  REAL*8, DIMENSION(:,:),POINTER     ::   gdepw_ps_interp => NULL() 
  REAL*8  ::   Cell_lonmin,Cell_lonmax,Cell_latmin,Cell_latmax,wghts
  LOGICAL ::   Pacifique = .FALSE.
  INTEGER ::   boundary,iimin,iimax,jjmax,jjmin
  INTEGER ::   nxhr,nyhr,nxyhr,ji,jj,nbiter

  TYPE(Coordinates) :: G0,G1 
  !      
  narg = iargc()      
  IF (narg == 0) THEN
     namelistname = 'namelist.input'
  ELSE
     CALL getarg(1,namelistname)
  ENDIF
  !
  ! read input file (namelist.input)
  CALL read_namelist(namelistname)

  ! if level or meter name is missing
  IF( TRIM(parent_level_name) == '' )   parent_level_name='mbathy'
  IF( TRIM(parent_meter_name) == '' )   parent_meter_name='Bathymetry'

  ! define names of child grid files
  CALL set_child_name(parent_coordinate_file,child_coordinates) 
  IF( TRIM(parent_bathy_level) /= '' )   CALL set_child_name(parent_bathy_level,child_level)            
  IF( TRIM(parent_bathy_meter) /= '' )   CALL set_child_name(parent_bathy_meter,child_meter)
  IF( TRIM(parent_domcfg_out)  /= '' )   CALL set_child_name(parent_domcfg_out,child_domcfg) 
  !
  IF( TRIM(parent_bathy_level) == '' .AND. TRIM(parent_bathy_meter) == '') THEN
     WRITE(*,*) 'ERROR ***** one needs at least to define parent_bathy_level or parent_bathy_meter ...'
     STOP
  ENDIF
  !
  ! read fine and coarse grids coordinates file
  status = Read_Coordinates(TRIM(parent_coordinate_file),G0)
  status = Read_Coordinates(TRIM(child_coordinates),G1,Pacifique)
  !
  ! check error in size
  IF( imax > SIZE(G0%nav_lon,1) .OR. jmax > SIZE(G0%nav_lon,2) .OR. imax <= imin .OR. jmax <= jmin ) THEN
     WRITE(*,*) 'ERROR ***** bad child grid definition ...'
     WRITE(*,*) 'please check imin,jmin,imax,jmax,jpizoom,jpjzoom values'       
     STOP
  ENDIF
  IF( SIZE(G1%nav_lon,1) .NE. nxfin .OR. SIZE(G1%nav_lon,2) .NE. nyfin ) THEN
     WRITE(*,*) 'ERROR ***** bad child coordinates file ...'
     WRITE(*,*) 'please check that child coordinates file has been created with the same namelist'
     STOP
  ENDIF
  !
  ! read bathymetry data set => G0%bathy_meter
  IF( new_topo ) THEN                                       ! read G0%bathy_meter (on a reduced grid) and G1 coordinates
     DEALLOCATE( G0%nav_lon, G0%nav_lat )
     status = read_bathy_coord(TRIM(elevation_database),G0,G1,Pacifique)
  ELSE                                                      ! read G0%bathy_meter (on the global grid)
     IF( TRIM(parent_bathy_meter) /= '') THEN
        status = read_bathy_meter(TRIM(parent_bathy_meter),G0)
     ELSE
        status = Read_bathy_level(TRIM(parent_bathy_level),G0)
        CALL levels_to_meter(G0)
     ENDIF     
     ! change longitudes (from -180:180 to 0:360)
     IF(Pacifique) THEN
        WHERE(G0%nav_lon < 0.001)   G0%nav_lon = G0%nav_lon + 360.
     ENDIF
  ENDIF
  !
  ! 1st allocation of child grid bathy
  ALLOCATE(G1%bathy_meter(nxfin,nyfin))
  G1%bathy_meter(:,:)=0.

  ! check grids: if identical then do not interpolate
  identical_grids = .FALSE.
  
  IF(  SIZE(G0%nav_lat,1) == SIZE(G1%nav_lat,1) .AND. SIZE(G0%nav_lat,2) == SIZE(G1%nav_lat,2) .AND. &
     & SIZE(G0%nav_lon,1) == SIZE(G1%nav_lon,1) .AND. SIZE(G0%nav_lon,2) == SIZE(G1%nav_lon,2) ) THEN
     IF(  MAXVAL( ABS(G0%nav_lat(:,:)- G1%nav_lat(:,:)) ) < 0.0001 .AND. &
        & MAXVAL( ABS(G0%nav_lon(:,:)- G1%nav_lon(:,:)) ) < 0.0001 ) THEN
        WRITE(*,*) ''
        WRITE(*,*) 'same grid between parent and child domains => NO INTERPOLATION'
        WRITE(*,*) ''
        G1%bathy_meter = G0%bathy_meter
        identical_grids = .TRUE.
     ENDIF
  ENDIF
  
  IF( .NOT.new_topo )   type_bathy_interp = 2   ! only one which works
  !
  !
  ! what type of interpolation for bathymetry
  IF( type_bathy_interp == 0 ) THEN
     WRITE(*,*) 'Interpolation of high resolution bathymetry on child grid: Arithmetic average ...'
  ELSE IF( type_bathy_interp == 1 ) THEN
     WRITE(*,*) 'Interpolation of high resolution bathymetry on child grid: Median average ...'
  ELSE IF( type_bathy_interp == 2 ) THEN     
     WRITE(*,*) 'Interpolation of high resolution bathymetry on child grid: Bilinear interpolation ...'
  ELSE     
     WRITE(*,*) 'bad value for type_bathy_interp variable ( must be 0, 1 or 2 )'
     STOP 
  ENDIF
  !
  !
  ! ---------------------------------------------------------------------------------
  ! ===                 Bathymetry of the fine grid (step1)                       ===
  ! ---------------------------------------------------------------------------------
  ! ==> It gives G1%bathy_meter from G0%bathy_meter
  ! ---------------------------------------------------------------------------------
  
  ! === Here: G0 is the grid associated with the new topography (as gebco or etopo) ===
  
  IF( .NOT. identical_grids ) THEN
     !                                                               ! ----------------------------- 
     IF( type_bathy_interp == 0 .OR. type_bathy_interp == 1 ) THEN   ! arithmetic or median averages
        !                                                            ! ----------------------------- 
        ALLOCATE(trouble_points(nxfin,nyfin))
        trouble_points(:,:) = 0
        !                       
        DO jj = 2, nyfin
           DO ji = 2, nxfin
              !	    
              ! fine grid cell extension               
              Cell_lonmin = MIN(G1%glamf(ji-1,jj-1),G1%glamf(ji,jj-1),G1%glamf(ji,jj),G1%glamf(ji-1,jj))
              Cell_lonmax = MAX(G1%glamf(ji-1,jj-1),G1%glamf(ji,jj-1),G1%glamf(ji,jj),G1%glamf(ji-1,jj))
              Cell_latmin = MIN(G1%gphif(ji-1,jj-1),G1%gphif(ji,jj-1),G1%gphif(ji,jj),G1%gphif(ji-1,jj))
              Cell_latmax = MAX(G1%gphif(ji-1,jj-1),G1%gphif(ji,jj-1),G1%gphif(ji,jj),G1%gphif(ji-1,jj)) 
              !               
              ! look for points in G0 (bathy dataset) contained in the fine grid cells  
              iimin = 1
              DO WHILE( G0%nav_lon(iimin,1) < Cell_lonmin ) 
                 iimin = iimin + 1
              ENDDO
              !               
              jjmin = 1
              DO WHILE( G0%nav_lat(iimin,jjmin) < Cell_latmin ) 
                 jjmin = jjmin + 1
              ENDDO
              !                
              iimax = iimin 
              DO WHILE( G0%nav_lon(iimax,1) <= Cell_lonmax ) 
                 iimax = iimax + 1
                 iimax = MIN( iimax,SIZE(G0%bathy_meter,1))
              ENDDO
              !                               
              jjmax = jjmin 
              DO WHILE( G0%nav_lat(iimax,jjmax) <= Cell_latmax ) 
                 jjmax = jjmax + 1
                 jjmax = MIN( jjmax,SIZE(G0%bathy_meter,2))
              ENDDO
              !
              IF( ln_agrif_domain ) THEN
                 iimax = iimax-1
                 jjmax = jjmax-1
              ELSE
                 iimax = MAX(iimin,iimax-1)
                 jjmax = MAX(jjmin,jjmax-1)
              ENDIF
              !               
              iimin = MAX( iimin,1 )
              jjmin = MAX( jjmin,1 )
              iimax = MIN( iimax,SIZE(G0%bathy_meter,1))
              jjmax = MIN( jjmax,SIZE(G0%bathy_meter,2))

              nxhr = iimax - iimin + 1
              nyhr = jjmax - jjmin + 1                    

              IF( nxhr == 0 .OR. nyhr == 0 ) THEN
                 !
                 trouble_points(ji,jj) = 1
                 !
              ELSE
                 !
                 ALLOCATE( vardep(nxhr,nyhr), mask_oce(nxhr,nyhr) )
                 vardep(:,:) = G0%bathy_meter(iimin:iimax,jjmin:jjmax)
                 !
                 WHERE( vardep(:,:) .GT. 0. )   ;   mask_oce = 1 ;
                 ELSEWHERE                      ;   mask_oce = 0 ;
                 ENDWHERE
                 !
                 nxyhr = nxhr*nyhr
                 IF( SUM(mask_oce) < 0.5*(nxyhr) ) THEN ! if more than half of the points are on land then bathy fine = 0
                    G1%bathy_meter(ji,jj) = 0.
                 ELSE
                    IF( type_bathy_interp == 0 ) THEN     ! Arithmetic average
                       G1%bathy_meter(ji,jj) = SUM( vardep(:,:) * mask_oce(:,:) ) / SUM( mask_oce(:,:) )
                    ELSE                                  ! Median average
                       ALLOCATE(vardep1d(nxyhr))
                       vardep1d = RESHAPE(vardep,(/ nxyhr /) )
                       !!CALL ssort(vardep1d,nxyhr)
                       CALL quicksort(vardep1d,1,nxyhr)
                       !
                       ! Calculate median
                       IF (MOD(nxyhr,2) .NE. 0) THEN
                          G1%bathy_meter(ji,jj) = vardep1d( nxyhr/2 + 1 )
                       ELSE
                          G1%bathy_meter(ji,jj) = 0.5 * ( vardep1d(nxyhr/2) + vardep1d(nxyhr/2+1) )
                       END IF
                       DEALLOCATE(vardep1d)   
                    ENDIF
                 ENDIF
                 DEALLOCATE (mask_oce,vardep)
                 !
              ENDIF
           ENDDO
        ENDDO

        IF( SUM( trouble_points ) > 0 ) THEN
           PRINT*,'too much empty cells, proceed to bilinear interpolation'
           type_bathy_interp = 2
        ENDIF

        DEALLOCATE(trouble_points)

     ENDIF
     !                                                       ! ----------------------------- 
     IF( type_bathy_interp == 2) THEN                        ! Bilinear interpolation
        !                                                    ! ----------------------------- 

        ALLOCATE(masksrc(SIZE(G0%bathy_meter,1),SIZE(G0%bathy_meter,2)))
        ALLOCATE(bathy_test(nxfin,nyfin))
        !
        WHERE(G0%bathy_meter.LE.0)   ;   masksrc = .FALSE.   ;
        ELSEWHERE                    ;   masksrc = .TRUE.    ;
        END WHERE
        !            
        ! compute remapping matrix thanks to SCRIP package            
        CALL get_remap_matrix(G0%nav_lat,G1%nav_lat,G0%nav_lon,G1%nav_lon,masksrc,matrix,src_add,dst_add)
        CALL make_remap(G0%bathy_meter,bathy_test,nxfin,nyfin,matrix,src_add,dst_add)  
        !                                  
        G1%bathy_meter = bathy_test               
        !            
        DEALLOCATE(masksrc)
        DEALLOCATE(bathy_test) 

     ENDIF
     !            
  ENDIF ! not identical grids
  ! ---
  ! At this stage bathymetry in meters has already been interpolated on fine grid
  !                    => G1%bathy_meter(nxfin,nyfin)
  !
  ! Also G0 was the grid from the new bathymetry data set (etopo, gebco...) and not the coarse grid
  ! ---
  !
  ! ---------------------------------------------------------------------------------
  ! ===                 Bathymetry of the fine grid (step2)                       ===
  ! ---------------------------------------------------------------------------------
  ! ==> It gives an update of G1%bathy_meter and G1%bathy_level
  ! ---------------------------------------------------------------------------------
  ! From here on: G0 is the coarse grid
  !
  ! Coarse grid bathymetry : G0%bathy_meter (on the global grid)
  IF( TRIM(parent_bathy_meter) /= '') THEN
     status = read_bathy_meter(TRIM(parent_bathy_meter),G0)
  ELSE
     status = Read_bathy_level(TRIM(parent_bathy_level),G0)
     CALL levels_to_meter(G0)
  ENDIF

  ! Coarse grid coordinatees : G0 coordinates
  DEALLOCATE(G0%nav_lat,G0%nav_lon)
  status = Read_coordinates(TRIM(parent_coordinate_file),G0)

  ! allocate temporary arrays                  
  IF (.NOT.ASSOCIATED(G0%gdepw_ps))       ALLOCATE(G0%gdepw_ps    (SIZE(G0%bathy_meter,1),SIZE(G0%bathy_meter,2)))
  IF (.NOT.ASSOCIATED(G1%gdepw_ps))       ALLOCATE(G1%gdepw_ps    (SIZE(G1%bathy_meter,1),SIZE(G1%bathy_meter,2)))
  IF (.NOT.ASSOCIATED(gdepw_ps_interp))   ALLOCATE(gdepw_ps_interp(SIZE(G1%bathy_meter,1),SIZE(G1%bathy_meter,2)))
  !                       
  IF( ln_agrif_domain ) THEN
     boundary = npt_copy*irafx + nbghostcellsfine + 1
  ELSE
     boundary = npt_copy*irafx
  ENDIF
  !
  ! compute G0%gdepw_ps and G1%gdepw_ps
  CALL get_partial_steps(G0) 
  CALL get_partial_steps(G1)
  CALL bathymetry_control(G0%Bathy_level)
  
  ! ---------------------------------------
  ! Bathymetry at the boundaries (npt_copy)                      
  ! ---------------------------------------
  ! 1st step: interpolate coarse bathy on the fine grid (using partial steps or not)
  IF( ln_agrif_domain ) THEN                   
     CALL Check_interp(G0,gdepw_ps_interp)
  ELSE
     gdepw_ps_interp = 0. * G1%gdepw_ps
     !!CALL agrif_interp(G0%gdepw_ps,gdepw_ps_interp,'T')
     CALL init_constant_bathy(G0%gdepw_ps,gdepw_ps_interp)
  ENDIF

  IF (.NOT.ASSOCIATED(G1%wgt))   ALLOCATE(G1%wgt(SIZE(G1%bathy_meter,1),SIZE(G1%bathy_meter,2)))
  G1%wgt(:,:) = 0.
  IF ((.NOT.ASSOCIATED(G0%wgt)).AND.bathy_update) THEN 
     ALLOCATE(G0%wgt(SIZE(G0%nav_lat,1),SIZE(G0%nav_lat,2)))
     G0%wgt(:,:) = 0.
  ENDIF
  !
!!$  IF( new_topo ) THEN ! clem: no, do it even when there is no new topo
  ! 2nd step: copy parent bathymetry at the boundaries
  DO jj=1,nyfin   ! West and East
     IF ( gdepw_ps_interp(nbghostcellsfine+1,jj) > 0. ) THEN
        G1%gdepw_ps(1:boundary,jj) = gdepw_ps_interp(1:boundary,jj) 
        G1%wgt(1:boundary,jj) = 1.
     ELSE
        G1%gdepw_ps(1:nbghostcellsfine+1,jj)=0. 
     ENDIF
     !
     IF ( gdepw_ps_interp(nxfin-nbghostcellsfine,jj) > 0.) THEN
        G1%gdepw_ps(nxfin-boundary+1:nxfin,jj)=gdepw_ps_interp(nxfin-boundary+1:nxfin,jj)
        G1%wgt(nxfin-boundary+1:nxfin,jj) = 1.
     ELSE
        G1%gdepw_ps(nxfin-nbghostcellsfine:nxfin,jj) = 0.
     ENDIF
  END DO
  !
  DO ji=1,nxfin    ! South and North
     IF (gdepw_ps_interp(ji,nbghostcellsfine+1)>0.) THEN
        G1%gdepw_ps(ji,1:boundary) = gdepw_ps_interp(ji,1:boundary)
        G1%wgt(ji,1:boundary) = 1.
     ELSE
        G1%gdepw_ps(ji,1:nbghostcellsfine+1)=0. 
     ENDIF
     !
     IF (gdepw_ps_interp(ji,nyfin-nbghostcellsfine)>0.) THEN
        G1%gdepw_ps(ji,nyfin-boundary+1:nyfin)=gdepw_ps_interp(ji,nyfin-boundary+1:nyfin)
        G1%wgt(ji,nyfin-boundary+1:nyfin) = 1.
     ELSE
        G1%gdepw_ps(ji,nyfin-nbghostcellsfine:nyfin) = 0.
     ENDIF
  END DO
  !
  !clem: recalculate interpolation everywhere before linear connection (useless to me??)
  IF( ln_agrif_domain ) THEN                
     gdepw_ps_interp = 0.
     CALL Check_interp(G0,gdepw_ps_interp)
  ENDIF
  !
  ! -------------------------------------------------------
  ! Bathymetry between boundaries and interior (npt_connect)                 
  ! --------------------------------------------------------
  ! Make linear connection (on npt_connect*irafx points) between the boundaries and the interior
  IF( ln_agrif_domain ) THEN
     boundary = (npt_copy + npt_connect)*irafx + nbghostcellsfine + 1
  ELSE
     boundary = (npt_copy + npt_connect)*irafx
  ENDIF

  IF( npt_connect > 0 ) THEN
     WRITE(*,*) ' linear connection on ',npt_connect,'coarse grid points'

     wghts = 1.
     DO ji = boundary - npt_connect*irafx + 1 , boundary
        wghts = wghts - (1. / (npt_connect*irafx + 1. ) )
        DO jj=1,nyfin
           IF (G1%gdepw_ps(nbghostcellsfine+1,jj) > 0.)       G1%wgt(ji,jj) = MAX(wghts, G1%wgt(ji,jj))  
        END DO
     END DO

     wghts = 1.
     DO ji = nxfin - (boundary - npt_connect*irafx), nxfin - boundary +1 , -1
        wghts = wghts - (1. / (npt_connect*irafx + 1. ) )
        DO jj=1,nyfin
           IF (G1%gdepw_ps(nxfin-nbghostcellsfine,jj) > 0.)   G1%wgt(ji,jj) = MAX(wghts, G1%wgt(ji,jj))
        END DO
     END DO

     wghts = 1.
     DO jj = boundary - npt_connect*irafy + 1 , boundary
        wghts = wghts - (1. / (npt_connect*irafy + 1. ) )
        DO ji=1,nxfin
           IF (G1%gdepw_ps(ji,nbghostcellsfine+1) > 0.)       G1%wgt(ji,jj) = MAX(wghts, G1%wgt(ji,jj))
        END DO
     END DO

     wghts = 1.
     DO jj = nyfin - (boundary - npt_connect*irafy) , nyfin - boundary +1, -1
        wghts = wghts - (1. / (npt_connect*irafy + 1. ) )
        DO ji=1,nxfin
           IF (G1%gdepw_ps(ji,nyfin-nbghostcellsfine) > 0.)   G1%wgt(ji,jj) = MAX(wghts, G1%wgt(ji,jj))
        END DO
     END DO
     IF (.NOT.identical_grids) THEN
        G1%gdepw_ps(:,:) = (1.-G1%wgt(:,:)) * G1%gdepw_ps(:,:) + gdepw_ps_interp(:,:)*G1%wgt(:,:)
     ENDIF

  ENDIF
!!$  ENDIF

  ! replace G1%bathy_meter by G1%gdepw_ps
  G1%bathy_meter = G1%gdepw_ps
  !                     
  ! --------------------
  ! Bathymetry smoothing                 
  ! --------------------
  IF( smoothing .AND. (.NOT.identical_grids) ) THEN
     ! Chanut: smoothing everywhere then discard result in connection zone
     CALL smooth_topo(G1%gdepw_ps(:,:),nbiter)
     WHERE (G1%wgt(:,:)==0.) G1%bathy_meter(:,:) = G1%gdepw_ps(:,:)
  ELSE
     WRITE(*,*) 'No smoothing process only connection is carried out'
  ENDIF
  !
  ! ------------------
  ! Remove closed seas
  ! ------------------
  IF (removeclosedseas) THEN
     ALLOCATE(bathy_test(nxfin,nyfin))
     bathy_test=0.
     WHERE (G1%bathy_meter(1,:)    .GT.0.)   bathy_test(1,:)=1
     WHERE (G1%bathy_meter(nxfin,:).GT.0.)   bathy_test(nxfin,:)=1
     WHERE (G1%bathy_meter(:,1)    .GT.0.)   bathy_test(:,1)=1
     WHERE (G1%bathy_meter(:,nyfin).GT.0.)   bathy_test(:,nyfin)=1

     nbadd = 1
     DO WHILE (nbadd.NE.0)
        nbadd = 0
        DO jj=2,nyfin-1
           DO ji=2,nxfin-1
              IF (G1%bathy_meter(ji,jj).GT.0.) THEN
                 IF (MAX(bathy_test(ji,jj+1),bathy_test(ji,jj-1),bathy_test(ji-1,jj),bathy_test(ji+1,jj)).EQ.1) THEN
                    IF (bathy_test(ji,jj).NE.1.) nbadd = nbadd + 1
                    bathy_test(ji,jj)=1.
                 ENDIF

              ENDIF
           ENDDO
        ENDDO
     ENDDO
     WHERE (bathy_test.EQ.0.)   G1%bathy_meter = 0.
     DEALLOCATE(bathy_test)
  ENDIF
  !
  CALL get_partial_steps(G1)  ! recompute bathy_level and gdepw_ps for G1 (and correct bathy_meter)
  !
  ! update parent grid
  IF(bathy_update) THEN
     CALL Update_Parent_Bathy( G0,G1 ) 
     status = Write_Bathy_meter(TRIM(parent_bathy_meter_updated),G0)
     status = write_domcfg(TRIM(parent_domcfg_updated),G0)
  ENDIF
  !
  ! store interpolation result in output file
  IF( TRIM(parent_bathy_level) /= '' )   status = Write_Bathy_level(TRIM(child_level),G1)
  IF( TRIM(parent_bathy_meter) /= '' )   status = Write_Bathy_meter(TRIM(child_meter),G1)
  IF( TRIM(parent_domcfg_out)  /= '' )   status = write_domcfg(TRIM(child_domcfg),G1)
  !
  WRITE(*,*) '****** Bathymetry successfully created ******'
  STOP
  !
END PROGRAM create_bathy


