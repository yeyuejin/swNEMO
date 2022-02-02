!
!************************************************************************
! Fortran 95 OPA Nesting tools						*
!									*
!     Copyright (C) 2005 Florian Lemari�(Florian.Lemarie@imag.fr)	*
!									*
!************************************************************************
!
PROGRAM create_rstrt_ice
  !
  USE NETCDF
  USE bilinear_interp
  USE bicubic_interp
  USE agrif_readwrite
  USE io_netcdf 
  USE agrif_extrapolation
  USE agrif_interpolation
  USE agrif_partial_steps
  USE agrif_connect_topo
  !
  IMPLICIT NONE
  !
  !************************************************************************
  ! 									*
  ! PROGRAM  CREATE_RSTRT_ICE     					*
  !									*
  ! program to interpolate parent grid restart file to child grid		*
  !									*
  !									*
  !Interpolation is carried out using bilinear interpolation		*
  !routine from SCRIP package						*		
  !									*
  !http://climate.lanl.gov/Software/SCRIP/				*
  !************************************************************************
  !
  ! variables declaration
  !      
  CHARACTER*20,DIMENSION(:),POINTER :: Ncdf_varname => NULL()
  CHARACTER*20 :: vert_coord_name
  CHARACTER*1 :: posvar
  CHARACTER*100 :: Child_file,Childcoordinates,varname,Child_Bathy_Level,Child_Bathy_Meter   
  REAL*8, POINTER, DIMENSION(:,:,:) :: tabvar00, tabvar3d,mask => NULL()
  REAL*8, POINTER, DIMENSION(:,:,:,:) :: tabinterp4d,tabvar0,tabvar1,tabvar2,tabvar3 => NULL()
  REAL*8, POINTER, DIMENSION(:) :: tabtemp1D,nav_lev => NULL()
  REAL*8, POINTER, DIMENSION(:,:) :: tabtemp2D => NULL()
  INTEGER,DIMENSION(:),POINTER :: src_add,dst_add  => NULL()
  REAL*8,DIMENSION(:,:),POINTER :: matrix => NULL()
  LOGICAL,DIMENSION(:,:),POINTER :: masksrc => NULL()
  LOGICAL, DIMENSION(:,:,:), POINTER :: detected_pts
  LOGICAL :: Interpolation,Extrapolation,Pacifique
  INTEGER :: narg,iargc,ncid,x,y,z
  INTEGER :: ii,jl,status,varid,numdims
  CHARACTER(len=20),DIMENSION(4) :: dimnames
  CHARACTER(len=80) :: namelistname
  TYPE(Coordinates) :: G0,G1
  INTEGER :: jpl 
  REAL*8 :: tabtemp0dreal

  LOGICAL, PARAMETER :: conservation = .FALSE.
  !       
  narg = iargc()
  IF (narg == 0) THEN
     namelistname = 'namelist.input'
  ELSE
     CALL getarg(1,namelistname)
  ENDIF
  !
  ! read input file
  !
  CALL read_namelist(namelistname)
  !      
  IF(TRIM(restart_ice_file) == '') THEN
     WRITE(*,*) 'no ice restart file specified in ',TRIM(namelistname)
     STOP
  ENDIF

  !
  WRITE(*,*) ''
  WRITE(*,*) 'Interpolation of restart file : ',TRIM(restart_ice_file)
  WRITE(*,*) ''
  !
  CALL Read_Ncdf_VarName(restart_ice_file,Ncdf_varname)
  !       
  CALL set_child_name(parent_coordinate_file,Childcoordinates)   
  IF( TRIM(parent_bathy_level) /= '' )   CALL set_child_name(parent_bathy_level,Child_Bathy_Level) 
  IF( TRIM(parent_bathy_meter) /= '' )   CALL set_child_name(parent_bathy_meter,Child_Bathy_Meter)   
  !
  ! create this file
  !
  CALL set_child_name(restart_ice_file,Child_file)
  status = nf90_create(Child_file,NF90_WRITE,ncid)
  status = nf90_close(ncid)
  WRITE(*,*) 'Child grid restart file name = ',TRIM(Child_file)      
  WRITE(*,*) ''
  ! 
  ! read dimensions in parent restart file
  !
  CALL Read_Ncdf_dim('x',restart_ice_file,x)
  CALL Read_Ncdf_dim('y',restart_ice_file,y) 
  CALL Read_Ncdf_dim('numcat',restart_ice_file,z)

  !
  ! mask initialization for extrapolation and interpolation
  ! 
  WRITE(*,*) 'mask initialisation on coarse and fine grids'
  !           
  status = Read_Local_Coordinates(parent_coordinate_file,G0,(/jpizoom,jpjzoom/),(/x,y/))
  status = Read_Coordinates(Childcoordinates,G1,Pacifique)
  !
  !longitude modification if child domain covers Pacific ocean area
  !      
  IF( Pacifique ) THEN
     WHERE( G0%nav_lon < 0 )
        G0%nav_lon = G0%nav_lon + 360.
     END WHERE
     WHERE( G1%nav_lon < 0 )
        G1%nav_lon = G1%nav_lon + 360.
     END WHERE
  ENDIF
  !
  ! one needs bathy_level
  IF( TRIM(parent_bathy_level) /= '' ) THEN
     status = Read_bathy_level(TRIM(parent_bathy_level),G0)
     status = Read_bathy_level(TRIM(child_bathy_level),G1)
  ELSE
     status = read_bathy_meter(TRIM(parent_bathy_meter),G0)
     status = read_bathy_meter(TRIM(child_bathy_meter),G1)
     CALL meter_to_levels(G0)
     CALL meter_to_levels(G1)
  ENDIF
  ! get masks
  CALL Init_mask(parent_bathy_level,G0,x,y)
  CALL Init_mask(child_bathy_level,G1,nxfin,nyfin)
  
  !
  ! write dimensions in output file
  WRITE(*,*) 'write dimensions'
  !          
  CALL Write_Ncdf_dim('x',Child_file,nxfin)
  CALL Write_Ncdf_dim('y',Child_file,nyfin)
  CALL Write_Ncdf_dim('numcat',Child_file,z)
  CALL Write_Ncdf_dim('time_counter',Child_file,0) 
  !
  !
  DO ii = 1,SIZE(Ncdf_varname)      
     !      
     ! loop on variables names
     varname = TRIM(Ncdf_varname(ii))
     !      
     WRITE(*,*) 'var = ',TRIM(varname)     
     !
     SELECT CASE (varname)
        !
        !copy nav_lon from child coordinates to output file      
        !
     CASE('nav_lon')
        CALL Read_Ncdf_var(varname,TRIM(Childcoordinates),tabtemp2D) 
        CALL Write_Ncdf_var(varname,(/'x','y'/),Child_file,tabtemp2D,'float')
        CALL Copy_Ncdf_att(varname,TRIM(restart_ice_file),Child_file, MINVAL(tabtemp2D),MAXVAL(tabtemp2D))
        DEALLOCATE(tabtemp2D)
        Interpolation = .FALSE.
        !	     
        !copy nav_lat from child coordinates to output file
        !
     CASE('nav_lat')             
        CALL Read_Ncdf_var(varname,TRIM(Childcoordinates),tabtemp2D) 
        CALL Write_Ncdf_var(varname,(/'x','y'/),Child_file,tabtemp2D,'float')
        CALL Copy_Ncdf_att(varname,TRIM(restart_ice_file),Child_file,MINVAL(tabtemp2D),MAXVAL(tabtemp2D)) 
        DEALLOCATE(tabtemp2D)
        Interpolation = .FALSE.
        !
        !copy numcat from restart_ice_file to output file
        !
     CASE('numcat')
        CALL Read_Ncdf_var(varname,TRIM(restart_ice_file),nav_lev) 
        CALL Write_Ncdf_var(varname,varname,Child_file,nav_lev,'float')
        CALL Copy_Ncdf_att(varname,TRIM(restart_ice_file),Child_file)      
        Interpolation = .FALSE.
        !
        !copy time from restart_ice_file to output file                       
        !
     CASE('time_counter')
        CALL Read_Ncdf_var(varname,TRIM(restart_ice_file),tabtemp1D) 
        tabtemp1D = tabtemp1D * rhot
        CALL Write_Ncdf_var(varname,'time_counter',Child_file,tabtemp1D,'double')
        CALL Copy_Ncdf_att(varname,TRIM(restart_ice_file),Child_file) 
        DEALLOCATE(tabtemp1D)
        Interpolation = .FALSE.
        !
        !copy info from restart_ice_file to output file
        !
     CASE('kt_ice') 
        CALL Read_Ncdf_var(varname,TRIM(restart_ice_file),tabtemp0dreal)  
        tabtemp0dreal = tabtemp0dreal * rhot
        CALL Write_Ncdf_var(varname,Child_file,tabtemp0dreal,'double')
        CALL Copy_Ncdf_att(varname,TRIM(restart_ice_file),Child_file) 
        Interpolation = .FALSE.
        !
     CASE('nn_fsbc') 
        CALL Read_Ncdf_var(varname,TRIM(restart_ice_file),tabtemp0dreal)  
        CALL Write_Ncdf_var(varname,Child_file,tabtemp0dreal,'double')
        CALL Copy_Ncdf_att(varname,TRIM(restart_ice_file),Child_file) 
        Interpolation = .FALSE.
        !
     CASE('frc_voltop','frc_volbot','frc_temtop','frc_tembot') 
        CALL Read_Ncdf_var(varname,TRIM(restart_ice_file),tabtemp0dreal)  
        CALL Write_Ncdf_var(varname,Child_file,tabtemp0dreal,'double')
        CALL Copy_Ncdf_att(varname,TRIM(restart_ice_file),Child_file) 
        Interpolation = .FALSE.
        !
     CASE('u_ice')
        vert_coord_name = '1'             
        posvar='U'
        Interpolation = .TRUE.            
        
     CASE('v_ice')
        vert_coord_name = '1'             
        posvar='V'
        Interpolation = .TRUE.            

     CASE('stress12_i')
        vert_coord_name = '1'             
        posvar='F'
        Interpolation = .TRUE.            
        
     CASE DEFAULT
        IF( Get_NbDims(TRIM(varname),TRIM(restart_ice_file)) == 4 ) THEN
           vert_coord_name = 'numcat'
        ELSEIF( Get_NbDims(TRIM(varname),TRIM(restart_ice_file)) == 3 ) THEN
           vert_coord_name = '1'
        ENDIF
        posvar='T'
        Interpolation = .TRUE.            
        !      
     END SELECT

     ! --- start interpolation --- !
     IF( Interpolation ) THEN
        !
        !
        IF( vert_coord_name == '1' ) THEN
           jpl = 1
        ELSE
           jpl = z
        ENDIF
        !
        ALLOCATE(detected_pts(SIZE(G0%tmask,1),SIZE(G0%tmask,2),jpl))                             
        ALLOCATE(tabvar00(x,y,1))
        ALLOCATE(tabvar0(x,y,jpl,1))
        ALLOCATE(tabvar1(x,y,1,1))
        ALLOCATE(tabvar2(x,y,1,1))
        ALLOCATE(tabvar3(x,y,1,1))
        ALLOCATE(masksrc(x,y))
        ALLOCATE(tabinterp4d(nxfin,nyfin,jpl,1)) 

        IF( vert_coord_name == '1' ) THEN
           CALL Read_Ncdf_var(varname,TRIM(restart_ice_file),tabvar00)
        ELSE
           CALL Read_Ncdf_var(varname,TRIM(restart_ice_file),tabvar0)
        ENDIF

        DO jl = 1, jpl
           !
           WRITE(*,*) 'interpolate/extrapolate for category = ',jl   
           !
           IF( vert_coord_name == '1' ) THEN
              tabvar1(:,:,1,1) = tabvar00(:,:,1)
              tabvar2(:,:,1,1) = tabvar00(:,:,1)
              tabvar3(:,:,1,1) = tabvar00(:,:,1)
           ELSE
              tabvar1(:,:,1,1) = tabvar0(:,:,jl,1)
              tabvar2(:,:,1,1) = tabvar0(:,:,jl,1)
              tabvar3(:,:,1,1) = tabvar0(:,:,jl,1)
           ENDIF
           !
           CALL extrap_detect(G0,G1,detected_pts(:,:,jl),1,posvar)                                            
           CALL correct_field(detected_pts(:,:,jl),tabvar1,tabvar2,tabvar3,G0,nav_lev,masksrc,1,posvar) !clem: nav_lev is useless here
           
           SELECT CASE(TRIM(interp_type))
           CASE('bilinear')                                                       
              CALL get_remap_matrix(G0%nav_lat,G1%nav_lat,G0%nav_lon,G1%nav_lon,masksrc,matrix,src_add,dst_add)
              CALL make_remap(tabvar1(:,:,1,1),tabinterp4d(:,:,jl,1),nxfin,nyfin,matrix,src_add,dst_add)
           CASE('bicubic')                                   
              CALL get_remap_bicub(G0%nav_lat,G1%nav_lat,G0%nav_lon,G1%nav_lon,masksrc,matrix,src_add,dst_add)
              CALL make_bicubic_remap(tabvar1(:,:,1,1),masksrc,tabinterp4d(:,:,jl,1),nxfin,nyfin,matrix,src_add,dst_add) 
           END SELECT
           !
           IF( conservation ) THEN ! clem: currently this does not work (and is only coded for T)
              SELECT CASE (posvar)
              CASE( 'T' )
                 CALL Correctforconservation(tabvar1(:,:,1,1),tabinterp4d(:,:,jl,1), &
                    &                        G0%e1t,G0%e2t,G1%e1t,G1%e2t,nxfin,nyfin,posvar,imin-jpizoom+1,jmin-jpjzoom+1)
              CASE( 'U' )
                 CALL Correctforconservation(tabvar1(:,:,1,1),tabinterp4d(:,:,jl,1), &
                    &                        G0%e1u,G0%e2u,G1%e1u,G1%e2u,nxfin,nyfin,posvar,imin-jpizoom+1,jmin-jpjzoom+1)
              CASE( 'V' )
                 CALL Correctforconservation(tabvar1(:,:,1,1),tabinterp4d(:,:,jl,1), &
                    &                        G0%e1v,G0%e2v,G1%e1v,G1%e2v,nxfin,nyfin,posvar,imin-jpizoom+1,jmin-jpjzoom+1)
              CASE( 'F' )
                 CALL Correctforconservation(tabvar1(:,:,1,1),tabinterp4d(:,:,jl,1), &
                    &                        G0%e1f,G0%e2f,G1%e1f,G1%e2f,nxfin,nyfin,posvar,imin-jpizoom+1,jmin-jpjzoom+1)
              END SELECT
           ENDIF
           tabinterp4d(:,:,jl,1) =  tabinterp4d(:,:,jl,1) * G1%tmask(:,:,1)

           IF(ASSOCIATED(matrix)) DEALLOCATE(matrix,src_add,dst_add)   

        ENDDO

        dimnames(1)='x'
        dimnames(2)='y'
        IF( vert_coord_name == '1' ) THEN
           dimnames(3)='time_counter'

           ALLOCATE(tabvar3d(SIZE(tabinterp4d,1),SIZE(tabinterp4d,2),SIZE(tabinterp4d,3)))
           tabvar3d=tabinterp4d(:,:,:,1)
           CALL Write_Ncdf_var(varname,dimnames,Child_file,tabvar3d,'double')
           DEALLOCATE(tabvar3d)
        ELSE
           dimnames(3)='numcat'
           dimnames(4)='time_counter'
           
           CALL Write_Ncdf_var(TRIM(varname),dimnames,Child_file,tabinterp4d,'double')
        ENDIF
        !             
        CALL Copy_Ncdf_att(TRIM(varname),TRIM(restart_ice_file),Child_file)

        DEALLOCATE(detected_pts)
        DEALLOCATE(tabinterp4d)
        DEALLOCATE(tabvar00, tabvar0, tabvar1,tabvar2,tabvar3)
        DEALLOCATE(masksrc)  
       
     ENDIF
  END DO
  !
  WRITE(*,*) ' '
  WRITE(*,*) ' --- list of all the variables that have been interpolated --- '
  WRITE(*,*) Ncdf_varname
  WRITE(*,*) ' '
  WRITE(*,*) '******* restart file successfully created *******' 
  WRITE(*,*) ' '
  !
  STOP 
END PROGRAM create_rstrt_ice


