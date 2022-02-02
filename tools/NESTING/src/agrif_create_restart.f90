!
!************************************************************************
! Fortran 95 OPA Nesting tools						*
!									*
!     Copyright (C) 2005 Florian Lemari�(Florian.Lemarie@imag.fr)	*
!									*
!************************************************************************
!
PROGRAM create_rstrt
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
  ! PROGRAM  CREATE_RSTRT							*
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
  REAL*8, POINTER, DIMENSION(:,:,:) :: tabvar3d => NULL()
  REAL*8, POINTER, DIMENSION(:,:,:,:) :: un,ub,vn,vb,tn,tb,sn,sb,e3t_n,e3t_b => NULL()
  REAL*8, POINTER, DIMENSION(:,:,:) :: sshn,sshb => NULL()      
  REAL*8, POINTER, DIMENSION(:,:,:,:) :: tabinterp4d,tabvar1,tabvar2,tabvar3 => NULL()
  REAL*8, POINTER, DIMENSION(:) :: tabtemp1D,nav_lev => NULL()
  REAL*8, POINTER, DIMENSION(:,:) :: tabtemp2D => NULL()
  REAL*8, POINTER, DIMENSION(:,:,:,:) :: tabtemp4D => NULL()
  INTEGER,DIMENSION(:),POINTER :: src_add,dst_add  => NULL()
  REAL*8,DIMENSION(:,:),POINTER :: matrix => NULL()
  LOGICAL,DIMENSION(:,:),POINTER :: masksrc => NULL()
  LOGICAL, DIMENSION(:,:,:), POINTER :: detected_pts
  LOGICAL :: Interpolation,Extrapolation,Pacifique
  INTEGER :: narg,iargc,ncid,x,y,z,z_a,x_a,y_a,z_b,nbvert_lev
  REAL*8 :: now_wght,before_wght
  INTEGER :: status,ii,jk
  CHARACTER(len=20),DIMENSION(4) :: dimnames
  CHARACTER(len=80) :: namelistname
  TYPE(Coordinates) :: G0,G1
  REAL*8 :: tabtemp0dreal
  CHARACTER(len=20) :: timedimname

  LOGICAL, PARAMETER :: conservation = .FALSE.
  !      
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
  IF(TRIM(restart_file) == '') THEN
     WRITE(*,*) 'no restart file specified in ',TRIM(namelistname)
     STOP
  END IF

  IF (iom_activated) THEN
     timedimname = 'time_counter'
  ELSE
     timedimname='time'
  ENDIF

  !
  WRITE(*,*) ''
  WRITE(*,*) 'Interpolation of restart file : ',TRIM(restart_file)
  WRITE(*,*) ''
  !
  CALL Read_Ncdf_VarName(restart_file,Ncdf_varname)
  !       
  CALL set_child_name(parent_coordinate_file,Childcoordinates)   
  IF( TRIM(parent_bathy_level) /= '' )   CALL set_child_name(parent_bathy_level,Child_Bathy_Level) 
  IF( TRIM(parent_bathy_meter) /= '' )   CALL set_child_name(parent_bathy_meter,Child_Bathy_Meter)   
  !
  ! create this file
  !
  CALL set_child_name(restart_file,Child_file)
  status = nf90_create(Child_file,NF90_WRITE,ncid)
  status = nf90_close(ncid)
  WRITE(*,*) 'Child grid restart file name = ',TRIM(Child_file)      
  WRITE(*,*) ''

  ! 
  ! read dimensions in parent restart file
  !
  CALL Read_Ncdf_dim('x',restart_file,x)
  CALL Read_Ncdf_dim('y',restart_file,y) 
  CALL Read_Ncdf_dim('nav_lev',restart_file,z)
  IF (.NOT.iom_activated) THEN
     CALL Read_Ncdf_dim('x_a',restart_file,x_a)
     CALL Read_Ncdf_dim('y_a',restart_file,y_a)
     CALL Read_Ncdf_dim('z_a',restart_file,z_a)
     CALL Read_Ncdf_dim('z_b',restart_file,z_b)
  ENDIF

  IF( z .NE. N ) THEN
     WRITE(*,*) '***'
     WRITE(*,*) 'Number of vertical levels doesn t match between namelist and restart file'
     WRITE(*,*) 'Please check the values in namelist file'
     STOP
  ENDIF
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
     !
     WHERE( G0%nav_lon < 0 )
        G0%nav_lon = G0%nav_lon + 360.
     END WHERE
     !              
     WHERE( G1%nav_lon < 0 )
        G1%nav_lon = G1%nav_lon + 360.
     END WHERE
     !
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
  CALL Init_mask(child_bathy_level,G1,1,1)

  G0%tmask = 1.    

  DO jk=1,z
     ALLOCATE(tabvar1(x,y,1,1))
     CALL Read_Ncdf_var('sn',TRIM(restart_file),tabvar1,1,jk)
     WHERE( tabvar1(:,:,1,1) == 0. ) 
        G0%tmask(:,:,jk) = 0.
     END WHERE
     DEALLOCATE(tabvar1)
  END DO
  !
  G0%umask(1:x-1,:,:) = G0%tmask(1:x-1,:,:)*G0%tmask(2:x,:,:)
  G0%umask(x,:,:)     = G0%tmask(x,:,:)
  G0%vmask(:,1:y-1,:) = G0%tmask(:,1:y-1,:)*G0%tmask(:,2:y,:)
  G0%vmask(:,y,:)     = G0%tmask(:,y,:)
  !      
  G0%fmask(1:x-1,1:y-1,:) = G0%tmask(1:x-1,1:y-1,:)*G0%tmask(2:x,1:y-1,:)* &
     &                      G0%tmask(1:x-1,2:y,:)*G0%tmask(2:x,2:y,:) 
  G0%fmask(x,:,:) = G0%tmask(x,:,:)
  G0%fmask(:,y,:) = G0%tmask(:,y,:)
  !
  !
  ! write dimensions in output file
  WRITE(*,*) 'write dimensions'
  !          
  CALL Write_Ncdf_dim('x',Child_file,nxfin)
  CALL Write_Ncdf_dim('y',Child_file,nyfin)
  CALL Write_Ncdf_dim('nav_lev',Child_file,z)
  CALL Write_Ncdf_dim(TRIM(timedimname),Child_file,0) 
  IF (.NOT.iom_activated) THEN
     CALL Write_Ncdf_dim('x_a',Child_file,x_a)
     CALL Write_Ncdf_dim('y_a',Child_file,y_a)
     CALL Write_Ncdf_dim('z_a',Child_file,z_a)
     CALL Write_Ncdf_dim('z_b',Child_file,z_b)
  ENDIF
  !
  !
  !
  !
  DO ii = 1,SIZE(Ncdf_varname)      
     !      
     ! loop on variables names
     varname = TRIM(Ncdf_varname(ii))
     WRITE(*,*) 'var = ',TRIM(varname)     
     !      
     SELECT CASE (TRIM(varname))
        !
     CASE('nav_lon')
        CALL Read_Ncdf_var('nav_lon',TRIM(Childcoordinates),tabtemp2D) 
        CALL Write_Ncdf_var('nav_lon',(/'x','y'/),Child_file,tabtemp2D,'float')
        CALL Copy_Ncdf_att('nav_lon',TRIM(restart_file),Child_file,MINVAL(tabtemp2D),MAXVAL(tabtemp2D))
        DEALLOCATE(tabtemp2D)
        Interpolation = .FALSE.
        !	     
     CASE('nav_lat')             
        CALL Read_Ncdf_var('nav_lat',TRIM(Childcoordinates),tabtemp2D) 
        CALL Write_Ncdf_var('nav_lat',(/'x','y'/),Child_file,tabtemp2D,'float')
        CALL Copy_Ncdf_att('nav_lat',TRIM(restart_file),Child_file,MINVAL(tabtemp2D),MAXVAL(tabtemp2D)) 
        DEALLOCATE(tabtemp2D)
        Interpolation = .FALSE.
        !
     CASE('nav_lev')
        CALL Read_Ncdf_var('nav_lev',TRIM(restart_file),nav_lev) 
        CALL Write_Ncdf_var('nav_lev','nav_lev',Child_file,nav_lev,'float')
        CALL Copy_Ncdf_att('nav_lev',TRIM(restart_file),Child_file)      
        Interpolation = .FALSE.
        !
     CASE('time')
        CALL Read_Ncdf_var('time',TRIM(restart_file),tabtemp1D) 
        CALL Write_Ncdf_var('time',TRIM(timedimname),Child_file,tabtemp1D,'float')
        CALL Copy_Ncdf_att('time',TRIM(restart_file),Child_file) 
        DEALLOCATE(tabtemp1D)
        Interpolation = .FALSE.
        !
     CASE('time_counter')
        CALL Read_Ncdf_var('time_counter',TRIM(restart_file),tabtemp1D) 
        tabtemp1D = tabtemp1D * rhot
        CALL Write_Ncdf_var('time_counter',TRIM(timedimname),Child_file,tabtemp1D,'double')
        CALL Copy_Ncdf_att('time_counter',TRIM(restart_file),Child_file) 
        DEALLOCATE(tabtemp1D)
        Interpolation = .FALSE.
        !
     CASE('kt','ndastp','adatrj','ntime','nn_fsbc','rdt') 
        IF (iom_activated) THEN
           CALL Read_Ncdf_var(TRIM(varname),TRIM(restart_file),tabtemp0dreal)  
           SELECT CASE (TRIM(varname))
           CASE('rdt')
              tabtemp0dreal = tabtemp0dreal / rhot
           CASE('kt')
              tabtemp0dreal = tabtemp0dreal * rhot
           END SELECT
           CALL Write_Ncdf_var(TRIM(varname),Child_file,tabtemp0dreal,'double')
        ELSE
           CALL Read_Ncdf_var(TRIM(varname),TRIM(restart_file),tabtemp4D) 
           dimnames(1)='x_a'
           dimnames(2)='y_a'
           dimnames(3)='z_b'
           dimnames(4)=TRIM(timedimname)            
           CALL Write_Ncdf_var(TRIM(varname),dimnames,Child_file,tabtemp4D,'double')
           DEALLOCATE(tabtemp4D)
        ENDIF
        CALL Copy_Ncdf_att(TRIM(varname),TRIM(restart_file),Child_file) 
        Interpolation = .FALSE.
        !
     CASE('frc_v','frc_t','frc_s') 
        CALL Read_Ncdf_var(varname,TRIM(restart_ice_file),tabtemp0dreal)  
        CALL Write_Ncdf_var(varname,Child_file,tabtemp0dreal,'double')
        CALL Copy_Ncdf_att(varname,TRIM(restart_ice_file),Child_file) 
        Interpolation = .FALSE.
        !
        ! Variable interpolation according to their position on grid
        !                                   
     CASE('ssu_m','utau_b','un_bf','un','ub')  
        IF( Get_NbDims(TRIM(varname),TRIM(restart_file)) == 4 ) THEN
           vert_coord_name = 'nav_lev'
        ELSEIF( Get_NbDims(TRIM(varname),TRIM(restart_file)) == 3 ) THEN
           vert_coord_name = '1'
        ENDIF
        posvar='U'
        Interpolation = .TRUE.  
        !
     CASE('ssv_m','vtau_b','vn_bf','vn','vb')  
        IF( Get_NbDims(TRIM(varname),TRIM(restart_file)) == 4 ) THEN
           vert_coord_name = 'nav_lev'
        ELSEIF( Get_NbDims(TRIM(varname),TRIM(restart_file)) == 3 ) THEN
           vert_coord_name = '1'
        ENDIF
        posvar='V'
        Interpolation = .TRUE.  
        !
     CASE DEFAULT
        IF( Get_NbDims(TRIM(varname),TRIM(restart_file)) == 4 ) THEN
           vert_coord_name = 'nav_lev'
        ELSEIF( Get_NbDims(TRIM(varname),TRIM(restart_file)) == 3 ) THEN
           vert_coord_name = '1'
        ENDIF
        posvar='T'
        Interpolation = .TRUE.            
        !      
     END SELECT
     !      
     ! --- start interpolation --- !
     IF( Interpolation ) THEN
        !		  
        IF( vert_coord_name == '1' ) THEN
           nbvert_lev = 1
        ELSE
           nbvert_lev = z
        ENDIF
        !

        ALLOCATE(detected_pts(SIZE(G0%tmask,1),SIZE(G0%tmask,2),nbvert_lev))                             
        ALLOCATE(tabvar1(x,y,1,2))
        ALLOCATE(tabvar2(x,y,1,1))
        ALLOCATE(tabvar3(x,y,1,1))
        ALLOCATE(masksrc(x,y))
        ALLOCATE(tabinterp4d(nxfin,nyfin,1,1)) 

        !	    
        DO n = 1,nbvert_lev
           !
           WRITE(*,*) 'interpolate/extrapolate for vertical level = ',n   
           !                           
           CALL Read_Ncdf_var(varname,TRIM(restart_file),tabvar1,1,n)
           IF(n==1) THEN
              !                            
           ELSE IF (n==2) THEN
              tabvar2(:,:,:,1) = tabvar1(:,:,:,2)
           ELSE
              tabvar3(:,:,:,1) = tabvar2(:,:,:,1)
              tabvar2(:,:,:,1) = tabvar1(:,:,:,2)

           ENDIF
           !                            
           SELECT CASE(posvar)
              ! 
           CASE('T')
              !
              IF(MAXVAL(G1%tmask(:,:,n)) == 0.) THEN		      
                 tabinterp4d = 0.0
                 WRITE(*,*) 'only land points on level ',n                     
              ELSE		       		      

                 CALL extrap_detect(G0,G1,detected_pts(:,:,n),n)                                            

                 CALL correct_field(detected_pts(:,:,n),tabvar1,tabvar2,tabvar3,G0,nav_lev,masksrc,n)                                 

                 ! for the following variables, you do not want to mask the values
                 IF(  TRIM(varname) == 'e3t_n' .OR. TRIM(varname) == 'e3t_b' .OR. TRIM(varname) == 'e3t_m' .OR. &
                    & TRIM(varname) == 'fraqsr_1lev' .OR. TRIM(varname) == 'frq_m' .OR. TRIM(varname) == 'surf_ini' ) THEN
                    masksrc(:,:) = .TRUE.
                 ENDIF
                 
                 SELECT CASE(TRIM(interp_type))
                 CASE('bilinear')                                                       
                    CALL get_remap_matrix(G0%nav_lat,G1%nav_lat, &
                       G0%nav_lon,G1%nav_lon,masksrc,matrix,src_add,dst_add)
                    CALL make_remap(tabvar1(:,:,1,1),tabinterp4d(:,:,1,1),nxfin,nyfin, &
                       matrix,src_add,dst_add)     
                 CASE('bicubic')                                   
                    CALL get_remap_bicub(G0%nav_lat,G1%nav_lat, &
                       G0%nav_lon,G1%nav_lon,masksrc,matrix,src_add,dst_add)
                    CALL make_bicubic_remap(tabvar1(:,:,1,1),masksrc,tabinterp4d(:,:,1,1),&
                       nxfin,nyfin,matrix,src_add,dst_add) 
                 END SELECT
                 !
                 IF( conservation ) THEN ! clem: it currently does not work
                    CALL Correctforconservation(tabvar1(:,:,1,1),tabinterp4d(:,:,1,1), &
                       G0%e1t,G0%e2t,G1%e1t,G1%e2t,nxfin,nyfin,posvar,imin-jpizoom+1,jmin-jpjzoom+1)
                 ENDIF

              ENDIF
              
              IF( ALL(masksrc) ) THEN
                 tabinterp4d(:,:,1,1) =  tabinterp4d(:,:,1,1)
              ELSE
                 tabinterp4d(:,:,1,1) =  tabinterp4d(:,:,1,1) * G1%tmask(:,:,n)
              ENDIF
              !
           CASE('U')
              !
              IF(MAXVAL(G1%umask(:,:,n)) == 0) THEN
                 tabinterp4d = 0.0
                 WRITE(*,*) 'only land points on level ',n  
              ELSE		
                 !
                 CALL extrap_detect(G0,G1,detected_pts(:,:,n),n,'U')  
                 CALL correct_field(detected_pts(:,:,n),tabvar1,tabvar2,tabvar3,G0,nav_lev,masksrc,n,'U')
                 !                                                           
                 SELECT CASE(TRIM(interp_type))
                 CASE('bilinear')                                                       
                    CALL get_remap_matrix(G0%gphiu,G1%gphiu,   &
                       G0%glamu,G1%glamu,masksrc,matrix,src_add,dst_add)
                    CALL make_remap(tabvar1(:,:,1,1),tabinterp4d(:,:,1,1),nxfin,nyfin, &
                       matrix,src_add,dst_add)     
                 CASE('bicubic')                                   
                    CALL get_remap_bicub(G0%gphiu,G1%gphiu,   &
                       G0%glamu,G1%glamu,masksrc,matrix,src_add,dst_add)
                    CALL make_bicubic_remap(tabvar1(:,:,1,1),masksrc,tabinterp4d(:,:,1,1),&
                       nxfin,nyfin,matrix,src_add,dst_add)                        
                 END SELECT
                 !                      
                 IF( conservation ) THEN ! clem: not coded for U
                    CALL Correctforconservation(tabvar1(:,:,1,1),tabinterp4d(:,:,1,1), &
                       G0%e1u,G0%e2u,G1%e1u,G1%e2u,nxfin,nyfin,posvar,imin-jpizoom+1,jmin-jpjzoom+1)
                 ENDIF
              ENDIF

              tabinterp4d(:,:,1,1) =  tabinterp4d(:,:,1,1) * G1%umask(:,:,n)
              !
           CASE('V')
              !		
              IF(MAXVAL(G1%vmask(:,:,n)) == 0) THEN
                 tabinterp4d = 0.0
                 WRITE(*,*) 'only land points on level ',n 
              ELSE		
                 !

                 CALL extrap_detect(G0,G1,detected_pts(:,:,n),n,'V')

                 CALL correct_field(detected_pts(:,:,n),tabvar1,tabvar2,tabvar3,G0,nav_lev,masksrc,n,'V')
                 !                                                           
                 SELECT CASE(TRIM(interp_type))
                 CASE('bilinear')                                                       
                    CALL get_remap_matrix(G0%gphiv,G1%gphiv,   &
                       G0%glamv,G1%glamv,masksrc,matrix,src_add,dst_add)
                    CALL make_remap(tabvar1(:,:,1,1),tabinterp4d(:,:,1,1),nxfin,nyfin, &
                       matrix,src_add,dst_add)     
                 CASE('bicubic')                                   
                    CALL get_remap_bicub(G0%gphiv,G1%gphiv,   &
                       G0%glamv,G1%glamv,masksrc,matrix,src_add,dst_add)
                    CALL make_bicubic_remap(tabvar1(:,:,1,1),masksrc,tabinterp4d(:,:,1,1),&
                       nxfin,nyfin,matrix,src_add,dst_add)                        
                 END SELECT
                 !                      
                 IF( conservation ) THEN ! clem: not coded for V
                    CALL Correctforconservation(tabvar1(:,:,1,1),tabinterp4d(:,:,1,1), &
                       G0%e1v,G0%e2v,G1%e1v,G1%e2v,nxfin,nyfin,posvar,imin-jpizoom+1,jmin-jpjzoom+1)
                 ENDIF
              ENDIF

              tabinterp4d(:,:,1,1) =  tabinterp4d(:,:,1,1) * G1%vmask(:,:,n)
              !
           END SELECT
           !
           !
           dimnames(1)='x'
           dimnames(2)='y'
           IF( vert_coord_name == '1' ) THEN
              dimnames(3)=TRIM(timedimname)
              
              ALLOCATE(tabvar3d(SIZE(tabinterp4d,1),SIZE(tabinterp4d,2),SIZE(tabinterp4d,3)))
              tabvar3d=tabinterp4d(:,:,:,1)
              CALL Write_Ncdf_var(TRIM(varname),dimnames,Child_file,tabvar3d,1,'double')
              DEALLOCATE(tabvar3d)
           ELSE
              dimnames(3)=vert_coord_name
              dimnames(4)=TRIM(timedimname)
              
              CALL Write_Ncdf_var(TRIM(varname),dimnames,Child_file,tabinterp4d,1,n,'double')
           ENDIF
           !             
           !
           CALL Copy_Ncdf_att(TRIM(varname),TRIM(restart_file),Child_file)
           !
           !
           IF(ASSOCIATED(matrix)) DEALLOCATE(matrix,src_add,dst_add)   
           !
        END DO
        ! 
        DEALLOCATE(detected_pts)
        DEALLOCATE(tabinterp4d)
        DEALLOCATE(tabvar1,tabvar2,tabvar3)
        DEALLOCATE(masksrc)  
        !                     
     ENDIF

  END DO

  ! change the before fields
  IF(rhot == 1) THEN
     WRITE(*,*) ''	 
     WRITE(*,*) 'no time interpolation (time refinement ratio = 1)'
  ELSE   
     now_wght = (rhot-1.)/rhot
     before_wght = 1./rhot
     !    
     ! --- 4D variables --- !               
     CALL Read_Ncdf_var('un',Child_file,un)      
     CALL Read_Ncdf_var('vn',Child_file,vn)
     CALL Read_Ncdf_var('ub',Child_file,ub)
     CALL Read_Ncdf_var('vb',Child_file,vb)
     ub = now_wght*un + before_wght*ub
     vb = now_wght*vn + before_wght*vb
     !
     CALL Read_Ncdf_var('tb',Child_file,tb)
     CALL Read_Ncdf_var('tn',Child_file,tn)
     tb = now_wght*tn + before_wght*tb
     !
     CALL Read_Ncdf_var('sb',Child_file,sb)
     CALL Read_Ncdf_var('sn',Child_file,sn)            
     sb = now_wght*sn + before_wght*sb
     !
     CALL Read_Ncdf_var('e3t_b',Child_file,e3t_b)
     CALL Read_Ncdf_var('e3t_n',Child_file,e3t_n)            
     e3t_b = now_wght*e3t_n + before_wght*e3t_b
     !
     dimnames(1)='x'
     dimnames(2)='y'
     dimnames(3)='nav_lev'
     dimnames(4)=TRIM(timedimname)
     CALL Write_Ncdf_var('ub',dimnames,Child_file,ub,'double')
     CALL Write_Ncdf_var('vb',dimnames,Child_file,vb,'double')
     CALL Write_Ncdf_var('tb',dimnames,Child_file,tb,'double')
     CALL Write_Ncdf_var('sb',dimnames,Child_file,sb,'double')
     CALL Write_Ncdf_var('e3t_b',dimnames,Child_file,e3t_b,'double')
     !
     DEALLOCATE(un,ub,vn,vb,tn,tb,sn,sb,e3t_n,e3t_b)
     !----------------------               
     !
     ! --- 3D variables --- !	       
     CALL Read_Ncdf_var('sshb',Child_file,sshb)
     CALL Read_Ncdf_var('sshn',Child_file,sshn)
     sshb = now_wght*sshn + before_wght*sshb
     !
     dimnames(1)='x'
     dimnames(2)='y'
     dimnames(3)=TRIM(timedimname)
     CALL Write_Ncdf_var('sshb',dimnames,Child_file,sshb,'double')
     !               
     DEALLOCATE(sshn,sshb) 
     !----------------------               
     !    
  ENDIF
  !
  !
  WRITE(*,*) ' '
  WRITE(*,*) ' --- list of all the variables that have been interpolated --- '
  WRITE(*,*) Ncdf_varname
  WRITE(*,*) ' '
  WRITE(*,*) '******* restart file successfully created *******' 
  WRITE(*,*) ' '
  !
  STOP 
END PROGRAM


