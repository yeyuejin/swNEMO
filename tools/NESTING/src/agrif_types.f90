!************************************************************************
! Fortran 95 OPA Nesting tools						*
!									*
!     Copyright (C) 2005 Florian Lemarié (Florian.Lemarie@imag.fr)	*
!									*
!************************************************************************
!      
MODULE agrif_types
  !
  PUBLIC
  !   
  !*****************************
  ! Coordinates type definition 
  !*****************************
  TYPE Coordinates
     !
     REAL*8,  DIMENSION(:,:),  POINTER :: nav_lon, nav_lat => NULL()
     REAL*8,  DIMENSION(:,:),  POINTER :: glamv, glamu, glamt, glamf => NULL()
     REAL*8,  DIMENSION(:,:),  POINTER :: gphit, gphiu, gphiv, gphif => NULL()
     REAL*8,  DIMENSION(:,:),  POINTER :: e1t, e1u, e1v, e1f => NULL()
     REAL*8,  DIMENSION(:,:),  POINTER :: e2t, e2u, e2v, e2f => NULL()
     REAL*8,  DIMENSION(:,:),  POINTER :: bathy_level => NULL()
     REAL*8,  DIMENSION(:,:),  POINTER :: bathy_meter => NULL()
     REAL*8,  DIMENSION(:,:),  POINTER :: wgt => NULL()
     REAL*8,  DIMENSION(:,:,:),POINTER :: fmask, umask, vmask, tmask => NULL()
     REAL*8,  DIMENSION(:,:,:),POINTER :: e3t_ps, e3w_ps, gdept_ps, gdepwps => NULL()
     REAL*8,  DIMENSION(:,:),  POINTER :: gdepw_ps => NULL()
     REAL*8,  DIMENSION(:),    POINTER :: gdeptht => NULL()
     INTEGER, DIMENSION(:) ,   POINTER :: time_steps => NULL()
     !     
  END TYPE Coordinates
  !
  !
  !
  CHARACTER*8,DIMENSION(10) :: flxtab = (/'socliot1','socliot2','socliopl', &
       'socliocl','socliohu','socliowi','soshfldo','sohefldo','sowaflup','sofbt   '/)
  !
  !
  !**************************************************************
  ! Declaration of various input file variables (namelist.input) 
  !**************************************************************
  !
  INTEGER ::   irafx, irafy
  INTEGER ::   nxfin, nyfin
  !      
  INTEGER ::   nbghostcellsfine, imin, jmin, imax, jmax, rho, rhot
  INTEGER ::   shlat
  INTEGER ::   N, type_bathy_interp
  ! 
  INTEGER ::   jpizoom, jpjzoom, npt_connect, npt_copy
  INTEGER ::   parent_jperio
  !      
  REAL*8 ::   rn_hmin
  REAL*8 ::   ppkth2, ppacr2, ppkth, ppacr, ppdzmin, pphmax, smoothing_factor, e3zps_min, e3zps_rat
  REAL*8 ::   psur, pa0, pa1, pa2, adatrj
  !       
  LOGICAL ::   ldbletanh, ln_e3_dep
  LOGICAL ::   partial_steps, smoothing, bathy_update
  LOGICAL ::   new_topo, removeclosedseas, dimg, iom_activated
  LOGICAL ::   ln_agrif_domain
  !       
  CHARACTER*100 ::   parent_bathy_level, parent_level_name, parent_bathy_meter, parent_meter_name, parent_domcfg_out
  CHARACTER*100 ::   elevation_name, elevation_database
  CHARACTER*100 ::   parent_coordinate_file, parent_bathy_meter_updated, parent_domcfg_updated, &
     &               restart_file, restart_trc_file, restart_ice_file
  CHARACTER*100 ::   dimg_output_file, interp_type
  !      
  CHARACTER(len=80) , DIMENSION(20) :: flx_Files, u_files, v_files
  CHARACTER(len=255), DIMENSION(20) :: VAR_INTERP
  !
  NAMELIST /input_output/iom_activated
  !
  NAMELIST /coarse_grid_files/parent_coordinate_file, parent_bathy_level, parent_level_name, &
     &                                                parent_bathy_meter, parent_meter_name, parent_domcfg_out, &
     &                                                parent_jperio
  !      
  NAMELIST /bathymetry/new_topo, elevation_database, elevation_name, smoothing, smoothing_factor,  &
                       ln_agrif_domain, npt_connect, npt_copy, removeclosedseas, type_bathy_interp, rn_hmin      
  !      
  NAMELIST /nesting/nbghostcellsfine, imin, imax, jmin, jmax, rho, rhot, &
     &              bathy_update, parent_bathy_meter_updated, parent_domcfg_updated      
  !
  NAMELIST /vertical_grid/ppkth, ppacr, ppdzmin, pphmax, psur, pa0, pa1, N, ldbletanh, ln_e3_dep, pa2, ppkth2, ppacr2
  ! 
  NAMELIST /partial_cells/partial_steps, e3zps_min, e3zps_rat      
  !
  NAMELIST /nemo_coarse_grid/ jpizoom, jpjzoom 
  !          
  NAMELIST /forcing_files/ flx_files,  u_files,  v_files 
  !           
  NAMELIST /interp/ VAR_INTERP
  !      
  NAMELIST /restart/ restart_file, shlat, dimg, dimg_output_file, adatrj, interp_type 
  !
  NAMELIST /restart_trc/ restart_trc_file, interp_type 
  !
  NAMELIST /restart_ice/ restart_ice_file, interp_type 
  !
CONTAINS
  !
  !********************************************************
  !subroutine agrif_grid_allocate				*
  !							*
  !allocation of grid type elements 			*
  !      according to nx and ny				*
  !							*
  !							*
  !********************************************************
  !       
  SUBROUTINE agrif_grid_allocate(Grid, nx, ny)
    !
    TYPE(Coordinates) :: Grid
    INTEGER :: nx, ny
    !
    ALLOCATE(Grid%nav_lon(nx,ny),Grid%nav_lat(nx,ny))
    !
    ALLOCATE(Grid%glamt(nx,ny),Grid%glamu(nx,ny),Grid%glamv(nx,ny),Grid%glamf(nx,ny))
    ALLOCATE(Grid%gphit(nx,ny),Grid%gphiu(nx,ny),Grid%gphiv(nx,ny),Grid%gphif(nx,ny))
    !
    ALLOCATE(Grid%e1t(nx,ny),Grid%e1u(nx,ny),Grid%e1v(nx,ny),Grid%e1f(nx,ny))
    ALLOCATE(Grid%e2t(nx,ny),Grid%e2u(nx,ny),Grid%e2v(nx,ny),Grid%e2f(nx,ny))
    !
    ALLOCATE(Grid%bathy_level(nx,ny))
    !
  END SUBROUTINE agrif_grid_allocate
  !
  !
  !************************************************************************
  !									*
  !   subroutine read_namelist						*
  !									*
  !   read variables contained in namelist.input file   			*
  !   filled in by user 							*
  !									*
  !************************************************************************
  !
  SUBROUTINE read_namelist(namelistname)
    !
    IMPLICIT NONE
    CHARACTER(len=80) :: namelistname
    CHARACTER*255 :: output
    LOGICAL :: is_it_there
    INTEGER unit_nml
    !      
    FLX_FILES  = ''
    U_FILES    = ''
    V_FILES    = ''
    VAR_INTERP = ''
    unit_nml = Agrif_Get_Unit()
    !      
    INQUIRE ( FILE = namelistname , EXIST = is_it_there )      
    !
    IF ( is_it_there ) THEN 
       !
       OPEN ( FILE   = namelistname , &
            UNIT   =  unit_nml        , &
            STATUS = 'OLD'            , &
            FORM   = 'FORMATTED'      , &
            ACTION = 'READ'           , &
            ACCESS = 'SEQUENTIAL'     )
       !
       REWIND(unit_nml)
       READ (unit_nml , NML = input_output)
       READ (unit_nml , NML = coarse_grid_files)
       READ (unit_nml , NML = bathymetry)
       READ (unit_nml , NML = nesting) 
       READ (unit_nml , NML = vertical_grid)
       READ (unit_nml , NML = partial_cells)                    
       READ (unit_nml , NML = nemo_coarse_grid ) 
       READ (unit_nml , NML = forcing_files )  
       READ (unit_nml , NML = interp )   
       READ (unit_nml , NML = restart )
       READ (unit_nml , NML = restart_trc )
       READ (unit_nml , NML = restart_ice )
       CLOSE(unit_nml)
       !	
       irafx = rho
       irafy = rho
       imin = imin + jpizoom - 1
       imax = imax + jpizoom - 1
       jmin = jmin + jpjzoom - 1
       jmax = jmax + jpjzoom - 1
       !
       IF( ln_agrif_domain ) THEN
          nxfin = (imax-imin)*irafx+2*nbghostcellsfine+2
          nyfin = (jmax-jmin)*irafy+2*nbghostcellsfine+2
       ELSE
          bathy_update = .FALSE.
          nbghostcellsfine = 0
          nxfin = (imax-imin+1)*irafx
          nyfin = (jmax-jmin+1)*irafy
       ENDIF
       !
       IF( .NOT.partial_steps ) THEN
          WRITE(*,*) 'E R R O R: partial_steps must be set to true otherwise very thin bottom layers can be created'
          STOP
       ENDIF
       !
    ELSE
       !
       PRINT *,'namelist file ''',TRIM(namelistname),''' not found'
       STOP 
       !
    END IF
    !
    !
  END SUBROUTINE read_namelist

  INTEGER FUNCTION agrif_int(x)

    REAL :: x
    INTEGER ::i

    i = FLOOR(x) + 1

    IF( ABS(x - i).LE.0.0001 )THEN
       agrif_int = i
    ELSE
       agrif_int = i-1
    ENDIF

  END FUNCTION agrif_int
  !
  !*************************************************
  !   function Agrif_Get_Unit                             
  !*************************************************
  !

  INTEGER FUNCTION Agrif_Get_Unit()
    !
    INTEGER n
    LOGICAL op
    INTEGER :: nunit
    INTEGER :: iii,out,iiimax 
    ! 
    DO n = 7,1000
       ! 
       INQUIRE(Unit=n,Opened=op)
       !
       IF (.NOT.op) EXIT
       !      
    ENDDO
    !
    Agrif_Get_Unit=n
    !
    !
  END FUNCTION Agrif_Get_Unit
  !
END MODULE agrif_types
