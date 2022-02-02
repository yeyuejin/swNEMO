MODULE utils

  USE netcdf

  IMPLICIT NONE
  PUBLIC
  
  INTEGER, PUBLIC, PARAMETER   :: dp=8 , sp=4, wp=dp
  INTEGER :: tmask_id, umask_id, vmask_id, fmask_id
  INTEGER :: gdept_id
  INTEGER :: gphit_id, gphiv_id, gphiu_id, gphif_id        ! Variable ids
  INTEGER :: glamt_id, glamv_id, glamu_id, glamf_id        ! Variable ids
  INTEGER :: resto_id                           ! Variable ID for output
  INTEGER :: jpi, jpj, jpk                      ! Size of domain
  INTEGER  :: ncin, ncout                              ! File handles for netCDF files
  REAL(wp), DIMENSION(:,:),   ALLOCATABLE :: gphit, glamt
  REAL(wp), DIMENSION(:,:),   ALLOCATABLE :: gphiu, glamu
  REAL(wp), DIMENSION(:,:),   ALLOCATABLE :: gphiv, glamv
  REAL(wp), DIMENSION(:,:),   ALLOCATABLE :: gphif, glamf
  REAL(wp), DIMENSION(:,:),   ALLOCATABLE :: tmask, umask, vmask, fmask
  REAL(wp), DIMENSION(:,:),   ALLOCATABLE :: gdept
  REAL(wp), DIMENSION(:,:),   ALLOCATABLE :: resto

  INTEGER,PARAMETER :: numout = 6
  INTEGER,PARAMETER :: numerr = 0
  INTEGER,PARAMETER :: numnam = 11
  REAL(wp),PARAMETER :: rday = 86400           ! seconds in a day
  REAL(wp),PARAMETER ::   rpi = 3.141592653589793
  REAL(wp),PARAMETER ::   rad = 3.141592653589793/180.
  REAL(wp),PARAMETER ::   ra =  6371229.

  ! Namelist variables
  CHARACTER(LEN=30) :: cp_cfg = 'ORCA'
  CHARACTER(LEN=30) :: cp_cfz = 'No zoom'
  INTEGER :: jp_cfg = 2
  REAL(KIND=8) :: pn_surf = 1
  REAL(KIND=8) :: pn_bot = 1
  REAL(KIND=8) :: pn_dep = 1000         
  INTEGER  :: nn_hdmp = 0                          ! damping option
  INTEGER  :: jperio = 0                          ! damping option
  LOGICAL :: lzoom = .false.
  LOGICAL :: ln_coast = .false.
  LOGICAL :: ln_full_field = .true.
  LOGICAL :: ln_med_red_seas = .false.
  LOGICAL :: ln_old_31_lev_code = .false.
  LOGICAL :: ln_zero_top_layer = .false.
  LOGICAL :: ln_custom = .false.

  NAMELIST/nam_dmp_create/cp_cfg, cp_cfz, jp_cfg, lzoom, ln_full_field, &
                          ln_med_red_seas, ln_old_31_lev_code, ln_coast, &
                          ln_zero_top_layer, ln_custom, &
                          pn_surf, pn_bot, pn_dep, nn_hdmp, jperio

  CONTAINS 

  SUBROUTINE grid_info(mesh)
     CHARACTER(LEN=*),INTENT(in) :: mesh

     ! Open meshfile
     CALL check_nf90( nf90_open(mesh, NF90_NOWRITE, ncin), 'Error opening mesh_mask file' )
  
     ! Get size of grid from meshfile
     CALL dimlen( ncin, 'x', jpi )
     CALL dimlen( ncin, 'y', jpj )
     CALL dimlen( ncin, 'z', jpk )

     ALLOCATE( tmask(jpi, jpj), gdept(jpi, jpj), gphit(jpi,jpj) )

     !Get ID of tmask in meshfile
     CALL check_nf90( nf90_inq_varid( ncin, 'tmask', tmask_id ), 'Cannot get variable ID for tmask')
     CALL check_nf90( nf90_inq_varid( ncin, 'umask', umask_id ), 'Cannot get variable ID for umask')
     CALL check_nf90( nf90_inq_varid( ncin, 'vmask', vmask_id ), 'Cannot get variable ID for vmask')
     CALL check_nf90( nf90_inq_varid( ncin, 'fmask', fmask_id ), 'Cannot get variable ID for fmask')
     CALL check_nf90( nf90_inq_varid( ncin, 'gdept_0', gdept_id ), 'Cannot get variable ID for gdept_0')
     CALL check_nf90( nf90_inq_varid( ncin, 'gphit', gphit_id ), 'Cannot get variable ID for gphit')
     CALL check_nf90( nf90_inq_varid( ncin, 'gphiu', gphiu_id ), 'Cannot get variable ID for gphiu')
     CALL check_nf90( nf90_inq_varid( ncin, 'gphiv', gphiv_id ), 'Cannot get variable ID for gphiv')
     CALL check_nf90( nf90_inq_varid( ncin, 'gphif', gphif_id ), 'Cannot get variable ID for gphif')
     CALL check_nf90( nf90_inq_varid( ncin, 'glamt', glamt_id ), 'Cannot get variable ID for glamt')
     CALL check_nf90( nf90_inq_varid( ncin, 'glamu', glamu_id ), 'Cannot get variable ID for glamu')
     CALL check_nf90( nf90_inq_varid( ncin, 'glamv', glamv_id ), 'Cannot get variable ID for glamv')
     CALL check_nf90( nf90_inq_varid( ncin, 'glamf', glamf_id ), 'Cannot get variable ID for glamf')
  
  END SUBROUTINE grid_info

  SUBROUTINE dimlen( ncid, dimname, len )
     ! Determine the length of dimension dimname
     INTEGER, INTENT(in)          :: ncid
     CHARACTER(LEN=*), INTENT(in) :: dimname
     INTEGER, INTENT(out)         :: len
     ! Local variables
     INTEGER :: id_var, istatus

     id_var = 1
     CALL check_nf90( nf90_inq_dimid(ncid, dimname, id_var), 'Dimension not found in file')
     CALL check_nf90( nf90_inquire_dimension(ncid,id_var,len=len))

  END SUBROUTINE dimlen

  SUBROUTINE make_outfile( filename )
     ! Create the output file
     ! Define dimensions and resto variable
     CHARACTER(LEN=*), INTENT(in) :: filename
     INTEGER :: id_x, id_y, id_z

     CALL check_nf90( nf90_create(filename, NF90_CLOBBER, ncout), 'Could not create output file')
     CALL check_nf90( nf90_def_dim(ncout, 'x', jpi, id_x) )
     CALL check_nf90( nf90_def_dim(ncout, 'y', jpj, id_y) )
     CALL check_nf90( nf90_def_dim(ncout, 'z', jpk, id_z) )

     CALL check_nf90( nf90_def_var(ncout, 'resto', nf90_double, (/id_x,id_y,id_z/), resto_id ) )
     CALL check_nf90( nf90_enddef(ncout) )

  END SUBROUTINE make_outfile


  SUBROUTINE check_nf90( istat, message )
     !Check for netcdf errors
     INTEGER, INTENT(in) :: istat
     CHARACTER(LEN=*), INTENT(in), OPTIONAL :: message
     
     IF (istat /= nf90_noerr) THEN
        WRITE(numerr,*) 'ERROR! : '//TRIM(nf90_strerror(istat))
        IF ( PRESENT(message) ) THEN ; WRITE(numerr,*) message ; ENDIF
        STOP
     ENDIF

  END SUBROUTINE check_nf90

END MODULE utils
