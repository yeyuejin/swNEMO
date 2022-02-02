!------------------------------------------------------------------------
! Copyright 2018/03, CERFACS, Toulouse, France.
! All rights reserved. Use is subject to OASIS3 license terms.
!=============================================================================
!
PROGRAM TOYATM
  !
  USE netcdf
  USE mod_oasis
  !
  IMPLICIT NONE
  !
  INTEGER, PARAMETER :: wp = 8
  !
  CHARACTER(len=30), PARAMETER   :: data_gridname='grids.nc' ! file with the grids
  CHARACTER(len=30), PARAMETER   :: data_maskname='masks.nc' ! file with the masks
  !
  ! Component name (6 characters) same as in the namcouple
  CHARACTER(len=6)   :: comp_name = 'toyatm'
  CHARACTER(len=128) :: comp_out       ! name of the output log file
  CHARACTER(len=4)   :: cl_grd_src     ! name of the source grid
  !
  ! Global grid parameters : 
  INTEGER, PARAMETER :: nlon = 180
  INTEGER, PARAMETER :: nlat = 90

  REAL (kind=wp)   :: gg_lon(nlon,nlat)
  REAL (kind=wp)   :: gg_lat(nlon,nlat)
  INTEGER          :: gg_mask(nlon,nlat)
  !
  ! Exchanged local fields arrays
  REAL (kind=wp), ALLOCATABLE :: field_send(:,:)
  !
  REAL (kind=wp), ALLOCATABLE :: field_recv(:,:)

  INTEGER :: mype, npes ! rank and  number of pe
  INTEGER :: localComm  ! local MPI communicator and Initialized
  INTEGER :: comp_id    ! component identification
  !
  INTEGER :: il_paral(3) ! Decomposition for each proc
  !
  INTEGER :: ierror, ios
  INTEGER, PARAMETER :: w_unit = 711
  INTEGER :: FILE_Debug=1
  !
  ! Names of exchanged Fields
  CHARACTER(len=8), DIMENSION(3), PARAMETER :: var_name = (/'ATSSTSST','ATSOLFLX','ATFLXEMP'/) ! 8 characters field
  !
  ! Used in oasis_def_var and oasis_def_var
  INTEGER                       :: var_id(3)
  INTEGER                       :: var_nodims(2) 
  INTEGER                       :: var_type
  !
  INTEGER                       :: niter, time_step, ib, it_sec
  !
  ! Grid parameters definition
  INTEGER                       :: part_id  ! use to connect the partition to the variables 
  INTEGER                       :: var_sh(4) ! local dimensions of the arrays; 2 x rank (=4)
  INTEGER :: ji, jj
  INTEGER :: auxfileid, auxdimid(2), auxvarid(2)
  !
  ! NEMO namelist parameters
  INTEGER                       :: numnam_cfg=80, nn_it000, nn_itend
  INTEGER                       :: nn_stocklist, nn_rstctl, nn_no
  LOGICAL                       :: ln_rst_list, ln_mskland  , ln_clobber,ln_cfmeta, ln_iscpl, ln_xios_read
  LOGICAL                       :: ln_rstart, nn_date0, nn_time0, nn_leapy  , nn_istate, nn_stock, nn_write ,nn_chunksz, nn_euler,nn_wxios
  CHARACTER (len=256)           :: cn_exp , cn_ocerst_in, cn_ocerst_indir, cn_ocerst_out, cn_ocerst_outdir
  REAL (kind=wp)                :: rn_rdt
  LOGICAL                       :: ln_linssh, ln_crs, ln_meshmask
  REAL (kind=wp)                :: rn_isfhmin, rn_atfp
  !
  ! NEMO namelists
!!  NAMELIST/namrun/ nn_it000, nn_itend
!!  NAMELIST/namdom/ rn_rdt
      NAMELIST/namrun/ cn_ocerst_indir, cn_ocerst_outdir, nn_stocklist, ln_rst_list,                 &
         &             nn_no   , cn_exp   , cn_ocerst_in, cn_ocerst_out, ln_rstart , nn_rstctl ,     &
         &             nn_it000, nn_itend , nn_date0    , nn_time0     , nn_leapy  , nn_istate ,     &
         &             nn_stock, nn_write , ln_mskland  , ln_clobber   , nn_chunksz, nn_euler  ,     &
         &             ln_cfmeta, ln_iscpl, ln_xios_read, nn_wxios
      NAMELIST/namdom/ ln_linssh, rn_isfhmin, rn_rdt, rn_atfp, ln_crs, ln_meshmask
  !
  !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !  INITIALISATION 
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !
  CALL oasis_init_comp (comp_id, comp_name, ierror )
  IF (ierror /= 0) THEN
      WRITE(0,*) 'oasis_init_comp abort by toyatm compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at oasis_init_comp')
  ENDIF
  !
  CALL oasis_get_localcomm ( localComm, ierror )
  IF (ierror /= 0) THEN
      WRITE (0,*) 'oasis_get_localcomm abort by toyatm compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at oasis_get_localcomm')
  ENDIF
  !
  ! Get MPI size and rank
  CALL MPI_Comm_Size ( localComm, npes, ierror )
  IF (ierror /= 0) THEN
      WRITE(0,*) 'MPI_comm_size abort by toyatm compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at MPI_Comm_Size')
  ENDIF
  !
  CALL MPI_Comm_Rank ( localComm, mype, ierror )
  IF (ierror /= 0) THEN
      WRITE (0,*) 'MPI_Comm_Rank abort by toyatm compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at MPI_Comm_Rank')
  ENDIF
  !
  IF (mype == 0) THEN
       FILE_Debug = 2
       comp_out=comp_name//'.root'
       OPEN(w_unit,file=TRIM(comp_out),form='formatted')
  ENDIF
  !
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) '-----------------------------------------------------------'
      WRITE(w_unit,*) TRIM(comp_name), ' running with reals compiled as kind ',wp
      WRITE(w_unit,*) '----------------------------------------------------------'
      WRITE (w_unit,*) 'Number of processors :',npes
      WRITE(w_unit,*) '----------------------------------------------------------'
      CALL FLUSH(w_unit)
  ENDIF
  !
  ! Simulation length definition (according to NEMO namelist_cfg)
  !
  OPEN (UNIT=numnam_cfg, FILE='namelist_cfg', STATUS='OLD' )
  READ  ( numnam_cfg, namrun, IOSTAT = ios )
  REWIND(numnam_cfg)
  READ  ( numnam_cfg, namdom, IOSTAT = ios )
  CLOSE(numnam_cfg)
  !
! Get time step and number of iterations from ocean
  time_step = INT(rn_rdt)
  niter = nn_itend - nn_it000 + 1 
  !
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) '-----------------------------------------------------------'
      WRITE (w_unit,*) 'Total time step # :', niter
      WRITE (w_unit,*) 'Simulation length :', niter*time_step
      WRITE(w_unit,*) '----------------------------------------------------------'
      CALL FLUSH(w_unit)
  ENDIF
  !
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !  GRID DEFINITION 
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !
  ! Reading global grids.nc and masks.nc netcdf files
  ! Get arguments giving source grid acronym and field type
  ! 
  cl_grd_src = 'lmdz'
  !
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'Source grid name : ',cl_grd_src
      CALL flush(w_unit)
  ENDIF
  !
  !
  ! Define global grid longitudes, latitudes, mask 
  DO jj = 1, nlat
     DO ji = 1, nlon
        gg_lon(ji ,jj) = ( ji - 1 ) * ( 360. / nlon )
        gg_lat(ji ,jj) = ( jj - 1 ) * ( 180. / nlon )
     ENDDO
  ENDDO

  gg_mask(:,:) = 0.

  ! Complete OASIS auxiliary files with yoy grid data
  !
  IF (mype == 0) THEN
     ! Define longitude and latitude
     CALL check_nf90( nf90_open( data_gridname, nf90_write, auxfileid ) )
     CALL check_nf90( nf90_redef( auxfileid ) )
     CALL check_nf90( nf90_def_dim( auxfileid, "toylon", nlon, auxdimid(1)) )
     CALL check_nf90( nf90_def_dim( auxfileid, "toylat", nlat, auxdimid(2)) )
     CALL check_nf90( nf90_def_var( auxfileid, cl_grd_src//'.lon', NF90_DOUBLE, auxdimid, auxvarid(1)))
     CALL check_nf90( nf90_def_var( auxfileid, cl_grd_src//'.lat', NF90_DOUBLE, auxdimid, auxvarid(2)))
     CALL check_nf90( nf90_enddef( auxfileid ) )
     CALL check_nf90( nf90_put_var( auxfileid, auxvarid(1), gg_lon ) )
     CALL check_nf90( nf90_put_var( auxfileid, auxvarid(2), gg_lat ) )
     CALL check_nf90( nf90_close( auxfileid ) )

     ! Define mask
     CALL check_nf90( nf90_open( data_maskname, nf90_write, auxfileid ) )
     CALL check_nf90( nf90_redef( auxfileid ) )
     CALL check_nf90( nf90_def_dim( auxfileid, "toylon", nlon, auxdimid(1)) )
     CALL check_nf90( nf90_def_dim( auxfileid, "toylat", nlat, auxdimid(2)) )
     CALL check_nf90( nf90_def_var( auxfileid, cl_grd_src//'.msk', NF90_INT, auxdimid, auxvarid(1)))
     CALL check_nf90( nf90_enddef( auxfileid ) )
     CALL check_nf90( nf90_put_var( auxfileid, auxvarid(1), gg_mask ) )
     CALL check_nf90( nf90_close( auxfileid ) )
  ENDIF
  !
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'After grid and mask reading'
      CALL FLUSH(w_unit)
  ENDIF
  !
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !  PARTITION DEFINITION 
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ !
  !
  il_paral(1) = 1  ! Apple decomposition
  il_paral(2) = mype * nlon * nlat / npes
  il_paral(3) = nlon * nlat / npes
  IF ( mype > ( npes - 1 ) ) &
     il_paral(3) =  nlon * nlat - ( mype * ( nlon * nlat / npes ) )
  !
  CALL oasis_def_partition (part_id, il_paral, ierror)
  !
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !  COUPLING LOCAL FIELD DECLARATION  
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !
  var_nodims(1) = 2    ! Rank of the field array is 2
  var_nodims(2) = 1    ! Bundles always 1 for OASIS3
  var_type = OASIS_Real
  !
  var_sh(1) = 1
  var_sh(2) = il_paral(3)
  var_sh(3) = 1 
  var_sh(4) = 1
  !
  ! Declaration of the field associated with the partition (recv)
  CALL oasis_def_var (var_id(1), var_name(1), part_id, &
                      var_nodims, OASIS_In, var_sh, var_type, ierror)
  IF (ierror /= 0) THEN
      WRITE(w_unit,*) 'oasis_def_var abort by toyatm compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at oasis_def_var')
  ENDIF

  ! Declaration of the field associated with the partition (send)
  CALL oasis_def_var (var_id(2), var_name(2), part_id, &
                      var_nodims, OASIS_Out, var_sh, var_type, ierror)
  IF (ierror /= 0) THEN
      WRITE(w_unit,*) 'oasis_def_var abort by toyatm compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at oasis_def_var')
  ENDIF
  CALL oasis_def_var (var_id(3), var_name(3), part_id, &
                      var_nodims, OASIS_Out, var_sh, var_type, ierror)
  IF (ierror /= 0) THEN
      WRITE(w_unit,*) 'oasis_def_var abort by toyatm compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at oasis_def_var')
  ENDIF
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'After def_var'
      CALL FLUSH(w_unit)
  ENDIF
  !
  !
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !  TERMINATION OF DEFINITION PHASE 
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !
  CALL oasis_enddef ( ierror )
  IF (ierror /= 0) THEN
      WRITE(w_unit,*) 'oasis_enddef abort by toyatm compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at oasis_enddef')
  ENDIF
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'After enddef'
      CALL FLUSH(w_unit)
  ENDIF
  !
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !  SEND ARRAYS 
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !
  ! Allocate the fields send and received by the model1
  !
  ALLOCATE(field_send(var_sh(2),var_sh(4)), STAT=ierror )
  IF ( ierror /= 0 ) WRITE(w_unit,*) 'Error allocating field_send'
  ALLOCATE(field_recv(var_sh(2),var_sh(4)), STAT=ierror )
  IF ( ierror /= 0 ) WRITE(w_unit,*) 'Error allocating field_recv'
  !
  DO ib=1, niter
    it_sec = time_step * (ib-1) ! Time
 
    ! QNS
    field_send(:,:) = 1. 
    !
    CALL oasis_put(var_id(2), it_sec, field_send, ierror )
    ! EMPs
    field_send(:,:) = 10./ 86400.
    CALL oasis_put(var_id(3), it_sec, field_send, ierror )
    ! SST
    CALL oasis_get(var_id(1), it_sec, &
                   field_recv, &
                   ierror )
    !
  END DO
  !
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !         TERMINATION 
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  IF (FILE_Debug >= 2) THEN
      WRITE(w_unit,*) 'End of the program, before oasis_terminate'
      CALL FLUSH(w_unit)
  ENDIF
  !
  CALL oasis_terminate (ierror)
  IF (ierror /= 0) THEN
      WRITE(w_unit,*) 'oasis_terminate abort by toyatm compid ',comp_id
      CALL oasis_abort(comp_id,comp_name,'Problem at oasis_terminate')
  ENDIF
  !
CONTAINS


   SUBROUTINE check_nf90(status, errorFlag)
   !---------------------------------------------------------------------
   !  Checks return code from nf90 library calls and warns if needed
   !  If errorFlag is present then it just increments this flag (OMP use)
   !
   !---------------------------------------------------------------------
      INTEGER, INTENT(IN   ) :: status
      INTEGER, INTENT(INOUT), OPTIONAL :: errorFlag
   !---------------------------------------------------------------------

      IF( status /= nf90_noerr ) THEN
         WRITE(w_unit,*) 'ERROR! : '//TRIM(nf90_strerror(status))
         IF( PRESENT( errorFlag ) ) THEN
            errorFlag = errorFlag + status
         ELSE
            WRITE(w_unit,*) "*** TOYATM failed on netcdf ***"
            WRITE(w_unit,*)
            STOP 5
         ENDIF
      ENDIF

   END SUBROUTINE check_nf90
  !
END PROGRAM TOYATM
!
