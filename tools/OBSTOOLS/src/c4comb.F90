PROGRAM c4comb
   !!---------------------------------------------------------------------
   !!
   !!                     ** PROGRAM c4comb **
   !!
   !!  ** Purpose : Combine MPI decomposed class4 files into one file
   !!
   !!  ** Method  : Use of utilities from obs_utils, ooo_utils.
   !!
   !!  ** Action  : 
   !!
   !!   Usage:
   !!     c4comb.exe outputfile inputfile1 inputfile2 ...
   !!
   !!   History :
   !!----------------------------------------------------------------------
   USE netcdf
   USE obs_const
   USE obs_utils
   USE ooo_utils, ONLY: date_format, obfilldbl
   USE toolspar_kind
   IMPLICIT NONE
   !! Command line setup
#ifndef NOIARGCPROTO
   INTEGER,EXTERNAL :: iargc
#endif
   INTEGER :: nargs,    & !: number of command line arguments
            & ia,       & !: argument loop index
            & ninfiles    !: number of input files
   !! Routine arguments
   CHARACTER(len=256) :: cdoutfile
   CHARACTER(len=256),ALLOCATABLE :: cdinfile(:)
   !! Routine variables
   CHARACTER(len=80) :: cpname
   INTEGER,PARAMETER :: nstr=8, n128=128
   INTEGER           ::    ncid,   & !: netcdf file id
                        &  dimid,  & !: netcdf dimension id
                        &  dpdim,  & !: netcdf dimension ids
                        &  fcdim,  &
                        &  vrdim,  &
                        &  obdim,  &
                        &  stdim,  &
                        &  sxdim,  &
                        &  fdvid,  & !: netcdf variable ids
                        &  lonid,  &
                        &  latid,  &
                        &  depid,  &
                        &  varid,  &
                        &  unitid, &
                        &  obvid,  &
                        &  fcvid,  &
                        &  prvid,  &
                        &  clvid,  &
                        &  dm2id,  &
                        &  dm1id,  &
                        &  mdtid,  &
                        &  altid,  &
                        &  qcvid,  &
                        &  jdvid,  &
                        &  mjdid,  &
                        &  typid,  &
                        &  idvid,  &
                        &  ndeps,  & !: number depths
                        &  nfcst,  & !: number forecast
                        &  nvars,  & !: number variables
                        &  nobs,   & !: number obs
                        &  sdeps,  &
                        &  sobs,   &
                        &  l_dex,  &
                        &  u_dex

   INTEGER                   :: iob, idep, istat
   INTEGER, DIMENSION(2)     :: dim2a, dim2b, dim2c, dim2d
   INTEGER, DIMENSION(3)     :: dim3a
   INTEGER, DIMENSION(4)     :: dim4a
   INTEGER,  ALLOCATABLE, DIMENSION(:)           :: fcday
   REAL(wp), ALLOCATABLE, DIMENSION(:)           :: modjd
   !: Global Attributes
   CHARACTER(len=40)                             :: nam_str, &
                                                 & version,  &
                                                 & contact,  &
                                                 & sys_str,  &
                                                 & cfg_str,  &
                                                 & ins_str,  &
                                                 & val_str,  &
                                                 & dat_str,  &
                                                 & obs_str
   !: Variable Attributes
   CHARACTER(len=100)                            :: lon_units, &
                                                 &  lat_units, &
                                                 &  dep_units, &
                                                 &  jul_units, &
                                                 &  mjd_units, &
                                                 &  fcd_units, &
                                                 &  lead_comment, &
                                                 &  fcst_comment, &
                                                 &  per_comment,  &
                                                 &  cli_comment,  &
                                                 &  dm2_comment,  &
                                                 &  dm1_comment
   CHARACTER(len=128)                            :: qc_comment,   &
                                                 &  qc_flag_meaning
   INTEGER, DIMENSION(2)                         :: qc_flag_value

   !: Global Arrays
   REAL(wp), ALLOCATABLE, DIMENSION(:)           :: g_lam, &
                                                 &  g_phi, & 
                                                 &  gjuld 
   CHARACTER(len=n128),ALLOCATABLE,DIMENSION(:)  :: gtype
   CHARACTER(len=nstr),ALLOCATABLE,DIMENSION(:)  ::        &
                                                 &  g_id,  & 
                                                 &  gvnam, & 
                                                 &  gunit
   REAL(wp), ALLOCATABLE, DIMENSION(:,:)         :: g_dep
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:)       :: g3dob, &
                                                 &  g3dcl, &
                                                 &  g3mdt, &
                                                 &  g3alt, &
                                                 &  g3dm2, &
                                                 &  g3dm1
   INTEGER(ik),  ALLOCATABLE, DIMENSION(:,:,:)   :: g3dqc
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:,:)     :: g3dmc, &
                                                 &  g3dpr
   !: Small Arrays  
   REAL(wp), ALLOCATABLE, DIMENSION(:)       ::     s_lam, &
                                                 &  s_phi, & 
                                                 &  sjuld

   CHARACTER(len=n128),ALLOCATABLE,DIMENSION(:) ::  stype
   CHARACTER(len=nstr),ALLOCATABLE,DIMENSION(:) ::        &
                                                 &  s_id 

   REAL(wp), ALLOCATABLE, DIMENSION(:,:)        :: s_dep
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:)      :: s3dob, &
                                                &  s3dcl, &
                                                &  s3mdt, &
                                                &  s3alt, &
                                                &  s3dm2, &
                                                &  s3dm1
   INTEGER(ik),  ALLOCATABLE, DIMENSION(:,:,:)  :: s3dqc
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:,:)    :: s3dmc, &
                                                &  s3dpr

   !: File creation logical
   LOGICAL                                     :: ln_cre

   !: Optional variable logicals
   LOGICAL                                     :: ln_init, &
                                                & ln_mdt,  &
                                                & ln_altbias, &
                                                & ln_best
   !! Command name
   cpname='c4comb.exe'

   !! Process command line
   nargs = IARGC()
   IF (nargs /= 2) THEN
      WRITE(*, *) "Usage: c4comb.exe outputfile inputfile1 inputfile2 ..."
      CALL abort()
   END IF
   CALL GETARG(1, cdoutfile)

   !! Process input files
   !! Set output file creation to off
   ln_cre = .false.

   !! Turn optional variables off
   ln_init = .false.
   ln_best = .false.
   ln_altbias = .false.
   ln_mdt = .false.

   !! Compute size of output file
   nobs = 0
   ndeps= 0
   ALLOCATE( cdinfile( nargs - 1 ) )
   ninfiles = nargs - 1
   DO ia = 1, ninfiles
      CALL GETARG(ia+1, cdinfile(ia))
      WRITE(*,*) "Opening : ", TRIM(cdinfile(ia))
      !! Open Netcdf file
      istat = nf90_open(TRIM(cdinfile(ia)),nf90_nowrite,ncid)
      IF (istat == nf90_noerr) THEN
         !! Turn output file creation on
         ln_cre = .true.
         !! Get Dimensions
         CALL chkerr( nf90_inq_dimid(ncid, 'numobs',  dimid),              cpname, __LINE__ )
         CALL chkerr( nf90_inquire_dimension(ncid,    dimid, len=sobs  ),  cpname, __LINE__ )
         CALL chkerr( nf90_inq_dimid(ncid, 'numdeps', dimid),              cpname, __LINE__ )
         CALL chkerr( nf90_inquire_dimension(ncid,    dimid, len=sdeps ),  cpname, __LINE__ )
         CALL chkerr( nf90_inq_dimid(ncid, 'numfcsts',dimid),              cpname, __LINE__ )
         CALL chkerr( nf90_inquire_dimension(ncid,    dimid, len=nfcst ),  cpname, __LINE__ )
         CALL chkerr( nf90_inq_dimid(ncid, 'numvars', dimid),              cpname, __LINE__ )
         CALL chkerr( nf90_inquire_dimension(ncid,    dimid, len=nvars ),  cpname, __LINE__ )
         !! Close Netcdf file
         CALL chkerr( nf90_close(ncid), cpname, __LINE__ )
         !! Report on file contents
         WRITE(*,'(2A)')'File = ', TRIM(cdinfile(ia))
         WRITE(*,'(A,I9,A)')'has', sobs, ' observations'
         !! Increment size
         nobs  = nobs + sobs       !: Accumulate number of profiles
         ndeps = MAX(ndeps, sdeps) !: Define maximum number of levels needed
      END IF ! istat
   END DO

   !! Allocate global arrays
   ALLOCATE( g_phi(nobs),                      &
          &  g_lam(nobs),                      &
          &  g_dep(ndeps, nobs),               &
          &  g3dob(ndeps,        nvars, nobs), &
          &  g3dmc(ndeps, nfcst, nvars, nobs), &
          &  g3dpr(ndeps, nfcst, nvars, nobs), &
          &  g3dcl(ndeps,        nvars, nobs), &
          &  g3dm2(ndeps,        nvars, nobs), &
          &  g3dm1(ndeps,        nvars, nobs), &
          &  g3mdt(ndeps,        nvars, nobs), &
          &  g3alt(ndeps,        nvars, nobs), &
          &  g3dqc(ndeps,        nvars, nobs), &
          &  gjuld(nobs),                      &
          &  gtype(nobs),                      &
          &  g_id(nobs),                       &
          &  gvnam(nvars),                     &
          &  gunit(nvars) )
   ALLOCATE(fcday(nfcst), modjd(nfcst))

   !! Fill with missing data value
   g_dep(:,:)     = 99999.
   g3dmc(:,:,:,:) = 99999.
   g3dpr(:,:,:,:) = 99999.
   g3dob(:,:,:)   = 99999.
   g3dcl(:,:,:)   = 99999.
   g3dm2(:,:,:)   = 99999.
   g3dm1(:,:,:)   = 99999.
   g3mdt(:,:,:)   = 99999.
   g3alt(:,:,:)   = 99999.
   g3dqc(:,:,:)   = NF90_FILL_SHORT

   !! Read in each file
   !  initialise global matrix indices
   l_dex = 0
   u_dex = 0

   !! initialise Global attribute strings
   nam_str = ''
   version = ''
   contact = ''
   sys_str = ''
   cfg_str = ''
   ins_str = ''
   val_str = ''
   dat_str = ''
   obs_str = ''

   !! initialise Variable attribute strings
   fcd_units = ''
   lon_units = ''
   lat_units = ''
   dep_units = ''
   jul_units = ''
   mjd_units = ''
   lead_comment = ''
   fcst_comment = ''
   per_comment  = ''
   cli_comment  = ''
   dm2_comment  = ''
   dm1_comment  = ''
   qc_comment  = ''
   qc_flag_meaning  = ''

   DO ia = 1, ninfiles
      WRITE(*,*) "Opening : ", TRIM(cdinfile(ia))
      !! Open Netcdf file
      istat = nf90_open(TRIM(cdinfile(ia)),nf90_nowrite,ncid)
      IF (istat == nf90_noerr) THEN
         !! Get Global Attributes
         CALL chkerr( nf90_get_att(ncid, nf90_global,'title',         nam_str),cpname, __LINE__)
         CALL chkerr( nf90_get_att(ncid, nf90_global,'version',       version),cpname, __LINE__)
         CALL chkerr( nf90_get_att(ncid, nf90_global,'contact',       contact),cpname, __LINE__)
         CALL chkerr( nf90_get_att(ncid, nf90_global,'obs_type',      obs_str),cpname, __LINE__)
         CALL chkerr( nf90_get_att(ncid, nf90_global,'system',        sys_str),cpname, __LINE__)
         CALL chkerr( nf90_get_att(ncid, nf90_global,'configuration', cfg_str),cpname, __LINE__)
         CALL chkerr( nf90_get_att(ncid, nf90_global,'institution',   ins_str),cpname, __LINE__)
         CALL chkerr( nf90_get_att(ncid, nf90_global,'validity_time', val_str),cpname, __LINE__)
         !! Get Dimensions of single file
         CALL chkerr( nf90_inq_dimid(ncid, 'numdeps', dimid),              cpname, __LINE__ )
         CALL chkerr( nf90_inquire_dimension(ncid,    dimid, len=sdeps ),  cpname, __LINE__ )
         CALL chkerr( nf90_inq_dimid(ncid, 'numfcsts',dimid),              cpname, __LINE__ )
         CALL chkerr( nf90_inquire_dimension(ncid,    dimid, len=nfcst ),  cpname, __LINE__ )
         CALL chkerr( nf90_inq_dimid(ncid, 'numvars', dimid),              cpname, __LINE__ )
         CALL chkerr( nf90_inquire_dimension(ncid,    dimid, len=nvars ),  cpname, __LINE__ )
         CALL chkerr( nf90_inq_dimid(ncid, 'numobs',  dimid),              cpname, __LINE__ )
         CALL chkerr( nf90_inquire_dimension(ncid,    dimid, len=sobs  ),  cpname, __LINE__ )
         !! Check for Optional variables in first file
         IF (ia == 1) THEN
            !! Best estimate
            istat = nf90_inq_varid(ncid,'best_estimate',dm2id)
            IF (istat == nf90_noerr) THEN
               ln_best = .TRUE.
            ENDIF 
            !! nrt_analysis
            istat = nf90_inq_varid(ncid,'nrt_analysis',dm1id)
            IF (istat == nf90_noerr) THEN
               ln_init = .TRUE.
            ENDIF 
            !! Mean Dynamic Topography
            istat = nf90_inq_varid(ncid,'mdt_reference',mdtid)
            IF (istat == nf90_noerr) THEN
               ln_mdt = .TRUE.
            ENDIF 
            !! Altimeter bias
            istat = nf90_inq_varid(ncid,'altimeter_bias',altid)
            IF (istat == nf90_noerr) THEN
               ln_altbias = .TRUE.
            ENDIF 
         END IF
         WRITE(*,*) TRIM(cdinfile(ia)), " contains ", sobs, " observations"
         WRITE(*,*) TRIM(cdinfile(ia)), " contains ", sdeps, " depths"
         WRITE(*,*) TRIM(cdinfile(ia)), " contains ", nfcst, " forecasts"
         WRITE(*,*) TRIM(cdinfile(ia)), " contains ", nvars, " vars"
         !! Read Variables
         IF (sobs /= 0) THEN
            !! Get Variable ids
            CALL chkerr(nf90_inq_varid(ncid,'leadtime', fdvid) ,cpname, __LINE__ )
            CALL chkerr(nf90_inq_varid(ncid,'longitude',   lonid) ,cpname, __LINE__ )
            CALL chkerr(nf90_inq_varid(ncid,'latitude',    latid) ,cpname, __LINE__ )
            CALL chkerr(nf90_inq_varid(ncid,'depth',       depid) ,cpname, __LINE__ )
            CALL chkerr(nf90_inq_varid(ncid,'varname',     varid) ,cpname, __LINE__ )
            CALL chkerr(nf90_inq_varid(ncid,'unitname',   unitid) ,cpname, __LINE__ )
            CALL chkerr(nf90_inq_varid(ncid,'observation', obvid) ,cpname, __LINE__ )
            CALL chkerr(nf90_inq_varid(ncid,'forecast',    fcvid) ,cpname, __LINE__ )
            CALL chkerr(nf90_inq_varid(ncid,'persistence', prvid) ,cpname, __LINE__ )
            CALL chkerr(nf90_inq_varid(ncid,'climatology', clvid) ,cpname, __LINE__ )
            CALL chkerr(nf90_inq_varid(ncid,'qc',          qcvid) ,cpname, __LINE__ )
            CALL chkerr(nf90_inq_varid(ncid,'juld',        jdvid) ,cpname, __LINE__ )
            CALL chkerr(nf90_inq_varid(ncid,'modeljuld',   mjdid) ,cpname, __LINE__ )
            CALL chkerr(nf90_inq_varid(ncid,'type',        typid) ,cpname, __LINE__ )
            CALL chkerr(nf90_inq_varid(ncid,'id',          idvid) ,cpname, __LINE__ )
            !! Get variable attributes
            CALL chkerr(nf90_get_att(ncid, fdvid, 'units', fcd_units) ,cpname, __LINE__ )
            CALL chkerr(nf90_get_att(ncid, lonid, 'units', lon_units) ,cpname, __LINE__ )
            CALL chkerr(nf90_get_att(ncid, latid, 'units', lat_units) ,cpname, __LINE__ )
            CALL chkerr(nf90_get_att(ncid, depid, 'units', dep_units) ,cpname, __LINE__ )
            CALL chkerr(nf90_get_att(ncid, jdvid, 'units', jul_units) ,cpname, __LINE__ )
            CALL chkerr(nf90_get_att(ncid, mjdid, 'units', mjd_units) ,cpname, __LINE__ )
            CALL chkerr(nf90_get_att(ncid, fcvid, 'comment', fcst_comment) ,cpname, __LINE__ )
            CALL chkerr(nf90_get_att(ncid, prvid, 'comment', per_comment)  ,cpname, __LINE__ )
            CALL chkerr(nf90_get_att(ncid, clvid, 'comment', cli_comment)  ,cpname, __LINE__ )
            CALL chkerr(nf90_get_att(ncid, fdvid, 'comment', lead_comment) ,cpname, __LINE__ )
            CALL chkerr(nf90_get_att(ncid, qcvid, 'comment', qc_comment) ,cpname, __LINE__ )
            CALL chkerr(nf90_get_att(ncid, qcvid, 'flag_value', qc_flag_value) ,cpname, __LINE__ )
            CALL chkerr(nf90_get_att(ncid, qcvid, 'flag_meaning', qc_flag_meaning) ,cpname, __LINE__ )
            !! Optional variables
            IF (ln_best) THEN
               CALL chkerr(nf90_inq_varid(ncid,'best_estimate',dm2id) ,cpname, __LINE__ )
               CALL chkerr(nf90_get_att(ncid, dm2id, 'comment', dm2_comment)  ,cpname, __LINE__ )
            ENDIF
            IF (ln_init) THEN
               CALL chkerr(nf90_inq_varid(ncid,'nrt_analysis', dm1id) ,cpname, __LINE__ )
               CALL chkerr(nf90_get_att(ncid, dm1id, 'comment', dm1_comment)  ,cpname, __LINE__ )
            ENDIF
            IF (ln_mdt) THEN
               CALL chkerr(nf90_inq_varid(ncid,'mdt_reference', mdtid) ,cpname, __LINE__ )
            ENDIF
            IF (ln_altbias) THEN
               CALL chkerr(nf90_inq_varid(ncid,'altimeter_bias', altid) ,cpname, __LINE__ )
            ENDIF

            !! Allocate small arrays
            ALLOCATE( s_lam(sobs), s_phi(sobs),  s_dep(sdeps, sobs),               &
                   &  s3dob(sdeps,        nvars, sobs), & !: observations 
                   &  s3dmc(sdeps, nfcst, nvars, sobs), & !: model data
                   &  s3dpr(sdeps, nfcst, nvars, sobs), & !: persistence
                   &  s3dcl(sdeps,        nvars, sobs), & !: climatology
                   &  s3dm2(sdeps,        nvars, sobs), & !: best estimate
                   &  s3dm1(sdeps,        nvars, sobs), & !: nrt_analysis
                   &  s3mdt(sdeps,        nvars, sobs), & !: mdt
                   &  s3alt(sdeps,        nvars, sobs), & !: altbias
                   &  s3dqc(sdeps,        nvars, sobs), & !: QC
                   &  sjuld(sobs), stype( sobs),  &
                   &  s_id(sobs) )
            !! Fill with missing data value
            s3dmc(:,:,:,:) = 99999.
            s3dpr(:,:,:,:) = 99999.
            s3dob(:,:,:)   = 99999.
            s3dcl(:,:,:)   = 99999.
            s3dm2(:,:,:)   = 99999.
            s3dm1(:,:,:)   = 99999.
            s3mdt(:,:,:)   = 99999.
            s3alt(:,:,:)   = 99999.
            s3dqc(:,:,:)   = NF90_FILL_SHORT

            !! Read variables into small arrays
            CALL chkerr( nf90_get_var(ncid, fdvid, fcday), cpname, __LINE__ )       
            CALL chkerr( nf90_get_var(ncid, lonid, s_lam), cpname, __LINE__ )       
            CALL chkerr( nf90_get_var(ncid, latid, s_phi), cpname, __LINE__ )       
            CALL chkerr( nf90_get_var(ncid, depid, s_dep), cpname, __LINE__ )       
            CALL chkerr( nf90_get_var(ncid, obvid, s3dob), cpname, __LINE__ )       
            CALL chkerr( nf90_get_var(ncid, fcvid, s3dmc), cpname, __LINE__ )       
            CALL chkerr( nf90_get_var(ncid, prvid, s3dpr), cpname, __LINE__ )       
            CALL chkerr( nf90_get_var(ncid, clvid, s3dcl), cpname, __LINE__ )       
            CALL chkerr( nf90_get_var(ncid, qcvid, s3dqc), cpname, __LINE__ )       
            CALL chkerr( nf90_get_var(ncid, jdvid, sjuld), cpname, __LINE__ )       
            CALL chkerr( nf90_get_var(ncid, mjdid, modjd), cpname, __LINE__ )       
            CALL chkerr( nf90_get_var(ncid, typid, stype), cpname, __LINE__ )       
            CALL chkerr( nf90_get_var(ncid, idvid, s_id),  cpname, __LINE__ )       
            !! Read unitname and varname into global arrays
            CALL chkerr( nf90_get_var(ncid, varid, gvnam), cpname, __LINE__ )       
            CALL chkerr( nf90_get_var(ncid, unitid,gunit), cpname, __LINE__ )       
            !! Optional variables read
            IF (ln_best) THEN
               CALL chkerr( nf90_get_var(ncid, dm2id, s3dm2), cpname, __LINE__ )       
            ENDIF
            IF (ln_init) THEN
               CALL chkerr( nf90_get_var(ncid, dm1id, s3dm1), cpname, __LINE__ )       
            ENDIF
            IF (ln_mdt) THEN
               CALL chkerr( nf90_get_var(ncid, mdtid, s3mdt), cpname, __LINE__ )       
            ENDIF
            IF (ln_altbias) THEN
               CALL chkerr( nf90_get_var(ncid, altid, s3alt), cpname, __LINE__ )       
            ENDIF

            !! Fill Global arrays
            !  increment numobs indices
            l_dex = u_dex + 1
            u_dex = l_dex + sobs -1

            g_lam(l_dex:u_dex)                          = s_lam(:)
            g_phi(l_dex:u_dex)                          = s_phi(:)
            g_dep(1:sdeps,  l_dex:u_dex)                = s_dep(1:sdeps,:)
            g3dob(1:sdeps,1:nvars,l_dex:u_dex)          = s3dob(1:sdeps,1:nvars,:)
            g3dmc(1:sdeps,1:nfcst,1:nvars,l_dex:u_dex)  = s3dmc(1:sdeps,1:nfcst,1:nvars,:)
            g3dpr(1:sdeps,1:nfcst,1:nvars,l_dex:u_dex)  = s3dpr(1:sdeps,1:nfcst,1:nvars,:)
            g3dcl(1:sdeps,1:nvars,l_dex:u_dex)          = s3dcl(1:sdeps,1:nvars,:)
            g3dm2(1:sdeps,1:nvars,l_dex:u_dex)          = s3dm2(1:sdeps,1:nvars,:)
            g3dm1(1:sdeps,1:nvars,l_dex:u_dex)          = s3dm1(1:sdeps,1:nvars,:)
            g3mdt(1:sdeps,1:nvars,l_dex:u_dex)          = s3mdt(1:sdeps,1:nvars,:)
            g3alt(1:sdeps,1:nvars,l_dex:u_dex)          = s3alt(1:sdeps,1:nvars,:)
            g3dqc(1:sdeps,1:nvars,l_dex:u_dex)          = s3dqc(1:sdeps,1:nvars,:)
            gjuld(l_dex:u_dex)                          = sjuld(:)
            gtype(l_dex:u_dex)                          = stype(:)
            g_id(l_dex:u_dex)                           = s_id(:)

            !! Deallocate small array
            DEALLOCATE( s_lam, s_phi, s_dep, s3dob, s3dmc, s3dpr, s3dcl, s3dqc, s3dm2, s3dm1, s3mdt, s3alt, sjuld, stype, s_id)
         ENDIF ! sobs    
         !! Close Netcdf file
         CALL chkerr( nf90_close(ncid), cpname, __LINE__ )
      END IF ! istat
   END DO

   !! Create Output file
   IF (ln_cre) THEN
      WRITE(*,*) 'Create the output file, ',trim(cdoutfile)
      CALL chkerr( nf90_create(trim(cdoutfile),nf90_clobber,ncid),      cpname, __LINE__ )
      !! Put Global Attributes
      CALL date_format(dat_str) 
      CALL chkerr( nf90_put_att(ncid, nf90_global,'title',          trim(nam_str)),cpname, __LINE__)
      CALL chkerr( nf90_put_att(ncid, nf90_global,'version',        trim(version)),cpname, __LINE__)
      CALL chkerr( nf90_put_att(ncid, nf90_global,'creation_date',  trim(dat_str)),cpname, __LINE__)
      CALL chkerr( nf90_put_att(ncid, nf90_global,'contact',        trim(contact)),cpname, __LINE__)
      CALL chkerr( nf90_put_att(ncid, nf90_global,'obs_type',       trim(obs_str)),cpname, __LINE__)
      CALL chkerr( nf90_put_att(ncid, nf90_global,'system',         trim(sys_str)),cpname, __LINE__)
      CALL chkerr( nf90_put_att(ncid, nf90_global,'configuration',  trim(cfg_str)),cpname, __LINE__)
      CALL chkerr( nf90_put_att(ncid, nf90_global,'institution',    trim(ins_str)),cpname, __LINE__)
      CALL chkerr( nf90_put_att(ncid, nf90_global,'validity_time',  trim(val_str)),cpname, __LINE__)
      CALL chkerr( nf90_put_att(ncid, nf90_global,'best_estimate_description', &
                              &       'analysis produced 2 days behind real time'),cpname, __LINE__)
      CALL chkerr( nf90_put_att(ncid, nf90_global,'time_interp', 'daily average fields'),cpname, __LINE__)
      WRITE(*,*) 'Succesfully put global attributes '

      !! Define Dimensions
      CALL chkerr( nf90_def_dim(ncid, 'numdeps',          ndeps, dpdim) ,cpname, __LINE__ )
      CALL chkerr( nf90_def_dim(ncid, 'numfcsts',         nfcst, fcdim) ,cpname, __LINE__ )
      CALL chkerr( nf90_def_dim(ncid, 'numvars',          nvars, vrdim) ,cpname, __LINE__ )
      CALL chkerr( nf90_def_dim(ncid, 'numobs',           nobs,  obdim) ,cpname, __LINE__ )
      CALL chkerr( nf90_def_dim(ncid, 'string_length8',   nstr,  stdim) ,cpname, __LINE__ )
      CALL chkerr( nf90_def_dim(ncid, 'string_length128', n128,  sxdim) ,cpname, __LINE__ )
      WRITE(*,*) 'Succesfully defined dimensions'

      !! Define possible dimension permutations
      ! 2d
      dim2a(:) = (/ dpdim, obdim /) !: (/ ndeps, nobs  /) 
      dim2b(:) = (/ stdim, obdim /) !: (/ nstr,  nobs  /)
      dim2c(:) = (/ stdim, vrdim /) !: (/ nstr,  nvars /)
      dim2d(:) = (/ sxdim, obdim /) !: (/ nstr,  nobs  /)
      ! 3d
      dim3a(:) = (/ dpdim, vrdim, obdim/) !: (/ ndeps, nvars, nobs /)
      ! 4d
      dim4a(:) = (/ dpdim, fcdim, vrdim, obdim /) !: (/ ndeps, nfcst, nvars, nobs /)


      !! Create the variables
      !  Forecast day
      CALL chkerr( nf90_def_var(ncid, 'leadtime',  nf90_double, fcdim, fdvid)          ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, fdvid, 'long_name', 'Model forecast day offset') ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, fdvid, 'units',     trim(fcd_units))             ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, fdvid, 'comment',   trim(lead_comment))          ,cpname, __LINE__ )
      WRITE(*,*) 'leadtime created'
      !  longitude
      CALL chkerr( nf90_def_var(ncid, 'longitude',    nf90_float, obdim, lonid)        ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, lonid, 'long_name', 'Longitudes')                ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, lonid, 'units',     trim(lon_units))             ,cpname, __LINE__ )
      WRITE(*,*) 'lon created'
      !  latitude
      CALL chkerr( nf90_def_var(ncid, 'latitude',     nf90_float, obdim, latid)        ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, latid, 'long_name', 'Latitudes')                 ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, latid, 'units',     trim(lat_units))             ,cpname, __LINE__ )
      WRITE(*,*) 'lat created'
      !  depth
      CALL chkerr( nf90_def_var(ncid, 'depth',        nf90_float, dim2a, depid)        ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, depid, 'long_name', 'Depths')                    ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, depid, 'units',     trim(dep_units))             ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, depid, '_FillValue',obfillflt)                    ,cpname, __LINE__ )
      WRITE(*,*) 'dep created'
      !  varname
      CALL chkerr( nf90_def_var(ncid, 'varname',      nf90_char,  dim2c, varid)        ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, varid, 'long_name', 'Variable name')             ,cpname, __LINE__ )
      WRITE(*,*) 'varname created'
      !  unitname
      CALL chkerr( nf90_def_var(ncid, 'unitname',     nf90_char,  dim2c, unitid)        ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid,  unitid, 'long_name', 'Unit name')              ,cpname, __LINE__ )
      WRITE(*,*) 'unitname created'
      !  obs
      CALL chkerr( nf90_def_var(ncid, 'observation', nf90_float,  dim3a,  obvid)        ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, obvid, '_FillValue',obfillflt)                    ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid,  obvid, 'long_name', 'Observation value')         ,cpname, __LINE__ )
      WRITE(*,*) 'obs created'
      !  forecast
      CALL chkerr( nf90_def_var(ncid, 'forecast',    nf90_float,  dim4a,  fcvid)        ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, fcvid, '_FillValue',obfillflt)                    ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, fcvid, 'long_name', 'Model forecast counterpart of obs. value')   ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, fcvid, 'comment',   trim(fcst_comment))           ,cpname, __LINE__ )
      WRITE(*,*) 'forecast created'
      !  persistence
      CALL chkerr( nf90_def_var(ncid, 'persistence', nf90_float,  dim4a,  prvid)        ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, prvid, '_FillValue',obfillflt)                    ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, prvid, 'long_name', 'Model persistence counterpart of obs. value'),cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, prvid, 'comment',   trim(per_comment))            ,cpname, __LINE__ )
      WRITE(*,*) 'persistence created'
      !  clim
      CALL chkerr( nf90_def_var(ncid, 'climatology', nf90_float,  dim3a, clvid)        ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, clvid, '_FillValue',obfillflt)                    ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, clvid, 'long_name', 'Climatological value')       ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, clvid, 'comment',   trim(cli_comment))            ,cpname, __LINE__ )
      WRITE(*,*) 'clim created'
      IF (ln_best) THEN
         !  daym2
         CALL chkerr( nf90_def_var(ncid, 'best_estimate', nf90_float,  dim3a, dm2id)    ,cpname, __LINE__ )
         CALL chkerr( nf90_put_att(ncid, dm2id, '_FillValue',obfillflt)                 ,cpname, __LINE__ )
         CALL chkerr( nf90_put_att(ncid, dm2id, 'long_name', 'Best estimate')           ,cpname, __LINE__ )
         CALL chkerr( nf90_put_att(ncid, dm2id, 'comment',   trim(dm2_comment))         ,cpname, __LINE__ )
         WRITE(*,*) 'daym2 created'
      ENDIF
      IF (ln_init) THEN
         !  daym1
         CALL chkerr( nf90_def_var(ncid, 'nrt_analysis', nf90_float,  dim3a, dm1id)        ,cpname, __LINE__ )
         CALL chkerr( nf90_put_att(ncid, dm1id, '_FillValue',obfillflt)                    ,cpname, __LINE__ )
         CALL chkerr( nf90_put_att(ncid, dm1id, 'long_name', 'Near real time analysis')    ,cpname, __LINE__ )
         CALL chkerr( nf90_put_att(ncid, dm1id, 'comment',   trim(dm1_comment))            ,cpname, __LINE__ )
         WRITE(*,*) 'daym1 created'
      ENDIF
      IF (ln_mdt) THEN
         !  mdt
         CALL chkerr( nf90_def_var(ncid, 'mdt_reference', nf90_float,  dim3a, mdtid)      ,cpname, __LINE__ )
         CALL chkerr( nf90_put_att(ncid, mdtid, '_FillValue',obfillflt)                    ,cpname, __LINE__ )
         CALL chkerr( nf90_put_att(ncid, mdtid, 'long_name', 'Mean dynamic topography')    ,cpname, __LINE__ )
         WRITE(*,*) 'mdt created'
      ENDIF
      IF (ln_altbias) THEN
         !  altbias
         CALL chkerr( nf90_def_var(ncid, 'altimeter_bias', nf90_float,  dim3a, altid)     ,cpname, __LINE__ )
         CALL chkerr( nf90_put_att(ncid, altid, '_FillValue',obfillflt)                    ,cpname, __LINE__ )
         CALL chkerr( nf90_put_att(ncid, altid, 'long_name', 'Altimeter bias')             ,cpname, __LINE__ )
         WRITE(*,*) 'altbias created'
      ENDIF
      !  qc
      CALL chkerr( nf90_def_var(ncid, 'qc',          nf90_short,  dim3a, qcvid)         ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, qcvid, '_FillValue', NF90_FILL_SHORT)             ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, qcvid, 'long_name', 'Quality flags')              ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, qcvid, 'flag_value', qc_flag_value)               ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, qcvid, 'flag_meaning', qc_flag_meaning)           ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, qcvid, 'comment', qc_comment)                     ,cpname, __LINE__ )
      WRITE(*,*) 'qc created'
      !  juld
      CALL chkerr( nf90_def_var(ncid, 'juld',       nf90_double,  obdim, jdvid)         ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, jdvid, '_FillValue',99999.)                       ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, jdvid, 'long_name', 'Observation time in Julian days'),cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, jdvid, 'units',     trim(jul_units))              ,cpname, __LINE__ )
      WRITE(*,*) 'juld created'
      !  modeljuld 
      CALL chkerr( nf90_def_var(ncid, 'modeljuld',  nf90_double,  fcdim, mjdid)         ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, mjdid, 'long_name', 'Model field date in Julian days'),cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, mjdid, 'units',     trim(mjd_units))              ,cpname, __LINE__ )
      WRITE(*,*) 'modeljuld created'
      !  type
      CALL chkerr( nf90_def_var(ncid, 'type',     nf90_char,      dim2d, typid)         ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, typid, 'long_name', 'Observation type')           ,cpname, __LINE__ )
      WRITE(*,*) 'type created'
      !  id
      CALL chkerr( nf90_def_var(ncid, 'id',       nf90_char,      dim2b, idvid)         ,cpname, __LINE__ )
      CALL chkerr( nf90_put_att(ncid, idvid, 'long_name', 'Observation id')             ,cpname, __LINE__ )
      WRITE(*,*) 'id created'
      ! Close Netcdf file 
      CALL chkerr( nf90_close(ncid) ,cpname, __LINE__ )  
      !! Fill in the variables
      CALL chkerr( nf90_open(trim(cdoutfile),nf90_write,ncid),                   cpname, __LINE__ )
      WRITE(*,*) 'Create the variables ',trim(cdoutfile)
      !  Forecast day
      CALL chkerr( nf90_inq_varid(ncid, 'leadtime', fdvid)                ,cpname, __LINE__ )
      CALL chkerr( nf90_put_var(ncid, fdvid, fcday) ,cpname, __LINE__ )
      WRITE(*,*) 'forecast day put'
      !  longitude
      CALL chkerr( nf90_inq_varid(ncid, 'longitude', lonid)                  ,cpname, __LINE__ )
      CALL chkerr( nf90_put_var(ncid, lonid, g_lam) ,cpname, __LINE__ )
      WRITE(*,*) 'lon put'
      !  latitude
      CALL chkerr( nf90_inq_varid(ncid, 'latitude', latid)                   ,cpname, __LINE__ )
      CALL chkerr( nf90_put_var(ncid, latid, g_phi) ,cpname, __LINE__ )
      WRITE(*,*) 'lat put'
      !  depth
      CALL chkerr( nf90_inq_varid(ncid, 'depth',depid)                       ,cpname, __LINE__ )
      CALL chkerr( nf90_put_var(ncid, depid, g_dep) ,cpname, __LINE__ )
      WRITE(*,*) 'dep put'
      ! varname
      CALL chkerr( nf90_inq_varid(ncid, 'varname', varid)                    ,cpname, __LINE__ )
      CALL chkerr( nf90_put_var(ncid, varid, gvnam,(/1,1/), (/nstr,nvars/) ) ,cpname, __LINE__ )   
      WRITE(*,*) 'var put'
      ! unitname
      CALL chkerr( nf90_inq_varid(ncid, 'unitname',unitid)                   ,cpname, __LINE__ )
      CALL chkerr( nf90_put_var(ncid, unitid, gunit,(/1,1/),(/nstr,nvars/) ) ,cpname, __LINE__ )   
      WRITE(*,*) 'unitnam put'
      ! obs
      CALL chkerr( nf90_inq_varid(ncid, 'observation', obvid)                ,cpname, __LINE__ )
      CALL chkerr( nf90_put_var(ncid, obvid,g3dob ) ,cpname, __LINE__ )   
      WRITE(*,*) 'obs put'
      ! clim
      CALL chkerr( nf90_inq_varid(ncid, 'climatology', clvid)                ,cpname, __LINE__ )
      CALL chkerr( nf90_put_var(ncid, clvid,g3dcl ) ,cpname, __LINE__ )   
      WRITE(*,*) 'cli put'
      IF (ln_best) THEN
         ! daym2
         CALL chkerr( nf90_inq_varid(ncid, 'best_estimate',dm2id)            ,cpname, __LINE__ )
         CALL chkerr( nf90_put_var(ncid, dm2id,g3dm2 ) ,cpname, __LINE__ )   
         WRITE(*,*) 'daym2 put'
      ENDIF
      IF (ln_init) THEN 
         ! daym1
         CALL chkerr( nf90_inq_varid(ncid, 'nrt_analysis',dm1id)              ,cpname, __LINE__ )
         CALL chkerr( nf90_put_var(ncid, dm1id,g3dm1 ) ,cpname, __LINE__ )   
         WRITE(*,*) 'daym1 put'
      ENDIF
      IF (ln_mdt) THEN 
         ! mdt
         CALL chkerr( nf90_inq_varid(ncid, 'mdt_reference', mdtid)              ,cpname, __LINE__ )
         CALL chkerr( nf90_put_var(ncid, mdtid, g3mdt ) ,cpname, __LINE__ )   
         WRITE(*,*) 'mdt put'
      ENDIF
      IF (ln_altbias) THEN 
         ! altbias
         CALL chkerr( nf90_inq_varid(ncid, 'altimeter_bias', altid)             ,cpname, __LINE__ )
         CALL chkerr( nf90_put_var(ncid, altid, g3alt ) ,cpname, __LINE__ )   
         WRITE(*,*) 'altbias put'
      ENDIF
      ! persistence
      CALL chkerr( nf90_inq_varid(ncid, 'persistence',prvid)                                 ,cpname, __LINE__ )
      CALL chkerr( nf90_put_var(ncid, prvid, g3dpr, (/1,1,1,1/) ,(/ ndeps,nfcst,nvars,nobs/) ) ,cpname, __LINE__ )   
      WRITE(*,*) 'per put'
      ! forecast
      CALL chkerr( nf90_inq_varid(ncid, 'forecast',fcvid)                                      ,cpname, __LINE__ )
      CALL chkerr( nf90_put_var(ncid, fcvid, g3dmc, (/1,1,1,1/), (/ ndeps,nfcst,nvars,nobs/) ) ,cpname, __LINE__ )   
      WRITE(*,*) 'fcst put'
      ! qc
      CALL chkerr( nf90_inq_varid(ncid, 'qc', qcvid)                         ,cpname, __LINE__ )
      CALL chkerr( nf90_put_var(ncid, qcvid,g3dqc ) ,cpname, __LINE__ )   
      WRITE(*,*) 'qc put'
      ! juld
      CALL chkerr( nf90_inq_varid(ncid, 'juld',jdvid)                        ,cpname, __LINE__ )
      CALL chkerr( nf90_put_var(ncid, jdvid, gjuld) ,cpname, __LINE__ )   
      WRITE(*,*) 'juld put'
      ! modeljuld
      CALL chkerr( nf90_inq_varid(ncid, 'modeljuld', mjdid)                  ,cpname, __LINE__ )
      CALL chkerr( nf90_put_var(ncid, mjdid, modjd,(/1/),(/nfcst/))          ,cpname, __LINE__ )   
      WRITE(*,*) 'modjuld put'
      ! type
      CALL chkerr( nf90_inq_varid(ncid, 'type', typid)                       ,cpname, __LINE__ )
      CALL chkerr( nf90_put_var(ncid, typid, gtype,(/1,1/) , (/n128,nobs/) ) ,cpname, __LINE__ )   
      WRITE(*,*) 'type put'
      ! id
      CALL chkerr( nf90_inq_varid(ncid, 'id', idvid)                         ,cpname, __LINE__ )
      CALL chkerr( nf90_put_var(ncid, idvid, g_id,(/1,1/)  , (/nstr,nobs/) ) ,cpname, __LINE__ )   
      WRITE(*,*) 'id put'
      ! Close netcdf file
      CALL chkerr( nf90_close(ncid),                                          cpname, __LINE__ )
   END IF ! ln_cre
   !! Deallocate Global arrays
   DEALLOCATE( g_lam, g_phi, g_dep, g3dob, g3dmc, g3dpr, g3dcl, g3dm2, g3dm1, g3mdt, g3alt, g3dqc, gjuld, gtype, g_id, gvnam, gunit)
   DEALLOCATE( fcday, modjd )

   !! Deallocate input argument list
   DEALLOCATE(cdinfile)
END PROGRAM c4comb
