PROGRAM fbgenerate
   !!---------------------------------------------------------------------
   !!
   !!                     ** PROGRAM fbmatchup **
   !!
   !!  ** Purpose : Generate a feedback file of pseudo obs
   !!
   !!  ** Method  : Use of utilities from obs_fbm.
   !!
   !!  ** Action  : Read in data from a namelist file and generate a feedback
   !!               file of pseudo observations
   !! 
   !!   Usage: 		For any parameter, if 3 values are given, these will be treated as
   !!                   bounds (start, stop, step), EXCEPT when only 3 values are expected.
   !!
   !!                For spatial coord, date and time
	!! 						3 values with nlat=3 will be treated as three separate values
	!! 						3 values with nobs=3 will be treated as three separate values
	!! 						3 values with nobs>3 will be treated as a start, end and step
	!! 						Giving a FillValue in one of namelist entries may have unexpected consequences
	!! 							e.g., discarding rest of list of values
	!!
	!! 					For time bounds, start(HHMM), end(HHMM), step(minutes)
	!!						For date bounds, start(YYYYMMDD), end(YYYYMMDD), step(days)
   !!
   !!                To split obs evenly in time, date and time should be given two values 
   !!                describing the start and end times. 
   !!
   !!                Can use logical flag to shuffle timestamps to remove position-time correlation.
   !!
   !!                Can use logical flag to add uniformly sampled perturbation (up to defined limit) to positions
	!!
	!! 					For obs values, can list either single value for all (x,y,z)
	!! 					or can specify single profile for all (x,y)
	!! 					or can specify all values 1 profile after another.
	!!	 					
	!! 					In namelist to set vals(nlev,nobs,nvar)
	!! 						set all obs (for variable 1) at all levels to one value
	!! 							vals(:,:,1)= 1.0
	!! 						set all obs (for variable 2) at all levels to one value
	!! 							vals(:,:,2)= 35.0
	!! 						set one profile (for variable 1), apply to all	
	!! 							vals(:,:,1)= 1.0, 2.0, 3.0, 4.0
	!! 						set all profiles profile (for variable 1)	
	!! 							vals(:,1,1)= 1.0, 2.0, 3.0, 4.0
	!! 							vals(:,2,1)= 1.0, 2.0, 3.0, 4.0
	!! 							vals(:,nobs,1)= 1.0, 2.0, 3.0, 4.0
   !!     
   !!
   !!   Limitations: Uses an allocatable array in a namelist. This is not meant 
	!!                 to be allowed in F95, but it works when compiled with xlf90_r
	!!                 (and is allowed in Fortran2003).
	!!
	!!                Forces same depth levels on all profiles.
	!!
   !!   History :
   !!        ! 2014    (J. Waters) Initial version
   !!        ! 2014-02 (R. King)   Adapted for profiles and bounds on vars
   !!----------------------------------------------------------------------
   USE obs_fbm
	USE fbgenerate_coords
	USE test_fbgenerate
   IMPLICIT NONE
   TYPE(obfbdata) :: fbobsdata
   CHARACTER(len=256) :: filenameout
#ifndef NOIARGCPROTO
   INTEGER,EXTERNAL :: iargc
#endif
   INTEGER :: nobs, nvar, nlev, ntyp, nadd, next, lgrid, ierr, nlats, nlons
   INTEGER :: i, j
   INTEGER :: FillValue_int
	REAL(KIND=fbdp) :: FillValue_real
   INTEGER, ALLOCATABLE :: dates(:), times(:)
   REAL(KIND=fbdp), ALLOCATABLE :: julian_dates(:)
	REAL(KIND=fbdp) :: phys_spacing, lat_step, current_lat, perturb_limit
   REAL(KIND=fbdp), ALLOCATABLE :: lats(:), lons(:), deps(:), vals(:,:,:), model_vals(:,:,:) 
   CHARACTER(LEN=ilenwmo) :: cdwmo, cdtyp
   CHARACTER(LEN=ilenwmo), ALLOCATABLE :: variable(:)
	CHARACTER(LEN=ilenlong), ALLOCATABLE :: variable_longname(:)
	CHARACTER(LEN=ilenunit), ALLOCATABLE :: variable_units(:)
	LOGICAL, PARAMETER :: test_prog = .FALSE.
	LOGICAL :: ln_listobs, ln_physpace, ln_gridobs, ln_shuffletimes, ln_perturb_posn
   REAL(KIND=fbdp), PARAMETER :: earth_radius = 6371.0_fbdp
   REAL(KIND=fbdp), PARAMETER :: pi = 3.14159_fbdp
	
	!!!! set up default values and read in namelist namfbgen.in !!!!!!!!!!!!!!!!!!!!
   NAMELIST/namfbgeneratesetup/ln_listobs,ln_physpace,ln_gridobs,ln_shuffletimes,ln_perturb_posn,perturb_limit,nobs,nlats,nlons,phys_spacing,nlev,nvar,nadd
   NAMELIST/namfbgenerate/ntyp,lats,lons,deps,times,dates,vals,model_vals,variable,variable_longname,variable_units,cdwmo,cdtyp

	IF (test_prog) THEN
		WRITE(*,*) "In test mode:"
		CALL tester
	ELSE
	
   
	IF (iargc()/=1) THEN
      WRITE(*,*)'Usage:'
      WRITE(*,*)'fbgenerate <output filename>'
      CALL abort
   ENDIF

   CALL getarg(1,filenameout)
	
   ! set up default values for namfbgeneratesetup
   ln_listobs=.FALSE.
   ln_physpace=.FALSE.
   ln_gridobs=.FALSE.
   ln_shuffletimes=.FALSE.
   ln_perturb_posn=.FALSE.
   nobs=0
   nlats=0
   nlons=0
   phys_spacing =100.0_fbdp
   nlev=0
	nvar=0

   ! Read in number of obs and number of depths from namelist (before reading other variables)
	FillValue_real = 99999.0_fbdp
	FillValue_int = 99999
   OPEN(10,file='namfbgen.in')
   READ(10,namfbgeneratesetup) 
   CLOSE(10)
   
   ! Calculate number of observations
   IF ((ln_gridobs) .AND. (.NOT.ln_physpace) .AND. (.NOT.ln_listobs)) THEN
      IF( ( nlats .LE. 0 ) .OR. (nlons .LE. 0) )THEN
         WRITE(6,*)'ERROR: nlats and nlons must be greater than 0 if ln_gridobs set'
      ELSE
         nobs=nlats*nlons
         WRITE(*,*) "Constructing a grid of lat-lons"
      ENDIF

   ELSE IF ((.NOT.ln_gridobs) .AND. (ln_physpace) .AND. (.NOT.ln_listobs)) THEN
      IF ((phys_spacing .LE. 0) .OR.(phys_spacing .GE. earth_radius)) THEN
         WRITE(6,*)'ERROR: phys_spacing must be greater than 0 and less than 6371km.'
      ELSE
        nlats = INT(pi * earth_radius / phys_spacing)
        IF (MOD(nlats,2)==0) nlats=nlats-1   ! Ensure nlats is odd to have even number around equator.  
        lat_step = 180.0_fbdp / (nlats-1)
        nobs=0
        current_lat=0.0_fbdp
        ! Sum up number of lons at each latitude
        DO i=1,(nlats-1)/2
           current_lat = current_lat + lat_step * (pi/180.0_fbdp)
           nobs = nobs + INT(2.0_fbdp * pi * earth_radius * cos(current_lat) / phys_spacing) 
        END DO
        nobs = nobs * 2 + INT(2.0_fbdp * pi * earth_radius / phys_spacing)
        WRITE(*,*) "Using a physical seperation of ~", INT(phys_spacing),"km"
      END IF

   ELSE IF ((.NOT.ln_gridobs) .AND. (.NOT.ln_physpace) .AND. (ln_listobs)) THEN
      IF( nobs .LE. 0 ) THEN
         WRITE(6,*)'ERROR: nobs must be greater than 0'
      ENDIF
      WRITE(*,*) "Constructing a list of lat-lons"
   ELSE
      WRITE(*,*)'ERROR: one (and only one) of the logical flags must be true!'
      CALL abort 
   ENDIF


   ALLOCATE(dates(nobs),       	&
	    		times(nobs),       	&
	    		julian_dates(nobs),  &
	    		lats(nobs),	       	&
	    		lons(nobs),	       	&
	    		deps(nlev),        	&
	    		variable(nvar),    	&
	    		variable_longname(nvar),   &
	    		variable_units(nvar),&
	    		vals(nlev,nobs,nvar),&
	    		model_vals(nlev,nobs,nvar),&
				STAT=ierr          	&
	    		)
	IF (ierr /= 0) THEN
		WRITE(*,*) "Could not allocate observation arrays:"
		WRITE(*,*) "dates(", nobs, ")"
		WRITE(*,*) "times(", nobs, ")"
		WRITE(*,*) "lats(", nobs, ")"
		WRITE(*,*) "lons(", nobs, ")"
		WRITE(*,*) "deps(", nlev, ")"
		WRITE(*,*) "variable(", nvar, ")"
		WRITE(*,*) "variable_longname(", nvar, ")"
		WRITE(*,*) "variable_units(", nvar, ")"
		WRITE(*,*) "vals(", nlev, nobs, nvar, ")"
		WRITE(*,*) "model_vals(", nlev, nobs, nvar, ")"

		CALL abort
	END IF			    
	    
		 
   !!!! set up default values and read in namelist namfbgen.in !!!!!!!!!!!!!!!!!!!!
   ntyp=-1
   lats(:) = FillValue_real
   lons(:) = FillValue_real
   deps(:) = FillValue_real
   times(:) = FillValue_int
   dates(:) = FillValue_int
   vals(:,:,:) = FillValue_real
   model_vals(:,:,:) = FillValue_real
   variable=REPEAT('X',ilenwmo)
   variable_longname=REPEAT('X',ilenwmo)
   variable_units=REPEAT('X',ilenwmo)
   cdwmo=REPEAT('X',ilenwmo)  !station identifier
   cdtyp="90"    !station type

   WRITE(*,*) "Creating a fdbk file with", nobs, "observations"
   WRITE(*,*) "with", nvar, "variables and",nlev, "depths."

   OPEN(10,file='namfbgen.in')
   READ(10,namfbgenerate)
   CLOSE(10)
	
	! Use bounds to construct the full arrays of coords and values

   IF (ln_physpace) THEN
      CALL set_spatial_coords_physpace(lats,lons,nobs,FillValue_real,phys_spacing)
   ELSE IF (ln_gridobs) THEN
      CALL set_spatial_coords_grid(lats,lons,nobs,nlats,nlons,FillValue_real)
   ELSE
      CALL set_spatial_coords(lats,lons,nobs,FillValue_real)
   END IF
   
   ! Add random perturbation to position if ln_perturb_posn=.TRUE.   
   IF (ln_perturb_posn) CALL perturb_positions(lats,lons,perturb_limit)
   
   CALL set_datetime(dates,times,julian_dates,nobs,FillValue_int)
   CALL set_depths(deps,nlev,FillValue_real)
   CALL set_obs_values(vals,nvar,nobs,nlev,FillValue_real)
   CALL set_obs_values(model_vals,nvar,nobs,nlev,FillValue_real)


   !nadd=1	! Hx
   next=0	! e.g. TEMP 
	! Add TEMP as extra variable if POTM is defined.
	DO i=1,nvar
		IF (variable(i)(1:4) == "POTM") THEN
		   next=1	! e.g. If set to zero, will not add extra TEMP variable			
			EXIT
		END IF
	END DO

   CALL init_obfbdata(fbobsdata)
   
   !!!! Allocate the obfb type !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   
   CALL alloc_obfbdata( fbobsdata, nvar, nobs, nlev, nadd, next, .FALSE.)
   
   !!!! Read data specified in the namelist into obfd type!!!!!!!!!!!!!!!
	    
   fbobsdata%cname = variable(:)
	fbobsdata%coblong = variable_longname(:)
	fbobsdata%cobunit = variable_units(:)
   fbobsdata%cdjuldref = "19500101000000"
   fbobsdata%cdwmo = cdwmo
   fbobsdata%cdtyp = cdtyp
	
   ! Add model counterpart values if nadd=1
	IF (nadd > 0) THEN
		IF (nadd /= 1) THEN
			CALL abort
			WRITE(*,*) "Not set-up to add more than 1 additional variable."
		ELSE
			fbobsdata%caddname = "Hx"
   		fbobsdata%caddlong(1,:) = "Model interpolated " // variable_longname(:)
   		fbobsdata%caddunit(1,:) = variable_units(:)
         fbobsdata%padd(:,:,1,:)=model_vals(:,:,:)
		END IF
	END IF
	
	! Add TEMP as extra variable if POTM is defined.
	IF (next==1) THEN
		fbobsdata%cextname = "TEMP"
	END IF
	
   ! set up pos/time variables
   fbobsdata%plam(:)=lons(:)
   fbobsdata%pphi(:)=lats(:)
	DO i=1,nobs 
      fbobsdata%pdep(:,i)=deps(:)
   END DO

   
   fbobsdata%ptim(:)= julian_dates(:)
   
   ! Shuffle the time array to spread the profiles across the time period
   IF (ln_shuffletimes) CALL shuffle(fbobsdata%ptim)

   ! read in variable data and flags
   fbobsdata%ivqc(:,:)=1
   fbobsdata%ivqcf(:,:,:)=1
   fbobsdata%ivlqc(:,:,:)=1
   fbobsdata%pob(:,:,:)=vals(:,:,:)

   !set up all QC flags
   fbobsdata%ioqc(:)=1
   fbobsdata%ioqcf(:,:)=0
   fbobsdata%ipqc(:)=1  
   fbobsdata%ipqcf(:,:)=0 
   fbobsdata%idqc(:,:)=1		!0
   fbobsdata%idqcf(:,:,:)=0
   fbobsdata%itqc(:)=1
   fbobsdata%kindex(:)=0
   
   !!!! Write out the obfb type !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
   CALL write_obfbdata(TRIM(filenameout),fbobsdata)

   DEALLOCATE(dates,       &
              times,       &
              julian_dates,&
	    		  lats,	      &
	    		  lons,	      &
	    		  deps,        &
	    		  variable,    &
	    		  variable_longname,    &
	    		  variable_units,       &
	    		  vals,        &
	    		  model_vals   &
	    		  )

	END IF !test prog


END PROGRAM fbgenerate


