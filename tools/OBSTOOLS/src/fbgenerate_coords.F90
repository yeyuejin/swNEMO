MODULE fbgenerate_coords
USE obs_fbm
USE date_utils

CONTAINS

   REAL(KIND=fbdp) FUNCTION fb_dates(timein,datein) 
      IMPLICIT NONE   
      INTEGER, INTENT(IN) :: timein 	! Format: HHMM
		INTEGER, INTENT(IN) :: datein 	! Format: YYYYMMDD
      INTEGER :: iyea,imon,iday
      INTEGER :: ihr,imin,isec
  
      iyea=datein/10000
      imon=datein/100-iyea*100
      iday=datein-iyea*10000-imon*100
		
		ihr = timein/100
		imin = timein - ihr*100
		isec = 0
		
      CALL greg2jul(isec,imin,ihr,iday,imon,iyea,fb_dates)
   
   END FUNCTION fb_dates



	SUBROUTINE set_spatial_coords(lats,lons,n,FillVal)
	IMPLICIT NONE	
	INTEGER :: i, j, k, p, nlats, nlons, nlats_in_list, nlons_in_list
	INTEGER, INTENT(IN) :: n
	REAL(KIND=fbdp), INTENT(INOUT) :: lats(:), lons(:)
	REAL(KIND=fbdp) :: FillVal 
	
	! A single non-FillVal value should be replicated n times
	! Three non-FillVal values should be used as the start, end, step in conjunction with n
	!	unless n is three, in which case treat as list, not bounds.
	! A full list of n non-FillVals should be left unaltered.
	
	!Find number of lats and lons
	! by finding number of non-FillVal entries in arrays
	! and expanding bounds if necessary
	
	nlats_in_list = last_element(lats,FillVal)
	nlons_in_list = last_element(lons,FillVal)

   !Expand bounds only if 3 values given and number obs>3
	IF (nlats_in_list == 1) THEN
		lats(1:nlats) = lats(1)
	ELSE IF ((nlats_in_list==3).AND.((n>3))) THEN ! Treat as list of three, not bounds
		CALL expand_bounds(lats, FillVal)
	END IF

	IF (nlons_in_list == 1) THEN
		lons(1:nlons) = lons(1)
	ELSE IF ((nlons_in_list==3).AND.((n>3))) THEN ! Treat as list of three, not bounds
		CALL expand_bounds(lons, FillVal)
	END IF

	nlats = last_element(lats,FillVal)
	nlons = last_element(lons,FillVal)
	
	IF ((n /= nlats) .AND. (n /= nlons)) THEN
		WRITE(*,*) "ERROR: Number of lat/lons not equal to nobs", nlats, nlons, n
		CALL abort	
	END IF
	
	END SUBROUTINE set_spatial_coords


	SUBROUTINE set_spatial_coords_grid(lats,lons,n,nlats,nlons,FillVal)
	IMPLICIT NONE	
	INTEGER :: i, j, k, p, nlats_in_list, nlons_in_list
	INTEGER, INTENT(IN) :: n, nlats, nlons
	REAL(KIND=fbdp), INTENT(INOUT) :: lats(:), lons(:)
	REAL(KIND=fbdp), ALLOCATABLE :: tmp_lats(:), tmp_lons(:)
	REAL(KIND=fbdp) :: FillVal 
	
	! A single non-FillVal value should be replicated n times
	! Three non-FillVal values should be used as the start, end, step in conjunction with n
	!	unless n is three, in which case treat as list, not bounds.
	! A full list of n non-FillVals should be left unaltered.
	
	!Find number of lats and lons in list
	! by finding number of non-FillVal entries in arrays
	! and then expand bounds if necessary
	
	nlats_in_list = last_element(lats,FillVal)
	nlons_in_list = last_element(lons,FillVal)
	
   !Expand bounds only if 3 values given and number lats (or lons) not equal to 3
	IF (nlats_in_list == 1) THEN
		lats(1:nlats) = lats(1)
	ELSE IF ((nlats_in_list==3).AND.((nlats>3))) THEN ! Treat as bounds
		CALL expand_bounds(lats, FillVal)
	END IF

	IF (nlons_in_list == 1) THEN
		lons(1:nlons) = lons(1)
	ELSE IF ((nlons_in_list==3).AND.((nlons>3))) THEN ! Treat as bounds
		CALL expand_bounds(lons, FillVal)
	END IF

	nlats_in_list = last_element(lats,FillVal)
	nlons_in_list = last_element(lons,FillVal)


	IF ((nlats_in_list * nlons_in_list) == n) THEN
		ALLOCATE(tmp_lons(n))
		ALLOCATE(tmp_lats(n))
		k = 0
		p = 0
		DO i = 1, nlats
		   p = p + 1
			DO j = 1, nlons
				k = k + 1
				tmp_lons(k) = lons(j)   
				tmp_lats(k) = lats(p)   
			END DO
		END DO
		lats(:) = tmp_lats(:)
		lons(:) = tmp_lons(:)
		DEALLOCATE(tmp_lons)
		DEALLOCATE(tmp_lats)
	ELSE 
		WRITE(*,*) 
		WRITE(*,*) "ERROR: Number of lat * lon values not equal to nobs", nlats_in_list, nlons_in_list, n
		CALL abort	
	END IF
	
	END SUBROUTINE set_spatial_coords_grid


	SUBROUTINE set_spatial_coords_physpace(lats,lons,n,FillVal,phys_spacing)
	IMPLICIT NONE	
	INTEGER :: i, j, k, nlats, nlons
	INTEGER, INTENT(IN) :: n
	REAL(KIND=fbdp), INTENT(INOUT) :: lats(:), lons(:)
	REAL(KIND=fbdp), ALLOCATABLE :: tmp_lats(:)
	REAL(KIND=fbdp) :: FillVal 
   REAL(KIND=fbdp), PARAMETER :: earth_radius = 6371.0_fbdp ! km
   REAL(KIND=fbdp), PARAMETER :: pi = 3.141592654_fbdp
	REAL(KIND=fbdp), INTENT(IN) :: phys_spacing
   REAL(KIND=fbdp) :: lat_step
	
	! Use physical spacing to set lats and lons

   nlats = INT(pi * earth_radius / phys_spacing)
   IF (MOD(nlats,2)==0) nlats=nlats-1   ! Ensure nlats is odd to have even number around equator.  
      
   ALLOCATE(tmp_lats(nlats))

   lat_step = 180.0_fbdp / (nlats-1)

   tmp_lats(1) = 0.0_fbdp
   DO i=1,(nlats-1)/2
      tmp_lats(i+1) = tmp_lats(i) + lat_step
      tmp_lats(((nlats-1)/2)+i+1) = tmp_lats(i+1) * (-1.0_fbdp)
   END DO

   k = 0
   DO i=1,nlats
      nlons = INT(2.0_fbdp * pi * earth_radius * cos(tmp_lats(i)* (pi/180.0_fbdp)) / phys_spacing)
      IF (nlons>0) THEN
         DO j = 1, nlons
            k = k + 1
            lons(k) = REAL(j-1) * 360.0_fbdp / nlons
            lats(k) = tmp_lats(i)
         END DO
      END IF
   END DO
   DEALLOCATE(tmp_lats)
	
	END SUBROUTINE set_spatial_coords_physpace


	SUBROUTINE expand_bounds(array, fillval)
	!
	! Checks if there are three entries and expands bounds.
	!
	IMPLICIT NONE
	INTEGER :: i, nentries
	REAL(KIND=fbdp), INTENT(INOUT) :: array(:)
	REAL(KIND=fbdp), INTENT(IN) :: fillval
	REAL(KIND=fbdp) :: nsteps, start, step

	! Find number of elements given in list
	nentries = last_element(array, fillval)

	IF (nentries==3) THEN 			! Bounds given	
		nsteps = (array(2) - array(1)) /  array(3) 
	   IF ((array(2) < array(1)) .AND. (array(3) >= 0.0)) THEN
		   WRITE(*,*) "Error: if upper bound is less than lower bound, step must be negative.", array
			CALL abort
		ELSE IF ( NINT(nsteps)+1 > SIZE(array) ) THEN
		   WRITE(*,*) "Error: bounds not compatible with length of array.", array
		   WRITE(*,*) "Error: bounds not compatible with length of array.", nsteps
			CALL abort
		END IF
		start = array(1) 
		step = array(3)
		DO i=1,NINT(nsteps)+1
			array(i) = start + (i-1)*step 	   	
		END DO
	END IF
	
	END SUBROUTINE expand_bounds



	INTEGER FUNCTION last_element(array,fillval)
	!
	! Returns index of the last non-fill value element in a list
	! Will return 1, even in first element is the FillValue
	!
	IMPLICIT NONE
	REAL(KIND=fbdp), INTENT(IN) :: array(:), fillval
	INTEGER :: i

	last_element = 1
	IF (SIZE(array) > 1) THEN
		DO i=2, SIZE(array)
			IF (array(i) /= fillval) THEN 
				last_element = i
			END IF
		END DO
	END IF
		
	END FUNCTION last_element



	SUBROUTINE set_depths(array,n,FillVal)
	IMPLICIT NONE	
	INTEGER :: i
	INTEGER, INTENT(IN) :: n
	REAL(KIND=fbdp), INTENT(INOUT) :: array(:)
	REAL(KIND=fbdp) :: FillVal
	REAL(KIND=fbdp) :: start, step, nstep
	! Top bound not necessarily inclusive - will use start, step and number of obs
	
	! A single non-FillVal value should be replicated n times
	! Three non-FillVal values should be used as the start, end, step in conjunction with n
	!	unless n is three, in which case treat as list, not bounds.
	! A full list of n non-FillVals should be left unaltered.
	IF (n==1) THEN
		array(:) = array(1)
	ELSE IF ((n > 1) .AND. (array(2)==FillVal)) THEN
		array(:) = array(1)
	ELSE IF ((n==2) .AND. (array(2)/=FillVal)) THEN
		array(:) = array(:)
	ELSE IF (n==3) THEN ! Treat as list of three, not bounds
		array(:) = array(:)
	ELSE IF ((n>=4) .AND. (array(4)==FillVal)) THEN ! Assume start, stop, step
		nstep = (array(2) - array(1)) /  array(3) 
	   IF ((array(2) < array(1)) .AND. (array(3) >= 0.0)) THEN
		   WRITE(*,*) "Error: if upper bound is less than lower bound, step must be negative.", array
			CALL abort
		ELSE IF ( ( nstep < (0.99 * real(n-1)) ) .OR. &
		          ( nstep > (1.01 * real(n-1)) ) ) THEN
		   WRITE(*,*) "Error: depth bounds not compatible.", array
		   WRITE(*,*) "Error: depth bounds not compatible.", nstep
			CALL abort
		END IF
		start = array(1) 
		step = array(3)
		DO i=1,n
			array(i) = start + (i-1)*step 	   	
		END DO
	END IF
	
	END SUBROUTINE set_depths





	SUBROUTINE set_datetime(date,time,julian_date,n,FillVal)
	!
	! Transform input array from m values describing n dates to a list of n dates.
	!
	IMPLICIT NONE	
	INTEGER :: i
	INTEGER, INTENT(IN) :: n
	INTEGER, INTENT(INOUT) :: date(:), time(:)
	INTEGER :: FillVal
	REAL(KIND=fbdp), INTENT(INOUT) :: julian_date(:)
	REAL(KIND=fbdp) :: start, step, finish

   ! Check if user has supplied start and end times to be split amongst number of obs
   ! (i.e. more than 2 obs, but only 2 dates and 2 times given.)
   IF ((n>2) .AND. (date(2)/=FillVal) .AND. (date(3)==FillVal) &
      &      .AND. (time(2)/=FillVal) .AND. (time(3)==FillVal)) THEN
      
      !work out JD of each
      start = fb_dates(time(1),date(1)) 
      finish = fb_dates(time(2),date(2)) 

      ! use info to calc step
      step = ( finish - start ) / n
      ! calc all JDs and put output in dates
      DO i=1,n 
	      ! Could make this an elemental function if dateutils were pure funcs
         julian_date(i)= start + (i-1)*step
      END DO

   ELSE
      CALL set_date(date,n,FillVal)
      CALL set_time(time,n,FillVal)
      ! Replace dates with JD including time info
      DO i=1,n 
	      ! Could make this an elemental function if dateutils were pure funcs
         julian_date(i)= fb_dates(time(i),date(i))
      END DO
   END IF


	END SUBROUTINE set_datetime


	SUBROUTINE set_date(array,n,FillVal)
	!
	! Transform input array from m values describing n dates to a list of n dates.
	!
	IMPLICIT NONE	
	INTEGER :: i
	INTEGER, INTENT(IN) :: n
	INTEGER, INTENT(INOUT) :: array(:)
	INTEGER :: FillVal
	INTEGER :: start, step, diff
	! Top bound not necessarily inclusive - will use start, step and number of obs
	
	! A single non-FillVal value should be replicated n times
	! Three non-FillVal values should be used as the start, end, step in conjunction with n
	!	unless n is three, in which case treat as list, not bounds.
	! A full list of n non-FillVals should be left unaltered.
	IF (n==1) THEN
		array(:) = array(1)
	ELSE IF ((n > 1) .AND. (array(2)==FillVal)) THEN
		array(:) = array(1)
	ELSE IF ((n==2) .AND. (array(2)/=FillVal)) THEN
		array(:) = array(:)
	ELSE IF (n==3) THEN ! Treat as list of three, not bounds
		array(:) = array(:)
	ELSE IF ((n>=4) .AND. (array(4)==FillVal)) THEN ! Assume start, stop, step
   	diff = diffdate(array(2),array(1))	! in number of days
	   IF ((array(2) < array(1)) .AND. (array(3) >= 0)) THEN
		   WRITE(*,*) "Error: if upper bound is less than lower bound, step must be negative.", array
			CALL abort
		ELSE IF ( ( diff / ABS(array(3)) ) /= (n-1) ) THEN
		   WRITE(*,*) "Error: date bounds not compatible.", array
			CALL abort
		END IF
		start = array(1) 
		step = array(3)
		DO i=1,n
			CALL add_days_to_date(start,(i-1)*step,array(i)) 	   	
		END DO
	END IF
	
	END SUBROUTINE set_date



	SUBROUTINE set_time(array,n,FillVal)
	!
	! Transform input array from m (m<=n) values describing n times to a list of n times.
	!
	IMPLICIT NONE	
	INTEGER :: i
	INTEGER, INTENT(IN) :: n
	INTEGER, INTENT(INOUT) :: array(:)
	INTEGER :: FillVal
	INTEGER :: start, step, nstep
	! Top bound not necessarily inclusive - will use start, step and number of obs
	
	! A single non-FillVal value should be replicated n times
	! Three non-FillVal values should be used as the start, end, step in conjunction with n
	!	unless n is three, in which case treat as list, not bounds.
	! A full list of n non-FillVals should be left unaltered.
	IF (n==1) THEN
		array(:) = array(1)
	ELSE IF ((n > 1) .AND. (array(2)==FillVal)) THEN
		array(:) = array(1)
	ELSE IF ((n==2) .AND. (array(2)/=FillVal)) THEN
		array(:) = array(:)
	ELSE IF (n==3) THEN ! Treat as list of three, not bounds
		array(:) = array(:)
	ELSE IF ((n>=4) .AND. (array(4)==FillVal)) THEN ! Assume start, stop, step
		IF (array(3)<0) nstep = difftime(array(2),array(1)) / ABS(array(3)) 
		IF (array(3)>=0) nstep = difftime(array(1),array(2)) / ABS(array(3)) 
		IF ( nstep /= (n-1) ) THEN
		   WRITE(*,*) "Error: time bounds not compatible.", array
			CALL abort
		END IF
		start = array(1) 
		step = array(3)
		DO i=1,n
			array(i) = add_mins_to_time(start,(i-1)*step) 	   	
		END DO
	END IF
	
	END SUBROUTINE set_time
	
	

	SUBROUTINE set_obs_values(array,p,n,m,FillVal)
	!
	! n profiles at m depths
	!
	IMPLICIT NONE	
	INTEGER :: i, j, k
	INTEGER, INTENT(IN) :: n, m, p
	REAL(KIND=fbdp), INTENT(INOUT) :: array(:,:,:) 			! (nlevels, nprofiles, nvars) = (m,n,p)
	REAL(KIND=fbdp) :: FillVal 
	
	
	DO k=1,p
	   IF ((k > 1) .AND. (array(1,1,k) == FillVal)) THEN  ! set to same values as first variable
			array(:,:,k) = array(:,:,1)
		ELSE
			IF ((n==1).AND.(m==1)) THEN
				array(:,:,k) = array(1,1,k)

			! If mult depths, but not specified, set all to one value	
			ELSE IF ((m > 1) .AND. (array(2,1,k) == FillVal)) THEN
				array(:,:,k) = array(1,1,k)

			! If mult profiles and mult depths and only one value set in first profile	
			ELSE IF ((n > 1) .AND. (m > 1) .AND. (array(2,1,k) == FillVal)) THEN
				array(:,:,k) = array(1,1,k)
				
			! If mult profiles and mult depths and only one first profile set	
			ELSE IF ((n > 1) .AND. (m > 1) .AND. (array(1,2,k) == FillVal)) THEN
				DO j=1,m
					array(j,:,k) = array(j,1,k)
				END DO
				
			ELSE 
	   		array(:,:,k) = array(:,:,k)
			END IF
		END IF
	END DO	
	
	END SUBROUTINE set_obs_values


   ! Unbiased shuffle of array
   SUBROUTINE shuffle(a)
   REAL(KIND=fbdp), INTENT(INOUT) :: a(:)
   INTEGER :: i, randpos
   REAL(KIND=fbdp) :: r, temp

   CALL random_seed()
   DO i = SIZE(a), 2, -1
      CALL random_number(r)
      randpos = int(r * i) + 1
      temp = a(randpos)
      a(randpos) = a(i)
      a(i) = temp
   END DO
   
   END SUBROUTINE shuffle
   
   
   ! Add a random perturbation to the lats and lons
   ! Perturbation is sampled from a uniform distribution +/-perturb_limit
   SUBROUTINE perturb_positions(lats,lons,perturb_limit)
   INTEGER :: i
   REAL(KIND=fbdp), INTENT(INOUT) :: lats(:), lons(:)
   REAL(KIND=fbdp), INTENT(IN) :: perturb_limit
   REAL(KIND=fbdp) :: randpos, lat_perturb, lon_perturb
   REAL(KIND=fbdp), PARAMETER :: earth_radius = 6371.0_fbdp
   REAL(KIND=fbdp), PARAMETER :: pi = 3.141592654_fbdp
   
 	IF ( SIZE(lats) /= SIZE(lons) ) THEN
		WRITE(*,*) "Error: different number of lat and lon elements", SIZE(lats), SIZE(lons)
		CALL abort
	END IF

   CALL random_seed()
   
   ! Convert physical sep into a latidue sep in degrees
   lat_perturb = 360.0_fbdp * perturb_limit / (2.0_fbdp * pi * earth_radius)
   
   DO i=1,SIZE(lats)
   
      ! Perturb lats first, as lon conversion uses lat
      CALL random_number(randpos)
      lats(i) = lats(i) + randpos*lat_perturb
      IF (lats(i) > 90.0_fbdp) lats(i) = 180.0_fbdp - lats(i) 
      IF (lats(i) < -90.0_fbdp) lats(i) = (lats(i) + 180.0_fbdp) * (-1.0_fbdp)
      
      ! Use lat to convert physical size to delta_longitude   
      IF (ABS(lats(i)) == 90.0_fbdp) THEN 
         lon_perturb = 0.0_fbdp
      ELSE
         lon_perturb = 360.0_fbdp * perturb_limit / (2.0_fbdp * pi * earth_radius * cos(lats(i)*(pi/180.0_fbdp)))
      END IF
      CALL random_number(randpos)
      lons(i) = lons(i) + randpos*lon_perturb
      IF (lons(i) >= 360.0_fbdp) lons(i) = lons(i) - 360.0_fbdp 
      IF (lons(i) < 0.0_fbdp) lons(i) = lons(i) + 360.0_fbdp 
   
   END DO
   
   END SUBROUTINE perturb_positions
    
END MODULE fbgenerate_coords
