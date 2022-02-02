MODULE test_fbgenerate	
USE obs_fbm
USE test_arrays_mod
USE fbgenerate_coords

CONTAINS

   SUBROUTINE tester
	IMPLICIT NONE
	REAL(KIND=fbdp) :: array10_in(10), array1_in(1), array2_in(2), array3_in(3)
	REAL(KIND=fbdp) :: array10_out(10), array1_out(1), array2_out(2), array3_out(3)
	REAL(KIND=fbdp) :: array_4x3_in(3,4,1), array_4x3_out(3,4,1)
	REAL(KIND=fbdp), ALLOCATABLE :: lat_array_in(:), lat_array_out(:)
	REAL(KIND=fbdp), ALLOCATABLE :: lon_array_in(:), lon_array_out(:)
	INTEGER :: iarray10_in(10), iarray1_in(1), iarray2_in(2), iarray3_in(3)
	INTEGER :: iarray10_out(10), iarray1_out(1), iarray2_out(2), iarray3_out(3)
	LOGICAL :: okay = .TRUE.
	LOGICAL :: okay_too = .TRUE.
	LOGICAL :: all_okay = .TRUE.
	INTEGER, PARAMETER :: FV_int = 99999
	REAL(KIND=fbdp), PARAMETER :: FV_real = 99999.0_fbdp
	
   ! A single non-FillVal value should be replicated n times
	array1_in(:)=(/1.0_fbdp/)
	array1_out(:)=(/1.0_fbdp/)
	CALL set_depths(array1_in,1,FV_real)
	okay = test_arrays(array1_in,array1_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 1 failed"
		all_okay = .FALSE.
	END IF

   ! A single non-FillVal value should be replicated n times
	array2_in(:) = (/1.0_fbdp, FV_real/)
	array2_out(:) = (/1.0_fbdp, 1.0_fbdp/)
	CALL set_depths(array2_in,2,FV_real)
	okay = test_arrays(array2_in,array2_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 2 failed"
		all_okay = .FALSE.
	END IF

	! A single non-FillVal value should be replicated n times
	array10_in(:) = (/1.0_fbdp,FV_real,FV_real,FV_real,FV_real,FV_real,FV_real,FV_real,FV_real,FV_real/)
	array10_out(:) = 1.0_fbdp
	CALL set_depths(array10_in,10,FV_real)
	okay = test_arrays(array10_in,array10_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 3 failed"
		all_okay = .FALSE.
	END IF

	! A single non-FillVal value should be replicated n times
	array3_in(:) = (/1.0_fbdp,FV_real,FV_real/)
	array3_out(:) = 1.0_fbdp
	CALL set_depths(array3_in,3,FV_real)
	okay = test_arrays(array3_in,array3_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 4 failed"
		all_okay = .FALSE.
	END IF
		
	! A full list of n non-FillVals should be left unaltered.
	array2_in(:) = (/1.0_fbdp, 2.0_fbdp/)
	array2_out(:) = (/1.0_fbdp, 2.0_fbdp/)
	CALL set_depths(array2_in,2,FV_real)
	okay = test_arrays(array2_in,array2_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 5 failed"
		all_okay = .FALSE.
	END IF

	! A full list of n non-FillVals should be left unaltered.
	array3_in(:) = (/3.0_fbdp, 2.0_fbdp, 56.23_fbdp/)
	array3_out(:) = (/3.0_fbdp, 2.0_fbdp, 56.23_fbdp/)
	CALL set_depths(array3_in,3,FV_real)
	okay = test_arrays(array3_in,array3_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 6 failed"
		all_okay = .FALSE.
	END IF

	! A full list of n non-FillVals should be left unaltered.
	array10_in(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, -5.0_fbdp, 5.0_fbdp, 67.0_fbdp, 7.0_fbdp, 8.0_fbdp, 9.0_fbdp, 10.0_fbdp/)
	array10_out(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, -5.0_fbdp, 5.0_fbdp, 67.0_fbdp, 7.0_fbdp, 8.0_fbdp, 9.0_fbdp, 10.0_fbdp/)
	CALL set_depths(array10_in,10,FV_real)
	okay = test_arrays(array10_in,array10_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 7 failed"
		all_okay = .FALSE.
	END IF

	! Three non-FillVal values should be used as the start, end, step in conjunction with n
	array10_in(:) = (/1.0_fbdp,10.0_fbdp,1.0_fbdp,FV_real,FV_real,FV_real,FV_real,FV_real,FV_real,FV_real/)
	array10_out(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp, 6.0_fbdp, 7.0_fbdp, 8.0_fbdp, 9.0_fbdp, 10.0_fbdp/)
	CALL set_depths(array10_in,10,FV_real)
	okay = test_arrays(array10_in,array10_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 8 failed"
		all_okay = .FALSE.
	END IF

	! Three non-FillVal values should be used as the start, end, step in conjunction with n
	array10_in(:) = (/1.0_fbdp, 3.25_fbdp, 0.25_fbdp, FV_real,FV_real,FV_real,FV_real,FV_real,FV_real,FV_real/)
	array10_out(:) = (/1.0_fbdp, 1.25_fbdp, 1.5_fbdp, 1.75_fbdp, 2.0_fbdp, 2.25_fbdp, 2.5_fbdp, 2.75_fbdp, 3.0_fbdp, 3.25_fbdp/)
	CALL set_depths(array10_in,10,FV_real)
	okay = test_arrays(array10_in,array10_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 9 failed"
		all_okay = .FALSE.
	END IF

	! Three non-FillVal values should be used as the start, end, step in conjunction with n
	array10_in(:) = (/3.25_fbdp, 1.0_fbdp, -0.25_fbdp, FV_real,FV_real,FV_real,FV_real,FV_real,FV_real,FV_real/)
	array10_out(:) = (/3.25_fbdp, 3.0_fbdp, 2.75_fbdp, 2.5_fbdp, 2.25_fbdp, 2.0_fbdp, 1.75_fbdp, 1.5_fbdp, 1.25_fbdp, 1.0_fbdp/)
	CALL set_depths(array10_in,10,FV_real)
	okay = test_arrays(array10_in,array10_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 11 failed"
		all_okay = .FALSE.
	END IF
	
   ! A single non-FillVal value should be replicated n times
	iarray1_in(:)=(/19991231/)
	iarray1_out(:)=(/19991231/)
	CALL set_date(iarray1_in,1,FV_int)
	okay = test_arrays(iarray1_in,iarray1_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 12 failed"
		all_okay = .FALSE.
	END IF

   ! A single non-FillVal value should be replicated n times
	iarray2_in(:) = (/19991231, FV_int/)
	iarray2_out(:) = (/19991231, 19991231/)
	CALL set_date(iarray2_in,2,FV_int)
	okay = test_arrays(iarray2_in,iarray2_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 13 failed"
		all_okay = .FALSE.
	END IF

	! A single non-FillVal value should be replicated n times
	iarray10_in(:) = (/20110101,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int/)
	iarray10_out(:) = 20110101
	CALL set_date(iarray10_in,10,FV_int)
	okay = test_arrays(iarray10_in,iarray10_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 14 failed"
		all_okay = .FALSE.
	END IF

	! A single non-FillVal value should be replicated n times
	iarray3_in(:) = (/20110101,FV_int,FV_int/)
	iarray3_out(:) = 20110101
	CALL set_date(iarray3_in,3,FV_int)
	okay = test_arrays(iarray3_in,iarray3_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 15 failed"
		all_okay = .FALSE.
	END IF
		
	! A full list of n non-FillVals should be left unaltered.
	iarray2_in(:) = (/19991231, 20000101/)
	iarray2_out(:) = (/19991231, 20000101/)
	CALL set_date(iarray2_in,2,FV_int)
	okay = test_arrays(iarray2_in,iarray2_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 16 failed"
		all_okay = .FALSE.
	END IF

	! A full list of n non-FillVals should be left unaltered.
	iarray3_in(:) = (/19840101, 19001231, 20500612/)
	iarray3_out(:) = (/19840101, 19001231, 20500612/)
	CALL set_date(iarray3_in,3,FV_int)
	okay = test_arrays(iarray3_in,iarray3_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 17 failed"
		all_okay = .FALSE.
	END IF

	! A full list of n non-FillVals should be left unaltered.
	iarray10_in(:) = (/20091231, 20100101, 20100102, 20100103, 20091231, &
	                    20100105, 20100106, 20091231, 20100108, 20100109/)
	iarray10_out(:) = (/20091231, 20100101, 20100102, 20100103, 20091231, &
	                    20100105, 20100106, 20091231, 20100108, 20100109/)
	CALL set_date(iarray10_in,10,FV_int)
	okay = test_arrays(iarray10_in,iarray10_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 18 failed"
		all_okay = .FALSE.
	END IF

	! Three non-FillVal values should be used as the start, end, step in conjunction with n
	iarray10_in(:) = (/20091231,20100109,1,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int/)
	iarray10_out(:) = (/20091231, 20100101, 20100102, 20100103, 20100104, &
	                    20100105, 20100106, 20100107, 20100108, 20100109/)
	CALL set_date(iarray10_in,10,FV_int)
	okay = test_arrays(iarray10_in,iarray10_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 19 failed"
		all_okay = .FALSE.
	END IF

	! Three non-FillVal values should be used as the start, end, step in conjunction with n
	iarray10_in(:) = (/20100109,20091231,-1,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int/)
	iarray10_out(:) = (/20100109, 20100108, 20100107, 20100106, 20100105, &
	                    20100104, 20100103, 20100102, 20100101, 20091231/)
	CALL set_date(iarray10_in,10,FV_int)
	okay = test_arrays(iarray10_in,iarray10_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 20 failed"
		all_okay = .FALSE.
	END IF

	! Three non-FillVal values should be used as the start, end, step in conjunction with n
	iarray10_in(:) = (/20100101,20100401,10,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int/)
	iarray10_out(:) = (/20100101, 20100111, 20100121, 20100131, 20100210, &
	                    20100220, 20100302, 20100312, 20100322, 20100401/)
	CALL set_date(iarray10_in,10,FV_int)
	okay = test_arrays(iarray10_in,iarray10_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 21 failed"
		all_okay = .FALSE.
	END IF

   ! A single non-FillVal value should be replicated n times
	iarray1_in(:)=(/0000/)
	iarray1_out(:)=(/0/)
	CALL set_time(iarray1_in,1,FV_int)
	okay = test_arrays(iarray1_in,iarray1_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 22 failed"
		all_okay = .FALSE.
	END IF

   ! A single non-FillVal value should be replicated n times
	iarray2_in(:) = (/0600, FV_int/)
	iarray2_out(:) = (/600, 600/)
	CALL set_time(iarray2_in,2,FV_int)
	okay = test_arrays(iarray2_in,iarray2_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 23 failed"
		all_okay = .FALSE.
	END IF

	! A full list of n non-FillVals should be left unaltered.
	iarray2_in(:) = (/0600, 1200/)
	iarray2_out(:) = (/0600, 1200/)
	CALL set_time(iarray2_in,2,FV_int)
	okay = test_arrays(iarray2_in,iarray2_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 24 failed"
		all_okay = .FALSE.
	END IF

	! A full list of n non-FillVals should be left unaltered.
	iarray10_in(:) = (/1, 2, 3, -5, 5, 67, 7, 8, 9, 10/)
	iarray10_out(:) = (/1, 2, 3, -5, 5, 67, 7, 8, 9, 10/)
	CALL set_time(iarray10_in,10,FV_int)
	okay = test_arrays(iarray10_in,iarray10_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 25 failed"
		all_okay = .FALSE.
	END IF

	! Three non-FillVal values should be used as the start, end, step in conjunction with n
	iarray10_in(:) = (/0000,0430,0030,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int/)
	iarray10_out(:) = (/0, 30, 100, 130, 200, &
	                    230, 300, 330, 400, 430/)
	CALL set_time(iarray10_in,10,FV_int)
	okay = test_arrays(iarray10_in,iarray10_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 26 failed"
		WRITE(*,*) iarray10_in
		all_okay = .FALSE.
	END IF

	! Three non-FillVal values should be used as the start, end, step in conjunction with n
	iarray10_in(:) = (/2200,0230,0030,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int/)
	iarray10_out(:) = (/2200, 2230, 2300, 2330, 0, &
	                    30, 100, 130, 200, 230/)
	CALL set_time(iarray10_in,10,FV_int)
	okay = test_arrays(iarray10_in,iarray10_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 27 failed"
		WRITE(*,*) iarray10_in
		all_okay = .FALSE.
	END IF

	! Three non-FillVal values should be used as the start, end, step in conjunction with n
	iarray10_in(:) = (/2200,1730,-0030,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int/)
	iarray10_out(:) = (/2200, 2130, 2100, 2030, 2000, &
	                    1930, 1900, 1830, 1800, 1730/)
	CALL set_time(iarray10_in,10,FV_int)
	okay = test_arrays(iarray10_in,iarray10_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 28 failed"
		WRITE(*,*) iarray10_in
		all_okay = .FALSE.
	END IF

	! Three non-FillVal values should be used as the start, end, step in conjunction with n
	iarray10_in(:) = (/0100,2200,-0020,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int/)
	iarray10_out(:) = (/0100, 0040, 0020, 0000, 2340, &
	                    2320, 2300, 2240, 2220, 2200/)
	CALL set_time(iarray10_in,10,FV_int)
	okay = test_arrays(iarray10_in,iarray10_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 29 failed"
		WRITE(*,*) iarray10_in
		all_okay = .FALSE.
	END IF

	! Three non-FillVal values should be used as the start, end, step in conjunction with n
	iarray10_in(:) = (/1700,0630,90,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int,FV_int/)
	iarray10_out(:) = (/1700, 1830, 2000, 2130, 2300, &
	                    0030, 0200, 0330, 0500, 0630/)
	CALL set_time(iarray10_in,10,FV_int)
	okay = test_arrays(iarray10_in,iarray10_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 30 failed"
		WRITE(*,*) iarray10_in
		all_okay = .FALSE.
	END IF

	! A single obs value should be replicated throught the array(obs,levels)
	array_4x3_in(:,1,1) = (/1.0,FV_real,FV_real/)
	array_4x3_in(:,2,1) = (/FV_real,FV_real,FV_real/)
	array_4x3_in(:,3,1) = (/FV_real,FV_real,FV_real/)
	array_4x3_in(:,4,1) = (/FV_real,FV_real,FV_real/)
	array_4x3_out(:,1,1) = (/1.0_fbdp, 1.0_fbdp, 1.0_fbdp/)
	array_4x3_out(:,2,1) = (/1.0_fbdp, 1.0_fbdp, 1.0_fbdp/)
	array_4x3_out(:,3,1) = (/1.0_fbdp, 1.0_fbdp, 1.0_fbdp/)
	array_4x3_out(:,4,1) = (/1.0_fbdp, 1.0_fbdp, 1.0_fbdp/)
	CALL set_obs_values(array_4x3_in,1,4,3,FV_real)
	okay = test_arrays(array_4x3_in,array_4x3_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 31 failed"
		WRITE(*,*) array_4x3_in
		WRITE(*,*)
		WRITE(*,*) array_4x3_out
		all_okay = .FALSE.
	END IF

	! A full specification of the obs values should be unaltered
	array_4x3_in(:,1,1) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp/)
	array_4x3_in(:,2,1) = (/4.0_fbdp, 5.0_fbdp, 6.0_fbdp/)
	array_4x3_in(:,3,1) = (/7.0_fbdp, 8.0_fbdp, 9.0_fbdp/)
	array_4x3_in(:,4,1) = (/10.0_fbdp, 11.0_fbdp, 12.0_fbdp/)
	array_4x3_out(:,1,1) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp/)
	array_4x3_out(:,2,1) = (/4.0_fbdp, 5.0_fbdp, 6.0_fbdp/)
	array_4x3_out(:,3,1) = (/7.0_fbdp, 8.0_fbdp, 9.0_fbdp/)
	array_4x3_out(:,4,1) = (/10.0_fbdp, 11.0_fbdp, 12.0_fbdp/)
	CALL set_obs_values(array_4x3_in,1,4,3,FV_real)
	okay = test_arrays(array_4x3_in,array_4x3_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 32 failed"
		WRITE(*,*) array_4x3_in
		WRITE(*,*)
		WRITE(*,*) array_4x3_out
		all_okay = .FALSE.
	END IF

	! A single profile should be replicated at all lat/lons
	array_4x3_in(:,1,1) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp/)
	array_4x3_in(:,2,1) = (/FV_real, FV_real, FV_real/)
	array_4x3_in(:,3,1) = (/FV_real, FV_real, FV_real/)
	array_4x3_in(:,4,1) = (/FV_real, FV_real, FV_real/)
	array_4x3_out(:,1,1) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp/)
	array_4x3_out(:,2,1) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp/)
	array_4x3_out(:,3,1) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp/)
	array_4x3_out(:,4,1) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp/)
	CALL set_obs_values(array_4x3_in,1,4,3,FV_real)
	okay = test_arrays(array_4x3_in,array_4x3_out)
	IF (okay .EQV. .FALSE.) THEN
	   WRITE(*,*) "Test 33 failed"
		WRITE(*,*) array_4x3_in
		WRITE(*,*)
		WRITE(*,*) array_4x3_out
		all_okay = .FALSE.
	END IF

	! A single profile position should only be given where nobs=1
	ALLOCATE(lat_array_in(1),	&
				lon_array_in(1),	&
				lat_array_out(1), &
				lon_array_out(1)	)
	lat_array_in(:) = (/1.0_fbdp/)
	lon_array_in(:) = (/1.0_fbdp/)
	lat_array_out(:) = (/1.0_fbdp/)
	lon_array_out(:) = (/1.0_fbdp/)
	CALL set_spatial_coords(lat_array_in,lon_array_in,1,FV_real)
	okay = test_arrays(lat_array_in,lat_array_out)
	okay_too = test_arrays(lon_array_in,lon_array_out)
	IF ((okay .EQV. .FALSE.) .OR. (okay_too .EQV. .FALSE.)) THEN
	   WRITE(*,*) "Test 34 failed"
		WRITE(*,*) lat_array_in
		WRITE(*,*) lon_array_in
		WRITE(*,*) lat_array_out
		WRITE(*,*) lon_array_out
		all_okay = .FALSE.
	END IF
	DEALLOCATE(lat_array_in,	&
				  lon_array_in,	&
				  lat_array_out,	&
				  lon_array_out	)

	! A single lat and list of lons should give nobs observations
	ALLOCATE(lat_array_in(4),	&
				lon_array_in(4),	&
				lat_array_out(4), &
				lon_array_out(4)	)
	lat_array_in(:) = (/1.0_fbdp, FV_real, FV_real, FV_real/)
	lon_array_in(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp/)
	lat_array_out(:) = (/1.0_fbdp, 1.0_fbdp, 1.0_fbdp, 1.0_fbdp/)
	lon_array_out(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp/)
	CALL set_spatial_coords(lat_array_in,lon_array_in,4,FV_real)
	okay = test_arrays(lat_array_in,lat_array_out)
	okay_too = test_arrays(lon_array_in,lon_array_out)
	IF ((okay .EQV. .FALSE.) .OR. (okay_too .EQV. .FALSE.)) THEN
	   WRITE(*,*) "Test 35 failed"
		WRITE(*,*) lat_array_in
		WRITE(*,*) lon_array_in
		WRITE(*,*) lat_array_out
		WRITE(*,*) lon_array_out
		all_okay = .FALSE.
	END IF
	DEALLOCATE(lat_array_in,	&
				  lon_array_in,	&
				  lat_array_out,	&
				  lon_array_out	)

	! A single lat and bounded lons should give nobs = no elements in expanded bounds
	ALLOCATE(lat_array_in(3),	&
				lon_array_in(3),	&
				lat_array_out(3), &
				lon_array_out(3)	)
	lat_array_in(:) = (/1.0_fbdp, FV_real, FV_real/)
	lon_array_in(:) = (/1.0_fbdp, 3.0_fbdp, 1.0_fbdp/)
	lat_array_out(:) = (/1.0_fbdp, 1.0_fbdp, 1.0_fbdp/)
	lon_array_out(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp/)
	CALL set_spatial_coords(lat_array_in,lon_array_in,3,FV_real)
	okay = test_arrays(lat_array_in,lat_array_out)
	okay_too = test_arrays(lon_array_in,lon_array_out)
	IF ((okay .EQV. .FALSE.) .OR. (okay_too .EQV. .FALSE.)) THEN
	   WRITE(*,*) "Test 36 failed"
		WRITE(*,*) lat_array_in
		WRITE(*,*) lon_array_in
		WRITE(*,*) lat_array_out
		WRITE(*,*) lon_array_out
		all_okay = .FALSE.
	END IF
	DEALLOCATE(lat_array_in,	&
				  lon_array_in,	&
				  lat_array_out,	&
				  lon_array_out	)

	! A single lat and bounded lons should give nobs = no elements in expanded bounds
	ALLOCATE(lat_array_in(5),	&
				lon_array_in(5),	&
				lat_array_out(5), &
				lon_array_out(5)	)
	lat_array_in(:) = (/1.0_fbdp, FV_real, FV_real, FV_real, FV_real/)
	lon_array_in(:) = (/1.0_fbdp, 5.0_fbdp, 1.0_fbdp, FV_real, FV_real/)
	lat_array_out(:) = (/1.0_fbdp, 1.0_fbdp, 1.0_fbdp, 1.0_fbdp, 1.0_fbdp/)
	lon_array_out(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp/)
	CALL set_spatial_coords(lat_array_in,lon_array_in,5,FV_real)
	okay = test_arrays(lat_array_in,lat_array_out)
	okay_too = test_arrays(lon_array_in,lon_array_out)
	IF ((okay .EQV. .FALSE.) .OR. (okay_too .EQV. .FALSE.)) THEN
	   WRITE(*,*) "Test 37 failed"
		WRITE(*,*) lat_array_in
		WRITE(*,*) lon_array_in
		WRITE(*,*) lat_array_out
		WRITE(*,*) lon_array_out
		all_okay = .FALSE.
	END IF
	DEALLOCATE(lat_array_in,	&
				  lon_array_in,	&
				  lat_array_out,	&
				  lon_array_out	)

	! A single lon and bounded lats should give nobs = no elements in expanded bounds
	ALLOCATE(lat_array_in(5),	&
				lon_array_in(5),	&
				lat_array_out(5), &
				lon_array_out(5)	)
	lat_array_in(:) = (/1.0_fbdp, 5.0_fbdp, 1.0_fbdp, FV_real, FV_real/)
	lon_array_in(:) = (/1.0_fbdp, FV_real, FV_real, FV_real, FV_real/)
	lat_array_out(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp/)
	lon_array_out(:) = (/1.0_fbdp, 1.0_fbdp, 1.0_fbdp, 1.0_fbdp, 1.0_fbdp/)
	CALL set_spatial_coords(lat_array_in,lon_array_in,5,FV_real)
	okay = test_arrays(lat_array_in,lat_array_out)
	okay_too = test_arrays(lon_array_in,lon_array_out)
	IF ((okay .EQV. .FALSE.) .OR. (okay_too .EQV. .FALSE.)) THEN
	   WRITE(*,*) "Test 38 failed"
		WRITE(*,*) lat_array_in
		WRITE(*,*) lon_array_in
		WRITE(*,*) lat_array_out
		WRITE(*,*) lon_array_out
		all_okay = .FALSE.
	END IF
	DEALLOCATE(lat_array_in,	&
				  lon_array_in,	&
				  lat_array_out,	&
				  lon_array_out	)

	! A list of lats and lons with same number of elements as nobs should be unaltered
	ALLOCATE(lat_array_in(5),	&
				lon_array_in(5),	&
				lat_array_out(5), &
				lon_array_out(5)	)
	lat_array_in(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp/)
	lon_array_in(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp/)
	lat_array_out(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp/)
	lon_array_out(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp/)
	CALL set_spatial_coords(lat_array_in,lon_array_in,5,FV_real)
	okay = test_arrays(lat_array_in,lat_array_out)
	okay_too = test_arrays(lon_array_in,lon_array_out)
	IF ((okay .EQV. .FALSE.) .OR. (okay_too .EQV. .FALSE.)) THEN
	   WRITE(*,*) "Test 39 failed"
		WRITE(*,*) lat_array_in
		WRITE(*,*) lon_array_in
		WRITE(*,*) lat_array_out
		WRITE(*,*) lon_array_out
		all_okay = .FALSE.
	END IF
	DEALLOCATE(lat_array_in,	&
				  lon_array_in,	&
				  lat_array_out,	&
				  lon_array_out	)

	! A list of lats and lons with fewer elements as nobs should be gridded 
	! to produce nobs observations - list all lons at same lat, before stepping in lat.
	ALLOCATE(lat_array_in(25),	&
				lon_array_in(25),	&
				lat_array_out(25), &
				lon_array_out(25)	)
	lat_array_in(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp,&
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real/)
	lon_array_in(:) = (/10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp, 50.0_fbdp,&
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real/)
	lat_array_out(:) = (/1.0_fbdp, 1.0_fbdp, 1.0_fbdp, 1.0_fbdp, 1.0_fbdp, & 
								2.0_fbdp, 2.0_fbdp, 2.0_fbdp, 2.0_fbdp, 2.0_fbdp, &
								3.0_fbdp, 3.0_fbdp, 3.0_fbdp, 3.0_fbdp, 3.0_fbdp, &
								4.0_fbdp, 4.0_fbdp, 4.0_fbdp, 4.0_fbdp, 4.0_fbdp, &
								5.0_fbdp, 5.0_fbdp, 5.0_fbdp, 5.0_fbdp, 5.0_fbdp/)
	lon_array_out(:) = (/10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp, 50.0_fbdp,&
								10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp, 50.0_fbdp,&
								10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp, 50.0_fbdp,&
								10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp, 50.0_fbdp,&
								10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp, 50.0_fbdp/)
	CALL set_spatial_coords(lat_array_in,lon_array_in,25,FV_real)
	okay = test_arrays(lat_array_in,lat_array_out)
	okay_too = test_arrays(lon_array_in,lon_array_out)
	IF ((okay .EQV. .FALSE.) .OR. (okay_too .EQV. .FALSE.)) THEN
	   WRITE(*,*) "Test 40 failed"
		WRITE(*,*) lat_array_in
		WRITE(*,*) lon_array_in
		WRITE(*,*) lat_array_out
		WRITE(*,*) lon_array_out
		all_okay = .FALSE.
	END IF
	DEALLOCATE(lat_array_in,	&
				  lon_array_in,	&
				  lat_array_out,	&
				  lon_array_out	)

	! A list of lats and lon bounds with fewer elements as nobs should be gridded 
	! to produce nobs observations - list all lons at same lat, before stepping in lat.
	ALLOCATE(lat_array_in(25),	&
				lon_array_in(25),	&
				lat_array_out(25), &
				lon_array_out(25)	)
	lat_array_in(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp,&
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real/)
	lon_array_in(:) = (/10.0_fbdp, 50.0_fbdp, 10.0_fbdp, FV_real, FV_real,&
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real/)
	lat_array_out(:) = (/1.0_fbdp, 1.0_fbdp, 1.0_fbdp, 1.0_fbdp, 1.0_fbdp, & 
								2.0_fbdp, 2.0_fbdp, 2.0_fbdp, 2.0_fbdp, 2.0_fbdp, &
								3.0_fbdp, 3.0_fbdp, 3.0_fbdp, 3.0_fbdp, 3.0_fbdp, &
								4.0_fbdp, 4.0_fbdp, 4.0_fbdp, 4.0_fbdp, 4.0_fbdp, &
								5.0_fbdp, 5.0_fbdp, 5.0_fbdp, 5.0_fbdp, 5.0_fbdp/)
	lon_array_out(:) = (/10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp, 50.0_fbdp,&
								10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp, 50.0_fbdp,&
								10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp, 50.0_fbdp,&
								10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp, 50.0_fbdp,&
								10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp, 50.0_fbdp/)
	CALL set_spatial_coords(lat_array_in,lon_array_in,25,FV_real)
	okay = test_arrays(lat_array_in,lat_array_out)
	okay_too = test_arrays(lon_array_in,lon_array_out)
	IF ((okay .EQV. .FALSE.) .OR. (okay_too .EQV. .FALSE.)) THEN
	   WRITE(*,*) "Test 41 failed"
		WRITE(*,*) lat_array_in
		WRITE(*,*) lon_array_in
		WRITE(*,*) lat_array_out
		WRITE(*,*) lon_array_out
		all_okay = .FALSE.
	END IF
	DEALLOCATE(lat_array_in,	&
				  lon_array_in,	&
				  lat_array_out,	&
				  lon_array_out	)


	! A list of lats and lon bounds with the same number of elements as nobs should be 
	! unaltered expect for bounds expansion
	ALLOCATE(lat_array_in(5),	&
				lon_array_in(5),	&
				lat_array_out(5), &
				lon_array_out(5)	)
	lat_array_in(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp/)
	lon_array_in(:) = (/10.0_fbdp, 50.0_fbdp, 10.0_fbdp, FV_real, FV_real/)
	lat_array_out(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp/)
	lon_array_out(:) = (/10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp, 50.0_fbdp/)
	CALL set_spatial_coords(lat_array_in,lon_array_in,5,FV_real)
	okay = test_arrays(lat_array_in,lat_array_out)
	okay_too = test_arrays(lon_array_in,lon_array_out)
	IF ((okay .EQV. .FALSE.) .OR. (okay_too .EQV. .FALSE.)) THEN
	   WRITE(*,*) "Test 42 failed"
		WRITE(*,*) lat_array_in
		WRITE(*,*) lon_array_in
		WRITE(*,*) lat_array_out
		WRITE(*,*) lon_array_out
		all_okay = .FALSE.
	END IF
	DEALLOCATE(lat_array_in,	&
				  lon_array_in,	&
				  lat_array_out,	&
				  lon_array_out	)
	

	! A list of lons and lat bounds with the same number of elements as nobs should be 
	! unaltered expect for bounds expansion
	ALLOCATE(lat_array_in(5),	&
				lon_array_in(5),	&
				lat_array_out(5), &
				lon_array_out(5)	)
	lat_array_in(:) = (/10.0_fbdp, 50.0_fbdp, 10.0_fbdp, FV_real, FV_real/)
	lon_array_in(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp/)
	lat_array_out(:) = (/10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp, 50.0_fbdp/)
	lon_array_out(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp/)
	CALL set_spatial_coords(lat_array_in,lon_array_in,5,FV_real)
	okay = test_arrays(lat_array_in,lat_array_out)
	okay_too = test_arrays(lon_array_in,lon_array_out)
	IF ((okay .EQV. .FALSE.) .OR. (okay_too .EQV. .FALSE.)) THEN
	   WRITE(*,*) "Test 43 failed"
		WRITE(*,*) lat_array_in
		WRITE(*,*) lon_array_in
		WRITE(*,*) lat_array_out
		WRITE(*,*) lon_array_out
		all_okay = .FALSE.
	END IF
	DEALLOCATE(lat_array_in,	&
				  lon_array_in,	&
				  lat_array_out,	&
				  lon_array_out	)
	

	! A list of lons and lat bounds with fewer elements as nobs should be gridded 
	! to produce nobs observations - list all lons at same lat, before stepping in lat.
	ALLOCATE(lat_array_in(25),	&
				lon_array_in(25),	&
				lat_array_out(25), &
				lon_array_out(25)	)
	lat_array_in(:) = (/10.0_fbdp, 50.0_fbdp, 10.0_fbdp, FV_real, FV_real,&
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real/)
	lon_array_in(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp,&
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real/)
	lat_array_out(:) = (/10.0_fbdp, 10.0_fbdp, 10.0_fbdp, 10.0_fbdp, 10.0_fbdp, & 
								20.0_fbdp, 20.0_fbdp, 20.0_fbdp, 20.0_fbdp, 20.0_fbdp, &
								30.0_fbdp, 30.0_fbdp, 30.0_fbdp, 30.0_fbdp, 30.0_fbdp, &
								40.0_fbdp, 40.0_fbdp, 40.0_fbdp, 40.0_fbdp, 40.0_fbdp, &
								50.0_fbdp, 50.0_fbdp, 50.0_fbdp, 50.0_fbdp, 50.0_fbdp/)
	lon_array_out(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp,&
								1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp,&
								1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp,&
								1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp,&
								1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp/)
	CALL set_spatial_coords(lat_array_in,lon_array_in,25,FV_real)
	okay = test_arrays(lat_array_in,lat_array_out)
	okay_too = test_arrays(lon_array_in,lon_array_out)
	IF ((okay .EQV. .FALSE.) .OR. (okay_too .EQV. .FALSE.)) THEN
	   WRITE(*,*) "Test 44 failed"
		WRITE(*,*) lat_array_in
		WRITE(*,*) lon_array_in
		WRITE(*,*) lat_array_out
		WRITE(*,*) lon_array_out
		all_okay = .FALSE.
	END IF
	DEALLOCATE(lat_array_in,	&
				  lon_array_in,	&
				  lat_array_out,	&
				  lon_array_out	)


	! A single lat and bounded lons should give nobs = no elements in expanded bounds
	ALLOCATE(lat_array_in(5),	&
				lon_array_in(5),	&
				lat_array_out(5), &
				lon_array_out(5)	)
	lat_array_in(:) = (/1.0_fbdp, FV_real, FV_real, FV_real, FV_real/)
	lon_array_in(:) = (/1.0_fbdp, 5.0_fbdp, 1.0_fbdp, FV_real, FV_real/)
	lat_array_out(:) = (/1.0_fbdp, 1.0_fbdp, 1.0_fbdp, 1.0_fbdp, 1.0_fbdp/)
	lon_array_out(:) = (/1.0_fbdp, 2.0_fbdp, 3.0_fbdp, 4.0_fbdp, 5.0_fbdp/)
	CALL set_spatial_coords(lat_array_in,lon_array_in,5,FV_real)
	okay = test_arrays(lat_array_in,lat_array_out)
	okay_too = test_arrays(lon_array_in,lon_array_out)
	IF ((okay .EQV. .FALSE.) .OR. (okay_too .EQV. .FALSE.)) THEN
	   WRITE(*,*) "Test 45 failed"
		WRITE(*,*) lat_array_in
		WRITE(*,*) lon_array_in
		WRITE(*,*) lat_array_out
		WRITE(*,*) lon_array_out
		all_okay = .FALSE.
	END IF
	DEALLOCATE(lat_array_in,	&
				  lon_array_in,	&
				  lat_array_out,	&
				  lon_array_out	)


	! Bounded lat and bounded lons - list all lons at same lat, before stepping in lat.
	ALLOCATE(lat_array_in(20),	&
				lon_array_in(20),	&
				lat_array_out(20), &
				lon_array_out(20)	)
	lat_array_in(:) = (/1.0_fbdp, 5.0_fbdp, 1.0_fbdp, FV_real, FV_real,&
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real/)
	lon_array_in(:) = (/10.0_fbdp, 40.0_fbdp, 10.0_fbdp, FV_real, FV_real,&
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real, &
	                    FV_real, FV_real, FV_real, FV_real, FV_real/)
	lat_array_out(:) = (/1.0_fbdp, 1.0_fbdp, 1.0_fbdp, 1.0_fbdp, & 
								2.0_fbdp, 2.0_fbdp, 2.0_fbdp, 2.0_fbdp, &
								3.0_fbdp, 3.0_fbdp, 3.0_fbdp, 3.0_fbdp, &
								4.0_fbdp, 4.0_fbdp, 4.0_fbdp, 4.0_fbdp, &
								5.0_fbdp, 5.0_fbdp, 5.0_fbdp, 5.0_fbdp/)
	lon_array_out(:) = (/10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp,&
								10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp,&
								10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp,&
								10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp,&
								10.0_fbdp, 20.0_fbdp, 30.0_fbdp, 40.0_fbdp/)
	CALL set_spatial_coords(lat_array_in,lon_array_in,20,FV_real)
	okay = test_arrays(lat_array_in,lat_array_out)
	okay_too = test_arrays(lon_array_in,lon_array_out)
	IF ((okay .EQV. .FALSE.) .OR. (okay_too .EQV. .FALSE.)) THEN
	   WRITE(*,*) "Test 46 failed"
		WRITE(*,*) lat_array_in
		WRITE(*,*) lon_array_in
		WRITE(*,*) lat_array_out
		WRITE(*,*) lon_array_out
		all_okay = .FALSE.
	END IF
	DEALLOCATE(lat_array_in,	&
				  lon_array_in,	&
				  lat_array_out,	&
				  lon_array_out	)


	IF (all_okay) WRITE(*,*) "All tests passed"
	
	END SUBROUTINE tester

END MODULE test_fbgenerate	
