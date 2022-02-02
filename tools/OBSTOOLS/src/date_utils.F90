MODULE date_utils

   USE toolspar_kind
   IMPLICIT NONE

CONTAINS

   SUBROUTINE add_date(initial_date,hours,final_date)

      ! Add a number of hours to initial_date and return it in final_date

      IMPLICIT NONE


      !! Arguments
      INTEGER,INTENT(in) :: initial_date ! Initial date (YYYYMMDDHH)
      INTEGER,INTENT(in) :: hours        ! Number of hours to add
      INTEGER,INTENT(out) :: final_date  ! Final date (YYYYMMDDHH)

      !! Local variables

      INTEGER :: isec,imin,ihours,iyear,imon,iday ! temporary results
      REAL(dp):: juld

      CALL split_date(initial_date,iyear,imon,iday,ihours)

      CALL greg2jul(0,0,ihours,iday,imon,iyear,juld)

      juld=juld+REAL(hours)/24.0

      CALL jul2greg(isec,imin,ihours,iday,imon,iyear,juld)

      final_date=iyear*1000000+imon*10000+iday*100+ihours

   END SUBROUTINE add_date


   SUBROUTINE add_days_to_date(initial_date,days,final_date)

      ! Add a number of days to initial_date and return it in final_date

      IMPLICIT NONE


      !! Arguments
      INTEGER,INTENT(in) :: initial_date ! Initial date (YYYYMMDD)
      INTEGER,INTENT(in) :: days         ! Number of days to add
      INTEGER,INTENT(out) :: final_date  ! Final date (YYYYMMDD)

      !! Local variables

      INTEGER :: isec,imin,ihours,iyear,imon,iday ! temporary results
      REAL(dp):: juld

		! Account for lack of hours in date format (initial_date*100)
      CALL split_date(initial_date*100,iyear,imon,iday,ihours)

      CALL greg2jul(0,0,ihours,iday,imon,iyear,juld)

      juld=juld+REAL(days)

      CALL jul2greg(isec,imin,ihours,iday,imon,iyear,juld)

      final_date=(iyear*1000000+imon*10000+iday*100+ihours)/100

   END SUBROUTINE add_days_to_date


   SUBROUTINE split_date(iyyyymmddhh,iyyyy,imm,idd,ihh)

      ! Splits a date in YYYYMMDDHH format into iyyyy, imm, idd, ihh

      IMPLICIT NONE
      INTEGER,INTENT(in) :: iyyyymmddhh
      INTEGER,INTENT(out) :: iyyyy,imm,idd,ihh

      iyyyy=iyyyymmddhh/1000000
      imm=iyyyymmddhh/10000-iyyyy*100
      idd=iyyyymmddhh/100-(iyyyy*10000+imm*100)
      ihh=MOD(iyyyymmddhh,100)

   END SUBROUTINE split_date

   SUBROUTINE jul2greg( ksec, kminut, khour, kday, kmonth, kyear, &
      &                           prelday )

      IMPLICIT NONE
      !!-----------------------------------------------------------------------
      !!
      !!                     ***  ROUTINE jul2greg  ***
      !!
      !! ** Purpose : Take the relative time in days and re-express in terms of
      !!              seconds, minutes, hours, days, month, year.
      !!
      !! ** Method  : Reference date : 19500101
      !!
      !! ** Action  :
      !!
      !! History
      !!      ! 06-04  (A. Vidard) Original
      !!      ! 06-05  (A. Vidard) Reformatted and refdate      
      !!      ! 06-10  (A. Weaver) Cleanup
      !!-----------------------------------------------------------------------

      ! * Arguments
      INTEGER, INTENT(OUT) :: &
         & ksec,   &
         & kminut, &
         & khour,  &
         & kday,   &
         & kmonth, &
         & kyear
      REAL(KIND=dp), INTENT(IN) :: &
         & prelday

      !! * Local declarations
      INTEGER, PARAMETER :: &
         & jpgreg = 2299161, &
         & jporef = 2433283, &
         & jparef = 2415021
      INTEGER :: &
         & ijulian, &
         & ij1,     &
         & ija,     &
         & ijb,     &
         & ijc,     &
         & ijd,     &
         & ije,     &
         & isec,    &
         & imin,    &
         & ihou,    &
         & iday,    &
         & imon,    &
         & iyea,    &
         & iref
      REAL(KIND=wp) :: &
         & zday, &
         & zref

      ! Main computation
      iref = jporef 

      zday = prelday
      ksec = NINT( 86400. * MOD( zday, 1.0_wp ) )

      IF ( ksec < 0. ) ksec = 86400. + ksec

      khour  = ksec / 3600
      kminut = ( ksec - 3600 * khour ) / 60
      ksec   = MOD( ksec , 60 )

      ijulian = iref + INT( zday )
      IF ( zday < 0. ) ijulian = ijulian - 1

      ! If input date after 10/15/1582 :
      IF ( ijulian >= jpgreg ) THEN
         ij1 = INT( ( DBLE( ijulian - 1867216 ) - 0.25 ) / 36524.25 )
         ija = ijulian + 1 + ij1 - INT( ( 0.25 * ij1 ) )
      ELSE
         ija = ijulian
      ENDIF

      ijb = ija + 1524
      ijc = INT( 6680. + ( DBLE ( ijb - 2439870 ) - 122.1 ) / 365.25 )
      ijd = 365 * ijc + INT( 0.25 * ijc )
      ije = INT( ( ijb - ijd ) / 30.6001 )
      kday = ijb - ijd - INT( 30.6001 * ije )
      kmonth = ije - 1
      IF ( kmonth > 12 ) kmonth = kmonth - 12
      kyear = ijc - 4715
      IF ( kmonth > 2 ) kyear = kyear - 1
      IF ( kyear <= 0 ) kyear = kyear - 1

   END SUBROUTINE jul2greg

   SUBROUTINE greg2jul( ksec, kmin, khour, kday, kmonth, kyear, pjulian )

      IMPLICIT NONE
      !!-----------------------------------------------------------------------
      !!
      !!                     ***  ROUTINE greg2jul  ***
      !!
      !! ** Purpose : Produce the time relative to the current date and time.
      !!
      !! ** Method  : The units are days, so hours and minutes transform to
      !!              fractions of a day. 
      !!
      !!              Reference date : 19500101
      !! ** Action  :
      !!
      !! History :
      !!      ! 06-04  (A. Vidard) Original
      !!      ! 06-04  (A. Vidard) Reformatted
      !!      ! 06-10  (A. Weaver) Cleanup
      !!-----------------------------------------------------------------------

      ! * Arguments
      INTEGER, INTENT(IN) :: &
         & ksec,   &
         & kmin,   &
         & khour,  & 
         & kday,   &
         & kmonth, & 
         & kyear
      REAL(KIND=dp), INTENT(OUT) :: &
         & pjulian

      !! * Local declarations
      INTEGER, PARAMETER :: &
         & jpgreg = 15 + 31 * ( 10 + 12 * 1582 ), &  ! Gregorian calendar introduction date
         & jpjref = 2433283                          ! Julian reference date: 19500101
      INTEGER :: &
         & ija,     &
         & ijy,     &
         & ijm,     &
         & ijultmp, &
         & ijyear

      ! Main computation
      ijyear = kyear
      IF ( ijyear < 0 ) ijyear = ijyear + 1
      IF ( kmonth > 2 ) THEN
         ijy = ijyear
         ijm = kmonth + 1
      ELSE
         ijy = ijyear  - 1
         ijm = kmonth + 13
      ENDIF
      ijultmp = INT( 365.25 * ijy ) + INT( 30.6001 * ijm ) + kday + 1720995
      IF ( kday + 31 * ( kmonth + 12 * ijyear ) >= jpgreg ) THEN
         ija = INT( 0.01 * ijy )
         ijultmp = ijultmp + 2 - ija + INT( 0.25 * ija )
      ENDIF
      pjulian = ( ijultmp - jpjref ) + ( ( 60 * khour + kmin ) * 60 + ksec ) / 86400.

   END SUBROUTINE greg2jul


   SUBROUTINE addseconds(iyear,imon,iday,ihour,imin,isec,iaddsec)

      ! Add iaddsecs to the date and return the new date (in place)

      !! Arguments

      INTEGER,intent(inout) :: iyear,imon,iday,ihour,imin,isec,iaddsec

      !! Local variables

      INTEGER :: itotsec,idays,isecs
      INTEGER :: mday(12) = (/31,28,31,30,31,30,31,31,30,31,30,31/) 

      itotsec=iaddsec+ihour*3600*imin*60+isec

      IF (itotsec<0) THEN
         WRITE(*,*)'Negative itotsec in addseconds'
         WRITE(*,*)'This does not work'
         RETURN
      ENDIF

      ihour=0
      imin=0
      isec=0

      idays=itotsec/86400
      isecs=itotsec-idays*86400
      iday=iday+idays

      ! Compute the date
      DO
         ! Leap year 
         mday(2)=28
         IF (MOD(iyear,4).EQ.0) mday(2)=29
         IF (MOD(iyear,100).EQ.0) mday(2)=28
         IF (MOD(iyear,400).EQ.0) mday(2)=29
         IF (MOD(iyear,4000).EQ.0) mday(2)=28

         IF (iday.GT.mday(imon))THEN
            iday=iday-mday(imon)
            imon=imon+1
            IF(imon.GT.12)THEN
               imon=1
               iyear=iyear+1
            ENDIF
         ELSE
            EXIT
         ENDIF

      ENDDO

      ! Set the time
      ihour=isecs/3600
      imin=isecs/60-ihour*60
      isec=isecs-ihour*3600-imin*60

   END SUBROUTINE addseconds

   INTEGER FUNCTION nextdate(idate)

      ! Return next date.
      ! Date format is assumed to be YYYYMMDD

      IMPLICIT NONE

      !! Arguments

      INTEGER :: idate ! Initial date

      !! Local variables

      INTEGER :: year,day,mon
      INTEGER :: mday(12) = (/31,28,31,30,31,30,31,31,30,31,30,31/)


      day=MOD(idate,100)
      mon=MOD((idate-day)/100,100)
      year=idate/10000

      mday(2)=28
      IF (MOD(year,4).EQ.0) mday(2)=29
      IF (MOD(year,100).EQ.0) mday(2)=28
      IF (MOD(year,400).EQ.0) mday(2)= 29
      IF (MOD(year,4000).EQ.0) mday(2) = 28

      day=day+1
      IF (day.GT.mday(mon))THEN
         day=1
         mon=mon+1
         IF(mon.GT.12)THEN
            mon=1
            year=year+1
         ENDIF
      ENDIF
      nextdate=year*10000+mon*100+day
      RETURN

   END FUNCTION nextdate

   INTEGER FUNCTION prevdate(idate)

      ! Return previous date.
      ! Date format is assumed to be YYYYMMDD

      IMPLICIT NONE

      !! Arguments

      INTEGER :: idate ! Initial date

      !! Local variables

      INTEGER :: year,day,mon
      INTEGER :: mday(12) = (/31,28,31,30,31,30,31,31,30,31,30,31/)


      day=MOD(idate,100)
      mon=MOD((idate-day)/100,100)
      year=idate/10000

      mday(2)=28
      IF (MOD(year,4).EQ.0) mday(2)=29
      IF (MOD(year,100).EQ.0) mday(2)=28
      IF (MOD(year,400).EQ.0) mday(2)= 29
      IF (MOD(year,4000).EQ.0) mday(2) = 28

      day=day-1
      IF (day.LT.1)THEN
         mon=mon-1
         IF(mon.LT.1)THEN
            mon=12
            year=year-1
         ENDIF
         day=mday(mon)
      ENDIF
      prevdate=year*10000+mon*100+day
      RETURN

   END FUNCTION prevdate

   INTEGER FUNCTION diffdate(idate1,idate2)

      ! Compute difference in days between dates
      ! Assumes YYYYMMDD format for dates

      IMPLICIT NONE

      !! Argument

      INTEGER :: idate1,idate2    ! Dates to be diffed.

      !! Local variables

      INTEGER :: itdate1,itdate2
      INTEGER :: it

      itdate1=MIN(idate1,idate2)
      itdate2=MAX(idate1,idate2)

      IF (itdate1==itdate2) THEN
         diffdate=0
         RETURN
      ENDIF
      diffdate=0
      it=itdate1
      DO 
         it=nextdate(it)
         diffdate=diffdate+1
         IF (it==itdate2) EXIT
      ENDDO
      RETURN

   END FUNCTION diffdate

   INTEGER FUNCTION difftime(itime1,itime2)

      ! Compute difference in minutes between times
      ! Assumes HHMM or HMM or MM or M format for dates
		!
		! ORDER MATTERS - itime1 is ealier time
		! Result is an integer number of minutes

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: itime1,itime2    ! Times to be diffed.
      INTEGER :: imin1, imin2, ihr1, ihr2

		ihr1 = (itime1/100)
		ihr2 = (itime2/100)
		
		imin1 = (itime1 - ihr1*100) + (60 * ihr1)
		imin2 = (itime2 - ihr2*100) + (60 * ihr2)
		
		! Assume that itime2 is later, so wrap around midnight if necessary.
		IF (imin2 < imin1) THEN
			imin2 = imin2 + 24*60
		END IF
		
		difftime = imin2 - imin1
		
   END FUNCTION difftime


   INTEGER FUNCTION add_mins_to_time(itime1, imin_add)

      ! Add number of minutes onto given time
      ! Assumes time in HHMM or HMM or MM or M format
		!
		! Result is in HHMM format

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: itime1,imin_add
      INTEGER :: imin1, ihr1, imin2, ihr2

		ihr1 = (itime1/100)
		
		! itime1 in minutes from previous midnight
		imin1 = (itime1 - ihr1*100) + (60 * ihr1)
		
		imin1 = imin1 + imin_add
		
		! Add 1day if time went nagative
		IF (imin1 < 0) THEN
			imin1 = imin1 + 24*60
		END IF
		
		! Turn number of minutes back into HHMM
		ihr2 = imin1/60
		imin2 = imin1 - ihr2*60

		DO
			IF (ihr2<0) THEN
				ihr2 = ihr2 + 24
			ELSE IF (ihr2>=24) THEN
				ihr2 = ihr2 - 24
			END IF
			IF ((ihr2>=0).OR.(ihr2<24)) EXIT
		END DO

		add_mins_to_time = ihr2*100 + imin2
	
   END FUNCTION add_mins_to_time


END MODULE date_utils
