MODULE daymod
   !!======================================================================
   !!                       ***  MODULE  daymod  ***
   !! Ocean        :  calendar
   !!=====================================================================
   !! History :  OPA  ! 1994-09  (M. Pontaud M. Imbard)  Original code
   !!                 ! 1997-03  (O. Marti)
   !!                 ! 1997-05  (G. Madec)
   !!                 ! 1997-08  (M. Imbard)
   !!   NEMO     1.0  ! 2003-09  (G. Madec)  F90 + nyear, nmonth, nday
   !!                 ! 2004-01  (A.M. Treguier) new calculation based on adatrj
   !!                 ! 2006-08  (G. Madec)  surface module major update
   !!                 ! 2015-11  (D. Lea) Allow non-zero initial time of day
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   day        : calendar
   !!
   !!           -------------------------------
   !!           ----------- WARNING -----------
   !!
   !!   we suppose that the time step is deviding the number of second of in a day
   !!             ---> MOD( rday, rdt ) == 0
   !!
   !!           ----------- WARNING -----------
   !!           -------------------------------
   !!
   !!----------------------------------------------------------------------
   USE dom_oce        ! ocean space and time domain
   USE phycst         ! physical constants
   USE in_out_manager ! I/O manager
   USE iom            !
   USE ioipsl  , ONLY :   ymds2ju   ! for calendar
   USE prtctl         ! Print control
   USE timing         ! Timing

   IMPLICIT NONE
   PRIVATE

   PUBLIC   day        ! called by step.F90
   PUBLIC   day_init   ! called by istate.F90
   PUBLIC   day_mth    ! Needed by TAM

   INTEGER, PUBLIC ::   nsecd, nsecd05, ndt, ndt05   !: (PUBLIC for TAM)

   !!----------------------------------------------------------------------
   !! NEMO/OCE 4.0 , NEMO Consortium (2018)
   !! $Id: daymod.F90 6140 2015-12-21 11:35:23Z timgraham $
   !! Software governed by the CeCILL licence     (./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE day_init
      !!----------------------------------------------------------------------
      !!                   ***  ROUTINE day_init  ***
      !!
      !! ** Purpose :   Initialization of the calendar values to their values 1 time step before nit000
      !!                because day will be called at the beginning of step
      !!
      !! ** Action  : - nyear        : current year
      !!              - nmonth       : current month of the year nyear
      !!              - nday         : current day of the month nmonth
      !!              - nday_year    : current day of the year nyear
      !!              - nsec_year    : current time step counted in second since 00h jan 1st of the current year
      !!              - nsec_month   : current time step counted in second since 00h 1st day of the current month
      !!              - nsec_day     : current time step counted in second since 00h of the current day
      !!              - nsec1jan000  : second since Jan. 1st 00h of nit000 year and Jan. 1st 00h of the current year
      !!              - nmonth_len, nyear_len, nmonth_half, nmonth_end through day_mth
      !!----------------------------------------------------------------------
      INTEGER  ::   inbday, idweek
      REAL(wp) ::   zjul
      !!----------------------------------------------------------------------
      !
      ! max number of seconds between each restart
      IF( REAL( nitend - nit000 + 1 ) * rdt > REAL( HUGE( nsec1jan000 ) ) ) THEN
         CALL ctl_stop( 'The number of seconds between each restart exceeds the integer 4 max value: 2^31-1. ',   &
            &           'You must do a restart at higher frequency (or remove this stop and recompile the code in I8)' )
      ENDIF
      ! all calendar staff is based on the fact that MOD( rday, rdt ) == 0
      IF( MOD( rday     , rdt   ) /= 0. )   CALL ctl_stop( 'the time step must devide the number of second of in a day' )
      IF( MOD( rday     , 2.    ) /= 0. )   CALL ctl_stop( 'the number of second of in a day must be an even number'    )
      IF( MOD( rdt      , 2.    ) /= 0. )   CALL ctl_stop( 'the time step (in second) must be an even number'           )
      nsecd   = NINT(rday       )
      nsecd05 = NINT(0.5 * rday )
      ndt     = NINT(      rdt  )
      ndt05   = NINT(0.5 * rdt  )


      ! set the calandar from ndastp (read in restart file and namelist)

      nyear   =   ndastp / 10000
      nmonth  = ( ndastp - (nyear * 10000) ) / 100
      nday    =   ndastp - (nyear * 10000) - ( nmonth * 100 )

      nhour   =   nn_time0 / 100
      nminute = ( nn_time0 - nhour * 100 )

      CALL ymds2ju( nyear, nmonth, nday, nhour*3600._wp+nminute*60._wp, fjulday )  
      IF( ABS(fjulday - REAL(NINT(fjulday),wp)) < 0.1 / rday )   fjulday = REAL(NINT(fjulday),wp)   ! avoid truncation error
      IF( nn_time0*3600 - ndt05 .lt. 0 ) fjulday = fjulday + 1.                    ! move back to the day at nit000 (and not at nit000 - 1)

      nsec1jan000 = 0
      CALL day_mth

      IF ( nday == 0 ) THEN     !   for ex if ndastp = ndate0 - 1
         nmonth = nmonth - 1
         nday = nmonth_len(nmonth)
      ENDIF
      IF ( nmonth == 0 ) THEN   ! go at the end of previous year
         nmonth = 12
         nyear = nyear - 1
         nsec1jan000 = nsec1jan000 - nsecd * nyear_len(0)
         IF( nleapy == 1 )   CALL day_mth
      ENDIF

      ! day since january 1st
      nday_year = nday + SUM( nmonth_len(1:nmonth - 1) )

      !compute number of days between last monday and today
      CALL ymds2ju( 1900, 01, 01, 0.0, zjul )  ! compute julian day value of 01.01.1900 (our reference that was a Monday)
      inbday = FLOOR(fjulday - zjul)            ! compute nb day between  01.01.1900 and start of current day
      idweek = MOD(inbday, 7)                  ! compute nb day between last monday and current day
      IF (idweek .lt. 0) idweek=idweek+7       ! Avoid negative values for dates before 01.01.1900

      ! number of seconds since the beginning of current year/month/week/day at the middle of the time-step
      IF (nhour*3600+nminute*60-ndt05 .gt. 0) THEN
         ! 1 timestep before current middle of first time step is still the same day
         nsec_year  = (nday_year-1) * nsecd + nhour*3600+nminute*60 - ndt05 
         nsec_month = (nday-1)      * nsecd + nhour*3600+nminute*60 - ndt05    
      ELSE
         ! 1 time step before the middle of the first time step is the previous day 
         nsec_year  = nday_year * nsecd + nhour*3600+nminute*60 - ndt05 
         nsec_month = nday      * nsecd + nhour*3600+nminute*60 - ndt05   
      ENDIF
      nsec_week  = idweek    * nsecd + nhour*3600+nminute*60 - ndt05
      nsec_day   =             nhour*3600+nminute*60 - ndt05 
      IF( nsec_day .lt. 0 ) nsec_day = nsec_day + nsecd
      IF( nsec_week .lt. 0 ) nsec_week = nsec_week + nsecd*7

      ! control print
      IF(lwp) WRITE(numout,'(a,i6,a,i2,a,i2,a,i8,a,i8,a,i8,a,i8)')' =======>> 1/2 time step before the start of the run DATE Y/M/D = ',   &
           &                   nyear, '/', nmonth, '/', nday, '  nsec_day:', nsec_day, '  nsec_week:', nsec_week, '  &
           &                   nsec_month:', nsec_month , '  nsec_year:' , nsec_year

      ! Up to now, calendar parameters are related to the end of previous run (nit000-1)
      ! call day to set the calendar parameters at the begining of the current simulaton. needed by iom_init
      CALL day( nit000 )
      !
   END SUBROUTINE day_init


   SUBROUTINE day_mth
      !!----------------------------------------------------------------------
      !!                   ***  ROUTINE day_init  ***
      !!
      !! ** Purpose :   calendar values related to the months
      !!
      !! ** Action  : - nmonth_len    : length in days of the months of the current year
      !!              - nyear_len     : length in days of the previous/current year
      !!              - nmonth_half   : second since the beginning of the year and the halft of the months
      !!              - nmonth_end    : second since the beginning of the year and the end of the months
      !!----------------------------------------------------------------------
      INTEGER  ::   jm               ! dummy loop indice
      !!----------------------------------------------------------------------

      ! length of the month of the current year (from nleapy, read in namelist)
      IF ( nleapy < 2 ) THEN
         nmonth_len(:) = (/ 31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31 /)
         nyear_len(:) = 365
         IF ( nleapy == 1 ) THEN   ! we are using calandar with leap years
            IF ( MOD(nyear-1, 4) == 0 .AND. ( MOD(nyear-1, 400) == 0 .OR. MOD(nyear-1, 100) /= 0 ) ) THEN
               nyear_len(0)  = 366
            ENDIF
            IF ( MOD(nyear  , 4) == 0 .AND. ( MOD(nyear  , 400) == 0 .OR. MOD(nyear  , 100) /= 0 ) ) THEN
               nmonth_len(2) = 29
               nyear_len(1)  = 366
            ENDIF
            IF ( MOD(nyear+1, 4) == 0 .AND. ( MOD(nyear+1, 400) == 0 .OR. MOD(nyear+1, 100) /= 0 ) ) THEN
               nyear_len(2)  = 366
            ENDIF
         ENDIF
      ELSE
         nmonth_len(:) = nleapy   ! all months with nleapy days per year
         nyear_len(:) = 12 * nleapy
      ENDIF

      ! half month in second since the begining of the year:
      ! time since Jan 1st   0     1     2    ...    11    12    13
      !          ---------*--|--*--|--*--| ... |--*--|--*--|--*--|--------------------------------------
      !                 <---> <---> <--->  ...  <---> <---> <--->
      ! month number      0     1     2    ...    11    12    13
      !
      ! nmonth_half(jm) = rday * REAL( 0.5 * nmonth_len(jm) + SUM(nmonth_len(1:jm-1)) )
      nmonth_half(0) = - nsecd05 * nmonth_len(0)
      DO jm = 1, 13
         nmonth_half(jm) = nmonth_half(jm-1) + nsecd05 * ( nmonth_len(jm-1) + nmonth_len(jm) )
      END DO

      nmonth_end(0) = 0
      DO jm = 1, 13
         nmonth_end(jm) = nmonth_end(jm-1) + nsecd * nmonth_len(jm)
      END DO
      !
   END SUBROUTINE


   SUBROUTINE day( kt )
      !!----------------------------------------------------------------------
      !!                      ***  ROUTINE day  ***
      !!
      !! ** Purpose :   Compute the date with a day iteration IF necessary.
      !!
      !! ** Method  : - ???
      !!
      !! ** Action  : - nyear     : current year
      !!              - nmonth    : current month of the year nyear
      !!              - nday      : current day of the month nmonth
      !!              - nday_year : current day of the year nyear
      !!              - ndastp    : = nyear*10000 + nmonth*100 + nday
      !!              - adatrj    : date in days since the beginning of the run
      !!              - nsec_year : current time of the year (in second since 00h, jan 1st)
      !!----------------------------------------------------------------------
      INTEGER, INTENT(in) ::   kt        ! ocean time-step indices
      !
      CHARACTER (len=25) ::   charout
      REAL(wp)           ::   zprec      ! fraction of day corresponding to 0.1 second
      !!----------------------------------------------------------------------
      !
      IF( nn_timing == 1 )  CALL timing_start('day')
      !
      zprec = 0.1 / rday
      !                                                 ! New time-step
      nsec_year  = nsec_year  + ndt
      nsec_month = nsec_month + ndt
      nsec_week  = nsec_week  + ndt
      nsec_day   = nsec_day   + ndt
      adatrj  = adatrj  + rdt / rday
      fjulday = fjulday + rdt / rday
      IF( ABS(fjulday - REAL(NINT(fjulday),wp)) < zprec )   fjulday = REAL(NINT(fjulday),wp)   ! avoid truncation error
      IF( ABS(adatrj  - REAL(NINT(adatrj ),wp)) < zprec )   adatrj  = REAL(NINT(adatrj ),wp)   ! avoid truncation error

      IF( nsec_day > nsecd ) THEN                       ! New day
         !
         nday      = nday + 1
         nday_year = nday_year + 1
         nsec_day  = ndt05
         !
         IF( nday == nmonth_len(nmonth) + 1 ) THEN      ! New month
            nday   = 1
            nmonth = nmonth + 1
            nsec_month = ndt05
            IF( nmonth == 13 ) THEN                     ! New year
               nyear     = nyear + 1
               nmonth    = 1
               nday_year = 1
               nsec_year = ndt05
               nsec1jan000 = nsec1jan000 + nsecd * nyear_len(1)
               IF( nleapy == 1 )   CALL day_mth
            ENDIF
         ENDIF
         !
         ndastp = nyear * 10000 + nmonth * 100 + nday   ! New date
         !
         !compute first day of the year in julian days
         CALL ymds2ju( nyear, 01, 01, 0.0, fjulstartyear )
         !
         IF(lwp) WRITE(numout,'(a,i8,a,i4.4,a,i2.2,a,i2.2,a,i3.3)') '======>> time-step =', kt,   &
              &   '      New day, DATE Y/M/D = ', nyear, '/', nmonth, '/', nday, '      nday_year = ', nday_year
         IF(lwp) WRITE(numout,'(a,i8,a,i7,a,i5)') '         nsec_year = ', nsec_year,   &
              &   '   nsec_month = ', nsec_month, '   nsec_day = ', nsec_day, '   nsec_week = ', nsec_week
      ENDIF

      IF( nsec_week > 7*nsecd )   nsec_week = ndt05     ! New week

      IF(ln_ctl) THEN
         WRITE(charout,FMT="('kt =', I4,'  d/m/y =',I2,I2,I4)") kt, nday, nmonth, nyear
         CALL prt_ctl_info(charout)
      ENDIF

      !
      IF( nn_timing == 1 )  CALL timing_stop('day')
      !
   END SUBROUTINE day


   !!======================================================================
END MODULE daymod
