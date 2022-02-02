MODULE restart
   !!======================================================================
   !!                     ***  MODULE  restart  ***
   !! Ocean restart :  write the ocean restart file
   !!======================================================================
   !! History :  OPA  !  1999-11  (M. Imbard)  Original code
   !!   NEMO     1.0  !  2002-08  (G. Madec)  F90: Free form
   !!            2.0  !  2006-07  (S. Masson)  use IOM for restart
   !!            3.3  !  2010-04  (M. Leclair, G. Madec)  modified LF-RA
   !!            - -  !  2010-10  (C. Ethe, G. Madec) TRC-TRA merge (T-S in 4D)
   !!            3.7  !  2014-01  (G. Madec) suppression of curl and hdiv from the restart
   !!             -   !  2014-12  (G. Madec) remove KPP scheme
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   rst_opn    : open the ocean restart file
   !!   rst_write  : write the ocean restart file
   !!   rst_read   : read the ocean restart file
   !!----------------------------------------------------------------------
   USE oce             ! ocean dynamics and tracers 
   USE dom_oce         ! ocean space and time domain
   USE sbc_ice         ! only lk_lim3 
   USE phycst          ! physical constants
   USE eosbn2          ! equation of state            (eos bn2 routine)
   USE trdmxl_oce      ! ocean active mixed layer tracers trends variables
   !
   USE in_out_manager  ! I/O manager
   USE iom             ! I/O module
   USE diurnal_bulk
   
   IMPLICIT NONE
   PRIVATE

   PUBLIC   rst_opn         ! routine called by step module
   PUBLIC   rst_write       ! routine called by step module
   PUBLIC   rst_read        ! routine called by istate module
   PUBLIC   rst_read_open   ! routine called in rst_read and (possibly) in dom_vvl_init

   !! * Substitutions
   !!----------------------------------------------------------------------
   !!                   ***  vectopt_loop_substitute  ***
   !!----------------------------------------------------------------------
   !! ** purpose :   substitute the inner loop start/end indices with CPP macro
   !!                allow unrolling of do-loop (useful with vector processors)
   !!----------------------------------------------------------------------
   !!----------------------------------------------------------------------
   !! NEMO/OPA 3.7 , NEMO Consortium (2014)
   !! $Id: vectopt_loop_substitute.h90 4990 2014-12-15 16:42:49Z timgraham $ 
   !! Software governed by the CeCILL licence (./LICENSE)
   !!----------------------------------------------------------------------
   !!----------------------------------------------------------------------
   !! NEMO/OCE 4.0 , NEMO Consortium (2018)
   !! $Id: restart.F90 6140 2015-12-21 11:35:23Z timgraham $
   !! Software governed by the CeCILL licence     (./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE rst_opn( kt )
      !!---------------------------------------------------------------------
      !!                   ***  ROUTINE rst_opn  ***
      !!                     
      !! ** Purpose : + initialization (should be read in the namelist) of nitrst 
      !!              + open the restart when we are one time step before nitrst
      !!                   - restart header is defined when kt = nitrst-1
      !!                   - restart data  are written when kt = nitrst
      !!              + define lrst_oce to .TRUE. when we need to define or write the restart
      !!----------------------------------------------------------------------
      INTEGER, INTENT(in) ::   kt     ! ocean time-step
      !!
      CHARACTER(LEN=20)   ::   clkt     ! ocean time-step deine as a character
      CHARACTER(LEN=50)   ::   clname   ! ocean output restart file name
      CHARACTER(lc)       ::   clpath   ! full path to ocean output restart file
      !!----------------------------------------------------------------------
      !
      IF( kt == nit000 ) THEN   ! default definitions
         lrst_oce = .FALSE.   
         IF( ln_rst_list ) THEN
            nrst_lst = 1
            nitrst = nstocklist( nrst_lst )
         ELSE
            nitrst = nitend
         ENDIF
      ENDIF

      ! frequency-based restart dumping (nn_stock)
      IF( .NOT. ln_rst_list .AND. MOD( kt - 1, nstock ) == 0 ) THEN   
         ! we use kt - 1 and not kt - nit000 to keep the same periodicity from the beginning of the experiment
         nitrst = kt + nstock - 1                  ! define the next value of nitrst for restart writing
         IF( nitrst > nitend )   nitrst = nitend   ! make sure we write a restart at the end of the run
      ENDIF
      ! to get better performances with NetCDF format:
      ! we open and define the ocean restart file one time step before writing the data (-> at nitrst - 1)
      ! except if we write ocean restart files every time step or if an ocean restart file was writen at nitend - 1
      IF( kt == nitrst - 1 .OR. nstock == 1 .OR. ( kt == nitend .AND. .NOT. lrst_oce ) ) THEN
         IF( nitrst <= nitend .AND. nitrst > 0 ) THEN 
            ! beware of the format used to write kt (default is i8.8, that should be large enough...)
            IF( nitrst > 999999999 ) THEN   ;   WRITE(clkt, *       ) nitrst
            ELSE                            ;   WRITE(clkt, '(i8.8)') nitrst
            ENDIF
            ! create the file
            clname = TRIM(cexper)//"_"//TRIM(ADJUSTL(clkt))//"_"//TRIM(cn_ocerst_out)
            clpath = TRIM(cn_ocerst_outdir)
            IF( clpath(LEN_TRIM(clpath):) /= '/' ) clpath = TRIM(clpath) // '/'
            IF(lwp) THEN
               WRITE(numout,*)
               SELECT CASE ( jprstlib )
               CASE DEFAULT         ;   WRITE(numout,*)                            &
                   '             open ocean restart NetCDF file: ',TRIM(clpath)//clname
               END SELECT
               IF ( snc4set%luse )      WRITE(numout,*) '             opened for NetCDF4 chunking and compression'
               IF( kt == nitrst - 1 ) THEN   ;   WRITE(numout,*) '             kt = nitrst - 1 = ', kt
               ELSE                          ;   WRITE(numout,*) '             kt = '             , kt
               ENDIF
            ENDIF
            !
            CALL iom_open( TRIM(clpath)//TRIM(clname), numrow, ldwrt = .TRUE., kiolib = jprstlib )
            lrst_oce = .TRUE.
         ENDIF
      ENDIF
      !
   END SUBROUTINE rst_opn


   SUBROUTINE rst_write( kt )
      !!---------------------------------------------------------------------
      !!                   ***  ROUTINE rstwrite  ***
      !!                     
      !! ** Purpose :   Write restart fields in the format corresponding to jprstlib
      !!
      !! ** Method  :   Write in numrow when kt == nitrst in NetCDF
      !!              file, save fields which are necessary for restart
      !!----------------------------------------------------------------------
      INTEGER, INTENT(in) ::   kt   ! ocean time-step
      !!----------------------------------------------------------------------

                     CALL iom_rstput( kt, nitrst, numrow, 'rdt'    , rdt       )   ! dynamics and tracer time step

      IF ( .NOT. ln_diurnal_only ) THEN
                     CALL iom_rstput( kt, nitrst, numrow, 'ub'     , ub        )     ! before fields
                     CALL iom_rstput( kt, nitrst, numrow, 'vb'     , vb        )
                     CALL iom_rstput( kt, nitrst, numrow, 'tb'     , tsb(:,:,:,jp_tem) )
                     CALL iom_rstput( kt, nitrst, numrow, 'sb'     , tsb(:,:,:,jp_sal) )
                     CALL iom_rstput( kt, nitrst, numrow, 'sshb'   , sshb      )
                     !
                     CALL iom_rstput( kt, nitrst, numrow, 'un'     , un        )     ! now fields
                     CALL iom_rstput( kt, nitrst, numrow, 'vn'     , vn        )
                     CALL iom_rstput( kt, nitrst, numrow, 'tn'     , tsn(:,:,:,jp_tem) )
                     CALL iom_rstput( kt, nitrst, numrow, 'sn'     , tsn(:,:,:,jp_sal) )
                     CALL iom_rstput( kt, nitrst, numrow, 'sshn'   , sshn      )
                     CALL iom_rstput( kt, nitrst, numrow, 'rhop'   , rhop      )

                  ! extra variable needed for the ice sheet coupling
                  IF ( ln_iscpl ) THEN 
                     CALL iom_rstput( kt, nitrst, numrow, 'tmask'  , tmask     ) ! need to extrapolate T/S
                     CALL iom_rstput( kt, nitrst, numrow, 'umask'  , umask     ) ! need to correct barotropic velocity
                     CALL iom_rstput( kt, nitrst, numrow, 'vmask'  , vmask     ) ! need to correct barotropic velocity
                     CALL iom_rstput( kt, nitrst, numrow, 'smask'  , ssmask    ) ! need to correct barotropic velocity
                     CALL iom_rstput( kt, nitrst, numrow, 'e3t_n', e3t_n(:,:,:) )   ! need to compute temperature correction
                     CALL iom_rstput( kt, nitrst, numrow, 'e3u_n', e3u_n(:,:,:) )   ! need to compute bt conservation
                     CALL iom_rstput( kt, nitrst, numrow, 'e3v_n', e3v_n(:,:,:) )   ! need to compute bt conservation
                     CALL iom_rstput( kt, nitrst, numrow, 'gdepw_n', gdepw_n(:,:,:) ) ! need to compute extrapolation if vvl
                  END IF
      ENDIF
      
      IF (ln_diurnal) CALL iom_rstput( kt, nitrst, numrow, 'Dsst', x_dsst    )  

      IF( kt == nitrst ) THEN
         CALL iom_close( numrow )     ! close the restart file (only at last time step)
!!gm         IF( .NOT. lk_trdmld )   lrst_oce = .FALSE.
!!gm  not sure what to do here   ===>>>  ask to Sebastian
         lrst_oce = .FALSE.
            IF( ln_rst_list ) THEN
               nrst_lst = MIN(nrst_lst + 1, SIZE(nstocklist,1))
               nitrst = nstocklist( nrst_lst )
            ENDIF
            lrst_oce = .FALSE.
      ENDIF
      !
   END SUBROUTINE rst_write


   SUBROUTINE rst_read_open
      !!---------------------------------------------------------------------- 
      !!                   ***  ROUTINE rst_read_open  ***
      !! 
      !! ** Purpose :   Open read files for restart (format fixed by jprstlib )
      !! 
      !! ** Method  :   Use a non-zero, positive value of numror to assess whether or not
      !!                the file has already been opened
      !!----------------------------------------------------------------------
      INTEGER        ::   jlibalt = jprstlib
      LOGICAL        ::   llok
      CHARACTER(lc)  ::   clpath   ! full path to ocean output restart file
      !!----------------------------------------------------------------------
      !
      IF( numror <= 0 ) THEN
         IF(lwp) THEN                                             ! Contol prints
            WRITE(numout,*)
            SELECT CASE ( jprstlib )
            CASE ( jpnf90    )   ;   WRITE(numout,*) 'rst_read : read oce NetCDF restart file'
            END SELECT
            IF ( snc4set%luse )      WRITE(numout,*) 'rst_read : configured with NetCDF4 support'
            WRITE(numout,*) '~~~~~~~~'
         ENDIF

         clpath = TRIM(cn_ocerst_indir)
         IF( clpath(LEN_TRIM(clpath):) /= '/' ) clpath = TRIM(clpath) // '/'
         CALL iom_open( TRIM(clpath)//cn_ocerst_in, numror, kiolib = jlibalt )
      ENDIF
   END SUBROUTINE rst_read_open


   SUBROUTINE rst_read
      !!---------------------------------------------------------------------- 
      !!                   ***  ROUTINE rst_read  ***
      !! 
      !! ** Purpose :   Read files for restart (format fixed by jprstlib )
      !! 
      !! ** Method  :   Read in restart.nc file fields which are necessary for restart
      !!----------------------------------------------------------------------
      REAL(wp) ::   zrdt
      INTEGER  ::   jk
      !!----------------------------------------------------------------------

      CALL rst_read_open           ! open restart for reading (if not already opened)

      ! Check dynamics and tracer time-step consistency and force Euler restart if changed
      IF( iom_varid( numror, 'rdt', ldstop = .FALSE. ) > 0 )   THEN
         CALL iom_get( numror, 'rdt', zrdt )
         IF( zrdt /= rdt )   neuler = 0
      ENDIF

      ! Diurnal DSST 
      IF( ln_diurnal ) CALL iom_get( numror, jpdom_autoglo, 'Dsst' , x_dsst  ) 
      IF ( ln_diurnal_only ) THEN 
         IF(lwp) WRITE( numout, * ) &
         &   "rst_read:- ln_diurnal_only set, setting rhop=rau0" 
         rhop = rau0
         CALL iom_get( numror, jpdom_autoglo, 'tn'     , tsn(:,:,1,jp_tem) ) 
         RETURN 
      ENDIF  
      
      IF( iom_varid( numror, 'ub', ldstop = .FALSE. ) > 0 ) THEN
         CALL iom_get( numror, jpdom_autoglo, 'ub'     , ub      )   ! before fields
         CALL iom_get( numror, jpdom_autoglo, 'vb'     , vb      )
         CALL iom_get( numror, jpdom_autoglo, 'tb'     , tsb(:,:,:,jp_tem) )
         CALL iom_get( numror, jpdom_autoglo, 'sb'     , tsb(:,:,:,jp_sal) )
         CALL iom_get( numror, jpdom_autoglo, 'sshb'   , sshb    )
      ELSE
         neuler = 0
      ENDIF
      !
      CALL iom_get( numror, jpdom_autoglo, 'un'     , un      )   ! now    fields
      CALL iom_get( numror, jpdom_autoglo, 'vn'     , vn      )
      CALL iom_get( numror, jpdom_autoglo, 'tn'     , tsn(:,:,:,jp_tem) )
      CALL iom_get( numror, jpdom_autoglo, 'sn'     , tsn(:,:,:,jp_sal) )
      CALL iom_get( numror, jpdom_autoglo, 'sshn'   , sshn    )
      IF( iom_varid( numror, 'rhop', ldstop = .FALSE. ) > 0 ) THEN
         CALL iom_get( numror, jpdom_autoglo, 'rhop'   , rhop    )   ! now    potential density
      ELSE
         CALL eos( tsn, rhd, rhop, gdept_n(:,:,:) )   
      ENDIF
      !
      IF( neuler == 0 ) THEN                                  ! Euler restart (neuler=0)
         tsb  (:,:,:,:) = tsn  (:,:,:,:)                             ! all before fields set to now values
         ub   (:,:,:)   = un   (:,:,:)
         vb   (:,:,:)   = vn   (:,:,:)
         sshb (:,:)     = sshn (:,:)
         !
         IF( .NOT.ln_linssh ) THEN
            DO jk = 1, jpk
               e3t_b(:,:,jk) = e3t_n(:,:,jk)
            END DO
         ENDIF
         !
      ENDIF
      !
   END SUBROUTINE rst_read

   !!=====================================================================
END MODULE restart
