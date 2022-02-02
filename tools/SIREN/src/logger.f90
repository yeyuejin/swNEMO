!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief This module manage log file.
!> @details
!> This module create log file and fill it depending of verbosity.
!>
!> verbosity could be choosen between :
!>    - trace : Most detailed information.
!>    - debug : Detailed information on the flow through the system.
!>    - info  : Interesting runtime events (startup/shutdown).
!>    - warning: Use of deprecated APIs, poor use of API, 'almost' errors, 
!> other runtime situations that are undesirable or unexpected, 
!> but not necessarily "wrong". 
!>    - error : Other runtime errors or unexpected conditions.
!>    - fatal : Severe errors that cause premature termination.
!>    - none  : to not create and write any information in logger file.<br />
!>       @warn in this case only FATAL ERROR will be detected.<br />
!>
!> @note default verbosity is warning
!>
!> If total number of error exceeded maximum number 
!> authorized, program stop.
!>
!> to open/create logger file:<br/>
!> @code
!>    CALL logger_open(cd_file, [cd_verbosity,] [id_maxerror,] [id_loggerid])
!> @endcode
!> - cd_file is logger file name
!> - cd_verbosity is verbosity to be used [optional, default 'warning']
!> - id_loggerid is file id [optional, use only to flush]
!> - id_maxerror is the maximum number of error authorized before program stop [optional, default 5]
!>
!> to close logger file:<br/>
!> @code
!> CALL logger_close()
!> @endcode
!>
!> to clean logger file:<br/>
!> @code
!> CALL logger_clean()
!> @endcode
!>
!> to write header in logger file:<br/>
!> @code
!> CALL logger_header()
!> @endcode
!>
!> to write footer in logger file:<br/>
!> @code
!> CALL logger_footer()
!> @endcode
!>
!> to flushing output:<br/>
!> @code
!> CALL logger_flush()
!> @endcode
!>
!> to write TRACE message in logger file:<br/>
!> @code
!> CALL logger_trace(cd_msg [,ld_flush])
!> @endcode
!>    - cd_msg is TRACE message
!>    - ld_flush to flush output [optional]
!>
!> to write DEBUG message in logger file:<br/>
!> @code
!> CALL logger_debug(cd_msg [,ld_flush])
!> @endcode
!>    - cd_msg is DEBUG message
!>    - ld_flush to flush output [optional]
!>
!> to write INFO message in logger file:<br/>
!> @code
!> CALL logger_info(cd_msg [,ld_flush])
!> @endcode
!>    - cd_msg is INFO message
!>    - ld_flush to flush output [optional]
!>
!> to write WARNING message in logger file:<br/>
!> @code
!> CALL logger_warn(cd_msg [,ld_flush])
!> @endcode
!>    - cd_msg is WARNING message
!>    - ld_flush to flush output [optional]
!>
!> to write ERROR message in logger file:<br/>
!> @code
!> CALL logger_error(cd_msg [,ld_flush])
!> @endcode
!>    - cd_msg is ERROR message
!>    - ld_flush to flush output [optional]
!>
!> to write FATAL message in logger file:<br/>
!> @code
!> CALL logger_fatal(cd_msg)
!> @endcode
!>    - cd_msg is FATAL message
!>
!> Examples :<br />
!> @code
!>   CALL logger_open('loggerfile.txt','info')
!>
!>   CALL logger_header()
!>   CALL logger_debug('une info de debug')
!>   CALL logger_info('une info')
!>   CALL logger_warn('un warning')
!>   CALL logger_error('une erreur')
!>   CALL logger_footer()
!>   CALL logger_close()
!>   CALL logger_clean()
!> @endcode
!>
!> @code
!>   CALL logger_open('loggerfile.txt')
!>
!>   CALL logger_header()
!>   CALL logger_debug('une info de debug')
!>   CALL logger_info('une info')
!>   CALL logger_warn('un warning')
!>   CALL logger_error('une erreur')
!>   CALL logger_footer()
!>   CALL logger_close()
!>   CALL logger_clean()
!> @endcode
!>
!> @author
!> J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date February, 2015
!> - check verbosity validity
!> - add 'none' verbosity level to not used logger file
!> @date January, 2016
!> - add logger_clean subroutine
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE logger

   USE kind                            ! F90 kind parameter
   USE fct                             ! basic useful function
   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! type and variable
   PRIVATE :: TLOGGER            !< logger structure

   PRIVATE :: tm_logger          !< logger structure
   PRIVATE :: im_nverbosity      !< number of log level
   PRIVATE :: cm_verbosity       !< verbosity array

   ! function and subroutine
   PUBLIC :: logger_open        !< create a log file with given verbosity
   PUBLIC :: logger_close       !< close log file
   PUBLIC :: logger_clean       !< clean log structure
   PUBLIC :: logger_header      !< write header on log file
   PUBLIC :: logger_footer      !< write footer on log file
   PUBLIC :: logger_flush       !< flushing output
   PUBLIC :: logger_trace       !< write trace    message in log file
   PUBLIC :: logger_debug       !< write debug    message in log file 
   PUBLIC :: logger_info        !< write info     message in log file
   PUBLIC :: logger_warn        !< write warning  message in log file
   PUBLIC :: logger_error       !< write error    message in log file
   PUBLIC :: logger_fatal       !< write fatal    message in log file, and stop

   PRIVATE :: logger__write     ! cut message to get maximum of 80 character by line in log file
   PRIVATE :: logger__check_verb! check verbosity validity

   TYPE TLOGGER   !< logger structure
      INTEGER(i4)       :: i_id = 0                 !< log file id
      LOGICAL           :: l_use=.TRUE.             !< use logger or not
      CHARACTER(LEN=lc) :: c_name                   !< log file name
      CHARACTER(LEN=lc) :: c_verbosity = "warning"  !< verbosity choose
      CHARACTER(LEN=lc) :: c_verb = ""              !< array of "verbosities" to used 
      INTEGER(i4)       :: i_nerror   = 0           !< number of error
      INTEGER(i4)       :: i_nfatal   = 0           !< number of fatal error
      INTEGER(i4)       :: i_maxerror = 5           !< maximum number of error before stoping program
   END TYPE TLOGGER   

   !  module variable
   INTEGER(i4), PARAMETER :: im_nverbosity=7     !< number of log level
   CHARACTER(len=*), DIMENSION(im_nverbosity), PARAMETER :: cm_verbosity= & !< verbosity array 
   &               (/ 'trace   ',&
   &                  'debug   ',&
   &                  'info    ',& 
   &                  'warning ',&
   &                  'error   ',&
   &                  'fatal   ',&
   &                  'none    '/)

   TYPE(TLOGGER), SAVE :: tm_logger      !< logger structure
                                                 
CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE logger_open(cd_file, cd_verbosity, id_maxerror, id_logid)
   !-------------------------------------------------------------------
   !> @brief This subroutine create a log file with default verbosity
   !> ('warning').
   !> @details
   !> Optionally verbosity could be change to 
   !> ('trace','debug','info',warning','error','fatal').<br/>
   !> Optionally maximum number of error allowed could be change.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_file      log file name
   !> @param[in] cd_verbosity log file verbosity
   !> @param[in] id_maxerror  maximum number of error
   !> @param[in] id_logid     log file id (use to flush)
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      CHARACTER(len=*), INTENT(IN) :: cd_file                ! log file name
      CHARACTER(len=*), INTENT(IN), OPTIONAL :: cd_verbosity ! log file verbosity
      INTEGER(i4),      INTENT(IN), OPTIONAL :: id_maxerror  ! log max error
      INTEGER(i4),      INTENT(IN), OPTIONAL :: id_logid     ! log file id

      ! local variable
      INTEGER(i4) :: il_status

      LOGICAL     :: ll_valid

      ! loop
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! if present, change verbosity value
      IF( PRESENT(cd_verbosity) )THEN
         ll_valid=logger__check_verb(TRIM(ADJUSTL(cd_verbosity)))
         IF( ll_valid )THEN
            tm_logger%c_verbosity=TRIM(ADJUSTL(cd_verbosity))
         ENDIF
      ENDIF

      IF( TRIM(tm_logger%c_verbosity) == 'none' ) tm_logger%l_use=.FALSE.
      
      IF( tm_logger%l_use )THEN

         ! get id if not already define
         IF( PRESENT(id_logid) )THEN
            tm_logger%i_id=id_logid
         ELSE
            tm_logger%i_id=fct_getunit()
         ENDIF

         ! open log file
         OPEN( tm_logger%i_id, &
         &     STATUS="unknown",    &
         &     FILE=TRIM(ADJUSTL(cd_file)),  &
         &     ACTION="write",      &
         &     POSITION="append",   &
         &     IOSTAT=il_status)
         CALL fct_err(il_status)

         ! keep filename
         tm_logger%c_name=TRIM(ADJUSTL(cd_file))

         ! compute "tab" of verbosity to be used
         IF( TRIM(ADJUSTL(tm_logger%c_verb)) == "" )THEN
            DO ji=im_nverbosity,1,-1
               tm_logger%c_verb = &
               &  TRIM(tm_logger%c_verb)//" "//TRIM(ADJUSTL(cm_verbosity(ji)))
               IF( TRIM(tm_logger%c_verbosity) == TRIM(cm_verbosity(ji)) )THEN
                  EXIT
               ENDIF
            ENDDO
         ENDIF

         IF( PRESENT(id_maxerror) )THEN
            tm_logger%i_maxerror=id_maxerror
         ENDIF

      ENDIF

   END SUBROUTINE logger_open
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE logger_close()
   !-------------------------------------------------------------------
   !> @brief This subroutine close a log file.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! local variable
      INTEGER(i4)   :: il_status
      !----------------------------------------------------------------

      IF( tm_logger%l_use )THEN
         IF( tm_logger%i_id /= 0 )THEN
            !tm_logger%i_id = 0
            CLOSE( tm_logger%i_id, &
            &      IOSTAT=il_status)      
            CALL fct_err(il_status)
         ELSE
             CALL logger_open('logger.log')
             CALL logger_header()
             CALL logger_fatal('you must have create logger to use logger_close')
         ENDIF
      ENDIF

   END SUBROUTINE logger_close
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE logger_clean()
   !-------------------------------------------------------------------
   !> @brief This subroutine clean a log structure.
   !>
   !> @author J.Paul
   !> @date January, 2016 - Initial Version
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! local variable
      TYPE(TLOGGER) :: tl_logger
      !----------------------------------------------------------------

      tm_logger = tl_logger

   END SUBROUTINE logger_clean
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   !-------------------------------------------------------------------
   !> @brief This subroutine flushing output into log file.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !-------------------------------------------------------------------
   SUBROUTINE logger_flush()

      IMPLICIT NONE
      
      !----------------------------------------------------------------

      IF( tm_logger%l_use )THEN
         IF( tm_logger%i_id /= 0 )THEN
            CALL logger_close()
            CALL logger_open( tm_logger%c_name, tm_logger%c_verbosity, &
            &                 tm_logger%i_maxerror, tm_logger%i_id )     
         ELSE
             CALL logger_open('logger.log')
             CALL logger_header()
             CALL logger_fatal('you must have create logger to use logger_flush')
         ENDIF
      ENDIF

   END SUBROUTINE logger_flush
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   RECURSIVE SUBROUTINE logger_header()
   !-------------------------------------------------------------------
   !> @brief This subroutine write header on log file.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! local variable
      INTEGER(i4)       :: il_status
      !----------------------------------------------------------------

      IF( tm_logger%l_use )THEN
         IF( tm_logger%i_id /= 0 )THEN
            WRITE( tm_logger%i_id,    &
               &   FMT='(4(a/))',     &
               &   IOSTAT=il_status ) &
               &   "--------------------------------------------------",&
               &   "INIT     : verbosity "//TRIM(tm_logger%c_verbosity),&
               &   "INIT     : max error "//TRIM(fct_str(tm_logger%i_maxerror)), &
               &   "--------------------------------------------------"
            CALL fct_err(il_status)
         ELSE
             CALL logger_open('logger.log')
             CALL logger_header()
             CALL logger_fatal('you must have create logger to use logger_header')
         ENDIF
      ENDIF

   END SUBROUTINE logger_header
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE logger_footer()
   !-------------------------------------------------------------------
   !> @brief This subroutine write footer on log file.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! local variable
      INTEGER(i4)       :: il_status
      !----------------------------------------------------------------

      IF( tm_logger%l_use )THEN
         IF( tm_logger%i_id /= 0 )THEN
            WRITE( tm_logger%i_id,    &
               &   FMT='(4(/a))',     &
               &   IOSTAT=il_status ) &
               &   "--------------------------------------------------",&
               &   "END      : log ended ",              &
               &   "END      : "//TRIM(fct_str(tm_logger%i_nerror))//   &
               &   " ERROR detected ",                                  &
               &   "END      : "//TRIM(fct_str(tm_logger%i_nfatal))//   &
               &   " FATAL detected ",                                  &
               &   "--------------------------------------------------"
            CALL fct_err(il_status)
         ELSE
             CALL logger_open('logger.log')
             CALL logger_header()
             CALL logger_fatal('you must have create logger to use logger_footer')
         ENDIF
      ENDIF

   END SUBROUTINE logger_footer
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE logger_trace(cd_msg, ld_flush)
   !-------------------------------------------------------------------
   !> @brief This subroutine write trace message on log file.
   !> @details
   !> Optionally you could flush output.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_msg    message to write
   !> @param[in] ld_flush  flushing ouput
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN)  :: cd_msg
      LOGICAL,          INTENT(IN), OPTIONAL :: ld_flush
      !----------------------------------------------------------------

      IF( tm_logger%l_use )THEN
         IF( tm_logger%i_id /= 0 )THEN
            IF( INDEX(TRIM(tm_logger%c_verb),'trace')/=0 )THEN

               CALL logger__write("TRACE   :",cd_msg)

               IF( PRESENT(ld_flush) )THEN
                  IF( ld_flush )THEN
                     CALL logger_flush()
                  ENDIF
               ENDIF      
            ENDIF
         ELSE
             CALL logger_open('logger.log')
             CALL logger_header()
             CALL logger_fatal('you must have create logger to use logger_trace')
         ENDIF
      ENDIF

   END SUBROUTINE logger_trace
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE logger_debug(cd_msg, ld_flush)
   !-------------------------------------------------------------------
   !> @brief This subroutine write debug message on log file.
   !> @details
   !> Optionally you could flush output.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_msg    message to write
   !> @param[in] ld_flush  flushing ouput
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN)  :: cd_msg
      LOGICAL,          INTENT(IN), OPTIONAL :: ld_flush
      !----------------------------------------------------------------

      IF( tm_logger%l_use )THEN
         IF( tm_logger%i_id /= 0 )THEN
            IF( INDEX(TRIM(tm_logger%c_verb),'debug')/=0 )THEN

               CALL logger__write("DEBUG   :",cd_msg)

               IF( PRESENT(ld_flush) )THEN
                  IF( ld_flush )THEN
                     CALL logger_flush()
                  ENDIF
               ENDIF      
            ENDIF
         ELSE
             CALL logger_open('logger.log')
             CALL logger_header()
             CALL logger_fatal('you must have create logger to use logger_debug')
         ENDIF
      ENDIF

   END SUBROUTINE logger_debug
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE logger_info(cd_msg, ld_flush)
   !-------------------------------------------------------------------
   !> @brief This subroutine write info message on log file.
   !> @details
   !> Optionally you could flush output.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_msg    message to write
   !> @param[in] ld_flush  flushing ouput
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN)  :: cd_msg
      LOGICAL,          INTENT(IN), OPTIONAL :: ld_flush
      !----------------------------------------------------------------

      IF( tm_logger%l_use )THEN
         IF( tm_logger%i_id /= 0 )THEN
            IF( INDEX(TRIM(tm_logger%c_verb),'info')/=0 )THEN

               CALL logger__write("INFO    :",cd_msg)

               IF( PRESENT(ld_flush) )THEN
                  IF( ld_flush )THEN
                     CALL logger_flush()
                  ENDIF
               ENDIF      
            ENDIF
         ELSE
             CALL logger_open('logger.log')
             CALL logger_header()
             CALL logger_fatal('you must have create logger to use logger_info')
         ENDIF
      ENDIF
   END SUBROUTINE logger_info
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE logger_warn(cd_msg, ld_flush)
   !-------------------------------------------------------------------
   !> @brief This subroutine write warning message on log file.
   !> @details
   !> Optionally you could flush output.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_msg    message to write
   !> @param[in] ld_flush  flushing ouput
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN)  :: cd_msg
      LOGICAL,          INTENT(IN), OPTIONAL :: ld_flush
      !----------------------------------------------------------------

      IF( tm_logger%l_use )THEN
         IF( tm_logger%i_id /= 0 )THEN
            IF( INDEX(TRIM(tm_logger%c_verb),'warn')/=0 )THEN

               CALL logger__write("WARNING :",cd_msg)

               IF( PRESENT(ld_flush) )THEN
                  IF( ld_flush )THEN
                     CALL logger_flush()
                  ENDIF
               ENDIF      
            ENDIF
         ELSE
             CALL logger_open('logger.log')
             CALL logger_header()
             CALL logger_fatal('you must have create logger to use logger_warn')
         ENDIF
      ENDIF

   END SUBROUTINE logger_warn
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE logger_error(cd_msg, ld_flush)
   !-------------------------------------------------------------------
   !> @brief This subroutine write error message on log file.
   !> @details
   !> Optionally you could flush output.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_msg    message to write
   !> @param[in] ld_flush  flushing ouput
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN)  :: cd_msg
      LOGICAL,          INTENT(IN), OPTIONAL :: ld_flush

      ! local variable
      CHARACTER(LEN=lc) :: cl_nerror
      !----------------------------------------------------------------

      IF( tm_logger%l_use )THEN
         IF( tm_logger%i_id /= 0 )THEN
            ! increment the error number
            tm_logger%i_nerror=tm_logger%i_nerror+1

            IF( INDEX(TRIM(tm_logger%c_verb),'error')/=0 )THEN

               CALL logger__write("ERROR   :",cd_msg)

               IF( PRESENT(ld_flush) )THEN
                  IF( ld_flush )THEN
                     CALL logger_flush()
                  ENDIF
               ENDIF      
            ENDIF

            IF( tm_logger%i_nerror >= tm_logger%i_maxerror )THEN
               WRITE(cl_nerror,*) tm_logger%i_maxerror
               CALL logger_fatal(&
               &  'Error count reached limit of '//TRIM(ADJUSTL(cl_nerror)) )
            ENDIF
         ELSE
             CALL logger_open('logger.log')
             CALL logger_header()
             CALL logger_fatal('you must have create logger to use logger_error')
         ENDIF
      ENDIF

   END SUBROUTINE logger_error
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   RECURSIVE SUBROUTINE logger_fatal(cd_msg)
   !-------------------------------------------------------------------
   !> @brief This subroutine write fatal error message on log file, 
   !> close log file and stop process.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2015
   !> - stop program for FATAL ERROR if verbosity is none
   !>
   !> @param[in] cd_msg message to write
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),           INTENT(IN) :: cd_msg
      !----------------------------------------------------------------

      IF( tm_logger%l_use )THEN
         IF( tm_logger%i_id /= 0 )THEN
            IF( INDEX(TRIM(tm_logger%c_verb),'fatal')/=0 )THEN
               ! increment the error number
               tm_logger%i_nfatal=tm_logger%i_nfatal+1

               CALL logger__write("FATAL   :",cd_msg)

               CALL logger_footer()
               CALL logger_close()

               WRITE(*,*) 'FATAL ERROR, see ',TRIM(tm_logger%c_name)
               STOP
            ENDIF
         ELSE
             CALL logger_open('logger.log')
             CALL logger_header()
             CALL logger_fatal('you must have create logger to use logger_fatal')
         ENDIF
      ELSE
         PRINT *,"FATAL ERROR :"//TRIM(cd_msg)
         STOP
      ENDIF

   END SUBROUTINE logger_fatal
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE logger__write(cd_verb, cd_msg)
   !-------------------------------------------------------------------
   !> @brief This subroutine cut message to get maximum of 80 character 
   !> by line in log file.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_verb   verbosity of the message to write
   !> @param[in] cd_msg    message to write
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),           INTENT(IN) :: cd_verb
      CHARACTER(LEN=*),           INTENT(IN) :: cd_msg

      ! local variable
      INTEGER(i4)       :: il_status
      INTEGER(i4)       :: il_verb
      INTEGER(i4)       :: il_msg
      CHARACTER(LEN=lc) :: cl_verb
      CHARACTER(LEN=lc) :: cl_msg
      CHARACTER(LEN=lc) :: cl_tmp
      !----------------------------------------------------------------

      cl_verb=TRIM(ADJUSTL(cd_verb))
      cl_msg=TRIM(ADJUSTL(cd_msg))

      il_verb=LEN_TRIM(cl_verb)
      il_msg=LEN_TRIM(cl_msg)
      DO WHILE( il_verb + il_msg > 78 )
         cl_tmp=TRIM(cl_verb)//' '//TRIM(cl_msg(1:78-il_verb))

         WRITE( tm_logger%i_id,  &
         &      FMT=*,           &
         &      IOSTAT=il_status &
         &      ) TRIM(cl_tmp)
         CALL fct_err(il_status)


         cl_msg=cl_msg(78-il_verb+1:il_msg)
         cl_verb="        :"

         il_msg=LEN_TRIM(cl_msg)

      ENDDO

      cl_tmp=TRIM(cl_verb)//' '//TRIM(cl_msg)
      WRITE( tm_logger%i_id,  &
      &      FMT=*,           &
      &      IOSTAT=il_status &
      &      ) TRIM(cl_tmp)
      CALL fct_err(il_status)

   END SUBROUTINE logger__write
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION logger__check_verb(cd_verb) &
         & RESULT (lf_show)
   !-------------------------------------------------------------------
   !> @brief This function check validity of verbosity.
   !>
   !> @author J.Paul
   !> @date February, 2015 - Initial Version
   !>
   !> @param[in] cd_verb   verbosity of the message to write
   !> @return verbosity is valid or not
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN) :: cd_verb

      !function
      LOGICAL                      :: lf_show

      ! local variable
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      lf_show=.FALSE.

      DO ji=1,im_nverbosity
         IF( TRIM(cd_verb) == TRIM(cm_verbosity(ji)) )THEN
            lf_show=.TRUE.
            EXIT
         ENDIF
      ENDDO

      IF( .NOT. lf_show )THEN
         CALL logger_open('logger.log')
         CALL logger_header()
         CALL logger_fatal('LOGGER : invalid verbosity, check namelist.'//&
         &                 ' default one will be used.')
         CALL logger_footer()
      ENDIF

   END FUNCTION logger__check_verb
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE logger

