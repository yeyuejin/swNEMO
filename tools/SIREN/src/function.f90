!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief
!> This module groups some basic useful function.
!>
!> @details
!>  to get free I/O unit number:<br/>
!> @code
!>  il_id=fct_getunit()
!> @endcode
!>
!>  to convert "numeric" to string character:<br/>
!> @code
!>  cl_string=fct_str(numeric)
!> @endcode
!>  - "numeric" could be integer, real, or logical
!>
!>  to concatenate "numeric" to a string character:<br/>
!> @code
!>  cl_str=cd_char//num
!> @endcode
!>  - cd_char is the string character
!>  - num is the numeric value (integer, real or logical)  
!>
!>  to concatenate all the element of a character array:<br/> 
!> @code
!>  cl_string=fct_concat(cd_arr [,cd_sep])
!> @endcode
!>  - cd_arr is a 1D array of character
!>  - cd_sep is a separator character to add between each element of cd_arr 
!> [optional]
!>
!>  to convert character from lower to upper case:<br/>
!> @code
!>  cl_upper=fct_upper(cd_var)
!> @endcode
!>
!>  to convert character from upper to lower case:<br/>
!> @code
!>  cl_lower=fct_lower(cd_var)
!> @endcode
!>
!>  to check if character is numeric
!> @code
!>  ll_is_num=fct_is_num(cd_var)
!> @endcode
!>
!>  to check if character is real
!> @code
!>  ll_is_real=fct_is_real(cd_var)
!> @endcode
!>
!>  to split string into substring and return one of the element:<br/>  
!> @code
!>  cl_str=fct_split(cd_string ,id_ind [,cd_sep])
!> @endcode
!>  - cd_string is a string of character
!>  - id_ind is the indice of the lement to extract
!>  - cd_sep is the separator use to split cd_string (default '|')
!>
!>  to get basename (name without path):<br/>
!> @code
!>  cl_str=fct_basename(cd_string [,cd_sep])
!> @endcode
!>  - cd_string is the string filename
!>  - cd_sep is the separator ti be used (default '/')
!>
!>  to get dirname (path of the filename):<br/>
!> @code
!>  cl_str=fct_dirname(cd_string [,cd_sep])
!> @endcode
!>  - cd_string is the string filename
!>  - cd_sep is the separator ti be used (default '/')
!>  
!> to create a pause statement:<br/>
!> @code
!> CALL fct_pause(cd_msg)
!> @endcode
!>    - cd_msg : message to be added [optional]
!>
!> to handle frotran error:<br/>
!> @code
!> CALL fct_err(id_status)
!> @endcode
!> 
!> to show help message:<br/>
!> @code
!> CALL fct_help(cd_filename, cd_err)
!> @endcode
!>    - cd_filename : file name
!>    - cd_err      : error message [optional]
!>
!> to show Siren's version:<br/>
!> @code
!> CALL fct_version(cd_filename)
!> @endcode
!>    - cd_filename : file name
!>
!> 
!> @author
!> J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date September, 2014 
!> - add header
!> @date October, 2019
!> - add help and version function
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE fct

   USE global                          ! global variable
   USE kind                            ! F90 kind parameter

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! function and subroutine
   PUBLIC :: fct_getunit  !< returns free unit number
   PUBLIC :: fct_str      !< convert numeric to string character
   PUBLIC :: OPERATOR(//) !< concatenate operator
   PUBLIC :: fct_concat   !< concatenate all the element of a character array
   PUBLIC :: fct_upper    !< convert character from lower to upper case
   PUBLIC :: fct_lower    !< convert character from upper to lower case
   PUBLIC :: fct_is_num   !< check if character is numeric
   PUBLIC :: fct_is_real  !< check if character is real
   PUBLIC :: fct_split    !< split string into substring
   PUBLIC :: fct_basename !< return basename (name without path)
   PUBLIC :: fct_dirname  !< return dirname (path without filename)
   PUBLIC :: fct_pause    !< pause statement
   PUBLIC :: fct_err      !< handle fortran error status
   PUBLIC :: fct_help     !< show help message
   PUBLIC :: fct_version  !< show Siren's version

   PRIVATE :: fct__i1_str ! convert integer(1) to string character
   PRIVATE :: fct__i2_str ! convert integer(2) to string character
   PRIVATE :: fct__i4_str ! convert integer(4) to string character
   PRIVATE :: fct__i8_str ! convert integer(8) to string character
   PRIVATE :: fct__r4_str ! convert real(4) to string character
   PRIVATE :: fct__r8_str ! convert real(8) to string character
   PRIVATE :: fct__l_str  ! convert logical to string character
   PRIVATE :: fct__i1_cat ! concatenate integer(1) to string character
   PRIVATE :: fct__i2_cat ! concatenate integer(2) to string character
   PRIVATE :: fct__i4_cat ! concatenate integer(4) to string character
   PRIVATE :: fct__i8_cat ! concatenate integer(8) to string character
   PRIVATE :: fct__r4_cat ! concatenate real(4) to string character
   PRIVATE :: fct__r8_cat ! concatenate real(8) to string character
   PRIVATE :: fct__l_cat  ! concatenate logical to string character
   PRIVATE :: fct__split_space ! split string into substring using space as separator

   INTERFACE fct_str
      MODULE PROCEDURE fct__i1_str ! convert integer(1) to string character
      MODULE PROCEDURE fct__i2_str ! convert integer(2) to string character
      MODULE PROCEDURE fct__i4_str ! convert integer(4) to string character
      MODULE PROCEDURE fct__i8_str ! convert integer(8) to string character
      MODULE PROCEDURE fct__r4_str ! convert real(4) to string character
      MODULE PROCEDURE fct__r8_str ! convert real(8) to string character
      MODULE PROCEDURE fct__l_str  ! convert logical to string character
   END INTERFACE fct_str

   INTERFACE OPERATOR(//)
      MODULE PROCEDURE fct__i1_cat ! concatenate integer(1) to string character
      MODULE PROCEDURE fct__i2_cat ! concatenate integer(2) to string character
      MODULE PROCEDURE fct__i4_cat ! concatenate integer(4) to string character
      MODULE PROCEDURE fct__i8_cat ! concatenate integer(8) to string character
      MODULE PROCEDURE fct__r4_cat ! concatenate real(4) to string character
      MODULE PROCEDURE fct__r8_cat ! concatenate real(8) to string character
      MODULE PROCEDURE fct__l_cat  ! concatenate logical to string character
   END INTERFACE

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct__i1_cat(cd_char, bd_val) &
         & RESULT(cf_str)
   !------------------------------------------------------------------- 
   !> @brief This function concatenate character and integer(1) (as character). 
   !>
   !> @author J.Paul 
   !> @date September, 2014 - Initial Version 
   !>
   !> @param[in] cd_char   string character
   !> @param[in] bd_val    integer(1) variable value
   !> @return string character 
   !------------------------------------------------------------------- 
 
      IMPLICIT NONE
 
      ! arguments
      CHARACTER(LEN=lc), INTENT(IN) :: cd_char
      INTEGER(i1),       INTENT(IN) :: bd_val

      ! function
      CHARACTER(LEN=lc)             :: cf_str

      ! local variable
      CHARACTER(LEN=lc) :: cl_val
      !---------------------------------------------------------------- 
 
      cl_val = fct_str(bd_val)
      cf_str = TRIM(cd_char)//TRIM(cl_val)

   END FUNCTION fct__i1_cat 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct__i2_cat(cd_char, sd_val) &
         & RESULT(cf_str)
   !------------------------------------------------------------------- 
   !> @brief This function concatenate character and integer(2) (as character). 
   !>
   !> @author J.Paul 
   !> @date September, 2014 - Initial Version 
   !>
   !> @param[in] cd_char   string character
   !> @param[in] sd_val    integer(2) variable value
   !> @return string character 
   !------------------------------------------------------------------- 
 
      IMPLICIT NONE
 
      ! arguments
      CHARACTER(LEN=lc), INTENT(IN) :: cd_char
      INTEGER(i2),       INTENT(IN) :: sd_val

      ! function
      CHARACTER(LEN=lc)             :: cf_str

      ! local variable
      CHARACTER(LEN=lc) :: cl_val
      !---------------------------------------------------------------- 
 
      cl_val = fct_str(sd_val)
      cf_str = TRIM(cd_char)//TRIM(cl_val)

   END FUNCTION fct__i2_cat 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct__i4_cat(cd_char, id_val) &
         & RESULT(cf_str)
   !------------------------------------------------------------------- 
   !> @brief This function concatenate character and integer(4) (as character). 
   !>
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   !>
   !> @param[in] cd_char   string character
   !> @param[in] id_val    integer(4) variable value
   !> @return string character 
   !------------------------------------------------------------------- 
 
      IMPLICIT NONE
 
      ! arguments
      CHARACTER(LEN=lc), INTENT(IN) :: cd_char
      INTEGER(i4),       INTENT(IN) :: id_val

      ! function
      CHARACTER(LEN=lc)             :: cf_str

      ! local variable
      CHARACTER(LEN=lc) :: cl_val
      !---------------------------------------------------------------- 
 
      cl_val = fct_str(id_val)
      cf_str = TRIM(cd_char)//TRIM(cl_val)

   END FUNCTION fct__i4_cat 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct__i8_cat(cd_char, kd_val) &
         & RESULT(cf_str)
   !------------------------------------------------------------------- 
   !> @brief This function concatenate character and integer(8) (as character). 
   !>
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   !>
   !> @param[in] cd_char   string character
   !> @param[in] kd_val    integer(8) variable value
   !> @return string character 
   !------------------------------------------------------------------- 
 
      IMPLICIT NONE
 
      ! arguments
      CHARACTER(LEN=lc), INTENT(IN) :: cd_char
      INTEGER(i8),       INTENT(IN) :: kd_val

      ! function
      CHARACTER(LEN=lc)             :: cf_str

      ! local variable
      CHARACTER(LEN=lc) :: cl_val
      !---------------------------------------------------------------- 
 
      cl_val = fct_str(kd_val)
      cf_str = TRIM(cd_char)//TRIM(cl_val)

   END FUNCTION fct__i8_cat 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct__r4_cat(cd_char, rd_val) &
         & RESULT(cf_str)
   !------------------------------------------------------------------- 
   !> @brief This function concatenate character and real(4) (as character). 
   !>
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   !>
   !> @param[in] cd_char   string character
   !> @param[in] rd_val    real(4) variable value
   !> @return string character 
   !------------------------------------------------------------------- 
 
      IMPLICIT NONE
 
      ! arguments
      CHARACTER(LEN=lc), INTENT(IN) :: cd_char
      REAL(sp),          INTENT(IN) :: rd_val

      ! function
      CHARACTER(LEN=lc)             :: cf_str

      ! local variable
      CHARACTER(LEN=lc) :: cl_val
      !---------------------------------------------------------------- 
 
      cl_val = fct_str(rd_val)
      cf_str = TRIM(cd_char)//TRIM(cl_val)

   END FUNCTION fct__r4_cat 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct__r8_cat(cd_char, dd_val) &
         & RESULT(cf_str)
   !------------------------------------------------------------------- 
   !> @brief This function concatenate character and real(8) (as character). 
   !>
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   !>
   !> @param[in] cd_char   string character
   !> @param[in] dd_val    real(8) variable value
   !> @return string character 
   !------------------------------------------------------------------- 
 
      IMPLICIT NONE
 
      ! arguments
      CHARACTER(LEN=lc), INTENT(IN) :: cd_char
      REAL(dp),          INTENT(IN) :: dd_val

      ! function
      CHARACTER(LEN=lc)             :: cf_str

      ! local variable
      CHARACTER(LEN=lc) :: cl_val
      !---------------------------------------------------------------- 
 
      cl_val = fct_str(dd_val)
      cf_str = TRIM(cd_char)//TRIM(cl_val)

   END FUNCTION fct__r8_cat 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct__l_cat(cd_char, ld_val) &
         & RESULT(cf_str)
   !------------------------------------------------------------------- 
   !> @brief This function concatenate character and logical (as character). 
   !>
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   !>
   !> @param[in] cd_char   string character
   !> @param[in] ld_val    logical variable value
   !> @return string character 
   !------------------------------------------------------------------- 

      IMPLICIT NONE
 
      ! arguments
      CHARACTER(LEN=lc), INTENT(IN) :: cd_char
      LOGICAL,           INTENT(IN) :: ld_val

      ! function
      CHARACTER(LEN=lc)             :: cf_str

      ! local variable
      CHARACTER(LEN=lc) :: cl_val
      !---------------------------------------------------------------- 
 
      cl_val = fct_str(ld_val)
      cf_str = TRIM(cd_char)//TRIM(cl_val)

   END FUNCTION fct__l_cat 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION fct_getunit() &
         & RESULT(if_unit)
   !------------------------------------------------------------------- 
   !> @brief This function returns the next available I/O unit number. 
   !>
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   !>
   !> @return file id 
   !------------------------------------------------------------------- 

      IMPLICIT NONE
 
      ! function
      INTEGER(i4) :: if_unit

      ! local variable 
      LOGICAL ::  ll_opened 
      !---------------------------------------------------------------- 
      ! initialise 
      if_unit = 10 
 
      INQUIRE(UNIT=if_unit, OPENED=ll_opened) 
      DO WHILE( ll_opened ) 
         if_unit = if_unit + 1 
         INQUIRE(UNIT=if_unit, OPENED=ll_opened)  
      ENDDO 
 
   END FUNCTION fct_getunit 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE fct_err(id_status)
   !-------------------------------------------------------------------
   !> @brief This subroutine handle Fortran status.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] id_status
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      INTEGER(i4),       INTENT(IN) :: id_status
      !----------------------------------------------------------------

      IF( id_status /= 0 )THEN
         !CALL ERRSNS() ! not F95 standard
         PRINT *, "FORTRAN ERROR ", id_status
         !STOP
      ENDIF

   END SUBROUTINE fct_err
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE fct_pause(cd_msg)
   !-------------------------------------------------------------------
   !> @brief This subroutine  create a pause statement
   !>
   !> @author J.Paul
   !> @date November, 2014 - Initial Version
   !>
   !> @param[in] cd_msg optional message to be added
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL  :: cd_msg
      !----------------------------------------------------------------

      IF( PRESENT(cd_msg) )THEN
         WRITE( *, * ) 'Press Enter to continue '//TRIM(cd_msg)
      ELSE
         WRITE( *, * ) 'Press Enter to continue'
      ENDIF
      READ( *, * )

   END SUBROUTINE fct_pause
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct__l_str(ld_var) &
         & RESULT(cf_str)
   !-------------------------------------------------------------------
   !> @brief This function convert logical to string character.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] ld_var logical variable
   !> @return character of this integer variable
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      LOGICAL, INTENT(IN) :: ld_var

      ! function
      CHARACTER(LEN=lc)   :: cf_str

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      !----------------------------------------------------------------

      WRITE(cl_tmp,*) ld_var
      cf_str=TRIM(ADJUSTL(cl_tmp))

   END FUNCTION fct__l_str
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct__i1_str(bd_var) &
         & RESULT(cf_str)
   !-------------------------------------------------------------------
   !> @brief This function convert integer(1) to string character.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] bd_var integer(1) variable
   !> @return character of this integer variable
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      INTEGER(i1), INTENT(IN) :: bd_var

      ! function
      CHARACTER(LEN=lc)       :: cf_str

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      !----------------------------------------------------------------

      WRITE(cl_tmp,*) bd_var
      cf_str=TRIM(ADJUSTL(cl_tmp))

   END FUNCTION fct__i1_str
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct__i2_str(sd_var) &
         & RESULT(cf_str)
   !-------------------------------------------------------------------
   !> @brief This function convert integer(2) to string character.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] sd_var integer(2) variable
   !> @return character of this integer variable
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      INTEGER(i2), INTENT(IN) :: sd_var

      ! function
      CHARACTER(LEN=lc)       :: cf_str

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      !----------------------------------------------------------------

      WRITE(cl_tmp,*) sd_var
      cf_str=TRIM(ADJUSTL(cl_tmp))

   END FUNCTION fct__i2_str
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct__i4_str(id_var) &
         & RESULT(cf_str)
   !-------------------------------------------------------------------
   !> @brief This function convert integer(4) to string character.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] id_var integer(4) variable
   !> @return character of this integer variable
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      INTEGER(i4), INTENT(IN) :: id_var

      ! function
      CHARACTER(LEN=lc)       :: cf_str

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      !----------------------------------------------------------------

      WRITE(cl_tmp,*) id_var
      cf_str=TRIM(ADJUSTL(cl_tmp))

   END FUNCTION fct__i4_str
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct__i8_str(kd_var) &
         & RESULT(cf_str)
   !-------------------------------------------------------------------
   !> @brief This function convert integer(8) to string character.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] kd_var integer(8) variable
   !> @return character of this integer variable
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      INTEGER(i8), INTENT(IN) :: kd_var

      ! function
      CHARACTER(LEN=lc)       :: cf_str

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      !----------------------------------------------------------------

      WRITE(cl_tmp,*) kd_var
      cf_str=TRIM(ADJUSTL(cl_tmp))

   END FUNCTION fct__i8_str
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct__r4_str(rd_var) &
         & RESULT(cf_str)
   !-------------------------------------------------------------------
   !> @brief This function convert real(4) to string character.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] rd_var real(4) variable
   !> @return character of this real variable
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(sp), INTENT(IN) :: rd_var

      ! function
      CHARACTER(LEN=lc)    :: cf_str

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      !----------------------------------------------------------------

      WRITE(cl_tmp,*) rd_var
      cf_str=TRIM(ADJUSTL(cl_tmp))

   END FUNCTION fct__r4_str
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct__r8_str(dd_var) &
         & RESULT(cf_str)
   !-------------------------------------------------------------------
   !> @brief This function convert real(8) to string character.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_var real(8) variable
   !> @return character of this real variable
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp), INTENT(IN) :: dd_var

      ! function
      CHARACTER(LEN=lc)    :: cf_str

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      !----------------------------------------------------------------

      WRITE(cl_tmp,*) dd_var
      cf_str=TRIM(ADJUSTL(cl_tmp))

   END FUNCTION fct__r8_str
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct_concat(cd_arr,cd_sep) &
         & RESULT(cf_str)
   !-------------------------------------------------------------------
   !> @brief This function concatenate all the element of a character array 
   !> in a character string.
   !> @details
   !> optionnally a separator could be added between each element.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_arr array of character
   !> @param[in] cd_sep separator character
   !> @return character
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(*), DIMENSION(:), INTENT(IN) :: cd_arr
      CHARACTER(*),               INTENT(IN), OPTIONAL :: cd_sep

      ! function
      CHARACTER(LEN=lc)                      :: cf_str

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      CHARACTER(LEN=lc) :: cl_sep
      INTEGER(i4)       :: il_size

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      cl_sep=''
      IF(PRESENT(cd_sep)) cl_sep=cd_sep

      il_size=SIZE(cd_arr)
      cf_str=''
      cl_tmp=''
      DO ji=1,il_size

         WRITE(cl_tmp,*) TRIM(cf_str)//TRIM(ADJUSTL(cd_arr(ji)))//TRIM(cl_sep)
         cf_str=TRIM(ADJUSTL(cl_tmp))
      
      ENDDO

   END FUNCTION fct_concat
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct_lower(cd_var) &
         & RESULT(cf_str)
   !-------------------------------------------------------------------
   !> @brief This function convert string character upper case to lower case.
   !>
   !> @details
   !> The function IACHAR returns the ASCII value of the character passed 
   !> as argument. The ASCII code has the uppercase alphabet starting at 
   !> code 65, and the lower case one at code 101, therefore 
   !> IACHAR('a')- IACHAR('A') would be the difference between the uppercase
   !> and the lowercase codes.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_var character
   !> @return lower case character
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(*), INTENT(IN) :: cd_var

      ! function
      CHARACTER(LEN=lc)        :: cf_str

      ! local variable
      INTEGER(i4)                                  :: il_nletter ! number of letters in variable
      CHARACTER(LEN=lc)                            :: cl_var
      CHARACTER(LEN=lc), DIMENSION(:), ALLOCATABLE :: cl_tmp

      INTEGER(i4) :: il_icode    ! ASCII value
      INTEGER(i4) :: il_lacode   ! ASCII value of the lower case 'a'
      INTEGER(i4) :: il_uacode   ! ASCII value of the upper case 'A'
      INTEGER(i4) :: il_uzcode   ! ASCII value of the upper case 'z'

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      il_lacode=IACHAR('a')
      il_uacode=IACHAR('A')
      il_uzcode=IACHAR('Z')

      cl_var=TRIM(ADJUSTL(cd_var))
      il_nletter=LEN(TRIM(cl_var))
      ALLOCATE(cl_tmp(il_nletter))
      DO ji=1,il_nletter
         il_icode=IACHAR(cl_var(ji:ji))
         IF( il_icode >= il_uacode .AND. il_icode <= il_uzcode )THEN
            ! upper case
            cl_tmp(ji)=TRIM(CHAR(il_icode + (il_lacode - il_uacode) ))
         ELSE
            ! lower case and other character
            cl_tmp(ji)=TRIM(CHAR(il_icode))
         ENDIF
      ENDDO

      cf_str=TRIM(ADJUSTL(fct_concat(cl_tmp(:))))
      DEALLOCATE(cl_tmp)

   END FUNCTION fct_lower
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct_upper(cd_var) &
         & RESULT(cf_str)
   !-------------------------------------------------------------------
   !> @brief This function convert string character lower case to upper case.
   !>
   !> @details
   !> The function IACHAR returns the ASCII value of the character passed 
   !> as argument. The ASCII code has the uppercase alphabet starting at 
   !> code 65, and the lower case one at code 101, therefore 
   !> IACHAR('a')- IACHAR('A') would be the difference between the uppercase
   !> and the lowercase codes.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_var character
   !> @return upper case character
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(*), INTENT(IN) :: cd_var

      ! function
      CHARACTER(LEN=lc)        :: cf_str

      ! local variable
      INTEGER(i4)                                  :: il_nletter ! number of letters in cd_var
      CHARACTER(LEN=lc)                            :: cl_var
      CHARACTER(LEN=lc), DIMENSION(:), ALLOCATABLE :: cl_tmp

      INTEGER(i4) :: il_icode    ! ASCII value
      INTEGER(i4) :: il_lacode   ! ASCII value of the lower case 'a'
      INTEGER(i4) :: il_uacode   ! ASCII value of the upper case 'A'
      INTEGER(i4) :: il_lzcode   ! ASCII value of the lower case 'z'

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      il_lacode=ICHAR('a')
      il_uacode=ICHAR('A')
      il_lzcode=IACHAR('z')

      cl_var=TRIM(ADJUSTL(cd_var))
      il_nletter=LEN(TRIM(cl_var))
      ALLOCATE(cl_tmp(il_nletter))
      DO ji=1,il_nletter
         il_icode=IACHAR(cl_var(ji:ji))
         IF( il_icode >= il_lacode .AND. il_icode <= il_lzcode )THEN
            ! lower case
            cl_tmp(ji)=CHAR(il_icode - (il_lacode - il_uacode) )
         ELSE
            ! upper case and other character
            cl_tmp(ji)=CHAR(il_icode)
         ENDIF
      ENDDO

      cf_str=TRIM(ADJUSTL(fct_concat(cl_tmp(:))))
      DEALLOCATE(cl_tmp)

   END FUNCTION fct_upper
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct_is_num(cd_var) &
         & RESULT(lf_numeric)
   !-------------------------------------------------------------------
   !> @brief This function check if character is numeric.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_var character
   !> @return character is numeric
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=*), INTENT(IN) :: cd_var

      ! function
      LOGICAL                      :: lf_numeric

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      DO ji=1,LEN(TRIM(cd_var))
         IF( IACHAR(cd_var(ji:ji)) >= IACHAR('0') .AND. &
         &   IACHAR(cd_var(ji:ji)) <= IACHAR('9') )THEN
            lf_numeric=.TRUE.
         ELSE
            lf_numeric=.FALSE.
            EXIT
         ENDIF
      ENDDO

   END FUNCTION fct_is_num
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct_is_real(cd_var) &
         & RESULT(lf_real)
   !-------------------------------------------------------------------
   !> @brief This function check if character is real number.
   !>
   !> @details
   !> it permits exponantial and decimal number
   !> exemple :  1e6, 2.3
   !>
   !> @author J.Paul
   !> @date June, 2015 - Initial Version
   !> @date April, 2018
   !> - permit negative exposant 
   !> - permit sign as first character 
   !>
   !> @param[in] cd_var character
   !> @return character is real number
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=*), INTENT(IN) :: cd_var
   
      ! function
      LOGICAL                      :: lf_real

      ! local variables
      LOGICAL :: ll_exp
      LOGICAL :: ll_dec
   
      ! loop indices
      INTEGER :: ji
      !----------------------------------------------------------------
   
      ll_exp=.TRUE.
      ll_dec=.FALSE.
      DO ji=1,LEN(TRIM(cd_var))
         IF( IACHAR(cd_var(ji:ji)) >= IACHAR('0') .AND. &
         &   IACHAR(cd_var(ji:ji)) <= IACHAR('9') )THEN
   
            lf_real=.TRUE.
            ll_exp=.FALSE.
      
         ELSEIF( TRIM(fct_lower(cd_var(ji:ji)))=='e' )THEN
         
            IF( ll_exp .OR. ji== LEN(TRIM(cd_var)) )THEN
               lf_real=.FALSE.
               EXIT
            ELSE 
               ll_exp=.TRUE.
            ENDIF

         ELSEIF( TRIM(cd_var(ji:ji))=='+' )THEN
            IF( ji /= 1 )THEN
               lf_real=.FALSE.
               EXIT
            ELSE
               lf_real=.TRUE.
            ENDIF
         
         ELSEIF( TRIM(cd_var(ji:ji))=='-' )THEN
         
            IF( ji <= 1 )THEN
               IF( ji /= 1 )THEN
                  lf_real=.FALSE.
                  EXIT
               ELSE
                  lf_real=.TRUE.
               ENDIF
            ELSE ! ji > 1
               IF( TRIM(fct_lower(cd_var(ji-1:ji-1)))/='e' )THEN
                  lf_real=.FALSE.
                  EXIT
               ELSE
                  lf_real=.TRUE.
               ENDIF
            ENDIF

         ELSEIF( TRIM(cd_var(ji:ji))=='.' )THEN
   
            IF( ll_dec )THEN
               lf_real=.FALSE.
               EXIT
            ELSE
               lf_real=.TRUE.
               ll_dec=.TRUE.
            ENDIF
   
         ELSE
   
            lf_real=.FALSE.
            EXIT
   
         ENDIF
      ENDDO
   
   END FUNCTION fct_is_real
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct_split(cd_string, id_ind, cd_sep) &
         & RESULT(cf_elt)
   !-------------------------------------------------------------------
   !> @brief This function split string of character 
   !> using separator character, by default '|',
   !> and return the element on index ind.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_string string of character
   !> @param[in] id_ind    indice
   !> @param[in] cd_sep    separator character
   !> @return return the element of index id_ind
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=*), INTENT(IN) :: cd_string
      INTEGER(i4)     , INTENT(IN) :: id_ind
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: cd_sep

      ! function
      CHARACTER(LEN=lc)            :: cf_elt

      ! local variable
      CHARACTER(LEN=lc) :: cl_sep
      CHARACTER(LEN=lc) :: cl_string

      INTEGER(i4) :: il_sep
      INTEGER(i4) :: il_lsep
      
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! initialize
      cf_elt=''
      cl_string=ADJUSTL(cd_string)

      ! get separator
      cl_sep='|'
      IF( PRESENT(cd_sep) )THEN
         IF( cd_sep==' ')THEN
            cl_sep=' '
         ELSE
            cl_sep=TRIM(ADJUSTL(cd_sep))
         ENDIF
      ENDIF
      
      IF( cl_sep /= ' ' )THEN
         ! get separator index
         il_sep=INDEX( TRIM(cl_string), TRIM(cl_sep) )
         il_lsep=LEN(TRIM(cl_sep)) 

         IF( il_sep /= 0 )THEN
            cf_elt=TRIM(ADJUSTL(cl_string(1:il_sep-1)))
         ELSE
            cf_elt=TRIM(ADJUSTL(cl_string))
         ENDIF

         ji=1
         DO WHILE( il_sep /= 0 .AND. ji /= id_ind )
            
            ji=ji+1
            
            cl_string=TRIM(cl_string(il_sep+il_lsep:))
            il_sep=INDEX( TRIM(cl_string), TRIM(cl_sep) )

            IF( il_sep /= 0 )THEN
               cf_elt=TRIM(ADJUSTL(cl_string(1:il_sep-1)))
            ELSE
               cf_elt=TRIM(ADJUSTL(cl_string))
            ENDIF

         ENDDO

         IF( ji /= id_ind ) cf_elt=''
      ELSE
         cf_elt=fct__split_space(TRIM(cl_string), id_ind)
      ENDIF

   END FUNCTION fct_split
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct__split_space(cd_string, id_ind) &
         & RESULT(cf_elt)
   !-------------------------------------------------------------------
   !> @brief This function split string of character 
   !> using space as separator, 
   !> and return the element on index ind.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_string string of character
   !> @param[in] id_ind    indice
   !> @return return the element of index id_ind
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=*), INTENT(IN) :: cd_string
      INTEGER(i4)     , INTENT(IN) :: id_ind

      ! function
      CHARACTER(LEN=lc)            :: cf_elt

      ! local variable
      CHARACTER(LEN=lc) :: cl_string

      INTEGER(i4) :: il_sep
      INTEGER(i4) :: il_lsep
      
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! initialize
      cf_elt=''
      cl_string=ADJUSTL(cd_string)

      ! get separator index
      il_sep=INDEX( TRIM(cl_string), ' ' )
      il_lsep=LEN(' ') 

      IF( il_sep /= 0 )THEN
         cf_elt=TRIM(ADJUSTL(cl_string(1:il_sep-1)))
      ELSE
         cf_elt=TRIM(ADJUSTL(cl_string))
      ENDIF

      ji=1
      DO WHILE( il_sep /= 0 .AND. ji /= id_ind )
         
         ji=ji+1
         
         cl_string=TRIM(cl_string(il_sep+il_lsep:))
         il_sep=INDEX( TRIM(cl_string), ' ' )

         IF( il_sep /= 0 )THEN
            cf_elt=TRIM(ADJUSTL(cl_string(1:il_sep-1)))
         ELSE
            cf_elt=TRIM(ADJUSTL(cl_string))
         ENDIF

      ENDDO

      IF( ji /= id_ind ) cf_elt=''

   END FUNCTION fct__split_space
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct_basename(cd_string, cd_sep) &
         & RESULT(cf_file)
   !-------------------------------------------------------------------
   !> @brief This function return basename of a filename.
   !>
   !> @details
   !> Actually it splits filename using sperarator '/'
   !> and return last string character.<br/>
   !> Optionally you could specify another separator.
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_string filename
   !> @param[in] cd_sep    separator character
   !> @return basename (filename without path)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=*), INTENT(IN) :: cd_string
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: cd_sep

      ! function
      CHARACTER(LEN=lc)            :: cf_file

      ! local variable
      CHARACTER(LEN=lc) :: cl_sep
      CHARACTER(LEN=lc) :: cl_string
      INTEGER(i4)       :: il_sep
      
      ! loop indices
      !----------------------------------------------------------------
      ! initialize
      cl_string=TRIM(ADJUSTL(cd_string))

      ! get separator
      cl_sep='/'
      IF( PRESENT(cd_sep) ) cl_sep=TRIM(ADJUSTL(cd_sep))

      il_sep=INDEX( TRIM(cl_string), TRIM(cl_sep), BACK=.TRUE.)
      cf_file=TRIM(cl_string(il_sep+1:))

   END FUNCTION fct_basename
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   PURE FUNCTION fct_dirname(cd_string, cd_sep) &
         & RESULT(cf_dir)
   !-------------------------------------------------------------------
   !> @brief This function return dirname of a filename.
   !>
   !> @details
   !> Actually it splits filename using sperarator '/'
   !> and return all except last string character.<br/>
   !> Optionally you could specify another separator.
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] cd_string filename
   !> @param[in] cd_sep    separator character
   !> @return dirname (path of the filename)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=*), INTENT(IN) :: cd_string
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: cd_sep

      ! function
      CHARACTER(LEN=lc)            :: cf_dir

      ! local variable
      CHARACTER(LEN=lc) :: cl_sep
      CHARACTER(LEN=lc) :: cl_string
      INTEGER(i4)       :: il_sep
      
      ! loop indices
      !----------------------------------------------------------------
      ! initialize
      cl_string=TRIM(ADJUSTL(cd_string))

      ! get separator
      cl_sep='/'
      IF( PRESENT(cd_sep) ) cl_sep=TRIM(ADJUSTL(cd_sep))

      il_sep=INDEX( TRIM(cl_string), TRIM(cl_sep), BACK=.TRUE.)
      IF( il_sep == 0 )THEN
         cf_dir=''
      ELSE
         cf_dir=TRIM(cl_string(1:il_sep))
      ENDIF

   END FUNCTION fct_dirname
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE fct_help(cd_filename, cd_err)
   !-------------------------------------------------------------------
   !> @brief
   !> This function show help message.
   !> 
   !> @details 
   !>  Optionaly, print error detected
   !>
   !> @author J.Paul
   !> @date October, 2019 - Initial Version
   !>
   !> @param[in] cd_filename   file name 
   !> @param[in] cd_err        error message
   !>
   !> @return print help message
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN) :: cd_filename
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: cd_err
      !----------------------------------------------------------------

      PRINT '( /,   a,/)', 'USAGE: '//TRIM(cd_filename)//' namelist [-v] [-h]'
      PRINT '(   2x,a,/)', 'positional arguments:'
      PRINT '(   5x,a   )',    'namelist                       '//TRIM(cd_filename)//" namelist"
      PRINT '( /,5x,a,/)', 'NB : a template of the namelist could be created running (in templates directory):'
      PRINT '(   8x,a  )',    'python create_templates.py '//TRIM(cd_filename)
      PRINT '( /,2x,a,/)', 'optional arguments:'
      PRINT '(   5x,a  )',    "-h, --help                      display this help and exit"
      PRINT '(   5x,a,/)',    "-v, --version                   output Siren's version information and exit"
      IF (PRESENT(cd_err)) THEN
         PRINT '(2x,a,/)', 'ERROR DETECTED:'
         PRINT '(5x,a,/)', TRIM(cd_err)
      ENDIF

   END SUBROUTINE fct_help
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE fct_version(cd_filename)
   !-------------------------------------------------------------------
   !> @brief
   !> This function show the version of Siren.
   !> 
   !> @author J.Paul
   !> @date October, 2019 - Initial Version
   !>
   !> @param[in] cd_filename   file name 
   !>
   !> @return print version message
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN) :: cd_filename
      !----------------------------------------------------------------

      PRINT '( /, a,/)', 'PROGRAM: Siren - '//TRIM(cd_filename)
      PRINT '(2x,2a  )', 'Revision of last commit : ', TRIM(fct_split(fct_split(cp_version,2,'$'),2,'Revision:'))
      PRINT '(2x,2a  )', 'Author   of last commit : ', TRIM(fct_split(fct_split(cp_author,2,'$'),2,'Author:'))
      PRINT '(2x,2a  )', 'Date     of last commit : ', TRIM(fct_split(fct_split(cp_date,2,'$'),2,'Date:'))
      PRINT '(2x,2a,/)', 'SVN URL                 : ', TRIM(fct_split(fct_split(fct_split(cp_url,2,'$'),2,'URL:'),1,'/src/global.f90'))

   END SUBROUTINE fct_version
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE fct

