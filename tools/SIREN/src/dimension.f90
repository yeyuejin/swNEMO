!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief 
!> This module manage dimension and how to change order of those dimension.
!>
!> @details
!>    define type TDIM:<br/>
!> @code
!>    TYPE(TDIM) :: tl_dim
!> @endcode
!>
!>    to initialize a dimension structure:<br/>
!> @code
!>    tl_dim=dim_init( cd_name, [id_len,] [ld_uld,] [cd_sname])
!> @endcode
!>       - cd_name is the dimension name
!>       - id_len is the dimension size [optional]
!>       - ld_uld is true if this dimension is the unlimited one [optional]
!>       - cd_sname is the dimension short name ('x','y','z','t') [optional]
!>
!>    to clean dimension structure:<br/>
!> @code
!>    CALL dim_clean(tl_dim)
!> @endcode
!>       - tl_dim : dimension strucutre or array of dimension structure
!>
!>    to print information about dimension structure:<br/>
!> @code
!>    CALL dim_print(tl_dim)
!> @endcode
!>
!>    to copy dimension structure in another one (using different memory cell):<br/>
!> @code
!>    tl_dim2=dim_copy(tl_dim1) 
!> @endcode
!>
!>    to get dimension name:<br/>
!>    - tl_dim\%c_name
!>
!>    to get dimension short name:<br/>
!>    - tl_dim\%c_sname
!>
!>    to get dimension length:<br/>
!>    - tl_dim\%i_len
!>
!>    to know if dimension is the unlimited one:<br/>
!>    - tl_dim\%l_uld
!>
!>    to get dimension id (for variable or file dimension):<br/>
!>    - tl_dim\%i_id 
!>
!>    to know if dimension is used (for variable or file dimension):<br/>
!>    - tl_dim\%l_use
!>
!>    Former function or information concern only one dimension. However
!>    variables as well as files use usually 4 dimensions.<br/>
!>    To easily work with variable we want they will be all 4D and ordered as
!>    following: ('x','y','z','t').<br/>
!>    Functions and subroutines below, allow to reorder dimension of 
!>    variable.<br/>
!>    
!>    Suppose we defined the array of dimension structure below:<br/>
!> @code
!>    TYPE(TDIM), DIMENSION(4) :: tl_dim 
!>    tl_dim(1)=dim_init( 'X', id_len=10)
!>    tl_dim(2)=dim_init( 'T', id_len=3, ld_uld=.TRUE.)
!> @endcode
!>
!>    to reorder dimension (default order: ('x','y','z','t')):<br/>
!> @code
!>    CALL dim_reorder(tl_dim(:))
!> @endcode
!>
!>    This subroutine filled dimension structure with unused dimension, 
!>    then switch from "disordered" dimension to "ordered" dimension.<br/>
!>    The dimension structure return will be:<br/>
!>    tl_dim(1) => 'X', i_len=10, l_use=T, l_uld=F<br/>
!>    tl_dim(2) => 'Y', i_len=1,  l_use=F, l_uld=F<br/> 
!>    tl_dim(3) => 'Z', i_len=1,  l_use=F, l_uld=F<br/>
!>    tl_dim(4) => 'T', i_len=3,  l_use=T, l_uld=T<br/>
!>
!>    After using subroutine dim_reorder you could use functions and subroutine
!>    below.<br/>
!>
!>    to use another dimension order.<br/>
!> @code
!>    CALL dim_reorder(tl(dim(:), cl_neworder)
!> @endcode
!>    - cl_neworder : character(len=4) (example: 'yxzt')
!>
!>    to switch dimension array from ordered dimension to disordered
!> dimension:<br/>
!> @code
!>    CALL dim_disorder(tl_dim(:))
!> @endcode
!>
!>    to fill unused dimension of an array of dimension structure.<br/>
!> @code
!>    tl_dimout(:)=dim_fill_unused(tl_dimin(:))
!> @endcode
!>    - tl_dimout(:) : 1D array (4elts) of dimension strcuture
!>    - tl_dimin(:)  : 1D array (<=4elts) of dimension structure
!>
!>    to reshape array of value in "ordered" dimension:<br/>
!> @code
!>    CALL dim_reshape_2xyzt(tl_dim(:), value(:,:,:,:))
!> @endcode
!>       - value must be a 4D array of real(8) value "disordered"
!>
!>    to reshape array of value in "disordered" dimension:<br/>
!> @code
!>    CALL dim_reshape_xyzt2(tl_dim(:), value(:,:,:,:))
!> @endcode
!>       - value must be a 4D array of real(8) value "ordered"
!>
!>    to reorder a 1D array of 4 elements in "ordered" dimension:<br/>
!> @code
!>    CALL dim_reorder_2xyzt(tl_dim(:), tab(:))
!> @endcode
!>       - tab must be a 1D array with 4 elements "disordered".
!>       It could be composed of character, integer(4), or logical
!> 
!>    to reorder a 1D array of 4 elements in "disordered" dimension:<br/>
!> @code
!>    CALL dim_reorder_xyzt2(tl_dim(:), tab(:))
!> @endcode
!>       - tab must be a 1D array with 4 elements "ordered".
!>       It could be composed of character, integer(4), or logical
!>
!>    to get dimension index from a array of dimension structure, 
!>    given dimension name or short name :<br/>
!> @code
!>    index=dim_get_index( tl_dim(:), [cl_name, cl_sname] )
!> @endcode
!>       - tl_dim(:) : array of dimension structure
!>       - cl_name : dimension name [optional]
!>       - cl_sname: dimension short name [optional]
!>
!>    to get dimension id used in an array of dimension structure, 
!>    given dimension name or short name :<br/>
!> @code
!>    id=dim_get_id( tl_dim(:), [cl_name, cl_sname] )
!> @endcode
!>       - tl_dim(:) : array of dimension structure
!>       - cl_name : dimension name [optional]
!>       - cl_sname: dimension short name [optional]
!>
!> @author J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date September, 2015
!> - manage useless (dummy) dimension
!> @date October, 2016
!> - dimension allowed read in configuration file
!> @date May, 2019
!> - read number of element for each dimension allowed in configuration file 
!> - read number of element for each dummy array in configuration file
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE dim

   USE global                          ! global variable
   USE kind                            ! F90 kind parameter
   USE logger                          ! log file manager
   USE fct                             ! basic useful function

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! type and variable
   PUBLIC :: TDIM              !< dimension structure

   PRIVATE :: im_ndumdim       !< number of elt in dummy dimension array
   PRIVATE :: cm_dumdim        !< dummy dimension array
   PRIVATE :: im_dimX          !< number of elt in x dimension array
   PRIVATE :: im_dimY          !< number of elt in y dimension array
   PRIVATE :: im_dimZ          !< number of elt in z dimension array
   PRIVATE :: im_dimT          !< number of elt in t dimension array
   PRIVATE :: cm_dimX          !< x dimension array
   PRIVATE :: cm_dimY          !< y dimension array
   PRIVATE :: cm_dimZ          !< z dimension array
   PRIVATE :: cm_dimT          !< t dimension array

   ! function and subroutine
   PUBLIC :: dim_init          !< initialize dimension structure
   PUBLIC :: dim_clean         !< clean dimension structuree
   PUBLIC :: dim_print         !< print dimension information
   PUBLIC :: dim_copy          !< copy dimension structure
   PUBLIC :: dim_reorder       !< filled dimension structure to switch from disordered to ordered dimension
   PUBLIC :: dim_disorder      !< switch dimension array from ordered to disordered dimension
   PUBLIC :: dim_fill_unused   !< filled dimension structure with unused dimension 
   PUBLIC :: dim_reshape_2xyzt !< reshape array dimension to ('x','y','z','t')
   PUBLIC :: dim_reshape_xyzt2 !< reshape array dimension from ('x','y','z','t')
   PUBLIC :: dim_reorder_2xyzt !< reorder 1D array to ('x','y','z','t')
   PUBLIC :: dim_reorder_xyzt2 !< reorder 1D array from ('x','y','z','t')
   PUBLIC :: dim_get_index     !< get dimension index in array of dimension structure
   PUBLIC :: dim_get_id        !< get dimension id in array of dimension structure
   PUBLIC :: dim_get_dummy     !< fill dummy dimension array
   PUBLIC :: dim_is_dummy      !< check if dimension is defined as dummy dimension
   PUBLIC :: dim_def_extra     !< read dimension configuration file, and save dimension allowed.

   PRIVATE :: dim__reshape_2xyzt_dp ! reshape real(8) 4D array to ('x','y','z','t')
   PRIVATE :: dim__reshape_xyzt2_dp ! reshape real(8) 4D array from ('x','y','z','t')
   PRIVATE :: dim__reorder_2xyzt_i4 ! reorder integer(4) 1D array to ('x','y','z','t')
   PRIVATE :: dim__reorder_xyzt2_i4 ! reorder integer(4) 1D array from ('x','y','z','t')
   PRIVATE :: dim__reorder_2xyzt_l  ! reorder logical 1D array to ('x','y','z','t')
   PRIVATE :: dim__reorder_xyzt2_l  ! reorder logical 1D array from ('x','y','z','t')
   PRIVATE :: dim__reorder_2xyzt_c  ! reorder string 1D array to ('x','y','z','t')
   PRIVATE :: dim__reorder_xyzt2_c  ! reorder string 1D array from ('x','y','z','t')
   PRIVATE :: dim__clean_unit       ! clean one dimension structure
   PRIVATE :: dim__clean_arr        ! clean a array of dimension structure
   PRIVATE :: dim__print_unit       ! print information on one dimension structure
   PRIVATE :: dim__print_arr        ! print information on a array of dimension structure
   PRIVATE :: dim__copy_unit        ! copy dimension structure
   PRIVATE :: dim__copy_arr         ! copy array of dimension structure
   PRIVATE :: dim__is_allowed

   TYPE TDIM !< dimension structure
      CHARACTER(LEN=lc) :: c_name = ''       !< dimension name
      CHARACTER(LEN=lc) :: c_sname = 'u'     !< dimension short name
      INTEGER(i4)       :: i_id  = 0         !< dimension id
      INTEGER(i4)       :: i_len = 1         !< dimension length
      LOGICAL           :: l_uld = .FALSE.   !< dimension unlimited or not
      LOGICAL           :: l_use = .FALSE.   !< dimension used or not
      INTEGER(i4)       :: i_2xyzt = 0       !< indices to reshape array to ('x','y','z','t')
      INTEGER(i4)       :: i_xyzt2 = 0       !< indices to reshape array from ('x','y','z','t')
   END TYPE

   INTEGER(i4)                               , SAVE :: im_ndumdim !< number of elt in dummy dimension array
   CHARACTER(LEN=lc), DIMENSION(ip_maxdumcfg), SAVE :: cm_dumdim  !< dummy dimension
   INTEGER(i4)                               , SAVE :: im_dimX    !< number of elt in x dimension array
   INTEGER(i4)                               , SAVE :: im_dimY    !< number of elt in y dimension array
   INTEGER(i4)                               , SAVE :: im_dimZ    !< number of elt in z dimension array
   INTEGER(i4)                               , SAVE :: im_dimT    !< number of elt in t dimension array
   CHARACTER(LEN=lc), DIMENSION(ip_maxdimcfg), SAVE :: cm_dimX    !< x dimension
   CHARACTER(LEN=lc), DIMENSION(ip_maxdimcfg), SAVE :: cm_dimY    !< y dimension
   CHARACTER(LEN=lc), DIMENSION(ip_maxdimcfg), SAVE :: cm_dimZ    !< z dimension
   CHARACTER(LEN=lc), DIMENSION(ip_maxdimcfg), SAVE :: cm_dimT    !< t dimension

   INTERFACE dim_print
      MODULE PROCEDURE dim__print_unit ! print information on one dimension
      MODULE PROCEDURE dim__print_arr  ! print information on a array of dimension
   END INTERFACE dim_print

   INTERFACE dim_clean
      MODULE PROCEDURE dim__clean_unit ! clean one dimension
      MODULE PROCEDURE dim__clean_arr  ! clean a array of dimension
   END INTERFACE dim_clean

   INTERFACE dim_copy
      MODULE PROCEDURE dim__copy_unit  ! copy dimension structure
      MODULE PROCEDURE dim__copy_arr   ! copy array of dimension structure
   END INTERFACE

   INTERFACE dim_reshape_2xyzt
      MODULE PROCEDURE dim__reshape_2xyzt_dp   ! reshape real(8) 4D array to ('x','y','z','t')
   END INTERFACE dim_reshape_2xyzt

   INTERFACE dim_reshape_xyzt2
      MODULE PROCEDURE dim__reshape_xyzt2_dp   ! reshape real(8) 4D array from ('x','y','z','t')
   END INTERFACE dim_reshape_xyzt2

   INTERFACE dim_reorder_2xyzt
      MODULE PROCEDURE dim__reorder_2xyzt_i4   ! reorder integer(4) 1D array to ('x','y','z','t')
      MODULE PROCEDURE dim__reorder_2xyzt_c    ! reorder string 1D array to ('x','y','z','t')
      MODULE PROCEDURE dim__reorder_2xyzt_l    ! reorder logical 1D array to ('x','y','z','t')
   END INTERFACE dim_reorder_2xyzt

   INTERFACE dim_reorder_xyzt2
      MODULE PROCEDURE dim__reorder_xyzt2_i4   ! reorder integer(4) 1D array from ('x','y','z','t')
      MODULE PROCEDURE dim__reorder_xyzt2_c    ! reorder string 1D array from ('x','y','z','t')
      MODULE PROCEDURE dim__reorder_xyzt2_l    ! reorder logical 1D array from ('x','y','z','t')  
   END INTERFACE dim_reorder_xyzt2

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dim__copy_arr(td_dim) &
         & RESULT (tf_dim)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine copy a array of dimension structure in another one
   !> @details 
   !> see dim__copy_unit
   !>
   !> @warning do not use on the output of a function who create or read an
   !> structure (ex: tl_dim=dim_copy(dim_init()) is forbidden).
   !> This will create memory leaks.
   !> @warning to avoid infinite loop, do not use any function inside 
   !> this subroutine
   !>
   !> @author J.Paul
   !> @date November, 2014 - Initial Version
   !
   !> @param[in] td_dim   array of dimension structure
   !> @return copy of input array of dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDIM), DIMENSION(:), INTENT(IN)   :: td_dim

      ! function
      TYPE(TDIM), DIMENSION(SIZE(td_dim(:))) :: tf_dim

      ! local variable
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      DO ji=1,SIZE(td_dim(:))
         tf_dim(ji)=dim_copy(td_dim(ji))
      ENDDO

   END FUNCTION dim__copy_arr
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dim__copy_unit(td_dim) &
         & RESULT (tf_dim)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine copy an dimension structure in another one
   !> @details 
   !> dummy function to get the same use for all structure
   !>
   !> @warning do not use on the output of a function who create or read an
   !> structure (ex: tl_dim=dim_copy(dim_init()) is forbidden).
   !> This will create memory leaks.
   !> @warning to avoid infinite loop, do not use any function inside 
   !> this subroutine
   !>
   !> @author J.Paul
   !> @date November, 2014 - Initial Version
   !>
   !> @param[in] td_dim   dimension structure
   !> @return copy of input dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDIM), INTENT(IN)  :: td_dim

      ! function
      TYPE(TDIM)              :: tf_dim

      ! local variable
      !----------------------------------------------------------------

      tf_dim=td_dim

   END FUNCTION dim__copy_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dim_get_index(td_dim, cd_name, cd_sname) &
         & RESULT (if_idx)
   !-------------------------------------------------------------------
   !> @brief This function returns dimension index,
   !> given dimension name or short name.
   !>
   !> @details
   !> the function check dimension name, in the array of dimension structure.
   !> dimension could be used or not.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014 
   !> - do not check if dimension used
   !>
   !> @param[in] td_dim    array of dimension structure
   !> @param[in] cd_name   dimension name
   !> @param[in] cd_sname  dimension short name
   !> @return dimension index
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDIM)      , DIMENSION(:), INTENT(IN) :: td_dim
      CHARACTER(LEN=*),               INTENT(IN) :: cd_name
      CHARACTER(LEN=*),               INTENT(IN), OPTIONAL :: cd_sname

      ! function
      INTEGER(i4)                                :: if_idx

      ! local variable
      CHARACTER(LEN=lc) :: cl_name
      CHARACTER(LEN=lc) :: cl_dim_name
      CHARACTER(LEN=lc) :: cl_sname
      CHARACTER(LEN=lc) :: cl_dim_sname

      INTEGER(i4) :: il_ndim

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! init
      if_idx=0

      il_ndim=SIZE(td_dim(:))

      ! look for dimension name
      cl_name=fct_lower(cd_name)
      ! check if dimension is in array of dimension structure
      DO ji=1,il_ndim
         cl_dim_name=fct_lower(td_dim(ji)%c_name)
         IF( TRIM(cl_dim_name) == TRIM(cl_name) )THEN
            if_idx=ji
            EXIT
         ENDIF
      ENDDO

      ! look for dimension short name
      IF(  if_idx == 0 )THEN

         cl_sname=fct_lower(cd_name)
         ! check if dimension is in array of dimension structure
         DO ji=1,il_ndim
            cl_dim_sname=fct_lower(td_dim(ji)%c_sname)
            IF( (TRIM(cl_dim_sname) == TRIM(cl_sname) ) )THEN
               CALL logger_debug("DIM GET INDEX: variable short name "//&
                  &              TRIM(ADJUSTL(cd_name))//" already in file")
               if_idx=ji
               EXIT
            ENDIF
         ENDDO

      ENDIF

      ! look for dimension short name
      IF( PRESENT(cd_sname) )THEN
         IF( if_idx == 0 )THEN

            cl_sname=fct_lower(cd_sname)
            ! check if dimension is in array of dimension structure
            DO ji=1,il_ndim
               cl_dim_sname=fct_lower(td_dim(ji)%c_sname)
               IF( (TRIM(cl_dim_sname) == TRIM(cl_sname) ) )THEN
                  CALL logger_debug("DIM GET INDEX: variable short name "//&
                     &              TRIM(ADJUSTL(cd_sname))//" already in file")
                  if_idx=ji
                  EXIT
               ENDIF
            ENDDO

         ENDIF
      ENDIF

   END FUNCTION dim_get_index
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dim_get_id(td_dim, cd_name, cd_sname) &
         & RESULT (if_id)
   !-------------------------------------------------------------------
   !> @brief This function returns dimension id, in a array of dimension structure,
   !> given dimension name, or short name. 
   !> @note only dimension used are checked.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[in] td_dim    dimension structure
   !> @param[in] cd_name   dimension name or short name
   !> @param[in] cd_sname  dimension short name
   !> @return dimension id
   !-------------------------------------------------------------------
   
      IMPLICIT NONE

      ! Argument
      TYPE(TDIM),       DIMENSION(:), INTENT(IN) :: td_dim
      CHARACTER(LEN=*),               INTENT(IN) :: cd_name
      CHARACTER(LEN=*),               INTENT(IN), OPTIONAL :: cd_sname

      ! function
      INTEGER(i4)                                :: if_id

      ! local variable
      CHARACTER(LEN=lc) :: cl_name
      CHARACTER(LEN=lc) :: cl_dim_name
      CHARACTER(LEN=lc) :: cl_sname
      CHARACTER(LEN=lc) :: cl_dim_sname

      INTEGER(i4) :: il_ndim

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------
      ! init
      if_id=0

      il_ndim=SIZE(td_dim(:))

      ! look for dimension name
      cl_name=fct_lower(cd_name)
      ! check if dimension is in array of dimension structure and used
      jj=0
      DO ji=1,il_ndim
         cl_dim_name=fct_lower(td_dim(ji)%c_name)
         IF( TRIM(cl_dim_name) == TRIM(cl_name) .AND. &
         &   td_dim(ji)%l_use )THEN
            IF( td_dim(ji)%i_id /= 0 )THEN
               if_id=td_dim(ji)%i_id
               EXIT
            ENDIF
         ENDIF
      ENDDO

      ! look for dimension short name
      IF(  if_id == 0 )THEN

         cl_sname=fct_lower(cd_name)
         ! check if dimension is in array of dimension structure and used
         jj=0
         DO ji=1,il_ndim
            cl_dim_sname=fct_lower(td_dim(ji)%c_sname)
            IF( (TRIM(cl_dim_sname) == TRIM(cl_sname) ).AND.&
            &   td_dim(ji)%l_use )THEN
               IF( td_dim(ji)%i_id /= 0 )THEN
                  if_id=td_dim(ji)%i_id
                  EXIT
               ENDIF
            ENDIF
         ENDDO

      ENDIF

      ! look for dimension short name
      IF( PRESENT(cd_sname) )THEN
         IF(  if_id == 0 )THEN

            cl_sname=fct_lower(cd_sname)
            ! check if dimension is in array of dimension structure and used
            jj=0
            DO ji=1,il_ndim
               cl_dim_sname=fct_lower(td_dim(ji)%c_sname)
               IF( (TRIM(cl_dim_sname) == TRIM(cl_sname) ).AND.&
                  & td_dim(ji)%l_use )THEN
                  IF( td_dim(ji)%i_id /= 0 )THEN
                     if_id=td_dim(ji)%i_id
                     EXIT
                  ENDIF
               ENDIF
            ENDDO

         ENDIF
      ENDIF

   END FUNCTION dim_get_id
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dim_init(cd_name, id_len, ld_uld, cd_sname, ld_use) &
         & RESULT (tf_dim)
   !-------------------------------------------------------------------
   !> @brief This function initialize a dimension structure with given
   !> name.<br/>
   !> @details
   !> Optionally length could be inform, as well as short name and if dimension
   !> is unlimited or not.<br/>
   !> By default, define dimension is supposed to be used.
   !> Optionally you could force a defined dimension to be unused. 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date February, 2015 
   !> - add optional argument to define dimension unused
   !> @date July, 2015
   !> - Bug fix: inform order to disorder table instead of disorder to order
   !> table
   !> @date May, 2019
   !> - use number of element for each dimention allowed, instead of while loop
   !>
   !> @param[in] cd_name   dimension name
   !> @param[in] id_len    dimension length
   !> @param[in] ld_uld    dimension unlimited
   !> @param[in] cd_sname  dimension short name
   !> @param[in] ld_use    dimension use or not
   !> @return dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument 
      CHARACTER(LEN=*), INTENT(IN)  :: cd_name
      INTEGER(i4),      INTENT(IN), OPTIONAL :: id_len
      LOGICAL,          INTENT(IN), OPTIONAL :: ld_uld
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: cd_sname
      LOGICAL,          INTENT(IN), OPTIONAL :: ld_use

      ! function
      TYPE(TDIM)                    :: tf_dim

      ! local variable
      CHARACTER(LEN=lc) :: cl_name
      CHARACTER(LEN=lc) :: cl_sname
      !----------------------------------------------------------------

      ! clean dimension
      CALL dim_clean(tf_dim)

      cl_name=fct_upper(cd_name)

      CALL logger_debug( &
      &  " DIM INIT: dimension name: "//TRIM(cl_name) )
      tf_dim%c_name=TRIM(ADJUSTL(cd_name))

      IF( PRESENT(id_len) )THEN
         CALL logger_debug( &
         &  " DIM INIT: dimension length: "//fct_str(id_len) )
         tf_dim%i_len=id_len
      ENDIF

      ! define dimension is supposed to be used
      IF( PRESENT(ld_use) )THEN
         tf_dim%l_use=ld_use
      ELSE
         tf_dim%l_use=.TRUE.
      ENDIF

      IF( PRESENT(cd_sname) )THEN

         cl_sname=fct_lower(cd_sname)

         IF( TRIM(cl_sname) == 'x' .OR. &
         &   TRIM(cl_sname) == 'y' .OR. & 
         &   TRIM(cl_sname) == 'z' .OR. & 
         &   TRIM(cl_sname) == 't' )THEN
            CALL logger_debug( &
            &  " DIM INIT: dimension short name: "//TRIM(cd_sname) )
            tf_dim%c_sname=TRIM(cd_sname)
         ELSE
            CALL logger_warn("DIM INIT: invalid short name."//&
            " choose between ('x','y','z','t')")
         ENDIF
      ENDIF

      IF( TRIM(fct_lower(tf_dim%c_sname)) == 'u' )THEN

         cl_name=fct_lower(cd_name)

         IF(     dim__is_allowed(TRIM(cl_name), cm_dimX(:), im_dimX) )THEN
            tf_dim%c_sname='x'
         ELSEIF( dim__is_allowed(TRIM(cl_name), cm_dimY(:), im_dimY) )THEN
            tf_dim%c_sname='y'
         ELSEIF( dim__is_allowed(TRIM(cl_name), cm_dimZ(:), im_dimZ) )THEN
            tf_dim%c_sname='z'
         ELSEIF( dim__is_allowed(TRIM(cl_name), cm_dimT(:), im_dimT) )THEN
            tf_dim%c_sname='t'
         ELSE
            CALL logger_warn("DIM INIT: "//TRIM(cd_name)//&
            " not allowed.")
         ENDIF

      ENDIF

      IF( PRESENT(ld_uld) )THEN
         CALL logger_debug( &
         &  " DIM INIT: unlimited dimension: "//fct_str(ld_uld) )
         tf_dim%l_uld=ld_uld
      ELSE
         IF( TRIM(fct_lower(tf_dim%c_sname)) =='t'  )THEN
            tf_dim%l_uld=.TRUE.
         ENDIF
      ENDIF
      
      ! get dimension order indices
      tf_dim%i_xyzt2=SCAN(TRIM(cp_dimorder),TRIM(tf_dim%c_sname))

   END FUNCTION dim_init
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dim__print_arr(td_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine print informations of an array of dimension. 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[in] td_dim array of dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TDIM), DIMENSION(:), INTENT(IN) :: td_dim

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      DO ji=1,SIZE(td_dim(:))
         CALL dim_print(td_dim(ji))
      ENDDO

   END SUBROUTINE dim__print_arr
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dim__print_unit(td_dim)
   !-------------------------------------------------------------------
   !> @brief This subrtoutine print dimension information. 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[in] td_dim dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TDIM), INTENT(IN) :: td_dim
      !----------------------------------------------------------------

      WRITE(*,'((3x,a,a),(/6x,a,a),(a,i1),(a,i5),2(a,a),2(a,i1))')   &
      &        " dimension : ",TRIM(td_dim%c_name),               &
      &        " short name : ",TRIM(td_dim%c_sname),        &
      &        " id : ",td_dim%i_id,                         &
      &        " len : ",td_dim%i_len,                       &
      &        " use : ",TRIM(fct_str(td_dim%l_use)),        &
      &        " uld : ",TRIM(fct_str(td_dim%l_uld)),        &
      &        " xyzt2 : ",td_dim%i_xyzt2,                   &
      &        " 2xyzt : ",td_dim%i_2xyzt

   END SUBROUTINE dim__print_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dim_fill_unused(td_dim) &
         & RESULT (tf_dim)
   !-------------------------------------------------------------------
   !> @brief This function fill unused dimension of an array of dimension
   !> and return a 4 elts array of dimension structure. 
   !> @details
   !> output dimensions 'x','y','z' and 't' are all informed.
   !>
   !> @note without input array of dimension, return 
   !> a 4 elts array of dimension structure all unused
   !> (case variable 0d)
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2015 
   !> - Bug fix: use order to disorder table (see dim_init)
   !>
   !> @param[in] td_dim array of dimension structure
   !> @return  4elts array of dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TDIM), DIMENSION(:), INTENT(IN), OPTIONAL :: td_dim

      ! function
      TYPE(TDIM), DIMENSION(ip_maxdim)               :: tf_dim

      ! local variable
      CHARACTER(LEN=lc)                       :: cl_dimin
      INTEGER(i4)      , DIMENSION(1)         :: il_ind  ! index
      
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( PRESENT(td_dim) )THEN
         tf_dim(1:SIZE(td_dim(:)))=td_dim(:)
      ENDIF
      ! concatenate short nem dimension in a character string
      cl_dimin=fct_lower(fct_concat(tf_dim(:)%c_sname))
      DO ji = 1, ip_maxdim

         ! search missing dimension
         IF( INDEX(cl_dimin,TRIM(fct_lower(cp_dimorder(ji:ji)))) == 0 )THEN
            ! search first empty dimension (see dim_init)
            il_ind(:)=MINLOC( tf_dim(:)%i_xyzt2, tf_dim(:)%i_xyzt2 == 0 )

            ! put missing dimension instead of empty one
            tf_dim(il_ind(1))%c_sname=fct_lower(cp_dimorder(ji:ji))
            ! update output structure
            tf_dim(il_ind(1))%c_name=fct_lower(cp_dimorder(ji:ji))
            tf_dim(il_ind(1))%i_xyzt2=ji
            tf_dim(il_ind(1))%i_len=1
            tf_dim(il_ind(1))%l_use=.FALSE.
         ENDIF

      ENDDO
         
   END FUNCTION dim_fill_unused
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dim_reorder(td_dim, cd_dimorder)
   !-------------------------------------------------------------------
   !> @brief 
   !> This subroutine switch element of an array (4 elts) of dimension 
   !> structure 
   !> from disordered dimension to ordered dimension <br/>
   !>
   !> @details
   !> Optionally you could specify dimension order to output
   !> (default 'xyzt')
   !> Example: (/'z','x','t','y'/) => (/'x','y','z','t'/)
   !>
   !> @warning this subroutine change dimension order 
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014 
   !> - allow to choose ordered dimension to be output
   !>
   !> @param[inout] td_dim    array of dimension structure
   !> @param[in] cd_dimorder  dimension order to be output 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TDIM)              , DIMENSION(:), INTENT(INOUT) :: td_dim
      CHARACTER(LEN=ip_maxdim)              , INTENT(IN   ), OPTIONAL :: cd_dimorder

      ! local variable
      INTEGER(i4)                             :: il_ind

      CHARACTER(LEN=lc)                       :: cl_dimin
      CHARACTER(LEN=lc)                       :: cl_dimorder

      TYPE(TDIM)       , DIMENSION(ip_maxdim) :: tl_dim

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( SIZE(td_dim(:)) /= ip_maxdim )THEN
         CALL logger_error("DIM REORDER: invalid dimension of array dimension.")
      ELSE

         cl_dimorder=TRIM(cp_dimorder)
         IF( PRESENT(cd_dimorder) ) cl_dimorder=TRIM(ADJUSTL(cd_dimorder))

         ! add id if dimension used and no id
         DO ji=1, ip_maxdim

            IF( td_dim(ji)%l_use )THEN
               IF( td_dim(ji)%i_id == 0 )THEN
                  td_dim(ji)%i_id=MAXVAL(td_dim(:)%i_id)+1
               ENDIF
            ELSE
               td_dim(ji)%i_id=0
               td_dim(ji)%i_xyzt2=0
               td_dim(ji)%i_2xyzt=0
               td_dim(ji)%c_sname='u'
               td_dim(ji)%c_name=''
               td_dim(ji)%l_uld=.FALSE.
            ENDIF

         ENDDO

         ! fill unused dimension
         tl_dim(:)=dim_fill_unused(td_dim(:))
         cl_dimin=fct_lower(fct_concat(tl_dim(:)%c_sname))

         ! compute input id from output id (xyzt) 
         DO ji = 1, ip_maxdim
             
            il_ind=SCAN(TRIM(cl_dimorder),TRIM(cl_dimin(ji:ji)))
            IF( il_ind /= 0 )THEN
               tl_dim(ji)%i_xyzt2=il_ind
            ENDIF
            
         ENDDO

         ! compute output id (xyzt) from input id 
         DO ji = 1, ip_maxdim
             
            il_ind=SCAN(TRIM(cl_dimin),TRIM(cl_dimorder(ji:ji)))
            IF( il_ind /= 0 )THEN
               tl_dim(ji)%i_2xyzt=il_ind
            ENDIF
            
         ENDDO

         ! change dimension order to ('x','y','z','t')
         td_dim(:)%c_name  = dim_reorder_2xyzt(tl_dim(:),tl_dim(:)%c_name)
         td_dim(:)%c_sname = dim_reorder_2xyzt(tl_dim(:),tl_dim(:)%c_sname)
         td_dim(:)%i_id    = dim_reorder_2xyzt(tl_dim(:),tl_dim(:)%i_id  )
         td_dim(:)%i_len   = dim_reorder_2xyzt(tl_dim(:),tl_dim(:)%i_len )
         td_dim(:)%l_uld   = dim_reorder_2xyzt(tl_dim(:),tl_dim(:)%l_uld )
         td_dim(:)%l_use   = dim_reorder_2xyzt(tl_dim(:),tl_dim(:)%l_use )
         td_dim(:)%i_2xyzt = tl_dim(:)%i_2xyzt
         td_dim(:)%i_xyzt2 = tl_dim(:)%i_xyzt2

         ! clean
         CALL dim_clean(tl_dim(:))
      ENDIF

   END SUBROUTINE dim_reorder
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dim_disorder(td_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine switch dimension array from ordered dimension ('x','y','z','t')
   !> to disordered dimension. <br/>
   !> @details
   !> Example: (/'x','y','z','t'/) => (/'z','x','t','y'/)<br/>
   !  This is useful to add dimension in a variable or file.
   !> @warning this subroutine change dimension order 
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[inout] td_dim array of dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TDIM), DIMENSION(:), INTENT(INOUT) :: td_dim

      ! local variable

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      IF( SIZE(td_dim(:)) /= ip_maxdim )THEN
         CALL logger_error("DIM DISORDER: invalid dimension of array dimension.")
      ELSE      
         ! add dummy xyzt2 id to unused dimension
         jj=1
         DO ji = 1, ip_maxdim
            IF( .NOT. td_dim(ji)%l_use .AND. td_dim(ji)%i_xyzt2 == 0 )THEN
               DO WHILE( ANY( td_dim(:)%i_xyzt2 == jj ))
                  jj=jj+1
               ENDDO
               td_dim(ji)%i_xyzt2=jj
            ENDIF
         ENDDO

         ! change dimension order from ('x','y','z','t')
         td_dim(:)%c_name  = dim_reorder_xyzt2(td_dim,td_dim(:)%c_name)
         td_dim(:)%c_sname = dim_reorder_xyzt2(td_dim,td_dim(:)%c_sname)
         td_dim(:)%i_id    = dim_reorder_xyzt2(td_dim,td_dim(:)%i_id  )
         td_dim(:)%i_len   = dim_reorder_xyzt2(td_dim,td_dim(:)%i_len )
         td_dim(:)%l_uld   = dim_reorder_xyzt2(td_dim,td_dim(:)%l_uld )
         td_dim(:)%l_use   = dim_reorder_xyzt2(td_dim,td_dim(:)%l_use )

         ! remove dummy xyzt2 id from unused dimension
         DO ji = 1, ip_maxdim
            IF( .NOT. td_dim(ji)%l_use )THEN
               td_dim(ji)%i_id=0
               td_dim(ji)%i_xyzt2=0
               td_dim(ji)%c_sname='u'
               td_dim(ji)%c_name=''
               td_dim(ji)%l_uld=.FALSE.
            ENDIF
         ENDDO
      ENDIF

   END SUBROUTINE dim_disorder
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dim__reshape_2xyzt_dp(td_dim, dd_value) &
         & RESULT (df_value)
   !-------------------------------------------------------------------
   !> @brief This function reshape real(8) 4D array   
   !> to an ordered array, as defined by dim_reorder.<br/>
   !> @details
   !> Example: (/'z','x','t','y'/) => (/'x','y','z','t'/)
   !
   !> @note you must have run dim_reorder before use this subroutine
   !
   !> @warning output array dimension differ from input array dimension
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - do not reshape array already order
   !>
   !> @param[in] td_dim    array of dimension structure
   !> @param[in] dd_value  array of value to reshape
   !> @return array of value reshaped 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TDIM), DIMENSION(:)      , INTENT(IN) :: td_dim
      REAL(dp)  , DIMENSION(:,:,:,:), INTENT(IN) :: dd_value
      
      ! function
      REAL(dp), DIMENSION(td_dim(1)%i_len, &
         &                td_dim(2)%i_len, &
         &                td_dim(3)%i_len, &
         &                td_dim(4)%i_len)       :: df_value

      ! local variable
      INTEGER(i4)      , DIMENSION(ip_maxdim) :: il_shape
      CHARACTER(LEN=lc)                       :: cl_dim
      
      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      IF( SIZE(td_dim(:)) /= ip_maxdim )THEN
         CALL logger_error("DIM RESHAPE 2 XYZT: invalid dimension of "//&
            &  "array dimension.")
      ELSE      

         IF( ANY(td_dim(:)%i_2xyzt==0) .OR. ANY(td_dim(:)%i_xyzt2==0) )THEN

            CALL logger_fatal( &
            &  "  DIM RESHAPE 2 XYZT: you should have run dim_reorder"// &
            &  "   before running RESHAPE" )

         ENDIF

         il_shape=SHAPE(dd_value)
         ! check input dimension
         IF( ANY(il_shape(:) /= (/ td_dim(td_dim(1)%i_xyzt2)%i_len, &
                               &   td_dim(td_dim(2)%i_xyzt2)%i_len, &
                               &   td_dim(td_dim(3)%i_xyzt2)%i_len, &
                               &   td_dim(td_dim(4)%i_xyzt2)%i_len /)) )THEN

            DO ji=1,ip_maxdim
               CALL logger_debug(" DIM RESHAPE 2 XYZT: dim "//&
                  &     TRIM(td_dim(td_dim(ji)%i_xyzt2)%c_name)//" "//&
                  &     TRIM(fct_str(td_dim(td_dim(ji)%i_xyzt2)%i_len))//" vs "//&
                  &     TRIM(fct_str(il_shape(ji))) )
            ENDDO
            CALL logger_fatal(" DIM RESHAPE 2 XYZT: wrong input dimensions " )

         ELSE

            ! write some informations
            cl_dim="(/"
            DO ji=1,ip_maxdim-1
               cl_dim=TRIM(cl_dim)//TRIM(fct_str(il_shape(ji)))//','
            ENDDO
            cl_dim=TRIM(cl_dim)//TRIM(fct_str(il_shape(ip_maxdim)))//"/)"

            CALL logger_debug(" DIM RESHAPE 2 XYZT: input dimensions are "//&
            &  TRIM(cl_dim) )

            cl_dim="(/"
            DO ji=1,ip_maxdim-1
               cl_dim=TRIM(cl_dim)//TRIM(fct_str(td_dim(ji)%i_len))//','
            ENDDO
            cl_dim=TRIM(cl_dim)//TRIM(fct_str(td_dim(ip_maxdim)%i_len))//"/)"

            CALL logger_debug(" DIM RESHAPE 2 XYZT: ouput dimensions should be "//&
            &  TRIM(cl_dim) )

            IF( td_dim(1)%i_xyzt2 == 1 .AND. &
              & td_dim(2)%i_xyzt2 == 2 .AND. &   
              & td_dim(3)%i_xyzt2 == 3 .AND. &   
              & td_dim(4)%i_xyzt2 == 4 )THEN     
   
               DO jl=1,td_dim(4)%i_len
                  DO jk=1,td_dim(3)%i_len
                     DO jj=1,td_dim(2)%i_len
                        DO ji=1,td_dim(1)%i_len
                           df_value(ji,jj,jk,jl)=dd_value(ji,jj,jk,jl)
                        ENDDO
                     ENDDO   
                  ENDDO
               ENDDO

            ELSE

               ! reorder dimension to x,y,z,t
               df_value(:,:,:,:)=RESHAPE(SOURCE=dd_value(:,:,:,:),&
                  &                 SHAPE = (/ td_dim(1)%i_len,   &
                  &                            td_dim(2)%i_len,   &
                  &                            td_dim(3)%i_len,   &
                  &                            td_dim(4)%i_len /),&
                  &                 ORDER = (/ td_dim(1)%i_2xyzt, &
                  &                            td_dim(2)%i_2xyzt, &
                  &                            td_dim(3)%i_2xyzt, &
                  &                            td_dim(4)%i_2xyzt /))      
            ENDIF
         ENDIF
      ENDIF

   END FUNCTION dim__reshape_2xyzt_dp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dim__reshape_xyzt2_dp(td_dim, dd_value) &
         & RESULT (df_value)
   !-------------------------------------------------------------------
   !> @brief This function reshape ordered real(8) 4D array with dimension 
   !> (/'x','y','z','t'/) to an "disordered" array.<br/>
   !> @details
   !> Example: (/'x','y','z','t'/) => (/'z','x','t','y'/)
   !
   !> @note you must have run dim_reorder before use this subroutine
   !
   !> @warning output array dimension differ from input array dimension
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[in] td_dim    array of dimension structure
   !> @param[in] dd_value  array of value to reshape
   !> @return array of value reshaped 
   !-------------------------------------------------------------------

      IMPLICIT NONE
      
      ! Argument      
      TYPE(TDIM), DIMENSION(:)      , INTENT(IN) :: td_dim
      REAL(dp),   DIMENSION(:,:,:,:), INTENT(IN) :: dd_value
      
      ! function
      REAL(dp), DIMENSION(td_dim(td_dim(1)%i_xyzt2)%i_len, &
         &                td_dim(td_dim(2)%i_xyzt2)%i_len, &
         &                td_dim(td_dim(3)%i_xyzt2)%i_len, &
         &                td_dim(td_dim(4)%i_xyzt2)%i_len) :: df_value

      ! local variable
      INTEGER(i4),      DIMENSION(ip_maxdim) :: il_shape
      CHARACTER(LEN=lc)                      :: cl_dim
      
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( SIZE(td_dim(:)) /= ip_maxdim )THEN
         CALL logger_error("DIM RESHAPE XYZT 2: invalid dimension of "//&
            &  "array dimension.")
      ELSE

         IF( ANY(td_dim(:)%i_xyzt2==0) .OR. ANY(td_dim(:)%i_2xyzt==0) )THEN

            CALL logger_fatal( &
            &  "  DIM RESHAPE XYZT 2: you should have run dim_reorder"// &
            &  "   before running RESHAPE" )

         ENDIF        

         ! check input dimension
         il_shape=SHAPE(dd_value)
         IF( ANY(il_shape(:)/=td_dim(:)%i_len))THEN

            DO ji=1,ip_maxdim
               CALL logger_trace(" DIM RESHAPE XYZT 2: dim "//&
               &              TRIM(td_dim(ji)%c_name)//" "//&
               &              TRIM(fct_str(td_dim(ji)%i_len))//" vs "//&
               &              TRIM(fct_str(il_shape(ji))) )
            ENDDO
            CALL logger_fatal( "DIM RESHAPE XYZT 2: wrong input dimensions ")

         ELSE      

            ! write some informations
            cl_dim="(/"
            DO ji=1,ip_maxdim-1
               cl_dim=TRIM(cl_dim)//TRIM(fct_str(il_shape(ji)))//','
            ENDDO
            cl_dim=TRIM(cl_dim)//TRIM(fct_str(il_shape(ip_maxdim)))//"/)"

            CALL logger_debug(" DIM RESHAPE XYZT 2: input dimensions are "//&
            &  TRIM(cl_dim) )

            cl_dim="(/"
            DO ji=1,ip_maxdim-1
               cl_dim=TRIM(cl_dim)//&
               &      TRIM(fct_str(td_dim(td_dim(ji)%i_xyzt2)%i_len))//','
            ENDDO
            cl_dim=TRIM(cl_dim)//&
            &      TRIM(fct_str(td_dim(td_dim(ip_maxdim)%i_xyzt2)%i_len))//"/)"

            CALL logger_debug(" DIM RESHAPE XYZT 2: ouput dimensions should be "//&
            &  TRIM(cl_dim) )

            ! reshape array
            df_value(:,:,:,:)=RESHAPE(SOURCE=dd_value,  &
               &           SHAPE = (/ td_dim(td_dim(1)%i_xyzt2)%i_len,   &
               &                      td_dim(td_dim(2)%i_xyzt2)%i_len,   &
               &                      td_dim(td_dim(3)%i_xyzt2)%i_len,   &
               &                      td_dim(td_dim(4)%i_xyzt2)%i_len /),&
               &           ORDER = (/        td_dim(1)%i_xyzt2,          &
               &                             td_dim(2)%i_xyzt2,          &
               &                             td_dim(3)%i_xyzt2,          &
               &                             td_dim(4)%i_xyzt2        /))

         ENDIF      
      ENDIF      

   END FUNCTION dim__reshape_xyzt2_dp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dim__reorder_2xyzt_i4(td_dim, id_arr) &
         & RESULT (if_value)
   !-------------------------------------------------------------------
   !> @brief  This function reordered integer(4) 1D array to be suitable 
   !> with dimension ordered as defined in dim_reorder.
   !> @note you must have run dim_reorder before use this subroutine
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[in] td_dim array of dimension structure
   !> @param[in] id_arr array of value to reshape
   !> @return array of value reshaped 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TDIM) , DIMENSION(:), INTENT(IN) :: td_dim
      INTEGER(i4), DIMENSION(:), INTENT(IN) :: id_arr
      
      ! function
      INTEGER(i4), DIMENSION(ip_maxdim)     :: if_value

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( SIZE(td_dim(:)) /= ip_maxdim .OR. &
      &   SIZE(id_arr(:)) /= ip_maxdim )THEN
         CALL logger_error("DIM REORDER 2 XYZT: invalid dimension of array dimension"//&
         &              " or of array of value.")
      ELSE      
         IF( ANY(td_dim(:)%i_2xyzt==0) )THEN

            CALL logger_error( &
            &  "  DIM REORDER 2 XYZT: you should have run dim_reorder"//&
            &  "   before running REORDER" )

         ENDIF        

         DO ji=1,ip_maxdim
            if_value(ji)=id_arr(td_dim(ji)%i_2xyzt)
         ENDDO
      ENDIF

   END FUNCTION dim__reorder_2xyzt_i4
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dim__reorder_xyzt2_i4(td_dim, id_arr) &
         & RESULT (if_value)
   !-------------------------------------------------------------------
   !> @brief This function disordered integer(4) 1D array to be suitable with
   !> initial dimension order (ex: dimension read in file).
   !> @note you must have run dim_reorder before use this subroutine
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[in] td_dim array of dimension structure
   !> @param[in] id_arr array of value to reshape
   !> @return array of value reshaped 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TDIM) , DIMENSION(:), INTENT(IN) :: td_dim
      INTEGER(i4), DIMENSION(:), INTENT(IN) :: id_arr
 
      ! function
      INTEGER(i4), DIMENSION(ip_maxdim)     :: if_value
 
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( SIZE(td_dim(:)) /= ip_maxdim .OR. &
      &   SIZE(id_arr(:)) /= ip_maxdim )THEN
         CALL logger_error("DIM REORDER XYZT 2: invalid dimension of "//&
            &  "array dimension or of array of value.")
      ELSE      
         IF( ANY(td_dim(:)%i_xyzt2==0) )THEN

            CALL logger_error( &
            &  "  DIM REORDER XYZT 2: you should have run dim_reorder"// &
            &  "   before running REORDER" )

         ENDIF        

         DO ji=1,ip_maxdim
            if_value(ji)=id_arr(td_dim(ji)%i_xyzt2)
         ENDDO
      ENDIF

   END FUNCTION dim__reorder_xyzt2_i4
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dim__reorder_2xyzt_l(td_dim, ld_arr) &
         & RESULT (lf_arr)
   !-------------------------------------------------------------------
   !> @brief  This function reordered logical 1D array to be suitable 
   !> with dimension ordered as defined in dim_reorder.
   !> @note you must have run dim_reorder before use this subroutine
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[in] td_dim array of dimension structure
   !> @param[in] ld_arr array of value to reordered
   !> @return array of value reordered 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TDIM) , DIMENSION(:), INTENT(IN) :: td_dim
      LOGICAL    , DIMENSION(:), INTENT(IN) :: ld_arr
      
      ! function
      LOGICAL, DIMENSION(ip_maxdim)         :: lf_arr

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( SIZE(td_dim(:)) /= ip_maxdim .OR. &
      &   SIZE(ld_arr(:)) /= ip_maxdim )THEN
         CALL logger_error("DIM REORDER 2 XYZT: invalid dimension of array dimension"//&
         &              " or of array of value.")
      ELSE      
         IF( ANY(td_dim(:)%i_2xyzt==0) )THEN

            CALL logger_error( &
            &  "  DIM REORDER 2 XYZT: you should have run dim_reorder"// &
            &  "   before running REORDER" )

         ENDIF        

         DO ji=1,ip_maxdim
            lf_arr(ji)=ld_arr(td_dim(ji)%i_2xyzt)
         ENDDO
      ENDIF

   END FUNCTION dim__reorder_2xyzt_l
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dim__reorder_xyzt2_l(td_dim, ld_arr) &
         & RESULT (lf_arr)
   !-------------------------------------------------------------------
   !> @brief This function disordered logical 1D array to be suitable with
   !> initial dimension order (ex: dimension read in file).
   !> @note you must have run dim_reorder before use this subroutine
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[in] td_dim array of dimension structure
   !> @param[in] ld_arr array of value to reordered
   !> @return array of value reordered 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TDIM) , DIMENSION(:), INTENT(IN) :: td_dim
      LOGICAL    , DIMENSION(:), INTENT(IN) :: ld_arr
 
      ! function
      LOGICAL, DIMENSION(ip_maxdim)         :: lf_arr
 
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( SIZE(td_dim(:)) /= ip_maxdim .OR. &
      &   SIZE(ld_arr(:)) /= ip_maxdim )THEN
         CALL logger_error("DIM REORDER XYZT 2: invalid dimension of array dimension"//&
         &              " or of array of value.")
      ELSE
         IF( ANY(td_dim(:)%i_xyzt2==0) )THEN

            CALL logger_error( &
            &  "  DIM REORDER XYZT 2: you should have run dim_reorder"//&
            &  "  before running REORDER" )

         ENDIF        

         DO ji=1,ip_maxdim
            lf_arr(ji)=ld_arr(td_dim(ji)%i_xyzt2)
         ENDDO
      ENDIF

   END FUNCTION dim__reorder_xyzt2_l
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dim__reorder_2xyzt_c(td_dim, cd_arr) &
         & RESULT (cf_arr)
   !-------------------------------------------------------------------
   !> @brief  This function reordered string 1D array to be suitable 
   !> with dimension ordered as defined in dim_reorder.
   !> @note you must have run dim_reorder before use this subroutine
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[in] td_dim array of dimension structure
   !> @param[in] cd_arr array of value to reordered
   !> @return array of value reordered 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TDIM),       DIMENSION(:), INTENT(IN) :: td_dim
      CHARACTER(LEN=*), DIMENSION(:), INTENT(IN) :: cd_arr
 
      ! function
      CHARACTER(LEN=lc), DIMENSION(ip_maxdim)    :: cf_arr

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( SIZE(td_dim(:)) /= ip_maxdim .OR. &
      &   SIZE(cd_arr(:)) /= ip_maxdim )THEN
         CALL logger_error("DIM REORDER 2 XYZT: invalid dimension of array dimension"//&
         &              " or of array of value.")
      ELSE      
         IF( ANY(td_dim(:)%i_2xyzt==0) )THEN

            CALL logger_error( &
            &  "  DIM REORDER 2 XYZT: you should have run dim_reorder"//&
            &  " before running REORDER" )

         ENDIF        

         DO ji=1,ip_maxdim
            cf_arr(ji)=TRIM(cd_arr(td_dim(ji)%i_2xyzt))
         ENDDO
      ENDIF

   END FUNCTION dim__reorder_2xyzt_c
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dim__reorder_xyzt2_c(td_dim, cd_arr) &
         & RESULT (cf_arr)
   !-------------------------------------------------------------------
   !> @brief This function disordered string 1D array to be suitable with
   !> initial dimension order (ex: dimension read in file).
   !> @note you must have run dim_reorder before use this subroutine
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[in] td_dim array of dimension structure
   !> @param[in] cd_arr array of value to reordered
   !> @return array of value reordered 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TDIM),       DIMENSION(:), INTENT(IN) :: td_dim
      CHARACTER(LEN=*), DIMENSION(:), INTENT(IN) :: cd_arr
 
      ! function
      CHARACTER(LEN=lc), DIMENSION(ip_maxdim) :: cf_arr
      
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( SIZE(td_dim(:)) /= ip_maxdim .OR. &
      &   SIZE(cd_arr(:)) /= ip_maxdim )THEN
         CALL logger_error("DIM REORDER XYZT 2: invalid dimension of array dimension"//&
         &              " or of array of value.")
      ELSE
         IF( ANY(td_dim(:)%i_xyzt2==0) )THEN
            CALL logger_error( &
            &  "  DIM REORDER XYZT 2: you should have run dim_reorder"// &
            &  "   before running REORDER" )

         ENDIF        

         DO ji=1,ip_maxdim
            cf_arr(ji)=TRIM(cd_arr(td_dim(ji)%i_xyzt2))
         ENDDO
      ENDIF

   END FUNCTION dim__reorder_xyzt2_c
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dim__clean_unit(td_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine clean dimension structure.
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[in] td_dim dimension strucutre
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDIM), INTENT(INOUT) :: td_dim

      ! local variable
      TYPE(TDIM) :: tl_dim ! empty dimension strucutre
      !----------------------------------------------------------------

      CALL logger_trace( &
      &  " DIM CLEAN: reset dimension "//TRIM(td_dim%c_name) )

      ! replace by empty structure
      td_dim=tl_dim

   END SUBROUTINE dim__clean_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dim__clean_arr(td_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine clean array of dimension structure
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[in] td_dim array of dimension strucutre
   !-------------------------------------------------------------------

      IMPLICIT NONE
      ! Argument
      TYPE(TDIM), DIMENSION(:), INTENT(INOUT) :: td_dim

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      DO ji=1,SIZE(td_dim(:))
         CALL dim_clean(td_dim(ji))
      ENDDO

   END SUBROUTINE dim__clean_arr
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dim_get_dummy(cd_dummy)
   !-------------------------------------------------------------------
   !> @brief This subroutine fill dummy dimension array
   !
   !> @author J.Paul
   !> @date September, 2015 - Initial Version
   !> @date May, 2019
   !> - read number of dummy element
   !>
   !> @param[in] cd_dummy dummy configuration file
   !-------------------------------------------------------------------

      IMPLICIT NONE
      ! Argument
      CHARACTER(LEN=*), INTENT(IN) :: cd_dummy

      ! local variable
      INTEGER(i4)   :: il_fileid
      INTEGER(i4)   :: il_status

      LOGICAL       :: ll_exist

      ! loop indices
      ! namelist
      INTEGER(i4)                                :: in_ndumvar
      INTEGER(i4)                                :: in_ndumdim
      INTEGER(i4)                                :: in_ndumatt
      CHARACTER(LEN=lc), DIMENSION(ip_maxdumcfg) :: cn_dumvar
      CHARACTER(LEN=lc), DIMENSION(ip_maxdumcfg) :: cn_dumdim
      CHARACTER(LEN=lc), DIMENSION(ip_maxdumcfg) :: cn_dumatt

      !----------------------------------------------------------------
      NAMELIST /namdum/ &   !< dummy namelist
      &  in_ndumvar,&       !< number of variable  name
      &  in_ndumdim,&       !< number of dimension name
      &  in_ndumatt,&       !< number of attribute name
      &  cn_dumvar, &       !< variable  name
      &  cn_dumdim, &       !< dimension name
      &  cn_dumatt          !< attribute name
      !----------------------------------------------------------------

      ! init
      cm_dumdim(:)=''

      ! read namelist
      INQUIRE(FILE=TRIM(cd_dummy), EXIST=ll_exist)
      IF( ll_exist )THEN

         il_fileid=fct_getunit()

         OPEN( il_fileid, FILE=TRIM(cd_dummy), &
         &                FORM='FORMATTED',       &
         &                ACCESS='SEQUENTIAL',    &
         &                STATUS='OLD',           &
         &                ACTION='READ',          &
         &                IOSTAT=il_status)
         CALL fct_err(il_status)
         IF( il_status /= 0 )THEN
            CALL logger_fatal("DIM GET DUMMY: opening "//TRIM(cd_dummy))
         ENDIF

         READ( il_fileid, NML = namdum )
         im_ndumdim  = in_ndumdim
         cm_dumdim(:)= cn_dumdim(:)

         CLOSE( il_fileid )

         IF( im_ndumdim > ip_maxdumcfg )THEN
            CALL logger_fatal("DIM GET dUMMY : too much dummy dimension &
            &     ( >"//fct_str(ip_maxdumcfg)//" ). &
            &     set ip_maxdumcfg to higher value.")
         ENDIF
      ENDIF

   END SUBROUTINE dim_get_dummy
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dim_is_dummy(td_dim) &
         & RESULT (lf_dummy)
   !-------------------------------------------------------------------
   !> @brief This function check if dimension is defined as dummy dimension
   !> in configuraton file
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial Version
   !> @date, May, 2019
   !> - use number of dummy elt in do-loop
   !>
   !> @param[in] td_dim dimension structure
   !> @return true if dimension is dummy dimension 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TDIM), INTENT(IN) :: td_dim
      
      ! function
      LOGICAL                :: lf_dummy
      
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      lf_dummy=.FALSE.
      DO ji=1,im_ndumdim
         IF( fct_lower(td_dim%c_name) == fct_lower(cm_dumdim(ji)) )THEN
            lf_dummy=.TRUE.
            EXIT
         ENDIF
      ENDDO

   END FUNCTION dim_is_dummy
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dim_def_extra(cd_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine read dimension configuration file, 
   !> and fill array of dimension allowed.
   !>
   !> @author J.Paul
   !> @date Ocotber, 2016 - Initial Version
   !> @date May, 2019
   !> - read number of element for each dimention 
   !
   !> @param[in] cd_file input file (dimension configuration file)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      CHARACTER(LEN=*), INTENT(IN) :: cd_file

      ! local variable
      INTEGER(i4)   :: il_fileid
      INTEGER(i4)   :: il_status

      LOGICAL       :: ll_exist
      
      ! loop indices
      ! namelist
      INTEGER(i4)                                :: in_dimX = 0
      INTEGER(i4)                                :: in_dimY = 0
      INTEGER(i4)                                :: in_dimZ = 0
      INTEGER(i4)                                :: in_dimT = 0
      CHARACTER(LEN=lc), DIMENSION(ip_maxdimcfg) :: cn_dimX = '' 
      CHARACTER(LEN=lc), DIMENSION(ip_maxdimcfg) :: cn_dimY = ''
      CHARACTER(LEN=lc), DIMENSION(ip_maxdimcfg) :: cn_dimZ = ''
      CHARACTER(LEN=lc), DIMENSION(ip_maxdimcfg) :: cn_dimT = ''

      !----------------------------------------------------------------
      NAMELIST /namdim/ &   !< dimension namelist
      &  in_dimX, &       !< number of x dimension name allowed
      &  in_dimY, &       !< number of y dimension name allowed
      &  in_dimZ, &       !< number of z dimension name allowed
      &  in_dimT, &       !< number of t dimension name allowed
      &  cn_dimX, &       !< x dimension name allowed
      &  cn_dimY, &       !< y dimension name allowed
      &  cn_dimZ, &       !< z dimension name allowed
      &  cn_dimT          !< t dimension name allowed

      !----------------------------------------------------------------

      ! init
      cm_dimX(:)=''
      cm_dimY(:)=''
      cm_dimZ(:)=''
      cm_dimT(:)=''

      ! read config variable file
      INQUIRE(FILE=TRIM(cd_file), EXIST=ll_exist)
      IF( ll_exist )THEN

         il_fileid=fct_getunit()
   
         OPEN( il_fileid, FILE=TRIM(cd_file), &
         &                FORM='FORMATTED',       &
         &                ACCESS='SEQUENTIAL',    &
         &                STATUS='OLD',           &
         &                ACTION='READ',          &
         &                IOSTAT=il_status)
         CALL fct_err(il_status)
         IF( il_status /= 0 )THEN
            CALL logger_fatal("DIM GET DUMMY: opening "//TRIM(cd_file))
         ENDIF
   
         READ( il_fileid, NML = namdim )
         im_dimX   =in_dimX
         im_dimY   =in_dimY
         im_dimZ   =in_dimZ
         im_dimT   =in_dimT
         cm_dimX(:)=cn_dimX(:)
         cm_dimY(:)=cn_dimY(:)
         cm_dimZ(:)=cn_dimZ(:)
         cm_dimT(:)=cn_dimT(:)

         CLOSE( il_fileid )

      ELSE

         CALL logger_fatal("DIM DEF EXTRA: can't find configuration"//&
            &              " file "//TRIM(cd_file))

      ENDIF         

   END SUBROUTINE dim_def_extra
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dim__is_allowed(cd_name, cd_dim, id_ndim) &
         & RESULT (lf_allowed)
   !-------------------------------------------------------------------
   !> @brief This function check if dimension is allowed, i.e defined 
   !> in dimension configuraton file
   !>
   !> @author J.Paul
   !> @date October, 2016 - Initial Version
   !> @date May, 2019
   !> - use number of element for each dimention allowed, instead of while loop 
   !
   !> @param[in] cd_name dimension name
   !> @param[in] cd_dim  array dimension name allowed
   !> @param[in] id_ndim number of elt in array dimension name allowed
   !> @return true if dimension is allowed 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),               INTENT(IN) :: cd_name
      CHARACTER(LEN=*), DIMENSION(:), INTENT(IN) :: cd_dim
      INTEGER(i4)     ,               INTENT(IN) :: id_ndim
      
      ! function
      LOGICAL                                    :: lf_allowed
      
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      lf_allowed=.FALSE.
      DO ji=1,id_ndim
         IF( TRIM(fct_lower(cd_name)) == TRIM(fct_lower(cd_dim(ji))) )THEN
            lf_allowed=.TRUE.
            EXIT
         ENDIF
      ENDDO

   END FUNCTION dim__is_allowed
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE dim

