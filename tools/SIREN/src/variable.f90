!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief 
!> This module manage variable structure.
!>
!> @details
!> to define type TVAR:<br/>
!> @code
!>    TYPE(TVAR) :: tl_var
!> @endcode
!>
!>    @note the variable value inside structure will always be 4D array of real(8).<br/>
!>    However the variable value could be initialised with
!>    array of real(4), real(8), integer(4) or integer(8).
!>        
!> to initialise a variable structure:<br/>
!> @code
!>    tl_var=var_init( cd_name, [value,] [id_start, [id_count,]] [id_type,] [td_dim,] [td_att]... )
!> @endcode
!>       - cd_name is the variable name
!>       - value is a 1D,2D,3D or 4D array, see var_init for more information [optional]
!>       - id_start is a integer(4) 1D array of index from which the data 
!>          values will be read [optional]
!>       - id_count is a integer(4) 1D array of the number of indices selected
!>          along each dimension [optional]
!>       - id_type is the type of the variable to be used [optional]
!>       - td_dim is the array of dimension structure [optional]
!>       - td_att is the array of attribute structure [optional]
!>    Note:<br/>
!>       - others optionals arguments could be added, see var_init.
!>       - to put scalar variable (OD), use td_dim with all dimension unused
!> (td_dim(:)%l_use=.FALSE.)
!>    
!>    to print information about variable structure:<br/>
!> @code
!>    CALL var_print(td_var [,ld_more])
!> @endcode
!>       - td_var is the variable structure
!>       - ld_more to print more information about variable
!>
!> to clean variable structure:<br/>
!> @code
!>    CALL var_clean(tl_var)
!> @endcode
!>
!> to copy variable structure in another one (using different memory cell):<br/>
!> @code
!>    tl_var2=var_copy(tl_var1) 
!> @endcode
!>    @note as we use pointer for the value array of the variable structure,
!>    the use of the assignment operator (=) to copy variable structure 
!>    create a pointer on the same array. 
!>    This is not the case with this copy function.
!>
!> to get variable name:<br/>
!>    - tl_var\%c_name
!>
!> to get grid point of the variable
!>    - tl_var\%c_point
!>
!> to get EW overlap:<br/>
!>    - tl_var\%i_ew
!>
!> to get variable value:<br/>
!>    - tl_var\%d_value(:,:,:,:)
!>    
!> to get the type number (based on NETCDF type constants) of the variable 
!>    (as define initially or read in file):<br/>
!>    - tl_var\%i_type
!>
!> to get variable id (read from a file):<br/>
!>    - tl_var\%i_id
!>
!> **Variable dimension**<br/>
!> to get the number of dimension used in the variable:<br/>
!>    - tl_var\%i_ndim
!>
!> to get the array of dimension structure (4 elts) associated to the
!> variable:<br/>
!>    - tl_var\%t_dim(:)
!>
!> **Variable attributes**<br/>
!> @note attribue value are always character or real(8) 1D array.<br/>
!>
!> to get the number of attributes of the variable:<br/>
!>    - tl_var\%i_natt
!>
!> to get the array of attribute structure associated to the
!> variable:<br/>
!>    - tl_var\%t_att(:)
!>
!> Some attribute are highlight, to be easily used.
!> to get variable standard name:<br/>
!>    - tl_var\%c_stdname
!>
!> to get variable longname:<br/>
!>    - tl_var\%c_longname
!>
!> to get variable units:<br/>
!>    - tl_var\%c_units
!>
!> to get variable axis:<br/>
!>    - tl_var\%c_axis
!>
!> to get variable scale factor:<br/>
!>    - tl_var\%d_scf
!>
!> to get variable add offset:<br/>
!>    - tl_var\%d_ofs
!>
!> to get variable FillValue:<br/>
!>    - tl_var\%d_fill
!>
!> to add value to a variable structure:<br/>
!> @code
!>    CALL var_add_value(tl_var, value, [id_type,] [id_start, [id_count]]) 
!> @endcode
!>       - value : 4D array of value (real(4), real(8), integer(1), integer(2), integer(4), integer(8))
!>       - id_type is the type of the variable to be used (default is the type
!> of array value)
!>       - id_start : 1D array of the index in the variable from which the data
!>       values will be read (integer(4), optional)
!>       - id_count : 1D array of the number of indices selected along each 
!>       dimension (integer(4), optional)
!>
!> to add attribute to a variable structure:<br/>
!> @code
!>    CALL var_add_att(tl_var, td_att)
!> @endcode
!>       - td_att is an attribute structure, or array of attribute structure 
!>
!> to add dimension to a variable structure:<br/>
!> @code
!>    CALL var_add_dim(tl_var, td_dim)
!> @endcode
!>       - td_dim is a dimension structure, or array of dimension structure
!> 
!> to delete value of a variable structure:<br/>
!> @code
!>    CALL var_del_value(tl_var)
!> @endcode
!>
!> to delete one attribute of a variable structure:<br/>
!> @code
!>    CALL var_del_att(tl_var, td_att)
!> @endcode
!>       - td_att is an attribute structure 
!> or
!> @code
!>    CALL var_del_att(tl_var, cd_name)
!> @endcode
!>       - cd_name is attribute name 
!>
!> to delete one dimension of a variable structure:<br/>
!> @code
!>    CALL var_del_dim(tl_var, td_dim)
!> @endcode
!>       - td_dim is a dimension structure
!>
!> to overwrite one attribute structure in variable structure:<br/>
!> @code
!>    CALL var_move_att(tl_var, td_att)
!> @endcode
!>       - td_att is an attribute structure
!>
!> to overwrite one dimension structure in variable structure:<br/>
!> @code
!>    CALL var_move_dim(tl_var, td_dim)
!> @endcode
!>       - td_dim is a dimension structure
!>
!> to get the mask of a variable strucutre, (based on its FillValue):<br/>
!> @code
!>    mask(:,:)=var_get_mask(tl_var)
!> @endcode
!>
!> to change FillValue to standard NETCDF Fill Value:<br/>
!> @code
!>    CALL  var_chg_FillValue(tl_var, [dd_fill])
!> @endcode
!>       - dd_fill is the FillValue to be used [optional]
!>
!> to concatenate two variables:<br/>
!> @code
!>    tl_var=var_concat(tl_var1, tl_var2, [DIM])
!> @endcode
!>       - tl_var1 : variable structure 
!>       - tl_var2 : variable structure 
!>       - DIM : number of the dimension following which concatenate (1=>I, 2=>J, 3=>Z, 4=>T) [optional, default=4]
!>
!> to forced min and max value of a variable:<br/>
!>    - define min and max value of the variable:<br/>
!>       - tl_var\%d_min=min<br/>
!>       - tl_var\%d_max=max<br/>
!>          - min and max : real(8) value
!>    - then <br/>
!> @code
!>    CALL  var_limit_value( tl_var )
!> @endcode
!>
!> to get the biggest dimensions use in a array of variable:<br/>
!> @code
!>    tl_dim(:)=var_max_dim(tl_var(:))
!> @endcode
!>       - tl_var(:) : array of variable structure
!>       - tl_dim(:) : array (4 elts) of dimension structure
!>
!> to reorder dimension of a variable (default 'x','y','z','t'):<br/>
!> @code
!>    CALL var_reorder( td_var, cd_dimorder )
!> @endcode
!>       - td_var is variable structure
!>       - cd_dimorder string character(LEN=4) of dimension order to be used (example:
!> 'yxzt') [optional]
!>
!> to get variable index, in an array of variable structure:<br/>
!> @code
!>   il_index=var_get_index( td_var, cd_name )
!> @endcode
!>    - td_var array of variable structure
!>    - cd_name variable name
!> 
!> to get variable id, read from a file:<br/>
!> @code
!>  il_id=var_get_id( td_var, cd_name )
!> @endcode
!>    - td_var array of variable structure
!>    - cd_name variable name
!>
!> to get free variable unit in an array of variable structure:<br/>
!> @code
!>  il_unit=var_get_unit(td_var)
!> @endcode
!>    - td_var array of variable structure
!>
!> to convert time variable structure in date structure:<br/>
!> @code
!>   tl_date=var_to_date(td_var) 
!> @endcode
!>    - td_var is time variable structure
!>    - tl_date is date structure
!>
!> to read matrix value from character string in namelist
!> @code
!>    CALL var_read_matrix(td_var, cd_matrix)
!> @endcode
!>    - td_var is variable structure
!>    - cd_matrix is matrix value
!>
!> to read variable configuration file ('variable.cfg') and fill global array
!> of variable structure:<br/>
!> @code
!>    CALL var_def_extra( cd_file )
!> @endcode
!>    - cd_file is filename
!>
!> to add variable information get from namelist, in global array of variable 
!> structure:
!> @code
!>    CALL var_chg_extra( cd_varinfo )
!> @endcode
!>    - cd_varinfo is variable information from namelist
!>
!> to clean global array of variable structure:<br/>
!> @code
!>    CALL var_clean_extra( )
!> @endcode
!>
!> to check variable dimension expected, as defined in file 'variable.cfg':<br/>
!> @code
!>    CALL var_check_dim( td_var )
!> @endcode
!>    - td_var is variable structure
!> 
!> @author
!> J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date September, 2014
!>  - add var_reorder
!> @date November, 2014 
!> - Fix memory leaks bug
!> @date June, 2015
!> - change way to get variable information in namelist
!> @date July, 2015 
!> - add subroutine var_chg_unit to change unit of output variable
!> @date Spetember, 2015
!> - manage useless (dummy) variable
!> @date October, 2016
!> - add subroutine to clean global array of extra information.
!> - define logical for variable to be used
!> @date May, 2019
!> - read number of element for each dummy array in configuration file
!>
!> @todo
!> - var_copy_value qui copie le tableau de valeur mais verifie que tous les
!> attribut sont egaux
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE var

   USE netcdf                          ! nf90 library
   USE global                          ! global variable
   USE kind                            ! F90 kind parameter
   USE logger                          ! log file manager
   USE date                            ! date manager
   USE fct                             ! basic useful function
   USE att                             ! attribute manager
   USE dim                             ! dimension manager
   USE math                            ! mathematical function

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! type and variable
   PUBLIC :: TVAR        !< variable structure

   PUBLIC :: tg_varextra !< array of variable structure with extra information.

   PRIVATE :: im_ndumvar !< number of elt in dummy variable array
   PRIVATE :: cm_dumvar  !< dummy variable array

   ! function and subroutine
   PUBLIC :: var_init          !< initialize variable structure
   PUBLIC :: var_print         !< print variable information
   PUBLIC :: var_clean         !< clean variable structure
   PUBLIC :: var_copy          !< copy variable structure
   PUBLIC :: var_add_value     !< add array of value in variable structure
   PUBLIC :: var_add_att       !< add attribute structure in variable structure
   PUBLIC :: var_add_dim       !< add dimension structure in variable structure
   PUBLIC :: var_del_value     !< delete array of value of variable structure
   PUBLIC :: var_del_att       !< delete one attribute structure of variable structure
   PUBLIC :: var_del_dim       !< delete one dimension structure of variable structure
   PUBLIC :: var_move_att      !< overwrite one attribute structure in variable structure
   PUBLIC :: var_move_dim      !< overwrite one dimension structure in variable structure 
   PUBLIC :: var_get_mask      !< return the mask of variable
   PUBLIC :: var_chg_FillValue !< change FillValue to standard NETCDF Fill Value
   PUBLIC :: var_concat        !< concatenate two variables
   PUBLIC :: var_limit_value   !< forced min and max value
   PUBLIC :: var_chg_unit      !< change variable unit and value
   PUBLIC :: var_chg_name      !< change variable name
   PUBLIC :: var_max_dim       !< get array of maximum dimension use
   PUBLIC :: var_reorder       !< reorder table of value in variable structure
   PUBLIC :: var_get_index     !< return the variable index, in an array of variable structure 
   PUBLIC :: var_get_id        !< return the variable id, read from a file 
   PUBLIC :: var_get_unit      !< get free variable unit in an array of variable structure
   PUBLIC :: var_to_date       !< convert time variable structure in date structure
   PUBLIC :: var_read_matrix   !< read matrix value from character string in namelist 
   PUBLIC :: var_def_extra     !< read variable configuration file, and save extra information.
   PUBLIC :: var_chg_extra     !< read variable namelist information, and modify extra information.
   PUBLIC :: var_clean_extra   !< clean gloabl array of extra information.
   PUBLIC :: var_check_dim     !< check variable dimension expected
   PUBLIC :: var_get_dummy     !< fill dummy variable array
   PUBLIC :: var_is_dummy      !< check if variable is defined as dummy variable

   PRIVATE :: var__init          ! initialize variable structure without array of value
   PRIVATE :: var__init_dp       ! initialize variable structure with real(8) 4D array of value
   PRIVATE :: var__init_1D_dp    ! initialize variable structure with real(8) 1D array of value
   PRIVATE :: var__init_2D_dp    ! initialize variable structure with real(8) 2D array of value
   PRIVATE :: var__init_3D_dp    ! initialize variable structure with real(8) 3D array of value
   PRIVATE :: var__init_sp       ! initialize variable structure with real(4) 4D array of value
   PRIVATE :: var__init_1D_sp    ! initialize variable structure with real(4) 1D array of value
   PRIVATE :: var__init_2D_sp    ! initialize variable structure with real(4) 2D array of value
   PRIVATE :: var__init_3D_sp    ! initialize variable structure with real(4) 3D array of value
   PRIVATE :: var__init_i1       ! initialize variable structure with integer(1) 4D array of value
   PRIVATE :: var__init_1D_i1    ! initialize variable structure with integer(1) 1D array of value
   PRIVATE :: var__init_2D_i1    ! initialize variable structure with integer(1) 2D array of value
   PRIVATE :: var__init_3D_i1    ! initialize variable structure with integer(1) 3D array of value
   PRIVATE :: var__init_i2       ! initialize variable structure with integer(2) 4D array of value
   PRIVATE :: var__init_1D_i2    ! initialize variable structure with integer(2) 1D array of value
   PRIVATE :: var__init_2D_i2    ! initialize variable structure with integer(2) 2D array of value
   PRIVATE :: var__init_3D_i2    ! initialize variable structure with integer(2) 3D array of value
   PRIVATE :: var__init_i4       ! initialize variable structure with integer(4) 4D array of value
   PRIVATE :: var__init_1D_i4    ! initialize variable structure with integer(4) 1D array of value
   PRIVATE :: var__init_2D_i4    ! initialize variable structure with integer(4) 2D array of value
   PRIVATE :: var__init_3D_i4    ! initialize variable structure with integer(4) 3D array of value
   PRIVATE :: var__init_i8       ! initialize variable structure with integer(8) 4D array of value
   PRIVATE :: var__init_1D_i8    ! initialize variable structure with integer(8) 1D array of value
   PRIVATE :: var__init_2D_i8    ! initialize variable structure with integer(8) 2D array of value
   PRIVATE :: var__init_3D_i8    ! initialize variable structure with integer(8) 3D array of value
   PRIVATE :: var__print_unit    ! print information on one variable
   PRIVATE :: var__print_arr     ! print information on a array of variables
   PRIVATE :: var__clean_unit    ! clean variable structure
   PRIVATE :: var__clean_arr_1D  ! clean 1D array of variable structure
   PRIVATE :: var__clean_arr_2D  ! clean 2D array of variable structure
   PRIVATE :: var__clean_arr_3D  ! clean 3D array of variable structure
   PRIVATE :: var__add_value_dp  ! add array of value real(8) in variable structure
   PRIVATE :: var__add_value_rp  ! add array of value real(4) in variable structure
   PRIVATE :: var__add_value_i1  ! add array of value integer(1) in variable structure
   PRIVATE :: var__add_value_i2  ! add array of value integer(2) in variable structure
   PRIVATE :: var__add_value_i4  ! add array of value integer(4) in variable structure
   PRIVATE :: var__add_value_i8  ! add array of value integer(8) in variable structure
   PRIVATE :: var__add_att_unit  ! add one attribute structure in variable structure
   PRIVATE :: var__add_att_arr   ! add a array of attribute structure in variable structure
   PRIVATE :: var__del_att_name  ! delete one attribute given attribute name
   PRIVATE :: var__del_att_str   ! delete one attribute given attribute structure
   PRIVATE :: var__add_dim_unit  ! add one dimension structure in variable structure
   PRIVATE :: var__add_dim_arr   ! add a array of dimension structure in variable structure
   PRIVATE :: var__add_value     ! add a 4D array of real(8) value in a variable structure.
   PRIVATE :: var__copy_unit     ! copy variable structure
   PRIVATE :: var__copy_arr      ! copy a array of variable structure
   PRIVATE :: var__get_extra     ! add extra information in variable structure
   PRIVATE :: var__concat_i      ! concatenate varibales in i-direction
   PRIVATE :: var__concat_j      ! concatenate varibales in j-direction
   PRIVATE :: var__concat_k      ! concatenate varibales in k-direction
   PRIVATE :: var__concat_l      ! concatenate varibales in l-direction
   PRIVATE :: var__get_max       ! get maximum value from namelist 
   PRIVATE :: var__get_min       ! get minimum value from namelist
   PRIVATE :: var__get_unf       ! get scale factor value from namelist
   PRIVATE :: var__get_unt       ! get output unit from namelist
   PRIVATE :: var__get_namout    ! get output variable name from namelist
   PRIVATE :: var__get_interp    ! get interpolation method from namelist
   PRIVATE :: var__get_extrap    ! get extrapolation method from namelist
   PRIVATE :: var__get_filter    ! get filter method from namelist

   TYPE TVAR   !< variable structure

      CHARACTER(LEN=lc) :: c_name  = ''  !< variable name
      CHARACTER(LEN=lc) :: c_point = 'T' !< ARAKAWA C-grid point name (T,U,V,F)
      INTEGER(i4)       :: i_id = 0      !< variable id
      INTEGER(i4)       :: i_ew = -1     !< east-west overlap

      REAL(dp)   , DIMENSION(:,:,:,:), POINTER :: d_value => NULL() !< variable value

      !!! netcdf
      INTEGER(i4)       :: i_type = 0           !< variable type
      INTEGER(i4)       :: i_natt = 0           !< number of attributes
      INTEGER(i4)       :: i_ndim = 0           !< number of dimensions
      TYPE(TATT), DIMENSION(:), POINTER :: t_att => NULL() !< variable attributes
      TYPE(TDIM), DIMENSION(ip_maxdim)  :: t_dim           !< variable dimension
 
      LOGICAL           :: l_file = .FALSE. !< variable read in a file
      LOGICAL           :: l_use  = .TRUE.  !< variable to be used

      ! highlight some attributes
      CHARACTER(LEN=lc) :: c_stdname  = ''!< variable standard name
      CHARACTER(LEN=lc) :: c_longname = ''!< variable long name
      CHARACTER(LEN=lc) :: c_units    = ''!< variable units
      CHARACTER(LEN=lc) :: c_axis     = ''!< variable axis
      REAL(dp)          :: d_scf = 1.           !< scale factor
      REAL(dp)          :: d_ofs = 0.           !< offset
      REAL(dp)          :: d_fill= 0.           !< fill value     ! NF90_FILL_DOUBLE 
      REAL(dp)          :: d_min = dp_fill      !< minimum value 
      REAL(dp)          :: d_max = dp_fill      !< maximum value 
 
      ! will be changed in output
      CHARACTER(LEN=lc) :: c_unt = ''           !< output variable unit (linked to unit factor)
      REAL(dp)          :: d_unf = 1._dp        !< unit factor

      CHARACTER(LEN=lc) :: c_namout = ''        !< output variable name (renamed variable)

      !!! netcdf4
      LOGICAL           :: l_contiguous = .FALSE. !< use contiguous storage or not
      LOGICAL           :: l_shuffle    = .FALSE. !< shuffle filter is turned on or not
      LOGICAL           :: l_fletcher32 = .FALSE. !< fletcher32 filter is turned on or not
      INTEGER(i4)       :: i_deflvl = 0           !< deflate level from 0 to 9, 0 indicates no deflation is in use
      INTEGER(i4), DIMENSION(ip_maxdim) :: i_chunksz = (/1,1,1,1/) !< chunk size

      !!! dimg
      INTEGER(i4) :: i_rec  = 0 !< record number

      CHARACTER(LEN=lc), DIMENSION(2) :: c_interp = '' !< interpolation method
      CHARACTER(LEN=lc), DIMENSION(1) :: c_extrap = '' !< extrapolation method
      CHARACTER(LEN=lc), DIMENSION(5) :: c_filter = '' !< filter method

   END TYPE TVAR

   TYPE(TVAR), DIMENSION(:), ALLOCATABLE :: tg_varextra !< array of variable structure with extra information.
                                                        !< fill when running var_def_extra() 

   INTEGER(i4)                               , SAVE :: im_ndumvar !< number of elt in dummy variable array
   CHARACTER(LEN=lc), DIMENSION(ip_maxdumcfg), SAVE :: cm_dumvar  !< dummy variable

   INTERFACE var_init
      MODULE PROCEDURE var__init       ! initialize variable structure without array of value
      MODULE PROCEDURE var__init_dp    ! initialize variable structure with real(8) 4D array of value
      MODULE PROCEDURE var__init_1D_dp ! initialize variable structure with real(8) 1D array of value
      MODULE PROCEDURE var__init_2D_dp ! initialize variable structure with real(8) 2D array of value
      MODULE PROCEDURE var__init_3D_dp ! initialize variable structure with real(8) 3D array of value
      MODULE PROCEDURE var__init_sp    ! initialize variable structure with real(4) 4D array of value
      MODULE PROCEDURE var__init_1D_sp ! initialize variable structure with real(4) 1D array of value
      MODULE PROCEDURE var__init_2D_sp ! initialize variable structure with real(4) 2D array of value
      MODULE PROCEDURE var__init_3D_sp ! initialize variable structure with real(4) 3D array of value
      MODULE PROCEDURE var__init_i1    ! initialize variable structure with integer(1) 4D array of value
      MODULE PROCEDURE var__init_1D_i1 ! initialize variable structure with integer(1) 1D array of value
      MODULE PROCEDURE var__init_2D_i1 ! initialize variable structure with integer(1) 2D array of value
      MODULE PROCEDURE var__init_3D_i1 ! initialize variable structure with integer(1) 3D array of value
      MODULE PROCEDURE var__init_i2    ! initialize variable structure with integer(2) 4D array of value
      MODULE PROCEDURE var__init_1D_i2 ! initialize variable structure with integer(2) 1D array of value
      MODULE PROCEDURE var__init_2D_i2 ! initialize variable structure with integer(2) 2D array of value
      MODULE PROCEDURE var__init_3D_i2 ! initialize variable structure with integer(2) 3D array of value
      MODULE PROCEDURE var__init_i4    ! initialize variable structure with integer(4) 4D array of value
      MODULE PROCEDURE var__init_1D_i4 ! initialize variable structure with integer(4) 1D array of value
      MODULE PROCEDURE var__init_2D_i4 ! initialize variable structure with integer(4) 2D array of value
      MODULE PROCEDURE var__init_3D_i4 ! initialize variable structure with integer(4) 3D array of value
      MODULE PROCEDURE var__init_i8    ! initialize variable structure with integer(8) 4D array of value
      MODULE PROCEDURE var__init_1D_i8 ! initialize variable structure with integer(8) 1D array of value
      MODULE PROCEDURE var__init_2D_i8 ! initialize variable structure with integer(8) 2D array of value
      MODULE PROCEDURE var__init_3D_i8 ! initialize variable structure with integer(8) 3D array of value
   END INTERFACE var_init

   INTERFACE var_print
      MODULE PROCEDURE var__print_unit ! print information on one variable
      MODULE PROCEDURE var__print_arr  ! print information on a array of variables
   END INTERFACE var_print
   
   INTERFACE var_clean
      MODULE PROCEDURE var__clean_unit   
      MODULE PROCEDURE var__clean_arr_1D    
      MODULE PROCEDURE var__clean_arr_2D    
      MODULE PROCEDURE var__clean_arr_3D    
   END INTERFACE

   INTERFACE var_add_value
      MODULE PROCEDURE var__add_value_dp  ! add array of value real(8) in variable structure
      MODULE PROCEDURE var__add_value_rp  ! add array of value real(4) in variable structure
      MODULE PROCEDURE var__add_value_i1  ! add array of value integer(1) in variable structure
      MODULE PROCEDURE var__add_value_i2  ! add array of value integer(2) in variable structure
      MODULE PROCEDURE var__add_value_i4  ! add array of value integer(4) in variable structure
      MODULE PROCEDURE var__add_value_i8  ! add array of value integer(8) in variable structure
   END INTERFACE var_add_value

   INTERFACE var_add_att
      MODULE PROCEDURE var__add_att_unit ! add one attribute structure in variable structure
      MODULE PROCEDURE var__add_att_arr  ! add a array of attribute structure in variable structure
   END INTERFACE var_add_att

   INTERFACE var_del_att               ! delete one attribute in variable structure
      MODULE PROCEDURE var__del_att_name ! - given attribute name
      MODULE PROCEDURE var__del_att_str  ! - given attribute structure
   END INTERFACE var_del_att

   INTERFACE var_add_dim
      MODULE PROCEDURE var__add_dim_unit ! add one dimension structure in variable structure
      MODULE PROCEDURE var__add_dim_arr  ! add a array of dimension structure in variable structure
   END INTERFACE var_add_dim

   INTERFACE var_copy
      MODULE PROCEDURE var__copy_unit   ! copy variable structure
      MODULE PROCEDURE var__copy_arr    ! copy variable structure
   END INTERFACE
CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__copy_unit(td_var, ld_value) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine copy variable structure in another one
   !> @details 
   !> variable values are copied in a transitional variable, so input and output
   !> variable structure values do not point on the same "memory cell", and so
   !> are independant. 
   !>
   !> @warning do not use on the output of a function who create or read an
   !> structure (ex: tl_var=var_copy(var_init()) is forbidden).
   !> This will create memory leaks.
   !> @warning to avoid infinite loop, do not use any function inside 
   !> this subroutine
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date November, 2014
   !> - use function instead of overload assignment operator (to avoid memory leak)
   !> @date July, 2017
   !> - permit to copy variable structure without value
   !> @date January, 2019
   !> - use scalar instead of array, as transitional variable
   !> @date February, 2019
   !> - copy namout
   !>
   !> @param[in] td_var   variable structure
   !> @param[in] ld_value copy variable value (default .TRUE.)
   !> @return copy of input variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), INTENT(IN) :: td_var
      LOGICAL   , INTENT(IN), OPTIONAL :: ld_value

      ! function
      TYPE(TVAR)             :: tf_var

      ! local variable
      TYPE(TATT) :: tl_att
      REAL(dp)   :: dl_value
      LOGICAL    :: ll_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      ll_value=.TRUE.
      IF( PRESENT(ld_value) ) ll_value=ld_value
      ! copy variable name, id, ..
      tf_var%c_name     = TRIM(td_var%c_name)
      tf_var%c_point    = TRIM(td_var%c_point)
      tf_var%i_id       = td_var%i_id
      tf_var%i_ew       = td_var%i_ew

      tf_var%d_min      = td_var%d_min
      tf_var%d_max      = td_var%d_max

      tf_var%c_unt      = TRIM(td_var%c_unt)
      tf_var%d_unf      = td_var%d_unf

      tf_var%c_namout   = TRIM(td_var%c_namout)

      tf_var%i_type     = td_var%i_type
      tf_var%i_natt     = td_var%i_natt
      tf_var%i_ndim     = td_var%i_ndim
      tf_var%i_ndim     = td_var%i_ndim

      ! copy dimension
      tf_var%t_dim(:)   = dim_copy(td_var%t_dim(:))

      ! copy attribute
      IF( ASSOCIATED(tf_var%t_att) )THEN
         CALL att_clean( tf_var%t_att(:) )
         DEALLOCATE(tf_var%t_att)
      ENDIF
      IF( ASSOCIATED(td_var%t_att) .AND. tf_var%i_natt > 0 )THEN
         ALLOCATE( tf_var%t_att(tf_var%i_natt) )
         DO ji=1,tf_var%i_natt
            tl_att=att_copy(td_var%t_att(ji))
            tf_var%t_att(ji)=att_copy(tl_att)
         ENDDO
         ! clean
         CALL att_clean(tl_att)
      ENDIF

      tf_var%l_file     = td_var%l_file
      tf_var%l_use      = td_var%l_use

      ! copy highlight attribute
      tf_var%c_stdname  = TRIM(td_var%c_stdname)
      tf_var%c_longname = TRIM(td_var%c_longname)
      tf_var%c_units    = TRIM(td_var%c_units)
      tf_var%c_axis     = TRIM(td_var%c_axis)
      tf_var%d_unf      = td_var%d_unf
      tf_var%d_scf      = td_var%d_scf
      tf_var%d_ofs      = td_var%d_ofs
      tf_var%d_fill     = td_var%d_fill
      
      ! copy netcdf4 variable
      tf_var%l_contiguous  = td_var%l_contiguous
      tf_var%l_shuffle     = td_var%l_shuffle
      tf_var%l_fletcher32  = td_var%l_fletcher32 
      tf_var%i_deflvl      = td_var%i_deflvl
      tf_var%i_chunksz(:)  = td_var%i_chunksz(:)
      
      ! copy dimg variable
      tf_var%i_rec = td_var%i_rec

      ! copy pointer in an independant variable
      IF( ASSOCIATED(tf_var%d_value) ) DEALLOCATE(tf_var%d_value)
      IF( ll_value .AND. ASSOCIATED(td_var%d_value) )THEN
         ALLOCATE( tf_var%d_value( tf_var%t_dim(1)%i_len, &
            &                      tf_var%t_dim(2)%i_len, &
            &                      tf_var%t_dim(3)%i_len, &
            &                      tf_var%t_dim(4)%i_len ) )
         DO jl=1,td_var%t_dim(4)%i_len
            DO jk=1,td_var%t_dim(3)%i_len
               DO jj=1,td_var%t_dim(2)%i_len
                  DO ji=1,td_var%t_dim(1)%i_len
                     dl_value=td_var%d_value(ji,jj,jk,jl)
                     tf_var%d_value(ji,jj,jk,jl)=dl_value
                  ENDDO
               ENDDO
            ENDDO
         ENDDO

      ENDIF

      tf_var%c_interp(:)=td_var%c_interp(:)
      tf_var%c_extrap(:)=td_var%c_extrap(:)
      tf_var%c_filter(:)=td_var%c_filter(:)

   END FUNCTION var__copy_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__copy_arr(td_var) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine copy a array of variable structure in another one
   !> @details 
   !> see var__copy_unit
   !>
   !> @warning do not use on the output of a function who create or read an
   !> structure (ex: tl_var=var_copy(var_init()) is forbidden).
   !> This will create memory leaks.
   !> @warning to avoid infinite loop, do not use any function inside 
   !> this subroutine
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date November, 2014
   !> - use function instead of overload assignment operator 
   !> (to avoid memory leak)
   !>
   !> @param[in] td_var   array of variable structure
   !> @return copy of input array of variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), DIMENSION(:), INTENT(IN   ) :: td_var

      ! function
      TYPE(TVAR), DIMENSION(SIZE(td_var(:)))  :: tf_var

      ! local variable
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      DO ji=1,SIZE(td_var(:))
         tf_var(ji)=var_copy(td_var(ji))
      ENDDO

   END FUNCTION var__copy_arr
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__clean_unit(td_var)
   !-------------------------------------------------------------------
   !> @brief This subroutine clean variable structure
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - nullify attributes structure inside variable strcuture
   !>
   !> @param[inout] td_var variable strucutre
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), INTENT(INOUT) :: td_var

      ! local variable
      TYPE(TVAR) :: tl_var ! empty variable strucutre

      ! loop indices
      !----------------------------------------------------------------

      ! del attribute
      IF( ASSOCIATED(td_var%t_att) )THEN
         CALL att_clean( td_var%t_att(:) )
         DEALLOCATE(td_var%t_att)
         NULLIFY(td_var%t_att)
      ENDIF

      ! del dimension
      IF( td_var%i_ndim /= 0 )THEN
         CALL dim_clean(td_var%t_dim(:))
      ENDIF

      ! del value
      IF( ASSOCIATED(td_var%d_value) )THEN
         CALL var_del_value(td_var)
      ENDIF

      ! replace by empty structure
      td_var=var_copy(tl_var)

   END SUBROUTINE var__clean_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__clean_arr_1D(td_var)
   !-------------------------------------------------------------------
   !> @brief This subroutine clean 1D array of variable structure
   !>
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[inout] td_var array of variable strucutre
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), DIMENSION(:), INTENT(INOUT) :: td_var

      ! local variable
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      DO ji=SIZE(td_var(:)),1,-1
         CALL var_clean(td_var(ji))
      ENDDO

   END SUBROUTINE var__clean_arr_1D
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__clean_arr_2D(td_var)
   !-------------------------------------------------------------------
   !> @brief This subroutine clean 2D array of variable structure
   !>
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[inout] td_var array of variable strucutre
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), DIMENSION(:,:), INTENT(INOUT) :: td_var

      ! local variable
      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      DO jj=SIZE(td_var(:,:),DIM=2),1,-1
         DO ji=SIZE(td_var(:,:),DIM=1),1,-1
            CALL var_clean(td_var(ji,jj))
         ENDDO
      ENDDO

   END SUBROUTINE var__clean_arr_2D
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__clean_arr_3D(td_var)
   !-------------------------------------------------------------------
   !> @brief This subroutine clean 3D array of variable structure
   !>
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[inout] td_var array of variable strucutre
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), DIMENSION(:,:,:), INTENT(INOUT) :: td_var

      ! local variable
      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      DO jk=SIZE(td_var(:,:,:),DIM=3),1,-1
         DO jj=SIZE(td_var(:,:,:),DIM=2),1,-1
            DO ji=SIZE(td_var(:,:,:),DIM=1),1,-1
               CALL var_clean(td_var(ji,jj,jk))
            ENDDO
         ENDDO
      ENDDO

   END SUBROUTINE var__clean_arr_3D
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init(cd_name, id_type, td_dim,             &
         &            td_att, dd_fill, cd_units, cd_axis,   &
         &            cd_stdname, cd_longname,              &
         &            cd_point, id_id, id_ew,               &
         &            dd_scf, dd_ofs,  id_rec,              &
         &            dd_min, dd_max,                       &
         &            ld_contiguous, ld_shuffle,            &
         &            ld_fletcher32, id_deflvl, id_chunksz, &
         &            cd_interp, cd_extrap, cd_filter,      &
         &            cd_unt, dd_unf,                       &
         &            cd_namout) &
         &  RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure, given variable name.
   !>
   !> @details
   !> Optionally you could add 1D,2D,3D or 4D array of value, 
   !> see var__init_1D_dp, var__init_2D_dp... for more information.
   !>
   !> you could also add more information with the following optional arguments:
   !>   - id_type :  integer(4) variable type, (as defined by NETCDF type constants).
   !>   - td_dim : array of dimension structure.
   !>   - td_att  : array of attribute structure.
   !>   - dd_fill : real(8) variable FillValue. if none NETCDF FillValue will be used.
   !>   - cd_units : string character of units.
   !>   - cd_axis : string character of axis expected to be used
   !>   - cd_stdname : string character of variable standard name.
   !>   - cd_longname : string character of variable long name.
   !>   - cd_point : one character for ARAKAWA C-grid point name (T,U,V,F).
   !>   - id_id : variable id (read from a file).
   !>   - id_ew : number of point composing east west wrap band.
   !>   - dd_unf : real(8) value for units factor attribute.
   !>   - dd_scf : real(8) value for scale factor attribute.
   !>   - dd_ofs : real(8) value for add offset attribute.
   !>   - id_rec : record id (for rstdimg file).
   !>   - dd_min : real(8) value for minimum value.
   !>   - dd_max : real(8) value for maximum value.
   !>   - ld_contiguous : use contiguous storage or not (for netcdf4).
   !>   - ld_shuffle :  shuffle filter is turned on or not (for netcdf4).
   !>   - ld_fletcher32 : fletcher32 filter is turned on or not (for netcdf4).
   !>   - id_deflvl : deflate level from 0 to 9, 0 indicates no deflation is in use (for netcdf4).
   !>   - id_chunksz : chunk size (for netcdf4).
   !>   - cd_interp  : a array of character defining interpolation method.
   !>   - cd_extrap  : a array of character defining extrapolation method.
   !>   - cd_filter  : a array of character defining filtering method.
   !>   - cd_unt : a string character to define output unit
   !>   - dd_unf : real(8) factor applied to change unit
   !>
   !>  @note most of these optionals arguments will be inform automatically,
   !>  when reading variable from a file, or using confiuguration file variable.cfg.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date February, 2015 
   !> - Bug fix: conversion of the FillValue type (float case)
   !> @date June, 2015 
   !> - add unit factor (to change unit)
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] dd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle      shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          output unit (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_att
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4) :: il_ind

      TYPE(TATT)  :: tl_att

      ! loop indices
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)

      tf_var%c_name=TRIM(ADJUSTL(cd_name))

      ! standard name
      IF( PRESENT(cd_stdname) )THEN
         tf_var%c_stdname=TRIM(ADJUSTL(cd_stdname))
      ENDIF

      ! long name
      IF( PRESENT(cd_longname) )THEN
         tf_var%c_longname=TRIM(ADJUSTL(cd_longname))
      ENDIF

      ! point
      IF( PRESENT(cd_point) )THEN
         tf_var%c_point=TRIM(ADJUSTL(cd_point))
      ENDIF

      ! variable id
      IF( PRESENT(id_id) )THEN
         tf_var%i_id=id_id
      ENDIF

      ! east west wrap
      IF( PRESENT(id_ew) )THEN
         tf_var%i_ew=id_ew
      ENDIF

      ! type
      IF( PRESENT(id_type) )THEN
         tf_var%i_type=id_type
      ELSE
         tf_var%i_type=NF90_DOUBLE
      ENDIF

      ! add attribute
      IF( PRESENT(td_att) )THEN
         CALL var_add_att(tf_var, td_att(:))
      ENDIF

      ! add _FillValue
      IF( PRESENT(dd_fill) )THEN
         SELECT CASE( tf_var%i_type )
            CASE(NF90_BYTE)
               tl_att=att_init('_FillValue', INT(dd_fill,i1) )
            CASE(NF90_SHORT)
               tl_att=att_init('_FillValue', INT(dd_fill,i2) )
            CASE(NF90_INT)
               tl_att=att_init('_FillValue', INT(dd_fill,i4) )
            CASE(NF90_FLOAT)
               tl_att=att_init('_FillValue', REAL(dd_fill,sp) )
            CASE DEFAULT ! NF90_DOUBLE
               tl_att=att_init('_FillValue', dd_fill )
         END SELECT
         CALL var_move_att(tf_var, tl_att)
      ELSE
         il_ind=0
         IF( ASSOCIATED(tf_var%t_att) )THEN
            il_ind=att_get_index(tf_var%t_att(:),'_FillValue')
         ENDIF
         IF( il_ind == 0 )THEN
            SELECT CASE( tf_var%i_type )
               CASE(NF90_BYTE)
                  tl_att=att_init('_FillValue',NF90_FILL_BYTE)
               CASE(NF90_SHORT)
                  tl_att=att_init('_FillValue',NF90_FILL_SHORT)
               CASE(NF90_INT)
                  tl_att=att_init('_FillValue',NF90_FILL_INT)
               CASE(NF90_FLOAT)
                  tl_att=att_init('_FillValue',NF90_FILL_FLOAT)
               CASE DEFAULT ! NF90_DOUBLE
                  tl_att=att_init('_FillValue',NF90_FILL_DOUBLE)
            END SELECT         
            CALL var_add_att(tf_var, tl_att)
         ENDIF
      ENDIF

      ! scale factor
      IF( PRESENT(dd_scf) )THEN
         tl_att=att_init('scale_factor',dd_scf)
         CALL var_move_att(tf_var, tl_att)
      ENDIF

      ! add offset
      IF( PRESENT(dd_ofs) )THEN
         tl_att=att_init('add_offset',dd_ofs)
         CALL var_move_att(tf_var, tl_att)
      ENDIF

      IF( PRESENT(cd_units) )THEN
         tl_att=att_init('units',cd_units)
         CALL var_move_att(tf_var, tl_att)
      ENDIF

      IF( PRESENT(cd_axis) )THEN
         tf_var%c_axis=TRIM(cd_axis)
      ENDIF

      ! add dimension
      IF( PRESENT(td_dim) )THEN
         CALL var_add_dim(tf_var, td_dim(:))
      ELSE
         CALL var_add_dim(tf_var, dim_fill_unused())
      ENDIF

      IF( PRESENT(id_rec) )THEN
         tf_var%i_rec=id_rec
      ENDIF

      ! add minimum value
      IF( PRESENT(dd_min) )THEN
         tf_var%d_min=dd_min
      ENDIF

      ! add maximum value
      IF( PRESENT(dd_max) )THEN
         tf_var%d_max=dd_max
      ENDIF

      ! netcdf4
      IF( PRESENT(ld_contiguous) )THEN
         tf_var%l_contiguous=ld_contiguous
      ENDIF

      IF( PRESENT(ld_shuffle) )THEN
         tf_var%l_shuffle=ld_shuffle
      ENDIF

      IF( PRESENT(ld_fletcher32) )THEN
         tf_var%l_fletcher32=ld_fletcher32
      ENDIF

       IF( PRESENT(id_deflvl) )THEN
         tf_var%i_deflvl=id_deflvl
      ENDIF     
      
      IF( PRESENT(id_chunksz) )THEN
         tf_var%i_chunksz(:)=id_chunksz(:)
      ENDIF

      ! interp
      IF( PRESENT(cd_interp) )THEN
         tf_var%c_interp(:)=cd_interp(:)
      ENDIF

      !extrap
      IF( PRESENT(cd_extrap) )THEN
         tf_var%c_extrap(:)=cd_extrap(:)
      ENDIF

      !filter
      IF( PRESENT(cd_filter) )THEN
         tf_var%c_filter(:)=cd_filter(:)
      ENDIF

      ! unit factor
      IF( PRESENT(dd_unf) )THEN
         tl_att=att_init('units_factor',dd_unf)
         CALL var_move_att(tf_var, tl_att)
      ENDIF

      ! output unit (linked to unit factor)
      IF( PRESENT(cd_unt) )THEN
         tl_att=att_init('new_units',cd_unt)
         CALL var_move_att(tf_var, tl_att)
      ENDIF

      ! output name (renamed variable)
      IF( PRESENT(cd_unt) )THEN
         tl_att=att_init('output_name',cd_namout)
         CALL var_move_att(tf_var, tl_att)
      ENDIF

      ! add extra information
      CALL var__get_extra(tf_var)

      ! delete some attribute cause linked to file where variable come from
      CALL var_del_att(tf_var, 'refinment_factor')
      CALL var_del_att(tf_var, 'interpolation')
      CALL var_del_att(tf_var, 'extrapolation')
      CALL var_del_att(tf_var, 'filter')
      CALL var_del_att(tf_var, 'src_file')
      CALL var_del_att(tf_var, 'src_i_indices')
      CALL var_del_att(tf_var, 'src_j_indices')
      CALL var_del_att(tf_var, 'valid_min')
      CALL var_del_att(tf_var, 'valid_max')
      CALL var_del_att(tf_var, 'missing_value')

      ! clean
      CALL att_clean(tl_att)

   END FUNCTION var__init
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_1D_dp(cd_name, dd_value,                    &
         &                  id_start, id_count, id_type, td_dim,  &
         &                  td_att, dd_fill, cd_units, cd_axis,   &
         &                  cd_stdname, cd_longname,              &
         &                  cd_point, id_id, id_ew,               &
         &                  dd_scf, dd_ofs,  id_rec,              &
         &                  dd_min, dd_max,                       &
         &                  ld_contiguous, ld_shuffle,            &
         &                  ld_fletcher32, id_deflvl, id_chunksz, &
         &                  cd_interp, cd_extrap, cd_filter,      &
         &                  cd_unt, dd_unf,                       &
         &                  cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a real(8) 1D array of value.
   !> @details 
   !> Optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> Dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('z') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date November, 2016
   !> - allow to add scalar value
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] dd_value        1D array of real(8) value
   !> @param[in] id_start        index in the variable from which the data values 
   !> will be read
   !> @param[in] id_count        number of indices selected along each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] dd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle      shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      REAL(dp)        , DIMENSION(:)        , INTENT(IN) :: dd_value
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      ,                       INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:)        , INTENT(IN), OPTIONAL :: td_att
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                    :: il_type
      INTEGER(i4), DIMENSION(ip_maxdim)              :: il_start
      INTEGER(i4), DIMENSION(ip_maxdim)              :: il_count

      REAL(dp)   , DIMENSION(:,:,:,:)  , ALLOCATABLE :: dl_value

      TYPE(TDIM) , DIMENSION(ip_maxdim)              :: tl_dim

      ! loop indices
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)

      ! dummy call to avoid warning
      il_type=NF90_DOUBLE
      IF( PRESENT(id_type) ) il_type=id_type

      tl_dim(1)=dim_init( 'Z', id_len=SIZE(dd_value(:)) )
      IF( PRESENT(td_dim) )THEN
         tl_dim(1)=dim_copy(td_dim)
      ENDIF

      il_start(:)=1
      IF( PRESENT(id_start) )THEN
         il_start(1)=id_start
      ENDIF

      il_count(:)=tl_dim(:)%i_len
      IF( PRESENT(id_count) )THEN
         il_count(1)=id_count
      ENDIF

      ! reorder dimension
      CALL dim_reorder(tl_dim(:))
      ! reorder array
      il_start(:)=dim_reorder_2xyzt(tl_dim(:),il_start(:))
      il_count(:)=dim_reorder_2xyzt(tl_dim(:),il_count(:))

      tf_var=var__init( cd_name, id_type=il_type,           &
         &              td_dim=tl_dim(:), td_att=td_att,    &
         &              dd_fill=dd_fill, cd_units=cd_units, &
         &              cd_axis=cd_axis,                    &
         &              cd_stdname=cd_stdname,              &
         &              cd_longname=cd_longname,            &
         &              cd_point=cd_point, id_id=id_id,     &
         &              id_ew=id_ew, dd_scf=dd_scf,         &
         &              dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &              dd_min=dd_min, dd_max=dd_max,       &
         &              ld_contiguous=ld_contiguous,        &
         &              ld_shuffle=ld_shuffle,              &
         &              ld_fletcher32=ld_fletcher32,        &
         &              id_deflvl=id_deflvl,                &
         &              id_chunksz=id_chunksz(:),           &
         &              cd_interp=cd_interp(:),             &
         &              cd_extrap=cd_extrap(:),             &
         &              cd_filter=cd_filter(:),             &
         &              cd_unt=cd_unt, dd_unf=dd_unf,       &
         &              cd_namout=cd_namout )
   
      ! add value
      ALLOCATE( dl_value(tl_dim(1)%i_len, &
      &                  tl_dim(2)%i_len, &
      &                  tl_dim(3)%i_len, &
      &                  tl_dim(4)%i_len) )

      IF( tl_dim(1)%l_use )THEN
         dl_value(:,1,1,1) = dd_value(:)
      ELSEIF( tl_dim(2)%l_use )THEN
         dl_value(1,:,1,1) = dd_value(:)
      ELSEIF( tl_dim(3)%l_use )THEN
         dl_value(1,1,:,1) = dd_value(:)
      ELSEIF( tl_dim(4)%l_use )THEN
         dl_value(1,1,1,:) = dd_value(:)
      ELSE
         IF( SIZE(dd_value(:)) > 1 )THEN
            CALL logger_fatal("VAR INIT: can not add value from variable "//&
            &  TRIM(cd_name)//". invalid dimension to be used")
         ELSE
            dl_value(1,1,1,1) = dd_value(1)
            CALL logger_warn("VAR INIT: add scalar value for variable "//&
            &  TRIM(cd_name))

         ENDIF
      ENDIF

      CALL var_add_value( tf_var, dl_value(:,:,:,:), il_type, &
         &                il_start(:), il_count(:) )

      ! clean
      DEALLOCATE( dl_value )
      CALL dim_clean(tl_dim)

   END FUNCTION var__init_1D_dp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_2D_dp(cd_name, dd_value,                    &
         &                  id_start, id_count, id_type, td_dim,  &
         &                  td_att, dd_fill, cd_units, cd_axis,   &
         &                  cd_stdname, cd_longname,              &
         &                  cd_point, id_id, id_ew,               &
         &                  dd_scf, dd_ofs,  id_rec,              &
         &                  dd_min, dd_max,                       &
         &                  ld_contiguous, ld_shuffle,            &
         &                  ld_fletcher32, id_deflvl, id_chunksz, &
         &                  cd_interp, cd_extrap, cd_filter,      &
         &                  cd_unt, dd_unf,                       &
         &                  cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a real(8) 2D array of value. 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> @details 
   !> array of 2 dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date February, 2015 
   !> - bug fix: array initialise with dimension
   !> array not only one value
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> - Bux fix: dimension array initialise not only one value
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] dd_value        1D array of real(8) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] dd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle      shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates 
   !> no deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      REAL(dp)        , DIMENSION(:,:)      , INTENT(IN) :: dd_value
      INTEGER(i4)     , DIMENSION(:)        , INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:)        , INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:)        , INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:)        , INTENT(IN), OPTIONAL :: td_att
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                    :: il_type
      INTEGER(i4), DIMENSION(ip_maxdim)              :: il_start
      INTEGER(i4), DIMENSION(ip_maxdim)              :: il_count

      REAL(dp)   , DIMENSION(:,:,:,:)  , ALLOCATABLE :: dl_value

      TYPE(TDIM) , DIMENSION(ip_maxdim)              :: tl_dim

      ! loop indices
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)

      ! dummy call to avoid warning
      il_type=NF90_DOUBLE
      IF( PRESENT(id_type) ) il_type=id_type

      tl_dim(1)=dim_init( 'X', id_len=SIZE(dd_value(:,:),DIM=1) )
      tl_dim(2)=dim_init( 'Y', id_len=SIZE(dd_value(:,:),DIM=2) )
      IF( PRESENT(td_dim) )THEN
         IF( SIZE(td_dim(:)) /= 2 )THEN
            CALL logger_error("VAR INIT: dimension of dimension structure "//&
            &                 " not conform")
         ELSE
            tl_dim(1)=dim_copy(td_dim(1))
            tl_dim(2)=dim_copy(td_dim(2))
         ENDIF
      ENDIF

      il_start(:)=1
      IF( PRESENT(id_start) )THEN
         IF( SIZE(id_start(:)) /= 2 )THEN
            CALL logger_error("VAR INIT: dimension of start array "//&
            &                 " not conform")
         ELSE
            il_start(1)=id_start(1)
            il_start(2)=id_start(2)
         ENDIF
      ENDIF

      il_count(:)=tl_dim(:)%i_len
      IF( PRESENT(id_count) )THEN
         IF( SIZE(id_count(:)) /= 2 )THEN
            CALL logger_error("VAR INIT: dimension of count array "//&
            &                 " not conform")
         ELSE
            il_count(1)=id_count(1)
            il_count(2)=id_count(2)
         ENDIF
      ENDIF

      ! reorder dimension
      CALL dim_reorder(tl_dim(:))
      ! reorder array
      il_start(:)=dim_reorder_2xyzt(tl_dim(:),il_start(:))
      il_count(:)=dim_reorder_2xyzt(tl_dim(:),il_count(:))

      tf_var=var__init( cd_name, id_type=il_type,           &
         &              td_dim=tl_dim(:), td_att=td_att,    &
         &              dd_fill=dd_fill, cd_units=cd_units, &
         &              cd_axis=cd_axis,                    &
         &              cd_stdname=cd_stdname,              &
         &              cd_longname=cd_longname,            &
         &              cd_point=cd_point, id_id=id_id,     &
         &              id_ew=id_ew, dd_scf=dd_scf,         &
         &              dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &              dd_min=dd_min, dd_max=dd_max,       &
         &              ld_contiguous=ld_contiguous,        &
         &              ld_shuffle=ld_shuffle,              &
         &              ld_fletcher32=ld_fletcher32,        &
         &              id_deflvl=id_deflvl,                &
         &              id_chunksz=id_chunksz(:),           &
         &              cd_interp=cd_interp(:),             &
         &              cd_extrap=cd_extrap(:),             &
         &              cd_filter=cd_filter(:),             &
         &              cd_unt=cd_unt, dd_unf=dd_unf,       &
         &              cd_namout=cd_namout )
   
      ! add value
      ALLOCATE( dl_value(tl_dim(1)%i_len, &
         &               tl_dim(2)%i_len, &
         &               tl_dim(3)%i_len, &
         &               tl_dim(4)%i_len) )

      IF( tl_dim(1)%l_use .AND. tl_dim(2)%l_use )THEN
         dl_value(:,:,1,1)=dd_value(:,:)
      ELSEIF( tl_dim(1)%l_use .AND. tl_dim(3)%l_use )THEN
         dl_value(:,1,:,1)=dd_value(:,:)
      ELSEIF( tl_dim(1)%l_use .AND. tl_dim(4)%l_use )THEN
         dl_value(:,1,1,:)=dd_value(:,:)
      ELSEIF( tl_dim(2)%l_use .AND. tl_dim(3)%l_use )THEN
         dl_value(1,:,:,1)=dd_value(:,:)
      ELSEIF( tl_dim(2)%l_use .AND. tl_dim(4)%l_use )THEN
         dl_value(1,:,1,:)=dd_value(:,:)
      ELSEIF( tl_dim(3)%l_use .AND. tl_dim(4)%l_use )THEN
         dl_value(1,1,:,:)=dd_value(:,:)
      ELSE
         CALL logger_fatal("VAR INIT: can not add value from variable "//&
         &  TRIM(cd_name)//". invalid dimension to be used")
      ENDIF

      CALL var_add_value( tf_var, dl_value(:,:,:,:), il_type, &
         &                il_start(:), il_count(:) )

      ! clean
      DEALLOCATE( dl_value )
      CALL dim_clean(tl_dim)

   END FUNCTION var__init_2D_dp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_3D_dp(cd_name, dd_value,                    &
         &                  id_start, id_count, id_type, td_dim,  &
         &                  td_att, dd_fill, cd_units, cd_axis,   &
         &                  cd_stdname, cd_longname,              &
         &                  cd_point, id_id, id_ew,               &
         &                  dd_scf, dd_ofs,  id_rec,              &
         &                  dd_min, dd_max,                       &
         &                  ld_contiguous, ld_shuffle,            &
         &                  ld_fletcher32, id_deflvl, id_chunksz, &
         &                  cd_interp, cd_extrap, cd_filter,      &
         &                  cd_unt, dd_unf,                       &
         &                  cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a real(8) 3D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> array of 3 dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y','z') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] dd_value        1D array of real(8) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] dd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      REAL(dp)        , DIMENSION(:,:,:)    , INTENT(IN) :: dd_value
      INTEGER(i4)     , DIMENSION(:)        , INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:)        , INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:)        , INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:)        , INTENT(IN), OPTIONAL :: td_att
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                    :: il_type
      INTEGER(i4), DIMENSION(ip_maxdim)              :: il_start
      INTEGER(i4), DIMENSION(ip_maxdim)              :: il_count

      REAL(dp)   , DIMENSION(:,:,:,:)  , ALLOCATABLE :: dl_value

      TYPE(TDIM) , DIMENSION(ip_maxdim)              :: tl_dim

      ! loop indices
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)

      ! dummy call to avoid warning
      il_type=NF90_DOUBLE
      IF( PRESENT(id_type) ) il_type=id_type

      tl_dim(1)=dim_init( 'X', id_len=SIZE(dd_value(:,:,:),DIM=1) )
      tl_dim(2)=dim_init( 'Y', id_len=SIZE(dd_value(:,:,:),DIM=2) )
      tl_dim(3)=dim_init( 'Z', id_len=SIZE(dd_value(:,:,:),DIM=3) )
      IF( PRESENT(td_dim) )THEN
         IF( SIZE(td_dim(:)) /= 3 )THEN
            CALL logger_error("VAR INIT: dimension of dimension structure "//&
            &                 " not conform")
         ELSE
            tl_dim(1)=dim_copy(td_dim(1))
            tl_dim(2)=dim_copy(td_dim(2))
            tl_dim(3)=dim_copy(td_dim(3))
         ENDIF
      ENDIF

      il_start(:)=1
      IF( PRESENT(id_start) )THEN
         IF( SIZE(id_start(:)) /= 3 )THEN
            CALL logger_error("VAR INIT: dimension of start array "//&
            &                 " not conform")
         ELSE
            il_start(1)=id_start(1)
            il_start(2)=id_start(2)
            il_start(3)=id_start(3)
         ENDIF
      ENDIF

      il_count(:)=tl_dim(:)%i_len
      IF( PRESENT(id_count) )THEN
         IF( SIZE(id_count(:)) /= 3 )THEN
            CALL logger_error("VAR INIT: dimension of count array "//&
            &                 " not conform")
         ELSE
            il_count(1)=id_count(1)
            il_count(2)=id_count(2)
            il_count(3)=id_count(3)
         ENDIF
      ENDIF

      ! reorder dimension
      CALL dim_reorder(tl_dim(:))
      ! reorder array
      il_start(:)=dim_reorder_2xyzt(tl_dim(:),il_start(:))
      il_count(:)=dim_reorder_2xyzt(tl_dim(:),il_count(:))

      tf_var=var__init( cd_name, id_type=il_type,           &
         &              td_dim=tl_dim(:), td_att=td_att,    &
         &              dd_fill=dd_fill, cd_units=cd_units, &
         &              cd_axis=cd_axis,                    &
         &              cd_stdname=cd_stdname,              &
         &              cd_longname=cd_longname,            &
         &              cd_point=cd_point, id_id=id_id,     &
         &              id_ew=id_ew, dd_scf=dd_scf,         &
         &              dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &              dd_min=dd_min, dd_max=dd_max,       &
         &              ld_contiguous=ld_contiguous,        &
         &              ld_shuffle=ld_shuffle,              &
         &              ld_fletcher32=ld_fletcher32,        &
         &              id_deflvl=id_deflvl,                &
         &              id_chunksz=id_chunksz(:),           &
         &              cd_interp=cd_interp(:),             &
         &              cd_extrap=cd_extrap(:),             &
         &              cd_filter=cd_filter(:),             &
         &              cd_unt=cd_unt, dd_unf=dd_unf,       &
         &              cd_namout=cd_namout )
   
      ! add value
      ALLOCATE( dl_value(tl_dim(1)%i_len, &
      &                  tl_dim(2)%i_len, &
      &                  tl_dim(3)%i_len, &
      &                  tl_dim(4)%i_len) )

      IF( tl_dim(1)%l_use .AND. tl_dim(2)%l_use .AND. tl_dim(3)%l_use )THEN
         dl_value(:,:,:,1)=dd_value(:,:,:)
      ELSEIF( tl_dim(1)%l_use .AND. tl_dim(2)%l_use .AND. tl_dim(4)%l_use  )THEN
         dl_value(:,:,1,:)=dd_value(:,:,:)
      ELSEIF( tl_dim(1)%l_use .AND. tl_dim(3)%l_use .AND. tl_dim(4)%l_use )THEN
         dl_value(:,1,:,:)=dd_value(:,:,:)
      ELSEIF( tl_dim(2)%l_use .AND. tl_dim(3)%l_use .AND. tl_dim(4)%l_use )THEN
         dl_value(1,:,:,:)=dd_value(:,:,:)
      ELSE
         CALL logger_fatal("VAR INIT: can not add value from variable "//&
         &  TRIM(cd_name)//". invalid dimension to be used")
      ENDIF

      CALL var_add_value( tf_var, dl_value(:,:,:,:), il_type, &
         &                il_start(:), il_count(:) )

      ! clean
      DEALLOCATE( dl_value )
      CALL dim_clean(tl_dim)

   END FUNCTION var__init_3D_dp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_dp(cd_name, dd_value,                    &
         &               id_start, id_count, id_type, td_dim,  &
         &               td_att, dd_fill, cd_units, cd_axis,   &
         &               cd_stdname, cd_longname,              &
         &               cd_point, id_id, id_ew,               &
         &               dd_scf, dd_ofs,  id_rec,              &
         &               dd_min, dd_max,                       &
         &               ld_contiguous, ld_shuffle,            &
         &               ld_fletcher32, id_deflvl, id_chunksz, &
         &               cd_interp, cd_extrap, cd_filter,      &
         &               cd_unt, dd_unf,                       &
         &               cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a real(8) 4D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> Dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y','z','t') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] dd_value        4D array of real(8) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] dd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      REAL(dp)        , DIMENSION(:,:,:,:),   INTENT(IN) :: dd_value
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_att
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_shape
      TYPE(TDIM)                        :: tl_dim

      INTEGER(i4) :: il_type

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)

      ! dummy call to avoid warning
      il_type=NF90_DOUBLE
      IF( PRESENT(id_type) ) il_type=id_type

      tf_var=var__init( cd_name, id_type=il_type,           &
         &              td_dim=td_dim, td_att=td_att,       &
         &              dd_fill=dd_fill, cd_units=cd_units, &
         &              cd_axis=cd_axis,                    &
         &              cd_stdname=cd_stdname,              &
         &              cd_longname=cd_longname,            &
         &              cd_point=cd_point, id_id=id_id,     &
         &              id_ew=id_ew, dd_scf=dd_scf,         &
         &              dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &              dd_min=dd_min, dd_max=dd_max,       &
         &              ld_contiguous=ld_contiguous,        &
         &              ld_shuffle=ld_shuffle,              &
         &              ld_fletcher32=ld_fletcher32,        &
         &              id_deflvl=id_deflvl,                &
         &              id_chunksz=id_chunksz(:),           &
         &              cd_interp=cd_interp(:),             &
         &              cd_extrap=cd_extrap(:),             &
         &              cd_filter=cd_filter(:),             &
         &              cd_unt=cd_unt, dd_unf=dd_unf,       &
         &              cd_namout=cd_namout )
 
      ! add value
      IF( .NOT. PRESENT(td_dim) )THEN
         il_shape(:)=SHAPE(dd_value(:,:,:,:)) 
         DO ji=1,ip_maxdim
            tl_dim=dim_init( cp_dimorder(ji:ji), id_len=il_shape(ji)) 
            CALL var_add_dim(tf_var, tl_dim)
         ENDDO
      ENDIF

      CALL var_add_value( tf_var, dd_value(:,:,:,:), il_type, &
         &                id_start(:), id_count(:) )

      ! clean
      CALL dim_clean(tl_dim)

   END FUNCTION var__init_dp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_1D_sp(cd_name, rd_value,                    &
         &                  id_start, id_count, id_type, td_dim,  &
         &                  td_att, rd_fill, cd_units, cd_axis,   &
         &                  cd_stdname, cd_longname,              &
         &                  cd_point, id_id, id_ew,               &
         &                  dd_scf, dd_ofs,  id_rec,              &
         &                  dd_min, dd_max,                       &
         &                  ld_contiguous, ld_shuffle,            &
         &                  ld_fletcher32, id_deflvl, id_chunksz, &
         &                  cd_interp, cd_extrap, cd_filter,      &
         &                  cd_unt, dd_unf,                       &
         &                  cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a real(4) 1D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('z') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] rd_value        1D array of real(4) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] rd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      REAL(sp)        , DIMENSION(:)        , INTENT(IN) :: rd_value
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      ,                       INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_att
      REAL(sp)        ,                       INTENT(IN), OPTIONAL :: rd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                       :: il_type
      INTEGER(i4)                                       :: il_shape

      REAL(dp)                                          :: dl_fill
      REAL(dp)        , DIMENSION(:)      , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_FLOAT
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_FLOAT
      IF( PRESENT(rd_fill) ) dl_fill=REAL(rd_fill,dp)

      il_shape=SIZE(rd_value(:))
      ALLOCATE( dl_value( il_shape) )

      DO ji=1,il_shape
         dl_value(ji)=REAL(rd_value(ji),dp)
      ENDDO

      tf_var=var_init( cd_name, dl_value(:),               &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
 
      DEALLOCATE( dl_value )
 
   END FUNCTION var__init_1D_sp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_2D_sp(cd_name, rd_value,                    &
         &                  id_start, id_count, id_type, td_dim,  &
         &                  td_att, rd_fill, cd_units, cd_axis,   &
         &                  cd_stdname, cd_longname,              &
         &                  cd_point, id_id, id_ew,               &
         &                  dd_scf, dd_ofs,  id_rec,              &
         &                  dd_min, dd_max,                       &
         &                  ld_contiguous, ld_shuffle,            &
         &                  ld_fletcher32, id_deflvl, id_chunksz, &
         &                  cd_interp, cd_extrap, cd_filter,      &
         &                  cd_unt, dd_unf,                       &
         &                  cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a real(4) 2D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> array of 2 dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         : variable name
   !> @param[in] rd_value        : 2D array of real(4) value
   !> @param[in] id_start        : index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        : number of indices selected along 
   !> each dimension
   !> @param[in] id_type         : variable type 
   !> @param[in] td_dim          : array of dimension structure
   !> @param[in] td_att          : array of attribute structure
   !> @param[in] rd_fill         : fill value
   !> @param[in] cd_units        : units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      : variable standard name
   !> @param[in] cd_longname     : variable long name
   !> @param[in] cd_point        : point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           : variable id
   !> @param[in] id_ew           : east west wrap
   !> @param[in] dd_scf          : scale factor
   !> @param[in] dd_ofs          : add offset
   !> @param[in] id_rec          : record id (for rstdimg file)
   !> @param[in] dd_min          : minimum value
   !> @param[in] dd_max          : maximum value
   !> @param[in] ld_contiguous   : use contiguous storage or not 
   !> @param[in] ld_shuffle      :  shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   : fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       : deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      : chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      REAL(sp)        , DIMENSION(:,:)     ,  INTENT(IN) :: rd_value
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_att
      REAL(sp)        ,                       INTENT(IN), OPTIONAL :: rd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                  :: il_type
      INTEGER(i4), DIMENSION(2)                    :: il_shape

      REAL(dp)                                     :: dl_fill
      REAL(dp)   , DIMENSION(:,:)    , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_FLOAT
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_FLOAT
      IF( PRESENT(rd_fill) ) dl_fill=REAL(rd_fill,dp)

      il_shape(:)=SHAPE(rd_value(:,:))

      ALLOCATE( dl_value( il_shape(1), &
         &                il_shape(2)) )

      DO jj=1,il_shape(2)
         DO ji=1,il_shape(1)
            dl_value(ji,jj)=REAL(rd_value(ji,jj),dp)
         ENDDO
      ENDDO

      tf_var=var_init( cd_name, dl_value(:,:),             &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
      
      DEALLOCATE( dl_value )
      
   END FUNCTION var__init_2D_sp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_3D_sp(cd_name, rd_value,        &
         &                             id_start, id_count, id_type, td_dim, &
         &                             td_att, rd_fill, cd_units, cd_axis,&
         &                             cd_stdname, cd_longname,  &
         &                             cd_point, id_id, id_ew,   &
         &                             dd_scf, dd_ofs,  id_rec,  &
         &                             dd_min, dd_max,           &
         &                             ld_contiguous, ld_shuffle,&
         &                             ld_fletcher32, id_deflvl, id_chunksz, &
         &                             cd_interp, cd_extrap, cd_filter, &
         &                             cd_unt, dd_unf, &
         &                             cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a real(4) 3D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> array of 3 dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y','z') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         : variable name
   !> @param[in] rd_value        : 2D array of real(4) value
   !> @param[in] id_start        : index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        : number of indices selected along 
   !> each dimension
   !> @param[in] id_type         : variable type 
   !> @param[in] td_dim          : array of dimension structure
   !> @param[in] td_att          : array of attribute structure
   !> @param[in] rd_fill         : fill value
   !> @param[in] cd_units        : units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      : variable standard name
   !> @param[in] cd_longname     : variable long name
   !> @param[in] cd_point        : point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           : variable id
   !> @param[in] id_ew           : east west wrap
   !> @param[in] dd_scf          : scale factor
   !> @param[in] dd_ofs          : add offset
   !> @param[in] id_rec          : record id (for rstdimg file)
   !> @param[in] dd_min          : minimum value
   !> @param[in] dd_max          : maximum value
   !> @param[in] ld_contiguous   : use contiguous storage or not 
   !> @param[in] ld_shuffle      :  shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   : fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       : deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      : chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      REAL(sp)        , DIMENSION(:,:,:)   ,  INTENT(IN) :: rd_value
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_att
      REAL(sp)        ,                       INTENT(IN), OPTIONAL :: rd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                  :: il_type
      INTEGER(i4), DIMENSION(3)                    :: il_shape

      REAL(dp)                                     :: dl_fill
      REAL(dp)   , DIMENSION(:,:,:)  , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_FLOAT
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_FLOAT
      IF( PRESENT(rd_fill) ) dl_fill=REAL(rd_fill,dp)

      il_shape(:)=SHAPE(rd_value(:,:,:))

      ALLOCATE( dl_value( il_shape(1), &
         &                il_shape(2), &
         &                il_shape(3)) )

      DO jk=1,il_shape(3)
         DO jj=1,il_shape(2)
            DO ji=1,il_shape(1)
               dl_value(ji,jj,jk)=REAL(rd_value(ji,jj,jk),dp)
            ENDDO
         ENDDO
      ENDDO

      tf_var=var_init( cd_name, dl_value(:,:,:),           &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
      
      DEALLOCATE( dl_value )
      
   END FUNCTION var__init_3D_sp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_sp(cd_name, rd_value,                    &
         &               id_start, id_count, id_type, td_dim,  &
         &               td_att, rd_fill, cd_units, cd_axis,   &
         &               cd_stdname, cd_longname,              &
         &               cd_point, id_id, id_ew,               &
         &               dd_scf, dd_ofs,  id_rec,              &
         &               dd_min, dd_max,                       &
         &               ld_contiguous, ld_shuffle,            &
         &               ld_fletcher32, id_deflvl, id_chunksz, &
         &               cd_interp, cd_extrap, cd_filter,      &
         &               cd_unt, dd_unf,                       &
         &               cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a real(4) 4D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> Dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y','z','t') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] rd_value        4D array of real(4) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] rd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle      shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      REAL(sp)        , DIMENSION(:,:,:,:),   INTENT(IN) :: rd_value
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_att
      REAL(sp)        ,                       INTENT(IN), OPTIONAL :: rd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                       :: il_type
      INTEGER(i4), DIMENSION(ip_maxdim)                 :: il_shape

      REAL(dp)                                          :: dl_fill
      REAL(dp)        , DIMENSION(:,:,:,:), ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_FLOAT
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_FLOAT
      IF( PRESENT(rd_fill) ) dl_fill=REAL(rd_fill,dp)

      il_shape(:)=SHAPE(rd_value(:,:,:,:))

      ALLOCATE( dl_value( il_shape(1), &
         &                il_shape(2), &
         &                il_shape(3), &
         &                il_shape(4)) )

      DO jl=1,il_shape(4)
         DO jk=1,il_shape(3)
            DO jj=1,il_shape(2)
               DO ji=1,il_shape(1)
                  dl_value(ji,jj,jk,jl)=REAL(rd_value(ji,jj,jk,jl),dp)
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      tf_var=var_init( cd_name, dl_value(:,:,:,:),         &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
      
      DEALLOCATE( dl_value )
      
   END FUNCTION var__init_sp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_1D_i8(cd_name, kd_value,                    &
         &                  id_start, id_count, id_type, td_dim,  &
         &                  td_att, kd_fill, cd_units, cd_axis,   &
         &                  cd_stdname, cd_longname,              &
         &                  cd_point, id_id, id_ew,               &
         &                  dd_scf, dd_ofs,  id_rec,              &
         &                  dd_min, dd_max,                       &
         &                  ld_contiguous, ld_shuffle,            &
         &                  ld_fletcher32, id_deflvl, id_chunksz, &
         &                  cd_interp, cd_extrap, cd_filter,      &
         &                  cd_unt, dd_unf,                       &
         &                  cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a integer(8) 1D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('z') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         : variable name
   !> @param[in] kd_value        : 1D array of integer(8) value
   !> @param[in] id_start        : index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        : number of indices selected along 
   !> each dimension
   !> @param[in] id_type         : variable type 
   !> @param[in] td_dim          : array of dimension structure
   !> @param[in] td_att          : array of attribute structure
   !> @param[in] kd_fill         : fill value
   !> @param[in] cd_units        : units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      : variable standard name
   !> @param[in] cd_longname     : variable long name
   !> @param[in] cd_point        : point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           : variable id
   !> @param[in] id_ew           : east west wrap
   !> @param[in] dd_scf          : scale factor
   !> @param[in] dd_ofs          : add offset
   !> @param[in] id_rec          : record id (for rstdimg file)
   !> @param[in] dd_min          : minimum value
   !> @param[in] dd_max          : maximum value
   !> @param[in] ld_contiguous   : use contiguous storage or not 
   !> @param[in] ld_shuffle      :  shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   : fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       : deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      : chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i8)     , DIMENSION(:)        , INTENT(IN) :: kd_value
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      ,                       INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_att
      INTEGER(i8)     ,                       INTENT(IN), OPTIONAL :: kd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                       :: il_type
      INTEGER(i4)                                       :: il_shape

      REAL(dp)                                          :: dl_fill
      REAL(dp)        , DIMENSION(:)      , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_INT
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_INT
      IF( PRESENT(kd_fill) ) dl_fill=REAL(kd_fill,dp)

      il_shape=SIZE(kd_value(:))
      ALLOCATE( dl_value( il_shape) )

      DO ji=1,il_shape
         dl_value(ji)=REAL(kd_value(ji),dp)
      ENDDO

      tf_var=var_init( cd_name, dl_value(:),               &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
 
      DEALLOCATE( dl_value )
 
   END FUNCTION var__init_1D_i8
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_2D_i8(cd_name, kd_value,        &
         &                             id_start, id_count, id_type, td_dim, &
         &                             td_att, kd_fill, cd_units, cd_axis,&
         &                             cd_stdname, cd_longname,  &
         &                             cd_point, id_id, id_ew,   &
         &                             dd_scf, dd_ofs,  id_rec,  &
         &                             dd_min, dd_max,           &
         &                             ld_contiguous, ld_shuffle,&
         &                             ld_fletcher32, id_deflvl, id_chunksz, &
         &                             cd_interp, cd_extrap, cd_filter, &
         &                             cd_unt, dd_unf, &
         &                             cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a integer(8) 2D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> array of 2 dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] kd_value        2D array of integer(8) value
   !> @param[in] id_start        index in the variable from which the data values 
   !> will be read
   !> @param[in] id_count        number of indices selected along each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] kd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i8)     , DIMENSION(:,:)     ,  INTENT(IN) :: kd_value
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_att
      INTEGER(i8)     ,                       INTENT(IN), OPTIONAL :: kd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                  :: il_type
      INTEGER(i4), DIMENSION(2)                    :: il_shape

      REAL(dp)                                     :: dl_fill
      REAL(dp)   , DIMENSION(:,:)    , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_INT
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_INT
      IF( PRESENT(kd_fill) ) dl_fill=REAL(kd_fill,dp)

      il_shape(:)=SHAPE(kd_value(:,:))

      ALLOCATE( dl_value( il_shape(1), &
         &                il_shape(2)) )

      DO jj=1,il_shape(2)
         DO ji=1,il_shape(1)
            dl_value(ji,jj)=REAL(kd_value(ji,jj),dp)
         ENDDO
      ENDDO

      tf_var=var_init( cd_name, dl_value(:,:),             &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
      
      DEALLOCATE( dl_value )
      
   END FUNCTION var__init_2D_i8
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_3D_i8(cd_name, kd_value,                    &
         &                  id_start, id_count, id_type, td_dim,  &
         &                  td_att, kd_fill, cd_units, cd_axis,   &
         &                  cd_stdname, cd_longname,              &
         &                  cd_point, id_id, id_ew,               &
         &                  dd_scf, dd_ofs,  id_rec,              &
         &                  dd_min, dd_max,                       &
         &                  ld_contiguous, ld_shuffle,            &
         &                  ld_fletcher32, id_deflvl, id_chunksz, &
         &                  cd_interp, cd_extrap, cd_filter,      &
         &                  cd_unt, dd_unf,                       &
         &                  cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a integer(8) 3D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> array of 3 dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y','z') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] kd_value        2D array of integer(8) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] kd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i8)     , DIMENSION(:,:,:)   ,  INTENT(IN) :: kd_value
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_att
      INTEGER(i8)     ,                       INTENT(IN), OPTIONAL :: kd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                  :: il_type
      INTEGER(i4), DIMENSION(3)                    :: il_shape

      REAL(dp)                                     :: dl_fill
      REAL(dp)   , DIMENSION(:,:,:)  , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_INT
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_INT
      IF( PRESENT(kd_fill) ) dl_fill=REAL(kd_fill,dp)

      il_shape(:)=SHAPE(kd_value(:,:,:))

      ALLOCATE( dl_value( il_shape(1), &
         &                il_shape(2), &
         &                il_shape(3)) )

      DO jk=1,il_shape(3)
         DO jj=1,il_shape(2)
            DO ji=1,il_shape(1)
               dl_value(ji,jj,jk)=REAL(kd_value(ji,jj,jk),dp)
            ENDDO
         ENDDO
      ENDDO

      tf_var=var_init( cd_name, dl_value(:,:,:),           &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
      
      DEALLOCATE( dl_value )
      
   END FUNCTION var__init_3D_i8
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_i8(cd_name, kd_value,                    &
         &               id_start, id_count, id_type, td_dim,  &
         &               td_att, kd_fill, cd_units, cd_axis,   &
         &               cd_stdname, cd_longname,              &
         &               cd_point, id_id, id_ew,               &
         &               dd_scf, dd_ofs,  id_rec,              &
         &               dd_min, dd_max,                       &
         &               ld_contiguous, ld_shuffle,            &
         &               ld_fletcher32, id_deflvl, id_chunksz, &
         &               cd_interp, cd_extrap, cd_filter,      &
         &               cd_unt, dd_unf,                       &
         &               cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a integer(8) 4D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> Dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y','z','t') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] kd_value        4D array of integer(8) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] kd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i8)     , DIMENSION(:,:,:,:),   INTENT(IN) :: kd_value
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_att
      INTEGER(i8)     ,                       INTENT(IN), OPTIONAL :: kd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                       :: il_type
      INTEGER(i4), DIMENSION(ip_maxdim)                 :: il_shape

      REAL(dp)                                          :: dl_fill
      REAL(dp)        , DIMENSION(:,:,:,:), ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_INT
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_INT
      IF( PRESENT(kd_fill) ) dl_fill=REAL(kd_fill,dp)

      il_shape(:)=SHAPE(kd_value(:,:,:,:))

      ALLOCATE( dl_value( il_shape(1), &
         &                il_shape(2), &
         &                il_shape(3), &
         &                il_shape(4)) )

      DO jl=1,il_shape(4)
         DO jk=1,il_shape(3)
            DO jj=1,il_shape(2)
               DO ji=1,il_shape(1)
                  dl_value(ji,jj,jk,jl)=REAL(kd_value(ji,jj,jk,jl),dp)
               ENDDO
            ENDDO
         ENDDO
      ENDDO                  

      tf_var=var_init( cd_name, dl_value(:,:,:,:),         &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
      
      DEALLOCATE( dl_value )
      
   END FUNCTION var__init_i8
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_1D_i4(cd_name, id_value,                    &
         &                  id_start, id_count, id_type, td_dim,  &
         &                  td_att, id_fill, cd_units, cd_axis,   &
         &                  cd_stdname, cd_longname,              &
         &                  cd_point, id_id, id_ew,               &
         &                  dd_scf, dd_ofs,  id_rec,              &
         &                  dd_min, dd_max,                       &
         &                  ld_contiguous, ld_shuffle,            &
         &                  ld_fletcher32, id_deflvl, id_chunksz, &
         &                  cd_interp, cd_extrap, cd_filter,      &
         &                  cd_unt, dd_unf,                       &
         &                  cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a integer(4) 1D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('z') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] id_value        1D array of integer(4) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] id_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i4)     , DIMENSION(:)        , INTENT(IN) :: id_value
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      ,                       INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_att
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                       :: il_type
      INTEGER(i4)                                       :: il_shape

      REAL(dp)                                          :: dl_fill
      REAL(dp)        , DIMENSION(:)      , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_INT
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_INT
      IF( PRESENT(id_fill) ) dl_fill=REAL(id_fill,dp)

      il_shape=SIZE(id_value(:))
      ALLOCATE( dl_value( il_shape) )

      DO ji=1,il_shape
         dl_value(ji)=REAL(id_value(ji),dp)
      ENDDO

      tf_var=var_init( cd_name, dl_value(:),               &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
 
      DEALLOCATE( dl_value )
 
   END FUNCTION var__init_1D_i4
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_2D_i4(cd_name, id_value,                    &
         &                  id_start, id_count, id_type, td_dim,  &
         &                  td_att, id_fill, cd_units, cd_axis,   &
         &                  cd_stdname, cd_longname,              &
         &                  cd_point, id_id, id_ew,               &
         &                  dd_scf, dd_ofs,  id_rec,              &
         &                  dd_min, dd_max,                       &
         &                  ld_contiguous, ld_shuffle,            &
         &                  ld_fletcher32, id_deflvl, id_chunksz, &
         &                  cd_interp, cd_extrap, cd_filter,      &
         &                  cd_unt, dd_unf,                       &
         &                  cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a integer(4) 2D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> array of 2 dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] id_value        2D array of integer(4) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] id_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i4)     , DIMENSION(:,:)     ,  INTENT(IN) :: id_value
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_att
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                  :: il_type
      INTEGER(i4), DIMENSION(2)                    :: il_shape

      REAL(dp)                                     :: dl_fill
      REAL(dp)   , DIMENSION(:,:)    , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_INT
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_INT
      IF( PRESENT(id_fill) ) dl_fill=REAL(id_fill,dp)

      il_shape(:)=SHAPE(id_value(:,:))

      ALLOCATE( dl_value( il_shape(1), &
         &                il_shape(2)) )

      DO jj=1,il_shape(2)
         DO ji=1,il_shape(1)
            dl_value(ji,jj)=REAL(id_value(ji,jj),dp)
         ENDDO
      ENDDO

      tf_var=var_init( cd_name, dl_value(:,:),             &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
      
      DEALLOCATE( dl_value )
      
   END FUNCTION var__init_2D_i4
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_3D_i4(cd_name, id_value,                    &
         &                  id_start, id_count, id_type, td_dim,  &
         &                  td_att, id_fill, cd_units, cd_axis,   &
         &                  cd_stdname, cd_longname,              &
         &                  cd_point, id_id, id_ew,               &
         &                  dd_scf, dd_ofs,  id_rec,              &
         &                  dd_min, dd_max,                       &
         &                  ld_contiguous, ld_shuffle,            &
         &                  ld_fletcher32, id_deflvl, id_chunksz, &
         &                  cd_interp, cd_extrap, cd_filter,      &
         &                  cd_unt, dd_unf,                       &
         &                  cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a integer(4) 3D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> array of 3 dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y','z') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] id_value        3D array of integer(4) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] id_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i4)     , DIMENSION(:,:,:)   ,  INTENT(IN) :: id_value
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_att
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                  :: il_type
      INTEGER(i4), DIMENSION(3)                    :: il_shape

      REAL(dp)                                     :: dl_fill
      REAL(dp)   , DIMENSION(:,:,:)  , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_INT
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_INT
      IF( PRESENT(id_fill) ) dl_fill=REAL(id_fill,dp)

      il_shape(:)=SHAPE(id_value(:,:,:))

      ALLOCATE( dl_value( il_shape(1), &
         &                il_shape(2), &
         &                il_shape(3)) )

      DO jk=1,il_shape(3)
         DO jj=1,il_shape(2)
            DO ji=1,il_shape(1)
               dl_value(ji,jj,jk)=REAL(id_value(ji,jj,jk),dp)
            ENDDO
         ENDDO
      ENDDO

      tf_var=var_init( cd_name, dl_value(:,:,:),           &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
      
      DEALLOCATE( dl_value )
      
   END FUNCTION var__init_3D_i4
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_i4(cd_name, id_value,                    &
         &               id_start, id_count, id_type, td_dim,  &
         &               td_att, id_fill, cd_units, cd_axis,   &
         &               cd_stdname, cd_longname,              &
         &               cd_point, id_id, id_ew,               &
         &               dd_scf, dd_ofs,  id_rec,              &
         &               dd_min, dd_max,                       &
         &               ld_contiguous, ld_shuffle,            &
         &               ld_fletcher32, id_deflvl, id_chunksz, &
         &               cd_interp, cd_extrap, cd_filter,      &
         &               cd_unt, dd_unf,                       &
         &               cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a integer(4) 4D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> Dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y','z','t') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] id_value        4D array of integer(4) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] id_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i4)     , DIMENSION(:,:,:,:),   INTENT(IN) :: id_value
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_att
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                       :: il_type
      INTEGER(i4), DIMENSION(ip_maxdim)                 :: il_shape

      REAL(dp)                                          :: dl_fill
      REAL(dp)        , DIMENSION(:,:,:,:), ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_INT
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_INT
      IF( PRESENT(id_fill) ) dl_fill=REAL(id_fill,dp)

      il_shape(:)=SHAPE(id_value(:,:,:,:))

      ALLOCATE( dl_value( il_shape(1), &
         &                il_shape(2), &
         &                il_shape(3), &
         &                il_shape(4)) )

      DO jl=1,il_shape(4)
         DO jk=1,il_shape(3)
            DO jj=1,il_shape(2)
               DO ji=1,il_shape(1)
                  dl_value(ji,jj,jk,jl)=REAL(id_value(ji,jj,jk,jl),dp)
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      tf_var=var_init( cd_name, dl_value(:,:,:,:),         &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
      
      DEALLOCATE( dl_value )
      
   END FUNCTION var__init_i4
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_1D_i2(cd_name, sd_value,                    &
         &                  id_start, id_count, id_type, td_dim,  &
         &                  td_att, sd_fill, cd_units, cd_axis,   &
         &                  cd_stdname, cd_longname,              &
         &                  cd_point, id_id, id_ew,               &
         &                  dd_scf, dd_ofs,  id_rec,              &
         &                  dd_min, dd_max,                       &
         &                  ld_contiguous, ld_shuffle,            &
         &                  ld_fletcher32, id_deflvl, id_chunksz, &
         &                  cd_interp, cd_extrap, cd_filter,      &
         &                  cd_unt, dd_unf,                       &
         &                  cd_namout) &
         &  RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a integer(2) 1D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('z') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] sd_value        1D array of integer(2) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] sd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i2)     , DIMENSION(:)        , INTENT(IN) :: sd_value
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      ,                       INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_att
      INTEGER(i2)     ,                       INTENT(IN), OPTIONAL :: sd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                       :: il_type
      INTEGER(i4)                                       :: il_shape

      REAL(dp)                                          :: dl_fill
      REAL(dp)        , DIMENSION(:)      , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_SHORT
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_SHORT
      IF( PRESENT(sd_fill) ) dl_fill=REAL(sd_fill,dp)

      il_shape=SIZE(sd_value(:))
      ALLOCATE( dl_value( il_shape) )

      DO ji=1,il_shape
         dl_value(ji)=REAL(sd_value(ji),dp)
      ENDDO

      tf_var=var_init( cd_name, dl_value(:),               &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
 
      DEALLOCATE( dl_value )
 
   END FUNCTION var__init_1D_i2
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_2D_i2(cd_name, sd_value,                    &
         &                  id_start, id_count, id_type, td_dim,  &
         &                  td_att, sd_fill, cd_units, cd_axis,   &
         &                  cd_stdname, cd_longname,              &
         &                  cd_point, id_id, id_ew,               &
         &                  dd_scf, dd_ofs,  id_rec,              &
         &                  dd_min, dd_max,                       &
         &                  ld_contiguous, ld_shuffle,            &
         &                  ld_fletcher32, id_deflvl, id_chunksz, &
         &                  cd_interp, cd_extrap, cd_filter,      &
         &                  cd_unt, dd_unf,                       &
         &                  cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a integer(2) 2D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> array of 2 dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] sd_value        2D array of integer(2) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] sd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i2)     , DIMENSION(:,:)     ,  INTENT(IN) :: sd_value
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_att
      INTEGER(i2)     ,                       INTENT(IN), OPTIONAL :: sd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                  :: il_type
      INTEGER(i4), DIMENSION(2)                    :: il_shape

      REAL(dp)                                     :: dl_fill
      REAL(dp)   , DIMENSION(:,:)    , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_SHORT
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_SHORT
      IF( PRESENT(sd_fill) ) dl_fill=REAL(sd_fill,dp)

      il_shape(:)=SHAPE(sd_value(:,:))

      ALLOCATE( dl_value( il_shape(1), &
         &                il_shape(2)) )

      DO jj=1,il_shape(2)
         DO ji=1,il_shape(1)
            dl_value(ji,jj)=REAL(sd_value(ji,jj),dp)
         ENDDO
      ENDDO

      tf_var=var_init( cd_name, dl_value(:,:),             &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
      
      DEALLOCATE( dl_value )
      
   END FUNCTION var__init_2D_i2
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_3D_i2(cd_name, sd_value,                    &
         &                  id_start, id_count, id_type, td_dim,  &
         &                  td_att, sd_fill, cd_units, cd_axis,   &
         &                  cd_stdname, cd_longname,              &
         &                  cd_point, id_id, id_ew,               &
         &                  dd_scf, dd_ofs,  id_rec,              &
         &                  dd_min, dd_max,                       &
         &                  ld_contiguous, ld_shuffle,            &
         &                  ld_fletcher32, id_deflvl, id_chunksz, &
         &                  cd_interp, cd_extrap, cd_filter,      &
         &                  cd_unt, dd_unf,                       &
         &                  cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a integer(2) 3D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> array of 3 dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y','z') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] sd_value        3D array of integer(2) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] sd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i2)     , DIMENSION(:,:,:)   ,  INTENT(IN) :: sd_value
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_att
      INTEGER(i2)     ,                       INTENT(IN), OPTIONAL :: sd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                  :: il_type
      INTEGER(i4), DIMENSION(3)                    :: il_shape

      REAL(dp)                                     :: dl_fill
      REAL(dp)   , DIMENSION(:,:,:)  , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_SHORT
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_SHORT
      IF( PRESENT(sd_fill) ) dl_fill=REAL(sd_fill,dp)

      il_shape(:)=SHAPE(sd_value(:,:,:))

      ALLOCATE( dl_value( il_shape(1), &
         &                il_shape(2), &
         &                il_shape(3)) )

      DO jk=1,il_shape(3)
         DO jj=1,il_shape(2)
            DO ji=1,il_shape(1)
               dl_value(ji,jj,jk)=REAL(sd_value(ji,jj,jk),dp)
            ENDDO
         ENDDO
      ENDDO

      tf_var=var_init( cd_name, dl_value(:,:,:),           &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
      
      DEALLOCATE( dl_value )
      
   END FUNCTION var__init_3D_i2
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_i2(cd_name, sd_value,        &
         &                          id_start, id_count, id_type, td_dim, &
         &                          td_att, sd_fill, cd_units, cd_axis,&
         &                          cd_stdname, cd_longname,  &
         &                          cd_point, id_id, id_ew,   &
         &                          dd_scf, dd_ofs,  id_rec,  &
         &                          dd_min, dd_max,           &
         &                          ld_contiguous, ld_shuffle,&
         &                          ld_fletcher32, id_deflvl, id_chunksz, &
         &                          cd_interp, cd_extrap, cd_filter, &
         &                          cd_unt, dd_unf, &
         &                          cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a integer(2) 4D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> Dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y','z','t') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] sd_value        4D array of integer(2) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] sd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i2)     , DIMENSION(:,:,:,:),   INTENT(IN) :: sd_value
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_att
      INTEGER(i2)     ,                       INTENT(IN), OPTIONAL :: sd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                       :: il_type
      INTEGER(i4), DIMENSION(ip_maxdim)                 :: il_shape

      REAL(dp)                                          :: dl_fill
      REAL(dp)        , DIMENSION(:,:,:,:), ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_SHORT
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_SHORT
      IF( PRESENT(sd_fill) ) dl_fill=REAL(sd_fill,dp)

      il_shape(:)=SHAPE(sd_value(:,:,:,:))

      ALLOCATE( dl_value( il_shape(1), &
         &                il_shape(2), &
         &                il_shape(3), &
         &                il_shape(4)) )

      DO jl=1,il_shape(4)
         DO jk=1,il_shape(3)
            DO jj=1,il_shape(2)
               DO ji=1,il_shape(1)
                  dl_value(ji,jj,jk,jl)=REAL(sd_value(ji,jj,jk,jl),dp)
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      tf_var=var_init( cd_name, dl_value(:,:,:,:),         &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
      
      DEALLOCATE( dl_value )
      
   END FUNCTION var__init_i2
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_1D_i1(cd_name, bd_value,                    &
         &                  id_start, id_count, id_type, td_dim,  &
         &                  td_att, bd_fill, cd_units, cd_axis,   &
         &                  cd_stdname, cd_longname,              &
         &                  cd_point, id_id, id_ew,               &
         &                  dd_scf, dd_ofs,  id_rec,              &
         &                  dd_min, dd_max,                       &
         &                  ld_contiguous, ld_shuffle,            &
         &                  ld_fletcher32, id_deflvl, id_chunksz, &
         &                  cd_interp, cd_extrap, cd_filter,      &
         &                  cd_unt, dd_unf,                       &
         &                  cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a integer(1) 1D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('z') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] bd_value        1D array of integer(1) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] bd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i1)     , DIMENSION(:)        , INTENT(IN) :: bd_value
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      ,                       INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_att
      INTEGER(i1)     ,                       INTENT(IN), OPTIONAL :: bd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                       :: il_type
      INTEGER(i4)                                       :: il_shape

      REAL(dp)                                          :: dl_fill
      REAL(dp)        , DIMENSION(:)      , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_BYTE
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_BYTE
      IF( PRESENT(bd_fill) ) dl_fill=REAL(bd_fill,dp)

      il_shape=SIZE(bd_value(:))
      ALLOCATE( dl_value( il_shape) )

      DO ji=1,il_shape
         dl_value(ji)=REAL(bd_value(ji),dp)
      ENDDO

      tf_var=var_init( cd_name, dl_value(:),               &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
 
      DEALLOCATE( dl_value )
 
   END FUNCTION var__init_1D_i1
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_2D_i1(cd_name, bd_value,                    &
         &                  id_start, id_count, id_type, td_dim,  &
         &                  td_att, bd_fill, cd_units, cd_axis,   &
         &                  cd_stdname, cd_longname,              &
         &                  cd_point, id_id, id_ew,               &
         &                  dd_scf, dd_ofs,  id_rec,              &
         &                  dd_min, dd_max,                       &
         &                  ld_contiguous, ld_shuffle,            &
         &                  ld_fletcher32, id_deflvl, id_chunksz, &
         &                  cd_interp, cd_extrap, cd_filter,      &
         &                  cd_unt, dd_unf,                       &
         &                  cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a integer(1) 2D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> array of 2 dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] bd_value        2D array of integer(1) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] bd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i1)     , DIMENSION(:,:)     ,  INTENT(IN) :: bd_value
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_att
      INTEGER(i1)     ,                       INTENT(IN), OPTIONAL :: bd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                  :: il_type
      INTEGER(i4), DIMENSION(2)                    :: il_shape

      REAL(dp)                                     :: dl_fill
      REAL(dp)   , DIMENSION(:,:)    , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_BYTE
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_BYTE
      IF( PRESENT(bd_fill) ) dl_fill=REAL(bd_fill,dp)

      il_shape(:)=SHAPE(bd_value(:,:))

      ALLOCATE( dl_value( il_shape(1), &
         &                il_shape(2)) )

      DO jj=1,il_shape(2)
         DO ji=1,il_shape(1)
            dl_value(ji,jj)=REAL(bd_value(ji,jj),dp)
         ENDDO
      ENDDO

      tf_var=var_init( cd_name, dl_value(:,:),             &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
      
      DEALLOCATE( dl_value )
      
   END FUNCTION var__init_2D_i1
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_3D_i1(cd_name, bd_value,                    &
         &                  id_start, id_count, id_type, td_dim,  &
         &                  td_att, bd_fill, cd_units, cd_axis,   &
         &                  cd_stdname, cd_longname,              &
         &                  cd_point, id_id, id_ew,               &
         &                  dd_scf, dd_ofs,  id_rec,              &
         &                  dd_min, dd_max,                       &
         &                  ld_contiguous, ld_shuffle,            &
         &                  ld_fletcher32, id_deflvl, id_chunksz, &
         &                  cd_interp, cd_extrap, cd_filter,      &
         &                  cd_unt, dd_unf,                       &
         &                  cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a integer(1) 3D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> array of 3 dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y','z') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case.    
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] bd_value        3D array of integer(1) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] bd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i1)     , DIMENSION(:,:,:)   ,  INTENT(IN) :: bd_value
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:)       ,  INTENT(IN), OPTIONAL :: td_att
      INTEGER(i1)     ,                       INTENT(IN), OPTIONAL :: bd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                  :: il_type
      INTEGER(i4), DIMENSION(3)                    :: il_shape

      REAL(dp)                                     :: dl_fill
      REAL(dp)   , DIMENSION(:,:,:)  , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_BYTE
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_BYTE
      IF( PRESENT(bd_fill) ) dl_fill=REAL(bd_fill,dp)

      il_shape(:)=SHAPE(bd_value(:,:,:))

      ALLOCATE( dl_value( il_shape(1), &
         &                il_shape(2), &
         &                il_shape(3)) )

      DO jk=1,il_shape(3)
         DO jj=1,il_shape(2)
            DO ji=1,il_shape(1)
               dl_value(:,:,:)=REAL(bd_value(:,:,:),dp)
            ENDDO
         ENDDO
      ENDDO

      tf_var=var_init( cd_name, dl_value(:,:,:),           &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
      
      DEALLOCATE( dl_value )
      
   END FUNCTION var__init_3D_i1
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__init_i1(cd_name, bd_value,                    &
         &               id_start, id_count, id_type, td_dim,  &
         &               td_att, bd_fill, cd_units, cd_axis,   &
         &               cd_stdname, cd_longname,              &
         &               cd_point, id_id, id_ew,               &
         &               dd_scf, dd_ofs,  id_rec,              &
         &               dd_min, dd_max,                       &
         &               ld_contiguous, ld_shuffle,            &
         &               ld_fletcher32, id_deflvl, id_chunksz, &
         &               cd_interp, cd_extrap, cd_filter,      &
         &               cd_unt, dd_unf,                       &
         &               cd_namout) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief This function initialize a variable structure,
   !> with a integer(1) 4D array of value.
   !> @details 
   !> optionally could be added:<br/>
   !> - dimension structure.
   !> - attribute structure.
   !>
   !> Dimension structure is needed to put value in variable structure. 
   !> If none is given, we assume array is ordered as ('x','y','z','t') and we 
   !> use array size as lentgh dimension. 
   !>
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. Dimension structure is needed in that 
   !> case. 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015
   !> - add interp, extrap, and filter argument
   !> @date July, 2015
   !> - add unit factor (to change unit)
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !> @date February, 2019
   !> - add output name (to change name)
   !>
   !> @param[in] cd_name         variable name
   !> @param[in] bd_value        4D array of integer(1) value
   !> @param[in] id_start        index in the variable from which the 
   !> data values will be read
   !> @param[in] id_count        number of indices selected along 
   !> each dimension
   !> @param[in] id_type         variable type 
   !> @param[in] td_dim          array of dimension structure
   !> @param[in] td_att          array of attribute structure
   !> @param[in] bd_fill         fill value
   !> @param[in] cd_units        units
   !> @param[in] cd_axis         axis expected to be used
   !> @param[in] cd_stdname      variable standard name
   !> @param[in] cd_longname     variable long name
   !> @param[in] cd_point        point on Arakawa-C grid (T,U,V,F)
   !> @param[in] id_id           variable id
   !> @param[in] id_ew           east west wrap
   !> @param[in] dd_scf          scale factor
   !> @param[in] dd_ofs          add offset
   !> @param[in] id_rec          record id (for rstdimg file)
   !> @param[in] dd_min          minimum value
   !> @param[in] dd_max          maximum value
   !> @param[in] ld_contiguous   use contiguous storage or not 
   !> @param[in] ld_shuffle       shuffle filter is turned on or not
   !> @param[in] ld_fletcher32   fletcher32 filter is turned on or not
   !> @param[in] id_deflvl       deflate level from 0 to 9, 0 indicates no 
   !> deflation is in use
   !> @param[in] id_chunksz      chunk size
   !> @param[in] cd_interp       interpolation method
   !> @param[in] cd_extrap       extrapolation method
   !> @param[in] cd_filter       filter method
   !> @param[in] cd_unt          new units (linked to units factor)
   !> @param[in] dd_unf          units factor
   !> @param[in] cd_namout       output name (renamed variable)
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                       INTENT(IN) :: cd_name
      INTEGER(i1)     , DIMENSION(:,:,:,:),   INTENT(IN) :: bd_value
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_start
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_count
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_type 
      TYPE(TDIM)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_dim
      TYPE(TATT)      , DIMENSION(:),         INTENT(IN), OPTIONAL :: td_att
      INTEGER(i1)     ,                       INTENT(IN), OPTIONAL :: bd_fill
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_units
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_axis 
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_stdname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_longname
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_point
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_id 
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_ew 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_scf
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_ofs
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_rec 
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_min
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_max
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_contiguous
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_shuffle
      LOGICAL         ,                       INTENT(IN), OPTIONAL :: ld_fletcher32
      INTEGER(i4)     ,                       INTENT(IN), OPTIONAL :: id_deflvl
      INTEGER(i4)     , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: id_chunksz
      CHARACTER(LEN=*), DIMENSION(2)        , INTENT(IN), OPTIONAL :: cd_interp
      CHARACTER(LEN=*), DIMENSION(1)        , INTENT(IN), OPTIONAL :: cd_extrap
      CHARACTER(LEN=*), DIMENSION(5)        , INTENT(IN), OPTIONAL :: cd_filter
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_unt
      REAL(dp)        ,                       INTENT(IN), OPTIONAL :: dd_unf
      CHARACTER(LEN=*),                       INTENT(IN), OPTIONAL :: cd_namout

      ! function
      TYPE(TVAR)                                         :: tf_var

      ! local variable
      INTEGER(i4)                                       :: il_type
      INTEGER(i4), DIMENSION(ip_maxdim)                 :: il_shape

      REAL(dp)                                          :: dl_fill
      REAL(dp)        , DIMENSION(:,:,:,:), ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      ! clean variable
      CALL var_clean(tf_var)      

      il_type=NF90_BYTE
      IF( PRESENT(id_type) ) il_type=id_type

      dl_fill=NF90_FILL_BYTE
      IF( PRESENT(bd_fill) ) dl_fill=REAL(bd_fill,dp)

      il_shape(:)=SHAPE(bd_value(:,:,:,:))

      ALLOCATE( dl_value( il_shape(1), &
         &                il_shape(2), &
         &                il_shape(3), &
         &                il_shape(4)) )

      DO jl=1,il_shape(4)
         DO jk=1,il_shape(3)
            DO jj=1,il_shape(2)
               DO ji=1,il_shape(1)
                  dl_value(ji,jj,jk,jl)=REAL(bd_value(ji,jj,jk,jl),dp)
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      tf_var=var_init( cd_name, dl_value(:,:,:,:),         &
         &             id_start=id_start,                  &
         &             id_count=id_count,                  &
         &             id_type=il_type,                    &
         &             td_dim=td_dim, td_att=td_att,       &
         &             dd_fill=dl_fill,                    &
         &             cd_units=cd_units,                  &
         &             cd_axis=cd_axis,                    &
         &             cd_stdname=cd_stdname,              &
         &             cd_longname=cd_longname,            &
         &             cd_point=cd_point, id_id=id_id,     &
         &             id_ew=id_ew, dd_scf=dd_scf,         &
         &             dd_ofs=dd_ofs,  id_rec=id_rec,      &
         &             dd_min=dd_min, dd_max=dd_max,       &
         &             ld_contiguous=ld_contiguous,        &
         &             ld_shuffle=ld_shuffle,              &
         &             ld_fletcher32=ld_fletcher32,        &
         &             id_deflvl=id_deflvl,                &
         &             id_chunksz=id_chunksz(:),           &
         &             cd_interp=cd_interp(:),             &
         &             cd_extrap=cd_extrap(:),             &
         &             cd_filter=cd_filter(:),             &
         &             cd_unt=cd_unt, dd_unf=dd_unf,       &
         &             cd_namout=cd_namout )
      
      DEALLOCATE( dl_value )
      
   END FUNCTION var__init_i1
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var_concat(td_var1, td_var2, id_dim) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief  This function concatenate variable value following id_dim direction.
   !> 
   !> @details
   !> By default variable are concatenate following time dimension. To
   !> concatenate following another dimension, specify id_dim=x where x is the
   !> dimension number (jp_I, jp_J,jp_K, jp_L). 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_var1   variable structure
   !> @param[in] td_var2   variable structure
   !> @param[in] DIM       dimension following which concatenate
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) , INTENT(IN) :: td_var1
      TYPE(TVAR) , INTENT(IN) :: td_var2
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_dim

      ! function
      TYPE(TVAR)              :: tf_var

      ! local variable
      INTEGER(i4) :: il_dim
      !----------------------------------------------------------------
      il_dim=4
      IF( PRESENT(id_dim) ) il_dim=id_dim

      IF( .NOT. ASSOCIATED(td_var1%d_value) )THEN
         CALL logger_error("VAR CONCAT: no value associated to variable "//&
            &  TRIM(td_var1%c_name) )
      ELSEIF( .NOT. ASSOCIATED(td_var2%d_value) )THEN
         CALL logger_error("VAR CONCAT: no value associated to variable "//&
            &  TRIM(td_var2%c_name) )
      ELSEIF( il_dim < 0 .OR. il_dim > 4 )THEN
         CALL logger_error("VAR CONCAT: invalid concatenate dimension ")
      ELSE
         ! check other dimension
         SELECT CASE(il_dim)
         CASE(jp_I)
            tf_var=var__concat_i(td_var1, td_var2)
         CASE(jp_J)
            tf_var=var__concat_j(td_var1, td_var2)
         CASE(jp_K)
            tf_var=var__concat_k(td_var1, td_var2)
         CASE(jp_L)
            tf_var=var__concat_l(td_var1, td_var2)
         END SELECT
      ENDIF

   END FUNCTION var_concat
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__concat_i(td_var1, td_var2) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief  This function concatenate variable value following i-direction.
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - decompose array copy on each dimension
   !>
   !> @param[in] td_var1   variable structure
   !> @param[in] td_var2   variable structure
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) , INTENT(IN) :: td_var1
      TYPE(TVAR) , INTENT(IN) :: td_var2

      ! function
      TYPE(TVAR)              :: tf_var

      ! local variable
      TYPE(TVAR)        :: tl_var
      CHARACTER(LEN=lc) :: cl_tmp

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------
      IF( .NOT. td_var1%t_dim(1)%l_use .OR. &
         &.NOT. td_var1%t_dim(1)%l_use )THEN

         CALL logger_error("VAR CONCAT: can not concatenate variable "//&
            &              TRIM(td_var1%c_name)//" on an unused dimension I")
   
      ELSEIF( ANY(td_var1%t_dim(2:4)%i_len /=  td_var2%t_dim(2:4)%i_len) )THEN

         cl_tmp='('//":"//","//&
            &   TRIM(fct_str(td_var1%t_dim(2)%i_len))//','//&
            &   TRIM(fct_str(td_var1%t_dim(3)%i_len))//','//&
            &   TRIM(fct_str(td_var1%t_dim(4)%i_len))//')'
         CALL logger_debug("VAR CONCAT: first variable dimensions "//&
            &   TRIM(cl_tmp) )
         cl_tmp='('//":"//","//&
            &   TRIM(fct_str(td_var2%t_dim(2)%i_len))//','//&
            &   TRIM(fct_str(td_var2%t_dim(3)%i_len))//','//&
            &   TRIM(fct_str(td_var2%t_dim(4)%i_len))//')'         
         CALL logger_debug("VAR CONCAT: second variable dimensions "//&
            &   TRIM(cl_tmp) )

         CALL logger_error("VAR CONCAT: dimension not conform")

      ELSE
         tl_var=var_copy(td_var1)

         DEALLOCATE(tl_var%d_value)
         ! change dimension length
         tl_var%t_dim(1)%i_len=td_var1%t_dim(1)%i_len+td_var2%t_dim(1)%i_len
         
         ALLOCATE(tl_var%d_value(tl_var%t_dim(1)%i_len, &
            &                    tl_var%t_dim(2)%i_len, &
            &                    tl_var%t_dim(3)%i_len, &
            &                    tl_var%t_dim(4)%i_len) )

         ! copy first variable value
         DO jl=1,tl_var%t_dim(4)%i_len
            DO jk=1,tl_var%t_dim(3)%i_len
               DO jj=1,tl_var%t_dim(2)%i_len
                  DO ji=1,td_var1%t_dim(1)%i_len
                     tl_var%d_value(ji,jj,jk,jl) = td_var1%d_value(ji,jj,jk,jl)
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
         !            tl_var%d_value(1:td_var1%t_dim(1)%i_len,:,:,:) = &
         !&  td_var1%d_value(:,:,:,:)

         ! copy second variable value
         DO jl=1,tl_var%t_dim(4)%i_len
            DO jk=1,tl_var%t_dim(3)%i_len
               DO jj=1,tl_var%t_dim(2)%i_len
                  DO ji=1,td_var2%t_dim(1)%i_len
                     tl_var%d_value(ji+td_var1%t_dim(1)%i_len,jj,jk,jl) = td_var2%d_value(ji,jj,jk,jl)
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
         !tl_var%d_value(td_var1%t_dim(1)%i_len+1:tl_var%t_dim(1)%i_len,:,:,:)=&
         !&  td_var2%d_value(:,:,:,:)

         ! save result
         tf_var=var_copy(tl_var)

         ! clean
         CALL var_clean(tl_var)
      ENDIF

   END FUNCTION var__concat_i
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__concat_j(td_var1, td_var2) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief  This function concatenate variable value following j-direction.
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - decompose array copy on each dimension
   !>
   !> @param[in] td_var1   variable structure
   !> @param[in] td_var2   variable structure
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) , INTENT(IN) :: td_var1
      TYPE(TVAR) , INTENT(IN) :: td_var2

      ! function
      TYPE(TVAR)              :: tf_var

      ! local variable
      TYPE(TVAR)        :: tl_var
      CHARACTER(LEN=lc) :: cl_tmp

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------
      IF( .NOT. td_var1%t_dim(2)%l_use .OR. &
      &   .NOT. td_var1%t_dim(2)%l_use )THEN

         CALL logger_error("VAR CONCAT: can not concatenate variable "//&
         &  TRIM(td_var1%c_name)//" on an unused dimension J")      

      ELSEIF(     td_var1%t_dim(1)%i_len   /=  td_var2%t_dim(1)%i_len  .OR. &
      &   ANY(td_var1%t_dim(3:4)%i_len /=  td_var2%t_dim(3:4)%i_len) )THEN

         cl_tmp='('//&
            &    TRIM(fct_str(td_var1%t_dim(1)%i_len))//','//&
            &    ":"//','//&
            &    TRIM(fct_str(td_var1%t_dim(3)%i_len))//','//&
            &    TRIM(fct_str(td_var1%t_dim(4)%i_len))//')'
         CALL logger_debug("VAR CONCAT: first variable dimensions "//&
            &    TRIM(cl_tmp) )
         cl_tmp='('//&
            &    TRIM(fct_str(td_var1%t_dim(1)%i_len))//','//&
            &    ":"//','//&
            &    TRIM(fct_str(td_var2%t_dim(3)%i_len))//','//&
            &    TRIM(fct_str(td_var2%t_dim(4)%i_len))//')'         
         CALL logger_debug("VAR CONCAT: second variable dimensions "//&
            &    TRIM(cl_tmp) )

         CALL logger_error("VAR CONCAT: dimension not conform")

      ELSE
         tl_var=var_copy(td_var1)

         DEALLOCATE(tl_var%d_value)
         ! change dimension length
         tl_var%t_dim(2)%i_len=td_var1%t_dim(2)%i_len+td_var2%t_dim(2)%i_len
         
         ALLOCATE(tl_var%d_value(tl_var%t_dim(1)%i_len, &
            &                    tl_var%t_dim(2)%i_len, &
            &                    tl_var%t_dim(3)%i_len, &
            &                    tl_var%t_dim(4)%i_len) )

         ! copy first variable value
         DO jl=1,tl_var%t_dim(4)%i_len
            DO jk=1,tl_var%t_dim(3)%i_len
               DO jj=1,td_var1%t_dim(2)%i_len
                  DO ji=1,tl_var%t_dim(1)%i_len
                     tl_var%d_value(ji,jj,jk,jl) = td_var1%d_value(ji,jj,jk,jl)
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
         !         tl_var%d_value(:,1:td_var1%t_dim(2)%i_len,:,:)= &
         !&  td_var1%d_value(:,:,:,:)

         ! copy second variable value
         DO jl=1,tl_var%t_dim(4)%i_len
            DO jk=1,tl_var%t_dim(3)%i_len
               DO jj=1,td_var2%t_dim(2)%i_len
                  DO ji=1,tl_var%t_dim(1)%i_len
                     tl_var%d_value(ji,jj+td_var2%t_dim(2)%i_len,jk,jl) = td_var2%d_value(ji,jj,jk,jl)
                  ENDDO
               ENDDO
            ENDDO
         ENDDO    
         !         tl_var%d_value(:,td_var1%t_dim(2)%i_len+1:tl_var%t_dim(2)%i_len,:,:)=&
         !&  td_var2%d_value(:,:,:,:)

         ! save result
         tf_var=var_copy(tl_var)

         ! clean
         CALL var_clean(tl_var)
      ENDIF

   END FUNCTION var__concat_j
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__concat_k(td_var1, td_var2) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief  This function concatenate variable value following k-direction.
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - decompose array copy on each dimension
   !
   !> @param[in] td_var1   variable structure
   !> @param[in] td_var2   variable structure
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) , INTENT(IN) :: td_var1
      TYPE(TVAR) , INTENT(IN) :: td_var2

      ! function
      TYPE(TVAR)              :: tf_var

      ! local variable
      TYPE(TVAR)        :: tl_var
      CHARACTER(LEN=lc) :: cl_tmp

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------
      IF( .NOT. td_var1%t_dim(3)%l_use .OR. &
      &   .NOT. td_var1%t_dim(3)%l_use )THEN

         CALL logger_error("VAR CONCAT: can not concatenate variable "//&
         &  TRIM(td_var1%c_name)//" on an unused dimension K")      

      ELSEIF(     td_var1%t_dim(4)%i_len   /=  td_var2%t_dim(4)%i_len  .OR. &
      &   ANY(td_var1%t_dim(1:2)%i_len /=  td_var2%t_dim(1:2)%i_len) )THEN

         cl_tmp='('//&
            &    TRIM(fct_str(td_var1%t_dim(1)%i_len))//','//&
            &    TRIM(fct_str(td_var1%t_dim(2)%i_len))//','//&
            &    ":"//','//&
            &    TRIM(fct_str(td_var1%t_dim(4)%i_len))//')'
         CALL logger_debug("VAR CONCAT: first variable dimensions "//&
            &    TRIM(cl_tmp) )
         cl_tmp='('//&
            &    TRIM(fct_str(td_var1%t_dim(1)%i_len))//','//&
            &    TRIM(fct_str(td_var2%t_dim(2)%i_len))//','//&
            &    ":"//','//&
            &    TRIM(fct_str(td_var2%t_dim(4)%i_len))//')'         
         CALL logger_debug("VAR CONCAT: second variable dimensions "//&
            &    TRIM(cl_tmp) )

         CALL logger_error("VAR CONCAT: dimension not conform")

      ELSE
         tl_var=var_copy(td_var1)

         DEALLOCATE(tl_var%d_value)
         ! change dimension length
         tl_var%t_dim(3)%i_len=td_var1%t_dim(3)%i_len+td_var2%t_dim(3)%i_len
         
         ALLOCATE(tl_var%d_value(tl_var%t_dim(1)%i_len, &
            &                    tl_var%t_dim(2)%i_len, &
            &                    tl_var%t_dim(3)%i_len, &
            &                    tl_var%t_dim(4)%i_len) )

         ! copy first variable value
         DO jl=1,tl_var%t_dim(4)%i_len
            DO jk=1,td_var1%t_dim(3)%i_len
               DO jj=1,tl_var%t_dim(2)%i_len
                  DO ji=1,tl_var%t_dim(1)%i_len
                     tl_var%d_value(ji,jj,jk,jl) = td_var1%d_value(ji,jj,jk,jl)
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
         !tl_var%d_value(:,:,1:td_var1%t_dim(3)%i_len,:)= &
         ! &  td_var1%d_value(:,:,:,:)

         ! copy second variable value
         DO jl=1,tl_var%t_dim(4)%i_len
            DO jk=1,td_var2%t_dim(3)%i_len
               DO jj=1,tl_var%t_dim(2)%i_len
                  DO ji=1,tl_var%t_dim(1)%i_len
                     tl_var%d_value(ji,jj,jk+td_var1%t_dim(3)%i_len,jl) = td_var2%d_value(ji,jj,jk,jl)
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
         !tl_var%d_value(:,:,td_var1%t_dim(3)%i_len+1:tl_var%t_dim(3)%i_len,:)=&
         !&  td_var2%d_value(:,:,:,:)

         ! save result
         tf_var=var_copy(tl_var)

         ! clean
         CALL var_clean(tl_var)
      ENDIF

   END FUNCTION var__concat_k
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__concat_l(td_var1, td_var2) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief  This function concatenate variable value following l-direction.
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - decompose array copy on each dimension
   !>
   !> @param[in] td_var1   variable structure
   !> @param[in] td_var2   variable structure
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) , INTENT(IN) :: td_var1
      TYPE(TVAR) , INTENT(IN) :: td_var2

      ! function
      TYPE(TVAR)              :: tf_var

      ! local variable
      TYPE(TVAR)        :: tl_var
      CHARACTER(LEN=lc) :: cl_tmp

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------
      IF( .NOT. td_var1%t_dim(4)%l_use .OR. &
      &   .NOT. td_var1%t_dim(4)%l_use )THEN

         CALL logger_error("VAR CONCAT: can not concatenate variable "//&
         &  TRIM(td_var1%c_name)//" on an unused dimension L")      

      ELSEIF( ANY(td_var1%t_dim(1:3)%i_len /=  td_var2%t_dim(1:3)%i_len) )THEN

         cl_tmp='('//&
            &    TRIM(fct_str(td_var1%t_dim(1)%i_len))//','//&
            &    TRIM(fct_str(td_var1%t_dim(2)%i_len))//','//&
            &    TRIM(fct_str(td_var1%t_dim(3)%i_len))//','//&
            &    ":"//','//')'
         CALL logger_debug("VAR CONCAT: first variable dimensions "//&
            &   TRIM(cl_tmp) )
         cl_tmp='('//&
            &    TRIM(fct_str(td_var1%t_dim(1)%i_len))//','//&
            &    TRIM(fct_str(td_var2%t_dim(2)%i_len))//','//&
            &    TRIM(fct_str(td_var2%t_dim(3)%i_len))//','//&
            &    ":"//','//')'         
         CALL logger_debug("VAR CONCAT: second variable dimensions "//&
            &    TRIM(cl_tmp) )

         CALL logger_error("VAR CONCAT: dimension not conform")

      ELSE
         tl_var=var_copy(td_var1)

         DEALLOCATE(tl_var%d_value)
         ! change dimension length
         tl_var%t_dim(4)%i_len=td_var1%t_dim(4)%i_len+td_var2%t_dim(4)%i_len
         
         ALLOCATE(tl_var%d_value(tl_var%t_dim(1)%i_len, &
            &                    tl_var%t_dim(2)%i_len, &
            &                    tl_var%t_dim(3)%i_len, &
            &                    tl_var%t_dim(4)%i_len) )

         ! copy first variable value
         DO jl=1,td_var1%t_dim(4)%i_len
            DO jk=1,tl_var%t_dim(3)%i_len
               DO jj=1,tl_var%t_dim(2)%i_len
                  DO ji=1,tl_var%t_dim(1)%i_len
                     tl_var%d_value(ji,jj,jk,jl) = td_var1%d_value(ji,jj,jk,jl)
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
         !tl_var%d_value(:,:,1:td_var1%t_dim(4)%i_len,:)= &
         !&  td_var1%d_value(:,:,:,:)

         ! copy second variable value
         DO jl=1,td_var2%t_dim(4)%i_len
            DO jk=1,tl_var%t_dim(3)%i_len
               DO jj=1,tl_var%t_dim(2)%i_len
                  DO ji=1,tl_var%t_dim(1)%i_len
                     tl_var%d_value(ji,jj,jk,jl+td_var2%t_dim(4)%i_len) = td_var2%d_value(ji,jj,jk,jl)
                  ENDDO
               ENDDO
            ENDDO
         ENDDO
         !tl_var%d_value(:,:,td_var1%t_dim(4)%i_len+1:tl_var%t_dim(4)%i_len,:)=&
         !&  td_var2%d_value(:,:,:,:)

         ! save result
         tf_var=var_copy(tl_var)

         ! clean
         CALL var_clean(tl_var)
      ENDIF

   END FUNCTION var__concat_l
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__add_att_arr(td_var, td_att)
   !-------------------------------------------------------------------
   !> @brief This subroutine add an array of attribute structure 
   !> in a variable structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015 
   !> - add all element of the array in the same time
   !> @date January, 2019
   !> - deallocate attribute strucure whatever happens
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] td_att    array of attribute structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR),               INTENT(INOUT) :: td_var
      TYPE(TATT), DIMENSION(:), INTENT(IN)    :: td_att

      ! local variable
      INTEGER(i4) :: il_natt
      INTEGER(i4) :: il_status
      INTEGER(i4) :: il_ind
      TYPE(TATT), DIMENSION(:), ALLOCATABLE :: tl_att

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      il_natt=SIZE(td_att(:))

      IF( td_var%i_natt > 0 )THEN
      ! already other attribute in variable structure
         ALLOCATE( tl_att(td_var%i_natt), stat=il_status )
         IF(il_status /= 0 )THEN

            CALL logger_error( &
               &  " VAR ADD ATT: not enough space to put attributes from "//&
               &  TRIM(td_var%c_name)//" in temporary attribute structure")

         ELSE

            ! save temporary global attribute's variable structure
            tl_att(:)=att_copy(td_var%t_att(:))

            CALL att_clean(td_var%t_att(:))
            DEALLOCATE( td_var%t_att )
            ALLOCATE( td_var%t_att(td_var%i_natt+il_natt), stat=il_status )
            IF(il_status /= 0 )THEN

               CALL logger_error( &
                  &  " VAR ADD ATT: not enough space to put attributes "//&
                  &  "in variable structure "//TRIM(td_var%c_name) )

            ENDIF

            ! copy attribute in variable before
            td_var%t_att(1:td_var%i_natt)=att_copy(tl_att(:))

            ! clean
            CALL att_clean(tl_att(:))
         ENDIF
         DEALLOCATE(tl_att)

      ELSE
      ! no attribute in variable structure
         IF( ASSOCIATED(td_var%t_att) )THEN
            CALL att_clean(td_var%t_att(:))
            DEALLOCATE(td_var%t_att)
         ENDIF
         ALLOCATE( td_var%t_att(td_var%i_natt+il_natt), stat=il_status )
         IF(il_status /= 0 )THEN

            CALL logger_error( &
               &  " VAR ADD ATT: not enough space to put attributes "//&
               &  "in variable structure "//TRIM(td_var%c_name) )

         ENDIF
      ENDIF

      ALLOCATE( tl_att(il_natt) )
      tl_att(:)=att_copy(td_att(:))

      ! check if attribute already in variable structure
      DO ji=1,il_natt
         il_ind=0
         il_ind=att_get_index( td_var%t_att(:), tl_att(ji)%c_name )
         IF( il_ind /= 0 )THEN
            CALL logger_error( &
               &  " VAR ADD ATT: attribute "//TRIM(tl_att(ji)%c_name)//&
               &  ", already in variable "//TRIM(td_var%c_name) )
            CALL att_clean(tl_att(ji))
         ENDIF
      ENDDO

      ! add new attributes
      td_var%t_att(td_var%i_natt+1:td_var%i_natt+il_natt)=att_copy(tl_att(:))

      DEALLOCATE(tl_att)

      DO ji=1,il_natt
         ! highlight some attribute
         IF( ASSOCIATED(td_var%t_att(td_var%i_natt+ji)%d_value) .OR. &
           & td_var%t_att(td_var%i_natt+ji)%c_value /= 'none' )THEN
            SELECT CASE(TRIM(td_var%t_att(td_var%i_natt+ji)%c_name))

               CASE("add_offset")
                  td_var%d_ofs = td_var%t_att(td_var%i_natt+ji)%d_value(1)
               CASE("scale_factor")
                  td_var%d_scf = td_var%t_att(td_var%i_natt+ji)%d_value(1)
               CASE("_FillValue")
                  td_var%d_fill = td_var%t_att(td_var%i_natt+ji)%d_value(1)
               CASE("ew_overlap")
                  td_var%i_ew = INT(td_var%t_att(td_var%i_natt+ji)%d_value(1),i4) 
               CASE("standard_name")
                  td_var%c_stdname = TRIM(td_var%t_att(td_var%i_natt+ji)%c_value)
               CASE("long_name")
                  td_var%c_longname = TRIM(td_var%t_att(td_var%i_natt+ji)%c_value)
               CASE("units")
                  td_var%c_units = TRIM(td_var%t_att(td_var%i_natt+ji)%c_value)
               CASE("grid_point")
                  td_var%c_point = TRIM(td_var%t_att(td_var%i_natt+ji)%c_value)

            END SELECT
         ENDIF
      ENDDO

      ! update number of attribute
      td_var%i_natt=td_var%i_natt+il_natt


   END SUBROUTINE var__add_att_arr
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__add_att_unit(td_var, td_att)
   !-------------------------------------------------------------------
   !> @brief This subroutine add an attribute structure 
   !> in a variable structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015 
   !> - use var__add_att_arr subroutine
   !> @date January, 2019
   !> - clean attribute strucure
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] td_att    attribute structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR), INTENT(INOUT) :: td_var
      TYPE(TATT), INTENT(IN)    :: td_att

      ! local variable
      TYPE(TATT), DIMENSION(1) :: tl_att

      ! loop indices
      !----------------------------------------------------------------

      ! copy structure in an array
      tl_att(1)=att_copy(td_att)

      ! 
      CALL var_add_att( td_var, tl_att(:) )

      ! clean
      CALL att_clean(tl_att)

   END SUBROUTINE var__add_att_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__del_att_name(td_var, cd_name)
   !-------------------------------------------------------------------
   !> @brief This subroutine delete an attribute 
   !> from variable structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date February, 2015 
   !> - define local attribute structure to avoid mistake 
   !> with pointer
   !> @date January, 2019
   !> - clean attribute strucure
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] cd_name   attribute name
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR)      , INTENT(INOUT) :: td_var
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_name

      ! local variable
      INTEGER(i4) :: il_ind

      TYPE(TATT)  :: tl_att
      ! loop indices
      !----------------------------------------------------------------

      ! check if attribute already in variable structure
      il_ind=0
      IF( ASSOCIATED(td_var%t_att) )THEN
         il_ind=att_get_index( td_var%t_att(:), TRIM(cd_name) )
      ENDIF

      IF( il_ind == 0 )THEN

         CALL logger_debug( &
         &  " VAR DEL ATT: no attribute "//TRIM(cd_name)//&
         &  ", in variable "//TRIM(td_var%c_name) )

      ELSE
         
         tl_att=att_copy(td_var%t_att(il_ind))
         CALL var_del_att(td_var, tl_att)
         ! clean
         CALL att_clean(tl_att)
      ENDIF

   END SUBROUTINE var__del_att_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__del_att_str(td_var, td_att)
   !-------------------------------------------------------------------
   !> @brief This subroutine delete an attribute 
   !> from variable structure.
   !>
   !> @author J.Paul
   !> @date November, 2013- Initial Version
   !> @date February, 2015 
   !> - delete highlight attribute too, when attribute 
   !> is deleted
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] td_att    attribute structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR), INTENT(INOUT) :: td_var
      TYPE(TATT), INTENT(IN)    :: td_att

      ! local variable
      INTEGER(i4) :: il_status
      INTEGER(i4) :: il_ind
      TYPE(TATT), DIMENSION(:), ALLOCATABLE :: tl_att

      ! loop indices
      !----------------------------------------------------------------

      ! check if attribute already in variable structure
      il_ind=0
      IF( ASSOCIATED(td_var%t_att) )THEN
         il_ind=att_get_index( td_var%t_att(:), td_att%c_name )
      ENDIF

      IF( il_ind == 0 )THEN

         CALL logger_debug( &
         &  " VAR DEL ATT: no attribute "//TRIM(td_att%c_name)//&
         &  ", in variable "//TRIM(td_var%c_name) )

      ELSE
         
         CALL logger_trace( &
         &  " VAR DEL ATT: del attribute "//TRIM(td_att%c_name)//&
         &  ", in var "//TRIM(td_var%c_name) )

         IF( td_var%i_natt == 1 )THEN

            CALL att_clean(td_var%t_att(:))
            DEALLOCATE(td_var%t_att)

            ! new number of attribute in variable
            td_var%i_natt=td_var%i_natt-1

         ELSE
            ALLOCATE( tl_att(td_var%i_natt-1), stat=il_status )
            IF(il_status /= 0 )THEN

               CALL logger_error( &
               &  " VAR ADD ATT: not enough space to put attributes from "//&
               &  TRIM(td_var%c_name)//" in temporary attribute structure")

            ELSE

               ! save temporary global attribute's variable structure
               tl_att(1:il_ind-1)=att_copy(td_var%t_att(1:il_ind-1))
               IF( il_ind < td_var%i_natt )THEN
                  tl_att(il_ind:)=att_copy(td_var%t_att(il_ind+1:))
               ENDIF

               CALL att_clean(td_var%t_att(:))
               DEALLOCATE( td_var%t_att )

               ! new number of attribute in variable
               td_var%i_natt=td_var%i_natt-1

               ALLOCATE( td_var%t_att(td_var%i_natt), stat=il_status )
               IF(il_status /= 0 )THEN

                  CALL logger_error( &
                  &  " VAR ADD ATT: not enough space to put attributes "//&
                  &  "in variable structure "//TRIM(td_var%c_name) )

               ENDIF

               ! copy attribute in variable before
               td_var%t_att(1:td_var%i_natt)=att_copy(tl_att(:))

               ! clean
               CALL att_clean(tl_att(:))
            ENDIF 
            DEALLOCATE(tl_att)
         ENDIF

         ! highlight attribute
         SELECT CASE( TRIM(td_att%c_name) )

            CASE("add_offset")
               td_var%d_ofs = 0._dp
            CASE("scale_factor")
               td_var%d_scf = 1._dp
            CASE("_FillValue")
               td_var%d_fill = 0._dp
            CASE("ew_overlap")
               td_var%i_ew = -1 
            CASE("standard_name")
               td_var%c_stdname = ''
            CASE("long_name")
               td_var%c_longname = ''
            CASE("units")
               td_var%c_units = ''
            CASE("grid_point")
               td_var%c_point = ''

         END SELECT

      ENDIF

   END SUBROUTINE var__del_att_str
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var_move_att(td_var, td_att)
   !-------------------------------------------------------------------
   !> @brief This subroutine move an attribute structure 
   !> from variable structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] td_att    attribute structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR), INTENT(INOUT) :: td_var
      TYPE(TATT), INTENT(IN)    :: td_att

      ! local variable
      TYPE(TATT) :: tl_att

      !----------------------------------------------------------------
      ! copy attribute
      tl_att=att_copy(td_att)

      ! remove attribute with same name
      CALL var_del_att(td_var, tl_att)

      ! add new attribute
      CALL var_add_att(td_var, tl_att)

      ! clean
      CALL att_clean(tl_att)

   END SUBROUTINE var_move_att
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__add_dim_arr(td_var, td_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine add an array of dimension structure in a variable 
   !> structure.
   !> - number of dimension in variable can't be greater than 4
   !> - dimension can't be already uses in variable structure
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] td_dim    dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR),               INTENT(INOUT) :: td_var
      TYPE(TDIM), DIMENSION(:), INTENT(IN)    :: td_dim

      ! local variable
      INTEGER(i4) :: il_ndim

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      il_ndim=SIZE(td_dim(:))
      IF( il_ndim <= ip_maxdim )THEN

         DO ji=1,il_ndim
            CALL var_add_dim(td_var, td_dim(ji))
         ENDDO

      ELSE
         CALL logger_error( &
         &  " VAR ADD DIM: too much dimension to put in structure "//&
         &  "("//TRIM(fct_str(il_ndim))//")" )
      ENDIF

   END SUBROUTINE var__add_dim_arr
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__add_dim_unit(td_var, td_dim) 
   !-------------------------------------------------------------------
   !> @brief This subroutine add one dimension in a variable 
   !> structure.
   !> @details
   !> - number of dimension in variable can't be greater than 4
   !> - dimension can't be already uses in variable structure 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] td_dim    dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR)      , INTENT(INOUT) :: td_var
      TYPE(TDIM)      , INTENT(IN   ) :: td_dim

      ! local variable
      INTEGER(i4) :: il_ind
      !----------------------------------------------------------------

      IF( td_var%i_ndim <= ip_maxdim )THEN

         ! check if dimension already used in variable structure
         il_ind=SCAN(TRIM(cp_dimorder),TRIM(td_dim%c_sname))
         IF( il_ind == 0 )THEN
            CALL logger_warn( &
            &  " VAR ADD DIM: dimension "//TRIM(td_dim%c_name)//&
            &  ", short name "//TRIM(td_dim%c_sname)//&
            &  ", will not be added in variable "//TRIM(td_var%c_name) )
         ELSEIF( td_var%t_dim(il_ind)%l_use )THEN
            CALL logger_error( &
            &  " VAR ADD DIM: dimension "//TRIM(td_dim%c_name)//&
            &  ", short name "//TRIM(td_dim%c_sname)//&
            &  ", already used in variable "//TRIM(td_var%c_name) )
         ELSE

            ! back to disorder dimension array 
            CALL dim_disorder(td_var%t_dim(:))

            ! add new dimension
            td_var%t_dim(td_var%i_ndim+1)=dim_copy(td_dim)

            ! update number of attribute
            td_var%i_ndim=COUNT(td_var%t_dim(:)%l_use)

         ENDIF
         ! reorder dimension to ('x','y','z','t')
         CALL dim_reorder(td_var%t_dim(:))

      ELSE
         CALL logger_error( &
         &  " VAR ADD DIM: too much dimension in variable "//&
         &  TRIM(td_var%c_name)//" ("//TRIM(fct_str(td_var%i_ndim))//")")
      ENDIF

   END SUBROUTINE var__add_dim_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var_del_dim(td_var, td_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine delete a dimension structure in a variable 
   !> structure.
   !>
   !> @warning delete variable value too.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] td_dim    dimension structure
   !-------------------------------------------------------------------
      
      IMPLICIT NONE
      
      ! Argument      
      TYPE(TVAR)      , INTENT(INOUT) :: td_var
      TYPE(TDIM)      , INTENT(IN   ) :: td_dim

      ! local variable
      INTEGER(i4) :: il_ind
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_shape

      TYPE(TDIM)  :: tl_dim ! empty dimension structure
      !----------------------------------------------------------------

      IF( td_var%i_ndim <= ip_maxdim )THEN

         CALL logger_trace( &
         &  " VAR DEL DIM: delete dimension "//TRIM(td_dim%c_name)//&
         &  ", short name "//TRIM(td_dim%c_sname)//&
         &  ", in variable "//TRIM(td_var%c_name) )
         
         ! check if dimension already in variable structure
         il_ind=SCAN(TRIM(cp_dimorder),TRIM(td_dim%c_sname))

         ! replace dimension by empty one
         td_var%t_dim(il_ind)=dim_copy(tl_dim)

         ! update number of dimension
         td_var%i_ndim=COUNT(td_var%t_dim(:)%l_use)

         ! remove variable value using this dimension
         IF( ASSOCIATED(td_var%d_value) )THEN
            il_shape(:)=SHAPE(td_var%d_value(:,:,:,:))
            IF(il_shape(il_ind)/=td_dim%i_len)THEN
               CALL logger_warn("VAR DEL DIM: remove value of variable "//&
               &  TRIM(td_var%c_name) )
               CALL var_del_value(td_var)
            ENDIF
         ENDIF

         ! reorder dimension to ('x','y','z','t')
         CALL dim_reorder(td_var%t_dim)

      ELSE
         CALL logger_error( &
         &  " VAR DEL DIM: too much dimension in variable "//&
         &  TRIM(td_var%c_name)//" ("//TRIM(fct_str(td_var%i_ndim))//")")
      ENDIF

   END SUBROUTINE var_del_dim
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var_move_dim(td_var, td_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine move a dimension structure 
   !> in variable structure.
   !>
   !> @warning 
   !> - dimension order could be changed
   !> - delete variable value 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] td_dim    dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR)      , INTENT(INOUT) :: td_var
      TYPE(TDIM)      , INTENT(IN   ) :: td_dim

      ! local variable
      INTEGER(i4) :: il_ind
      INTEGER(i4) :: il_dimid
      !----------------------------------------------------------------

      IF( td_var%i_ndim <= ip_maxdim )THEN

         ! check if dimension already in mpp structure
         il_ind=dim_get_index(td_var%t_dim(:), td_dim%c_name, td_dim%c_sname)
         IF( il_ind /= 0 )THEN

            il_dimid=td_var%t_dim(il_ind)%i_id
            ! replace dimension
            td_var%t_dim(il_ind)=dim_copy(td_dim)
            td_var%t_dim(il_ind)%i_id=il_dimid
            td_var%t_dim(il_ind)%l_use=.TRUE.

         ELSE
            CALL var_add_dim(td_var, td_dim)
         ENDIF

      ELSE
         CALL logger_error( &
         &  "VAR MOVE DIM: too much dimension in variale "//&
         &  TRIM(td_var%c_name)//" ("//TRIM(fct_str(td_var%i_ndim))//")")
      ENDIF

   END SUBROUTINE var_move_dim
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__print_arr(td_var)
   !-------------------------------------------------------------------
   !> @brief This subroutine print informations of an array of variables. 
   !>
   !> @author J.Paul
   !> @date June, 2014 - Initial Version
   !>
   !> @param[in] td_var array of variables structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR), DIMENSION(:), INTENT(IN) :: td_var

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      DO ji=1,SIZE(td_var(:))
         CALL var_print(td_var(ji))
      ENDDO

   END SUBROUTINE var__print_arr
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__print_unit(td_var, ld_more)
   !-------------------------------------------------------------------
   !> @brief This subroutine print variable information.</br/>
   !> @details
   !> If ld_more is TRUE (default), print information about variable dimensions
   !> and variable attributes.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_var    variable structure
   !> @param[in] ld_more   print more infomration about variable
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR), INTENT(IN) :: td_var
      LOGICAL,    INTENT(IN), OPTIONAL :: ld_more

      ! local vairbale
      CHARACTER(LEN=lc) :: cl_type
      REAL(dp)          :: dl_min
      REAL(dp)          :: dl_max
      LOGICAL           :: ll_more

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ll_more=.TRUE.
      IF( PRESENT(ld_more) )THEN
         ll_more=ld_more
      ENDIF

      SELECT CASE( td_var%i_type )

         CASE(NF90_CHAR)
            cl_type='CHAR'
         CASE(NF90_BYTE)
            cl_type='BYTE'
         CASE(NF90_SHORT)
            cl_type='SHORT'
         CASE(NF90_INT)
            cl_type='INT'
         CASE(NF90_FLOAT)
            cl_type='FLOAT'
         CASE(NF90_DOUBLE)
            cl_type='DOUBLE'
         CASE DEFAULT
            !cl_type='unknown'
            cl_type=''
      END SELECT      

      WRITE(*,'((/a,a),4(/3x,a,a),4(/3x,a,i3),&
         &     (/3x,a,a),3(/3x,a,ES12.4))')&
         &    " Variable : ",TRIM(td_var%c_name),    &
         &    " standard name : ",TRIM(td_var%c_stdname), &
         &    " long name     : ",TRIM(td_var%c_longname), &
         &    " units         : ",TRIM(td_var%c_units),   &
         &    " point         : ",TRIM(td_var%c_point),   &
         &    " id            : ",td_var%i_id,            &
         &    " rec           : ",td_var%i_rec,           &
         &    " ndim          : ",td_var%i_ndim,          &
         &    " natt          : ",td_var%i_natt,          &
         &    " type          : ",TRIM(cl_type),          &
         &    " scale factor  : ",td_var%d_scf,           &
         &    " add offset    : ",td_var%d_ofs,           &
         &    " _FillValue    : ",td_var%d_fill

      IF( ASSOCIATED(td_var%d_value) )THEN
         dl_min=MINVAL(td_var%d_value(:,:,:,:), &
            &          mask=(td_var%d_value(:,:,:,:)/=td_var%d_fill) )&
            &   *td_var%d_scf+td_var%d_ofs
         dl_max=MAXVAL(td_var%d_value(:,:,:,:), &
            &          mask=(td_var%d_value(:,:,:,:)/=td_var%d_fill) )&
            &   *td_var%d_scf+td_var%d_ofs

         WRITE(*,'((3x,a),2(/3x,a,ES12.4))')&
            &     "VALUE ASSOCIATED" ,       &
            &     " min value     : ",dl_min,&        
            &     " max value     : ",dl_max
      ENDIF      

      IF( ll_more )THEN
         ! print dimension
         IF(  td_var%i_ndim /= 0 )THEN
            WRITE(*,'(a)') " Variable dimension"
            DO ji=1,ip_maxdim
               IF( td_var%t_dim(ji)%l_use )THEN
                  CALL dim_print(td_var%t_dim(ji))
               ENDIF
            ENDDO
         ENDIF

         ! print attribute
         IF( td_var%i_natt /= 0 )THEN
            WRITE(*,'(a)') " Variable attribute"
            DO ji=1,td_var%i_natt
               CALL att_print(td_var%t_att(ji))
            ENDDO
         ENDIF
      ENDIF

   END SUBROUTINE var__print_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__add_value(td_var, dd_value, id_start, id_count)
   !-------------------------------------------------------------------
   !> @brief This subroutine add a 4D array of real(8) value in a variable 
   !> structure.
   !>
   !> @details 
   !> indices in the variable where value will be written could be specify if
   !> start and count array are given. 
   !> @warning Dimension of the array must be ordered as ('x','y','z','t')
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - decompose array copy on each dimension
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] dd_value  array of variable value
   !> @param[in] id_start  index in the variable from which the data values 
   !> will be read
   !> @param[in] id_count  number of indices selected along each dimension
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR),                        INTENT(INOUT) :: td_var
      REAL(dp),    DIMENSION(:,:,:,:),   INTENT(IN)    :: dd_value
      INTEGER(i4), DIMENSION(ip_maxdim), INTENT(IN),   OPTIONAL  :: id_start
      INTEGER(i4), DIMENSION(ip_maxdim), INTENT(IN),   OPTIONAL  :: id_count

      ! local variable
      INTEGER(i4) :: il_status
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_start
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_count
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_shape

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      ! check id_count and id_start optionals parameters...
      IF( (       PRESENT(id_start)  .AND. (.NOT. PRESENT(id_count))) .OR. &
          ((.NOT. PRESENT(id_start)) .AND.        PRESENT(id_count) ) )THEN
         CALL logger_warn( &
         &  " VAR ADD VALUE: id_start and id_count should be both specified")
      ENDIF

      IF( PRESENT(id_start).AND.PRESENT(id_count) )THEN

         ! keep ordered array ('x','y','z','t')
         il_start(:)=id_start(:)
         il_count(:)=id_count(:)

      ELSE

         ! keep ordered array ('x','y','z','t')
         il_start(:)=(/1,1,1,1/)
         il_count(:)=td_var%t_dim(:)%i_len

      ENDIF

      ! check dimension of input array
      il_shape(:)=SHAPE(dd_value(:,:,:,:))
      IF(.NOT.ALL( il_count(:) == il_shape(:)) )THEN

         CALL logger_debug(" ADD VALUE: check dimension order !!")
         DO ji = 1, ip_maxdim
            CALL logger_debug( &
            &  " VAR ADD VALUE: count : "//TRIM(fct_str(il_count(ji)))//&
            &  " array dimension : "//TRIM(fct_str(il_shape(ji))))
         ENDDO
         CALL logger_error( &
         &  " VAR ADD VALUE: dimension of input array, and count array differ " )

      ELSE

         ! check dimension of variable
         IF(.NOT.ALL(il_start(:)+il_count(:)-1 <= td_var%t_dim(:)%i_len) )THEN

            CALL logger_debug(" VAR ADD VALUE: check dimension order !!")
            DO ji = 1, ip_maxdim
               CALL logger_debug( &
               &  " VAR ADD VALUE: start ("//TRIM(fct_str(il_start(ji)))//") "//&
               &  "+ count ("//TRIM(fct_str(il_count(ji)))//") "//&
               &  "variable dimension "//TRIM(fct_str(td_var%t_dim(ji)%i_len)))
            ENDDO

            CALL logger_error( &
            &  " VAR ADD VALUE: start + count exceed variable dimension bound. " )
         ELSE

            ! special case for scalar variable
            IF( td_var%i_ndim == 0 )THEN
               ! reorder dimension to ('x','y','z','t')
               CALL dim_reorder(td_var%t_dim)
            ENDIF

            IF( ASSOCIATED(td_var%d_value) )THEN

               CALL logger_warn( &
               &  "VAR ADD VALUE: value already in variable "//&
               &  TRIM(td_var%c_name)//&
               &  " (standard name "//TRIM(td_var%c_stdname)//")" )

            ELSE

               ! Allocate space to hold variable value in structure
               ALLOCATE(td_var%d_value( td_var%t_dim(1)%i_len, &
                  &                     td_var%t_dim(2)%i_len, &
                  &                     td_var%t_dim(3)%i_len, &
                  &                     td_var%t_dim(4)%i_len),&
                  &     stat=il_status)
               IF(il_status /= 0 )THEN

                 CALL logger_error( &
                  &  " VAR ADD VALUE: not enough space to put variable "//&
                  &  TRIM(td_var%c_name)//&
                  &  " in variable structure")

               ENDIF
               
               ! initialise array
               CALL logger_trace( &
                  & " VAR ADD VALUE: value in variable "//TRIM(td_var%c_name)//&
                  & ", initialise to FillValue "//TRIM(fct_str(td_var%d_fill)) )
               td_var%d_value(:,:,:,:)=td_var%d_fill

            ENDIF

            CALL logger_debug( &
            &  " VAR ADD VALUE: put value in variable "//TRIM(td_var%c_name)//&
            &  " (standard name "//TRIM(td_var%c_stdname)//")" )

            ! put value in variable structure
            DO jl=1,il_count(4)
               DO jk=1,il_count(3)
                  DO jj=1,il_count(2)
                     DO ji=1,il_count(1)
                        td_var%d_value(ji+il_start(1)-1, &
                           &           jj+il_start(2)-1, &
                           &           jk+il_start(3)-1, &
                           &           jl+il_start(4)-1 ) = dd_value(ji,jj,jk,jl)
                     ENDDO
                  ENDDO
               ENDDO
            ENDDO

!            td_var%d_value( il_start(1):il_start(1)+il_count(1)-1, &
!            &               il_start(2):il_start(2)+il_count(2)-1, &
!            &               il_start(3):il_start(3)+il_count(3)-1, &
!            &               il_start(4):il_start(4)+il_count(4)-1 ) = dd_value(:,:,:,:)

         ENDIF
      ENDIF

   END SUBROUTINE var__add_value
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__add_value_dp(td_var, dd_value, id_type, id_start, id_count)
   !-------------------------------------------------------------------
   !> @brief This subroutine add a 4D array of real(8) value in a variable 
   !> structure. Dimension of the array must be ordered as ('x','y','z','t')
   !>
   !> @details 
   !> Optionally, you could specify the type of the variable to be used (default real(8)),
   !> and indices of the variable where value will be written with start and count array.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] dd_value  array of variable value
   !> @param[in] id_type   type of the variable to be used (default real(8)) 
   !> @param[in] id_start  start indices of the variable where data values 
   !> will be written
   !> @param[in] id_count  number of indices selected along each dimension
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR),                        INTENT(INOUT) :: td_var
      REAL(dp),    DIMENSION(:,:,:,:),   INTENT(IN)    :: dd_value
      INTEGER(i4),                       INTENT(IN),   OPTIONAL  :: id_type
      INTEGER(i4), DIMENSION(ip_maxdim), INTENT(IN),   OPTIONAL  :: id_start
      INTEGER(i4), DIMENSION(ip_maxdim), INTENT(IN),   OPTIONAL  :: id_count

      ! local variable
      CHARACTER(LEN=lc) :: cl_type
      !----------------------------------------------------------------

      IF( PRESENT(id_type) )THEN
         td_var%i_type=id_type

         cl_type=''
         SELECT CASE(td_var%i_type)
         CASE(NF90_DOUBLE)
            cl_type='DOUBLE'
         CASE(NF90_FLOAT)
            cl_type='FLOAT'
         CASE(NF90_INT)
            cl_type='INT'
         CASE(NF90_SHORT)
            cl_type='SHORT'
         CASE(NF90_BYTE)
            cl_type='BYTE'
         END SELECT
         CALL logger_trace("VAR ADD VALUE: "//TRIM(td_var%c_name)//&
         &                " value will be saved as "//TRIM(cl_type)) 
      ENDIF

      CALL var__add_value(td_var, dd_value, id_start, id_count)

   END SUBROUTINE var__add_value_dp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__add_value_rp(td_var, rd_value, id_type, id_start, id_count)
   !-------------------------------------------------------------------
   !> @brief This subroutine add a 4D array of real(4) value in a variable 
   !> structure. Dimension of the array must be ordered as ('x','y','z','t')
   !>
   !> @details 
   !> Optionally, you could specify the type of the variable to be used (default real(4)),
   !> and indices of the variable where value will be written with start and count array.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] rd_value  array of variable value
   !> @param[in] id_type   type of the variable to be used (default real(4)) 
   !> @param[in] id_start  start indices of the variable where data values 
   !> will be written
   !> @param[in] id_count  number of indices selected along each dimension
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR),                        INTENT(INOUT) :: td_var
      REAL(sp),    DIMENSION(:,:,:,:),   INTENT(IN)    :: rd_value
      INTEGER(i4),                       INTENT(IN),   OPTIONAL  :: id_type
      INTEGER(i4), DIMENSION(ip_maxdim), INTENT(IN),   OPTIONAL  :: id_start
      INTEGER(i4), DIMENSION(ip_maxdim), INTENT(IN),   OPTIONAL  :: id_count

      ! local variable
      CHARACTER(LEN=lc)                                    :: cl_type

      INTEGER(i4)                                          :: il_status
      INTEGER(i4)      , DIMENSION(ip_maxdim)              :: il_shape

      REAL(dp)         , DIMENSION(:,:,:,:)  , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      IF( PRESENT(id_type) )THEN
         td_var%i_type=id_type

         cl_type=''
         SELECT CASE(td_var%i_type)
         CASE(NF90_DOUBLE)
            cl_type='DOUBLE'
         CASE(NF90_FLOAT)
            cl_type='FLOAT'
         CASE(NF90_INT)
            cl_type='INT'
         CASE(NF90_SHORT)
            cl_type='SHORT'
         CASE(NF90_BYTE)
            cl_type='BYTE'
         END SELECT
         CALL logger_trace("VAR ADD VALUE: "//TRIM(td_var%c_name)//&
         &                " value will be saved as "//TRIM(cl_type)) 
      ENDIF      

      il_shape=SHAPE(rd_value)
      ALLOCATE( dl_value(il_shape(1), il_shape(2), il_shape(3), il_shape(4)),&
      &         stat=il_status)
      IF(il_status /= 0 )THEN

        CALL logger_error( &
         &  " VAR ADD VALUE: not enough space to put variable "//&
         &  TRIM(td_var%c_name)//&
         &  " in variable structure")

      ENDIF

      DO jl=1,il_shape(4)
         DO jk=1,il_shape(3)
            DO jj=1,il_shape(2)
               DO ji=1,il_shape(1)
                  dl_value(ji,jj,jk,jl)=REAL(rd_value(ji,jj,jk,jl), dp)
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      CALL var__add_value(td_var, dl_value(:,:,:,:), id_start(:), id_count(:))

      DEALLOCATE(dl_value)

   END SUBROUTINE var__add_value_rp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__add_value_i1(td_var, bd_value, id_type, id_start, id_count)
   !-------------------------------------------------------------------
   !> @brief This subroutine add a 4D array of integer(1) value in a variable 
   !> structure. Dimension of the array must be ordered as ('x','y','z','t')
   !>
   !> @details 
   !> Optionally, you could specify the type of the variable to be used (default integer(1)),
   !> and indices of the variable where value will be written with start and count array.
   !>
   !> @note variable type is forced to BYTE
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !>
   !> @param[inout] td_var variabele structure
   !> @param[in] bd_value  array of variable value
   !> @param[in] id_type   type of the variable to be used (default integer(1)) 
   !> @param[in] id_start  start indices of the variable where data values 
   !> will be read
   !> @param[in] id_count  number of indices selected along each dimension
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR),                        INTENT(INOUT) :: td_var
      INTEGER(i1), DIMENSION(:,:,:,:),   INTENT(IN)    :: bd_value
      INTEGER(i4),                       INTENT(IN),   OPTIONAL  :: id_type
      INTEGER(i4), DIMENSION(ip_maxdim), INTENT(IN),   OPTIONAL  :: id_start
      INTEGER(i4), DIMENSION(ip_maxdim), INTENT(IN),   OPTIONAL  :: id_count

      ! local variable
      CHARACTER(LEN=lc)                                    :: cl_type

      INTEGER(i4)                                          :: il_status
      INTEGER(i4)      , DIMENSION(ip_maxdim)              :: il_shape

      REAL(dp)         , DIMENSION(:,:,:,:)  , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      IF( PRESENT(id_type) )THEN
         td_var%i_type=id_type

         cl_type=''
         SELECT CASE(td_var%i_type)
         CASE(NF90_DOUBLE)
            cl_type='DOUBLE'
         CASE(NF90_FLOAT)
            cl_type='FLOAT'
         CASE(NF90_INT)
            cl_type='INT'
         CASE(NF90_SHORT)
            cl_type='SHORT'
         CASE(NF90_BYTE)
            cl_type='BYTE'
         END SELECT
         CALL logger_trace("VAR ADD VALUE: "//TRIM(td_var%c_name)//&
         &                " value will be saved as "//TRIM(cl_type)) 
      ENDIF

      il_shape=SHAPE(bd_value)
      ALLOCATE( dl_value(il_shape(1), il_shape(2), il_shape(3), il_shape(4)),&
      &         stat=il_status)
      IF(il_status /= 0 )THEN

        CALL logger_error( &
         &  " VAR ADD VALUE: not enough space to put variable "//&
         &  TRIM(td_var%c_name)//&
         &  " in variable structure")

      ENDIF

      DO jl=1,il_shape(4)
         DO jk=1,il_shape(3)
            DO jj=1,il_shape(2)
               DO ji=1,il_shape(1)
                  dl_value(ji,jj,jk,jl)=REAL(bd_value(ji,jj,jk,jl),dp)      
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      CALL var__add_value(td_var, dl_value(:,:,:,:), id_start(:), id_count(:))

      DEALLOCATE(dl_value)

   END SUBROUTINE var__add_value_i1
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__add_value_i2(td_var, sd_value, id_type, id_start, id_count)
   !-------------------------------------------------------------------
   !> @brief This subroutine add a 4D array of integer(2) value in a variable 
   !> structure. Dimension of the array must be ordered as ('x','y','z','t')
   !>
   !> @details 
   !> Optionally, you could specify the type of the variable to be used (default integer(2)),
   !> and indices of the variable where value will be written with start and count array.
   !>
   !> @note variable type is forced to SHORT
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !>
   !> @param[inout] td_var variabele structure
   !> @param[in] sd_value  array of variable value
   !> @param[in] id_type   type of the variable to be used (default integer(2)) 
   !> @param[in] id_start  start indices of the variable where data values 
   !> will be read
   !> @param[in] id_count  number of indices selected along each dimension
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR),                        INTENT(INOUT) :: td_var
      INTEGER(i2), DIMENSION(:,:,:,:),   INTENT(IN)    :: sd_value
      INTEGER(i4),                       INTENT(IN),   OPTIONAL  :: id_type
      INTEGER(i4), DIMENSION(ip_maxdim), INTENT(IN),   OPTIONAL  :: id_start
      INTEGER(i4), DIMENSION(ip_maxdim), INTENT(IN),   OPTIONAL  :: id_count

      ! local variable
      CHARACTER(LEN=lc)                                    :: cl_type

      INTEGER(i4)                                          :: il_status
      INTEGER(i4)      , DIMENSION(ip_maxdim)              :: il_shape

      REAL(dp)         , DIMENSION(:,:,:,:)  , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      IF( PRESENT(id_type) )THEN
         td_var%i_type=id_type

         cl_type=''
         SELECT CASE(td_var%i_type)
         CASE(NF90_DOUBLE)
            cl_type='DOUBLE'
         CASE(NF90_FLOAT)
            cl_type='FLOAT'
         CASE(NF90_INT)
            cl_type='INT'
         CASE(NF90_SHORT)
            cl_type='SHORT'
         CASE(NF90_BYTE)
            cl_type='BYTE'
         END SELECT
         CALL logger_trace("VAR ADD VALUE: "//TRIM(td_var%c_name)//&
         &                " value will be saved as "//TRIM(cl_type)) 
      ENDIF

      il_shape=SHAPE(sd_value)
      ALLOCATE( dl_value(il_shape(1), il_shape(2), il_shape(3), il_shape(4)),&
      &         stat=il_status)
      IF(il_status /= 0 )THEN

        CALL logger_error( &
         &  " VAR ADD VALUE: not enough space to put variable "//&
         &  TRIM(td_var%c_name)//&
         &  " in variable structure")

      ENDIF

      DO jl=1,il_shape(4)
         DO jk=1,il_shape(3)
            DO jj=1,il_shape(2)
               DO ji=1,il_shape(1)
                  dl_value(ji,jj,jk,jl)=REAL(sd_value(ji,jj,jk,jl),dp)      
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      CALL var__add_value(td_var, dl_value(:,:,:,:), id_start(:), id_count(:))

      DEALLOCATE(dl_value)

   END SUBROUTINE var__add_value_i2
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__add_value_i4(td_var, id_value, id_type, id_start, id_count)
   !-------------------------------------------------------------------
   !> @brief This subroutine add a 4D array of integer(4) value in a variable 
   !> structure. Dimension of the array must be ordered as ('x','y','z','t')
   !>
   !> @details 
   !> Optionally, you could specify the type of the variable to be used (default integer(4)),
   !> and indices of the variable where value will be written with start and count array.
   !>
   !> @note variable type is forced to INT
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !>
   !> @param[inout] td_var variabele structure
   !> @param[in] id_value  array of variable value
   !> @param[in] id_type   type of the variable to be used (default integer(4)) 
   !> @param[in] id_start  start indices of the variable where data values 
   !> will be read
   !> @param[in] id_count  number of indices selected along each dimension
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR),                        INTENT(INOUT) :: td_var
      INTEGER(i4), DIMENSION(:,:,:,:),   INTENT(IN)    :: id_value
      INTEGER(i4),                       INTENT(IN),   OPTIONAL  :: id_type
      INTEGER(i4), DIMENSION(ip_maxdim), INTENT(IN),   OPTIONAL  :: id_start
      INTEGER(i4), DIMENSION(ip_maxdim), INTENT(IN),   OPTIONAL  :: id_count

      ! local variable
      CHARACTER(LEN=lc)                                    :: cl_type

      INTEGER(i4)                                          :: il_status
      INTEGER(i4)      , DIMENSION(ip_maxdim)              :: il_shape

      REAL(dp)         , DIMENSION(:,:,:,:)  , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      IF( PRESENT(id_type) )THEN
         td_var%i_type=id_type

         cl_type=''
         SELECT CASE(td_var%i_type)
         CASE(NF90_DOUBLE)
            cl_type='DOUBLE'
         CASE(NF90_FLOAT)
            cl_type='FLOAT'
         CASE(NF90_INT)
            cl_type='INT'
         CASE(NF90_SHORT)
            cl_type='SHORT'
         CASE(NF90_BYTE)
            cl_type='BYTE'
         END SELECT
         CALL logger_trace("VAR ADD VALUE: "//TRIM(td_var%c_name)//&
         &                " value will be saved as "//TRIM(cl_type)) 
      ENDIF

      il_shape=SHAPE(id_value)
      ALLOCATE( dl_value(il_shape(1), il_shape(2), il_shape(3), il_shape(4)),&
      &         stat=il_status)
      IF(il_status /= 0 )THEN

        CALL logger_error( &
         &  " VAR ADD VALUE: not enough space to put variable "//&
         &  TRIM(td_var%c_name)//&
         &  " in variable structure")

      ENDIF

      DO jl=1,il_shape(4)
         DO jk=1,il_shape(3)
            DO jj=1,il_shape(2)
               DO ji=1,il_shape(1)
                  dl_value(ji,jj,jk,jl)=REAL(id_value(ji,jj,jk,jl),dp)      
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      CALL var__add_value(td_var, dl_value(:,:,:,:), id_start(:), id_count(:))

      DEALLOCATE(dl_value)

   END SUBROUTINE var__add_value_i4
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__add_value_i8(td_var, kd_value, id_type, id_start, id_count)
   !-------------------------------------------------------------------
   !> @brief This subroutine add a 4D array of integer(8) value in a variable 
   !> structure. Dimension of the array must be ordered as ('x','y','z','t')
   !>
   !> @details 
   !> Optionally, you could specify the type of the variable to be used (default integer(4)),
   !> and indices of the variable where value will be written with start and count array.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - decompose array conversion on each dimension
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] kd_value  array of variable value
   !> @param[in] id_type   type of the variable to be used (default integer(8)) 
   !> @param[in] id_start  start indices of the variable where data values 
   !> will be read
   !> @param[in] id_count  number of indices selected along each dimension
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR),                        INTENT(INOUT) :: td_var
      INTEGER(i8), DIMENSION(:,:,:,:),   INTENT(IN)    :: kd_value
      INTEGER(i4),                       INTENT(IN),   OPTIONAL  :: id_type
      INTEGER(i4), DIMENSION(ip_maxdim), INTENT(IN),   OPTIONAL  :: id_start
      INTEGER(i4), DIMENSION(ip_maxdim), INTENT(IN),   OPTIONAL  :: id_count

      ! local variable
      CHARACTER(LEN=lc)                                    :: cl_type

      INTEGER(i4)                                          :: il_status
      INTEGER(i4)      , DIMENSION(ip_maxdim)              :: il_shape

      REAL(dp)         , DIMENSION(:,:,:,:)  , ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      IF( PRESENT(id_type) )THEN
         td_var%i_type=id_type

         cl_type=''
         SELECT CASE(td_var%i_type)
         CASE(NF90_DOUBLE)
            cl_type='DOUBLE'
         CASE(NF90_FLOAT)
            cl_type='FLOAT'
         CASE(NF90_INT)
            cl_type='INT'
         CASE(NF90_SHORT)
            cl_type='SHORT'
         CASE(NF90_BYTE)
            cl_type='BYTE'
         END SELECT
         CALL logger_trace("VAR ADD VALUE: "//TRIM(td_var%c_name)//&
         &                " value will be saved as "//TRIM(cl_type)) 
      ENDIF

      il_shape=SHAPE(kd_value)
      ALLOCATE( dl_value(il_shape(1), il_shape(2), il_shape(3), il_shape(4)),&
      &         stat=il_status)
      IF(il_status /= 0 )THEN

        CALL logger_error( &
         &  " VAR ADD VALUE: not enough space to put variable "//&
         &  TRIM(td_var%c_name)//&
         &  " in variable structure")

      ENDIF

      DO jl=1,il_shape(4)
         DO jk=1,il_shape(3)
            DO jj=1,il_shape(2)
               DO ji=1,il_shape(1)
                  dl_value(ji,jj,jk,jl)=REAL(kd_value(ji,jj,jk,jl),dp)
               ENDDO
            ENDDO
         ENDDO
      ENDDO

      CALL var__add_value(td_var, dl_value, id_start, id_count)

      DEALLOCATE(dl_value)

   END SUBROUTINE var__add_value_i8
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var_del_value(td_var)
   !-------------------------------------------------------------------
   !> @brief This subroutine remove variable value in a variable 
   !> structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - nullify array inside variable structure
   !>
   !> @param[inout] td_var variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR), INTENT(INOUT) :: td_var

      !----------------------------------------------------------------
      CALL logger_debug( &
      &  " VAR DEL VALUE: value in variable "//TRIM(td_var%c_name)//&
      &  ", standard name "//TRIM(td_var%c_stdname)//&
      &  " will be remove ")

      DEALLOCATE(td_var%d_value)
      NULLIFY(td_var%d_value)

   END SUBROUTINE var_del_value
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var_get_index(td_var, cd_name, cd_stdname) &
         & RESULT (if_idx)
   !-------------------------------------------------------------------
   !> @brief This function return the variable index, in a array of variable
   !> structure,  given variable name or standard name. 
   !>
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[in] td_var       array of variable structure
   !> @param[in] cd_name      variable name
   !> @param[in] cd_stdname   variable standard name
   !> @return variable index in array of variable structure (0 if not found)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR)      , DIMENSION(:), INTENT(IN) :: td_var
      CHARACTER(LEN=*),               INTENT(IN) :: cd_name
      CHARACTER(LEN=*),               INTENT(IN), OPTIONAL :: cd_stdname

      ! function
      INTEGER(i4)                                :: if_idx

      ! local variable
      INTEGER(i4) :: il_size

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      if_idx=0
      il_size=SIZE(td_var(:))

      ! check if variable is in array of variable structure
      DO ji=1,il_size

         ! look for variable name
         IF( fct_lower(td_var(ji)%c_name) == fct_lower(cd_name) )THEN
         
            if_idx=ji
            EXIT

         ! look for variable standard name
         ELSE IF( fct_lower(td_var(ji)%c_stdname) == fct_lower(cd_name) .AND.&
            &     TRIM(fct_lower(td_var(ji)%c_stdname)) /= '' )THEN
            
            if_idx=ji
            EXIT

         ELSE IF( PRESENT(cd_stdname) )THEN 

            IF( fct_lower(td_var(ji)%c_stdname) == fct_lower(cd_stdname) .AND.&
               &TRIM(fct_lower(td_var(ji)%c_stdname)) /= '' )THEN
            
               if_idx=ji
               EXIT
            ENDIF

         ENDIF

         ! look for variable longname
         IF( fct_lower(td_var(ji)%c_longname) == fct_lower(cd_name) .AND.&
            &TRIM(fct_lower(td_var(ji)%c_longname)) /= '' )THEN
            
            if_idx=ji
            EXIT

         ELSE IF( PRESENT(cd_stdname) )THEN

            IF( fct_lower(td_var(ji)%c_longname) == fct_lower(cd_stdname) .AND.&
               &TRIM(fct_lower(td_var(ji)%c_longname)) /= '' )THEN
            
               if_idx=ji
               EXIT
            ENDIF

         ENDIF

      ENDDO

   END FUNCTION var_get_index
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var_get_id(td_var, cd_name, cd_stdname) &
         & RESULT (if_id)
   !-------------------------------------------------------------------
   !> @brief This function return the variable id, 
   !> given variable name or standard name. 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2015
   !> - check long name
   !>
   !> @param[in] td_var       array of variable structure
   !> @param[in] cd_name      variable name
   !> @param[in] cd_stdname   variable standard name
   !> @return variable id in array of variable structure (0 if not found)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR)      , DIMENSION(:), INTENT(IN) :: td_var
      CHARACTER(LEN=*),               INTENT(IN) :: cd_name
      CHARACTER(LEN=*),               INTENT(IN), OPTIONAL :: cd_stdname

      ! function
      INTEGER(i4)                                :: if_id

      ! local variable
      INTEGER(i4) :: il_size

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      if_id=0
      il_size=SIZE(td_var(:))

      ! check if variable is in array of variable structure
      DO ji=1,il_size
      
         ! look for variable name
         IF( fct_lower(td_var(ji)%c_name) == fct_lower(cd_name) )THEN
         
            if_id=td_var(ji)%i_id
            EXIT

         ! look for variable standard name
         ELSE IF( fct_lower(td_var(ji)%c_stdname) == fct_lower(cd_name) .AND.&
            &     TRIM(fct_lower(td_var(ji)%c_stdname)) /= '' )THEN
            
            if_id=td_var(ji)%i_id
            EXIT

         ELSE IF( PRESENT(cd_stdname) )THEN

            IF( fct_lower(td_var(ji)%c_stdname) == fct_lower(cd_stdname) .AND.&
               &TRIM(fct_lower(td_var(ji)%c_stdname)) /= '' )THEN
            
               if_id=td_var(ji)%i_id
               EXIT
            ENDIF

         ENDIF

         ! look for variable long name
         IF( fct_lower(td_var(ji)%c_longname) == fct_lower(cd_name) .AND.&
            &TRIM(fct_lower(td_var(ji)%c_longname)) /= '' )THEN
            
            if_id=td_var(ji)%i_id
            EXIT

         ELSE IF( PRESENT(cd_stdname) )THEN 

            IF( fct_lower(td_var(ji)%c_longname) == fct_lower(cd_stdname) .AND.&
               &TRIM(fct_lower(td_var(ji)%c_longname)) /= '' )THEN
            
               if_id=td_var(ji)%i_id
               EXIT
            ENDIF

         ENDIF

      ENDDO

   END FUNCTION var_get_id
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var_get_mask(td_var) &
         & RESULT (if_mask)
   !-------------------------------------------------------------------
   !> @brief
   !> This function return the mask 3D of variable, given variable structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_var array of variable structure
   !> @return variable mask(3D)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR), INTENT(IN) :: td_var

      ! function
      INTEGER(i4), DIMENSION(td_var%t_dim(1)%i_len, &
         &                   td_var%t_dim(2)%i_len, &
         &                   td_var%t_dim(3)%i_len ) :: if_mask

      ! local variable
      !----------------------------------------------------------------

      IF( ASSOCIATED(td_var%d_value) )THEN

         CALL logger_debug( "VAR GET MASK: create mask from variable "//&
            &               TRIM(td_var%c_name)//", FillValue ="//&
            &               TRIM(fct_str(td_var%d_fill)))
         if_mask(:,:,:)=1
         WHERE( td_var%d_value(:,:,:,1) == td_var%d_fill )
            if_mask(:,:,:)=0
         ENDWHERE

      ELSE
         CALL logger_error("VAR GET MASK: variable value not define.")
      ENDIF

   END FUNCTION var_get_mask
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var_chg_FillValue(td_var, dd_fill)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine change FillValue of the variable to 
   !> standard NETCDF FillValue. 
   !>
   !> @details
   !> optionally, you could specify a dummy _FillValue to be used
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - write fill value on array level by level
   !>
   !> @param[inout] td_var array of variable structure
   !> @param[in] dd_fill _FillValue to be used
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR), INTENT(INOUT) :: td_var
      REAL(dp)  , INTENT(IN)   , OPTIONAL :: dd_fill

      ! local variable
      TYPE(TATT) :: tl_att

      INTEGER(i1) :: bl_fill
      INTEGER(i2) :: sl_fill
      INTEGER(i4) :: il_fill
      REAL(sp)    :: rl_fill

      ! loop indices
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      CALL logger_trace( "VAR CHG FILL VALUE: change _FillValue in variable "//&
      &  TRIM(td_var%c_name) )

      ! define attribute FillValue
      SELECT CASE( td_var%i_type )

         CASE(NF90_BYTE)
            IF( PRESENT(dd_fill) )THEN
               bl_fill=INT(dd_fill,i1)
               tl_att=att_init('_FillValue',bl_fill)
            ELSE
               tl_att=att_init('_FillValue',NF90_FILL_BYTE)
            ENDIF
         CASE(NF90_SHORT)
            IF( PRESENT(dd_fill) )THEN
               sl_fill=INT(dd_fill,i2)
               tl_att=att_init('_FillValue',sl_fill)
            ELSE
               tl_att=att_init('_FillValue',NF90_FILL_SHORT)
            ENDIF
         CASE(NF90_INT)
            IF( PRESENT(dd_fill) )THEN
               il_fill=INT(dd_fill,i4)
               tl_att=att_init('_FillValue',il_fill)
            ELSE
               tl_att=att_init('_FillValue',NF90_FILL_INT)
            ENDIF
         CASE(NF90_FLOAT)
            IF( PRESENT(dd_fill) )THEN
               rl_fill=REAL(dd_fill,sp)
               tl_att=att_init('_FillValue',rl_fill)
            ELSE
               tl_att=att_init('_FillValue',NF90_FILL_FLOAT)
            ENDIF
         CASE DEFAULT ! NF90_DOUBLE
            IF( PRESENT(dd_fill) )THEN
               tl_att=att_init('_FillValue',dd_fill)
            ELSE
               tl_att=att_init('_FillValue',NF90_FILL_DOUBLE)
            ENDIF

      END SELECT

      IF( ASSOCIATED(td_var%d_value) )THEN
         ! change FillValue in variable value
         DO jl=1,td_var%t_dim(jp_L)%i_len
            WHERE( td_var%d_value(:,:,:,jl) == td_var%d_fill )
               td_var%d_value(:,:,:,jl)=tl_att%d_value(1)
            END WHERE
         ENDDO
      ENDIF

      ! change attribute _FillValue
      CALL var_move_att(td_var, tl_att)

      ! clean
      CALL att_clean(tl_att)

   END SUBROUTINE var_chg_FillValue
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var_def_extra( cd_file )
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine read variable configuration file. And save 
   !> global array of variable structure with extra information: tg_varextra.
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015 
   !> - new namelist format to get extra information (interpolation,...)
   !>
   !> @param[in] cd_file   configuration file of variable
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN) :: cd_file

      ! local variable
      CHARACTER(LEN=lc) :: cl_line
      CHARACTER(LEN=lc) :: cl_interp

      INTEGER(i4)       :: il_nvar
      INTEGER(i4)       :: il_fileid
      INTEGER(i4)       :: il_status

      LOGICAL           :: ll_exist

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( ALLOCATED(tg_varextra) )THEN
         CALL var_clean(tg_varextra(:))
         DEALLOCATE(tg_varextra)
      ENDIF

      ! read config variable file
      INQUIRE(FILE=TRIM(cd_file), EXIST=ll_exist)
      IF( ll_exist )THEN
         
         ! get number of variable to be read

         il_fileid=fct_getunit()
         OPEN( il_fileid, FILE=TRIM(cd_file), &
            &             FORM='FORMATTED',   &
            &             ACCESS='SEQUENTIAL',&
            &             STATUS='OLD',       &
            &             ACTION='READ',      &
            &             IOSTAT=il_status)
         CALL fct_err(il_status)
         IF( il_status /= 0 )THEN
            CALL logger_fatal("VAR DEF EXTRA: can not open file "//&
            &                 TRIM(cd_file))
         ENDIF

         ! read file
         READ( il_fileid, FMT='(a)', IOSTAT=il_status ) cl_line
         cl_line=TRIM(ADJUSTL(cl_line))
         il_nvar=0
         DO WHILE( il_status == 0 )

         ! search line not beginning with comment character
            IF( SCAN( TRIM(fct_concat(cp_com(:))) ,cl_line(1:1)) == 0 )THEN
               il_nvar=il_nvar+1
            ENDIF
            
            READ( il_fileid, FMT='(a)', IOSTAT=il_status ) cl_line
            cl_line=TRIM(ADJUSTL(cl_line))
         ENDDO

         IF( il_nvar <= 0 )THEN
            CALL logger_warn("VAR DEF EXTRA: no variable to be read")

            CLOSE( il_fileid, IOSTAT=il_status )
            CALL fct_err(il_status)
            IF( il_status /= 0 )THEN
               CALL logger_error("VAR DEF EXTRA: closing file "//TRIM(cd_file))
            ENDIF

         ELSE
            CALL logger_info("VAR DEF EXTRA: "//TRIM(fct_str(il_nvar))//&
               &             " variable to be read on varaible config file"//&
               &             TRIM(cd_file))

            CALL logger_trace("VAR DEF EXTRA: rewind "//TRIM(cd_file))
            REWIND( il_fileid, IOSTAT=il_status)
            CALL fct_err(il_status)
            IF( il_status /= 0 )THEN
               CALL logger_error("VAR DEF EXTRA: opening file "//TRIM(cd_file))
            ENDIF

            ALLOCATE( tg_varextra(il_nvar) )

            ! read file
            READ( il_fileid, FMT='(a)', IOSTAT=il_status ) cl_line
            cl_line=TRIM(ADJUSTL(cl_line))
            ji=1
            DO WHILE( il_status == 0 )

               IF( SCAN( TRIM(fct_concat(cp_com(:))) ,cl_line(1:1)) == 0 )THEN
                  tg_varextra(ji)%i_id      = ji
                  tg_varextra(ji)%c_name    =TRIM(fct_split(cl_line,1))
                  tg_varextra(ji)%c_units   =TRIM(fct_split(cl_line,2))
                  tg_varextra(ji)%c_axis    =TRIM(fct_split(cl_line,3))
                  tg_varextra(ji)%c_point   =TRIM(fct_split(cl_line,4))

                  cl_interp='int='//TRIM(fct_split(cl_line,5))
                  tg_varextra(ji)%c_interp(:) = &
                     &  var__get_interp(TRIM(tg_varextra(ji)%c_name), cl_interp)
                  CALL logger_debug("VAR DEF EXTRA: "//&
                     &  TRIM(tg_varextra(ji)%c_name)//&
                     &  " "//TRIM(tg_varextra(ji)%c_interp(1)))

                  tg_varextra(ji)%c_longname=TRIM(fct_split(cl_line,6))
                  tg_varextra(ji)%c_stdname =TRIM(fct_split(cl_line,7))
               ELSE
                  ji=ji-1
               ENDIF
               
               READ( il_fileid, FMT='(a)', IOSTAT=il_status ) cl_line
               cl_line=TRIM(ADJUSTL(cl_line))
               ji=ji+1
            ENDDO

            CLOSE( il_fileid, IOSTAT=il_status )
            CALL fct_err(il_status)
            IF( il_status /= 0 )THEN
               CALL logger_error("VAR DEF EXTRA: closing file "//TRIM(cd_file))
            ENDIF
         ENDIF

      ELSE

         CALL logger_error("VAR DEF EXTRA: can't find file "//TRIM(cd_file))

      ENDIF      

   END SUBROUTINE var_def_extra
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var_chg_extra( cd_varinfo )
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine add variable information get from namelist in 
   !> global array of variable structure with extra information: tg_varextra.
   !> 
   !> @details 
   !> string character format must be : <br/>
   !> "varname:int=interp; flt=filter; ext=extrap; min=min; max=max"<br/>
   !> you could specify only interpolation, filter or extrapolation method, 
   !> whatever the order. you could find more
   !> information about available method in \ref interp, \ref filter, and
   !> \ref extrap module.<br/>
   !> Examples: 
   !> cn_varinfo='Bathymetry:flt=2*hamming(2,3); min=10.'
   !> cn_varinfo='votemper:int=cubic; ext=dist_weight; max=40.'
   !>
   !>
   !> @warning variable should be define in tg_varextra (ie in configuration
   !> file, to be able to add information from namelist
   !>
   !> @note If you do not specify a method which is required, default one is
   !> apply.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2015 
   !> - get unit and unit factor (to change unit) 
   !> @date February, 2019
   !> - get variable output name
   !>
   !> @param[in] cd_varinfo   variable information from namelist
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), DIMENSION(:), INTENT(IN) :: cd_varinfo

      ! local variable
      CHARACTER(LEN=lc)                            :: cl_name
      CHARACTER(LEN=lc)                            :: cl_method
      CHARACTER(LEN=lc), DIMENSION(2)              :: cl_interp
      CHARACTER(LEN=lc), DIMENSION(1)              :: cl_extrap
      CHARACTER(LEN=lc), DIMENSION(5)              :: cl_filter
      CHARACTER(LEN=lc)                            :: cl_unt
      CHARACTER(LEN=lc)                            :: cl_namout

      INTEGER(i4)                                  :: il_ind
      INTEGER(i4)                                  :: il_nvar

      REAL(dp)                                     :: dl_min
      REAL(dp)                                     :: dl_max
      REAL(dp)                                     :: dl_unf

      TYPE(TVAR)       , DIMENSION(:), ALLOCATABLE :: tl_varextra

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( ALLOCATED(tg_varextra) )THEN
         ji=1
         DO WHILE( TRIM(cd_varinfo(ji)) /= '' )

            cl_name  =fct_lower(fct_split(cd_varinfo(ji),1,':'))
            cl_method=fct_split(cd_varinfo(ji),2,':')

            dl_min=var__get_min(cl_name, cl_method)
            dl_max=var__get_max(cl_name, cl_method)
            dl_unf=var__get_unf(cl_name, cl_method)
            cl_interp(:)=var__get_interp(cl_name, cl_method)
            cl_extrap(:)=var__get_extrap(cl_name, cl_method)
            cl_filter(:)=var__get_filter(cl_name, cl_method)
            cl_unt=var__get_unt(cl_name, cl_method)
            cl_namout=var__get_namout(cl_name, cl_method)


            il_ind=var_get_index(tg_varextra(:), TRIM(cl_name))
            IF( il_ind /= 0 )THEN
               IF( dl_min /= dp_fill ) tg_varextra(il_ind)%d_min=dl_min
               IF( dl_max /= dp_fill ) tg_varextra(il_ind)%d_max=dl_max
               IF( dl_unf /= dp_fill ) tg_varextra(il_ind)%d_unf=dl_unf
               IF(cl_unt      /='') tg_varextra(il_ind)%c_unt      =cl_unt
               IF(cl_namout   /='') tg_varextra(il_ind)%c_namout   =cl_namout
               IF(cl_interp(1)/='') tg_varextra(il_ind)%c_interp(:)=cl_interp(:)
               IF(cl_extrap(1)/='') tg_varextra(il_ind)%c_extrap(:)=cl_extrap(:)
               IF(cl_filter(1)/='') tg_varextra(il_ind)%c_filter(:)=cl_filter(:)
            ELSE

               IF( ALLOCATED(tg_varextra) )THEN
                  il_nvar=SIZE(tg_varextra(:))
                  ! save older variable
                  ALLOCATE( tl_varextra(il_nvar) )
                  tl_varextra(:)=var_copy(tg_varextra(:))

                  CALL var_clean(tg_varextra(:))
                  DEALLOCATE(tg_varextra)
                  ALLOCATE( tg_varextra(il_nvar+1) )
                  
                  tg_varextra(1:il_nvar)=var_copy(tl_varextra(:))

                  ! clean
                  CALL var_clean(tl_varextra(:))
                  DEALLOCATE(tl_varextra)

               ELSE

                  il_nvar=0
                  ALLOCATE( tg_varextra(1) )

               ENDIF

               ! add new variable
               il_ind=il_nvar+1
               tg_varextra(il_ind)=var_init( TRIM(cl_name), &
                  &                          cd_interp=cl_interp(:), &
                  &                          cd_extrap=cl_extrap(:), &
                  &                          cd_filter=cl_filter(:), &
                  &                          dd_min = dl_min, &
                  &                          dd_max = dl_max, &
                  &                          cd_unt = cl_unt, &
                  &                          dd_unf = dl_unf, &
                  &                          cd_namout = cl_namout )

            ENDIF

            ji=ji+1
            CALL logger_debug( "VAR CHG EXTRA: name       "//&
               &               TRIM(tg_varextra(il_ind)%c_name) )
            CALL logger_debug( "VAR CHG EXTRA: interp     "//&
               &               TRIM(tg_varextra(il_ind)%c_interp(1)) )         
            CALL logger_debug( "VAR CHG EXTRA: filter     "//&
               &               TRIM(tg_varextra(il_ind)%c_filter(1)) )         
            CALL logger_debug( "VAR CHG EXTRA: extrap     "//&
               &               TRIM(tg_varextra(il_ind)%c_extrap(1)) )
            IF( tg_varextra(il_ind)%d_min /= dp_fill )THEN
               CALL logger_debug( "VAR CHG EXTRA: min value  "//&
                  &               TRIM(fct_str(tg_varextra(il_ind)%d_min)) )
            ENDIF
            IF( tg_varextra(il_ind)%d_max /= dp_fill )THEN
               CALL logger_debug( "VAR CHG EXTRA: max value  "//&
                  &               TRIM(fct_str(tg_varextra(il_ind)%d_max)) )
            ENDIF
            IF( TRIM(tg_varextra(il_ind)%c_unt) /= '' )THEN
               CALL logger_debug( "VAR CHG EXTRA: new unit  "//&
                  &               TRIM(tg_varextra(il_ind)%c_unt) )
            ENDIF
            IF( tg_varextra(il_ind)%d_unf /= 1. )THEN
               CALL logger_debug( "VAR CHG EXTRA: new unit factor  "//&
                  &               TRIM(fct_str(tg_varextra(il_ind)%d_unf)) )
            ENDIF
            IF( TRIM(tg_varextra(il_ind)%c_namout) /= '' )THEN
               CALL logger_debug( "VAR CHG EXTRA: new name output  "//&
                  &               TRIM(tg_varextra(il_ind)%c_namout) )
            ENDIF
         ENDDO
      ENDIF

   END SUBROUTINE var_chg_extra
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var_clean_extra( )
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine clean global array of variable structure 
   !> with extra information: tg_varextra.
   !> 
   !> @author J.Paul
   !> @date October, 2016 - Initial Version
   !> @date January, 2019
   !> - check if tg_varextra is allocated before clean it
   !>
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      !----------------------------------------------------------------

      IF( ALLOCATED(tg_varextra) )THEN
         CALL var_clean(tg_varextra(:))
         DEALLOCATE(tg_varextra)
      ENDIF

   END SUBROUTINE var_clean_extra
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var_read_matrix(td_var, cd_matrix)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine read matrix value from character string in namelist
   !> and fill variable structure value.
   !>
   !> @details
   !> to split matrix, separator use are:<br/>
   !> - ',' for line
   !> - '/' for row
   !> - '\' for level<br/>
   !> Example:<br/>
   !> 3,2,3/1,4,5  => 
   !> @f$ \left( \begin{array}{ccc}
   !> 3 & 2 & 3 \\
   !> 1 & 4 & 5 \end{array} \right) @f$
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable structure
   !> @param[in] cd_matrix matrix value
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR)      , INTENT(INOUT) :: td_var
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_matrix

      ! local variable
      CHARACTER(LEN=lc)                                  :: cl_array
      CHARACTER(LEN=lc)                                  :: cl_line
      CHARACTER(LEN=lc)                                  :: cl_elt

      REAL(dp)         , DIMENSION(:,:,:)  , ALLOCATABLE :: dl_matrix
      REAL(dp)         , DIMENSION(:,:,:,:), ALLOCATABLE :: dl_value

      TYPE(TDIM)       , DIMENSION(:)      , ALLOCATABLE :: tl_dim

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk      
      !----------------------------------------------------------------

      IF( TRIM(cd_matrix) == '' )THEN
         CALL logger_debug("VAR READ MATRIX: no matrix to be read")
      ELSE

         !1- read matrix
         ALLOCATE( dl_matrix(ip_maxmtx, ip_maxmtx, ip_maxmtx) )
         dl_matrix(:,:,:)=td_var%d_fill

         jk=1
         cl_array=fct_split(TRIM(cd_matrix),jk,'\ ')
         CALL logger_debug("VAR MATRIX array "//TRIM(cl_array) )
         DO WHILE( TRIM(cl_array) /= '' )
            jj=1
            cl_line=fct_split(TRIM(cl_array),jj,'/')
            CALL logger_debug("VAR MATRIX line "//TRIM(cl_line) )
            DO WHILE( TRIM(cl_line) /= '' )
               ji=1
               cl_elt=fct_split(TRIM(cl_line),ji,',')
               CALL logger_debug("VAR MATRIX elt "//TRIM(cl_elt) )
               DO WHILE( TRIM(cl_elt) /= '') 
                  READ(cl_elt,*) dl_matrix(ji,jj,jk)
                  ji=ji+1
                  cl_elt=fct_split(TRIM(cl_line),ji,',')
                  CALL logger_debug("VAR MATRIX elt "//TRIM(cl_elt) )
               ENDDO
               jj=jj+1
               cl_line=fct_split(TRIM(cl_array),jj,'/')
               CALL logger_debug("VAR MATRIX line "//TRIM(cl_line) )
            ENDDO
            jk=jk+1
            cl_array=fct_split(TRIM(cd_matrix),jk,'\ ')
            CALL logger_debug("VAR MATRIX array "//TRIM(cl_array) )
         ENDDO

         ! save useful value
         ALLOCATE( dl_value(ji-1,jj-1,jk-1,1) )
         dl_value(:,:,:,1)=dl_matrix(1:ji-1,1:jj-1,1:jk-1)

         DEALLOCATE(dl_matrix)

         ALLOCATE( tl_dim(3) )

         IF( ji-1 > 0 ) tl_dim(1)=dim_init('x',ji-1)
         IF( jj-1 > 0 ) tl_dim(2)=dim_init('y',jj-1)
         IF( jk-1 > 0 ) tl_dim(3)=dim_init('z',jk-1)
         
         CALL var_add_dim(td_var, tl_dim(:))
         ! clean
         CALL dim_clean(tl_dim)
         DEALLOCATE( tl_dim )

         IF( ASSOCIATED(td_var%d_value) ) DEALLOCATE(td_var%d_value)
         CALL var_add_value(td_var, dl_value(:,:,:,:), id_type=NF90_FLOAT)

         DEALLOCATE( dl_value )
      ENDIF

   END SUBROUTINE var_read_matrix
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var__get_extra(td_var)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine add extra information in variable structure.
   !> 
   !> @details 
   !> if variable name is informed in global array of variable structure (tg_varextra).
   !> fill empty parameter on variable structure.
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), INTENT(INOUT) :: td_var

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp

      INTEGER(i4)       :: il_ind
      
      TYPE(TATT)        :: tl_att

      ! loop indices
      INTEGER(i4)       :: ji
      !----------------------------------------------------------------

      IF( ALLOCATED(tg_varextra) )THEN

         il_ind=var_get_index( tg_varextra(:), TRIM(td_var%c_name),  &
                                               TRIM(td_var%c_stdname))
         IF( il_ind /= 0 )THEN

            ! name
            IF( TRIM(td_var%c_name) == '' .AND. &
            &   TRIM(tg_varextra(il_ind)%c_name) /= '' )THEN
               td_var%c_name=TRIM(tg_varextra(il_ind)%c_name)
            ENDIF

            ! standard name
            IF( TRIM(tg_varextra(il_ind)%c_stdname) /= '' .AND. &
            &   ( TRIM(td_var%c_stdname) == '' .OR. &
            &     TRIM(tg_varextra(il_ind)%c_stdname) /= &
            &     TRIM(td_var%c_stdname) ) )THEN
               td_var%c_stdname=TRIM(tg_varextra(il_ind)%c_stdname)
               ! create attibute
               tl_att=att_init('standard_name',TRIM(td_var%c_stdname))
               CALL var_move_att(td_var, tl_att)
            ENDIF

            ! long_name
            IF( TRIM(tg_varextra(il_ind)%c_longname) /= '' .AND. &
            &   ( TRIM(td_var%c_longname) == '' .OR. &
            &     TRIM(tg_varextra(il_ind)%c_longname) /= &
            &     TRIM(td_var%c_longname) ) )THEN
               td_var%c_longname=TRIM(tg_varextra(il_ind)%c_longname)
               ! create attibute
               tl_att=att_init('long_name',TRIM(td_var%c_longname))
               CALL var_move_att(td_var, tl_att)               
            ENDIF

            ! units
            IF( TRIM(td_var%c_units) == '' .AND. & 
            &   TRIM(tg_varextra(il_ind)%c_units) /= '' )THEN
               td_var%c_units=TRIM(tg_varextra(il_ind)%c_units)
               ! create attibute
               tl_att=att_init('units',TRIM(td_var%c_units))
               CALL var_move_att(td_var, tl_att)               
            ENDIF

            ! axis
            IF( TRIM(tg_varextra(il_ind)%c_axis) /= '' .AND. &
            &   ( TRIM(td_var%c_axis) == '' .OR. & 
            &     TRIM(tg_varextra(il_ind)%c_axis) /= &
            &     TRIM(td_var%c_axis) ) )THEN
               td_var%c_axis=TRIM(tg_varextra(il_ind)%c_axis)
               ! create attibute
               IF( TRIM(fct_upper(td_var%c_name)) == TRIM(td_var%c_axis) )THEN
                  tl_att=att_init('axis',TRIM(td_var%c_axis))
               ELSE
                  cl_tmp=""
                  DO ji=LEN(TRIM(td_var%c_axis)),1,-1
                     cl_tmp=TRIM(cl_tmp)//" "//TRIM(td_var%c_axis(ji:ji))
                  ENDDO
                  tl_att=att_init('associate',TRIM(ADJUSTL(cl_tmp)))
               ENDIF
               CALL var_move_att(td_var, tl_att)
            ENDIF

            ! grid point
            IF( TRIM(tg_varextra(il_ind)%c_point) /= '' .AND. &
            &   ( TRIM(td_var%c_point) == '' .OR. &
            &     TRIM(tg_varextra(il_ind)%c_point) /= &
            &     TRIM(td_var%c_point) ) )THEN
               td_var%c_point=TRIM(tg_varextra(il_ind)%c_point)
            ELSE
               IF( TRIM(td_var%c_point) == '' )THEN
                  CALL logger_warn("VAR GET EXTRA: unknown grid point "//&
                  &  "for variable "//TRIM(td_var%c_name)//&
                  &  ". assume it is a T-point.")
                  td_var%c_point='T'
               ENDIF
            ENDIF
            ! create attibute
            tl_att=att_init('grid_point',TRIM(td_var%c_point))
            CALL var_move_att(td_var, tl_att)               

            ! clean
            CALL att_clean(tl_att)

            ! interp
            IF( TRIM(td_var%c_interp(1)) == '' .AND. &
            &   TRIM(tg_varextra(il_ind)%c_interp(1)) /= '' )THEN
               td_var%c_interp(:)=tg_varextra(il_ind)%c_interp(:)
            ENDIF

            ! extrap
            IF( TRIM(td_var%c_extrap(1)) == '' .AND. &
            &   TRIM(tg_varextra(il_ind)%c_extrap(1)) /= '' )THEN
               td_var%c_extrap(:)=tg_varextra(il_ind)%c_extrap(:)
            ENDIF

            ! filter
            IF( TRIM(td_var%c_filter(1)) == '' .AND. &
            &   TRIM(tg_varextra(il_ind)%c_filter(1)) /= '' )THEN
               td_var%c_filter(:)=tg_varextra(il_ind)%c_filter(:)
            ENDIF            

            ! min value
            IF( td_var%d_min == dp_fill .AND. &
            &   tg_varextra(il_ind)%d_min /= dp_fill )THEN
               td_var%d_min=tg_varextra(il_ind)%d_min
            ENDIF

            ! max value
            IF( td_var%d_max == dp_fill .AND. &
            &   tg_varextra(il_ind)%d_max /= dp_fill )THEN
               td_var%d_max=tg_varextra(il_ind)%d_max
            ENDIF

            ! unt
            IF( TRIM(td_var%c_unt) == '' .AND. &
            &   TRIM(tg_varextra(il_ind)%c_unt) /= '' )THEN
               td_var%c_unt=TRIM(tg_varextra(il_ind)%c_unt)
            ENDIF

            ! units factor
            IF( td_var%d_unf == 1._dp .AND. &
            &   tg_varextra(il_ind)%d_unf /= 1._dp )THEN
               td_var%d_unf=tg_varextra(il_ind)%d_unf
            ENDIF

            ! namout
            IF( TRIM(td_var%c_namout) == '' .AND. &
            &   TRIM(tg_varextra(il_ind)%c_namout) /= '' )THEN
               td_var%c_namout=TRIM(tg_varextra(il_ind)%c_namout)
            ENDIF

         ELSE
            CALL logger_warn("VAR GET EXTRA: no extra information on "//&
               &  "variable "//TRIM(td_var%c_name)//". you should define it"//&
               &  " (see variable.cfg).")
         ENDIF

      ELSE

         CALL logger_debug("VAR GET EXTRA: no extra information on variable "//&
         &             " you should have run var_def_extra. ")

      ENDIF

   END SUBROUTINE var__get_extra
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__get_min(cd_name, cd_varinfo) &
         & RESULT (df_min)
   !-------------------------------------------------------------------
   !> @brief
   !> This function check if variable information read in namelist contains 
   !> minimum value and return it if true. 
   !> 
   !> @details
   !> minimum value is assume to follow string "min ="
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015 
   !> - change way to get information in namelist, 
   !> value follows string "min ="
   !> @date Feb, 2016
   !> - check character just after keyword
   !>
   !> @param[in] cd_name      variable name
   !> @param[in] cd_varinfo   variable information read in namelist 
   !> @return minimum value to be used (FillValue if none)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_name
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_varinfo

      ! function
      REAL(dp)                        :: df_min

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      CHARACTER(LEN=lc) :: cl_min
      
      INTEGER(i4)       :: il_ind

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------
      ! init
      cl_min=''
      df_min=dp_fill

      ji=1
      cl_tmp=fct_split(cd_varinfo,ji,';')
      DO WHILE( TRIM(cl_tmp) /= '' )
         il_ind=INDEX(TRIM(cl_tmp),'min')
         IF( il_ind /= 0 )THEN
            ! check character just after
            jj=il_ind+LEN('min')
            IF(  TRIM(cl_tmp(jj:jj)) == ' ' .OR. &
            &    TRIM(cl_tmp(jj:jj)) == '=' )THEN
               cl_min=fct_split(cl_tmp,2,'=')
               EXIT
            ENDIF
         ENDIF
         ji=ji+1
         cl_tmp=fct_split(cd_varinfo,ji,';')         
      ENDDO

      IF( TRIM(cl_min) /= '' )THEN
         IF( fct_is_real(cl_min) )THEN
            READ(cl_min,*) df_min
            CALL logger_debug("VAR GET MIN: will use minimum value of "//&
               &  TRIM(fct_str(df_min))//" for variable "//TRIM(cd_name) )
         ELSE
            CALL logger_error("VAR GET MIN: invalid minimum value ("//&
               & TRIM(cl_min)//") for variable "//TRIM(cd_name)//&
               & ". check namelist." )
         ENDIF
      ENDIF

   END FUNCTION var__get_min
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__get_max(cd_name, cd_varinfo) &
         & RESULT (df_max)
   !-------------------------------------------------------------------
   !> @brief
   !> This function check if variable information read in namelist contains 
   !> maximum value and return it if true. 
   !> 
   !> @details
   !> maximum value is assume to follow string "max ="
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015 
   !> - change way to get information in namelist, 
   !> value follows string "max ="
   !> @date Feb, 2016
   !> - check character just after keyword
   !>
   !> @param[in] cd_name      variable name
   !> @param[in] cd_varinfo   variable information read in namelist 
   !> @return maximum value to be used (FillValue if none)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_name
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_varinfo

      ! function
      REAL(dp)                        :: df_max

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      CHARACTER(LEN=lc) :: cl_max
      
      INTEGER(i4)       :: il_ind

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------
      ! init
      cl_max=''
      df_max=dp_fill

      ji=1
      cl_tmp=fct_split(cd_varinfo,ji,';')
      DO WHILE( TRIM(cl_tmp) /= '' )
         il_ind=INDEX(TRIM(cl_tmp),'max')
         IF( il_ind /= 0 )THEN
            ! check character just after
            jj=il_ind+LEN('max')
            IF(  TRIM(cl_tmp(jj:jj)) == ' ' .OR. &
            &    TRIM(cl_tmp(jj:jj)) == '=' )THEN
               cl_max=fct_split(cl_tmp,2,'=')
               EXIT
            ENDIF
         ENDIF
         ji=ji+1
         cl_tmp=fct_split(cd_varinfo,ji,';')         
      ENDDO

      IF( TRIM(cl_max) /= '' )THEN
         IF( fct_is_real(cl_max) )THEN
            READ(cl_max,*) df_max
            CALL logger_debug("VAR GET MAX: will use maximum value of "//&
               &  TRIM(fct_str(df_max))//" for variable "//TRIM(cd_name) )
         ELSE
            CALL logger_error("VAR GET MAX: invalid maximum value for "//&
               &  "variable "//TRIM(cd_name)//". check namelist." )
         ENDIF
      ENDIF

   END FUNCTION var__get_max
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__get_unf(cd_name, cd_varinfo) &
         & RESULT (df_unf)
   !-------------------------------------------------------------------
   !> @brief
   !> This function check if variable information read in namelist contains 
   !> units factor value and return it if true. 
   !> 
   !> @details
   !> units factor value is assume to follow string "unf ="
   !>
   !> @author J.Paul
   !> @date June, 2015 - Initial Version
   !> @date Feb, 2016
   !> - check character just after keyword
   !>
   !> @param[in] cd_name      variable name
   !> @param[in] cd_varinfo   variable information read in namelist 
   !> @return untis factor value to be used (FillValue if none)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_name
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_varinfo

      ! function
      REAL(dp)                        :: df_unf

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      CHARACTER(LEN=lc) :: cl_unf
      
      INTEGER(i4)       :: il_ind

      REAL(dp)          :: dl_unf

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------
      ! init
      cl_unf=''
      df_unf=dp_fill

      ji=1
      cl_tmp=fct_split(cd_varinfo,ji,';')
      DO WHILE( TRIM(cl_tmp) /= '' )
         il_ind=INDEX(TRIM(cl_tmp),'unf')
         IF( il_ind /= 0 )THEN
            ! check character just after
            jj=il_ind+LEN('unf')
            IF(  TRIM(cl_tmp(jj:jj)) == ' ' .OR. &
               & TRIM(cl_tmp(jj:jj)) == '=' )THEN
               cl_unf=fct_split(cl_tmp,2,'=')
               EXIT
            ENDIF
         ENDIF
         ji=ji+1
         cl_tmp=fct_split(cd_varinfo,ji,';')         
      ENDDO

      IF( TRIM(cl_unf) /= '' )THEN
         dl_unf=math_compute(cl_unf)
         IF( dl_unf /= dp_fill )THEN
            df_unf = dl_unf
            CALL logger_debug("VAR GET UNITS FACTOR: will use units factor "//&
               &  "value of "//TRIM(fct_str(df_unf))//" for variable "//&
               &   TRIM(cd_name) )
         ELSE
            CALL logger_error("VAR GET UNITS FACTOR: invalid units factor "//&
               &  "value for variable "//TRIM(cd_name)//". check namelist." )
         ENDIF
      ENDIF

   END FUNCTION var__get_unf
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__get_interp(cd_name, cd_varinfo) &
         & RESULT (cf_interp)
   !-------------------------------------------------------------------
   !> @brief
   !> This function check if variable information read in namelist contains 
   !> interpolation method and return it if true. 
   !> 
   !> @details 
   !> interpolation method is assume to follow string "int ="
   !>
   !> compare method name with the list of interpolation method available (see
   !> module global).
   !> check if factor (*rhoi, /rhoj..) are present.<br/>
   !> Example:<br/> 
   !> - int=cubic/rhoi ; ext=dist_weight
   !> - int=bilin
   !> see @ref interp module for more information.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015 
   !> - change way to get information in namelist, 
   !> value follows string "int ="
   !> @date Feb, 2016
   !> - check character just after keyword
   !>
   !> @param[in] cd_name      variable name
   !> @param[in] cd_varinfo   variable information read in namelist
   !> @return array of character information about interpolation 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_name
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_varinfo

      ! function
      CHARACTER(LEN=lc), DIMENSION(2) :: cf_interp

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      CHARACTER(LEN=lc) :: cl_int
      CHARACTER(LEN=lc) :: cl_factor
      
      INTEGER(i4)       :: il_ind
      INTEGER(i4)       :: il_len

      INTEGER(i4)       :: il_mul
      INTEGER(i4)       :: il_div

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      cf_interp(:)=''

      ji=1
      cl_tmp=fct_split(cd_varinfo,ji,';')
      DO WHILE( TRIM(cl_tmp) /= '' )
         il_ind=INDEX(TRIM(cl_tmp),'int')
         IF( il_ind /= 0 )THEN
            ! check character just after
            jj=il_ind+LEN('int')
            IF(  TRIM(cl_tmp(jj:jj)) == ' ' .OR. &
            &    TRIM(cl_tmp(jj:jj)) == '=' )THEN
               cl_int=fct_split(cl_tmp,2,'=')
               EXIT
            ENDIF
         ENDIF
         ji=ji+1
         cl_tmp=fct_split(cd_varinfo,ji,';')         
      ENDDO

      IF( TRIM(cl_int) /= '' )THEN
         DO jj=1,ip_ninterp
            il_ind= INDEX(fct_lower(cl_int),TRIM(cp_interp_list(jj)))
            IF( il_ind /= 0 )THEN

               cf_interp(1)=TRIM(cp_interp_list(jj))
               il_len=LEN(TRIM(cp_interp_list(jj)))
               
               ! look for factor
               IF( il_ind==1 )THEN
                  cl_factor=cl_int(il_len+1:)
               ELSE
                  cl_factor=cl_int(1:il_ind-1)
               ENDIF
               il_mul=SCAN(TRIM(cl_factor),'*')
               il_div=SCAN(TRIM(cl_factor),'/')
               
               il_len=LEN(cl_factor)
               IF( il_mul /= 0 )THEN
                  IF( il_mul==1 )THEN
                     cl_factor=cl_factor(2:il_len)
                  ELSE
                     cl_factor=cl_factor(1:il_mul-1)
                  ENDIF

               ELSE IF( il_div /=0 )THEN
                  IF( il_div==1 )THEN
                     cl_factor=cl_factor(2:il_len)
                  ELSE
                     cl_factor=cl_factor(1:il_div-1)
                  ENDIF

               ELSE
                  cl_factor=''
               ENDIF

               SELECT CASE(TRIM(cl_factor))
                  CASE('rhoi','rhoj','rhok')
                     IF( il_mul /= 0 ) cf_interp(2)='*'//TRIM(cl_factor)
                     IF( il_div /= 0 ) cf_interp(2)='/'//TRIM(cl_factor)
                  CASE('')
                     cf_interp(2)=''
                  CASE DEFAULT
                     cf_interp(2)=''
                     CALL logger_error("VAR GET INTERP: variable "//&
                        &     TRIM(cd_name)//&
                        &     " invalid factor coefficient. check namelist. "//&
                        &     " factor should be choose between rhox rhoy rhoz.")
               END SELECT

               EXIT
            ENDIF
         ENDDO
      ENDIF

   END FUNCTION var__get_interp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__get_extrap(cd_name, cd_varinfo) &
         & RESULT (cf_extrap)
   !-------------------------------------------------------------------
   !> @brief
   !> This function check if variable information read in namelist contains 
   !> extrapolation method and return it if true. 
   !> 
   !> @details 
   !> extrapolation method is assume to follow string "ext ="
   !> 
   !> compare method name with the list of extrapolation method available (see
   !> module global).<br/>
   !> Example:<br/>
   !> - int=cubic ; ext=dist_weight
   !> - ext=min_error
   !> see @ref extrap module for more information.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015 
   !> - change way to get information in namelist, 
   !> value follows string "ext ="
   !> @date Feb, 2016
   !> - check character just after keyword
   !>
   !> @param[in] cd_name      variable name
   !> @param[in] cd_varinfo   variable information read in namelist
   !> @return array of character information about extrapolation
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_name
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_varinfo

      ! function
      CHARACTER(LEN=lc), DIMENSION(1) :: cf_extrap

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      CHARACTER(LEN=lc) :: cl_ext

      INTEGER(i4)       :: il_ind

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      cf_extrap(:)=''

      ji=1
      cl_tmp=fct_split(cd_varinfo,ji,';')
      DO WHILE( TRIM(cl_tmp) /= '' )
         il_ind=INDEX(TRIM(cl_tmp),'ext')
         IF( il_ind /= 0 )THEN
            ! check character just after
            jj=il_ind+LEN('ext')
            IF(  TRIM(cl_tmp(jj:jj)) == ' ' .OR. &
            &    TRIM(cl_tmp(jj:jj)) == '=' )THEN
               cl_ext=fct_split(cl_tmp,2,'=')
               EXIT
            ENDIF
         ENDIF
         ji=ji+1
         cl_tmp=fct_split(cd_varinfo,ji,';')         
      ENDDO

      IF( TRIM(cl_ext) /= '' )THEN
         DO jj=1,ip_nextrap
            IF( TRIM(fct_lower(cl_ext)) == TRIM(cp_extrap_list(jj)) )THEN
               cf_extrap(1)=TRIM(cp_extrap_list(jj))

               CALL logger_trace("VAR GET EXTRAP: variable "//TRIM(cd_name)//&
                  &  " will use extrapolation method "//TRIM(cf_extrap(1)) )

               EXIT
            ENDIF
         ENDDO
      ENDIF


   END FUNCTION var__get_extrap
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__get_filter(cd_name, cd_varinfo) &
         & RESULt (cf_filter)
   !-------------------------------------------------------------------
   !> @brief
   !> This function check if variable information read in namelist contains 
   !> filter method and return it if true 
   !> 
   !> @details 
   !> filter method is assume to follow string "flt ="
   !>
   !> compare method name with the list of filter method available (see
   !> module global).
   !> look for the number of run, using '*' separator, and method parameters inside
   !> bracket.<br/>
   !> Example:<br/>
   !> - int=cubic ; flt=2*hamming(2,3)
   !> - flt=hann
   !> see @ref filter module for more information.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date June, 2015 
   !> - change way to get information in namelist, 
   !> value follows string "flt ="
   !> @date Feb, 2016
   !> - check character just after keyword
   !>
   !> @param[in] cd_name      variable name
   !> @param[in] cd_varinfo   variable information read in namelist 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_name
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_varinfo

      ! function
      CHARACTER(LEN=lc), DIMENSION(5) :: cf_filter

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      CHARACTER(LEN=lc) :: cl_flt
      INTEGER(i4)       :: il_ind

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      cf_filter(:)=''

      ji=1
      cl_tmp=fct_split(cd_varinfo,ji,';')
      DO WHILE( TRIM(cl_tmp) /= '' )
         il_ind=INDEX(TRIM(cl_tmp),'flt')
         IF( il_ind /= 0 )THEN
            ! check character just after
            jj=il_ind+LEN('flt')
            IF(  TRIM(cl_tmp(jj:jj)) == ' ' .OR. &
            &    TRIM(cl_tmp(jj:jj)) == '=' )THEN
               cl_flt=fct_split(cl_tmp,2,'=')
               EXIT
            ENDIF
         ENDIF
         ji=ji+1
         cl_tmp=fct_split(cd_varinfo,ji,';')         
      ENDDO
      
      IF( TRIM(cl_flt) /= '' )THEN
         DO jj=1,ip_nfilter
            il_ind=INDEX(fct_lower(cl_flt),TRIM(cp_filter_list(jj)))
            IF( il_ind /= 0 )THEN
               cf_filter(1)=TRIM(cp_filter_list(jj))

               ! look for number of run
               il_ind=SCAN(fct_lower(cl_flt),'*')
               IF( il_ind /=0 )THEN
                  IF( fct_is_num(cl_flt(1:il_ind-1)) )THEN
                     cf_filter(2)=TRIM(cl_flt(1:il_ind-1))
                  ELSE IF( fct_is_num(cl_flt(il_ind+1:)) )THEN
                     cf_filter(2)=TRIM(cl_flt(il_ind+1:))
                  ELSE
                     cf_filter(2)='1'
                  ENDIF
               ELSE
                  cf_filter(2)='1'
               ENDIF

               ! look for filter parameter
               il_ind=SCAN(fct_lower(cl_flt),'(')
               IF( il_ind /=0 )THEN
                  cl_flt=TRIM(cl_flt(il_ind+1:))
                  il_ind=SCAN(fct_lower(cl_flt),')')
                  IF( il_ind /=0 )THEN
                     cl_flt=TRIM(cl_flt(1:il_ind-1))
                     ! look for cut-off frequency
                     cf_filter(3)=fct_split(cl_flt,1,',')
                     ! look for halo size
                     cf_filter(4)=fct_split(cl_flt,2,',')
                     ! look for alpha parameter
                     cf_filter(5)=fct_split(cl_flt,3,',')
                  ELSE
                     CALL logger_error("VAR GET FILTER: variable "//&
                     &  TRIM(cd_name)//&
                     &  " unclosed parentheses. check namelist. ")
                  ENDIF
               ELSE
                  cf_filter(3)=''
                  cf_filter(4)=''
                  cf_filter(5)=''
               ENDIF
   
               CALL logger_trace("VAR GET FILTER: name   "//TRIM(cf_filter(1)))
               CALL logger_trace("VAR GET FILTER: nturn  "//TRIM(cf_filter(2)))
               CALL logger_trace("VAR GET FILTER: cutoff "//TRIM(cf_filter(3)))
               CALL logger_trace("VAR GET FILTER: halo   "//TRIM(cf_filter(4)))
               CALL logger_trace("VAR GET FILTER: alpha  "//TRIM(cf_filter(5)))

               EXIT
            ENDIF
         ENDDO
      ENDIF

   END FUNCTION var__get_filter
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__get_unt(cd_name, cd_varinfo) &
         & RESULT (cf_unt)
   !-------------------------------------------------------------------
   !> @brief
   !> This function check if variable information read in namelist contains 
   !> output unit and return it if true. 
   !> 
   !> @details 
   !> output unit is assume to follow string "unt ="
   !>
   !> @author J.Paul
   !> @date June, 2015 - Initial Version
   !> @date February, 2016
   !> - check character just after keyword
   !>
   !> @param[in] cd_name      variable name
   !> @param[in] cd_varinfo   variable information read in namelist
   !> @return unit string character 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_name
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_varinfo

      ! function
      CHARACTER(LEN=lc)               :: cf_unt

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      
      INTEGER(i4)       :: il_ind

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      cf_unt=''

      ji=1
      cl_tmp=fct_split(cd_varinfo,ji,';')
      DO WHILE( TRIM(cl_tmp) /= '' )
         il_ind=INDEX(TRIM(cl_tmp),'unt')
         IF( il_ind /= 0 )THEN
            ! check character just after
            jj=il_ind+LEN('unt')
            IF(  TRIM(cl_tmp(jj:jj)) == ' ' .OR. &
               & TRIM(cl_tmp(jj:jj)) == '=' )THEN
               cf_unt=fct_split(cl_tmp,2,'=')
               EXIT
            ENDIF
         ENDIF
         ji=ji+1
         cl_tmp=fct_split(cd_varinfo,ji,';')         
      ENDDO

      IF( TRIM(cf_unt) /= '' )THEN
         CALL logger_debug("VAR GET UNIT: will use output unit "//&
            &  TRIM(cf_unt)//" for variable "//&
            &  TRIM(cd_name) )
      ENDIF

   END FUNCTION var__get_unt
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var__get_namout(cd_name, cd_varinfo) &
         & RESULT (cf_namout)
   !-------------------------------------------------------------------
   !> @brief
   !> This function check if variable information read in namelist contains 
   !> variable ouptut name and return it if true. 
   !> 
   !> @details 
   !> output name is assume to follow string "out ="
   !>
   !> @author J.Paul
   !> @date February, 2019 - Initial Version
   !>
   !> @param[in] cd_name      variable name
   !> @param[in] cd_varinfo   variable information read in namelist
   !> @return ouptut name string character 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_name
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_varinfo

      ! function
      CHARACTER(LEN=lc)               :: cf_namout

      ! local variable
      CHARACTER(LEN=lc) :: cl_tmp
      
      INTEGER(i4)       :: il_ind

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      cf_namout=''

      ji=1
      cl_tmp=fct_split(cd_varinfo,ji,';')
      DO WHILE( TRIM(cl_tmp) /= '' )
         il_ind=INDEX(TRIM(cl_tmp),'out')
         IF( il_ind /= 0 )THEN
            ! check character just after
            jj=il_ind+LEN('out')
            IF(  TRIM(cl_tmp(jj:jj)) == ' ' .OR. &
               & TRIM(cl_tmp(jj:jj)) == '=' )THEN
               cf_namout=fct_split(cl_tmp,2,'=')
               EXIT
            ENDIF
         ENDIF
         ji=ji+1
         cl_tmp=fct_split(cd_varinfo,ji,';')         
      ENDDO

      IF( TRIM(cf_namout) /= '' )THEN
         CALL logger_debug("VAR GET NAMOUT: will use output name "//&
            &  TRIM(cf_namout)//" for variable "//&
            &  TRIM(cd_name) )
      ENDIF

   END FUNCTION var__get_namout
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var_max_dim(td_var) &
         & RESULT (tf_dim)
   !-------------------------------------------------------------------
   !> @brief 
   !> This function search and save the biggest dimensions use 
   !> in an array of variable structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_var array of variable structure
   !> @return array of dimension 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR), DIMENSION(:), INTENT(IN) :: td_var
      
      ! function
      TYPE(TDIM), DIMENSION(ip_maxdim)     :: tf_dim

      ! local variable
      INTEGER(i4) :: il_nvar
      
      ! loop indices
      INTEGER(i4) :: ji   
      !-------------------------------------------------------------------
                          
      il_nvar=SIZE(td_var(:))
                          
      tf_dim(:)=dim_copy(td_var(1)%t_dim(:))

      IF( il_nvar > 1 )THEN
         DO ji=2,il_nvar     
                             
            IF( td_var(ji)%t_dim(1)%l_use .AND. &
            &   td_var(ji)%t_dim(1)%i_len >= tf_dim(1)%i_len )THEN
               tf_dim(1)=dim_copy(td_var(ji)%t_dim(1))
            ENDIF            
                             
            IF( td_var(ji)%t_dim(2)%l_use .AND. &
            &   td_var(ji)%t_dim(2)%i_len >= tf_dim(2)%i_len )THEN 
               tf_dim(2)=dim_copy(td_var(ji)%t_dim(2))
            ENDIF            
                             
            IF( td_var(ji)%t_dim(3)%l_use .AND. &
            &   td_var(ji)%t_dim(3)%i_len >= tf_dim(3)%i_len )THEN 
               tf_dim(3)=dim_copy(td_var(ji)%t_dim(3))
            ENDIF            
                             
            IF( td_var(ji)%t_dim(4)%l_use .AND. &
            &   td_var(ji)%t_dim(4)%i_len >= tf_dim(4)%i_len )THEN
               tf_dim(4)=dim_copy(td_var(ji)%t_dim(4))
            ENDIF

         ENDDO
      ENDIF

   END FUNCTION var_max_dim
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var_limit_value(td_var)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine forced minimum and maximum value of variable,
   !> with value of variable structure attribute d_min and d_max.
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable structure
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), INTENT(INOUT) :: td_var

      ! local variable

      ! loop indices
      !----------------------------------------------------------------

      IF( ASSOCIATED(td_var%d_value) )THEN
         !1- forced minimum value
         IF( td_var%d_min /= dp_fill )THEN
            WHERE( td_var%d_value(:,:,:,:) /= td_var%d_fill .AND. &
               &   td_var%d_value(:,:,:,:) <  td_var%d_min )
               td_var%d_value(:,:,:,:)=td_var%d_min
            END WHERE
         ENDIF

         !2- forced maximum value 
         IF( td_var%d_max /= dp_fill )THEN
            WHERE( td_var%d_value(:,:,:,:) /= td_var%d_fill .AND. &
               &   td_var%d_value(:,:,:,:) >  td_var%d_max )
               td_var%d_value(:,:,:,:)=td_var%d_max
            END WHERE
         ENDIF

      ENDIF

   END SUBROUTINE var_limit_value
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var_chg_name(td_var)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine replace name of the variable,
   !> 
   !> @details
   !> output name (namout) is read from the namelist.
   !>
   !> @note the variable value should be already read.
   !>
   !> @author J.Paul
   !> @date February, 2019 - Initial Version
   !>
   !> @param[inout] td_var variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), INTENT(INOUT) :: td_var

      ! local variable
      ! loop indices
      !----------------------------------------------------------------

      IF( ASSOCIATED(td_var%d_value) )THEN
         !- change variable name 
         IF( TRIM(td_var%c_namout) /= TRIM(td_var%c_name) .AND. &
         &   TRIM(td_var%c_namout) /= '' )THEN
            td_var%c_name = TRIM(td_var%c_namout)
         ENDIF

      ENDIF

   END SUBROUTINE var_chg_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var_chg_unit(td_var)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine replace unit name of the variable,
   !> and apply unit factor to the value of this variable.
   !> 
   !> @details
   !> new unit name (unt) and unit factor (unf) are read from the namelist.
   !>
   !> @note the variable value should be already read.
   !>
   !> @author J.Paul
   !> @date June, 2015 - Initial Version
   !>
   !> @param[inout] td_var variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), INTENT(INOUT) :: td_var

      ! local variable
      TYPE(TATT)                :: tl_att

      ! loop indices
      !----------------------------------------------------------------

      IF( ASSOCIATED(td_var%d_value) )THEN
         !- change value
         IF( td_var%d_unf /= 1._dp )THEN
            WHERE( td_var%d_value(:,:,:,:) /= td_var%d_fill )
               td_var%d_value(:,:,:,:)=td_var%d_value(:,:,:,:)*td_var%d_unf
            END WHERE

            !- change scale factor and offset to avoid mistake
            tl_att=att_init('scale_factor',1._dp)
            CALL var_move_att(td_var, tl_att)

            tl_att=att_init('add_offset',0._dp)
            CALL var_move_att(td_var, tl_att)
         ENDIF

         !- change unit name 
         IF( TRIM(td_var%c_unt) /= TRIM(td_var%c_units) .AND. &
         &   TRIM(td_var%c_unt) /= '' )THEN
            tl_att=att_init('units',TRIM(td_var%c_unt))
            CALL var_move_att(td_var,tl_att)
         ENDIF
         ! clean
         CALL att_clean(tl_att)

      ENDIF

   END SUBROUTINE var_chg_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var_check_dim(td_var)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine check variable dimension expected, as defined in
   !> file 'variable.cfg'.
   !> 
   !> @details 
   !> compare dimension used in variable structure with string character
   !> axis from configuration file. 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var    variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), INTENT(INOUT) :: td_var

      ! local variable
      INTEGER(i4)       :: il_naxis
      INTEGER(i4)       :: il_ndim
      CHARACTER(LEN=lc) :: cl_dim

      LOGICAL           :: ll_warn

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( TRIM(td_var%c_axis) /= '' )THEN

         cl_dim=''
         DO ji=1,ip_maxdim
            IF( td_var%t_dim(ji)%l_use )THEN
               cl_dim=TRIM(cl_dim)//TRIM(fct_upper(td_var%t_dim(ji)%c_sname))
            ENDIF
         ENDDO

         il_naxis=LEN( TRIM(ADJUSTL(td_var%c_axis)) )
         il_ndim =LEN( TRIM(ADJUSTL(cl_dim)) )
         IF( il_naxis >= il_ndim )THEN
            ll_warn=.FALSE.
            DO ji=1,il_naxis
               IF( INDEX(TRIM(cl_dim),td_var%c_axis(ji:ji)) == 0 )THEN
                  CALL logger_debug("VAR CHECK DIM: "//TRIM(cl_dim)//&
                     &              " "//TRIM(td_var%c_axis(ji:ji)) )
                  ll_warn=.TRUE.
                  EXIT
               ENDIF
            ENDDO

            IF( ll_warn )THEN
               CALL logger_warn("VAR CHECK DIM: variable dimension ("//&
                  &             TRIM(cl_dim)//") not conform with dimension"//&
                  &             " expected ("//TRIM(td_var%c_axis)//"). ") 
            ENDIF
         ELSE
            ! too much dimension
            CALL logger_warn("VAR CHECK DIM: too much dimension for "//&
               &             "variable "//TRIM(td_var%c_name)//".")
            cl_dim=TRIM(fct_upper(cp_dimorder))
            il_ndim =LEN( TRIM(ADJUSTL(cl_dim)) )
            DO ji=1,il_ndim 
               IF( INDEX(TRIM(td_var%c_axis),cl_dim(ji:ji)) == 0 )THEN
                  IF( td_var%t_dim(ji)%l_use )THEN
                     IF( td_var%t_dim(ji)%i_len == 1 )THEN
                        ! remove useless dimension
                        CALL var_del_dim(td_var,td_var%t_dim(ji))
                     ELSE
                        CALL logger_warn("VAR CHECK DIM: variable "//&
                           &     TRIM(td_var%c_name)//" should not use"//&
                           &     " dimension "//TRIM(td_var%t_dim(ji)%c_name))
                     ENDIF
                  ENDIF
               ENDIF
            ENDDO
         ENDIF

      ELSE
         ! no information on variable dimension expected
      ENDIF

   END SUBROUTINE var_check_dim
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var_reorder(td_var, cd_dimorder)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine reshape variable value and dimension 
   !> in variable structure.
   !> @details
   !> output dimension will be ordered as defined in 
   !> input array of dimension
   !> Optionaly you could specify output dimension order with
   !> string character of dimension
   !> 
   !> @author J.Paul
   !> @date August, 2014 - Initial Version
   !> @date July 2015 
   !> - do not use dim_disorder anymore
   !>
   !> @param[inout] td_var       variable structure
   !> @param[in]    cd_dimorder  string character of dimension order to be used
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR)              , INTENT(INOUT) :: td_var
      CHARACTER(LEN=ip_maxdim), INTENT(IN   ), OPTIONAL :: cd_dimorder

      ! local variable
      CHARACTER(LEN=lc)                             :: cl_dimorder

      REAL(dp)  , DIMENSION(:,:,:,:)  , ALLOCATABLE :: dl_value

      TYPE(TDIM), DIMENSION(ip_maxdim)              :: tl_dim

      ! loop indices
      !----------------------------------------------------------------

      cl_dimorder=TRIM(cp_dimorder)
      IF( PRESENT(cd_dimorder) ) cl_dimorder=TRIM(ADJUSTL(cd_dimorder))

      CALL logger_debug("VAR REORDER: work on "//TRIM(td_var%c_name)//&
         &  " new dimension order "//TRIM(cl_dimorder))

      tl_dim(:)=dim_copy(td_var%t_dim(:))

      CALL dim_reorder(tl_dim(:),TRIM(cl_dimorder))

      ALLOCATE(dl_value(tl_dim(1)%i_len, &
         &              tl_dim(2)%i_len, &
         &              tl_dim(3)%i_len, &
         &              tl_dim(4)%i_len ))

      dl_value(:,:,:,:)=dim_reshape_2xyzt(tl_dim, &
         &                                td_var%d_value(:,:,:,:))

      ! change dimension
      td_var%t_dim(:)=dim_copy(tl_dim(:))
      ! change value
      DEALLOCATE( td_var%d_value )
      CALL var_add_value(td_var, dl_value(:,:,:,:))

      ! clean
      DEALLOCATE(dl_value)
      CALL dim_clean(tl_dim(:))

   END SUBROUTINE var_reorder
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var_get_unit(td_var) &
         & RESULT (if_unit)
   !-------------------------------------------------------------------
   !> @brief
   !> This function get the next unused unit in array of variable structure. 
   !> 
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[in] td_var array of variable structure 
   !> @return free variable id
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), DIMENSION(:), INTENT(IN) :: td_var

      ! function
      INTEGER(i4)                          :: if_unit

      ! local variable
      ! loop indices
      !----------------------------------------------------------------

      if_unit=MAXVAL(td_var(:)%i_id)+1

   END FUNCTION var_get_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var_to_date(td_var) &
         & RESULT (tf_date)
   !-------------------------------------------------------------------
   !> @brief
   !> This function convert a time variable structure in date structure. 
   !> 
   !> @author J.Paul
   !> @date November, 2014 - Initial Version
   !> @date January, 2019
   !> -  add case for units in hours  
   !>
   !> @param[in] td_var time variable structure 
   !> @return date structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), INTENT(IN) :: td_var

      ! function
      TYPE(TDATE)            :: tf_date

      ! local variable
      CHARACTER(LEN=lc) :: cl_step
      CHARACTER(LEN=lc) :: cl_date

      INTEGER(i4) :: il_attid

      INTEGER(i8) :: kl_nsec

      TYPE(TDATE) :: tl_dateo
      ! loop indices
      !----------------------------------------------------------------

      IF( INDEX(TRIM(td_var%c_name),'time') /= 0 )THEN
         IF( ASSOCIATED(td_var%d_value) )THEN

            il_attid=att_get_index(td_var%t_att(:),'units')
            IF( il_attid /=0 )THEN
               cl_step=fct_split(td_var%t_att(il_attid)%c_value,1,'since')
               cl_date=fct_split(td_var%t_att(il_attid)%c_value,2,'since')

               SELECT CASE(TRIM(cl_step))
                  CASE('seconds')
                     kl_nsec=INT(td_var%d_value(1,1,1,1),i8)
                  CASE('hours')
                     kl_nsec=INT(td_var%d_value(1,1,1,1)*3600,i8)
                  CASE('days')
                     kl_nsec=INT(td_var%d_value(1,1,1,1)*86400,i8)
                  CASE DEFAULT
                     CALL logger_error("VAR TO DATE: unknown units format "//&
                     &  "in variable "//TRIM(td_var%c_name))
               END SELECT

               CALL logger_trace("VAR TO DATE: "//fct_str(kl_nsec)//&
                  &              "seconds since "//TRIM(cl_date))

               tl_dateo=date_init(cl_date)

               tf_date=date_init(kl_nsec,tl_dateo)

            ELSE
               CALL logger_error("VAR TO DATE: no attribute units in "//&
                  &              "variable "//TRIM(td_var%c_name))
            ENDIF
         ELSE
            CALL logger_error("VAR TO DATE: no value associated to "//&
               &              "variable "//TRIM(td_var%c_name))
         ENDIF
      ELSE
         CALL logger_error("VAR TO DATE: variable "//TRIM(td_var%c_name)//&
            &              "can not be convert in date.")
      ENDIF

   END FUNCTION var_to_date
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE var_get_dummy(cd_dummy)
   !-------------------------------------------------------------------
   !> @brief This subroutine fill dummy variable array
   !>
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
      cm_dumvar(:)=''

      ! read namelist
      INQUIRE(FILE=TRIM(cd_dummy), EXIST=ll_exist)
      IF( ll_exist )THEN
    
         il_fileid=fct_getunit()
   
         OPEN( il_fileid, FILE=TRIM(cd_dummy), &
            &             FORM='FORMATTED',    &
            &             ACCESS='SEQUENTIAL', &
            &             STATUS='OLD',        &
            &             ACTION='READ',       &
            &             IOSTAT=il_status)
         CALL fct_err(il_status)
         IF( il_status /= 0 )THEN
            CALL logger_fatal("DIM GET DUMMY: opening "//TRIM(cd_dummy))
         ENDIF
   
         READ( il_fileid, NML = namdum )
         im_ndumvar  = in_ndumvar
         cm_dumvar(:)= cn_dumvar(:)

         CLOSE( il_fileid )

         IF( im_ndumvar > ip_maxdumcfg )THEN
            CALL logger_fatal("VAR GET dUMMY : too much dummy variables &
               &              ( >"//fct_str(ip_maxdumcfg)//" ). &
               &              set ip_maxdumcfg to higher value.")
         ENDIF

      ENDIF
   
   END SUBROUTINE var_get_dummy
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION var_is_dummy(td_var) &
         & RESULT (lf_dummy)
   !-------------------------------------------------------------------
   !> @brief This function check if variable is defined as dummy variable
   !> in configuraton file
   !>
   !> @author J.Paul
   !> @date September, 2015 - Initial Version
   !> @date, May, 2019
   !> - use number of dummy elt in do-loop
   !>
   !> @param[in] td_var variable structure
   !> @return true if variable is dummy variable 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR), INTENT(IN) :: td_var
      
      ! function
      LOGICAL                :: lf_dummy
      
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      lf_dummy=.FALSE.
      DO ji=1,im_ndumvar !ip_maxdumcfg
         IF( fct_lower(td_var%c_name) == fct_lower(cm_dumvar(ji)) )THEN
            lf_dummy=.TRUE.
            EXIT
         ENDIF
      ENDDO

   END FUNCTION var_is_dummy
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE var

