!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief
!> This module manage boundary.
!>
!> @details
!>    define type TBDY:<br/>
!> @code
!>    TYPE(TBDY) :: tl_bdy<br/>
!> @endcode
!>
!>    to initialise boundary structure:<br/>
!> @code
!>    tl_bdy=boundary_init(td_var, [ld_north,] [ld_south,] [ld_east,] [ld_west,]
!>    [cd_north,] [cd_south,] [cd_east,] [cd_west,] [ld_oneseg])
!> @endcode
!>       - td_var is variable structure
!>       - ld_north is logical to force used of north boundary [optional]
!>       - ld_south is logical to force used of north boundary [optional]
!>       - ld_east  is logical to force used of north boundary [optional]
!>       - ld_west  is logical to force used of north boundary [optional]
!>       - cd_north is string character description of north boundary [optional]
!>       - cd_south is string character description of south boundary [optional]
!>       - cd_east  is string character description of east  boundary [optional]
!>       - cd_west  is string character description of west  boundary [optional]
!>       - ld_oneseg is logical to force to use only one segment for each boundary [optional]
!>
!>    to get boundary cardinal:<br/>
!>    - tl_bdy\%c_card
!>
!>    to know if boundary is use:<br/>
!>    - tl_bdy\%l_use
!>
!>    to know if boundary come from namelist (cn_north,..):<br/>
!>    - tl_bdy\%l_nam
!>
!>    to get the number of segment in boundary:<br/>
!>    - tl_bdy\%i_nseg
!>
!>    to get array of segment in boundary:<br/>
!>    - tl_bdy\%t_seg(:)
!>
!>    to get orthogonal segment index of north boundary:<br/>
!>    - tl_bdy\%t_seg(jp_north)%\i_index
!>
!>    to get segment width of south boundary:<br/>
!>    - tl_bdy\%t_seg(jp_south)%\i_width
!>
!>    to get segment first indice of east boundary:<br/>
!>    - tl_bdy\%t_seg(jp_east)%\i_first
!>
!>    to get segment last indice of west boundary:<br/>
!>    - tl_bdy\%t_seg(jp_west)%\i_last
!>
!>    to print information about boundary:<br/>
!> @code
!>    CALL boundary_print(td_bdy)
!> @endcode
!>       - td_bdy is boundary structure or a array of boundary structure
!>
!>    to clean boundary structure:<br/>
!> @code
!>    CALL boundary_clean(td_bdy)
!> @endcode
!>
!>    to get indices of each semgent for each boundary:<br/>
!> @code
!>    CALL boundary_get_indices( td_bdy, td_var, ld_oneseg)
!> @endcode
!>       - td_bdy is boundary structure
!>       - td_var is variable structure
!>       - ld_oneseg is logical to force to use only one segment for each boundary [optional]
!>
!>    to check boundary indices and corner:<br/>
!> @code
!>    CALL boundary_check(td_bdy, td_var)
!> @endcode
!>       - td_bdy is boundary structure
!>       - td_var is variable structure
!>
!>    to check boundary corner:<br/>
!> @code
!>    CALL boundary_check_corner(td_bdy, td_var)
!> @endcode
!>       - td_bdy is boundary structure
!>       - td_var is variable structure
!>
!>    to create filename with cardinal name inside:<br/>
!> @code
!>    cl_filename=boundary_set_filename(cd_file, cd_card)
!> @endcode
!>       - cd_file = original file name
!>       - cd_card = cardinal name
!>
!>    to swap array for east and north boundary:<br/>
!> @code
!>    CALL boundary_swap( td_var, td_bdy )
!> @endcode
!>       - td_var is variable strucutre
!>       - td_bdy is boundary strucutre
!>
!> @author J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date September, 2014 
!> - add boundary description
!> @date November, 2014 
!> - Fix memory leaks bug
!> @date February, 2015 
!> - Do not change indices read from namelist
!> - Change string character format of boundary read from namelist, 
!>  see boundary__get_info
!> 
!> @todo add schematic to boundary structure description
!> 
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE boundary

   USE netcdf                          ! nf90 library                           
   USE global                          ! global parameter
   USE phycst                          ! physical constant
   USE kind                            ! F90 kind parameter
   USE logger                          ! log file manager
   USE fct                             ! basic useful function
   USE var                             ! variable manager

   IMPLICIT NONE

   ! NOTE_avoid_public_variables_if_possible

   ! type and variable
   PUBLIC :: TBDY     !< boundary structure
   PUBLIC :: TSEG     !< segment structure

   PRIVATE :: im_width !< boundary width

   ! function and subroutine
   PUBLIC :: boundary_copy         !< copy boundary structure
   PUBLIC :: boundary_init         !< initialise boundary structure
   PUBLIC :: boundary_print        !< print information about boundary
   PUBLIC :: boundary_clean        !< clean boundary structure 
   PUBLIC :: boundary_get_indices  !< get indices of each semgent for each boundary.
   PUBLIC :: boundary_check        !< check boundary indices and corner.
   PUBLIC :: boundary_check_corner !< check boundary corner
   PUBLIC :: boundary_set_filename !< set boundary filename
   PUBLIC :: boundary_swap         !< swap array for north and east boundary

   PRIVATE :: boundary__clean_unit      ! clean boundary structure 
   PRIVATE :: boundary__clean_arr       ! clean array of boundary structure 
   PRIVATE :: boundary__init_wrapper    ! initialise a boundary structure
   PRIVATE :: boundary__init            ! initialise basically a boundary structure
   PRIVATE :: boundary__copy_unit       ! copy boundary structure in another
   PRIVATE :: boundary__copy_arr        ! copy boundary structure in another
   PRIVATE :: boundary__add_seg         ! add one segment structure to a boundary 
   PRIVATE :: boundary__del_seg         ! remove all segments of a boundary
   PRIVATE :: boundary__get_info        ! get boundary information from boundary description string character.
   PRIVATE :: boundary__get_seg_number  ! compute the number of sea segment for one boundary
   PRIVATE :: boundary__get_seg_indices ! get segment indices for one boundary 
   PRIVATE :: boundary__print_unit      ! print information about one boundary
   PRIVATE :: boundary__print_arr       ! print information about a array of boundary
   
   PRIVATE :: seg__init       ! initialise segment structure
   PRIVATE :: seg__clean      ! clean segment structure
   PRIVATE :: seg__clean_unit ! clean one segment structure
   PRIVATE :: seg__clean_arr  ! clean array of segment structure
   PRIVATE :: seg__copy       ! copy segment structure in another
   PRIVATE :: seg__copy_unit  ! copy segment structure in another
   PRIVATE :: seg__copy_arr   ! copy array of segment structure in another

   TYPE TSEG   !< segment structure
      INTEGER(i4) :: i_index = 0 !< segment index
      INTEGER(i4) :: i_width = 0 !< segment width
      INTEGER(i4) :: i_first = 0 !< segment first indice 
      INTEGER(i4) :: i_last  = 0 !< segment last indices
   END TYPE TSEG

   TYPE TBDY !< boundary structure
      CHARACTER(LEN=lc) :: c_card = ''          !< boundary cardinal
      LOGICAL           :: l_use  = .FALSE.     !< boundary use or not 
      LOGICAL           :: l_nam  = .FALSE.     !< boundary get from namelist
      INTEGER(i4)       :: i_nseg = 0           !< number of segment in boundary
      TYPE(TSEG), DIMENSION(:), POINTER :: t_seg => NULL() !<  array of segment structure
   END TYPE TBDY

   ! module variable
   INTEGER(i4), PARAMETER :: im_width=10

   INTERFACE boundary_init
      MODULE PROCEDURE boundary__init_wrapper 
   END INTERFACE boundary_init

   INTERFACE boundary_print
      MODULE PROCEDURE boundary__print_unit 
      MODULE PROCEDURE boundary__print_arr 
   END INTERFACE boundary_print

   INTERFACE boundary_clean
      MODULE PROCEDURE boundary__clean_unit   
      MODULE PROCEDURE boundary__clean_arr    
   END INTERFACE

   INTERFACE seg__clean
      MODULE PROCEDURE seg__clean_unit   
      MODULE PROCEDURE seg__clean_arr    
   END INTERFACE

   INTERFACE boundary_copy
      MODULE PROCEDURE boundary__copy_unit 
      MODULE PROCEDURE boundary__copy_arr 
   END INTERFACE   

   INTERFACE seg__copy
      MODULE PROCEDURE seg__copy_unit   ! copy segment structure
      MODULE PROCEDURE seg__copy_arr    ! copy array of segment structure
   END INTERFACE   

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION boundary__copy_arr(td_bdy) &
         & RESULT (tf_bdy)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine copy a array of boundary structure in another one
   !> @details 
   !>
   !> @warning do not use on the output of a function who create or read an
   !> attribute (ex: tl_bdy=boundary_copy(boundary_init()) is forbidden).
   !> This will create memory leaks.
   !> @warning to avoid infinite loop, do not use any function inside 
   !> this subroutine
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date November, 2014
   !> - use function instead of overload assignment operator 
   !> (to avoid memory leak)
   !
   !> @param[in] td_bdy   array of boundary structure
   !> @return copy of input array of boundary structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TBDY), DIMENSION(:)   , INTENT(IN) :: td_bdy

      ! function
      TYPE(TBDY), DIMENSION(SIZE(td_bdy(:)))  :: tf_bdy

      ! local variable
      ! loop indices
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      DO jk=1,SIZE(td_bdy(:))
         tf_bdy(jk)=boundary_copy(td_bdy(jk))
      ENDDO

   END FUNCTION boundary__copy_arr
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION boundary__copy_unit(td_bdy) &
         & RESULT (tf_bdy)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine copy boundary structure in another one
   !> @details 
   !>
   !> @warning do not use on the output of a function who create or read an
   !> attribute (ex: tl_bdy=boundary_copy(boundary_init()) is forbidden).
   !> This will create memory leaks.
   !> @warning to avoid infinite loop, do not use any function inside 
   !> this subroutine
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date November, 2014
   !> - use function instead of overload assignment operator 
   !> (to avoid memory leak)
   !
   !> @param[in] td_bdy   boundary structure
   !> @return copy of input boundary structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TBDY), INTENT(IN)  :: td_bdy

      ! function
      TYPE(TBDY)              :: tf_bdy

      ! local variable
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! copy variable name, id, ..
      tf_bdy%c_card     = TRIM(td_bdy%c_card)
      tf_bdy%i_nseg     = td_bdy%i_nseg
      tf_bdy%l_use      = td_bdy%l_use

      ! copy segment
      IF( ASSOCIATED(tf_bdy%t_seg) )THEN
         CALL seg__clean(tf_bdy%t_seg(:))
         DEALLOCATE(tf_bdy%t_seg)
      ENDIF
      IF( ASSOCIATED(td_bdy%t_seg) .AND. tf_bdy%i_nseg > 0 )THEN
         ALLOCATE( tf_bdy%t_seg(tf_bdy%i_nseg) )
         DO ji=1,tf_bdy%i_nseg
            tf_bdy%t_seg(ji)=td_bdy%t_seg(ji)
         ENDDO
      ENDIF

   END FUNCTION boundary__copy_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE boundary__clean_unit(td_bdy)
   !-------------------------------------------------------------------
   !> @brief This subroutine clean boundary structure
   !
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2019
   !> - nullify segment structure inside boundary structure 
   !
   !> @param[inout] td_bdy boundary strucutre
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TBDY), INTENT(INOUT) :: td_bdy

      ! local variable
      TYPE(TBDY) :: tl_bdy ! empty boundary strucutre

      ! loop indices
      !----------------------------------------------------------------

      CALL logger_info( &
      &  " CLEAN: reset boundary "//TRIM(td_bdy%c_card) )

      ! del segment
      IF( ASSOCIATED(td_bdy%t_seg) )THEN
         ! clean each segment
         CALL seg__clean(td_bdy%t_seg(:) )
         DEALLOCATE( td_bdy%t_seg )
         NULLIFY(td_bdy%t_seg)
      ENDIF

      ! replace by empty structure
      td_bdy=boundary_copy(tl_bdy)

   END SUBROUTINE boundary__clean_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE boundary__clean_arr(td_bdy)
   !-------------------------------------------------------------------
   !> @brief This subroutine clean array of boundary structure
   !
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !
   !> @param[inout] td_bdy boundary strucutre
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TBDY), DIMENSION(:), INTENT(INOUT) :: td_bdy

      ! local variable
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      DO ji=SIZE(td_bdy(:)),1,-1
         CALL boundary_clean( td_bdy(ji) )
      ENDDO

   END SUBROUTINE boundary__clean_arr
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION boundary_set_filename(cd_file, cd_card, id_seg, cd_date) &
         &  RESULT (cf_file)
   !------------------------------------------------------------------- 
   !> @brief This function put cardinal name and date inside file name.
   ! 
   !> @details 
   !>    Examples :
   !>       cd_file="boundary.nc"
   !>       cd_card="west" 
   !>       id_seg =2
   !>       cd_date=y2015m07d16
   !> 
   !>       function return "boundary_west_2_y2015m07d16.nc"
   !> 
   !>       cd_file="boundary.nc"
   !>       cd_card="west" 
   !> 
   !>       function return "boundary_west.nc"
   !> 
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   ! 
   !> @param[in] cd_file   file name 
   !> @param[in] cd_card   cardinal name 
   !> @param[in] id_seg    segment number 
   !> @param[in] cd_date   file date (format: y????m??d??) 
   !> @return file name with cardinal name inside
   !------------------------------------------------------------------- 

      IMPLICIT NONE 

      ! Argument
      CHARACTER(LEN=*), INTENT(IN) :: cd_file
      CHARACTER(LEN=*), INTENT(IN) :: cd_card
      INTEGER(i4)     , INTENT(IN), OPTIONAL :: id_seg
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: cd_date

      ! function 
      CHARACTER(LEN=lc)            :: cf_file

      ! local variable 
      CHARACTER(LEN=lc) :: cl_dirname
      CHARACTER(LEN=lc) :: cl_basename
      CHARACTER(LEN=lc) :: cl_base
      CHARACTER(LEN=lc) :: cl_suffix
      CHARACTER(LEN=lc) :: cl_segnum
      CHARACTER(LEN=lc) :: cl_date
      CHARACTER(LEN=lc) :: cl_name

      INTEGER(i4)       :: il_ind
      INTEGER(i4)       :: il_indend

      ! loop indices 
      !---------------------------------------------------------------- 
      ! init
      cf_file=''

      IF( TRIM(cd_file) /= '' .AND. TRIM(cd_card) /= '' )THEN

         cl_dirname = fct_dirname( TRIM(cd_file))
         IF( TRIM(cl_dirname) == '' ) cl_dirname='.'

         cl_basename= fct_basename(TRIM(cd_file))

         cl_base  =fct_split(TRIM(cl_basename),1,'.')
         cl_suffix=fct_split(TRIM(cl_basename),2,'.')
         
         ! add segment number
         IF( PRESENT(id_seg) )THEN
            cl_segnum="_"//TRIM(fct_str(id_seg))
         ELSE
            cl_segnum=""
         ENDIF

         ! add date
         IF( PRESENT(cd_date) )THEN
            cl_date="_"//TRIM(ADJUSTL(cd_date))
         ELSE
            cl_date=""
         ENDIF

         ! special case for obcdta
         il_ind=INDEX(cl_base,'_obcdta_')
         IF( il_ind/=0 )THEN
            il_ind=il_ind-1+8
            il_indend=LEN_TRIM(cl_base)

            cl_name=TRIM(cl_base(1:il_ind))//TRIM(cd_card)//&
               &     TRIM(cl_segnum)//"_"//TRIM(cl_base(il_ind+1:il_indend))//&
               &     TRIM(cl_date)//"."//TRIM(cl_suffix)
         ELSE
            cl_name=TRIM(cl_base)//"_"//TRIM(cd_card)//TRIM(cl_segnum)//&
               &     TRIM(cl_date)//"."//TRIM(cl_suffix)
         ENDIF

         cf_file=TRIM(cl_dirname)//"/"//TRIM(cl_name)
      ELSE
         CALL logger_error("BOUNDARY SET FILENAME: file or cardinal name "//&
         &  " are empty")
      ENDIF
 
   END FUNCTION boundary_set_filename 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION boundary__init_wrapper(td_var,                               &
         &                         ld_north, ld_south, ld_east, ld_west, &
         &                         cd_north, cd_south, cd_east, cd_west, &
         &                         ld_oneseg) &
         &  RESULT (tf_bdy)
   !------------------------------------------------------------------- 
   !> @brief This function initialise a boundary structure.
   ! 
   !> @details 
   !>  Boundaries for each cardinal will be compute with variable structure.
   !>  It means that orthogonal index, first and last indices of each 
   !>  sea segment will be compute automatically.
   !>  However you could specify which boundary to use or not with
   !>  arguments ln_north, ln_south, ln_east, ln_west.
   !>  And boundary description could be specify with argument
   !>  cn_north, cn_south, cn_east, cn_west.
   !>  For each cardinal you could specify orthogonal index, 
   !>  first and last indices (in this order) and boundary width (between
   !>  parentheses).
   !> ex : cn_north='index,first,last(width)'
   !> You could specify more than one segment for each boundary. 
   !> However each segment will have the same width. So you do not need to
   !> specify it for each segment.
   !> ex : cn_north='index1,first1,last1(width)|index2,first2,last2'
   !>
   !> @warn Boundaries are compute on T point, but expressed on U,V point.
   !> change will be done to get data on other point when need be. 
   !>
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   !> @date September, 2014
   !> - add boolean to use only one segment for each boundary
   !> - check boundary width
   ! 
   !> @param[in] td_var    variable structure 
   !> @param[in] ld_north  use north boundary or not 
   !> @param[in] ld_south  use south boundary or not 
   !> @param[in] ld_east   use east  boundary or not 
   !> @param[in] ld_west   use west  boundary or not 
   !> @param[in] cd_north  north boundary description 
   !> @param[in] cd_south  south boundary description 
   !> @param[in] cd_east   east  boundary description 
   !> @param[in] cd_west   west  boundary description 
   !> @param[in] ld_oneseg force to use only one segment for each boundary 
   !> @return boundary structure
   !------------------------------------------------------------------- 

      IMPLICIT NONE 

      ! Argument
      TYPE(TVAR)       , INTENT(IN) :: td_var
      LOGICAL          , INTENT(IN), OPTIONAL :: ld_north
      LOGICAL          , INTENT(IN), OPTIONAL :: ld_south
      LOGICAL          , INTENT(IN), OPTIONAL :: ld_east 
      LOGICAL          , INTENT(IN), OPTIONAL :: ld_west 
      CHARACTER(LEN=lc), INTENT(IN), OPTIONAL :: cd_north
      CHARACTER(LEN=lc), INTENT(IN), OPTIONAL :: cd_south
      CHARACTER(LEN=lc), INTENT(IN), OPTIONAL :: cd_east 
      CHARACTER(LEN=lc), INTENT(IN), OPTIONAL :: cd_west
      LOGICAL          , INTENT(IN), OPTIONAL :: ld_oneseg 

      ! function 
      TYPE(TBDY)       , DIMENSION(ip_ncard)  :: tf_bdy

      ! local variable 
      INTEGER(i4)                            :: il_width
      INTEGER(i4)      , DIMENSION(ip_ncard) :: il_max_width
      INTEGER(i4)      , DIMENSION(ip_ncard) :: il_index
      INTEGER(i4)      , DIMENSION(ip_ncard) :: il_min
      INTEGER(i4)      , DIMENSION(ip_ncard) :: il_max

      CHARACTER(LEN=lc), DIMENSION(ip_ncard) :: cl_card

      TYPE(TBDY)                             :: tl_tmp

      TYPE(TSEG)                             :: tl_seg

      LOGICAL                                :: ll_oneseg

      ! loop indices 
      INTEGER(i4) :: ji
      INTEGER(i4) :: jk
      !---------------------------------------------------------------- 
      IF( .NOT. ASSOCIATED(td_var%d_value) )THEN
         CALL logger_error("BOUNDARY INIT: no value associated to variable "//&
         &              TRIM(td_var%c_name) )
      ELSEIF( TRIM(td_var%c_point) /= 'T' )THEN
         CALL logger_error("BOUNDARY INIT: can not compute boundary with "//&
         &                 "variable "//TRIM(td_var%c_name)//&
         &                 ". need a variable on T point." )
      ELSE
         ll_oneseg=.TRUE.
         IF( PRESENT(ld_oneseg) ) ll_oneseg=ld_oneseg

         ! init
         tf_bdy(jp_north)=boundary__init('north',ld_north)
         tf_bdy(jp_south)=boundary__init('south',ld_south)
         tf_bdy(jp_east )=boundary__init('east ',ld_east )
         tf_bdy(jp_west )=boundary__init('west ',ld_west )

         ! if EW cyclic no east west boundary and force to use one segment
         IF( td_var%i_ew >= 0 )THEN
            CALL logger_info("BOUNDARY INIT: cyclic domain, "//&
            &  "no East West boundary")
            tf_bdy(jp_east )%l_use=.FALSE.
            tf_bdy(jp_west )%l_use=.FALSE.

            CALL logger_info("BOUNDARY INIT: force to use one segment due"//&
            &  " to EW cyclic domain")
            ll_oneseg=.TRUE.
         ENDIF

         il_index(jp_north)=td_var%t_dim(2)%i_len-ip_ghost
         il_index(jp_south)=1+ip_ghost
         il_index(jp_east )=td_var%t_dim(1)%i_len-ip_ghost
         il_index(jp_west )=1+ip_ghost

         il_min(jp_north)=1
         il_min(jp_south)=1
         il_min(jp_east )=1
         il_min(jp_west )=1

         il_max(jp_north)=td_var%t_dim(1)%i_len
         il_max(jp_south)=td_var%t_dim(1)%i_len
         il_max(jp_east )=td_var%t_dim(2)%i_len
         il_max(jp_west )=td_var%t_dim(2)%i_len
 
         cl_card=(/'','','',''/)
         IF( PRESENT(cd_north) ) cl_card(jp_north)=TRIM(cd_north)
         IF( PRESENT(cd_south) ) cl_card(jp_south)=TRIM(cd_south)
         IF( PRESENT(cd_east ) ) cl_card(jp_east )=TRIM(cd_east )
         IF( PRESENT(cd_west ) ) cl_card(jp_west )=TRIM(cd_west )

         il_max_width(jp_north)=INT(0.5*(td_var%t_dim(2)%i_len-2*ip_ghost))
         il_max_width(jp_south)=INT(0.5*(td_var%t_dim(2)%i_len-2*ip_ghost))
         il_max_width(jp_east )=INT(0.5*(td_var%t_dim(1)%i_len-2*ip_ghost))
         il_max_width(jp_west )=INT(0.5*(td_var%t_dim(1)%i_len-2*ip_ghost))

         DO jk=1,ip_ncard

            ! check boundary width
            IF( il_max_width(jk) <= im_width )THEN
               IF( il_max_width(jk) <= 0 )THEN
                  CALL logger_fatal("BOUNDARY INIT: domain too small to define"//&
                  &                " boundaries.")
               ELSE
                  CALL logger_warn("BOUNDARY INIT: default boundary width too "//&
                  &                "large for boundaries. force to use boundary"//&
                  &                " on one point")
                  il_width=1
               ENDIF
            ELSE
               il_width=im_width
            ENDIF

            ! define default segment
            tl_seg=seg__init(il_index(jk),il_width,il_min(jk),il_max(jk))

            IF( tf_bdy(jk)%l_use )THEN

               ! get namelist information
               tl_tmp=boundary__get_info(cl_card(jk),jk)

               ! get segments indices
               DO ji=1,tl_tmp%i_nseg
                  CALL boundary__add_seg(tf_bdy(jk),tl_tmp%t_seg(ji))
               ENDDO
               ! indices from namelist or not
               tf_bdy(jk)%l_nam=tl_tmp%l_nam

               CALL boundary_clean(tl_tmp)

               IF( tf_bdy(jk)%i_nseg == 0 )THEN
                  ! add default segment
                  CALL boundary__add_seg(tf_bdy(jk),tl_seg)
               ELSE
                  ! fill undefined value
                  WHERE( tf_bdy(jk)%t_seg(:)%i_index == 0 ) 
                     tf_bdy(jk)%t_seg(:)%i_index = tl_seg%i_index
                  END WHERE               
                  WHERE( tf_bdy(jk)%t_seg(:)%i_width == 0 ) 
                     tf_bdy(jk)%t_seg(:)%i_width = tl_seg%i_width
                  END WHERE
                  WHERE( tf_bdy(jk)%t_seg(:)%i_first == 0 ) 
                     tf_bdy(jk)%t_seg(:)%i_first = tl_seg%i_first
                  END WHERE
                  WHERE( tf_bdy(jk)%t_seg(:)%i_last == 0 ) 
                     tf_bdy(jk)%t_seg(:)%i_last = tl_seg%i_last
                  END WHERE
               ENDIF

            ENDIF
            ! clean
            CALL seg__clean(tl_seg)

         ENDDO

         CALL boundary_get_indices(tf_bdy(:), td_var, ll_oneseg)

         CALL boundary_check(tf_bdy, td_var)

      ENDIF
 
   END FUNCTION boundary__init_wrapper 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION boundary__init(cd_card, ld_use, ld_nam, td_seg) &
         &  RESULT (tf_bdy)
   !------------------------------------------------------------------- 
   !> @brief This function initialise basically a boundary structure with
   !> cardinal name.
   ! 
   !> @details 
   !> optionnaly you could specify if this boundary is used or not, 
   !> and add one segment structure.
   ! 
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   ! 
   !> @param[in]  cd_card  cardinal name
   !> @param[in]  ld_use   boundary use or not
   !> @param[in]  td_seg   segment structure
   !> @return boundary structure
   !------------------------------------------------------------------- 

      IMPLICIT NONE 

      ! Argument
      CHARACTER(LEN=*), INTENT(IN) :: cd_card
      LOGICAL         , INTENT(IN), OPTIONAL :: ld_use 
      LOGICAL         , INTENT(IN), OPTIONAL :: ld_nam 
      TYPE(TSEG)      , INTENT(IN), OPTIONAL :: td_seg

      ! function 
      TYPE(TBDY)                   :: tf_bdy

      ! local variable 
      ! loop indices 
      !---------------------------------------------------------------- 

      SELECT CASE(TRIM(cd_card))
         CASE ('north','south','east','west')
         
            tf_bdy%c_card=TRIM(cd_card)

            tf_bdy%l_use=.TRUE.
            IF( PRESENT(ld_use) ) tf_bdy%l_use=ld_use

            tf_bdy%l_nam=.FALSE.
            IF( PRESENT(ld_nam) ) tf_bdy%l_nam=ld_nam

            IF( PRESENT(td_seg) )THEN
               CALL boundary__add_seg(tf_bdy, td_seg)
            ENDIF

         CASE DEFAULT
            CALL logger_error("BOUNDARY INIT: invalid cardinal name")
      END SELECT

   END FUNCTION boundary__init
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE boundary__add_seg(td_bdy, td_seg) 
   !------------------------------------------------------------------- 
   !> @brief This subroutine add one segment structure to a boundary structure 
   ! 
   !> @details 
   ! 
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   ! 
   !> @param[inout] td_bdy boundary structure  
   !> @param[in] td_seg    segment structure  
   !------------------------------------------------------------------- 

      IMPLICIT NONE 

      ! Argument 
      TYPE(TBDY), INTENT(INOUT) :: td_bdy
      TYPE(TSEG), INTENT(IN   ) :: td_seg

      ! local variable 
      INTEGER(i4)                            :: il_status
      TYPE(TSEG) , DIMENSION(:), ALLOCATABLE :: tl_seg

      ! loop indices 
      !---------------------------------------------------------------- 

      IF( td_bdy%i_nseg > 0 )THEN
         ! already other segment in boundary structure
         ALLOCATE( tl_seg(td_bdy%i_nseg), stat=il_status )
         IF(il_status /= 0 )THEN
            CALL logger_error( &
            &  " BOUNDARY ADD SEG: not enough space to put segments ")
         ELSE
            ! save temporary segment
            tl_seg(:)=seg__copy(td_bdy%t_seg(:))

            CALL seg__clean(td_bdy%t_seg(:))
            DEALLOCATE( td_bdy%t_seg )
            ALLOCATE( td_bdy%t_seg(td_bdy%i_nseg+1), stat=il_status )
            IF(il_status /= 0 )THEN
               CALL logger_error( &
               &  " BOUNDARY ADD SEG: not enough space to put segments ")
            ENDIF

            ! copy segment in boundary before
            td_bdy%t_seg(1:td_bdy%i_nseg)=seg__copy(tl_seg(:))

            ! clean
            CALL seg__clean(tl_seg(:))
            DEALLOCATE(tl_seg)            
            
         ENDIF
      ELSE
         ! no segment in boundary structure
         IF( ASSOCIATED(td_bdy%t_seg) )THEN
            CALL seg__clean(td_bdy%t_seg(:))
            DEALLOCATE(td_bdy%t_seg)
         ENDIF
         ALLOCATE( td_bdy%t_seg(td_bdy%i_nseg+1), stat=il_status )
         IF(il_status /= 0 )THEN
            CALL logger_error( &
            &  " BOUNDARY ADD SEG: not enough space to put segments ")
         ENDIF         
      ENDIF
 
      ! update number of segment
      td_bdy%i_nseg=td_bdy%i_nseg+1

      ! add new segment
      td_bdy%t_seg(td_bdy%i_nseg)=seg__copy(td_seg)

   END SUBROUTINE boundary__add_seg 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE boundary__del_seg(td_bdy) 
   !------------------------------------------------------------------- 
   !> @brief This subroutine remove all segments of a boundary structure 
   ! 
   !> @details 
   ! 
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   ! 
   !> @param[inout]  td_bdy   boundary structure
   !------------------------------------------------------------------- 

      IMPLICIT NONE 

      ! Argument 
      TYPE(TBDY), INTENT(INOUT) :: td_bdy

      ! local variable 
      ! loop indices 
      !---------------------------------------------------------------- 

      IF( ASSOCIATED(td_bdy%t_seg) )THEN
         CALL seg__clean(td_bdy%t_seg(:))
         DEALLOCATE(td_bdy%t_seg)
      ENDIF
      !update number of segment
      td_bdy%i_nseg=0

   END SUBROUTINE boundary__del_seg 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION boundary__get_info(cd_card, id_jcard) &
         & RESULT (tf_bdy)
   !------------------------------------------------------------------- 
   !> @brief This function get information about boundary from string character. 
   ! 
   !> @details
   !> This string character that will be passed through namelist could contains
   !> orthogonal index, first and last indices, of each segment. 
   !> And also the width of all segments of this boundary.
   !>   cn_north='index1,first1:last1(width)|index2,first2:last2'
   !> 
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   !> @date february, 2015 
   !> - do not change indices read from namelist
   !> - change format cn_north
   ! 
   !> @param[in] cd_card   boundary description
   !> @param[in] id_jcard  boundary index
   !> @return boundary structure
   !------------------------------------------------------------------- 

      IMPLICIT NONE 

      ! Argument 
      CHARACTER(LEN=lc), INTENT(IN) :: cd_card
      INTEGER(i4)      , INTENT(IN) :: id_jcard

      ! function 
      TYPE(TBDY)                    :: tf_bdy

      ! local variable 
      INTEGER(i4)       :: il_width
      INTEGER(i4)       :: il_ind1
      INTEGER(i4)       :: il_ind2

      CHARACTER(LEN=lc) :: cl_seg
      CHARACTER(LEN=lc) :: cl_index
      CHARACTER(LEN=lc) :: cl_width
      CHARACTER(LEN=lc) :: cl_tmp
      CHARACTER(LEN=lc) :: cl_first
      CHARACTER(LEN=lc) :: cl_last 

      TYPE(TSEG)        :: tl_seg

      ! loop indices 
      INTEGER(i4) :: ji
      !---------------------------------------------------------------- 
 
      ji=1
      cl_seg=fct_split(cd_card,ji)

      il_width=0
      ! look for segment width 
      ! width should be the same for all segment of one boundary
      IF( TRIM(cl_seg)   /= '' )THEN

         ! initialise boundary
         ! temporaty boundary, so it doesn't matter which caridnal is used
         tf_bdy=boundary__init('north',ld_nam=.TRUE.)

         il_ind1=SCAN(fct_lower(cl_seg),'(')
         IF( il_ind1 /=0 )THEN
            cl_width=TRIM(cl_seg(il_ind1+1:))

            il_ind2=SCAN(fct_lower(cl_width),')')
            IF( il_ind2 /=0 )THEN
               cl_width=TRIM(cl_width(1:il_ind2-1))
               READ(cl_width,*) il_width
            ELSE
               CALL logger_error("BOUNDARY INIT: unclosed parentheses."//&
               &  " check namelist. ")
            ENDIF
         ENDIF

      ENDIF 

      DO WHILE( TRIM(cl_seg) /= '' )

         cl_index=fct_split(cl_seg,1,',')
         ! remove potential width information
         il_ind1=SCAN(fct_lower(cl_index),'(')
         IF( il_ind1 /=0 )THEN
            il_ind2=SCAN(fct_lower(cl_index),')')
            IF( il_ind2 /=0 )THEN
               cl_index=TRIM(cl_index(:il_ind1-1))//TRIM(cl_index(il_ind2+1:))
            ELSE
               CALL logger_error("BOUNDARY INIT: unclosed parentheses."//&
               &  " check namelist. ")
            ENDIF
         ENDIF
      
         
         cl_tmp=fct_split(cl_seg,2,',')


         cl_first=fct_split(cl_tmp,1,':')
         ! remove potential width information
         il_ind1=SCAN(fct_lower(cl_first),'(')
         IF( il_ind1 /=0 )THEN
            il_ind2=SCAN(fct_lower(cl_first),')')
            IF( il_ind2 /=0 )THEN
               cl_first=TRIM(cl_first(:il_ind1-1))//TRIM(cl_first(il_ind2+1:))
            ELSE
               CALL logger_error("BOUNDARY INIT: unclosed parentheses."//&
               &  " check namelist. ")
            ENDIF
         ENDIF         
         
         cl_last =fct_split(cl_tmp,2,':')
         ! remove potential width information
         il_ind1=SCAN(fct_lower(cl_last),'(')
         IF( il_ind1 /=0 )THEN
            il_ind2=SCAN(fct_lower(cl_last),')')
            IF( il_ind2 /=0 )THEN
               cl_last=TRIM(cl_last(:il_ind1-1))//TRIM(cl_last(il_ind2+1:))
            ELSE
               CALL logger_error("BOUNDARY INIT: unclosed parentheses."//&
               &  " check namelist. ")
            ENDIF
         ENDIF

         IF( il_width /= 0 ) tl_seg%i_width=il_width

         IF( TRIM(cl_index) /= '' ) READ(cl_index,*) tl_seg%i_index
         IF( TRIM(cl_first) /= '' ) READ(cl_first,*) tl_seg%i_first
         IF( TRIM(cl_last)  /= '' ) READ(cl_last ,*) tl_seg%i_last

         ! index expressed on U,V point, move on T point.
         SELECT CASE(id_jcard)
            CASE(jp_north, jp_east)
               tl_seg%i_index=tl_seg%i_index+1
         END SELECT

         IF( (tl_seg%i_first == 0 .AND.  tl_seg%i_last == 0) .OR. &
         &   (tl_seg%i_first /= 0 .AND.  tl_seg%i_last /= 0) )THEN
            CALL boundary__add_seg(tf_bdy, tl_seg)
         ELSE
            CALL logger_error("BOUNDARY INIT: first or last segment indices "//&
            &              "are missing . check namelist.")
         ENDIF

         ji=ji+1
         cl_seg=fct_split(cd_card,ji)

         ! clean
         CALL seg__clean(tl_seg)
      ENDDO 

   END FUNCTION boundary__get_info 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE boundary_get_indices(td_bdy, td_var, ld_oneseg) 
   !------------------------------------------------------------------- 
   !> @brief This subroutine get indices of each semgent for each boundary.
   ! 
   !> @details 
   !> indices are compute from variable value, actually variable fill value,
   !> which is assume to be land mask. 
   !> Boundary structure should have been initialized before running 
   !> this subroutine. Segment indices will be search between first and last
   !> indies, at this orthogonal index.
   !> 
   !> Optionnally you could forced to use only one segment for each boundary.
   !> 
   !> @warning number of segment (i_nseg) will be change, before the number 
   !> of segment structure
   ! 
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   ! 
   !> @param[inout] td_bdy boundary structure  
   !> @param[in] td_var    variable structure 
   !> @param[in] ld_onseg  use only one sgment for each boundary 
   !------------------------------------------------------------------- 

      IMPLICIT NONE 

      ! Argument
      TYPE(TBDY) , DIMENSION(ip_ncard), INTENT(INOUT) :: td_bdy
      TYPE(TVAR)                      , INTENT(IN   ) :: td_var
      LOGICAL                         , INTENT(IN   ), OPTIONAL :: ld_oneseg

      ! local variable 
      INTEGER(i4) :: il_index
      INTEGER(i4) :: il_width
      INTEGER(i4) :: il_first
      INTEGER(i4) :: il_last 

      LOGICAL     :: ll_oneseg

      TYPE(TSEG)  :: tl_seg

      ! loop indices
      INTEGER(i4) :: jk
      !---------------------------------------------------------------- 
 
      ll_oneseg=.TRUE.
      IF( PRESENT(ld_oneseg) ) ll_oneseg=ld_oneseg

      DO jk=1,ip_ncard
         IF( .NOT. td_bdy(jk)%l_use .OR. td_bdy(jk)%l_nam )THEN
            ! nothing to be done
         ELSE

            IF( .NOT. ASSOCIATED(td_bdy(jk)%t_seg) )THEN
               CALL logger_error("BOUNDARY GET INDICES: no segment "//&
               &  " associated to "//TRIM(td_bdy(jk)%c_card)//&
               &  " boundary. you should have run boundary_init before"//&
               &  " running boundary_get_indices" )
            ELSE
               il_index=td_bdy(jk)%t_seg(1)%i_index
               il_width=td_bdy(jk)%t_seg(1)%i_width
               il_first=td_bdy(jk)%t_seg(1)%i_first
               il_last =td_bdy(jk)%t_seg(1)%i_last
 
               CALL boundary__get_seg_number( td_bdy(jk), td_var)

               CALL boundary__get_seg_indices( td_bdy(jk), td_var, &
               &                               il_index, il_width, &
               &                               il_first, il_last )

               IF( ll_oneseg .AND. td_bdy(jk)%l_use )THEN
                  tl_seg=seg__copy(td_bdy(jk)%t_seg(1))
                  ! use last indice of last segment
                  tl_seg%i_last=td_bdy(jk)%t_seg(td_bdy(jk)%i_nseg)%i_last

                  ! remove all segment from boundary
                  CALL boundary__del_seg(td_bdy(jk))

                  ! add one segment
                  CALL boundary__add_seg(td_bdy(jk),tl_seg)

                  ! clean
                  CALL seg__clean(tl_seg)
               ENDIF

            ENDIF

         ENDIF

      ENDDO

   END SUBROUTINE boundary_get_indices 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE boundary__get_seg_number(td_bdy, td_var) 
   !------------------------------------------------------------------- 
   !> @brief This subroutine compute the number of sea segment. 
   ! 
   !> @details 
   !> It use variable value, actually variable fill value
   !> (which is assume to be land mask), to compute the number of segment between
   !> first and last indices at boundary orthogonal index.
   !> @warning number of segment (i_nseg) will be change, before the number 
   !> of segment structure
   ! 
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   ! 
   !> @param[inout] td_bdy boundary structure 
   !> @param[in] td_var    variable structure 
   !------------------------------------------------------------------- 

      IMPLICIT NONE 

      ! Argument
      TYPE(TBDY) , INTENT(INOUT) :: td_bdy
      TYPE(TVAR) , INTENT(IN   ) :: td_var

      ! local variable 
      REAL(dp)   , DIMENSION(:)        , ALLOCATABLE :: dl_value
      LOGICAL                                        :: ll_sea
      INTEGER(i4)                                    :: il_index

      ! loop indices
      INTEGER(i4) :: ji
      !---------------------------------------------------------------- 
 
      IF( td_bdy%l_use .AND. td_bdy%i_nseg == 1 )THEN

         il_index=td_bdy%t_seg(1)%i_index

         SELECT CASE(TRIM(td_bdy%c_card))
            CASE('north','south')

               ALLOCATE( dl_value(td_var%t_dim(1)%i_len) )
               dl_value(:)=td_var%d_value(:,il_index,1,1)

               IF( ANY(dl_value(:) /= td_var%d_fill) )THEN
                  
                  td_bdy%l_use=.TRUE.
                  td_bdy%i_nseg=0

                  ll_sea=.FALSE.
                  DO ji=1,td_var%t_dim(1)%i_len
                     IF( dl_value(ji)/= td_var%d_fill )THEN
                        IF( .NOT. ll_sea )THEN
                           td_bdy%i_nseg=td_bdy%i_nseg+1
                        ENDIF
                        ll_sea=.TRUE.
                     ELSE
                        ll_sea=.FALSE.
                     ENDIF
                  ENDDO

               ELSE
                  td_bdy%l_use=.FALSE.
                  td_bdy%i_nseg=0
               ENDIF

               DEALLOCATE( dl_value )

            CASE('east','west')

               ALLOCATE( dl_value(td_var%t_dim(2)%i_len) )
               dl_value(:)=td_var%d_value(il_index,:,1,1)

               IF( ANY(dl_value(:) /= td_var%d_fill) )THEN
                  
                  td_bdy%l_use=.TRUE.
                  td_bdy%i_nseg=0

                  ll_sea=.FALSE.
                  DO ji=1,td_var%t_dim(2)%i_len
                     IF( dl_value(ji)/= td_var%d_fill )THEN
                        IF( .NOT. ll_sea )THEN
                           td_bdy%i_nseg=td_bdy%i_nseg+1
                        ENDIF
                        ll_sea=.TRUE.
                     ELSE
                        ll_sea=.FALSE.
                     ENDIF
                  ENDDO

               ELSE
                  td_bdy%l_use=.FALSE.
                  td_bdy%i_nseg=0
               ENDIF

               DEALLOCATE( dl_value )

         END SELECT
      ENDIF
 
   END SUBROUTINE boundary__get_seg_number 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE boundary__get_seg_indices(td_bdy, td_var, &
         &                              id_index, id_width, id_first, id_last) 
   !------------------------------------------------------------------- 
   !> @brief This subroutine get segment indices for one boundary.
   ! 
   !> @details 
   ! 
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   ! 
   !> @param[inout] td_bdy boundary structure  
   !> @param[in] td_var    variable structure  
   !> @param[in] id_index  boundary orthogonal index  
   !> @param[in] id_width  bounary width 
   !> @param[in] id_first  boundary first indice
   !> @param[in] id_last   boundary last  indice
   !------------------------------------------------------------------- 

      IMPLICIT NONE 

      ! Argument
      TYPE(TBDY) , INTENT(INOUT) :: td_bdy
      TYPE(TVAR) , INTENT(IN   ) :: td_var
      INTEGER(i4), INTENT(IN   ) :: id_index
      INTEGER(i4), INTENT(IN   ) :: id_width
      INTEGER(i4), INTENT(IN   ) :: id_first
      INTEGER(i4), INTENT(IN   ) :: id_last

      ! local variable 
      INTEGER(i4)                                    :: il_nseg
      INTEGER(i4), DIMENSION(ip_ncard)               :: il_max
      INTEGER(i4), DIMENSION(ip_ncard)               :: il_min

      REAL(dp)   , DIMENSION(:)        , ALLOCATABLE :: dl_value

      LOGICAL                                        :: ll_sea
      LOGICAL                                        :: ll_first
      LOGICAL                                        :: ll_last

      TYPE(TSEG)                                     :: tl_seg

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !---------------------------------------------------------------- 
 
      SELECT CASE(TRIM(td_bdy%c_card))
         CASE('north')
            jk=jp_north
            
            ALLOCATE( dl_value(td_var%t_dim(1)%i_len) )
            dl_value(:)=td_var%d_value(:,id_index,1,1)

         CASE('south')
            jk=jp_south

            ALLOCATE( dl_value(td_var%t_dim(1)%i_len) )
            dl_value(:)=td_var%d_value(:,id_index,1,1)

         CASE('east ')
            jk=jp_east 

            ALLOCATE( dl_value(td_var%t_dim(2)%i_len) )
            dl_value(:)=td_var%d_value(id_index,:,1,1)

         CASE('west ')
            jk=jp_west 

            ALLOCATE( dl_value(td_var%t_dim(2)%i_len) )
            dl_value(:)=td_var%d_value(id_index,:,1,1)

      END SELECT

      il_max(jp_north)=td_var%t_dim(1)%i_len-ip_ghost
      il_max(jp_south)=td_var%t_dim(1)%i_len-ip_ghost
      il_max(jp_east )=td_var%t_dim(2)%i_len-ip_ghost
      il_max(jp_west )=td_var%t_dim(2)%i_len-ip_ghost

      il_min(jp_north)=1+ip_ghost
      il_min(jp_south)=1+ip_ghost
      il_min(jp_east )=1+ip_ghost
      il_min(jp_west )=1+ip_ghost
         
      ! special case for EW cyclic 
      IF( td_var%i_ew >= 0 )THEN
         il_min(jp_north)=1
         il_min(jp_south)=1

         il_max(jp_north)=td_var%t_dim(1)%i_len
         il_max(jp_south)=td_var%t_dim(1)%i_len
      ENDIF
      
      il_nseg=td_bdy%i_nseg
      ! remove all segment from boundary
      CALL boundary__del_seg(td_bdy)

      ll_first=.FALSE.
      ll_last =.FALSE.
      DO jl=1,il_nseg

         ! init
         tl_seg=seg__init(id_index,id_width,id_first,id_last)

         IF( .NOT. (ll_first .AND. ll_last) )THEN
            ! first loop
            tl_seg%i_first=MAX(id_first,il_min(jk))
            tl_seg%i_last =MIN(id_last ,il_max(jk))
         ELSE
            ! load new min and max
            tl_seg%i_first=MAX(td_bdy%t_seg(jl-1)%i_last,il_min(jk))
            tl_seg%i_last =MIN(id_last                  ,il_max(jk))
         ENDIF

         ll_first=.FALSE.
         ll_last =.FALSE.
         ll_sea  =.FALSE.

         DO ji=tl_seg%i_first,tl_seg%i_last

            IF( ll_first .AND. ll_last )THEN
               ! first and last point already loaded
               ! look for next segment
               EXIT
            ENDIF

            IF( dl_value(ji)/= td_var%d_fill )THEN
               IF( .NOT. ll_sea )THEN
                  tl_seg%i_first=MAX(tl_seg%i_first,ji-1)
                  ll_first=.true.
               ENDIF
               ll_sea=.TRUE.
            ELSE
               IF( ll_sea )THEN
                  tl_seg%i_last=ji
                  ll_last=.TRUE.
               ENDIF
               ll_sea=.FALSE.
            ENDIF
            
         ENDDO

         CALL boundary__add_seg(td_bdy,tl_seg)

         ! clean
         CALL seg__clean(tl_seg)
         
      ENDDO

      DEALLOCATE(dl_value)
      
   END SUBROUTINE boundary__get_seg_indices 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE boundary_check_corner(td_bdy, td_var)
   !------------------------------------------------------------------- 
   !> @brief This subroutine check if there is boundary at corner, and 
   !> adjust boundary indices if necessary. 
   ! 
   !> @details 
   !> If there is a north west corner, first indices of north boundary
   !> should be the same as the west boundary indices. 
   !> And the last indices of the west boundary should be the same as
   !> the north indices.
   !> More over the width of west and north boundary should be the same.
   ! 
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   ! 
   !> @param[inout] td_bdy boundary structure
   !> @param[in] td_var    variable structure
   !------------------------------------------------------------------- 

      IMPLICIT NONE 

      ! Argument
      TYPE(TBDY) , DIMENSION(ip_ncard), INTENT(INOUT) :: td_bdy
      TYPE(TVAR)                      , INTENT(IN   ) :: td_var

      ! local variable 
      TYPE(TSEG)  :: tl_north
      TYPE(TSEG)  :: tl_south
      TYPE(TSEG)  :: tl_east 
      TYPE(TSEG)  :: tl_west

      INTEGER(i4) :: il_width

      ! loop indices
      !---------------------------------------------------------------- 
 
      IF( .NOT. ASSOCIATED(td_var%d_value) )THEN
         CALL logger_error("BOUNDARY CHEKC CORNER: no value associated "//&
         &              "to variable "//TRIM(td_var%c_name))
      ENDIF

      ! check north west corner
      IF( td_bdy(jp_north)%l_use .AND. td_bdy(jp_west)%l_use )THEN
         tl_west =seg__copy(td_bdy(jp_west )%t_seg(td_bdy(jp_west)%i_nseg))
         tl_north=seg__copy(td_bdy(jp_north)%t_seg(1))

         IF( tl_west%i_last  >= tl_north%i_index .AND. &
         &   tl_west%i_index >= tl_north%i_first ) THEN
            CALL logger_debug("BOUNDARY CHEKC CORNER: there is "//&
            &                 "a north west corner")

            tl_west%i_last   = tl_north%i_index
            tl_north%i_first = tl_west%i_index

            IF( tl_west%i_width /= tl_north%i_width )THEN
               CALL logger_error("BOUNDARY CHEKC CORNER: discordant "//&
               &  " width between north and west boundary ")

               il_width=MIN(tl_west%i_width,tl_north%i_width)
               
               tl_west%i_width =il_width
               tl_north%i_width=il_width

            ENDIF

            td_bdy(jp_west )%t_seg(td_bdy(jp_west)%i_nseg)=seg__copy(tl_west)
            td_bdy(jp_north)%t_seg(1)                     =seg__copy(tl_north)

         ELSE

            IF( td_var%d_value(tl_north%i_first,tl_north%i_index,1,1) /= &
            &   td_var%d_fill )THEN
               CALL logger_error("BOUNDARY CHEKC CORNER: wrong "//&
               &              "north boundary first indice ")
            ENDIF

            IF( td_var%d_value(tl_west%i_index,tl_west%i_last,1,1) /= &
            &   td_var%d_fill )THEN
               CALL logger_error("BOUNDARY CHEKC CORNER: wrong "//&
               &              "west boundary last indice")
            ENDIF
         ENDIF
      ENDIF

      ! check north east corner
      IF( td_bdy(jp_north)%l_use .AND. td_bdy(jp_east)%l_use )THEN
         tl_east =seg__copy(td_bdy(jp_east )%t_seg(td_bdy(jp_east )%i_nseg))
         tl_north=seg__copy(td_bdy(jp_north)%t_seg(td_bdy(jp_north)%i_nseg))

         IF( tl_east%i_last  >= tl_north%i_index .AND. &
         &   tl_east%i_index <= tl_north%i_last ) THEN
            CALL logger_debug("BOUNDARY CHEKC CORNER: there is "//&
            &              "a north east corner")

            tl_east%i_last  = tl_north%i_index
            tl_north%i_last = tl_east%i_index

            IF( tl_east%i_width /= tl_north%i_width )THEN
               CALL logger_error("BOUNDARY CHEKC CORNER: discordant "//&
               &  " width between north and east boundary ")

               il_width=MIN(tl_east%i_width,tl_north%i_width)
               
               tl_east%i_width =il_width
               tl_north%i_width=il_width

            ENDIF

            td_bdy(jp_east )%t_seg(td_bdy(jp_east )%i_nseg)=seg__copy(tl_east)
            td_bdy(jp_north)%t_seg(td_bdy(jp_north)%i_nseg)=seg__copy(tl_north)
         ELSE

            IF( td_var%d_value(tl_north%i_last,tl_north%i_index,1,1) /= &
            &   td_var%d_fill )THEN
               CALL logger_error("BOUNDARY CHEKC CORNER: wrong "//&
               &              "north boundary last indice ")
            ENDIF

            IF( td_var%d_value(tl_east%i_index,tl_east%i_last,1,1) /= &
            &   td_var%d_fill )THEN
               CALL logger_error("BOUNDARY CHEKC CORNER: wrong "//&
               &              "east boundary last indice")
            ENDIF
         ENDIF
      ENDIF

      ! check south east corner
      IF( td_bdy(jp_south)%l_use .AND. td_bdy(jp_east)%l_use )THEN
         tl_east =seg__copy(td_bdy(jp_east )%t_seg(1))
         tl_south=seg__copy(td_bdy(jp_south)%t_seg(td_bdy(jp_south)%i_nseg))

         IF( tl_east%i_first <= tl_south%i_index .AND. &
         &   tl_east%i_index <= tl_south%i_last ) THEN
            CALL logger_debug("BOUNDARY CHEKC CORNER: there is "//&
            &              "a south east corner")

            tl_east%i_first = tl_south%i_index
            tl_south%i_last = tl_east%i_index

            IF( tl_east%i_width /= tl_south%i_width )THEN
               CALL logger_error("BOUNDARY CHEKC CORNER: discordant "//&
               &  " width between south and east boundary ")

               il_width=MIN(tl_east%i_width,tl_south%i_width)
               
               tl_east%i_width =il_width
               tl_south%i_width=il_width

            ENDIF

            td_bdy(jp_east )%t_seg(1)                      =seg__copy(tl_east)
            td_bdy(jp_south)%t_seg(td_bdy(jp_south)%i_nseg)=seg__copy(tl_south)
         ELSE

            IF( td_var%d_value(tl_south%i_last,tl_south%i_index,1,1) /= &
            &   td_var%d_fill )THEN
               CALL logger_error("BOUNDARY CHEKC CORNER: wrong "//&
               &              "south boundary last indice ")
            ENDIF

            IF( td_var%d_value(tl_east%i_index,tl_east%i_first,1,1) /= &
            &   td_var%d_fill )THEN
               CALL logger_error("BOUNDARY CHEKC CORNER: wrong "//&
               &              "east boundary first indice")
            ENDIF
         ENDIF
      ENDIF

      ! check south west corner
      IF( td_bdy(jp_south)%l_use .AND. td_bdy(jp_west)%l_use )THEN
         tl_west =seg__copy(td_bdy(jp_west )%t_seg(1))
         tl_south=seg__copy(td_bdy(jp_south)%t_seg(1))

         IF( tl_west%i_first <= tl_south%i_index .AND. &
         &   tl_west%i_index >= tl_south%i_first ) THEN
            CALL logger_debug("BOUNDARY CHEKC CORNER: there is "//&
            &              "a south west corner")

            tl_west%i_first = tl_south%i_index
            tl_south%i_first= tl_west%i_index

            IF( tl_west%i_width /= tl_south%i_width )THEN
               CALL logger_error("BOUNDARY CHEKC CORNER: discordant "//&
               &  " width between south and west boundary ")

               il_width=MIN(tl_west%i_width,tl_south%i_width)
               
               tl_west%i_width =il_width
               tl_south%i_width=il_width

            ENDIF

            td_bdy(jp_west )%t_seg(1) = seg__copy(tl_west)
            td_bdy(jp_south)%t_seg(1) = seg__copy(tl_south)
         ELSE

            IF( td_var%d_value(tl_south%i_first,tl_south%i_index,1,1) /= &
            &   td_var%d_fill )THEN
               CALL logger_error("BOUNDARY CHEKC CORNER: wrong "//&
               &              "south boundary first indice ")
            ENDIF

            IF( td_var%d_value(tl_west%i_index,tl_west%i_first,1,1) /= &
            &   td_var%d_fill )THEN
               CALL logger_error("BOUNDARY CHEKC CORNER: wrong "//&
               &              "west boundary first indice")
            ENDIF
         ENDIF
      ENDIF

      ! clean
      CALL seg__clean(tl_north)
      CALL seg__clean(tl_south)
      CALL seg__clean(tl_east )
      CALL seg__clean(tl_west )

   END SUBROUTINE boundary_check_corner 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE boundary_check(td_bdy, td_var) 
   !------------------------------------------------------------------- 
   !> @brief This subroutine check boundary.
   ! 
   !> @details 
   !> It checks that first and last indices as well as orthogonal index are
   !> inside domain, and check corner (see boundary_check_corner).
   ! 
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   !> @date June, 2016
   !> - Bug fix: take into account that boundaries are compute on T point,
   !>   but expressed on U,V point
   !>
   !> @param[inout] td_bdy boundary structure 
   !> @param[in] td_var    variable structure 
   !------------------------------------------------------------------- 

      IMPLICIT NONE 

      ! Argument
      TYPE(TBDY) , DIMENSION(ip_ncard), INTENT(INOUT) :: td_bdy
      TYPE(TVAR)                      , INTENT(IN   ) :: td_var

      ! local variable 
      INTEGER(i4)      , DIMENSION(ip_ncard) :: il_max
      INTEGER(i4)      , DIMENSION(ip_ncard) :: il_maxindex

      ! loop indices 
      INTEGER(i4) :: jk
      !---------------------------------------------------------------- 
 
      il_max(jp_north)=td_var%t_dim(1)%i_len
      il_max(jp_south)=td_var%t_dim(1)%i_len
      il_max(jp_east )=td_var%t_dim(2)%i_len
      il_max(jp_west )=td_var%t_dim(2)%i_len
 
      ! index expressed on U,V point, move on T point.
      il_maxindex(jp_north)=td_var%t_dim(2)%i_len-ip_ghost+1
      il_maxindex(jp_south)=td_var%t_dim(2)%i_len-ip_ghost
      il_maxindex(jp_east )=td_var%t_dim(1)%i_len-ip_ghost+1
      il_maxindex(jp_west )=td_var%t_dim(1)%i_len-ip_ghost

      DO jk=1,ip_ncard
         IF( td_bdy(jk)%l_use )THEN
            IF( .NOT. ASSOCIATED(td_bdy(jk)%t_seg) )THEN
               CALL logger_error("BOUNDARY CHECK: no segment associted "//&
               &                 "to "//TRIM(td_bdy(jk)%c_card)//" boundary")
            ELSE
               ! check indices
               IF( ANY(td_bdy(jk)%t_seg(:)%i_first < 1         ) .OR. &
               &   ANY(td_bdy(jk)%t_seg(:)%i_first > il_max(jk)) .OR. &
               &   ANY(td_bdy(jk)%t_seg(:)%i_last  < 1         ) .OR. &
               &   ANY(td_bdy(jk)%t_seg(:)%i_last  > il_max(jk)) .OR. &
               &   ANY(td_bdy(jk)%t_seg(:)%i_first > td_bdy(jk)%t_seg(:)%i_last)&
               & )THEN
                  CALL logger_error("BOUNDARY CHECK: invalid segment "//&
                  &              "first and/or last indice for "//&
                  &              TRIM(td_bdy(jk)%c_card)//&
                  &              " boundary. check namelist")
               ENDIF

               IF( ANY(td_bdy(jk)%t_seg(:)%i_index < 1         ) .OR. &
               &   ANY(td_bdy(jk)%t_seg(:)%i_index > il_maxindex(jk)) &
               & )THEN
                  CALL logger_error("BOUNDARY CHECK: invalid index "//&
                  &              "for "//TRIM(td_bdy(jk)%c_card)//&
                  &              " boundary. check namelist")
               ENDIF
            ENDIF
         ENDIF
      ENDDO
 
      CALL boundary_check_corner(td_bdy, td_var)

   END SUBROUTINE boundary_check
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE boundary_swap(td_var, td_bdy)
   !-------------------------------------------------------------------
   !> @brief This subroutine swap array for east and north boundary.
   !
   !> @detail
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !
   !> @param[inout] td_var variable strucutre
   !> @param[in   ] td_bdy boundary strucutre
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), INTENT(INOUT) :: td_var
      TYPE(TBDY), INTENT(IN   ) :: td_bdy

      ! local variable
      REAL(dp), DIMENSION(:,:,:,:), ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      IF( .NOT. ASSOCIATED(td_var%d_value) )THEN
         CALL logger_error("BOUNDARY SWAP: no array of value "//&
         &  "associted to variable "//TRIM(td_var%c_name) )
      ELSE      

         SELECT CASE(TRIM(td_bdy%c_card))
         CASE('north')
            ALLOCATE( dl_value(td_var%t_dim(1)%i_len, &
            &                  td_var%t_dim(2)%i_len, &
            &                  td_var%t_dim(3)%i_len, &
            &                  td_var%t_dim(4)%i_len) )

            dl_value(:,:,:,:)=td_var%d_value(:,:,:,:)

            DO jj=1, td_var%t_dim(2)%i_len
               td_var%d_value(:,jj,:,:) = &
               &  dl_value(:,td_var%t_dim(2)%i_len-jj+1,:,:)
            ENDDO

            DEALLOCATE( dl_value )         
         CASE('east')
            ALLOCATE( dl_value(td_var%t_dim(1)%i_len, &
            &                  td_var%t_dim(2)%i_len, &
            &                  td_var%t_dim(3)%i_len, &
            &                  td_var%t_dim(4)%i_len) )

            dl_value(:,:,:,:)=td_var%d_value(:,:,:,:)

            DO ji=1, td_var%t_dim(1)%i_len
               td_var%d_value(ji,:,:,:) = &
               &  dl_value(td_var%t_dim(1)%i_len-ji+1,:,:,:)
            ENDDO

            DEALLOCATE( dl_value )
         CASE DEFAULT
         ! nothing to be done
         END SELECT

      ENDIF
   END SUBROUTINE boundary_swap
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE boundary__print_unit(td_bdy) 
   !------------------------------------------------------------------- 
   !> @brief This subroutine print information about one boundary. 
   ! 
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   ! 
   !> @param[in] td_bdy boundary structure 
   !------------------------------------------------------------------- 

      IMPLICIT NONE 

      ! Argument
      TYPE(TBDY), INTENT(IN) :: td_bdy

      ! local variable 
      ! loop indices 
      INTEGER(i4) :: ji
      !---------------------------------------------------------------- 

      WRITE(*,'(a,/1x,a,/1x,a)') "Boundary "//TRIM(td_bdy%c_card), &
      &  " use  "//TRIM(fct_str(td_bdy%l_use)), &
      &  " nseg "//TRIM(fct_str(td_bdy%i_nseg))
      DO ji=1,td_bdy%i_nseg
         WRITE(*,'(4(/1x,a))') &
         &  " index "//TRIM(fct_str(td_bdy%t_seg(ji)%i_index)), &
         &  " width "//TRIM(fct_str(td_bdy%t_seg(ji)%i_width)), &
         &  " first "//TRIM(fct_str(td_bdy%t_seg(ji)%i_first)), &
         &  " last  "//TRIM(fct_str(td_bdy%t_seg(ji)%i_last))
      ENDDO
 
   END SUBROUTINE boundary__print_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE boundary__print_arr(td_bdy) 
   !------------------------------------------------------------------- 
   !> @brief This subroutine print information about a array of boundary 
   ! 
   !> @details 
   ! 
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   ! 
   !> @param[in] td_bdy boundary structure 
   !------------------------------------------------------------------- 

      IMPLICIT NONE 

      ! Argument
      TYPE(TBDY), DIMENSION(:), INTENT(IN) :: td_bdy

      ! local variable 
      ! loop indices 
      INTEGER(i4) :: ji
      !---------------------------------------------------------------- 

      DO ji=1,SIZE(td_bdy(:))
         CALL boundary_print(td_bdy(ji))
      ENDDO
 
   END SUBROUTINE boundary__print_arr
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION seg__copy_unit(td_seg) &
         & RESULT (tf_seg)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine copy segment structure in another one.
   !>
   !> @warning do not use on the output of a function who create or read a
   !> structure (ex: tl_seg=seg__copy(seg__init()) is forbidden).
   !> This will create memory leaks.
   !> @warning to avoid infinite loop, do not use any function inside 
   !> this subroutine
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date November, 2014
   !> - use function instead of overload assignment operator 
   !> (to avoid memory leak)
   !
   !> @param[in] td_seg   segment structure
   !> @return copy of input segment structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TSEG), INTENT(IN)  :: td_seg

      ! function
      TYPE(TSEG)              :: tf_seg

      ! local variable
      ! loop indices
      !----------------------------------------------------------------

      ! copy segment index, width, ..
      tf_seg%i_index    = td_seg%i_index
      tf_seg%i_width    = td_seg%i_width
      tf_seg%i_first    = td_seg%i_first
      tf_seg%i_last     = td_seg%i_last 

   END FUNCTION seg__copy_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION seg__copy_arr(td_seg) &
         & RESULT (tf_seg)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine copy segment structure in another one.
   !>
   !> @warning do not use on the output of a function who create or read a
   !> structure (ex: tl_seg=seg__copy(seg__init()) is forbidden).
   !> This will create memory leaks.   
   !> @warning to avoid infinite loop, do not use any function inside 
   !> this subroutine
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date November, 2014
   !> - use function instead of overload assignment operator 
   !> (to avoid memory leak)
   !
   !> @param[in] td_seg   segment structure
   !> @return copy of input array of segment structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TSEG), DIMENSION(:), INTENT(IN)   :: td_seg

      ! function
      TYPE(TSEG), DIMENSION(SIZE(td_seg(:))) :: tf_seg

      ! local variable
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      DO ji=1,SIZE(td_seg(:))
         tf_seg(ji)=seg__copy(td_seg(ji))
      ENDDO

   END FUNCTION seg__copy_arr
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION seg__init(id_index, id_width, id_first, id_last) &
         &  RESULT(tf_seg)
   !------------------------------------------------------------------- 
   !> @brief This function  initialise segment structure.
   ! 
   !> @details 
   !> It simply add orthogonal index, and optionnaly width, first
   !> and last indices of the segment. 
   ! 
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   ! 
   !> @param[in] id_index  orthogonal index
   !> @param[in] id_width  width of the segment 
   !> @param[in] id_first  first indices 
   !> @param[in] id_last   last  indices
   !> @return segment structure
   !------------------------------------------------------------------- 

      IMPLICIT NONE 

      ! Argument
      INTEGER(i4), INTENT(IN) :: id_index
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_width
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_first
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_last 

      ! function 
      TYPE(TSEG)              :: tf_seg

      ! local variable 
      
      ! loop indices 
      !---------------------------------------------------------------- 

      tf_seg%i_index=id_index

      IF( PRESENT(id_width) ) tf_seg%i_width=id_width
      IF( PRESENT(id_first) ) tf_seg%i_first=id_first
      IF( PRESENT(id_last ) ) tf_seg%i_last =id_last

   END FUNCTION seg__init 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE seg__clean_unit(td_seg) 
   !------------------------------------------------------------------- 
   !> @brief This subroutine clean segment structure. 
   ! 
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   ! 
   !> @param[inout] td_seg segment structure
   !------------------------------------------------------------------- 

      IMPLICIT NONE 

      ! Argument       
      TYPE(TSEG), INTENT(INOUT) :: td_seg

      ! local variable 
      TYPE(TSEG) :: tl_seg
      ! loop indices 
      !---------------------------------------------------------------- 

      td_seg=seg__copy(tl_seg)
 
   END SUBROUTINE seg__clean_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE seg__clean_arr(td_seg) 
   !------------------------------------------------------------------- 
   !> @brief This subroutine clean segment structure. 
   ! 
   !> @author J.Paul 
   !> @date November, 2013 - Initial Version 
   ! 
   !> @param[inout] td_seg array of segment structure
   !------------------------------------------------------------------- 

      IMPLICIT NONE 

      ! Argument       
      TYPE(TSEG), DIMENSION(:), INTENT(INOUT) :: td_seg

      ! local variable 
      ! loop indices 
      INTEGER(i4) :: ji
      !---------------------------------------------------------------- 

      DO ji=SIZE(td_seg(:)),1,-1
         CALL seg__clean(td_seg(ji))
      ENDDO
 
   END SUBROUTINE seg__clean_arr 
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE boundary
