!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief
!> This module manage domain computation.
!>
!> @details
!>    define type TDOM:<br/>
!> @code
!>    TYPE(TDOM) :: tl_dom
!> @endcode
!>
!>    to initialize domain structure:<br/>
!> @code
!>    tl_dom=dom_init(td_mpp, [id_imin,] [id_imax,] [id_jmin,] [id_jmax],[cd_card])
!> @endcode
!>       - td_mpp  is mpp structure of an opened file.
!>       - id_imin is i-direction sub-domain lower left  point indice
!>       - id_imax is i-direction sub-domain upper right point indice 
!>       - id_jmin is j-direction sub-domain lower left  point indice
!>       - id_jmax is j-direction sub-domain upper right point indice
!>       - cd_card is the cardinal name (for boundary case) 
!> 
!>    to get global domain dimension:<br/>
!>    - tl_dom\%t_dim0
!>
!>    to get NEMO periodicity index of global domain:<br/>
!>    - tl_dom\%i_perio0
!>
!>    to get NEMO pivot point index F(0),T(1):<br/>
!>    - tl_dom\%i_pivot
!>
!>    to get East-West overlap of global domain:<br/>
!>    - tl_dom\%i_ew0
!>
!>    to get selected sub domain dimension:<br/>
!>    - tl_dom\%t_dim
!>
!>    to get NEMO periodicity index of sub domain:<br/>
!>    - tl_dom\%i_perio
!>
!>    to get East-West overlap of sub domain:<br/>
!>    - tl_dom\%i_ew
!>
!>    to get i-direction sub-domain lower left  point indice:<br/>
!>    - tl_dom\%i_imin
!>
!>    to get i-direction sub-domain upper right point indice:<br/>
!>    - tl_dom\%i_imax
!>
!>    to get j-direction sub-domain lower left  point indice:<br/>
!>    - tl_dom\%i_jmin
!>
!>    to get j-direction sub-domain upper right point indice:<br/>
!>    - tl_dom\%i_jmax
!>
!>    to get size of i-direction extra band:<br/>
!>    - tl_dom\%i_iextra
!>
!>    to get size of j-direction extra band:<br/>
!>    - tl_dom\%i_jextra
!>
!>    to get i-direction ghost cell number:<br/>
!>    - tl_dom\%i_ighost
!>
!>    to get j-direction ghost cell number:<br/>
!>    - tl_dom\%i_jghost
!>
!>    to get boundary index:<br/>
!>    - tl_dom\%i_bdy
!>       - 0 = no boundary
!>       - 1 = north
!>       - 2 = south 
!>       - 3 = east 
!>       - 4 = west
!>
!>    to clean domain structure:<br/>
!> @code
!>    CALL dom_clean(td_dom)
!> @endcode
!>       - td_dom is domain structure
!>
!>    to print information about domain structure:<br/>
!> @code
!>    CALL dom_print(td_dom)
!> @endcode
!>
!>    to get East-West overlap (if any):<br/>
!> @code
!>    il_ew=dom_get_ew_overlap(td_lon)
!> @endcode
!>       - td_lon : longitude variable structure
!>
!>    to add extra bands to coarse grid domain (for interpolation):<br/>
!> @code
!>    CALL dom_add_extra( td_dom, id_iext, id_jext ) 
!> @endcode
!>       - td_dom is domain structure
!>       - id_iext is i-direction size of extra bands
!>       - id_jext is j-direction size of extra bands
!> 
!>    to remove extra bands from fine grid (after interpolation):<br/>
!> @code
!>    CALL dom_del_extra( td_var, td_dom, id_rho )
!> @endcode
!>       - td_var is variable structure to be changed
!>       - td_dom is domain structure
!>       - id_rho is a array of refinement factor following i- and j-direction
!>    
!>    to reset coarse grid domain witouht extra bands:<br/>
!> @code
!>    CALL dom_clean_extra( td_dom )
!> @endcode
!>
!> @author
!> J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date September, 2014
!> - add header
!> - use zero indice to defined cyclic or global domain
!> @date October, 2014
!> - use mpp file structure instead of file
!> 
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE dom

   USE kind                            ! F90 kind parameter
   USE global                          ! global parameter
   USE fct                             ! basic useful function
   USE logger                          ! log file manager
   USE dim                             ! dimension manager
   USE var                             ! variable manager
   USE mpp                             ! mpp file manager

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! type and variable
   PUBLIC :: TDOM     !< domain structure

   PRIVATE :: im_minext !< default minumum number of extraband

   ! function and subroutine
   PUBLIC :: dom_copy            !< copy domain structure
   PUBLIc :: dom_clean           !< clean domain structure
   PUBLIC :: dom_init            !< initialise domain structure
   PUBLIC :: dom_print           !< print information about domain
   PUBLIC :: dom_add_extra       !< add useful extra bands to coarse grid for interpolation
   PUBLIC :: dom_clean_extra     !< reset domain without extra bands
   PUBLIC :: dom_del_extra       !< remove extra point from fine grid after interpolation
    
   PRIVATE :: dom__init_mpp                 ! initialise domain structure, given mpp file structure
   PRIVATE :: dom__define                   ! define sub domain indices
                                            ! define sub domain indices for input domain with
   PRIVATE :: dom__define_cyclic_north_fold ! - cyclic east-west boundary and north fold boundary condition.
   PRIVATE :: dom__define_north_fold        ! - north fold boundary condition.
   PRIVATE :: dom__define_symmetric         ! - symmetric boundary condition across the equator.
   PRIVATE :: dom__define_cyclic            ! - cyclic east-west boundary.
   PRIVATE :: dom__define_closed            ! - cyclic east-west boundary.
                                            ! compute size of sub domain
   PRIVATE :: dom__size_no_pole             ! - without north fold condition
   PRIVATE :: dom__size_no_pole_overlap     ! - without north fold condition, and which overlap east-west boundary
   PRIVATE :: dom__size_no_pole_no_overlap  ! - without north fold condition, and which do not overlap east-west boundary
   PRIVATE :: dom__size_pole                ! - with north fold condition
   PRIVATE :: dom__size_pole_overlap        ! - with north fold condition, and which overlap east-west boundary
   PRIVATE :: dom__size_pole_no_overlap     ! - with north fold condition, and which do not overlap east-west boundary
                                            ! compute size of
   PRIVATE :: dom__size_global              ! - global domain
   PRIVATE :: dom__size_semi_global         ! - semi global domain 
   PRIVATE :: dom__copy_unit                ! copy attribute structure

   TYPE TDOM !< domain structure
      TYPE(TDIM), DIMENSION(ip_maxdim)  :: t_dim0  !< global domain dimension
      TYPE(TDIM), DIMENSION(ip_maxdim)  :: t_dim   !< sub domain dimension
      INTEGER(i4) :: i_perio0                      !< NEMO periodicity index of global domain
      INTEGER(i4) :: i_ew0                         !< East-West overlap of global domain
      INTEGER(i4) :: i_perio                       !< NEMO periodicity index of sub domain
      INTEGER(i4) :: i_pivot                       !< NEMO pivot point index F(0),T(1)
      INTEGER(i4) :: i_imin = 0                    !< i-direction sub-domain lower left  point indice
      INTEGER(i4) :: i_imax = 0                    !< i-direction sub-domain upper right point indice
      INTEGER(i4) :: i_jmin = 0                    !< j-direction sub-domain lower left  point indice
      INTEGER(i4) :: i_jmax = 0                    !< j-direction sub-domain upper right point indice

      INTEGER(i4) :: i_bdy = 0                     !< boundary index : 0 = no boundary
                                                   !<                  1 = north
                                                   !<                  2 = south 
                                                   !<                  3 = east 
                                                   !<                  4 = west 
      INTEGER(i4), DIMENSION(2,2) :: i_ghost0 = 0   !< array of ghost cell factor of global domain
      INTEGER(i4), DIMENSION(2,2) :: i_ghost  = 0   !< array of ghost cell factor of sub domain

      INTEGER(i4), DIMENSION(2) :: i_iextra = 0    !< i-direction extra point
      INTEGER(i4), DIMENSION(2) :: i_jextra = 0    !< j-direction extra point

   END TYPE TDOM

   INTEGER(i4), PARAMETER :: im_minext  = 2  !< default minumum number of extraband

   INTERFACE dom_init
      MODULE PROCEDURE dom__init_file
      MODULE PROCEDURE dom__init_mpp
   END INTERFACE dom_init

   INTERFACE dom_copy
      MODULE PROCEDURE dom__copy_unit  ! copy attribute structure
   END INTERFACE

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dom__copy_unit(td_dom) &
         & RESULT (tf_dom)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine copy an domain structure in another one
   !> @details 
   !> dummy function to get the same use for all structure
   !>
   !> @warning do not use on the output of a function who create or read an
   !> structure (ex: tl_dom=dom_copy(dom_init()) is forbidden).
   !> This will create memory leaks.
   !> @warning to avoid infinite loop, do not use any function inside 
   !> this subroutine
   !>
   !> @author J.Paul
   !> @date November, 2014 - Initial Version
   !>
   !> @param[in] td_dom   domain structure
   !> @return copy of input domain structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM), INTENT(IN)  :: td_dom

      ! function
      TYPE(TDOM)              :: tf_dom

      ! local variable
      !----------------------------------------------------------------

      tf_dom=td_dom
      
   END FUNCTION dom__copy_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom_print(td_dom)
   !-------------------------------------------------------------------
   !> @brief This subroutine print some information about domain strucutre.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_dom dom structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TDOM), INTENT(IN) :: td_dom

      ! local argument
      CHARACTER(LEN=lc) :: cl_pivot
      !----------------------------------------------------------------
      SELECT CASE(td_dom%i_pivot)
         CASE(0)
            cl_pivot='F-point'
         CASE(1)
            cl_pivot='T-point'
         CASE DEFAULT
            cl_pivot='unknown'
      END SELECT

      WRITE(*,'((a,4(i0,1x)),(/a,i2,a,a),2(/a,2(i0,1x)),(/a,4(i0,1x)),(/a,i2/),&
      &          4(/a,i0),4(/a,2(i0,1x)))') &
      &  " global domain size ",td_dom%t_dim0(:)%i_len, &
      &  " domain periodicity ",td_dom%i_perio0,", pivot: ",TRIM(cl_pivot),   &
      &  " i-direction ghost cell factor of global domain  ",td_dom%i_ghost0(jp_I,:), &
      &  " j-direction ghost cell factor of global domain  ",td_dom%i_ghost0(jp_J,:), &
      &  " sub-domain size : ",td_dom%t_dim(:)%i_len,                         &
      &  " sub domain periodicity ",td_dom%i_perio,                           &
      &  " i-direction sub-domain lower left  point indice ",td_dom%i_imin,   &
      &  " i-direction sub-domain upper right point indice ",td_dom%i_imax,   &
      &  " j-direction sub-domain lower left  point indice ",td_dom%i_jmin,   &
      &  " j-direction sub-domain upper right point indice ",td_dom%i_jmax,   &
      &  " i-direction ghost cell factor                   ",td_dom%i_ghost(jp_I,:), &
      &  " j-direction ghost cell factor                   ",td_dom%i_ghost(jp_J,:), &
      &  " i-direction extra point for interpolation       ",td_dom%i_iextra(:), &
      &  " j-direction extra point for interpolation       ",td_dom%i_jextra(:)

   END SUBROUTINE dom_print
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dom__init_mpp(td_mpp, id_imin, id_imax, id_jmin, id_jmax, cd_card) &
         & RESULT (tf_dom)
   !-------------------------------------------------------------------
   !> @brief 
   !> This function intialise domain structure, given open file structure,
   !> and sub domain indices. 
   !> @details
   !> sub domain indices are computed, taking into account coarse grid
   !> periodicity, pivot point, and East-West overlap.
   !>
   !> @author J.Paul
   !> @date June, 2013 - Initial Version
   !> @date September, 2014
   !> - add boundary index
   !> - add ghost cell factor
   !> @date October, 2014
   !> - work on mpp file structure instead of file structure
   !>
   !> @param[in] td_mpp    mpp structure
   !> @param[in] id_perio  grid periodicity
   !> @param[in] id_imin   i-direction sub-domain lower left  point indice
   !> @param[in] id_imax   i-direction sub-domain upper right point indice
   !> @param[in] id_jmin   j-direction sub-domain lower left  point indice
   !> @param[in] id_jmax   j-direction sub-domain upper right point indice
   !> @param[in] cd_card   name of cardinal (for boundary)
   !> @return domain structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP)      , INTENT(IN) :: td_mpp 

      INTEGER(i4)     , INTENT(IN), OPTIONAL :: id_imin
      INTEGER(i4)     , INTENT(IN), OPTIONAL :: id_imax
      INTEGER(i4)     , INTENT(IN), OPTIONAL :: id_jmin
      INTEGER(i4)     , INTENT(IN), OPTIONAL :: id_jmax

      CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: cd_card

      ! function
      TYPE(TDOM)                   :: tf_dom

      !local variable
      !----------------------------------------------------------------

      ! clean domain structure
      CALL dom_clean(tf_dom)

      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN

         CALL logger_error( &
         &  " DOM INIT: no processor file associated to mpp "//&
         &  TRIM(td_mpp%c_name))

      ELSE
         ! global domain define by file

         ! look for boundary index
         IF( PRESENT(cd_card) )THEN
            SELECT CASE(TRIM(cd_card))
               CASE('north')
                  tf_dom%i_bdy=jp_north
               CASE('south')
                  tf_dom%i_bdy=jp_south
               CASE('east')
                  tf_dom%i_bdy=jp_east
               CASE('west')
                  tf_dom%i_bdy=jp_west
               CASE DEFAULT
                  ! no boundary
                  tf_dom%i_bdy=0
            END SELECT
         ELSE
            ! no boundary
            tf_dom%i_bdy=0
         ENDIF

         ! use global dimension define by mpp file
         tf_dom%t_dim0(:) = dim_copy(td_mpp%t_dim(:))

         IF( td_mpp%i_perio < 0 .OR. td_mpp%i_perio > 6 )THEN
            CALL logger_error("DOM INIT: invalid grid periodicity ("//&
            &  TRIM(fct_str(td_mpp%i_perio))//&
            &  ") you should use grid_get_perio to compute it")
         ELSE
            tf_dom%i_perio0=td_mpp%i_perio
         ENDIF

         ! global domain pivot point
         SELECT CASE(tf_dom%i_perio0)
            CASE(3,4)
               tf_dom%i_pivot = 0
            CASE(5,6)
               tf_dom%i_pivot = 1
            CASE DEFAULT
               tf_dom%i_pivot = 0
         END SELECT

         ! add ghost cell factor of global domain
         tf_dom%i_ghost0(:,:)=0
         SELECT CASE(tf_dom%i_perio0)
            CASE(0)
               tf_dom%i_ghost0(:,:)=1
            CASE(1)
               tf_dom%i_ghost0(jp_J,:)=1
            CASE(2)
               tf_dom%i_ghost0(jp_I,:)=1
               tf_dom%i_ghost0(jp_J,2)=1
            CASE(3,5)
               tf_dom%i_ghost0(jp_I,:)=1
               tf_dom%i_ghost0(jp_J,1)=1
            CASE(4,6)
               tf_dom%i_ghost0(jp_J,1)=1
         END SELECT

         ! look for EW overlap
         tf_dom%i_ew0=td_mpp%i_ew

         ! initialise domain as global
         tf_dom%i_imin = 1 
         tf_dom%i_imax = tf_dom%t_dim0(1)%i_len

         tf_dom%i_jmin = 1 
         tf_dom%i_jmax = tf_dom%t_dim0(2)%i_len

         ! sub domain dimension
         tf_dom%t_dim(:) = dim_copy(td_mpp%t_dim(:))

         ! define sub domain indices 
         CALL dom__define(tf_dom, id_imin, id_imax, id_jmin, id_jmax)

      ENDIF

   END FUNCTION dom__init_mpp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION dom__init_file(td_file, id_imin, id_imax, id_jmin, id_jmax, cd_card) &
         & RESULT (tf_dom)
   !-------------------------------------------------------------------
   !> @brief 
   !> This function intialise domain structure, given open file structure,
   !> and sub domain indices. 
   !> @details
   !> sub domain indices are computed, taking into account coarse grid
   !> periodicity, pivot point, and East-West overlap.
   !>
   !> @author J.Paul
   !> @date June, 2013 - Initial Version
   !> @date September, 2014
   !> - add boundary index
   !> - add ghost cell factor
   !>
   !> @param[in] td_file   file structure
   !> @param[in] id_perio  grid periodicity
   !> @param[in] id_imin   i-direction sub-domain lower left  point indice
   !> @param[in] id_imax   i-direction sub-domain upper right point indice
   !> @param[in] id_jmin   j-direction sub-domain lower left  point indice
   !> @param[in] id_jmax   j-direction sub-domain upper right point indice
   !> @param[in] cd_card   name of cardinal (for boundary)
   !> @return domain structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TFILE)      , INTENT(IN) :: td_file 

      INTEGER(i4)      , INTENT(IN), OPTIONAL :: id_imin
      INTEGER(i4)      , INTENT(IN), OPTIONAL :: id_imax
      INTEGER(i4)      , INTENT(IN), OPTIONAL :: id_jmin
      INTEGER(i4)      , INTENT(IN), OPTIONAL :: id_jmax

      CHARACTER(LEN=*) , INTENT(IN), OPTIONAL :: cd_card

      ! function
      TYPE(TDOM)                    :: tf_dom

      !local variable
      !----------------------------------------------------------------

      ! clean domain structure
      CALL dom_clean(tf_dom)

      IF( td_file%i_id == 0 )THEN

         CALL logger_error( &
         &  " DOM INIT: no id associated to file "//TRIM(td_file%c_name))

      ELSE
         ! global domain define by file

         ! look for boundary index
         IF( PRESENT(cd_card) )THEN
            SELECT CASE(TRIM(cd_card))
               CASE('north')
                  tf_dom%i_bdy=jp_north
               CASE('south')
                  tf_dom%i_bdy=jp_south
               CASE('east')
                  tf_dom%i_bdy=jp_east
               CASE('west')
                  tf_dom%i_bdy=jp_west
               CASE DEFAULT
                  ! no boundary
                  tf_dom%i_bdy=0
            END SELECT
         ELSE
            ! no boundary
            tf_dom%i_bdy=0
         ENDIF

         ! use global dimension define by file
         tf_dom%t_dim0(:) = dim_copy(td_file%t_dim(:))

         IF( td_file%i_perio < 0 .OR. td_file%i_perio > 6 )THEN
            CALL logger_error("DOM INIT: invalid grid periodicity ("//&
            &  TRIM(fct_str(td_file%i_perio))//&
            &  ") you should use grid_get_perio to compute it")
         ELSE
            tf_dom%i_perio0=td_file%i_perio
         ENDIF

         ! global domain pivot point
         SELECT CASE(tf_dom%i_perio0)
            CASE(3,4)
               tf_dom%i_pivot = 0
            CASE(5,6)
               tf_dom%i_pivot = 1
            CASE DEFAULT
               tf_dom%i_pivot = 0
         END SELECT

         ! add ghost cell factor of global domain
         tf_dom%i_ghost0(:,:)=0
         SELECT CASE(tf_dom%i_perio0)
            CASE(0)
               tf_dom%i_ghost0(:,:)=1
            CASE(1)
               tf_dom%i_ghost0(jp_J,:)=1
            CASE(2)
               tf_dom%i_ghost0(jp_I,:)=1
               tf_dom%i_ghost0(jp_J,2)=1
            CASE(3,5)
               tf_dom%i_ghost0(jp_I,:)=1
               tf_dom%i_ghost0(jp_J,1)=1
            CASE(4,6)
               tf_dom%i_ghost0(jp_J,1)=1
         END SELECT

         ! look for EW overlap
         tf_dom%i_ew0=td_file%i_ew

         ! initialise domain as global
         tf_dom%i_imin = 1 
         tf_dom%i_imax = tf_dom%t_dim0(1)%i_len

         tf_dom%i_jmin = 1 
         tf_dom%i_jmax = tf_dom%t_dim0(2)%i_len

         ! sub domain dimension
         tf_dom%t_dim(:) = dim_copy(td_file%t_dim(:))

         ! define sub domain indices 
         CALL dom__define(tf_dom, id_imin, id_imax, id_jmin, id_jmax)

      ENDIF

   END FUNCTION dom__init_file
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom__define(td_dom, &
         &                id_imin, id_imax, id_jmin, id_jmax)
   !-------------------------------------------------------------------
   !> @brief 
   !> This subroutine define sub domain indices, and compute the size 
   !> of the sub domain.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_dom domain structure
   !> @param[in] id_imin   i-direction sub-domain lower left  point indice
   !> @param[in] id_imax   i-direction sub-domain upper right point indice
   !> @param[in] id_jmin   j-direction sub-domain lower left  point indice
   !> @param[in] id_jmax   j-direction sub-domain upper right point indice
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TDOM),  INTENT(INOUT) :: td_dom
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_imin
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_imax
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_jmin
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_jmax
      !----------------------------------------------------------------

      IF( PRESENT(id_imin) ) td_dom%i_imin = id_imin
      IF( PRESENT(id_imax) ) td_dom%i_imax = id_imax

      IF( PRESENT(id_jmin) ) td_dom%i_jmin = id_jmin
      IF( PRESENT(id_jmax) ) td_dom%i_jmax = id_jmax

      ! check indices
      IF(( td_dom%i_imin < -1 .OR. td_dom%i_imin > td_dom%t_dim0(1)%i_len ).OR. &
      &  ( td_dom%i_imax < -1 .OR. td_dom%i_imax > td_dom%t_dim0(1)%i_len ).OR. &
      &  ( td_dom%i_jmin < -1 .OR. td_dom%i_jmin > td_dom%t_dim0(2)%i_len ).OR. &
      &  ( td_dom%i_jmax < -1 .OR. td_dom%i_jmax > td_dom%t_dim0(2)%i_len ))THEN
         CALL logger_debug("0 <= imin ("//TRIM(fct_str(id_imin))//") < "//&
         &              TRIM(fct_str(td_dom%t_dim0(1)%i_len)))
         CALL logger_debug("0 <= imax ("//TRIM(fct_str(id_imax))//") < "//&
         &              TRIM(fct_str(td_dom%t_dim0(1)%i_len)))
         CALL logger_debug("0 <= jmin ("//TRIM(fct_str(id_jmin))//") < "//&
         &              TRIM(fct_str(td_dom%t_dim0(2)%i_len)))
         CALL logger_debug("0 <= jmax ("//TRIM(fct_str(id_jmax))//") < "//&
         &              TRIM(fct_str(td_dom%t_dim0(2)%i_len)))
         CALL logger_fatal( "DOM INIT DEFINE: invalid grid definition."// &
         &               " check min and max indices")
      ELSE

         ! force to select north fold
         IF( td_dom%i_perio0 > 2 .AND. &
         &   ( td_dom%i_jmax >= td_dom%t_dim0(2)%i_len-1 .OR. &
         &     td_dom%i_jmax < td_dom%i_jmin .OR. &
         &     td_dom%i_jmin == 0 ) )THEN
            td_dom%i_jmax=0
         ENDIF

         ! force to use cyclic boundary
         IF( ( td_dom%i_perio0 == 1 .OR. &
         &     td_dom%i_perio0 == 4 .OR. &
         &     td_dom%i_perio0 == 6 ) .AND. &
         &   ( td_dom%i_imin == 0 .OR. td_dom%i_imax == 0 .OR. &
         &     ABS(td_dom%i_imax-td_dom%i_imin)+1 == td_dom%t_dim0(1)%i_len ) &
         &  )THEN
            td_dom%i_imin = 0
            td_dom%i_imax = 0
         ENDIF

         SELECT CASE(td_dom%i_perio0)
            CASE(0) ! closed boundary
               CALL logger_trace("DOM INIT DEFINE: closed boundary")
               CALL dom__define_closed( td_dom )
            CASE(1) ! cyclic east-west boundary
               CALL logger_trace("DOM INIT DEFINE: cyclic east-west boundary")
               CALL dom__define_cyclic( td_dom )
            CASE(2) ! symmetric boundary condition across the equator
               CALL logger_trace("DOM INIT DEFINE: symmetric boundary condition "//&
               &                 " across the equator")
               CALL dom__define_symmetric( td_dom )
            CASE(3) ! North fold boundary (with a F-point pivot)  
               CALL logger_trace("DOM INIT DEFINE: North fold boundary "//&
               &                 "(with a F-point pivot)")
               CALL dom__define_north_fold( td_dom )
            CASE(5) ! North fold boundary (with a T-point pivot)
               CALL logger_trace("DOM INIT DEFINE: North fold boundary "//&
               &                 "(with a T-point pivot)")
               CALL dom__define_north_fold( td_dom )
            CASE(4) ! North fold boundary (with a F-point pivot) 
                    ! and cyclic east-west boundary
               CALL logger_trace("DOM INIT DEFINE:  North fold boundary "//&
               &                 "(with a F-point pivot) and cyclic "//&
               &                 "east-west boundary")
               CALL dom__define_cyclic_north_fold( td_dom )
            CASE(6) ! North fold boundary (with a T-point pivot) 
                    ! and cyclic east-west boundary
               CALL logger_trace("DOM INIT DEFINE: North fold boundary "//&
               &                 "(with a T-point pivot) and cyclic "//&
               &                 "east-west boundary")
               CALL dom__define_cyclic_north_fold( td_dom )
            CASE DEFAULT
               CALL logger_error("DOM INIT DEFINE: invalid grid periodicity index")
         END SELECT

      ENDIF

   END SUBROUTINE dom__define
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom__define_cyclic_north_fold(td_dom)
   !-------------------------------------------------------------------
   !> @brief 
   !> This subroutine define sub domain indices from global domain with
   !> cyclic east-west boundary and north fold boundary condition.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !> @date September, 2014
   !> - use zero indice to defined cyclic or global domain
   !>
   !> @param[inout] td_dom domain strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM), INTENT(INOUT) :: td_dom
      !----------------------------------------------------------------

      !CALL dom__check_EW_index( td_dom )

      IF( td_dom%i_imin == 0 .AND. td_dom%i_imax == 0 .AND. &
      &   td_dom%i_jmin == 0 .AND. td_dom%i_jmax == 0 )THEN

         CALL logger_trace("DOM DEFINE CYCLIC NORTH FOLD: "//&
         &  "domain to extract is global" )
         ! coarse domain is global domain

         CALL dom__size_global( td_dom )

      ELSEIF( td_dom%i_imin == 0 .AND. td_dom%i_imax == 0 .AND. &
      &       td_dom%i_jmax == 0 )THEN

         CALL logger_trace("DOM DEFINE CYCLIC NORTH FOLD: "//&
         &  "domain to extract is semi-global" )

         CALL dom__size_semi_global( td_dom )

      ELSEIF( td_dom%i_imin == 0 .AND. td_dom%i_imax == 0 .AND. &
      &       td_dom%i_jmax /= 0 )THEN

         CALL logger_trace("DOM DEFINE CYCLIC NORTH FOLD: "//&
         &  "domain to extract is band of latidue" )

         CALL dom__size_no_pole( td_dom )

      ELSEIF( td_dom%i_jmax == 0 )THEN

         CALL logger_trace("DOM DEFINE CYCLIC NORTH FOLD: "//&
         &  "domain to extract use north fold" )

         CALL dom__size_pole( td_dom )

      ELSEIF( td_dom%i_jmax /= 0 )THEN

         CALL logger_trace("DOM DEFINE CYCLIC NORTH FOLD: "//&
         &  "domain to extract do not use north fold" )
         ! no North Pole
         
         CALL dom__size_no_pole( td_dom )

      ELSE

         CALL logger_error("DOM DEFINE CYCLIC NORTH FOLD: "//&
         &  "should have been an impossible case" )

      ENDIF
      
   END SUBROUTINE dom__define_cyclic_north_fold
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom__define_north_fold(td_dom)
   !-------------------------------------------------------------------
   !> @brief 
   !> This subroutine define sub domain indices from global domain 
   !> with north fold boundary condition.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial verison
   !>
   !> @param[inout] td_dom domain strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM), INTENT(INOUT) :: td_dom
      !----------------------------------------------------------------

      IF( td_dom%i_jmax /= 0 )THEN

         CALL logger_trace("DOM DEFINE NORTH FOLD: "//&
         &  "domain to extract has no north boundary" )
         ! no North Pole
         
         CALL dom__size_no_pole_no_overlap( td_dom )

      ELSE

         CALL logger_trace("DOM DEFINE NORTH FOLD: "//&
         &  "sub domain has north boundary" )

         CALL dom__size_pole_no_overlap( td_dom )

      ENDIF      

   END SUBROUTINE dom__define_north_fold
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom__define_symmetric(td_dom)
   !-------------------------------------------------------------------
   !> @brief 
   !> This subroutine define sub domain indices from global domain 
   !> with symmetric boundary condition across the equator.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_dom domain strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM), INTENT(INOUT) :: td_dom
      !----------------------------------------------------------------

      CALL dom__size_no_pole_no_overlap( td_dom )

   END SUBROUTINE dom__define_symmetric
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom__define_cyclic(td_dom)
   !-------------------------------------------------------------------
   !> @brief 
   !> This subroutine define sub domain indices from global domain
   !> with cyclic east-west boundary.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_dom domain strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM), INTENT(INOUT) :: td_dom
      !----------------------------------------------------------------
      
      IF( td_dom%i_imin >= td_dom%i_imax )THEN
         CALL logger_trace("DOM DEFINE CYCLIC: "//&
         &  "domain to extract overlap east-west boundary")

         CALL dom__size_no_pole_overlap( td_dom )

      ELSE
         ! id_imin < id_imax
         CALL logger_trace("DOM DEFINE CYCLIC: "//&
         &  "domain to extract do not overlap east-west boundary")

         CALL dom__size_no_pole_no_overlap( td_dom )

      ENDIF

   END SUBROUTINE dom__define_cyclic
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom__define_closed(td_dom)
   !-------------------------------------------------------------------
   !> @brief 
   !> This subroutine define sub domain indices from global domain
   !> with closed boundaries.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_dom domain strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM), INTENT(INOUT) :: td_dom
      !----------------------------------------------------------------

      CALL dom__size_no_pole_no_overlap( td_dom )

   END SUBROUTINE dom__define_closed
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom__size_global(td_dom)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute size of global domain 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_dom domain strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM), INTENT(INOUT) :: td_dom
      !----------------------------------------------------------------

      td_dom%i_imin = 1                     
      td_dom%i_imax = td_dom%t_dim0(1)%i_len

      td_dom%i_jmin = 1 
      td_dom%i_jmax = td_dom%t_dim0(2)%i_len

      ! domain size
      td_dom%t_dim(1)%i_len = td_dom%t_dim0(1)%i_len
      td_dom%t_dim(2)%i_len = td_dom%t_dim0(2)%i_len

      ! no ghost cell to add
      td_dom%i_ghost(:,:)=0

      ! periodicity
      IF( td_dom%i_pivot == 0 )THEN ! 0-F
         td_dom%i_perio=4
         td_dom%i_pivot=0
      ELSE ! 1-T
         td_dom%i_perio=6
         td_dom%i_pivot=1
      ENDIF

   END SUBROUTINE dom__size_global
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom__size_semi_global(td_dom)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute size of a semi global domain 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_dom domain strcuture
   !> @note never tested
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM), INTENT(INOUT) :: td_dom

      ! local variable
      INTEGER(i4) :: il_imid   ! canadian bipole index (middle of global domain) 
      !----------------------------------------------------------------

      il_imid = td_dom%t_dim0(1)%i_len/2 + td_dom%i_pivot

      td_dom%i_imin = 2
      td_dom%i_imax = il_imid !td_dom%t_dim0(1)%i_len

      IF( td_dom%i_jmin == 0 ) td_dom%i_jmin=1
      td_dom%i_jmax = td_dom%t_dim0(2)%i_len

      ! domain size
      td_dom%t_dim(1)%i_len = td_dom%i_imax - &
      &                       td_dom%i_imin + 1

      td_dom%t_dim(2)%i_len = ( td_dom%t_dim0(2)%i_len - &
      &                         td_dom%i_jmin + 1 ) +      &
      &                         ( td_dom%t_dim0(2)%i_len - &
      &                           td_dom%i_jmin + 1 ) - 2    ! remove north fold condition ?

      ! ghost cell to add
      td_dom%i_ghost(:,:)=1

      ! periodicity
      IF( td_dom%i_pivot == 0 )THEN !0-F
         td_dom%i_perio=3
         td_dom%i_pivot=0
      ELSE !1-T
         td_dom%i_perio=5
         td_dom%i_pivot=1
      ENDIF

   END SUBROUTINE dom__size_semi_global
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom__size_no_pole(td_dom)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute size of sub domain without north fold
   !> condition
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_dom domain strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM), INTENT(INOUT) :: td_dom
      !----------------------------------------------------------------

      IF( td_dom%i_jmax == 0 )THEN
         CALL logger_fatal("DOM SIZE NO POLE: invalid domain. "//&
         &  "can not get north pole from this coarse grid. "//&
         &  "check namelist and coarse grid periodicity." )
      ENDIF

      IF( td_dom%i_imin == 0 .AND. td_dom%i_imax == 0 .OR. &
      &   td_dom%i_imin > td_dom%i_imax )THEN
         CALL logger_trace("DOM SIZE NO POLE: "// &
         &  "domain to extract overlap east-west boundary")

         CALL dom__size_no_pole_overlap( td_dom )

      ELSE
         ! id_imin < id_imax
         CALL logger_trace("DOM SIZE NO POLE: "// &
         &  "domain to extract do not overlap east-west boundary")

         CALL dom__size_no_pole_no_overlap( td_dom )

      ENDIF

   END SUBROUTINE dom__size_no_pole
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom__size_pole(td_dom)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute size of sub domain with north fold
   !> condition.
   !>
   !> @author J.Paul
   !> @date April, 2013 - Initial version
   !>
   !> @param[inout] td_dom domain strcuture
   !> @note never tested
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM), INTENT(INOUT) :: td_dom
      !----------------------------------------------------------------

      IF( td_dom%i_imin >= td_dom%i_imax )THEN
         CALL logger_trace("DOM SIZE POLE: "//&
         &  "domain to extract overlap east-west boundary")
         CALL dom__size_pole_overlap( td_dom )
      ELSEIF( td_dom%i_imin < td_dom%i_imax )THEN
         CALL logger_trace("DOM SIZE POLE: "//&
         &  "domain to extract do not overlap east-west boundary")
         CALL dom__size_pole_no_overlap( td_dom )
      ENDIF

   END SUBROUTINE dom__size_pole
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom__size_no_pole_overlap(td_dom)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute size of sub domain without north fold
   !> condition, and which overlap east-west boundary
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_dom domain strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM), INTENT(INOUT) :: td_dom
      !----------------------------------------------------------------

      IF( td_dom%i_jmax == 0 )THEN
         CALL logger_fatal("DOM SIZE NO POLE OVERLAP: invalid domain. "//&
         &  "can not get north pole from this coarse grid. "//&
         &  "check namelist and coarse grid periodicity." )
      ENDIF

      IF( td_dom%i_imin == 0 .AND. td_dom%i_imax == 0 )THEN
         ! domain to extract with east west cyclic boundary
         CALL logger_trace("DOM SIZE NO POLE OVERLAP: "//&
         &  "domain to extract has cyclic east-west boundary")

         td_dom%i_imin = 1
         td_dom%i_imax = td_dom%t_dim0(1)%i_len

         td_dom%t_dim(1)%i_len = td_dom%t_dim0(1)%i_len

         ! no ghost cell
         td_dom%i_ghost(jp_I,:)=0

         ! periodicity
         td_dom%i_perio=1

      ELSE

         ! id_imin > id_imax
         ! extract domain overlap east-west boundary

         td_dom%t_dim(1)%i_len = td_dom%i_imax + &
         &                       td_dom%t_dim0(1)%i_len - td_dom%i_imin + 1 - & 
         &                       td_dom%i_ew0     ! remove cyclic boundary

         ! add ghost cell
         td_dom%i_ghost(jp_I,:)=1

         ! periodicity
         td_dom%i_perio=0

      ENDIF

      td_dom%t_dim(2)%i_len = td_dom%i_jmax - &
      &                       td_dom%i_jmin + 1

      ! add ghost cell
      td_dom%i_ghost(jp_J,:)=1

   END SUBROUTINE dom__size_no_pole_overlap
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom__size_no_pole_no_overlap(td_dom)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute size of sub domain without north fold
   !> condition, and which do not overlap east-west boundary
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_dom domain strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM), INTENT(INOUT) :: td_dom
      !----------------------------------------------------------------

      IF( td_dom%i_jmax == 0 )THEN
         CALL logger_fatal("DOM SIZE NO POLE NO OVERLAP: invalid domain. "//&
         &  "can not get north pole from this coarse grid. "//&
         &  "check domain indices and grid periodicity." )
      ENDIF

      IF( td_dom%i_imin == 0 .OR. td_dom%i_imax == 0 )THEN
         CALL logger_fatal("DOM SIZE NO POLE NO OVERLAP: invalid domain. "//&
         &  "can not overlap East-West boundary with this coarse grid. "//&
         &  "check domain indices and grid periodicity." )
      ENDIF

      td_dom%t_dim(1)%i_len = td_dom%i_imax - & 
      &                       td_dom%i_imin + 1 

      td_dom%t_dim(2)%i_len = td_dom%i_jmax - &
      &                       td_dom%i_jmin + 1
      
      ! add ghost cell
      td_dom%i_ghost(:,:)=1

      ! periodicity
      td_dom%i_perio=0

   END SUBROUTINE dom__size_no_pole_no_overlap
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom__size_pole_overlap(td_dom)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute size of sub domain with north fold
   !> condition, and which overlap east-west boundary
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_dom domain strcuture
   !> @note never tested
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM), INTENT(INOUT) :: td_dom

      ! local variable
      INTEGER(i4) :: il_idom1  ! extract domain size, east part
      INTEGER(i4) :: il_idom2  ! extract domain size, west part
      INTEGER(i4) :: il_imid   ! cananadian bipole index (middle of global domain) 
      !----------------------------------------------------------------

      CALL logger_trace("DOM SIZE POLE OVERLAP: "//&
      &  "asian bipole inside domain to extract")

      il_imid = td_dom%t_dim0(1)%i_len/2 + td_dom%i_pivot

      il_idom1 = td_dom%t_dim0(1)%i_len - td_dom%i_imin + 1
      il_idom2 = td_dom%i_imax
      
      IF( il_idom1 > il_imid .OR. il_idom2 > il_imid )THEN

         CALL logger_trace("DOM SIZE POLE OVERLAP: "//&
         &  "canadian bipole inside domain to extract")
         td_dom%i_imin = 0
         td_dom%i_imax = 0

         IF( td_dom%i_jmin == 0 .AND. td_dom%i_jmax == 0 )THEN
            CALL dom__size_global( td_dom )
         ELSE
            CALL dom__size_semi_global( td_dom )
         ENDIF

         ! periodicity
         td_dom%i_perio=0

      ELSEIF( il_idom1 > il_idom2 )THEN

         ! east part bigger than west part
         CALL logger_trace("DOM SIZE POLE OVERLAP: east part bigger than west part ")
         ! to respect symmetry around asian bipole
         td_dom%i_imax = il_idom1

         IF( td_dom%i_jmin == 0 ) td_dom%i_jmin = 1 
         ! north pole
         td_dom%i_jmax = td_dom%t_dim0(2)%i_len

         ! compute size
         td_dom%t_dim(1)%i_len = il_idom1  !! no ghost cell ??
         td_dom%t_dim(2)%i_len = ( td_dom%t_dim0(2)%i_len - &
         &                         td_dom%i_jmin + 1 ) + &   
         &                         ( td_dom%t_dim0(2)%i_len - &
         &                         td_dom%i_jmin + 1 ) - 2   ! remove north fold condition ?

         ! add ghost cell
         td_dom%i_ghost(:,:)=1

         ! periodicity
         td_dom%i_perio=0

      ELSE ! il_idom2 >= il_idom1

         ! west part bigger than east part
         CALL logger_trace("DOM SIZE POLE OVERLAP: west part bigger than east part ")

         ! to respect symmetry around asian bipole
         td_dom%i_imin = td_dom%t_dim0(1)%i_len - il_idom2 + 1

         IF( td_dom%i_jmin == 0 ) td_dom%i_jmin = 1 
         ! north pole
         td_dom%i_jmax=td_dom%t_dim0(2)%i_len

         ! compute size
         td_dom%t_dim(1)%i_len = il_idom2
         td_dom%t_dim(2)%i_len = ( td_dom%t_dim0(2)%i_len - &
         &                         td_dom%i_jmin + 1 ) + &
         &                         ( td_dom%t_dim0(2)%i_len - &
         &                         td_dom%i_jmin + 1 ) - 2

         ! add ghost cell
         td_dom%i_ghost(:,:)=1
         
         ! periodicity
         td_dom%i_perio=0

      ENDIF

   END SUBROUTINE dom__size_pole_overlap
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom__size_pole_no_overlap(td_dom)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute size of sub domain with north fold
   !> condition, and which do not overlap east-west boundary
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_dom domain strcuture
   !> @note never tested
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM), INTENT(INOUT) :: td_dom

      ! local variable
      INTEGER(i4) :: il_idom1  ! extract domain size, east part
      INTEGER(i4) :: il_idom2  ! extract domain size, west part
      INTEGER(i4) :: il_mid    ! canadian biple index ?
      !----------------------------------------------------------------

      IF( td_dom%i_imin == 0 .OR. td_dom%i_imax == 0 .OR. &
      &   td_dom%i_imin > td_dom%i_imax )THEN
         CALL logger_fatal("DOM SIZE POLE NO OVERLAP: invalid domain. "//&
         &  "can not overlap East-West boundary with this coarse grid. "//&
         &  "check namelist and coarse grid periodicity." )
      ENDIF

      CALL logger_trace("DOM SIZE POLE NO OVERLAP: "//&
      &  "no asian bipole inside domain to extract")

      IF( td_dom%i_jmin==0 ) td_dom%i_jmin = 1
      IF( td_dom%i_jmax==0 ) td_dom%i_jmax = td_dom%t_dim0(2)%i_len

      ! 
      il_mid = td_dom%t_dim0(1)%i_len/2 + td_dom%i_pivot

      IF( (td_dom%i_imin < il_mid .AND. td_dom%i_imax < il_mid) .OR. &
      &   (td_dom%i_imin > il_mid .AND. td_dom%i_imax > il_mid) )THEN
         CALL logger_trace("DOM SIZE POLE NO OVERLAP: "//&
         &  "no canadian bipole inside domain to extract")

         td_dom%t_dim(1)%i_len = td_dom%i_imax - &
         &                       td_dom%i_imin + 1

         td_dom%t_dim(2)%i_len = ( td_dom%t_dim0(2)%i_len - &
         &                       td_dom%i_jmin + 1 ) + &
         &                       ( td_dom%t_dim0(2)%i_len - &
         &                       td_dom%i_jmin + 1 ) - 2 ! remove north fold condition ?

         ! add ghost cell
         td_dom%i_ghost(:,:)=1

         ! periodicity
         td_dom%i_perio=0

      ELSE ! id_imin < il_mid .AND. id_imax > il_mid
         CALL logger_trace("DOM SIZE POLE NO OVERLAP: "//&
         &  "canadian bipole inside domain to extract")

         il_idom1 = td_dom%i_imax - (il_mid - 1)
         il_idom2 = il_mid  - td_dom%i_imin
         IF( il_idom1 > il_idom2 )THEN
            ! east part bigger than west part
            CALL logger_trace("DOM SIZE POLE NO OVERLAP: east part bigger than west part ")
            ! to respect symmetry around canadian bipole
            td_dom%i_imin = il_mid - il_idom1

            td_dom%t_dim(1)%i_len = il_idom1 + 1
            td_dom%t_dim(2)%i_len = ( td_dom%t_dim0(2)%i_len - &
            &                         td_dom%i_jmin + 1 ) + & 
            &                         ( td_dom%t_dim0(2)%i_len - &
            &                         td_dom%i_jmin + 1 ) &   
            &                         - 2 - 2 * td_dom%i_pivot    ! remove north fold condition ?

            ! add ghost cell
            td_dom%i_ghost(:,:)=1

            ! periodicity
            td_dom%i_perio=0

         ELSE ! il_idom2 >= il_idom1
            ! west part bigger than east part
            CALL logger_trace("DOM SIZE POLE NO OVERLAP: west part bigger than east part ")
            ! to respect symmetry around canadian bipole

            td_dom%i_imax = il_mid + il_idom2

            td_dom%t_dim(1)%i_len = il_idom2 + 1
            td_dom%t_dim(2)%i_len = ( td_dom%t_dim0(2)%i_len -  &
            &                         td_dom%i_jmin + 1 ) +     & 
            &                         ( td_dom%t_dim0(2)%i_len -  &
            &                         td_dom%i_jmax + 1 )       & 
            &                         - 2 - 2 * td_dom%i_pivot  !  remove north fold condition ?

            ! add ghost cell
            td_dom%i_ghost(:,:)=1

            ! periodicity
            td_dom%i_perio=0            

         ENDIF
      ENDIF

   END SUBROUTINE dom__size_pole_no_overlap
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom_add_extra(td_dom, id_iext, id_jext)
   !-------------------------------------------------------------------
   !> @brief 
   !>  This subroutine add extra bands to coarse domain to get enough point for
   !>  interpolation...
   !>
   !> @details
   !>  - domain periodicity is take into account.<br/>
   !>  - domain indices are changed, and size of extra bands are saved.<br/>
   !>  - optionaly, i- and j- direction size of extra bands could be specify 
   !> (default=im_minext)
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !> @date September, 2014
   !> - take into account number of ghost cell
   !> @date February, 2016
   !> - number of extra point is the MAX (not the MIN) of zero and asess value. 
   !>
   !> @param[inout] td_dom domain strcuture
   !> @param [in] id_iext  i-direction size of extra bands (default=im_minext)
   !> @param [in] id_jext  j-direction size of extra bands (default=im_minext)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM) ,  INTENT(INOUT) :: td_dom
      INTEGER(i4),  INTENT(IN   ), OPTIONAL :: id_iext
      INTEGER(i4),  INTENT(IN   ), OPTIONAL :: id_jext

      ! local variable
      INTEGER(i4) :: il_iext
      INTEGER(i4) :: il_jext

      ! loop indices
      !----------------------------------------------------------------
      ! init
      il_iext=im_minext
      IF( PRESENT(id_iext) ) il_iext=id_iext

      il_jext=im_minext
      IF( PRESENT(id_jext) ) il_jext=id_jext

      td_dom%i_iextra(:)=0
      td_dom%i_jextra(:)=0

      IF( td_dom%i_imin == 1                       .AND. &
      &   td_dom%i_imax == td_dom%t_dim0(1)%i_len  .AND. &
      &   td_dom%i_jmin == 1                       .AND. &
      &   td_dom%i_jmax == td_dom%t_dim0(2)%i_len )THEN
         ! global
         ! nothing to be done
      ELSE

         IF( td_dom%i_imin == 1                       .AND. &
         &   td_dom%i_imax == td_dom%t_dim0(1)%i_len )THEN
            ! EW cyclic
            ! nothing to be done
         ELSE
            IF( td_dom%i_ew0 < 0 )THEN
               ! EW not cyclic
               IF( td_dom%i_imin - il_iext > td_dom%i_ghost0(jp_I,1)*ip_ghost )THEN
                  td_dom%i_iextra(1) = il_iext
                  td_dom%i_imin      = td_dom%i_imin - td_dom%i_iextra(1)
               ELSE ! td_dom%i_imin - il_iext <= td_dom%i_ghost0(jp_I,1)*ip_ghost
                  td_dom%i_iextra(1) = MAX(0, &
                  &                         td_dom%i_imin - &
                  &                         td_dom%i_ghost0(jp_I,1)*ip_ghost -1)
                  td_dom%i_imin      = td_dom%i_imin - td_dom%i_iextra(1)
               ENDIF

               IF( td_dom%i_imax + il_iext < &
               &   td_dom%t_dim0(1)%i_len - td_dom%i_ghost0(jp_I,2)*ip_ghost )THEN
                  td_dom%i_iextra(2) = il_iext
                  td_dom%i_imax      = td_dom%i_imax + td_dom%i_iextra(2)
               ELSE ! td_dom%i_imax + il_iext >= &
                    !  td_dom%t_dim0(1)%i_len - td_dom%i_ghost0(jp_I,2)*ip_ghost
                  td_dom%i_iextra(2) = MAX( 0, &
                  &                         td_dom%t_dim0(1)%i_len - &
                  &                         td_dom%i_ghost0(jp_I,2)*ip_ghost - &
                  &                         td_dom%i_imax )
                  td_dom%i_imax      = td_dom%i_imax + td_dom%i_iextra(2)
               ENDIF

            ELSE ! td_dom%i_ew0 >= 0

               ! EW cyclic
               IF( td_dom%i_imin - il_iext > 0 )THEN
                  td_dom%i_iextra(1) = il_iext
                  td_dom%i_imin      = td_dom%i_imin - td_dom%i_iextra(1)
               ELSE ! td_dom%i_imin - il_iext <= 0
                  td_dom%i_iextra(1) = il_iext
                  td_dom%i_imin      = td_dom%t_dim0(1)%i_len + &
                  &                     td_dom%i_imin - td_dom%i_iextra(1) -&
                  &                     td_dom%i_ew0
               ENDIF

               IF( td_dom%i_imax + il_iext <= td_dom%t_dim0(1)%i_len )THEN
                  td_dom%i_iextra(2) = il_iext
                  td_dom%i_imax      = td_dom%i_imax + td_dom%i_iextra(2)
               ELSE ! td_dom%i_imax + il_iext > td_dom%t_dim0(1)%i_len
                  td_dom%i_iextra(2) = il_iext
                  td_dom%i_imax      = td_dom%i_imax + td_dom%i_iextra(2) - &
                  &                     (td_dom%t_dim0(1)%i_len-td_dom%i_ew0) 
               ENDIF               
            ENDIF

         ENDIF

         IF( td_dom%i_jmin == 1                       .AND. &
         &   td_dom%i_jmax == td_dom%t_dim0(2)%i_len )THEN
            ! nothing to be done
         ELSE

            IF( td_dom%i_jmin - il_jext > td_dom%i_ghost0(jp_J,1)*ip_ghost )THEN
               td_dom%i_jextra(1) = il_jext
               td_dom%i_jmin      = td_dom%i_jmin - td_dom%i_jextra(1)
            ELSE ! td_dom%i_jmin - il_jext <= td_dom%i_ghost0(jp_J,1)*ip_ghost
               td_dom%i_jextra(1) = MAX( 0, &
               &                         td_dom%i_jmin - &
               &                         td_dom%i_ghost0(jp_J,1)*ip_ghost - 1)
               td_dom%i_jmin      = td_dom%i_jmin - td_dom%i_jextra(1)
            ENDIF

            IF( td_dom%i_jmax + il_jext < &
            &   td_dom%t_dim0(2)%i_len - td_dom%i_ghost0(jp_J,2)*ip_ghost )THEN
               td_dom%i_jextra(2) = il_jext
               td_dom%i_jmax      = td_dom%i_jmax + td_dom%i_jextra(2)
            ELSE ! td_dom%i_jmax + il_jext >= &
                 !  td_dom%t_dim0(2)%i_len - td_dom%i_ghost0(jp_J,2)*ip_ghost
               td_dom%i_jextra(2) = MAX( 0, &
               &                         td_dom%t_dim0(2)%i_len - &
               &                         td_dom%i_ghost0(jp_J,2)*ip_ghost - &
               &                         td_dom%i_jmax )
               td_dom%i_jmax      = td_dom%i_jmax + td_dom%i_jextra(2)
            ENDIF
         ENDIF         

      ENDIF

      IF( td_dom%i_imin <= td_dom%i_imax )THEN
         td_dom%t_dim(1)%i_len = td_dom%i_imax - td_dom%i_imin +1 
      ELSE ! td_dom%i_imin > td_dom%i_imax
         td_dom%t_dim(1)%i_len = td_dom%i_imax + &
         &                       td_dom%t_dim0(1)%i_len - td_dom%i_imin + 1 - &
         &                       td_dom%i_ew0 ! remove overlap
      ENDIF

      td_dom%t_dim(2)%i_len = td_dom%i_jmax-td_dom%i_jmin+1


   END SUBROUTINE dom_add_extra
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom_clean_extra(td_dom)
   !-------------------------------------------------------------------
   !> @brief 
   !>  This subroutine clean coarse grid domain structure. 
   !> it remove extra point added. 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_dom domain strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM) , INTENT(INOUT) :: td_dom

      ! local variable
      ! loop indices
      !----------------------------------------------------------------

      ! change domain
      td_dom%i_imin         = td_dom%i_imin + td_dom%i_iextra(1)
      td_dom%i_jmin         = td_dom%i_jmin + td_dom%i_jextra(1)

      td_dom%i_imax         = td_dom%i_imax - td_dom%i_iextra(2)
      td_dom%i_jmax         = td_dom%i_jmax - td_dom%i_jextra(2)

      td_dom%t_dim(1)%i_len = td_dom%t_dim(1)%i_len - &
      &                          td_dom%i_iextra(1) - &
      &                          td_dom%i_iextra(2)
      td_dom%t_dim(2)%i_len = td_dom%t_dim(2)%i_len - &
      &                          td_dom%i_jextra(1) - &
      &                          td_dom%i_jextra(2)

      td_dom%i_iextra(:)=0
      td_dom%i_jextra(:)=0

   END SUBROUTINE dom_clean_extra
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom_del_extra(td_var, td_dom, id_rho, ld_coord)
   !-------------------------------------------------------------------
   !> @brief 
   !>  This subroutine delete extra band, from fine grid variable value, 
   !> and dimension, taking into account refinement factor.
   !>
   !> @details
   !> @note This subroutine should be used before clean domain structure.
   !>
   !> @warning if work on coordinates grid, do not remove all extra point.
   !> save value on ghost cell. 
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !> @date September, 2014
   !> - take into account boundary for one point size domain
   !> @date December, 2014
   !> - add special case for coordinates file.
   !>
   !> @param[inout] td_var variable strcuture
   !> @param[in] td_dom    domain strcuture
   !> @param[in] id_rho    array of refinement factor
   !> @param[in] ld_coord  work on coordinates file or not
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR)               , INTENT(INOUT) :: td_var
      TYPE(TDOM)               , INTENT(IN   ) :: td_dom
      INTEGER(i4), DIMENSION(:), INTENT(IN   ), OPTIONAL :: id_rho
      LOGICAL                  , INTENT(IN   ), OPTIONAL :: ld_coord

      ! local variable
      INTEGER(i4) :: il_iextra
      INTEGER(i4) :: il_jextra

      INTEGER(i4) :: il_imin
      INTEGER(i4) :: il_imax
      INTEGER(i4) :: il_jmin
      INTEGER(i4) :: il_jmax
      
      INTEGER(i4), DIMENSION(2)   :: il_rho
      INTEGER(i4), DIMENSION(2,2) :: il_ghost

      REAL(dp)   , DIMENSION(:,:,:,:), ALLOCATABLE :: dl_value

      LOGICAL     :: ll_coord
      ! loop indices
      !----------------------------------------------------------------

      IF( PRESENT(id_rho) )THEN
         ! work on coarse grid
         il_rho(:)=id_rho(jp_I:jp_J)
      ELSE
         ! work on fine grid
         il_rho(:)=1
      ENDIF

      ll_coord=.false.
      IF( PRESENT(ld_coord) ) ll_coord=ld_coord

      IF( .NOT. ASSOCIATED(td_var%d_value) )THEN
         CALL logger_error("DOM DEL EXTRA: no value associated to "//&
         &     "variable "//TRIM(td_var%c_name) )
      ELSE
         ! get variable right domain
         IF( ALL(td_var%t_dim(1:2)%l_use) )THEN

            ALLOCATE(dl_value(td_var%t_dim(1)%i_len, &
            &                 td_var%t_dim(2)%i_len, &
            &                 td_var%t_dim(3)%i_len, &
            &                 td_var%t_dim(4)%i_len) )
            dl_value(:,:,:,:)=td_var%d_value(:,:,:,:)

            il_iextra=SUM(td_dom%i_iextra(:))*il_rho(jp_I)
            il_jextra=SUM(td_dom%i_jextra(:))*il_rho(jp_J)

            il_ghost(:,:)=0
            IF( ll_coord )THEN
               il_ghost(:,:)=td_dom%i_ghost(:,:)
            ENDIF

            IF( il_iextra >= td_var%t_dim(1)%i_len )THEN
               ! case one point size dimension
               SELECT CASE(td_dom%i_bdy) 

                  CASE(jp_north,jp_east)

                     CALL logger_info("DOM DEL EXTRA: special case for north"//&
                     &                " or east boundary.")
                     IF( td_dom%i_iextra(1) <= 0 )THEN
                        il_imin= 1
                        il_ghost(jp_I,1) = 0
                     ELSE
                        il_imin= 1 + (td_dom%i_iextra(1)-1)*il_rho(jp_I) + 1 &
                        &        - il_ghost(jp_I,1)
                     ENDIF
                     IF( td_dom%i_iextra(2) <= 0 )THEN;
                        il_imax= td_var%t_dim(1)%i_len
                        il_ghost(jp_I,2) = 0
                     ELSE
                        il_imax= td_var%t_dim(1)%i_len - &
                        &          td_dom%i_iextra(2)*il_rho(jp_I) &
                        &        + il_ghost(jp_I,2)
                     ENDIF

                  CASE(jp_south,jp_west)

                     CALL logger_info("DOM DEL EXTRA: special case for south"//&
                     &                " or west boundary.")
                     IF( td_dom%i_iextra(1) <= 0 )THEN
                        il_imin= 1
                        il_ghost(jp_I,1) = 0
                     ELSE
                        il_imin= 1 + td_dom%i_iextra(1)*il_rho(jp_I) &
                        &        - il_ghost(jp_I,1)
                     ENDIF
                     IF( td_dom%i_iextra(2) <= 0 )THEN
                        il_imax= td_var%t_dim(1)%i_len
                        il_ghost(jp_I,2) = 0
                     ELSE
                        il_imax= td_var%t_dim(1)%i_len - &
                        &          (td_dom%i_iextra(2)-1)*il_rho(jp_I) - 1 &
                        &        + il_ghost(jp_I,2)
                     ENDIF

                  CASE DEFAULT

                     IF( MOD(il_iextra-td_var%t_dim(1)%i_len,2)==0 )THEN
                        ! case one point size dimension with even refinment
                        CALL logger_fatal("DOM DEL EXTRA: should have been"//&
                        &                 "an impossible case: domain of "//&
                        &                 " one point size and even refinment.")
                     ELSE
                        il_imin= 1 + &
                        &        (td_dom%i_iextra(1)-1)*il_rho(jp_I) + &
                        &        (il_rho(jp_I)-1)/2 + 1                &
                        &        - il_ghost(jp_I,1)
                        il_imax= td_var%t_dim(1)%i_len - &
                        &        (td_dom%i_iextra(2)-1)*il_rho(jp_I) - &
                        &        (il_rho(jp_I)-1)/2 - 1                &
                        &        + il_ghost(jp_I,2)
                     ENDIF

               END SELECT

               td_var%t_dim(1)%i_len = 1 + SUM(il_ghost(jp_I,:))

            ELSE
               ! general case
               il_imin=1                     + td_dom%i_iextra(1)*il_rho(jp_I) &
               &                             - il_ghost(jp_I,1)
               il_imax=td_var%t_dim(1)%i_len - td_dom%i_iextra(2)*il_rho(jp_I) &
               &                             + il_ghost(jp_I,2)

               td_var%t_dim(1)%i_len=td_var%t_dim(1)%i_len - il_iextra &
               &                                         + SUM(il_ghost(jp_I,:))
            ENDIF

            IF( il_jextra >= td_var%t_dim(2)%i_len )THEN
               ! case one point size dimension
               SELECT CASE(td_dom%i_bdy) 

                  CASE(jp_north,jp_east)

                     IF( td_dom%i_jextra(1) <= 0 )THEN
                        il_jmin= 1
                        il_ghost(jp_J,1) = 0
                     ELSE
                        il_jmin= 1 + (td_dom%i_jextra(1)-1)*il_rho(jp_J) + 1 &
                        &        - il_ghost(jp_J,1)
                     ENDIF
                     IF( td_dom%i_jextra(2) <= 0 )THEN
                        il_jmax= td_var%t_dim(2)%i_len
                        il_ghost(jp_J,2) = 0
                     ELSE
                        il_jmax= td_var%t_dim(2)%i_len - &
                        &          td_dom%i_jextra(2)*il_rho(jp_J) &
                        &        + il_ghost(jp_J,2)
                     ENDIF

                  CASE(jp_south,jp_west)

                     IF( td_dom%i_iextra(2) <= 0 )THEN
                        il_jmin= 1
                        il_ghost(jp_J,1) = 0
                     ELSE
                        il_jmin= 1 + td_dom%i_jextra(1)*il_rho(jp_J) &
                        &        - il_ghost(jp_J,1)
                     ENDIF
                     IF( td_dom%i_jextra(2) <= 0 )THEN
                        il_jmax= td_var%t_dim(2)%i_len
                        il_ghost(jp_J,2) = 0
                     ELSE
                        il_jmax= td_var%t_dim(2)%i_len - &
                        &          (td_dom%i_jextra(2)-1)*il_rho(jp_J) - 1 &
                        &        + il_ghost(jp_J,2)
                     ENDIF

                  CASE DEFAULT

                     IF( MOD(il_jextra-td_var%t_dim(2)%i_len,2)==0 )THEN
                        ! case one point size dimension with even refinment
                        CALL logger_fatal("DOM DEL EXTRA: should have been"//&
                        &                 "an impossible case: domain of "//&
                        &                 " one point size and even refinment.")
                     ELSE
                        il_jmin= 1 + &
                        &        (td_dom%i_jextra(1)-1)*il_rho(jp_J) + &
                        &        (il_rho(jp_J)-1)/2 + 1 &
                        &        - il_ghost(jp_J,1)
                        il_jmax= td_var%t_dim(2)%i_len - &
                        &        (td_dom%i_jextra(2)-1)*il_rho(jp_J) - &
                        &        (il_rho(jp_J)-1)/2 - 1 &
                        &        + il_ghost(jp_J,2)
                     ENDIF

               END SELECT

               td_var%t_dim(2)%i_len = 1 + SUM(il_ghost(jp_J,:))

            ELSE
               ! general case
               il_jmin=1                     + td_dom%i_jextra(1)*il_rho(jp_J) &
               &                             - il_ghost(jp_J,1)
               il_jmax=td_var%t_dim(2)%i_len - td_dom%i_jextra(2)*il_rho(jp_J) &
               &                             + il_ghost(jp_J,2)

                td_var%t_dim(2)%i_len= td_var%t_dim(2)%i_len - il_jextra &
                &                                        + SUM(il_ghost(jp_J,:))
            ENDIF

            DEALLOCATE(td_var%d_value)
            ALLOCATE(td_var%d_value(td_var%t_dim(1)%i_len, &
            &                       td_var%t_dim(2)%i_len, &
            &                       td_var%t_dim(3)%i_len, &
            &                       td_var%t_dim(4)%i_len) )

            td_var%d_value(:,:,:,:)=dl_value(il_imin:il_imax, &
            &                                il_jmin:il_jmax, &
            &                                :, :)
            DEALLOCATE(dl_value)
         ENDIF

      ENDIF

   END SUBROUTINE dom_del_extra
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE dom_clean(td_dom)
   !-------------------------------------------------------------------
   !> @brief 
   !>  This subroutine clean domain structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_dom domain strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TDOM),  INTENT(INOUT) :: td_dom

      ! local variable
      TYPE(TDOM) :: tl_dom ! empty dom structure

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      CALL logger_info( "DOM CLEAN: reset domain " )

      ! del dimension
      DO ji=ip_maxdim,1,-1
         CALL dim_clean( td_dom%t_dim0(ji) )
      ENDDO

      ! replace by empty structure
      td_dom=tl_dom

   END SUBROUTINE dom_clean
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE dom
