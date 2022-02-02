!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief This module is grid manager.
!>
!> @details
!>    to get NEMO pivot point index:<br/>
!> @code
!>    il_pivot=grid_get_pivot(td_file)
!> @endcode
!>       - il_pivot is NEMO pivot point index F(0), T(1)
!>       - td_file is mpp structure 
!>
!>    to get NEMO periodicity index:<br/>
!> @code
!>    il_perio=grid_get_perio(td_file)
!> @endcode
!>       - il_perio is NEMO periodicity index (0,1,2,3,4,5,6)
!>       - td_file is mpp structure
!>
!>    to check domain validity:<br/>
!> @code
!>    CALL grid_check_dom(td_coord, id_imin, id_imax, id_jmin, id_jmax)
!> @endcode
!>       - td_coord is coordinates mpp structure
!>       - id_imin is i-direction lower left  point indice
!>       - id_imax is i-direction upper right point indice
!>       - id_jmin is j-direction lower left  point indice
!>       - id_jmax is j-direction upper right point indice
!>
!>    to get closest coarse grid indices of fine grid domain:<br/>
!> @code
!>    il_index(:,:)=grid_get_coarse_index(td_coord0, td_coord1,
!>                                      [id_rho,] [cd_point])
!> @endcode
!>    or 
!> @code
!>    il_index(:,:)=grid_get_coarse_index(td_lon0, td_lat0, td_coord1,
!>                                      [id_rho,] [cd_point])
!> @endcode
!>    or
!> @code
!>    il_index(:,:)=grid_get_coarse_index(td_coord0, td_lon1, td_lat1,
!>                                      [id_rho,] [cd_point])
!> @endcode
!>    or
!> @code
!>    il_index(:,:)=grid_get_coarse_index(td_lon0, td_lat0, td_lon1, td_lat1,
!>                                      [id_rho,] [cd_point])
!> @endcode
!>       - il_index(:,:) is  coarse grid indices (/ (/ imin0, imax0 /), 
!> (/ jmin0, jmax0 /) /)
!>       - td_coord0 is coarse grid coordinate mpp structure 
!>       - td_coord1 is fine grid coordinate mpp structure
!>       - td_lon0 is coarse grid longitude variable structure 
!>       - td_lat0 is coarse grid latitude  variable structure 
!>       - td_lon1 is fine   grid longitude variable structure 
!>       - td_lat1 is fine   grid latitude  variable structure 
!>       - id_rho is array of refinment factor (default 1)
!>       - cd_point is Arakawa grid point (default 'T')
!>
!>    to know if grid is global:<br/>
!> @code
!>    ll_global=grid_is_global(td_lon, td_lat)
!> @endcode
!>       - td_lon is longitude variable structure
!>       - td_lat is latitude variable structure
!>   
!>    to know if grid contains north fold:<br/>
!> @code
!>    ll_north=grid_is_north_fold(td_lat)
!> @endcode
!>       - td_lat is latitude variable structure    
!>
!>    to get coarse grid indices of the closest point from one fine grid 
!> point:<br/>
!> @code
!>    il_index(:)=grid_get_closest(dd_lon0(:,:), dd_lat0(:,:), dd_lon1, dd_lat1
!>                                 [,dd_fill] [,cd_pos])
!> @endcode
!>       - il_index(:) is  coarse grid indices (/ i0, j0 /)
!>       - dd_lon0 is coarse grid array of longitude value (real(8))
!>       - dd_lat0 is coarse grid array of latitude  value (real(8))
!>       - dd_lon1 is fine grid longitude value (real(8))
!>       - dd_lat1 is fine grid latitude  value (real(8))
!>       - dd_fill
!>       - cd_pos
!>
!>    to compute distance between a point A and grid points:<br/>
!> @code
!>    il_dist(:,:)=grid_distance(dd_lon, dd_lat, dd_lonA, dd_latA)
!> @endcode
!>       - il_dist(:,:) is array of distance between point A and grid points
!>       - dd_lon is array of longitude value (real(8))
!>       - dd_lat is array of longitude value (real(8))
!>       - dd_lonA is longitude of point A (real(8))
!>       - dd_latA is latitude  of point A (real(8))
!>
!>    to get offset between fine grid and coarse grid:<br/> 
!> @code
!>    il_offset(:,:)=grid_get_fine_offset(td_coord0, 
!>                                        id_imin0, id_jmin0, id_imax0, id_jmax0,
!>                                        td_coord1
!>                                        [,id_rho] [,cd_point])
!> @endcode
!>    or
!> @code
!>    il_offset(:,:)=grid_get_fine_offset(dd_lon0, dd_lat0, 
!>                                        id_imin0, id_jmin0,id_imax0, id_jmax0,
!>                                        td_coord1 
!>                                        [,id_rho] [,cd_point])
!> @endcode
!>    or
!> @code
!>    il_offset(:,:)=grid_get_fine_offset(td_coord0, 
!>                                        id_imin0, id_jmin0, id_imax0, id_jmax0,
!>                                        dd_lon1, dd_lat1 
!>                                        [,id_rho] [,cd_point])
!> @endcode
!>    or
!> @code
!>    il_offset(:,:)=grid_get_fine_offset(dd_lon0, dd_lat0,
!>                                        id_imin0, id_jmin0, id_imax0, id_jmax0,
!>                                        dd_lon1, dd_lat1
!>                                        [,id_rho] [,cd_point])
!> @endcode
!>       - il_offset(:,:) is offset array 
!>    (/ (/ i_offset_left, i_offset_right /), (/ j_offset_lower, j_offset_upper /) /)
!>       - td_coord0 is coarse grid coordinate mpp structure 
!>       - dd_lon0  is coarse grid longitude array (real(8))
!>       - dd_lat0  is coarse grid latitude  array (real(8))
!>       - id_imin0 is coarse grid lower left  corner i-indice of fine grid 
!> domain
!>       - id_jmin0 is coarse grid lower left  corner j-indice of fine grid 
!> domain
!>       - id_imax0 is coarse grid upper right corner i-indice of fine grid 
!> domain
!>       - id_jmax0 is coarse grid upper right corner j-indice of fine grid 
!> domain
!>       - td_coord1 is fine grid coordinate mpp structure
!>       - dd_lon1  is fine   grid longitude array (real(8))
!>       - dd_lat1  is fine   grid latitude  array (real(8))
!>       - id_rho is array of refinment factor (default 1)
!>       - cd_point is Arakawa grid point (default 'T')
!>
!>    to check fine and coarse grid coincidence:<br/>
!> @code
!>    CALL grid_check_coincidence(td_coord0, td_coord1, 
!>                                id_imin0, id_imax0, id_jmin0, id_jmax0
!>                                ,id_rho)
!> @endcode
!>       - td_coord0 is coarse grid coordinate mpp structure
!>       - td_coord1 is fine   grid coordinate mpp structure
!>       - id_imin0  is coarse grid lower left  corner i-indice of fine grid 
!> domain
!>       - id_imax0  is coarse grid upper right corner i-indice of fine grid 
!> domain
!>       - id_jmin0  is coarse grid lower left  corner j-indice of fine grid 
!> domain
!>       - id_jmax0  is coarse grid upper right corner j-indice of fine grid 
!> domain 
!>       - id_rho    is array of refinement factor 
!>
!>    to add ghost cell at boundaries:<br/>
!> @code
!>    CALL grid_add_ghost(td_var, id_ghost)
!> @endcode
!>       - td_var is array of variable structure
!>       - id_ghost is 2D array of ghost cell factor
!>
!>    to delete ghost cell at boundaries:<br/>
!> @code
!>    CALL grid_del_ghost(td_var, id_ghost)
!> @endcode
!>       - td_var is array of variable structure
!>       - id_ghost is 2D array of ghost cell factor
!>
!>    to get ghost cell factor (use or not):<br/>
!> @code
!>    il_factor(:)= grid_get_ghost( td_var )
!> @endcode
!>    or
!> @code
!>    il_factor(:)= grid_get_ghost( td_mpp )
!> @endcode
!>       - il_factor(:) is  array of ghost cell factor (0 or 1)
!>       - td_var  is variable structure
!>       - td_mpp is mpp sturcture
!>
!>    to compute closed sea domain:<br/>
!> @code
!>    il_mask(:,:)=grid_split_domain(td_var, [id_level])
!> @endcode
!>       - il_mask(:,:) is domain mask
!>       - td_var is variable strucutre
!>       - id_level is level to be used [optional] 
!>
!>    to fill small closed sea with _FillValue:<br/>
!> @code
!>    CALL grid_fill_small_dom(td_var, id_mask, [id_minsize])
!> @endcode
!>       - td_var  is variable structure
!>       - id_mask is domain mask (from grid_split_domain)
!>       - id_minsize is minimum size of sea to be kept [optional]
!>
!> @author
!> J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date September, 2014
!> - add header
!> @date October, 2014
!> - use mpp file structure instead of file
!> @date February, 2015
!> - add function grid_fill_small_msk to fill small domain inside bigger one
!> @date February, 2016
!> - improve way to check coincidence (bug fix)
!> - manage grid cases for T,U,V or F point, with even or odd refinment (bug fix)
!> @date April, 2016
!> - add function to get closest grid point using coarse grid coordinates strucutre 
!> @date May, 2019
!> - define as module variable im_max_overlap
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE grid

   USE netcdf
   USE kind                            ! F90 kind parameter
   USE fct                             ! basic usefull function
   USE global                          ! global parameter
   USE phycst                          ! physical constant
   USE logger                          ! log file manager
   USE file                            ! file manager
   USE att                             ! attribute manager
   USE var                             ! variable manager
   USE dim                             ! dimension manager
   USE iom                             ! I/O manager
   USE mpp                             ! MPP manager
   USE dom                             ! domain manager
   USE iom_mpp                         ! MPP I/O manager
   USE iom_dom                         ! DOM I/O manager

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! type and variable
   INTEGER(i4), PARAMETER :: im_max_overlap = 5

   ! function and subroutine
   PUBLIC :: grid_get_info             !< get information about mpp global domain (pivot, perio, ew)
   PUBLIC :: grid_get_pivot            !< get NEMO pivot point index
   PUBLIC :: grid_get_perio            !< get NEMO periodicity index
   PUBLIC :: grid_get_ew_overlap       !< get East West overlap
   PUBLIC :: grid_check_dom            !< check domain validity 
   PUBLIC :: grid_get_coarse_index     !< get closest coarse grid indices of fine grid domain.
   PUBLIC :: grid_is_global            !< check if grid is global or not
   PUBLIC :: grid_is_north_fold
   PUBLIC :: grid_get_closest          !< return closest coarse grid point from another point
   PUBLIC :: grid_distance             !< compute grid distance to a point
   PUBLIC :: grid_get_fine_offset      !< get fine grid offset
   PUBLIC :: grid_check_coincidence    !< check fine and coarse grid coincidence
   PUBLIC :: grid_add_ghost            !< add ghost cell at boundaries.
   PUBLIC :: grid_del_ghost            !< delete ghost cell at boundaries.
   PUBLIC :: grid_get_ghost            !< return ghost cell factor
   PUBLIC :: grid_split_domain         !< compute closed sea domain 
   PUBLIC :: grid_fill_small_dom       !< fill small closed sea with fill value 
   PUBLIC :: grid_fill_small_msk       !< fill small domain inside bigger one 

                                     ! get closest coarse grid indices of fine grid domain
   PRIVATE :: grid__get_coarse_index_ff ! - using coarse and fine grid coordinates files
   PRIVATE :: grid__get_coarse_index_cf ! - using coarse grid array of lon,lat and fine grid coordinates files
   PRIVATE :: grid__get_coarse_index_fc ! - using coarse grid coordinates files, and fine grid array of lon,lat
   PRIVATE :: grid__get_coarse_index_cc ! - using coarse and fine grid array of lon,lat

                                     ! return closest coarse grid point from another point
   PRIVATE :: grid__get_closest_str    ! - using coarse grid coordinates strucutre
   PRIVATE :: grid__get_closest_arr    ! - using coarse grid array of lon,lat

                                     ! get offset between fine and coarse grid
   PRIVATE :: grid__get_fine_offset_ff ! - using coarse and fine grid coordinates files
   PRIVATE :: grid__get_fine_offset_cf ! - using coarse grid array of lon,lat and fine grid coordinates files
   PRIVATE :: grid__get_fine_offset_fc ! - using coarse grid coordinates files, and fine grid array of lon,lat
   PRIVATE :: grid__get_fine_offset_cc ! - using coarse and fine grid array of lon,lat

                                     ! get information about global domain (pivot, perio, ew)
   PRIVATE :: grid__get_info_mpp      ! - using mpp files structure
   PRIVATE :: grid__get_info_file     ! - using files structure

                                     ! get NEMO pivot point index
   PRIVATE :: grid__get_pivot_mpp      ! - using mpp files structure
   PRIVATE :: grid__get_pivot_file     ! - using files structure
   PRIVATE :: grid__get_pivot_var      ! - using variable structure
   PRIVATE :: grid__get_pivot_varT   ! compute NEMO pivot point index for variable on grid T 
   PRIVATE :: grid__get_pivot_varU   ! compute NEMO pivot point index for variable on grid U
   PRIVATE :: grid__get_pivot_varV   ! compute NEMO pivot point index for variable on grid V
   PRIVATE :: grid__get_pivot_varF   ! compute NEMO pivot point index for variable on grid F

                                     ! get NEMO periodicity index
   PRIVATE :: grid__get_perio_mpp      ! - using mpp files structure
   PRIVATE :: grid__get_perio_file     ! - using files structure
   PRIVATE :: grid__get_perio_var      ! - using variable structure

                                     ! get East West overlap
   PRIVATE :: grid__get_ew_overlap_mpp  ! - using mpp files structure
   PRIVATE :: grid__get_ew_overlap_file ! - using files structure
   PRIVATE :: grid__get_ew_overlap_var  ! - using longitude variable structure

                                    ! return ghost cell factor
   PRIVATE :: grid__get_ghost_mpp      ! - using mpp files structure
   PRIVATE :: grid__get_ghost_var      ! - using array of lon,lat
   PRIVATE :: grid__check_corner    ! check that fine grid is inside coarse grid
   PRIVATE :: grid__check_lat       ! check that fine grid latitude are inside coarse grid latitude
   
   INTERFACE  grid_get_info
      MODULE PROCEDURE grid__get_info_mpp
      MODULE PROCEDURE grid__get_info_file
   END INTERFACE grid_get_info

   INTERFACE  grid_get_pivot
      MODULE PROCEDURE grid__get_pivot_mpp
      MODULE PROCEDURE grid__get_pivot_file
      MODULE PROCEDURE grid__get_pivot_var
   END INTERFACE grid_get_pivot

   INTERFACE  grid_get_perio
      MODULE PROCEDURE grid__get_perio_mpp
      MODULE PROCEDURE grid__get_perio_file
      MODULE PROCEDURE grid__get_perio_var
   END INTERFACE grid_get_perio

   INTERFACE  grid_get_ew_overlap
      MODULE PROCEDURE grid__get_ew_overlap_mpp
      MODULE PROCEDURE grid__get_ew_overlap_file
      MODULE PROCEDURE grid__get_ew_overlap_var
   END INTERFACE grid_get_ew_overlap

   INTERFACE  grid_get_ghost
      MODULE PROCEDURE grid__get_ghost_var
      MODULE PROCEDURE grid__get_ghost_mpp
   END INTERFACE  grid_get_ghost

   INTERFACE  grid_get_closest
      MODULE PROCEDURE grid__get_closest_str
      MODULE PROCEDURE grid__get_closest_arr
   END INTERFACE  grid_get_closest

   INTERFACE  grid_get_coarse_index
      MODULE PROCEDURE grid__get_coarse_index_ff
      MODULE PROCEDURE grid__get_coarse_index_cf
      MODULE PROCEDURE grid__get_coarse_index_fc
      MODULE PROCEDURE grid__get_coarse_index_cc
   END INTERFACE grid_get_coarse_index

   INTERFACE  grid_get_fine_offset
      MODULE PROCEDURE grid__get_fine_offset_ff
      MODULE PROCEDURE grid__get_fine_offset_fc
      MODULE PROCEDURE grid__get_fine_offset_cf
      MODULE PROCEDURE grid__get_fine_offset_cc
   END INTERFACE grid_get_fine_offset   

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid__get_info_file(td_file)
   !-------------------------------------------------------------------
   !> @brief This subroutine get information about global domain, given file
   !> strucutre.
   !>
   !> @details
   !> open edge files then:
   !> - compute NEMO pivot point 
   !> - compute NEMO periodicity
   !> - compute East West overlap
   !>
   !> @note need all processor files to be there
   !> @author J.Paul
   !> @date October, 2014 - Initial Version
   !>
   !> @param[inout] td_file file structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file

      ! local variable
      INTEGER(i4) :: il_ew
      INTEGER(i4) :: il_pivot
      INTEGER(i4) :: il_perio
      INTEGER(i4) :: il_attid

      TYPE(TATT)  :: tl_att

      TYPE(TFILE) :: tl_file

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! initialise
      il_pivot=-1
      il_perio=-1
      il_ew   =-1 

      ! copy structure
      tl_file=file_copy(td_file)

      ! open file to be used
      CALL iom_open(tl_file)

      IF( td_file%i_perio >= 0 .AND. td_file%i_perio <= 6 )THEN
         il_perio=td_file%i_perio
      ELSE
         ! look for attribute in file
         il_attid=att_get_index(tl_file%t_att(:),'periodicity')
         IF( il_attid /= 0 )THEN
            il_perio=INT(tl_file%t_att(il_attid)%d_value(1),i4)
         ENDIF
      ENDIF

      IF( td_file%i_ew >= 0 )THEN
         il_ew=td_file%i_ew
      ELSE
         ! look for attribute in file
         il_attid=att_get_index(tl_file%t_att(:),'ew_overlap')
         IF( il_attid /= 0 )THEN
            il_ew=INT(tl_file%t_att(il_attid)%d_value(1),i4)
         ENDIF
      ENDIF

      SELECT CASE(il_perio)
      CASE(3,4)
         il_pivot=0
      CASE(5,6)
         il_pivot=1
      CASE(0,1,2)
         il_pivot=1
      END SELECT
      
      IF( il_pivot < 0 .OR. il_pivot > 1 )THEN
         ! get pivot
         il_pivot=grid_get_pivot(tl_file)
      ENDIF

      IF( il_perio < 0 .OR. il_perio > 6 )THEN
         ! get periodicity
         il_perio=grid_get_perio(tl_file, il_pivot)
      ENDIF

      IF( il_ew < 0 )THEN
         ! get periodicity
         il_ew=grid_get_ew_overlap(tl_file)
      ENDIF

      ! close
      CALL iom_close(tl_file)

      !save in file structure
      td_file%i_ew=il_ew
      td_file%i_pivot=il_pivot
      td_file%i_perio=il_perio

      ! save in variable of file structure
      tl_att=att_init("ew_overlap",il_ew)
      DO ji=1,td_file%i_nvar
         IF( td_file%t_var(ji)%t_dim(jp_I)%l_use )THEN
            CALL var_move_att(td_file%t_var(ji),tl_att)
         ENDIF
      ENDDO

      ! clean 
      CALL file_clean(tl_file)
      CALL att_clean(tl_att)

      IF( td_file%i_perio == -1 )THEN
         CALL logger_fatal("GRID GET INFO: can not read or compute "//&
         &  "domain periodicity from file "//TRIM(td_file%c_name)//"."//&
         &  " you have to inform periodicity in namelist.")
      ENDIF

   END SUBROUTINE grid__get_info_file
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid__get_info_mpp(td_mpp)
   !-------------------------------------------------------------------
   !> @brief This subroutine get information about global domain, given mpp
   !> structure.
   !>
   !> @details
   !> open edge files then:
   !> - compute NEMO pivot point 
   !> - compute NEMO periodicity
   !> - compute East West overlap
   !>
   !> @note need all processor files
   !> @author J.Paul
   !> @date October, 2014 - Initial Version
   !>
   !> @param[in] td_mpp mpp structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP) , INTENT(INOUT) :: td_mpp

      ! local variable
      INTEGER(i4) :: il_ew
      INTEGER(i4) :: il_pivot
      INTEGER(i4) :: il_perio
      INTEGER(i4) :: il_attid

      TYPE(TATT)  :: tl_att

      TYPE(TMPP)  :: tl_mpp

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------
      ! initialise
      il_pivot=-1
      il_perio=-1
      il_ew   =-1

      CALL logger_info("GRID GET INFO: look for "//TRIM(td_mpp%c_name))
      ! copy structure
      tl_mpp=mpp_copy(td_mpp)
      ! select edge files
      CALL mpp_get_contour(tl_mpp)
      ! open mpp file to be used
      CALL iom_mpp_open(tl_mpp)

      IF( td_mpp%i_perio >= 0 .AND. td_mpp%i_perio <= 6 )THEN
         il_perio=td_mpp%i_perio
      ELSE
         ! look for attribute in mpp files
         il_attid=att_get_index(tl_mpp%t_proc(1)%t_att(:),'periodicity')
         IF( il_attid /= 0 )THEN
            il_perio=INT(tl_mpp%t_proc(1)%t_att(il_attid)%d_value(1),i4)
         ENDIF
      ENDIF
 
      IF( td_mpp%i_ew >= 0 )THEN
         il_ew=td_mpp%i_ew
      ELSE
         ! look for attribute in mpp files
         il_attid=att_get_index(tl_mpp%t_proc(1)%t_att(:),'ew_overlap')
         IF( il_attid /= 0 )THEN
            il_ew=INT(tl_mpp%t_proc(1)%t_att(il_attid)%d_value(1),i4)
         ENDIF
      ENDIF

      CALL logger_info("GRID GET INFO: perio "//TRIM(fct_str(il_perio)))

      SELECT CASE(il_perio)
         CASE(3,4)
            il_pivot=1
         CASE(5,6)
            il_pivot=0
         CASE(0,1,2)
            il_pivot=1
      END SELECT

      IF( il_pivot < 0 .OR. il_pivot > 1 )THEN
         ! get pivot
         CALL logger_info("GRID GET INFO: look for pivot ")
         il_pivot=grid_get_pivot(tl_mpp)
      ENDIF

      IF( il_perio < 0 .OR. il_perio > 6 )THEN
         ! get periodicity
         CALL logger_info("GRID GET INFO: look for perio ")
         il_perio=grid_get_perio(tl_mpp, il_pivot)
      ENDIF

      IF( il_ew < 0 )THEN
         ! get periodicity
         CALL logger_info("GRID GET INFO: look for overlap ")
         il_ew=grid_get_ew_overlap(tl_mpp)
      ENDIF

      ! close
      CALL iom_mpp_close(tl_mpp)

      !save in mpp structure
      td_mpp%i_ew=il_ew
      td_mpp%i_pivot=il_pivot
      td_mpp%i_perio=il_perio

      ! save in variable of mpp structure
      IF( ASSOCIATED(td_mpp%t_proc) )THEN
         tl_att=att_init("ew_overlap",il_ew)
         DO jj=1,td_mpp%i_nproc
            DO ji=1,td_mpp%t_proc(jj)%i_nvar
               IF( td_mpp%t_proc(jj)%t_var(ji)%t_dim(jp_I)%l_use )THEN
                  CALL var_move_att(td_mpp%t_proc(jj)%t_var(ji),tl_att)
               ENDIF
            ENDDO
         ENDDO
      ENDIF

      ! clean 
      CALL mpp_clean(tl_mpp)
      CALL att_clean(tl_att)

      IF( td_mpp%i_perio == -1 )THEN
         CALL logger_fatal("GRID GET INFO: can not read or compute "//&
         &  "domain periodicity from mpp "//TRIM(td_mpp%c_name)//"."//&
         &  " you have to inform periodicity in namelist.")
      ENDIF

   END SUBROUTINE grid__get_info_mpp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_pivot_var(td_var) &
         & RESULT (if_pivot)
   !-------------------------------------------------------------------
   !> @brief 
   !> This function compute NEMO pivot point index of the input variable.
   !> - F-point : 0
   !> - T-point : 1
   !>
   !> @details
   !> check north points of latitude grid (indices jpj to jpj-3) depending on which grid point
   !> (T,F,U,V) variable is defined
   !> 
   !> @note variable must be at least 2D variable, and should not be coordinate
   !> variable (i.e lon, lat)
   !> 
   !> @warning 
   !> - do not work with ORCA2 grid (T-point)
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !> @date September, 2014
   !> - add dummy loop in case variable not over right point.
   !> @date October, 2014
   !> - work on variable structure instead of file structure
   !>
   !> @param[in] td_lat  latitude variable structure
   !> @param[in] td_var  variable structure
   !> @return pivot point index
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR), INTENT(IN) :: td_var

      ! function
      INTEGER(i4)            :: if_pivot

      ! local variable
      INTEGER(i4), DIMENSION(ip_maxdim)            :: il_dim

      REAL(dp)   , DIMENSION(:,:,:,:), ALLOCATABLE :: dl_value

      ! loop indices
      INTEGER(i4) :: jj
      !----------------------------------------------------------------
      ! intitalise
      if_pivot=-1

      IF( .NOT. ASSOCIATED(td_var%d_value) .OR. &
         &.NOT. ALL(td_var%t_dim(1:2)%l_use) )THEN
         CALL logger_error("GRID GET PIVOT: can not compute pivot point"//&
            &  " with variable "//TRIM(td_var%c_name)//"."//&
            &  " no value associated or missing dimension.")
      ELSE
         il_dim(:)=td_var%t_dim(:)%i_len

         ALLOCATE(dl_value(il_dim(1),4,1,1))
         ! extract value
         dl_value(:,:,:,:)=td_var%d_value( 1:il_dim(1),          &
            &                              il_dim(2)-3:il_dim(2),&
            &                              1:1,                  &
            &                              1:1 )

         SELECT CASE(TRIM(td_var%c_point))
         CASE('T')
            if_pivot=grid__get_pivot_varT(dl_value)
         CASE('U')
            if_pivot=grid__get_pivot_varU(dl_value)
         CASE('V')
            if_pivot=grid__get_pivot_varV(dl_value)
         CASE('F')
            if_pivot=grid__get_pivot_varF(dl_value)
         END SELECT

         ! dummy loop in case variable not over right point 
         ! (ex: nav_lon over U-point)
         IF( if_pivot == -1 )THEN
            
            ! no pivot point found
            CALL logger_warn("GRID GET PIVOT: something wrong "//&
               &  "when computing pivot point with variable "//&
               &  TRIM(td_var%c_name))

            DO jj=1,ip_npoint
               SELECT CASE(TRIM(cp_grid_point(jj)))
               CASE('T')
                  CALL logger_debug("GRID GET PIVOT: check variable on point T")
                  if_pivot=grid__get_pivot_varT(dl_value)
               CASE('U')
                  CALL logger_debug("GRID GET PIVOT: check variable on point U")
                  if_pivot=grid__get_pivot_varU(dl_value)
               CASE('V')
                  CALL logger_debug("GRID GET PIVOT: check variable on point V")
                  if_pivot=grid__get_pivot_varV(dl_value)
               CASE('F')
                  CALL logger_debug("GRID GET PIVOT: check variable on point F")
                  if_pivot=grid__get_pivot_varF(dl_value)
               END SELECT

               IF( if_pivot /= -1 )THEN
                  CALL logger_info("GRID GET PIVOT: variable "//&
                     &  TRIM(td_var%c_name)//" seems to be on grid point "//&
                     &  TRIM(cp_grid_point(jj)) )
                  EXIT
               ENDIF

            ENDDO
         ENDIF

         IF( if_pivot == -1 )THEN
            CALL logger_warn("GRID GET PIVOT: not able to found pivot point. "//&
               &             "Force to use pivot point T.")
            if_pivot = 1
         ENDIF

         ! clean
         DEALLOCATE(dl_value)

      ENDIF

   END FUNCTION grid__get_pivot_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_pivot_varT(dd_value) &
         & RESULT (if_pivot)
   !-------------------------------------------------------------------
   !> @brief 
   !> This function compute NEMO pivot point index for variable on grid T.
   !>
   !> @details
   !> - F-point : 0
   !> - T-point : 1
   !>
   !> @note array of value must be only the top border of the domain.
   !> 
   !> @author J.Paul
   !> @date October, 2014 - Initial version
   !>
   !> @param[in] dd_value array of value
   !> @return pivot point index
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp), DIMENSION(:,:,:,:), INTENT(IN) :: dd_value

      ! function
      INTEGER(i4)                              :: if_pivot

      ! local variable
      INTEGER(i4)                       :: il_midT
      INTEGER(i4)                       :: il_midF

      INTEGER(i4)                       :: it1
      INTEGER(i4)                       :: it2
      INTEGER(i4)                       :: jt1
      INTEGER(i4)                       :: jt2

      INTEGER(i4)                       :: if1
      INTEGER(i4)                       :: if2
      INTEGER(i4)                       :: jf1
      INTEGER(i4)                       :: jf2

      INTEGER(i4), DIMENSION(ip_maxdim) :: il_dim

      LOGICAL                           :: ll_check

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! intitalise
      if_pivot=-1

      il_dim(:)=SHAPE(dd_value(:,:,:,:))

      ! T-point pivot !case of ORCA2, ORCA025, ORCA12 grid
      jt1=4  ; jt2=2 
      il_midT=il_dim(1)/2+1

      ! F-point pivot !case of ORCA05 grid
      jf1=4 ; jf2=3
      il_midF=il_dim(1)/2

      ! check T-point pivot
      DO ji=2,il_midT
         ll_check=.TRUE.
         it1=ji
         it2=il_dim(1)-(ji-2)
         IF( dd_value(it1,jt1,1,1) /= dd_value(it2,jt2,1,1) )THEN
            ll_check=.FALSE.
            EXIT
         ENDIF
      ENDDO

      IF( ll_check )THEN
         CALL logger_info("GRID GET PIVOT: T-pivot")
         if_pivot=1
      ELSE

         ! check F-point pivot
         DO ji=1,il_midF
            ll_check=.TRUE.
            if1=ji
            if2=il_dim(1)-(ji-1)
            IF( dd_value(if1,jf1,1,1) /= dd_value(if2,jf2,1,1) )THEN
               ll_check=.FALSE.
               EXIT
            ENDIF
         ENDDO

         IF( ll_check )THEN
            CALL logger_info("GRID GET PIVOT: F-pivot")
            if_pivot=0
         ENDIF

      ENDIF

   END FUNCTION grid__get_pivot_varT
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_pivot_varU(dd_value) &
         & RESULT (if_pivot)
   !-------------------------------------------------------------------
   !> @brief 
   !> This function compute NEMO pivot point index for variable on grid U.
   !>
   !> @details
   !> - F-point : 0
   !> - T-point : 1
   !>
   !> @note array of value must be only the top border of the domain.
   !> 
   !> @author J.Paul
   !> @date October, 2014 - Initial version
   !>
   !> @param[in] dd_value array of value
   !> @return pivot point index
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp), DIMENSION(:,:,:,:), INTENT(IN) :: dd_value

      ! function
      INTEGER(i4)                              :: if_pivot

      ! local variable
      INTEGER(i4)                       :: il_midT
      INTEGER(i4)                       :: il_midF

      INTEGER(i4)                       :: it1
      INTEGER(i4)                       :: it2
      INTEGER(i4)                       :: jt1
      INTEGER(i4)                       :: jt2

      INTEGER(i4)                       :: if1
      INTEGER(i4)                       :: if2
      INTEGER(i4)                       :: jf1
      INTEGER(i4)                       :: jf2

      INTEGER(i4), DIMENSION(ip_maxdim) :: il_dim

      LOGICAL                           :: ll_check

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! intitalise
      if_pivot=-1

      il_dim(:)=SHAPE(dd_value(:,:,:,:))

      ! T-point pivot !case of ORCA2, ORCA025, ORCA12 grid
      jt1=4 ; jt2=2 
      il_midT=il_dim(1)/2+1

      ! F-point pivot !case of ORCA05 grid
      jf1=4 ; jf2=3
      il_midF=il_dim(1)/2

      ! check T-point pivot
      DO ji=1,il_midT
         ll_check=.TRUE.
         it1=ji
         it2=il_dim(1)-(ji-2)
         IF( dd_value(it1,jt1,1,1) /= dd_value(it2-1,jt2,1,1) )THEN
            ll_check=.FALSE.
            EXIT
         ENDIF
      ENDDO

      IF( ll_check )THEN
         CALL logger_info("GRID GET PIVOT: T-pivot")
         if_pivot=1
      ELSE

         ! check F-point pivot
         DO ji=1,il_midF
            ll_check=.TRUE.
            if1=ji
            if2=il_dim(1)-(ji-1)
            IF( dd_value(if1,jf1,1,1) /= dd_value(if2-1,jf2,1,1) )THEN
               ll_check=.FALSE.
               EXIT
            ENDIF
         ENDDO

         IF( ll_check )THEN
            CALL logger_info("GRID GET PIVOT: F-pivot")
            if_pivot=0
         ENDIF

      ENDIF

   END FUNCTION grid__get_pivot_varU
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_pivot_varV(dd_value) &
         & RESULT (if_pivot)
   !-------------------------------------------------------------------
   !> @brief 
   !> This function compute NEMO pivot point index for variable on grid V.
   !>
   !> @details
   !> - F-point : 0
   !> - T-point : 1
   !>
   !> @note array of value must be only the top border of the domain.
   !> 
   !> @author J.Paul
   !> @date October, 2014 - Initial version
   !>
   !> @param[in] dd_value array of value
   !> @return pivot point index
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp), DIMENSION(:,:,:,:), INTENT(IN) :: dd_value

      ! function
      INTEGER(i4)                              :: if_pivot

      ! local variable
      INTEGER(i4)                       :: il_midT
      INTEGER(i4)                       :: il_midF

      INTEGER(i4)                       :: it1
      INTEGER(i4)                       :: it2
      INTEGER(i4)                       :: jt1
      INTEGER(i4)                       :: jt2

      INTEGER(i4)                       :: if1
      INTEGER(i4)                       :: if2
      INTEGER(i4)                       :: jf1
      INTEGER(i4)                       :: jf2

      INTEGER(i4), DIMENSION(ip_maxdim) :: il_dim

      LOGICAL                           :: ll_check

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! intitalise
      if_pivot=-1

      il_dim(:)=SHAPE(dd_value(:,:,:,:))

      ! T-point pivot !case of ORCA2, ORCA025, ORCA12 grid
      jt1=4 ; jt2=2 
      il_midT=il_dim(1)/2+1

      ! F-point pivot !case of ORCA05 grid
      jf1=4 ; jf2=3
      il_midF=il_dim(1)/2

      ! check T-point pivot
      DO ji=2,il_midT
         ll_check=.TRUE.
         it1=ji
         it2=il_dim(1)-(ji-2)
         IF( dd_value(it1,jt1,1,1) /= dd_value(it2,jt2-1,1,1) )THEN
            ll_check=.FALSE.
            EXIT
         ENDIF
      ENDDO

      IF( ll_check )THEN
         CALL logger_info("GRID GET PIVOT: T-pivot")
         if_pivot=1
      ELSE

         ! check F-point pivot
         DO ji=1,il_midF
            ll_check=.TRUE.
            if1=ji
            if2=il_dim(1)-(ji-1)
            IF( dd_value(if1,jf1,1,1) /= dd_value(if2,jf2-1,1,1) )THEN
               ll_check=.FALSE.
               EXIT
            ENDIF
         ENDDO

         IF( ll_check )THEN
            CALL logger_info("GRID GET PIVOT: F-pivot")
            if_pivot=0
         ENDIF

      ENDIF

   END FUNCTION grid__get_pivot_varV
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_pivot_varF(dd_value) &
         & RESULT (if_pivot)
   !-------------------------------------------------------------------
   !> @brief 
   !> This function compute NEMO pivot point index for variable on grid F.
   !>
   !> @details
   !> - F-point : 0
   !> - T-point : 1
   !>
   !> @note array of value must be only the top border of the domain.
   !> 
   !> @author J.Paul
   !> @date October, 2014 - Initial version
   !>
   !> @param[in] dd_value array of value
   !> @return pivot point index
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp), DIMENSION(:,:,:,:), INTENT(IN) :: dd_value

      ! function
      INTEGER(i4)                              :: if_pivot

      ! local variable
      INTEGER(i4)                       :: il_midT
      INTEGER(i4)                       :: il_midF

      INTEGER(i4)                       :: it1
      INTEGER(i4)                       :: it2
      INTEGER(i4)                       :: jt1
      INTEGER(i4)                       :: jt2

      INTEGER(i4)                       :: if1
      INTEGER(i4)                       :: if2
      INTEGER(i4)                       :: jf1
      INTEGER(i4)                       :: jf2

      INTEGER(i4), DIMENSION(ip_maxdim) :: il_dim

      LOGICAL                           :: ll_check

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! intitalise
      if_pivot=-1

      il_dim(:)=SHAPE(dd_value(:,:,:,:))

      ! T-point pivot !case of ORCA2, ORCA025, ORCA12 grid
      jt1=4 ; jt2=2 
      il_midT=il_dim(1)/2+1

      ! F-point pivot !case of ORCA05 grid
      jf1=4 ; jf2=3
      il_midF=il_dim(1)/2

      ! check T-point pivot
      DO ji=1,il_midT
         ll_check=.TRUE.
         it1=ji
         it2=il_dim(1)-(ji-2)
         IF( dd_value(it1,jt1,1,1) /= dd_value(it2-1,jt2-1,1,1) )THEN
            ll_check=.FALSE.
            EXIT
         ENDIF
      ENDDO

      IF( ll_check )THEN
         CALL logger_info("GRID GET PIVOT: T-pivot")
         if_pivot=1
      ELSE

         ! check F-point pivot
         DO ji=1,il_midF
            ll_check=.TRUE.
            if1=ji
            if2=il_dim(1)-(ji-1)
            IF( dd_value(if1,jf1,1,1) /= dd_value(if2-1,jf2-1,1,1) )THEN
               ll_check=.FALSE.
               EXIT
            ENDIF
         ENDDO

         IF( ll_check )THEN
            CALL logger_info("GRID GET PIVOT: F-pivot")
            if_pivot=0
         ENDIF

      ENDIF

   END FUNCTION grid__get_pivot_varF
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_pivot_file(td_file) &
         & RESULT (if_pivot)
   !-------------------------------------------------------------------
   !> @brief 
   !> This function compute NEMO pivot point index from input file variable.
   !> - F-point : 0
   !> - T-point : 1
   !>
   !> @details
   !> check north points symmetry of a 2D variable (indices jpj to jpj-3), depending on which grid point
   !> (T,F,U,V) variable is defined.
   !> 
   !> @warning 
   !> - do not work with ORCA2 grid (T-point)
   !>
   !> @author J.Paul
   !> @date Ocotber, 2014 - Initial version
   !> @date August, 2017
   !> - if can't find latitude variable, assume there is a north fold
   !> - do not use latitude variable to get pivot (to avoid mistake with regular
   !> grid)
   !>
   !> @param[in] td_file file structure
   !> @return pivot point index
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(IN) :: td_file

      ! function
      INTEGER(i4)             :: if_pivot

      ! local variable
      INTEGER(i4)                       :: il_varid
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_dim

      LOGICAL                           :: ll_north

      TYPE(TVAR)                        :: tl_var
      TYPE(TVAR)                        :: tl_lat

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! intitalise
      if_pivot=-1

      ! look for north fold
      il_varid=var_get_index(td_file%t_var(:), 'latitude')
      IF( il_varid == 0 )THEN
         CALL logger_error("GRID GET PIVOT: no variable with name "//&
            &  "or standard name latitude in file structure "//&
            &  TRIM(td_file%c_name)//". Assume there is north fold and "//&
            &  "try to get pivot point")

         ll_north=.TRUE.
      ELSE      
         IF( ASSOCIATED(td_file%t_var(il_varid)%d_value) )THEN
            tl_lat=var_copy(td_file%t_var(il_varid))
         ELSE
            tl_lat=iom_read_var(td_file, 'latitude')
         ENDIF

         ll_north=grid_is_north_fold(tl_lat)
         ! clean
         CALL var_clean(tl_lat)
      ENDIF

      IF( ll_north )THEN      
         ! look for suitable variable
         DO ji=1,td_file%i_nvar
            IF( .NOT. ALL(td_file%t_var(ji)%t_dim(1:2)%l_use) ) CYCLE

            IF( ASSOCIATED(td_file%t_var(ji)%d_value) )THEN
               tl_var=var_copy(td_file%t_var(ji))
            ELSE
               il_dim(:)=td_file%t_var(ji)%t_dim(:)%i_len
               tl_var=iom_read_var(td_file, &
                  &                td_file%t_var(ji)%c_name, &
                  &                id_start=(/1,il_dim(2)-3,1,1/), &
                  &                id_count=(/il_dim(1),4,1,1/) )
            ENDIF
         ENDDO

         IF( ASSOCIATED(tl_var%d_value) )THEN

            if_pivot=grid_get_pivot(tl_var)

         ENDIF

         ! clean
         CALL var_clean(tl_var)
      ELSE
         CALL logger_warn("GRID GET PIVOT: no north fold. force to use T-PIVOT")
         if_pivot=1
      ENDIF

   END FUNCTION grid__get_pivot_file
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_pivot_mpp(td_mpp) &
         & RESULT (if_pivot)
   !-------------------------------------------------------------------
   !> @brief 
   !> This function compute NEMO pivot point index from input mpp variable.
   !> - F-point : 0
   !> - T-point : 1
   !>
   !> @details
   !> check north points symmetry of a 2D variable (indices jpj to jpj-3), depending 
   !> on which grid point (T,F,U,V) variable is defined.
   !> 
   !> @warning 
   !> - do not work with ORCA2 grid (T-point)
   !>
   !> @author J.Paul
   !> @date October, 2014 - Initial version
   !> @date August, 2017
   !> - if can't find latitude variable, assume there is a north fold
   !> - do not use latitude variable to get pivot (to avoid mistake with regular
   !> grid)
   !>
   !> @param[in] td_mpp   mpp file structure
   !> @return pivot point index
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP), INTENT(IN) :: td_mpp

      ! function
      INTEGER(i4)            :: if_pivot

      ! local variable
      INTEGER(i4)                       :: il_varid
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_dim

      LOGICAL                           :: ll_north

      TYPE(TVAR)                        :: tl_var
      TYPE(TVAR)                        :: tl_lat
 
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! intitalise
      if_pivot=-1

      ! look for north fold
      il_varid=var_get_index(td_mpp%t_proc(1)%t_var(:), 'latitude')
      IF( il_varid == 0 )THEN
         CALL logger_error("GRID GET PIVOT: no variable with name "//&
            &  "or standard name latitude in mpp structure "//&
            &  TRIM(td_mpp%c_name)//". Assume there is north fold and "//&
            &  "try to get pivot point")

         ll_north=.TRUE.
      ELSE
         IF( ASSOCIATED(td_mpp%t_proc(1)%t_var(il_varid)%d_value) )THEN
            ! 
            tl_lat=mpp_recombine_var(td_mpp, 'latitude')
         ELSE
            tl_lat=iom_mpp_read_var(td_mpp, 'latitude')
         ENDIF      

         ll_north=grid_is_north_fold(tl_lat)
         ! clean
         CALL var_clean(tl_lat)
      ENDIF

      IF( ll_north )THEN

         ! look for suitable variable
         DO ji=1,td_mpp%t_proc(1)%i_nvar
            IF(.NOT. ALL(td_mpp%t_proc(1)%t_var(ji)%t_dim(1:2)%l_use)) CYCLE

            IF( ASSOCIATED(td_mpp%t_proc(1)%t_var(ji)%d_value) )THEN
               CALL logger_debug("GRID GET PIVOT: mpp_recombine_var"//&
                  &              TRIM(td_mpp%t_proc(1)%t_var(ji)%c_name))
               tl_var=mpp_recombine_var(td_mpp, &
                  &                     TRIM(td_mpp%t_proc(1)%t_var(ji)%c_name))
            ELSE
               CALL logger_debug("GRID GET PIVOT: iom_mpp_read_var "//&
               &        TRIM(td_mpp%t_proc(1)%t_var(ji)%c_name))
               il_dim(:)=td_mpp%t_dim(:)%i_len

               ! read variable
               tl_var=iom_mpp_read_var(td_mpp, &
                  &                    td_mpp%t_proc(1)%t_var(ji)%c_name, &
                  &                    id_start=(/1,il_dim(2)-3,1,1/), &
                  &                    id_count=(/il_dim(1),4,1,1/) )
            ENDIF
            EXIT
         ENDDO

         IF( ASSOCIATED(tl_var%d_value) )THEN

            if_pivot=grid_get_pivot(tl_var)

         ELSE
            CALL logger_warn("GRID GET PIVOT: force to use T-PIVOT")
            if_pivot=1
         ENDIF

         ! clean
         CALL var_clean(tl_var)
      ELSE
         CALL logger_warn("GRID GET PIVOT: no north fold. force to use T-PIVOT")
         if_pivot=1
      ENDIF

   END FUNCTION grid__get_pivot_mpp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_perio_var(td_var, id_pivot) &
         & RESULT (if_perio)
   !-------------------------------------------------------------------
   !> @brief 
   !> This subroutine search NEMO periodicity index given variable structure and
   !> pivot point index.
   !> @details
   !> The variable must be on T point.
   !>
   !> 0: closed boundaries
   !> 1: cyclic east-west boundary
   !> 2: symmetric boundary condition across the equator
   !> 3: North fold boundary (with a T-point pivot)
   !> 4: North fold boundary (with a T-point pivot) and cyclic east-west boundary
   !> 5: North fold boundary (with a F-point pivot)
   !> 6: North fold boundary (with a F-point pivot) and cyclic east-west boundary
   !>
   !> @warning pivot point should have been computed before run this script. see grid_get_pivot.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !> @date October, 2014
   !> - work on variable structure instead of file structure
   !>
   !> @param[in] td_var   variable structure
   !> @param[in] id_pivot pivot point index
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR) , INTENT(IN) :: td_var
      INTEGER(i4), INTENT(IN) :: id_pivot

      ! function
      INTEGER(i4)             :: if_perio

      ! local variable
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_dim

      ! loop indices
      !----------------------------------------------------------------
      ! intitalise
      if_perio=-1

      IF( id_pivot < 0 .OR. id_pivot > 1 )THEN
         CALL logger_error("GRID GET PERIO: invalid pivot point index. "//&
         &  "you should use grid_get_pivot to compute it")
      ENDIF

      IF( .NOT. ASSOCIATED(td_var%d_value) .OR. &
      &   .NOT. ALL(td_var%t_dim(1:2)%l_use) )THEN
         CALL logger_error("GRID GET PERIO: can not compute periodicity"//&
         &  " with variable "//TRIM(td_var%c_name)//"."//&
         &  " no value associated or missing dimension.")
      ELSE

         il_dim(:)=td_var%t_dim(:)%i_len

         CALL logger_debug("GRID GET PERIO: use variable "//TRIM(td_var%c_name))
         CALL logger_debug("GRID GET PERIO: fill value "//TRIM(fct_str(td_var%d_fill)))
         CALL logger_debug("GRID GET PERIO: first value "//TRIM(fct_str(td_var%d_value(1,1,1,1))))

         IF(ALL(td_var%d_value(    1    ,    :    ,1,1)/=td_var%d_fill).AND.&
         &  ALL(td_var%d_value(il_dim(1),    :    ,1,1)/=td_var%d_fill).AND.&
         &  ALL(td_var%d_value(    :    ,    1    ,1,1)/=td_var%d_fill).AND.&
         &  ALL(td_var%d_value(    :    ,il_dim(2),1,1)/=td_var%d_fill))THEN
         ! no boundary closed
            CALL logger_error("GRID GET PERIO: can't determined periodicity. "//&
            &             "there is no boundary closed for variable "//&
            &              TRIM(td_var%c_name) )
            ! check pivot
            SELECT CASE(id_pivot)
               CASE(0)
                  ! F pivot
                  CALL logger_warn("GRID GET PERIO: assume domain is global")
                  if_perio=6
               CASE(1)
                  ! T pivot
                  CALL logger_warn("GRID GET PERIO: assume domain is global")
                  if_perio=4
            END SELECT
         ELSE
            if_perio=-1
            ! check periodicity
            IF(ANY(td_var%d_value(   1     ,:,1,1)/=td_var%d_fill).OR.&
            &  ANY(td_var%d_value(il_dim(1),:,1,1)/=td_var%d_fill))THEN
            ! East-West cyclic (1,4,6)

               IF( ANY(td_var%d_value(:, 1, 1, 1) /= td_var%d_fill) )THEN
                  ! South boundary not closed 

                  CALL logger_debug("GRID GET PERIO: East_West cyclic")
                  CALL logger_debug("GRID GET PERIO: South boundary not closed")
                  CALL logger_error("GRID GET PERIO: should have been an "//&
                  &              "impossible case")

               ELSE
                  ! South boundary closed (1,4,6)
                  CALL logger_info("GRID GET PERIO: South boundary closed")

                  IF(ANY(td_var%d_value(:,il_dim(2),1,1)/=td_var%d_fill) )THEN
                     ! North boundary not closed (4,6)
                     CALL logger_info("GRID GET PERIO: North boundary not closed")
                     ! check pivot
                     SELECT CASE(id_pivot)
                        CASE(0)
                           ! F pivot
                           if_perio=6
                        CASE(1)
                           ! T pivot
                           if_perio=4
                        CASE DEFAULT
                           CALL logger_error("GRID GET PERIO: invalid pivot ")
                     END SELECT
                  ELSE
                  ! North boundary closed
                     CALL logger_info("GRID GET PERIO: North boundary closed")
                     if_perio=1 ! North and South boundaries closed
                  ENDIF

               ENDIF

            ELSE
            ! East-West boundaries closed (0,2,3,5)
               CALL logger_info("GRID GET PERIO: East West boundaries closed")

               IF( ANY(td_var%d_value(:, 1, 1, 1) /= td_var%d_fill) )THEN
               ! South boundary not closed (2)
                  CALL logger_info("GRID GET PERIO: South boundary not closed")

                  IF(ANY(td_var%d_value(:,il_dim(2),1,1)/=td_var%d_fill))THEN
                     ! North boundary not closed
                     CALL logger_debug("GRID GET PERIO: East West boundaries "//&
                        &              "closed")
                     CALL logger_debug("GRID GET PERIO: South boundary not closed")
                     CALL logger_debug("GRID GET PERIO: North boundary not closed")
                     CALL logger_error("GRID GET PERIO: should have been "//&
                        &              "an impossible case")
                  ELSE
                     ! North boundary closed
                     if_perio=2   ! East-West and North boundaries closed 
                  ENDIF

               ELSE
               ! South boundary closed (0,3,5)
                  CALL logger_info("GRID GET PERIO: South boundary closed")

                  IF(ANY(td_var%d_value(:,il_dim(2),1,1)/=td_var%d_fill))THEN
                     ! North boundary not closed (3,5)
                     CALL logger_info("GRID GET PERIO: North boundary not closed")
                     ! check pivot
                     SELECT CASE(id_pivot)
                        CASE(0)
                           ! F pivot
                           if_perio=5
                        CASE(1)
                           ! T pivot
                           if_perio=3
                        CASE DEFAULT
                           CALL logger_error("GRID GET PERIO: invalid pivot")
                     END SELECT
                  ELSE
                  ! North boundary closed   
                     CALL logger_info("GRID GET PERIO: North boundary closed")
                     if_perio=0   ! all boundary closed
                  ENDIF

               ENDIF

            ENDIF

         ENDIF

      ENDIF

   END FUNCTION grid__get_perio_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_perio_file(td_file, id_pivot) &
         & RESULT (if_perio)
   !-------------------------------------------------------------------
   !> @brief 
   !> This subroutine search NEMO periodicity index given file structure, and
   !> optionaly pivot point index.
   !> @details
   !> The variable used must be on T point.
   !>
   !> 0: closed boundaries
   !> 1: cyclic east-west boundary
   !> 2: symmetric boundary condition across the equator
   !> 3: North fold boundary (with a F-point pivot)
   !> 4: North fold boundary (with a F-point pivot) and cyclic east-west boundary
   !> 5: North fold boundary (with a T-point pivot)
   !> 6: North fold boundary (with a T-point pivot) and cyclic east-west boundary
   !>
   !> @warning pivot point should have been computed before run this script. see grid_get_pivot.
   !>
   !> @author J.Paul
   !> @date October, 2014 - Initial version
   !> @date August, 2017
   !> - read only grid boundaries to handle huge file 
   !>
   !> @param[in] td_file   file structure
   !> @param[in] id_pivot  pivot point index
   !> @return NEMO periodicity index
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(IN) :: td_file
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_pivot

      ! function
      INTEGER(i4)             :: if_perio

      ! local variable
      INTEGER(i4)                       :: il_idx
      INTEGER(i4)                       :: il_pivot
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_dim

      TYPE(TVAR)                        :: tl_var
      TYPE(TVAR)                        :: tl_tmp

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      !initialise
      if_perio=-1

      IF(PRESENT(id_pivot) )THEN
         il_pivot=id_pivot
      ELSE
         il_pivot=grid_get_pivot(td_file)
      ENDIF

      IF( il_pivot < 0 .OR. il_pivot > 1 )THEN
         CALL logger_error("GRID GET PERIO: invalid pivot point index. "//&
         &  "you should use grid_get_pivot to compute it")
      ENDIF

      ! look for suitable variable
      il_idx=0
      DO ji=1,td_file%i_nvar
         IF( .NOT. ALL(td_file%t_var(ji)%t_dim(1:2)%l_use) ) CYCLE
         SELECT CASE(TRIM(fct_lower(td_file%t_var(ji)%c_stdname)) )
            CASE('longitude','latitude')
            CASE DEFAULT
               il_idx=ji
               EXIT
         END SELECT
      ENDDO

      IF( il_idx==0 )THEN
 
         CALL logger_error("GRID GET PERIO: no suitable variable to compute "//&
         &              " periodicity in file "//TRIM(td_file%c_name))

      ELSE

         DO ji=1,ip_maxdim
            IF( td_file%t_var(il_idx)%t_dim(ji)%l_use )THEN
               il_dim(ji)= td_file%t_var(il_idx)%t_dim(ji)%i_len
            ELSE
               il_dim(ji)=1
            ENDIF
         ENDDO

         ! read variable (full array)
         !tl_var=iom_read_var(td_file, &
         !&                   td_file%t_var(il_idx)%c_name, &
         !&                   id_start=(/1,1,1,1/), &
         !&                   id_count=(/il_dim(1),il_dim(2),1,1/) )

         ! read variable (only usefull part)
         tl_tmp=iom_read_var(td_file, &
            &                td_file%t_var(il_idx)%c_name, &
            &                id_start=(/1,1,1,1/), &
            &                id_count=(/il_dim(1),1,1,1/) )

         ! copy variable struct here, to get change done inside read_var too.
         tl_var=var_copy(tl_tmp,ld_value=.false.)
         ! force dimension to be full domain dimension 
         ! (instead of proc dimension)
         tl_var%t_dim(:)%i_len=il_dim(:)
         ALLOCATE(tl_var%d_value(il_dim(jp_I), &
            &                    il_dim(jp_J), &
            &                    il_dim(jp_K), &
            &                    il_dim(jp_L)))

         tl_var%d_value(:,1,1,1)=tl_tmp%d_value(:,1,1,1)
         ! clean 
         CALL var_clean(tl_tmp)
         
         ! read variable (only usefull part)
         tl_tmp=iom_read_var(td_file, &
            &                td_file%t_var(il_idx)%c_name, &
            &                id_start=(/1,il_dim(2),1,1/), &
            &                id_count=(/il_dim(1),1,1,1/) )

         tl_var%d_value(:,il_dim(2),1,1)=tl_tmp%d_value(:,1,1,1)
         ! clean 
         CALL var_clean(tl_tmp)

         ! read variable (only usefull part)
         tl_tmp=iom_read_var(td_file, &
            &                td_file%t_var(il_idx)%c_name, &
            &                id_start=(/1,1,1,1/), &
            &                id_count=(/1,il_dim(2),1,1/) )

         tl_var%d_value(1,:,1,1)=tl_tmp%d_value(1,:,1,1)
         ! clean 
         CALL var_clean(tl_tmp)

         ! read variable (only usefull part)
         tl_tmp=iom_read_var(td_file, &
            &                td_file%t_var(il_idx)%c_name, &
            &                id_start=(/il_dim(1),1,1,1/), &
            &                id_count=(/1,il_dim(2),1,1/) )

         tl_var%d_value(il_dim(1),:,1,1)=tl_tmp%d_value(1,:,1,1)
         ! clean 
         CALL var_clean(tl_tmp)

         if_perio=grid_get_perio(tl_var,il_pivot)
 
         ! clean 
         CALL var_clean(tl_var)

      ENDIF

   END FUNCTION grid__get_perio_file
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_perio_mpp(td_mpp, id_pivot) &
         & RESULT (if_perio)
   !-------------------------------------------------------------------
   !> @brief 
   !> This subroutine search NEMO periodicity given mpp structure and optionaly
   !> pivot point index.
   !> @details
   !> The variable used must be on T point.
   !>
   !> 0: closed boundaries
   !> 1: cyclic east-west boundary
   !> 2: symmetric boundary condition across the equator
   !> 3: North fold boundary (with a T-point pivot)
   !> 4: North fold boundary (with a T-point pivot) and cyclic east-west boundary
   !> 5: North fold boundary (with a F-point pivot)
   !> 6: North fold boundary (with a F-point pivot) and cyclic east-west boundary
   !>
   !> @warning pivot point should have been computed before run this script. see grid_get_pivot.
   !>
   !> @author J.Paul
   !> @date October, 2014 - Initial version
   !> @date August, 2017
   !> - read only grid boundaries to handle huge file 
   !> @date January, 2019
   !> - do not use silicalim, or silicamax to get pivot point
   !>
   !> @todo
   !> do not check silicalim, or silicamax
   !>
   !> @param[in] td_mpp   mpp file structure
   !> @param[in] id_pivot pivot point index
   !> @return NEMO periodicity index
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP) , INTENT(IN) :: td_mpp
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_pivot

      ! function
      INTEGER(i4)             :: if_perio

      ! local variable
      INTEGER(i4)                       :: il_idx
      INTEGER(i4)                       :: il_pivot
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_dim

      TYPE(TVAR)                        :: tl_var
      TYPE(TVAR)                        :: tl_tmp

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! initialise
      if_perio=-1

      IF(PRESENT(id_pivot) )THEN
         il_pivot=id_pivot
      ELSE
         il_pivot=grid_get_pivot(td_mpp)
      ENDIF

      IF( il_pivot < 0 .OR. il_pivot > 1 )THEN
         CALL logger_error("GRID GET PERIO: invalid pivot point index. "//&
            &  "you should use grid_get_pivot to compute it")
      ENDIF

      ! look for suitable variable
      il_idx=0
      DO ji=1,td_mpp%t_proc(1)%i_nvar
         IF( .NOT. ALL(td_mpp%t_proc(1)%t_var(ji)%t_dim(1:2)%l_use) ) CYCLE
         SELECT CASE(TRIM(fct_lower(td_mpp%t_proc(1)%t_var(ji)%c_stdname)) )
            CASE('longitude','latitude')
            CASE DEFAULT
               SELECT CASE(TRIM(fct_lower(td_mpp%t_proc(1)%t_var(ji)%c_name)))
                  CASE('silicalim','silicamax')
                  CASE DEFAULT
                     il_idx=ji
                     EXIT
               END SELECT
         END SELECT
      ENDDO

      IF( il_idx==0 )THEN
 
         CALL logger_error("GRID GET PERIO: no suitable variable to compute "//&
         &              " periodicity in file "//TRIM(td_mpp%c_name))

      ELSE
 
         ! full domain dimension
         DO ji=1,ip_maxdim
            IF( td_mpp%t_proc(1)%t_var(il_idx)%t_dim(ji)%l_use )THEN
               il_dim(ji)=td_mpp%t_dim(ji)%i_len
            ELSE
               il_dim(ji)=1
            ENDIF
         ENDDO

         ! read variable (full array)
         !tl_var=iom_mpp_read_var(td_mpp, td_mpp%t_proc(1)%t_var(il_idx)%c_name)

         ! read variable (only usefull part)
         tl_tmp=iom_mpp_read_var(td_mpp, &
            &                    td_mpp%t_proc(1)%t_var(il_idx)%c_name, &
            &                    id_start=(/1,1,1,1/), &
            &                    id_count=(/il_dim(1),1,1,1/) )

         ! copy variable struct here, to get change done inside read_var too.
         tl_var=var_copy(tl_tmp,ld_value=.false.)
         ! force dimension to be full domain dimension 
         ! (instead of proc dimension)
         tl_var%t_dim(:)%i_len=il_dim(:)
         ALLOCATE(tl_var%d_value(il_dim(jp_I), &
            &                    il_dim(jp_J), &
            &                    il_dim(jp_K), &
            &                    il_dim(jp_L)))

         tl_var%d_value(:,1,1,1)=tl_tmp%d_value(:,1,1,1)
         ! clean 
         CALL var_clean(tl_tmp)
 
         ! read variable (only usefull part)
         tl_tmp=iom_mpp_read_var(td_mpp, &
            &                    td_mpp%t_proc(1)%t_var(il_idx)%c_name, &
            &                    id_start=(/1,il_dim(2),1,1/), &
            &                    id_count=(/il_dim(1),1,1,1/) )

         tl_var%d_value(:,il_dim(2),1,1)=tl_tmp%d_value(:,1,1,1)
         ! clean 
         CALL var_clean(tl_tmp)

         ! read variable (only usefull part)
         tl_tmp=iom_mpp_read_var(td_mpp, &
            &                    td_mpp%t_proc(1)%t_var(il_idx)%c_name, &
            &                    id_start=(/1,1,1,1/), &
            &                    id_count=(/1,il_dim(2),1,1/) )

         tl_var%d_value(1,:,1,1)=tl_tmp%d_value(1,:,1,1)
         ! clean 
         CALL var_clean(tl_tmp)

         ! read variable (only usefull part)
         tl_tmp=iom_mpp_read_var(td_mpp, &
            &                    td_mpp%t_proc(1)%t_var(il_idx)%c_name, &
            &                    id_start=(/il_dim(1),1,1,1/), &
            &                    id_count=(/1,il_dim(2),1,1/) )

         tl_var%d_value(il_dim(1),:,1,1)=tl_tmp%d_value(1,:,1,1)
         ! clean 
         CALL var_clean(tl_tmp)

         if_perio=grid_get_perio(tl_var, il_pivot)

         ! clean 
         CALL var_clean(tl_var)
      ENDIF

   END FUNCTION grid__get_perio_mpp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_ew_overlap_var(td_var) &
         & RESULT (if_overlap)
   !-------------------------------------------------------------------
   !> @brief This function get East-West overlap.
   !>
   !> @details
   !> If no East-West wrap return -1, 
   !> else return the size of the ovarlap band.
   !> East-West overlap is computed comparing longitude value of the  
   !> South part of the domain, to avoid  north fold boundary.
   !>
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date October, 2014
   !> - work on mpp file structure instead of file structure
   !> @date October, 2016
   !> - check longitude as longname
   !>
   !> @param[in] td_lon longitude variable structure 
   !> @return East West overlap
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR), INTENT(INOUT) :: td_var

      ! function
      INTEGER(i4)               :: if_overlap

      ! local variable
      REAL(dp), DIMENSION(:,:), ALLOCATABLE :: dl_value
      REAL(dp), DIMENSION(:)  , ALLOCATABLE :: dl_vare
      REAL(dp), DIMENSION(:)  , ALLOCATABLE :: dl_varw

      REAL(dp)    :: dl_delta
      REAL(dp)    :: dl_varmax
      REAL(dp)    :: dl_varmin

      INTEGER(i4) :: il_east
      INTEGER(i4) :: il_west
      INTEGER(i4) :: il_jmin
      INTEGER(i4) :: il_jmax

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      ! initialise
      if_overlap=-1

      IF( ASSOCIATED(td_var%d_value) )THEN

         IF( td_var%t_dim(1)%i_len > 1 )THEN
            il_west=1
            il_east=td_var%t_dim(1)%i_len

            ALLOCATE( dl_value(td_var%t_dim(1)%i_len, &
               &               td_var%t_dim(2)%i_len) )

            dl_value(:,:)=td_var%d_value(:,:,1,1)

            ! we do not use jmax as dimension length due to north fold boundary
            IF( td_var%t_dim(2)%i_len > 1 )THEN
               il_jmin=1+ip_ghost
               il_jmax=(td_var%t_dim(2)%i_len-ip_ghost)/2
            ELSE
               il_jmin=1
               il_jmax=1
            ENDIF

            ALLOCATE( dl_vare(il_jmax-il_jmin+1) )
            ALLOCATE( dl_varw(il_jmax-il_jmin+1) )

            dl_vare(:)=dl_value(il_east,il_jmin:il_jmax)
            dl_varw(:)=dl_value(il_west,il_jmin:il_jmax)

            IF( .NOT.( ALL(dl_vare(:)==td_var%d_fill) .AND. &
            &          ALL(dl_varw(:)==td_var%d_fill) ) )THEN

               IF( TRIM(td_var%c_stdname) == 'longitude' .OR. &
                 & SCAN( TRIM(td_var%c_longname), 'longitude') == 0 )THEN
                  WHERE( dl_value(:,:) > 180._dp .AND. &
                     &   dl_value(:,:) /= td_var%d_fill ) 
                     dl_value(:,:)=360.-dl_value(:,:)
                  END WHERE

                  dl_varmax=MAXVAL(dl_value(:,il_jmin:il_jmax))
                  dl_varmin=MINVAL(dl_value(:,il_jmin:il_jmax))

                  dl_delta=(dl_varmax-dl_varmin)/td_var%t_dim(1)%i_len

                  IF( ALL(ABS(dl_vare(:)) - ABS(dl_varw(:)) == dl_delta) )THEN
                     if_overlap=0
                  ENDIF
               ENDIF

               IF( if_overlap == -1 )THEN
                  DO ji=0,im_max_overlap

                     IF( il_east-ji == il_west )THEN
                        ! case of small domain
                        EXIT
                     ELSE
                        dl_vare(:)=dl_value(il_east-ji,il_jmin:il_jmax)

                        IF( ALL( dl_varw(:) == dl_vare(:) ) )THEN
                           if_overlap=ji+1
                           EXIT
                        ENDIF
                     ENDIF

                  ENDDO
               ENDIF
            ENDIF

         ENDIF
      ELSE
         CALL logger_error("GRID GET EW OVERLAP: input variable standard name"//&
         &  TRIM(td_var%c_stdname)//" can not be used to compute East West "//&
         &  "overalp. no value associated. ")
      ENDIF

   END FUNCTION grid__get_ew_overlap_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_ew_overlap_file(td_file) &
         & RESULT (if_overlap)
   !-------------------------------------------------------------------
   !> @brief This function get East-West overlap.
   !>
   !> @details
   !> If no East-West wrap return -1, 
   !> else return the size of the ovarlap band.
   !> East-West overlap is computed comparing longitude value of the  
   !> South part of the domain, to avoid  north fold boundary.
   !>
   !> @author J.Paul
   !> @date October, 2014 - Initial Version
   !> @date October, 2016
   !> - check varid for longitude_T
   !> @date August, 2017
   !> - read only grid boundaries to handle huge file 
   !>
   !> @param[in] td_file file structure 
   !> @return East West overlap
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE), INTENT(INOUT) :: td_file

      ! function
      INTEGER(i4)                :: if_overlap

      ! local variable
      INTEGER(i4)                       :: il_idx
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_dim

      TYPE(TVAR)                        :: tl_var
      TYPE(TVAR)                        :: tl_tmp

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: i1
      INTEGER(i4) :: i2
      INTEGER(i4) :: j1
      INTEGER(i4) :: j2
      INTEGER(i4) :: ic
      INTEGER(i4) :: jc
      !----------------------------------------------------------------

      ! look for suitable variable
      il_idx=var_get_index(td_file%t_var(:), 'longitude', 'longitude_T')
      IF( il_idx == 0 )THEN
         DO jj=1,td_file%i_nvar
            IF( ALL(td_file%t_var(jj)%t_dim(1:2)%l_use) )THEN
               il_idx=jj
               EXIT
            ENDIF
         ENDDO
      ENDIF

      IF( il_idx==0 )THEN
 
         CALL logger_error("GRID GET EW OVERLAP: no suitable variable to compute "//&
         &              " east west overlap in file "//TRIM(td_file%c_name))

      ELSE
         ! read variable (full array)
         !tl_var=iom_read_var(td_file, td_file%t_var(il_idx)%c_name)

         ! full domain dimension
         DO ji=1,ip_maxdim
            IF( td_file%t_var(il_idx)%t_dim(ji)%l_use )THEN
               il_dim(ji)= td_file%t_var(il_idx)%t_dim(ji)%i_len
            ELSE
               il_dim(ji)=1
            ENDIF
         ENDDO

         ! read variable (only usefull part)
         i1=1                ; j1=1
         i2=im_max_overlap   ; j2=il_dim(jp_J)
         ic=i2-i1+1          ; jc=j2-j1+1
         tl_tmp=iom_read_var(td_file, &
            &                td_file%t_var(il_idx)%c_name, &
            &                id_start=(/i1,j1,1,1/), &
            &                id_count=(/ic,jc,1,1/) )

         ! copy variable struct here, to get change done inside read_var too.
         tl_var=var_copy(tl_tmp,ld_value=.false.)
         ! force dimension to be full domain dimension 
         ! (instead of proc dimension)
         tl_var%t_dim(:)%i_len=il_dim(:)
         ALLOCATE(tl_var%d_value(il_dim(jp_I), &
            &                    il_dim(jp_J), &
            &                    il_dim(jp_K), &
            &                    il_dim(jp_L)))
         ! init array
         tl_var%d_value(:,:,:,:)=tl_var%d_fill

         tl_var%d_value(i1:i2,:,1,1)=tl_tmp%d_value(:,:,1,1)
         ! clean 
         CALL var_clean(tl_tmp)
         
         ! read variable (only usefull part)
         i1=il_dim(jp_I)-im_max_overlap ; j1=1
         i2=il_dim(jp_I)                ; j2=il_dim(jp_J)
         ic=i2-i1+1                     ; jc=j2-j1+1
         tl_tmp=iom_read_var(td_file, &
            &                td_file%t_var(il_idx)%c_name, &
            &                id_start=(/i1,j1,1,1/), &
            &                id_count=(/ic,jc,1,1/) )

         tl_var%d_value(i1:i2,:,1,1)=tl_tmp%d_value(:,:,1,1)
         ! clean 
         CALL var_clean(tl_tmp)

         if_overlap=grid_get_ew_overlap(tl_var)

         ! clean
         CALL var_clean(tl_var)

      ENDIF

   END FUNCTION grid__get_ew_overlap_file
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_ew_overlap_mpp(td_mpp) &
         & RESULT (if_overlap)
   !-------------------------------------------------------------------
   !> @brief This function get East-West overlap.
   !>
   !> @details
   !> If no East-West wrap return -1, 
   !> else return the size of the ovarlap band.
   !> East-West overlap is computed comparing longitude value of the  
   !> South part of the domain, to avoid  north fold boundary.
   !>
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date October, 2014
   !> - work on mpp file structure instead of file structure
   !> @date October, 2016
   !> - check varid for longitude_T
   !> @date August, 2017
   !> - read only grid boundaries to handle huge file 
   !>
   !> @param[in] td_mpp mpp structure 
   !> @return East West overlap
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP), INTENT(INOUT) :: td_mpp

      ! function
      INTEGER(i4)               :: if_overlap

      ! local variable
      INTEGER(i4)                       :: il_idx
      INTEGER(i4)                       :: il_ew
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_dim

      TYPE(TVAR)                        :: tl_var
      TYPE(TVAR)                        :: tl_tmp

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: i1
      INTEGER(i4) :: i2
      INTEGER(i4) :: j1
      INTEGER(i4) :: j2
      INTEGER(i4) :: ic
      INTEGER(i4) :: jc
      !----------------------------------------------------------------

      ! initialise
      if_overlap=td_mpp%i_ew

      il_idx=var_get_index(td_mpp%t_proc(1)%t_var(:), 'longitude', 'longitude_T')
      IF( il_idx == 0 )THEN
         DO jj=1,td_mpp%t_proc(1)%i_nvar
            IF( ALL(td_mpp%t_proc(1)%t_var(jj)%t_dim(1:2)%l_use) )THEN
               il_idx=jj
               EXIT
            ENDIF
         ENDDO
      ENDIF

      IF( il_idx==0 )THEN
 
         CALL logger_error("GRID GET EW OVERLAP: no suitable variable to compute "//&
         &              " east west overlap in mppfile "//TRIM(td_mpp%c_name))

      ELSE
         ! read variable (full array)
         !tl_var=iom_mpp_read_var(td_mpp, il_idx)

         ! full domain dimension
         DO ji=1,ip_maxdim
            IF( td_mpp%t_proc(1)%t_var(il_idx)%t_dim(ji)%l_use )THEN
               !il_dim(ji)=td_mpp%t_proc(1)%t_var(il_idx)%t_dim(ji)%i_len
               il_dim(ji)=td_mpp%t_dim(ji)%i_len
            ELSE
               il_dim(ji)=1
            ENDIF
         ENDDO

         ! read variable (only usefull part)
         i1=1                ; j1=1
         i2=im_max_overlap   ; j2=il_dim(jp_J)
         ic=i2-i1+1          ; jc=j2-j1+1
         tl_tmp=iom_mpp_read_var(td_mpp, &
         &                       td_mpp%t_proc(1)%t_var(il_idx)%c_name, &
         &                       id_start=(/i1,j1,1,1/), &
         &                       id_count=(/ic,jc,1,1/) )

         ! copy variable struct here, to get change done inside read_var too.
         tl_var=var_copy(tl_tmp,ld_value=.false.)
         ! force dimension to be full domain dimension 
         ! (instead of proc dimension)
         tl_var%t_dim(:)%i_len=il_dim(:)
         ALLOCATE(tl_var%d_value(il_dim(jp_I), &
            &                    il_dim(jp_J), &
            &                    il_dim(jp_K), &
            &                    il_dim(jp_L)))
         ! init array
         tl_var%d_value(:,:,:,:)=tl_var%d_fill

         tl_var%d_value(i1:i2,:,1,1)=tl_tmp%d_value(:,:,1,1)
         ! clean 
         CALL var_clean(tl_tmp)
         
         ! read variable (only usefull part)
         i1=il_dim(jp_I)-im_max_overlap ; j1=1
         i2=il_dim(jp_I)                ; j2=il_dim(jp_J)
         ic=i2-i1+1                     ; jc=j2-j1+1
         tl_tmp=iom_mpp_read_var(td_mpp, &
         &                       td_mpp%t_proc(1)%t_var(il_idx)%c_name, &
         &                       id_start=(/i1,j1,1,1/), &
         &                       id_count=(/ic,jc,1,1/) )

         tl_var%d_value(i1:i2,:,1,1)=tl_tmp%d_value(:,:,1,1)
         ! clean 
         CALL var_clean(tl_tmp)

         il_ew=grid_get_ew_overlap(tl_var)
         IF( il_ew >= 0 )THEN
            if_overlap=il_ew
         ENDIF

         ! clean
         CALL var_clean(tl_var)

      ENDIF

   END FUNCTION grid__get_ew_overlap_mpp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid_is_north_fold(td_lat) &
         & RESULT (lf_north)
   !-------------------------------------------------------------------
   !> @brief This subroutine check if there is north fold.
   !>
   !> @details
   !> check if maximum latitude greater than 88°N 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_lat latitude variable structure 
   !> @return true if there is north fold
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR), INTENT(IN) :: td_lat

      ! function
      LOGICAL                :: lf_north

      ! local variable
      ! loop indices
      !----------------------------------------------------------------
   
      ! init
      lf_north=.FALSE.

      IF( .NOT. ASSOCIATED(td_lat%d_value) )THEN
         CALL logger_error("GRID IS NORTH FOLD: "//&
         &                 " no value associated to latitude")
      ELSE      
         IF( MAXVAL(td_lat%d_value(:,:,:,:), &
         &          td_lat%d_value(:,:,:,:)/= td_lat%d_fill) >= 88.0 )THEN

            lf_north=.TRUE.
            
         ENDIF
      ENDIF

   END FUNCTION grid_is_north_fold
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_check_dom(td_coord, id_imin, id_imax, id_jmin, id_jmax)
   !-------------------------------------------------------------------
   !> @brief This subroutine check domain validity.
   !>
   !> @details
   !> If maximum latitude greater than 88°N, program will stop. 
   !> @note Not able to manage north fold for now.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date October, 2014
   !> - work on mpp file structure instead of file structure
   !>
   !> @param[in] cd_coord  coordinate file 
   !> @param[in] id_imin   i-direction lower left  point indice  
   !> @param[in] id_imax   i-direction upper right point indice 
   !> @param[in] id_jmin   j-direction lower left  point indice 
   !> @param[in] id_jmax   j-direction upper right point indice 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP) , INTENT(IN) :: td_coord
      INTEGER(i4), INTENT(IN) :: id_imin
      INTEGER(i4), INTENT(IN) :: id_imax
      INTEGER(i4), INTENT(IN) :: id_jmin
      INTEGER(i4), INTENT(IN) :: id_jmax

      ! local variable
      CHARACTER(LEN=lc) :: cl_name

      INTEGER(i4)       :: il_ind

      TYPE(TVAR)        :: tl_var

      TYPE(TMPP)        :: tl_coord

      TYPE(TDOM)        :: tl_dom
      ! loop indices
      !----------------------------------------------------------------

      IF( id_jmin > id_jmax .OR. id_jmax == 0 )THEN

         CALL logger_fatal("GRID CHECK DOM: invalid domain. "//&
         &  "can not create configuration with north pole.")

      ELSE

            IF( id_imin == id_imax .AND. td_coord%i_ew < 0 )THEN
               CALL logger_fatal("GRID CHECK DOM: invalid domain."//&
               &  " can not create east-west cyclic fine grid"//&
               &  " inside closed coarse grid")
            ENDIF

            ! copy structure
            tl_coord=mpp_copy(td_coord)

            ! compute domain
            tl_dom=dom_init( tl_coord,        &
            &                id_imin, id_imax,&
            &                id_jmin, id_jmax )
            
            ! open mpp files to be used
            CALL iom_dom_open(tl_coord, tl_dom)

            ! read variable value on domain
            WRITE(cl_name,*) 'latitude'
            il_ind=var_get_id(tl_coord%t_proc(1)%t_var(:), cl_name)
            IF( il_ind == 0 )THEN
               CALL logger_warn("GRID CHECK DOM: no variable "//&
               &  TRIM(cl_name)//" in file "//TRIM(tl_coord%c_name)//". &
               &  try to use latitude_T.")
               WRITE(cl_name,*) 'latitude_T'
            ENDIF
            tl_var=iom_dom_read_var(tl_coord,TRIM(cl_name),tl_dom)

            ! close mpp files
            CALL iom_dom_close(tl_coord)

            ! clean structure
            CALL mpp_clean(tl_coord)

            IF( MAXVAL(tl_var%d_value(:,:,:,:), &
            &          tl_var%d_value(:,:,:,:)/= tl_var%d_fill) >= 88.0 )THEN
               
               CALL logger_debug("GRID CHECK DOM: max latitude "//&
               &  TRIM(fct_str(MAXVAL(tl_var%d_value(:,:,:,:)))) )
               CALL logger_fatal("GRID CHECK DOM: invalid domain. "//&
               &  "can not create configuration too close from north pole.")

            ENDIF

            ! clean
            CALL dom_clean(tl_dom)
            CALL var_clean(tl_var)

      ENDIF

   END SUBROUTINE grid_check_dom
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_coarse_index_ff(td_coord0, td_coord1, &
         &                            id_rho, cd_point) &
         & RESULT (if_idx)
   !-------------------------------------------------------------------
   !> @brief This function get closest coarse grid indices of fine grid domain.
   !>
   !> @details
   !> it use coarse and fine grid coordinates files.
   !> optionally, you could specify the array of refinment factor (default 1.)
   !> optionally, you could specify on which Arakawa grid point you want to
   !> work (default 'T')
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014
   !> - use grid point to read coordinates variable. 
   !> @date October, 2014
   !> - work on mpp file structure instead of file structure
   !> @date February, 2015
   !> - use longitude or latitude as standard name, if can not find 
   !> longitude_T, latitude_T...
   !>
   !> @param[in] td_coord0 coarse grid coordinate mpp structure
   !> @param[in] td_coord1 fine grid coordinate mpp structure
   !> @param[in] id_rho    array of refinment factor (default 1.)
   !> @param[in] cd_point  Arakawa grid point (default 'T').
   !> @return coarse grid indices(/(/imin0, imax0/), (/jmin0, jmax0/)/)
   !>                                     
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP)                    , INTENT(IN) :: td_coord0
      TYPE(TMPP)                    , INTENT(IN) :: td_coord1
      INTEGER(i4)     , DIMENSION(:), INTENT(IN), OPTIONAL :: id_rho
      CHARACTER(LEN=*)              , INTENT(IN), OPTIONAL :: cd_point

      ! function
      INTEGER(i4)     , DIMENSION(2,2)           :: if_idx

      ! local variable
      CHARACTER(LEN= 1)                        :: cl_point
      CHARACTER(LEN=lc)                        :: cl_name

      INTEGER(i4)                              :: il_imin0
      INTEGER(i4)                              :: il_imax0
      INTEGER(i4)                              :: il_jmin0
      INTEGER(i4)                              :: il_jmax0
      INTEGER(i4)                              :: il_ind

      INTEGER(i4), DIMENSION(2,2)              :: il_xghost0
      INTEGER(i4), DIMENSION(2,2)              :: il_xghost1

      INTEGER(i4), DIMENSION(:)  , ALLOCATABLE :: il_rho

      TYPE(TVAR)                               :: tl_lon0
      TYPE(TVAR)                               :: tl_lat0
      TYPE(TVAR)                               :: tl_lon1
      TYPE(TVAR)                               :: tl_lat1

      TYPE(TMPP)                               :: tl_coord0
      TYPE(TMPP)                               :: tl_coord1

      ! loop indices
      !----------------------------------------------------------------

      ! init
      if_idx(:,:)=0

      ALLOCATE(il_rho(ip_maxdim))
      il_rho(:)=1
      IF( PRESENT(id_rho) ) il_rho(:)=id_rho(:)

      cl_point='T'
      IF( PRESENT(cd_point) ) cl_point=TRIM(fct_upper(cd_point))

      ! copy structure
      tl_coord0=mpp_copy(td_coord0)
      tl_coord1=mpp_copy(td_coord1)

      IF( .NOT. ASSOCIATED(tl_coord0%t_proc) .OR. &
      &   .NOT. ASSOCIATED(tl_coord1%t_proc) )THEN
         CALL logger_error("GRID GET COARSE INDEX: can not get coarse "//&
         &  "grid indices. decompsition of mpp file "//TRIM(tl_coord0%c_name)//&
         &  " and/or "//TRIM(tl_coord1%c_name)//" not defined." )
      ELSE
         ! Coarse grid
         ! get ghost cell factor on coarse grid
         il_xghost0(:,:)=grid_get_ghost( tl_coord0 )

         ! open mpp files
         CALL iom_mpp_open(tl_coord0)

         ! read coarse longitue and latitude
         WRITE(cl_name,*) 'longitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord0%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET COARSE INDEX: no variable "//&
            &  TRIM(cl_name)//" in file "//TRIM(tl_coord0%c_name)//". &
            &  try to use longitude.")
            WRITE(cl_name,*) 'longitude'
         ENDIF
         tl_lon0=iom_mpp_read_var(tl_coord0, TRIM(cl_name))

         WRITE(cl_name,*) 'latitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord0%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET COARSE INDEX: no variable "//&
            &  TRIM(cl_name)//" in file "//TRIM(tl_coord0%c_name)//". &
            &  try to use latitude.")
            WRITE(cl_name,*) 'latitude'
         ENDIF
         tl_lat0=iom_mpp_read_var(tl_coord0, TRIM(cl_name))
         
         CALL grid_del_ghost(tl_lon0, il_xghost0(:,:))
         CALL grid_del_ghost(tl_lat0, il_xghost0(:,:))

         ! close mpp files
         CALL iom_mpp_close(tl_coord0)

         ! Fine grid

         ! get ghost cell factor on fine grid
         il_xghost1(:,:)=grid_get_ghost( tl_coord1 )

         ! open mpp files
         CALL iom_mpp_open(tl_coord1)

         ! read fine longitue and latitude
         WRITE(cl_name,*) 'longitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord1%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET COARSE INDEX: no variable "//&
               &  TRIM(cl_name)//" in file "//TRIM(tl_coord1%c_name)//". &
               &  try to use longitude.")
            WRITE(cl_name,*) 'longitude'
         ENDIF
         tl_lon1=iom_mpp_read_var(tl_coord1, TRIM(cl_name)) 

         WRITE(cl_name,*) 'latitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord1%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET COARSE INDEX: no variable "//&
            &  TRIM(cl_name)//" in file "//TRIM(tl_coord1%c_name)//". &
            &  try to use latitude.")
            WRITE(cl_name,*) 'latitude'
         ENDIF
         tl_lat1=iom_mpp_read_var(tl_coord1, TRIM(cl_name))
 
         CALL grid_del_ghost(tl_lon1, il_xghost1(:,:))
         CALL grid_del_ghost(tl_lat1, il_xghost1(:,:))

         ! close mpp files
         CALL iom_mpp_close(tl_coord1)

         ! compute
         if_idx(:,:)=grid_get_coarse_index(tl_lon0,tl_lat0,&
            &                              tl_lon1,tl_lat1,&
            &                              il_rho(:) )

         ! add ghost cell to indices
         il_imin0=if_idx(1,1)+il_xghost0(jp_I,1)*ip_ghost
         il_imax0=if_idx(1,2)+il_xghost0(jp_I,1)*ip_ghost

         il_jmin0=if_idx(2,1)+il_xghost0(jp_J,1)*ip_ghost
         il_jmax0=if_idx(2,2)+il_xghost0(jp_J,1)*ip_ghost

         if_idx(jp_I,1)=il_imin0
         if_idx(jp_I,2)=il_imax0
         if_idx(jp_J,1)=il_jmin0
         if_idx(jp_J,2)=il_jmax0

         CALL var_clean(tl_lon0)
         CALL var_clean(tl_lat0)         
         CALL var_clean(tl_lon1)
         CALL var_clean(tl_lat1)         

      ENDIF

      ! clean
      CALL mpp_clean(tl_coord0)
      CALL mpp_clean(tl_coord1)
      DEALLOCATE(il_rho)

   END FUNCTION grid__get_coarse_index_ff
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_coarse_index_cf(td_lon0, td_lat0, td_coord1, &
         &                            id_rho, cd_point) &
         & RESULT (if_id)
   !-------------------------------------------------------------------
   !> @brief This function get closest coarse grid indices of fine grid domain.
   !>
   !> @details
   !> it use coarse array of longitude and latitude and fine grid coordinates file.
   !> optionaly, you could specify the array of refinment factor (default 1.)
   !> optionally, you could specify on which Arakawa grid point you want to
   !> work (default 'T')
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014
   !> - use grid point to read coordinates variable.
   !> @date October, 2014
   !> - work on mpp file structure instead of file structure
   !> @date February, 2015
   !> - use longitude or latitude as standard name, if can not find 
   !> longitude_T, latitude_T...
   !>
   !> @param[in] td_longitude0   coarse grid longitude
   !> @param[in] td_latitude0    coarse grid latitude
   !> @param[in] td_coord1       fine grid coordinate mpp structure
   !> @param[in] id_rho          array of refinment factor
   !> @param[in] cd_point        Arakawa grid point (default 'T')
   !> @return coarse grid indices (/(/imin0, imax0/), (/jmin0, jmax0/)/)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR )                   , INTENT(IN) :: td_lon0
      TYPE(TVAR )                   , INTENT(IN) :: td_lat0
      TYPE(TMPP )                   , INTENT(IN) :: td_coord1
      INTEGER(i4)     , DIMENSION(:), INTENT(IN), OPTIONAL :: id_rho
      CHARACTER(LEN=*)              , INTENT(IN), OPTIONAL :: cd_point

      ! function
      INTEGER(i4), DIMENSION(2,2) :: if_id

      ! local variable
      CHARACTER(LEN= 1)                        :: cl_point
      CHARACTER(LEN=lc)                        :: cl_name

      INTEGER(i4)                              :: il_ind

      INTEGER(i4), DIMENSION(:)  , ALLOCATABLE :: il_rho

      INTEGER(i4), DIMENSION(2,2)              :: il_xghost

      TYPE(TVAR)                               :: tl_lon1
      TYPE(TVAR)                               :: tl_lat1

      TYPE(TMPP)                               :: tl_coord1

      ! loop indices
      !----------------------------------------------------------------

      ! init
      if_id(:,:)=0

      ALLOCATE(il_rho(ip_maxdim) )
      il_rho(:)=1
      IF( PRESENT(id_rho) ) il_rho(:)=id_rho(:)

      ! copy structure
      tl_coord1=mpp_copy(td_coord1)
      
      cl_point='T'
      IF( PRESENT(cd_point) ) cl_point=TRIM(fct_upper(cd_point))

      IF( .NOT. ASSOCIATED(tl_coord1%t_proc) )THEN
         CALL logger_error("GRID GET COARSE INDEX: decompsition of mpp "//&
         &  "file "//TRIM(tl_coord1%c_name)//" not defined." )

      ELSE IF( .NOT. ASSOCIATED(td_lon0%d_value) .OR. &
      &        .NOT. ASSOCIATED(td_lat0%d_value) )THEN

         CALL logger_error("GRID GET COARSE INDEX: some coarse grid"//&
         &                 " coordinate value are not associated.")         

      ELSE
         
         IF( TRIM(td_lon0%c_point)/='' )THEN
            cl_point=TRIM(td_lon0%c_point)
         ELSEIF( TRIM(td_lat0%c_point)/='' )THEN
            cl_point=TRIM(td_lat0%c_point)
         ENDIF

         ! Fine grid
         ! get ghost cell factor on fine grid
         il_xghost(:,:)=grid_get_ghost( tl_coord1 )

         ! open mpp files
         CALL iom_mpp_open(tl_coord1)

         ! read fine longitue and latitude
         WRITE(cl_name,*) 'longitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord1%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET COARSE INDEX: no variable "//&
               &  TRIM(cl_name)//"in file "//TRIM(tl_coord1%c_name)//". &
               &  try to use longitude.")
            WRITE(cl_name,*) 'longitude'
         ENDIF
         tl_lon1=iom_mpp_read_var(tl_coord1, TRIM(cl_name))

         WRITE(cl_name,*) 'latitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord1%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET COARSE INDEX: no variable "//&
               &  TRIM(cl_name)//"in file "//TRIM(tl_coord1%c_name)//". &
               &  try to use longitude.")
            WRITE(cl_name,*) 'latitude'
         ENDIF
         tl_lat1=iom_mpp_read_var(tl_coord1, TRIM(cl_name))
         
         CALL grid_del_ghost(tl_lon1, il_xghost(:,:))
         CALL grid_del_ghost(tl_lat1, il_xghost(:,:))

         ! close mpp files
         CALL iom_mpp_close(tl_coord1)

         ! compute
         if_id(:,:)=grid_get_coarse_index(td_lon0,td_lat0,&
            &                             tl_lon1,tl_lat1,&
            &                             il_rho(:), cl_point )

         CALL var_clean(tl_lon1)
         CALL var_clean(tl_lat1)         

      ENDIF

      DEALLOCATE(il_rho)
      CALL mpp_clean(tl_coord1)

   END FUNCTION grid__get_coarse_index_cf
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_coarse_index_fc(td_coord0, td_lon1, td_lat1, &
         &                            id_rho, cd_point) &
         & RESULT (if_idx)
   !-------------------------------------------------------------------
   !> @brief This function get closest coarse grid indices of fine grid domain.
   !>
   !> @details
   !> it use coarse grid coordinates file and fine grid array of longitude and latitude.
   !> optionaly, you could specify the array of refinment factor (default 1.)
   !> optionally, you could specify on which Arakawa grid point you want to
   !> work (default 'T')
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014
   !> - use grid point to read coordinates variable.
   !> @date October, 2014
   !> - work on mpp file structure instead of file structure
   !> @date February, 2015
   !> - use longitude or latitude as standard name, if can not find 
   !> longitude_T, latitude_T...
   !> 
   !> @param[in] td_coord0 coarse grid coordinate mpp structure
   !> @param[in] td_lon1   fine grid longitude
   !> @param[in] td_lat1   fine grid latitude
   !> @param[in] id_rho    array of refinment factor (default 1.)
   !> @param[in] cd_point  Arakawa grid point (default 'T')
   !> @return coarse grid indices (/(/imin0, imax0/), (/jmin0, jmax0/)/)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP )                   , INTENT(IN) :: td_coord0
      TYPE(TVAR )                   , INTENT(IN) :: td_lon1
      TYPE(TVAR )                   , INTENT(IN) :: td_lat1
      INTEGER(i4)     , DIMENSION(:), INTENT(IN), OPTIONAL :: id_rho
      CHARACTER(LEN=*)              , INTENT(IN), OPTIONAL :: cd_point

      ! function
      INTEGER(i4)     , DIMENSION(2,2)           :: if_idx

      ! local variable
      CHARACTER(LEN= 1)                        :: cl_point
      CHARACTER(LEN=lc)                        :: cl_name      

      INTEGER(i4)                              :: il_imin0
      INTEGER(i4)                              :: il_imax0
      INTEGER(i4)                              :: il_jmin0
      INTEGER(i4)                              :: il_jmax0
      INTEGER(i4)                              :: il_ind

      INTEGER(i4), DIMENSION(:), ALLOCATABLE   :: il_rho

      INTEGER(i4), DIMENSION(2,2)              :: il_xghost

      TYPE(TVAR)                               :: tl_lon0
      TYPE(TVAR)                               :: tl_lat0

      TYPE(TMPP)                               :: tl_coord0

      ! loop indices
      !----------------------------------------------------------------

      ! init
      if_idx(:,:)=0

      ALLOCATE(il_rho(ip_maxdim))
      il_rho(:)=1
      IF( PRESENT(id_rho) ) il_rho(:)=id_rho(:)

      cl_point='T'
      IF( PRESENT(cd_point) ) cl_point=TRIM(fct_upper(cd_point))

      ! copy structure
      tl_coord0=mpp_copy(td_coord0)

      IF( .NOT. ASSOCIATED(tl_coord0%t_proc) )THEN
         CALL logger_error("GRID GET COARSE INDEX: decompsition of mpp "//&
            &              "file "//TRIM(tl_coord0%c_name)//" not defined." )

      ELSE IF( .NOT. ASSOCIATED(td_lon1%d_value) .OR. &
      &        .NOT. ASSOCIATED(td_lat1%d_value) )THEN

         CALL logger_error("GRID GET COARSE INDEX: some fine grid"//&
            &              " coordinate value are not associated.")

      ELSE

         IF( TRIM(td_lon1%c_point)/='' )THEN
            cl_point=TRIM(td_lon1%c_point)
         ELSEIF( TRIM(td_lat1%c_point)/='' )THEN
            cl_point=TRIM(td_lat1%c_point)
         ENDIF

         ! get ghost cell factor on coarse grid
         il_xghost(:,:)=grid_get_ghost( tl_coord0 )

         ! open mpp files
         CALL iom_mpp_open(tl_coord0)

         ! read coarse longitue and latitude
         WRITE(cl_name,*) 'longitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord0%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET COARSE INDEX: no variable "//&
               &  TRIM(cl_name)//"in file "//TRIM(tl_coord0%c_name)//". &
               &  try to use longitude.")
            WRITE(cl_name,*) 'longitude'
         ENDIF
         tl_lon0=iom_mpp_read_var(tl_coord0, TRIM(cl_name))
         
         WRITE(cl_name,*) 'latitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord0%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET COARSE INDEX: no variable "//&
               &  TRIM(cl_name)//"in file "//TRIM(tl_coord0%c_name)//". &
               &  try to use latitude.")
            WRITE(cl_name,*) 'latitude'
         ENDIF
         tl_lat0=iom_mpp_read_var(tl_coord0, TRIM(cl_name))

         CALL grid_del_ghost(tl_lon0, il_xghost(:,:))
         CALL grid_del_ghost(tl_lat0, il_xghost(:,:))

         ! close mpp files
         CALL iom_mpp_close(tl_coord0)

         if_idx(:,:)=grid_get_coarse_index(tl_lon0,tl_lat0,&
            &                              td_lon1,td_lat1,&
            &                              il_rho(:), cl_point )

         ! remove ghost cell
         il_imin0=if_idx(1,1)+il_xghost(jp_I,1)*ip_ghost
         il_imax0=if_idx(1,2)+il_xghost(jp_I,1)*ip_ghost

         il_jmin0=if_idx(2,1)+il_xghost(jp_J,1)*ip_ghost
         il_jmax0=if_idx(2,2)+il_xghost(jp_J,1)*ip_ghost

         if_idx(1,1)=il_imin0
         if_idx(1,2)=il_imax0
         if_idx(2,1)=il_jmin0
         if_idx(2,2)=il_jmax0

         CALL var_clean(tl_lon0)
         CALL var_clean(tl_lat0)

      ENDIF

      CALL mpp_clean(tl_coord0)
      DEALLOCATE(il_rho)

   END FUNCTION grid__get_coarse_index_fc
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_coarse_index_cc(td_lon0, td_lat0, td_lon1, td_lat1, &
         &                            id_rho, cd_point) &
         & RESULT (if_idx)
   !-------------------------------------------------------------------
   !> @brief This function get closest coarse grid indices of fine grid domain.
   !
   !> @details
   !> it use coarse and fine grid array of longitude and latitude.
   !> optionaly, you could specify the array of refinment factor (default 1.)
   !> optionally, you could specify on which Arakawa grid point you want to
   !> work (default 'T')
   !>
   !> @note do not use ghost cell
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014
   !> - check grid point
   !> - take into account EW overlap
   !> @date February, 2016
   !> - use delta (lon or lat)
   !> - manage cases for T,U,V or F point, with even or odd refinment
   !>
   !> @param[in] td_lon0   coarse grid longitude
   !> @param[in] td_lat0   coarse grid latitude
   !> @param[in] td_lon1   fine grid longitude
   !> @param[in] td_lat1   fine grid latitude
   !> @param[in] id_rho    array of refinment factor
   !> @param[in] cd_point  Arakawa grid point ('T','U','V','F')
   !> @return coarse grid indices (/(/imin0, imax0/), (/jmin0, jmax0/)/)
   !>
   !> @todo
   !> -check case boundary domain on overlap band
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR)                    , INTENT(IN) :: td_lon0
      TYPE(TVAR)                    , INTENT(IN) :: td_lat0
      TYPE(TVAR)                    , INTENT(IN) :: td_lon1
      TYPE(TVAR)                    , INTENT(IN) :: td_lat1
      INTEGER(i4)     , DIMENSION(:), INTENT(IN), OPTIONAL :: id_rho
      CHARACTER(LEN=*)              , INTENT(IN), OPTIONAL :: cd_point

      ! function
      INTEGER(i4)     , DIMENSION(2,2)           :: if_idx

      ! local variable
      CHARACTER(LEN= 1)                      :: cl_point0
      CHARACTER(LEN= 1)                      :: cl_point1

      LOGICAL    , DIMENSION(2)              :: ll_even

      REAL(dp)                               :: dl_lon1
      REAL(dp)                               :: dl_dlon
      REAL(dp)                               :: dl_lat1
      REAL(dp)                               :: dl_dlat

      INTEGER(i4)                            :: il_ew0 
      INTEGER(i4)                            :: il_imin0
      INTEGER(i4)                            :: il_imax0
      INTEGER(i4)                            :: il_jmin0
      INTEGER(i4)                            :: il_jmax0

      INTEGER(i4)                            :: il_ew1 
      INTEGER(i4)                            :: il_imin1
      INTEGER(i4)                            :: il_imax1
      INTEGER(i4)                            :: il_jmin1
      INTEGER(i4)                            :: il_jmax1

      INTEGER(i4)                            :: il_imin
      INTEGER(i4)                            :: il_imax
      INTEGER(i4)                            :: il_jmin
      INTEGER(i4)                            :: il_jmax      

      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_rho

      INTEGER(i4), DIMENSION(2)              :: il_ill
      INTEGER(i4), DIMENSION(2)              :: il_ilr
      INTEGER(i4), DIMENSION(2)              :: il_iul
      INTEGER(i4), DIMENSION(2)              :: il_iur

      INTEGER(i4), DIMENSION(2,2)            :: il_xghost0
      INTEGER(i4), DIMENSION(2,2)            :: il_yghost0
      INTEGER(i4), DIMENSION(2,2)            :: il_xghost1
      INTEGER(i4), DIMENSION(2,2)            :: il_yghost1

      TYPE(TVAR)                             :: tl_lon0
      TYPE(TVAR)                             :: tl_lat0
      TYPE(TVAR)                             :: tl_lon1
      TYPE(TVAR)                             :: tl_lat1

      ! loop indices
      !----------------------------------------------------------------
      ! init
      if_idx(:,:)=0

      ALLOCATE( il_rho(ip_maxdim) )
      il_rho(:)=1
      IF( PRESENT(id_rho) ) il_rho(:)=id_rho(:)

      ll_even(:)=(/ (MOD(id_rho(jp_I),2)==0), (MOD(id_rho(jp_J),2)==0) /)

      cl_point0='T'
      cl_point1='T'
      IF( PRESENT(cd_point) )THEN
         cl_point0=TRIM(fct_upper(cd_point))
         cl_point1=TRIM(fct_upper(cd_point))
      ENDIF
      
      IF( .NOT. ASSOCIATED(td_lon0%d_value) .OR. &
      &   .NOT. ASSOCIATED(td_lat0%d_value) .OR. &
      &   .NOT. ASSOCIATED(td_lon1%d_value) .OR. &
      &   .NOT. ASSOCIATED(td_lat1%d_value) )THEN
         CALL logger_error("GRID GET COARSE INDEX: some fine or coarse grid"//&
         &                 " coordinate value not associated.")
      ELSE

         IF( TRIM(td_lon0%c_point)/='' )THEN
            cl_point0=TRIM(td_lon0%c_point)
         ELSEIF( TRIM(td_lat0%c_point)/='' )THEN
            cl_point0=TRIM(td_lat0%c_point)
         ENDIF
         IF( TRIM(td_lon1%c_point)/='' )THEN
            cl_point1=TRIM(td_lon1%c_point)
         ELSEIF( TRIM(td_lat1%c_point)/='' )THEN
            cl_point1=TRIM(td_lat1%c_point)
         ENDIF
         IF( cl_point0 /= cl_point1 )THEN
            CALL logger_error("GRID GET COARSE INDEX: fine and coarse grid"//&
         &                 " coordinate not on same grid point.")
         ENDIF

         IF( grid_is_global(td_lon1, td_lat1) )THEN

            IF( grid_is_global(td_lon0, td_lat0) )THEN
               CALL logger_trace("GRID GET COARSE INDEX: fine grid is global ")
               if_idx(:,:) = 1
               if_idx(:,:) = 0
            ELSE
               CALL logger_error("GRID GET COARSE INDEX: fine grid is "//&
               &                 "global, coarse grid not.")
            ENDIF

         ELSE

            il_xghost0(:,:)=grid_get_ghost( td_lon0 )
            il_yghost0(:,:)=grid_get_ghost( td_lat0 )
            IF( ANY(il_xghost0(:,:) /= il_yghost0(:,:)) )THEN
               CALL logger_error("GRID GET COARSE INDEX: coarse grid "//&
               &        "coordinate do not share same ghost cell")
            ENDIF

            tl_lon0=var_copy(td_lon0)
            tl_lat0=var_copy(td_lat0)
            CALL grid_del_ghost(tl_lon0, il_xghost0(:,:))
            CALL grid_del_ghost(tl_lat0, il_xghost0(:,:))
 
            ! "global" coarse grid indice
            il_imin0=1
            il_jmin0=1

            il_imax0=tl_lon0%t_dim(1)%i_len
            il_jmax0=tl_lon0%t_dim(2)%i_len

            ! get east west overlap for coarse grid
            il_ew0=tl_lon0%i_ew
            IF( il_ew0 >= 0 )THEN
               ! last point before overlap
               il_imax0=il_imax0-il_ew0
            ENDIF

            il_xghost1(:,:)=grid_get_ghost( td_lon1 )
            il_yghost1(:,:)=grid_get_ghost( td_lat1 )
            IF( ANY(il_xghost1(:,:) /= il_yghost1(:,:)) )THEN
               CALL logger_error("GRID GET COARSE INDEX: fine grid "//&
               &        "coordinate do not share same ghost cell")
            ENDIF

            tl_lon1=var_copy(td_lon1)
            tl_lat1=var_copy(td_lat1)
            CALL grid_del_ghost(tl_lon1, il_xghost1(:,:))
            CALL grid_del_ghost(tl_lat1, il_xghost1(:,:))
            
            ! "global" fine grid indice
            il_imin1=1
            il_jmin1=1

            il_imax1=tl_lon1%t_dim(1)%i_len
            il_jmax1=tl_lon1%t_dim(2)%i_len

            ! get east west overlap for fine grid
            il_ew1=tl_lon1%i_ew
            IF( il_ew1 >= 0 )THEN
               ! last point before overlap
               il_imax1=il_imax1-il_ew1
            ENDIF

            ! get indices for each corner
            !1- search lower left corner indices
            dl_lon1=tl_lon1%d_value( il_imin1, il_jmin1, 1, 1 )
            dl_lat1=tl_lat1%d_value( il_imin1, il_jmin1, 1, 1 )

            IF( dl_lon1 == tl_lon1%d_fill .OR. &
            &   dl_lat1 == tl_lat1%d_fill )THEN
               CALL logger_error("GRID GET COARSE INDEX: lower left corner "//&
                  &                 "point is FillValue. remove ghost cell "//&
                  &                 "before running grid_get_coarse_index.")
            ENDIF

            !!!!! i-direction !!!!!
            IF( ll_even(jp_I) )THEN
               ! even
               SELECT CASE(TRIM(cl_point1))
                  CASE('F','U')
                     dl_dlon= ( tl_lon1%d_value(il_imin1+1,il_jmin1,1,1) -   &
                        &       tl_lon1%d_value(il_imin1  ,il_jmin1,1,1) ) / &
                        &     2.
                  CASE DEFAULT
                     dl_dlon=0
               END SELECT
            ELSE
               ! odd
               dl_dlon= ( tl_lon1%d_value(il_imin1+1,il_jmin1,1,1) -   &
                  &       tl_lon1%d_value(il_imin1  ,il_jmin1,1,1) ) / &
                  &     2.
            ENDIF

            !!!!! j-direction !!!!!
            IF( ll_even(jp_J) )THEN
               ! even
               SELECT CASE(TRIM(cl_point1))
                  CASE('F','V')
                     dl_dlat= ( tl_lat1%d_value(il_imin1,il_jmin1+1,1,1) -   &
                        &       tl_lat1%d_value(il_imin1,il_jmin1  ,1,1) ) / &
                        &     2.
                  CASE DEFAULT
                     dl_dlat=0
               END SELECT
            ELSE
               ! odd
               dl_dlat= ( tl_lat1%d_value(il_imin1,il_jmin1+1,1,1) -   &
                  &       tl_lat1%d_value(il_imin1,il_jmin1  ,1,1) ) / &
                  &     2.
            ENDIF

            dl_lon1 = dl_lon1 + dl_dlon
            dl_lat1 = dl_lat1 + dl_dlat

            ! look for closest point on coarse grid
            il_ill(:)= grid_get_closest(tl_lon0%d_value(il_imin0:il_imax0, &
               &                                        il_jmin0:il_jmax0, &
               &                                        1,1), &
               &                        tl_lat0%d_value(il_imin0:il_imax0, &
               &                                        il_jmin0:il_jmax0, &
               &                                        1,1), &
               &                        dl_lon1, dl_lat1, 'll'   )


            !2- search upper left corner indices
            dl_lon1=tl_lon1%d_value( il_imin1, il_jmax1, 1, 1 )
            dl_lat1=tl_lat1%d_value( il_imin1, il_jmax1, 1, 1 )

            IF( dl_lon1 == tl_lon1%d_fill .OR. &
            &   dl_lat1 == tl_lat1%d_fill )THEN
               CALL logger_error("GRID GET COARSE INDEX: upper left corner "//&
               &                 "point is FillValue. remove ghost cell "//&
               &                 "running grid_get_coarse_index.")
            ENDIF            

            !!!!! i-direction !!!!!
            IF( ll_even(jp_I) )THEN
               ! even
               SELECT CASE(TRIM(cl_point1))
                  CASE('F','U')
                     dl_dlon= ( tl_lon1%d_value(il_imin1+1,il_jmax1,1,1) -   &
                        &       tl_lon1%d_value(il_imin1  ,il_jmax1,1,1) ) / &
                        &     2.
                  CASE DEFAULT
                     dl_dlon=0
               END SELECT
            ELSE
               ! odd
               dl_dlon= ( tl_lon1%d_value(il_imin1+1,il_jmax1,1,1) -   &
                  &       tl_lon1%d_value(il_imin1  ,il_jmax1,1,1) ) / &
                  &     2.
            ENDIF

            !!!!! j-direction !!!!!
            IF( ll_even(jp_J) )THEN
               ! even
               SELECT CASE(TRIM(cl_point1))
                  CASE('F','V')
                     dl_dlat= ( tl_lat1%d_value(il_imin1,il_jmax1  ,1,1) -   &
                        &       tl_lat1%d_value(il_imin1,il_jmax1-1,1,1) ) / &
                        &     2.
                  CASE DEFAULT
                     dl_dlat=0
               END SELECT
            ELSE
               ! odd
               dl_dlat= ( tl_lat1%d_value(il_imin1,il_jmax1  ,1,1) -   &
                  &       tl_lat1%d_value(il_imin1,il_jmax1-1,1,1) ) / &
                  &     2.
            ENDIF

            dl_lon1 = dl_lon1 + dl_dlon
            dl_lat1 = dl_lat1 - dl_dlat

            ! look for closest point on coarse grid
            il_iul(:)= grid_get_closest(tl_lon0%d_value(il_imin0:il_imax0, &
               &                                        il_jmin0:il_jmax0, &
               &                                        1,1), &
               &                        tl_lat0%d_value(il_imin0:il_imax0, &
               &                                        il_jmin0:il_jmax0, &
               &                                        1,1), &
               &                        dl_lon1, dl_lat1, 'ul' )

            !3- search lower right corner indices
            dl_lon1=tl_lon1%d_value( il_imax1, il_jmin1, 1, 1 )
            dl_lat1=tl_lat1%d_value( il_imax1, il_jmin1, 1, 1 )

            IF( dl_lon1 == tl_lon1%d_fill .OR. &
            &   dl_lat1 == tl_lat1%d_fill )THEN
               CALL logger_error("GRID GET COARSE INDEX: lower right corner "//&
                  &                 "point is FillValue. remove ghost cell "//&
                  &                 "running grid_get_coarse_index.")
            ENDIF            

            !!!!! i-direction !!!!!
            IF( ll_even(jp_I) )THEN
               ! even
               SELECT CASE(TRIM(cl_point1))
                  CASE('F','U')
                     dl_dlon= ( tl_lon1%d_value(il_imax1  ,il_jmin1,1,1) -   &
                        &       tl_lon1%d_value(il_imax1-1,il_jmin1,1,1) ) / &
                        &     2.
                  CASE DEFAULT
                     dl_dlon=0
               END SELECT
            ELSE
               ! odd
               dl_dlon= ( tl_lon1%d_value(il_imax1  ,il_jmin1,1,1) -   &
                  &       tl_lon1%d_value(il_imax1-1,il_jmin1,1,1) ) / &
                  &     2.
            ENDIF

            !!!!! j-direction !!!!!
            IF( ll_even(jp_J) )THEN
               ! even
               SELECT CASE(TRIM(cl_point1))
                  CASE('F','V')
                     dl_dlat= ( tl_lat1%d_value(il_imax1,il_jmin1+1,1,1) -   &
                        &       tl_lat1%d_value(il_imax1,il_jmin1  ,1,1) ) / &
                        &     2.
                  CASE DEFAULT
                     dl_dlat=0
               END SELECT
            ELSE
               ! odd
               dl_dlat= ( tl_lat1%d_value(il_imax1,il_jmin1+1,1,1) -   &
                  &       tl_lat1%d_value(il_imax1,il_jmin1  ,1,1) ) / &
                  &     2.
            ENDIF

            dl_lon1 = dl_lon1 - dl_dlon
            dl_lat1 = dl_lat1 + dl_dlat

            ! look for closest point on coarse grid
            il_ilr(:)= grid_get_closest(tl_lon0%d_value(il_imin0:il_imax0, &
               &                                        il_jmin0:il_jmax0, &
               &                                        1,1), &
               &                        tl_lat0%d_value(il_imin0:il_imax0, &
               &                                        il_jmin0:il_jmax0, &
               &                                        1,1), &
               &                        dl_lon1, dl_lat1, 'lr' )

            !4- search upper right corner indices
            dl_lon1=tl_lon1%d_value( il_imax1, il_jmax1, 1, 1 )
            dl_lat1=tl_lat1%d_value( il_imax1, il_jmax1, 1, 1 )

            IF( dl_lon1 == tl_lon1%d_fill .OR. &
            &   dl_lat1 == tl_lat1%d_fill )THEN
               CALL logger_error("GRID GET COARSE INDEX: upper right corner "//&
                  &                 "point is FillValue. remove ghost cell "//&
                  &                 "before running grid_get_coarse_index.")
            ENDIF            

            !!!!! i-direction !!!!!
            IF( ll_even(jp_I) )THEN
               ! even
               SELECT CASE(TRIM(cl_point1))
                  CASE('F','U')
                     dl_dlon= ( tl_lon1%d_value(il_imax1  ,il_jmax1,1,1) -   &
                        &       tl_lon1%d_value(il_imax1-1,il_jmax1,1,1) ) / &
                        &     2.
                  CASE DEFAULT
                     dl_dlon=0
               END SELECT
            ELSE
               ! odd
               dl_dlon= ( tl_lon1%d_value(il_imax1  ,il_jmax1,1,1) -   &
                  &       tl_lon1%d_value(il_imax1-1,il_jmax1,1,1) ) / &
                  &     2.
            ENDIF

            !!!!! j-direction !!!!!
            IF( ll_even(jp_J) )THEN
               ! even
               SELECT CASE(TRIM(cl_point1))
                  CASE('F','V')
                     dl_dlat= ( tl_lat1%d_value(il_imax1,il_jmax1  ,1,1) -   &
                        &       tl_lat1%d_value(il_imax1,il_jmax1-1,1,1) ) / &
                        &     2.
                  CASE DEFAULT
                     dl_dlat=0
               END SELECT
            ELSE
               ! odd
               dl_dlat= ( tl_lat1%d_value(il_imax1,il_jmax1  ,1,1) -   &
                  &       tl_lat1%d_value(il_imax1,il_jmax1-1,1,1) ) / &
                  &     2.
            ENDIF

            dl_lon1 = dl_lon1 - dl_dlon
            dl_lat1 = dl_lat1 - dl_dlat

            ! look for closest point on coarse grid
            il_iur(:)= grid_get_closest(tl_lon0%d_value(il_imin0:il_imax0, &
               &                                        il_jmin0:il_jmax0, &
               &                                        1,1), &
               &                        tl_lat0%d_value(il_imin0:il_imax0, &
               &                                        il_jmin0:il_jmax0, &
               &                                        1,1), &
               &                        dl_lon1, dl_lat1, 'ur' )

            ! coarse grid indices
            il_imin = il_imin0-1+MIN(il_ill(1), il_iul(1))
            il_imax = il_imin0-1+MAX(il_ilr(1), il_iur(1))

            IF( il_imax <= il_ew0 )THEN
               !il_imin = 1
               il_imax = tl_lon0%t_dim(1)%i_len - il_ew0 + il_imax 
            ENDIF

            il_jmin = il_jmin0-1+MIN(il_ill(2), il_ilr(2))
            il_jmax = il_jmin0-1+MAX(il_iul(2), il_iur(2))

            ! special case if east west overlap
            IF( il_ew1 >= 0 )THEN
               CALL logger_debug("GRID GET COARSE INDEX: East-West overlap "//&
               &                 "found for fine grid " )

               il_imin = 1
               il_imax = tl_lon0%t_dim(1)%i_len

            ENDIF
         ENDIF

         if_idx(1,1) = il_imin
         if_idx(1,2) = il_imax

         if_idx(2,1) = il_jmin
         if_idx(2,2) = il_jmax
 
         ! clean 
         CALL var_clean(tl_lon1)
         CALL var_clean(tl_lat1)
         CALL var_clean(tl_lon0)
         CALL var_clean(tl_lat0)
      ENDIF

      DEALLOCATE( il_rho )

   END FUNCTION grid__get_coarse_index_cc
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid_is_global(td_lon, td_lat) &
         & RESULT (lf_global)
   !-------------------------------------------------------------------
   !> @brief This function check if grid is global or not
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_lon longitude structure 
   !> @param[in] td_lat latitude structure 
   !> @return true if grid is global
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR), INTENT(IN) :: td_lon
      TYPE(TVAR), INTENT(IN) :: td_lat

      ! function
      LOGICAL                :: lf_global
      
      ! local variable
      INTEGER(i4)               :: il_ew
      INTEGER(i4)               :: il_south
      INTEGER(i4)               :: il_north

      REAL(dp)                  :: dl_lat_min
      REAL(dp)                  :: dl_lat_max

      ! loop indices
      !----------------------------------------------------------------

      ! init
      lf_global=.FALSE.

      IF( ANY( td_lon%t_dim(:)%i_len /= td_lat%t_dim(:)%i_len )  )THEN
         CALL logger_fatal("GRID IS GLOBAL: dimension of longitude and "//&
         &              " latitude differ")
      ENDIF

      IF( .NOT. ASSOCIATED(td_lon%d_value) .OR. &
      &   .NOT. ASSOCIATED(td_lat%d_value) )THEN
         CALL logger_error("GRID IS GLOBAL: no value associated to "//&
         &              " longitude or latitude strucutre")      
      ELSE

         il_south=1
         il_north=td_lon%t_dim(2)%i_len

         dl_lat_min=MINVAL(td_lat%d_value(:,il_south,1,1))
         dl_lat_max=MAXVAL(td_lat%d_value(:,il_north,1,1))

         IF( dl_lat_min < -77.0 .AND. dl_lat_max > 89.0 )THEN

            il_ew=td_lon%i_ew
            IF( il_ew >= 0 )THEN

               lf_global=.TRUE.

            ENDIF

         ENDIF
      ENDIF

   END FUNCTION grid_is_global
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_closest_str( td_coord0, dd_lon1, dd_lat1, cd_pos, dd_fill ) &
         &  RESULT(if_idx)
   !-------------------------------------------------------------------
   !> @brief This function return grid indices of the closest point
   !> from point (lon1,lat1) 
   !> 
   !> @details
   !>
   !> @note overlap band should have been already removed from coarse grid array 
   !> of longitude and latitude, before running this function
   !>
   !> if you add cd_pos argument, you could choice to return closest point at
   !> - lower left  (ll) of the point
   !> - lower right (lr) of the point
   !> - upper left  (ul) of the point
   !> - upper right (ur) of the point
   !> - lower       (lo) of the point
   !> - upper       (up) of the point
   !> -       left  (le) of the point
   !> -       right (ri) of the point
   !>
   !> @author J.Paul
   !> @date April, 2016 - Initial Version
   !> @date October, 2016
   !> - use max of zero and east-west overlap instead of east-west overlap
   !>
   !> @param[in] td_coord0 coarse grid coordinate mpp structure
   !> @param[in] dd_lon1   fine   grid longitude
   !> @param[in] dd_lat1   fine   grid latitude
   !> @param[in] cd_pos    relative position of grid point from point 
   !> @param[in] dd_fill   fill value
   !> @return coarse grid indices of closest point of fine grid point
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP )     , INTENT(IN) :: td_coord0
      REAL(dp),         INTENT(IN) :: dd_lon1
      REAL(dp),         INTENT(IN) :: dd_lat1
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: cd_pos
      REAL(dp),         INTENT(IN), OPTIONAL :: dd_fill

      ! function
      INTEGER(i4), DIMENSION(2)    :: if_idx

      ! local variable
      CHARACTER(LEN=lc)                        :: cl_point
      CHARACTER(LEN=lc)                        :: cl_name

      INTEGER(i4)                              :: il_ind
      INTEGER(i4)                              :: il_ew

      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_lon0
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_lat0

      TYPE(TVAR)                               :: tl_lon0
      TYPE(TVAR)                               :: tl_lat0
      TYPE(TMPP)                               :: tl_coord0
      !----------------------------------------------------------------

      if_idx(:)=-1
      cl_point='T'

      ! copy structure
      tl_coord0=mpp_copy(td_coord0)

      IF( .NOT. ASSOCIATED(tl_coord0%t_proc) )THEN

         CALL logger_error("GRID GET CLOSEST: decompsition of mpp "//&
         &  "file "//TRIM(tl_coord0%c_name)//" not defined." )

      ELSE

         ! open mpp files
         CALL iom_mpp_open(tl_coord0)
 
         ! read coarse longitue and latitude
         WRITE(cl_name,*) 'longitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord0%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET CLOSEST: no variable "//&
            &  TRIM(cl_name)//"in file "//TRIM(tl_coord0%c_name)//". &
            &  try to use longitude.")
            WRITE(cl_name,*) 'longitude'
         ENDIF
         tl_lon0=iom_mpp_read_var(tl_coord0, TRIM(cl_name))
 
         WRITE(cl_name,*) 'latitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord0%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET CLOSEST: no variable "//&
            &  TRIM(cl_name)//"in file "//TRIM(tl_coord0%c_name)//". &
            &  try to use latitude.")
            WRITE(cl_name,*) 'latitude'
         ENDIF
         tl_lat0=iom_mpp_read_var(tl_coord0, TRIM(cl_name))

         ! close mpp files
         CALL iom_mpp_close(tl_coord0)

         il_ew=MAX(0,tl_coord0%i_ew)
         ALLOCATE(dl_lon0(tl_coord0%t_dim(jp_I)%i_len-il_ew, &
            &             tl_coord0%t_dim(jp_J)%i_len) )             
         ALLOCATE(dl_lat0(tl_coord0%t_dim(jp_I)%i_len-il_ew, &
            &             tl_coord0%t_dim(jp_J)%i_len) )

         dl_lon0(:,:)=tl_lon0%d_value(il_ew+1:,:,1,1)
         dl_lat0(:,:)=tl_lat0%d_value(il_ew+1:,:,1,1)

         if_idx(:)=grid_get_closest( dl_lon0, dl_lat0, dd_lon1, dd_lat1, cd_pos, dd_fill )

         DEALLOCATE(dl_lon0, dl_lat0)
         CALL var_clean(tl_lon0)
         CALL var_clean(tl_lat0)
         CALL mpp_clean(tl_coord0)

      ENDIF

   END FUNCTION  grid__get_closest_str
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_closest_arr(dd_lon0, dd_lat0, dd_lon1, dd_lat1, cd_pos, dd_fill) &
         & RESULT (if_idx)
   !-------------------------------------------------------------------
   !> @brief This function return grid indices of the closest point
   !> from point (lon1,lat1) 
   !> 
   !> @details
   !>
   !> @note overlap band should have been already removed from coarse grid array 
   !> of longitude and latitude, before running this function
   !>
   !> if you add cd_pos argument, you could choice to return closest point at
   !> - lower left  (ll) of the point
   !> - lower right (lr) of the point
   !> - upper left  (ul) of the point
   !> - upper right (ur) of the point
   !> - lower       (lo) of the point
   !> - upper       (up) of the point
   !> -       left  (le) of the point
   !> -       right (ri) of the point
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date February, 2015
   !> - change dichotomy method to manage ORCA grid
   !> @date February, 2016
   !> - add optional use of relative position
   !>
   !> @param[in] dd_lon0   coarse grid array of longitude
   !> @param[in] dd_lat0   coarse grid array of latitude
   !> @param[in] dd_lon1   fine   grid longitude
   !> @param[in] dd_lat1   fine   grid latitude
   !> @param[in] cd_pos    relative position of grid point from point 
   !> @param[in] dd_fill   fill value
   !> @return coarse grid indices of closest point of fine grid point
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:,:), INTENT(IN) :: dd_lon0
      REAL(dp), DIMENSION(:,:), INTENT(IN) :: dd_lat0
      REAL(dp),                 INTENT(IN) :: dd_lon1
      REAL(dp),                 INTENT(IN) :: dd_lat1
      CHARACTER(LEN=*),         INTENT(IN), OPTIONAL :: cd_pos
      REAL(dp),                 INTENT(IN), OPTIONAL :: dd_fill

      ! function
      INTEGER(i4), DIMENSION(2)            :: if_idx

      ! local variable
      INTEGER(i4)                              :: il_iinf
      INTEGER(i4)                              :: il_imid
      INTEGER(i4)                              :: il_isup
      INTEGER(i4)                              :: il_jinf
      INTEGER(i4)                              :: il_jmid
      INTEGER(i4)                              :: il_jsup
      INTEGER(i4), DIMENSION(2)                :: il_shape
      INTEGER(i4), DIMENSION(1)                :: il_ind
    
      LOGICAL                                  :: ll_north
      LOGICAL                                  :: ll_continue

      REAL(dp)                                 :: dl_lon1
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_dist
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_lon0

      ! loop indices
      !----------------------------------------------------------------

      IF( ANY( SHAPE(dd_lon0(:,:)) /= SHAPE(dd_lat0(:,:)) ) )THEN
         CALL logger_fatal("GRID GET CLOSEST: dimension of longitude and "//&
         &              " latitude differ")
      ENDIF

      il_shape(:)=SHAPE(dd_lon0(:,:))
      
      ALLOCATE( dl_lon0(il_shape(1),il_shape(2)) ) 
      
      dl_lon0(:,:) = dd_lon0(:,:)
      WHERE(dd_lon0(:,:) < 0 ) dl_lon0(:,:) = dd_lon0(:,:) + 360.

      dl_lon1 = dd_lon1
      IF( dd_lon1 < 0 ) dl_lon1 = dd_lon1 + 360.

      ! first, use dichotomy to reduce domain
      il_iinf = 1              ; il_jinf = 1
      il_isup = il_shape(1)    ; il_jsup = il_shape(2)

      il_shape(1)= il_isup - il_iinf + 1
      il_shape(2)= il_jsup - il_jinf + 1

      ll_north=.FALSE.
      ll_continue=.FALSE.

      ! avoid to use fillvalue for reduce domain on first time
      IF( PRESENT(dd_fill) )THEN
         DO WHILE( ALL(dl_lon0(il_isup,:) == dd_fill) )
            il_isup=il_isup-1
         ENDDO
         DO WHILE( ALL(dl_lon0(il_iinf,:) == dd_fill) )
            il_iinf=il_iinf+1
         ENDDO
         DO WHILE( ALL(dd_lat0(:,il_jsup) == dd_fill) )
            il_jsup=il_jsup-1
         ENDDO
         DO WHILE( ALL(dd_lat0(:,il_jinf) == dd_fill) )
            il_jinf=il_jinf+1
         ENDDO

         il_shape(1)= il_isup - il_iinf + 1
         il_shape(2)= il_jsup - il_jinf + 1

      ENDIF

      ! special case for north ORCA grid
      IF( dd_lat1 > 19. .AND. dl_lon1 < 74.  )THEN
         ll_north=.TRUE.
      ENDIF

      IF( .NOT. ll_north )THEN
         ! look for meridian 0°/360°
         il_jmid = il_jinf + INT(il_shape(2)/2)
         il_ind(:) = MAXLOC( dl_lon0(il_iinf:il_isup,il_jmid), &
         &                   dl_lon0(il_iinf:il_isup,il_jmid) <= 360._dp )

         il_imid=il_ind(1)

         IF( dl_lon1 == dl_lon0(il_imid,il_jmid) .AND. &
         &   dd_lat1 == dd_lat0(il_imid,il_jmid) )THEN

            il_iinf = il_imid ;  il_isup = il_imid
            il_jinf = il_jmid ;  il_jsup = il_jmid

         ELSE
            IF( ALL(dl_lon0(il_isup,il_jinf:il_jsup) >  dl_lon1 ) .AND. &
            &   il_imid /= il_isup )THEN
               ! 0 < lon1 < lon0(isup)
               ! point east
               il_iinf = il_imid+1
               ll_continue=.TRUE.
         
            ELSE IF( ALL(dl_lon0(il_iinf,il_jinf:il_jsup) <  dl_lon1 ) .AND. &
            &        il_imid /= il_iinf )THEN
               ! lon0(iinf) < lon1 < 360
               ! point west
               il_isup = il_imid
               ll_continue=.TRUE.

            ENDIF

            il_shape(1)= il_isup - il_iinf + 1
            il_shape(2)= il_jsup - il_jinf + 1

            il_imid = il_iinf + INT(il_shape(1)/2) 
            il_jmid = il_jinf + INT(il_shape(2)/2)

            ! exit when close enough of point
            IF( ANY(il_shape(:) < 10 ) ) ll_continue=.FALSE.
         ENDIF
      ENDIF

      !
      DO WHILE( ll_continue .AND. .NOT. ll_north )

         ll_continue=.FALSE.
         IF( dl_lon1 == dl_lon0(il_imid,il_jmid) .AND. &
         &   dd_lat1 == dd_lat0(il_imid,il_jmid) )THEN

            il_iinf = il_imid ;  il_isup = il_imid
            il_jinf = il_jmid ;  il_jsup = il_jmid

         ELSE
            IF( ALL(dl_lon0(il_imid,il_jinf:il_jsup) <  dl_lon1) )THEN    

               ! point east
               il_iinf = il_imid
               ll_continue=.TRUE.
        
            ELSE IF( ALL(dl_lon0(il_imid,il_jinf:il_jsup) >  dl_lon1) )THEN    

               ! point west
               il_isup = il_imid
               ll_continue=.TRUE.

            ENDIF

            IF( ALL(dd_lat0(il_iinf:il_isup,il_jmid) <  dd_lat1) )THEN    
               
               ! point north
               il_jinf = il_jmid
               ll_continue=.TRUE.

            ELSE IF( ALL(dd_lat0(il_iinf:il_isup,il_jmid) > dd_lat1) )THEN    

               ! point south
               il_jsup = il_jmid
               ll_continue=.TRUE.
            
            ENDIF

            il_shape(1)= il_isup - il_iinf + 1
            il_shape(2)= il_jsup - il_jinf + 1

            il_imid = il_iinf + INT(il_shape(1)/2) 
            il_jmid = il_jinf + INT(il_shape(2)/2)

            ! exit when close enough of point
            IF( ANY(il_shape(:) < 10 ) ) ll_continue=.FALSE.
         ENDIF
         
      ENDDO

      ! then find closest point by computing distances
      il_shape(1)= il_isup - il_iinf + 1
      il_shape(2)= il_jsup - il_jinf + 1

      ALLOCATE( dl_dist(il_shape(1), il_shape(2)) )

      dl_dist(:,:)=grid_distance(dl_lon0(il_iinf:il_isup,il_jinf:il_jsup), &
         &                       dd_lat0(il_iinf:il_isup,il_jinf:il_jsup), &
         &                       dl_lon1, dd_lat1 )

      IF( PRESENT(cd_pos) )THEN
         ! 
         SELECT CASE(TRIM(cd_pos))
            CASE('le')
               WHERE( dd_lat0(il_iinf:il_isup,il_jinf:il_jsup) > dd_lat1 )
                  dl_dist(:,:)=NF90_FILL_DOUBLE
               END WHERE
            CASE('ri')
               WHERE( dd_lat0(il_iinf:il_isup,il_jinf:il_jsup) < dd_lat1 )
                  dl_dist(:,:)=NF90_FILL_DOUBLE
               END WHERE
            CASE('up')
               WHERE( dl_lon0(il_iinf:il_isup,il_jinf:il_jsup) < dl_lon1 )
                  dl_dist(:,:)=NF90_FILL_DOUBLE
               END WHERE
            CASE('lo')
               WHERE( dl_lon0(il_iinf:il_isup,il_jinf:il_jsup) > dl_lon1 )
                  dl_dist(:,:)=NF90_FILL_DOUBLE
               END WHERE
            CASE('ll')
               WHERE( dl_lon0(il_iinf:il_isup,il_jinf:il_jsup) > dl_lon1 .OR. &
                    & dd_lat0(il_iinf:il_isup,il_jinf:il_jsup) > dd_lat1 )
                  dl_dist(:,:)=NF90_FILL_DOUBLE
               END WHERE
            CASE('lr')
               WHERE( dl_lon0(il_iinf:il_isup,il_jinf:il_jsup) < dl_lon1 .OR. &
                    & dd_lat0(il_iinf:il_isup,il_jinf:il_jsup) > dd_lat1 )
                  dl_dist(:,:)=NF90_FILL_DOUBLE
               END WHERE               
            CASE('ul')
               WHERE( dl_lon0(il_iinf:il_isup,il_jinf:il_jsup) > dl_lon1 .OR. &
                    & dd_lat0(il_iinf:il_isup,il_jinf:il_jsup) < dd_lat1 )
                  dl_dist(:,:)=NF90_FILL_DOUBLE
               END WHERE               
            CASE('ur')
               WHERE( dl_lon0(il_iinf:il_isup,il_jinf:il_jsup) < dl_lon1 .OR. &
                    & dd_lat0(il_iinf:il_isup,il_jinf:il_jsup) < dd_lat1 )
                  dl_dist(:,:)=NF90_FILL_DOUBLE
               END WHERE
         END SELECT
      ENDIF
      if_idx(:)=MINLOC(dl_dist(:,:),dl_dist(:,:)/=NF90_FILL_DOUBLE)

      if_idx(1)=if_idx(1)+il_iinf-1
      if_idx(2)=if_idx(2)+il_jinf-1

      DEALLOCATE( dl_dist )
      DEALLOCATE( dl_lon0 )

   END FUNCTION grid__get_closest_arr
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid_distance(dd_lon, dd_lat, dd_lonA, dd_latA) &
         & RESULT (df_dist)
   !-------------------------------------------------------------------
   !> @brief This function compute the distance between a point A and grid points.  
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_lon    grid longitude array
   !> @param[in] dd_lat    grid latitude  array
   !> @param[in] dd_lonA   longitude of point A
   !> @param[in] dd_latA   latitude  of point A
   !> @param[in] dd_fill
   !> @return array of distance between point A and grid points.
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp), DIMENSION(:,:), INTENT(IN) :: dd_lon
      REAL(dp), DIMENSION(:,:), INTENT(IN) :: dd_lat
      REAL(dp),                 INTENT(IN) :: dd_lonA
      REAL(dp),                 INTENT(IN) :: dd_latA

      ! function
      REAL(dp), DIMENSION(SIZE(dd_lon(:,:),DIM=1),&
      &                   SIZE(dd_lon(:,:),DIM=2)) :: df_dist

      ! local variable
      INTEGER(i4), DIMENSION(2) :: il_shape

      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_lon
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_lat
      REAL(dp)                                 :: dl_lonA
      REAL(dp)                                 :: dl_latA

      REAL(dp)                                 :: dl_tmp

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      IF( ANY( SHAPE(dd_lon(:,:)) /= SHAPE(dd_lat(:,:)) ) )THEN
         CALL logger_fatal("GRID DISTANCE: dimension of longitude and "//&
         &              " latitude differ")
      ENDIF
      il_shape(:)=SHAPE(dd_lon(:,:))
      
      ALLOCATE(dl_lon(il_shape(1),il_shape(2)))
      ALLOCATE(dl_lat(il_shape(1),il_shape(2)))

      dl_lon(:,:) = dd_lon(:,:)
      dl_lonA     = dd_lonA

      WHERE(dd_lon(:,:) < 0 ) dl_lon(:,:) = dd_lon(:,:) + 360.
      IF(   dd_lonA     < 0 ) dl_lonA     = dd_lonA     + 360.
      
      dl_lonA = dd_lonA * dp_deg2rad
      dl_latA = dd_latA * dp_deg2rad

      dl_lon(:,:) = dl_lon(:,:) * dp_deg2rad
      dl_lat(:,:) = dd_lat(:,:) * dp_deg2rad

      df_dist(:,:)=NF90_FILL_DOUBLE

      DO jj=1,il_shape(2)
         DO ji=1,il_shape(1)
            IF( dl_lon(ji,jj) == dl_lonA .AND. &
            &   dl_lat(ji,jj) == dl_latA )THEN
               df_dist(ji,jj)=0.0
            ELSE
               dl_tmp= SIN(dl_latA)*SIN(dl_lat(ji,jj)) + &
               &       COS(dl_latA)*COS(dl_lat(ji,jj)) * &
               &       COS(dl_lon(ji,jj)-dl_lonA)
               ! check to avoid mistake with ACOS
               IF( dl_tmp < -1.0 ) dl_tmp = -1.0
               IF( dl_tmp >  1.0 ) dl_tmp =  1.0
               df_dist(ji,jj)=ACOS(dl_tmp)*dp_rearth
            ENDIF
         ENDDO
      ENDDO

      DEALLOCATE(dl_lon)
      DEALLOCATE(dl_lat)

   END FUNCTION grid_distance
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_fine_offset_ff(td_coord0, &
         &                           id_imin0, id_jmin0, id_imax0, id_jmax0, &
         &                           td_coord1, id_rho, cd_point) &
         & RESULT (if_offset)
   !-------------------------------------------------------------------
   !> @brief This function get offset between fine grid and coarse grid.
   !>
   !> @details
   !> optionally, you could specify on which Arakawa grid point you want to
   !> work (default 'T')
   !> offset value could be 0,1,..,rho-1
   !>
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !> @date October, 2014
   !> - work on mpp file structure instead of file structure
   !>
   !> @param[in] td_coord0 coarse grid coordinate 
   !> @param[in] id_imin0  coarse grid lower left corner i-indice of fine grid domain
   !> @param[in] id_jmin0  coarse grid lower left corner j-indice of fine grid domain
   !> @param[in] id_imax0  coarse grid upper right corner i-indice of fine grid domain
   !> @param[in] id_jmax0  coarse grid upper right corner j-indice of fine grid domain
   !> @param[in] td_coord1 fine   grid coordinate 
   !> @param[in] id_rho    array of refinement factor
   !> @param[in] cd_point  Arakawa grid point
   !> @return offset array (/ (/i_offset_left,i_offset_right/),(/j_offset_lower,j_offset_upper/) /)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP)                    , INTENT(IN) :: td_coord0
      TYPE(TMPP)                    , INTENT(IN) :: td_coord1

      INTEGER(i4)                   , INTENT(IN) :: id_imin0
      INTEGER(i4)                   , INTENT(IN) :: id_jmin0
      INTEGER(i4)                   , INTENT(IN) :: id_imax0
      INTEGER(i4)                   , INTENT(IN) :: id_jmax0

      INTEGER(i4)     , DIMENSION(:), INTENT(IN), OPTIONAL :: id_rho
      CHARACTER(LEN=*)              , INTENT(IN), OPTIONAL :: cd_point

      ! function
      INTEGER(i4)     , DIMENSION(2,2)           :: if_offset

      ! local variable
      INTEGER(i4)                              :: il_imin0
      INTEGER(i4)                              :: il_jmin0
      INTEGER(i4)                              :: il_imax0
      INTEGER(i4)                              :: il_jmax0
      INTEGER(i4)                              :: il_ind
      
      INTEGER(i4), DIMENSION(:), ALLOCATABLE   :: il_rho
      
      INTEGER(i4), DIMENSION(2,2)              :: il_xghost0
      INTEGER(i4), DIMENSION(2,2)              :: il_xghost1

      CHARACTER(LEN= 1)                        :: cl_point
      CHARACTER(LEN=lc)                        :: cl_name

      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_lon0
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_lat0
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_lon1
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_lat1

      TYPE(TVAR)                               :: tl_lon0
      TYPE(TVAR)                               :: tl_lat0
      TYPE(TVAR)                               :: tl_lon1
      TYPE(TVAR)                               :: tl_lat1

      TYPE(TMPP)                               :: tl_coord0
      TYPE(TMPP)                               :: tl_coord1
      
      ! loop indices
      !----------------------------------------------------------------
      ! init
      if_offset(:,:)=-1

      ALLOCATE(il_rho(ip_maxdim))
      il_rho(:)=1
      IF( PRESENT(id_rho) ) il_rho(:)=id_rho(:)

      cl_point='T'
      IF( PRESENT(cd_point) ) cl_point=TRIM(fct_upper(cd_point))

      ! copy structure
      tl_coord0=mpp_copy(td_coord0)
      tl_coord1=mpp_copy(td_coord1)

      IF( .NOT. ASSOCIATED(tl_coord0%t_proc) .OR. &
      &   .NOT. ASSOCIATED(tl_coord1%t_proc) )THEN
         CALL logger_error("GRID GET FINE OFFSET: can not get coarse "//&
         &  "grid indices. decompsition of mpp file "//TRIM(tl_coord0%c_name)//&
         &  " and/or "//TRIM(tl_coord1%c_name)//" not defined." )
      ELSE      
         !1- Coarse grid
         ! get ghost cell factor on coarse grid
         il_xghost0(:,:)=grid_get_ghost( tl_coord0 )

         ! open mpp files
         CALL iom_mpp_open(tl_coord0)

         ! read coarse longitue and latitude
         WRITE(cl_name,*) 'longitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord0%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET FINE OFFSET: no variable "//&
            &  TRIM(cl_name)//" in file "//TRIM(tl_coord0%c_name)//". &
            &  try to use longitude.")
            WRITE(cl_name,*) 'longitude'
         ENDIF
         tl_lon0=iom_mpp_read_var(tl_coord0, TRIM(cl_name))

         WRITE(cl_name,*) 'latitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord0%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET FINE OFFSET: no variable "//&
            &  TRIM(cl_name)//" in file "//TRIM(tl_coord0%c_name)//". &
            &  try to use latitude.")
            WRITE(cl_name,*) 'latitude'
         ENDIF
         tl_lat0=iom_mpp_read_var(tl_coord0, TRIM(cl_name))

         ! close mpp files
         CALL iom_mpp_close(tl_coord0)

         CALL grid_del_ghost(tl_lon0, il_xghost0(:,:))
         CALL grid_del_ghost(tl_lat0, il_xghost0(:,:))

         ALLOCATE(dl_lon0(tl_lon0%t_dim(jp_I)%i_len, &
         &                tl_lon0%t_dim(jp_J)%i_len ))

         dl_lon0(:,:)=tl_lon0%d_value(:,:,1,1)

         ALLOCATE(dl_lat0(tl_lat0%t_dim(jp_I)%i_len, &
         &                tl_lat0%t_dim(jp_J)%i_len ))

         dl_lat0(:,:)=tl_lat0%d_value(:,:,1,1)

         ! clean
         CALL var_clean(tl_lon0)
         CALL var_clean(tl_lat0)

         ! adjust coarse grid indices
         il_imin0=id_imin0-il_xghost0(jp_I,1)
         il_imax0=id_imax0-il_xghost0(jp_I,1)

         il_jmin0=id_jmin0-il_xghost0(jp_J,1)
         il_jmax0=id_jmax0-il_xghost0(jp_J,1)

         !2- Fine grid
         ! get ghost cell factor on fine grid
         il_xghost1(:,:)=grid_get_ghost( tl_coord1 )

         ! open mpp files
         CALL iom_mpp_open(tl_coord1)

         ! read fine longitue and latitude
         WRITE(cl_name,*) 'longitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord1%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET FINE OFFSET: no variable "//&
            &  TRIM(cl_name)//" in file "//TRIM(tl_coord1%c_name)//". &
            &  try to use longitude.")
            WRITE(cl_name,*) 'longitude'
         ENDIF
         tl_lon1=iom_mpp_read_var(tl_coord1, TRIM(cl_name))

         WRITE(cl_name,*) 'latitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord1%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET FINE OFFSET: no variable "//&
            &  TRIM(cl_name)//" in file "//TRIM(tl_coord1%c_name)//". &
            &  try to use latitude.")
            WRITE(cl_name,*) 'latitude'
         ENDIF
         tl_lat1=iom_mpp_read_var(tl_coord1, TRIM(cl_name))
 
         ! close mpp files
         CALL iom_mpp_close(tl_coord1)

         CALL grid_del_ghost(tl_lon1, il_xghost1(:,:))
         CALL grid_del_ghost(tl_lat1, il_xghost1(:,:))

         ALLOCATE(dl_lon1(tl_lon1%t_dim(jp_I)%i_len, &
         &                tl_lon1%t_dim(jp_J)%i_len ))

         dl_lon1(:,:)=tl_lon1%d_value(:,:,1,1)

         ALLOCATE(dl_lat1(tl_lat1%t_dim(jp_I)%i_len, &
         &                tl_lat1%t_dim(jp_J)%i_len ))

         dl_lat1(:,:)=tl_lat1%d_value(:,:,1,1)

         ! clean
         CALL var_clean(tl_lon1)
         CALL var_clean(tl_lat1)
 
         !3- compute
         if_offset(:,:)=grid_get_fine_offset( dl_lon0(:,:), dl_lat0(:,:),&
            &                                 il_imin0, il_jmin0, &
            &                                 il_imax0, il_jmax0, &
            &                                 dl_lon1(:,:), dl_lat1(:,:),&
            &                                 id_rho(:), cl_point )
 
         DEALLOCATE(dl_lon0, dl_lat0)
         DEALLOCATE(dl_lon1, dl_lat1)
      ENDIF

      ! clean 
      CALL mpp_clean(tl_coord0)
      CALL mpp_clean(tl_coord1)
      DEALLOCATE(il_rho)

   END FUNCTION grid__get_fine_offset_ff
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_fine_offset_cf(dd_lon0, dd_lat0, &
         &                           id_imin0, id_jmin0, id_imax0, id_jmax0, &
         &                           td_coord1, id_rho, cd_point) &
         & RESULT (if_offset)
   !-------------------------------------------------------------------
   !> @brief This function get offset between fine grid and coarse grid.
   !>
   !> @details
   !> optionally, you could specify on which Arakawa grid point you want to
   !> work (default 'T')
   !> offset value could be 0,1,..,rho-1
   !>
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !> @date October, 2014
   !> - work on mpp file structure instead of file structure
   !>
   !> @param[in] dd_lon0   coarse grid longitude array 
   !> @param[in] dd_lat0   coarse grid latitude  array
   !> @param[in] id_imin0  coarse grid lower left corner i-indice of fine grid domain
   !> @param[in] id_jmin0  coarse grid lower left corner j-indice of fine grid domain
   !> @param[in] id_imax0  coarse grid upper right corner i-indice of fine grid domain
   !> @param[in] id_jmax0  coarse grid upper right corner j-indice of fine grid domain
   !> @param[in] td_coord1 fine   grid coordinate 
   !> @param[in] id_rho    array of refinement factor
   !> @param[in] cd_point  Arakawa grid point
   !> @return offset array (/ (/i_offset_left,i_offset_right/),(/j_offset_lower,j_offset_upper/) /)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)       , DIMENSION(:,:), INTENT(IN) :: dd_lon0
      REAL(dp)       , DIMENSION(:,:), INTENT(IN) :: dd_lat0
      TYPE(TMPP)                     , INTENT(IN) :: td_coord1

      INTEGER(i4)                    , INTENT(IN) :: id_imin0
      INTEGER(i4)                    , INTENT(IN) :: id_jmin0
      INTEGER(i4)                    , INTENT(IN) :: id_imax0
      INTEGER(i4)                    , INTENT(IN) :: id_jmax0

      INTEGER(i4)     , DIMENSION(:) , INTENT(IN), OPTIONAL :: id_rho
      CHARACTER(LEN=*)               , INTENT(IN), OPTIONAL :: cd_point

      ! function
      INTEGER(i4)     , DIMENSION(2,2)            :: if_offset

      ! local variable
      INTEGER(i4)                              :: il_ind
      INTEGER(i4), DIMENSION(2,2)              :: il_xghost1
      INTEGER(i4), DIMENSION(:), ALLOCATABLE   :: il_rho
      
      CHARACTER(LEN= 1)                        :: cl_point
      CHARACTER(LEN=lc)                        :: cl_name

      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_lon1
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_lat1

      TYPE(TVAR)                               :: tl_lon1
      TYPE(TVAR)                               :: tl_lat1

      TYPE(TMPP)                               :: tl_coord1
      ! loop indices
      !----------------------------------------------------------------
      ! init
      if_offset(:,:)=-1

      ALLOCATE(il_rho(ip_maxdim))
      il_rho(:)=1
      IF( PRESENT(id_rho) ) il_rho(:)=id_rho(:)

      cl_point='T'
      IF( PRESENT(cd_point) ) cl_point=TRIM(fct_upper(cd_point))

      ! copy structure
      tl_coord1=mpp_copy(td_coord1)

      IF( .NOT. ASSOCIATED(tl_coord1%t_proc) )THEN
         CALL logger_error("GRID GET FINE OFFSET: decompsition of mpp "//&
         &  "file "//TRIM(tl_coord1%c_name)//" not defined." )
      ELSE      

         ! Fine grid
         ! get ghost cell factor on fine grid
         il_xghost1(:,:)=grid_get_ghost( tl_coord1 )

         ! open mpp files
         CALL iom_mpp_open(tl_coord1)

         ! read fine longitue and latitude
         WRITE(cl_name,*) 'longitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord1%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET FINE OFFSET: no variable "//&
            &  TRIM(cl_name)//" in file "//TRIM(tl_coord1%c_name)//". &
            &  try to use longitude.")
            WRITE(cl_name,*) 'longitude'
         ENDIF
         tl_lon1=iom_mpp_read_var(tl_coord1, TRIM(cl_name))

         WRITE(cl_name,*) 'latitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord1%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET FINE OFFSET: no variable "//&
            &  TRIM(cl_name)//" in file "//TRIM(tl_coord1%c_name)//". &
            &  try to use latitude.")
            WRITE(cl_name,*) 'latitude'
         ENDIF
         tl_lat1=iom_mpp_read_var(tl_coord1, TRIM(cl_name))
 
         ! close mpp files
         CALL iom_mpp_close(tl_coord1)

         CALL grid_del_ghost(tl_lon1, il_xghost1(:,:))
         CALL grid_del_ghost(tl_lat1, il_xghost1(:,:))

         ALLOCATE(dl_lon1(tl_lon1%t_dim(jp_I)%i_len, &
         &                tl_lon1%t_dim(jp_J)%i_len ))

         dl_lon1(:,:)=tl_lon1%d_value(:,:,1,1)

         ALLOCATE(dl_lat1(tl_lat1%t_dim(jp_I)%i_len, &
         &                tl_lat1%t_dim(jp_J)%i_len ))

         dl_lat1(:,:)=tl_lat1%d_value(:,:,1,1)

         ! clean
         CALL var_clean(tl_lon1)
         CALL var_clean(tl_lat1)
      
         ! compute
         if_offset(:,:)=grid_get_fine_offset( dd_lon0(:,:), dd_lat0(:,:),&
            &                                 id_imin0, id_jmin0, &
            &                                 id_imax0, id_jmax0, &
            &                                 dl_lon1(:,:), dl_lat1(:,:),&
            &                                 id_rho(:), cl_point )
         
         DEALLOCATE(dl_lon1, dl_lat1)
      ENDIF

      ! clean 
      CALL mpp_clean(tl_coord1)
      DEALLOCATE(il_rho)

   END FUNCTION grid__get_fine_offset_cf
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_fine_offset_fc(td_coord0, &
         &                           id_imin0, id_jmin0, id_imax0, id_jmax0, &
         &                           dd_lon1, dd_lat1, &
         &                           id_rho, cd_point) &
         & RESULT (if_offset)
   !-------------------------------------------------------------------
   !> @brief This function get offset between fine grid and coarse grid.
   !>
   !> @details
   !> optionally, you could specify on which Arakawa grid point you want to
   !> work (default 'T')
   !> offset value could be 0,1,..,rho-1
   !>
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !> @date October, 2014
   !> - work on mpp file structure instead of file structure
   !>
   !> @param[in] td_coord0 coarse grid coordinate 
   !> @param[in] id_imin0  coarse grid lower left corner i-indice of fine grid domain
   !> @param[in] id_jmin0  coarse grid lower left corner j-indice of fine grid domain
   !> @param[in] id_imax0  coarse grid upper right corner i-indice of fine grid domain
   !> @param[in] id_jmax0  coarse grid upper right corner j-indice of fine grid domain
   !> @param[in] dd_lon1   fine   grid longitude array 
   !> @param[in] dd_lat1   fine   grid latitude  array
   !> @param[in] id_rho    array of refinement factor
   !> @param[in] cd_point  Arakawa grid point
   !> @return offset array (/ (/i_offset_left,i_offset_right/),(/j_offset_lower,j_offset_upper/) /)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP)                      , INTENT(IN) :: td_coord0
      REAL(dp)        , DIMENSION(:,:), INTENT(IN) :: dd_lon1
      REAL(dp)        , DIMENSION(:,:), INTENT(IN) :: dd_lat1

      INTEGER(i4)                     , INTENT(IN) :: id_imin0
      INTEGER(i4)                     , INTENT(IN) :: id_jmin0
      INTEGER(i4)                     , INTENT(IN) :: id_imax0
      INTEGER(i4)                     , INTENT(IN) :: id_jmax0

      INTEGER(i4)     , DIMENSION(:)  , INTENT(IN), OPTIONAL :: id_rho
      CHARACTER(LEN=*)                , INTENT(IN), OPTIONAL :: cd_point

      ! function
      INTEGER(i4)     , DIMENSION(2,2)             :: if_offset

      ! local variable
      INTEGER(i4)                              :: il_imin0
      INTEGER(i4)                              :: il_jmin0
      INTEGER(i4)                              :: il_imax0
      INTEGER(i4)                              :: il_jmax0
      INTEGER(i4)                              :: il_ind
      
      INTEGER(i4), DIMENSION(:), ALLOCATABLE   :: il_rho
      
      INTEGER(i4), DIMENSION(2,2)              :: il_xghost0

      CHARACTER(LEN= 1)                        :: cl_point
      CHARACTER(LEN=lc)                        :: cl_name

      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_lon0
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_lat0

      TYPE(TVAR)                               :: tl_lon0
      TYPE(TVAR)                               :: tl_lat0

      TYPE(TMPP)                               :: tl_coord0
      ! loop indices
      !----------------------------------------------------------------
      ! init
      if_offset(:,:)=-1
      ALLOCATE(il_rho(ip_maxdim))
      il_rho(:)=1
      IF( PRESENT(id_rho) ) il_rho(:)=id_rho(:)

      cl_point='T'
      IF( PRESENT(cd_point) ) cl_point=TRIM(fct_upper(cd_point))

      ! copy structure
      tl_coord0=mpp_copy(td_coord0)

      IF( .NOT. ASSOCIATED(tl_coord0%t_proc) )THEN
         CALL logger_error("GRID GET FINE OFFSET: decompsition of mpp "//&
         &  "file "//TRIM(tl_coord0%c_name)//" not defined." )
      ELSE      
         !1- Coarse grid
         ! get ghost cell factor on coarse grid
         il_xghost0(:,:)=grid_get_ghost( tl_coord0 )

         ! open mpp files
         CALL iom_mpp_open(tl_coord0)

         ! read coarse longitude and latitude
         WRITE(cl_name,*) 'longitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord0%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET FINE OFFSET: no variable "//&
            &  TRIM(cl_name)//" in file "//TRIM(tl_coord0%c_name)//". &
            &  try to use longitude.")
            WRITE(cl_name,*) 'longitude'
         ENDIF
         tl_lon0=iom_mpp_read_var(tl_coord0, TRIM(cl_name))

         WRITE(cl_name,*) 'latitude_'//TRIM(cl_point)
         il_ind=var_get_id(tl_coord0%t_proc(1)%t_var(:), cl_name)
         IF( il_ind == 0 )THEN
            CALL logger_warn("GRID GET FINE OFFSET: no variable "//&
            &  TRIM(cl_name)//" in file "//TRIM(tl_coord0%c_name)//". &
            &  try to use latitude.")
            WRITE(cl_name,*) 'latitude'
         ENDIF
         tl_lat0=iom_mpp_read_var(tl_coord0, TRIM(cl_name))
 
         ! close mpp files
         CALL iom_mpp_close(tl_coord0)

         CALL grid_del_ghost(tl_lon0, il_xghost0(:,:))
         CALL grid_del_ghost(tl_lat0, il_xghost0(:,:))


         ALLOCATE(dl_lon0(tl_lon0%t_dim(jp_I)%i_len, &
         &                tl_lon0%t_dim(jp_J)%i_len ))

         dl_lon0(:,:)=tl_lon0%d_value(:,:,1,1)

         ALLOCATE(dl_lat0(tl_lat0%t_dim(jp_I)%i_len, &
         &                tl_lat0%t_dim(jp_J)%i_len ))

         dl_lat0(:,:)=tl_lat0%d_value(:,:,1,1)

         ! clean
         CALL var_clean(tl_lon0)
         CALL var_clean(tl_lat0)

         ! adjust coarse grid indices
         il_imin0=id_imin0-il_xghost0(jp_I,1)
         il_imax0=id_imax0-il_xghost0(jp_I,1)

         il_jmin0=id_jmin0-il_xghost0(jp_J,1)
         il_jmax0=id_jmax0-il_xghost0(jp_J,1)

         !3- compute
         if_offset(:,:)=grid_get_fine_offset( dl_lon0(:,:), dl_lat0(:,:),&
            &                                 il_imin0, il_jmin0, &
            &                                 il_imax0, il_jmax0, &
            &                                 dd_lon1(:,:), dd_lat1(:,:),&
            &                                 id_rho(:), cl_point )
         
         DEALLOCATE(dl_lon0, dl_lat0)
      ENDIF

      ! clean
      CALL mpp_clean(tl_coord0)
      DEALLOCATE(il_rho)

   END FUNCTION grid__get_fine_offset_fc
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_fine_offset_cc(dd_lon0, dd_lat0, &
         &                           id_imin0, id_jmin0, id_imax0, id_jmax0, &
         &                           dd_lon1, dd_lat1, id_rho, cd_point) &
         & RESULT (if_offset)
   !-------------------------------------------------------------------
   !> @brief This function get offset between fine grid and coarse grid.
   !>
   !> @details
   !> offset value could be 0,1,..,rho-1
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014 
   !> - rename from grid_get_fine_offset
   !> @date May, 2015 
   !> - improve way to find offset
   !> @date July, 2015
   !> - manage case close to greenwich meridian
   !> @date February, 2016
   !> - use grid_get_closest to assess offset
   !> - use delta (lon or lat)
   !> - manage cases for T,U,V or F point, with even or odd refinment
   !> - check lower left(upper right) fine grid point inside lower left(upper
   !> right) coarse grid cell.
   !> 
   !> @todo check case close from North fold.
   !>
   !> @param[in] dd_lon0   coarse grid longitude array 
   !> @param[in] dd_lat0   coarse grid latitude  array
   !> @param[in] id_imin0  coarse grid lower left corner i-indice of fine grid domain
   !> @param[in] id_jmin0  coarse grid lower left corner j-indice of fine grid domain
   !> @param[in] id_imax0  coarse grid upper right corner i-indice of fine grid domain
   !> @param[in] id_jmax0  coarse grid upper right corner j-indice of fine grid domain
   !> @param[in] dd_lon1   fine   grid longitude array 
   !> @param[in] dd_lat1   fine   grid latitude  array
   !> @param[in] id_rho    array of refinement factor
   !> @param[in] cd_point  Arakawa grid point
   !> @return offset array (/ (/i_offset_left,i_offset_right/),(/j_offset_lower,j_offset_upper/) /)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp)        , DIMENSION(:,:), INTENT(IN) :: dd_lon0
      REAL(dp)        , DIMENSION(:,:), INTENT(IN) :: dd_lat0
      REAL(dp)        , DIMENSION(:,:), INTENT(IN) :: dd_lon1
      REAL(dp)        , DIMENSION(:,:), INTENT(IN) :: dd_lat1

      INTEGER(i4)     ,                 INTENT(IN) :: id_imin0
      INTEGER(i4)     ,                 INTENT(IN) :: id_jmin0
      INTEGER(i4)     ,                 INTENT(IN) :: id_imax0
      INTEGER(i4)     ,                 INTENT(IN) :: id_jmax0

      INTEGER(i4)     , DIMENSION(:)  , INTENT(IN) :: id_rho
      CHARACTER(LEN=*)                , INTENT(IN), OPTIONAL :: cd_point

      ! function
      INTEGER(i4)     , DIMENSION(2,2)             :: if_offset

      ! local variable
      CHARACTER(LEN= 1)                        :: cl_point

      INTEGER(i4)                              :: i1
      INTEGER(i4)                              :: i2
      INTEGER(i4)                              :: j1
      INTEGER(i4)                              :: j2

      INTEGER(i4), DIMENSION(2)                :: il_shape0
      INTEGER(i4), DIMENSION(2)                :: il_shape1

      INTEGER(i4), DIMENSION(2)                :: il_ind

      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_lon0
      REAL(dp)   , DIMENSION(:,:), ALLOCATABLE :: dl_lon1

      REAL(dp)                                 :: dl_lonmax0
      REAL(dp)                                 :: dl_latmax0
      REAL(dp)                                 :: dl_lonmin0
      REAL(dp)                                 :: dl_latmin0

      REAL(dp)                                 :: dl_lon0F
      REAL(dp)                                 :: dl_lat0F
      REAL(dp)                                 :: dl_dlon
      REAL(dp)                                 :: dl_dlat

      LOGICAL    , DIMENSION(2)                :: ll_even
      LOGICAL                                  :: ll_greenwich
      
      ! loop indices
      INTEGER(i4) :: ii
      INTEGER(i4) :: ij
      !----------------------------------------------------------------
      IF( ANY( SHAPE(dd_lon0(:,:)) /= SHAPE(dd_lat0(:,:)) ) )THEN
         CALL logger_fatal("GRID GET FINE OFFSET: dimension of coarse "//&
         &              "longitude and latitude differ")
      ENDIF

      IF( ANY( SHAPE(dd_lon1(:,:)) /= SHAPE(dd_lat1(:,:)) ) )THEN
         CALL logger_fatal("GRID GET FINE OFFSET: dimension of fine "//&
         &              "longitude and latitude differ")
      ENDIF

      ll_even(:)=(/ (MOD(id_rho(jp_I),2)==0), (MOD(id_rho(jp_J),2)==0) /)

      cl_point='T'
      IF( PRESENT(cd_point) ) cl_point=TRIM(fct_upper(cd_point))

      il_shape0(:)=SHAPE(dd_lon0(:,:))
      ALLOCATE( dl_lon0(il_shape0(1),il_shape0(2)) )

      il_shape1(:)=SHAPE(dd_lon1(:,:))
      ALLOCATE( dl_lon1(il_shape1(1),il_shape1(2)) )

      dl_lon0(:,:)=dd_lon0(:,:)
      WHERE( dd_lon0(:,:) < 0 ) dl_lon0(:,:)=dd_lon0(:,:)+360.

      dl_lon1(:,:)=dd_lon1(:,:)
      WHERE( dd_lon1(:,:) < 0 ) dl_lon1(:,:)=dd_lon1(:,:)+360.

      ! init
      if_offset(:,:)=-1
      ll_greenwich=.FALSE.

      IF( il_shape1(jp_J) == 1 )THEN
 
         if_offset(jp_J,:)=((id_rho(jp_J)-1)/2)

         !!! work on i-direction
         !!! look for i-direction left offset
         i1=1 ; i2=MIN((id_rho(jp_I)+2),il_shape1(jp_I))
         j1=1 ; j2=1

         ! check if cross greenwich meridien
         IF( minval(dl_lon0(id_imin0:id_imin0+1,id_jmin0))<5. .OR. &
           & maxval(dl_lon0(id_imin0:id_imin0+1,id_jmin0))>355. )THEN
            ! close to greenwich meridien
            ll_greenwich=.TRUE.
            ! 0:360 => -180:180
            WHERE( dl_lon0(id_imin0:id_imin0+1,id_jmin0) > 180. )
               dl_lon0(id_imin0:id_imin0+1,id_jmin0) = &
                  & dl_lon0(id_imin0:id_imin0+1,id_jmin0)-360.
            END WHERE

            WHERE( dl_lon1(i1:i2,j1:j2) > 180. )
               dl_lon1(i1:i2,j1:j2)=dl_lon1(i1:i2,j1:j2)-360.
            END WHERE
         ENDIF

         ! max lognitude of the left cell
         dl_lonmax0=dl_lon0(id_imin0+1,id_jmin0)
         IF( dl_lon1(1,1) < dl_lonmax0 )THEN

            !!!!! i-direction !!!!!
            IF( ll_even(jp_I) )THEN
               ! even
               SELECT CASE(TRIM(cl_point))
                  CASE('F','U')
                     dl_dlon= ( dl_lon0(id_imin0+1,id_jmin0) -   &
                        &       dl_lon0(id_imin0  ,id_jmin0) ) / &
                        &     ( 2.*id_rho(jp_I) )
                  CASE DEFAULT
                     dl_dlon=0
               END SELECT
            ELSE
               ! odd
               dl_dlon= ( dl_lon0(id_imin0+1,id_jmin0) -   &
                  &       dl_lon0(id_imin0  ,id_jmin0) ) / &
                  &     ( 2.*id_rho(jp_I) )
            ENDIF

            dl_lon0F= dl_lon0(id_imin0+1,id_jmin0) + dl_dlon
            dl_lat0F= dd_lat0(id_imin0+1,id_jmin0)

            il_ind(:)=grid_get_closest( dl_lon1(i1:i2,j1:j2), dd_lat1(i1:i2,j1:j2), &
            &                           dl_lon0F, dl_lat0F, 'le' )
      
            ii=il_ind(1)

            !!!!! i-direction !!!!!
            IF( ll_even(jp_I) )THEN
               ! even
               SELECT CASE(TRIM(cl_point))
                  CASE('T','V')
                     if_offset(jp_I,1)=id_rho(jp_I)-ii
                  CASE DEFAULT !'F','U' 
                     if_offset(jp_I,1)=(id_rho(jp_I)+1)-ii
               END SELECT
            ELSE
               ! odd
               if_offset(jp_I,1)=(id_rho(jp_I)+1)-ii
            ENDIF

         ELSE
            CALL logger_error("GRID GET FINE OFFSET: coarse grid indices do "//&
            &                 " not match fine grid left corner.")
         ENDIF

         IF( ll_greenwich )THEN
            ! close to greenwich meridien
            ll_greenwich=.FALSE.
            ! -180:180 => 0:360
            WHERE( dl_lon0(id_imin0:id_imin0+1,id_jmin0) < 0. )
               dl_lon0(id_imin0:id_imin0+1,id_jmin0) = &
                  & dl_lon0(id_imin0:id_imin0+1,id_jmin0)+360.
            END WHERE

            WHERE( dl_lon1(i1:i2,j1:j2) < 0. )
               dl_lon1(i1:i2,j1:j2)=dl_lon1(i1:i2,j1:j2)+360.
            END WHERE
         ENDIF

         !!!!!! look for i-direction right offset !!!!!!
         i1=MAX(1,il_shape1(jp_I)-(id_rho(jp_I)+2)+1) ; i2=il_shape1(jp_I)
         j1=1                                         ; j2=1

         ! check if cross greenwich meridien
         IF( minval(dl_lon0(id_imax0-1:id_imax0,id_jmin0))<5. .OR. &
           & maxval(dl_lon0(id_imax0-1:id_imax0,id_jmin0))>355. )THEN
            ! close to greenwich meridien
            ll_greenwich=.TRUE.
            ! 0:360 => -180:180
            WHERE( dl_lon0(id_imax0-1:id_imax0,id_jmin0) > 180. )
               dl_lon0(id_imax0-1:id_imax0,id_jmin0) = &
                  & dl_lon0(id_imax0-1:id_imax0,id_jmin0)-360.
            END WHERE

            WHERE( dl_lon1(i1:i2,j1:j2) > 180. )
               dl_lon1(i1:i2,j1:j2)=dl_lon1(i1:i2,j1:j2)-360.
            END WHERE
         ENDIF

         ! min lognitude of the right cell
         dl_lonmin0=dl_lon0(id_imax0-1,id_jmin0)
         IF( dl_lon1(il_shape1(jp_I),il_shape1(jp_J)) > dl_lonmin0 )THEN

            !!!!! i-direction !!!!!
            IF( ll_even(jp_I) )THEN
               ! even
               SELECT CASE(TRIM(cl_point))
                  CASE('F','U')
                     dl_dlon= ( dl_lon0(id_imax0  ,id_jmin0) -   &
                        &       dl_lon0(id_imax0-1,id_jmin0) ) / &
                        &     ( 2.*id_rho(jp_I) )
                  CASE DEFAULT
                     dl_dlon=0
               END SELECT
            ELSE
               ! odd
               dl_dlon= ( dl_lon0(id_imax0  ,id_jmin0) -   &
                  &       dl_lon0(id_imax0-1,id_jmin0) ) / &
                  &     ( 2.*id_rho(jp_I) )
            ENDIF

            dl_lon0F= dl_lon0(id_imax0-1,id_jmin0) - dl_dlon
            dl_lat0F= dd_lat0(id_imax0-1,id_jmin0)

            il_ind(:)=grid_get_closest( dl_lon1(i1:i2,j1:j2), dd_lat1(i1:i2,j1:j2), &
            &                           dl_lon0F, dl_lat0F, 'ri' )

            ii=(MIN(il_shape1(jp_I),(id_rho(jp_I)+2))-il_ind(1)+1)

            !!!!! i-direction !!!!!
            IF( ll_even(jp_I) )THEN
               ! even
               SELECT CASE(TRIM(cl_point))
                  CASE('T','V')
                     if_offset(jp_I,2)=id_rho(jp_I)-ii
                  CASE DEFAULT !'F','U' 
                     if_offset(jp_I,2)=(id_rho(jp_I)+1)-ii
               END SELECT
            ELSE
               ! odd
               if_offset(jp_I,2)=(id_rho(jp_I)+1)-ii
            ENDIF

         ELSE
            CALL logger_error("GRID GET FINE OFFSET: coarse grid indices do "//&
            &                 " not match fine grid right corner.")
         ENDIF

         IF( ll_greenwich )THEN
            ! close to greenwich meridien
            ll_greenwich=.FALSE.
            ! -180:180 => 0:360
            WHERE( dl_lon0(id_imax0-1:id_imax0,id_jmin0) < 0. )
               dl_lon0(id_imax0-1:id_imax0,id_jmin0) = &
                  & dl_lon0(id_imax0-1:id_imax0,id_jmin0)+360.
            END WHERE

            WHERE( dl_lon1(i1:i2,j1:j2) < 0. )
               dl_lon1(i1:i2,j1:j2)=dl_lon1(i1:i2,j1:j2)+360.
            END WHERE
         ENDIF

      ELSEIF( il_shape1(jp_I) == 1 )THEN
         
         if_offset(jp_I,:)=((id_rho(jp_I)-1)/2)
         
         !!! work on j-direction
         !!! look for j-direction lower offset 
         i1=1 ; i2=1
         j1=1 ; j2=MIN((id_rho(jp_J)+2),il_shape1(jp_J))


         ! max latitude of the lower cell
         dl_latmax0=dd_lat0(id_imin0,id_jmin0+1)
         IF( dd_lat1(1,1) < dl_latmax0 )THEN

            IF( ll_even(jp_J) )THEN
               ! even
               SELECT CASE(TRIM(cl_point))
                  CASE('F','V')
                     dl_dlat= ( dd_lat0(id_imin0,id_jmin0+1) -   &
                        &       dd_lat0(id_imin0,id_jmin0  ) ) / &
                        &     ( 2.*id_rho(jp_J) )
                  CASE DEFAULT
                     dl_dlat=0
               END SELECT
            ELSE
               ! odd
               dl_dlat= ( dd_lat0(id_imin0,id_jmin0+1) -   &
                  &       dd_lat0(id_imin0,id_jmin0  ) ) / &
                  &     ( 2.*id_rho(jp_J) )
            ENDIF

            dl_lon0F= dl_lon0(id_imin0,id_jmin0+1)
            dl_lat0F= dd_lat0(id_imin0,id_jmin0+1) + dl_dlat 
            
            il_ind(:)=grid_get_closest( dl_lon1(i1:i2,j1:j2), dd_lat1(i1:i2,j1:j2), &
            &                           dl_lon0F, dl_lat0F, 'lo' )

            ij=il_ind(2)

            !!!!! j-direction !!!!!
            IF( ll_even(jp_J) )THEN
               ! even
               SELECT CASE(TRIM(cl_point))
                  CASE('T','V')
                     if_offset(jp_J,1)=id_rho(jp_J)-ij
                  CASE DEFAULT !'F','U' 
                     if_offset(jp_J,1)=(id_rho(jp_J)+1)-ij
               END SELECT
            ELSE
               ! odd
               if_offset(jp_J,1)=(id_rho(jp_J)+1)-ij
            ENDIF

         ELSE
            CALL logger_error("GRID GET FINE OFFSET: coarse grid indices do "//&
            &                 " not match fine grid lower corner.")
         ENDIF

         !!! look for j-direction upper offset 
         i1=1                                         ; i2=1
         j1=MAX(1,il_shape1(jp_J)-(id_rho(jp_J)+2)+1) ; j2=il_shape1(jp_J)

         ! min latitude of the upper cell
         dl_latmin0=dd_lat0(id_imin0,id_jmax0-1)
         IF( dd_lat1(il_shape1(jp_I),il_shape1(jp_J)) > dl_latmin0 )THEN

            IF( ll_even(jp_J) )THEN
               ! even
               SELECT CASE(TRIM(cl_point))
                  CASE('F','V')
                     dl_dlat= ( dd_lat0(id_imin0,id_jmax0  ) -   &
                        &       dd_lat0(id_imin0,id_jmax0-1) ) / &
                        &     ( 2.*id_rho(jp_J) )
                  CASE DEFAULT
                     dl_dlat=0
               END SELECT
            ELSE
               ! odd
               dl_dlat= ( dd_lat0(id_imin0,id_jmax0  ) -   &
                  &       dd_lat0(id_imin0,id_jmax0-1) ) / &
                  &     ( 2*id_rho(jp_J) )
            ENDIF

            dl_lon0F= dl_lon0(id_imin0,id_jmax0-1) 
            dl_lat0F= dd_lat0(id_imin0,id_jmax0-1) - dl_dlat
            
            il_ind(:)=grid_get_closest( dl_lon1(i1:i2,j1:j2), dd_lat1(i1:i2,j1:j2), &
            &                           dl_lon0F, dl_lat0F, 'up' )

            ij=(MIN(il_shape1(jp_J),(id_rho(jp_J)+2))-il_ind(2)+1)

            !!!!! j-direction !!!!!
            IF( ll_even(jp_J) )THEN
               ! even
               SELECT CASE(TRIM(cl_point))
                  CASE('T','U')
                     if_offset(jp_J,2)=id_rho(jp_J)-ij
                  CASE DEFAULT !'F','V'
                     if_offset(jp_J,2)=(id_rho(jp_J)+1)-ij
               END SELECT
            ELSE
               ! odd
               if_offset(jp_J,2)=(id_rho(jp_J)+1)-ij
            ENDIF

         ELSE
            CALL logger_error("GRID GET FINE OFFSET: coarse grid indices do "//&
            &                 " not match fine grid upper corner.")
         ENDIF

      ELSE ! il_shape1(1) > 1 .AND. il_shape1(2) > 1 

         !!!!!! look for lower left offset !!!!!!
         i1=1 ; i2=MIN((id_rho(jp_I)+2),il_shape1(jp_I))
         j1=1 ; j2=MIN((id_rho(jp_J)+2),il_shape1(jp_J))

         ! check if cross greenwich meridien
         IF( minval(dl_lon0(id_imin0:id_imin0+1,id_jmin0:id_jmin0+1))<5. .OR. &
           & maxval(dl_lon0(id_imin0:id_imin0+1,id_jmin0:id_jmin0+1))>355. )THEN
            ! close to greenwich meridien
            ll_greenwich=.TRUE.
            ! 0:360 => -180:180
            WHERE( dl_lon0(id_imin0:id_imin0+1,id_jmin0:id_jmin0+1) > 180. )
               dl_lon0(id_imin0:id_imin0+1,id_jmin0:id_jmin0+1) = &
                  & dl_lon0(id_imin0:id_imin0+1,id_jmin0:id_jmin0+1)-360.
            END WHERE

            WHERE( dl_lon1(i1:i2,j1:j2) > 180. )
               dl_lon1(i1:i2,j1:j2)=dl_lon1(i1:i2,j1:j2)-360.
            END WHERE
         ENDIF

         ! max longitude of the lower left cell
         dl_lonmax0=MAX(dl_lon0(id_imin0+1,id_jmin0),dl_lon0(id_imin0+1,id_jmin0+1))
         ! max latitude of the lower left cell
         dl_latmax0=MAX(dd_lat0(id_imin0,id_jmin0+1),dd_lat0(id_imin0+1,id_jmin0+1))
         IF( dl_lon1(1,1) < dl_lonmax0 .AND. &
           & dd_lat1(1,1) < dl_latmax0 )THEN

            !!!!! i-direction !!!!!
            IF( ll_even(jp_I) )THEN
               ! even
               SELECT CASE(TRIM(cl_point))
                  CASE('F','U')
                     dl_dlon= ( dl_lon0(id_imin0+1,id_jmin0+1) -   &
                        &       dl_lon0(id_imin0  ,id_jmin0+1) ) / &
                        &     ( 2.*id_rho(jp_I) )
                  CASE DEFAULT
                     dl_dlon=0
               END SELECT
            ELSE
               ! odd
               dl_dlon= ( dl_lon0(id_imin0+1,id_jmin0+1) -   &
                  &       dl_lon0(id_imin0  ,id_jmin0+1) ) / &
                  &     ( 2.*id_rho(jp_I) )
            ENDIF

            !!!!! j-direction !!!!!
            IF( ll_even(jp_J) )THEN
               ! even
               SELECT CASE(TRIM(cl_point))
                  CASE('F','V')
                     dl_dlat= ( dd_lat0(id_imin0+1,id_jmin0+1) -   &
                        &       dd_lat0(id_imin0+1,id_jmin0  ) ) / &
                        &     ( 2.*id_rho(jp_J) )
                  CASE DEFAULT
                     dl_dlat=0
               END SELECT
            ELSE
               ! odd
               dl_dlat= ( dd_lat0(id_imin0+1,id_jmin0+1) -   &
                  &       dd_lat0(id_imin0+1,id_jmin0  ) ) / &
                  &     ( 2.*id_rho(jp_J) )
            ENDIF

            dl_lon0F= dl_lon0(id_imin0+1,id_jmin0+1) + dl_dlon
            dl_lat0F= dd_lat0(id_imin0+1,id_jmin0+1) + dl_dlat

            il_ind(:)=grid_get_closest( dl_lon1(i1:i2,j1:j2), dd_lat1(i1:i2,j1:j2), &
            &                           dl_lon0F, dl_lat0F, 'll' )

            ii=il_ind(1)
            ij=il_ind(2)

            !!!!! i-direction !!!!!
            IF( ll_even(jp_I) )THEN
               ! even
               SELECT CASE(TRIM(cl_point))
                  CASE('T','V')
                     if_offset(jp_I,1)=id_rho(jp_I)-ii
                  CASE DEFAULT !'F','U' 
                     if_offset(jp_I,1)=(id_rho(jp_I)+1)-ii
               END SELECT
            ELSE
               ! odd
               if_offset(jp_I,1)=(id_rho(jp_I)+1)-ii
            ENDIF

            !!!!! j-direction !!!!!
            IF( ll_even(jp_J) )THEN
               ! even
               SELECT CASE(TRIM(cl_point))
                  CASE('T','U')
                     if_offset(jp_J,1)=id_rho(jp_J)-ij
                  CASE DEFAULT !'F','V'
                     if_offset(jp_J,1)=(id_rho(jp_J)+1)-ij
               END SELECT
            ELSE
               ! odd
               if_offset(jp_J,1)=(id_rho(jp_J)+1)-ij
            ENDIF

         ELSE
            CALL logger_error("GRID GET FINE OFFSET: coarse grid indices do"//&
            &                 " not match fine grid lower left corner.")
         ENDIF

         IF( ll_greenwich )THEN
            ! close to greenwich meridien
            ll_greenwich=.FALSE.
            ! -180:180 => 0:360 
            WHERE( dl_lon0(id_imin0:id_imin0+1,id_jmin0:id_jmin0+1) < 0. )
               dl_lon0(id_imin0:id_imin0+1,id_jmin0:id_jmin0+1) = &
                  & dl_lon0(id_imin0:id_imin0+1,id_jmin0:id_jmin0+1)+360.
            END WHERE

            WHERE( dl_lon1(i1:i2,j1:j2) < 0. )
               dl_lon1(i1:i2,j1:j2)=dl_lon1(i1:i2,j1:j2)+360.
            END WHERE
         ENDIF

         !!!!!! look for upper right offset !!!!!!
         i1=MAX(1,il_shape1(jp_I)-(id_rho(jp_I)+2)+1) ; i2=il_shape1(jp_I)
         j1=MAX(1,il_shape1(jp_J)-(id_rho(jp_J)+2)+1) ; j2=il_shape1(jp_J)

         ! check if cross greenwich meridien
         IF( minval(dl_lon0(id_imax0-1:id_imax0,id_jmax0-1:id_jmax0))<5. .OR. &
           & maxval(dl_lon0(id_imax0-1:id_imax0,id_jmax0-1:id_jmax0))>355. )THEN
            ! close to greenwich meridien
            ll_greenwich=.TRUE.
            ! 0:360 => -180:180
            WHERE( dl_lon0(id_imax0-1:id_imax0,id_jmax0-1:id_jmax0) > 180. )
               dl_lon0(id_imax0-1:id_imax0,id_jmax0-1:id_jmax0) = &
                  & dl_lon0(id_imax0-1:id_imax0,id_jmax0-1:id_jmax0)-360.
            END WHERE

            WHERE( dl_lon1(i1:i2,j1:j2) > 180. )
               dl_lon1(i1:i2,j1:j2)=dl_lon1(i1:i2,j1:j2)-360.
            END WHERE
         ENDIF

         ! min latitude of the upper right cell
         dl_lonmin0=MIN(dl_lon0(id_imax0-1,id_jmax0-1),dl_lon0(id_imax0-1,id_jmax0))
         ! min latitude of the upper right cell
         dl_latmin0=MIN(dd_lat0(id_imax0-1,id_jmax0-1),dd_lat0(id_imax0,id_jmax0-1))
         IF( dl_lon1(il_shape1(jp_I),il_shape1(jp_J)) > dl_lonmin0 .AND. &
           & dd_lat1(il_shape1(jp_I),il_shape1(jp_J)) > dl_latmin0 )THEN

            !!!!! i-direction !!!!!
            IF( ll_even(jp_I) )THEN
               ! even
               SELECT CASE(TRIM(cl_point))
                  CASE('F','U')
                     dl_dlon= ( dl_lon0(id_imax0  ,id_jmax0-1) -   &
                        &       dl_lon0(id_imax0-1,id_jmax0-1) ) / &
                        &     ( 2.*id_rho(jp_I) )
                  CASE DEFAULT
                     dl_dlon=0
               END SELECT               
            ELSE
               ! odd
               dl_dlon= ( dl_lon0(id_imax0  ,id_jmax0-1) -   &
                  &       dl_lon0(id_imax0-1,id_jmax0-1) ) / &
                  &     ( 2*id_rho(jp_I) )
            ENDIF

            !!!!! j-direction !!!!!
            IF( ll_even(jp_J) )THEN
               ! even
               SELECT CASE(TRIM(cl_point))
                  CASE('F','V')
                     dl_dlat= ( dd_lat0(id_imax0-1,id_jmax0  ) -   &
                        &       dd_lat0(id_imax0-1,id_jmax0-1) ) / &
                        &     ( 2.*id_rho(jp_J) )
                  CASE DEFAULT
                     dl_dlat=0
               END SELECT
            ELSE
               ! odd
               dl_dlat= ( dd_lat0(id_imax0-1,id_jmax0  ) -   &
                  &       dd_lat0(id_imax0-1,id_jmax0-1) ) / &
                  &     ( 2*id_rho(jp_J) )
            ENDIF

            dl_lon0F= dl_lon0(id_imax0-1,id_jmax0-1) - dl_dlon
            dl_lat0F= dd_lat0(id_imax0-1,id_jmax0-1) - dl_dlat

            il_ind(:)=grid_get_closest( dl_lon1(i1:i2,j1:j2), dd_lat1(i1:i2,j1:j2), &
               &                        dl_lon0F, dl_lat0F, 'ur' )

            ii=(MIN(il_shape1(jp_I),(id_rho(jp_I)+2))-il_ind(1)+1)
            ij=(MIN(il_shape1(jp_J),(id_rho(jp_J)+2))-il_ind(2)+1)

            !!!!! i-direction !!!!!
            IF( ll_even(jp_I) )THEN
               ! even
               SELECT CASE(TRIM(cl_point))
                  CASE('T','V')
                     if_offset(jp_I,2)=id_rho(jp_I)-ii
                  CASE DEFAULT !'F','U' 
                     if_offset(jp_I,2)=(id_rho(jp_I)+1)-ii
               END SELECT
            ELSE
               ! odd
               if_offset(jp_I,2)=(id_rho(jp_I)+1)-ii
            ENDIF

            !!!!! j-direction !!!!!
            IF( ll_even(jp_J) )THEN
               ! even
               SELECT CASE(TRIM(cl_point))
                  CASE('T','U')
                     if_offset(jp_J,2)=id_rho(jp_J)-ij
                  CASE DEFAULT !'F','V'
                     if_offset(jp_J,2)=(id_rho(jp_J)+1)-ij
               END SELECT
            ELSE
               ! odd
               if_offset(jp_J,2)=(id_rho(jp_J)+1)-ij
            ENDIF

         ELSE
            CALL logger_error("GRID GET FINE OFFSET: coarse grid indices do"//&
            &                 " not match fine grid upper right corner.")
         ENDIF

         IF( ll_greenwich )THEN
            ! close to greenwich meridien
            ll_greenwich=.FALSE.
            ! -180:180 => 0:360
            WHERE( dl_lon0(id_imax0-1:id_imax0,id_jmax0-1:id_jmax0) < 0. )
               dl_lon0(id_imax0-1:id_imax0,id_jmax0-1:id_jmax0) = &
                  & dl_lon0(id_imax0-1:id_imax0,id_jmax0-1:id_jmax0)+360.
            END WHERE

            WHERE( dl_lon1(i1:i2,j1:j2) < 0. )
               dl_lon1(i1:i2,j1:j2)=dl_lon1(i1:i2,j1:j2)+360.
            END WHERE
         ENDIF

      ENDIF

      DEALLOCATE( dl_lon0 )
      DEALLOCATE( dl_lon1 )

      IF( ANY(if_offset(:,:)==-1) )THEN
         CALL logger_fatal("GRID GET FINE OFFSET: can not found "//&
         &                 " offset between coarse and fine grid.")
      ENDIF

   END FUNCTION grid__get_fine_offset_cc
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_check_coincidence(td_coord0, td_coord1, &
         &                           id_imin0, id_imax0, &
         &                           id_jmin0, id_jmax0, &
         &                           id_rho)
   !-------------------------------------------------------------------
   !> @brief This subroutine check fine and coarse grid coincidence.
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date November, 2013- Initial Version
   !> @date October, 2014
   !> - work on mpp file structure instead of file structure
   !> @date February, 2016
   !> - use F-point to check coincidence for even refinment
   !> - use F-point estimation, if can not read it.
   !>
   !> @param[in] td_coord0 coarse grid coordinate file structure 
   !> @param[in] td_coord1 fine   grid coordinate file structure 
   !> @param[in] id_imin0  coarse grid lower left  corner i-indice of fine grid domain 
   !> @param[in] id_imax0  coarse grid upper right corner i-indice of fine grid domain
   !> @param[in] id_jmin0  coarse grid lower left  corner j-indice of fine grid domain 
   !> @param[in] id_jmax0  coarse grid upper right corner j-indice of fine grid domain  
   !> @param[in] id_rho    array of refinement factor 
   !-------------------------------------------------------------------

      IMPLICIT NONE
      
      ! Argument      
      TYPE(TMPP)               , INTENT(IN) :: td_coord0
      TYPE(TMPP)               , INTENT(IN) :: td_coord1
      INTEGER(i4)              , INTENT(IN) :: id_imin0
      INTEGER(i4)              , INTENT(IN) :: id_imax0
      INTEGER(i4)              , INTENT(IN) :: id_jmin0
      INTEGER(i4)              , INTENT(IN) :: id_jmax0
      INTEGER(i4), DIMENSION(:), INTENT(IN) :: id_rho

      ! local variable
      INTEGER(i4)               :: il_imid1
      INTEGER(i4)               :: il_jmid1
      
      INTEGER(i4)               :: il_ew0
      INTEGER(i4)               :: il_ew1

      INTEGER(i4)               :: il_ind

      INTEGER(i4)               :: il_imin1
      INTEGER(i4)               :: il_imax1
      INTEGER(i4)               :: il_jmin1
      INTEGER(i4)               :: il_jmax1

      INTEGER(i4), DIMENSION(2) :: il_ind0
      INTEGER(i4), DIMENSION(2) :: il_ind1

      INTEGER(i4), DIMENSION(2) :: il_ill1
      INTEGER(i4), DIMENSION(2) :: il_ilr1
      INTEGER(i4), DIMENSION(2) :: il_iul1
      INTEGER(i4), DIMENSION(2) :: il_iur1

      REAL(dp)                  :: dl_lon0F
      REAL(dp)                  :: dl_lat0F
      REAL(dp)                  :: dl_lon0
      REAL(dp)                  :: dl_lat0
      REAL(dp)                  :: dl_lon1F
      REAL(dp)                  :: dl_lat1F
      REAL(dp)                  :: dl_lon1
      REAL(dp)                  :: dl_lat1

      REAL(dp)                  :: dl_delta

      LOGICAL                   :: ll_coincidence
      LOGICAL                   :: ll_even
      LOGICAL                   :: ll_grid0F
      LOGICAL                   :: ll_grid1F

      TYPE(TVAR)                :: tl_lon0
      TYPE(TVAR)                :: tl_lat0
      TYPE(TVAR)                :: tl_lon0F
      TYPE(TVAR)                :: tl_lat0F
      TYPE(TVAR)                :: tl_lon1
      TYPE(TVAR)                :: tl_lat1
      TYPE(TVAR)                :: tl_lon1F
      TYPE(TVAR)                :: tl_lat1F

      TYPE(TMPP)                :: tl_coord0
      TYPE(TMPP)                :: tl_coord1

      TYPE(TDOM)                :: tl_dom0

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      ll_coincidence=.TRUE.

      ll_even=.FALSE.
      IF( MOD(id_rho(jp_I)*id_rho(jp_J),2) == 0 )THEN
         ll_even=.TRUE.
      ENDIF

      ! copy structure
      tl_coord0=mpp_copy(td_coord0)

      ! compute domain
      tl_dom0=dom_init( tl_coord0,         &
      &                 id_imin0, id_imax0,&
      &                 id_jmin0, id_jmax0 )

      ! open mpp files
      CALL iom_dom_open(tl_coord0, tl_dom0)

      ! read variable value on domain
      il_ind=var_get_index(tl_coord0%t_proc(1)%t_var(:), 'longitude_T')
      IF( il_ind /= 0 )THEN
         tl_lon0=iom_dom_read_var(tl_coord0,'longitude_T',tl_dom0)
      ELSE
         tl_lon0=iom_dom_read_var(tl_coord0,'longitude',tl_dom0)
      ENDIF

      il_ind=var_get_index(tl_coord0%t_proc(1)%t_var(:), 'latitude_T')
      IF( il_ind /= 0 )THEN
         tl_lat0=iom_dom_read_var(tl_coord0,'latitude_T' ,tl_dom0)
      ELSE
         tl_lat0=iom_dom_read_var(tl_coord0,'latitude' ,tl_dom0)
      ENDIF

      IF( ll_even )THEN

         ! look for variable value on domain for F point
         il_ind=var_get_index(tl_coord0%t_proc(1)%t_var(:), 'longitude_F')
         IF( il_ind /= 0 )THEN
            tl_lon0F=iom_dom_read_var(tl_coord0,'longitude_F',tl_dom0)
         ENDIF

         il_ind=var_get_index(tl_coord0%t_proc(1)%t_var(:), 'latitude_F')
         IF( il_ind /= 0 )THEN
            tl_lat0F=iom_dom_read_var(tl_coord0,'latitude_F' ,tl_dom0)
         ENDIF

         ll_grid0F=.FALSE.
         IF( ASSOCIATED(tl_lon0F%d_value) .AND. &
         &   ASSOCIATED(tl_lat0F%d_value) )THEN
            ll_grid0F=.TRUE.
         ENDIF

      ENDIF

      ! close mpp files
      CALL iom_dom_close(tl_coord0)

      ! clean structure
      CALL mpp_clean(tl_coord0)
      CALL dom_clean(tl_dom0)

      ! copy structure
      tl_coord1=mpp_copy(td_coord1)

      ! open mpp files
      CALL iom_mpp_open(tl_coord1)

      ! read fine longitue and latitude
      il_ind=var_get_index(tl_coord1%t_proc(1)%t_var(:), TRIM(tl_lon0%c_longname))
      IF( il_ind /= 0 )THEN
         tl_lon1=iom_mpp_read_var(tl_coord1,TRIM(tl_lon0%c_longname))
      ELSE
         tl_lon1=iom_mpp_read_var(tl_coord1,'longitude')
      ENDIF
      il_ind=var_get_index(tl_coord1%t_proc(1)%t_var(:), TRIM(tl_lat0%c_longname))
      IF( il_ind /= 0 )THEN
         tl_lat1=iom_mpp_read_var(tl_coord1,TRIM(tl_lat0%c_longname))
      ELSE
         tl_lat1=iom_mpp_read_var(tl_coord1,'latitude')
      ENDIF
      
      IF( ll_even )THEN

         ! look for variable value on domain for F point
         il_ind=var_get_index(tl_coord1%t_proc(1)%t_var(:), 'longitude_F')
         IF( il_ind /= 0 )THEN
            tl_lon1F=iom_mpp_read_var(tl_coord1,'longitude_F')
         ENDIF

         il_ind=var_get_index(tl_coord1%t_proc(1)%t_var(:), 'latitude_F')
         IF( il_ind /= 0 )THEN
            tl_lat1F=iom_mpp_read_var(tl_coord1,'latitude_F')
         ENDIF

         ll_grid1F=.FALSE.
         IF( ASSOCIATED(tl_lon1F%d_value) .AND. &
         &   ASSOCIATED(tl_lat1F%d_value) )THEN
            ll_grid1F=.TRUE.
         ENDIF

      ENDIF

      ! close mpp files
      CALL iom_mpp_close(tl_coord1)
      ! clean structure
      CALL mpp_clean(tl_coord1)

      CALL logger_debug("GRID CHECK COINCIDENCE:"//&
      &        " fine   grid "//TRIM(td_coord1%c_name) )
      CALL logger_debug("GRID CHECK COINCIDENCE:"//&
      &        " coarse grid "//TRIM(td_coord0%c_name) )

      ! check domain
      ! check global grid
      IF( .NOT. grid_is_global(tl_lon0, tl_lat0) )THEN
         IF( grid_is_global(tl_lon1, tl_lat1) )THEN

            ll_coincidence=.FALSE.
            CALL logger_fatal("GRID CHECK COINCIDENCE:"//&
            &        " fine   grid is global,"//&
            &        " coarse grid is not ")

         ELSE
            il_ew1=tl_lon1%i_ew
            IF( il_ew1 >= 0 )THEN
               ! ew overlap

               il_ew0=tl_lon0%i_ew
               IF( il_ew0 < 0 )THEN
                  CALL logger_fatal("GRID CHECK COINCIDENCE: "//&
                  &        "fine grid has east west overlap,"//&
                  &        " coarse grid not ")
               ENDIF

               il_jmin1=1+ip_ghost
               il_jmax1=tl_lon1%t_dim(2)%i_len-ip_ghost

               ll_coincidence=grid__check_lat(&
               &                     tl_lat0%d_value(1,:,1,1),&
               &                     tl_lat1%d_value(1,il_jmin1:il_jmax1,1,1))

            ELSE
               ! other case
               il_imin1=1+ip_ghost
               il_jmin1=1+ip_ghost

               il_imax1=tl_lon1%t_dim(1)%i_len-ip_ghost
               il_jmax1=tl_lon1%t_dim(2)%i_len-ip_ghost

               ll_coincidence=grid__check_corner(&
                  &                   tl_lon0%d_value(:,:,1,1),&
                  &                   tl_lat0%d_value(:,:,1,1),&
                  &                   tl_lon1%d_value(il_imin1:il_imax1, &
                  &                                   il_jmin1:il_jmax1, &
                  &                                   1,1),&
                  &                   tl_lat1%d_value(il_imin1:il_imax1, &
                  &                                   il_jmin1:il_jmax1, &
                  &                                   1,1) )

            ENDIF
 
         ENDIF

         IF( .NOT. ll_coincidence )THEN
            CALL logger_fatal("GRID CHECK COINCIDENCE: no coincidence "//&
            &              "between fine grid and coarse grid: invalid domain." )
         ENDIF

      ENDIF
 
      ! check refinement factor
      ! select point in middle of fine grid
      il_imid1=INT(tl_lon1%t_dim(1)%i_len*0.5)
      il_jmid1=INT(tl_lon1%t_dim(2)%i_len*0.5)
 
      dl_lon1=tl_lon1%d_value(il_imid1, il_jmid1,1,1)
      dl_lat1=tl_lat1%d_value(il_imid1, il_jmid1,1,1)

      ! select closest point on coarse grid
      il_ind0(:)=grid_get_closest(tl_lon0%d_value(:,:,1,1),&
      &                           tl_lat0%d_value(:,:,1,1),&
      &                           dl_lon1, dl_lat1   )

      IF( ANY(il_ind0(:)==0) )THEN
         CALL logger_fatal("GRID CHECK COINCIDENCE: can not find valid "//&
         &              "coarse grid indices: invalid domain." )
      ENDIF

      IF( .NOT. ll_even )THEN
         ! case odd refinment in both direction
         ! work on T-point

         dl_lon0=tl_lon0%d_value(il_ind0(1),il_ind0(2),1,1)
         dl_lat0=tl_lat0%d_value(il_ind0(1),il_ind0(2),1,1)

         il_ind1(:)=grid_get_closest(tl_lon1%d_value(:,:,1,1),&
            &                        tl_lat1%d_value(:,:,1,1),&
            &                        dl_lon0, dl_lat0 )

         ! check i-direction refinement factor
         DO ji=0,MIN(3,il_imid1)

            IF( il_ind1(1)+ji*id_rho(jp_I)+1 > tl_lon1%t_dim(1)%i_len )THEN
               CALL logger_warn("GRID CHECK COINCIDENCE: domain to small "//&
               &  " to check i-direction refinement factor ")
               EXIT
            ELSE
               dl_lon0=tl_lon0%d_value(il_ind0(1)+ji             ,il_ind0(2),1,1)
               dl_lon1=tl_lon1%d_value(il_ind1(1)+ji*id_rho(jp_I),il_ind1(2),1,1)

               ! assume there could be little difference due to interpolation
               IF( ABS(dl_lon1 - dl_lon0) > dp_delta )THEN
                  ll_coincidence=.FALSE.
                  CALL logger_debug("GRID CHECK COINCIDENCE: invalid "//&
                  &  "i-direction refinement factor ("//&
                  &   TRIM(fct_str(id_rho(jp_I)))//&
                  &  ") between fine grid and coarse grid ")
               ENDIF
            ENDIF

         ENDDO

         ! check j-direction refinement factor
         DO jj=0,MIN(3,il_jmid1)

            IF( il_ind1(2)+jj*id_rho(jp_J)+1 > tl_lat1%t_dim(2)%i_len )THEN
               CALL logger_warn("GRID CHECK COINCIDENCE: domain to small "//&
                  &  " to check j-direction refinement factor ")
               EXIT
            ELSE      
               dl_lat0=tl_lat0%d_value(il_ind0(1),il_ind0(2)+jj             ,1,1)
               dl_lat1=tl_lat1%d_value(il_ind1(1),il_ind1(2)+jj*id_rho(jp_J),1,1)

               ! assume there could be little difference due to interpolation
               IF( ABS(dl_lat1-dl_lat0) > dp_delta )THEN
                  ll_coincidence=.FALSE.
                  CALL logger_debug("GRID CHECK COINCIDENCE: invalid "//&
                     &  "j-direction refinement factor ("//&
                     &   TRIM(fct_str(id_rho(jp_J)))//&
                     &  ") between fine grid and coarse grid ")
               ENDIF
            ENDIF

         ENDDO

      ELSE
         ! case even refinment at least in one direction
         ! work on F-point

         dl_delta=dp_delta
         ! look for lower left fine point in coarse cell.
         IF( ll_grid0F )THEN
      
            ! lower left corner of coarse cell
            dl_lon0F=tl_lon0F%d_value(il_ind0(1)-1,il_ind0(2)-1,1,1)
            dl_lat0F=tl_lat0F%d_value(il_ind0(1)-1,il_ind0(2)-1,1,1)

         ELSE

            ! approximate lower left corner of coarse cell (with T point)
            dl_lon0F=( tl_lon0%d_value(il_ind0(1)  ,il_ind0(2)  ,1,1) + &
               &       tl_lon0%d_value(il_ind0(1)  ,il_ind0(2)-1,1,1) + &
               &       tl_lon0%d_value(il_ind0(1)-1,il_ind0(2)  ,1,1) + &
               &       tl_lon0%d_value(il_ind0(1)-1,il_ind0(2)-1,1,1) ) * 0.25

            dl_lat0F=( tl_lat0%d_value(il_ind0(1)  ,il_ind0(2)  ,1,1) + &
               &       tl_lat0%d_value(il_ind0(1)  ,il_ind0(2)-1,1,1) + &
               &       tl_lat0%d_value(il_ind0(1)-1,il_ind0(2)  ,1,1) + &
               &       tl_lat0%d_value(il_ind0(1)-1,il_ind0(2)-1,1,1) ) * 0.25

            ! as we use approximation of F-point we relax condition
            dl_delta=100*dp_delta

         ENDIF

         IF( ll_grid1F )THEN
      
            il_ind1(:)=grid_get_closest(tl_lon1F%d_value(:,:,1,1),&
               &                        tl_lat1F%d_value(:,:,1,1),&
               &                        dl_lon0F, dl_lat0F )

         ELSE

            il_ill1(:)=grid_get_closest(tl_lon1%d_value(:,:,1,1),&
               &                        tl_lat1%d_value(:,:,1,1),&
               &                        dl_lon0F, dl_lat0F, 'll' )

            il_ilr1(:)=grid_get_closest(tl_lon1%d_value(:,:,1,1),&
               &                        tl_lat1%d_value(:,:,1,1),&
               &                        dl_lon0F, dl_lat0F, 'lr' )

            il_iul1(:)=grid_get_closest(tl_lon1%d_value(:,:,1,1),&
               &                        tl_lat1%d_value(:,:,1,1),&
               &                        dl_lon0F, dl_lat0F, 'ul' )

            il_iur1(:)=grid_get_closest(tl_lon1%d_value(:,:,1,1),&
               &                        tl_lat1%d_value(:,:,1,1),&
               &                        dl_lon0F, dl_lat0F, 'ur' )

            ! as we use approximation of F-point we relax condition
            dl_delta=100*dp_delta

         ENDIF

         ! check i-direction refinement factor
         DO ji=0,MIN(3,il_imid1)

            IF( il_ind1(1)+ji*id_rho(jp_I)+1 > tl_lon1%t_dim(1)%i_len )THEN
               CALL logger_warn("GRID CHECK COINCIDENCE: domain to small "//&
               &  " to check i-direction refinement factor ")
               EXIT
            ELSE
               IF( ll_grid0F )THEN
                  dl_lon0F=tl_lon0F%d_value(il_ind0(1)+ji-1, il_ind0(2)-1,1,1)
               ELSE
                  dl_lon0F= 0.25 * &
                     & ( tl_lon0%d_value(il_ind0(1)+ji  , il_ind0(2)  ,1,1) + &
                     &   tl_lon0%d_value(il_ind0(1)+ji-1, il_ind0(2)  ,1,1) + &
                     &   tl_lon0%d_value(il_ind0(1)+ji  , il_ind0(2)-1,1,1) + &
                     &   tl_lon0%d_value(il_ind0(1)+ji-1, il_ind0(2)-1,1,1) )
               ENDIF

               IF( ll_grid1F )THEN
                  dl_lon1F= tl_lon1F%d_value( il_ind1(1)+ji*id_rho(jp_I), &
                                            & il_ind1(2),1,1)
               ELSE
                  dl_lon1F= 0.25 * &
                     & ( tl_lon1%d_value( il_ill1(1)+ji*id_rho(jp_I), &
                                        & il_ill1(2),1,1) + &
                     &   tl_lon1%d_value( il_ilr1(1)+ji*id_rho(jp_I), &
                                        & il_ilr1(2),1,1) + &
                     &   tl_lon1%d_value( il_iul1(1)+ji*id_rho(jp_I), &
                                        & il_iul1(2),1,1) + &
                     &   tl_lon1%d_value( il_iur1(1)+ji*id_rho(jp_I), &
                                        & il_iur1(2),1,1) )
                  
               ENDIF

               ! assume there could be little difference due to interpolation
               IF( ABS(dl_lon1F - dl_lon0F) > dl_delta )THEN
                  ll_coincidence=.FALSE.
                  CALL logger_debug("GRID CHECK COINCIDENCE: invalid "//&
                  &  "i-direction refinement factor ("//&
                  &   TRIM(fct_str(id_rho(jp_I)))//&
                  &  ") between fine grid and coarse grid ")
               ENDIF
            ENDIF

         ENDDO

         ! check j-direction refinement factor
         DO jj=0,MIN(3,il_jmid1)

            IF( il_ind1(2)+jj*id_rho(jp_J)+1 > tl_lat1%t_dim(2)%i_len )THEN
               CALL logger_warn("GRID CHECK COINCIDENCE: domain to small "//&
               &  " to check j-direction refinement factor ")
               EXIT
            ELSE      
               IF( ll_grid0F )THEN
                  dl_lat0F=tl_lat0F%d_value(il_ind0(1)-1, il_ind0(2)+jj-1,1,1)
               ELSE
                  dl_lat0F= 0.25 * &
                  & ( tl_lat0%d_value(il_ind0(1)  , il_ind0(2)+jj  ,1,1) + &
                  &   tl_lat0%d_value(il_ind0(1)-1, il_ind0(2)+jj  ,1,1) + &
                  &   tl_lat0%d_value(il_ind0(1)  , il_ind0(2)+jj-1,1,1) + &
                  &   tl_lat0%d_value(il_ind0(1)-1, il_ind0(2)+jj-1,1,1) )
               ENDIF

               IF( ll_grid1F )THEN
                  dl_lat1F= tl_lat1F%d_value( il_ind1(1), &
                                            & il_ind1(2)+jj*id_rho(jp_J),1,1)
               ELSE
                  dl_lat1F= 0.25 * &
                     & ( tl_lat1%d_value( il_ill1(1), &
                                        & il_ill1(2)+jj*id_rho(jp_J),1,1) + &
                     &   tl_lat1%d_value( il_ilr1(1), &
                                        & il_ilr1(2)+jj*id_rho(jp_J),1,1) + &
                     &   tl_lat1%d_value( il_iul1(1), &
                                        & il_iul1(2)+jj*id_rho(jp_J),1,1) + &
                     &   tl_lat1%d_value( il_iur1(1), &
                                        & il_iur1(2)+jj*id_rho(jp_J),1,1) )
                  
               ENDIF

               ! assume there could be little difference due to interpolation
               IF( ABS(dl_lat1F - dl_lat0F) > dl_delta )THEN
                  ll_coincidence=.FALSE.
                  CALL logger_debug("GRID CHECK COINCIDENCE: invalid "//&
                     &  "i-direction refinement factor ("//&
                     &   TRIM(fct_str(id_rho(jp_I)))//&
                     &  ") between fine grid and coarse grid ")
               ENDIF
            ENDIF

         ENDDO
      ENDIF

      ! clean 
      CALL var_clean(tl_lon1)
      CALL var_clean(tl_lat1)
      CALL var_clean(tl_lon0)
      CALL var_clean(tl_lat0)

      IF( .NOT. ll_coincidence )THEN
         CALL logger_fatal("GRID CHECK COINCIDENCE: no coincidence "//&
         &              "between fine and coarse grid: "//&
         &              "invalid refinement factor" )
      ENDIF

   END SUBROUTINE grid_check_coincidence
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__check_corner(dd_lon0, dd_lat0, &
         &                     dd_lon1, dd_lat1) &
         & RESULT (lf_inside)
   !-------------------------------------------------------------------
   !> @brief This function check that fine grid is 
   !> inside coarse grid 
   !>
   !> @details
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_lon0   array of coarse grid longitude
   !> @param[in] dd_lat0   array of coarse grid latitude
   !> @param[in] dd_lon1   array of fine   grid longitude 
   !> @param[in] dd_lat1   array of fine   grid latitude
   !> @return true if fine grid is inside coarse grid
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      REAL(dp), DIMENSION(:,:), INTENT(IN) :: dd_lon0
      REAL(dp), DIMENSION(:,:), INTENT(IN) :: dd_lat0
      REAL(dp), DIMENSION(:,:), INTENT(IN) :: dd_lon1
      REAL(dp), DIMENSION(:,:), INTENT(IN) :: dd_lat1

      ! function
      LOGICAL                              :: lf_inside

      ! local variable
      INTEGER(i4), DIMENSION(2) :: il_shape0
      INTEGER(i4), DIMENSION(2) :: il_shape1

      INTEGER(i4) :: il_imin0
      INTEGER(i4) :: il_jmin0
      INTEGER(i4) :: il_imax0
      INTEGER(i4) :: il_jmax0

      INTEGER(i4) :: il_imin1
      INTEGER(i4) :: il_jmin1
      INTEGER(i4) :: il_imax1
      INTEGER(i4) :: il_jmax1

      REAL(dp)    :: dl_lon0
      REAL(dp)    :: dl_lat0

      REAL(dp)    :: dl_lon1
      REAL(dp)    :: dl_lat1
      ! loop indices
      !----------------------------------------------------------------

      ! init
      lf_inside=.TRUE.

      il_shape0=SHAPE(dd_lon0(:,:))
      il_shape1=SHAPE(dd_lon1(:,:))

      !1- check if fine grid inside coarse grid domain
      il_imin0=1 ; il_imax0=il_shape0(1)
      il_jmin0=1 ; il_jmax0=il_shape0(2)

      il_imin1=1 ; il_imax1=il_shape1(1)
      il_jmin1=1 ; il_jmax1=il_shape1(2)

      ! check lower left corner
      dl_lon0 = dd_lon0(il_imin0, il_jmin0)
      dl_lat0 = dd_lat0(il_imin0, il_jmin0)

      dl_lon1 = dd_lon1(il_imin1, il_jmin1)
      dl_lat1 = dd_lat1(il_imin1, il_jmin1)

      IF( (ABS(dl_lon1-dl_lon0)>dp_delta) .AND. (dl_lon1 < dl_lon0 ) .OR. & 
         &(ABS(dl_lat1-dl_lat0)>dp_delta) .AND. (dl_lat1 < dl_lat0 ) )THEN

         CALL logger_error("GRID CHECK COINCIDENCE: fine grid lower left "//&
            &     "corner  not north east of coarse grid (imin,jmin) ")
         CALL logger_debug(" fine   grid lower left ( "//&
            &              TRIM(fct_str(dl_lon1))//","//&
            &              TRIM(fct_str(dl_lat1))//")" )
         CALL logger_debug(" coarse grid lower left ( "//&
            &              TRIM(fct_str(dl_lon0))//","//&
            &              TRIM(fct_str(dl_lat0))//")" )
         lf_inside=.FALSE.

      ENDIF

      ! check upper left corner
      dl_lon0 = dd_lon0(il_imin0, il_jmax0)
      dl_lat0 = dd_lat0(il_imin0, il_jmax0)

      dl_lon1 = dd_lon1(il_imin1, il_jmax1)
      dl_lat1 = dd_lat1(il_imin1, il_jmax1)

      IF( (ABS(dl_lon1-dl_lon0)>dp_delta) .AND. (dl_lon1 < dl_lon0) .OR. &
      &   (ABS(dl_lat1-dl_lat0)>dp_delta) .AND. (dl_lat1 > dl_lat0) )THEN

         CALL logger_error("GRID CHECK COINCIDENCE: fine grid upper left "//&
            &     "corner not south east of coarse grid (imin,jmax) ")
         CALL logger_debug(" fine   grid upper left ("//&
            &              TRIM(fct_str(dl_lon1))//","//&
            &              TRIM(fct_str(dl_lat1))//")")
         CALL logger_debug(" coasre grid upper left ("//&
            &              TRIM(fct_str(dl_lon0))//","//&
            &              TRIM(fct_str(dl_lat0))//")")
         lf_inside=.FALSE.

      ENDIF

      ! check lower right corner
      dl_lon0 = dd_lon0(il_imax0, il_jmin0)
      dl_lat0 = dd_lat0(il_imax0, il_jmin0)

      dl_lon1 = dd_lon1(il_imax1, il_jmin1)
      dl_lat1 = dd_lat1(il_imax1, il_jmin1)


      IF( (ABS(dl_lon1-dl_lon0)>dp_delta) .AND. (dl_lon1 > dl_lon0) .OR. &
         &(ABS(dl_lat1-dl_lat0)>dp_delta) .AND. (dl_lat1 < dl_lat0) )THEN

         CALL logger_error("GRID CHECK COINCIDENCE: fine grid lower right "//&
            &     "corner not north west of coarse grid (imax,jmin) ")
         CALL logger_debug(" fine   grid lower right ( "//&
            &              TRIM(fct_str(dl_lon1))//","//&
            &              TRIM(fct_str(dl_lat1))//")" )
         CALL logger_debug(" coarse grid lower right ( "//&
            &              TRIM(fct_str(dl_lon0))//","//&
            &              TRIM(fct_str(dl_lat0))//")" )   
         lf_inside=.FALSE.

      ENDIF

      ! check upper right corner
      dl_lon0 = dd_lon0(il_imax0, il_jmax0)
      dl_lat0 = dd_lat0(il_imax0, il_jmax0)

      dl_lon1 = dd_lon1(il_imax1, il_jmax1)
      dl_lat1 = dd_lat1(il_imax1, il_jmax1)

      IF( (ABS(dl_lon1-dl_lon0)>dp_delta) .AND. (dl_lon1 > dl_lon0) .OR. &
      &   (ABS(dl_lat1-dl_lat0)>dp_delta) .AND. (dl_lat1 > dl_lat0) )THEN

         CALL logger_error("GRID CHECK COINCIDENCE: fine grid upper right "//&
            &     "corner not south west of coarse grid (imax,jmax) ")
         CALL logger_debug(" fine   grid upper right ( "//&
            &              TRIM(fct_str(dl_lon1))//","//&
            &              TRIM(fct_str(dl_lat1))//")" )
         CALL logger_debug(" fine   imax1 jmax1 ( "//&
            &              TRIM(fct_str(il_imax1))//","//&
            &              TRIM(fct_str(il_jmax1))//")" )
         CALL logger_debug(" coarse grid upper right ( "//&
            &              TRIM(fct_str(dl_lon0))//","//&
            &              TRIM(fct_str(dl_lat0))//")" )    
         CALL logger_debug(" fine   imax0 jmax0 ( "//&
            &              TRIM(fct_str(il_imax0))//","//&
            &              TRIM(fct_str(il_jmax0))//")" )
         lf_inside=.FALSE.

      ENDIF      

   END FUNCTION grid__check_corner
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__check_lat(dd_lat0, dd_lat1) &
         & RESULT (lf_inside)
   !-------------------------------------------------------------------
   !> @brief This function check that fine grid latitude are 
   !> inside coarse grid latitude
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] dd_lat0   array of coarse grid latitude 
   !> @param[in] dd_lat1   array of fine grid latitude
   !> @return true if fine grid is inside coarse grid
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument 
      REAL(dp), DIMENSION(:), INTENT(IN) :: dd_lat0
      REAL(dp), DIMENSION(:), INTENT(IN) :: dd_lat1

      ! function
      LOGICAL                            :: lf_inside

      ! local variable
      INTEGER(i4), DIMENSION(1) :: il_shape0
      INTEGER(i4), DIMENSION(1) :: il_shape1

      INTEGER(i4) :: il_jmin0
      INTEGER(i4) :: il_jmax0

      INTEGER(i4) :: il_jmin1
      INTEGER(i4) :: il_jmax1
      ! loop indices
      !----------------------------------------------------------------

      ! init
      lf_inside=.TRUE.

      il_shape0(:)=SHAPE(dd_lat0(:))
      il_shape1(:)=SHAPE(dd_lat1(:))

      !1- check if fine grid inside coarse grid domain
      il_jmin0=1 ; il_jmax0=il_shape0(1)
      il_jmin1=1 ; il_jmax1=il_shape1(1)

      ! check lower left fine grid
      IF( ABS(dd_lat1(il_jmin1)-dd_lat0(il_jmin0)) > dp_delta .AND. &
         &dd_lat1(il_jmin1) < dd_lat0(il_jmin0) )THEN

         CALL logger_error("GRID CHECK COINCIDENCE: fine grid lower point"//&
            &     " not north of coarse grid (jmin) ")
         CALL logger_debug(" fine grid lower point ( "//&
            &              TRIM(fct_str(dd_lat1(il_jmin1)))//")" )
         CALL logger_debug(" coarse grid lower point ( "//&
            &              TRIM(fct_str(dd_lat0(il_jmin0)))//")" )
         lf_inside=.FALSE.

      ENDIF

      ! check upper left fine grid
      IF( ABS(dd_lat1(il_jmax1)-dd_lat0(il_jmax0)) > dp_delta .AND. &
         &dd_lat1(il_jmax1) > dd_lat0(il_jmax0) )THEN

         CALL logger_error("GRID CHECK COINCIDENCE: fine grid upper point"//&
            &     " not south of coarse grid (jmax) ")
         CALL logger_debug(" fine grid upper point ("//&
            &              TRIM(fct_str(dd_lat1(il_jmax1)))//")")
         CALL logger_debug(" coasre grid upper point ("//&
            &              TRIM(fct_str(dd_lat0(il_jmax0)))//")")
         lf_inside=.FALSE.

      ENDIF
      
   END FUNCTION grid__check_lat
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_add_ghost(td_var, id_ghost)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine add ghost cell at boundaries.
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_var array of variable structure 
   !> @param[in] id_ghost  array of ghost cell factor 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) ,                 INTENT(INOUT) :: td_var
      INTEGER(i4), DIMENSION(2,2), INTENT(IN   ) :: id_ghost

      ! local variable
      INTEGER(i4) :: il_imin
      INTEGER(i4) :: il_jmin
      INTEGER(i4) :: il_imax
      INTEGER(i4) :: il_jmax

      REAL(dp), DIMENSION(:,:,:,:) , ALLOCATABLE :: dl_value
      
      TYPE(TVAR) :: tl_var

      ! loop indices
      !----------------------------------------------------------------

      IF( ALL(td_var%t_dim(1:2)%l_use) )THEN

         CALL logger_warn( "ADD GHOST: dimension change in variable "//&
         &              TRIM(td_var%c_name) )

         ! copy variable
         tl_var=var_copy(td_var)

         CALL var_del_value(td_var)

         ! compute indice to fill center
         il_imin=1+id_ghost(jp_I,1)*ip_ghost
         il_jmin=1+id_ghost(jp_J,1)*ip_ghost

         il_imax=tl_var%t_dim(1)%i_len+id_ghost(jp_I,1)*ip_ghost
         il_jmax=tl_var%t_dim(2)%i_len+id_ghost(jp_J,1)*ip_ghost

         ! compute new dimension
         td_var%t_dim(1)%i_len = tl_var%t_dim(1)%i_len + &
         &                             SUM(id_ghost(jp_I,:))*ip_ghost
         td_var%t_dim(2)%i_len = tl_var%t_dim(2)%i_len + &
         &                             SUM(id_ghost(jp_J,:))*ip_ghost

         ALLOCATE(dl_value(td_var%t_dim(1)%i_len, &
         &                 td_var%t_dim(2)%i_len, &
         &                 td_var%t_dim(3)%i_len, &
         &                 td_var%t_dim(4)%i_len) )

         dl_value(:,:,:,:)=tl_var%d_fill

         dl_value(il_imin:il_imax, &
         &        il_jmin:il_jmax, &
         &              :,:)  =  tl_var%d_value(:,:,:,:)

         ! add variable value
         CALL var_add_value(td_var,dl_value(:,:,:,:))

         ! save variable type
         td_var%i_type=tl_var%i_type
         
         DEALLOCATE( dl_value )

         CALL var_clean(tl_var)

      ENDIF

   END SUBROUTINE grid_add_ghost
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_del_ghost(td_var, id_ghost)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine delete ghost cell at boundaries.
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_var array of variable structure 
   !> @param[in] id_ghost  array of ghost cell factor 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) ,                 INTENT(INOUT) :: td_var
      INTEGER(i4), DIMENSION(2,2), INTENT(IN   ) :: id_ghost

      ! local variable
      INTEGER(i4) :: il_imin
      INTEGER(i4) :: il_jmin
      INTEGER(i4) :: il_imax
      INTEGER(i4) :: il_jmax

      REAL(dp), DIMENSION(:,:,:,:) , ALLOCATABLE :: dl_value
      
      TYPE(TVAR) :: tl_var

      ! loop indices
      !----------------------------------------------------------------

      IF( ALL(td_var%t_dim(1:2)%l_use) )THEN

         IF( ANY(id_ghost(:,:)/=0) )THEN
            CALL logger_warn( "GRID DEL GHOST: dimension change in variable "//&
            &              TRIM(td_var%c_name) )
         ENDIF

         ! copy variable
         tl_var=var_copy(td_var)

         CALL var_del_value(td_var)

         ! compute indice to get center
         il_imin=1+id_ghost(jp_I,1)*ip_ghost
         il_jmin=1+id_ghost(jp_J,1)*ip_ghost

         il_imax=tl_var%t_dim(1)%i_len-id_ghost(jp_I,2)*ip_ghost
         il_jmax=tl_var%t_dim(2)%i_len-id_ghost(jp_J,2)*ip_ghost

         ! compute new dimension
         td_var%t_dim(1)%i_len = il_imax - il_imin +1 
         td_var%t_dim(2)%i_len = il_jmax - il_jmin +1 

         ALLOCATE(dl_value(td_var%t_dim(1)%i_len, &
            &              td_var%t_dim(2)%i_len, &
            &              td_var%t_dim(3)%i_len, &
            &              td_var%t_dim(4)%i_len) )

         dl_value(:,:,:,:)=tl_var%d_fill

         dl_value(:,:,:,:)  =  tl_var%d_value(il_imin:il_imax, &
            &                                 il_jmin:il_jmax, &
            &                                 :,:)

         ! add variable value
         CALL var_add_value(td_var,dl_value(:,:,:,:))

         ! save variable type
         td_var%i_type=tl_var%i_type
         
         DEALLOCATE( dl_value )

         CALL var_clean(tl_var)

      ENDIF

   END SUBROUTINE grid_del_ghost
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_ghost_var(td_var) &
         & RESULT (if_ghost)
   !-------------------------------------------------------------------
   !> @brief This function check if ghost cell are used or not, and return ghost
   !> cell factor (0,1) in horizontal plan.
   !>
   !> @details
   !> check if domain is global, and if there is an East-West overlap.
   !>
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !>
   !> @param[in] td_var variable sturcture 
   !> @return array of ghost cell factor
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), INTENT(IN) :: td_var

      ! function
      INTEGER(i4), DIMENSION(2,2) :: if_ghost

      ! local variable
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_dim

      ! loop indices
      !----------------------------------------------------------------
      ! init
      if_ghost(:,:)=0

      IF( .NOT. ALL(td_var%t_dim(1:2)%l_use) )THEN
         CALL logger_error("GRID GET GHOST: "//TRIM(td_var%c_name)//" is not a suitable"//&
            &              " variable to look for ghost cell (not 2D).")
      ELSE
         IF( .NOT. ASSOCIATED(td_var%d_value) )THEN
            CALL logger_error("GRID GET GHOST: no value associated to "//TRIM(td_var%c_name)//&
               &              ". can't look for ghost cell.")
         ELSE
            il_dim(:)=td_var%t_dim(:)%i_len

            IF(ALL(td_var%d_value(    1    ,    :    ,1,1)/=td_var%d_fill).AND.&
            &  ALL(td_var%d_value(il_dim(1),    :    ,1,1)/=td_var%d_fill).AND.&
            &  ALL(td_var%d_value(    :    ,    1    ,1,1)/=td_var%d_fill).AND.&
            &  ALL(td_var%d_value(    :    ,il_dim(2),1,1)/=td_var%d_fill))THEN
            ! no boundary closed
               CALL logger_warn("GRID GET GHOST: can't determined ghost cell. "//&
                  &             "there is no boundary closed for variable "//&
                  &              TRIM(td_var%c_name))

            ELSE
               ! check periodicity
               IF(ANY(td_var%d_value(   1     ,:,1,1)/=td_var%d_fill).OR.&
               &  ANY(td_var%d_value(il_dim(1),:,1,1)/=td_var%d_fill))THEN
               ! East-West cyclic (1,4,6)
                  CALL logger_info("GRID GET GHOST: East West cyclic")
                  if_ghost(jp_I,:)=0

                  IF( ANY(td_var%d_value(:, 1, 1, 1) /= td_var%d_fill) )THEN
                  ! South boundary not closed 

                     CALL logger_debug("GRID GET GHOST: East_West cyclic")
                     CALL logger_debug("GRID GET GHOST: South boundary not closed")
                     CALL logger_error("GRID GET GHOST: should have been an "//&
                        &              "impossible case")

                  ELSE
                  ! South boundary closed (1,4,6)
                     CALL logger_info("GRID GET GHOST: South boundary closed")
                     if_ghost(jp_J,1)=1

                     IF(ANY(td_var%d_value(:,il_dim(2),1,1)/=td_var%d_fill) )THEN
                     ! North boundary not closed (4,6)
                        CALL logger_info("GRID GET GHOST: North boundary not closed")
                        if_ghost(jp_J,2)=0
                     ELSE
                     ! North boundary closed
                        CALL logger_info("GRID GET GHOST: North boundary closed")
                        if_ghost(jp_J,2)=1
                     ENDIF

                  ENDIF

               ELSE
               ! East-West boundaries closed (0,2,3,5)
                  CALL logger_info("GRID GET GHOST: East West boundaries closed")
                  if_ghost(jp_I,:)=1

                  IF( ANY(td_var%d_value(:, 1, 1, 1) /= td_var%d_fill) )THEN
                  ! South boundary not closed (2)
                     CALL logger_info("GRID GET GHOST: South boundary not closed")
                     if_ghost(jp_J,1)=0

                     IF(ANY(td_var%d_value(:,il_dim(2),1,1)/=td_var%d_fill))THEN
                     ! North boundary not closed
                        CALL logger_debug("GRID GET GHOST: East West boundaries "//&
                           &              "closed")
                        CALL logger_debug("GRID GET GHOST: South boundary not closed")
                        CALL logger_debug("GRID GET GHOST: North boundary not closed")
                        CALL logger_error("GRID GET GHOST: should have been "//&
                        &              "an impossible case")
                     ELSE
                     ! North boundary closed
                        if_ghost(jp_J,2)=1
                     ENDIF

                  ELSE
                  ! South boundary closed (0,3,5)
                     CALL logger_info("GRID GET GHOST: South boundary closed")
                     if_ghost(jp_J,1)=1

                     IF(ANY(td_var%d_value(:,il_dim(2),1,1)/=td_var%d_fill))THEN
                     ! North boundary not closed (3,5)
                        CALL logger_info("GRID GET GHOST: North boundary not closed")
                        if_ghost(jp_J,2)=0
                     ELSE
                     ! North boundary closed   
                        CALL logger_info("GRID GET GHOST: North boundary closed")
                        if_ghost(jp_J,2)=1
                     ENDIF

                  ENDIF

               ENDIF

            ENDIF

         ENDIF
      ENDIF

   END FUNCTION grid__get_ghost_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid__get_ghost_mpp(td_mpp) &
         & RESULT (if_ghost)
   !-------------------------------------------------------------------
   !> @brief This function check if ghost cell are used or not, and return ghost
   !> cell factor (0,1) in i- and j-direction.
   !>
   !> @details
   !> get longitude an latitude array, then
   !> check if domain is global, and if there is an East-West overlap
   !> 
   !> @author J.Paul
   !> @date September, 2014 - Initial Version
   !> @date October, 2014
   !> - work on mpp file structure instead of file structure
   !>
   !> @param[in] td_file   file sturcture
   !> @return array of ghost cell factor
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP), INTENT(IN) :: td_mpp

      ! function
      INTEGER(i4), DIMENSION(2,2) :: if_ghost

      ! local variable
      !TYPE(TVAR)  :: tl_lon
      !TYPE(TVAR)  :: tl_lat

      TYPE(TMPP) :: tl_mpp

      !INTEGER(i4) :: il_lonid
      !INTEGER(i4) :: il_latid
      ! loop indices
      !----------------------------------------------------------------
      ! init
      if_ghost(:,:)=0

      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN
         CALL logger_error("GRID GET GHOST: decomposition of mpp file "//&
         &                 TRIM(td_mpp%c_name)//" not defined." )

      ELSE

         ! copy structure
         tl_mpp=mpp_copy(td_mpp)

         CALL logger_info("GRID GET FINE GHOST perio"//TRIM(fct_str(tl_mpp%i_perio)))
         IF( tl_mpp%i_perio < 0 )THEN
            ! compute NEMO periodicity index
            CALL grid_get_info(tl_mpp)
         ENDIF

         SELECT CASE(tl_mpp%i_perio)
            CASE(0)
               if_ghost(:,:)=1
            CASE(1)
               if_ghost(jp_J,:)=1
            CASE(2)
               if_ghost(jp_I,:)=1
               if_ghost(jp_J,2)=1
            CASE(3,5)
               if_ghost(jp_I,:)=1
               if_ghost(jp_J,1)=1
            CASE(4,6)
               if_ghost(jp_J,1)=1
            CASE DEFAULT
         END SELECT

         ! clean 
         CALL mpp_clean(tl_mpp)

      ENDIF

   END FUNCTION grid__get_ghost_mpp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION grid_split_domain(td_var, id_level) &
         & RESULT (if_mask)
   !-------------------------------------------------------------------
   !> @brief This subroutine compute closed sea domain.
   !>
   !> @details
   !> to each domain is associated a negative value id (from -1 to ...)<br/>
   !> optionaly you could specify which level use (default 1)
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_var    variable strucutre 
   !> @param[in] id_level  level
   !> @return domain mask  
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) , INTENT(IN) :: td_var
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_level

      ! function
      INTEGER(i4), DIMENSION(td_var%t_dim(1)%i_len, &
      &                      td_var%t_dim(2)%i_len ) :: if_mask

      ! local variable
      INTEGER(i4)                              :: il_domid
      INTEGER(i4)                              :: il_level
      INTEGER(i4), DIMENSION(2)                :: il_shape
      INTEGER(i4), DIMENSION(2)                :: il_ind
      INTEGER(i4), DIMENSION(:,:), ALLOCATABLE :: il_tmp

      LOGICAL                                  :: ll_full

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jim
      INTEGER(i4) :: jip
      INTEGER(i4) :: jj
      INTEGER(i4) :: jjm
      INTEGER(i4) :: jjp
      !----------------------------------------------------------------
      il_level=1
      IF( PRESENT(id_level) ) il_level=id_level

      ! init
      il_domid=-1

      il_shape(:)=td_var%t_dim(1:2)%i_len
      if_mask(:,:)=0
      WHERE( td_var%d_value(:,:,il_level,1)/=td_var%d_fill ) if_mask(:,:)=1

      il_ind(:)=MAXLOC( if_mask(:,:) )
      DO WHILE( if_mask(il_ind(1), il_ind(2)) == 1 )

         if_mask(il_ind(1),il_ind(2))=il_domid
         ll_full=.FALSE.

         ALLOCATE( il_tmp(il_shape(1),il_shape(2)) )

         DO WHILE( .NOT. ll_full )
            il_tmp(:,:)=0

            ll_full=.TRUE.
            DO jj=1,il_shape(2)
               DO ji=1,il_shape(1)
                  IF( if_mask(ji,jj)==il_domid )THEN
                     jim=MAX(1,ji-1)   ;  jip=MIN(il_shape(1),ji+1)
                     jjm=MAX(1,jj-1)   ;  jjp=MIN(il_shape(2),jj+1)
                     
                     WHERE( if_mask(jim:jip,jjm:jjp)==1 )
                        if_mask(jim:jip,jjm:jjp)=il_domid
                        il_tmp(jim:jip,jjm:jjp)=1
                     END WHERE

                  ENDIF
               ENDDO
            ENDDO
            IF( SUM(il_tmp(:,:))/=0 ) ll_full=.FALSE.

         ENDDO
         DEALLOCATE( il_tmp )

         il_ind(:)=MAXLOC( if_mask(:,:) )
         il_domid=il_domid-1

      ENDDO

      CALL logger_info("GRID SPLIT DOMAIN: "//TRIM( fct_str(ABS(il_domid+1)) )//&
      &             " domain found" ) 

   END FUNCTION grid_split_domain
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_fill_small_dom(td_var, id_mask, id_minsize)
   !-------------------------------------------------------------------
   !> @brief This subroutine fill small closed sea with fill value. 
   !>
   !> @details
   !> the minimum size (number of point) of closed sea to be kept could be
   !> sepcify with id_minsize.
   !> By default only the biggest sea is preserve.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var    variable structure
   !> @param[in] id_mask      domain mask (from grid_split_domain)
   !> @param[in] id_minsize   minimum size of sea to be kept
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TVAR) ,                 INTENT(INOUT) :: td_var
      INTEGER(i4), DIMENSION(:,:), INTENT(IN   ) :: id_mask
      INTEGER(i4),                 INTENT(IN   ), OPTIONAL :: id_minsize

      ! local variable
      INTEGER(i4)                              :: il_ndom
      INTEGER(i4)                              :: il_minsize
      INTEGER(i4), DIMENSION(2)                :: il_shape
      INTEGER(i4), DIMENSION(:,:), ALLOCATABLE :: il_tmp

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      il_shape(:)=SHAPE(id_mask(:,:))
      IF( ANY(il_shape(:) /= td_var%t_dim(1:2)%i_len) )THEN
         CALL logger_error("GRID FILL SMALL DOM: variable and mask "//&
            &              "dimension differ")
      ELSE

         il_ndom=MINVAL(id_mask(:,:))

         ALLOCATE( il_tmp(il_shape(1),il_shape(2)) )
         il_tmp(:,:)=0
         DO ji=-1,il_ndom,-1
            WHERE( id_mask(:,:)==ji ) 
               il_tmp(:,:)=SUM(id_mask(:,:),id_mask(:,:)==ji)/ji
            END WHERE
         ENDDO

         il_minsize=MAXVAL(il_tmp(:,:))
         IF( PRESENT(id_minsize) ) il_minsize=id_minsize

         DO jl=1,td_var%t_dim(4)%i_len
            DO jk=1,td_var%t_dim(3)%i_len
               WHERE( il_tmp(:,:) < il_minsize ) 
                  td_var%d_value(:,:,jk,jl)=td_var%d_fill
               END WHERE
            ENDDO
         ENDDO

         DEALLOCATE( il_tmp )

      ENDIF

   END SUBROUTINE grid_fill_small_dom
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE grid_fill_small_msk(id_mask, id_minsize)
   !-------------------------------------------------------------------
   !> @brief This subroutine fill small domain inside bigger one. 
   !>
   !> @details
   !> the minimum size (number of point) of domain sea to be kept 
   !> is specified by id_minsize.
   !> smaller domain are included in the one they are embedded.
   !>
   !> @author J.Paul
   !> @date Ferbruay, 2015 - Initial Version
   !>
   !> @param[inout] id_mask   domain mask (from grid_split_domain)
   !> @param[in] id_minsize   minimum size of sea to be kept
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      INTEGER(i4), DIMENSION(:,:), INTENT(INOUT) :: id_mask
      INTEGER(i4),                 INTENT(IN   ) :: id_minsize

      ! local variable
      INTEGER(i4)                              :: il_ndom
      INTEGER(i4)                              :: il_minsize
      INTEGER(i4)                              :: il_msk
      
      INTEGER(i4)                              :: jim
      INTEGER(i4)                              :: jjm
      INTEGER(i4)                              :: jip
      INTEGER(i4)                              :: jjp

      INTEGER(i4), DIMENSION(2)                :: il_shape
      INTEGER(i4), DIMENSION(:,:), ALLOCATABLE :: il_tmp

      ! loop indices
      INTEGER(i4) :: ii
      INTEGER(i4) :: ij

      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      il_shape(:)=SHAPE(id_mask(:,:))
      il_ndom=MINVAL(id_mask(:,:))

      ALLOCATE( il_tmp(il_shape(1),il_shape(2)) )
      il_tmp(:,:)=0
      DO ji=-1,il_ndom,-1
         WHERE( id_mask(:,:)==ji ) 
            il_tmp(:,:)=SUM(id_mask(:,:),id_mask(:,:)==ji)/ji
         END WHERE
      ENDDO

      DO WHILE( id_minsize > MINVAL(il_tmp(:,:)) )

         DO jj=1,il_shape(2)
            DO ji=1,il_shape(1)

               IF( il_tmp(ji,jj) < id_minsize )THEN
                  jim=MAX(1,ji-1)   ;  jip=MIN(il_shape(1),ji+1)
                  jjm=MAX(1,jj-1)   ;  jjp=MIN(il_shape(2),jj+1)
                  
                  il_msk=0
                  DO ij=jjm,jjp
                     DO ii=jim,jip
                        IF( id_mask(ii,ij) /= id_mask(ji,jj) )THEN
                           IF( il_msk == 0 )THEN
                              il_msk=id_mask(ii,ij)
                           ELSEIF( il_msk /= id_mask(ii,ij) )THEN
                              CALL logger_error("GRID FILL SMALL MSK:"//&
                                 &  " small domain not embedded in bigger one."//&
                                 &  " point should be between two different"//&
                                 &  " domain.")
                           ENDIF
                        ENDIF
                     ENDDO
                  ENDDO
                  IF( il_msk /= 0 ) id_mask(ji,jj)=il_msk

               ENDIF

            ENDDO
         ENDDO


         il_tmp(:,:)=0
         DO ji=-1,il_ndom,-1
            WHERE( id_mask(:,:)==ji ) 
               il_tmp(:,:)=SUM(id_mask(:,:),id_mask(:,:)==ji)/ji
            END WHERE
         ENDDO            

      ENDDO

      DEALLOCATE( il_tmp )

   END SUBROUTINE grid_fill_small_msk
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE grid

