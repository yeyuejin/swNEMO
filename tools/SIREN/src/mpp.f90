!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief
!> This module manage massively parallel processing.
!>
!> @details
!> define type TMPP:<br/>
!> @code
!> TYPE(TMPP) :: tl_mpp
!> @endcode
!>
!>    to initialise a mpp structure:<br/>
!> @code
!>    tl_mpp=mpp_init( cd_file, id_mask, 
!>                       [id_niproc,] [id_njproc,] [id_nproc,]
!>                       [id_preci,] [id_precj,] 
!>                       [cd_type,] [id_ew])
!> @endcode
!> or
!> @code
!>    tl_mpp=mpp_init( cd_file, td_var, 
!>                      [id_niproc,] [id_njproc,] [id_nproc,]
!>                      [id_preci,] [id_precj,]
!>                      [cd_type] )
!> @endcode
!> or
!> @code
!>    tl_mpp=mpp_init( td_file [,id_ew] )
!> @endcode
!>       - cd_file is the filename of the global domain file, in which 
!>         MPP will be done (example: Bathymetry) 
!>       - td_file is the file structure of one processor file composing an MPP
!>       - id_mask is the 2D mask of global domain [optional]
!>       - td_var is a variable structure (on T-point) from global domain file.
!>         mask of the domain will be computed using FillValue [optional]
!>       - id_niproc is the number of processor following I-direction to be used
!>         [optional]
!>       - id_njproc is the number of processor following J-direction to be used 
!>         [optional]
!>       - id_nproc is the total number of processor to be used [optional]
!>       - id_preci is the size of the overlap region following I-direction [optional]
!>       - id_precj is the size of the overlap region following J-direction [optional]
!>       - cd_type is the type of files composing MPP [optional]
!>       - id_ew is east-west overlap [optional]<br/> 
!>  
!>    to get mpp name:<br/>
!>    - tl_mpp\%c_name
!>
!>    to get the total number of processor:<br/>
!>    - tl_mpp\%i_nproc
!>
!>    to get the number of processor following I-direction:<br/>
!>    - tl_mpp\%i_niproc
!>
!>    to get the number of processor following J-direction:<br/>
!>    - tl_mpp\%i_njproc
!>
!>    to get the length of the overlap region following I-direction:<br/>
!>    - tl_mpp\%i_preci
!>
!>    to get the length of the overlap region following J-direction:<br/>
!>    - tl_mpp\%i_precj
!>
!>    to get the type of files composing mpp structure:<br/>
!>    - tl_mpp\%c_type
!>
!>    to get the type of the global domain:<br/>
!>    - tl_mpp\%c_dom
!>
!>    MPP dimensions (global domain)<br/>
!>    to get the number of dimensions to be used in mpp strcuture:<br/>
!>    - tl_mpp\%i_ndim
!>
!>    to get the array of dimension structure (4 elts) associated to the
!>    mpp structure:<br/>
!>    - tl_mpp\%t_dim(:)
!>
!>    MPP processor (files composing domain)<br/>
!>    - tl_mpp\%t_proc(:)
!>
!>    to clean a mpp structure:<br/>
!> @code
!>    CALL mpp_clean(tl_mpp)
!> @endcode
!>
!>    to print information about mpp:<br/>
!> @code
!>    CALL mpp_print(tl_mpp)
!> @endcode
!>
!>    to add variable to mpp:<br/>
!> @code
!>    CALL mpp_add_var(td_mpp, td_var)
!> @endcode
!>       - td_var is a variable structure
!>
!>    to add dimension to mpp:<br/>
!> @code
!>    CALL mpp_add_dim(td_mpp, td_dim)
!> @endcode
!>       - td_dim is a dimension structure
!>
!>    to add attribute to mpp:<br/>
!> @code
!>    CALL mpp_add_att(td_mpp, td_att)
!> @endcode
!>       - td_att is a attribute structure
!>
!>    to delete variable from mpp:<br/>
!> @code
!>    CALL mpp_del_var(td_mpp, td_var)
!> @endcode
!>    or
!> @code
!>    CALL mpp_del_var(td_mpp, cd_name)
!> @endcode
!>       - td_var is a variable structure
!>       - cd_name is variable name or standard name
!>
!>    to delete dimension from mpp:<br/>
!> @code
!>    CALL mpp_del_dim(td_mpp, td_dim)
!> @endcode
!>       - td_dim is a dimension structure
!>
!>    to delete attribute from mpp:<br/>
!> @code
!>    CALL mpp_del_att(td_mpp, td_att)
!> @endcode
!>    or
!> @code
!>    CALL mpp_del_att(td_mpp, cd_name)
!> @endcode
!>       - td_att is a attribute structure
!>       - cd_name is attribute name
!>
!>    to overwrite variable to mpp:<br/>
!> @code
!>    CALL mpp_move_var(td_mpp, td_var)
!> @endcode
!>       - td_var is a variable structure
!>
!>    to overwrite dimension to mpp:<br/>
!> @code
!>    CALL mpp_move_dim(td_mpp, td_dim)
!> @endcode
!>       - td_dim is a dimension structure
!>
!>    to overwrite attribute to mpp:<br/>
!> @code
!>    CALL mpp_move_att(td_mpp, td_att)
!> @endcode
!>       - td_att is a attribute structure
!>
!>    to determine domain decomposition type:<br/>
!> @code
!>    CALL mpp_get_dom(td_mpp)
!> @endcode
!>
!>    to get processors to be used:<br/>
!> @code
!>    CALL mpp_get_use( td_mpp, id_imin, id_imax, & 
!>    &                         id_jmin, id_jmax )
!> @endcode
!>       - id_imin 
!>       - id_imax 
!>       - id_jmin 
!>       - id_jmax 
!>
!>    to get sub domains which form global domain contour:<br/>
!> @code
!>    CALL mpp_get_contour( td_mpp ) 
!> @endcode
!>
!>    to get global domain indices of one processor:<br/>
!> @code
!>    il_ind(1:4)=mpp_get_proc_index( td_mpp, id_procid )
!> @endcode
!>       - il_ind(1:4) are global domain indices (i1,i2,j1,j2)
!>       - id_procid is the processor id
!>
!>    to get the processor domain size:<br/>
!> @code
!>    il_size(1:2)=mpp_get_proc_size( td_mpp, id_procid )
!> @endcode
!>       - il_size(1:2) are the size of domain following I and J
!>       - id_procid is the processor id
!>
!> @author
!>  J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date November, 2014 
!> - Fix memory leaks bug
!> @date October, 2015
!> - improve way to compute domain layout
!> @date January, 2016
!> - allow to print layout file (use lm_layout, hard coded)
!> - add mpp__compute_halo and mpp__read_halo
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!>
!> @todo
!> - ECRIRE ET TESTER add_proc_array pour optimiser codes (voir old/MO_mpp.f90)
!----------------------------------------------------------------------
MODULE mpp

   USE global                          ! global parameter
   USE kind                            ! F90 kind parameter
   USE logger                          ! log file manager
   USE fct                             ! basic useful function
   USE dim                             ! dimension manager
   USE att                             ! attribute manager
   USE var                             ! variable manager
   USE file                            ! file manager
   USE iom                             ! I/O manager 

   IMPLICIT NONE
   ! NOTE_avoid_public_variables_if_possible

   ! type and variable
   PUBLIC  :: TMPP       !< mpp structure
   PRIVATE :: TLAY       !< domain layout structure

   ! function and subroutine
   PUBLIC :: mpp_copy           !< copy mpp structure
   PUBLIC :: mpp_init           !< initialise mpp structure
   PUBLIC :: mpp_clean          !< clean mpp strcuture
   PUBLIC :: mpp_print          !< print information about mpp structure
   PUBLIC :: mpp_add_var        !< split/add one variable strucutre in mpp structure
   PUBLIC :: mpp_add_dim        !< add one dimension to mpp structure
   PUBLIC :: mpp_add_att        !< add one attribute strucutre in mpp structure
   PUBLIC :: mpp_del_var        !< delete one variable strucutre in mpp structure
   PUBLIC :: mpp_del_dim        !< delete one dimension strucutre in mpp structure
   PUBLIC :: mpp_del_att        !< delete one attribute strucutre in mpp structure
   PUBLIC :: mpp_move_var       !< overwrite variable structure in mpp structure
   PUBLIC :: mpp_move_dim       !< overwrite one dimension strucutre in mpp structure
   PUBLIC :: mpp_move_att       !< overwrite one attribute strucutre in mpp structure
   PUBLIC :: mpp_recombine_var  !< recombine variable from mpp structure
   PUBLIC :: mpp_get_index      !< return index of mpp 

   PUBLIC :: mpp_get_dom        !< determine domain decomposition type (full, overlap, noverlap)
   PUBLIC :: mpp_get_use        !< get sub domains to be used (which cover "zoom domain")
   PUBLIC :: mpp_get_contour    !< get sub domains which form global domain contour
   PUBLIC :: mpp_get_proc_index !< get processor domain indices
   PUBLIC :: mpp_get_proc_size  !< get processor domain size

   PRIVATE :: mpp__add_proc            ! add proc strucutre in mpp structure
   PRIVATE :: mpp__add_proc_unit       ! add one proc strucutre in mpp structure
   PRIVATE :: mpp__add_proc_arr        ! add array of proc strucutre in mpp structure
   PRIVATE :: mpp__del_proc            ! delete one proc strucutre in mpp structure
   PRIVATE :: mpp__del_proc_id         ! delete one proc strucutre in mpp structure, given procesor id
   PRIVATE :: mpp__del_proc_str        ! delete one proc strucutre in mpp structure, given procesor file structure 
   PRIVATE :: mpp__move_proc           ! overwrite proc strucutre in mpp structure
   PRIVATE :: mpp__create_layout       ! create mpp structure using domain layout
   PRIVATE :: mpp__optimiz             ! compute optimum domain decomposition
   PRIVATE :: mpp__check_dim           ! check mpp structure dimension with proc or variable dimension
   PRIVATE :: mpp__check_proc_dim      ! check if processor and mpp structure use same dimension
   PRIVATE :: mpp__check_var_dim       ! check if variable  and mpp structure use same dimension
   PRIVATE :: mpp__del_var_name        ! delete variable in mpp structure, given variable name
   PRIVATE :: mpp__del_var_mpp         ! delete all variable in mpp structure
   PRIVATE :: mpp__del_var_str         ! delete variable in mpp structure, given variable structure
   PRIVATE :: mpp__del_att_name        ! delete variable in mpp structure, given variable name
   PRIVATE :: mpp__del_att_str         ! delete variable in mpp structure, given variable structure
   PRIVATE :: mpp__split_var           ! extract variable part that will be written in processor 
   PRIVATE :: mpp__copy_unit           ! copy mpp structure
   PRIVATE :: mpp__copy_arr            ! copy array of mpp structure
   PRIVATE :: mpp__get_use_unit        ! get sub domains to be used (which cover "zoom domain")
   PRIVATE :: mpp__init_mask           ! initialise mpp structure, given mask array
   PRIVATE :: mpp__init_var            ! initialise mpp structure, given variable strcuture
   PRIVATE :: mpp__init_file           ! initialise a mpp structure, given file structure 
   PRIVATE :: mpp__init_file_cdf       ! initialise a mpp structure with cdf file
   PRIVATE :: mpp__init_file_rstdimg   ! initialise a mpp structure with rstdimg file
   PRIVATE :: mpp__clean_unit          ! clean mpp strcuture
   PRIVATE :: mpp__clean_arr           ! clean array of mpp strcuture
   PRIVATE :: mpp__compute_halo        ! compute subdomain indices defined with halo 
   PRIVATE :: mpp__read_halo           ! read subdomain indices defined with halo

   PRIVATE :: layout__init             ! initialise domain layout structure
   PRIVATE :: layout__copy             ! clean domain layout structure
   PRIVATE :: layout__clean            ! copy  domain layout structure

   TYPE TMPP !< mpp structure
      ! general 
      CHARACTER(LEN=lc)                  :: c_name = ''   !< base name 
      INTEGER(i4)                        :: i_id   = 0    !< mpp id

      INTEGER(i4)                        :: i_niproc = 0  !< number of processors following i
      INTEGER(i4)                        :: i_njproc = 0  !< number of processors following j
      INTEGER(i4)                        :: i_nproc  = 0  !< total number of proccessors used
      INTEGER(i4)                        :: i_preci = 1   !< i-direction overlap region length
      INTEGER(i4)                        :: i_precj = 1   !< j-direction overlap region length
      INTEGER(i4)                        :: i_ew    = -1  !< east-west overlap
      INTEGER(i4)                        :: i_perio = -1  !< NEMO periodicity index
      INTEGER(i4)                        :: i_pivot = -1  !< NEMO pivot point index F(0),T(1)

      CHARACTER(LEN=lc)                  :: c_type = ''   !< type of the files (cdf, cdf4, dimg)
      CHARACTER(LEN=lc)                  :: c_dom  = ''   !< type of domain (full, noextra, nooverlap)

      INTEGER(i4)                        :: i_ndim = 0    !< number of dimensions used in mpp
      TYPE(TDIM),  DIMENSION(ip_maxdim)  :: t_dim         !< global domain dimension

      TYPE(TFILE), DIMENSION(:), POINTER :: t_proc => NULL()  !< files/processors composing mpp

      LOGICAL                            :: l_usempp = .TRUE. !< use mpp decomposition for writing netcdf
   END TYPE

   TYPE TLAY !< domain layout structure
      INTEGER(i4)                          :: i_niproc = 0  !< number of processors following i
      INTEGER(i4)                          :: i_njproc = 0  !< number of processors following j
      INTEGER(i4)                          :: i_nland  = 0       !< number of land processors
      INTEGER(i4)                          :: i_nsea   = 0       !< number of sea  processors
      INTEGER(i4)                          :: i_mean   = 0       !< mean sea point per proc
      INTEGER(i4)                          :: i_min    = 0       !< min  sea point per proc
      INTEGER(i4)                          :: i_max    = 0       !< max  sea point per proc
      INTEGER(i4), DIMENSION(:,:), POINTER :: i_msk   => NULL()  !< sea/land processor mask 
      INTEGER(i4), DIMENSION(:,:), POINTER :: i_impp  => NULL()  !< i-indexes for mpp-subdomain left bottom 
      INTEGER(i4), DIMENSION(:,:), POINTER :: i_jmpp  => NULL()  !< j-indexes for mpp-subdomain left bottom 
      INTEGER(i4), DIMENSION(:,:), POINTER :: i_lci   => NULL()  !< i-dimensions of subdomain 
      INTEGER(i4), DIMENSION(:,:), POINTER :: i_lcj   => NULL()  !< j-dimensions of subdomain 
   END TYPE

   ! module variable
   INTEGER(i4) :: im_psize  = 2000        !< processor dimension length for huge file 

   INTEGER(i4) :: im_iumout = 44
   LOGICAL     :: lm_layout =.FALSE.

   INTERFACE mpp_get_use
      MODULE PROCEDURE mpp__get_use_unit 
   END INTERFACE mpp_get_use

   INTERFACE mpp__add_proc
      MODULE PROCEDURE mpp__add_proc_unit 
      MODULE PROCEDURE mpp__add_proc_arr
   END INTERFACE mpp__add_proc

   INTERFACE mpp_clean
      MODULE PROCEDURE mpp__clean_unit 
      MODULE PROCEDURE mpp__clean_arr
   END INTERFACE mpp_clean

   INTERFACE mpp__check_dim
      MODULE PROCEDURE mpp__check_proc_dim !< check if processor and mpp structure use same dimension
      MODULE PROCEDURE mpp__check_var_dim  !< check if variable  and mpp structure use same dimension
   END INTERFACE mpp__check_dim

   INTERFACE mpp__del_proc
      MODULE PROCEDURE mpp__del_proc_id
      MODULE PROCEDURE mpp__del_proc_str
   END INTERFACE mpp__del_proc

   INTERFACE mpp_del_var
      MODULE PROCEDURE mpp__del_var_name
      MODULE PROCEDURE mpp__del_var_str
      MODULE PROCEDURE mpp__del_var_mpp
   END INTERFACE mpp_del_var

   INTERFACE mpp_del_att
      MODULE PROCEDURE mpp__del_att_name
      MODULE PROCEDURE mpp__del_att_str
   END INTERFACE mpp_del_att

   INTERFACE mpp_init
      MODULE PROCEDURE mpp__init_mask
      MODULE PROCEDURE mpp__init_var
      MODULE PROCEDURE mpp__init_file
   END INTERFACE mpp_init

   INTERFACE mpp_copy
      MODULE PROCEDURE mpp__copy_unit  ! copy mpp structure
      MODULE PROCEDURE mpp__copy_arr   ! copy array of mpp structure
   END INTERFACE

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION mpp__copy_unit(td_mpp) &
         &  RESULT(tf_mpp)      
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine copy mpp structure in another one
   !> @details 
   !> mpp file are copied in a temporary array, 
   !> so input and output mpp structure do not point on the same 
   !> "memory cell", and so on are independant. 
   !>
   !> @warning do not use on the output of a function who create or read an
   !> structure (ex: tl_file=file_copy(file_init()) is forbidden).
   !> This will create memory leaks.
   !> @warning to avoid infinite loop, do not use any function inside 
   !> this subroutine
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date November, 2014
   !> - use function instead of overload assignment operator 
   !> (to avoid memory leak)
   !> @date January, 2019
   !> - clean file structure
   !>
   !> @param[in] td_mpp   mpp structure
   !> @return copy of input mpp structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP), INTENT(IN)  :: td_mpp

      ! function
      TYPE(TMPP)              :: tf_mpp

      ! local variable
      TYPE(TFILE) :: tl_file

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      CALL logger_trace("MPP COPY: "//TRIM(td_mpp%c_name)//" in "//&
      &  TRIM(tf_mpp%c_name))

      ! copy mpp variable
      tf_mpp%c_name     = TRIM(td_mpp%c_name)
      tf_mpp%i_id       = td_mpp%i_id
      tf_mpp%i_niproc   = td_mpp%i_niproc
      tf_mpp%i_njproc   = td_mpp%i_njproc
      tf_mpp%i_nproc    = td_mpp%i_nproc
      tf_mpp%i_preci    = td_mpp%i_preci
      tf_mpp%i_precj    = td_mpp%i_precj
      tf_mpp%c_type     = TRIM(td_mpp%c_type)
      tf_mpp%c_dom      = TRIM(td_mpp%c_dom)
      tf_mpp%i_ndim     = td_mpp%i_ndim
      tf_mpp%i_ew       = td_mpp%i_ew
      tf_mpp%i_perio    = td_mpp%i_perio
      tf_mpp%i_pivot    = td_mpp%i_pivot
      tf_mpp%l_usempp   = td_mpp%l_usempp

      ! copy dimension
      tf_mpp%t_dim(:) = dim_copy(td_mpp%t_dim(:))
      
      ! copy file structure
      IF( ASSOCIATED(tf_mpp%t_proc) )THEN
         CALL file_clean(tf_mpp%t_proc(:))
         DEALLOCATE(tf_mpp%t_proc)
      ENDIF
      IF( ASSOCIATED(td_mpp%t_proc) .AND. tf_mpp%i_nproc > 0 )THEN
         ALLOCATE( tf_mpp%t_proc(tf_mpp%i_nproc) )
         DO ji=1,tf_mpp%i_nproc
            tl_file = file_copy(td_mpp%t_proc(ji))
            tf_mpp%t_proc(ji) = file_copy(tl_file)
         ENDDO
         ! clean
         CALL file_clean(tl_file)
      ENDIF

   END FUNCTION mpp__copy_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION mpp__copy_arr(td_mpp) &
         &  RESULT(tf_mpp)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine copy an array of mpp structure in another one
   !> @details 
   !> mpp file are copied in a temporary array, 
   !> so input and output mpp structure do not point on the same 
   !> "memory cell", and so on are independant. 
   !>
   !> @warning do not use on the output of a function who create or read an
   !> structure (ex: tl_file=file_copy(file_init()) is forbidden).
   !> This will create memory leaks.
   !> @warning to avoid infinite loop, do not use any function inside 
   !> this subroutine
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date November, 2014
   !>    - use function instead of overload assignment operator 
   !> (to avoid memory leak)
   !>
   !> @param[in] td_mpp   mpp structure
   !> @return copy of input array of mpp structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP), DIMENSION(:),  INTENT(IN)  :: td_mpp

      ! function
      TYPE(TMPP), DIMENSION(SIZE(td_mpp(:))) :: tf_mpp

      ! local variable
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      DO ji=1,SIZE(td_mpp(:))
         tf_mpp(ji)=mpp_copy(td_mpp(ji))
      ENDDO

   END FUNCTION mpp__copy_arr
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp_print(td_mpp)
   !-------------------------------------------------------------------
   !> @brief This subroutine print some information about mpp strucutre.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_mpp mpp structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP), INTENT(IN) :: td_mpp

      ! local variable
      INTEGER(i4), PARAMETER :: ip_freq = 4
      INTEGER(i4), PARAMETER :: ip_min  = 5

      INTEGER(i4)                              :: il_min
      INTEGER(i4), DIMENSION(:,:), ALLOCATABLE :: il_proc
      INTEGER(i4), DIMENSION(:,:), ALLOCATABLE :: il_lci
      INTEGER(i4), DIMENSION(:,:), ALLOCATABLE :: il_lcj

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      INTEGER(i4) :: jm
      !----------------------------------------------------------------

      WRITE(*,'((a,a),2(/3x,a,a),9(/3x,a,i0))')&
      &  "MPP : ",TRIM(td_mpp%c_name), &
      &  " type   : ",TRIM(td_mpp%c_type), &
      &  " dom    : ",TRIM(td_mpp%c_dom), &
      &  " nproc  : ",td_mpp%i_nproc, &
      &  " niproc : ",td_mpp%i_niproc, &
      &  " njproc : ",td_mpp%i_njproc, &
      &  " preci  : ",td_mpp%i_preci, &
      &  " precj  : ",td_mpp%i_precj, &
      &  " ndim   : ",td_mpp%i_ndim,  &
      &  " overlap: ",td_mpp%i_ew, &
      &  " perio  : ",td_mpp%i_perio, &
      &  " pivot  : ",td_mpp%i_pivot

      ! print dimension
      IF(  td_mpp%i_ndim /= 0 )THEN
         WRITE(*,'(/a)') " MPP dimension"
         DO ji=1,ip_maxdim
            IF( td_mpp%t_dim(ji)%l_use )THEN
               CALL dim_print(td_mpp%t_dim(ji))
            ENDIF
         ENDDO
      ENDIF

      ! print file
      IF( td_mpp%i_nproc /= 0 .AND. ASSOCIATED(td_mpp%t_proc) )THEN
         IF( ALL( td_mpp%t_proc(:)%i_iind==0 ) .OR. &
         &   ALL( td_mpp%t_proc(:)%i_jind==0 ) )THEN

            il_min=MIN(td_mpp%i_nproc,ip_min)
            DO ji=1,il_min
               CALL file_print(td_mpp%t_proc(ji))
               WRITE(*,'((a),(/3x,a,i0),2(/3x,a,a),4(/3x,a,i0,a,i0)/)')&
               &  " Domain decomposition : ", &
               &  " id          : ",td_mpp%t_proc(ji)%i_pid, &
               &  " used        : ",TRIM(fct_str(td_mpp%t_proc(ji)%l_use)), &
               &  " contour     : ",TRIM(fct_str(td_mpp%t_proc(ji)%l_ctr)), &
               &  " left-bottom : ",td_mpp%t_proc(ji)%i_impp,', ',&
               &  td_mpp%t_proc(ji)%i_jmpp, &
               &  " dimension   : ",td_mpp%t_proc(ji)%i_lci,' x ',&
               &  td_mpp%t_proc(ji)%i_lcj, &
               &  " first indoor indices : ",td_mpp%t_proc(ji)%i_ldi,', ',&
               &  td_mpp%t_proc(ji)%i_ldj, &
               &  " last  indoor indices : ",td_mpp%t_proc(ji)%i_lei,', ',&
               &  td_mpp%t_proc(ji)%i_lej

            ENDDO
            IF( td_mpp%i_nproc > ip_min )THEN
               WRITE(*,'(a)') "...etc" 
            ENDIF

            IF( td_mpp%t_proc(1)%i_nvar > 0 )THEN
               WRITE(*,'(/a)') " Variable(s) used : "
               DO ji=1,td_mpp%t_proc(1)%i_nvar
                  WRITE(*,'(3x,a)') TRIM(td_mpp%t_proc(1)%t_var(ji)%c_name) 
               ENDDO
            ENDIF

         ELSE

            il_min=MIN(td_mpp%i_nproc,ip_min)
            DO ji=1,il_min
               CALL file_print(td_mpp%t_proc(ji))
               WRITE(*,'((a, a),(/3x,a,i0),(/3x,a,a),4(/3x,a,i0,a,i0)/)')&
               &  " Domain decomposition : ",TRIM(td_mpp%t_proc(ji)%c_name),&
               &  " id          : ",td_mpp%t_proc(ji)%i_pid, &
               &  " used        : ",TRIM(fct_str(td_mpp%t_proc(ji)%l_use)),&
               &  " left-bottom : ",td_mpp%t_proc(ji)%i_impp,', ',&
               &  td_mpp%t_proc(ji)%i_jmpp, &
               &  " dimension   : ",td_mpp%t_proc(ji)%i_lci,' x ',&
               &  td_mpp%t_proc(ji)%i_lcj, &
               &  " first indoor indices : ",td_mpp%t_proc(ji)%i_ldi,',',&
               &  td_mpp%t_proc(ji)%i_ldj, &
               &  " last  indoor indices : ",td_mpp%t_proc(ji)%i_lei,', ',&
               &  td_mpp%t_proc(ji)%i_lej

            ENDDO
            IF( td_mpp%i_nproc > ip_min )THEN
               WRITE(*,'(a)') "...etc" 
            ENDIF
            
            IF( td_mpp%t_proc(1)%i_nvar > 0 )THEN
               WRITE(*,'(/a)') " Variable(s) used : "
               DO ji=1,td_mpp%t_proc(1)%i_nvar
                  WRITE(*,'(3x,a)') TRIM(td_mpp%t_proc(1)%t_var(ji)%c_name)
               ENDDO
            ENDIF

            IF( td_mpp%l_usempp )THEN
               ALLOCATE( il_proc(td_mpp%i_niproc,td_mpp%i_njproc) )
               ALLOCATE( il_lci(td_mpp%i_niproc,td_mpp%i_njproc) )
               ALLOCATE( il_lcj(td_mpp%i_niproc,td_mpp%i_njproc) )
               il_proc(:,:)=-1
               il_lci(:,:) =-1
               il_lcj(:,:) =-1

               DO jk=1,td_mpp%i_nproc
                  ji=td_mpp%t_proc(jk)%i_iind
                  jj=td_mpp%t_proc(jk)%i_jind
                  il_proc(ji,jj)=jk-1
                  il_lci(ji,jj)=td_mpp%t_proc(jk)%i_lci
                  il_lcj(ji,jj)=td_mpp%t_proc(jk)%i_lcj
               ENDDO

               jl = 1
               DO jk = 1,(td_mpp%i_niproc-1)/ip_freq+1
                  jm = MIN(td_mpp%i_niproc, jl+ip_freq-1)
                  WRITE(*,*)
                  WRITE(*,9401) (ji, ji = jl,jm)
                  WRITE(*,9400) ('***', ji = jl,jm-1)
                  DO jj = 1, td_mpp%i_njproc
                     WRITE(*,9403) ('   ', ji = jl,jm-1)
                     WRITE(*,9402) jj, ( il_lci(ji,jj), il_lcj(ji,jj), ji = jl,jm)
                     WRITE(*,9404) (il_proc(ji,jj), ji= jl,jm)
                     WRITE(*,9403) ('   ', ji = jl,jm-1)
                     WRITE(*,9400) ('***', ji = jl,jm-1)
                  ENDDO
                  jl = jl+ip_freq
               ENDDO
         
               DEALLOCATE( il_proc )
               DEALLOCATE( il_lci )
               DEALLOCATE( il_lcj )
            ENDIF

         ENDIF
      ELSE
         WRITE(*,'(/a)') " Domain decomposition : none"
      ENDIF

9400   FORMAT('     ***',20('*************',a3))
9403   FORMAT('     *     ',20('         *   ',a3))
9401   FORMAT('        ',20('   ',i3,'          '))
9402   FORMAT(' ',i3,' *  ',20(i3,'  x',i3,'   *   '))
9404   FORMAT('     *  ',20('      ',i3,'   *   '))

   END SUBROUTINE mpp_print
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION mpp__init_mask(cd_file, id_mask,                   &
         &                 id_niproc, id_njproc, id_nproc,     &
         &                 id_preci, id_precj,                 &
         &                 cd_type, id_ew, id_perio, id_pivot, &
         &                 td_dim, ld_usempp)                  &
         & RESULT(tf_mpp)
   !-------------------------------------------------------------------
   !> @brief
   !> This function initialise mpp structure, given file name, 
   !> and optionaly mask and number of processor following I and J
   !> @detail
   !> - If no total number of processor is defined (id_nproc), optimize 
   !> the domain decomposition (look for the domain decomposition with 
   !> the most land processor to remove)
   !> - length of the overlap region (id_preci, id_precj) could be specify
   !> in I and J direction (default value is 1)
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !> @date September, 2015
   !> - allow to define dimension with array of dimension structure
   !> @date January, 2016
   !> - use RESULT to rename output
   !> - mismatch with "halo" indices
   !>
   !> @param[in] cd_file   file name of one file composing mpp domain
   !> @param[in] id_mask   domain mask
   !> @param[in] id_niproc number of processors following i
   !> @param[in] id_njproc number of processors following j
   !> @param[in] id_nproc  total number of processors
   !> @param[in] id_preci  i-direction overlap region
   !> @param[in] id_precj  j-direction overlap region
   !> @param[in] cd_type   type of the files (cdf, cdf4, dimg)
   !> @param[in] id_ew     east-west overlap
   !> @param[in] id_perio  NEMO periodicity index
   !> @param[in] id_pivot  NEMO pivot point index F(0),T(1)
   !> @param[in] td_dim    array of dimension structure
   !> @return mpp structure
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*),                  INTENT(IN) :: cd_file
      INTEGER(i4), DIMENSION(:,:),       INTENT(IN) :: id_mask
      INTEGER(i4),                       INTENT(IN), OPTIONAL :: id_niproc
      INTEGER(i4),                       INTENT(IN), OPTIONAL :: id_njproc
      INTEGER(i4),                       INTENT(IN), OPTIONAL :: id_nproc
      INTEGER(i4),                       INTENT(IN), OPTIONAL :: id_preci
      INTEGER(i4),                       INTENT(IN), OPTIONAL :: id_precj
      CHARACTER(LEN=*),                  INTENT(IN), OPTIONAL :: cd_type
      INTEGER(i4),                       INTENT(IN), OPTIONAL :: id_ew
      INTEGER(i4),                       INTENT(IN), OPTIONAL :: id_perio
      INTEGER(i4),                       INTENT(IN), OPTIONAL :: id_pivot
      TYPE(TDIM) , DIMENSION(ip_maxdim), INTENT(IN), OPTIONAL :: td_dim
      LOGICAL                          , INTENT(IN), OPTIONAL :: ld_usempp

      ! function
      TYPE(TMPP)                                   :: tf_mpp

      ! local variable
      CHARACTER(LEN=lc)                            :: cl_type

      INTEGER(i4)      , DIMENSION(2)              :: il_shape
      INTEGER(i4)                                  :: il_niproc
      INTEGER(i4)                                  :: il_njproc

      TYPE(TDIM)                                   :: tl_dim

      TYPE(TATT)                                   :: tl_att

      TYPE(TLAY)                                   :: tl_lay

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! clean mpp
      CALL mpp_clean(tf_mpp)

      ! check type
      cl_type=''
      IF( PRESENT(cd_type) ) cl_type=TRIM(ADJUSTL(cd_type))

      IF( TRIM(cl_type) /= '' )THEN
         SELECT CASE(TRIM(cd_type))
            CASE('cdf')
               tf_mpp%c_type='cdf'
            CASE('dimg')
               tf_mpp%c_type='dimg'
            CASE DEFAULT
               CALL logger_warn("MPP INIT: type "//TRIM(cd_type)//&
                  &             " unknown. type dimg will be used for mpp "//&
                  &             TRIM(tf_mpp%c_name) )
               tf_mpp%c_type='dimg'
         END SELECT
      ELSE
         tf_mpp%c_type=TRIM(file_get_type(cd_file))
      ENDIF

      ! get mpp name
      tf_mpp%c_name=TRIM(file_rename(cd_file))

      ! get global domain dimension
      il_shape(:)=SHAPE(id_mask)

      IF( PRESENT(td_dim) )THEN
         DO ji=1,ip_maxdim
            IF( td_dim(ji)%l_use )THEN
               CALL mpp_add_dim(tf_mpp, td_dim(ji))
            ENDIF
         ENDDO
      ELSE
         tl_dim=dim_init('X',il_shape(1))
         CALL mpp_add_dim(tf_mpp, tl_dim)

         tl_dim=dim_init('Y',il_shape(2))
         CALL mpp_add_dim(tf_mpp, tl_dim)

         ! clean
         CALL dim_clean(tl_dim)
      ENDIF

      IF( (       PRESENT(id_niproc)  .AND. (.NOT. PRESENT(id_njproc))) .OR. &
          ((.NOT. PRESENT(id_niproc)) .AND.        PRESENT(id_njproc) ) )THEN
          CALL logger_warn( "MPP INIT: number of processors following I and J "//&
          & "should be both specified")
      ELSE
         ! get number of processors following I and J
         IF( PRESENT(id_niproc) ) tf_mpp%i_niproc=id_niproc
         IF( PRESENT(id_njproc) ) tf_mpp%i_njproc=id_njproc
      ENDIF

      ! get maximum number of processors to be used
      IF( PRESENT(id_nproc) ) tf_mpp%i_nproc = id_nproc

      ! get overlap region length
      IF( PRESENT(id_preci) ) tf_mpp%i_preci= id_preci
      IF( PRESENT(id_precj) ) tf_mpp%i_precj= id_precj

      ! east-west overlap
      IF( PRESENT(id_ew) ) tf_mpp%i_ew= id_ew
      ! NEMO periodicity
      IF( PRESENT(id_perio) ) tf_mpp%i_perio= id_perio
      IF( PRESENT(id_pivot) ) tf_mpp%i_pivot= id_pivot
      !
      IF( PRESENT(ld_usempp) ) tf_mpp%l_usempp = ld_usempp

      IF( tf_mpp%i_nproc  /= 0 .AND. &
      &   tf_mpp%i_niproc /= 0 .AND. &
      &   tf_mpp%i_njproc /= 0 .AND. &
      &   tf_mpp%i_nproc  >  tf_mpp%i_niproc * tf_mpp%i_njproc )THEN

         CALL logger_error("MPP INIT: invalid domain decomposition ")
         CALL logger_debug("MPP INIT: "// &
            &              TRIM(fct_str(tf_mpp%i_nproc))//" > "//&
            &              TRIM(fct_str(tf_mpp%i_niproc))//" x "//&
            &              TRIM(fct_str(tf_mpp%i_njproc)) )

      ELSE
         IF( lm_layout )THEN
            OPEN(im_iumout,FILE='processor.layout')
            WRITE(im_iumout,*)
            WRITE(im_iumout,*) ' optimisation de la partition'
            WRITE(im_iumout,*) ' ----------------------------'
            WRITE(im_iumout,*)
         ENDIF

         IF( tf_mpp%i_niproc /= 0 .AND. tf_mpp%i_njproc /= 0 .AND. &
           &(tf_mpp%i_niproc  > 1 .OR.  tf_mpp%i_njproc  > 1) )THEN

            ! compute domain layout
            tl_lay=layout__init(tf_mpp, id_mask, &
               &                tf_mpp%i_niproc, tf_mpp%i_njproc)
            ! create mpp domain layout
            CALL mpp__create_layout( tf_mpp, tl_lay )

            ! clean
            CALL layout__clean( tl_lay )

         ELSEIF( tf_mpp%i_nproc  > 1 )THEN

            ! optimiz
            CALL mpp__optimiz( tf_mpp, id_mask, tf_mpp%i_nproc )

         ELSE

            CALL logger_warn("MPP INIT: number of processor to be used "//&
            &                "not specify. force output on one file.")
            ! number of proc to get proc size close to im_psize
            il_niproc=INT(il_shape(jp_I)/im_psize)+1
            il_njproc=INT(il_shape(jp_J)/im_psize)+1
         
            tf_mpp%l_usempp=.FALSE.
            tl_lay=layout__init( tf_mpp, id_mask,  &
                               & il_niproc, il_njproc )

            ! create mpp domain layout
            CALL mpp__create_layout( tf_mpp, tl_lay )

            ! clean
            CALL layout__clean( tl_lay )

         ENDIF

         CALL logger_info("MPP INIT: domain decoposition : "//&
         &  'niproc('//TRIM(fct_str(tf_mpp%i_niproc))//') * '//&
         &  'njproc('//TRIM(fct_str(tf_mpp%i_njproc))//') = '//&
         &  'nproc('//TRIM(fct_str(tf_mpp%i_nproc))//')' )

         ! get domain type
         CALL mpp_get_dom( tf_mpp )

         DO ji=1,tf_mpp%i_nproc

            ! get processor size
            il_shape(:)=mpp_get_proc_size( tf_mpp, ji )

            tl_dim=dim_init('X',il_shape(1))
            CALL file_move_dim(tf_mpp%t_proc(ji), tl_dim)

            tl_dim=dim_init('Y',il_shape(2))
            CALL file_move_dim(tf_mpp%t_proc(ji), tl_dim)            

            IF( PRESENT(td_dim) )THEN
               IF( td_dim(jp_K)%l_use )THEN
                  CALL file_move_dim(tf_mpp%t_proc(ji), td_dim(jp_K))
               ENDIF
               IF( td_dim(jp_L)%l_use )THEN
                  CALL file_move_dim(tf_mpp%t_proc(ji), td_dim(jp_L))
               ENDIF
            ENDIF
            ! add type
            tf_mpp%t_proc(ji)%c_type=TRIM(tf_mpp%c_type)

            ! clean
            CALL dim_clean(tl_dim)

         ENDDO

         ! add global attribute
         tl_att=att_init("DOMAIN_number_total",tf_mpp%i_nproc)
         CALL mpp_add_att(tf_mpp, tl_att)

         tl_att=att_init("DOMAIN_LOCAL",TRIM(tf_mpp%c_dom))
         CALL mpp_add_att(tf_mpp, tl_att)

         tl_att=att_init("DOMAIN_I_number_total",tf_mpp%i_niproc)
         CALL mpp_add_att(tf_mpp, tl_att)

         tl_att=att_init("DOMAIN_J_number_total",tf_mpp%i_njproc)
         CALL mpp_add_att(tf_mpp, tl_att)

         tl_att=att_init("DOMAIN_size_global",tf_mpp%t_dim(1:2)%i_len)
         CALL mpp_add_att(tf_mpp, tl_att)

         CALL mpp__compute_halo(tf_mpp) 
      ENDIF

   END FUNCTION mpp__init_mask
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION mpp__init_var(cd_file, td_var,               &
         &                id_niproc, id_njproc, id_nproc,&
         &                id_preci, id_precj, cd_type,   &
         &                id_perio, id_pivot, ld_usempp) &
         &  RESULT(tf_mpp)
   !-------------------------------------------------------------------
   !> @brief
   !> This function initialise mpp structure, given variable strcuture 
   !> and optionaly number of processor following I and J
   !> @detail
   !> - If no total number of processor is defined (id_nproc), optimize 
   !> the domain decomposition (look for the domain decomposition with 
   !> the most land processor to remove)
   !> - length of the overlap region (id_preci, id_precj) could be specify
   !> in I and J direction (default value is 1)
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[in] cd_file   file name of one file composing mpp domain
   !> @param[in] td_var    variable structure
   !> @param[in] id_niproc number of processors following i
   !> @param[in] id_njproc number of processors following j
   !> @param[in] id_nproc  total number of processors
   !> @param[in] id_preci  i-direction overlap region
   !> @param[in] id_precj  j-direction overlap region
   !> @param[in] cd_type   type of the files (cdf, cdf4, dimg)
   !> @param[in] id_perio  NEMO periodicity index
   !> @param[in] id_pivot  NEMO pivot point index F(0),T(1)
   !> @return mpp structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      CHARACTER(LEN=*), INTENT(IN) :: cd_file
      TYPE(TVAR),       INTENT(IN) :: td_var
      INTEGER(i4),      INTENT(IN), OPTIONAL :: id_niproc
      INTEGER(i4),      INTENT(IN), OPTIONAL :: id_njproc
      INTEGER(i4),      INTENT(IN), OPTIONAL :: id_nproc
      INTEGER(i4),      INTENT(IN), OPTIONAL :: id_preci
      INTEGER(i4),      INTENT(IN), OPTIONAL :: id_precj
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: cd_type
      INTEGER(i4),      INTENT(IN), OPTIONAL :: id_perio
      INTEGER(i4),      INTENT(IN), OPTIONAL :: id_pivot
      LOGICAL,          INTENT(IN), OPTIONAL :: ld_usempp
     
      ! function
      TYPE(TMPP)                   :: tf_mpp

      ! local variable
      INTEGER(i4), DIMENSION(:,:,:), ALLOCATABLE :: il_mask
      !----------------------------------------------------------------

      IF( ASSOCIATED(td_var%d_value) )THEN
         ALLOCATE( il_mask(td_var%t_dim(1)%i_len, &
         &                 td_var%t_dim(2)%i_len, &
         &                 td_var%t_dim(3)%i_len) )
         il_mask(:,:,:)=var_get_mask(td_var)
         
         CALL logger_info("MPP INIT: mask compute from variable "//&
            &             TRIM(td_var%c_name))
         tf_mpp = mpp_init( cd_file, il_mask(:,:,1),       &
            &               id_niproc, id_njproc, id_nproc,&
            &               id_preci, id_precj, cd_type,   &
            &               id_ew=td_var%i_ew,             &
            &               id_perio=id_perio, id_pivot=id_pivot,&
            &               ld_usempp=ld_usempp)

         DEALLOCATE(il_mask)
      ELSE
         CALL logger_error("MPP INIT: variable value not define.")
      ENDIF

   END FUNCTION mpp__init_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION mpp__init_file(td_file, id_ew, id_perio, id_pivot) &
         & RESULT(tf_mpp)
   !-------------------------------------------------------------------
   !> @brief This function initalise a mpp structure given file structure. 
   !> @details 
   !> It reads restart dimg files, or some netcdf files.
   !>
   !> @warning 
   !>  netcdf file must contains some attributes:
   !>    - DOMAIN_number_total 
   !>    - DOMAIN_size_global
   !>    - DOMAIN_number
   !>    - DOMAIN_position_first
   !>    - DOMAIN_position_last
   !>    - DOMAIN_halo_size_start
   !>    - DOMAIN_halo_size_end
   !>  or the file is assume to be no mpp file.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2016
   !> - mismatch with "halo" indices, use mpp__compute_halo
   !> @date Marsh, 2017
   !> - netcdf proc indices from zero to N-1
   !> - copy file periodicity to mpp structure 
   !> @date August, 2017
   !> - force to use domain decomposition to enhance read of monoproc file
   !>
   !> @param[in] td_file   file strcuture
   !> @param[in] id_ew     east-west overlap
   !> @param[in] id_perio  NEMO periodicity index
   !> @param[in] id_pivot  NEMO pivot point index F(0),T(1)
   !> @return mpp structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TFILE), INTENT(IN)           :: td_file
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_ew
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_perio
      INTEGER(i4), INTENT(IN), OPTIONAL :: id_pivot

      ! function
      TYPE(TMPP)                        :: tf_mpp

      ! local variable
      INTEGER(i4)                               :: il_nproc
      INTEGER(i4)                               :: il_attid
      INTEGER(i4), DIMENSION(2)                 :: il_shape

      INTEGER(i4), DIMENSION(:,:), ALLOCATABLE  :: il_mask
      INTEGER(i4), DIMENSION(ip_maxdim)         :: il_dim
      INTEGER(i4)                               :: il_niproc
      INTEGER(i4)                               :: il_njproc

      TYPE(TDIM)                                :: tl_dim

      TYPE(TATT)                                :: tl_att

      TYPE(TFILE)                               :: tl_file

      TYPE(TMPP)                                :: tl_mpp

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! clean mpp
      CALL mpp_clean(tf_mpp)

      ! check file type
      SELECT CASE( TRIM(td_file%c_type) )
         CASE('cdf')
            ! need to read all file to get domain decomposition
            tl_file=file_copy(td_file)

            ! open file
            CALL iom_open(tl_file)

            ! read first file domain decomposition
            tl_mpp=mpp__init_file_cdf(tl_file)

            ! get number of processor/file to be read
            il_nproc = 1
            il_attid = 0

            IF( ASSOCIATED(tl_file%t_att) )THEN
               il_attid=att_get_id( tl_file%t_att, "DOMAIN_number_total" )
            ENDIF
            IF( il_attid /= 0 )THEN
               il_nproc = INT(tl_file%t_att(il_attid)%d_value(1))
            ENDIF

            ! close file
            CALL iom_close(tl_file)

            IF( il_nproc /= 1 )THEN
               DO ji=1,il_nproc

                  ! clean mpp strcuture
                  CALL mpp_clean(tl_mpp)
 
                  ! get filename (from 0 to n-1) 
                  tl_file=file_rename(td_file,ji-1)
 
                  ! open file
                  CALL iom_open(tl_file)

                  ! read domain decomposition
                  tl_mpp = mpp__init_file_cdf(tl_file)
                  IF( ji == 1 )THEN
                     tf_mpp=mpp_copy(tl_mpp)
                  ELSE
                     IF( ANY( tf_mpp%t_dim(1:2)%i_len /= &
                              tl_mpp%t_dim(1:2)%i_len) )THEN

                        CALL logger_error("MPP INIT READ: dimension from file "//&
                        &     TRIM(tl_file%c_name)//" and mpp strcuture "//&
                        &     TRIM(tf_mpp%c_name)//"differ ")

                     ELSE

                        ! add processor to mpp strcuture
                        CALL mpp__add_proc(tf_mpp, tl_mpp%t_proc(1))

                     ENDIF
                  ENDIF

                  ! close file
                  CALL iom_close(tl_file)

               ENDDO
               IF( tf_mpp%i_nproc /= il_nproc )THEN
                  CALL logger_error("MPP INIT READ: some processors can't be added &
                  &               to mpp structure")
               ENDIF

            ELSE

               ! force to use domain decomposition to enhance read of input

               ! create pseudo mask 
               il_dim(:)=tl_mpp%t_dim(:)%i_len
               ALLOCATE(il_mask(il_dim(jp_I),il_dim(jp_J)))
               il_mask(:,:)=1

               ! number of proc to get proc size close to im_psize
               il_niproc=INT(il_dim(jp_I)/im_psize)+1
               il_njproc=INT(il_dim(jp_J)/im_psize)+1

               ! compute domain layout
               ! output will be written on one file
               tf_mpp=mpp_init(tl_mpp%c_name, il_mask, il_niproc, il_njproc,&
                  &            id_perio=tl_file%i_perio, &   
                  &            ld_usempp=.FALSE. )

               ! add var
               DO ji=1,tl_mpp%t_proc(1)%i_nvar
                  CALL mpp_add_var(tf_mpp, tl_mpp%t_proc(1)%t_var(ji))
               ENDDO

            ENDIF

            ! mpp type
            tf_mpp%c_type=TRIM(td_file%c_type)

            ! mpp domain type
            CALL mpp_get_dom(tf_mpp)

            ! create some attributes for domain decomposition (use with dimg file)
            tl_att=att_init( "DOMAIN_number_total", tf_mpp%i_nproc )
            CALL mpp_move_att(tf_mpp, tl_att)

            CALL mpp__compute_halo(tf_mpp)
 
            ! clean
            CALL mpp_clean(tl_mpp)
            CALL att_clean(tl_att)

         CASE('dimg')
            ! domain decomposition could be read in one file

            tl_file=file_copy(td_file)
            ! open file
            CALL logger_debug("MPP INIT READ: open file "//TRIM(tl_file%c_name))
            CALL iom_open(tl_file)

            CALL logger_debug("MPP INIT READ: read mpp structure ")
            ! read mpp structure
            tf_mpp=mpp__init_file_rstdimg(tl_file)

            ! mpp type
            tf_mpp%c_type=TRIM(td_file%c_type)

            ! mpp domain type
            CALL logger_debug("MPP INIT READ: mpp_get_dom ")
            CALL mpp_get_dom(tf_mpp)

            ! get processor size
            CALL logger_debug("MPP INIT READ: get processor size ")
            DO ji=1,tf_mpp%i_nproc

               il_shape(:)=mpp_get_proc_size( tf_mpp, ji )

               tl_dim=dim_init('X',il_shape(1))
               CALL file_add_dim(tf_mpp%t_proc(ji), tl_dim)

               tl_dim=dim_init('Y',il_shape(2))
               CALL file_add_dim(tf_mpp%t_proc(ji), tl_dim)            

               ! clean
               CALL dim_clean(tl_dim)

            ENDDO

            ! close file
            CALL iom_close(tl_file)

         CASE DEFAULT
            CALL logger_error("MPP INIT READ: invalid type for file "//&
            &              TRIM(tl_file%c_name))
      END SELECT

      ! east west overlap
      IF( PRESENT(id_ew) ) tf_mpp%i_ew=id_ew
      ! NEMO periodicity
      IF( PRESENT(id_perio) )THEN
         tf_mpp%i_perio= id_perio
         SELECT CASE(id_perio)
         CASE(3,4)
            tf_mpp%i_pivot=1
         CASE(5,6)
            tf_mpp%i_pivot=0
         CASE DEFAULT
            tf_mpp%i_pivot=1
         END SELECT
      ENDIF

      IF( PRESENT(id_pivot) ) tf_mpp%i_pivot= id_pivot

      ! clean 
      CALL file_clean(tl_file)

   END FUNCTION mpp__init_file
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION mpp__init_file_cdf(td_file) &
         &  RESULT(tf_mpp)
   !-------------------------------------------------------------------
   !> @brief This function initalise a mpp structure, 
   !> reading some netcdf files.
   !>
   !> @details 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2015 
   !> - add only use dimension in MPP structure
   !> @date January, 2016
   !> - mismatch with "halo" indices, use mpp__read_halo
   !>
   !> @param[in] td_file   file strcuture
   !> @return mpp structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TFILE), INTENT(IN) :: td_file

      ! function
      TYPE(TMPP)              :: tf_mpp

      ! local variable
      INTEGER(i4) :: il_attid  ! attribute id
      
      LOGICAL     :: ll_exist
      LOGICAL     :: ll_open

      TYPE(TATT)  :: tl_att

      TYPE(TDIM)  :: tl_dim
      
      TYPE(TFILE) :: tl_proc
      !----------------------------------------------------------------

      CALL logger_trace("MPP INIT READ: netcdf file "//TRIM(td_file%c_name))

      INQUIRE( FILE=TRIM(td_file%c_name), EXIST=ll_exist, OPENED=ll_open )
      ! ll_open do not work for netcdf file, return always FALSE
      IF( ll_exist )THEN

         IF( td_file%i_id == 0 )THEN
            CALL logger_info(" id "//TRIM(fct_str(td_file%i_id))) 
            CALL logger_error("MPP INIT READ: netcdf file "//&
               &  TRIM(td_file%c_name)//" not opened")
         ELSE

            ! get mpp name
            tf_mpp%c_name=TRIM( file_rename(td_file%c_name) )

            ! add type
            tf_mpp%c_type="cdf"

            ! global domain size
            il_attid = 0
            IF( ASSOCIATED(td_file%t_att) )THEN
               il_attid=att_get_id( td_file%t_att, "DOMAIN_size_global" )
            ENDIF
            IF( il_attid /= 0 )THEN
               tl_dim=dim_init('X',INT(td_file%t_att(il_attid)%d_value(1)))
               CALL mpp_add_dim(tf_mpp,tl_dim)

               tl_dim=dim_init('Y',INT(td_file%t_att(il_attid)%d_value(2)))
               CALL mpp_add_dim(tf_mpp,tl_dim)
            ELSE ! assume only one file (not mpp)
               tl_dim=dim_init( td_file%t_dim(1)%c_name, td_file%t_dim(1)%i_len)
               CALL mpp_add_dim(tf_mpp,tl_dim)

               tl_dim=dim_init( td_file%t_dim(2)%c_name, td_file%t_dim(2)%i_len)
               CALL mpp_add_dim(tf_mpp,tl_dim)
            ENDIF

            IF( td_file%t_dim(3)%l_use )THEN
               tl_dim=dim_init( td_file%t_dim(3)%c_name, td_file%t_dim(3)%i_len)
               CALL mpp_add_dim(tf_mpp,tl_dim)
            ENDIF

            IF( td_file%t_dim(4)%l_use )THEN
               tl_dim=dim_init( td_file%t_dim(4)%c_name, td_file%t_dim(4)%i_len)
               CALL mpp_add_dim(tf_mpp,tl_dim)
            ENDIF

            ! initialise file/processor
            tl_proc=file_copy(td_file)

            ! processor id
            il_attid = 0
            IF( ASSOCIATED(td_file%t_att) )THEN
               il_attid=att_get_id( td_file%t_att, "DOMAIN_number" )
            ENDIF
            IF( il_attid /= 0 )THEN
               tl_proc%i_pid = INT(td_file%t_att(il_attid)%d_value(1))
            ELSE
               tl_proc%i_pid = 1
            ENDIF

            ! processor dimension
            tl_proc%t_dim(:)=dim_copy(td_file%t_dim(:))

            CALL mpp__read_halo(tl_proc, tf_mpp%t_dim(:) )

            ! add attributes
            tl_att=att_init( "DOMAIN_size_global", tf_mpp%t_dim(:)%i_len)
            CALL file_move_att(tl_proc, tl_att)

            tl_att=att_init( "DOMAIN_number", tl_proc%i_pid )
            CALL file_move_att(tl_proc, tl_att)

            ! add processor to mpp structure
            CALL mpp__add_proc(tf_mpp, tl_proc)

            ! clean 
            CALL file_clean(tl_proc)
            CALL dim_clean(tl_dim)
            CALL att_clean(tl_att)
         ENDIF

      ELSE

         CALL logger_error("MPP INIT READ: netcdf file "//TRIM(td_file%c_name)//&
         &  " do not exist")

      ENDIF

   END FUNCTION mpp__init_file_cdf
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION mpp__init_file_rstdimg(td_file) &
         &  RESULT(tf_mpp)
   !-------------------------------------------------------------------
   !> @brief This function initalise a mpp structure, 
   !> reading one dimg restart file.
   !>
   !> @details 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date January, 2016
   !> - mismatch with "halo" indices, use mpp__compute_halo
   !> @date January,2019
   !> - clean file structure
   !>
   !> @param[in] td_file   file strcuture
   !> @return mpp structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TFILE), INTENT(IN) :: td_file

      ! function
      TYPE(TMPP)              :: tf_mpp

      ! local variable
      INTEGER(i4)       :: il_status
      INTEGER(i4)       :: il_recl                          ! record length 
      INTEGER(i4)       :: il_nx, il_ny, il_nz              ! x,y,z dimension 
      INTEGER(i4)       :: il_n0d, il_n1d, il_n2d, il_n3d   ! number of 0/1/2/3D variables 
      INTEGER(i4)       :: il_iglo, il_jglo                 ! domain global size
      INTEGER(i4)       :: il_rhd                           ! record of the header infos
      INTEGER(i4)       :: il_pni, il_pnj, il_pnij          ! domain decomposition
      INTEGER(i4)       :: il_area                          ! domain index

      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_lci
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_ldi
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_lei
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_impp
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_lcj
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_ldj
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_lej
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_jmpp

      LOGICAL           ::  ll_exist
      LOGICAL           ::  ll_open

      CHARACTER(LEN=lc) :: cl_file

      TYPE(TDIM)        :: tl_dim ! dimension structure
      TYPE(TATT)        :: tl_att
      TYPE(TFILE)       :: tl_proc

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      INQUIRE( FILE=TRIM(td_file%c_name), EXIST=ll_exist, OPENED=ll_open)
      IF( ll_exist )THEN

         IF( .NOT. ll_open )THEN
            CALL logger_error("MPP INIT READ: dimg file "//TRIM(td_file%c_name)//&
            &  " not opened")
         ELSE

            ! read first record 
            READ( td_file%i_id, IOSTAT=il_status, REC=1 )& 
               &  il_recl,                         &
               &  il_nx, il_ny, il_nz,             &
               &  il_n0d, il_n1d, il_n2d, il_n3d,  &
               &  il_rhd,                          &
               &  il_pni, il_pnj, il_pnij,         &
               &  il_area
            CALL fct_err(il_status)
            IF( il_status /= 0 )THEN
               CALL logger_error("MPP INIT READ: read first line header of "//&
               &              TRIM(td_file%c_name))
            ENDIF

            ! get mpp name
            tf_mpp%c_name=TRIM( file_rename(td_file%c_name) )

            ! add type
            tf_mpp%c_type="dimg"

            ! number of processors to be read
            tf_mpp%i_nproc  = il_pnij
            tf_mpp%i_niproc = il_pni
            tf_mpp%i_njproc = il_pnj

            IF( ASSOCIATED(tf_mpp%t_proc) )THEN
               CALL file_clean(tf_mpp%t_proc(:))
               DEALLOCATE(tf_mpp%t_proc)
            ENDIF
            ALLOCATE( tf_mpp%t_proc(il_pnij) , stat=il_status )

            ALLOCATE(il_lci (il_pnij))
            ALLOCATE(il_lcj (il_pnij))
            ALLOCATE(il_ldi (il_pnij))
            ALLOCATE(il_ldj (il_pnij))
            ALLOCATE(il_lei (il_pnij))
            ALLOCATE(il_lej (il_pnij))
            ALLOCATE(il_impp(il_pnij))
            ALLOCATE(il_jmpp(il_pnij))

            tl_proc=file_copy(td_file)
            ! remove dimension from file
            CALL dim_clean(tl_proc%t_dim(:))
            ! initialise file/processors
            DO ji=1,tf_mpp%i_nproc
               tf_mpp%t_proc(ji)=file_copy(tl_proc)
            ENDDO

            IF( il_status /= 0 )THEN
               CALL logger_error("MPP INIT READ: not enough space to read domain &
               &              decomposition in file "//TRIM(td_file%c_name))
            ENDIF

            ! read first record 
            READ( td_file%i_id, IOSTAT=il_status, REC=1 )& 
               &  il_recl,                         &
               &  il_nx, il_ny, il_nz,             &
               &  il_n0d, il_n1d, il_n2d, il_n3d,  &
               &  il_rhd,                          &
               &  il_pni, il_pnj, il_pnij,         &
               &  il_area,                         &
               &  il_iglo, il_jglo,                &
               &  il_lci(1:il_pnij),               &
               &  il_lcj(1:il_pnij),               &
               &  il_ldi(1:il_pnij),               &
               &  il_ldj(1:il_pnij),               &
               &  il_lei(1:il_pnij),               &
               &  il_lej(1:il_pnij),               &
               &  il_impp(1:il_pnij),              &
               &  il_jmpp(1:il_pnij)
            CALL fct_err(il_status)
            IF( il_status /= 0 )THEN
               CALL logger_error("MPP INIT READ: read first line of "//&
               &              TRIM(td_file%c_name))
            ENDIF

            tf_mpp%t_proc(1:il_pnij)%i_lci = il_lci (1:il_pnij)
            tf_mpp%t_proc(1:il_pnij)%i_lcj = il_lcj (1:il_pnij) 
            tf_mpp%t_proc(1:il_pnij)%i_ldi = il_ldi (1:il_pnij) 
            tf_mpp%t_proc(1:il_pnij)%i_ldj = il_ldj (1:il_pnij) 
            tf_mpp%t_proc(1:il_pnij)%i_lei = il_lei (1:il_pnij) 
            tf_mpp%t_proc(1:il_pnij)%i_lej = il_lej (1:il_pnij) 
            tf_mpp%t_proc(1:il_pnij)%i_impp= il_impp(1:il_pnij)
            tf_mpp%t_proc(1:il_pnij)%i_jmpp= il_jmpp(1:il_pnij)

            DEALLOCATE(il_lci) 
            DEALLOCATE(il_lcj) 
            DEALLOCATE(il_ldi) 
            DEALLOCATE(il_ldj) 
            DEALLOCATE(il_lei) 
            DEALLOCATE(il_lej) 
            DEALLOCATE(il_impp)
            DEALLOCATE(il_jmpp)

            ! global domain size
            tl_dim=dim_init('X',il_iglo)
            CALL mpp_add_dim(tf_mpp,tl_dim)
            tl_dim=dim_init('Y',il_jglo)
            CALL mpp_add_dim(tf_mpp,tl_dim)

            tl_dim=dim_init('Z',il_nz)
            CALL mpp_add_dim(tf_mpp,tl_dim)

            DO ji=1,tf_mpp%i_nproc

               ! get file name
               cl_file =  file_rename(td_file%c_name,ji)
               tf_mpp%t_proc(ji)%c_name = TRIM(cl_file)
               ! update processor id
               tf_mpp%t_proc(ji)%i_pid=ji

               ! add attributes
               tl_att=att_init( "DOMAIN_number", ji )
               CALL file_move_att(tf_mpp%t_proc(ji), tl_att) 

            ENDDO
 
            ! add type
            tf_mpp%t_proc(:)%c_type="dimg"

            ! add attributes
            tl_att=att_init("DOMAIN_size_global", tf_mpp%t_dim(:)%i_len)
            CALL mpp_move_att(tf_mpp, tl_att)

            tl_att=att_init("DOMAIN_number_total", tf_mpp%i_nproc)
            CALL mpp_move_att(tf_mpp, tl_att)

            tl_att=att_init("DOMAIN_I_number_total", tf_mpp%i_niproc)
            CALL mpp_move_att(tf_mpp, tl_att)

            tl_att=att_init("DOMAIN_J_number_total", tf_mpp%i_njproc)
            CALL mpp_move_att(tf_mpp, tl_att)

            CALL mpp_get_dom( tf_mpp )

            CALL mpp__compute_halo( tf_mpp )

            ! clean
            CALL dim_clean(tl_dim)
            CALL att_clean(tl_att)
            CALL file_clean(tl_proc)
         ENDIF

      ELSE

         CALL logger_error("MPP INIT READ: dimg file "//TRIM(td_file%c_name)//&
         &  " do not exist")

      ENDIF

   END FUNCTION mpp__init_file_rstdimg
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION mpp__check_proc_dim(td_mpp, td_proc) &
         & RESULT(lf_check)
   !-------------------------------------------------------------------
   !> @brief This function check if variable and mpp structure use same
   !> dimension.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_mpp    mpp structure
   !> @param[in] td_proc   processor structure
   !> @return dimension of processor and mpp structure agree (or not)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP),  INTENT(IN) :: td_mpp
      TYPE(TFILE), INTENT(IN) :: td_proc

      !function
      LOGICAL                 :: lf_check

      ! local variable
      INTEGER(i4) :: il_isize !< i-direction maximum sub domain size 
      INTEGER(i4) :: il_jsize !< j-direction maximum sub domain size
      !----------------------------------------------------------------

      lf_check=.TRUE.
      ! check used dimension 
      IF( td_mpp%i_niproc /= 0 .AND. td_mpp%i_njproc /= 0 )THEN
         ! check with maximum size of sub domain
         il_isize = ( td_mpp%t_dim(1)%i_len - 2*td_mpp%i_preci + &
            &        (td_mpp%i_niproc-1) ) / td_mpp%i_niproc + 2*td_mpp%i_preci
         il_jsize = ( td_mpp%t_dim(2)%i_len - 2*td_mpp%i_precj + &
            &        (td_mpp%i_njproc-1) ) / td_mpp%i_njproc + 2*td_mpp%i_precj

         IF( il_isize < td_proc%i_lci .OR.                     &
            &il_jsize < td_proc%i_lcj )THEN

            lf_check=.FALSE.

            CALL logger_error( "MPP CHECK DIM: processor and mpp dimension differ" )

         ENDIF

      ELSE
         ! check with global domain size
         IF( td_mpp%t_dim(1)%i_len < td_proc%i_lci .OR.                     &
            &td_mpp%t_dim(2)%i_len < td_proc%i_lcj )THEN

            lf_check=.FALSE.

            CALL logger_error( "MPP CHECK DIM: processor and mpp dimension differ" )

         ENDIF
      ENDIF

   END FUNCTION mpp__check_proc_dim
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp_add_var(td_mpp, td_var)
   !-------------------------------------------------------------------
   !> @brief
   !>    This subroutine add variable in all files of mpp structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !> @date January, 2019
   !> - do not split variable on domain decomposition, if only one procesor 
   !>
   !> @param[inout] td_mpp mpp strcuture
   !> @param[in]    td_var variable strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP), INTENT(INOUT) :: td_mpp
      TYPE(TVAR), INTENT(INOUT) :: td_var

      ! local variable
      INTEGER(i4) :: il_varid
      TYPE(TVAR)  :: tl_var

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! check if mpp exist
      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN

         CALL logger_error( "MPP ADD VAR: processor decomposition not "//&
         &  "define for mpp "//TRIM(td_mpp%c_name))

      ELSE
         ! check if variable exist
         IF( TRIM(td_var%c_name) == '' .AND. &
         &   TRIM(td_var%c_stdname) == '' )THEN
            CALL logger_error("MPP ADD VAR: variable not define ")
         ELSE
            ! check if variable already in mpp structure
            il_varid=0
            IF( ASSOCIATED(td_mpp%t_proc(1)%t_var) )THEN
               il_varid=var_get_index( td_mpp%t_proc(1)%t_var(:), &
               &                       td_var%c_name, td_var%c_stdname )
            ENDIF

            IF( il_varid /= 0 )THEN

               DO ji=1,td_mpp%t_proc(1)%i_nvar
                  CALL logger_debug( " MPP ADD VAR: in mpp structure : &
                  &  variable "//TRIM(td_mpp%t_proc(1)%t_var(ji)%c_name)//&
                  &  ", standard name "//&
                  &  TRIM(td_mpp%t_proc(1)%t_var(ji)%c_stdname) )
               ENDDO
               CALL logger_error( " MPP ADD VAR: variable "//TRIM(td_var%c_name)//&
               &  ", standard name "//TRIM(td_var%c_stdname)//&
               &  ", already in mpp "//TRIM(td_mpp%c_name) )

            ELSE
 
               CALL logger_info( &
               &  " MPP ADD VAR: add variable "//TRIM(td_var%c_name)//&
               &  ", standard name "//TRIM(td_var%c_stdname)//&
               &  ", in mpp "//TRIM(td_mpp%c_name) )
               ! check used dimension 
               IF( mpp__check_dim(td_mpp, td_var) )THEN
 
                  ! check variable dimension expected
                  CALL var_check_dim(td_var)

                  ! update dimension if need be
                  DO ji=1,ip_maxdim
                     IF( td_var%t_dim(ji)%l_use .AND. &
                     &   .NOT. td_mpp%t_dim(ji)%l_use )THEN
                        CALL mpp_add_dim(td_mpp,td_var%t_dim(ji))
                     ENDIF
                  ENDDO

                  ! add variable in each processor
                  IF( td_mpp%i_nproc == 1 )THEN
                     CALL file_add_var(td_mpp%t_proc(1), td_var)
                  ELSE
                     DO ji=1,td_mpp%i_nproc
                        ! split variable on domain decomposition
                        tl_var=mpp__split_var(td_mpp, td_var, ji)

                        CALL file_add_var(td_mpp%t_proc(ji), tl_var)

                        ! clean
                        CALL var_clean(tl_var)
                     ENDDO
                  ENDIF

               ENDIF
            ENDIF
         ENDIF
      ENDIF

   END SUBROUTINE mpp_add_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION mpp__split_var(td_mpp, td_var, id_procid) &
         & RESULT(tf_var)
   !-------------------------------------------------------------------
   !> @brief This function extract, from variable structure, part that will 
   !> be written in processor id_procid.<br/>
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_mpp    mpp structure
   !> @param[in] td_var    variable structure
   !> @param[in] id_procid processor id
   !> @return variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP),  INTENT(IN) :: td_mpp
      TYPE(TVAR),  INTENT(IN) :: td_var
      INTEGER(i4), INTENT(IN) :: id_procid

      ! function
      TYPE(TVAR)              :: tf_var
   
      ! local variable
      TYPE(TDIM)  :: tl_dim

      INTEGER(i4), DIMENSION(4) :: il_ind
      INTEGER(i4), DIMENSION(2) :: il_size
      INTEGER(i4) :: il_i1
      INTEGER(i4) :: il_i2
      INTEGER(i4) :: il_j1
      INTEGER(i4) :: il_j2
      !----------------------------------------------------------------

      ! copy mpp
      tf_var=var_copy(td_var, ld_value=.FALSE.)

      ! get processor indices
      il_ind(:)=mpp_get_proc_index( td_mpp, id_procid )
      il_i1 = il_ind(1)
      il_i2 = il_ind(2)
      il_j1 = il_ind(3)
      il_j2 = il_ind(4)

      IF( .NOT. td_var%t_dim(1)%l_use )THEN
         il_i1=1 
         il_i2=1 
      ENDIF

      IF( .NOT. td_var%t_dim(2)%l_use )THEN
         il_j1=1 
         il_j2=1 
      ENDIF      

      IF( ASSOCIATED(td_var%d_value) )THEN
         ! remove value over global domain from pointer
         !CALL var_del_value( tf_var )

         ! get processor dimension
         il_size(:)=mpp_get_proc_size( td_mpp, id_procid )

         ! define new dimension in variable structure 
         IF( td_var%t_dim(1)%l_use )THEN
            tl_dim=dim_init( TRIM(td_var%t_dim(1)%c_name), il_size(1) )
            CALL var_move_dim( tf_var, tl_dim )
         ENDIF
         IF( td_var%t_dim(2)%l_use )THEN
            tl_dim=dim_init( TRIM(td_var%t_dim(2)%c_name), il_size(2) )
            CALL var_move_dim( tf_var, tl_dim )      
         ENDIF

         ! add variable value on processor
         CALL var_add_value( tf_var, &
            &                td_var%d_value(il_i1:il_i2, il_j1:il_j2, :, :) )
   
      ELSE

         tf_var%t_dim(jp_I)%i_len=il_i2-il_i1+1
         tf_var%t_dim(jp_J)%i_len=il_j2-il_j1+1

      ENDIF

   END FUNCTION mpp__split_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp__del_var_mpp(td_mpp)
   !-------------------------------------------------------------------
   !> @brief 
   !>  This subroutine delete all variable in mpp strcuture.
   !>
   !> @author J.Paul
   !> @date October, 2014 - Initial version
   !>
   !> @param[inout] td_mpp mpp strcuture
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      TYPE(TMPP), INTENT(INOUT) :: td_mpp

      ! local variable
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      CALL logger_info( &
      &  "MPP CLEAN VAR: reset all variable "//&
      &  "in mpp strcuture "//TRIM(td_mpp%c_name) )

      IF( ASSOCIATED(td_mpp%t_proc) )THEN
         DO ji=td_mpp%t_proc(1)%i_nvar,1,-1
            CALL mpp_del_var(td_mpp, td_mpp%t_proc(1)%t_var(ji))
         ENDDO
      ENDIF

   END SUBROUTINE mpp__del_var_mpp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp__del_var_str(td_mpp, td_var)
   !-------------------------------------------------------------------
   !> @brief
   !>    This subroutine delete variable in mpp structure, given variable
   !> structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_mpp mpp strcuture
   !> @param[in]    td_var variable strcuture
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      TYPE(TMPP), INTENT(INOUT) :: td_mpp
      TYPE(TVAR), INTENT(IN)    :: td_var

      ! local variable
      INTEGER(i4)       :: il_varid
      CHARACTER(LEN=lc) :: cl_name

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! check if mpp exist
      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN

         CALL logger_error( "MPP DEL VAR: domain decomposition not define "//&
         &               " in mpp strcuture "//TRIM(td_mpp%c_name))

      ELSE

         ! check if variable already in mpp structure
         il_varid = 0
         IF( ASSOCIATED(td_mpp%t_proc(1)%t_var) )THEN
            il_varid=var_get_index( td_mpp%t_proc(1)%t_var(:), &
            &                       td_var%c_name, td_var%c_stdname )
         ENDIF
         IF( il_varid == 0 )THEN
            CALL logger_error( &
            &  "MPP DEL VAR: no variable "//TRIM(td_var%c_name)//&
            &  ", in mpp structure "//TRIM(td_mpp%c_name) )

            DO ji=1,td_mpp%t_proc(1)%i_nvar
               CALL logger_debug( "MPP DEL VAR: in mpp structure : &
               &  variable : "//TRIM(td_mpp%t_proc(1)%t_var(ji)%c_name)//&
               &  ", standard name "//&
               &  TRIM(td_mpp%t_proc(1)%t_var(ji)%c_stdname) )
            ENDDO

         ELSE

            cl_name=TRIM(td_var%c_name)
            DO ji=1,td_mpp%i_nproc
               CALL file_del_var(td_mpp%t_proc(ji), TRIM(cl_name)) 
            ENDDO

         ENDIF
      ENDIF

   END SUBROUTINE mpp__del_var_str
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp__del_var_name(td_mpp, cd_name)
   !-------------------------------------------------------------------
   !> @brief
   !>    This subroutine delete variable in mpp structure, given variable name.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !> @date February, 2015 
   !> - define local variable structure to avoid mistake with pointer
   !> @date January, 2019
   !> - clean variable strcuture
   !>
   !> @param[inout] td_mpp    mpp strcuture
   !> @param[in]    cd_name   variable name
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      TYPE(TMPP)      , INTENT(INOUT) :: td_mpp
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_name

      ! local variable
      INTEGER(i4)       :: il_varid
      TYPE(TVAR)        :: tl_var
      !----------------------------------------------------------------

      ! check if mpp exist
      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN

         CALL logger_error( "MPP DEL VAR: domain decomposition not define "//&
         &               " in mpp strcuture "//TRIM(td_mpp%c_name))

      ELSE

         IF( td_mpp%t_proc(1)%i_nvar == 0 )THEN
            CALL logger_debug( "MPP DEL VAR NAME: no variable associated to mpp &
            &                 structure "//TRIM(td_mpp%c_name) )
         ELSE

            ! get the variable id, in file variable structure
            il_varid=0
            IF( ASSOCIATED(td_mpp%t_proc(1)%t_var) )THEN
               il_varid=var_get_index( td_mpp%t_proc(1)%t_var(:), &
               &                       cd_name )
            ENDIF

            IF( il_varid == 0 )THEN

               CALL logger_warn( &
               &  "MPP DEL VAR : there is no variable with name "//&
               &  "or standard name "//TRIM(ADJUSTL(cd_name))//&
               &  " in mpp structure "//TRIM(td_mpp%c_name))

            ELSE

               tl_var=var_copy(td_mpp%t_proc(1)%t_var(il_varid))
               CALL mpp_del_var(td_mpp, tl_var)
               ! clean
               CALL var_clean(tl_var)
            ENDIF
         ENDIF
      ENDIF

   END SUBROUTINE mpp__del_var_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp_move_var(td_mpp, td_var)
   !-------------------------------------------------------------------
   !> @brief
   !>    This subroutine overwrite variable in mpp structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_mpp mpp strcuture
   !> @param[in]    td_var variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP), INTENT(INOUT) :: td_mpp
      TYPE(TVAR), INTENT(IN)    :: td_var

      !local variable
      TYPE(TVAR) :: tl_var
      !----------------------------------------------------------------

      ! copy variablie
      tl_var=var_copy(td_var)

      ! remove processor
      CALL mpp_del_var(td_mpp, tl_var)

      ! add processor
      CALL mpp_add_var(td_mpp, tl_var)

      ! clean 
      CALL var_clean(tl_var)

   END SUBROUTINE mpp_move_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp__add_proc_unit(td_mpp, td_proc)
   !-------------------------------------------------------------------
   !> @brief
   !>    This subroutine add processor to mpp structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !> @date January, 2019
   !> - deallocate file structure whatever happens 
   !
   !> @param[inout] td_mpp    mpp strcuture
   !> @param[in]    td_proc   processor strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP) , INTENT(INOUT) :: td_mpp
      TYPE(TFILE), INTENT(IN)    :: td_proc

      ! local variable
      INTEGER(i4)                                  :: il_status
      INTEGER(i4)                                  :: il_procid
      INTEGER(i4)      , DIMENSION(1)              :: il_ind

      TYPE(TFILE)      , DIMENSION(:), ALLOCATABLE :: tl_proc

      CHARACTER(LEN=lc)                            :: cl_name
      !----------------------------------------------------------------

      ! check file name
      cl_name=TRIM( file_rename(td_proc%c_name) )
      IF( TRIM(cl_name) /=  TRIM(td_mpp%c_name) )THEN
         CALL logger_warn("MPP ADD PROC: processor name do not match mpp name")
      ENDIF

      il_procid=0
      IF( ASSOCIATED(td_mpp%t_proc) )THEN
         ! check if processor already in mpp structure
         il_ind(:)=MINLOC( td_mpp%t_proc(:)%i_pid, &
                     mask=(td_mpp%t_proc(:)%i_pid==td_proc%i_pid) )
         il_procid=il_ind(1)
      ENDIF

      IF( il_procid /= 0 )THEN

            CALL logger_error( &
            &  "MPP ADD PROC: processor "//TRIM(fct_str(td_proc%i_pid))//&
            &  ", already in mpp structure " )

      ELSE
 
         CALL logger_trace("MPP ADD PROC: add processor "//&
         &               TRIM(fct_str(td_mpp%i_nproc+1))//" in mpp structure")

         IF( td_mpp%i_nproc > 0 )THEN
            ! 
            il_ind(:)=MAXLOC( td_mpp%t_proc(:)%i_pid, &
                        mask=(td_mpp%t_proc(:)%i_pid < td_proc%i_pid) )
            il_procid=il_ind(1)

            ! already other processor in mpp structure
            ALLOCATE( tl_proc(td_mpp%i_nproc), stat=il_status )
            IF(il_status /= 0 )THEN

               CALL logger_error( "MPP ADD PROC: not enough space to put processor &
               &               in mpp structure")

            ELSE
               ! save temporary mpp structure
               tl_proc(:)=file_copy(td_mpp%t_proc(:))

               CALL file_clean( td_mpp%t_proc(:) )
               DEALLOCATE(td_mpp%t_proc)
               ALLOCATE( td_mpp%t_proc(td_mpp%i_nproc+1), stat=il_status)
               IF(il_status /= 0 )THEN

                  CALL logger_error( "MPP ADD PROC: not enough space to put "//&
                  &  "processor in mpp structure ")

               ENDIF

               ! copy processor in mpp before
               ! processor with lower id than new processor
               td_mpp%t_proc( 1:il_procid ) = file_copy(tl_proc( 1:il_procid ))

               ! processor with greater id than new processor
               td_mpp%t_proc( il_procid+1 : td_mpp%i_nproc+1 ) = &
               &              file_copy(tl_proc( il_procid : td_mpp%i_nproc ))

               ! clean
               CALL file_clean(tl_proc(:))
            ENDIF
            DEALLOCATE(tl_proc)

         ELSE
            
            ! no processor in mpp structure
            IF( ASSOCIATED(td_mpp%t_proc) )THEN
               CALL file_clean(td_mpp%t_proc(:))
               DEALLOCATE(td_mpp%t_proc)
            ENDIF
            ALLOCATE( td_mpp%t_proc(td_mpp%i_nproc+1), stat=il_status )
            IF(il_status /= 0 )THEN

               CALL logger_error( "MPP ADD PROC: not enough space to put "//&
               &  "processor in mpp structure " )

            ENDIF
         ENDIF

         ! check dimension
         IF( ANY(td_mpp%t_dim(1:2)%i_len < td_proc%t_dim(1:2)%i_len) )THEN
            CALL logger_error( "MPP ADD PROC: mpp structure and new processor "//&
            &  " dimension differ. ")
            CALL logger_debug("MPP ADD PROC: mpp dimension ("//&
            &  TRIM(fct_str(td_mpp%t_dim(1)%i_len))//","//&
            &  TRIM(fct_str(td_mpp%t_dim(2)%i_len))//")" )
            CALL logger_debug("MPP ADD PROC: processor dimension ("//&
            &  TRIM(fct_str(td_proc%t_dim(1)%i_len))//","//&
            &  TRIM(fct_str(td_proc%t_dim(2)%i_len))//")" )
         ELSE
            td_mpp%i_nproc=td_mpp%i_nproc+1

            ! add new processor
            td_mpp%t_proc(td_mpp%i_nproc)=file_copy(td_proc)
         ENDIF

      ENDIF

   END SUBROUTINE mpp__add_proc_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp__add_proc_arr(td_mpp, td_proc)
   !-------------------------------------------------------------------
   !> @brief
   !>    This subroutine add array of processor to mpp structure.
   !>    @note mpp structure should be empty
   !>
   !> @author J.Paul
   !> @date August, 2017 - Initial version
   !>
   !> @param[inout] td_mpp    mpp strcuture
   !> @param[in]    td_proc   array of processor strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP)               , INTENT(INOUT) :: td_mpp
      TYPE(TFILE), DIMENSION(:), INTENT(IN   ) :: td_proc

      ! local variable
      INTEGER(i4)                                  :: il_status
      INTEGER(i4)                                  :: il_nproc

      CHARACTER(LEN=lc)                            :: cl_name
      !----------------------------------------------------------------

      ! check file name
      cl_name=TRIM( file_rename(td_proc(1)%c_name) )
      IF( TRIM(cl_name) /=  TRIM(td_mpp%c_name) )THEN
         CALL logger_warn("MPP ADD PROC: processor name do not match mpp name")
      ENDIF

      IF( ASSOCIATED(td_mpp%t_proc) )THEN
            CALL logger_error( &
            &  "MPP ADD PROC: some processor(s) already in mpp structure " )

      ELSE
 
         CALL logger_trace("MPP ADD PROC: add array of processor "//&
         &                 " in mpp structure")

         il_nproc=SIZE(td_proc)
         ALLOCATE( td_mpp%t_proc(il_nproc), stat=il_status )
         IF(il_status /= 0 )THEN
            CALL logger_error( "MPP ADD PROC: not enough space to put "//&
            &  "processor in mpp structure " )
         ENDIF

         ! check dimension
         IF( ANY(td_mpp%t_dim(1:2)%i_len < td_proc(1)%t_dim(1:2)%i_len) )THEN
            CALL logger_error( "MPP ADD PROC: mpp structure and new processor "//&
            &  " dimension differ. ")
            CALL logger_debug("MPP ADD PROC: mpp dimension ("//&
            &  TRIM(fct_str(td_mpp%t_dim(1)%i_len))//","//&
            &  TRIM(fct_str(td_mpp%t_dim(2)%i_len))//")" )
            CALL logger_debug("MPP ADD PROC: processor dimension ("//&
            &  TRIM(fct_str(td_proc(1)%t_dim(1)%i_len))//","//&
            &  TRIM(fct_str(td_proc(1)%t_dim(2)%i_len))//")" )
         ELSE
            td_mpp%i_nproc=il_nproc

            ! add new processor
            td_mpp%t_proc(:)=file_copy(td_proc(:))
         ENDIF

      ENDIF

   END SUBROUTINE mpp__add_proc_arr   
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp__del_proc_id(td_mpp, id_procid)
   !-------------------------------------------------------------------
   !> @brief
   !>    This subroutine delete processor in mpp structure, given processor id.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !> @date January, 2019
   !> - clean file structure
   !>
   !> @param[inout] td_mpp    mpp strcuture
   !> @param[in]    id_procid processor id
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP),   INTENT(INOUT) :: td_mpp
      INTEGER(i4),  INTENT(IN)    :: id_procid

      ! local variable
      INTEGER(i4) :: il_status
      INTEGER(i4) :: il_procid
      INTEGER(i4), DIMENSION(1) :: il_ind
      TYPE(TFILE), DIMENSION(:), ALLOCATABLE :: tl_proc

      ! loop indices
      !----------------------------------------------------------------

      il_ind(:)=MINLOC(td_mpp%t_proc(:)%i_pid,td_mpp%t_proc(:)%i_pid==id_procid)
      il_procid=il_ind(1)
      IF( il_procid == 0 )THEN
         CALL logger_error("MPP DEL PROC: no processor "//&
         &                 TRIM(fct_str(id_procid))//&
         &                 " associated to mpp structure")
      ELSE
         CALL logger_trace("DEL PROC: remove processor "//&
         &                 TRIM(fct_str(id_procid)))

         IF( td_mpp%i_nproc > 1 )THEN
            ALLOCATE( tl_proc(td_mpp%i_nproc-1), stat=il_status )
            IF(il_status /= 0 )THEN
               CALL logger_error( "MPP DEL PROC: not enough space to put &
               &  processor in temporary mpp structure")

            ELSE

               ! save temporary processor's mpp structure
               IF( il_procid > 1 )THEN
                  tl_proc(1:il_procid-1)=file_copy(td_mpp%t_proc(1:il_procid-1))
               ENDIF

               IF( il_procid < td_mpp%i_nproc )THEN
                  tl_proc(il_procid:)=file_copy(td_mpp%t_proc(il_procid+1:))
               ENDIF

               ! new number of processor in mpp
               td_mpp%i_nproc=td_mpp%i_nproc-1

               CALL file_clean( td_mpp%t_proc(:) )
               DEALLOCATE(td_mpp%t_proc)
               ALLOCATE( td_mpp%t_proc(td_mpp%i_nproc), stat=il_status )
               IF(il_status /= 0 )THEN

                  CALL logger_error( "MPP DEL PROC: not enough space &
                  &  to put processors in mpp structure " )

               ELSE

                  ! copy processor in mpp before
                  td_mpp%t_proc(:)=file_copy(tl_proc(:))

                  ! update processor id
                  td_mpp%t_proc( il_procid : td_mpp%i_nproc )%i_pid = &
                  &     td_mpp%t_proc( il_procid : td_mpp%i_nproc )%i_pid - 1

               ENDIF
            ENDIF
            ! clean
            CALL file_clean( tl_proc(:) )
            DEALLOCATE(tl_proc)
         ELSE
            CALL file_clean( td_mpp%t_proc(:) )
            DEALLOCATE(td_mpp%t_proc)

            ! new number of processor in mpp
            td_mpp%i_nproc=td_mpp%i_nproc-1
         ENDIF
      ENDIF

   END SUBROUTINE mpp__del_proc_id
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp__del_proc_str(td_mpp, td_proc)
   !-------------------------------------------------------------------
   !> @brief
   !>    This subroutine delete processor in mpp structure, given processor
   !>    structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_mpp : mpp strcuture
   !> @param[in]    td_proc : file/processor structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP),   INTENT(INOUT) :: td_mpp
      TYPE(TFILE),  INTENT(IN)    :: td_proc
      !----------------------------------------------------------------

      IF( td_proc%i_pid >= 0 )THEN
         CALL mpp__del_proc( td_mpp, td_proc%i_pid )
      ELSE
         CALL logger_error("MPP DEL PROC: processor not defined")
      ENDIF

   END SUBROUTINE mpp__del_proc_str
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp__move_proc(td_mpp, td_proc)
   !-------------------------------------------------------------------
   !> @brief
   !>    This subroutine overwrite processor in mpp structure.
   !>
   !> @detail
   !>
   !> @author J.Paul
   !> @date Nov, 2013 - Initial version
   !>
   !> @param[inout] td_mpp    mpp strcuture
   !> @param[in]    id_procid processor id
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP),  INTENT(INOUT) :: td_mpp
      TYPE(TFILE), INTENT(IN)    :: td_proc
      !----------------------------------------------------------------

      ! remove processor
      CALL mpp__del_proc(td_mpp, td_proc)

      ! add processor
      CALL mpp__add_proc(td_mpp, td_proc)

   END SUBROUTINE mpp__move_proc
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp_add_dim(td_mpp, td_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine add a dimension structure in a mpp 
   !> structure.
   !> Do not overwrite, if dimension already in mpp structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2015 
   !> - rewrite the same as way var_add_dim
   !>
   !> @param[inout] td_mpp mpp structure
   !> @param[in] td_dim    dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP), INTENT(INOUT) :: td_mpp
      TYPE(TDIM), INTENT(IN)    :: td_dim

      ! local variable
      INTEGER(i4) :: il_ind

      ! loop indices
      !----------------------------------------------------------------

      IF( td_mpp%i_ndim <= ip_maxdim )THEN

         ! check if dimension already used in mpp structure
         il_ind=SCAN(TRIM(cp_dimorder),TRIM(td_dim%c_sname))
         IF( il_ind == 0 )THEN
            CALL logger_warn( &
            &  " MPP ADD DIM: dimension "//TRIM(td_dim%c_name)//&
            &  ", short name "//TRIM(td_dim%c_sname)//&
            &  ", will not be added in mpp "//TRIM(td_mpp%c_name) )
         ELSEIF( td_mpp%t_dim(il_ind)%l_use )THEN
            CALL logger_error( &
            &  " MPP ADD DIM: dimension "//TRIM(td_dim%c_name)//&
            &  ", short name "//TRIM(td_dim%c_sname)//&
            &  ", already used in mpp "//TRIM(td_mpp%c_name) )
         ELSE

            ! back to disorder dimension array 
            CALL dim_disorder(td_mpp%t_dim(:))

            ! add new dimension
            td_mpp%t_dim(td_mpp%i_ndim+1)=dim_copy(td_dim)

            ! update number of attribute
            td_mpp%i_ndim=COUNT(td_mpp%t_dim(:)%l_use)

         ENDIF
         ! reorder dimension to ('x','y','z','t')
         CALL dim_reorder(td_mpp%t_dim(:))

      ELSE
         CALL logger_error( &
         &  "MPP ADD DIM: too much dimension in mpp "//&
         &  TRIM(td_mpp%c_name)//" ("//TRIM(fct_str(td_mpp%i_ndim))//")")
      ENDIF

   END SUBROUTINE mpp_add_dim
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp_del_dim(td_mpp, td_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine delete a dimension structure in a mpp 
   !> structure.<br/>
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date July, 2015 
   !> - rewrite the same as way var_del_dim
   !>
   !> @param[inout] td_mpp mpp structure
   !> @param[in] td_dim    dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP), INTENT(INOUT) :: td_mpp
      TYPE(TDIM), INTENT(IN)    :: td_dim

      ! local variable
      INTEGER(i4) :: il_ind
      TYPE(TDIM)  :: tl_dim

      ! loop indices
      !----------------------------------------------------------------

      IF( td_mpp%i_ndim <= ip_maxdim )THEN

         CALL logger_trace( &
         &  " MPP DEL DIM: delete dimension "//TRIM(td_dim%c_name)//&
         &  ", short name "//TRIM(td_dim%c_sname)//&
         &  ", in mpp "//TRIM(td_mpp%c_name) )
         
         ! check if dimension already in variable structure
         il_ind=SCAN(TRIM(cp_dimorder),TRIM(td_dim%c_sname))

         ! replace dimension by empty one
         td_mpp%t_dim(il_ind)=dim_copy(tl_dim)

         ! update number of dimension
         td_mpp%i_ndim=COUNT(td_mpp%t_dim(:)%l_use)

         ! reorder dimension to ('x','y','z','t')
         CALL dim_reorder(td_mpp%t_dim)

      ELSE
         CALL logger_error( &
         &  " MPP DEL DIM: too much dimension in mpp "//&
         &  TRIM(td_mpp%c_name)//" ("//TRIM(fct_str(td_mpp%i_ndim))//")")
      ENDIF

   END SUBROUTINE mpp_del_dim
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp_move_dim(td_mpp, td_dim)
   !-------------------------------------------------------------------
   !> @brief This subroutine move a dimension structure 
   !> in mpp structure.
   !> @warning dimension order may have changed
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_mpp mpp structure
   !> @param[in] td_dim    dimension structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP), INTENT(INOUT) :: td_mpp
      TYPE(TDIM), INTENT(IN)    :: td_dim

      ! local variable
      INTEGER(i4) :: il_ind
      INTEGER(i4) :: il_dimid
      !----------------------------------------------------------------

      IF( td_mpp%i_ndim <= ip_maxdim )THEN

         ! check if dimension already in mpp structure
         il_ind=dim_get_index(td_mpp%t_dim(:), td_dim%c_name, td_dim%c_sname)
         IF( il_ind /= 0 )THEN

            il_dimid=td_mpp%t_dim(il_ind)%i_id
            ! replace dimension
            td_mpp%t_dim(il_ind)=dim_copy(td_dim)
            td_mpp%t_dim(il_ind)%i_id=il_dimid
            td_mpp%t_dim(il_ind)%l_use=.TRUE.

         ELSE
            CALL mpp_add_dim(td_mpp, td_dim)
         ENDIF

      ELSE
         CALL logger_error( &
         &  "MPP MOVE DIM: too much dimension in mpp "//&
         &  TRIM(td_mpp%c_name)//" ("//TRIM(fct_str(td_mpp%i_ndim))//")")
      ENDIF

   END SUBROUTINE mpp_move_dim
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp_add_att(td_mpp, td_att)
   !-------------------------------------------------------------------
   !> @brief
   !>    This subroutine add global attribute to mpp structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_mpp mpp strcuture
   !> @param[in]    td_att attribute strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP), INTENT(INOUT) :: td_mpp
      TYPE(TATT), INTENT(IN)    :: td_att

      ! local variable
      INTEGER(i4) :: il_attid

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! check if mpp exist
      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN

         CALL logger_error( "MPP ADD ATT: domain decomposition not define "//&
         &               "for mpp "//TRIM(td_mpp%c_name))

      ELSE
         ! check if variable exist
         IF( TRIM(td_att%c_name) == '' )THEN
            CALL logger_error("MPP ADD ATT: attribute not define ")
         ELSE
            ! check if attribute already in mpp structure
            il_attid=0
            IF( ASSOCIATED(td_mpp%t_proc(1)%t_att) )THEN
               il_attid=att_get_index( td_mpp%t_proc(1)%t_att(:), &
               &                    td_att%c_name )
            ENDIF
            IF( il_attid /= 0 )THEN

               CALL logger_error( " MPP ADD ATT: attribute "//&
               &                 TRIM(td_att%c_name)//&
               &                 ", already in mpp "//TRIM(td_mpp%c_name) )

               DO ji=1,td_mpp%t_proc(1)%i_natt
                  CALL logger_debug( " MPP ADD ATT: in mpp structure : &
                  &  attribute "//TRIM(td_mpp%t_proc(1)%t_att(ji)%c_name) )
               ENDDO

            ELSE
            
               CALL logger_info( &
               &  " MPP ADD ATT: add attribute "//TRIM(td_att%c_name)//&
               &  ", in mpp "//TRIM(td_mpp%c_name) )

               ! add attribute in each processor
               DO ji=1,td_mpp%i_nproc

                  CALL file_add_att(td_mpp%t_proc(ji), td_att)

               ENDDO

            ENDIF
         ENDIF
      ENDIF

   END SUBROUTINE mpp_add_att
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp__del_att_str(td_mpp, td_att)
   !-------------------------------------------------------------------
   !> @brief
   !>    This subroutine delete attribute in mpp structure, given attribute
   !> structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_mpp mpp strcuture
   !> @param[in]    td_att attribute strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP), INTENT(INOUT) :: td_mpp
      TYPE(TATT), INTENT(IN)    :: td_att

      ! local variable
      INTEGER(i4)       :: il_attid
      CHARACTER(LEN=lc) :: cl_name

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ! check if mpp exist
      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN

         CALL logger_warn( "MPP DEL VAR: domain decomposition not define "//&
         &               " in mpp strcuture "//TRIM(td_mpp%c_name))

      ELSE

         ! check if attribute already in mpp structure
         il_attid=0
         IF( ASSOCIATED(td_mpp%t_proc(1)%t_att) )THEN
            il_attid=att_get_index( td_mpp%t_proc(1)%t_att(:), &
            &                    td_att%c_name )
         ENDIF
         IF( il_attid == 0 )THEN
            CALL logger_warn( &
            &  "MPP DEL VAR: no attribute "//TRIM(td_att%c_name)//&
            &  ", in mpp structure "//TRIM(td_mpp%c_name) )

            IF( ASSOCIATED(td_mpp%t_proc(1)%t_att) )THEN
               DO ji=1,td_mpp%t_proc(1)%i_natt
                  CALL logger_debug( "MPP DEL ATT: in mpp structure : &
                  &  attribute : "//TRIM(td_mpp%t_proc(1)%t_att(ji)%c_name) )
               ENDDO
            ENDIF

         ELSE

            cl_name=TRIM(td_att%c_name)
            CALL logger_debug( "MPP DEL ATT: delete in mpp structure : &
            &  attribute : "//TRIM(cl_name) )
            DO ji=1,td_mpp%i_nproc
               CALL file_del_att(td_mpp%t_proc(ji), TRIM(cl_name)) 
            ENDDO

         ENDIF
      ENDIF

   END SUBROUTINE mpp__del_att_str
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp__del_att_name(td_mpp, cd_name)
   !-------------------------------------------------------------------
   !> @brief
   !>    This subroutine delete attribute in mpp structure, given attribute name.
   !>
   !> @detail
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !> @date February, 2015 
   !> - define local attribute structure to avoid mistake with pointer
   !> @date January, 2019
   !> - clean attributes structure
   !>
   !> @param[inout] td_mpp    mpp strcuture
   !> @param[in]    cd_name   attribute name
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP)       , INTENT(INOUT) :: td_mpp
      CHARACTER(LEN=*) , INTENT(IN   ) :: cd_name

      ! local variable
      INTEGER(i4) :: il_attid
      TYPE(TATT)  :: tl_att
      !----------------------------------------------------------------

      ! check if mpp exist
      IF( .NOT. ASSOCIATED(td_mpp%t_proc) )THEN

         CALL logger_warn( "MPP DEL ATT: domain decomposition not define "//&
         &               " in mpp strcuture "//TRIM(td_mpp%c_name))

      ELSE

         IF( td_mpp%t_proc(1)%i_natt == 0 )THEN
            CALL logger_debug( "MPP DEL ATT NAME: no attribute associated to mpp &
            &                 structure "//TRIM(td_mpp%c_name) )
         ELSE

            ! get the attribute id, in file variable structure
            il_attid=0
            IF( ASSOCIATED(td_mpp%t_proc(1)%t_att) )THEN
               il_attid=att_get_id( td_mpp%t_proc(1)%t_att(:), &
               &                    cd_name )
            ENDIF

            IF( il_attid == 0 )THEN

               CALL logger_debug( &
               &  "MPP DEL ATT : there is no attribute with "//&
               &  "name "//TRIM(cd_name)//" in mpp structure "//&
               &  TRIM(td_mpp%c_name))

            ELSE

               tl_att=att_copy(td_mpp%t_proc(1)%t_att(il_attid))
               CALL mpp_del_att(td_mpp, tl_att) 
               ! clean
               CALL att_clean(tl_att)
            ENDIF
         ENDIF
      ENDIF

   END SUBROUTINE mpp__del_att_name
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp_move_att(td_mpp, td_att)
   !-------------------------------------------------------------------
   !> @brief
   !>    This subroutine overwrite attribute in mpp structure.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_mpp mpp strcuture
   !> @param[in]    td_att attribute structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP), INTENT(INOUT) :: td_mpp
      TYPE(TATT), INTENT(IN)    :: td_att

      !local variable
      TYPE(TATT)  :: tl_att
      !----------------------------------------------------------------

      ! copy variable
      tl_att=att_copy(td_att)

      ! remove processor
      CALL mpp_del_att(td_mpp, tl_att)

      ! add processor
      CALL mpp_add_att(td_mpp, tl_att)

      ! clean
      CALL att_clean(tl_att)

   END SUBROUTINE mpp_move_att
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION layout__init(td_mpp, id_mask, id_niproc, id_njproc) &
         & RESULT(tf_lay)
   !-------------------------------------------------------------------
   !> @brief
   !>    This function initialise domain layout
   !> 
   !> @detail
   !> Domain layout is first computed, with domain dimension, overlap between subdomain,
   !> and the number of processors following I and J.
   !> Then the number of sea/land processors is compute with mask
   !>
   !> @author J.Paul
   !> @date October, 2015 - Initial version
   !> @date October, 2016
   !> - compare index to tf_lay number of proc instead of td_mpp (bug fix)
   !>
   !> @param[in] td_mpp mpp strcuture
   !> @param[in] id_mask   sub domain mask (sea=1, land=0)
   !> @pâram[in] id_niproc number of processors following I
   !> @pâram[in] id_njproc number of processors following J
   !> @return domain layout structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP)                 , INTENT(IN) :: td_mpp
      INTEGER(i4), DIMENSION(:,:), INTENT(IN) :: id_mask
      INTEGER(i4)                , INTENT(IN) :: id_niproc
      INTEGER(i4)                , INTENT(IN) :: id_njproc

      ! function
      TYPE(TLAY)                              :: tf_lay

      ! local variable
      INTEGER(i4) :: ii1, ii2
      INTEGER(i4) :: ij1, ij2

      INTEGER(i4) :: il_ldi
      INTEGER(i4) :: il_ldj
      INTEGER(i4) :: il_lei
      INTEGER(i4) :: il_lej

      INTEGER(i4) :: il_isize !< i-direction maximum sub domain size 
      INTEGER(i4) :: il_jsize !< j-direction maximum sub domain size
      INTEGER(i4) :: il_resti !<  
      INTEGER(i4) :: il_restj !<  

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      ! intialise
      tf_lay%i_niproc=id_niproc
      tf_lay%i_njproc=id_njproc

      CALL logger_info( "MPP COMPUTE LAYOUT: compute domain layout with "//&
         &              TRIM(fct_str(tf_lay%i_niproc))//" x "//&
         &              TRIM(fct_str(tf_lay%i_njproc))//" processors")

      ! maximum size of sub domain
      il_isize = ((td_mpp%t_dim(1)%i_len - 2*td_mpp%i_preci + (tf_lay%i_niproc-1))/ &
         &         tf_lay%i_niproc) + 2*td_mpp%i_preci
      il_jsize = ((td_mpp%t_dim(2)%i_len - 2*td_mpp%i_precj + (tf_lay%i_njproc-1))/ &
         &         tf_lay%i_njproc) + 2*td_mpp%i_precj

      il_resti = MOD(td_mpp%t_dim(1)%i_len - 2*td_mpp%i_preci, tf_lay%i_niproc)
      il_restj = MOD(td_mpp%t_dim(2)%i_len - 2*td_mpp%i_precj, tf_lay%i_njproc)
      IF( il_resti == 0 ) il_resti = tf_lay%i_niproc
      IF( il_restj == 0 ) il_restj = tf_lay%i_njproc

      ! compute dimension of each sub domain
      ALLOCATE( tf_lay%i_lci(tf_lay%i_niproc,tf_lay%i_njproc) )
      ALLOCATE( tf_lay%i_lcj(tf_lay%i_niproc,tf_lay%i_njproc) )

      tf_lay%i_lci( 1          : il_resti       , : ) = il_isize
      tf_lay%i_lci( il_resti+1 : tf_lay%i_niproc, : ) = il_isize-1

      tf_lay%i_lcj( : , 1          : il_restj       ) = il_jsize
      tf_lay%i_lcj( : , il_restj+1 : tf_lay%i_njproc) = il_jsize-1

      ! compute first index of each sub domain
      ALLOCATE( tf_lay%i_impp(tf_lay%i_niproc,tf_lay%i_njproc) )
      ALLOCATE( tf_lay%i_jmpp(tf_lay%i_niproc,tf_lay%i_njproc) )

      tf_lay%i_impp(:,:)=1
      tf_lay%i_jmpp(:,:)=1

      IF( tf_lay%i_niproc > 1 )THEN
         DO jj=1,tf_lay%i_njproc
            DO ji=2,tf_lay%i_niproc
               tf_lay%i_impp(ji,jj) = tf_lay%i_impp(ji-1,jj) + &
                  &                   tf_lay%i_lci (ji-1,jj) - 2*td_mpp%i_preci
            ENDDO
         ENDDO
      ENDIF

      IF( tf_lay%i_njproc > 1 )THEN
         DO jj=2,tf_lay%i_njproc
            DO ji=1,tf_lay%i_niproc
               tf_lay%i_jmpp(ji,jj) = tf_lay%i_jmpp(ji,jj-1) + &
                  &                   tf_lay%i_lcj (ji,jj-1) - 2*td_mpp%i_precj
            ENDDO
         ENDDO 
      ENDIF

      ALLOCATE( tf_lay%i_msk(tf_lay%i_niproc,tf_lay%i_njproc) )
      tf_lay%i_msk(:,:)=0
      ! init number of sea/land proc
      tf_lay%i_nsea=0
      tf_lay%i_nland=tf_lay%i_njproc*tf_lay%i_niproc

      ! check if processor is land or sea
      DO jj = 1,tf_lay%i_njproc
         DO ji = 1,tf_lay%i_niproc

            ! compute first and last indoor indices
            ! west boundary
            IF( ji == 1 )THEN
               il_ldi = 1 
            ELSE
               il_ldi = 1 + td_mpp%i_preci
            ENDIF

            ! south boundary
            IF( jj == 1 )THEN
               il_ldj = 1 
            ELSE
               il_ldj = 1 + td_mpp%i_precj
            ENDIF

            ! east boundary
            IF( ji == tf_lay%i_niproc )THEN
               il_lei = tf_lay%i_lci(ji,jj)
            ELSE
               il_lei = tf_lay%i_lci(ji,jj) - td_mpp%i_preci
            ENDIF

            ! north boundary
            IF( jj == tf_lay%i_njproc )THEN
               il_lej = tf_lay%i_lcj(ji,jj)
            ELSE
               il_lej = tf_lay%i_lcj(ji,jj) - td_mpp%i_precj
            ENDIF

            ii1=tf_lay%i_impp(ji,jj) + il_ldi - 1
            ii2=tf_lay%i_impp(ji,jj) + il_lei - 1

            ij1=tf_lay%i_jmpp(ji,jj) + il_ldj - 1
            ij2=tf_lay%i_jmpp(ji,jj) + il_lej - 1

            tf_lay%i_msk(ji,jj)=SUM( id_mask(ii1:ii2,ij1:ij2) )
            IF( tf_lay%i_msk(ji,jj) > 0 )THEN ! sea
               tf_lay%i_nsea =tf_lay%i_nsea +1
               tf_lay%i_nland=tf_lay%i_nland-1
            ENDIF

         ENDDO
      ENDDO

      CALL logger_info( "MPP COMPUTE LAYOUT: sea proc "//TRIM(fct_str(tf_lay%i_nsea)))
      CALL logger_info( "MPP COMPUTE LAYOUT: land proc "//TRIM(fct_str(tf_lay%i_nland)))
      CALL logger_info( "MPP COMPUTE LAYOUT: sum "//TRIM(fct_str( SUM(tf_lay%i_msk(:,:)))))

      tf_lay%i_mean= SUM(tf_lay%i_msk(:,:)) / tf_lay%i_nsea
      tf_lay%i_min = MINVAL(tf_lay%i_msk(:,:),tf_lay%i_msk(:,:)/=0)
      tf_lay%i_max = MAXVAL(tf_lay%i_msk(:,:))

      IF( lm_layout )THEN
         ! print info 
         WRITE(im_iumout,*) ' '
         WRITE(im_iumout,*) " jpni=",tf_lay%i_niproc ," jpnj=",tf_lay%i_njproc
         WRITE(im_iumout,*) " jpi= ",il_isize," jpj= ",il_jsize
         WRITE(im_iumout,*) " iresti=",td_mpp%i_preci," irestj=",td_mpp%i_precj


         WRITE(im_iumout,*) ' nombre de processeurs       ',tf_lay%i_niproc*tf_lay%i_njproc
         WRITE(im_iumout,*) ' nombre de processeurs mer   ',tf_lay%i_nsea
         WRITE(im_iumout,*) ' nombre de processeurs terre ',tf_lay%i_nland
         WRITE(im_iumout,*) ' moyenne de recouvrement     ',tf_lay%i_mean
         WRITE(im_iumout,*) ' minimum de recouvrement     ',tf_lay%i_min
         WRITE(im_iumout,*) ' maximum de recouvrement     ',tf_lay%i_max
      ENDIF

   END FUNCTION layout__init
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE layout__clean(td_lay)
   !-------------------------------------------------------------------
   !> @brief 
   !>  This subroutine clean domain layout strcuture.
   !>
   !> @author J.Paul
   !> @date October, 2015 - Initial version
   !> @date January, 2019
   !> - nullify array in layout structure
   !>
   !> @param[inout] td_lay domain layout strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TLAY),  INTENT(INOUT) :: td_lay
      !----------------------------------------------------------------

      IF( ASSOCIATED(td_lay%i_msk) )THEN
         DEALLOCATE(td_lay%i_msk)
         NULLIFY(td_lay%i_msk)
      ENDIF
      IF( ASSOCIATED(td_lay%i_impp) )THEN
         DEALLOCATE(td_lay%i_impp)
         NULLIFY(td_lay%i_impp)
      ENDIF
      IF( ASSOCIATED(td_lay%i_jmpp) )THEN
         DEALLOCATE(td_lay%i_jmpp)
         NULLIFY(td_lay%i_jmpp)
      ENDIF
      IF( ASSOCIATED(td_lay%i_lci) )THEN
         DEALLOCATE(td_lay%i_lci)
         NULLIFY(td_lay%i_lci)
      ENDIF
      IF( ASSOCIATED(td_lay%i_lcj) )THEN
         DEALLOCATE(td_lay%i_lcj)
         NULLIFY(td_lay%i_lcj)
      ENDIF

      td_lay%i_niproc=0
      td_lay%i_njproc=0
      td_lay%i_nland =0
      td_lay%i_nsea  =0

      td_lay%i_mean  =0
      td_lay%i_min   =0
      td_lay%i_max   =0

   END SUBROUTINE layout__clean
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION layout__copy(td_lay) &
         & RESULT(tf_lay)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine copy domain layout structure in another one.
   !>
   !> @warning do not use on the output of a function who create or read a
   !> structure (ex: tl_seg=seg__copy(seg__init()) is forbidden).
   !> This will create memory leaks.
   !> @warning to avoid infinite loop, do not use any function inside 
   !> this subroutine
   !>
   !> @author J.Paul
   !> @date October, 2015 - Initial Version
   !>
   !> @param[in] td_lay   domain layout structure
   !> @return copy of input domain layout structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TLAY), INTENT(IN)  :: td_lay
      ! function
      TYPE(TLAY) :: tf_lay

      ! local variable
      INTEGER(i4), DIMENSION(2)                :: il_shape
      INTEGER(i4), DIMENSION(:,:), ALLOCATABLE :: il_tmp
      ! loop indices
      !----------------------------------------------------------------

      ! copy scalar 
      tf_lay%i_niproc   = td_lay%i_niproc
      tf_lay%i_njproc   = td_lay%i_njproc
      tf_lay%i_nland    = td_lay%i_nland 
      tf_lay%i_nsea     = td_lay%i_nsea  
      tf_lay%i_mean     = td_lay%i_mean  
      tf_lay%i_min      = td_lay%i_min   
      tf_lay%i_max      = td_lay%i_max   

      ! copy pointers
      IF( ASSOCIATED(tf_lay%i_msk) )THEN
         DEALLOCATE(tf_lay%i_msk)
      ENDIF
      IF( ASSOCIATED(td_lay%i_msk) )THEN
         il_shape(:)=SHAPE(td_lay%i_msk(:,:))
         ALLOCATE( tf_lay%i_msk(il_shape(jp_I),il_shape(jp_J)) )
         tf_lay%i_msk(:,:)=td_lay%i_msk(:,:)
      ENDIF

      IF( ASSOCIATED(tf_lay%i_msk) ) DEALLOCATE(tf_lay%i_msk)
      IF( ASSOCIATED(td_lay%i_msk) )THEN
         il_shape(:)=SHAPE(td_lay%i_msk(:,:))
         ALLOCATE(il_tmp(il_shape(jp_I),il_shape(jp_J)))
         il_tmp(:,:)=td_lay%i_msk(:,:)

         ALLOCATE( tf_lay%i_msk(il_shape(jp_I),il_shape(jp_J)) )
         tf_lay%i_msk(:,:)=il_tmp(:,:)

         DEALLOCATE(il_tmp)
      ENDIF

      IF( ASSOCIATED(tf_lay%i_impp) ) DEALLOCATE(tf_lay%i_impp)
      IF( ASSOCIATED(td_lay%i_impp) )THEN
         il_shape(:)=SHAPE(td_lay%i_impp(:,:))
         ALLOCATE(il_tmp(il_shape(jp_I),il_shape(jp_J)))
         il_tmp(:,:)=td_lay%i_impp(:,:)

         ALLOCATE( tf_lay%i_impp(il_shape(jp_I),il_shape(jp_J)) )
         tf_lay%i_impp(:,:)=il_tmp(:,:)

         DEALLOCATE(il_tmp)
      ENDIF

      IF( ASSOCIATED(tf_lay%i_jmpp) ) DEALLOCATE(tf_lay%i_jmpp)
      IF( ASSOCIATED(td_lay%i_jmpp) )THEN
         il_shape(:)=SHAPE(td_lay%i_jmpp(:,:))
         ALLOCATE(il_tmp(il_shape(jp_I),il_shape(jp_J)))
         il_tmp(:,:)=td_lay%i_jmpp(:,:)

         ALLOCATE( tf_lay%i_jmpp(il_shape(jp_I),il_shape(jp_J)) )
         tf_lay%i_jmpp(:,:)=il_tmp(:,:)

         DEALLOCATE(il_tmp)
      ENDIF

      IF( ASSOCIATED(tf_lay%i_lci) ) DEALLOCATE(tf_lay%i_lci)
      IF( ASSOCIATED(td_lay%i_lci) )THEN
         il_shape(:)=SHAPE(td_lay%i_lci(:,:))
         ALLOCATE(il_tmp(il_shape(jp_I),il_shape(jp_J)))
         il_tmp(:,:)=td_lay%i_lci(:,:)

         ALLOCATE( tf_lay%i_lci(il_shape(jp_I),il_shape(jp_J)) )
         tf_lay%i_lci(:,:)=il_tmp(:,:)

         DEALLOCATE(il_tmp)
      ENDIF

      IF( ASSOCIATED(tf_lay%i_lcj) ) DEALLOCATE(tf_lay%i_lcj)
      IF( ASSOCIATED(td_lay%i_lcj) )THEN
         il_shape(:)=SHAPE(td_lay%i_lcj(:,:))
         ALLOCATE(il_tmp(il_shape(jp_I),il_shape(jp_J)))
         il_tmp(:,:)=td_lay%i_lcj(:,:)

         ALLOCATE( tf_lay%i_lcj(il_shape(jp_I),il_shape(jp_J)) )
         tf_lay%i_lcj(:,:)=il_tmp(:,:)

         DEALLOCATE(il_tmp)
      ENDIF

   END FUNCTION layout__copy
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp__create_layout(td_mpp, td_lay)
   !-------------------------------------------------------------------
   !> @brief
   !>    This subroutine create mpp structure using domain layout
   !>
   !> @detail
   !>
   !> @author J.Paul
   !> @date October, 2015 - Initial version
   !> @date August, 2017 
   !> - handle use of domain decomposition for monoproc file
   !>
   !> @param[inout] td_mpp mpp strcuture
   !> @param[in] td_lay domain layout structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP), INTENT(INOUT) :: td_mpp
      TYPE(TLAY), INTENT(IN   ) :: td_lay

      ! local variable
      CHARACTER(LEN=lc)                        :: cl_file
      TYPE(TATT)                               :: tl_att

      TYPE(TFILE), DIMENSION(:), ALLOCATABLE   :: tl_proc

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      ! intialise
      td_mpp%i_nproc=0

      CALL logger_debug( "MPP CREATE LAYOUT: create domain decomposition with "//&
      &               TRIM(fct_str(td_lay%i_niproc))//" x "//&
      &               TRIM(fct_str(td_lay%i_njproc))//" = "//&
      &               TRIM(fct_str(td_lay%i_nsea))//" processors")

      IF( lm_layout )THEN
         WRITE(im_iumout,*) ' choix optimum'
         WRITE(im_iumout,*) ' ============='
         WRITE(im_iumout,*)
         ! print info 
         WRITE(im_iumout,*) ' '
         WRITE(im_iumout,*) " jpni=",td_lay%i_niproc ," jpnj=",td_lay%i_njproc
         WRITE(im_iumout,*) " iresti=",td_mpp%i_preci," irestj=",td_mpp%i_precj


         WRITE(im_iumout,*) ' nombre de processeurs       ',td_lay%i_niproc*td_lay%i_njproc
         WRITE(im_iumout,*) ' nombre de processeurs mer   ',td_lay%i_nsea
         WRITE(im_iumout,*) ' nombre de processeurs terre ',td_lay%i_nland
         WRITE(im_iumout,*) ' moyenne de recouvrement     ',td_lay%i_mean
         WRITE(im_iumout,*) ' minimum de recouvrement     ',td_lay%i_min
         WRITE(im_iumout,*) ' maximum de recouvrement     ',td_lay%i_max
      ENDIF

      td_mpp%i_niproc=td_lay%i_niproc
      td_mpp%i_njproc=td_lay%i_njproc
      !td_mpp%i_nproc =td_lay%i_nsea

      IF( td_mpp%i_niproc*td_mpp%i_njproc == td_lay%i_nsea )THEN
         IF( td_lay%i_nsea == 1 )THEN
            td_mpp%c_dom='full'
         ELSE
            td_mpp%c_dom='nooverlap'
         ENDIF
      ELSE
            td_mpp%c_dom='noextra'
      ENDIF
      
      ALLOCATE(tl_proc(td_lay%i_nsea))
      jk=1
      DO jj=1,td_lay%i_njproc
         DO ji=1,td_lay%i_niproc

            IF( td_lay%i_msk(ji,jj) >= 1 )THEN

               ! get processor file name
               IF( td_mpp%l_usempp )THEN
                  cl_file=file_rename(td_mpp%c_name,jk)
               ELSE
                  cl_file=TRIM(td_mpp%c_name)
               ENDIF
               ! initialise file structure
               tl_proc(jk)=file_init(cl_file,td_mpp%c_type)

               ! procesor id
               tl_proc(jk)%i_pid=jk-1

               tl_att=att_init("DOMAIN_number",tl_proc(jk)%i_pid)
               CALL file_add_att(tl_proc(jk), tl_att)

               ! processor indices
               tl_proc(jk)%i_iind=ji
               tl_proc(jk)%i_jind=jj

               ! fill processor dimension and first indices
               tl_proc(jk)%i_impp = td_lay%i_impp(ji,jj)
               tl_proc(jk)%i_jmpp = td_lay%i_jmpp(ji,jj)

               tl_proc(jk)%i_lci  = td_lay%i_lci(ji,jj)
               tl_proc(jk)%i_lcj  = td_lay%i_lcj(ji,jj)

               ! compute first and last indoor indices
 
               ! west boundary
               IF( ji == 1 )THEN
                  tl_proc(jk)%i_ldi = 1 
                  tl_proc(jk)%l_ctr = .TRUE.
               ELSE
                  tl_proc(jk)%i_ldi = 1 + td_mpp%i_preci
               ENDIF

               ! south boundary
               IF( jj == 1 )THEN
                  tl_proc(jk)%i_ldj = 1 
                  tl_proc(jk)%l_ctr = .TRUE.
               ELSE
                  tl_proc(jk)%i_ldj = 1 + td_mpp%i_precj
               ENDIF

               ! east boundary
               IF( ji == td_mpp%i_niproc )THEN
                  tl_proc(jk)%i_lei = td_lay%i_lci(ji,jj)
                  tl_proc(jk)%l_ctr = .TRUE.
               ELSE
                  tl_proc(jk)%i_lei = td_lay%i_lci(ji,jj) - td_mpp%i_preci
               ENDIF

               ! north boundary
               IF( jj == td_mpp%i_njproc )THEN
                  tl_proc(jk)%i_lej = td_lay%i_lcj(ji,jj)
                  tl_proc(jk)%l_ctr = .TRUE.
               ELSE
                  tl_proc(jk)%i_lej = td_lay%i_lcj(ji,jj) - td_mpp%i_precj
               ENDIF

               ! clean
               CALL att_clean(tl_att)

               ! update proc number
               jk=jk+1 

            ENDIF
         ENDDO
      ENDDO
!
      CALL mpp__add_proc(td_mpp, tl_proc(:))
      DEALLOCATE(tl_proc)

   END SUBROUTINE mpp__create_layout
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp__optimiz(td_mpp, id_mask, id_nproc)
   !-------------------------------------------------------------------
   !> @brief 
   !>  This subroutine optimize the number of sub domain to be used, given mask.
   !> @details
   !>  Actually it get the domain decomposition with the most land 
   !>  processors removed.
   !>  If no land processor could be removed, it get the decomposition with the
   !>  most sea processors.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !> @date October, 2015
   !> - improve way to compute domain layout 
   !> @date February, 2016
   !> - new criteria for domain layout in case no land proc
   !>
   !> @param[inout] td_mpp mpp strcuture
   !> @param[in] id_mask   sub domain mask (sea=1, land=0) 
   !> @pram[in] id_nproc maximum number of processor to be used
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      TYPE(TMPP),                  INTENT(INOUT) :: td_mpp
      INTEGER(i4), DIMENSION(:,:), INTENT(IN)    :: id_mask
      INTEGER(i4)                , INTENT(IN)    :: id_nproc

      ! local variable
      TYPE(TLAY) :: tl_lay
      TYPE(TLAY) :: tl_sav

      REAL(dp)   :: dl_min
      REAL(dp)   :: dl_max
      REAL(dp)   :: dl_ratio
      REAL(dp)   :: dl_sav

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      !----------------------------------------------------------------

      CALL logger_trace("MPP OPTIMIZ: look for best domain decomposition")
      dl_sav=0
      ! 
      DO ji=1,id_nproc
         DO jj=1,id_nproc

            ! compute domain layout
            tl_lay=layout__init( td_mpp, id_mask, ji,jj )
            IF( tl_lay%i_nsea <= id_nproc )THEN

               IF( ASSOCIATED(tl_sav%i_lci) )THEN
                  IF( tl_sav%i_nland /= 0 )THEN
                     ! look for layout with most land proc
                     IF( tl_lay%i_nland > tl_sav%i_nland    .OR. &
                     &   ( tl_lay%i_nland == tl_sav%i_nland .AND. &
                     &     tl_lay%i_min   >  tl_sav%i_min   ) )THEN
                        ! save optimiz layout
                        CALL logger_info("MPP OPTIMIZ:save this decomposition "//&
                        &   TRIM(fct_str(ji))//"x"//TRIM(fct_str(jj))//"="//&
                        &   TRIM(fct_str(tl_lay%i_nsea)) )

                        tl_sav=layout__copy(tl_lay)
                     ENDIF
                  ELSE ! tl_sav%i_nland == 0
                     ! look for layout with most sea proc 
                     ! and "square" cell 
                     dl_min=MIN(tl_lay%i_lci(1,1),tl_lay%i_lcj(1,1))
                     dl_max=MAX(tl_lay%i_lci(1,1),tl_lay%i_lcj(1,1))
                     dl_ratio=dl_min/dl_max
                     IF( tl_lay%i_nsea > tl_sav%i_nsea    .OR. &
                     &   ( tl_lay%i_nsea == tl_sav%i_nsea .AND. &
                     &     dl_ratio   >  dl_sav ) )THEN
                        ! save optimiz layout
                        CALL logger_info("MPP OPTIMIZ:save this decomposition "//&
                        &   TRIM(fct_str(ji))//"x"//TRIM(fct_str(jj))//"="//&
                        &   TRIM(fct_str(tl_lay%i_nsea)) )

                        tl_sav=layout__copy(tl_lay)
                        dl_sav=dl_ratio
                     ENDIF
                  ENDIF
               ELSE
                  ! init tl_sav
                  tl_sav=layout__copy(tl_lay)

                  dl_min=MIN(tl_sav%i_lci(1,1),tl_sav%i_lcj(1,1))
                  dl_max=MAX(tl_sav%i_lci(1,1),tl_sav%i_lcj(1,1))
                  dl_sav=dl_min/dl_max
               ENDIF

            ENDIF

            ! clean
            CALL layout__clean( tl_lay )

         ENDDO
      ENDDO

      ! create mpp domain layout
      CALL mpp__create_layout(td_mpp, tl_sav)

      ! clean
      CALL layout__clean( tl_sav )

   END SUBROUTINE mpp__optimiz
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp__clean_unit(td_mpp)
   !-------------------------------------------------------------------
   !> @brief 
   !>  This subroutine clean mpp strcuture.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !> @date January, 2019
   !> - nullify file structure inside mpp structure
   !>
   !> @param[inout] td_mpp mpp strcuture
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP),  INTENT(INOUT) :: td_mpp

      ! local variable
      TYPE(TMPP) :: tl_mpp ! empty mpp structure

      ! loop indices
      !----------------------------------------------------------------

      CALL logger_info( &
      &  "MPP CLEAN: reset mpp "//TRIM(td_mpp%c_name) )

      ! del dimension
      IF( td_mpp%i_ndim /= 0 )THEN
         CALL dim_clean( td_mpp%t_dim(:) )
      ENDIF

      IF( ASSOCIATED(td_mpp%t_proc) )THEN
         ! clean array of file processor
         CALL file_clean( td_mpp%t_proc(:) )
         DEALLOCATE(td_mpp%t_proc)
         NULLIFY(td_mpp%t_proc)
      ENDIF

      ! replace by empty structure
      td_mpp=mpp_copy(tl_mpp)

   END SUBROUTINE mpp__clean_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   !-------------------------------------------------------------------
   !> @brief 
   !>  This subroutine clean mpp strcuture.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_mpp mpp strcuture
   !-------------------------------------------------------------------
   SUBROUTINE mpp__clean_arr(td_mpp)

      IMPLICIT NONE
      ! Argument
      TYPE(TMPP),  DIMENSION(:), INTENT(INOUT) :: td_mpp

      ! local variable
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      DO ji=SIZE(td_mpp(:)),1,-1
         CALL mpp_clean(td_mpp(ji))
      ENDDO

   END SUBROUTINE mpp__clean_arr
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp__get_use_unit(td_mpp, id_imin, id_imax, id_jmin, id_jmax)
   !-------------------------------------------------------------------
   !> @brief 
   !>  This subroutine get sub domains which cover "zoom domain".
   !>                      proc use in "zoom domain" 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_mpp mpp strcuture
   !> @param[in] id_imin   i-direction lower indice
   !> @param[in] id_imax   i-direction upper indice
   !> @param[in] id_jmin   j-direction lower indice
   !> @param[in] id_jmax   j-direction upper indice
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP) ,  INTENT(INOUT) :: td_mpp
      INTEGER(i4),  INTENT(IN), OPTIONAL :: id_imin
      INTEGER(i4),  INTENT(IN), OPTIONAL :: id_imax
      INTEGER(i4),  INTENT(IN), OPTIONAL :: id_jmin
      INTEGER(i4),  INTENT(IN), OPTIONAL :: id_jmax

      ! local variable
      LOGICAL     :: ll_iuse
      LOGICAL     :: ll_juse

      INTEGER(i4) :: il_imin
      INTEGER(i4) :: il_imax
      INTEGER(i4) :: il_jmin
      INTEGER(i4) :: il_jmax

      ! loop indices
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      IF( ASSOCIATED(td_mpp%t_proc) )THEN
   
         il_imin=1
         il_imax=td_mpp%t_dim(1)%i_len
         IF( PRESENT(id_imin) ) il_imin=id_imin
         IF( PRESENT(id_imax) ) il_imax=id_imax
         il_jmin=1
         il_jmax=td_mpp%t_dim(2)%i_len
         IF( PRESENT(id_jmin) ) il_jmin=id_jmin
         IF( PRESENT(id_jmax) ) il_jmax=id_jmax

         ! check domain
         IF( il_imin < 1 .OR. il_imin > td_mpp%t_dim(1)%i_len .OR. &
         &   il_imax < 1 .OR. il_imax > td_mpp%t_dim(1)%i_len .OR. &
         &   il_jmin < 1 .OR. il_jmin > td_mpp%t_dim(2)%i_len .OR. &
         &   il_jmax < 1 .OR. il_jmax > td_mpp%t_dim(2)%i_len )THEN
            CALL logger_debug("MPP GET USE: mpp gloabl size "//&
            &        TRIM(fct_str(td_mpp%t_dim(1)%i_len))//","//&
            &        TRIM(fct_str(td_mpp%t_dim(2)%i_len)))
            CALL logger_debug("MPP GET USE: i-indices "//&
            &        TRIM(fct_str(il_imin))//","//TRIM(fct_str(il_imax)))
            CALL logger_debug("MPP GET USE: j-indices "//&
            &        TRIM(fct_str(il_jmin))//","//TRIM(fct_str(il_jmax)))
            CALL logger_error("MPP GET USE: invalid indices ")
         ELSE
            td_mpp%t_proc(:)%l_use=.FALSE.
            DO jk=1,td_mpp%i_nproc

               ! check i-direction
               ll_iuse=.FALSE.
               IF( il_imin < il_imax )THEN

                  ! not overlap east west boundary
                  IF( td_mpp%t_proc(jk)%i_impp + td_mpp%t_proc(jk)%i_lci > &
                  &   il_imin .AND.                                  &
                  &   td_mpp%t_proc(jk)%i_impp < il_imax )THEN
                      ll_iuse=.TRUE.
                  ENDIF

               ELSEIF( il_imin == il_imax )THEN

                  ! east west cyclic
                  ll_iuse=.TRUE.

               ELSE ! il_imin > id_imax

                  ! overlap east west boundary
                  IF( ( td_mpp%t_proc(jk)%i_impp + td_mpp%t_proc(jk)%i_lci >  &
                  &     il_imin )                                             &
                  &   .OR.                                                    &
                  &   ( td_mpp%t_proc(jk)%i_impp < il_imax) )THEN
                     ll_iuse=.TRUE.
                  ENDIF

               ENDIF

               ! check j-direction
               ll_juse=.FALSE.
               IF( il_jmin < il_jmax )THEN

                  ! not overlap north fold
                  IF( td_mpp%t_proc(jk)%i_jmpp + td_mpp%t_proc(jk)%i_lcj > &
                  &   il_jmin .AND.                                  &
                  &   td_mpp%t_proc(jk)%i_jmpp < il_jmax )THEN
                     ll_juse=.TRUE.
                  ENDIF

               ELSE ! id_jmin >= id_jmax

                  IF( td_mpp%t_proc(jk)%i_jmpp + td_mpp%t_proc(jk)%i_lcj > &
                  &  il_jmin )THEN
                     ll_juse=.TRUE.
                  ENDIF

               ENDIF

               IF( ll_iuse .AND. ll_juse ) td_mpp%t_proc(jk)%l_use=.TRUE.

            ENDDO
         ENDIF

      ELSE
         CALL logger_error("MPP GET USE: mpp decomposition not define.")
      ENDIF

   END SUBROUTINE mpp__get_use_unit
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp_get_contour(td_mpp)
   !-------------------------------------------------------------------
   !> @brief 
   !>  This subroutine get sub domains which form global domain border.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_mpp mpp strcuture
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      TYPE(TMPP),  INTENT(INOUT) :: td_mpp

      ! loop indices
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      IF( ASSOCIATED(td_mpp%t_proc) )THEN

         td_mpp%t_proc(:)%l_use = .FALSE.
         DO jk=1,td_mpp%i_nproc
            IF( td_mpp%t_proc(jk)%i_ldi == 1 .OR. &
            &   td_mpp%t_proc(jk)%i_ldj == 1 .OR. &
            &   td_mpp%t_proc(jk)%i_lei == td_mpp%t_proc(jk)%i_lci .OR. &
            &   td_mpp%t_proc(jk)%i_lej == td_mpp%t_proc(jk)%i_lcj )THEN

               td_mpp%t_proc(jk)%l_use = .TRUE.
 
            ENDIF
         ENDDO
   
      ELSE
         CALL logger_error("MPP GET CONTOUR: domain decomposition not define.")
      ENDIF

   END SUBROUTINE mpp_get_contour
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION mpp_get_proc_index(td_mpp, id_procid) &
         & RESULT(if_idx)
   !-------------------------------------------------------------------
   !> @brief
   !> This function return processor indices, without overlap boundary,
   !> given processor id. 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[in] td_mpp    mpp strcuture
   !> @param[in] id_procid processor id
   !> @return array of index (/ i1, i2, j1, j2 /)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP) , INTENT(IN) :: td_mpp
      INTEGER(i4), INTENT(IN) :: id_procid

      ! function
      INTEGER(i4), DIMENSION(4) :: if_idx

      ! local variable
      INTEGER(i4) :: il_i1, il_i2
      INTEGER(i4) :: il_j1, il_j2
      !----------------------------------------------------------------

      IF( ASSOCIATED(td_mpp%t_proc) )THEN

         IF( TRIM(td_mpp%c_dom) == '' )THEN
            CALL logger_fatal("MPP GET PROC INDEX: decomposition type unknown. "//&
            &                 "you should ahve run mpp_get_dom before.")
         ENDIF

         SELECT CASE(TRIM(td_mpp%c_dom))
            CASE('full')
               il_i1 = 1 
               il_j1 = 1 

               il_i2 = td_mpp%t_dim(1)%i_len
               il_j2 = td_mpp%t_dim(2)%i_len
            CASE('noextra')
               il_i1 = td_mpp%t_proc(id_procid)%i_impp
               il_j1 = td_mpp%t_proc(id_procid)%i_jmpp

               il_i2 = il_i1 + td_mpp%t_proc(id_procid)%i_lci - 1 
               il_j2 = il_j1 + td_mpp%t_proc(id_procid)%i_lcj - 1 
            CASE('nooverlap')
               il_i1 = td_mpp%t_proc(id_procid)%i_impp + &
               &        td_mpp%t_proc(id_procid)%i_ldi - 1
               il_j1 = td_mpp%t_proc(id_procid)%i_jmpp + &
               &        td_mpp%t_proc(id_procid)%i_ldj - 1

               il_i2 = td_mpp%t_proc(id_procid)%i_impp + &
               &        td_mpp%t_proc(id_procid)%i_lei - 1
               il_j2 = td_mpp%t_proc(id_procid)%i_jmpp + &
               &        td_mpp%t_proc(id_procid)%i_lej - 1
            CASE DEFAULT
               CALL logger_error("MPP GET PROC INDEX: invalid "//&
                  &              "decomposition type.")
         END SELECT

         if_idx(:)=(/il_i1, il_i2, il_j1, il_j2/)

      ELSE
         CALL logger_error("MPP GET PROC INDEX: domain decomposition not define.")
      ENDIF

   END FUNCTION mpp_get_proc_index
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION mpp_get_proc_size(td_mpp, id_procid) &
         & RESULT(if_size)
   !-------------------------------------------------------------------
   !> @brief
   !> This function return processor domain size, depending of domain 
   !> decompisition type, given sub domain id. 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[in] td_mpp    mpp strcuture
   !> @param[in] id_procid sub domain id
   !> @return array of index (/ isize, jsize /)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP),  INTENT(IN) :: td_mpp
      INTEGER(i4), INTENT(IN) :: id_procid

      ! function
      INTEGER(i4), DIMENSION(2) :: if_size

      ! local variable
      INTEGER(i4) :: il_isize
      INTEGER(i4) :: il_jsize
      !----------------------------------------------------------------

      IF( ASSOCIATED(td_mpp%t_proc) )THEN

         IF( TRIM(td_mpp%c_dom) == '' )THEN
            CALL logger_fatal("MPP GET PROC SIZE: decomposition type unknown. "//&
            &                 "you should ahve run mpp_get_dom before.")
         ENDIF

         SELECT CASE(TRIM(td_mpp%c_dom))
            CASE('full')
               
               il_isize = td_mpp%t_dim(1)%i_len
               il_jsize = td_mpp%t_dim(2)%i_len

            CASE('noextra')

                il_isize = td_mpp%t_proc(id_procid)%i_lci
                il_jsize = td_mpp%t_proc(id_procid)%i_lcj

            CASE('nooverlap')
               il_isize = td_mpp%t_proc(id_procid)%i_lei - &
               &          td_mpp%t_proc(id_procid)%i_ldi + 1
               il_jsize = td_mpp%t_proc(id_procid)%i_lej - &
               &          td_mpp%t_proc(id_procid)%i_ldj + 1
            CASE DEFAULT
               CALL logger_error("MPP GET PROC SIZE: invalid decomposition type : "//&
               &  TRIM(td_mpp%c_dom) )
         END SELECT

         if_size(:)=(/il_isize, il_jsize/)

      ELSE
         CALL logger_error("MPP GET PROC SIZE: domain decomposition not define.")
      ENDIF

   END FUNCTION mpp_get_proc_size
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp_get_dom(td_mpp)
   !-------------------------------------------------------------------
   !> @brief 
   !>  This subroutine determine domain decomposition type.
   !>  (full, overlap, noverlap)
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial version
   !>
   !> @param[inout] td_mpp mpp strcuture
   !-------------------------------------------------------------------
      
      IMPLICIT NONE

      ! Argument
      TYPE(TMPP),  INTENT(INOUT) :: td_mpp

      ! local variable
      INTEGER(i4) :: il_isize
      INTEGER(i4) :: il_jsize
      !----------------------------------------------------------------

      IF( ASSOCIATED(td_mpp%t_proc) )THEN

         IF( td_mpp%i_niproc == 0 .AND. td_mpp%i_njproc == 0 )THEN
            CALL logger_info("MPP GET DOM: use indoor indices to get domain "//&
            &             "decomposition type.")
            IF((td_mpp%t_proc(1)%t_dim(1)%i_len ==                         &
            &   td_mpp%t_proc(1)%i_lei - td_mpp%t_proc(1)%i_ldi + 1) .AND. &
            &  (td_mpp%t_proc(1)%t_dim(2)%i_len ==                         &
            &   td_mpp%t_proc(1)%i_lej - td_mpp%t_proc(1)%i_ldj + 1) )THEN

               td_mpp%c_dom='nooverlap'

            ELSEIF((td_mpp%t_proc(1)%t_dim(1)%i_len ==                     &
            &       td_mpp%t_proc(1)%i_lci                     )     .AND. &
            &      (td_mpp%t_proc(1)%t_dim(2)%i_len ==                     &
            &       td_mpp%t_proc(1)%i_lcj                     )     )THEN

               td_mpp%c_dom='noextra'

            ELSEIF((td_mpp%t_proc(1)%t_dim(1)%i_len ==                     &
            &       td_mpp%t_dim(1)%i_len             )              .AND. &
            &      (td_mpp%t_proc(1)%t_dim(2)%i_len ==                     &
            &       td_mpp%t_dim(2)%i_len )                          )THEN

               td_mpp%c_dom='full'

            ELSE

               CALL logger_error("MPP GET DOM: should have been an impossible case")

               il_isize=td_mpp%t_proc(1)%t_dim(1)%i_len
               il_jsize=td_mpp%t_proc(1)%t_dim(2)%i_len
               CALL logger_debug("MPP GET DOM: proc size "//&
               &  TRIM(fct_str(il_isize))//" x "//TRIM(fct_str(il_jsize)) )

               il_isize=td_mpp%t_proc(1)%i_lei - td_mpp%t_proc(1)%i_ldi + 1
               il_jsize=td_mpp%t_proc(1)%i_lej - td_mpp%t_proc(1)%i_ldj + 1
               CALL logger_debug("MPP GET DOM: no overlap size "//&
               &  TRIM(fct_str(il_isize))//" x "//TRIM(fct_str(il_jsize)) )

               il_isize=td_mpp%t_proc(1)%i_lci
               il_jsize=td_mpp%t_proc(1)%i_lcj
               CALL logger_debug("MPP GET DOM: overlap size "//&
               &  TRIM(fct_str(il_isize))//" x "//TRIM(fct_str(il_jsize)) )

               il_isize=td_mpp%t_dim(1)%i_len
               il_jsize=td_mpp%t_dim(2)%i_len
               CALL logger_debug("MPP GET DOM: full size "//&
               &  TRIM(fct_str(il_isize))//" x "//TRIM(fct_str(il_jsize)) )

            ENDIF

         ELSE

            CALL logger_info("MPP GET DOM: use number of processors following "//&
            &             "I and J to get domain decomposition type.")
            IF( td_mpp%i_niproc*td_mpp%i_njproc==td_mpp%i_nproc )THEN
               IF( td_mpp%i_nproc == 1 )THEN
                  td_mpp%c_dom='full'
               ENDIF
               td_mpp%c_dom='nooverlap'
            ELSE
               td_mpp%c_dom='noextra'
            ENDIF

         ENDIF

      ELSE
         CALL logger_error("MPP GET DOM: domain decomposition not define.")
      ENDIF

   END SUBROUTINE mpp_get_dom
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION mpp__check_var_dim(td_mpp, td_var) &
         & RESULT(lf_check)
   !-------------------------------------------------------------------
   !> @brief This function check if variable  and mpp structure use same
   !> dimension.
   !>
   !> @details
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September 2015
   !> - do not check used dimension here
   !>
   !> @param[in] td_mpp mpp structure
   !> @param[in] td_var variable structure
   !> @return dimension of variable and mpp structure agree (or not)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP), INTENT(IN) :: td_mpp
      TYPE(TVAR), INTENT(IN) :: td_var

      ! function
      LOGICAL                :: lf_check

      ! local variable
      CHARACTER(LEN=lc) :: cl_dim
      LOGICAL :: ll_error
      LOGICAL :: ll_warn

      INTEGER(i4)       :: il_ind

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      lf_check=.TRUE.

      ! check used dimension 
      ll_error=.FALSE.
      ll_warn=.FALSE.
      DO ji=1,ip_maxdim
         il_ind=dim_get_index( td_mpp%t_dim(:), &
            &                  TRIM(td_var%t_dim(ji)%c_name), &
            &                  TRIM(td_var%t_dim(ji)%c_sname))
         IF( il_ind /= 0 )THEN
            IF( td_var%t_dim(ji)%l_use  .AND. &
               &td_mpp%t_dim(il_ind)%l_use .AND. &
               &td_var%t_dim(ji)%i_len /= td_mpp%t_dim(il_ind)%i_len )THEN
               IF( INDEX( TRIM(td_var%c_axis), &
                  &       TRIM(fct_upper(td_var%t_dim(ji)%c_name))) == 0 )THEN
                  ll_warn=.TRUE.
               ELSE
                  ll_error=.TRUE.
               ENDIF
            ENDIF
         ENDIF
      ENDDO

      IF( ll_error )THEN

         cl_dim='(/'
         DO ji = 1, td_mpp%i_ndim
            IF( td_mpp%t_dim(ji)%l_use )THEN
               cl_dim=TRIM(cl_dim)//&
                  &   TRIM(fct_upper(td_mpp%t_dim(ji)%c_sname))//':'//&
                  &   TRIM(fct_str(td_mpp%t_dim(ji)%i_len))//','
            ENDIF
         ENDDO
         cl_dim=TRIM(cl_dim)//'/)'
         CALL logger_debug( " mpp dimension: "//TRIM(cl_dim) )

         cl_dim='(/'
         DO ji = 1, td_var%i_ndim
            IF( td_var%t_dim(ji)%l_use )THEN
               cl_dim=TRIM(cl_dim)//&
                  &   TRIM(fct_upper(td_var%t_dim(ji)%c_sname))//':'//&
                  &   TRIM(fct_str(td_var%t_dim(ji)%i_len))//','
            ENDIF
         ENDDO
         cl_dim=TRIM(cl_dim)//'/)'
         CALL logger_debug( " variable dimension: "//TRIM(cl_dim) )

         lf_check=.FALSE.

         CALL logger_error( &
            &     " MPP CHECK VAR DIM: variable and file dimension differ"//&
            &     " for variable "//TRIM(td_var%c_name)//&
            &     " and file "//TRIM(td_mpp%c_name))

      ELSEIF( ll_warn )THEN
         CALL logger_warn( &
            &     " MPP CHECK VAR DIM: variable and file dimension differ"//&
            &     " for variable "//TRIM(td_var%c_name)//&
            &     " and file "//TRIM(td_mpp%c_name)//". you should use"//&
            &     " var_check_dim to remove useless dimension.")
      ELSE

         IF( td_var%i_ndim >  td_mpp%i_ndim )THEN
            CALL logger_info("MPP CHECK VAR DIM: variable "//&
               &     TRIM(td_var%c_name)//" use more dimension than file "//&
               &     TRIM(td_mpp%c_name)//" do until now.")
         ENDIF

      ENDIF

   END FUNCTION mpp__check_var_dim
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION mpp_get_index(td_mpp, cd_name) &
         & RESULT(if_idx)
   !-------------------------------------------------------------------
   !> @brief This function return the mpp id, in a array of mpp
   !> structure,  given mpp base name. 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_file   array of file structure
   !> @param[in] cd_name   file name
   !> @return file id in array of file structure (0 if not found)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP)      , DIMENSION(:), INTENT(IN) :: td_mpp
      CHARACTER(LEN=*),               INTENT(IN) :: cd_name

      ! function
      INTEGER(i4)                                :: if_idx

      ! local variable
      CHARACTER(LEN=lc) :: cl_name
      INTEGER(i4)       :: il_size

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      if_idx=0
      il_size=SIZE(td_mpp(:))

      cl_name=TRIM( file_rename(cd_name) )

      ! check if mpp is in array of mpp structure
      DO ji=1,il_size
         ! look for file name
         IF( TRIM(fct_lower(td_mpp(ji)%c_name)) == TRIM(fct_lower(cd_name)) )THEN
 
            if_idx=ji
            EXIT

         ENDIF
      ENDDO

   END FUNCTION mpp_get_index
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION mpp_recombine_var(td_mpp, cd_name) &
         & RESULT(tf_var)
   !-------------------------------------------------------------------
   !> @brief This function recombine variable splitted in mpp structure. 
   !>
   !> @author J.Paul
   !> @date October, 2014 - Initial Version
   !>
   !> @param[in] td_mpp   mpp file structure
   !> @param[in] cd_name  variable name
   !> @return variable strucutre
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP)      , INTENT(IN) :: td_mpp
      CHARACTER(LEN=*), INTENT(IN) :: cd_name
      ! function
      TYPE(TVAR)                   :: tf_var

      ! local variable
      INTEGER(i4)                       :: il_varid
      INTEGER(i4)                       :: il_status
      INTEGER(i4)                       :: il_i1p
      INTEGER(i4)                       :: il_i2p
      INTEGER(i4)                       :: il_j1p
      INTEGER(i4)                       :: il_j2p
      INTEGER(i4), DIMENSION(4)         :: il_ind

      INTEGER(i4), DIMENSION(ip_maxdim) :: il_strt
      INTEGER(i4), DIMENSION(ip_maxdim) :: il_cnt

      TYPE(TVAR)                        :: tl_tmp

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      il_varid=var_get_index( td_mpp%t_proc(1)%t_var(:), cd_name)
      IF( il_varid /= 0 )THEN
      
         tf_var=var_copy(td_mpp%t_proc(1)%t_var(il_varid))
         ! Allocate space to hold variable value in structure 
         IF( ASSOCIATED(tf_var%d_value) )THEN
            DEALLOCATE(tf_var%d_value)   
         ENDIF
         ! 
         DO ji=1,ip_maxdim
            IF( tf_var%t_dim(ji)%l_use )THEN
               tf_var%t_dim(ji)%i_len=td_mpp%t_dim(ji)%i_len
            ENDIF
         ENDDO

         ALLOCATE(tf_var%d_value( tf_var%t_dim(1)%i_len, &
            &                     tf_var%t_dim(2)%i_len, &
            &                     tf_var%t_dim(3)%i_len, &
            &                     tf_var%t_dim(4)%i_len),&
            &     stat=il_status)
         IF(il_status /= 0 )THEN

           CALL logger_error( &
              &  " MPP RECOMBINE VAR: not enough space to put variable "//&
              &  TRIM(tf_var%c_name)//" in variable structure")

         ENDIF

         ! FillValue by default
         tf_var%d_value(:,:,:,:)=tf_var%d_fill

         ! read processor 
         DO jk=1,td_mpp%i_nproc
            IF( td_mpp%t_proc(jk)%l_use )THEN
               ! get processor indices
               il_ind(:)=mpp_get_proc_index( td_mpp, jk )
               il_i1p = il_ind(1)
               il_i2p = il_ind(2)
               il_j1p = il_ind(3)
               il_j2p = il_ind(4)
 
               il_strt(:)=(/ 1,1,1,1 /)

               il_cnt(:)=(/ il_i2p-il_i1p+1,         &
                  &         il_j2p-il_j1p+1,         &
                  &         tf_var%t_dim(3)%i_len, &
                  &         tf_var%t_dim(4)%i_len /)

               tl_tmp=iom_read_var( td_mpp%t_proc(jk), tf_var%c_name,&
                  &                 il_strt(:), il_cnt(:) )
               
               ! replace value in output variable structure
               tf_var%d_value( il_i1p : il_i2p,  &
                  &            il_j1p : il_j2p,  &
                  &            :,:) = tl_tmp%d_value(:,:,:,:)

               ! clean
               CALL var_clean(tl_tmp)

            ENDIF
         ENDDO

      ELSE

         CALL logger_error( &
            &  " MPP RECOMBINE VAR: there is no variable with "//&
            &  "name or standard name"//TRIM(cd_name)//&
            &  " in mpp file "//TRIM(td_mpp%c_name))
      ENDIF

   END FUNCTION mpp_recombine_var
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp__read_halo(td_file, td_dimglo) 
   !-------------------------------------------------------------------
   !> @brief This subroutine read subdomain indices defined with halo
   !> (NEMO netcdf way)
   !>
   !> @author J.Paul
   !> @date January, 2016 - Initial Version
   !>
   !> @param[inout] td_file   mpp structure
   !-------------------------------------------------------------------
   
      IMPLICIT NONE

      ! Argument      
      TYPE(TFILE)              , INTENT(INOUT) :: td_file
      TYPE(TDIM) , DIMENSION(:), INTENT(IN   ) :: td_dimglo

      ! local variable
      INTEGER(i4)       :: il_attid
      INTEGER(i4)       :: il_ifirst
      INTEGER(i4)       :: il_jfirst
      INTEGER(i4)       :: il_ilast
      INTEGER(i4)       :: il_jlast
      INTEGER(i4)       :: il_ihalostart
      INTEGER(i4)       :: il_jhalostart
      INTEGER(i4)       :: il_ihaloend
      INTEGER(i4)       :: il_jhaloend

      CHARACTER(LEN=lc) :: cl_dom
      !----------------------------------------------------------------

      ! DOMAIN_position_first
      il_attid = 0
      IF( ASSOCIATED(td_file%t_att) )THEN
         il_attid=att_get_id( td_file%t_att, "DOMAIN_position_first" )
      ENDIF
      IF( il_attid /= 0 )THEN
         il_ifirst = INT(td_file%t_att(il_attid)%d_value(1))
         il_jfirst = INT(td_file%t_att(il_attid)%d_value(2))
      ELSE
         il_ifirst = 1
         il_jfirst = 1
      ENDIF

      ! DOMAIN_position_last
      il_attid = 0
      IF( ASSOCIATED(td_file%t_att) )THEN
         il_attid=att_get_id( td_file%t_att, "DOMAIN_position_last" )
      ENDIF
      IF( il_attid /= 0 )THEN
         il_ilast = INT(td_file%t_att(il_attid)%d_value(1))
         il_jlast = INT(td_file%t_att(il_attid)%d_value(2))
      ELSE
         il_ilast = td_file%t_dim(1)%i_len
         il_jlast = td_file%t_dim(2)%i_len
      ENDIF

      ! DOMAIN_halo_size_start
      il_attid = 0
      IF( ASSOCIATED(td_file%t_att) )THEN
         il_attid=att_get_id( td_file%t_att, "DOMAIN_halo_size_start" )
      ENDIF
      IF( il_attid /= 0 )THEN
         il_ihalostart = INT(td_file%t_att(il_attid)%d_value(1))
         il_jhalostart = INT(td_file%t_att(il_attid)%d_value(2))
      ELSE
         il_ihalostart = 0
         il_jhalostart = 0
      ENDIF

      ! DOMAIN_halo_size_end
      il_attid = 0
      IF( ASSOCIATED(td_file%t_att) )THEN
         il_attid=att_get_id( td_file%t_att, "DOMAIN_halo_size_end" )
      ENDIF
      IF( il_attid /= 0 )THEN
         il_ihaloend = INT(td_file%t_att(il_attid)%d_value(1))
         il_jhaloend = INT(td_file%t_att(il_attid)%d_value(2))
      ELSE
         il_ihaloend = 0
         il_jhaloend = 0
      ENDIF

      IF( (td_dimglo(jp_I)%i_len == td_file%t_dim(jp_I)%i_len) .AND. &
        & (td_dimglo(jp_J)%i_len == td_file%t_dim(jp_J)%i_len) )THEN
         cl_dom='full'
      ELSEIF( il_ihalostart == 0 .AND. il_jhalostart == 0 .AND. &
           &  il_ihaloend == 0 .AND. il_jhaloend == 0 )THEN
         cl_dom='nooverlap'
      ELSE
         cl_dom='noextra'
      ENDIF

      SELECT CASE(TRIM(cl_dom))
         CASE('full')
            td_file%i_impp = il_ifirst 
            td_file%i_jmpp = il_jfirst
            td_file%i_lci  = td_file%t_dim(jp_I)%i_len 
            td_file%i_lcj  = td_file%t_dim(jp_J)%i_len
            td_file%i_ldi  = il_ihalostart + 1
            td_file%i_ldj  = il_jhalostart + 1
            td_file%i_lei  = td_file%t_dim(jp_I)%i_len - il_ihaloend
            td_file%i_lej  = td_file%t_dim(jp_J)%i_len - il_jhaloend
         CASE('noextra')
            td_file%i_impp = il_ifirst
            td_file%i_jmpp = il_jfirst
            td_file%i_lci  = td_file%t_dim(jp_I)%i_len
            td_file%i_lcj  = td_file%t_dim(jp_J)%i_len
            td_file%i_ldi  = il_ihalostart + 1
            td_file%i_ldj  = il_jhalostart + 1
            td_file%i_lei  = td_file%i_lci - il_ihaloend
            td_file%i_lej  = td_file%i_lcj - il_jhaloend
         CASE('nooverlap') !!!?????
            td_file%i_impp = il_ifirst
            td_file%i_jmpp = il_jfirst
            td_file%i_lci  = td_file%t_dim(jp_I)%i_len
            td_file%i_lcj  = td_file%t_dim(jp_J)%i_len
            td_file%i_ldi  = 1
            td_file%i_ldj  = 1 
            td_file%i_lei  = td_file%t_dim(jp_I)%i_len
            td_file%i_lej  = td_file%t_dim(jp_J)%i_len
      END SELECT

   END SUBROUTINE mpp__read_halo
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE mpp__compute_halo(td_mpp) 
   !-------------------------------------------------------------------
   !> @brief This subroutine compute subdomain indices defined with halo
   !> (NEMO netcdf way)
   !>
   !> @author J.Paul
   !> @date January, 2016 - Initial Version
   !>
   !> @param[inout] td_mpp   mpp structure
   !-------------------------------------------------------------------
   
      IMPLICIT NONE

      ! Argument      
      TYPE(TMPP)      , INTENT(INOUT) :: td_mpp

      ! local variable
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_ifirst
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_jfirst
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_ilast
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_jlast
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_ihalostart
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_jhalostart
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_ihaloend
      INTEGER(i4), DIMENSION(:), ALLOCATABLE :: il_jhaloend

      TYPE(TATT)                             :: tl_att

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      ALLOCATE( il_ifirst    (td_mpp%i_nproc) )
      ALLOCATE( il_jfirst    (td_mpp%i_nproc) )

      ALLOCATE( il_ilast     (td_mpp%i_nproc) )
      ALLOCATE( il_jlast     (td_mpp%i_nproc) )

      ALLOCATE( il_ihalostart(td_mpp%i_nproc) )
      ALLOCATE( il_jhalostart(td_mpp%i_nproc) )

      ALLOCATE( il_ihaloend  (td_mpp%i_nproc) )
      ALLOCATE( il_jhaloend  (td_mpp%i_nproc) )

      SELECT CASE(TRIM(td_mpp%c_dom))
         CASE('full')
            
            il_ifirst(:)=td_mpp%t_proc(:)%i_impp
            il_jfirst(:)=td_mpp%t_proc(:)%i_jmpp
            
            il_ilast(:)=td_mpp%t_proc(:)%i_impp + td_mpp%t_proc(:)%t_dim(jp_I)%i_len - 1
            il_jlast(:)=td_mpp%t_proc(:)%i_jmpp + td_mpp%t_proc(:)%t_dim(jp_J)%i_len - 1

            il_ihalostart(:)=td_mpp%t_proc(:)%i_ldi-1
            il_jhalostart(:)=td_mpp%t_proc(:)%i_ldj-1
            
            il_ihaloend(:)=td_mpp%t_proc(:)%t_dim(jp_I)%i_len - td_mpp%t_proc(:)%i_lei
            il_jhaloend(:)=td_mpp%t_proc(:)%t_dim(jp_J)%i_len - td_mpp%t_proc(:)%i_lej

         CASE('noextra')
            
            il_ifirst(:)=td_mpp%t_proc(:)%i_impp
            il_jfirst(:)=td_mpp%t_proc(:)%i_jmpp

            il_ilast(:) =td_mpp%t_proc(:)%i_impp + td_mpp%t_proc(:)%i_lci - 1
            il_jlast(:) =td_mpp%t_proc(:)%i_jmpp + td_mpp%t_proc(:)%i_lcj - 1
            
            il_ihalostart(:)=td_mpp%t_proc(:)%i_ldi-1
            il_jhalostart(:)=td_mpp%t_proc(:)%i_ldj-1
            
            il_ihaloend(:)=td_mpp%t_proc(:)%i_lci - td_mpp%t_proc(:)%i_lei
            il_jhaloend(:)=td_mpp%t_proc(:)%i_lcj - td_mpp%t_proc(:)%i_lej

         CASE('nooverlap')

            il_ifirst(:)=td_mpp%t_proc(:)%i_impp + td_mpp%t_proc(:)%i_ldi - 1
            il_jfirst(:)=td_mpp%t_proc(:)%i_jmpp + td_mpp%t_proc(:)%i_ldj - 1

            il_ilast(:)=td_mpp%t_proc(:)%i_impp + td_mpp%t_proc(:)%i_lei - 1
            il_jlast(:)=td_mpp%t_proc(:)%i_jmpp + td_mpp%t_proc(:)%i_lej - 1

            il_ihalostart(:)=0
            il_jhalostart(:)=0

            il_ihaloend(:)=0
            il_jhaloend(:)=0

         CASE DEFAULT
            CALL logger_fatal("MPP INIT: invalid "//&
            &              "decomposition type.")                     
      END SELECT

      DO ji=1,td_mpp%i_nproc
         tl_att=att_init( "DOMAIN_position_first", &
         &                (/ il_ifirst(ji), il_jfirst(ji) /) )
         CALL file_move_att(td_mpp%t_proc(ji), tl_att)      

         tl_att=att_init( "DOMAIN_position_last", &
         &                (/ il_ilast(ji), il_jlast(ji) /) )
         CALL file_move_att(td_mpp%t_proc(ji), tl_att)

         tl_att=att_init( "DOMAIN_halo_size_start", &
         &                (/ il_ihalostart(ji), il_jhalostart(ji) /) )
         CALL file_move_att( td_mpp%t_proc(ji), tl_att)               

         tl_att=att_init( "DOMAIN_halo_size_end", &
         &                (/ il_ihaloend(ji), il_jhaloend(ji) /) )
         CALL file_move_att( td_mpp%t_proc(ji), tl_att)
      ENDDO

      DEALLOCATE( il_ifirst    )
      DEALLOCATE( il_jfirst    )
 
      DEALLOCATE( il_ilast     )
      DEALLOCATE( il_jlast     )
 
      DEALLOCATE( il_ihalostart)
      DEALLOCATE( il_jhalostart)

      DEALLOCATE( il_ihaloend  )
      DEALLOCATE( il_jhaloend  )

      !impp
      tl_att=att_init( "SUBDOMAIN_I_left_bottom_indices", td_mpp%t_proc(:)%i_impp)
      CALL mpp_move_att(td_mpp, tl_att)

      tl_att=att_init( "SUBDOMAIN_J_left_bottom_indices", td_mpp%t_proc(:)%i_jmpp)
      CALL mpp_move_att(td_mpp, tl_att)

      ! lci
      tl_att=att_init( "SUBDOMAIN_I_dimensions", td_mpp%t_proc(:)%i_lci)
      CALL mpp_move_att(td_mpp, tl_att)

      tl_att=att_init( "SUBDOMAIN_J_dimensions", td_mpp%t_proc(:)%i_lcj)
      CALL mpp_move_att(td_mpp, tl_att)

      ! ldi
      tl_att=att_init( "SUBDOMAIN_I_first_indoor_indices", td_mpp%t_proc(:)%i_ldi)
      CALL mpp_move_att(td_mpp, tl_att)

      tl_att=att_init( "SUBDOMAIN_J_first_indoor_indices", td_mpp%t_proc(:)%i_ldj)
      CALL mpp_move_att(td_mpp, tl_att)

      ! lei
      tl_att=att_init( "SUBDOMAIN_I_last_indoor_indices", td_mpp%t_proc(:)%i_lei)
      CALL mpp_move_att(td_mpp, tl_att)

      tl_att=att_init( "SUBDOMAIN_J_last_indoor_indices", td_mpp%t_proc(:)%i_lej)
      CALL mpp_move_att(td_mpp, tl_att)         

      ! clean
      CALL att_clean(tl_att)

   END SUBROUTINE mpp__compute_halo
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE mpp

