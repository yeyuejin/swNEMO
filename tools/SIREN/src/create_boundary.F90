!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @file
!> @brief 
!> This program creates boundary files.
!>
!> @details
!> @section sec1 method
!> Variables are read from coarse grid standard output, 
!> extracted or interpolated on fine grid. 
!> Variables could also be manually written.<br/>
!> @note 
!>    method could be different for each variable.
!>
!>  <br/> 
!> @image html  boundary_NEATL36_70.png 
!> <center>@image latex boundary_NEATL36_70.png
!> </center>
!>
!> @section sec2 how to
!> USAGE: create_boundary create_bounary.nam [-v] [-h]<br/>
!>    - positional arguments:<br/>
!>       - create_boundary.nam<br/>
!>          namelist of create_boundary
!>          @note
!>             a template of the namelist could be created running (in templates directory):
!>             @code{.sh}
!>                python create_templates.py create_boundary
!>             @endcode
!>
!>    - optional arguments:<br/>
!>       - -h, --help<br/>
!>          show this help message (and exit)<br/>
!>       - -v, --version<br/>
!>          show Siren's version   (and exit)<br/>
!>    @note 
!>       compiled with @a key_mpp_mpi, could be run on multi processor :<br/>
!>       USAGE: create_boundary create_bounary.nam create_bounary2.nam ... [-v] [-h]<br/>
!>
!> @section sec_boundary create_boundary.nam
!>    create_boundary.nam contains 9 namelists:<br/>
!>       - **namlog** to set logger parameters
!>       - **namcfg** to set configuration file parameters
!>       - **namsrc** to set source/coarse grid parameters
!>       - **namtgt** to set target/fine grid parameters
!>       - **namvar** to set variable parameters
!>       - **namnst** to set sub domain and nesting paramters
!>       - **nambdy** to set boundary parameters
!>       - **namzgr** to set vertical grid parameters
!>       - **namout** to set output parameters
!>
!>    here after, each sub-namelist parameters is detailed.
!>    @note 
!>       default values are specified between brackets
!>
!> @subsection sublog namlog
!>    the logger sub-namelist parameters are :
!>
!>    - **cn_logfile** [@a create_boundary.log]<br/>
!>       logger filename
!>
!>    - **cn_verbosity** [@a warning]<br/>
!>       verbosity level, choose between :
!>          - trace
!>          - debug
!>          - info
!>          - warning
!>          - error
!>          - fatal
!>          - none
!>
!>    - **in_maxerror** [@a 5]<br/> 
!>       maximum number of error allowed
!>
!> @subsection subcfg namcfg
!>    the configuration sub-namelist parameters are :
!>
!>    - **cn_varcfg** [@a ./cfg/variable.cfg]<br/>
!>       path to the variable configuration file.<br/>
!>       the variable configuration file defines standard name, 
!>       default interpolation method, axis,... 
!>       to be used for some known variables.<br/> 
!>
!>    - **cn_dimcfg** [@a ./cfg/dimension.cfg]<br/> 
!>       path to the dimension configuration file.<br/> 
!>       the dimension configuration file defines dimensions allowed.<br/> 
!>
!>    - **cn_dumcfg** [@a ./cfg/dummy.cfg]<br/> 
!>       path to the useless (dummy) configuration file.<br/>
!>       the dummy configuration file defines useless 
!>       dimension or variable. these dimension(s) or variable(s) will not be
!>       processed.<br/>
!>
!> @subsection subcrs namcrs 
!>    the coarse grid sub-namelist parameters are :
!>
!>    - **cn_coord0** [@a ]<br/> 
!>       path to the coordinate file
!>
!>    - **in_perio0** [@a ]<br/> 
!>       NEMO periodicity index<br/> 
!>       the NEMO periodicity could be choose between 0 to 6:
!>       <dl>
!>          <dt>in_perio=0</dt>
!>          <dd>standard regional model</dd>
!>          <dt>in_perio=1</dt>
!>          <dd>east-west cyclic model</dd>
!>          <dt>in_perio=2</dt>
!>          <dd>model with symmetric boundary condition across the equator</dd>
!>          <dt>in_perio=3</dt>
!>          <dd>regional model with North fold boundary and T-point pivot</dd>
!>          <dt>in_perio=4</dt>
!>          <dd>global model with a T-point pivot.<br/>
!>          example: ORCA2, ORCA025, ORCA12</dd>
!>          <dt>in_perio=5</dt>
!>          <dd>regional model with North fold boundary and F-point pivot</dd>
!>          <dt>in_perio=6</dt>
!>          <dd>global model with a F-point pivot<br/>
!>          example: ORCA05</dd>
!>          </dd>
!>       </dl>
!>       @sa For more information see @ref md_src_docsrc_6_perio
!>       and Model Boundary Condition paragraph in the 
!>       [NEMO documentation](https://forge.ipsl.jussieu.fr/nemo/chrome/site/doc/NEMO/manual/pdf/NEMO_manual.pdf)
!>
!> @subsection subfin namfin 
!>    the fine grid sub-namelist parameters are :
!>
!>    - **cn_coord1** [@a ]<br/> 
!>       path to coordinate file
!>
!>    - **cn_bathy1** [@a ]<br/> 
!>       path to bathymetry file
!>       @warning 
!>
!>    - **in_perio1** [@a ]<br/>
!>       NEMO periodicity index (see above)
!>    @note if the fine/target coordinates file (cn_coord1) was created by SIREN, you do
!>    not need to fill this parameter. SIREN will read it on the global attributes of
!>    the coordinates file.
!>
!> @subsection subzgr namzgr
!>    the vertical grid sub-namelist parameters are :
!>
!>    - **dn_pp_to_be_computed** [@a 0]<br/>
!>
!>    - **dn_ppsur** [@a -3958.951371276829]<br/>
!>       coefficient to compute vertical grid
!>
!>    - **dn_ppa0** [@a 103.953009600000]<br/>
!>       coefficient to compute vertical grid
!>
!>    - **dn_ppa1** [@a 2.415951269000]<br/>
!>       coefficient to compute vertical grid
!>
!>    - **dn_ppa2** [@a 100.760928500000]<br/>
!>       double tanh function parameter
!>
!>    - **dn_ppkth** [@a 15.351013700000]<br/>
!>       coefficient to compute vertical grid
!>
!>    - **dn_ppkth2** [@a 48.029893720000]<br/>
!>       double tanh function parameter
!>
!>    - **dn_ppacr** [@a 7.000000000000]<br/>
!>       coefficient to compute vertical grid
!>
!>    - **dn_ppacr2** [@a 13.000000000000]<br/>
!>       double tanh function parameter
!>
!>    - **dn_ppdzmin** [@a 6.]<br/>
!>       minimum vertical spacing
!>
!>    - **dn_pphmax** [@a 5750.]<br/>
!>       maximum depth
!>
!>    - **in_nlevel** [@a 75]<br/>
!>       number of vertical level
!>
!>     @note 
!>       If *dn_ppa1*, *dn_ppa0* and *dn_ppsur* are undefined,
!>       NEMO will compute them from *dn_ppdzmin, dn_pphmax, dn_ppkth, dn_ppacr*
!>
!> @subsection subzps namzps
!>    the partial step sub-namelist parameters are :
!>
!>    - **dn_e3zps_min** [@a 25.]<br/>
!>       minimum thickness of partial step level (meters)
!>    - **dn_e3zps_rat** [@a 0.2]<br/>
!>       minimum thickness ratio of partial step level
!>
!> @subsection subvar namvar 
!>    the variable sub-namelist parameters are :
!>
!>    - **cn_varfile** [@a ]<br/> 
!>       list of variable, and associated file 
!>
!>       *cn_varfile* is the path and filename of the file where find
!>       variable.
!>       @note 
!>          *cn_varfile* could be a matrix of value, if you want to handwrite
!>          variable value.<br/>
!>          the variable array of value is split into equal subdomain.<br/>
!>          each subdomain is filled with the corresponding value 
!>          of the matrix.<br/>          
!>          separators used to defined matrix are:
!>             - ',' for line
!>             - '/' for row
!>             - '\' for level<br/>
!>             Example:<br/>
!>                3,2,3/1,4,5  =>  @f$ \left( \begin{array}{ccc}
!>                                      3 & 2 & 3 \\
!>                                      1 & 4 & 5 \end{array} \right) @f$
!>
!>          @warning 
!>             the same matrix is used for all boundaries.
!>
!>       Examples: 
!>          - 'votemper:gridT.nc', 'vozocrtx:gridU.nc'
!>          - 'votemper:10\25', 'vozocrtx:gridU.nc'<br/>
!>
!>       @note 
!>          Optionnaly, NEMO periodicity could be added following the filename.
!>          the periodicity must be separated by ';'
!>
!>       Example:
!>          - 'votemper:gridT.nc ; perio=4'
!>
!>    - **cn_varinfo** [@a ]<br/> 
!>       list of variable and extra information about request(s) to be used<br/>
!>
!>       each elements of *cn_varinfo* is a string character (separated by ',').<br/>
!>       it is composed of the variable name follow by ':', 
!>       then request(s) to be used on this variable.<br/> 
!>       request could be:
!>          - int = interpolation method
!>          - ext = extrapolation method
!>          - flt = filter method
!>          - min = minimum value
!>          - max = maximum value
!>          - unt = new units
!>          - unf = unit scale factor (linked to new units)
!>
!>             requests must be separated by ';'.<br/>
!>             order of requests does not matter.<br/>
!>
!>       informations about available method could be find in @ref interp,
!>       @ref extrap and @ref filter modules.<br/>
!>       Example: 
!>          - 'votemper: int=linear; flt=hann; ext=dist_weight',
!>            'vosaline: int=cubic'
!>
!>       @note 
!>          If you do not specify a method which is required, 
!>          default one is apply.
!>
!> @subsection subnst namnst 
!>    the nesting sub-namelist parameters are :
!>
!>    - **in_rhoi**  [@a 1]<br/> 
!>       refinement factor in i-direction
!>
!>    - **in_rhoj**  [@a 1]<br/> 
!>       refinement factor in j-direction
!>
!>    @note 
!>       coarse grid indices will be deduced from fine grid
!>       coordinate file.
!>
!> @subsection subbdy nambdy
!>    the boundary sub-namelist parameters are :
!>
!>    - **ln_north** [@a .TRUE.]<br/> 
!>       logical to use north boundary or not
!>    - **ln_south** [@a .TRUE.]<br/>
!>       logical to use south boundary or not
!>    - **ln_east**  [@a .TRUE.]<br/>
!>       logical to use east boundary or not
!>    - **ln_west**  [@a .TRUE.]<br/>
!>       logical to use west  boundary or not
!>    <br/> <br/>
!>    - **cn_north** [@a ]<br/>
!>       north boundary indices on fine grid<br/>
!>    - **cn_south** [@a ]<br/>
!>       south boundary indices on fine grid<br/>
!>    - **cn_east**  [@a ]<br/>
!>       east  boundary indices on fine grid<br/>
!>    - **cn_west**  [@a ]<br/>
!>       west  boundary indices on fine grid<br/>
!>
!>       *cn_north* is a string character defining boundary
!>       segmentation.<br/>
!>       segments are separated by '|'.<br/>
!>       each segments of the boundary is composed of:
!>          - indice of velocity (orthogonal to boundary .ie. 
!>             for north boundary, J-indice). 
!>          - indice of segment start (I-indice for north boundary) 
!>          - indice of segment end   (I-indice for north boundary)<br/>
!>             indices must be separated by ':' .<br/>
!>          - optionally, boundary size could be added between '(' and ')' 
!>          in the first segment defined.
!>             @note 
!>                boundary size is the same for all segments of one boundary.
!>
!>       Examples:
!>          - cn_north='index1,first1:last1(width)'
!>          - cn_north='index1(width),first1:last1|index2,first2:last2'
!>
!>       @image html  boundary_50.png 
!>       <center>@image latex boundary_50.png
!>       </center>
!>
!>    - **ln_oneseg** [@a .TRUE.]<br/>
!>       logical to use only one segment for each boundary or not
!>
!>    @note
!>       the number of point(s) with coarse value save at boundaries is
!>       defined with the *weight* variable (see @ref merge_bathy)
!>
!> @subsection subout namout 
!>    the output sub-namelist parameter is :
!>
!>    - **cn_fileout** [@a boundary.nc]<br/>
!>       output bathymetry filename
!>
!>       @note
!>          cardinal point and segment number will be automatically added
!>
!>    - **ln_extrap** [@a .FALSE.]<br/>
!>       extrapolate on land point
!>
!>    - **dn_dayofs** [@a 0]<br/>
!>       date offset in day (change only ouput file name)
!>
!>       Examples: 
!>          - cn_fileout='boundary.nc'<br/>
!>             if time_counter (16/07/2015 00h) is read on input file (see varfile), 
!>             west boundary will be named boundary_west_y2015m07d16
!>          - dn_dayofs=-2.<br/>
!>             if you use day offset you get boundary_west_y2015m07d14
!>
!> @subsection sub_nambdy How to fill Lateral Boundary Condition in NEMO namelist
!>    To use boundary condition within NEMO, you need to fill the NEMO namelist.<br/>
!>    As this is a little bit messy for lateral boundary condition, here after
!>    is an explanation of how to do it.
!>
!>    This will be done in 3 steps.
!>
!>    @subsubsection ss_nambdy nambdy
!>       The *nambdy* NEMO sub-namelist defines open boundaries.<br/>
!>       Here we indicate the number of open boundary (**nb_bdy**).
!>
!>       @note
!>          we have to fill most of the parameters with as many elements as there are open boundaries
!> 
!>       Regarding the width of the relaxation zone **nn_rimwidth**,
!>       this information is available as a global attribute (**bdy_width**) 
!>       in the metadata of boundary files created with SIREN 
!>
!> @code{.sh}
!>    ncdump -h boundary_east.nc
!> @endcode
!>       @warning
!>          The order of the boundaries must stay unchanged, in parameters list as well as 
!>          in the next sub-namelsits 
!>
!>    Example:<br/>
!>       here is an example for a domain with two boundaries East and North
!>
!> @code{.sh}
!> !-----------------------------------------------------------------------
!> &nambdy        !  unstructured open boundaries                          ("key_bdy")
!> !-----------------------------------------------------------------------
!>   nb_bdy         = 2                    !  number of open boundary sets
!>   ln_coords_file = .false.,.false.      !  =T : read bdy coordinates from file
!>   cn_coords_file = '',''                !  bdy coordinates files
!>   ln_mask_file   = .false.              !  =T : read mask from file
!>   cn_mask_file   = ''                   !  name of mask file (if ln_mask_file=.TRUE.)
!>   cn_dyn2d       = 'flather','flather'  !
!>   nn_dyn2d_dta   = 1,1                  !  = 0, bdy data are equal to the initial state
!>                                         !  = 1, bdy data are read in 'bdydata   .nc' files
!>                                         !  = 2, use tidal harmonic forcing data from files
!>                                         !  = 3, use external data AND tidal harmonic forcing
!>   cn_dyn3d       = 'specified','specified' !  
!>   nn_dyn3d_dta   = 1,1                  !  = 0, bdy data are equal to the initial state
!>                                         !  = 1, bdy data are read in 'bdydata   .nc' files
!>   cn_tra         = 'specified','specified' ! 
!>   nn_tra_dta     = 1,1                  !  = 0, bdy data are equal to the initial state
!>                                         !  = 1, bdy data are read in 'bdydata   .nc' files
!>                                         !
!>   ln_tra_dmp    =.true.,.true.          !  open boudaries conditions for tracers
!>   ln_dyn3d_dmp  =.true.,.true.          !  open boundary condition for baroclinic velocities
!>   rn_time_dmp     =  1.,1.              ! Damping time scale in days 
!>   rn_time_dmp_out =  1.,1.              ! Outflow damping time scale
!>   nn_rimwidth   = 10,10                 !  width of the relaxation zone
!>   ln_vol        = .false.               !  total volume correction (see nn_volctl parameter)
!>   nn_volctl     = 1                     !  = 0, the total water flux across open boundaries is zero
!> /
!> @endcode
!>
!>    @subsubsection ss_nambdy_index nambdy_index
!>       The *nambdy_index* NEMO sub-namelist describes the boundaries we will use.
!>
!>       @warning
!>          We have to add as many as sub namelist *nambdy_index* than open boundaries (nb_bdy),
!>          and keep them in the same order as above
!>
!>          Here we indicate if the open boundary is North, South, East, or West (**ctypebdy**).<br/>
!>          We also indicate indice of segment start and end (respectively **nbdybeg**  and **nbdyend**)
!>          as well as indice of velocity row or column (**nbdyind**).<br/>
!>
!>          Those informations are available as global attributes 
!>          (respectively **bdy_deb, bdy_end, bdy_ind**) in the metadata of our boundary files 
!>          created with SIREN.
!>
!>    Example:<br/>
!>       here is an example for a domain with two boundaries East and North
!>
!> @code{.sh}
!>    !-----------------------------------------------------------------------
!>    &nambdy_index  !  structured open boundaries definition     ("key_bdy")
!>    !-----------------------------------------------------------------------
!>      ctypebdy ='E'                   ! Open boundary type (W,E,S or N)
!>      nbdyind  = 407                  ! indice of velocity row or column
!>                                      ! if ==-1, set obc at the domain boundary
!>                                      !        , discard start and end indices
!>      nbdybeg  = 32                   ! indice of segment start
!>      nbdyend  = 300                  ! indice of segment end
!>    /
!>    !-----------------------------------------------------------------------
!>    &nambdy_index  !  structured open boundaries definition     ("key_bdy")
!>    !-----------------------------------------------------------------------
!>      ctypebdy ='N'                   ! Open boundary type (W,E,S or N)
!>      nbdyind  = 299                  ! indice of velocity row or column
!>                                      ! if ==-1, set obc at the domain boundary
!>                                      !        , discard start and end indices
!>      nbdybeg  = 200                  ! indice of segment start
!>      nbdyend  = 408                  ! indice of segment end
!>    /
!> @endcode
!>
!>    @subsubsection ss_nambdy_dat nambdy_dta
!>       The *nambdy_dta* NEMO sub-namelists describes the boundary data and files to be used.<br/>
!>       @warning
!>          We have to add as many as sub namelist *nambdy_dta* than open boundaries (nb_bdy),
!>          and keep them in the same order as above
!>
!>    Example:<br/>
!>       here is an example for a domain with two boundaries East and North
!>
!> @code{.sh}
!> !-----------------------------------------------------------------------
!> &nambdy_dta      !  open boundaries - external data           ("key_bdy")
!> !-----------------------------------------------------------------------
!> !          ! file name !  freq (hours)   ! variable ! time interp. ! clim  ! 'yearly'/ ! weights...
!> !          !           ! (if < 0 months) !   name   !   (logical)  ! (T/F) ! 'monthly' ! filename
!>   bn_ssh = 'boundary_east' , -12   , 'sossheig' , .false. , .true. , 'yearly' , '', '', ''
!>   bn_u2d = 'boundary_east' , -12   , 'vobtcrtx' , .false. , .true. , 'yearly' , '', '', ''
!>   bn_v2d = 'boundary_east' , -12   , 'vobtcrty' , .false. , .true. , 'yearly' , '', '', ''
!>   bn_u3d = 'boundary_east' , -12   , 'vozocrtx' , .false. , .true. , 'yearly' , '', '', ''
!>   bn_v3d = 'boundary_east' , -12   , 'vomecrty' , .false. , .true. , 'yearly' , '', '', ''
!>   bn_tem = 'boundary_east' , -12   , 'votemper' , .false. , .true. , 'yearly' , '', '', ''
!>   bn_sal = 'boundary_east' , -12   , 'vosaline' , .false. , .true. , 'yearly' , '', '', ''
!>   cn_dir = './'
!>   ln_full_vel = .true.
!> /
!> !-----------------------------------------------------------------------
!> &nambdy_dta      !  open boundaries - external data           ("key_bdy")
!> !-----------------------------------------------------------------------
!> !          ! file name !  freq (hours)   ! variable ! time interp. ! clim  ! 'yearly'/ ! weights...
!> !          !           ! (if < 0 months) !   name   !   (logical)  ! (T/F) ! 'monthly' ! filename
!>   bn_ssh = 'boundary_north' , -12   , 'sossheig' ,  .false. , .true. , 'yearly' , '', '', ''
!>   bn_u2d = 'boundary_north' , -12   , 'vobtcrtx' ,  .false. , .true. , 'yearly' , '', '', ''
!>   bn_v2d = 'boundary_north' , -12   , 'vobtcrty' ,  .false. , .true. , 'yearly' , '', '', ''
!>   bn_u3d = 'boundary_north' , -12   , 'vozocrtx' ,  .false. , .true. , 'yearly' , '', '', ''
!>   bn_v3d = 'boundary_north' , -12   , 'vomecrty' ,  .false. , .true. , 'yearly' , '', '', ''
!>   bn_tem = 'boundary_north' , -12   , 'votemper' ,  .false. , .true. , 'yearly' , '', '', ''
!>   bn_sal = 'boundary_north' , -12   , 'vosaline' ,  .false. , .true. , 'yearly' , '', '', ''
!>   cn_dir = './'
!>   ln_full_vel = .true.
!> /
!> @endcode
!>
!> <hr>
!> @author J.Paul
!>
!> @date November, 2013 - Initial Version
!> @date September, 2014
!> - add header for user
!> - take into account grid point to compue boundaries
!> - reorder output dimension for north and south boundaries
!> @date June, 2015
!> - extrapolate all land points, and add ln_extrap in namelist.
!> - allow to change unit.
!> @date July, 2015
!> - add namelist parameter to shift date of output file name.
!> @date September, 2015
!> - manage useless (dummy) variable, attributes, and dimension
!> - allow to run on multi processors with key_mpp_mpi
!> @date January, 2016
!> - same process use for variable extracted or interpolated from input file.
!> @date October, 2016
!> - dimension to be used select from configuration file
!> @date January, 2019
!> - add url path to global attributes of output file(s)
!> - create and clean file structure to avoid memory leaks
!> - explain how to fill Lateral Boundary Condition in NEMO namelist
!> @date February, 2019
!> - rename sub namelist namcrs to namsrc
!> - rename sub namelist namfin to namtgt
!> @date August, 2019
!> - use periodicity read from namelist, and store in multi structure
!> @date Ocober, 2019
!> - add help and version optional arguments
!>
!> @todo
!> - rewitre using meshmask instead of bathymetry and coordinates files.
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
PROGRAM create_boundary

   USE netcdf                          ! nf90 library
   USE global                          ! global variable
   USE phycst                          ! physical constant
   USE kind                            ! F90 kind parameter
   USE fct                             ! basic useful function
   USE date                            ! date manager
   USE att                             ! attribute manager
   USE dim                             ! dimension manager
   USE var                             ! variable manager
   USE file                            ! file manager
   USE multi                           ! multi file manager
   USE boundary                        ! boundary manager
   USE iom                             ! I/O manager
   USE dom                             ! domain manager
   USE grid                            ! grid manager
   USE vgrid                           ! vertical grid manager
   USE extrap                          ! extrapolation manager
   USE interp                          ! interpolation manager
   USE filter                          ! filter manager
   USE mpp                             ! MPP manager
   USE iom_mpp                         ! MPP I/O manager

   IMPLICIT NONE

   ! parameters
   CHARACTER(LEN=lc), PARAMETER  :: cp_myname = "create_boundary"

   ! local variable
   CHARACTER(LEN=lc)                                  :: cl_arg
   CHARACTER(LEN=lc)                                  :: cl_errormsg

   INTEGER(i4)                                        :: il_narg

#if defined key_mpp_mpi
   ! mpp variable
   CHARACTER(LEN=lc), DIMENSION(:)      , ALLOCATABLE :: cl_args

   INTEGER(i4)                                        :: ierror
   INTEGER(i4)                                        :: iproc
   INTEGER(i4)                                        :: nproc
   INTEGER(i4)      , DIMENSION(:)      , ALLOCATABLE :: il_nprog

   ! loop indices
   INTEGER(i4) :: jm
#else
   CHARACTER(LEN=lc)                                  :: cl_namelist
#endif
   !-------------------------------------------------------------------
#if defined key_mpp_mpi
   INCLUDE 'mpif.h'
#endif
   !-------------------------------------------------------------------

   !
   ! Initialisation
   ! --------------
   !
   il_narg=COMMAND_ARGUMENT_COUNT() !f03 intrinsec

#if ! defined key_mpp_mpi

   ! Traitement des arguments fournis
   ! --------------------------------
   IF( il_narg /= 1 )THEN
      WRITE(cl_errormsg,*) ' ERROR : one argument is needed '
      CALL fct_help(cp_myname,cl_errormsg) 
      CALL EXIT(1)
   ELSE

      CALL GET_COMMAND_ARGUMENT(1,cl_arg) !f03 intrinsec
      SELECT CASE (cl_arg)
         CASE ('-v', '--version')

            CALL fct_version(cp_myname)
            CALL EXIT(0)

         CASE ('-h', '--help')

            CALL fct_help(cp_myname)
            CALL EXIT(0)

         CASE DEFAULT

            cl_namelist=cl_arg
            
            CALL GET_COMMAND_ARGUMENT(1,cl_namelist) !f03 intrinsec
            CALL create_boundary__mono(cl_namelist)

      END SELECT
   ENDIF
#else

   ! Initialize MPI
   CALL mpi_init(ierror)
   CALL mpi_comm_rank(mpi_comm_world,iproc,ierror)
   CALL mpi_comm_size(mpi_comm_world,nproc,ierror)

   ! Traitement des arguments fournis
   ! --------------------------------
   IF( il_narg == 0 )THEN
      WRITE(cl_errormsg,*) ' ERROR : at least one argument is needed '
      CALL fct_help(cp_myname,cl_errormsg) 
      CALL EXIT(1)
   ELSE

      ALLOCATE(cl_args(il_narg))
      DO jm=1,il_narg

         CALL GET_COMMAND_ARGUMENT(jm,cl_arg) !f03 intrinsec
         SELECT CASE (cl_arg)
            CASE ('-v', '--version')

               CALL fct_version(cp_myname)
               CALL EXIT(0)

            CASE ('-h', '--help')

               CALL fct_help(cp_myname)
               CALL EXIT(0)

            CASE DEFAULT

               cl_args(jm)=TRIM(cl_arg)

         END SELECT
      ENDDO
   ENDIF

   ALLOCATE(il_nprog(il_narg))
   DO jm=1, il_narg
      il_nprog(jm)= MOD(jm,nproc)
   ENDDO

   DO jm=1, il_narg
      IF ( il_nprog(jm) .eq. iproc ) THEN
         CALL create_boundary__mono(cl_args(jm))
      ENDIF
   ENDDO

   CALL mpi_finalize(ierror)

   DEALLOCATE(cl_args)
   DEALLOCATE(il_nprog)
#endif

CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE create_boundary__mono(cd_namelist)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine create boundary files.
   !> 
   !> @details 
   !>
   !> @author J.Paul
   !> @date January, 2016 - Initial Version
   !>
   !> @param[in] cd_namelist namelist file 
   !-------------------------------------------------------------------

   USE logger                          ! log file manager

   IMPLICIT NONE
   ! Argument
   CHARACTER(LEN=lc), INTENT(IN) :: cd_namelist 

   ! local variable
   CHARACTER(LEN=lc)                                  :: cl_date
   CHARACTER(LEN=lc)                                  :: cl_name
   CHARACTER(LEN=lc)                                  :: cl_bdyout
   CHARACTER(LEN=lc)                                  :: cl_data
   CHARACTER(LEN=lc)                                  :: cl_dimorder
   CHARACTER(LEN=lc)                                  :: cl_fmt
   CHARACTER(LEN=lc)                                  :: cl_url

   INTEGER(i4)                                        :: il_status
   INTEGER(i4)                                        :: il_fileid
   INTEGER(i4)                                        :: il_imin0
   INTEGER(i4)                                        :: il_imax0
   INTEGER(i4)                                        :: il_jmin0
   INTEGER(i4)                                        :: il_jmax0
   INTEGER(i4)                                        :: il_shift
   INTEGER(i4)      , DIMENSION(ip_maxdim)            :: il_rho
   INTEGER(i4)      , DIMENSION(2,2)                  :: il_offset
   INTEGER(i4)      , DIMENSION(2,2)                  :: il_ind

   LOGICAL                                            :: ll_exist

   TYPE(TATT)                                         :: tl_att
   
   TYPE(TVAR)                                         :: tl_depth   
   TYPE(TVAR)                                         :: tl_time
   TYPE(TVAR)                                         :: tl_var1
   TYPE(TVAR)                                         :: tl_var0
   TYPE(TVAR)                                         :: tl_lon1
   TYPE(TVAR)                                         :: tl_lat1
   TYPE(TVAR)                                         :: tl_lvl1  
   TYPE(TVAR)       , DIMENSION(:)      , ALLOCATABLE :: tl_level
   TYPE(TVAR)       , DIMENSION(:,:,:)  , ALLOCATABLE :: tl_seglvl1
   TYPE(TVAR)       , DIMENSION(:,:,:)  , ALLOCATABLE :: tl_segvar1

   TYPE(TDIM)       , DIMENSION(ip_maxdim)            :: tl_dim

   TYPE(TDATE)                                        :: tl_date
   
   TYPE(TBDY)       , DIMENSION(ip_ncard)             :: tl_bdy
   
   TYPE(TDOM)                                         :: tl_dom0
   TYPE(TDOM)                                         :: tl_dom1
   TYPE(TDOM)       , DIMENSION(:,:,:)  , ALLOCATABLE :: tl_segdom1

   TYPE(TFILE)                                        :: tl_file
   TYPE(TFILE)                                        :: tl_fileout
   
   TYPE(TMPP)                                         :: tl_coord0
   TYPE(TMPP)                                         :: tl_coord1
   TYPE(TMPP)                                         :: tl_bathy1
   TYPE(TMPP)                                         :: tl_mpp

   TYPE(TMULTI)                                       :: tl_multi

   ! loop indices
   INTEGER(i4) :: jvar
   INTEGER(i4) :: jpoint
   INTEGER(i4) :: ji
   INTEGER(i4) :: jj
   INTEGER(i4) :: jk
   INTEGER(i4) :: jl

   ! namelist variable
   ! namlog
   CHARACTER(LEN=lc)                       :: cn_logfile    = 'create_boundary.log' 
   CHARACTER(LEN=lc)                       :: cn_verbosity  = 'warning' 
   INTEGER(i4)                             :: in_maxerror   = 5

   ! namcfg
   CHARACTER(LEN=lc)                       :: cn_varcfg  = './cfg/variable.cfg' 
   CHARACTER(LEN=lc)                       :: cn_dimcfg  = './cfg/dimension.cfg'
   CHARACTER(LEN=lc)                       :: cn_dumcfg  = './cfg/dummy.cfg'

   ! namsrc
   CHARACTER(LEN=lc)                       :: cn_coord0  = '' 
   INTEGER(i4)                             :: in_perio0  = -1

   ! namtgt
   CHARACTER(LEN=lc)                       :: cn_coord1  = '' 
   CHARACTER(LEN=lc)                       :: cn_bathy1  = '' 
   INTEGER(i4)                             :: in_perio1  = -1

   !namzgr
   REAL(dp)                                :: dn_pp_to_be_computed = 0._dp
   REAL(dp)                                :: dn_ppsur   = -3958.951371276829_dp
   REAL(dp)                                :: dn_ppa0    =   103.953009600000_dp
   REAL(dp)                                :: dn_ppa1    =     2.415951269000_dp
   REAL(dp)                                :: dn_ppa2    =   100.760928500000_dp
   REAL(dp)                                :: dn_ppkth   =    15.351013700000_dp
   REAL(dp)                                :: dn_ppkth2  =    48.029893720000_dp
   REAL(dp)                                :: dn_ppacr   =     7.000000000000_dp
   REAL(dp)                                :: dn_ppacr2  =    13.000000000000_dp
   REAL(dp)                                :: dn_ppdzmin = 6._dp
   REAL(dp)                                :: dn_pphmax  = 5750._dp
   INTEGER(i4)                             :: in_nlevel  = 75

   !namzps
   REAL(dp)                                :: dn_e3zps_min  = 25._dp
   REAL(dp)                                :: dn_e3zps_rat  = 0.2_dp

   ! namvar
   CHARACTER(LEN=lc), DIMENSION(ip_maxvar) :: cn_varfile = ''
   CHARACTER(LEN=lc), DIMENSION(ip_maxvar) :: cn_varinfo = ''

   ! namnst
   INTEGER(i4)                             :: in_rhoi    = 1
   INTEGER(i4)                             :: in_rhoj    = 1

   ! nambdy
   LOGICAL                                 :: ln_north   = .TRUE.
   LOGICAL                                 :: ln_south   = .TRUE.
   LOGICAL                                 :: ln_east    = .TRUE.
   LOGICAL                                 :: ln_west    = .TRUE.
   LOGICAL                                 :: ln_oneseg  = .TRUE.
   CHARACTER(LEN=lc)                       :: cn_north   = ''
   CHARACTER(LEN=lc)                       :: cn_south   = ''
   CHARACTER(LEN=lc)                       :: cn_east    = ''
   CHARACTER(LEN=lc)                       :: cn_west    = ''

   ! namout
   CHARACTER(LEN=lc)                       :: cn_fileout = 'boundary.nc' 
   REAL(dp)                                :: dn_dayofs  = 0._dp
   LOGICAL                                 :: ln_extrap  = .FALSE.
   !-------------------------------------------------------------------

   NAMELIST /namlog/ &  !< logger namelist
   &  cn_logfile,    &  !< log file
   &  cn_verbosity,  &  !< log verbosity
   &  in_maxerror       !< logger maximum error

   NAMELIST /namcfg/ &  !< configuration namelist
   &  cn_varcfg,     &  !< variable configuration file
   &  cn_dimcfg,     &  !< dimension configuration file
   &  cn_dumcfg         !< dummy configuration file

   NAMELIST /namsrc/ &  !< source/coarse grid namelist
   &  cn_coord0,     &  !< coordinate file
   &  in_perio0         !< periodicity index

   NAMELIST /namtgt/ &  !< target/fine grid namelist
   &  cn_coord1,     &  !< coordinate file
   &  cn_bathy1,     &  !< bathymetry file
   &  in_perio1         !< periodicity index
 
   NAMELIST /namzgr/ &
   &  dn_pp_to_be_computed, &
   &  dn_ppsur,      &
   &  dn_ppa0,       &
   &  dn_ppa1,       &
   &  dn_ppa2,       &
   &  dn_ppkth,      &
   &  dn_ppkth2,     &
   &  dn_ppacr,      &
   &  dn_ppacr2,     &
   &  dn_ppdzmin,    &
   &  dn_pphmax,     &
   &  in_nlevel         !< number of vertical level

   NAMELIST /namzps/ &
   &  dn_e3zps_min,  &
   &  dn_e3zps_rat

   NAMELIST /namvar/ &  !< variable namelist
   &  cn_varfile,    &  !< list of variable and file where find it. (ex: 'votemper:GLORYS_gridT.nc' ) 
   &  cn_varinfo        !< list of variable and method to apply on. (ex: 'votemper:linear','vosaline:cubic' )
 
   NAMELIST /namnst/ &  !< nesting namelist
   &  in_rhoi,       &  !< refinement factor in i-direction
   &  in_rhoj           !< refinement factor in j-direction

   NAMELIST /nambdy/ &  !< boundary namelist
   &  ln_north,      &  !< use north boundary
   &  ln_south,      &  !< use south boundary
   &  ln_east ,      &  !< use east  boundary
   &  ln_west ,      &  !< use west  boundary
   &  cn_north,      &  !< north boundary indices on fine grid
   &  cn_south,      &  !< south boundary indices on fine grid
   &  cn_east ,      &  !< east  boundary indices on fine grid
   &  cn_west ,      &  !< west  boundary indices on fine grid
   &  ln_oneseg         !< use only one segment for each boundary or not

   NAMELIST /namout/ &  !< output namelist
   &  cn_fileout,    &  !< fine grid boundary file basename   
   &  dn_dayofs,     &  !< date offset in day (change only ouput file name)
   &  ln_extrap         !< extrapolate or not
   !-------------------------------------------------------------------

   ! read namelist
   INQUIRE(FILE=TRIM(cd_namelist), EXIST=ll_exist)

   IF( ll_exist )THEN
      
      il_fileid=fct_getunit()

      OPEN( il_fileid, FILE=TRIM(cd_namelist), &
      &                FORM='FORMATTED',       &
      &                ACCESS='SEQUENTIAL',    &
      &                STATUS='OLD',           &
      &                ACTION='READ',          &
      &                IOSTAT=il_status)
      CALL fct_err(il_status)
      IF( il_status /= 0 )THEN
         PRINT *,"CREATE BOUNDARY: ERROR opening "//TRIM(cd_namelist)
         STOP
      ENDIF

      READ( il_fileid, NML = namlog )
      ! define log file
      CALL logger_open(TRIM(cn_logfile),TRIM(cn_verbosity),in_maxerror)
      CALL logger_header()

      READ( il_fileid, NML = namcfg )
      ! get variable extra information
      CALL var_def_extra(TRIM(cn_varcfg))

      ! get dimension allowed
      CALL dim_def_extra(TRIM(cn_dimcfg))

      ! get dummy variable
      CALL var_get_dummy(TRIM(cn_dumcfg))
      ! get dummy dimension
      CALL dim_get_dummy(TRIM(cn_dumcfg))
      ! get dummy attribute
      CALL att_get_dummy(TRIM(cn_dumcfg))

      READ( il_fileid, NML = namsrc )
      READ( il_fileid, NML = namtgt )
      READ( il_fileid, NML = namzgr )
      READ( il_fileid, NML = namvar )
      ! add user change in extra information
      CALL var_chg_extra(cn_varinfo)
      ! match variable with file
      tl_multi=multi_init(cn_varfile)

      READ( il_fileid, NML = namnst )
      READ( il_fileid, NML = nambdy )
      READ( il_fileid, NML = namout )

      CLOSE( il_fileid, IOSTAT=il_status )
      CALL fct_err(il_status)
      IF( il_status /= 0 )THEN
         CALL logger_error("CREATE BOUNDARY: ERROR closing "//TRIM(cd_namelist))
      ENDIF

   ELSE

      WRITE(cl_errormsg,*) " ERROR : can't find "//TRIM(cd_namelist)
      CALL fct_help(cp_myname,cl_errormsg) 
      CALL EXIT(1)

   ENDIF

   CALL multi_print(tl_multi)

   IF( tl_multi%i_nvar <= 0 )THEN
      CALL logger_fatal("CREATE BOUNDARY: no variable to be used."//&
      &  " check namelist.")
   ENDIF

   ! open files
   IF( TRIM(cn_coord0) /= '' )THEN
      tl_file= file_init(TRIM(cn_coord0))
      tl_coord0=mpp_init( tl_file, id_perio=in_perio0)
      ! clean
      CALL file_clean(tl_file)
      CALL grid_get_info(tl_coord0)
   ELSE
      CALL logger_fatal("CREATE BOUNDARY: can not find coarse grid "//&
      &  "coordinate file. check namelist")
   ENDIF

   IF( TRIM(cn_coord1) /= '' )THEN
      tl_file=file_init(TRIM(cn_coord1))
      tl_coord1=mpp_init( tl_file, id_perio=in_perio1)
      ! clean
      CALL file_clean(tl_file)
      CALL grid_get_info(tl_coord1)
   ELSE
      CALL logger_fatal("CREATE BOUNDARY: can not find fine grid coordinate "//&
      &  "file. check namelist")
   ENDIF

   IF( TRIM(cn_bathy1) /= '' )THEN
      tl_file=file_init(TRIM(cn_bathy1))
      tl_bathy1=mpp_init( tl_file, id_perio=in_perio1)
      ! clean
      CALL file_clean(tl_file)
      CALL grid_get_info(tl_bathy1)
   ELSE
      CALL logger_fatal("CREATE BOUNDARY: can not find fine grid bathymetry "//&
      &  "file. check namelist")
   ENDIF

   ! check
   ! check output file do not already exist
   ! WARNING: do not work when use time to create output file name
   DO jk=1,ip_ncard
      cl_bdyout=boundary_set_filename( TRIM(cn_fileout), &
      &                                TRIM(cp_card(jk)), 1 )
      INQUIRE(FILE=TRIM(cl_bdyout), EXIST=ll_exist)
      IF( ll_exist )THEN
         CALL logger_fatal("CREATE BOUNDARY: output file "//TRIM(cl_bdyout)//&
         &  " already exist.")
      ENDIF

      cl_bdyout=boundary_set_filename( TRIM(cn_fileout), &
         &                             TRIM(cp_card(jk)) )
      INQUIRE(FILE=TRIM(cl_bdyout), EXIST=ll_exist)
      IF( ll_exist )THEN
         CALL logger_fatal("CREATE BOUNDARY: output file "//TRIM(cl_bdyout)//&
         &  " already exist.")
      ENDIF
   ENDDO

   ! check namelist
   ! check refinement factor
   il_rho(:)=1
   IF( in_rhoi < 1 .OR. in_rhoj < 1 )THEN
      CALL logger_error("CREATE BOUNDARY: invalid refinement factor."//&
         &  " check namelist "//TRIM(cd_namelist))
   ELSE
      il_rho(jp_I)=in_rhoi
      il_rho(jp_J)=in_rhoj
   ENDIF

   !
   ! compute coarse grid indices around fine grid
   il_ind(:,:)=grid_get_coarse_index(tl_coord0, tl_coord1, &
      &                              id_rho=il_rho(:))

   il_imin0=il_ind(1,1) ; il_imax0=il_ind(1,2)
   il_jmin0=il_ind(2,1) ; il_jmax0=il_ind(2,2)

   ! check domain validity
   CALL grid_check_dom(tl_coord0, il_imin0, il_imax0, il_jmin0, il_jmax0)

   ! check coordinate file
   CALL grid_check_coincidence( tl_coord0, tl_coord1, &
      &                         il_imin0, il_imax0, &
      &                         il_jmin0, il_jmax0, &
      &                         il_rho(:) )      

   ! read or compute boundary
   CALL mpp_get_contour(tl_bathy1)

   CALL iom_mpp_open(tl_bathy1)
 
   tl_var1=iom_mpp_read_var(tl_bathy1,'Bathymetry')
 
   CALL iom_mpp_close(tl_bathy1)

   ! get boundaries indices
   tl_bdy(:)=boundary_init(tl_var1, ln_north, ln_south, ln_east, ln_west, &
      &                             cn_north, cn_south, cn_east, cn_west, &
      &                             ln_oneseg ) 


   CALL var_clean(tl_var1)

   ! compute level
   ALLOCATE(tl_level(ip_npoint))
   tl_level(:)=vgrid_get_level(tl_bathy1, cd_namelist )

   ! get coordinate for each segment of each boundary
   ALLOCATE( tl_segdom1(ip_npoint,ip_maxseg,ip_ncard) )
   ALLOCATE( tl_seglvl1(ip_npoint,ip_maxseg,ip_ncard) )
 
   DO jl=1,ip_ncard
      IF( tl_bdy(jl)%l_use )THEN
         DO jk=1,tl_bdy(jl)%i_nseg

            ! get fine grid segment domain
            tl_segdom1(:,jk,jl)=create_boundary_get_dom( tl_bathy1, &
               &                                         tl_bdy(jl), jk )

            IF( .NOT. ln_extrap )THEN
               ! get fine grid level
               tl_seglvl1(:,jk,jl)= &
                  & create_boundary_get_level( tl_level(:), &
                  &                            tl_segdom1(:,jk,jl))
            ENDIF

            ! add extra band to fine grid domain (if possible)
            ! to avoid dimension of one and so be able to compute offset
            DO jj=1,ip_npoint
               CALL dom_add_extra(tl_segdom1(jj,jk,jl), &
                  &               il_rho(jp_I), il_rho(jp_J))
            ENDDO

         ENDDO
      ENDIF
   ENDDO

   ! clean
   CALL var_clean(tl_level(:))
   DEALLOCATE(tl_level)

   ! clean bathy
   CALL mpp_clean(tl_bathy1)

   ALLOCATE( tl_segvar1(tl_multi%i_nvar,ip_maxseg,ip_ncard) )
   ! compute boundary for variable to be used (see namelist)
   IF( .NOT. ASSOCIATED(tl_multi%t_mpp) )THEN
      CALL logger_error("CREATE BOUNDARY: no file to work on. "//&
         &              "check cn_varfile in namelist.")
   ELSE

      jvar=0
      ! for each file
      DO ji=1,tl_multi%i_nmpp

         WRITE(cl_data,'(a,i2.2)') 'data-',jvar+1
         IF( .NOT. ASSOCIATED(tl_multi%t_mpp(ji)%t_proc(1)%t_var) )THEN

            CALL logger_error("CREATE BOUNDARY: no variable to work on for "//&
               &              "mpp "//TRIM(tl_multi%t_mpp(ji)%c_name)//&
               &              ". check cn_varfile in namelist.")

         ELSEIF( TRIM(tl_multi%t_mpp(ji)%c_name) == TRIM(cl_data) )THEN
         !- use input matrix to fill variable

            WRITE(*,'(a)') "work on data"
            ! for each variable initialise from matrix
            DO jj=1,tl_multi%t_mpp(ji)%t_proc(1)%i_nvar

               jvar=jvar+1
               WRITE(*,'(2x,a,a)') "work on variable "//&
               &  TRIM(tl_multi%t_mpp(ji)%t_proc(1)%t_var(jj)%c_name)

               tl_var1=var_copy(tl_multi%t_mpp(ji)%t_proc(1)%t_var(jj))

               SELECT CASE(TRIM(tl_var1%c_point))
               CASE DEFAULT !'T'
                  jpoint=jp_T
               CASE('U')
                  jpoint=jp_U
               CASE('V')
                  jpoint=jp_V
               CASE('F')
                  jpoint=jp_F
               END SELECT

               WRITE(*,'(4x,a,a)') 'work on '//TRIM(tl_var1%c_name)
               DO jl=1,ip_ncard
                  IF( tl_bdy(jl)%l_use )THEN

                     DO jk=1,tl_bdy(jl)%i_nseg

                        ! fill value with matrix data
                        tl_segvar1(jvar,jk,jl)=create_boundary_matrix( &
                           &                          tl_var1, &
                           &                          tl_segdom1(jpoint,jk,jl), &
                           &                          in_nlevel )

                        !del extra
                        CALL dom_del_extra( tl_segvar1(jvar,jk,jl), &
                           &                tl_segdom1(jpoint,jk,jl) )

                     ENDDO

                  ENDIF
               ENDDO

               ! clean
               CALL var_clean(tl_var1)

            ENDDO

         !- end of use input matrix to fill variable
         ELSE
         !- use mpp file to fill variable

            WRITE(*,'(a)') "work on file "//TRIM(tl_multi%t_mpp(ji)%c_name)
            ! 
            tl_file=file_init(TRIM(tl_multi%t_mpp(ji)%t_proc(1)%c_name), &
               &              id_perio=tl_multi%t_mpp(ji)%i_perio)
            tl_mpp=mpp_init( tl_file )
            !tl_mpp=mpp_init( tl_file, id_perio=tl_multi%t_mpp(ji)%t_proc(1)%i_perio)
            ! clean
            CALL file_clean(tl_file)
            CALL grid_get_info(tl_mpp)

            DO jl=1,ip_ncard
               IF( tl_bdy(jl)%l_use )THEN

                  WRITE(*,'(2x,a,a)') 'work on '//TRIM(tl_bdy(jl)%c_card)//&
                     &  ' boundary'
                  DO jk=1,tl_bdy(jl)%i_nseg

                     ! for each variable of this file
                     DO jj=1,tl_multi%t_mpp(ji)%t_proc(1)%i_nvar
 
                        WRITE(*,'(4x,a,a)') "work on variable "//&
                        &  TRIM(tl_multi%t_mpp(ji)%t_proc(1)%t_var(jj)%c_name)

                        tl_var0=var_copy(tl_multi%t_mpp(ji)%t_proc(1)%t_var(jj))

                        ! open mpp file
                        CALL iom_mpp_open(tl_mpp)

                        ! get or check depth value
                        CALL create_boundary_check_depth( tl_var0, tl_mpp, &
                        &                                 in_nlevel, tl_depth )

                        ! get or check time value
                        CALL create_boundary_check_time( tl_var0, tl_mpp, &
                        &                                tl_time )

                        ! close mpp file
                        CALL iom_mpp_close(tl_mpp)

                        ! open mpp file on domain
                        SELECT CASE(TRIM(tl_var0%c_point))
                           CASE DEFAULT !'T'
                              jpoint=jp_T
                           CASE('U')
                              jpoint=jp_U
                           CASE('V')
                              jpoint=jp_V
                           CASE('F')
                              jpoint=jp_F
                        END SELECT

                        tl_dom1=dom_copy(tl_segdom1(jpoint,jk,jl))

                        CALL create_boundary_get_coord( tl_coord1, tl_dom1, &
                        &                               tl_var0%c_point,    &
                        &                               tl_lon1, tl_lat1 )

                        ! get coarse grid indices of this segment
                        il_ind(:,:)=grid_get_coarse_index(tl_coord0, &
                        &                                 tl_lon1, tl_lat1, &
                        &                                 id_rho=il_rho(:) )

                        IF( ANY(il_ind(:,:)==0) )THEN
                           CALL logger_error("CREATE BOUNDARY: error "//&
                           &  "computing coarse grid indices")
                        ELSE
                           il_imin0=il_ind(1,1)
                           il_imax0=il_ind(1,2)

                           il_jmin0=il_ind(2,1)
                           il_jmax0=il_ind(2,2)
                        ENDIF

                        il_offset(:,:)= grid_get_fine_offset( &
                        &                    tl_coord0, &
                        &                    il_imin0, il_jmin0,&
                        &                    il_imax0, il_jmax0,&
                        &                    tl_lon1%d_value(:,:,1,1),&
                        &                    tl_lat1%d_value(:,:,1,1),&
                        &                    il_rho(:),&
                        &                    TRIM(tl_var0%c_point) )

                        ! compute coarse grid segment domain
                        tl_dom0=dom_init( tl_coord0,         &
                        &                 il_imin0, il_imax0,&
                        &                 il_jmin0, il_jmax0 )

                        ! add extra band (if possible) to compute interpolation
                        CALL dom_add_extra(tl_dom0)

                        ! open mpp files
                        CALL iom_dom_open(tl_mpp, tl_dom0)

                        cl_name=tl_var0%c_name
                        ! read variable value on domain
                        tl_segvar1(jvar+jj,jk,jl)= &
                        &    iom_dom_read_var(tl_mpp, TRIM(cl_name), tl_dom0)

                        IF( ANY(il_rho(:)/=1) )THEN
                           WRITE(*,'(4x,a,a)') "interp variable "//TRIM(cl_name)
                           ! work on variable
                           CALL create_boundary_interp( &
                           &                 tl_segvar1(jvar+jj,jk,jl),&
                           &                 il_rho(:), il_offset(:,:) )
                        ENDIF

                        ! remove extraband added to domain
                        CALL dom_del_extra( tl_segvar1(jvar+jj,jk,jl), &
                        &                   tl_dom0, il_rho(:) )

                        ! del extra point on fine grid
                        CALL dom_del_extra( tl_segvar1(jvar+jj,jk,jl), &
                        &                   tl_dom1 )

                        ! clean extra point information on coarse grid domain
                        CALL dom_clean_extra( tl_dom0 )

                        ! add attribute to variable
                        tl_att=att_init('src_file',&
                        &  TRIM(fct_basename(tl_mpp%c_name)))
                        CALL var_move_att(tl_segvar1(jvar+jj,jk,jl), &
                        &                 tl_att)

                        ! 
                        tl_att=att_init('src_i_indices',&
                        &  (/tl_dom0%i_imin, tl_dom0%i_imax/))
                        CALL var_move_att(tl_segvar1(jvar+jj,jk,jl), &
                        &                 tl_att)

                        tl_att=att_init('src_j_indices', &
                        &  (/tl_dom0%i_jmin, tl_dom0%i_jmax/))
                        CALL var_move_att(tl_segvar1(jvar+jj,jk,jl), &
                        &                 tl_att)

                        IF( ANY(il_rho(:)/=1) )THEN
                           tl_att=att_init("refinment_factor", &
                           &               (/il_rho(jp_I),il_rho(jp_J)/))
                           CALL var_move_att(tl_segvar1(jvar+jj,jk,jl), &
                           &                 tl_att)
                        ENDIF

                        ! clean structure
                        CALL att_clean(tl_att)

                        ! clean
                        CALL dom_clean(tl_dom0)
                        CALL dom_clean(tl_dom1)

                        ! close mpp files
                        CALL iom_dom_close(tl_mpp)

                        ! clean structure
                        CALL var_clean(tl_lon1)
                        CALL var_clean(tl_lat1)
                        CALL var_clean(tl_lvl1)

                     ENDDO ! jj

                     ! clean
                     CALL var_clean(tl_var0)

                  ENDDO ! jk
            
               ENDIF
            ENDDO ! jl

            jvar=jvar+tl_multi%t_mpp(ji)%t_proc(1)%i_nvar

            ! clean
            CALL mpp_clean(tl_mpp)

         !- end of use file to fill variable
         ENDIF
      ENDDO ! ji
   ENDIF

   IF( jvar /= tl_multi%i_nvar )THEN
      CALL logger_error("CREATE BOUNDARY: it seems some variable "//&
         &  "can not be read")
   ENDIF

   ! write file for each segment of each boundary
   DO jl=1,ip_ncard
      IF( tl_bdy(jl)%l_use )THEN

         DO jk=1,tl_bdy(jl)%i_nseg
            !- 
            CALL create_boundary_get_coord( tl_coord1, tl_segdom1(jp_T,jk,jl),&
               &                           'T', tl_lon1, tl_lat1 )

            ! force to use nav_lon, nav_lat as variable name
            tl_lon1%c_name='nav_lon'
            tl_lat1%c_name='nav_lat'

            ! del extra point on fine grid
            CALL dom_del_extra( tl_lon1, tl_segdom1(jp_T,jk,jl) )
            CALL dom_del_extra( tl_lat1, tl_segdom1(jp_T,jk,jl) )

            ! clean
            DO jpoint=1,ip_npoint
               CALL dom_clean(tl_segdom1(jpoint,jk,jl))
            ENDDO

            ! swap array
            CALL boundary_swap(tl_lon1, tl_bdy(jl))
            CALL boundary_swap(tl_lat1, tl_bdy(jl))
            DO jvar=1,tl_multi%i_nvar

               ! use additional request
               ! change unit and apply factor
               CALL var_chg_unit(tl_segvar1(jvar,jk,jl))

               ! forced min and max value
               CALL var_limit_value(tl_segvar1(jvar,jk,jl))

               ! filter
               CALL filter_fill_value(tl_segvar1(jvar,jk,jl))

               IF( .NOT. ln_extrap )THEN
                  ! use mask
                  SELECT CASE(TRIM(tl_segvar1(jvar,jk,jl)%c_point))
                  CASE DEFAULT !'T'
                     jpoint=jp_T
                  CASE('U')
                     jpoint=jp_U
                  CASE('V')
                     jpoint=jp_V
                  CASE('F')
                     jpoint=jp_F
                  END SELECT

                  CALL create_boundary_use_mask(tl_segvar1(jvar,jk,jl), &
                  &                             tl_seglvl1(jpoint,jk,jl))
               ENDIF

               ! swap dimension order
               CALL boundary_swap(tl_segvar1(jvar,jk,jl), tl_bdy(jl))

            ENDDO

            ! create file
            ! create file structure
            ! set file namearray of level variable structure
            IF( tl_bdy(jl)%i_nseg > 1 )THEN
               IF( ASSOCIATED(tl_time%d_value) )THEN
                  cl_fmt="('y',i0.4,'m',i0.2,'d',i0.2)"
                  tl_date=var_to_date(tl_time)
                  tl_date=tl_date+dn_dayofs
                  cl_date=date_print( tl_date, cl_fmt ) 

                  cl_bdyout=boundary_set_filename( TRIM(cn_fileout), &
                  &                                TRIM(tl_bdy(jl)%c_card), jk,&
                  &                                cd_date=TRIM(cl_date) )
               ELSE
                  cl_bdyout=boundary_set_filename( TRIM(cn_fileout), &
                  &                                TRIM(tl_bdy(jl)%c_card), jk )
               ENDIF
            ELSE
               IF( ASSOCIATED(tl_time%d_value) )THEN
                  cl_fmt="('y',i0.4,'m',i0.2,'d',i0.2)"
                  tl_date=var_to_date(tl_time)
                  tl_date=tl_date+dn_dayofs
                  cl_date=date_print( tl_date, cl_fmt )

                  cl_bdyout=boundary_set_filename( TRIM(cn_fileout), &
                  &                                TRIM(tl_bdy(jl)%c_card), &
                  &                                cd_date=TRIM(cl_date) )
               ELSE
                  cl_bdyout=boundary_set_filename( TRIM(cn_fileout), &
                  &                                TRIM(tl_bdy(jl)%c_card) )
               ENDIF
            ENDIF
            ! 
            tl_fileout=file_init(TRIM(cl_bdyout),id_perio=in_perio1)

            ! add dimension
            tl_dim(:)=var_max_dim(tl_segvar1(:,jk,jl))

            SELECT CASE(TRIM(tl_bdy(jl)%c_card))
               CASE DEFAULT ! 'north','south'
                  cl_dimorder='xyzt'
               CASE('east','west')
                  cl_dimorder='yxzt'
            END SELECT

            DO ji=1,ip_maxdim
               IF( tl_dim(ji)%l_use ) CALL file_add_dim(tl_fileout, tl_dim(ji))
            ENDDO

            ! add variables
            IF( ALL( tl_dim(1:2)%l_use ) )THEN
               ! add longitude
               CALL file_add_var(tl_fileout, tl_lon1)
               CALL var_clean(tl_lon1)

               ! add latitude
               CALL file_add_var(tl_fileout, tl_lat1)
               CALL var_clean(tl_lat1)
            ENDIF
            


            IF( tl_dim(3)%l_use )THEN
               IF( ASSOCIATED(tl_depth%d_value) )THEN
                  ! add depth
                  CALL file_add_var(tl_fileout, tl_depth)
               ENDIF
            ENDIF

            IF( tl_dim(4)%l_use )THEN
               IF( ASSOCIATED(tl_time%d_value) )THEN
                  ! add time
                  CALL file_add_var(tl_fileout, tl_time)
               ENDIF
            ENDIF

            ! add other variable
            DO jvar=tl_multi%i_nvar,1,-1
               CALL file_add_var(tl_fileout, tl_segvar1(jvar,jk,jl))
               CALL var_clean(tl_segvar1(jvar,jk,jl))
            ENDDO

            ! add some attribute
            tl_att=att_init("Created_by","SIREN create_boundary")
            CALL file_add_att(tl_fileout, tl_att)

            !add source url
            cl_url=fct_split(fct_split(cp_url,2,'$'),2,'URL:')
            tl_att=att_init("SIREN_url",cl_url)
            CALL file_add_att(tl_fileout, tl_att)

            ! add date of creation
            cl_date=date_print(date_now())
            tl_att=att_init("Creation_date",cl_date)
            CALL file_add_att(tl_fileout, tl_att)

            ! add shift on north and east boundary
            ! boundary compute on T point but express on U or V point
            SELECT CASE(TRIM(tl_bdy(jl)%c_card))
            CASE DEFAULT ! 'south','west'
               il_shift=0
            CASE('north','east')
               il_shift=1
            END SELECT

            ! add indice of velocity row or column
            tl_att=att_init('bdy_ind',tl_bdy(jl)%t_seg(jk)%i_index-il_shift)
            CALL file_move_att(tl_fileout, tl_att)

            ! add width of the relaxation zone
            tl_att=att_init('bdy_width',tl_bdy(jl)%t_seg(jk)%i_width)
            CALL file_move_att(tl_fileout, tl_att)
            
            ! add indice of segment start 
            tl_att=att_init('bdy_deb',tl_bdy(jl)%t_seg(jk)%i_first)
            CALL file_move_att(tl_fileout, tl_att)
            
            ! add indice of segment end 
            tl_att=att_init('bdy_end',tl_bdy(jl)%t_seg(jk)%i_last)
            CALL file_move_att(tl_fileout, tl_att)
                           
            ! clean
            CALL att_clean(tl_att)

            ! create file
            CALL iom_create(tl_fileout)

            ! write file
            CALL iom_write_file(tl_fileout, cl_dimorder)

            ! close file
            CALL iom_close(tl_fileout)
            CALL file_clean(tl_fileout)

         ENDDO ! jk

      ENDIF
      ! clean
      CALL boundary_clean(tl_bdy(jl))
   ENDDO !jl

   ! clean
   IF( ASSOCIATED(tl_depth%d_value) ) CALL var_clean(tl_depth)
   IF( ASSOCIATED(tl_time%d_value) ) CALL var_clean(tl_time)
   DEALLOCATE( tl_segdom1 )
   DEALLOCATE( tl_segvar1 )
   CALL var_clean(tl_seglvl1(:,:,:))
   DEALLOCATE( tl_seglvl1 )


   CALL mpp_clean(tl_coord1)
   CALL mpp_clean(tl_coord0)
   CALL var_clean_extra()

   CALL multi_clean(tl_multi)

   ! close log file
   CALL logger_footer()
   CALL logger_close()
   CALL logger_clean()

   END SUBROUTINE create_boundary__mono
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION create_boundary_get_dom(td_bathy1, td_bdy, id_seg) &
         & RESULT (tf_dom)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine compute boundary domain for each grid point (T,U,V,F) 
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014
   !> - take into account grid point to compute boundary indices
   !>
   !> @param[in] td_bathy1 file structure 
   !> @param[in] td_bdy    boundary structure
   !> @param[in] id_seg    segment indice 
   !> @return array of domain structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP) , INTENT(IN   ) :: td_bathy1
      TYPE(TBDY) , INTENT(IN   ) :: td_bdy
      INTEGER(i4), INTENT(IN   ) :: id_seg

      ! function
      TYPE(TDOM), DIMENSION(ip_npoint) :: tf_dom

      ! local variable
      INTEGER(i4) :: il_imin1
      INTEGER(i4) :: il_imax1
      INTEGER(i4) :: il_jmin1
      INTEGER(i4) :: il_jmax1

      INTEGER(i4) :: il_imin
      INTEGER(i4) :: il_imax
      INTEGER(i4) :: il_jmin
      INTEGER(i4) :: il_jmax

      INTEGER(i4), DIMENSION(ip_npoint) :: il_ishift
      INTEGER(i4), DIMENSION(ip_npoint) :: il_jshift

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jk
      !----------------------------------------------------------------
      ! init
      jk=id_seg

      il_ishift(:)=0
      il_jshift(:)=0

      ! get boundary definition
      SELECT CASE(TRIM(td_bdy%c_card))
         CASE('north')

            il_imin1=td_bdy%t_seg(jk)%i_first
            il_imax1=td_bdy%t_seg(jk)%i_last 
            il_jmin1=td_bdy%t_seg(jk)%i_index-(td_bdy%t_seg(jk)%i_width-1)
            il_jmax1=td_bdy%t_seg(jk)%i_index

            il_jshift(jp_V)=-1
            il_jshift(jp_F)=-1

         CASE('south')

            il_imin1=td_bdy%t_seg(jk)%i_first
            il_imax1=td_bdy%t_seg(jk)%i_last 
            il_jmin1=td_bdy%t_seg(jk)%i_index
            il_jmax1=td_bdy%t_seg(jk)%i_index+(td_bdy%t_seg(jk)%i_width-1)

         CASE('east')

            il_imin1=td_bdy%t_seg(jk)%i_index-(td_bdy%t_seg(jk)%i_width-1)
            il_imax1=td_bdy%t_seg(jk)%i_index
            il_jmin1=td_bdy%t_seg(jk)%i_first
            il_jmax1=td_bdy%t_seg(jk)%i_last 

            il_ishift(jp_U)=-1
            il_ishift(jp_F)=-1

         CASE('west')

            il_imin1=td_bdy%t_seg(jk)%i_index
            il_imax1=td_bdy%t_seg(jk)%i_index+(td_bdy%t_seg(jk)%i_width-1)
            il_jmin1=td_bdy%t_seg(jk)%i_first
            il_jmax1=td_bdy%t_seg(jk)%i_last 

      END SELECT         

      !-read fine grid domain
      DO ji=1,ip_npoint

         ! shift domain
         il_imin=il_imin1+il_ishift(ji)
         il_imax=il_imax1+il_ishift(ji)

         il_jmin=il_jmin1+il_jshift(ji)
         il_jmax=il_jmax1+il_jshift(ji)

         ! compute domain
         tf_dom(ji)=dom_init(td_bathy1,         &
            &                il_imin, il_imax,  &
            &                il_jmin, il_jmax,  &
            &                TRIM(td_bdy%c_card))

      ENDDO

   END FUNCTION create_boundary_get_dom
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE create_boundary_get_coord(td_coord1, td_dom1, cd_point, &
      &                                 td_lon1, td_lat1)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine get coordinates over boundary domain
   !> 
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !> @date September, 2014 
   !> - take into account grid point
   !>
   !> @param[in] td_coord1 coordinates file structure
   !> @param[in] td_dom1   boundary domain structure
   !> @param[in] cd_point  grid point
   !> @param[out] td_lon1  longitude variable structure
   !> @param[out] td_lat1  latitude variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TMPP)      , INTENT(IN   ) :: td_coord1
      TYPE(TDOM)      , INTENT(IN   ) :: td_dom1
      CHARACTER(LEN=*), INTENT(IN   ) :: cd_point
      TYPE(TVAR)      , INTENT(  OUT) :: td_lon1
      TYPE(TVAR)      , INTENT(  OUT) :: td_lat1 

      ! local variable
      TYPE(TMPP)        :: tl_coord1
      
      CHARACTER(LEN=lc) :: cl_name
      ! loop indices
      !----------------------------------------------------------------
      !read variables on domain (ugly way to do it, have to work on it)
      ! init mpp structure
      tl_coord1=mpp_copy(td_coord1)
      
      ! open mpp files
      CALL iom_dom_open(tl_coord1, td_dom1)

      ! read variable value on domain
      WRITE(cl_name,*) 'longitude_'//TRIM(cd_point)
      td_lon1=iom_dom_read_var( tl_coord1, TRIM(cl_name), td_dom1)
      WRITE(cl_name,*) 'latitude_'//TRIM(cd_point)
      td_lat1=iom_dom_read_var( tl_coord1, TRIM(cl_name), td_dom1)

      ! close mpp files
      CALL iom_dom_close(tl_coord1)

      ! clean structure
      CALL mpp_clean(tl_coord1)

   END SUBROUTINE create_boundary_get_coord
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE create_boundary_interp(td_var, id_rho, id_offset, &
         &                           id_iext, id_jext)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine interpolate variable on boundary
   !> 
   !> @details 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable structure 
   !> @param[in] id_rho    array of refinment factor
   !> @param[in] id_offset array of offset between fine and coarse grid
   !> @param[in] id_iext   i-direction size of extra bands (default=im_minext)
   !> @param[in] id_jext   j-direction size of extra bands (default=im_minext)
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) ,                 INTENT(INOUT) :: td_var
      INTEGER(I4), DIMENSION(:)  , INTENT(IN   ) :: id_rho
      INTEGER(i4), DIMENSION(:,:), INTENT(IN   ) :: id_offset

      INTEGER(i4),                 INTENT(IN   ), OPTIONAL :: id_iext
      INTEGER(i4),                 INTENT(IN   ), OPTIONAL :: id_jext


      ! local variable
      INTEGER(i4) :: il_iext
      INTEGER(i4) :: il_jext
      ! loop indices
      !----------------------------------------------------------------

      !WARNING: at least two extrabands are required for cubic interpolation
      il_iext=2
      IF( PRESENT(id_iext) ) il_iext=id_iext

      il_jext=2
      IF( PRESENT(id_jext) ) il_jext=id_jext

      IF( il_iext < 2 .AND. td_var%c_interp(1) == 'cubic' )THEN
         CALL logger_warn("CREATE BOUNDARY INTERP: at least extrapolation "//&
         &  "on two points are required with cubic interpolation ")
         il_iext=2
      ENDIF

      IF( il_jext < 2 .AND. td_var%c_interp(1) == 'cubic' )THEN
         CALL logger_warn("CREATE BOUNDARY INTERP: at least extrapolation "//&
         &  "on two points are required with cubic interpolation ")
         il_jext=2
      ENDIF

      ! work on variable
      ! add extraband
      CALL extrap_add_extrabands(td_var, il_iext, il_jext)

      ! extrapolate variable
      CALL extrap_fill_value( td_var )

      ! interpolate variable
      CALL interp_fill_value( td_var, id_rho(:), &
      &                       id_offset=id_offset(:,:) )

      ! remove extraband
      CALL extrap_del_extrabands(td_var, il_iext*id_rho(jp_I), &
         &                               il_jext*id_rho(jp_J))

   END SUBROUTINE create_boundary_interp
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION create_boundary_matrix(td_var, td_dom, id_nlevel) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief
   !> This function create variable, filled with matrix value
   !> 
   !> @details 
   !> A variable is create with the same name that the input variable, 
   !> and with dimension of the coordinate file. 
   !> Then the variable array of value is split into equal subdomain.
   !> Each subdomain is fill with the associated value of the matrix.
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_var    variable structure 
   !> @param[in] td_dom    domain structure 
   !> @param[in] id_nlevel number of levels 
   !> @return variable structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) , INTENT(IN) :: td_var
      TYPE(TDOM) , INTENT(IN) :: td_dom
      INTEGER(i4), INTENT(IN) :: id_nlevel

      ! function
      TYPE(TVAR)              :: tf_var

      ! local variable
      INTEGER(i4)      , DIMENSION(3)                    :: il_dim
      INTEGER(i4)      , DIMENSION(3)                    :: il_size
      INTEGER(i4)      , DIMENSION(3)                    :: il_rest

      INTEGER(i4)      , DIMENSION(:)      , ALLOCATABLE :: il_ishape
      INTEGER(i4)      , DIMENSION(:)      , ALLOCATABLE :: il_jshape
      INTEGER(i4)      , DIMENSION(:)      , ALLOCATABLE :: il_kshape

      REAL(dp)         , DIMENSION(:,:,:,:), ALLOCATABLE :: dl_value

      TYPE(TDIM)       , DIMENSION(ip_maxdim)            :: tl_dim

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jj
      INTEGER(i4) :: jk
      !----------------------------------------------------------------

      ! write value on grid
      ! get matrix dimension
      il_dim(:)=td_var%t_dim(1:3)%i_len

      tl_dim(jp_I:jp_J)=dim_copy(td_dom%t_dim(jp_I:jp_J))
      tl_dim(jp_K)%i_len=id_nlevel

      ! split output domain in N subdomain depending of matrix dimension 
      il_size(:) = tl_dim(1:3)%i_len / il_dim(:)
      il_rest(:) = MOD(tl_dim(1:3)%i_len, il_dim(:))

      ALLOCATE( il_ishape(il_dim(1)+1) )
      il_ishape(:)=0
      DO ji=2,il_dim(1)+1
         il_ishape(ji)=il_ishape(ji-1)+il_size(1)
      ENDDO
      ! add rest to last cell
      il_ishape(il_dim(1)+1)=il_ishape(il_dim(1)+1)+il_rest(1)
      
      ALLOCATE( il_jshape(il_dim(2)+1) )
      il_jshape(:)=0
      DO jj=2,il_dim(2)+1
         il_jshape(jj)=il_jshape(jj-1)+il_size(2)
      ENDDO
      ! add rest to last cell
      il_jshape(il_dim(2)+1)=il_jshape(il_dim(2)+1)+il_rest(2)

      ALLOCATE( il_kshape(il_dim(3)+1) )
      il_kshape(:)=0
      DO jk=2,il_dim(3)+1
         il_kshape(jk)=il_kshape(jk-1)+il_size(3)
      ENDDO
      ! add rest to last cell
      il_kshape(il_dim(3)+1)=il_kshape(il_dim(3)+1)+il_rest(3)

      ! write ouput array of value 
      ALLOCATE(dl_value( tl_dim(1)%i_len, &
      &                  tl_dim(2)%i_len, &
      &                  tl_dim(3)%i_len, &
      &                  tl_dim(4)%i_len) )

      dl_value(:,:,:,:)=0

      DO jk=2,il_dim(3)+1
         DO jj=2,il_dim(2)+1
            DO ji=2,il_dim(1)+1
               
               dl_value( 1+il_ishape(ji-1):il_ishape(ji), &
               &         1+il_jshape(jj-1):il_jshape(jj), &
               &         1+il_kshape(jk-1):il_kshape(jk), &
               &         1 ) = td_var%d_value(ji-1,jj-1,jk-1,1)

            ENDDO
         ENDDO
      ENDDO

      ! initialise variable with value
      tf_var=var_init(TRIM(td_var%c_name),dl_value(:,:,:,:))

      DEALLOCATE(dl_value)

   END FUNCTION create_boundary_matrix
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE create_boundary_use_mask(td_var, td_mask)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine use mask to filled land point with _FillValue
   !> 
   !> @details 
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[inout] td_var variable structure 
   !> @param[in] td_mask   mask variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), INTENT(INOUT) :: td_var
      TYPE(TVAR), INTENT(IN   ) :: td_mask

      ! local variable
      INTEGER(i4), DIMENSION(:,:), ALLOCATABLE :: il_mask

      ! loop indices
      INTEGER(i4) :: jk
      INTEGER(i4) :: jl
      !----------------------------------------------------------------

      IF( ANY(td_var%t_dim(1:2)%i_len /= &
      &       td_mask%t_dim(1:2)%i_len) )THEN
         CALL logger_debug("     mask dimension ( "//&
         &              TRIM(fct_str(td_mask%t_dim(1)%i_len))//","//&
         &              TRIM(fct_str(td_mask%t_dim(2)%i_len))//")" )
         CALL logger_debug(" variable dimension ( "//&
         &              TRIM(fct_str(td_var%t_dim(1)%i_len))//","//&
         &              TRIM(fct_str(td_var%t_dim(2)%i_len))//")" )
         CALL logger_fatal("CREATE BOUNDARY USE MASK: mask and "//&
         &                 "variable dimension differ."   )
      ENDIF

      ALLOCATE( il_mask(td_var%t_dim(1)%i_len, &
      &                 td_var%t_dim(2)%i_len) )

      il_mask(:,:)=INT(td_mask%d_value(:,:,1,1))

      DO jl=1,td_var%t_dim(4)%i_len
         DO jk=1,td_var%t_dim(3)%i_len
            WHERE( il_mask(:,:) < jk ) td_var%d_value(:,:,jk,jl)=td_var%d_fill
         ENDDO
      ENDDO

      DEALLOCATE( il_mask )

   END SUBROUTINE create_boundary_use_mask
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   FUNCTION create_boundary_get_level(td_level, td_dom) &
         & RESULT (tf_var)
   !-------------------------------------------------------------------
   !> @brief
   !> This function extract level over domain on each grid point, and return
   !> array of variable structure
   !>
   !> @author J.Paul
   !> @date November, 2013 - Initial Version
   !>
   !> @param[in] td_level  array of level variable structure
   !> @param[in] td_dom    array of domain structure
   !> @return array of variable structure
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), DIMENSION(:), INTENT(IN) :: td_level
      TYPE(TDOM), DIMENSION(:), INTENT(IN) :: td_dom

      ! function
      TYPE(TVAR), DIMENSION(ip_npoint)     :: tf_var

      ! local variable
      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------

      IF( SIZE(td_level(:)) /= ip_npoint .OR. &
      &   SIZE(td_dom(:)) /= ip_npoint )THEN
         CALL logger_error("CREATE BDY GET LEVEL: invalid dimension. "//&
         &  "check input array of level and domain.")
      ELSE

         DO ji=1,ip_npoint

            tf_var(ji)=var_copy(td_level(ji))

            IF( ASSOCIATED(tf_var(ji)%d_value) ) DEALLOCATE(tf_var(ji)%d_value)

            tf_var(ji)%t_dim(1)%i_len=td_dom(ji)%t_dim(1)%i_len
            tf_var(ji)%t_dim(2)%i_len=td_dom(ji)%t_dim(2)%i_len
            ALLOCATE(tf_var(ji)%d_value(tf_var(ji)%t_dim(1)%i_len, &
            &                           tf_var(ji)%t_dim(2)%i_len, &
            &                           tf_var(ji)%t_dim(3)%i_len, &
            &                           tf_var(ji)%t_dim(4)%i_len) )

            tf_var(ji)%d_value(:,:,:,:) = &
            &  td_level(ji)%d_value( td_dom(ji)%i_imin:td_dom(ji)%i_imax, &
            &                        td_dom(ji)%i_jmin:td_dom(ji)%i_jmax, :, : )

         ENDDO

      ENDIF

   END FUNCTION create_boundary_get_level
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE create_boundary_check_depth(td_var, td_mpp, id_nlevel, td_depth)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine check if variable need depth dimension, 
   !> get depth variable value in an open mpp structure
   !> and check if agree with already input depth variable.
   !> 
   !> @details 
   !>
   !> @author J.Paul
   !> @date November, 2014 - Initial Version
   !> @date January, 2016
   !> - check if variable need/use depth dimension
   !>
   !> @param[in] td_var       variable structure
   !> @param[in] td_mpp       mpp structure
   !> @param[in] id_nlevel    mpp structure
   !> @param[inout] td_depth  depth variable structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR) , INTENT(IN   ) :: td_var
      TYPE(TMPP) , INTENT(IN   ) :: td_mpp
      INTEGER(i4), INTENT(IN   ) :: id_nlevel
      TYPE(TVAR) , INTENT(INOUT) :: td_depth

      ! local variable
      INTEGER(i4) :: il_varid
      TYPE(TVAR)  :: tl_depth
      ! loop indices
      !----------------------------------------------------------------

      IF( td_var%t_dim(jp_K)%l_use .AND. &
      &   ( TRIM(td_var%c_axis) == '' .OR. &
      &     INDEX(TRIM(td_var%c_axis),'Z') /= 0 )&
      & )THEN

         ! check vertical dimension
         IF( td_mpp%t_dim(jp_K)%l_use )THEN
            IF( td_mpp%t_dim(jp_K)%i_len /= id_nlevel .AND. &
              & td_mpp%t_dim(jp_K)%i_len /= 1 )THEN
               CALL logger_error("CREATE BOUNDARY: dimension in file "//&
               &  TRIM(td_mpp%c_name)//" not agree with namelist in_nlevel ")
            ENDIF
         ENDIF

         ! get or check depth value
         IF( td_mpp%t_proc(1)%i_depthid /= 0 )THEN

            il_varid=td_mpp%t_proc(1)%i_depthid
            IF( ASSOCIATED(td_depth%d_value) )THEN

               tl_depth=iom_mpp_read_var(td_mpp, il_varid)

               IF( ANY( td_depth%d_value(:,:,:,:) /= &
               &        tl_depth%d_value(:,:,:,:) ) )THEN

                  CALL logger_error("CREATE BOUNDARY: depth value "//&
                  &  "for variable "//TRIM(td_var%c_name)//&
                  &  "from "//TRIM(td_mpp%c_name)//" not conform "//&
                  &  " to those from former file(s).")

               ENDIF
               CALL var_clean(tl_depth)

            ELSE
               td_depth=iom_mpp_read_var(td_mpp,il_varid)
            ENDIF

         ENDIF
      ELSE
         CALL logger_debug("CREATE BOUNDARY: no depth dimension use"//&
         &                 " for variable "//TRIM(td_var%c_name))
      ENDIF
      
   END SUBROUTINE create_boundary_check_depth
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE create_boundary_check_time(td_var, td_mpp, td_time)
   !-------------------------------------------------------------------
   !> @brief
   !> This subroutine check if variable need time dimension, 
   !> get date and time in an open mpp structure
   !> and check if agree with date and time already read.
   !> 
   !> @details 
   !>
   !> @author J.Paul
   !> @date November, 2014 - Initial Version
   !> @date January, 2016
   !> - check if variable need/use time dimension
   !>
   !> @param[in] td_var       variable structure
   !> @param[in] td_mpp      mpp structure
   !> @param[inout] td_time  time variable structure 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      TYPE(TVAR), INTENT(IN   ) :: td_var
      TYPE(TMPP), INTENT(IN   ) :: td_mpp
      TYPE(TVAR), INTENT(INOUT) :: td_time

      ! local variable
      INTEGER(i4) :: il_varid
      TYPE(TVAR)  :: tl_time

      TYPE(TDATE) :: tl_date1
      TYPE(TDATE) :: tl_date2
      ! loop indices
      !----------------------------------------------------------------
      IF( td_var%t_dim(jp_L)%l_use .AND. &
      &   ( TRIM(td_var%c_axis) == '' .OR. &
      &     INDEX(TRIM(td_var%c_axis),'T') /= 0 )&
      & )THEN

         ! get or check time value
         IF( td_mpp%t_proc(1)%i_timeid /= 0 )THEN

            il_varid=td_mpp%t_proc(1)%i_timeid
            IF( ASSOCIATED(td_time%d_value) )THEN

               tl_time=iom_mpp_read_var(td_mpp, il_varid)

               tl_date1=var_to_date(td_time)
               tl_date2=var_to_date(tl_time)
               IF( tl_date1 - tl_date2 /= 0 )THEN

                  CALL logger_warn("CREATE BOUNDARY: date from "//&
                  &  TRIM(td_mpp%c_name)//" not conform "//&
                  &  " to those from former file(s).")

               ENDIF
               CALL var_clean(tl_time)

            ELSE
               td_time=iom_mpp_read_var(td_mpp,il_varid)
            ENDIF

         ENDIF

      ELSE
         CALL logger_debug("CREATE BOUNDARY: no time dimension use"//&
         &                 " for variable "//TRIM(td_var%c_name))
      ENDIF

   END SUBROUTINE create_boundary_check_time
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END PROGRAM create_boundary
