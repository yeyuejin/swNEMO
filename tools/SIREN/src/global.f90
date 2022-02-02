!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief This module defines global variables and parameters.
!>
!> @author
!> J.paul
!>
!> @date November, 2013 - Initial Version
!> @date September, 2015
!> - define fill value for each variable type
!> @date January, 2019
!> - define svn URL variable
!> @date October, 2019
!> - define svn Revision, Date, and Author variable
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE global

   USE kind                         ! F90 kind parameter
   USE netcdf

   IMPLICIT NONE

   PUBLIC :: dp_fill         !< default fill value
   PUBLIC :: ip_nsep         !< number of separator listed
   PUBLIC :: ip_ncom         !< number of comment character listed
   PUBLIC :: cp_sep          !< list of separator
   PUBLIC :: cp_com          !< list of comment character

   PUBLIC :: ip_npoint       !< number of point on ARAKAWA C-grid 
   PUBLIC :: jp_T            !< indice for T-point on ARAKAWA C-grid
   PUBLIC :: jp_U            !< indice for U-point on ARAKAWA C-grid
   PUBLIC :: jp_V            !< indice for V-point on ARAKAWA C-grid
   PUBLIC :: jp_F            !< indice for F-point on ARAKAWA C-grid
   PUBLIC :: cp_grid_point   !< list of grid_point character

   PUBLIC :: ip_maxdim       !< maximum number of dimension to be used
   PUBLIC :: jp_I            !< indice for I-direction
   PUBLIC :: jp_J            !< indice for J-direction
   PUBLIC :: jp_K            !< indice for K-direction
   PUBLIC :: jp_L            !< indice for L-direction
   PUBLIC :: cp_dimorder     !< dimension order

   PUBLIC :: ip_maxvar       !< maximum number of variable
   PUBLIC :: ip_maxmtx       !< matrix variable maximum dimension
   PUBLIC :: ip_maxseg       !< maximum number of segment
   PUBLIC :: ip_ghost        !< number of ghost cell

   PUBLIC :: ip_ninterp      !< number of available interpolation method
   PUBLIC :: cp_interp_list  !< list of interpolation name
   
   PUBLIC :: ip_nextrap      !< number of available extrapolation method
   PUBLIC :: cp_extrap_list  !< list of extrapolation name
   
   PUBLIC :: ip_nfilter      !< number of available filter
   PUBLIC :: cp_filter_list  !< list of filter name

   PUBLIC :: ip_ncard        !< number of cardinal point
   PUBLIC :: cp_card         !< array of cardinal point
   PUBLIC :: jp_north        !< indice for north boundary
   PUBLIC :: jp_south        !< indice for south boundary
   PUBLIC :: jp_east         !< indice for east  boundary
   PUBLIC :: jp_west         !< indice for west  boundary

   PUBLIC :: ip_maxdimcfg    !< maximum dimension in configuration file
   PUBLIC :: ip_maxdumcfg    !< maximum dummy variable in configuration file
   PUBLIC :: cp_url          !< svn url

   ! NOTE_avoid_public_variables_if_possible

   INTEGER(i4)                             , PARAMETER :: ip_maxvar =200   !< maximum number of variable
   INTEGER(i4)                             , PARAMETER :: ip_maxmtx =50    !< matrix variable maximum dimension (cf create_bathy)
   INTEGER(i4)                             , PARAMETER :: ip_maxseg =10    !< maximum number of segment for each boundary

   INTEGER(i4)                             , PARAMETER :: ip_nsep=2        !< number of separator listed
   CHARACTER(1)     , DIMENSION(ip_nsep)   , PARAMETER :: cp_sep = (/'.','_'/) !< list of separator 

   INTEGER(i4)                             , PARAMETER :: ip_ncom=2        !< number of comment character listed
   CHARACTER(1)     , DIMENSION(ip_ncom)   , PARAMETER :: cp_com = (/'#','!'/) !< list of comment character 
    
   INTEGER(i4)                             , PARAMETER :: ip_ghost=1       !< number of ghost cell

   INTEGER(i4)                             , PARAMETER :: ip_ninterp=3
   CHARACTER(LEN=lc), DIMENSION(ip_ninterp), PARAMETER :: cp_interp_list = &
   &  (/ 'nearest', &
   &     'cubic  ', &
   &     'linear '  /)

   INTEGER(i4)                             , PARAMETER :: ip_nextrap=2
   CHARACTER(LEN=lc), DIMENSION(ip_nextrap), PARAMETER :: cp_extrap_list = &
   &  (/ 'dist_weight', &
   &     'min_error  ' /)

   INTEGER(i4)                             , PARAMETER :: ip_nfilter=5
   CHARACTER(LEN=lc), DIMENSION(ip_nfilter), PARAMETER :: cp_filter_list = &
   &  (/ 'butterworth', &
   &     'blackman   ', &
   &     'hamming    ', &
   &     'hann       ', &
   &     'gauss      '/)

   REAL(dp)                                , PARAMETER :: dp_fill_i1=NF90_FILL_BYTE   !< byte fill value
   REAL(dp)                                , PARAMETER :: dp_fill_i2=NF90_FILL_SHORT  !< short fill value
   REAL(dp)                                , PARAMETER :: dp_fill_i4=NF90_FILL_INT    !< INT fill value
   REAL(dp)                                , PARAMETER :: dp_fill_sp=NF90_FILL_FLOAT  !< real fill value
   REAL(dp)                                , PARAMETER :: dp_fill=NF90_FILL_DOUBLE    !< double fill value

   INTEGER(i4)                             , PARAMETER :: ip_npoint=4
   INTEGER(i4)                             , PARAMETER :: jp_T=1
   INTEGER(i4)                             , PARAMETER :: jp_U=2
   INTEGER(i4)                             , PARAMETER :: jp_V=3
   INTEGER(i4)                             , PARAMETER :: jp_F=4
   CHARACTER(LEN=1), DIMENSION(ip_npoint)  , PARAMETER :: cp_grid_point = &
   &  (/ 'T', 'U', 'V', 'F' /)


   INTEGER(i4)                             , PARAMETER :: ip_maxdimcfg=10 !< maximum dimension in configuration file
   INTEGER(i4)                             , PARAMETER :: ip_maxdim=4
   INTEGER(i4)                             , PARAMETER :: jp_I=1
   INTEGER(i4)                             , PARAMETER :: jp_J=2
   INTEGER(i4)                             , PARAMETER :: jp_K=3
   INTEGER(i4)                             , PARAMETER :: jp_L=4
   CHARACTER(LEN=ip_maxdim)                , PARAMETER :: cp_dimorder = 'xyzt' !< dimension order to output

   INTEGER(i4)                             , PARAMETER :: ip_ncard=4
   CHARACTER(LEN=lc), DIMENSION(ip_ncard)  , PARAMETER :: cp_card = &
   &  (/ 'north', &
   &     'south', &
   &     'east ', &
   &     'west ' /)

   INTEGER(i4)                             , PARAMETER :: jp_north=1
   INTEGER(i4)                             , PARAMETER :: jp_south=2
   INTEGER(i4)                             , PARAMETER :: jp_east =3
   INTEGER(i4)                             , PARAMETER :: jp_west =4

   INTEGER(i4)                             , PARAMETER :: ip_maxdumcfg = 10 !< maximum dummy variable, dimension, or attribute 
                                                                            !< in configuration file

   CHARACTER(LEN=lc)                       , PARAMETER :: cp_url="$URL: https://forge.ipsl.jussieu.fr/nemo/svn/utils/tools/SIREN/src/global.f90 $"    !< svn url
   CHARACTER(LEN=lc)                       , PARAMETER :: cp_version = "$Revision: 12080 $"
   CHARACTER(LEN=lc)                       , PARAMETER :: cp_author = "$Author: jpaul $"
   CHARACTER(LEN=lc)                       , PARAMETER :: cp_date = "$Date: 2019-12-06 17:30:14 +0800 (五, 06 12 2019) $"

END MODULE global

