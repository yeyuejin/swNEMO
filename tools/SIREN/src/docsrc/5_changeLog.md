# Change log

@tableofcontents

# Release 2019-12-03 {#rev_2019-12-03}
## New features
- M src/iom_cdf.f90 :
	- write netcdf file as netcdf4

# Release 2019-11-05 {#rev_2019-11-05}
## New features
- M src/function.f90
- M src/create_bathy.f90 :
	- add help and version optional arguments
	- update header for usage
- M src/create_boundary.F90 :
	- add help and version optional arguments
	- update header for usage
- M src/create_coord.f90 :
	- add help and version optional arguments
	- update header for usage
- M src/create_layout.f90 :
	- add help and version optional arguments
	- update header for usage
- M src/create_meshmask.f90 :
	- add help and version optional arguments
	- update header for usage
- M src/create_restart.f90 :
	- add help and version optional arguments
	- update header for usage
- M src/merge_bathy.f90 :
	- add help and version optional arguments
	- update header for usage
- M src/function.f90 :
	- add help and version functions
- M src/global.f90 :
	- add parameter for version, author, and date
	- set SVN keyword Revision, Author, and Date 

# Release 2019-10-18 {#rev_2019-10-18}
## New Features
- A templates/README
- A templates/create_templates.py :
	- script to create template of namelist from Siren sources
## Changes
- M templates/addline.nam :
	- update template of namelist (with default value)
- M templates/create_bathy.nam :
	- update template of namelist (with default value)
- M templates/create_boundary.nam :
	- update template of namelist (with default value)
- M templates/create_coord.nam :
	- update template of namelist (with default value)
- M templates/create_restart.nam :
	- update template of namelist (with default value)
- M templates/merge_bathy.nam :
	- update template of namelist (with default value)

# Release 2019-08-12 {#rev_2019-08-12}
## Changes
- src/create_bathy.f90 :
	- use periodicity read from namelist, and store in multi structure
- src/create_boundary.F90 :
	- use periodicity read from namelist, and store in multi structure
- src/create_restart.f90 :
	- use periodicity read from namelist, and store in multi structure
- src/mpp.f90 :
	- change print format
- src/multi.f90 :
	- use periodicity read from namelist, and store in multi structure
- templates/addline.nam :
	- rename sub namelist namcrs to namsrc
- templates/create_bathy.nam :
	- rename sub namelist namcrs to namsrc
	- rename sub namelist namfin to namtgt
- templates/create_boundary.nam :
	- rename sub namelist namcrs to namsrc
- rename sub namelist namfin to namtgt
- templates/create_coord.nam :
	- rename sub namelist namcrs to namsrc
- templates/create_restart.nam :
	- rename sub namelist namcrs to namsrc
	- rename sub namelist namfin to namtgt
## Bug fixes
- src/create_layout.f90 :
	- add missing variable in logger namelist
- src/multi.f90 :
	- rewrite function to subroutine
	- output filename string contains only filename (no more periodicity if given)

# Release 2019-05-21 {#rev_2019-05-21}
## New Features
- src/date.f90 :
	- add date_time subroutine
## Changes
- src/create_restart.f90 :
	- force number of proc to one by default
	- create and clean file structure to avoid memory leaks
- src/file.f90 :
	- add option to check dimension axis but not length
- src/grid.f90 :
	- do not use latitude variable to get pivot
	- read only grid boundaries to handle huge file
	- define as module variable im_max_overlap
- src/iom.f90 :
	- permit to write header and variable independantly
	- split iom_write_file into write_header and write_var
- src/iom_cdf.f90 :
	- permit to write header and variable independantly
	- split iom_cdf_write_file into write_header and write_var
	- add dimension structure as optional argument
	- do not check variable dimension if dimension forced
- src/iom_dom.f90 :
	- copy variable struct without array of value, then read array of value
- src/iom_mpp.f90 :
	- handle use of domain decomposition for monoproc file
- src/iom_rstdimg.f90 :
	- handle use of domain decomposition for monoproc file
	- split iom_rstdimg_write_file into write_header and write_var
- src/merge_bathy.f90 :
	- create and clean file structure to avoid memory leaks
- src/mpp.f90 :
	- cosmetic change
	- use mpp decomposition for writing netcdf
	- force to use domain decomposition to enhance read of monoproc file
	- add mpp__add_proc_arr
	- handle use of domain decomposition for monoproc file
- src/variable.f90 :
	- permit to copy variable structure without value
## Bug fixes
- src/math.f90 :
	- use the correct loop index to look for element bigger than pivot point
- src/multi.f90 :
	- compare each elt of cl_tabfile to cl_file

# Release 2019-05-15 {#rev_2019-05-15}
## New Features
- src/addline_deg.f9 :
	- new program to add line to all variable of the input file
- src/create_layout.f90 :
	- new program to create/compute the domain layout
- src/create_meshmask.f90 :
	- new program to create meshmask or domain.cfg
- src/grid_hgr.f90 :
	- new module needed to create meshmask
- src/grid_zgr.f90 :
	- new module needed to create meshmask
- src/lbc.f90 :
	- new module needed to create meshmask
- src/create_bathy.f90 :
	- optionaly, add random value to a flat bathymetry
- src/global.f90 :
	- add svn keyword properties
- src/math.f90 :
	- add function math_ortho
	- add function math_euclid
- src/multi.f90 :
	- add function multi__get_perio
- src/variable.f90 :
	- add function var_chg_name to rename variable
	- add output name to variable structure
## Changes
- src/create_bathy.f90 :
	- add url path to global attributes of output file(s)
	- create and clean file structure to avoid memory leaks
	- check dimension of matrix for 'handmade' bathymetry
	- check name and standard name for longitude and latitude 
	- rewrite header (doc)
	- rename sub namelist namcrs to namsrc
	- rename sub namelist namfin to namtgt
	- change NEMO path
- src/create_boundary.F90 :
	- add url path to global attributes of output file(s)
	- explain how to fill Lateral Boundary Condition in NEMO namelist
	- create and clean file structure to avoid memory leaks
	- rewrite header (doc)
	- rename sub namelist namcrs to namsrc
	- rename sub namelist namfin to namtgt
	- change NEMO path
- src/create_coord.f90 :
	- add url path to global attributes of output file(s)
	- create and clean file structure to avoid memory leaks
	- rewrite header (doc)
	- rename sub namelist namcrs to namsrc
- src/create_restart.f90 :
	- add url path to global attributes of output file(s)
	- check name and standard name for longitude and latitude
	- rewrite header (doc)
	- rename sub namelist namcrs to namsrc
	- rename sub namelist namfin to namtgt
	- change NEMO path
- src/global.f90 :
	- define svn url
- src/merge_bathy.f90 :
	- add url path to global attributes of output file(s)
	- rewrite header (doc)
	- rename sub namelist namcrs to namsrc
	- rename sub namelist namfin to namtgt
	- change NEMO path
- src/create_meshmask.f90 :
	- change NEMO path
- src/mpp.f90 :
	- copy file periodicity to mpp structure
	- clean file, variable, and attributes structures
	- deallocate file structure whatever happens
	- do not split variable on domain decomposition, if only one procesor
	- nullify array in layout structure
	- nullify file structure inside mpp structure
- src/attribute.f90 :
	- nullify array inside attribute structure
- src/boundary.f90 :
	- nullify segment structure inside boundary structure
- src/date.f90 :
	- check time units CF convention, raise error if not
- src/dimension.f90 :
	- do not reshape array already order
- src/file.f90 :
	- clean variable, attributes, and dimension structures
	- nullify variable, attributes structures inside file structure
	- netcdf4 files identify as netcdf file
- src/function.f90 :
	- permit sign as first character
- src/grid.f90 :
	- do not use silicalim, or silicamax to get pivot point 
	- check name and standard name for latitude
- src/iom_cdf.f90 :
	- clean variable, attributes, and dimension structures
	- read array in netcdf file, level by level, and time step by time step
	- apply scale factor and offset, level by level, and time step by time step
	- check attribute array is allocated, before use it
- src/iom_rstdimg.f90 :
	- clean dimension structure
- src/multi.f90 :
	- create and clean file structure to avoid memory leaks
	- fill value read from array of variable structure
	- nullify mpp structure in multi file structure
	- deallocate mpp structure whatever happens
	- print periodicity
- src/phycst.f90 :
	- half reduce epsilon value
- src/variable.f90 :
	- use scalar instead of array, as transitional variable
	- nullify attributes structure inside variable strcuture
	- decompose array conversion on each dimension
	- decompose array copy on each dimension
	- deallocate attribute strucure whatever happens
	- clean attribute strucure
	- nullify array inside variable structure
	- write fill value on array, level by level
- src/docsrc/main.dox :
	- add create_meshmask to SIREN tools list
	- add create_layout to SIREN tools list
- src/docsrc/1_install.md :
	- change NEMO path
- src/docsrc/2_quickstart.md :
	- change link to SIREN inputs
	- add description of create_meshmask
	- add description of create_layout 
## Bug fixes
- src/function.f90 :
	- permit negative exposant
- src/iom_cdf.f90 :
	- use dimid to look for the index of the dimension, and not as dimension index
- src/mpp.f90 :
	- netcdf proc indices from zero to N-1
- src/variable.f90 :
	- check if tg_varextra is allocated before clean it
	- add case for units in hours
	- read number of element for each dummy array in configuration file
- src/attributes.f90 :
	- read and use number of element for each dummy array in configuration file
- src/multi.f90 :
	- specify format output
- src/dimension.f90 :
	- read number of element for each dimension allowed in configuration file
	- read number of element for each dummy array in configuration file

# Release 2016-11-16 {#rev_2016-11-16}
## New Features
- create_meshmask.f90 program to create meshmask from coordinates and bathymetry files.
- create_meshmask.f90 allows to write domain_cfg file.
- merge_bathy.f90:
	- allow to choose the number of boundary point with coarse grid value.
- dimension.f90:
	- dimension allowed read in configuration file.
- variable.f90:
	- allow to add scalar value.
- create_meshmask.f90:
	- choose vertical scale factors (e3.=dk[depth] or old definition).
## Changes
- create_coord.f90:
	- allow to define sub domain with coarse grid indices or coordinates.
- grid.f90:
	-grid__get_closest_str:
	- add function to get closest grid point using coarse grid coordinates strucutre.
- iom_cdf.f90:
	-iom_cdf__get_info:
	- define cdf4 as cdf.
- variable.f90:
	- add subroutine to clean global array of extra information, and define logical for variable to be used.
- create_coord.f90:
	- dimension to be used select from configuration file.
- create_bathy.f90:
	- dimension to be used select from configuration file.
- merge_bathy.f90:
	- dimension to be used select from configuration file.
- create_boundary.f90:
	- dimension to be used select from configuration file.
- create_restart.f90:
	- dimension to be used select from configuration file.
## Bug fixes
- boundary.f90:
	-boundary_check:
	- take into account that boundaries are compute on T point, but expressed on U,V point.
- grid.f90:
	-grid__get_closest_str:
	- use max of zero and east-west overlap instead of east-west overlap.
- mpp.f90:
	- compare index to td_lay number of proc instead of td_mpp (bug fix) .
- iom_cdf.f90 :
	- check if attribute cl_value is not bug (in read file)

# Release 2016-07-01 {#rev_2016-07-01}

## Changes
## New Features
## Bug fixes
- correct check of boundary indices

# Initial Release 2016-04-11 {#rev_2016-04-11}
	<HR>
	<b>
	- @ref index
	- @ref md_src_docsrc_1_install
	- @ref md_src_docsrc_2_quickstart
	- @ref md_src_docsrc_3_support_bug
	- @ref md_src_docsrc_4_codingRules
	- @ref todo
	</b>
