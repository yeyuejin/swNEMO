# How To Use (Quick Start)

@tableofcontents

SIREN is a software to set up regional configuration with
 [NEMO](http://www.nemo-ocean.eu).<br/> 
Actually SIREN creates all the input files you need to run a NEMO regional configuration.<br/>
 
SIREN is composed of a set of Fortran programs :
<ul>
 <li>create_coord.f90 to create regional grid coordinates.</li>
 <li>create_bathy.f90 to create regional grid bathymetry.</li>
 <li>merge_bathy.f90 to merge regional grid bathymetry with wider grid bathymetry
 at boundaries. 
 @note the goal of this step is to avoid break in Bathymetry.
 This break may cause inconsistency between forcing fields  at boundary and regional fields.
 </li>
 <li>create_meshmask.f90 to create meshmask or domain_cfg file(s) which contain(s) all the ocean domain informations.</li>
 <li>create_layout.f90 to create/compute the domain layout of your configuration.</li>
 <li>create_restart.f90 to create initial state file from coarse grid restart
 or standard outputs.
 @note this program could also be used to refined other input fields from a wider
 configuations (as runoff, chlorophyll etc...)
 </li>
 <li>create_boundary.F90 to create boundaries conditions from wider configurations
 output fields.
 </li>
</ul>

@warning SIREN can not:
<ul>
<li>create global configuration.</li>
<li>create configuarion around or close to North pole.</li>
<li>change number of vertical level.</li>
</ul>

Here after we briefly describe how to use each programs,
and so how to create your own regional configuration.
@note As demonstrator for a first start a set of GLORYS files (global reanalysis on *ORCA025* grid), as well as examples of namelists are available [here](https://cloud.mercator-ocean.fr/public.php?service=files&t=acf44730538cdda0da548ffc5f99fb55).

<!-- ######################################################################  -->
# Create coordinates file # {#coord}

To create your own configuration, you first have to create a coordinates file on your domain of study.<br/>
SIREN allows you to create this coordinates file from a wider coordinates file.<br/>
The coordinates file created could simply be an extraction, or a refinment of
the wide grid.<br/>

To create this new cooridnates file, you have to run :
~~~~~~~~~~~~~~~~~~
./SIREN/create_coord.exe create_coord.nam
~~~~~~~~~~~~~~~~~~

Here after is an example of namelist for *create_coord.exe*.<br/>
In this example, you create a coordinates file named *coord_fine.nc*.<br/>
This new coordinates file is refined from an extraction of *coordinates_ORCA025.nc*. 
~~~~~~~~~~~
&namlog
/

&namcfg
   cn_varcfg = "PATH/NEMOGCM/TOOLS/SIREN/cfg/variable.cfg"
   cn_dimcfg = "PATH/NEMOGCM/TOOLS/SIREN/cfg/dimension.cfg"
   cn_dumcfg = "PATH/NEMOGCM/TOOLS/SIREN/cfg/dummy.cfg"
/

&namsrc
   cn_coord0 = "PATH/coordinates_ORCA025.nc"
   in_perio0 = 4
/

&namvar
/

&namnst
   in_imin0 = 1070
   in_imax0 = 1072
   in_jmin0 = 607
   in_jmax0 = 609

   in_rhoi = 2
   in_rhoj = 3
/

&namout
   cn_fileout = "PATH/coord_fine.nc"
/
~~~~~~~~~~~

@note you could define sub domain with coarse grid indices or with coordinates.

Let's get describe this namelist.<br/>
First we have the **namlog** sub-namelist. This sub-namelist set parameters of the log
file.<br/>
All the parameters of this sub-namelist have default value, so you could let it
empty, as done here.<br/> This will create a log file named *create_coord.log*

## namcfg
The **namcfg** sub-namelist defines where found SIREN configuration files.<br/>
- The variable configuration file defines standard name, default interpolation method,
axis,... to be used for some known variables.<br/>
Obviously, you could add other variables to those already list, in this file.
- The dimension configuration file defines dimensions allowed.
- The dummy configuration file defines useless dimension or variable. these dimension(s) or variable(s) will not be processed.

@note You could find the generic version of those configuration files in the directory *SIREN/cfg*.

## namsrc
The **namsrc** sub-namelist set parameters of the coarse/source
coordinates file name, path to find it, and the NEMO periodicity of the grid.<br/>

@note the NEMO periodicity could be choose between 0 to 6:
<dl>
<dt>in_perio=0</dt>
<dd>standard regional model</dd>
<dt>in_perio=1</dt>
<dd>east-west cyclic model</dd>
<dt>in_perio=2</dt>
<dd>model with symmetric boundary condition across the equator</dd>
<dt>in_perio=3</dt>
<dd>regional model with North fold boundary and T-point pivot</dd>
<dt>in_perio=4</dt>
<dd>global model with a T-point pivot.<br/>
example: ORCA2, ORCA025, ORCA12</dd>
<dt>in_perio=5</dt>
<dd>regional model with North fold boundary and F-point pivot</dd>
<dt>in_perio=6</dt>
<dd>global model with a F-point pivot<br/>
example: ORCA05</dd>
</dd>
</dl>
@sa For more information see @ref md_src_docsrc_6_perio

## namvar
The **namvar** sub-namelist lists variables to be used.<br/>
By default all the variables of the coarse/source coordinates file are used to create
the new coordinates file.<br/> 
The interpolation methods to be used are defined in the configuration variables file (see
above). So you do not need to fill this sub-namelist too.

## namnst
The **namnst** sub-namelist defines the subdomain to be used as well as refinment factor.<br/>

<ul>
<li> you could define sub domain with coarse grid indices</li> 

~~~~~~~~~~~
&namnst
   in_imin0 = 1070
   in_imax0 = 1072
   in_jmin0 = 607
   in_jmax0 = 609
/
~~~~~~~~~~~

<li>or with coordinates</li>

~~~~~~~~~~~
&namnst
   rn_lonmin0 = -97.9 
   rn_lonmax0 = -62.3 
   rn_latmin0 =   7.7
   rn_latmax0 =  30.8
/
~~~~~~~~~~~

<li>you can select area quite every where (excepted too close from the North
pole), and use the refinment factor you want.</li>

~~~~~~~~~~~
&namnst
   in_imin0 = 1070
   in_imax0 = 1072
   in_jmin0 = 607
   in_jmax0 = 609

   in_rhoi = 2
   in_rhoj = 3
/
~~~~~~~~~~~
@image html grid_zoom_60.png
<center>@image latex grid_zoom_40.png
</center>
<!-- @note ghost cells all around the grid are not shown here. -->

<li>you can select area crossing the east-west overlap of the global ORCA grid.</li>

~~~~~~~~~~~
&namnst          
   in_imin0 = 1402 
   in_imax0 = 62
   in_jmin0 = 490 
   in_jmax0 = 570 

   in_rhoi = 5
   in_rhoj = 5 
/                
~~~~~~~~~~~
@image html grid_glob_over_30.png
<center>@image latex grid_glob_over_20.png
</center>
<!-- @note in blue, the east-west overlap band of ORCA grid. -->

<li>you can select east-west cyclic area.</li>

~~~~~~~~~~~
&namnst
   in_imin0 = 0
   in_imax0 = 0
   in_jmin0 = 390
   in_jmax0 = 450

   in_rhoi = 1
   in_rhoj = 1
/
~~~~~~~~~~~
@image html grid_glob_band_30.png
<center>@image latex grid_glob_band_20.png
</center>

</ul>

Finally the **namout** sub-namelist defines the output file.<br/>

@note All the output files created by SIREN include information about NEMO
periodicity, as well as source file, indices and refinment used.

@sa For more information and options to create coordinates, see create_coord.f90

<!-- ######################################################################  -->
# Create bathymetry file # {#bathy}

Then you need a Bathymetry file.<br/>
SIREN allows you to create a Bathymetry extracted or refined from a wider
Bathymetry grid.<br/>

To create this new bathymetry, you have to run :
~~~~~~~~~~~~~~~~~~
./SIREN/create_bathy.exe create_bathy.nam
~~~~~~~~~~~~~~~~~~

Here after is an example of namelist for *create_bathy.exe*.<br/>
In this example, you create a bathymetry file named *bathy_fine.nc*.<br/>
This new bathymetry file is refined from an extraction of *bathy_meter_ORCA025.nc*.<br/>
Moreover a minimum value of 5m is imposed to the output Bathymetry.
~~~~~~~~~~~~~~~~~~
&namlog
/

&namcfg
   cn_varcfg = "PATH/NEMOGCM/TOOLS/SIREN/cfg/variable.cfg"
   cn_dimcfg = "PATH/NEMOGCM/TOOLS/SIREN/cfg/dimension.cfg"
/

&namsrc
   cn_coord0 = "PATH/coordinates_ORCA025.nc"
   in_perio0 = 4
/

&namtgt
   cn_coord1 = "PATH/coord_fine.nc"
/

&namvar
   cn_varfile = "Bathymetry:PATH/bathy_meter_ORCA025.nc"
   cn_varinfo = "Bathymetry: min=5"
/

&namnst
   in_rhoi = 2
   in_rhoj = 3
/

&namout
   cn_fileout = "PATH/bathy_fine.nc"      
/
~~~~~~~~~~~~~~~~~~

Let's get describe this namelist.<br/>

## namlog, namcfg
First as previously, we have the **namlog** and **namcfg** sub-namelist (see above for more
explanation).<br/>

## namsrc
Then the **namsrc** sub-namelist set parameters of the source/wide
coordinates file.<br/>
@note in all SIREN namelist: <br/>
**0** referred to the coarse/wide grid.<br/>
**1** referred to the fine grid.

## namtgt
In the same way, the **namtgt** sub-namelist  set parameters of the target/fine
coordinates file.<br/>
@note in this namelist example, there is no need to set the variable *in_perio1* to define the NEMO
periodicity of the fine grid. Indeed, if this variable is not inform, SIREN tries to read it 
in the global attributes of the file. So if you created the fine coordinates with SIREN, you do not have to
fill it. In other case, you should add it to the namelist.

## namvar
The **namvar** sub-namelist lists variables to be used:
<dl>
   <dt>cn_varfile</dt>
      <dd> defines the variable name ("Bathymetry" here) and the input file associated with.
            @warning The domain of the input Bathymetry have to be larger than the output domain.
         @note 
         <ul>
            <li>if the input file is at coarse grid resolution (same than *cn_coord0*), the ouptut Bathymetry will be refined on fine grid.</li>
            <li>if the input file is a wider bathymetry (already at fine grid resolution), the output Bathymetry will be extracted from this one.</li>
         </ul>
      </dd>
   <dt>cn_varinfo</dt>
      <dd> defines user's requests for a variable.
         @note Default interpolation method for the Bathymetry, is *cubic* interpolation.<br/>
         So you may want to specify a minimum value to avoid negative value, or to change interpolation method.<br/>
         example: <ul><li>cn_varinfo="Bathymetry: min=1"'</li>
                      <li>cn_varinfo="Bathymetry: int=linear"</li></ul>
      </dd>
</dl>

## namnst
The **namnst** sub-namelist defines the subdomain refinment factor.<br/>
Of course those refinment factors have to be convenient with refinment
from coarse grid *cn_coord0* to fine grid *cn_coord1*.
@note subdomain indices are automatically deduced from fine and coarse grid
coordinates.

## namout
Finally, this **namout** sub-namelist defines the output file.<br/>

@note All the output files create by SIREN include information about 
source file, indices, refinment and interpolation method used.

@sa For more information and options to create bathymetry, see
create_bathy.f90

<!-- ######################################################################  -->
# Merge bathymetry file # {#merge}

The Bathymetry you build, may differ from the wider one.<br/>
To avoid issue with boundaries forcing fields, you should merge fine and coarse Bathymetry on boundaries.<br/>
SIREN allows you to do this.<br/>

To create this merged bathymetry, you have to run :
~~~~~~~~~~~~~~~~~~
./SIREN/merge_bathy.exe merge_bathy.nam
~~~~~~~~~~~~~~~~~~

Here after is an example of namelist for *merge_bathy.exe*.<br/>
~~~~~~~~~~~~~~~~~~
&namlog
/

&namcfg
   cn_varcfg = "PATH/NEMOGCM/TOOLS/SIREN/cfg/variable.cfg"
   cn_dimcfg = "PATH/NEMOGCM/TOOLS/SIREN/cfg/dimension.cfg"
/

&namsrc
   cn_bathy0 = "PATH/bathy_meter_ORCA025.nc"
   in_perio0 = 4
/

&namtgt
   cn_bathy1 = "PATH/bathy_fine.nc"
/

&namnst
   in_rhoi = 3
   in_rhoj = 3
/

&nambdy
/

&namout
   cn_fileout = "PATH/bathy_merged.nc"      
/
~~~~~~~~~~~~~~~~~~

## namlog, namcfg
In this namelist, you find again the **namlog**, **namcfg** describe above.

## namsrc
Then the **namsrc** sub-namelist sets parameters of the source/wide grid.
However this time, this is the source/wide grid Bathymetry wich have to be informed.

## namtgt
The **namtgt** sub-namelist defines parameters of the target/fine grid Bathymetry.
@note here again you could add the *in_perio1* parameter if need be i.e. if your
fine grid Bathymetry was not created by SIREN.

## namnst
The **namnst** sub-namelist defines the subdomain refinment factor.

## nambdy
The **nambdy** sub-namelist defines the subdomain boundaries.<br/>
By default SIREN tries to create boundaries for each side. Boundary exist if there is at least one sea point on the second row of each side. So you could let this namelist empty.
@sa For more information about boundaries, see @ref boundary

## namout
Finally, this **namout** sub-namelist defines the output file.<br/>

@sa For more information and options to merge bathymetry, see
merge_bathy.f90

<!-- ######################################################################  -->
# Create meshmask (ocean domain informations) # {#meshmask}


Depending on the vertical grid you choose to use, NEMO may not see the bathymetry exactly as you defined it just before. To get the ocean domain informations as seen by NEMO, SIREN allows you to create the meshmask file(s) which contain(s) all those informations.<br/>
Morevoer SIREN allows you to create the *domain_cfg* file which is the new input file for NEMO (release 3.7 and upper).<br/>

To create the meshmask file(s), you have to run :
~~~~~~~~~~~~~~~~~~
./SIREN/create_meshmask.exe create_meshmask.nam
~~~~~~~~~~~~~~~~~~

Here after is an example of namelist for *create_meshmask.exe*.<br/>
In this example, you create one meshmask file named *mesh_mask.nc*.<br/>
It uses coordinates file *coord_fine.nc* to define horizontal grid.
and defines z-coordinate with partial steps.
The minimum depth of the final Bathymetry is 10m. 
~~~~~~~~~~~
&namlog
/

&namcfg
   cn_varcfg = "PATH/NEMOGCM/TOOLS/SIREN/cfg/variable.cfg"
   cn_dimcfg = "PATH/NEMOGCM/TOOLS/SIREN/cfg/dimension.cfg"
/

&namsrc
   cn_bathy = "PATH/bathy_merged.nc"
   cn_coord = "PATH/coord_fine.nc"
   in_perio = 4
/

&namhgr
   in_mshhgr = 0
/

&namzgr
   ln_zps   = .TRUE.
   in_nlevel= 75
/

&namdmin
   dn_hmin=10.
/

&namzco
   dn_ppsur   = -3958.951371276829
   dn_ppa0    =   103.953009600000
   dn_ppa1    =     2.415951269000
   dn_ppkth   =    15.351013700000
   dn_ppacr   =     7.000000000000
   dn_ppdzmin = 6.
   dn_pphmax  = 5750.
   ln_dbletanh= .TRUE.
   dn_ppa2    =   100.760928500000
   dn_ppkth2  =    48.029893720000
   dn_ppacr2  =    13.000000000000
/

&namzps
   dn_e3zps_min = 25.
   dn_e3zps_rat = 0.2
/

&namsco
/

&namlbc
/

&namwd
/

&namgrd
/

&namout
   in_msh = 1
/
~~~~~~~~~~~

Let's get describe this namelist more accurately.<br/>

## namlog, namcfg
As previously, we have the **namlog** and **namcfg** describe above.<br/>

## namsrc
The **namsrc** sub-namelist defines the Bathymetry to be used.
Mainly SIREN need Bathymetry to create meshmask.
Here we also read coordinates directly on a file. 
@note 
 1. here again you could add the *in_perio* parameter if need be i.e. if your
Bathymetry was not created by SIREN.
 2. by default SIREN suppress closed sea/lake from the ORCA domain.

## namhgr
The **namhgr** sub-namelist defines the horizontal grid.
the type of horizontal mesh is choose between :<ul>
<li> in_mshhgr=0 : curvilinear coordinate on the sphere read in coordinate.nc</li>
<li> in_mshhgr=1 : geographical mesh on the sphere with regular grid-spacing</li>
<li> in_mshhgr=2 : f-plane with regular grid-spacing</li>
<li> in_mshhgr=3 : beta-plane with regular grid-spacing</li>
<li> in_mshhgr=4 : Mercator grid with T/U point at the equator</li>
<li> in_mshhgr=5 : beta-plane with regular grid-spacing and rotated domain (GYRE configuration)</li>
</ul>

## namzgr
The **namzgr** sub-namelist allows to choose the type of vertical grid (z-coordinate full steps, partial steps, sigma or hybrid coordinates) and the number of level.

## namdmin
The **namdmin** sub-namelist defines the minimum ocean depth. It could be defines in meter (>0) or in number of level (<0).

## namzco
The **namzco** sub-namelist defines parameters to compute z-coordinate vertical grid (**needed for all type of vertical grid**)
<!-- By default, those parameters are defined the same way than in GLORYS (i.e. 75 vertical levels).<br/> -->

## namzps
The **namzps** sub-namelist defines extra parameters needed to define z-coordinates partial steps.

## namsco
The **namsco** sub-namelist defines extra parameters needed to define sigma or hybrid coordinates (not needed here).

<!--The **namcla** sub-namelist defines cross land advection for exchanges through some straits only used for ORCA2 (see namgrd).-->

## namlbc
The **namlbc** sub-namelist defines lateral boundary conditions at the coast. It is needed to modify the fmask.

## namwd
The **namwd** sub-namelist defines the wetting and drying parameters if activated (see namzgr sub-namelist)

## namgrd
The **namgrd** sub-namelist allows to use configuration 1D or to choose vertical scale factors (e3.=dk or old definition).

## namout
Finally, this **namout** sub-namelist defines the number output file(s).<br/>
@note To create the domain_cfg file, you should put **in_msh=0**.

@sa For more information about how to create meshmask, see create_meshmask.f90

<!-- ######################################################################  -->
# Create layout (domain layout) # {#layout}

To run faster your configuration you may need to run it on multiprocessor. To do so you first need to know on which domain layout and so on how many processor you could do it.
.<br/>

To create/compute the domain layout, you have to run :
~~~~~~~~~~~~~~~~~~
./SIREN/create_layout.exe create_layout.nam
~~~~~~~~~~~~~~~~~~

Here after is an example of namelist for *create_layout.exe*.<br/>
In this example, you compute the domain layout on 40 processors.<br/>
~~~~~~~~~~~
&namlog
/

&namcfg
   cn_varcfg = "PATH/NEMOGCM/TOOLS/SIREN/cfg/variable.cfg"
   cn_dimcfg = "PATH/NEMOGCM/TOOLS/SIREN/cfg/dimension.cfg"
/

&namvar
   cn_varfile = "Bathymetry:PATH/bathy_merged.nc"
/

&namout
   in_nproc = 40
/
~~~~~~~~~~~

Let's get describe this namelist more accurately.<br/>

## namlog, namcfg
As previously, we have the **namlog** and **namcfg** describe above.<br/>

## namvar
The **namvar** sub-namelist lists variables to be used.
Mainly SIREN need Bathymetry to compute the domain layout, or at least the mask of your domain.

## namout
The **namout** sub-namelist defines the number of processor you want to work on.

@sa For more information and options to create layout, see create_layout.f90

<!-- ######################################################################  -->
# Create initial state # {#restart}

To run your configuration you need an inital state of the ocean.<br/>
You could start from a restart file (with all NEMO variables fields at one
time step). Or you could start from "partial" information about ocean state (Temperature and Salinity for example).

SIREN allows you to create both of those initial state.<br/>
To create the initial state, you have to run:<br/>
~~~~~~~~~~~~~~~~~~
./SIREN/create_restart.exe create_restart.nam
~~~~~~~~~~~~~~~~~~

Here after is an example of namelist for *create_restart.exe*.<br/>
In this example, you create an initial state split on 81 "processors", and named restart_out.nc.<br/>
The initial state is composed of temperature and salinity refined from an extraction of GLORYS fields.
~~~~~~~~~~~~~~~~~~
&namlog
/

&namcfg
   cn_varcfg = "PATH/NEMOGCM/TOOLS/SIREN/cfg/variable.cfg"
   cn_dimcfg = "PATH/NEMOGCM/TOOLS/SIREN/cfg/dimension.cfg"
/

&namsrc
   cn_coord0 = "PATH/coordinates_ORCA025.nc"
   in_perio0 = 4
/

&namtgt
   cn_coord1 = "PATH/coord_fine.nc"
   cn_bathy1 = "PATH/bathy_merged.nc"
/

&namzgr
/

&namzps
/

&namvar
   cn_varfile = "votemper:GLORYS_gridT.nc",
                "vosaline:GLORYS_gridS.nc"
/

&namnst
   in_rhoi = 3
   in_rhoj = 3
/

&namout
   cn_fileout = "PATH/restart_out.nc"      
   in_nproc = 81
/
~~~~~~~~~~~~~~~~~~
Let's get describe this namelist more accurately.<br/>

## namlog, namcfg
As previously, we have the **namlog** and **namcfg** sub-namelists, as well as 

## namsrc
the **namsrc** sub-namelist to set parameters of the source/wide coordinates file (see above for more
explanation).<br/>

## namtgt
Then the **namtgt** sub-namelist set parameters of the target/fine
grid coordinates and bathymetry.<br/>

## namzgr, namzps
The **namzgr** and **namzps** sub-namelists define respectively parameters for vertical grid
and partial step.<br>
By default, those parameters are defined the same way than in GLORYS (i.e. 75 vertical levels).<br/>
So you could let it empty.
@note If you use forcing fields other than GLORYS, you have to be sure it uses the same vertical grid. In other case, you need to get information about the parametrisation use, and to put it in those sub-namelist (see create_restart.f90).

## namvar
the **namvar** sub-namelist lists variables to be used.<br/>
Here we use *votemper* (temperature) get from *GLORYS_gridT.nc* file, and *vosaline*
(salinity) get from *GLORYS_gridS.nc* file.
@note To get all variables of a restart file. You have to use:
~~~~~~~~~~~~~~~~~~
cn_varfile = "all:PATH/restart.dimg"
~~~~~~~~~~~~~~~~~~

## namnst
The **namnst** sub-namelist defines the subdomain refinment factor, as seen previously.<br/>

## namout
Finally, this **namout** sub-namelist defines the output files.<br/>
Here we ask for output on 81 processors, with *restart_out.nc* as file "basename".<br/>
So SIREN computes the optimal layout for 81 processors
available,<br/>
and split restart on output files named *restart_out_num.nc*, where *num* is the proc number.

@note SIREN could also create the other fields you may need for
your configuration.<br/>
To do so, you just have to run *create_restart.exe* with other variable(s) from other input file(s).<br/>
For example, to get runoff fields, you could use:
~~~~~~~~~~~~~~~~~~
cn_varfile = "sorunoff:PATH/runoff_GLORYS.nc"
...
cn_fileout = "PATH/runoff_out.nc"
~~~~~~~~~~~~~~~~~~

@sa For more information and options to create initial state or other fields, see
create_restart.f90

<!-- ######################################################################  -->
# Create boundaries conditions # {#boundary}

Finally to force your configuration, you may need boundaries conditions.<br/>
NEMO read physical boundaries conditions from temperature, salinity, currents, and sea
surface height.

To create the boundaries condition with SIREN, you have to run:<br/>
~~~~~~~~~~~~~~~~~~
./SIREN/create_boundary.exe create_boundary.nam
~~~~~~~~~~~~~~~~~~

Here after is an example of namelist for *create_boundary.exe*.<br/>
In this example, you create boundaries conditions named *boundary_out.nc* on
each side of the domain.<br/>
The boundaries contain information about temperature, salinity, currents and sea surface height refined from an extraction of GLORYS fields.
~~~~~~~~~~~~~~~~~~
&namlog
/

&namcfg
   cn_varcfg = "PATH/NEMOGCM/TOOLS/SIREN/cfg/variable.cfg"
   cn_dimcfg = "PATH/NEMOGCM/TOOLS/SIREN/cfg/dimension.cfg"
/

&namsrc
   cn_coord0 = "PATH/coordinates_ORCA025.nc"
   in_perio0 = 4
/

&namtgt
   cn_coord1 = "PATH/coord_fine.nc"
   cn_bathy1 = "PATH/bathy_fine.nc"
/

&namzgr
/

&namzps
/

&namvar
   cn_varfile="votemper:GLORYS_gridT.nc",
              "vosaline:GLORYS_gridS.nc",
              "vozocrtx:GLORYS_gridU.nc",
              "vomecrty:GLORYS_gridV.nc",
              "sossheig:GLORYS_grid2D.nc"
/

&namnst
   in_rhoi = 3
   in_rhoj = 3
/

&nambdy
/

&namout
   cn_fileout = "PATH/boundary_out.nc"      
/
~~~~~~~~~~~~~~~~~~

Let's get describe this namelist more accurately.<br/>

## namlog, namcfg
As previously, we have the **namlog** and **namcfg** sub-namelists, as well as 

## namsrc
the **namcrs** sub-namelist to set parameters of the source/wide coordinates file (see above for more
explanation).<br/>

## namtgt
Then the **namtgt** sub-namelist set parameters of the target/fine
grid coordinates and bathymetry.<br/>

## namzgr, namzps
The **namzgr** and **namzps** sub-namelists define respectively parameters for vertical grid
and partial step.<br>
By default, those parameters are defined the same way than in GLORYS (i.e. 75 vertical levels).<br/>
So you could let it empty.
@note If you use forcing fields other than GLORYS, you have to be sure it uses the same vertical grid. In other case, you need to get information about the parametrisation use, and to put it in those sub-namelist (see create_boundary.F90).

## namvar
the **namvar** sub-namelist lists variables to be used.<br/>
Here we get *votemper* (temperature) from *GLORYS_gridT.nc* file, *vosaline*
(salinity) from *GLORYS_gridS.nc* file, *vozocrtx* (zonal velocity) from
*GLORYS_gridU.nc*, *vomecrty* (meridional velocity) from *GLORYS_gridV.nc*, and sossheig (sea surface
height) from *GLORYS_grid2D.nc*.

## namnst
The **namnst** sub-namelist defines the subdomain refinment factor.<br/>

## nambdy
The **nambdy** sub-namelist defines the subdomain boundaries.<br/>
By default SIREN tries to create boundaries for each side (Boundary is created if sea point exist on the second row of each side).<br/>
So you could let this namelist empty.

@note SIREN allows you to place boundaries away from the side of the domain.
To do so you have to define your boundary.<br/>
That means you have to give on fine
grid the index of the boundary (how far from the border your boundary is),
the width of your boundary, and finally first and last point of your boundary (the length of your boundary).<br/>
So to define a north boundary, you have to add in the sub-namelist *nambdy*, the parameter:
~~~~~~~~~~~~~~~~~~
cn_north="index,first:last(width)"
~~~~~~~~~~~~~~~~~~

## namout
Finally, this **namout** sub-namelist defines the output files.<br/>
Here we ask for output with *boundary_out.nc* as file "basename".<br/>
So SIREN creates output files named *boundary_out_west.nc*,
*boundary_out_east.nc*, *boundary_out_north.nc*, and *boundary_out_south.nc*
depending if boundary exist or not.

@sa For more information and options to create boundaries condition, see
create_boundary.F90

# Create and run NEMO configuration # {#NEMOconf}

So now you created all the input files you need for your physical configuration, you have to create the "NEMO configuration".<br/>
To do so, go to the directory *NEMOGCM/CONFIG/*, and run:
~~~~~~~~~~~~~~~~~~
./makenemo -n MY_CONFIG -d "OPA_SRC"
~~~~~~~~~~~~~~~~~~
This creates your configuration "MY_CONFIG" in the directory *NEMOGCM/CONFIG*.<br/>
you could check the cpp_keys used in file *cpp_MY_CONFIG.fcm*, and re-run *makenemo*
if need be.

Once *makenemo* has run successfully, the *opa* executable is available in
directory *NEMOGCM/CONFIG/MY_CONFIG/EXP00*.<br/>
Then you just have to put all your input files in this directory, fill the namelist *namelist_cfg*, and run:
~~~~~~~~~~~~~~~~~~
mpirun ./opa
~~~~~~~~~~~~~~~~~~

@note no surface forcing here.
weighted function needed to do interpolation on the fly, could be created by WEIGHT tools already inside NEMO.

@sa For more information about how to create NEMO configuration see [NEMO Quick Start Guide](http://www.nemo-ocean.eu/Using-NEMO/User-Guides/Basics/NEMO-Quick-Start-Guide).

 <HR>
   <b>
   - @ref index
   - @ref md_src_docsrc_1_install
   - @ref md_src_docsrc_2_quickstart
   - @ref md_src_docsrc_3_support_bug
   - @ref md_src_docsrc_4_codingRules
   - @ref md_src_docsrc_5_changeLog
   - @ref todo
   </b>
