============
MPP_PREP
============

Description
===========
MPP_PREP proposes possible domain decompositions for a given 
bathymetric file, which is particularly intersting when
we want to eliminate land-only domain. 
All solution are proposed and written to output file.
The ratio between the effective number of computed 
point and the total number of points in the domain is 
given and is probably a major criteria for choosing a 
domain decomposition.

Tools mpp_optimiz_zoom_nc.exe as been tested on one eORCA12 and one eORCA025 configuration at trunk@10036

Tools mppopt_showproc_nc.exe has not been tested.

Method
======
Use mpp_init like code for setting up the decomposition
and evaluate the efficiency of the decomposition.

How to compile it
=================
MPP_PREP is compiled in the same manner as all the other tools. 
The compilation script is maketools and the option are very similar to makenemo. 

Here an example of how to compile MPP_PREP on the MetOffice XC40 HPC:

.. code-block:: console
                      
        $ ./maketools -n MPP_PREP -m XC40_METO

Usage
=====

the MPP_PREP executable is named mpp_optimiz_zoom_nc.exe. The input file needed are:

 * a netcdf file containing a variable with 0 on land and data > 0 on ocean (typically a bathymetry file)

 * a namelist to specify the number of vertical levels, netcdf input file, variable and dimension names (...).
   A namelist template is available in the file 'namelist'. Default namelist is set up for input file domain_cfg.nc (output of Domaincfg tool)
   and will find decomposition between 100 and 4000 processors.

.. code-block:: console

 $ ./mpp_optimiz_zoom_nc.exe -h
   usage : mpp_optimize [ -h ]  [-keep jpni jpnj] [ -o file out ] 
                [ -modulo val ] [-r ratio] [-minocean procs] -n namelist
       
      PURPOSE :
          This program is build to optimize the domain beakdown into
          subdomain for mpp computing.
          Once the grid size, and the land/sea mask is known, it looks
          for all the possibilities within a range of setting parameters
          and determine the optimal.
 
          Optimization is done with respect to the maximum number of
          sea processors and to the maximum numbers of procs (nn_procmax)
                 
          Optional optimization can be performed taking into account
          the maximum available processor memory rn_ppmcal. This is
          activated if ln_memchk is set true in the namelist
       
          Additional criteria can be given on the command line to reduce
          the amount of possible choices.
       
      ARGUMENTS :
          -n namelist : indicate the name of the namelist to use
       
      OPTIONS :
          -h : print this help message
          -keep jpni jpnj : print a file suitable for plotting,
                  corresponding to the given decomposition
          -o output file : give the name of the output file
                  default is processor.layout
          -modulo val : only retain decomposition whose total number
                  of util processors (sea) are a multiple of val
          -r ratio : only retain decomposition with a ratio computed/global
                  less or equal to the given ratio
          -minocean procs : only retain decomposition with a number of 
                  ocean procs greater of equal to procs
       
      REQUIRED FILES :
        A bathymetric file and an ad-hoc namelist are required.
        The file name of the bathymetry is specified in the namelist
       
      OUTPUT : 
        processor.layout : an ascii file with all found possibilities
       
      SEE ALSO :
        script screen.ksh helps a lot in the analysis of the output file.
       
 STOP  

Example
=======

Here is an example of usage of ./mpp_optimiz_zoom_nc.exe on the the eORCA025 bathymetry. We keep in the output only domain decomposition with a ratio (computed/global) lower than 1, using namelist_eORCA025 and output the list of domain decomposition in processor.layout_eORCA025

.. code-block:: console

 $ ./mpp_optimiz_zoom_nc.exe -r 1 -n namelist_eORCA025 -o processor.layout_eORCA025
 
  ocean/land file used is: domcfg_eORCA025.nc
  variable used to find ocean domain is: bottom_level
  Dimensions (jpi x jpj) are:  1442 x 1207
 
 Loop over all the decompositions (can take a while) ...
 
 STOP

The output for one specific decomposition contains this information:

.. code-block:: console

  iresti= 14  irestj= 9
 --> Total number of domains  1612

  jpni= 31  jpnj= 52
  jpi=  49  jpj=  26
  Number of ocean processors        1074
  Number of land processors         538
  Mean ocean coverage per domain    0.7542637596508307
  Minimum ocean coverage            7.849293761E-4
  Maximum ocean coverage            1.
  nb of proc with coverage         < 10 %  68
  nb of proc with coverage 10 < nb < 30 %  99
  nb of proc with coverage 30 < nb < 50 %  59
  Number of computed points         1368276
  Overhead of computed points       -372218
  % sup (computed / global)         0.786142349

Sorting phase
=============
The processor.layout can be very long and hard to exploit.
To sort out what is the best model decomposition for a specific application, there is a suggestion at the end of the processor.layout file. Otherwise you can use the python script find_layout.py to dig into it.

.. code-block:: console

 $ python2.7 find_layout.py                                                       
 usage: find_layout.py [-h] -f layout_file --rmax max_ratio --noce min/max_noce

Below an example to extract all decomposition with a ratio (computed/global) < 0.8 and a number of ocean domain between 300 and 350. All the decomposition fitting the criterions are listed. At the end, a summary of the one with the smallest ratio, the largest number of ocean domains and the smallest computed domain.

.. code-block:: console

 $ python2.7 find_layout.py -f processor.layout_eORCA025 --rmax 0.8 --noce 300 350
 Domain decomposition 0
 domain decomposition (jpni, jpnj) = (13, 32)
 number of ocean domain            = 300
 ratio computed/global             = 0.779089153
 domain size (jpi, jpj)            = (113, 40)
 ...
 Domain decomposition 76
 domain decomposition (jpni, jpnj) = (37, 13)
 number of ocean domain            = 350
 ratio computed/global             = 0.783254623
 domain size (jpi, jpj)            = (41, 95)
 
 =====================================================================
 
 Among the layouts fitting the constraint on : ratio (computed/global) < 0.8 and 300 <= number of ocean domain <= 350

  3 layouts are highlighted : 

 Domain decomposition SMALLEST RATIO
 domain decomposition (jpni, jpnj) = (24, 18)
 number of ocean domain            = 310
 ratio computed/global             = 0.761956096
 domain size (jpi, jpj)            = (62, 69)
 
 Domain decomposition LARGEST NUMBER OF OCEAN DOMAINS
 domain decomposition (jpni, jpnj) = (21, 23)
 number of ocean domain            = 350
 ratio computed/global             = 0.785265565
 domain size (jpi, jpj)            = (71, 55)
 
 Domain decomposition SMALLEST COMPUTED DOMAIN
 domain decomposition (jpni, jpnj) = (18, 27)
 number of ocean domain            = 350
 ratio computed/global             = 0.775009871
 domain size (jpi, jpj)            = (82, 47)
