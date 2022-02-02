============
REBUILD_NEMO
============
REBUILD_NEMO is a tool to rebuild NEMO output files from multiple processors (mesh_mask, restart or XIOS output files) into one file.

Description
===========

NEMO rebuild has the following features:
 * dynamically works out what variables require rebuilding
 * does not copy subdomain halo regions
 * works for 1,2,3 and 4d arrays or types for all valid NetCDF types
 * utilises OMP shared memory parallelisation where applicable
 * time 'slicing' for lower memory use  (only for 4D vars with unlimited dimension)

The code reads the filestem and number of subdomains from the namelist file nam_rebuild.
| The 1st subdomain file is used to determine the dimensions and variables in all the input files. 
It is also used to find which dimensions (and hence which variables) require rebuilding 
as well as information about the global domain.
| It then opens all the input files (unbuffered) and creates an array of netcdf identifiers
before looping through all the variables and updating the rebuilt output file (either by direct 
copying or looping over the number of domains and rebuilding as appropriate).
| The code looks more complicated than it is because it has lots of case statements to deal with all 
the various NetCDF data types and with various data dimensions (up to 4d).
| Diagnostic output is written to numout (default 6 - stdout)
and errors are written to numerr (default 0 - stderr).
| If time slicing is specified the code will use less memory but take a little longer.
It does this by breaking down the 4D input variables over their 4th dimension 
(generally time) by way of a while loop.

How to compile it
=================
REBUILD_NEMO is compiled in the same manner as all the other tools. 
The compilation script is maketools and the option are very similar to makenemo. 

Here an example of how to compile REBUILD_NEMO on the MetOffice XC40 HPC:

.. code-block:: console
                      
        $ ./maketools -n REBUILD_NEMO -m XC40_METO

Usage
=====
There is 2 manners to use REBUILD_NEMO:

* **rebuild_nemo shell script**

If the rebuild_nemo shell script it used, the namelist is filled automatically depending on the on-line arguments.

.. code-block:: console

 $ ./rebuild_nemo

  NEMO Rebuild
  ************

  usage: rebuild_nemo [-l -p -s -m -n -r -d -x -y -z -t -c] filebase ndomain [rebuild dimensions]

  flags:    -l arch            submit to compute node
            -p num             use num threads
            -s num             split 4D vars into time slice of size num
            -m                 force masking of global arrays (zero if no mdi)
            -n namelist        full path to namelist file to be created (otherwise default nam_rebuild+_process_id is used)
            -r memory          Memory to request on compute node including units (Default = 10Gb)

      key_netcdf4 only 
            -d deflate_level     deflate level for output files
            -x chunksize along x 
            -y chunksize along y 
            -z chunksize along z 
            -t chunksize along t 
            -c total size of the chunk cache 

In case the option '-l arch' is used, a template for the batch script and parameter for job submission has to be provided in BATCH_TEMPLATES (param_arch and rebuild_nemo_batch_arch). 
Exemple from the XC40_METO architecture can be found in the directory. 
Some keywords (NTHREADS, MEMORY, INDIR, NAMELIST and NOPEN) from the template are replaced by the rebuild_nemo script.

* **rebuild_nemo.exe + namelist**

If rebuild_nemo.exe is used directly, a namelist has to be provided. Default name is nam_rebuild. A specific name can be provided as argument.
The minimal namelist required is (here example to rebuild a mesh_mask split in 36 files):

.. code-block:: console

         &nam_rebuild
         filebase='mesh_mask'
         ndomain=36
         /

Some option can be added (the value mentioned here are the default value):

.. code-block:: console

         l_maskout=.false       !(-m option: useful if input file comes from a run using land suppression)
         nslicesize=0           !(-s option: 0 means no splitting in time slice)
         deflate_level=0        !(-d option)
         nc4_xchunk=206         !(-x option)
         nc4_ychunk=135         !(-y option)
         nc4_zchunk=1           !(-z option)
         nc4_tchunk=1           !(-t option)
         fchunksize=32000000    !(-c option)

Example
=======

Here is an example of usage of rebuild_nemo shelf script
(rebuild mesh_mask files on the XC40_METO computer with deflation level set to 1 and chunksize to (1,1,100,100)):

.. code-block:: console

          $ rebuild_nemo -l XC40_METO -m -d 1 -x 100 -y 100 -z 1 -t 1 mesh_mask 36

          output is mask using netcdf missing value (_Fillvalue attribute) or 0 if missing value not in the netcdf.

          file mesh_mask,  num_domains 36, num_threads 1
          Submitting job to compute node
          8510610.xcs00
         $
