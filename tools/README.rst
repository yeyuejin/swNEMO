*****
Tools
*****

.. todo::

   The 'Tools' chapter needs to be enriched

.. contents::
   :local:
   :depth: 1

A set of tools is provided with NEMO to setup user own configuration and (pre|post)process data.

How to compile a tool
=====================

The tool can be compiled using the maketools script in the tools directory as follows:

.. code-block:: console

   $ ./maketools -m 'my_arch' -n '<TOOL_NAME>'

where ``my_arch`` can be selected among available architecture files or providing a user defined one.

List of tools
=============

BDY_TOOLS
---------

It contains the utility *bdy_reorder* used to reorder old BDY data files used with
previous versions of the model (before 3.4) to make them compatible with NEMO 3.4.

DMP_TOOLS
---------

Used to create a netcdf file called :file:`resto.nc` containing
restoration coefficients for use with the :file:`tra_dmp` module in NEMO
(see :download:`DMP_TOOLS README <../../../tools/DMP_TOOLS/README>`).

DOMAINcfg
---------

This tool allows the creation of a domain configuration file (``domain_cfg.nc``) containing
the ocean domain information required to define an ocean configuration from scratch.
(see :download:`DOMAINcfg README <../../../tools/DOMAINcfg/README>`).

GRIDGEN
-------

A toolbox allowing the creation of regional configurations from curvilinear grid
(see :download:`GRIDGEN documentation <../../../tools/GRIDGEN/doc_cfg_tools.pdf>`).

MISCELLANEOUS
-------------

The tool allows to create alternative configurations to the community without
having to rely on system team sponsorship and support.

MPP_PREP
--------

This tool provides the user with information to choose the best domain decomposition.
The tool computes the number of water processors for all possible decompositions,
up to a maximum number of processors
(see :download:`MPP_PREP documentation <../../../tools/MPP_PREP/mpp_nc.pdf>` and
:download:`MPP_PREP archive <../../../tools/MPP_PREP/mpp_prep-1.0.tar.gz>`).

NESTING
-------

AGRIF nesting tool allows for the seamless two-way coupling of nested sub-models within
the NEMO framework as long as these are defined on subsets of the original root grid.
It allows to create the grid coordinates, the surface forcing and the initial conditions required by
each sub-model when running a NEMO/AGRIF embedded mode
(see :download:`NESTING README <../../../tools/NESTING/README>`).

OBSTOOLS
--------

A series of Fortran utilities which are helpful in handling observation files and
the feedback file output from the NEMO observation operator.
Further info are available in the :doc:`Nemo manual <cite>`.

REBUILD_NEMO
------------

REBUILD_NEMO is a tool to rebuild NEMO output files from multiple processors
(mesh_mask, restart or XIOS output files) into one file
(see :download:`REBUILD_NEMO README <../../../tools/REBUILD_NEMO/README.rst>`).

REBUILD
-------

It contains the old version of REBUILD_NEMO tool based on the IOIPSL code.

SCOORD_GEN
----------

Offline tool to generate a vertical coordinates input file for use with S coordinates.
This has been carried out by copying the model code to an offline tool and then
modifying it to suppress the use of 3D arrays (to reduce memory usage).
The tool has been created in preparation for the removal of the vertical grid definition from
the code.
The output file should contain all variables that are necessary to restart the model.

SECTIONS_DIADCT
---------------

When the Transport across sections diagnostic is activated (``key_diadct``),
this tool is used to build the binary file containing the pathways between
the extremities of each section.
Further info are available in the :doc:`Nemo manual <cite>`.

SIREN
-----

SIREN is a configuration management tool to set up regional configurations with NEMO
(see :download:`SIREN README <../../../tools/SIREN/README>`).

WEIGHTS
-------

This directory contains software for generating and manipulating interpolation weights for use with
the Interpolation On the Fly (IOF) option in NEMO v3 onwards
(see :download:`WEIGHTS README <../../../tools/WEIGHTS/README>`).

TOYATM
------

This directory contains a simplified model that send/receive atmospheric fields to/from NEMO.
Used to test the coupling interface.
This toy requires OASIS3-MCT to be installed and properly defined in the arch file.
