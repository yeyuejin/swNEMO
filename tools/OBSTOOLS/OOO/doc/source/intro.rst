
Introduction
------------

The offline observation operator quick start script involves simple
computation and manipulation of NEMO namelists.

Quick start script
------------------

.. code-block:: none

   ooo -h
   usage: ooo [-h] [-w WORK_DIR] [-f FORECAST_TYPES] [-l LEAD_TIMES]
              [-o OBS_TYPES] [--class4]
              DATE NAMELIST
   
   positional arguments:
     DATE                  Run date.
     NAMELIST              NEMO namelist to edit.
   
   optional arguments:
     -h, --help            show this help message and exit
     -w WORK_DIR, --work-dir WORK_DIR
     -f FORECAST_TYPES, --forecast-types FORECAST_TYPES
                           Choice of forecast,persistence,climatology
     -l LEAD_TIMES, --lead-times LEAD_TIMES
                           Forecast lead times
     -o OBS_TYPES, --obs-types OBS_TYPES
                           Choice of namobs types.
     --class4              Flag to choose class 4 file outputs
     --dry-run             Flag to test namelist building without submitting.
     --cmd COMMAND         Submit task to run.
     -v, --verbose         Prints difference between before and after namelists.


The above command line utility ``ooo`` can be used to edit **NEMO** namelists. 

Run
---

.. automodule:: ooo.run
   :members:


Locators
--------

.. automodule:: ooo.locator
   :members:

Fortran
-------

.. automodule:: ooo.nml
   :members:

