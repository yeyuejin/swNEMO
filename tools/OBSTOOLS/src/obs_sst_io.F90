MODULE obs_sst_io
   !!======================================================================
   !!                       ***  MODULE obs_sst_io  ***
   !! Observation operators : I/O for GHRSST files
   !!======================================================================
   !! History : 
   !!             !  09-01  (K. Mogensen) Initial version
   !!----------------------------------------------------------------------
   !!----------------------------------------------------------------------
   !!   read_sstfile    :  Read a obfbdata structure from an GHRSST file
   !!----------------------------------------------------------------------
   USE par_kind
   USE obs_utils
   USE obs_fbm
   USE julian
   USE netcdf

   IMPLICIT NONE

   !!----------------------------------------------------------------------
   !! NEMO/OCE 4.0 , NEMO Consortium (2018)
   !! $Id: obs_sst_io.F90 2287 2010-10-18 07:53:52Z smasson $
   !! Software governed by the CeCILL licence (./LICENSE)
   !!----------------------------------------------------------------------

CONTAINS

#include "obssst_io.h90"

END MODULE obs_sst_io
