# Download

# Download NEMO #
to install SIREN, you should first download NEMO.
see [NEMO quick start guide](https://forge.ipsl.jussieu.fr/nemo/wiki/Users/ModelInstall#DownloadNEMOreferencecodeandconfigurations)

# Compile SIREN #
when NEMO is installed, you just have to compile SIREN codes:
   1. go to ./NEMOGCM/TOOLS
   2. run maketools (ex: ./maketools -n SIREN -m ifort_mpi_beaufix)

      @note to get help on maketools: ./maketools -h

# Fortran Compiler #
SIREN codes were succesfully tested with :
  - ifort (version 18.0.1 20171018)
  - gfortran (version 4.8.5 20150623) 

<HR>
  <b>
  - @ref index
  - @ref md_src_docsrc_2_quickstart
  - @ref md_src_docsrc_3_support_bug
  - @ref md_src_docsrc_4_codingRules
  - @ref md_src_docsrc_5_changeLog
  - @ref todo
  </b>
