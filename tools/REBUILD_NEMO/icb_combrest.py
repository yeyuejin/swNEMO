import os
from netCDF4 import Dataset
from argparse import ArgumentParser
import numpy as np
import sys

#
# Basic iceberg trajectory restart post-processing python script.
# This script collects iceberg information from the distributed restarts written
# out by each processing region and writes the information into a global restart file.
# The global restart file must already exist and contain the collated 2D spatial fields
# as prepared by utilities such as rebuild_nemo. This python script simply adds the
# iceberg position and state data that is held using the unlimited dimension 'n' which
# has been ignored by rebuild_nemo. Each processing region that contains icebergs will
# (probably) have a different number of icebergs (indicated by differing values for the
# current size of the unlimited dimension). This script collects all icebergs into a 
# single unordered list.
#

parser = ArgumentParser(description='produce a global trajectory restart file from distributed output\
                                     files, e.g. \n python ./icb_pp.py \
                                     -f  icebergs_00692992_restart_ -n 480 -o icebergs_restart.nc [-O]')

parser.add_argument('-f',dest='froot',help='fileroot_of_distrbuted_data; root name of \
                     distributed trajectory restart file (usually completed with XXXX.nc, where \
                     XXXX is the 4 digit processor number)', 
                     default='icebergs_00692992_restart_')

parser.add_argument('-n',dest='fnum',help='number of distributed files to process', 
                     type=int, default=None)

parser.add_argument('-o',dest='fout',help='global_iceberg_restart; file name to append the \
                     global iceberg restart data to.', default='icebergs_restart.nc')

parser.add_argument('-O',dest='fcre',help='Create the output file from scratch rather than \
                     append to an existing file.', \
                     action='store_true', default=False)

args = parser.parse_args()

default_used = 0
if args.froot is None:
    pathstart = 'icebergs_00692992_restart_'
    default_used = 1
else:
    pathstart = args.froot

if args.fnum is None:
    procnum = 0
    default_used = 1
else:
    procnum = args.fnum

if args.fout is None:
    pathout = 'icebergs_restart.nc'
    default_used = 1
else:
    pathout = args.fout

if default_used == 1:
   print('At least one default value will be used; command executing is:')
   print('icb_combrest.py -f ',pathstart,' -n ',procnum,' -o ',pathout)

if procnum < 1:
   print('Need some files to collate! procnum = ',procnum)
   sys.exit(11)

icu = []
times = []
ntraj = 0
nk = 0
#
# Loop through all distributed datasets to obtain the total number 
# of icebergs to transfer
#
for n in range(procnum):
 nn = '%4.4d' % n
 try:
   fw = Dataset(pathstart+nn+'.nc')
 except:
   print 'Error: unable to open input file: ' + pathstart+nn+'.nc'
   sys.exit(12)
 for d in fw.dimensions :
  if d == 'n' :
   if len(fw.dimensions['n']) > 0:
#    print 'icebergs found in: ' + pathstart+nn+'.nc' 
     if len(fw.dimensions['k']) > nk :
       nk = len(fw.dimensions['k'])
     ntraj = ntraj + len(fw.dimensions['n'])
 fw.close()
#
print(ntraj, ' icebergs found across all datasets')
#
# Declare 2-D arrays to receive the data from all files
#
lons = np.zeros(ntraj)
lats = np.zeros(ntraj)
xis  = np.zeros(ntraj)
yjs  = np.zeros(ntraj)
uvs  = np.zeros(ntraj)
vvs  = np.zeros(ntraj)
mas  = np.zeros(ntraj)
ths  = np.zeros(ntraj)
wis  = np.zeros(ntraj)
les  = np.zeros(ntraj)
dys  = np.zeros(ntraj)
mss  = np.zeros(ntraj)
msb  = np.zeros(ntraj)
hds  = np.zeros(ntraj)
yrs  = np.zeros(ntraj      , dtype=int)
num  = np.zeros((ntraj, nk), dtype=int)
#
# loop through distributed datasets again, this time
# collecting all trajectory data
#
nt = 0
for n in range(procnum):
 nn = '%4.4d' % n
 fw = Dataset(pathstart+nn+'.nc')
 for d in fw.dimensions :
  if d == 'n' :
# Note many distributed datafiles will contain no iceberg data
# so skip quickly over these
    m  = len(fw.dimensions['n'])
    if m > 0:
#     print pathstart+nn+'.nc'
      lons[nt:nt+m]  = fw.variables['lon'][:]
      lats[nt:nt+m]  = fw.variables['lat'][:]
      xis[nt:nt+m]   = fw.variables['xi'][:]
      yjs[nt:nt+m]   = fw.variables['yj'][:]
      uvs[nt:nt+m]   = fw.variables['uvel'][:]
      vvs[nt:nt+m]   = fw.variables['vvel'][:]
      mas[nt:nt+m]   = fw.variables['mass'][:]
      ths[nt:nt+m]   = fw.variables['thickness'][:]
      wis[nt:nt+m]   = fw.variables['width'][:]
      les[nt:nt+m]   = fw.variables['length'][:]
      dys[nt:nt+m]   = fw.variables['day'][:]
      mss[nt:nt+m]   = fw.variables['mass_scaling'][:]
      msb[nt:nt+m]   = fw.variables['mass_of_bits'][:]
      hds[nt:nt+m]   = fw.variables['heat_density'][:]
      yrs[nt:nt+m]   = fw.variables['year'][:]
      num[nt:nt+m,:] = fw.variables['number'][:,:]
      nt = nt + m
 fw.close()

# Finally create the output file and write out the collated sets
#
if args.fcre :
  try:
    fo = Dataset(pathout, 'w', format='NETCDF4')
  except:
    print 'Error accessing output file: ' + pathout
    print 'Check it is a writable location.'
    sys.exit(13)
else :
  # Copy 2D variables across to output file from input file. This step avoids problems if rebuild_nemo 
  # has created an "n" dimension in the prototype rebuilt file (ie. if there are icebergs on the zeroth 
  # processor). 
  try:
    os.rename(pathout,pathout.replace('.nc','_WORK.nc'))
  except OSError:
    print 'Error: unable to move icebergs restart file: '+pathout
    sys.exit(14)
  #
  try:
    fi = Dataset(pathout.replace('.nc','_WORK.nc'), 'r')
  except:
    print 'Error: unable to open icebergs restart file: '+pathout.replace('.nc','_WORK.nc')
    sys.exit(15)
  fo = Dataset(pathout, 'w')
  for dim in ['x','y','c','k']:
    indim = fi.dimensions[dim]
    fo.createDimension(dim, len(indim))
  for var in ['kount','calving','calving_hflx','stored_ice','stored_heat']:
    invar = fi.variables[var]
    fo.createVariable(var, invar.datatype, invar.dimensions)
    fo.variables[var][:] = invar[:]
    if "long_name" in invar.ncattrs():
        fo.variables[var].long_name = invar.long_name
    if "units" in invar.ncattrs():
        fo.variables[var].units = invar.units
  os.remove(pathout.replace('.nc','_WORK.nc'))
#
add_k = 1
for d in fo.dimensions :
  if d == 'n' :
    print 'Error: dimension n already exists in output file'
    sys.exit(16)
  if d == 'k' :
    add_k = 0
onn  = fo.createDimension('n', None)
if add_k > 0 :
 onk = fo.createDimension('k', nk)
olon = fo.createVariable('lon', 'f8',('n'))
olat = fo.createVariable('lat', 'f8',('n'))
oxis = fo.createVariable('xi', 'f8',('n'))
oyjs = fo.createVariable('yj', 'f8',('n'))
ouvs = fo.createVariable('uvel', 'f8',('n'))
ovvs = fo.createVariable('vvel', 'f8',('n'))
omas = fo.createVariable('mass', 'f8',('n'))
oths = fo.createVariable('thickness', 'f8',('n'))
owis = fo.createVariable('width', 'f8',('n'))
oles = fo.createVariable('length', 'f8',('n'))
odys = fo.createVariable('day', 'f8',('n'))
omss = fo.createVariable('mass_scaling', 'f8',('n'))
omsb = fo.createVariable('mass_of_bits', 'f8',('n'))
ohds = fo.createVariable('heat_density', 'f8',('n'))
oyrs = fo.createVariable('year', 'i4',('n'))
onum = fo.createVariable('number', 'i4',('n','k'))
#
olon[:] = lons
olon.long_name = "longitude"
olon.units = "degrees_E" 
#
olat[:] = lats
olat.long_name = "latitude"
olat.units = "degrees_N"
#
oxis[:] = xis
oxis.long_name = "x grid box position"
oxis.units = "fractional"
#
oyjs[:] = yjs
oyjs.long_name = "y grid box position"
oyjs.units = "fractional"
#
ouvs[:] = uvs
ouvs.long_name = "zonal velocity"
ouvs.units = "m/s"
#
ovvs[:] = vvs
ovvs.long_name = "meridional velocity"
ovvs.units = "m/s"
#
omas[:] = mas
omas.long_name = "mass"
omas.units = "kg"
#
oths[:] = ths
oths.long_name = "thickness"
oths.units = "m"
#
owis[:] = wis
owis.long_name = "width"
owis.units = "m"
#
oles[:] = les
oles.long_name = "length"
oles.units = "m"
#
odys[:] = dys
odys.long_name = "year day of calving event"
odys.units = "days"
#
omss[:] = mss
omss.long_name = "scaling factor for mass of calving berg"
omss.units = "none"
#
omsb[:] = msb
omsb.long_name = "mass of bergy bits"
omsb.units = "kg"
#
ohds[:] = hds
ohds.long_name = "heat density"
ohds.units = "J/kg"
#
oyrs[:] = yrs
oyrs.long_name = "calendar year of calving event"
oyrs.units = "years"
#
onum[:,:] = num
onum.long_name = "iceberg number on this processor"
onum.units = "count"
#
fo.close()
