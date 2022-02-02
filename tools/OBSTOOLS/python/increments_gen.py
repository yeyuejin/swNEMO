'''
Generate a test increments file
and the SlaReference file

Requires coordinates and depth file

D. J. Lea   Sep 2014
'''

import netCDF4
import numpy as np

# example files to get lon and lat
exdir='/hpc/data/nwp/ofrd/frld/FORCING/ORCA2_LIM_nemo_v3.6'
exfile1='coordinates.nc'

infile1=exdir+'/'+exfile1
fileid=netCDF4.Dataset(infile1, mode='r')
nav_lon=fileid.variables['nav_lon'][:,:]
nav_lat=fileid.variables['nav_lat'][:,:]
fileid.close

print np.shape(nav_lon)

exfile2='data_1m_potential_temperature_nomask.nc'
infile2=exdir+'/'+exfile2
fileid=netCDF4.Dataset(infile2, mode='r')
nav_lev=fileid.variables['depth'][:]
fileid.close()

nshp=np.shape(nav_lon)
ny=nshp[0]
nx=nshp[1]
nz=np.size(nav_lev)
nt=1

#create a new file
outdir='.'
outfile='assim_background_increments_test.nc'

print "creating: ",outfile
ncfile=netCDF4.Dataset(outdir+'/'+outfile, mode='w', format='NETCDF3_CLASSIC')
ncfile.createDimension('x',nx)
ncfile.createDimension('y',ny)
ncfile.createDimension('z',nz)
ncfile.createDimension('t',size=0)
ncnav_lat = ncfile.createVariable('nav_lat','f4',('y','x'))
ncnav_lat[:,:]=nav_lat
ncnav_lon = ncfile.createVariable('nav_lon','f4',('y','x'))
ncnav_lon[:,:]=nav_lon
ncnav_lev = ncfile.createVariable('nav_lev','f4',('z'))
ncnav_lev[:]=nav_lev
ncvar = ncfile.createVariable('bckineta','f8',('t','y','x'))
ncvar[0,:,:]=0.1
ncvar = ncfile.createVariable('bckins','f8',('t','z','y','x'))
ncvar[0,:,:,:]=0.1
ncvar = ncfile.createVariable('bckint','f8',('t','z','y','x'))
ncvar[0,:,:,:]=0.1
ncvar = ncfile.createVariable('bckinu','f8',('t','z','y','x'))
ncvar[0,:,:,:]=0.1
ncvar = ncfile.createVariable('bckinv','f8',('t','z','y','x'))
ncvar[0,:,:,:]=0.1
ncvar = ncfile.createVariable('bckinseaice','f8',('t','y','x'))
ncvar[0,:,:]=0.1
ncvar = ncfile.createVariable('bckinsshobias','f8',('t','y','x'))
ncvar[0,:,:]=0.1
ncvar = ncfile.createVariable('time_counter','f8',('t'))
ncvar[0]=0
ncvar = ncfile.createVariable('time','f8')
ncvar[:]=10101
ncvar = ncfile.createVariable('z_inc_dateb','f8')
ncvar[:]=10101
ncvar = ncfile.createVariable('z_inc_datef','f8')
ncvar[:]=10102


ncfile.close()

#create sla reference file
outfile='slaReferenceLevel_test.nc'

print "creating: ",outfile
ncfile=netCDF4.Dataset(outdir+'/'+outfile, mode='w', format='NETCDF3_CLASSIC')
ncfile.createDimension('x',nx)
ncfile.createDimension('y',ny)
ncnav_lat = ncfile.createVariable('nav_lat','f4',('y','x'))
ncnav_lat[:,:]=nav_lat
ncnav_lon = ncfile.createVariable('nav_lon','f4',('y','x'))
ncnav_lon[:,:]=nav_lon
ncvar = ncfile.createVariable('sossheig','f4',('y','x'),fill_value=1e20)
ncvar[:,:]=0.01

ncfile.close()
