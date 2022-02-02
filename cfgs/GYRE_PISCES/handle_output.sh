#!/bin/bash
#
nc_path='/home/export/online1/mdt00/systest/swyyz/yeyuejin/full-nemo4/check_error'
da=`date +%Y%m%d`
time=`date +%H%M`
name=$(basename `pwd`)
output_path=$nc_path'/'$name'_'$da'_'$time
mkdir $output_path
cp EXP00/*nc $output_path
cd $output_path
../nocscombine4 -f *grid_U*.nc
../nocscombine4 -f *grid_T*.nc
../nocscombine4 -f *grid_W*.nc
../nocscombine4 -f *grid_V*.nc
cd -
echo ''
echo 'The handled output file is in '
echo $output_path

/home/export/online1/mdt00/systest/swyyz/yeyuejin/full-nemo4/python-3.8.5/bin/python3 $nc_path/readnc.py $nc_path/master_SP_small/GYRE_20h_00010101_00010102_grid_T.nc $output_path/GYRE_20h_00010101_00010102_grid_T.nc
/home/export/online1/mdt00/systest/swyyz/yeyuejin/full-nemo4/python-3.8.5/bin/python3 $nc_path/readnc.py $nc_path/master_SP_small/GYRE_20h_00010101_00010102_grid_W.nc $output_path/GYRE_20h_00010101_00010102_grid_W.nc
/home/export/online1/mdt00/systest/swyyz/yeyuejin/full-nemo4/python-3.8.5/bin/python3 $nc_path/readnc.py $nc_path/master_SP_small/GYRE_20h_00010101_00010102_grid_U.nc $output_path/GYRE_20h_00010101_00010102_grid_U.nc
/home/export/online1/mdt00/systest/swyyz/yeyuejin/full-nemo4/python-3.8.5/bin/python3 $nc_path/readnc.py $nc_path/master_SP_small/GYRE_20h_00010101_00010102_grid_V.nc $output_path/GYRE_20h_00010101_00010102_grid_V.nc
