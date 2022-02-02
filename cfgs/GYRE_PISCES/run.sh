#########################################################################
# File Name: run.sh
# Author: zhousc
# Created Time: 2020年12月09日 星期三 18时02分16秒
#########################################################################
#!/bin/bash

#n=1
n=65536
#q=q_test_ss
q=q_test_ss

cd EXP00
rm -f nemo
ln -s ../BLD/bin/nemo.exe nemo
#bsub -I -b -p -q $q -J nemo -n $n -cgsp 64 -host_stack 4000 -share_size 8000 nemo 2>&1 | tee run.log
#bsub -I -b -p -q $q -J nemo_z128 -perf -swrunarg '-P all' -n $n -cgsp 64 -host_stack 4000 -share_size 8000 nemo 2>&1 | tee run_all.log
#bsub -I -b -p -q $q -J nemo_z128 -perf -swrunarg '-P slave' -n $n -cgsp 64 -host_stack 4000 -share_size 8000 nemo 2>&1 | tee run_all.log
#bsub -I -b -p -q $q -J nemo_z128 -perf -swrunarg '-P master' -n $n -cgsp 64 -host_stack 4000 -share_size 8000 nemo 2>&1 | tee run_all.log
bsub -I -b -p -q $q -cache_size 0 -J nemo_debug -n $n -cgsp 64 -host_stack 4000 -share_size 8000 nemo 2>&1 | tee run_all.log
#bsub -I -b -p -q $q -cache_size 0 -J nemo_z128 -perf -swrunarg '-P master' -n $n -cgsp 64 -host_stack 4000 -share_size 8000 nemo 2>&1 | tee run_all.log


