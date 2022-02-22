# swNEMO_v4.0

This file describes the procedures to compile and run the program, along with reproducing the experimental results for swNEMO_V4.0 model.

To build this file, go to /arch folder and change the configuration of arch-sw.fcm file. The Sunway compliler for C file is cc=sw5gcc.
After configuration, return to the parent folder and run "make clean" then "make" for full compilation.
It is worth mentioning that the sw complier may not be accessible in some region, thus the compile options should be changed correspondingly, i.e., cc = your_compiler

Test case can be found in /cfgs/GYRE_PISCES/. First change the parameters in run.sh for your own computing resources, set q to your own queue name and set n for designated number of core groups. Then return to ./EXP00/ folder to alter the nn_GYRE parameters in namelist_cfg file. Due to the chosen resolution and other factors, the experimental value of nn_GYRE may vary, so choose the value wisely.

The localizatiion and upgrade for Sunway system is a very innovative idea, and will be highlighted when using Sunway compiler. All code of kernel innovations in our Sunway system can be found in file /src/MY/kernel_slave.c. 
