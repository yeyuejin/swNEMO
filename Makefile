CFG=GYRE_PISCES
ARCH=sw

OPT_DIR=src/MY
BLD_DIR=$(OPT_DIR)/bld

#FORT=sw5f90
CC=sw9gcc

.PHONY : all mkdir

all : mkdir $(BLD_DIR)/kernel_slave.o
	./makenemo -r $(CFG) -m $(ARCH) -j 8 2>&1 | tee build.log
	
mkdir:
	@if [ ! -d $(BLD_DIR) ]; then mkdir -p $(BLD_DIR); fi;

$(BLD_DIR)/kernel_slave.o : $(OPT_DIR)/kernel_slave.c
	$(CC) -mslave $^ -c -mftz -mieee -msimd -o $@ -lm_slave -I/usr/sw/yyzlib/math_lib_vect
	@rm -f cfgs/GYRE_PISCES/BLD/obj/nemo.o
	
clean:
#	@rm -rf $(BLD_DIR)
	./makenemo -r $(CFG) -m $(ARCH) clean

