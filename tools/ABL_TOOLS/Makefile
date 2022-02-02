SHELL = /bin/bash
#-------------------------------------------------------------------------------------------------------------
EXEC_BIN   = vinterp_abl_frc.exe
EXEC_BIN2  = uvg_hpg_abl_frc.exe 
EXEC_BIN3  = drown_abl_frc.exe

NETCDF     := $(NETCDF_DIR)
NETCDF_LIB := -L$(NETCDF)/lib -lnetcdf -lnetcdff
NETCDF_INC := -I$(NETCDF)/include

#FC         := gfortran
#FFLAGS     := -fdefault-real-8 -fdefault-double-8 -O3 $(NETCDF_INC)
#FFLAGS     := -g -O0 -fcheck=bounds -Wall -Wextra -fbacktrace -finit-real=snan -finit-integer=8888 -finit-character=90 $(NETCDF_INC)

FC         := ifort
FFLAGS     := -check noarg_temp_created -autodouble -O3 $(NETCDF_INC)
#FFLAGS     := -check noarg_temp_created -autodouble -O0 -g -traceback -check all $(NETCDF_INC)

#---------------------------------------------------------------------  
SRC_DIR        =  $(PWD)
OBJ_DIR        =  $(PWD)
FINAL_OBJS     =  $(OBJ_DIR)/main_vinterp.o $(OBJ_DIR)/module_io.o $(OBJ_DIR)/module_interp.o $(OBJ_DIR)/module_grid.o
FINAL_OBJS2    =  $(OBJ_DIR)/main_uvg_hpg.o $(OBJ_DIR)/module_io.o $(OBJ_DIR)/module_grid.o
FINAL_OBJS3    =  $(OBJ_DIR)/main_drown.o $(OBJ_DIR)/module_io.o  $(OBJ_DIR)/module_grid.o
#---------------------------------------------------------------------

all: | $(EXEC_BIN) $(EXEC_BIN2) $(EXEC_BIN3)
	@echo 
	@echo ===================================================
	@echo     ABL Preprocessing tools for ECMWF data: OK
	@echo ===================================================
	@echo

clean:
	$(RM)    $(OBJ_DIR)/*.o   *.mod 
	$(RM)    $(EXEC_BIN) $(EXEC_BIN2) $(EXEC_BIN3)
	
$(EXEC_BIN): $(FINAL_OBJS) 
	$(FC) -o $(EXEC_BIN) $(FINAL_OBJS) $(NETCDF_LIB)

$(EXEC_BIN2): $(FINAL_OBJS2) 
	$(FC) -o $(EXEC_BIN2) $(FINAL_OBJS2) $(NETCDF_LIB)

$(EXEC_BIN3): $(FINAL_OBJS3) 
	$(FC) -o $(EXEC_BIN3) $(FINAL_OBJS3) $(NETCDF_LIB)

# Rules
.DEFAULT:

.SUFFIXES:

.PRECIOUS: $(SRC_DIR)/%.F90
	
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.F90
	$(FC) $(FFLAGS) -o $@ -c $<

$(OBJ_DIR)/main_vinterp.o:	$(OBJ_DIR)/module_grid.o $(OBJ_DIR)/module_io.o $(OBJ_DIR)/module_interp.o
$(OBJ_DIR)/main_uvg_hpg.o:	$(OBJ_DIR)/module_grid.o $(OBJ_DIR)/module_io.o
$(OBJ_DIR)/main_drown.o:	$(OBJ_DIR)/module_grid.o $(OBJ_DIR)/module_io.o
$(OBJ_DIR)/module_io.o:
$(OBJ_DIR)/module_interp.o:
$(OBJ_DIR)/module_grid.o: $(OBJ_DIR)/module_io.o
