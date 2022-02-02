#include "crts.h"
#include "stdio.h"

void crts_init_(){
  CRTS_init();
}

void cyc_time_(unsigned long *t){
  *t = CRTS_time_cycle();	
}
