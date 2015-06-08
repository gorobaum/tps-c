#ifndef TPS_CUTPS_H_
#define TPS_CUTPS_H_

#include "cudamemory.h"

unsigned char* runTPSCUDA(tps::CudaMemory cm, int width, int height, int slices, int numberOfCPs);

#endif