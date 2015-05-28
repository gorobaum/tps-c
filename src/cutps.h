#ifndef TPS_CUTPS_H_
#define TPS_CUTPS_H_

#include <vector>
#include "cudamemory.h"

unsigned char* runTPSCUDA(tps::CudaMemory cm, int width, int height, int numberOfCPs);

#endif