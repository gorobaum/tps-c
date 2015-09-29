#ifndef TPS_CUTPS_H_
#define TPS_CUTPS_H_

#include <vector>

#include "utils/cudamemory.h"

int getBlockSize();

void runTPSCUDA(tps::CudaMemory cm, std::vector<int> dimensions, int numberOfCPs);

short* getGPUResult(tps::CudaMemory cm, std::vector<int> dimensions);

#endif