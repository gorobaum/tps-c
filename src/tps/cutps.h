#ifndef TPS_CUTPS_H_
#define TPS_CUTPS_H_

#include <vector>

#include "utils/cudamemory.h"

short* runTPSCUDA(tps::CudaMemory cm, std::vector<int> dimensions, int numberOfCPs);

#endif