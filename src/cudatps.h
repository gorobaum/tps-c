#ifndef TPS_CUDATPS_H_
#define TPS_CUDATPS_H_

#include "cuda.h"
#include "cuda_runtime.h"

#include "tps.h"

namespace tps {
	
class CudaTPS : public TPS {
using TPS::TPS;
public:
	void run();
private:
	void allocResources();
	void allocCudaResources();
	void freeResources();
	void freeCudaResources();
	float *imageCoord;
	float *cudaImageCoord;
	size_t pitch;
};

}

#endif