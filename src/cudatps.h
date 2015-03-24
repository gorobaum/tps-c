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
  void callKernel(float *cudaSolution, double *imageCoord, dim3 threadsPerBlock, dim3 numBlocks);
  void createCudaSolution();
  void createCudaKeyPoint();
	double *imageCoordX;
  double *imageCoordY;
	double *cudaImageCoord;
  float *cudaSolutionX, *cudaSolutionY;
  float *floatKeyX, *floatKeyY;
  float *floatSolX, *floatSolY;
  float *cudaKeyX, *cudaKeyY;
  std::vector<int> dimensions;
};

}

#endif