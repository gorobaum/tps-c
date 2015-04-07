#ifndef TPS_CUDATPS_H_
#define TPS_CUDATPS_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_occupancy.h"

#include "tps.h"
#include "cudalinearsystems.h"
#include "OPCVlinearsystems.h"

namespace tps {
  
class CudaTPS : public TPS {
public:
  CudaTPS(std::vector<cv::Point2f> referenceKeypoints, std::vector<cv::Point2f> targetKeypoints, tps::Image targetImage, std::string outputName, bool cudaLS) :
    TPS(referenceKeypoints, targetKeypoints, targetImage, outputName),
    opcvlienarSolver(referenceKeypoints, targetKeypoints),
    cudalienarSolver(referenceKeypoints, targetKeypoints),
    cudaLS_(cudaLS) {}; 
  void run();
private:
  tps::OPCVLinearSystems opcvlienarSolver;
  tps::CudaLinearSystems cudalienarSolver;
  bool cudaLS_;
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
  size_t cudaPitch, hostPitch;
  std::vector<int> dimensions;
};

}

#endif