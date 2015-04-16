#ifndef TPS_CUDATPS_H_
#define TPS_CUDATPS_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_occupancy.h"

#include "tps.h"
#include "cudalinearsystems.h"

namespace tps {
  
class CudaTPS : public TPS {
public:
  CudaTPS(std::vector<cv::Point2f> referenceKeypoints, std::vector<cv::Point2f> targetKeypoints, tps::Image targetImage, std::string outputName) :
    TPS(referenceKeypoints, targetKeypoints, targetImage, outputName),
    cudalienarSolver(referenceKeypoints, targetKeypoints) {}; 
  void run();
private:
  tps::CudaLinearSystems cudalienarSolver;
	void allocResources();
	void allocCudaResources();
	void freeResources();
	void freeCudaResources();
  void callKernel(double* cudaImageCoord, float *cudaSolution, dim3 threadsPerBlock, dim3 numBlocks);
  void createCudaSolution();
  void createCudaKeyPoint();
	double *cudaImageCoordX, *cudaImageCoordY;
  float *cudaSolutionX, *cudaSolutionY;
  float *floatKeyX, *floatKeyY;
  float *floatSolX, *floatSolY;
  float *cudaKeyX, *cudaKeyY;
  uchar *cudaRegImage, *cudaImage;
  uchar *regImage;
  std::vector<int> dimensions;
};

}

#endif