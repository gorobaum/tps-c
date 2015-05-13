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
    cudalienarSolver(referenceKeypoints, targetKeypoints),
    width(targetImage.getWidth()),
    height(targetImage.getHeight()) {}; 
  void run();
private:
  tps::CudaLinearSystems cudalienarSolver;
  int width;
  int height;
	void allocResources();
	void allocCudaResources();
	void freeResources();
	void freeCudaResources();
  void callKernel(double *cudaImageCoord, float *cudaSolution, dim3 threadsPerBlock, dim3 numBlocks);
  void createCudaSolution();
  void createCudaKeyPoint();
	double *cudaImageCoordCol, *cudaImageCoordRow;
  float *cudaSolutionCol, *cudaSolutionRow;
  float *floatSolCol, *floatSolRow;
  float *floatKeyCol, *floatKeyRow;
  float *cudaKeyCol, *cudaKeyRow;
  uchar *cudaRegImage, *cudaImage;
  uchar *regImage;  
};

}

#endif