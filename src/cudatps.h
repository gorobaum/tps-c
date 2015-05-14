#ifndef TPS_CUDATPS_H_
#define TPS_CUDATPS_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_occupancy.h"

#include "tps.h"
#include "cudalinearsystems.h"
#include "cudamemory.h"

namespace tps {
  
class CudaTPS : public TPS {
public:
  CudaTPS(std::vector<cv::Point2f> referenceKeypoints, std::vector<cv::Point2f> targetKeypoints, tps::Image targetImage, std::string outputName, tps::CudaMemory& cm) :
    TPS(referenceKeypoints, targetKeypoints, targetImage, outputName),
    cudalienarSolver(referenceKeypoints, targetKeypoints),
    width(targetImage.getWidth()),
    height(targetImage.getHeight()),
    cm_(cm) {}; 
  void run();
private:
  tps::CudaLinearSystems cudalienarSolver;
  int width;
  int height;
  tps::CudaMemory& cm_;
	void allocResources();
	void allocCudaResources();
	void freeResources();
	void freeCudaResources();
  void callKernel(double *cudaImageCoord, float *cudaSolution, dim3 threadsPerBlock, dim3 numBlocks);
  void createCudaKeyPoint();
  float *floatSolCol, *floatSolRow;
  float *floatKeyCol, *floatKeyRow;
  uchar *regImage;  
};

}

#endif