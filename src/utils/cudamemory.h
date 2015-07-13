#ifndef TPS_CUDAMEMORY_H_
#define TPS_CUDAMEMORY_H_

#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_occupancy.h"

#include "image/image.h"

namespace tps {
  
class CudaMemory {
public:
  CudaMemory(std::vector<int> dimensions, std::vector< std::vector<float> > referenceKeypoints) :
    imageSize(dimensions[0]*dimensions[1]*dimensions[2]),
    referenceKeypoints_(referenceKeypoints),
    numberOfCps(referenceKeypoints.size()),
    systemDim(numberOfCps+4) {};
  void freeMemory();
  void allocCudaMemory(tps::Image& image);
  double memoryEstimation();
  float* getSolutionX() { return solutionX; };
  float* getSolutionY() { return solutionY; };
  float* getSolutionZ() { return solutionZ; };
  float* getKeypointX() { return keypointX; };
  float* getKeypointY() { return keypointY; };
  float* getKeypointZ() { return keypointZ; };
  short* getTargetImage() { return targetImage; };
  cudaTextureObject_t getTexObj() { return texObj; };
  short* getRegImage() { return regImage; };
  std::vector<float> getHostSolX();
  std::vector<float> getHostSolY();
  std::vector<float> getHostSolZ();
private:
  void allocCudaSolution();
  void allocCudaKeypoints();
  void allocCudaImagePixels(tps::Image& image);
  std::vector<float> cudaToHost(float *cudaMemory);
  int imageSize;
  std::vector< std::vector<float> > referenceKeypoints_;
  int numberOfCps;
  int systemDim;
  float *solutionX, *solutionY, *solutionZ;
  float *keypointX, *keypointY, *keypointZ;
  short *targetImage, *regImage;
  cudaTextureObject_t texObj;
  cudaArray* cuArray;
};

} // namespace

#endif