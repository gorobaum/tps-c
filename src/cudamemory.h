#ifndef TPS_CUDAMEMORY_H_
#define TPS_CUDAMEMORY_H_

#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_occupancy.h"

#include "image.h"

namespace tps {
  
class CudaMemory {
public:
  CudaMemory(int width, int height, std::vector< std::vector<float> > referenceKeypoints) :
    imageWidth(width),
    imageHeight(height),
    referenceKeypoints_(referenceKeypoints),
    numberOfCps(referenceKeypoints.size()),
    systemDim(numberOfCps+4) {};
  void freeMemory();
  void allocCudaMemory(tps::Image& image);
  double memoryEstimation();
  float* getSolutionCol() { return solutionCol; };
  float* getSolutionRow() { return solutionRow; };
  float* getSolutionSlice() { return solutionSlice; };
  float* getKeypointCol() { return keypointCol; };
  float* getKeypointRow() { return keypointRow; };
  float* getKeypointSlice() { return keypointSlice; };
  unsigned char* getTargetImage() { return targetImage; };
  unsigned char* getRegImage() { return regImage; };
  std::vector<float> getHostSolCol();
  std::vector<float> getHostSolRow();
  std::vector<float> getHostSolSlice();
private:
  void allocCudaSolution();
  void allocCudaKeypoints();
  void allocCudaImagePixels(tps::Image& image);
  std::vector<float> cudaToHost(float *cudaMemory);
  int imageWidth;
  int imageHeight;
  std::vector< std::vector<float> > referenceKeypoints_;
  int numberOfCps;
  int systemDim;
  float *solutionCol, *solutionRow, *solutionSlice;
  float *keypointCol, *keypointRow, *keypointSlice;
  unsigned char *targetImage, *regImage;
};

} // namespace

#endif