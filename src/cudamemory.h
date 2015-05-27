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
    systemDim(numberOfCps+3) {};
  void freeMemory();
  void allocCudaMemory(tps::Image& image);
  double memoryEstimation();
  double* getCoordinateCol() { return coordinateCol; };
  double* getCoordinateRow() { return coordinateRow; };
  float* getSolutionCol() { return solutionCol; };
  float* getSolutionRow() { return solutionRow; };
  float* getKeypointCol() { return keypointCol; };
  float* getKeypointRow() { return keypointRow; };
  unsigned char* getTargetImage() { return targetImage; };
  unsigned char* getRegImage() { return regImage; };
  std::vector<float> getHostSolCol();
  std::vector<float> getHostSolRow();
private:
  void allocCudaCoord();
  void allocCudaSolution();
  void allocCudaKeypoints();
  void allocCudaImagePixels(tps::Image& image);
  std::vector<float> cudaToHost(float *cudaMemory);
  int imageWidth;
  int imageHeight;
  std::vector< std::vector<float> > referenceKeypoints_;
  int numberOfCps;
  int systemDim;
  double *coordinateCol, *coordinateRow;
  float *solutionCol, *solutionRow;
  float *keypointCol, *keypointRow;
  unsigned char *targetImage, *regImage;
};

} // namespace

#endif