#ifndef TPS_CUDAMEMORY_H_
#define TPS_CUDAMEMORY_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_occupancy.h"

namespace tps {
  
class CudaMemory {
public:
  CudaMemory(int width, int height, int cps) :
    imageWidth(width),
    imageHeight(height),
    numberOfCps(cps),
    systemDim(cps+3) {};
  void allocCudaCoord();
  void allocCudaSolution();
  void allocCudaKeypoints(float *hostKeypointCol, float *hostKeypointRow);
  void allocCudaImagePixels(unsigned char *hostTargetImage, unsigned char *hostRegImage);
  void freeMemory();
  double* getCoordinateCol() { return coordinateCol; };
  double* getCoordinateRow() { return coordinateRow; };
  float* getSolutionCol() { return solutionCol; };
  float* getSolutionRow() { return solutionRow; };
  float* getKeypointCol() { return keypointCol; };
  float* getKeypointRow() { return keypointRow; };
  unsigned char* getTargetImage() { return targetImage; };
  unsigned char* getRegImage() { return regImage; };
private:
  int imageWidth;
  int imageHeight;
  int numberOfCps;
  int systemDim;
  double *coordinateCol, *coordinateRow;
  float *solutionCol, *solutionRow;
  float *keypointCol, *keypointRow;
  unsigned char *targetImage, *regImage;
};

} // namespace

#endif