#ifndef TPS_CUDAMEMORY_H_
#define TPS_CUDAMEMORY_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_occupancy.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "image.h"

namespace tps {
  
class CudaMemory {
public:
  CudaMemory(int width, int height, std::vector<cv::Point2f> referenceKeypoints) :
    imageWidth(width),
    imageHeight(height),
    referenceKeypoints_(referenceKeypoints),
    numberOfCps(referenceKeypoints.size()),
    systemDim(numberOfCps+3) {};
  void allocCudaCoord();
  void allocCudaSolution();
  void allocCudaKeypoints();
  void allocCudaImagePixels(tps::Image& image);
  void freeMemory();
  double memoryEstimation();
  double* getCoordinateCol() { return coordinateCol; };
  double* getCoordinateRow() { return coordinateRow; };
  float* getSolutionCol() { return solutionCol; };
  float* getSolutionRow() { return solutionRow; };
  float* getKeypointCol() { return keypointCol; };
  float* getKeypointRow() { return keypointRow; };
  uchar* getTargetImage() { return targetImage; };
  uchar* getRegImage() { return regImage; };
private:
  int imageWidth;
  int imageHeight;
  std::vector<cv::Point2f> referenceKeypoints_;
  int numberOfCps;
  int systemDim;
  double *coordinateCol, *coordinateRow;
  float *solutionCol, *solutionRow;
  float *keypointCol, *keypointRow;
  uchar *targetImage, *regImage;
};

} // namespace

#endif