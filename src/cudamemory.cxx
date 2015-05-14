#include "cudamemory.h"

void tps::CudaMemory::allocCudaCoord() {
  cudaMalloc(&coordinateCol, imageWidth*imageHeight*sizeof(double));
  cudaMalloc(&coordinateRow, imageWidth*imageHeight*sizeof(double));
}

void tps::CudaMemory::allocCudaSolution() {
  cudaMalloc(&solutionCol, systemDim*sizeof(float));
  cudaMalloc(&solutionRow, systemDim*sizeof(float));
}

void tps::CudaMemory::allocCudaKeypoints() {
  float* hostKeypointCol = (float*)malloc(referenceKeypoints_.size()*sizeof(float));
  float* hostKeypointRow = (float*)malloc(referenceKeypoints_.size()*sizeof(float));
  for (uint i = 0; i < referenceKeypoints_.size(); i++) {
    hostKeypointCol[i] = referenceKeypoints_[i].x;
    hostKeypointRow[i] = referenceKeypoints_[i].y;
  }
  cudaMalloc(&keypointCol, numberOfCps*sizeof(float));
  cudaMemcpy(keypointCol, hostKeypointCol, numberOfCps*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&keypointRow, numberOfCps*sizeof(float));
  cudaMemcpy(keypointRow, hostKeypointRow, numberOfCps*sizeof(float), cudaMemcpyHostToDevice);
  free(hostKeypointCol);
  free(hostKeypointRow);
}

void tps::CudaMemory::allocCudaImagePixels(tps::Image& image) {
  cudaMalloc(&targetImage, imageWidth*imageHeight*sizeof(uchar));
  cudaMemcpy(targetImage, image.getPixelVector(), imageWidth*imageHeight*sizeof(uchar), cudaMemcpyHostToDevice);
  cudaMalloc(&regImage, imageWidth*imageHeight*sizeof(uchar));
}

double tps::CudaMemory::memoryEstimation() {
  int floatSize = sizeof(float);
  int doubleSize = sizeof(double);
  int ucharSize = sizeof(uchar);

  double solutionsMemory = 2.0*systemDim*floatSize/(1024*1024);
  double coordinatesMemory = 2.0*imageWidth*imageHeight*doubleSize/(1024*1024);
  double keypointsMemory = 2.0*numberOfCps*floatSize/(1024*1024);
  double pixelsMemory = 2.0*imageWidth*imageHeight*ucharSize/(1024*1024);

  double totalMemory = solutionsMemory+coordinatesMemory+keypointsMemory+pixelsMemory;
  return totalMemory;
}

void tps::CudaMemory::freeMemory() {
  cudaFree(coordinateCol);
  cudaFree(coordinateRow);
  cudaFree(solutionCol);
  cudaFree(solutionRow);
  cudaFree(keypointCol);
  cudaFree(keypointRow);
  cudaFree(targetImage);
  cudaFree(regImage);
}
