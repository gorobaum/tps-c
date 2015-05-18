#include "cudamemory.h"

inline
cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        std::cout << "CUDA Runtime Error: \n" << cudaGetErrorString(result) << std::endl;
        assert(result == cudaSuccess);
    }
    return result;
}

void tps::CudaMemory::allocCudaMemory(tps::Image& image) {
  allocCudaCoord();
  allocCudaSolution();
  allocCudaKeypoints();
  allocCudaImagePixels(image);
}

void tps::CudaMemory::allocCudaCoord() {
  checkCuda(cudaMalloc(&coordinateCol, imageWidth*imageHeight*sizeof(double)));
  checkCuda(cudaMalloc(&coordinateRow, imageWidth*imageHeight*sizeof(double)));
}

void tps::CudaMemory::allocCudaSolution() {
  checkCuda(cudaMalloc(&solutionCol, systemDim*sizeof(float)));
  checkCuda(cudaMalloc(&solutionRow, systemDim*sizeof(float)));
}

void tps::CudaMemory::allocCudaKeypoints() {
  float* hostKeypointCol = (float*)malloc(referenceKeypoints_.size()*sizeof(float));
  float* hostKeypointRow = (float*)malloc(referenceKeypoints_.size()*sizeof(float));
  for (uint i = 0; i < referenceKeypoints_.size(); i++) {
    hostKeypointCol[i] = referenceKeypoints_[i].x;
    hostKeypointRow[i] = referenceKeypoints_[i].y;
  }
  checkCuda(cudaMalloc(&keypointCol, numberOfCps*sizeof(float)));
  checkCuda(cudaMemcpy(keypointCol, hostKeypointCol, numberOfCps*sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMalloc(&keypointRow, numberOfCps*sizeof(float)));
  cudaMemcpy(keypointRow, hostKeypointRow, numberOfCps*sizeof(float), cudaMemcpyHostToDevice);
  free(hostKeypointCol);
  free(hostKeypointRow);
}

void tps::CudaMemory::allocCudaImagePixels(tps::Image& image) {
  checkCuda(cudaMalloc(&targetImage, imageWidth*imageHeight*sizeof(uchar)));
  checkCuda(cudaMemcpy(targetImage, image.getPixelVector(), imageWidth*imageHeight*sizeof(uchar), cudaMemcpyHostToDevice));
  checkCuda(cudaMalloc(&regImage, imageWidth*imageHeight*sizeof(uchar)));
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
