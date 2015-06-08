#include "cudamemory.h"

#include <cassert>

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
  allocCudaSolution();
  allocCudaKeypoints();
  allocCudaImagePixels(image);
}

void tps::CudaMemory::allocCudaSolution() {
  checkCuda(cudaMalloc(&solutionCol, systemDim*sizeof(float)));
  checkCuda(cudaMalloc(&solutionRow, systemDim*sizeof(float)));
  checkCuda(cudaMalloc(&solutionSlice, systemDim*sizeof(float)));
}

void tps::CudaMemory::allocCudaKeypoints() {
  float* hostKeypointCol = (float*)malloc(referenceKeypoints_.size()*sizeof(float));
  float* hostKeypointRow = (float*)malloc(referenceKeypoints_.size()*sizeof(float));
  float* hostKeypointSlice = (float*)malloc(referenceKeypoints_.size()*sizeof(float));
  for (uint i = 0; i < referenceKeypoints_.size(); i++) {
    hostKeypointCol[i] = referenceKeypoints_[i][0];
    hostKeypointRow[i] = referenceKeypoints_[i][1];
    hostKeypointSlice[i] = referenceKeypoints_[i][2];
  }
  checkCuda(cudaMalloc(&keypointCol, numberOfCps*sizeof(float)));
  checkCuda(cudaMemcpy(keypointCol, hostKeypointCol, numberOfCps*sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMalloc(&keypointRow, numberOfCps*sizeof(float)));
  cudaMemcpy(keypointRow, hostKeypointRow, numberOfCps*sizeof(float), cudaMemcpyHostToDevice);
  checkCuda(cudaMalloc(&keypointSlice, numberOfCps*sizeof(float)));
  cudaMemcpy(keypointSlice, hostKeypointSlice, numberOfCps*sizeof(float), cudaMemcpyHostToDevice);
  free(hostKeypointCol);
  free(hostKeypointRow);
  free(hostKeypointSlice);
}

void tps::CudaMemory::allocCudaImagePixels(tps::Image& image) {
  checkCuda(cudaMalloc(&targetImage, imageSize*sizeof(unsigned char)));
  checkCuda(cudaMemcpy(targetImage, image.getPixelVector(), imageSize*sizeof(unsigned char), cudaMemcpyHostToDevice));
  checkCuda(cudaMalloc(&regImage, imageSize*sizeof(unsigned char)));
}

std::vector<float> tps::CudaMemory::getHostSolCol() {
  return cudaToHost(solutionCol);
}

std::vector<float> tps::CudaMemory::getHostSolRow() {
  return cudaToHost(solutionRow);
}

std::vector<float> tps::CudaMemory::getHostSolSlice() {
  return cudaToHost(solutionSlice);
}

std::vector<float> tps::CudaMemory::cudaToHost(float *cudaMemory) {
    float *hostSolPointer = (float*)malloc(systemDim*sizeof(float));
    cudaMemcpy(hostSolPointer, cudaMemory, systemDim*sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> hostSol;
    for (int i =0; i < systemDim; i++) hostSol.push_back(hostSolPointer[i]);
    delete(hostSolPointer);
    return hostSol;
}

double tps::CudaMemory::memoryEstimation() {
  int floatSize = sizeof(float);
  int doubleSize = sizeof(double);
  int ucharSize = sizeof(unsigned char);

  double solutionsMemory = 3.0*systemDim*floatSize/(1024*1024);
  double keypointsMemory = 3.0*numberOfCps*floatSize/(1024*1024);
  double pixelsMemory = 2.0*imageSize*ucharSize/(1024*1024);

  double totalMemory = solutionsMemory+keypointsMemory+pixelsMemory;
  return totalMemory;
}

void tps::CudaMemory::freeMemory() {
  cudaFree(solutionCol);
  cudaFree(solutionRow);
  cudaFree(solutionSlice);
  cudaFree(keypointCol);
  cudaFree(keypointRow);
  cudaFree(keypointSlice);
  cudaFree(targetImage);
  cudaFree(regImage);
}
