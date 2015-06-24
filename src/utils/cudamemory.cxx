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
  checkCuda(cudaMalloc(&solutionX, systemDim*sizeof(float)));
  checkCuda(cudaMalloc(&solutionY, systemDim*sizeof(float)));
  checkCuda(cudaMalloc(&solutionZ, systemDim*sizeof(float)));
}

void tps::CudaMemory::allocCudaKeypoints() {
  float* hostKeypointX = (float*)malloc(referenceKeypoints_.size()*sizeof(float));
  float* hostKeypointY = (float*)malloc(referenceKeypoints_.size()*sizeof(float));
  float* hostKeypointZ = (float*)malloc(referenceKeypoints_.size()*sizeof(float));
  for (uint i = 0; i < referenceKeypoints_.size(); i++) {
    hostKeypointX[i] = referenceKeypoints_[i][0];
    hostKeypointY[i] = referenceKeypoints_[i][1];
    hostKeypointZ[i] = referenceKeypoints_[i][2];
  }

  checkCuda(cudaMalloc(&keypointX, numberOfCps*sizeof(float)));
  checkCuda(cudaMemcpy(keypointX, hostKeypointX, numberOfCps*sizeof(float), cudaMemcpyHostToDevice));

  checkCuda(cudaMalloc(&keypointY, numberOfCps*sizeof(float)));
  checkCuda(cudaMemcpy(keypointY, hostKeypointY, numberOfCps*sizeof(float), cudaMemcpyHostToDevice));

  checkCuda(cudaMalloc(&keypointZ, numberOfCps*sizeof(float)));
  checkCuda(cudaMemcpy(keypointZ, hostKeypointZ, numberOfCps*sizeof(float), cudaMemcpyHostToDevice));

  free(hostKeypointX);
  free(hostKeypointY);
  free(hostKeypointZ);
}

void tps::CudaMemory::allocCudaImagePixels(tps::Image& image) {
  checkCuda(cudaMalloc(&targetImage, imageSize*sizeof(short)));
  checkCuda(cudaMemcpy(targetImage, image.getPixelVector(), imageSize*sizeof(short), cudaMemcpyHostToDevice));
  checkCuda(cudaMalloc(&regImage, imageSize*sizeof(short)));
}

std::vector<float> tps::CudaMemory::getHostSolX() {
  return cudaToHost(solutionX);
}

std::vector<float> tps::CudaMemory::getHostSolY() {
  return cudaToHost(solutionY);
}

std::vector<float> tps::CudaMemory::getHostSolZ() {
  return cudaToHost(solutionZ);
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
  int ucharSize = sizeof(short);

  double solutionsMemory = 3.0*systemDim*floatSize/(1024*1024);
  double keypointsMemory = 3.0*numberOfCps*floatSize/(1024*1024);
  double pixelsMemory = 2.0*imageSize*ucharSize/(1024*1024);

  double totalMemory = solutionsMemory+keypointsMemory+pixelsMemory;
  return totalMemory;
}

void tps::CudaMemory::freeMemory() {
  cudaFree(solutionX);
  cudaFree(solutionY);
  cudaFree(solutionZ);
  cudaFree(keypointX);
  cudaFree(keypointY);
  cudaFree(keypointZ);
  cudaFree(targetImage);
  cudaFree(regImage);
}
