#include "cudamemory.h"

#include <cassert>
#include <cstring>

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
  std::vector<int> dimensions = image.getDimensions();

  cudaExtent volumeExtent = make_cudaExtent(dimensions[0], dimensions[1], dimensions[2]);

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

  cudaMalloc3DArray(&cuArray, &channelDesc, volumeExtent, 0);

  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr   = make_cudaPitchedPtr((void*)image.getFloatPixelVector(), volumeExtent.width*sizeof(float), volumeExtent.width, volumeExtent.height);
  copyParams.dstArray = cuArray;
  copyParams.extent   = volumeExtent;
  copyParams.kind     = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&copyParams);

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeBorder;
  texDesc.addressMode[1]   = cudaAddressModeBorder;
  texDesc.addressMode[2]   = cudaAddressModeBorder;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  // Create texture object
  texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

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

float* vectorToPointer(std::vector<float> input) {
  float* output = (float*)malloc(input.size()*sizeof(float));
  for (int i = 0; i < input.size(); i++)
    output[i] = input[i];
  return output;
}

void tps::CudaMemory::setSolutionX(std::vector<float> solution) {
  float* solPointer = vectorToPointer(solution);
  checkCuda(cudaMemcpy(solutionX, solPointer, systemDim*sizeof(float), cudaMemcpyHostToDevice));
  free(solPointer);
}

void tps::CudaMemory::setSolutionY(std::vector<float> solution) {
  float* solPointer = vectorToPointer(solution);
  checkCuda(cudaMemcpy(solutionY, solPointer, systemDim*sizeof(float), cudaMemcpyHostToDevice));
  free(solPointer);
}

void tps::CudaMemory::setSolutionZ(std::vector<float> solution) {
  float* solPointer = vectorToPointer(solution);
  checkCuda(cudaMemcpy(solutionZ, solPointer, systemDim*sizeof(float), cudaMemcpyHostToDevice));
  free(solPointer);
}

std::vector<float> tps::CudaMemory::cudaToHost(float *cudaMemory) {
    float *hostSolPointer = (float*)malloc(systemDim*sizeof(float));
    cudaMemcpy(hostSolPointer, cudaMemory, systemDim*sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> hostSol;
    for (int i =0; i < systemDim; i++) 
      hostSol.push_back(hostSolPointer[i]);
    delete(hostSolPointer);
    return hostSol;
}

double tps::CudaMemory::memoryEstimation() {
  int floatSize = sizeof(float);
  int doubleSize = sizeof(double);
  int ucharSize = sizeof(short);

  double solutionsMemory = 3.0*systemDim*floatSize/(1024*1024);
  // std::cout << "solutionsMemory = " << solutionsMemory << std::endl;
  double keypointsMemory = 3.0*numberOfCps*floatSize/(1024*1024);
  // std::cout << "keypointsMemory = " << keypointsMemory << std::endl;
  double pixelsMemory = 2.0*imageSize*ucharSize/(1024*1024);
  // std::cout << "pixelsMemory = " << pixelsMemory << std::endl;

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
