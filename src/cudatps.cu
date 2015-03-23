#include <iostream>
#include <cmath>

#include "cudatps.h"

#define MAXTHREADPBLOCK 1024

// Kernel definition
__global__ void tpsCuda(float* cudaImageCoord, size_t pitch, float* solution, float* cudaKeyX, float* cudaKeyY, uint numOfKeys)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  float newCoord = solution[0] + x*solution[1] + y*solution[2];

  for (uint i = 0; i < numOfKeys; i++) {
    float r = (x-cudaKeyX[i])*(x-cudaKeyX[i]) + (y-cudaKeyY[i])*(y-cudaKeyY[i]);
    if (r != 0.0) {
      newCoord += r*log(r) * solution[i+3];
    }
  }

  float* row = ((float*)cudaImageCoord + y*pitch);
  row[x] = 1.0;
}

void tps::CudaTPS::run() {
	findSolutions();
	dimensions = registredImage.getDimensions();
	allocResources();
	allocCudaResources();

  int tpb = (1024 > dimensions[0] ? dimensions[0] : 1024);
	dim3 threadsPerBlock(tpb, 1);
	dim3 numBlocks( std::ceil(dimensions[0]*dimensions[1]/threadsPerBlock.x) , 1);

	tpsCuda<<<numBlocks, threadsPerBlock>>>(cudaImageCoord, pitch, cudaSolutionX, cudaKeyX, cudaKeyY, targetKeypoints_.size());
  cudaMemcpy(imageCoordX, cudaImageCoord, dimensions[0]*dimensions[1]*sizeof(uchar), cudaMemcpyDeviceToHost);
  tpsCuda<<<numBlocks, threadsPerBlock>>>(cudaImageCoord, pitch, cudaSolutionY, cudaKeyX, cudaKeyY, targetKeypoints_.size());
  cudaMemcpy(imageCoordY, cudaImageCoord, dimensions[0]*dimensions[1]*sizeof(uchar), cudaMemcpyDeviceToHost);

  for (int x = 0; x < dimensions[0]; x++)
    for (int y = 0; y < dimensions[1]; y++) {
      float newX = imageCoordX[x*dimensions[0]+y];
      float newY = imageCoordY[x*dimensions[0]+y];
      std::cout << "[" << x << "][" << y << "] = (" << newX << ")(" << newY << ")" << std::endl;
      uchar value = targetImage_.bilinearInterpolation<uchar>(newX, newY);
      registredImage.changePixelAt(x, y, value);
    }
  registredImage.save();

	freeResources();
	freeCudaResources();

	cudaDeviceReset();
}

void tps::CudaTPS::allocResources() {
  imageCoordX = (float*)malloc(dimensions[0]*dimensions[1]*sizeof(float));
  imageCoordY = (float*)malloc(dimensions[0]*dimensions[1]*sizeof(float));
}

void tps::CudaTPS::createCudaSolution() {
  floatSolX = (float*)malloc(targetKeypoints_.size()*sizeof(float));
  floatSolY = (float*)malloc(targetKeypoints_.size()*sizeof(float));
  for (uint i = 0; i < targetKeypoints_.size(); i++) {
    floatSolX[i] = solutionX.at<float>(i);
    floatSolY[i] = solutionY.at<float>(i);
  }
}

void tps::CudaTPS::createCudaKeyPoint() {
  floatKeyX = (float*)malloc(targetKeypoints_.size()*sizeof(float));
  floatKeyY = (float*)malloc(targetKeypoints_.size()*sizeof(float));
  for (uint i = 0; i < referenceKeypoints_.size(); i++) {
    floatKeyX[i] = referenceKeypoints_[i].x;
    floatKeyY[i] = referenceKeypoints_[i].y;
  }
}

void tps::CudaTPS::allocCudaResources() {
  cudaMallocPitch(&cudaImageCoord, &pitch, dimensions[1]*sizeof(uchar), dimensions[0]);
  cudaMalloc(&cudaSolutionX, targetKeypoints_.size()*sizeof(float));
  cudaMalloc(&cudaSolutionY, targetKeypoints_.size()*sizeof(float));
  cudaMemcpy(cudaSolutionX, floatSolX, targetKeypoints_.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaSolutionY, floatSolY, targetKeypoints_.size()*sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&cudaKeyX, targetKeypoints_.size()*sizeof(float));
  cudaMalloc(&cudaKeyY, targetKeypoints_.size()*sizeof(float));
  cudaMemcpy(cudaKeyX, floatKeyX, targetKeypoints_.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaKeyY, floatKeyY, targetKeypoints_.size()*sizeof(float), cudaMemcpyHostToDevice);
}

void tps::CudaTPS::freeResources() {
  free(imageCoordX);
  free(imageCoordY);
  free(floatSolX);
  free(floatSolY);
  free(floatKeyX);
  free(floatKeyY);
}

void tps::CudaTPS::freeCudaResources() {
  cudaFree(cudaImageCoord);
  cudaFree(cudaSolutionX);
  cudaFree(cudaSolutionY);
  cudaFree(cudaKeyX);
  cudaFree(cudaKeyY);
	cudaDeviceSynchronize();
}
