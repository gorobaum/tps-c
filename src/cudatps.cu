#include <iostream>
#include <cmath>

#include "cudatps.h"

#define MAXTHREADPBLOCK 1024

// Kernel definition
__global__ void tpsCuda(float* cudaImageCoord, int width, float* solution, float* cudaKeyX, float* cudaKeyY, uint numOfKeys)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  float newCoord = solution[0] + x*solution[1] + y*solution[2];

  for (uint i = 0; i < numOfKeys; i++) {
    float r = (x-cudaKeyX[i])*(x-cudaKeyX[i]) + (y-cudaKeyY[i])*(y-cudaKeyY[i]);
    newCoord += r*log(r) * solution[i+3];
  }

  cudaImageCoord[x*width+y] = newCoord;
}


void tps::CudaTPS::run() {
	findSolutions();
	dimensions = registredImage.getDimensions();
	allocResources();
	allocCudaResources();

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(dimensions[0]/threadsPerBlock.x, dimensions[1]/threadsPerBlock.y);


  std::cout << numBlocks.x << std::endl;
  std::cout << numBlocks.y << std::endl;
  std::cout << dimensions[0] << std::endl;
  std::cout << dimensions[1] << std::endl;
	tpsCuda<<<numBlocks, threadsPerBlock>>>(cudaImageCoord, dimensions[1], cudaSolutionX, cudaKeyX, cudaKeyY, targetKeypoints_.size());
  cudaThreadSynchronize();
  cudaMemcpy(imageCoordX, cudaImageCoord, dimensions[0]*dimensions[1]*sizeof(float), cudaMemcpyDeviceToHost);
  tpsCuda<<<numBlocks, threadsPerBlock>>>(cudaImageCoord, dimensions[1], cudaSolutionY, cudaKeyX, cudaKeyY, targetKeypoints_.size());
  cudaThreadSynchronize();
  cudaMemcpy(imageCoordY, cudaImageCoord, dimensions[0]*dimensions[1]*sizeof(float), cudaMemcpyDeviceToHost);

  for (int x = 0; x < dimensions[0]; x++)
    for (int y = 0; y < dimensions[1]; y++) {
      float newX = imageCoordX[x*dimensions[1]+y];
      float newY = imageCoordY[x*dimensions[1]+y];
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
  createCudaSolution();
  createCudaKeyPoint();
}

void tps::CudaTPS::createCudaSolution() {
  floatSolX = (float*)malloc((targetKeypoints_.size()+3)*sizeof(float));
  floatSolY = (float*)malloc((targetKeypoints_.size()+3)*sizeof(float));
  for (uint i = 0; i < (targetKeypoints_.size()+3); i++) {
    floatSolX[i] = solutionX.at<float>(i);
    floatSolY[i] = solutionY.at<float>(i);
  }
}

void tps::CudaTPS::createCudaKeyPoint() {
  floatKeyX = (float*)malloc(targetKeypoints_.size()*sizeof(float));
  floatKeyY = (float*)malloc(targetKeypoints_.size()*sizeof(float));
  for (uint i = 0; i < targetKeypoints_.size(); i++) {
    floatKeyX[i] = referenceKeypoints_[i].x;
    floatKeyY[i] = referenceKeypoints_[i].y;
  }
}

void tps::CudaTPS::allocCudaResources() {
  cudaMalloc(&cudaImageCoord, dimensions[0]*dimensions[1]*sizeof(float));
  cudaMalloc(&cudaSolutionX, (targetKeypoints_.size()+3)*sizeof(float));
  cudaMalloc(&cudaSolutionY, (targetKeypoints_.size()+3)*sizeof(float));
  cudaMemcpy(cudaSolutionX, floatSolX, (targetKeypoints_.size()+3)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaSolutionY, floatSolY, (targetKeypoints_.size()+3)*sizeof(float), cudaMemcpyHostToDevice);

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
