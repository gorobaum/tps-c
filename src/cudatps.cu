#include <iostream>
#include <cmath>

#include "cudatps.h"

#define MAXTHREADPBLOCK 1024

// Kernel definition
__global__ void tpsCuda(double* cudaImageCoord, int width, int heigth, float* solution, float* cudaKeyX, float* cudaKeyY, uint numOfKeys)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  double newCoord = solution[0] + x*solution[1] + y*solution[2];

  for (uint i = 0; i < numOfKeys; i++) {
    double r = (x-cudaKeyX[i])*(x-cudaKeyX[i]) + (y-cudaKeyY[i])*(y-cudaKeyY[i]);
    newCoord += r*log(r) * solution[i+3];
  }
  if (x*width+y < width*heigth)
    cudaImageCoord[x*width+y] = newCoord;
}

void tps::CudaTPS::callKernel(float *cudaSolution, double *imageCoord, dim3 threadsPerBlock, dim3 numBlocks) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  tpsCuda<<<numBlocks, threadsPerBlock>>>(cudaImageCoord, dimensions[1], dimensions[0], cudaSolution, cudaKeyX, cudaKeyY, targetKeypoints_.size());
  cudaDeviceSynchronize(); 
  cudaMemcpy(imageCoord, cudaImageCoord, dimensions[0]*dimensions[1]*sizeof(double), cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::cout << "Time = " << elapsedTime << " ms\n";
}

void tps::CudaTPS::run() {
	dimensions = registredImage.getDimensions(); 
  findSolutions();
	allocResources();
	allocCudaResources();

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(std::ceil(1.0*dimensions[0]/threadsPerBlock.x), std::ceil(1.0*dimensions[1]/threadsPerBlock.y));

  callKernel(cudaSolutionX, imageCoordX, threadsPerBlock, numBlocks);
  callKernel(cudaSolutionY, imageCoordY, threadsPerBlock, numBlocks);

  std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

  for (int x = 0; x < dimensions[0]; x++)
    for (int y = 0; y < dimensions[1]; y++) {
      double newX = imageCoordX[x*dimensions[1]+y];
      double newY = imageCoordY[x*dimensions[1]+y];
      uchar value = targetImage_.bilinearInterpolation<uchar>(newX, newY);
      registredImage.changePixelAt(x, y, value);
    }
  registredImage.save();

	freeResources();
	freeCudaResources();

	cudaDeviceReset();
}

void tps::CudaTPS::allocResources() {
  imageCoordX = (double*)malloc(dimensions[0]*dimensions[1]*sizeof(double));
  imageCoordY = (double*)malloc(dimensions[0]*dimensions[1]*sizeof(double));
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
  cudaMalloc(&cudaImageCoord, dimensions[0]*dimensions[1]*sizeof(double));
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
