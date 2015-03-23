#include <iostream>
#include <cmath>

#include "cudatps.h"

#define MAXTHREADPBLOCK 1024

// __global__ void tpsCuda(uchar* cudaImageCoord, size_t pitch, int dimension)
// {
//   int x = blockDim.x*blockIdx.x + threadIdx.x;
//   int y = blockDim.y*blockIdx.y + threadIdx.y;
//   uchar* row = ((uchar*)cudaImageCoord + y*pitch);
//   row[x] = 255;
// }

// Kernel definition
__global__ void tpsCuda(float* cudaImageCoord, size_t pitch, float* solution, float* refKeyX, float* refKeyY, int numOfKeys)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  float newCoord = solution[0] + x*solution[1] + y*solution[2];

  for (uint i = 0; i < numOfKeys; i++) {
    float r = (x-refKeyX[i])*(x-refKeyX[i]) + (y-refKeyY[i])*(y-refKeyY[i]);
    if (r != 0.0) {
      newCoord += r*log(r) * solution[i+3];
    }
  }
  float* row = ((float*)cudaImageCoord + y*pitch);
  cudaImageCoord[x] = newCoord;
}

void tps::CudaTPS::run() {
	findSolutions();
	dimensions = registredImage.getDimensions();
	allocResources();
	allocCudaResources();

  int tpb = (1024 > dimensions[0] ? dimensions[0] : 1024);
	dim3 threadsPerBlock(tpb, 1);
	dim3 numBlocks( std::ceil(dimensions[0]*dimensions[1]/threadsPerBlock.x) , 1);

	tpsCuda<<<numBlocks, threadsPerBlock>>>(cudaImageCoord, pitch, dimensions[0]);
  cudaMemcpy(imageCoordX, cudaImageCoord, dimensions[0]*dimensions[1]*sizeof(uchar), cudaMemcpyDeviceToHost);
  tpsCuda<<<numBlocks, threadsPerBlock>>>(cudaImageCoord, pitch, dimensions[0]);
  cudaMemcpy(imageCoordY, cudaImageCoord, dimensions[0]*dimensions[1]*sizeof(uchar), cudaMemcpyDeviceToHost);

	freeResources();
	freeCudaResources();

	cudaDeviceReset();
}

void tps::CudaTPS::allocResources() {
  imageCoordX = (float*)malloc(dimensions[0]*dimensions[1]*sizeof(float));
  imageCoordY = (float*)malloc(dimensions[0]*dimensions[1]*sizeof(float));
  pSolutionX = solutionPointer(solutionX);
  pSolutionY = solutionPointer(solutionY);
  refKeyX = (float*)malloc(targetKeypoints_.size()*sizeof(float));
  refKeyY = (float*)malloc(targetKeypoints_.size()*sizeof(float));
}

float* tps::CudaTPS::solutionPointer(cv::Mat solution) {
  float* pointer = (float*)malloc(targetKeypoints_.size()*sizeof(float));
  for (uint i = 0; i < targetKeypoints_.size(); i++)
    pointer[i] = solution.at<float>(i);
  return pointer;
}

void tps::CudaTPS::allocCudaResources() {
  cudaMallocPitch(&cudaImageCoord, &pitch, dimensions[1]*sizeof(uchar), dimensions[0]);
}

void tps::CudaTPS::freeResources() {

}

void tps::CudaTPS::freeCudaResources() {
	cudaDeviceSynchronize();
}
