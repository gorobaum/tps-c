#include <iostream>

#include "cudatps.h"

#define MAXTHREADPBLOCK 1024

// Kernel definition
__global__ void tpsCuda(float* cudaImageCoord)
{
  int x = threadIdx.x;
	cudaImageCoord[x] = 1.0;
}

void tps::CudaTPS::run() {
	findSolutions();
	std::vector<int> dimensions = registredImage.getDimensions();
	allocResources();
	allocCudaResources();

	dim3 threadsPerBlock(256, 256);
	std::cout << "numBlocks[0] = " << dimensions[0] / threadsPerBlock.x << std::endl;
	std::cout << "numBlocks[1] = " << dimensions[1] / threadsPerBlock.y << std::endl;
	dim3 numBlocks(dimensions[0] / threadsPerBlock.x, dimensions[1] / threadsPerBlock.y);
	tpsCuda<<<1, threadsPerBlock>>>(cudaImageCoord);

	freeResources();
	freeCudaResources();

	cudaDeviceSynchronize();
	std::cout << "imageCoord[0] = " << imageCoord[0] << std::endl;
}

void tps::CudaTPS::allocResources() {
	std::vector<int> dimensions = registredImage.getDimensions();

	imageCoord = (float*)std::malloc(referenceKeypoints_.size()*dimensions[0]*dimensions[1]*sizeof(float*));
}

void tps::CudaTPS::allocCudaResources() {
	std::vector<int> dimensions = registredImage.getDimensions();
	cudaMalloc(&cudaImageCoord, referenceKeypoints_.size()*dimensions[0]*dimensions[1]*sizeof(float));
	// cudaMemcpy(cudaImageCoord, imageCoord, referenceKeypoints_.size()*dimensions[0]*dimensions[1]*sizeof(float), cudaMemcpyHostToDevice);
}

void tps::CudaTPS::freeResources() {

}

void tps::CudaTPS::freeCudaResources() {
	std::vector<int> dimensions = registredImage.getDimensions();
	cudaMemcpy(imageCoord, cudaImageCoord, referenceKeypoints_.size()*dimensions[0]*dimensions[1]*sizeof(float), cudaMemcpyDeviceToHost);
}
