#include <iostream>

#include "cudatps.h"

#define MAXTHREADPBLOCK 1024

// Kernel definition
__global__ void tpsCuda(float* cudaImageCoord, int dimension)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
	cudaImageCoord[x+dimension*y] = blockIdx.y;
}

void tps::CudaTPS::run() {
	findSolutions();
	std::vector<int> dimensions = registredImage.getDimensions();
	allocResources();
	allocCudaResources();

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(dimensions[0] / threadsPerBlock.x, dimensions[1] / threadsPerBlock.y);
	std::cout << "numBlocks[0] = " << numBlocks.x << std::endl;
	std::cout << "numBlocks[1] = " << numBlocks.y << std::endl;
	tpsCuda<<<numBlocks, threadsPerBlock>>>(cudaImageCoord, dimensions[0]);

	freeResources();
	freeCudaResources();

	for (int i = 0; i < 256*256; i++) {
		std::cout << "imageCoord[i] = " << imageCoord[i] << std::endl;		
	}
}

void tps::CudaTPS::allocResources() {
	std::vector<int> dimensions = registredImage.getDimensions();

	imageCoord = (float*)std::malloc(dimensions[0]*dimensions[1]*sizeof(float*));
}

void tps::CudaTPS::allocCudaResources() {
	std::vector<int> dimensions = registredImage.getDimensions();
	cudaMalloc(&cudaImageCoord, dimensions[0]*dimensions[1]*sizeof(float));
	// cudaMemcpy(cudaImageCoord, imageCoord, referenceKeypoints_.size()*dimensions[0]*dimensions[1]*sizeof(float), cudaMemcpyHostToDevice);
}

void tps::CudaTPS::freeResources() {

}

void tps::CudaTPS::freeCudaResources() {
	std::vector<int> dimensions = registredImage.getDimensions();
	cudaMemcpy(imageCoord, cudaImageCoord, dimensions[0]*dimensions[1]*sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
}
