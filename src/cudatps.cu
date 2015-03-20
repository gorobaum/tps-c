#include "cudatps.h"

#define MAXTHREADPBLOCK 1024

// Kernel definition
__global__ void tpsCuda(float* imageCoord, size_t pitch, int width, int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
	float* row = (float*)((char*)imageCoord + x*pitch);
	row[y] = 1.0;
}

void tps::CudaTPS::run() {
	findSolutions();

	std::vector<int> dimensions = registredImage.getDimensions();
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(dimensions[0] / threadsPerBlock.x, dimensions[1] / threadsPerBlock.y);

}

void tps::CudaTPS::allocResources() {
	std::vector<int> dimensions = registredImage.getDimensions();

	imageCoord = (float*)std::malloc(referenceKeypoints_.size()*dimensions[0]*dimensions[1]*sizeof(float**));
}

void tps::CudaTPS::allocCudaResources() {
	std::vector<int> dimensions = registredImage.getDimensions();
	size_t pitch;
	cudaMallocPitch(&imageCoord, &pitch, dimensions[1]*sizeof(float), dimensions[0]);
}

void tps::CudaTPS::freeResources() {

}

void tps::CudaTPS::freeCudaResources() {

}
