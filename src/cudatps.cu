#include <iostream>
#include <cmath>

#include "cudatps.h"

#define MAXTHREADPBLOCK 1024

// Kernel definition
__device__ double cudaGetPixel(int x, int y, uchar* image, int width, int height) {
  if (x > width-1 || x < 0) return 0;
  if (y > height-1 || y < 0) return 0;
  return image[x*height+y];
}

// Kernel definition
__device__ double cudaBilinearInterpolation(double col, double row, uchar* image, int width, int height) {
  int u = trunc(col);
  int v = trunc(row);

  uchar pixelOne = cudaGetPixel(u, v, image, width, height);
  uchar pixelTwo = cudaGetPixel(u+1, v, image, width, height);
  uchar pixelThree = cudaGetPixel(u, v+1, image, width, height);
  uchar pixelFour = cudaGetPixel(u+1, v+1, image, width, height);

  double interpolation = (u+1-col)*(v+1-row)*pixelOne
                        + (col-u)*(v+1-row)*pixelTwo 
                        + (u+1-col)*(row-v)*pixelThree
                        + (col-u)*(row-v)*pixelFour;
  return interpolation;
}

// Kernel definition
__global__ void cudaRegistredImage(double* cudaImageCoordX, double* cudaImageCoordY, uchar* cudaImage, uchar* cudaRegImage, int width, int height) {
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;

  double newX = cudaImageCoordX[x*height+y];
  double newY = cudaImageCoordY[x*height+y];
  cudaRegImage[x*height+y] = cudaBilinearInterpolation(newX, newY, cudaImage, width, height);
}

// Kernel definition
__global__ void tpsCuda(double* cudaImageCoord, int width, int height, float* solution, float* cudaKeyCol, float* cudaKeyRow, uint numOfKeys)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  double newCoord = solution[0] + x*solution[1] + y*solution[2];

  for (uint i = 0; i < numOfKeys; i++) {
    double r = (x-cudaKeyCol[i])*(x-cudaKeyCol[i]) + (y-cudaKeyRow[i])*(y-cudaKeyRow[i]);
    if (r != 0.0) newCoord += r*log(r) * solution[i+3];
  }
  if (x*height+y < width*height)
    cudaImageCoord[x*height+y] = newCoord;
}

void tps::CudaTPS::callKernel(double *cudaImageCoord, float *cudaSolution, dim3 threadsPerBlock, dim3 numBlocks) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  tpsCuda<<<numBlocks, threadsPerBlock>>>(cudaImageCoord, width, height, cudaSolution, cm_.getKeypointCol(), cm_.getKeypointRow(), targetKeypoints_.size());
  cudaDeviceSynchronize(); 
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::cout << "Time = " << elapsedTime << " ms\n";
}

void tps::CudaTPS::run() {
	allocResources();
  cudalienarSolver.solveLinearSystems(cm_);
  allocCudaResources();

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(std::ceil(1.0*width/threadsPerBlock.x), std::ceil(1.0*height/threadsPerBlock.y));

  callKernel(cm_.getCoordinateCol(), cm_.getSolutionCol(), threadsPerBlock, numBlocks);
  callKernel(cm_.getCoordinateRow(), cm_.getSolutionRow(), threadsPerBlock, numBlocks);

  // std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  cudaRegistredImage<<<numBlocks, threadsPerBlock>>>(cm_.getCoordinateCol(), cm_.getCoordinateRow(), cm_.getTargetImage(), cm_.getRegImage(), width, height);
  cudaMemcpy(regImage, cm_.getRegImage(), width*height*sizeof(uchar), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::cout << "Time = " << elapsedTime << " ms\n";

  registredImage.setPixelVector(regImage);
  registredImage.save(outputName_);

	freeResources();
	freeCudaResources();

	cudaDeviceReset();
}

void tps::CudaTPS::allocResources() {
  regImage = (uchar*)malloc(width*height*sizeof(uchar));
  for (int col = 0; col < width; col++)
    for (int row = 0; row < height; row++)
      regImage[col*height+row] = 0;
}

void tps::CudaTPS::allocCudaResources() {
  cm_.allocCudaCoord();
  cm_.allocCudaKeypoints(referenceKeypoints_);
  cm_.allocCudaImagePixels(targetImage_);
}

void tps::CudaTPS::freeResources() {
  free(floatKeyCol);
  free(floatKeyRow);
}

void tps::CudaTPS::freeCudaResources() {
  size_t avail;
  size_t total;
  cudaMemGetInfo( &avail, &total );
  size_t used = total - avail;
  std::cout << "Device memory used after kernel execution: " << used/(1024*1024) << "MB" << std::endl;
  cm_.freeMemory();
	cudaDeviceSynchronize();
}
