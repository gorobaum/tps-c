#include <iostream>
#include <cassert>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_occupancy.h"

#include "cutps.h"

inline
cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        std::cout << "CUDA Runtime Error: \n" << cudaGetErrorString(result) << std::endl;
        assert(result == cudaSuccess);
    }
    return result;
}

// Kernel definition
__device__ double cudaGetPixel(int x, int y, unsigned char* image, int width, int height) {
  if (x > width-1 || x < 0) return 0;
  if (y > height-1 || y < 0) return 0;
  return image[x*height+y];
}

// Kernel definition
__device__ double cudaBilinearInterpolation(double col, double row, unsigned char* image, int width, int height) {
  int u = trunc(col);
  int v = trunc(row);

  unsigned char pixelOne = cudaGetPixel(u, v, image, width, height);
  unsigned char pixelTwo = cudaGetPixel(u+1, v, image, width, height);
  unsigned char pixelThree = cudaGetPixel(u, v+1, image, width, height);
  unsigned char pixelFour = cudaGetPixel(u+1, v+1, image, width, height);

  double interpolation = (u+1-col)*(v+1-row)*pixelOne
                        + (col-u)*(v+1-row)*pixelTwo 
                        + (u+1-col)*(row-v)*pixelThree
                        + (col-u)*(row-v)*pixelFour;
  return interpolation;
}

// Kernel definition
__global__ void cudaRegistredImage(double* cudaImageCoordX, double* cudaImageCoordY, unsigned char* cudaImage, unsigned char* cudaRegImage, int width, int height) {
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

void runTPSCUDA(tps::CudaMemory cm, int width, int height, int numberOfCPs) {
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(std::ceil(1.0*width/threadsPerBlock.x), std::ceil(1.0*height/threadsPerBlock.y));

  runTPSCUDAForCoord(cm.getCoordinateCol(), cm.getSolutionCol(), threadsPerBlock, numBlocks, width, height,
              cm.getKeypointCol(), cm.getKeypointRow(), numberOfCPs);
  runTPSCUDAForCoord(cm.getCoordinateRow(), cm.getSolutionRow(), threadsPerBlock, numBlocks, width, height,
              cm.getKeypointCol(), cm.getKeypointRow(), numberOfCPs);
}

void runTPSCUDAForCoord(double* cudaImageCoord, float* cudaSolution, dim3 threadsPerBlock, dim3 numBlocks, int width, int height,
                float* keypointCol, float* keypointRow, int numberOfCP) {
  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));
  checkCuda(cudaEventRecord(start, 0));
  tpsCuda<<<numBlocks, threadsPerBlock>>>(cudaImageCoord, width, height, cudaSolution, keypointCol, keypointRow, numberOfCP);
  cudaDeviceSynchronize(); 
  checkCuda(cudaEventRecord(stop, 0));
  checkCuda(cudaEventSynchronize(stop));
  float elapsedTime;
  checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
  checkCuda(cudaEventDestroy(start));
  checkCuda(cudaEventDestroy(stop));
  std::cout << "callKernel execution time = " << elapsedTime << " ms\n";
}

unsigned char* runRegImage(tps::CudaMemory cm, int width, int height) {
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(std::ceil(1.0*width/threadsPerBlock.x), std::ceil(1.0*height/threadsPerBlock.y));

  unsigned char* regImage = (unsigned char*)malloc(width*height*sizeof(unsigned char));

  for (int col = 0; col < width; col++)
    for (int row = 0; row < height; row++)
      regImage[col*height+row] = 0;

  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start));
  checkCuda(cudaEventCreate(&stop));
  checkCuda(cudaEventRecord(start, 0));
  cudaRegistredImage<<<numBlocks, threadsPerBlock>>>(cm.getCoordinateCol(), cm.getCoordinateRow(), cm.getTargetImage(), cm.getRegImage(), width, height);
  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaMemcpy(regImage, cm.getRegImage(), width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost));
  checkCuda(cudaEventRecord(stop, 0));
  checkCuda(cudaEventSynchronize(stop));
  float elapsedTime;
  checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop));
  checkCuda(cudaEventDestroy(start));
  checkCuda(cudaEventDestroy(stop));
  std::cout << "cudaRegistredImage execution time = " << elapsedTime << " ms\n";
  return regImage;
}
