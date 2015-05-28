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
__global__ void tpsCuda(unsigned char* cudaImage, unsigned char* cudaRegImage, float* colSolutions, float* rowSolutions, int width, int height, float* cudaKeyCol, float* cudaKeyRow, uint numOfKeys) {
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  double newCol = colSolutions[0] + x*colSolutions[1] + y*colSolutions[2];
  double newRow = rowSolutions[0] + x*rowSolutions[1] + y*rowSolutions[2];

  for (uint i = 0; i < numOfKeys; i++) {
    double r = (x-cudaKeyCol[i])*(x-cudaKeyCol[i]) + (y-cudaKeyRow[i])*(y-cudaKeyRow[i]);
    if (r != 0.0) {
      newCol += r*log(r) * colSolutions[i+3];
      newRow += r*log(r) * rowSolutions[i+3];
    }
  }
  if (x*height+y < width*height) {
    cudaRegImage[x*height+y] = cudaBilinearInterpolation(newCol, newRow, cudaImage, width, height);
  }
}

void startTimeRecord(cudaEvent_t *start, cudaEvent_t *stop) {
  checkCuda(cudaEventCreate(start));
  checkCuda(cudaEventCreate(stop));
  checkCuda(cudaEventRecord(*start, 0));
}

void showExecutionTimestartTimeRecord(cudaEvent_t *start, cudaEvent_t *stop, std::string output) {
  checkCuda(cudaEventRecord(*stop, 0));
  checkCuda(cudaEventSynchronize(*stop));
  float elapsedTime;
  checkCuda(cudaEventElapsedTime(&elapsedTime, *start, *stop));
  checkCuda(cudaEventDestroy(*start));
  checkCuda(cudaEventDestroy(*stop));
  std::cout << output << elapsedTime << " ms\n";
}

unsigned char* runTPSCUDA(tps::CudaMemory cm, int width, int height, int numberOfCPs) {
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(std::ceil(1.0*width/threadsPerBlock.x), std::ceil(1.0*height/threadsPerBlock.y));

  unsigned char* regImage = (unsigned char*)malloc(width*height*sizeof(unsigned char));

  for (int col = 0; col < width; col++)
    for (int row = 0; row < height; row++)
      regImage[col*height+row] = 0;

  cudaEvent_t start, stop;
  startTimeRecord(&start, &stop);

  tpsCuda<<<numBlocks, threadsPerBlock>>>(cm.getTargetImage(), cm.getRegImage(), cm.getSolutionCol(), cm.getSolutionRow(), 
                                          width, height, cm.getKeypointCol(), cm.getKeypointRow(), numberOfCPs);
  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaMemcpy(regImage, cm.getRegImage(), width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost));

  showExecutionTimestartTimeRecord(&start, &stop, "callKernel execution time = ");
  return regImage;
}
