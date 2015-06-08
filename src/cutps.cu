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
__device__ double cudaGetPixel(int x, int y, int z, unsigned char* image, int width, int height, int slices) {
  if (x > width-1 || x < 0) return 0;
  if (y > height-1 || y < 0) return 0;
  if (z > slices-1 || z < 0) return 0;
  return image[z*height*width+x*height+y];
}

// Kernel definition
__device__ double cudaTrilinearInterpolation(double col, double row, double slice, unsigned char* image, 
                                            int width, int height, int slices) {
  int u = trunc(col);
  int v = trunc(row);
  int w = trunc(slice);

  int xd = (col - u);
  int yd = (row - v);
  int zd = (slice - w);

  double c00 = cudaGetPixel(u, v, w, image, width, height, slices)*(1-xd);
  c00 += cudaGetPixel(u+1, v, w, image, width, height, slices)*xd;
  double c10 = cudaGetPixel(u, v+1, w, image, width, height, slices)*(1-xd);
  c10 += cudaGetPixel(u+1, v+1, w, image, width, height, slices)*xd;
  double c01 = cudaGetPixel(u, v, w+1, image, width, height, slices)*(1-xd);
  c01 += cudaGetPixel(u+1, v, w+1, image, width, height, slices)*xd;
  double c11 = cudaGetPixel(u, v+1, w+1, image, width, height, slices);
  c11 += cudaGetPixel(u+1, v+1, w+1, image, width, height, slices)*xd;

  double c0 = c00*(1-yd)+c10*yd;
  double c1 = c01*(1-yd)+c11*yd;

  return c0*(1-zd)+c1*zd;
}

// Kernel definition
__global__ void tpsCuda(unsigned char* cudaImage, unsigned char* cudaRegImage, float* colSolutions, float* rowSolutions, 
                        float* sliceSolutions, int width, int height, int slices, float* cudaKeyCol, float* cudaKeyRow, 
                        float* cudaKeySlice, uint numOfKeys) {
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  int z = blockDim.z*blockIdx.z + threadIdx.z;

  double newCol = colSolutions[0] + x*colSolutions[1] + y*colSolutions[2];
  double newRow = rowSolutions[0] + x*rowSolutions[1] + y*rowSolutions[2];
  double newSlice = sliceSolutions[0] + x*sliceSolutions[1] + y*sliceSolutions[2];

  for (uint i = 0; i < numOfKeys; i++) {
    double r = (x-cudaKeyCol[i])*(x-cudaKeyCol[i]) + (y-cudaKeyRow[i])*(y-cudaKeyRow[i]) + (z-cudaKeySlice[i])*(z-cudaKeySlice[i]);
    if (r != 0.0) {
      newCol += r*log(r) * colSolutions[i+3];
      newRow += r*log(r) * rowSolutions[i+3];
      newSlice += r*log(r) * sliceSolutions[i+3];
    }
  }
  if (z*height*width+x*height+y < width*height*slices) {
    cudaRegImage[z*height*width+x*height+y] = cudaTrilinearInterpolation(newCol, newRow, newSlice, cudaImage, width, height, slices);
  }
}

void startTimeRecord(cudaEvent_t *start, cudaEvent_t *stop) {
  checkCuda(cudaEventCreate(start));
  checkCuda(cudaEventCreate(stop));
  checkCuda(cudaEventRecord(*start, 0));
}

void showExecutionTime(cudaEvent_t *start, cudaEvent_t *stop, std::string output) {
  checkCuda(cudaEventRecord(*stop, 0));
  checkCuda(cudaEventSynchronize(*stop));
  float elapsedTime;
  checkCuda(cudaEventElapsedTime(&elapsedTime, *start, *stop));
  checkCuda(cudaEventDestroy(*start));
  checkCuda(cudaEventDestroy(*stop));
  std::cout << output << elapsedTime << " ms\n";
}

unsigned char* runTPSCUDA(tps::CudaMemory cm, int width, int height, int slices, int numberOfCPs) {
  dim3 threadsPerBlock(10, 10, 10);
  dim3 numBlocks(std::ceil(1.0*width/threadsPerBlock.x), 
                 std::ceil(1.0*height/threadsPerBlock.y), 
                 std::ceil(1.0*slices/threadsPerBlock.z));

  unsigned char* regImage = (unsigned char*)malloc(width*height*slices*sizeof(unsigned char));

  for (int slice = 0; slice < slices; slice++)
    for (int col = 0; col < width; col++)
      for (int row = 0; row < height; row++)
        regImage[slice*height*width+col*height+row] = 0;

  cudaEvent_t start, stop;
  startTimeRecord(&start, &stop);

  tpsCuda<<<numBlocks, threadsPerBlock>>>(cm.getTargetImage(), cm.getRegImage(), cm.getSolutionCol(), cm.getSolutionRow(), 
                                          cm.getSolutionSlice(), width, height, slices, cm.getKeypointCol(), 
                                          cm.getKeypointRow(), cm.getKeypointSlice(), numberOfCPs);
  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaMemcpy(regImage, cm.getRegImage(), width*height*slices*sizeof(unsigned char), cudaMemcpyDeviceToHost));

  showExecutionTime(&start, &stop, "callKernel execution time = ");
  return regImage;
}
