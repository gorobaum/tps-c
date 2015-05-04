#include <iostream>
#include <cmath>

#include "cudatps.h"

#define MAXTHREADPBLOCK 1024

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

void tps::CudaTPS::callKernel(float *cudaSolution, double *imageCoord, dim3 threadsPerBlock, dim3 numBlocks) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  tpsCuda<<<numBlocks, threadsPerBlock>>>(cudaImageCoord, width, height, cudaSolution, cudaKeyCol, cudaKeyRow, targetKeypoints_.size());
  cudaDeviceSynchronize(); 
  cudaMemcpy(imageCoord, cudaImageCoord, width*height*sizeof(double), cudaMemcpyDeviceToHost);
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
	allocCudaResources();

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(std::ceil(1.0*width/threadsPerBlock.x), std::ceil(1.0*height/threadsPerBlock.y));

  callKernel(cudaSolutionCol, imageCoordCol, threadsPerBlock, numBlocks);
  callKernel(cudaSolutionRow, imageCoordRow, threadsPerBlock, numBlocks);

  // std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

  for (int col = 0; col < width; col++)
    for (int row = 0; row < height; row++) {
      double newCol = imageCoordCol[col*height+row];
      double newRow = imageCoordRow[col*height+row];
      int value = targetImage_.bilinearInterpolation(newCol, newRow);
      registredImage.changePixelAt(col, row, value);
    }
  registredImage.save(outputName_);

	freeResources();
	freeCudaResources();

	cudaDeviceReset();
}

void tps::CudaTPS::allocResources() {
  imageCoordCol = (double*)malloc(width*height*sizeof(double));
  imageCoordRow = (double*)malloc(width*height*sizeof(double));
  createCudaSolution();
  createCudaKeyPoint();
}

void tps::CudaTPS::createCudaSolution() {
  std::vector<float> solutionCol;
  std::vector<float> solutionRow;
  cudalienarSolver.solveLinearSystems();
  solutionCol = cudalienarSolver.getSolutionCol();
  solutionRow = cudalienarSolver.getSolutionRow();
  floatSolCol = (float*)malloc((targetKeypoints_.size()+3)*sizeof(float));
  floatSolRow = (float*)malloc((targetKeypoints_.size()+3)*sizeof(float));
  for (uint i = 0; i < (targetKeypoints_.size()+3); i++) {
    floatSolCol[i] = solutionCol[i];
    floatSolRow[i] = solutionRow[i];
  }
}

void tps::CudaTPS::createCudaKeyPoint() {
  floatKeyCol = (float*)malloc(targetKeypoints_.size()*sizeof(float));
  floatKeyRow = (float*)malloc(targetKeypoints_.size()*sizeof(float));
  for (uint i = 0; i < targetKeypoints_.size(); i++) {
    floatKeyCol[i] = referenceKeypoints_[i].x;
    floatKeyRow[i] = referenceKeypoints_[i].y;
  }
}

void tps::CudaTPS::allocCudaResources() {
  cudaMalloc(&cudaImageCoord, width*height*sizeof(double));
  cudaMalloc(&cudaSolutionCol, (targetKeypoints_.size()+3)*sizeof(float));
  cudaMalloc(&cudaSolutionRow, (targetKeypoints_.size()+3)*sizeof(float));
  cudaMemcpy(cudaSolutionCol, floatSolCol, (targetKeypoints_.size()+3)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaSolutionRow, floatSolRow, (targetKeypoints_.size()+3)*sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&cudaKeyCol, targetKeypoints_.size()*sizeof(float));
  cudaMalloc(&cudaKeyRow, targetKeypoints_.size()*sizeof(float));
  cudaMemcpy(cudaKeyCol, floatKeyCol, targetKeypoints_.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaKeyRow, floatKeyRow, targetKeypoints_.size()*sizeof(float), cudaMemcpyHostToDevice);
}

void tps::CudaTPS::freeResources() {
  free(imageCoordCol);
  free(imageCoordRow);
  free(floatSolCol);
  free(floatSolRow);
  free(floatKeyCol);
  free(floatKeyRow);
}

void tps::CudaTPS::freeCudaResources() {
  cudaFree(cudaImageCoord);
  cudaFree(cudaSolutionCol);
  cudaFree(cudaSolutionRow);
  cudaFree(cudaKeyCol);
  cudaFree(cudaKeyRow);
	cudaDeviceSynchronize();
}
