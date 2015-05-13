#include <iostream>
#include <cmath>

#include "cudatps.h"

#define MAXTHREADPBLOCK 1024

// Kernel definition
__device__ double cudaGetPixel(int x, int y, uchar* image, int width, int heigth) {
  if (x > heigth-1 || x < 0) return 0;
  if (y > width-1 || y < 0) return 0;
  return image[x*heigth+y];
}

// Kernel definition
__device__ double cudaBilinearInterpolation(double newX, double newY, uchar* image, int width, int heigth) {
  int u = trunc(newX);
  int v = trunc(newY);

  uchar pixelOne = cudaGetPixel(u, v, image, width, heigth);
  uchar pixelTwo = cudaGetPixel(u+1, v, image, width, heigth);
  uchar pixelThree = cudaGetPixel(u, v+1, image, width, heigth);
  uchar pixelFour = cudaGetPixel(u+1, v+1, image, width, heigth);

  double interpolation = (u+1-newX)*(v+1-newY)*pixelOne
                        + (newX-u)*(v+1-newY)*pixelTwo 
                        + (u+1-newX)*(newY-v)*pixelThree
                        + (newX-u)*(newY-v)*pixelFour;
  return interpolation;
}

// Kernel definition
__global__ void cudaRegistredImage(double* cudaImageCoordX, double* cudaImageCoordY, uchar* cudaImage, uchar* cudaRegImage, int width, int heigth) {
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;

  double newX = cudaImageCoordX[x*heigth+y];
  double newY = cudaImageCoordY[x*heigth+y];
  cudaRegImage[x*heigth+y] = cudaBilinearInterpolation(newX, newY, cudaImage, width, heigth);
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
  tpsCuda<<<numBlocks, threadsPerBlock>>>(cudaImageCoord, width, height, cudaSolution, cudaKeyCol, cudaKeyRow, targetKeypoints_.size());
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
	allocCudaResources();

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(std::ceil(1.0*width/threadsPerBlock.x), std::ceil(1.0*height/threadsPerBlock.y));

  callKernel(cudaImageCoordCol, cudaSolutionCol, threadsPerBlock, numBlocks);
  callKernel(cudaImageCoordRow, cudaSolutionRow, threadsPerBlock, numBlocks);

  // std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  cudaRegistredImage<<<numBlocks, threadsPerBlock>>>(cudaImageCoordCol, cudaImageCoordRow, cudaImage, cudaRegImage, width, height);
  cudaMemcpy(regImage, cudaRegImage, width*height*sizeof(uchar), cudaMemcpyDeviceToHost);
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
  for (uint i = 0; i < referenceKeypoints_.size(); i++) {
    floatKeyCol[i] = referenceKeypoints_[i].x;
    floatKeyRow[i] = referenceKeypoints_[i].y;
  }
}

void tps::CudaTPS::allocCudaResources() {
  cudaMalloc(&cudaImageCoordCol, width*height*sizeof(double));
  cudaMalloc(&cudaImageCoordRow, width*height*sizeof(double));
  cudaMalloc(&cudaSolutionCol, (targetKeypoints_.size()+3)*sizeof(float));
  cudaMalloc(&cudaSolutionRow, (targetKeypoints_.size()+3)*sizeof(float));
  cudaMemcpy(cudaSolutionCol, floatSolCol, (targetKeypoints_.size()+3)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaSolutionRow, floatSolRow, (targetKeypoints_.size()+3)*sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&cudaKeyCol, targetKeypoints_.size()*sizeof(float));
  cudaMalloc(&cudaKeyRow, targetKeypoints_.size()*sizeof(float));
  cudaMemcpy(cudaKeyCol, floatKeyCol, targetKeypoints_.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaKeyRow, floatKeyRow, targetKeypoints_.size()*sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&cudaRegImage, width*height*sizeof(uchar));
  cudaMalloc(&cudaImage, width*height*sizeof(uchar));
  cudaMemcpy(cudaImage, targetImage_.getPixelVector(), width*height*sizeof(uchar), cudaMemcpyHostToDevice);
}

void tps::CudaTPS::freeResources() {
  free(floatSolCol);
  free(floatSolRow);
  free(floatKeyCol);
  free(floatKeyRow);
}

void tps::CudaTPS::freeCudaResources() {
  cudaFree(cudaImageCoordCol);
  cudaFree(cudaImageCoordRow);
  cudaFree(cudaSolutionCol);
  cudaFree(cudaSolutionRow);
  cudaFree(cudaKeyCol);
  cudaFree(cudaKeyRow);
	cudaDeviceSynchronize();
}
