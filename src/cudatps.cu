#include <iostream>
#include <cmath>

#include "cudatps.h"

#define MAXTHREADPBLOCK 1024

// Kernel definition
__device__ double cudaGetPixel(int x, int y, uchar* image, int width, int heigth) {
  if (x > heigth-1) return 0;
  if (y > width-1) return 0;
  return image[x*width+y];
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

  double newX = cudaImageCoordX[x*width+y];
  double newY = cudaImageCoordY[x*width+y];
  cudaRegImage[x*width+y] = cudaBilinearInterpolation(newX, newY, cudaImage, width, heigth);
}

// Kernel definition
__global__ void cudaTPS(double* cudaImageCoord, int width, int heigth, float* solution, float* cudaKeyX, float* cudaKeyY, uint numOfKeys)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  double newCoord = solution[0] + x*solution[1] + y*solution[2];

  for (uint i = 0; i < numOfKeys; i++) {
    double r = (x-cudaKeyX[i])*(x-cudaKeyX[i]) + (y-cudaKeyY[i])*(y-cudaKeyY[i]);
    if (r != 0.0)
      newCoord += r*log(r) * solution[i+3];
  }
  if (x*width+y < width*heigth)
    cudaImageCoord[x*width+y] = newCoord;
}

void tps::CudaTPS::callKernel(double* cudaImageCoord, float *cudaSolution, dim3 threadsPerBlock, dim3 numBlocks) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  cudaTPS<<<numBlocks, threadsPerBlock>>>(cudaImageCoord, dimensions[1], dimensions[0], cudaSolution, cudaKeyX, cudaKeyY, targetKeypoints_.size());
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
	dimensions = registredImage.getDimensions();
	allocResources();
	allocCudaResources();

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(std::ceil(1.0*dimensions[0]/threadsPerBlock.x), std::ceil(1.0*dimensions[1]/threadsPerBlock.y));

  callKernel(cudaImageCoordX ,cudaSolutionX, threadsPerBlock, numBlocks);
  callKernel(cudaImageCoordY ,cudaSolutionY, threadsPerBlock, numBlocks);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  cudaRegistredImage<<<numBlocks, threadsPerBlock>>>(cudaImageCoordX, cudaImageCoordY, cudaImage, cudaRegImage, dimensions[1], dimensions[0]);
  cudaMemcpy(regImage, cudaRegImage, dimensions[0]*dimensions[1]*sizeof(uchar), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize(); 
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::cout << "Time = " << elapsedTime << " ms\n";

  registredImage.setPixelVector<uchar>(regImage);
  registredImage.save();

	freeResources();
	freeCudaResources();

	cudaDeviceReset();
}

void tps::CudaTPS::allocResources() {
  regImage = (uchar*)malloc(dimensions[0]*dimensions[1]*sizeof(uchar));
  createCudaSolution();
  createCudaKeyPoint();
}

void tps::CudaTPS::createCudaSolution() {
  std::vector<float> solutionX;
  std::vector<float> solutionY;
  cudalienarSolver.solveLinearSystems();
  solutionX = cudalienarSolver.getSolutionX();
  solutionY = cudalienarSolver.getSolutionY();
  floatSolX = (float*)malloc((targetKeypoints_.size()+3)*sizeof(float));
  floatSolY = (float*)malloc((targetKeypoints_.size()+3)*sizeof(float));
  for (uint i = 0; i < (targetKeypoints_.size()+3); i++) {
    floatSolX[i] = solutionX[i];
    floatSolY[i] = solutionY[i];
  }
}

void tps::CudaTPS::createCudaKeyPoint() {
  floatKeyX = (float*)malloc(targetKeypoints_.size()*sizeof(float));
  floatKeyY = (float*)malloc(targetKeypoints_.size()*sizeof(float));
  for (uint i = 0; i < targetKeypoints_.size(); i++) {
    floatKeyX[i] = referenceKeypoints_[i].x;
    floatKeyY[i] = referenceKeypoints_[i].y;
  }
}

void tps::CudaTPS::allocCudaResources() {
  cudaMalloc(&cudaImageCoordX, dimensions[0]*dimensions[1]*sizeof(double));
  cudaMalloc(&cudaImageCoordY, dimensions[0]*dimensions[1]*sizeof(double));
  cudaMalloc(&cudaSolutionX, (targetKeypoints_.size()+3)*sizeof(float));
  cudaMalloc(&cudaSolutionY, (targetKeypoints_.size()+3)*sizeof(float));
  cudaMemcpy(cudaSolutionX, floatSolX, (targetKeypoints_.size()+3)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaSolutionY, floatSolY, (targetKeypoints_.size()+3)*sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&cudaKeyX, targetKeypoints_.size()*sizeof(float));
  cudaMalloc(&cudaKeyY, targetKeypoints_.size()*sizeof(float));
  cudaMemcpy(cudaKeyX, floatKeyX, targetKeypoints_.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaKeyY, floatKeyY, targetKeypoints_.size()*sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&cudaRegImage, dimensions[0]*dimensions[1]*sizeof(uchar));
  cudaMalloc(&cudaImage, dimensions[0]*dimensions[1]*sizeof(uchar));
  cudaMemcpy(cudaImage, targetImage_.getPixelVector<uchar>(), dimensions[0]*dimensions[1]*sizeof(uchar), cudaMemcpyHostToDevice);
}

void tps::CudaTPS::freeResources() {
  free(floatSolX);
  free(floatSolY);
  free(floatKeyX);
  free(floatKeyY);
  free(regImage);
}

void tps::CudaTPS::freeCudaResources() {
  cudaFree(cudaImageCoordX);
  cudaFree(cudaImageCoordY);
  cudaFree(cudaSolutionX);
  cudaFree(cudaSolutionY);
  cudaFree(cudaKeyX);
  cudaFree(cudaKeyY);
	cudaDeviceSynchronize();
}
