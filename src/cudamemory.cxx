#include "cudamemory.h"

void tps::CudaMemory::allocCudaCoord() {
  cudaMalloc(&coordinateCol, imageWidth*imageHeight*sizeof(double));
  cudaMalloc(&coordinateRow, imageWidth*imageHeight*sizeof(double));
}

void tps::CudaMemory::allocCudaSolution() {
  cudaMalloc(&solutionCol, systemDim*sizeof(float));
  cudaMalloc(&solutionRow, systemDim*sizeof(float));
}

void tps::CudaMemory::allocCudaKeypoints(float *hostKeypointCol, float *hostKeypointRow) {
  cudaMalloc(&keypointCol, numberOfCps*sizeof(float));
  cudaMemcpy(keypointCol, hostKeypointCol, numberOfCps*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&keypointRow, numberOfCps*sizeof(float));
  cudaMemcpy(keypointRow, hostKeypointRow, numberOfCps*sizeof(float), cudaMemcpyHostToDevice);
}

void tps::CudaMemory::allocCudaImagePixels(tps::Image& image) {
  cudaMalloc(&targetImage, imageWidth*imageHeight*sizeof(uchar));
  cudaMemcpy(targetImage, image.getPixelVector(), imageWidth*imageHeight*sizeof(uchar), cudaMemcpyHostToDevice);
  cudaMalloc(&regImage, imageWidth*imageHeight*sizeof(uchar));
}

void tps::CudaMemory::freeMemory() {
  cudaFree(coordinateCol);
  cudaFree(coordinateRow);
  cudaFree(solutionCol);
  cudaFree(solutionRow);
  cudaFree(keypointCol);
  cudaFree(keypointRow);
  cudaFree(targetImage);
  cudaFree(regImage);
}