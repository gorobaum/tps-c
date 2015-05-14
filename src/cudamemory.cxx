#include "cudamemory.h"

void tps::CudaMemory::allocCudaCoord(double *hostCoordinateCol, double *hostCoordinateRow) {
  cudaMalloc(&coordinateCol, imageWidth*imageHeight*sizeof(double));
  cudaMemcpy(coordinateCol, hostCoordinateCol, imageWidth*imageHeight*sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc(&coordinateRow, imageWidth*imageHeight*sizeof(double));
  cudaMemcpy(coordinateRow, hostCoordinateRow, imageWidth*imageHeight*sizeof(double), cudaMemcpyHostToDevice);
}

void tps::CudaMemory::allocCudaSolution(float *hostSolutionCol, float *hostSolutionRow) {
  cudaMalloc(&solutionCol, systemDim*sizeof(float));
  cudaMemcpy(solutionCol, hostSolutionCol, systemDim*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&solutionRow, systemDim*sizeof(float));
  cudaMemcpy(solutionRow, hostSolutionRow, systemDim*sizeof(float), cudaMemcpyHostToDevice);
}

void tps::CudaMemory::allocCudaKeypoints(float *hostKeypointCol, float *hostKeypointRow) {
  cudaMalloc(&keypointCol, numberOfCps*sizeof(float));
  cudaMemcpy(keypointCol, hostKeypointCol, numberOfCps*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&keypointRow, numberOfCps*sizeof(float));
  cudaMemcpy(keypointRow, hostKeypointRow, numberOfCps*sizeof(float), cudaMemcpyHostToDevice);
}

void tps::CudaMemory::allocCudaImagePixels(unsigned char *hostTargetImage, unsigned char *hostRegImage) {
  cudaMalloc(&targetImage, numberOfCps*sizeof(float));
  cudaMemcpy(targetImage, hostTargetImage, numberOfCps*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&regImage, numberOfCps*sizeof(float));
  cudaMemcpy(regImage, hostRegImage, numberOfCps*sizeof(float), cudaMemcpyHostToDevice);
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