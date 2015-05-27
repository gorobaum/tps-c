#ifndef TPS_CUTPS_H_
#define TPS_CUTPS_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_occupancy.h"

void runTPSCUDA(double* cudaImageCoord, float* cudaSolution, dim3 threadsPerBlock, dim3 numBlocks, int width, int height,
                float* keypointCol, float* keypointRow, int numberOfCP);

unsigned char* runRegImage(double* cudaImageCoordX, double* cudaImageCoordY, unsigned char* cudaImage, unsigned char* cudaRegImage, int width, int height, 
                  dim3 threadsPerBlock, dim3 numBlocks);

#endif