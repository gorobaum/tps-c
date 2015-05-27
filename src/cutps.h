#ifndef TPS_CUTPS_H_
#define TPS_CUTPS_H_

#include "cudamemory.h"

void runTPSCUDA(tps::CudaMemory cm, int width, int height, int numberOfCPs);

void runTPSCUDAForCoord(double* cudaImageCoord, float* cudaSolution, dim3 threadsPerBlock, dim3 numBlocks, int width, int height,
                float* keypointCol, float* keypointRow, int numberOfCP);

unsigned char* runRegImage(tps::CudaMemory cm, int width, int height);

#endif