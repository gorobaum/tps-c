#include <iostream>
#include <cmath>

#include "cudatps.h"
#include "cutps.h"

#define MAXTHREADPBLOCK 1024

void tps::CudaTPS::callKernel(double *cudaImageCoord, float *cudaSolution, dim3 threadsPerBlock, dim3 numBlocks) {
  runTPSCUDA(cudaImageCoord, cudaSolution, threadsPerBlock, numBlocks, width, height,
              cm_.getKeypointCol(), cm_.getKeypointRow(), referenceKeypoints_.size());
}

void tps::CudaTPS::run() {
	allocResources();
  cudalienarSolver.solveLinearSystems(cm_);

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(std::ceil(1.0*width/threadsPerBlock.x), std::ceil(1.0*height/threadsPerBlock.y));

  callKernel(cm_.getCoordinateCol(), cm_.getSolutionCol(), threadsPerBlock, numBlocks);
  callKernel(cm_.getCoordinateRow(), cm_.getSolutionRow(), threadsPerBlock, numBlocks);

  // std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

  runRegImage(cm_.getCoordinateCol(), cm_.getCoordinateRow(), cm_.getTargetImage(), cm_.getRegImage(), width, height, threadsPerBlock, numBlocks, regImage);

  registredImage.setPixelVector(regImage);
  registredImage.save(outputName_);

  free(regImage);
}

void tps::CudaTPS::allocResources() {
  regImage = (uchar*)malloc(width*height*sizeof(uchar));
  for (int col = 0; col < width; col++)
    for (int row = 0; row < height; row++)
      regImage[col*height+row] = 0;
}