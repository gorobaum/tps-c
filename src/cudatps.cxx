#include <iostream>
#include <cmath>

#include "cudatps.h"
#include "cutps.h"

#define MAXTHREADPBLOCK 1024

void tps::CudaTPS::run() {
  cudalienarSolver.solveLinearSystems(cm_);

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(std::ceil(1.0*width/threadsPerBlock.x), std::ceil(1.0*height/threadsPerBlock.y));

  runTPSCUDA(cm_.getCoordinateCol(), cm_.getSolutionCol(), threadsPerBlock, numBlocks, width, height,
              cm_.getKeypointCol(), cm_.getKeypointRow(), referenceKeypoints_.size());
  runTPSCUDA(cm_.getCoordinateRow(), cm_.getSolutionRow(), threadsPerBlock, numBlocks, width, height,
              cm_.getKeypointCol(), cm_.getKeypointRow(), referenceKeypoints_.size());

  // std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

  regImage = runRegImage(cm_.getCoordinateCol(), cm_.getCoordinateRow(), cm_.getTargetImage(), cm_.getRegImage(), width, height, threadsPerBlock, numBlocks);

  registredImage.setPixelVector(regImage);
  std::cout << "ParaRegImage[100] = " << registredImage.getPixelAt(0, 100) << std::endl;
  std::cout << "ParaRegImage[100] = " << registredImage.getPixelAt(100, 0) << std::endl;
  registredImage.save(outputName_);

  free(regImage);
}
