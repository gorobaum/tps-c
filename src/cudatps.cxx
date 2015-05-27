#include <iostream>
#include <cmath>

#include "cudatps.h"
#include "cutps.h"

#define MAXTHREADPBLOCK 1024

void tps::CudaTPS::run() {
  cudalienarSolver.solveLinearSystems(cm_);

  runTPSCUDA(cm_, width, height, referenceKeypoints_.size());

  // std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

  regImage = runRegImage(cm_, width, height);

  registredImage.setPixelVector(regImage);
  registredImage.save(outputName_);

  free(regImage);
}
