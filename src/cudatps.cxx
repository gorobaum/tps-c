#include <iostream>
#include <cmath>

#include "cudatps.h"
#include "cutps.h"

#define MAXTHREADPBLOCK 1024

void tps::CudaTPS::run() {
  cudalienarSolver.solveLinearSystems(cm_);

  regImage = runTPSCUDA(cm_, width, height, slices, referenceKeypoints_.size());

  registredImage.setPixelVector(regImage);
  registredImage.save(outputName_);

  free(regImage);
}
