#include <iostream>
#include <cmath>

#include "cudatps.h"
#include "cutps.h"

#define MAXTHREADPBLOCK 1024

float* tps::CudaTPS::solutionPointer(std::vector<float> solution) {
  float* vector = (float*)malloc(solution.size()*sizeof(float));
  for(int i = 0; i < solution.size(); i++)
    vector[i] = solution[i];
  return vector;
}

void tps::CudaTPS::loadImage() {
  short *regImage = getGPUResult(cm_, dimensions_);
  registredImage.setPixelVector(regImage);
}

void tps::CudaTPS::run() {
  lienarSolver.solveLinearSystems(cm_);

  runTPSCUDA(cm_, dimensions_, referenceKeypoints_.size());
}