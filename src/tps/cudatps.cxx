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

void tps::CudaTPS::run() {
  lienarSolver.solveLinearSystems();

  cm_.setSolX(lienarSolver.getSolutionX());
  cm_.setSolY(lienarSolver.getSolutionY());
  cm_.setSolZ(lienarSolver.getSolutionZ());

  regImage = runTPSCUDA(cm_, dimensions_, referenceKeypoints_.size());

  registredImage.setPixelVector(regImage);
  
  tps::ITKImageHandler::saveImageData(registredImage, outputName_);

  free(regImage);
}