#include "cudalinearsystems.h"

#include <cstdlib>
#include <iostream>


void tps::CudaLinearSystems::solveLinearSystems() {
  createMatrixA();
  createBs();
}

void tps::CudaLinearSystems::createMatrixA() {
  A = (float*)malloc(systemDimension*systemDimension*sizeof(float));

 for (uint j = 0; j < 3; j++) {
    A[0*systemDimension+j] = 0.0;
    A[1*systemDimension+j] = 0.0;
    A[2*systemDimension+j] = 0.0;
  }

  for (uint j = 0; j < referenceKeypoints_.size(); j++) {
    A[0*systemDimension+j+3] = 1;
    A[1*systemDimension+j+3] = referenceKeypoints_[j].x;
    A[2*systemDimension+j+3] = referenceKeypoints_[j].y;
    A[(j+3)*systemDimension+0] = 1;
    A[(j+3)*systemDimension+1] = referenceKeypoints_[j].x;
    A[(j+3)*systemDimension+2] = referenceKeypoints_[j].y;
  }

  for (uint i = 0; i < referenceKeypoints_.size(); i++)
    for (uint j = 0; j < referenceKeypoints_.size(); j++) {
      float r = computeRSquared(referenceKeypoints_[i].x, referenceKeypoints_[j].x, referenceKeypoints_[i].y, referenceKeypoints_[j].y);
      if (r != 0.0) A[(i+3)*systemDimension+j+3] = r*log(r);
    }

  for (uint i = 0; i < referenceKeypoints_.size()+3; i++)
    std::cout << A[i] << std::endl;
}

void tps::CudaLinearSystems::createBs() {
  bx = (float*)malloc(systemDimension*sizeof(float));
  by = (float*)malloc(systemDimension*sizeof(float));
  for (uint j = 0; j < 3; j++) {
    bx[j] = 0.0;
    by[j] = 0.0;
  }
  for (uint i = 0; i < referenceKeypoints_.size(); i++) {
    bx[i+3] = targetKeypoints_[i].x;
    by[i+3] = targetKeypoints_[i].y;
  }
}

std::vector<float> solveLinearSystem(cv::Mat A, cv::Mat b) {
  std::vector<float> dae;
  return dae;
}