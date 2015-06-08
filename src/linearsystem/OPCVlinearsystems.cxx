#include "OPCVlinearsystems.h"

#include <iostream>

void tps::OPCVLinearSystems::solveLinearSystems() {
  createMatrixA();
  createBs();

  double solverExec = (double)cv::getTickCount();
  solutionCol = solveLinearSystem(A, bx);
  solutionRow = solveLinearSystem(A, by);
  solutionSlice = solveLinearSystem(A, bz);
  solverExec = ((double)cv::getTickCount() - solverExec)/cv::getTickFrequency();
  std::cout << "OPCV solver execution time: " << solverExec << std::endl;

}

std::vector<float> tps::OPCVLinearSystems::solveLinearSystem(cv::Mat A, cv::Mat b) {
  std::vector<float> solution;
  cv::Mat cvSolution = cv::Mat::zeros(systemDimension, 1, CV_32F);
  cv::solve(A, b, cvSolution, cv::DECOMP_EIG);

  for (uint i = 0; i < systemDimension; i++)
    solution.push_back(cvSolution.at<float>(i));
  
  return solution;
}

void tps::OPCVLinearSystems::createMatrixA() {
  A = cv::Mat::zeros(systemDimension,systemDimension, CV_32F);

  for (uint j = 0; j < referenceKeypoints_.size(); j++) {
    A.at<float>(0,j+3) = 1;
    A.at<float>(1,j+3) = referenceKeypoints_[j][0];
    A.at<float>(2,j+3) = referenceKeypoints_[j][1];
    A.at<float>(3,j+3) = referenceKeypoints_[j][2];
    A.at<float>(j+3,0) = 1;
    A.at<float>(j+3,1) = referenceKeypoints_[j][0];
    A.at<float>(j+3,2) = referenceKeypoints_[j][1];
    A.at<float>(j+3,3) = referenceKeypoints_[j][2];
  }

  for (uint i = 0; i < referenceKeypoints_.size(); i++)
    for (uint j = 0; j < referenceKeypoints_.size(); j++) {
      float r = computeRSquared(referenceKeypoints_[i][0], referenceKeypoints_[j][0], 
                                referenceKeypoints_[i][1], referenceKeypoints_[j][1], 
                                referenceKeypoints_[i][2], referenceKeypoints_[j][2]);
      if (r != 0.0) A.at<float>(i+3,j+3) = r*log(r);
    }
}

void tps::OPCVLinearSystems::createBs() {
  bx = cv::Mat::zeros(systemDimension, 1, CV_32F);
  by = cv::Mat::zeros(systemDimension, 1, CV_32F);
  bz = cv::Mat::zeros(systemDimension, 1, CV_32F);
  for (uint i = 0; i < referenceKeypoints_.size(); i++) {
    bx.at<float>(i+3) = targetKeypoints_[i][0];
    by.at<float>(i+3) = targetKeypoints_[i][1];
    bz.at<float>(i+3) = targetKeypoints_[i][2];
  }
}