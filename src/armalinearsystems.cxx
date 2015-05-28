#include "armalinearsystems.h"

void tps::ArmaLinearSystems::solveLinearSystems() {
  createMatrixA();
  createBs();
  double solverExec = (double)cv::getTickCount();
  solutionCol = solveLinearSystem(A, bx);
  solutionRow = solveLinearSystem(A, by);
  solverExec = ((double)cv::getTickCount() - solverExec)/cv::getTickFrequency();
  std::cout << "Arma solver execution time: " << solverExec << std::endl;

}

std::vector<float> tps::ArmaLinearSystems::solveLinearSystem(arma::mat A, arma::vec b) {
  std::vector<float> solution;
  arma::vec armaSol = arma::solve(A, b);

  for (uint i = 0; i < systemDimension; i++)
    solution.push_back(armaSol(i));
  
  return solution;
}

void tps::ArmaLinearSystems::createMatrixA() {
  A = arma::mat(systemDimension, systemDimension, arma::fill::zeros);

  for (uint j = 0; j < referenceKeypoints_.size(); j++) {
    A(0,j+3) = 1;
    A(1,j+3) = referenceKeypoints_[j][0];
    A(2,j+3) = referenceKeypoints_[j][1];
    A(j+3,0) = 1;
    A(j+3,1) = referenceKeypoints_[j][0];
    A(j+3,2) = referenceKeypoints_[j][1];
  }

  for (uint i = 0; i < referenceKeypoints_.size(); i++)
    for (uint j = 0; j < referenceKeypoints_.size(); j++) {
      float r = computeRSquared(referenceKeypoints_[i][0], referenceKeypoints_[j][0], referenceKeypoints_[i][1], referenceKeypoints_[j][1]);
      if (r != 0.0) A(j+3,i+3) = r*log(r);
    }
}

void tps::ArmaLinearSystems::createBs() {
  bx = arma::vec(systemDimension, arma::fill::zeros);
  by = arma::vec(systemDimension, arma::fill::zeros);
 for (uint i = 0; i < targetKeypoints_.size(); i++) {
    bx(i+3) = targetKeypoints_[i][0];
    by(i+3) = targetKeypoints_[i][1];
  }
}