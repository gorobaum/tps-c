#include "armalinearsystems.h"

void tps::ArmaLinearSystems::solveLinearSystems() {
  createMatrixA();
  createBs();

  arma::wall_clock timer;
  timer.tic();

  solutionX = solveLinearSystem(A, bx);
  solutionY = solveLinearSystem(A, by);
  solutionZ = solveLinearSystem(A, bz);

  double time = timer.toc();
  std::cout << "Arma solver execution time: " << time << std::endl;

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
    A(0,j+4) = 1;
    A(1,j+4) = referenceKeypoints_[j][0];
    A(2,j+4) = referenceKeypoints_[j][1];
    A(3,j+4) = referenceKeypoints_[j][2];
    A(j+4,0) = 1;
    A(j+4,1) = referenceKeypoints_[j][0];
    A(j+4,2) = referenceKeypoints_[j][1];
    A(j+4,3) = referenceKeypoints_[j][2];
  }

  for (uint i = 0; i < referenceKeypoints_.size(); i++)
    for (uint j = 0; j < referenceKeypoints_.size(); j++) {
      float r = computeRSquared(referenceKeypoints_[i][0], referenceKeypoints_[j][0], 
                                referenceKeypoints_[i][1], referenceKeypoints_[j][1],
                                referenceKeypoints_[i][2], referenceKeypoints_[j][2]);
      if (r != 0.0) A(j+4,i+4) = r*log(r);
    }
}

void tps::ArmaLinearSystems::createBs() {
  bx = arma::vec(systemDimension, arma::fill::zeros);
  by = arma::vec(systemDimension, arma::fill::zeros);
  bz = arma::vec(systemDimension, arma::fill::zeros);
  for (uint i = 0; i < targetKeypoints_.size(); i++) {
    bx(i+4) = targetKeypoints_[i][0];
    by(i+4) = targetKeypoints_[i][1];
    bz(i+4) = targetKeypoints_[i][2];
  }
}