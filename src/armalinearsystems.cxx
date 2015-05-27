#include "armalinearsystems.h"

void tps::ArmaLinearSystems::solveLinearSystems() {
  createMatrixA();
  createBs();
}

void tps::ArmaLinearSystems::createMatrixA() {
  A = arma::mat(systemDimension, systemDimension, arma::fill::zeros);
}

void tps::ArmaLinearSystems::createBs() {
  bx = arma::vec(systemDimension, arma::fill::zeros);
  by = arma::vec(systemDimension, arma::fill::zeros);
}