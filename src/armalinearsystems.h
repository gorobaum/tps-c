#ifndef TPS_ARMALINEARSYSTEMS_H_
#define TPS_ARMALINEARSYSTEMS_H_

#include "cplinearsystems.h"
#include <armadillo>

namespace tps {

class ArmaLinearSystems : public CPLinearSystems {
using CPLinearSystems::CPLinearSystems;
public:
  void solveLinearSystems();
private:
  void createMatrixA();
  void createBs();
  void freeResources();
  arma::mat A;
  arma::vec bx, by;
};

} //namepsace

#endif