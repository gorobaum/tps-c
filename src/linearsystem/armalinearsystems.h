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
  void transferMatrixA();
  void transferBs();
  std::vector<float> solveLinearSystem(arma::vec b);
  arma::mat ALSA;
  arma::vec ALSbx, ALSby, ALSbz;
};

} //namepsace

#endif