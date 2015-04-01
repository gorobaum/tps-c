#ifndef TPS_OPCVLINEARSYSTEMS_H_
#define TPS_OPCVLINEARSYSTEMS_H_

#include "cplinearsystems.h"

namespace tps {

class OPCVLinearSystems : public CPLinearSystems {
using CPLinearSystems::CPLinearSystems;
public:
  void solveLinearSystems();
private:
  std::vector<float> solveLinearSystem(cv::Mat A, cv::Mat b);
  void createMatrixA();
  void createBs();
  cv::Mat bx;
  cv::Mat by;
  cv::Mat A;
};

} //namepsace

#endif