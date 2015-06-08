#ifndef TPS_OPCVLINEARSYSTEMS_H_
#define TPS_OPCVLINEARSYSTEMS_H_

#include "cplinearsystems.h"

#include <opencv2/core/core.hpp>

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
  cv::Mat bz;
  cv::Mat A;
};

} //namepsace

#endif