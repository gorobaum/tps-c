#ifndef TPS_CUDALINEARSYSTEMS_H_
#define TPS_CUDALINEARSYSTEMS_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_occupancy.h"
#include "cusolver_common.h"
#include "cusolverDn.h"

#include "cplinearsystems.h"

namespace tps {

class CudaLinearSystems : public CPLinearSystems {
using CPLinearSystems::CPLinearSystems;
public:
  void solveLinearSystems();
private:
  std::vector<float> solveLinearSystem(cv::Mat A, cv::Mat b);
  void createMatrixA();
  void createBs();
  float *bx, *by, *A;
  float *cudaBx, *cudaBy, *cudaA;
};

} //namepsace

#endif