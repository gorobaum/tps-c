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
  void solveLinearSystem(float *B, float *solution);
  std::vector<float> pointerToVector(float *pointer);
  void createMatrixA();
  void createBs();
  void freeResources();
  float *bx, *by, *A, *floatSolX, *floatSolY;
};

} //namepsace

#endif