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
  float* getCudaSolCol() { return cudaSolutionCol; };
  float* getCudaSolRow() { return cudaSolutionRow; };
  void freeCuda();
private:
  void solveLinearSystem(float *B, float *cudaSolution);
  std::vector<float> pointerToVector(float *pointer);
  void createMatrixA();
  void createBs();
  void freeResources();
  float *cudaSolutionCol, *cudaSolutionRow;
  float *bx, *by, *A;
};

} //namepsace

#endif