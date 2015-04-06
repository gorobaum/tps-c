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
  std::vector<float> solveLinearSystem(float *A, float *b, float *cudaB, float *cudaSol);
  std::vector<float> pointerToVector(float *pointer);
  void createMatrixA();
  void createBs();
  void allocCudaResources();
  void freeResources();
  void freeCudaResources();
  float *bx, *by, *A, *floatSolX, *floatSolY;
  float *cudaBx, *cudaBy, *cudaA, *cudaSolX, *cudaSolY;
};

} //namepsace

#endif