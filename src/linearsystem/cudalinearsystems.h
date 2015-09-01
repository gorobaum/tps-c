#ifndef TPS_CUDALINEARSYSTEMS_H_
#define TPS_CUDALINEARSYSTEMS_H_

#include "cplinearsystems.h"
#include "utils/cudamemory.h"

namespace tps {

class CudaLinearSystems : public CPLinearSystems {
using CPLinearSystems::CPLinearSystems;
public:
  void solveLinearSystems(tps::CudaMemory& cm);
private:
  void solveLinearSystem(float *B, float *cudaSolution);
  std::vector<float> pointerToVector(float *pointer);
  void transferMatrixA();
  void transferBs();
  void freeResources();
  float *CLSbx, *CLSby, *CLSbz, *CLSA;
};

} //namepsace

#endif