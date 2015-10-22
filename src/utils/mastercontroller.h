#ifndef TPS_MASTERCONTROLLER_H_
#define TPS_MASTERCONTROLLER_H_

#include <iostream>
#include <vector>
#include <string>

#include "tpsinstance.h"
#include "cudastream.h"
#include "cudamemory.h"

namespace tps {
  
class MasterController {
public:
  MasterController(std::vector<TpsInstance> executionInstances) :
    executionInstances_(executionInstances) {
      execMemoryReady = 0;
      execFinished = 0;
    };
  void run();
  void loadGPUMemory();
  static void executeInstances(void* data);
private:
  std::vector<tps::TpsInstance> executionInstances_;
  std::vector<tps::CudaStream> streams;
  static const int numberOfThreads = 2;
  int execMemoryReady;
  int execFinished;
};

} // namespace

#endif