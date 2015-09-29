#include "mastercontroller.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_occupancy.h"

#include <iostream>

void tps::MasterController::run() {
  int numberOfExecs = executionInstances_.size();
  while (execFinished < numberOfExecs) {
    loadGPUMemory();
    executeInstances(false);
  }
}

void tps::MasterController::loadGPUMemory() {
  int numberOfExecs = executionInstances_.size();
  while (execMemoryReady < numberOfExecs) {
    if (!executionInstances_[execMemoryReady].canAllocGPUMemory()) break;
    executionInstances_[execMemoryReady].allocCudaMemory();
    execMemoryReady++;
  }
}

void tps::MasterController::executeInstances(bool cpu) {
  while (execFinished < execMemoryReady) {
    if (cpu) executionInstances_[execFinished].runParallelTPS();
    executionInstances_[execFinished].runCudaTPS();
    execFinished++;
  }
}