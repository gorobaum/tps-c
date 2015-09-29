#include "mastercontroller.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_occupancy.h"

#include <iostream>

void tps::MasterController::run() {
  int numberOfExecs = executionInstances_.size();
  std::cout << "numberOfExecs = " << numberOfExecs << std::endl;
  while (execFinished < numberOfExecs-1) {
    std::cout << "execFinished = " << execFinished << std::endl; 
    std::cout << "execMemoryReady = " << execMemoryReady << std::endl; 
    loadGPUMemory();
    std::cout << "execFinished = " << execFinished << std::endl; 
    std::cout << "execMemoryReady = " << execMemoryReady << std::endl; 
    executeInstances(false);
    std::cout << "execFinished = " << execFinished << std::endl; 
    std::cout << "execMemoryReady = " << execMemoryReady << std::endl; 
  }
}

void tps::MasterController::loadGPUMemory() {
  int count = execFinished+1;
  while (executionInstances_[count].canAllocGPUMemory()) {
    executionInstances_[count].allocCudaMemory();
    execMemoryReady++;
    count++;
  }
}

void tps::MasterController::executeInstances(bool cpu) {
  while (execFinished < execMemoryReady) {
    int currentExec = execFinished + 1;
    if (cpu) executionInstances_[currentExec].runParallelTPS();
    executionInstances_[currentExec].runCudaTPS();
    execFinished++;
  }
}