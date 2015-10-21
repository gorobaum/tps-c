#include "mastercontroller.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_occupancy.h"

#include <iostream>

struct ThreadData {
  int tId;
  std::vector<tps::TpsInstance> executionInstances;
  int numInstancesForNow;
};

void tps::MasterController::run() {
  int numberOfExecs = executionInstances_.size();
  while (execFinished < numberOfExecs) {
    loadGPUMemory();
    
    pthread_t threads[numberOfThreads];
    int numInstancesForNow = execMemoryReady - execFinished;

    for (int i = 0; i < numberOfThreads; i++) {
        ThreadData* data = new ThreadData;
        data->tId = i;
        data->executionInstances = executionInstances_;
        data->numInstancesForNow = numInstancesForNow;
        if (pthread_create(&threads[i], NULL, executeInstances, static_cast<void*>(data))) {
            fprintf(stderr, "Error creating threadn");
            return 1;
        }
    }

    for (int i = 0; i < numberOfThreads; i++) {
        if(pthread_join(threads[i], NULL)) {
            fprintf(stderr, "Error joining threadn");
            return 2;
        }
    }
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

void tps::MasterController::executeInstances(void* data) {
  ThreadData* threadData = static_cast<ThreadData*>(data);
  threadData->tId;
  std::cout << threadData->tId << std::endl;
  threadData->executionInstances;
  threadData->numInstancesForNow;
}