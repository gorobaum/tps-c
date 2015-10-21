#include "mastercontroller.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_occupancy.h"

#include <iostream>

struct ThreadData {
  int tId;
  std::vector<tps::TpsInstance> executionInstances;
  int startPoint;
  int endPoint;
};

void tps::MasterController::run() {
  int numberOfExecs = executionInstances_.size();
  std::cout << "numberOfExecs = " << numberOfExecs << std::endl;
  while (execFinished < numberOfExecs) {
    loadGPUMemory();
    
    pthread_t threads[numberOfThreads];

    for (int i = 0; i < numberOfThreads; i++) {
        ThreadData* data = new ThreadData;
        data->tId = i;
        data->executionInstances = executionInstances_;
        data->startPoint = execFinished;
        data->endPoint = execMemoryReady;
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
    execFinished = execMemoryReady;
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

  int execFinished = threadData->startPoint;
  int execMemoryReady = threadData->endPoint;
  int numInstancesForNow = execMemoryReady - execFinished;

  for (int i = threadData->tId; i < numInstancesForNow; i += 4) {
    threadData->executionInstances[execFinished+i].runCudaTPS();
  }
}