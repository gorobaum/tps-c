#ifndef TPS_CUDASTREAM_H_
#define TPS_CUDASTREAM_H_

#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_occupancy.h"

namespace tps {
  
class CudaStream {
public:
  CudaStream() {
    cudaStreamCreate(&stream);
  };
  void destroyStream() {
    cudaStreamDestroy(stream);
  };
  bool isStreamFree() {
    return (cudaStreamQuery(stream) == cudaSuccess);
  };
private:
  cudaStream_t stream;
};

} // namespace

#endif