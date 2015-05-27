#ifndef TPS_CUDATPS_H_
#define TPS_CUDATPS_H_

#include "tps.h"
#include "cudalinearsystems.h"
#include "cudamemory.h"

namespace tps {

class CudaTPS : public TPS {
  public:
    CudaTPS(std::vector< std::vector<float> > referenceKeypoints, std::vector< std::vector<float> > targetKeypoints, tps::Image targetImage, std::string outputName, tps::CudaMemory& cm) :
      TPS(referenceKeypoints, targetKeypoints, targetImage, outputName),
      cudalienarSolver(referenceKeypoints, targetKeypoints),
      cm_(cm) {}; 
    void run();
  private:
    tps::CudaLinearSystems cudalienarSolver;
    tps::CudaMemory& cm_;
    uchar *regImage;  
};

}

#endif