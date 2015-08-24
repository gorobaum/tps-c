#ifndef TPS_CUDATPS_H_
#define TPS_CUDATPS_H_

#include "tps.h"
#include "linearsystem/armalinearsystems.h"
#include "linearsystem/cudalinearsystems.h"
#include "utils/cudamemory.h"

namespace tps {

class CudaTPS : public TPS {
  public:
    CudaTPS(std::vector< std::vector<float> > referenceKeypoints, std::vector< std::vector<float> > targetKeypoints, tps::Image targetImage, std::string outputName, tps::CudaMemory& cm) :
      TPS(referenceKeypoints, targetKeypoints, targetImage, outputName),
      lienarSolver(referenceKeypoints, targetKeypoints),
      cm_(cm) {}; 
    void run();
  private:
    tps::CudaLinearSystems lienarSolver;
    tps::CudaMemory& cm_;
    float* solutionPointer(std::vector<float> solution);
    short *regImage;  
};

}

#endif