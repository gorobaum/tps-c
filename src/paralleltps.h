#ifndef TPS_PARALLELTPS_H_
#define TPS_PARALLELTPS_H_

#include "tps.h"
#include "OPCVlinearsystems.h"
#include "cudalinearsystems.h"
#include "cudamemory.h"

#include <thread>

namespace tps {

class ParallelTPS : public TPS {
public:
  ParallelTPS(std::vector< std::vector<float> > referenceKeypoints, std::vector< std::vector<float> > targetKeypoints, tps::Image targetImage, std::string outputName, tps::CudaMemory& cm) :
    TPS(referenceKeypoints, targetKeypoints, targetImage, outputName),
    cudalienarSolver(referenceKeypoints, targetKeypoints),
    cm_(cm) {}; 
	void run();
private:
  tps::CudaLinearSystems cudalienarSolver;
  tps::CudaMemory& cm_;
  uint numberOfThreads = std::thread::hardware_concurrency();
  std::vector<float> solutionCol;
  std::vector<float> solutionRow;
	void runThread(uint tid);
};

} // namespace

#endif