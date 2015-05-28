#ifndef TPS_PARALLELTPS_H_
#define TPS_PARALLELTPS_H_

#include "tps.h"
#include "OPCVlinearsystems.h"
#include "cudalinearsystems.h"
#include "armalinearsystems.h"
#include "cudamemory.h"

#include <thread>

namespace tps {

class ParallelTPS : public TPS {
public:
  ParallelTPS(std::vector< std::vector<float> > referenceKeypoints, std::vector< std::vector<float> > targetKeypoints, tps::Image targetImage, std::string outputName) :
    TPS(referenceKeypoints, targetKeypoints, targetImage, outputName),
    opcvlienarSolver(referenceKeypoints, targetKeypoints),
    cudalienarSolver(referenceKeypoints, targetKeypoints),
    armalienarSolver(referenceKeypoints, targetKeypoints) {}; 
	void run();
private:
  tps::OPCVLinearSystems opcvlienarSolver;
  tps::CudaLinearSystems cudalienarSolver;
  tps::ArmaLinearSystems armalienarSolver;
  uint numberOfThreads = std::thread::hardware_concurrency();
  std::vector<float> solutionCol;
  std::vector<float> solutionRow;
	void runThread(uint tid);
};

} // namespace

#endif