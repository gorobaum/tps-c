#ifndef TPS_PARALLELTPS_H_
#define TPS_PARALLELTPS_H_

#include "tps.h"
#include "linearsystem/armalinearsystems.h"
#include "linearsystem/cudalinearsystems.h"
#include "utils/cudamemory.h"

#include <thread>

namespace tps {

class ParallelTPS : public TPS {
public:
  ParallelTPS(std::vector< std::vector<float> > referenceKeypoints, std::vector< std::vector<float> > targetKeypoints, 
              tps::Image targetImage, std::string outputName) :
    TPS(referenceKeypoints, targetKeypoints, targetImage, outputName),
    lienarSolver(referenceKeypoints, targetKeypoints) {}; 
  void run();
private:
  tps::ArmaLinearSystems lienarSolver;
  uint numberOfThreads = std::thread::hardware_concurrency();
  std::vector<float> solutionX;
  std::vector<float> solutionY;
  std::vector<float> solutionZ;
  void runThread(uint tid);
};

} // namespace

#endif