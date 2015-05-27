#ifndef TPS_BASICTPS_H_
#define TPS_BASICTPS_H_

#include "tps.h"
#include "OPCVlinearsystems.h"

namespace tps {

class BasicTPS : public TPS {
public:
  BasicTPS(std::vector< std::vector<float> > referenceKeypoints, std::vector< std::vector<float> > targetKeypoints, tps::Image targetImage, std::string outputName) :
    TPS(referenceKeypoints, targetKeypoints, targetImage, outputName),
    lienarSolver(referenceKeypoints, targetKeypoints) {};	
  void run();
private:
  tps::OPCVLinearSystems lienarSolver;
  std::vector<float> solutionCol;
  std::vector<float> solutionRow;
};

} // namespace

#endif