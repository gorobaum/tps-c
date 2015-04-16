#ifndef TPS_BASICTPS_H_
#define TPS_BASICTPS_H_

#include "tps.h"
#include "cudalinearsystems.h"

namespace tps {

class BasicTPS : public TPS {
public:
  BasicTPS(std::vector<cv::Point2f> referenceKeypoints, std::vector<cv::Point2f> targetKeypoints, tps::Image targetImage, std::string outputName) :
    TPS(referenceKeypoints, targetKeypoints, targetImage, outputName),
    lienarSolver(referenceKeypoints, targetKeypoints) {};	
  void run();
private:
  tps::CudaLinearSystems lienarSolver;
  std::vector<float> solutionX;
  std::vector<float> solutionY;
};

} // namespace

#endif