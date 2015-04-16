#ifndef TPS_PARALLELTPS_H_
#define TPS_PARALLELTPS_H_

#include "tps.h"
#include "cudalinearsystems.h"


#include <thread>

namespace tps {

class ParallelTPS : public TPS {
public:
  ParallelTPS(std::vector<cv::Point2f> referenceKeypoints, std::vector<cv::Point2f> targetKeypoints, tps::Image targetImage, std::string outputName) :
    TPS(referenceKeypoints, targetKeypoints, targetImage, outputName),
    lienarSolver(referenceKeypoints, targetKeypoints) {}; 
	void run();
private:
	void runThread(uint tid);
	uint numberOfThreads = std::thread::hardware_concurrency();
  tps::CudaLinearSystems lienarSolver;
  std::vector<float> solutionX;
  std::vector<float> solutionY;
};

} // namespace

#endif