#ifndef TPS_PARALLELTPS_H_
#define TPS_PARALLELTPS_H_

#include "tps.h"
#include "OPCVlinearsystems.h"
#include "cudalinearsystems.h"

#include <thread>

namespace tps {

class ParallelTPS : public TPS {
public:
  ParallelTPS(std::vector<cv::Point2f> referenceKeypoints, std::vector<cv::Point2f> targetKeypoints, tps::Image targetImage, std::string outputName, tps::CudaMemory& cm) :
    TPS(referenceKeypoints, targetKeypoints, targetImage, outputName),
    cudalienarSolver(referenceKeypoints, targetKeypoints),
    cm_(cm) {}; 
	void run();
private:
	void runThread(uint tid);
  tps::CudaMemory& cm_;
	uint numberOfThreads = std::thread::hardware_concurrency();
  tps::CudaLinearSystems cudalienarSolver;
  std::vector<float> solutionCol;
  std::vector<float> solutionRow;
};

} // namespace

#endif