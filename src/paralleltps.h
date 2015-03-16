#ifndef TPS_PARALLELTPS_H_
#define TPS_PARALLELTPS_H_

#include "tps.h"

#include <thread>

namespace tps {

class ParallelTPS : public TPS {
using TPS::TPS;
public:
	void run();
private:
	void findSolutions();
	void runThread(int tid);
	cv::Mat createMatrixA();
	cv::Mat solveLinearSystem(cv::Mat A, cv::Mat b);
	unsigned numberOfThreads = std::thread::hardware_concurrency();
};

} // namespace

#endif