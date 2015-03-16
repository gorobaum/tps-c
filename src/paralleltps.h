#ifndef TPS_PARALLELTPS_H_
#define TPS_PARALLELTPS_H_

#include "tps.h"

namespace tps {

class ParallelTPS : public TPS {
using TPS::TPS;
public:
	void run();
private:
	void findSolutions() ;
	cv::Mat createMatrixA() ;
	cv::Mat solveLinearSystem(cv::Mat A, cv::Mat b);
};

} // namespace

#endif