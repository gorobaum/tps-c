#ifndef TPS_BASICTPS_H_
#define TPS_BASICTPS_H_

#include "tps.h"

namespace tps {

class BasicTPS : public TPS {
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