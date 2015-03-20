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
	void runThread(uint tid);
	uint numberOfThreads = std::thread::hardware_concurrency();
};

} // namespace

#endif