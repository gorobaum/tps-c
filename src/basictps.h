#ifndef TPS_BASICTPS_H_
#define TPS_BASICTPS_H_

#include "tps.h"

namespace tps {

class BasicTPS : public TPS {
using TPS::TPS;
public:
	void run();
};

} // namespace

#endif