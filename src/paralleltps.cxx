#include "paralleltps.h"

#include <cmath>
#include <iostream>

void tps::ParallelTPS::runThread(uint tid) {
	std::vector<int> dimensions = registredImage.getDimensions();
	int chunck = dimensions[0]/numberOfThreads;
	uint limit = (tid+1)*chunck;
	if ((tid+1) == numberOfThreads) limit = dimensions[0];
	for (uint x = tid*chunck; x < limit; x++)
		for (int y = 0; y < dimensions[1]; y++) {
			double newX = solutionX[0] + x * solutionX[1] + y * solutionX[2];
			double newY = solutionY[0] + x * solutionY[1] + y * solutionY[2];
			for (uint i = 0; i < referenceKeypoints_.size(); i++) {
				float r = computeRSquared(x, referenceKeypoints_[i].x, y, referenceKeypoints_[i].y);
				if (r != 0.0) {
					newX += r * log(r) * solutionX[i+3];
					newY += r * log(r) * solutionY[i+3];
				}
			}
			uchar value = targetImage_.bilinearInterpolation<uchar>(newX, newY);
			registredImage.changePixelAt(x, y, value);
		}
}

void tps::ParallelTPS::run() {
	lienarSolver.solveLinearSystems();
	solutionX = lienarSolver.getSolutionX();
	solutionY = lienarSolver.getSolutionY();
	std::vector<std::thread> th;

	for (uint i = 0; i < numberOfThreads; ++i) {
    th.push_back(std::thread(&tps::ParallelTPS::runThread, *this, i));
  }

  for(auto &t : th){
    t.join();
  }

	registredImage.save();
}