#include "basictps.h"

#include <cmath>
#include <iostream>

void tps::BasicTPS::run() {
	lienarSolver.solveLinearSystems();
	std::vector<int> dimensions = registredImage.getDimensions();
	solutionX = lienarSolver.getSolutionX();
	solutionY = lienarSolver.getSolutionY();
	for (int x = 0; x < dimensions[0]; x++)
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
		registredImage.save();
}