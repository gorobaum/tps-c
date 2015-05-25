#include "paralleltps.h"

#include <cmath>
#include <iostream>

void tps::ParallelTPS::runThread(uint tid) {
	int chunck = width/numberOfThreads;
	uint limit = (tid+1)*chunck;
	if ((tid+1) == numberOfThreads) limit = width;
	for (uint col = tid*chunck; col < limit; col++)
		for (int row = 0; row < height; row++) {
			double newCol = solutionCol[0] + col * solutionCol[1] + row * solutionCol[2];
			double newRow = solutionRow[0] + col * solutionRow[1] + row * solutionRow[2];
			for (uint i = 0; i < referenceKeypoints_.size(); i++) {
				float r = computeRSquared(col, referenceKeypoints_[i].x, row, referenceKeypoints_[i].y);
				if (r != 0.0) {
					newCol += r * log(r) * solutionCol[i+3];
					newRow += r * log(r) * solutionRow[i+3];
				}
			}
			uchar value = targetImage_.bilinearInterpolation(newCol, newRow);
			registredImage.changePixelAt(col, row, value);
		}
}

void tps::ParallelTPS::run() {
	cudalienarSolver.solveLinearSystems(cm_);
	solutionCol = cudalienarSolver.getSolutionCol();
	solutionRow = cudalienarSolver.getSolutionRow();
	std::vector<std::thread> th;

	for (uint i = 0; i < numberOfThreads; ++i)
    th.push_back(std::thread(&tps::ParallelTPS::runThread, this, i));

  for(uint i = 0; i < numberOfThreads; ++i) th[i].join();

	registredImage.save(outputName_);
}