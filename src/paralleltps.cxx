#include "paralleltps.h"

#include <cmath>
#include <iostream>

void tps::ParallelTPS::runThread(uint tid) {
	int chunck = width/numberOfThreads;
	uint limit = (tid+1)*chunck;
	if ((tid+1) == numberOfThreads) limit = width;
	for (int slice = 0; slice < slices; slice++)
		for (uint col = tid*chunck; col < limit; col++)
			for (int row = 0; row < height; row++) {
				double newCol = solutionCol[0] + col * solutionCol[1] + row * solutionCol[2] + slice * solutionCol[3];
				double newRow = solutionRow[0] + col * solutionRow[1] + row * solutionRow[2] + slice * solutionRow[3];
				double newSlice = solutionSlice[0] + col * solutionSlice[1] + row * solutionSlice[2] + slice * solutionSlice[3];
				for (uint i = 0; i < referenceKeypoints_.size(); i++) {
					float r = computeRSquared(col, referenceKeypoints_[i][0], 
																		row, referenceKeypoints_[i][1],
																		slice, referenceKeypoints_[i][2]);
					if (r != 0.0) {
						newCol += r * log(r) * solutionCol[i+3];
						newRow += r * log(r) * solutionRow[i+3];
						newSlice += r * log(r) * solutionSlice[i+3];
					}
				}
			uchar value = targetImage_.trilinearInterpolation(newCol, newRow, newSlice);
			registredImage.changePixelAt(col, row, slice, value);
		}
}

void tps::ParallelTPS::run() {
  armalienarSolver.solveLinearSystems();
	solutionCol = armalienarSolver.getSolutionCol();
	solutionRow = armalienarSolver.getSolutionRow();
	solutionSlice = armalienarSolver.getSolutionSlice();
	std::vector<std::thread> th;

	for (uint i = 0; i < numberOfThreads; ++i)
    th.push_back(std::thread(&tps::ParallelTPS::runThread, this, i));

  for(uint i = 0; i < numberOfThreads; ++i) th[i].join();

	registredImage.save(outputName_);
}