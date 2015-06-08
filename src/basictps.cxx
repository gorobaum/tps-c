#include "basictps.h"

#include <cmath>
#include <iostream>

void tps::BasicTPS::run() {
	lienarSolver.solveLinearSystems();
	solutionCol = lienarSolver.getSolutionCol();
	solutionRow = lienarSolver.getSolutionRow();
	solutionSlice = lienarSolver.getSolutionSlice();
	for (int col = 0; col < width; col++)
		for (int row = 0; row < height; row++) 
			for (int slice = 0; slice < slices; slice++) {
				double newCol = solutionCol[0] + col * solutionCol[1] + row * solutionCol[2] + slice * solutionCol[3];
				double newRow = solutionRow[0] + col * solutionRow[1] + row * solutionRow[2] + slice * solutionRow[3];
				double newSlice = solutionSlice[0] + col * solutionSlice[1] + row * solutionSlice[2] + slice * solutionSlice[3];
				for (uint i = 0; i < referenceKeypoints_.size(); i++) {
					float r = computeRSquared(col, referenceKeypoints_[i][0], row, referenceKeypoints_[i][1]);
					if (r != 0.0) {
						newCol += r * log(r) * solutionCol[i+3];
						newRow += r * log(r) * solutionRow[i+3];
						newSlice += r * log(r) * solutionSlice[i+3];
					}
				}
			uchar value = targetImage_.bilinearInterpolation(newCol, newRow, newSlice);
			registredImage.changePixelAt(col, row, slice, value);
		}
		registredImage.save(outputName_);
}