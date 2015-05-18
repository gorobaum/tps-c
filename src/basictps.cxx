#include "basictps.h"

#include <cmath>
#include <iostream>

void tps::BasicTPS::run() {
	lienarSolver.solveLinearSystems();
	solutionCol = lienarSolver.getSolutionCol();
	solutionRow = lienarSolver.getSolutionRow();
	for (int col = 0; col < width; col++)
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
		registredImage.save(outputName_);
}