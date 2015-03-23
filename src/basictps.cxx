#include "basictps.h"

#include <cmath>
#include <iostream>

void tps::BasicTPS::run() {
	findSolutions();
	std::cout << solutionX.at<float>(0) << " - " << solutionY.at<float>(0) << std::endl;
	std::vector<int> dimensions = registredImage.getDimensions();
	for (int x = 0; x < dimensions[0]; x++)
		for (int y = 0; y < dimensions[1]; y++) {
			double newX = solutionX.at<float>(0) + x*solutionX.at<float>(1) + y*solutionX.at<float>(2);
			double newY = solutionY.at<float>(0) + x*solutionY.at<float>(1) + y*solutionY.at<float>(2);
			for (uint i = 0; i < referenceKeypoints_.size(); i++) {
				float r = computeRSquared(x, referenceKeypoints_[i].x, y, referenceKeypoints_[i].y);
				if (r != 0.0) {
					newX += r*log(r) * solutionX.at<float>(i+3);
					newY += r*log(r) * solutionY.at<float>(i+3);
				}
			}
			std::cout << "[" << x << "][" << y << "] = (" << newX << ")(" << newY << ")" << std::endl;
			uchar value = targetImage_.bilinearInterpolation<uchar>(newX, newY);
			registredImage.changePixelAt(x, y, value);
		}
		registredImage.save();
}