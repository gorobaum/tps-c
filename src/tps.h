#ifndef TPS_TPS_H_
#define TPS_TPS_H_

#include "image.h"

#include <vector>

#include <opencv2/core/core.hpp>

namespace tps {

class TPS
{
public:
	TPS(std::vector<cv::Point2f> referenceKeypoints, std::vector<cv::Point2f> targetKeypoints, tps::Image targetImage, std::string outputName) :
		referenceKeypoints_(referenceKeypoints),
		targetKeypoints_(targetKeypoints),
		targetImage_(targetImage),
		registredImage(targetImage.getDimensions()[0], targetImage.getDimensions()[1], outputName) {
			solutionX = cv::Mat::zeros(referenceKeypoints_.size()+3, 1, CV_32F);
			solutionY = cv::Mat::zeros(referenceKeypoints_.size()+3, 1, CV_32F);
		}
	void run();
private:
	std::vector<cv::Point2f> referenceKeypoints_;
	std::vector<cv::Point2f> targetKeypoints_;
	tps::Image targetImage_;
	tps::Image registredImage;
	cv::Mat solutionX;
	cv::Mat solutionY;
	void findSolutions();
	cv::Mat createMatrixA();
	float computeRSquared(int x, int xi, int y, int yi);
	cv::Mat solveLinearSystem(cv::Mat A, cv::Mat b);
};

} // namespace

#endif