#ifndef TPS_TPS_H_
#define TPS_TPS_H_

#include "image.h"
#include "cplinearsystems.h"

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace tps {

class TPS
{
public:
	TPS(std::vector<cv::Point2f> referenceKeypoints, std::vector<cv::Point2f> targetKeypoints, tps::Image targetImage, std::string outputName) :
		referenceKeypoints_(referenceKeypoints),
		targetKeypoints_(targetKeypoints),
		targetImage_(targetImage),
		registredImage(targetImage.getDimensions()[0], targetImage.getDimensions()[1], outputName) {};
	virtual void run() = 0;
protected:
	std::vector<cv::Point2f> referenceKeypoints_;
	std::vector<cv::Point2f> targetKeypoints_;
	tps::Image targetImage_;
	tps::Image registredImage;
	float computeRSquared(float x, float xi, float y, float yi) {return pow(x-xi,2) + pow(y-yi,2);};
};

} // namespace

#endif