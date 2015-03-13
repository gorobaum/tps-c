#ifndef TPS_OPTIMIZEDTPS_H_
#define TPS_OPTIMIZEDTPS_H_

#include "tps.h"

namespace tps {

class OptimizedTPS : public TPS {
public:
	OptimizedTPS(std::vector<cv::Point2f> referenceKeypoints, std::vector<cv::Point2f> targetKeypoints, tps::Image targetImage, std::string outputName) :
		TPS(referenceKeypoints, targetKeypoints, targetImage, outputName) {
			rSquaredX = cv::Mat::zeros(referenceKeypoints.size()+3, targetImage.getDimensions()[0], CV_32F);
			rSquaredY = cv::Mat::zeros(referenceKeypoints.size()+3, targetImage.getDimensions()[1], CV_32F);
		}
	void run();
private:
	cv::Mat rSquaredX;
	cv::Mat rSquaredY;
	void findSolutions();
	cv::Mat createMatrixA() ;
	cv::Mat solveLinearSystem(cv::Mat A, cv::Mat b);
	double computeRSquaredOptimized(int x, int y, std::vector<cv::Point2f> keypoints, int pos);
};

} // namespace

#endif