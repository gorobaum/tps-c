#ifndef TPS_TPS_H_
#define TPS_TPS_H_

#include <vector>

#include <opencv2/core/core.hpp>

namespace tps {

class TPS
{
public:
	TPS(std::vector<cv::Point2f> referenceKeypoints, std::vector<cv::Point2f> targetKeypoints) :
		referenceKeypoints_(referenceKeypoints),
		targetKeypoints_(targetKeypoints) {}
	void run();
private:
	std::vector<cv::Point2f> referenceKeypoints_;
	std::vector<cv::Point2f> targetKeypoints_;
	void createLinearSystems();
	void createLinearSystem(std::vector<cv::Point2f> referenceKeypoints_, std::vector<float> coordinate);
	float computeRSquared(int x, int xi, int y, int yi);
	void solveLinearSystems();
	void solveLinearSystem(cv::Mat A, cv::Mat b, cv::Mat& x);
};

} // namespace

#endif