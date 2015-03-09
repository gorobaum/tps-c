#include "tps.h"

#include <cmath>

float tps::TPS::computeRSquared(int x, int xi, int y, int yi) {
	return pow(x-xi,2) + pow(y-yi,2);
}

void tps::TPS::createLinearSystem(std::vector<cv::Point2f> referenceKeypoints_, std::vector<float> coordinate) {
	cv::Mat A = cv::Mat::zeros(referenceKeypoints_.size()+3,referenceKeypoints_.size()+3, CV_32F);

	for (uint j = 3; j < referenceKeypoints_.size()+3; j++) {
		A.at<float>(0,j) = 1;
		A.at<float>(1,j) = referenceKeypoints_[j].x;
		A.at<float>(2,j) = referenceKeypoints_[j].y;
		A.at<float>(j+3,0) = 1;
		A.at<float>(j+3,1) = referenceKeypoints_[j].x;
		A.at<float>(j+3,2) = referenceKeypoints_[j].y;
	}

	for (uint i = 3; i < referenceKeypoints_.size()+3; i++)
		for (uint j = 3; j < referenceKeypoints_.size()+3; j++) {
			float r = computeRSquared(referenceKeypoints_[i].x, referenceKeypoints_[j].x, referenceKeypoints_[i].y, referenceKeypoints_[j].y);
			A.at<float>(i,j) = r*log(r);
		}
		
	cv::Mat b = cv::Mat::zeros(referenceKeypoints_.size()+3, 0, CV_32F);
	for (uint j = 3; j < referenceKeypoints_.size()+3; j++) b.at<float>(j) = coordinate[j];
}