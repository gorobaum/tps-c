#include "optimizedtps.h"

#include <cmath>
#include <iostream>

void tps::OptimizedTPS::run() {
	findSolutions();
	std::vector<int> dimensions = registredImage.getDimensions();
	for (int x = 0; x < dimensions[0]; x++)
		for (int y = 0; y < dimensions[0]; y++) {
			cv::Mat aux = cv::Mat::zeros(referenceKeypoints_.size()+3, 1, CV_32F);
			aux.at<float>(0) = 1.0;
			aux.at<float>(1) = x;
			aux.at<float>(2) = y;
			for (uint i = 0; i < referenceKeypoints_.size(); i++) {
				float r = computeRSquaredOptimized(x, y, referenceKeypoints_, i);
				if (r != 0.0) aux.at<float>(i+3) = r*log(r);
			}
			double newX = aux.dot(solutionX);
			double newY = aux.dot(solutionY);
			uchar value = targetImage_.bilinearInterpolation<uchar>(newX, newY);
			registredImage.changePixelAt(x, y, value);
		}
		registredImage.save();
}

void tps::OptimizedTPS::findSolutions() {
	cv::Mat A = createMatrixA();

	cv::Mat bx = cv::Mat::zeros(referenceKeypoints_.size()+3, 1, CV_32F);
	cv::Mat by = cv::Mat::zeros(referenceKeypoints_.size()+3, 1, CV_32F);
	for (uint i = 0; i < referenceKeypoints_.size(); i++) {
		bx.at<float>(i+3) = targetKeypoints_[i].x;
		by.at<float>(i+3) = targetKeypoints_[i].y;
	}

	solutionX = solveLinearSystem(A, bx);
	solutionY = solveLinearSystem(A, by);
}

cv::Mat tps::OptimizedTPS::solveLinearSystem(cv::Mat A, cv::Mat b) {
	cv::Mat solution = cv::Mat::zeros(referenceKeypoints_.size()+3, 1, CV_32F);
	cv::solve(A, b, solution, cv::DECOMP_EIG);
	return solution;
}

cv::Mat tps::OptimizedTPS::createMatrixA() {
	cv::Mat A = cv::Mat::zeros(referenceKeypoints_.size()+3,referenceKeypoints_.size()+3, CV_32F);

	for (uint j = 0; j < referenceKeypoints_.size(); j++) {
		A.at<float>(0,j+3) = 1;
		A.at<float>(1,j+3) = referenceKeypoints_[j].x;
		A.at<float>(2,j+3) = referenceKeypoints_[j].y;
		A.at<float>(j+3,0) = 1;
		A.at<float>(j+3,1) = referenceKeypoints_[j].x;
		A.at<float>(j+3,2) = referenceKeypoints_[j].y;
	}

	for (uint i = 0; i < referenceKeypoints_.size(); i++)
		for (uint j = 0; j < referenceKeypoints_.size(); j++) {
			float r = computeRSquared(referenceKeypoints_[i].x, referenceKeypoints_[j].x, referenceKeypoints_[i].y, referenceKeypoints_[j].y);
			if (r != 0.0) A.at<float>(i+3,j+3) = r*log(r);
		}

	return A;
}

double tps::OptimizedTPS::computeRSquaredOptimized(int x, int y, std::vector<cv::Point2f> keypoints, int pos) {
	if (rSquaredX.at<float>(pos,x) == 0.0 ) 
		rSquaredX.at<float>(pos,x) = pow(x-keypoints[pos].x, 2);
	if (rSquaredY.at<float>(pos,y) == 0.0 ) 
		rSquaredY.at<float>(pos,y) = pow(y-keypoints[pos].y, 2);
	return rSquaredX.at<float>(pos,x) + rSquaredY.at<float>(pos,y);
}