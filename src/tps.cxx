#include "tps.h"

#include <cmath>
#include <iostream>

void tps::TPS::run() {
	findSolutions();
	
}

void tps::TPS::findSolutions() {
	cv::Mat A = createMatrixA();

	cv::Mat bx = cv::Mat::zeros(referenceKeypoints_.size()+3, 1, CV_32F);
	cv::Mat by = cv::Mat::zeros(referenceKeypoints_.size()+3, 1, CV_32F);
	for (uint i = 3; i < referenceKeypoints_.size()+3; i++) {
		bx.at<float>(i) = targetKeypoints_[i-3].x;
		by.at<float>(i) = targetKeypoints_[i-3].y;
	}

	solutionX = solveLinearSystem(A, bx);
	solutionY = solveLinearSystem(A, by);
}

float tps::TPS::computeRSquared(int x, int xi, int y, int yi) {
	return pow(x-xi,2) + pow(y-yi,2);
}

cv::Mat tps::TPS::solveLinearSystem(cv::Mat A, cv::Mat b) {
	cv::Mat solution = cv::Mat::zeros(referenceKeypoints_.size()+3, 1, CV_32F);
	cv::solve(A, b, solution);
	return solution;
}

cv::Mat tps::TPS::createMatrixA() {
	cv::Mat A = cv::Mat::zeros(referenceKeypoints_.size()+3,referenceKeypoints_.size()+3, CV_32F);

	for (uint j = 3; j < referenceKeypoints_.size()+3; j++) {
		A.at<float>(0,j) = 1;
		A.at<float>(1,j) = referenceKeypoints_[j-3].x;
		A.at<float>(2,j) = referenceKeypoints_[j-3].y;
		A.at<float>(j+3,0) = 1;
		A.at<float>(j+3,1) = referenceKeypoints_[j-3].x;
		A.at<float>(j+3,2) = referenceKeypoints_[j-3].y;
	}

	for (uint i = 3; i < referenceKeypoints_.size()+3; i++)
		for (uint j = 3; j < referenceKeypoints_.size()+3; j++) {
			float r = computeRSquared(referenceKeypoints_[i-3].x, referenceKeypoints_[j-3].x, referenceKeypoints_[i-3].y, referenceKeypoints_[j-3].y);
			if (r != 0) A.at<float>(i,j) = r*log(r);
		}

	return A;
}