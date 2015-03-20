#include "tps.h"


void tps::TPS::findSolutions() {
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

cv::Mat tps::TPS::solveLinearSystem(cv::Mat A, cv::Mat b) {
	cv::Mat solution = cv::Mat::zeros(referenceKeypoints_.size()+3, 1, CV_32F);
	cv::solve(A, b, solution, cv::DECOMP_EIG);
	return solution;
}

cv::Mat tps::TPS::createMatrixA() {
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