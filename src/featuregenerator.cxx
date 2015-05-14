#include "featuregenerator.h"

#include <iostream>
#include <vector>
#include <cmath>

void tps::FeatureGenerator::run(bool createFeatureImage) {
  gridSizeCol = referenceImage_.getWidth()*percentage_;
  gridSizeRow = referenceImage_.getHeight()*percentage_;
  colStep = referenceImage_.getWidth()*1.0/(gridSizeCol-1);
  rowStep = referenceImage_.getHeight()*1.0/(gridSizeRow-1);
  std::cout << "gridSizeCol = " << gridSizeCol << std::endl;
  std::cout << "gridSizeRow = " << gridSizeRow << std::endl;
  std::cout << "colStep = " << colStep << std::endl;
  std::cout << "rowStep = " << rowStep << std::endl;
  createReferenceImageFeatures();
  createTargetImageFeatures();
  if (createFeatureImage) saveFeatureImage();
}

void tps::FeatureGenerator::createReferenceImageFeatures() {
  for (int col = 0; col < gridSizeCol; col++)
    for (int row = 0; row < gridSizeRow; row++) {
      cv::Point2i newCP(col*colStep, row*rowStep);
      referenceKeypoints.push_back(newCP);
      cv::KeyPoint newKP(col*colStep, row*rowStep, 0.1);
      keypoints_ref.push_back(newKP);
    }
}

void tps::FeatureGenerator::createTargetImageFeatures() {
  for (int col = 0; col < gridSizeCol; col++)
    for (int row = 0; row < gridSizeRow; row++) {
      int pos = col*gridSizeRow+row;
      cv::Point2i referenceCP = referenceKeypoints[pos];
      std::vector<int> newPoint = applySenoidalDeformationTo(referenceCP.x, referenceCP.y);
      cv::Point2i newCP(newPoint[0], newPoint[1]);
      targetKeypoints.push_back(newCP);
      cv::KeyPoint newKP(newPoint[0], newPoint[1], 0.1);
      keypoints_tar.push_back(newKP);
    }
}

std::vector<int> tps::FeatureGenerator::applySenoidalDeformationTo(int x, int y) {
  int newX = x-8*std::sin(y/16);
  int newY = y+4*std::cos(x/32);
  std::vector<int> newPoint;
  newPoint.push_back(newX);
  newPoint.push_back(newY);
  return newPoint;
}

void tps::FeatureGenerator::saveFeatureImage() {
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);

  std::vector<cv::DMatch> matches;
  for (int col = 0; col < gridSizeCol; col++)
    for (int row = 0; row < gridSizeRow; row++) {
      int pos = col*gridSizeRow+row;
      cv::DMatch match(pos, pos, -1);
      matches.push_back(match);
    }

  cv::Mat img_matches;
  drawMatches(refImg_, keypoints_ref, tarImg_, keypoints_tar,
              matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
              std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  drawKeypoints(tarImg_, keypoints_tar, img_matches, cv::Scalar::all(-1), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  cv::imwrite("refkeypoints-cp.png", img_matches, compression_params);
}