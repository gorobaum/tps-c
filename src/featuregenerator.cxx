#include "featuregenerator.h"

#include <iostream>
#include <vector>
#include <cmath>

void tps::FeatureGenerator::run() {
  gridSizeCol = referenceImage_.getWidth()*percentage_;
  gridSizeRow = referenceImage_.getHeight()*percentage_;
  gridSizeSlice = referenceImage_.getSlices()*percentage_;
  colStep = referenceImage_.getWidth()*1.0/(gridSizeCol-1);
  rowStep = referenceImage_.getHeight()*1.0/(gridSizeRow-1);
  sliceStep = referenceImage_.getSlices()*1.0/(gridSizeSlice-1);
  std::cout << "gridSizeCol = " << gridSizeCol << std::endl;
  std::cout << "gridSizeRow = " << gridSizeRow << std::endl;
  std::cout << "gridSizeRow = " << gridSizeSlice << std::endl;
  std::cout << "colStep = " << colStep << std::endl;
  std::cout << "rowStep = " << rowStep << std::endl;
  std::cout << "rowStep = " << sliceStep << std::endl;
  createReferenceImageFeatures();
  createTargetImageFeatures();
}

void tps::FeatureGenerator::createReferenceImageFeatures() {
  for (int slice = 0; slice < gridSizeSlice; slice++)
    for (int col = 0; col < gridSizeCol; col++)
      for (int row = 0; row < gridSizeRow; row++) {
        std::vector<float> newCP;
        newCP.push_back(col*colStep);
        newCP.push_back(row*rowStep);
        newCP.push_back(slice*sliceStep);
        referenceKeypoints.push_back(newCP);
        cv::KeyPoint newKP(col*colStep, row*rowStep, 0.1);
        keypoints_ref.push_back(newKP);
      }
}

void tps::FeatureGenerator::createTargetImageFeatures() {
  for (int slice = 0; slice < gridSizeSlice; slice++)
    for (int col = 0; col < gridSizeCol; col++)
      for (int row = 0; row < gridSizeRow; row++) {
        int pos = slice*gridSizeCol*gridSizeRow+col*gridSizeRow+row;
        std::vector<float> referenceCP = referenceKeypoints[pos];
        std::vector<float> newPoint = applySenoidalDeformationTo(referenceCP[0], referenceCP[1], referenceCP[2]);
        targetKeypoints.push_back(newPoint);
        cv::KeyPoint newKP(newPoint[0], newPoint[1], 0.1);
        keypoints_tar.push_back(newKP);
      }
}

std::vector<float> tps::FeatureGenerator::applySenoidalDeformationTo(float x, float y, float z) {
  float newX = x-8*std::sin(y/16);
  float newY = y+4*std::cos(x/32);
  std::vector<float> newPoint;
  newPoint.push_back(newX);
  newPoint.push_back(newY);
  newPoint.push_back(z);
  return newPoint;
}

void tps::FeatureGenerator::drawKeypointsImage(cv::Mat tarImg, std::string filename) {
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);
  
  cv::Mat img_matches;
  drawKeypoints(tarImg, keypoints_tar, img_matches, cv::Scalar::all(-1), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  cv::imwrite(filename.c_str(), img_matches, compression_params);
}

void tps::FeatureGenerator::drawFeatureImage(cv::Mat refImg, cv::Mat tarImg, std::string filename) {
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
  drawMatches(refImg, keypoints_ref, tarImg, keypoints_tar,
              matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
              std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  cv::imwrite(filename.c_str(), img_matches, compression_params);
}