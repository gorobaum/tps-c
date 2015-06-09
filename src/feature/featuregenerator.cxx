#include "featuregenerator.h"

#include <iostream>
#include <vector>
#include <cmath>

void tps::FeatureGenerator::run() {
  gridSizeCol = referenceImage_.getWidth()*percentage_;
  gridSizeRow = referenceImage_.getHeight()*percentage_;
  gridSizeSlice = 1;
  colStep = referenceImage_.getWidth()*1.0/(gridSizeCol-1);
  rowStep = referenceImage_.getHeight()*1.0/(gridSizeRow-1);
  sliceStep = 1;
  std::cout << "referenceImage_.getSlices() = " << referenceImage_.getSlices() << std::endl;
  std::cout << "gridSizeCol = " << gridSizeCol << std::endl;
  std::cout << "gridSizeRow = " << gridSizeRow << std::endl;
  std::cout << "gridSizeSlice = " << gridSizeSlice << std::endl;
  std::cout << "colStep = " << colStep << std::endl;
  std::cout << "rowStep = " << rowStep << std::endl;
  std::cout << "sliceStep = " << sliceStep << std::endl;
  createReferenceImageFeatures();
  createTargetImageFeatures();
  for (int i = 0; i < referenceKeypoints.size(); i++) {
    std::cout << "referenceKeypoints[" << i << "] = " << referenceKeypoints[i][0] << std::endl;
    std::cout << "referenceKeypoints[" << i << "] = " << referenceKeypoints[i][1] << std::endl;
    std::cout << "referenceKeypoints[" << i << "] = " << referenceKeypoints[i][2] << std::endl;
    std::cout << "targetKeypoints[" << i << "] = " << targetKeypoints[i][0] << std::endl;
    std::cout << "targetKeypoints[" << i << "] = " << targetKeypoints[i][1] << std::endl;
    std::cout << "targetKeypoints[" << i << "] = " << targetKeypoints[i][2] << std::endl;
  }
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
  for (int slice = 0; slice < gridSizeSlice; slice++)
    for (int col = 0; col < gridSizeCol; col++)
      for (int row = 0; row < gridSizeRow; row++) {
        int pos = slice*gridSizeCol*gridSizeRow+col*gridSizeRow+row;
        cv::DMatch match(pos, pos, -1);
        matches.push_back(match);
      }

  cv::Mat img_matches;
  drawMatches(refImg, keypoints_ref, tarImg, keypoints_tar,
              matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
              std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  cv::imwrite(filename.c_str(), img_matches, compression_params);
}