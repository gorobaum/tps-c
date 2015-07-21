#include "featuregenerator.h"

#include <iostream>
#include <vector>
#include <cmath>

void tps::FeatureGenerator::run() {
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
}

bool tps::FeatureGenerator::checkSector(float col, float row) {
  if (col < (referenceImage_.getWidth()/2))
    if (row > (referenceImage_.getHeight()/2)-10)
      return false;
  return true;
}

void tps::FeatureGenerator::createReferenceImageFeatures() {
  for (int col = 0; col < gridSizeCol; col++)
    for (int row = 0; row < gridSizeRow; row++) {
      if (checkSector(col*colStep, row*rowStep)) {
        std::vector<float> newCP;
        newCP.push_back(col*colStep);
        newCP.push_back(row*rowStep);
        referenceKeypoints.push_back(newCP);
        cv::KeyPoint newKP(col*colStep, row*rowStep, 0.1);
        keypoints_ref.push_back(newKP);
      }
    }
}

void tps::FeatureGenerator::createTargetImageFeatures() {
  targetKeypoints = referenceKeypoints;
  keypoints_tar = keypoints_ref;
}

std::vector<float> tps::FeatureGenerator::applySenoidalDeformationTo(float x, float y) {
  float newX = x-8*std::sin(y/16);
  float newY = y+4*std::cos(x/32);
  std::vector<float> newPoint;
  newPoint.push_back(newX);
  newPoint.push_back(newY);
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

void tps::FeatureGenerator::addRefKeypoints(std::vector< std::vector< float > > newKPs) {
  addKeypoints(keypoints_ref, newKPs);
  for (std::vector<std::vector< float >>::iterator it = newKPs.begin() ; it != newKPs.end(); ++it) {
    std::vector< float > newPoint = *it;
    referenceKeypoints.push_back(newPoint);
  }
}

void tps::FeatureGenerator::addTarKeypoints(std::vector< std::vector< float > > newKPs) {
  addKeypoints(keypoints_tar, newKPs);
  for (std::vector<std::vector< float >>::iterator it = newKPs.begin() ; it != newKPs.end(); ++it) {
    std::vector< float > newPoint = *it;
    targetKeypoints.push_back(newPoint);
  }
}

void tps::FeatureGenerator::addKeypoints(std::vector<cv::KeyPoint> &keypoints, std::vector< std::vector< float > > newKPs) {
  for (std::vector<std::vector< float >>::iterator it = newKPs.begin() ; it != newKPs.end(); ++it) {
    std::vector< float > newPoint = *it;
    cv::KeyPoint newKP(newPoint[0], newPoint[1], 0.1);
    keypoints.push_back(newKP);
  }
}

void tps::FeatureGenerator::drawFeatureImage(cv::Mat refImg, cv::Mat tarImg, std::string filename) {
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);

  std::vector<cv::DMatch> matches;
  for (int i = 0; i < targetKeypoints.size(); i++) {
    cv::DMatch match(i, i, -1);
    matches.push_back(match);
  }

  cv::Mat img_matches;
  drawMatches(refImg, keypoints_ref, tarImg, keypoints_tar,
              matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
              std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  cv::imwrite(filename.c_str(), img_matches, compression_params);
}