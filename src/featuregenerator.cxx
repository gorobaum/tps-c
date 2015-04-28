#include "featuregenerator.h"

#include <iostream>
#include <vector>
#include <cmath>

void tps::FeatureGenerator::run(bool createFeatureImage) {
  gridSizeX = referenceImage_.dimensions[0]*percentage_;
  gridSizeY = referenceImage_.dimensions[1]*percentage_;
  xStep = referenceImage_.dimensions[0]*1.0/(gridSizeX-1);
  yStep = referenceImage_.dimensions[1]*1.0/(gridSizeY-1);
  createReferenceImageFeatures();
  createTargetImageFeatures();
  matches = createMatches();
  if (createFeatureImage) saveFeatureImage();
}

void tps::FeatureGenerator::createReferenceImageFeatures() {
  for (int x = 0; x < gridSizeX; x++)
    for (int y = 0; y < gridSizeY; y++) {
      cv::Point2f newCP(x*xStep, y*yStep);
      referenceKeypoints.push_back(newCP);
      cv::KeyPoint newKP(x*xStep, y*yStep, 0.1);
      keypoints_ref.push_back(newKP);
    }
}

void tps::FeatureGenerator::createTargetImageFeatures() {
  for (int x = 0; x < gridSizeX; x++)
    for (int y = 0; y < gridSizeY; y++) {
      int pos = x*gridSizeY+y;
      cv::Point2f referenceCP = referenceKeypoints[pos];
      std::vector<float> newPoint = applySenoidalDeformationTo(referenceCP.x, referenceCP.y);
      cv::Point2f newCP(newPoint[0], newPoint[1]);
      targetKeypoints.push_back(newCP);
      cv::KeyPoint newKP(newPoint[0], newPoint[1], 0.1);
      keypoints_tar.push_back(newKP);
    }
}

std::vector<float> tps::FeatureGenerator::applySenoidalDeformationTo(float x, float y) {
  float newX = x-8.0*std::sin(y/16.0);
  float newY = y+4.0*std::cos(x/32.0);
  std::vector<float> newPoint;
  newPoint.push_back(newX);
  newPoint.push_back(newY);
  return newPoint;
}

std::vector<cv::DMatch> tps::FeatureGenerator::createMatches() {
  std::vector<cv::DMatch> matches;
  for (int x = 0; x < gridSizeX; x++)
    for (int y = 0; y < gridSizeY; y++) {
      int pos = x*gridSizeY+y;
      cv::DMatch match(pos, pos, -1);
      matches.push_back(match);
    }
  return matches;
}

void tps::FeatureGenerator::saveFeatureImage() {
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);

  cv::Mat img_matches;
  drawMatches(referenceImage_.image, keypoints_ref, targetImage_.image, keypoints_tar,
              matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
              std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  cv::imwrite("refkeypoints-cp.png", img_matches, compression_params);
}