#include "surf.h"

void tps::Surf::detectFeatures() {
	detector.detect(referenceImage_, keypoints_ref);
  detector.detect(targetImage_, keypoints_tar);
}

void tps::Surf::extractDescriptors() {
  extractor.compute(referenceImage_, keypoints_ref, descriptors_ref);
  extractor.compute(targetImage_, keypoints_tar, descriptors_tar);
}

void tps::Surf::matchDescriptors() {
	std::vector<cv::DMatch> matches;
  matcher.match(descriptors_ref, descriptors_tar, matches);

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for (int i = 0; i < descriptors_ref.rows; i++) { 
    double dist = matches[i].distance;
    if(dist < min_dist) min_dist = dist;
    if(dist > max_dist) max_dist = dist;
  }

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )

  for (int i = 0; i < descriptors_ref.rows; i++)
    if (matches[i].distance < 5*min_dist)
      good_matches.push_back( matches[i]);

  for (uint i = 0; i < good_matches.size(); i++) {
    //-- Get the keypoints from the good matches
    std::vector<float> refNewPt;
    refNewPt.push_back(keypoints_ref[ good_matches[i].queryIdx ].pt.x);
    refNewPt.push_back(keypoints_ref[ good_matches[i].queryIdx ].pt.y);
    referenceKeypoints.push_back(refNewPt);
    std::vector<float> tarNewPt;
    tarNewPt.push_back(keypoints_tar[ good_matches[i].trainIdx ].pt.x);
    tarNewPt.push_back(keypoints_tar[ good_matches[i].trainIdx ].pt.y);
    targetKeypoints.push_back(tarNewPt);
  }
}

void tps::Surf::drawKeypointsImage(cv::Mat tarImg, std::string filename) {
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);
  
  cv::Mat img_matches;
  drawKeypoints(tarImg, keypoints_tar, img_matches, cv::Scalar::all(-1), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  cv::imwrite(filename.c_str(), img_matches, compression_params);
}

void tps::Surf::drawFeatureImage(cv::Mat refImg, cv::Mat tarImg, std::string filename) {
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);

  cv::Mat img_matches;
  drawMatches(refImg, keypoints_ref, tarImg, keypoints_tar,
              good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
              std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  cv::imwrite(filename.c_str(), img_matches, compression_params);
}

void tps::Surf::addRefKeypoints(std::vector< std::vector< float > > newKPs) {
  addKeypoints(keypoints_ref, newKPs);
  for (std::vector<std::vector< float >>::iterator it = newKPs.begin() ; it != newKPs.end(); ++it) {
    std::vector< float > newPoint = *it;
    referenceKeypoints.push_back(newPoint);
  }
}

void tps::Surf::addTarKeypoints(std::vector< std::vector< float > > newKPs) {
  addKeypoints(keypoints_tar, newKPs);
  for (std::vector<std::vector< float >>::iterator it = newKPs.begin() ; it != newKPs.end(); ++it) {
    std::vector< float > newPoint = *it;
    targetKeypoints.push_back(newPoint);
  }
}

void tps::Surf::addKeypoints(std::vector<cv::KeyPoint> &keypoints, std::vector< std::vector< float > > newKPs) {
  for (std::vector<std::vector< float >>::iterator it = newKPs.begin() ; it != newKPs.end(); ++it) {
    std::vector< float > newPoint = *it;
    cv::KeyPoint newKP(newPoint[0], newPoint[1], 0.1);
    keypoints.push_back(newKP);
  }
}

void tps::Surf::addNewMatches(int numbersOfNewKPs) {
  for (int i = 1; i < numbersOfNewKPs+1; i++) {
    cv::DMatch match(keypoints_ref.size()-i, keypoints_tar.size()-i, -1);
    good_matches.push_back(match);
  }
}

void tps::Surf::run() {
	detectFeatures();
	extractDescriptors();
	matchDescriptors();
}