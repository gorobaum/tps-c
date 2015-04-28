#ifndef TPS_FEATURE_FACTORY_H_
#define TPS_FEATURE_FACTORY_H_

#include "featuredetector.h"

namespace tps {
	
class FeatureGenerator : public FeatureDetector {
public:
  FeatureGenerator(Image& referenceImage, Image& targetImage, float percentage):
    FeatureDetector(referenceImage, targetImage),
    percentage_(percentage) {};
  void run(bool createFeatureImage);
  void saveFeatureImage();
private:
  std::vector<float> applySenoidalDeformationTo(float x, float y);
  std::vector<cv::DMatch> createMatches();
  void createReferenceImageFeatures();
  void createTargetImageFeatures();
  std::vector<cv::DMatch> matches;
  std::vector<cv::KeyPoint> keypoints_ref, keypoints_tar;
  float percentage_;
  int gridSizeX, gridSizeY;
  float xStep, yStep;
};

} //namespace

#endif