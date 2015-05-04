#ifndef TPS_FEATURE_FACTORY_H_
#define TPS_FEATURE_FACTORY_H_

#include "featuredetector.h"

namespace tps {
	
class FeatureGenerator : public FeatureDetector {
public:
  FeatureGenerator(Image referenceImage, Image targetImage, float percentage, cv::Mat refImg, cv::Mat tarImg):
    FeatureDetector(referenceImage, targetImage),
    percentage_(percentage),
    refImg_(refImg),
    tarImg_(tarImg) {};
  void run(bool createFeatureImage);
  void saveFeatureImage();
private:
  float percentage_;
  cv::Mat refImg_;
  cv::Mat tarImg_;
  int gridSizeCol, gridSizeRow;
  float colStep, rowStep;
  std::vector<float> applySenoidalDeformationTo(float x, float y);
  void createReferenceImageFeatures();
  void createTargetImageFeatures();
  std::vector<cv::KeyPoint> keypoints_ref, keypoints_tar;
};

} //namespace

#endif