#ifndef TPS_FEATURE_FACTORY_H_
#define TPS_FEATURE_FACTORY_H_

#include "featuredetector.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

namespace tps {
	
class FeatureGenerator : public FeatureDetector {
public:
  FeatureGenerator(Image referenceImage, Image targetImage, float percentage):
    FeatureDetector(referenceImage, targetImage),
    percentage_(percentage) {};
  void run();
  void drawKeypointsImage(cv::Mat tarImg, std::string filename);
  void drawFeatureImage(cv::Mat refImg, cv::Mat tarImg, std::string filename);
private:
  float percentage_;
  int gridSizeCol, gridSizeRow, gridSizeSlice;
  float colStep, rowStep, sliceStep;
  std::vector<float> applySenoidalDeformationTo(float x, float y, float z);
  void createReferenceImageFeatures();
  void createTargetImageFeatures();
  std::vector<cv::KeyPoint> keypoints_ref, keypoints_tar;
};

} //namespace

#endif