#ifndef TPS_FEATURE_FACTORY_H_
#define TPS_FEATURE_FACTORY_H_

#include "featuredetector.h"

namespace tps {
	
class FeatureGenerator : public FeatureDetector {
public:
  FeatureGenerator(Image referenceImage, Image targetImage, float percentage):
    FeatureDetector(referenceImage, targetImage),
    percentage_(percentage) {};
  void run();
private:
  float percentage_;
  int gridSizeX, gridSizeY, gridSizeZ;
  float xStep, yStep, zStep;
  std::vector<float> applyXRotationalDeformationTo(float x, float y, float z, float ang);
  std::vector<float> applySinDeformationTo(float x, float y, float z);
  void createReferenceImageFeatures();
  void createTargetImageFeatures();
};

} //namespace

#endif