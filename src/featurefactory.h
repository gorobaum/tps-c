#ifndef TPS_FEATURE_FACTORY_H_
#define TPS_FEATURE_FACTORY_H_

namespace tps {
	
class FeatureFactory : public FeatureDetector {
public:
  FeatureFactory(Image referenceImage, Image targetImage, float percentage):
    FeatureDetector(referenceImage, targetImage),
    percentage_(percentage) {};
  void run(bool createFeatureImage);
  void saveFeatureImage();
private:
  void createReferenceImageFeatures();
  void createTargetImageFeatures();
  float percentage_;
};

} //namespace

#endif