#ifndef TPS_FEATURE_DETECTOR_H_
#define TPS_FEATURE_DETECTOR_H_

#include "image.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

namespace tps {

// code from http://docs.opencv.org/doc/tutorials/features2d/feature_homography/feature_homography.html
class FeatureDetector
{
public:
	FeatureDetector(Image referenceImage, Image targetImage):
		referenceImage_(referenceImage),
		targetImage_(targetImage) {};
  virtual void run(bool createFeatureImage) = 0;
  virtual void saveFeatureImage() = 0;
  std::vector< std::vector<float> > getReferenceKeypoints() {return referenceKeypoints;};
  std::vector< std::vector<float> > getTargetKeypoints() {return targetKeypoints;};
protected:
	Image referenceImage_;
	Image targetImage_;
	// Data structures
	std::vector< std::vector<float> > referenceKeypoints;
  std::vector< std::vector<float> > targetKeypoints;
};

} // namespace

#endif