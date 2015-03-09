#ifndef TPS_SURF_H_
#define TPS_SURF_H_

#include "image.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

namespace tps {

class Surf
{
public:
	Surf(Image referenceImage, Image targetImage, int minHessian):
		referenceImage_(referenceImage),
		targetImage_(targetImage),
		detector(minHessian) {}
	void run(bool createFeatureImage);
	void saveFeatureImage();
private:
	Image referenceImage_;
	Image targetImage_;
	// Surf aux functions
	cv::SurfFeatureDetector detector;
	cv::SurfDescriptorExtractor extractor;
	cv::FlannBasedMatcher matcher;
	// Data structures
	cv::Mat descriptors_object, descriptors_scene;
	std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;
	std::vector<cv::DMatch> good_matches;

	void detectFeatures();
	void extractDescriptors();
	void matchDescriptors();
};

} // namespace

#endif