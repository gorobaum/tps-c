#ifndef TPS_TPS_H_
#define TPS_TPS_H_

#include "image/image.h"
#include "linearsystem/cplinearsystems.h"

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace tps {

class TPS {
public:
  TPS(std::vector< std::vector<float> > referenceKeypoints, std::vector< std::vector<float> > targetKeypoints, tps::Image targetImage, std::string outputName) :
    referenceKeypoints_(referenceKeypoints),
    targetKeypoints_(targetKeypoints),
    outputName_(outputName),
    targetImage_(targetImage),
    width(targetImage.getWidth()),
    height(targetImage.getHeight()),
    slices(targetImage.getSlices()),
    registredImage(targetImage.getWidth(), targetImage.getHeight(), targetImage.getSlices()) {};
  virtual void run() = 0;
protected:
  std::vector< std::vector<float> > referenceKeypoints_;
  std::vector< std::vector<float> > targetKeypoints_;
  std::string outputName_;
  tps::Image targetImage_;
  int width;
  int height;
  int slices;
  tps::Image registredImage;
  float computeRSquared(float x, float xi, float y, float yi) {return pow(x-xi,2) + pow(y-yi,2);};
};

} // namespace

#endif