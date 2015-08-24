#ifndef TPS_TPS_H_
#define TPS_TPS_H_

#include "image/image.h"
#include "linearsystem/cplinearsystems.h"
#include "image/itkimagehandler.h"

#include <vector>

namespace tps {

class TPS {
public:
  TPS(std::vector< std::vector<float> > referenceKeypoints, std::vector< std::vector<float> > targetKeypoints, tps::Image targetImage, std::string outputName) :
    referenceKeypoints_(referenceKeypoints),
    targetKeypoints_(targetKeypoints),
    outputName_(outputName),
    targetImage_(targetImage),
    dimensions_(targetImage.getDimensions()),
    registredImage(targetImage.getDimensions()) {};
  virtual void run() = 0;
protected:
  std::vector< std::vector<float> > referenceKeypoints_;
  std::vector< std::vector<float> > targetKeypoints_;
  std::string outputName_;
  tps::Image targetImage_;
  std::vector<int> dimensions_;
  tps::Image registredImage;
  float computeRSquared(float x, float xi, float y, float yi, float z, float zi) {return pow(x-xi,2) + pow(y-yi,2) + pow(z-zi,2);};
};

} // namespace

#endif