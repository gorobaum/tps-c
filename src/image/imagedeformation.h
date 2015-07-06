#ifndef TPS_IMAGEDEFORMATION_H_
#define TPS_IMAGEDEFORMATION_H_

#include <vector>
#include <string>

#include "image.h"

namespace tps {
  
class ImageDeformation {
public:
  ImageDeformation(tps::Image image, std::string outputName) :
    image_(image),
    result(image.getDimensions()),
    outputName_(outputName) {};
  void apply3DSinDeformation();
  tps::Image getResult() { return result; };
private:
  std::vector<int> newPointSinDef(int x, int y, int z);
  tps::Image image_;
  tps::Image result;
  std::string outputName_;
  int getNearestInteger(float number) {
    if ((number - std::floor(number)) <= 0.5) return std::floor(number);
    return std::floor(number) + 1.0;
  }
};

}

#endif