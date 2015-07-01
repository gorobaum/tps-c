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
  std::vector<float> newPointSinDef(int x, int y, int z);
  tps::Image image_;
  tps::Image result;
  std::string outputName_;
};

}

#endif