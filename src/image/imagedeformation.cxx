#include <cmath>

#include "imagedeformation.h"
#include "itkimagehandler.h"

void tps::ImageDeformation::apply3DSinDeformation() {
  std::vector<int> dimensions = image_.getDimensions();
  for(int z = 0; z < dimensions[2]; z++)
    for(int y = 0; y < dimensions[1]; y++)
      for(int x = 0; x < dimensions[0]; x++) {
        std::vector<float> newPoint = newPointSinDef(x, y, z);
        short newVoxel = image_.trilinearInterpolation(newPoint[0], newPoint[1], newPoint[2]);
        result.changePixelAt(x, y, z, newVoxel);
      }
  tps::ITKImageHandler::saveImageData(result, outputName_);
}

std::vector<float> tps::ImageDeformation::newPointSinDef(int x, int y, int z) {
  std::vector<float> newPoint;
  float newX = x - 8*std::sin(y/16);
  float newY = y + 4*std::cos(x/32);
  float newZ = z;
  newPoint.push_back(newX);
  newPoint.push_back(newY);
  newPoint.push_back(newZ);
  return newPoint;
}