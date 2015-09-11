#ifndef TPS_TPSINSTANCE_H_
#define TPS_TPSINSTANCE_H_

#include <iostream>
#include <vector>
#include <string>

#include "image/image.h"
#include "image/imagehandler.h"
#include "cudamemory.h"

namespace tps {
  
class TpsInstance {
public:
  TpsInstance(std::string configurationFile, tps::Image referenceImage, tps::ImageHandler *imageHandler) :
    configurationFile_(configurationFile),
    referenceImage_(referenceImage),
    imageHandler_(imageHandler),
    twoDimension(false) {
      readConfigurationFile();
      createKeyPoints();
    };
  void runCudaTPS();
  void runParallelTPS();
  void runBasicTPS();
  bool isTwoDimension() {return twoDimension;};
  size_t allocCudaMemory(size_t usedMemory);
  std::string generateOutputName(std::string differentiator);
private:
  std::string configurationFile_;
  tps::Image referenceImage_;
  tps::ImageHandler *imageHandler_;
  bool twoDimension;
  tps::Image targetImage;
  tps::CudaMemory cm;
  std::vector< std::vector<float> > referenceKPs;
  std::vector< std::vector<float> > targetKPs;
  std::string outputName;
  std::string extension;
  float percentage;
  void createKeyPoints();
  void readConfigurationFile();
  void readKeypoints(std::ifstream& infile, std::vector< std::vector<float> >& kps);
};

} // namespace

#endif