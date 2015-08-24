#ifndef TPS_TPSINSTANCE_H_
#define TPS_TPSINSTANCE_H_

#include <iostream>

#include "image/image.h"
#include "cudamemory.h"

namespace tps {
  
class TpsInstance {
public:
  TpsInstance(std::string configurationFile, tps::Image referenceImage) :
    configurationFile_(configurationFile),
    referenceImage_(referenceImage) {
      readConfigurationFile();
      createKeyPoints();
    };
  void runCudaTPS();
  void runParallelTPS();
  size_t allocCudaMemory(size_t usedMemory);
  std::string generateOutputName(std::string differentiator);
private:
  std::string configurationFile_;
  tps::Image referenceImage_;
  tps::Image targetImage;
  tps::CudaMemory cm;
  std::vector< std::vector<float> > referenceKPs;
  std::vector< std::vector<float> > targetKPs;
  std::string outputName;
  std::string extension;
  float percentage;
  void createKeyPoints();
  void readConfigurationFile();
};

} // namespace

#endif