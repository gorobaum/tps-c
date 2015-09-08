#include "tpsinstance.h"

#include "feature/featuregenerator.h"

#include "tps/cudatps.h"
#include "tps/paralleltps.h"
#include "tps/basictps.h"

void tps::TpsInstance::readConfigurationFile() {
  std::ifstream infile;
  infile.open(configurationFile_.c_str());
  std::string line;
  
  std::getline(infile, line);
  targetImage = imageHandler_->loadImageData(line);

  std::size_t pos = line.find('.');
  extension = line.substr(pos);
  if (extension.compare(".nii.gz") != 0 )
    twoDimension = true;

  std::getline(infile, outputName);

  std::getline(infile, line);
  percentage = std::stof(line);
}

void tps::TpsInstance::createKeyPoints() {
  tps::FeatureGenerator fg = tps::FeatureGenerator(referenceImage_, targetImage, percentage);
  fg.run();

  referenceKPs = fg.getReferenceKeypoints();
  targetKPs = fg.getTargetKeypoints();
}

std::string tps::TpsInstance::generateOutputName(std::string differentiator) {
  return outputName+differentiator+extension;
}

void tps::TpsInstance::runCudaTPS() {
  std::string filename = generateOutputName("Cuda");
  std::cout << "filename = " << filename << std::endl; 
  tps::CudaTPS CUDActps = tps::CudaTPS(referenceKPs, targetKPs, targetImage, cm, twoDimension);
  tps::Image resultImage = CUDActps.run();
  imageHandler_->saveImageData(resultImage, filename);
  cm.freeMemory();
}

void tps::TpsInstance::runBasicTPS() {
  std::string filename = generateOutputName("Basic");
  tps::BasicTPS basic = tps::BasicTPS(referenceKPs, targetKPs, targetImage, twoDimension);
  tps::Image resultImage = basic.run();
  imageHandler_->saveImageData(resultImage, filename);
}

void tps::TpsInstance::runParallelTPS() {
  std::string filename = generateOutputName("Parallel");
  tps::ParallelTPS parallelTPS = tps::ParallelTPS(referenceKPs, targetKPs, targetImage, twoDimension);
  tps::Image resultImage = parallelTPS.run();
  imageHandler_->saveImageData(resultImage, filename);
}

size_t tps::TpsInstance::allocCudaMemory(size_t usedMemory) {
  cm = tps::CudaMemory(targetImage.getDimensions(), referenceKPs);
  if (usedMemory+cm.memoryEstimation() > 1800) return usedMemory;
  cm.allocCudaMemory(targetImage);
  size_t avail;
  size_t total;
  cudaMemGetInfo( &avail, &total );
  size_t used = (total - avail)/(1024*1024);
  return used;
}


