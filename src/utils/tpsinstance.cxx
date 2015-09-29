#include <sstream>
#include <armadillo>

#include "tpsinstance.h"

#include "feature/featuregenerator.h"

#include "tps/cudatps.h"
#include "tps/paralleltps.h"
#include "tps/basictps.h"

void tps::TpsInstance::readKeypoints(std::ifstream& infile, std::vector< std::vector<float> >& kps) {
  std::string line;
  while(std::getline(infile, line)) {
    if (line.compare("endKeypoints") != 0) {
      std::stringstream stream(line);
      float point;
      std::vector<float> newKP;
      for (int i = 0; i < 2; i++) {
        stream >> point;
        newKP.push_back(point);
      }
      kps.push_back(newKP);
    } 
    else break;
  }
}

void tps::TpsInstance::readBoundaries(std::ifstream& infile) {
  std::string line;
  int count = 0;
  while(std::getline(infile, line)) {
    if (line.compare("endBoundaries") != 0) {
      std::stringstream stream(line);
      float point;
      std::vector<int> newBoundary;
      for (int i = 0; i < 2; i++) {
        stream >> point;
        newBoundary.push_back(point);
      }
      count++;
      boundaries.push_back(newBoundary);
    } 
    else break;
  }
  while (count < 3) {
    std::vector<int> newBoundary;
    newBoundary[0] = 0;
    newBoundary[1] = targetImage.getDimensions()[count];
    boundaries.push_back(newBoundary);
    count++;
  }
  std::cout << "boundaries.size() = " << boundaries.size() << std::endl;
}

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

  while (std::getline(infile, line)) {
    if (line.compare("referenceKeypoints:") == 0)
      readKeypoints(infile, referenceKPs);
    else if (line.compare("targetKeypoints:") == 0)
      readKeypoints(infile, targetKPs);
    else if (line.compare("boundaries:") == 0)
      readBoundaries(infile);
    else break;
  }
  std::cout << "boundaries.size() = " << boundaries.size() << std::endl;
}

void tps::TpsInstance::addNewKeypoints(std::vector< std::vector<float> >& keyPoints, std::vector< std::vector<float> > newKeyPoints) {
  std::vector< std::vector<float> >::iterator it = keyPoints.begin();
  keyPoints.insert(it, newKeyPoints.begin(), newKeyPoints.end());
}

void tps::TpsInstance::createKeyPoints() {
  tps::FeatureGenerator fg = tps::FeatureGenerator(referenceImage_, targetImage, percentage, boundaries);
  fg.run();

  addNewKeypoints(referenceKPs, fg.getReferenceKeypoints());
  addNewKeypoints(targetKPs, fg.getTargetKeypoints());
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

size_t tps::TpsInstance::getAllocatedGPUMemory() {
  size_t avail;
  size_t total;
  cudaMemGetInfo( &avail, &total );
  size_t usedMemory = (total - avail)/(1024*1024);
  return usedMemory;
}

bool tps::TpsInstance::canAllocGPUMemory() {
  int usedMemory = getAllocatedGPUMemory();

  int floatSize = sizeof(float);
  int doubleSize = sizeof(double);
  int ucharSize = sizeof(short);
  bool ret = false;

  int numberOfCps = referenceKPs.size();
  int systemDim = numberOfCps + 4;
  int imageSize = targetImage.getNumberofPixels();

  double solutionsMemory = 3.0*systemDim*floatSize/(1024*1024);
  // std::cout << "solutionsMemory = " << solutionsMemory << std::endl;
  double keypointsMemory = 3.0*numberOfCps*floatSize/(1024*1024);
  // std::cout << "keypointsMemory = " << keypointsMemory << std::endl;
  double pixelsMemory = 2.0*imageSize*ucharSize/(1024*1024);
  // std::cout << "pixelsMemory = " << pixelsMemory << std::endl;

  double totalMemory = solutionsMemory+keypointsMemory+pixelsMemory;

  if (usedMemory+totalMemory <= 1800) ret = true;

  return ret;
}

void tps::TpsInstance::allocCudaMemory() {
  int usedMemory = getAllocatedGPUMemory();
  cm = tps::CudaMemory(targetImage.getDimensions(), referenceKPs);
  cm.allocCudaMemory(targetImage);
}