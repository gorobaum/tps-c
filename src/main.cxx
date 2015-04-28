#include "image.h"
#include "surf.h"
#include "featuregenerator.h"
#include "basictps.h"
#include "paralleltps.h"
#include "cudatps.h"
#include "cudalinearsystems.h"
#include "OPCVlinearsystems.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>

tps::Image* referenceImage = nullptr;
tps::Image* targetImage = nullptr;

void readConfigFile(std::string filename, std::string& outputName, float& percentage, std::string& extension) {
  std::ifstream infile;
  infile.open(filename.c_str());
  std::string line;
  std::getline(infile, line);
  referenceImage = new tps::Image(line);
  std::size_t pos = line.find('.');
  extension = line.substr(pos);
  std::getline(infile, line);
  targetImage = new tps::Image(line);
  std::getline(infile, outputName);
  std::getline(infile, line);
  percentage = std::stof(line);
}

int main(int argc, char** argv) {
  if (argc < 1) {
    std::cout << "Precisa passar o arquivo de configuração coração! \n";    
    return 0;
  }
  std::cout << "====================================================\n";
  std::string outputName;
  std::string extension;
  float percentage;
  readConfigFile(argv[1], outputName, percentage, extension);
  int minHessian = 400;

  double fgExecTime = (double)cv::getTickCount();
  tps::FeatureGenerator fg = tps::FeatureGenerator(*referenceImage, *targetImage, percentage);
  fg.run(true);
  fgExecTime = ((double)cv::getTickCount() - fgExecTime)/cv::getTickFrequency();
  std::cout << "FeatureGenerator execution time: " << fgExecTime << std::endl;

  std::cout << "Percentage: " << percentage << std::endl;
  std::cout << "Number of control points: " << fg.getReferenceKeypoints().size() << std::endl;

  // double basicTpsExecTime = (double)cv::getTickCount();
  // tps::BasicTPS tps = tps::BasicTPS(fg.getReferenceKeypoints(), fg.getTargetKeypoints(), *targetImage, outputName+"BasicTPS"+extension);
  // tps.run();
  // basicTpsExecTime = ((double)cv::getTickCount() - basicTpsExecTime)/cv::getTickFrequency();
  // std::cout << "Basic TPS execution time: " << basicTpsExecTime << std::endl;

  // double pTpsExecTime = (double)cv::getTickCount();
  // tps::ParallelTPS ptps = tps::ParallelTPS(fg.getReferenceKeypoints(), fg.getTargetKeypoints(), *targetImage, outputName+"ParallelTPS"+extension);
  // ptps.run();
  // pTpsExecTime = ((double)cv::getTickCount() - pTpsExecTime)/cv::getTickFrequency();
  // std::cout << "Parallel TPS execution time: " << pTpsExecTime << std::endl;

  double CUDAcTpsExecTime = (double)cv::getTickCount();
  tps::CudaTPS* CUDAtps = new tps::CudaTPS(fg.getReferenceKeypoints(), fg.getTargetKeypoints(), *targetImage, outputName+"CudaTPS"+extension);
  CUDAtps->run();
  CUDAcTpsExecTime = ((double)cv::getTickCount() - CUDAcTpsExecTime)/cv::getTickFrequency();
  std::cout << "Cuda TPS execution time: " << CUDAcTpsExecTime << std::endl;
  std::cout << "====================================================\n";

  delete referenceImage;
  delete targetImage;
  delete CUDAtps;
  return 0;
}