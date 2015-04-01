#include "image.h"
#include "surf.h"
#include "featuregenerator.h"
#include "basictps.h"
#include "paralleltps.h"
#include "cudatps.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>

void readConfigFile(std::string filename, tps::Image& referenceImage, tps::Image& targetImage, std::string& outputName, float& percentage) {
  std::ifstream infile;
  infile.open(filename.c_str());
  std::string line;
  std::getline(infile, line);
  referenceImage = tps::Image(line);
  std::getline(infile, line);
  targetImage = tps::Image(line);
  std::getline(infile, outputName);
  std::getline(infile, line);
  percentage = std::stof(line);
}

int main(int argc, char** argv) {
	if (argc < 1) {
		std::cout << "Precisa passar o arquivo de configuração coração! \n";    
		return 0;
	}

 	tps::Image referenceImage;
  tps::Image targetImage;
  std::string outputName;
  float percentage;
  readConfigFile(argv[1], referenceImage, targetImage, outputName, percentage);
  int minHessian = 400;

  tps::FeatureGenerator fg = tps::FeatureGenerator(referenceImage, targetImage, percentage);
  fg.run(true);

  double basicTpsExecTime = (double)cv::getTickCount();
  tps::BasicTPS tps = tps::BasicTPS(fg.getReferenceKeypoints(), fg.getTargetKeypoints(), targetImage, "regBasic.png");
  tps.run();
  basicTpsExecTime = ((double)cv::getTickCount() - basicTpsExecTime)/cv::getTickFrequency();
  std::cout << "Basic TPS execution time: " << basicTpsExecTime << std::endl;

  double pTpsExecTime = (double)cv::getTickCount();
  tps::ParallelTPS ptps = tps::ParallelTPS(fg.getReferenceKeypoints(), fg.getTargetKeypoints(), targetImage, "regParallel.png");
  ptps.run();
  pTpsExecTime = ((double)cv::getTickCount() - pTpsExecTime)/cv::getTickFrequency();
  std::cout << "Parallel TPS execution time: " << pTpsExecTime << std::endl;

  double cTpsExecTime = (double)cv::getTickCount();
  tps::CudaTPS ctps = tps::CudaTPS(fg.getReferenceKeypoints(), fg.getTargetKeypoints(), targetImage, "regCuda.png");
  ctps.run();
  cTpsExecTime = ((double)cv::getTickCount() - cTpsExecTime)/cv::getTickFrequency();
  std::cout << "Cuda TPS execution time: " << cTpsExecTime << std::endl;

  return 0;
}