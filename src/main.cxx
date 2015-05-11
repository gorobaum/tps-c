#include "image.h"
#include "featuregenerator.h"
#include "tps.h"
#include "cudatps.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>

cv::Mat cvRefImg;
cv::Mat cvTarImg;

void readConfigFile(std::string filename, tps::Image& referenceImage, tps::Image& targetImage, std::string& outputName, float& percentage, std::string& extension) {
  std::ifstream infile;
  infile.open(filename.c_str());
  std::string line;
  std::getline(infile, line);
  cvRefImg = cv::imread(line, CV_LOAD_IMAGE_GRAYSCALE);
  referenceImage = tps::Image(cvRefImg);
  std::size_t pos = line.find('.');
  extension = line.substr(pos);
  std::getline(infile, line);
  cvTarImg = cv::imread(line, CV_LOAD_IMAGE_GRAYSCALE);
  targetImage = tps::Image(cvTarImg);
  std::getline(infile, outputName);
  std::getline(infile, line);
  percentage = std::stof(line);
}

int main(int argc, char** argv) {
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);

  if (argc < 1) {
    std::cout << "Precisa passar o arquivo de configuração coração! \n";    
    return 0;
  }
 	tps::Image referenceImage;
  tps::Image targetImage;
  std::string outputName;
  std::string extension;
  float percentage;
  readConfigFile(argv[1], referenceImage, targetImage, outputName, percentage, extension);
  int minHessian = 400;

  // double fgExecTime = (double)cv::getTickCount();
  // tps::FeatureGenerator fg = tps::FeatureGenerator(referenceImage, targetImage, percentage, cvRefImg, cvTarImg);
  // fg.run(true);
  // fgExecTime = ((double)cv::getTickCount() - fgExecTime)/cv::getTickFrequency();
  // std::cout << "FeatureGenerator execution time: " << fgExecTime << std::endl;

  std::vector<cv::Point2f> referenceKeypoints;
  std::vector<cv::Point2f> targetKeypoints;
  int x[7] = {1,1,20,20,11,7,16};
  int y[7] = {1,20,1,20,8,17,12};
  int X[7] = {10,10,10,10,15,25,20};
  int Y[7] = {0,0,0,0,0,0,0};
  for ( int i = 0; i < 7; i++) {
    cv::Point2f newCP(x[i], y[i]);
    referenceKeypoints.push_back(newCP);
    cv::Point2f newCPT(X[i], Y[i]);
    targetKeypoints.push_back(newCPT);
  }
  

  double CUDAcTpsExecTime = (double)cv::getTickCount();
  tps::CudaTPS CUDActps = tps::CudaTPS(referenceKeypoints, targetKeypoints, targetImage, outputName+"TPSCUDA"+extension);
  CUDActps.run();
  CUDAcTpsExecTime = ((double)cv::getTickCount() - CUDAcTpsExecTime)/cv::getTickFrequency();
  std::cout << "CUDA Cuda TPS execution time: " << CUDAcTpsExecTime << std::endl;
  return 0;
}