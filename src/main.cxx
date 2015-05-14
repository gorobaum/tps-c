#include "image.h"
#include "featuregenerator.h"
#include "tps.h"
#include "cudatps.h"
#include "cudamemory.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>

void memoryEstimation(int width, int height, int numberOfCps) {
  int floatSize = sizeof(float);
  int doubleSize = sizeof(double);
  int ucharSize = sizeof(uchar);

  int systemDimention = numberOfCps+3;

  size_t avail;
  size_t total;
  cudaMemGetInfo( &avail, &total );
  size_t used = total - avail;
  std::cout << "Device memory used: " << used/(1024*1024) << "MB" << std::endl;

  double solutionsMemory = 2.0*systemDimention*floatSize/(1024*1024);
  std::cout << "GPU Memory occupied by the linear systems solutions = " << solutionsMemory << "MB" << std::endl;

  double coordinatesMemory = 2.0*width*height*doubleSize/(1024*1024);
  std::cout << "GPU Memory occupied by the coordinates calculation = " << coordinatesMemory << "MB" << std::endl;

  double keypointsMemory = 2.0*numberOfCps*floatSize/(1024*1024);
  std::cout << "GPU Memory occupied by the keypoints = " << keypointsMemory << "MB" << std::endl;

  double pixelsMemory = 2.0*width*height*ucharSize/(1024*1024);
  std::cout << "GPU Memory occupied by the pixels = " << pixelsMemory << "MB" << std::endl;

  double totalMemory = solutionsMemory+coordinatesMemory+keypointsMemory+pixelsMemory;
  std::cout << "Total GPU memory occupied = " << totalMemory << "MB" << std::endl;
}

void readConfigFile(std::string filename, std::vector< tps::Image >& targetImages,
                    std::vector< cv::Mat >& cvTarImgs, std::vector< std::string >& outputNames, 
                    std::vector< float >& percentages) {
  std::ifstream infile;
  infile.open(filename.c_str());
  std::string line;
  
  std::getline(infile, line);
  cv::Mat cvTarImg = cv::imread(line, CV_LOAD_IMAGE_GRAYSCALE);
  tps::Image targetImage = tps::Image(cvTarImg);
  cvTarImgs.push_back(cvTarImg);
  targetImages.push_back(targetImage);

  std::string outputName;
  std::getline(infile, outputName);
  outputNames.push_back(outputName);

  std::getline(infile, line);
  float percentage = std::stof(line);
  percentages.push_back(percentage);
}

int main(int argc, char** argv) {
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);

  if (argc < 1) {
    std::cout << "Precisa passar o arquivo de configuração coração! \n";    
    return 0;
  }

  // int minHessian = 400;
  std::ifstream infile;
  infile.open(argv[1]);

  std::string line; 
  std::getline(infile, line);

  std::size_t pos = line.find('.');
  std::string extension = line.substr(pos);

  cv::Mat cvRefImg = cv::imread(line, CV_LOAD_IMAGE_GRAYSCALE);
  tps::Image referenceImage = tps::Image(cvRefImg);
  std::vector< cv::Mat > cvTarImgs;
  std::vector< tps::Image > targetImages;
  std::vector< std::string > outputNames;
  std::vector< float > percentages;

  int count = 0;
  for (line; std::getline(infile, line); infile.eof(), count++) {
    readConfigFile(line, targetImages, cvTarImgs, outputNames, percentages);
  }

  std::vector< std::vector< cv::Point2f > > referencesKPs;
  std::vector< std::vector< cv::Point2f > > targetsKPs;

  for (int i = 0; i < count; i++) {
    double fgExecTime = (double)cv::getTickCount();
    tps::FeatureGenerator fg = tps::FeatureGenerator(referenceImage, targetImages[i], percentages[i], cvRefImg, cvTarImgs[i]);
    fg.run(true);
    fgExecTime = ((double)cv::getTickCount() - fgExecTime)/cv::getTickFrequency();
    referencesKPs.push_back(fg.getReferenceKeypoints());
    targetsKPs.push_back(fg.getTargetKeypoints());
    std::cout << "FeatureGenerator execution time: " << fgExecTime << std::endl;
  }

  for (int i = 0; i < count; i++) {
    memoryEstimation(targetImages[i].getWidth(), targetImages[i].getHeight(), referencesKPs[i].size());

    tps::CudaMemory cm = tps::CudaMemory(targetImages[i].getWidth(), targetImages[i].getHeight(), referencesKPs[i]);

    double CUDAcTpsExecTime = (double)cv::getTickCount();
    tps::CudaTPS CUDActps = tps::CudaTPS(referencesKPs[i], targetsKPs[i], targetImages[i], outputNames[i]+"TPSCUDA"+extension, cm);
    CUDActps.run();
    CUDAcTpsExecTime = ((double)cv::getTickCount() - CUDAcTpsExecTime)/cv::getTickFrequency();
    std::cout << "CUDA Cuda TPS execution time: " << CUDAcTpsExecTime << std::endl;
  }
  return 0;
}