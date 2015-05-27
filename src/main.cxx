#include "image.h"
#include "featuregenerator.h"
#include "tps.h"
#include "cudatps.h"
#include "basictps.h"
#include "paralleltps.h"
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
  std::vector< std::vector<int> > vecImage(cvTarImg.size().width, std::vector<int>(cvTarImg.size().height, 0));
    for (int col = 0; col < cvTarImg.size().width; col++)
      for (int row = 0; row < cvTarImg.size().height; row++)
        vecImage[col][row] = cvTarImg.at<uchar>(row, col);
  tps::Image targetImage = tps::Image(vecImage, cvTarImg.size().width, cvTarImg.size().height);
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
  std::vector< std::vector<int> > vecImage(cvRefImg.size().width, std::vector<int>(cvRefImg.size().height, 0));
    for (int col = 0; col < cvRefImg.size().width; col++)
      for (int row = 0; row < cvRefImg.size().height; row++)
        vecImage[col][row] = cvRefImg.at<uchar>(row, col);
  tps::Image referenceImage = tps::Image(vecImage, cvRefImg.size().width, cvRefImg.size().height);
  std::vector< cv::Mat > cvTarImgs;
  std::vector< tps::Image > targetImages;
  std::vector< std::string > outputNames;
  std::vector< float > percentages;

  int count = 0;
  for (line; std::getline(infile, line); infile.eof(), count++) {
    readConfigFile(line, targetImages, cvTarImgs, outputNames, percentages);
  }

  std::vector< std::vector< std::vector<float> > > referencesKPs;
  std::vector< std::vector< std::vector<float> > > targetsKPs;

  for (int i = 0; i < count; i++) {
    double fgExecTime = (double)cv::getTickCount();
    tps::FeatureGenerator fg = tps::FeatureGenerator(referenceImage, targetImages[i], percentages[i], cvRefImg, cvTarImgs[i]);
    fg.run(true);
    fgExecTime = ((double)cv::getTickCount() - fgExecTime)/cv::getTickFrequency();
    referencesKPs.push_back(fg.getReferenceKeypoints());
    targetsKPs.push_back(fg.getTargetKeypoints());
    std::cout << "FeatureGenerator execution time: " << fgExecTime << std::endl;
  }

  size_t avail;
  size_t total;
  cudaMemGetInfo( &avail, &total );
  size_t used = (total - avail)/(1024*1024);
  std::cout << "Device memory used: " << used/(1024*1024) << "MB" << std::endl;

  std::vector< tps::CudaMemory > cudaMemories;
  for (int i = 0; i < count; i++) {
    int lastI = i;
    while(used <= 1800) {
      std::cout << "============================================" << std::endl;
      std::cout << "Entry number " << i << " will run." << std::endl;
      tps::CudaMemory cm = tps::CudaMemory(targetImages[i].getWidth(), targetImages[i].getHeight(), referencesKPs[i]);
      if (used+cm.memoryEstimation() > 1800) break;
      cm.allocCudaMemory(targetImages[i]);
      cudaMemories.push_back(cm);
      cudaMemGetInfo( &avail, &total );
      used = (total - avail)/(1024*1024);
      std::cout << "Device used memory = " << used << "MB" << std::endl;
      std::cout << "============================================" << std::endl;
      i++;
      if (i >= count) break;
    }

    for (int j = lastI; j < i; j++) {
      std::cout << "============================================" << std::endl;
      std::cout << "#Execution = " << j << std::endl;
      std::cout << "#Keypoints = " << referencesKPs[j].size() << std::endl;
      std::cout << "#Percentage = " << percentages[j] << std::endl;

      // double BasicTPSExecTime = (double)cv::getTickCount();
      // tps::BasicTPS BasicTPS = tps::BasicTPS(referencesKPs[j], targetsKPs[j], targetImages[j], outputNames[j]+"Basic"+extension);
      // BasicTPS.run();
      // BasicTPSExecTime = ((double)cv::getTickCount() - BasicTPSExecTime)/cv::getTickFrequency();
      // std::cout << "Basic TPS execution time: " << BasicTPSExecTime << std::endl;
      tps::CudaMemory parallelCM = tps::CudaMemory(targetImages[j].getWidth(), targetImages[j].getHeight(), referencesKPs[j]);
      parallelCM.allocCudaMemory(targetImages[j]);
      double ParallelTpsExecTime = (double)cv::getTickCount();
      tps::ParallelTPS parallelTPS = tps::ParallelTPS(referencesKPs[j], targetsKPs[j], targetImages[j], outputNames[j]+"Parallel"+extension, parallelCM);
      parallelTPS.run();
      ParallelTpsExecTime = ((double)cv::getTickCount() - ParallelTpsExecTime)/cv::getTickFrequency();
      std::cout << "Parallel TPS execution time: " << ParallelTpsExecTime << std::endl;

      double CUDAcTpsExecTime = (double)cv::getTickCount();
      tps::CudaTPS CUDActps = tps::CudaTPS(referencesKPs[j], targetsKPs[j], targetImages[j], outputNames[j]+"Cuda"+extension, cudaMemories[j]);
      CUDActps.run();
      CUDAcTpsExecTime = ((double)cv::getTickCount() - CUDAcTpsExecTime)/cv::getTickFrequency();
      std::cout << "Cuda TPS execution time: " << CUDAcTpsExecTime << std::endl;
      cudaMemories[j].freeMemory();
      std::cout << "============================================" << std::endl;
    }
  }
  return 0;
}