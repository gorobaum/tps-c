#include "image/image.h"
#include "image/itkimagehandler.h"
#include "image/imagedeformation.h"
#include "feature/featuregenerator.h"
#include "utils/cudamemory.h"
#include "tps/tps.h"
#include "tps/cudatps.h"
#include "tps/basictps.h"
#include "tps/paralleltps.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_occupancy.h"

bool createKeypointImages = true;

void readConfigFile(std::string filename, std::vector< tps::Image >& targetImages, 
                    std::vector< std::string >& outputNames, 
                    std::vector< float >& percentages) {
  std::ifstream infile;
  infile.open(filename.c_str());
  std::string line;
  
  std::getline(infile, line);
  tps::Image targetImage = tps::ITKImageHandler::loadImageData(line);
  targetImages.push_back(targetImage);

  std::string outputName;
  std::getline(infile, outputName);
  outputNames.push_back(outputName);

  std::getline(infile, line);
  float percentage = std::stof(line);
  percentages.push_back(percentage);
}

void runFeatureGeneration(tps::Image referenceImage, tps::Image targetImage, float percentage,
                          std::vector< std::vector< std::vector<float> > >& referencesKPs, 
                          std::vector< std::vector< std::vector<float> > >& targetsKPs) {
    double fgExecTime = (double)cv::getTickCount();

    tps::FeatureGenerator fg = tps::FeatureGenerator(referenceImage, targetImage, percentage);
    fg.run();

    fgExecTime = ((double)cv::getTickCount() - fgExecTime)/cv::getTickFrequency();
    referencesKPs.push_back(fg.getReferenceKeypoints());
    targetsKPs.push_back(fg.getTargetKeypoints());
    std::cout << "FeatureGenerator execution time: " << fgExecTime << std::endl;
    std::cout << "============================================" << std::endl;
}

int main(int argc, char** argv) {
  if (argc < 1) {
    std::cout << "Precisa passar o arquivo de configuração coração! \n";    
    return 0;
  }

  cudaDeviceReset();
  cudaThreadExit();

  // Reading the main file
  std::ifstream infile;
  infile.open(argv[1]);

  std::string line; 
  std::getline(infile, line);

  std::size_t pos = line.find('.');
  std::string extension = line.substr(pos);

  tps::Image referenceImage = tps::ITKImageHandler::loadImageData(line);

  // tps::ImageDeformation id = tps::ImageDeformation(referenceImage, "bio-Def.nii.gz");
  // id.apply3DSinDeformation();

  std::vector< tps::Image > targetImages;
  std::vector< std::string > outputNames;
  std::vector< float > percentages;

  // Reading each iteration configuration file
  int nFiles = 0;
  for (line; std::getline(infile, line); infile.eof(), nFiles++) {
    readConfigFile(line, targetImages, outputNames, percentages);
  }

  std::vector< std::vector< std::vector<float> > > referencesKPs;
  std::vector< std::vector< std::vector<float> > > targetsKPs;

  // Generating the Control Points for each pair of reference and target images
  std::cout << "============================================" << std::endl;
  for (int i = 0; i < nFiles; i++) {
    std::cout << "Generating the CPs for entry number " << i << std::endl;
    runFeatureGeneration(referenceImage, targetImages[i], percentages[i], referencesKPs, targetsKPs);
  }

  // Verifying the total memory occupation inside the GPU
  size_t avail;
  size_t total;
  cudaMemGetInfo( &avail, &total );
  size_t used = (total - avail)/(1024*1024);
  std::cout << "Device memory used: " << used/(1024*1024) << "MB" << std::endl;

  // Allocating the maximun possible of free memory in the GPU
  std::vector< tps::CudaMemory > cudaMemories;
  std::cout << "============================================" << std::endl;
  for (int i = 0; i < nFiles; i++) {
    int lastI = i;
    while(used <= 1800) {
      std::cout << "Entry number " << i << " will run." << std::endl;
      tps::CudaMemory cm = tps::CudaMemory(targetImages[i].getDimensions(), referencesKPs[i]);
      if (used+cm.memoryEstimation() > 1800) break;
      cm.allocCudaMemory(targetImages[i]);
      cudaMemories.push_back(cm);
      cudaMemGetInfo( &avail, &total );
      used = (total - avail)/(1024*1024);
      std::cout << "Device used memory = " << used << "MB" << std::endl;
      i++;
      if (i >= nFiles) break;
      else std::cout << "--------------------------------------------" << std::endl;
    }

    // Execution of the TPS, both in the Host and in the Device
    std::cout << "============================================" << std::endl;
    for (int j = lastI; j < i; j++) {
      std::cout << "#Execution = " << j << std::endl;
      std::cout << "#Keypoints = " << referencesKPs[j].size() << std::endl;
      std::cout << "#Percentage = " << percentages[j] << std::endl;
      std::cout << "++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

      // double parallelTpsExecTime = (double)cv::getTickCount();
      // tps::ParallelTPS parallelTPS = tps::ParallelTPS(referencesKPs[j], targetsKPs[j], targetImages[j], outputNames[j]+"Parallel"+extension);
      // parallelTPS.run();
      // parallelTpsExecTime = ((double)cv::getTickCount() - parallelTpsExecTime)/cv::getTickFrequency();
      // std::cout << "Parallel TPS execution time: " << parallelTpsExecTime << std::endl;

      // std::cout << "++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

      double CUDAcTpsExecTime = (double)cv::getTickCount();
      tps::CudaTPS CUDActps = tps::CudaTPS(referencesKPs[j], targetsKPs[j], targetImages[j], outputNames[j]+"Cuda"+extension, cudaMemories[j]);
      CUDActps.run();
      CUDAcTpsExecTime = ((double)cv::getTickCount() - CUDAcTpsExecTime)/cv::getTickFrequency();
      std::cout << "Cuda TPS execution time: " << CUDAcTpsExecTime << std::endl;
      cudaMemories[j].freeMemory();
      std::cout << "============================================" << std::endl;
    }
  }
  cudaThreadExit();
  cudaDeviceReset();
  return 0;
}