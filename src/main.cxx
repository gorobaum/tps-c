#include "image/image.h"
#include "image/itkimagehandler.h"
#include "image/opcvimagehandler.h"
#include "image/imagehandler.h"
#include "image/imagedeformation.h"
#include "utils/tpsinstance.h"

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

  tps::ImageHandler *imageHandler;

  if (extension.compare(".nii.gz") == 0)
    imageHandler = new tps::ITKImageHandler();
  else
    imageHandler = new tps::OPCVImageHandler();

  tps::Image referenceImage = imageHandler->loadImageData(line);

  // tps::ImageDeformation id = tps::ImageDeformation(referenceImage);
  // id.apply3DSinDeformation();
  // tps::Image deformedImage = id.getResult();
  // imageHandler->saveImageData(deformedImage, "result.png");


  std::vector< tps::TpsInstance > tpsInstances;

  // Reading each iteration configuration file
  int nFiles = 0;
  for (line; std::getline(infile, line); infile.eof(), nFiles++) {
    tps::TpsInstance newInstance(line, referenceImage, imageHandler);
    tpsInstances.push_back(newInstance);
  }

  // Verifying the total memory occupation inside the GPU
  size_t avail;
  size_t total;
  cudaMemGetInfo( &avail, &total );
  size_t used = (total - avail)/(1024*1024);
  std::cout << "Device memory used: " << used/(1024*1024) << "MB" << std::endl;

  // Allocating the maximun possible of free memory in the GPU
  std::vector< tps::CudaMemory > cudaMemories;
  for (int currentExecution = 0; currentExecution < nFiles; currentExecution++) {
    int lastExecution = currentExecution;
    while(used <= 1800) {
      int newUsed = tpsInstances[currentExecution].allocCudaMemory(used);
      if (newUsed == used) break;
      else {
        used = newUsed;
        currentExecution++;
        if (currentExecution >= nFiles) break;
      }
    }

    // Execution of the TPS, both in the Host and in the Device
    for (int j = lastExecution; j < currentExecution; j++) {
      std::cout << "============================================" << std::endl;
      // tpsInstances[j].runParallelTPS();
      tpsInstances[j].runCudaTPS();
    }
  }
  cudaThreadExit();
  cudaDeviceReset();
  return 0;
}