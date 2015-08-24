#include "image/image.h"
#include "image/itkimagehandler.h"
#include "image/imagedeformation.h"
#include "feature/featuregenerator.h"
#include "utils/cudamemory.h"
#include "utils/tpsinstance.h"
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

  std::vector< tps::TpsInstance > tpsInstances;

  // Reading each iteration configuration file
  int nFiles = 0;
  for (line; std::getline(infile, line); infile.eof(), nFiles++) {
    tps::TpsInstance newInstance(line, referenceImage);
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
  std::cout << "============================================" << std::endl;
  for (int currentExecution = 0; currentExecution < nFiles; currentExecution++) {
    int lastExecution = currentExecution;
    while(used <= 1800) {
      std::cout << "Entry number " << currentExecution << " will run." << std::endl;
      int newUsed = tpsInstances[currentExecution].allocCudaMemory(used);
      if (newUsed == used) break;
      else {
        used = newUsed;
        currentExecution++;
        if (currentExecution >= nFiles) break;
      }
    }

    // Execution of the TPS, both in the Host and in the Device
    std::cout << "============================================" << std::endl;
    for (int j = lastExecution; j < currentExecution; j++) {
      tpsInstances[j].runCudaTPS();
    }
  }
  cudaThreadExit();
  cudaDeviceReset();
  return 0;
}