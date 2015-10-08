#include "image/itkimagehandler.h"
#include "image/opcvimagehandler.h"
#include "image/imagehandler.h"
#include "image/imagedeformation.h"
#include "utils/tpsinstance.h"
#include "utils/mastercontroller.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>

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

  tps::MasterController mc(tpsInstances);

  mc.run();

  cudaThreadExit();
  cudaDeviceReset();
  return 0;
}