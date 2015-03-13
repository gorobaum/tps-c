#include "image.h"
#include "surf.h"
#include "tps.h"

#include <string>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cout << "Precisa passar o nome dos arquivos coração! \n";    
		return 0;
	}

 	tps::Image referenceImage = tps::Image(argv[1]);
  tps::Image targetImage = tps::Image(argv[2]);
  std::string outputName = argv[3];

  int minHessian = 400;

  double surfExecTime = (double)cv::getTickCount();
  tps::Surf surf = tps::Surf(referenceImage, targetImage, minHessian);
  surf.run(true);
  surfExecTime = ((double)cv::getTickCount() - surfExecTime)/cv::getTickFrequency();
  std::cout << "Surf execution time: " << surfExecTime << std::endl;

  double tpsExecTime = (double)cv::getTickCount();
  tps::TPS tps = tps::TPS(surf.getReferenceKeypoints(), surf.getTargetKeypoints(), targetImage, outputName);
  tps.run();
  tpsExecTime = ((double)cv::getTickCount() - tpsExecTime)/cv::getTickFrequency();
  std::cout << "TPS execution time: " << tpsExecTime << std::endl;

	std::cout << "Total execution time: " << tpsExecTime+surfExecTime << std::endl << std::endl;  

  return 0;
}