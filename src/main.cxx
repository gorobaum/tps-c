#include "image.h"
#include "surf.h"
#include "basictps.h"

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

  double basicTpsExecTime = (double)cv::getTickCount();
  tps::BasicTPS tps = tps::BasicTPS(surf.getReferenceKeypoints(), surf.getTargetKeypoints(), targetImage, "basicReg.png");
  tps.run();
  basicTpsExecTime = ((double)cv::getTickCount() - basicTpsExecTime)/cv::getTickFrequency();
  std::cout << "Basic TPS execution time: " << basicTpsExecTime << std::endl;

	std::cout << "Total execution time: " << basicTpsExecTime+surfExecTime << std::endl << std::endl;  

  return 0;
}